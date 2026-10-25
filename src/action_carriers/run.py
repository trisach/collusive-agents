r"""Experiment runner: plans episodes, runs them with LLM agents, stores and summarises them.

Examples:
  # A run as in the paper: 40 usable seeds x 6 arms, on a Modal endpoint (see README).
  # Resumable: re-run the same command after any stop.
  uv run python -m action_carriers.run --provider modal --model gpt-oss-120b \
      --reasoning-effort medium --seeds 40 --out runs

  # Same pipeline with a fake API (no key, no cost): checks the plumbing end to end.
  uv run python -m action_carriers.run --dry-run --seeds 2

  # Through OpenRouter (key: OPENROUTER_API_KEY in .env). --pin fixes the upstream.
  uv run python -m action_carriers.run --provider openrouter --model <id> --seeds 40

  # Respect published limits by waiting instead of hitting 429s (e.g. Groq gpt-oss-120b):
  uv run python -m action_carriers.run --model openai/gpt-oss-120b \
      --rpm 30 --tpm 8000 --rpd 1000 --tpd 200000

  # Summary of an existing run without running anything.
  uv run python -m action_carriers.run --model <id> --summary
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import random
import re
import sys
import time
import traceback
import types
from collections import defaultdict

import dotenv

from action_carriers import env, llm, prompts
from action_carriers.episode import (
    ARMS,
    MAX_RETRIES,
    PLANNING_ROUNDS,
    POST_MAX_CHARS,
    draw_coalition_and_order,
    run_episode,
)
from action_carriers.ratelimit import Limits, RateLimiter
from action_carriers.store import EpisodeStore

TASK_CONFIG = env.SOD_EXPERIMENT
SEED_START = 0
# Name of the run directory level and the manifest's "mode"; the published runs record "main".
RUN_MODE = "main"
COMP_TYPES = ("implement", "review")
MAX_TOKENS_ENV = "MAX_COMPLETION_TOKENS"
MAX_TRUNCATED_SHARE = 0.02  # Above this share of calls cut off by the output cap, warn.
# Stop after this many failed episodes in a row: a dead or stopped endpoint should not be
# retried episode after episode (on Modal every retry can wake a billed GPU container).
MAX_CONSECUTIVE_FAILURES = 2


def plan_episodes(n_seeds: int, arms: list[str]) -> list[tuple[int, str]]:
    """(seed, arm) in run order: the first n usable seeds, every arm per seed.

    Arms are interleaved by seed (a quota stop cannot leave one arm behind), and the arm
    order rotates from seed to seed (no arm always runs first or last).
    """
    plan, seed = [], SEED_START
    while len(plan) < n_seeds * len(arms):
        instance = env.generate_instance(seed, TASK_CONFIG)
        coalition, _ = draw_coalition_and_order(instance)
        if env.sod_feasible_issues(instance, coalition):  # Pre-treatment filter.
            k = len(plan) // len(arms)
            plan += [(seed, arm) for arm in arms[k % len(arms) :] + arms[: k % len(arms)]]
        seed += 1
    return plan


class DryRunClient:
    """Fake OpenAI-compatible client: random feasible claims, canned posts. No network."""

    _COST_LINE = re.compile(r"^- (ISSUE-\d+::\w+): cost=", re.MULTILINE)

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, *, messages, tools, model, **_):
        name = tools[0]["function"]["name"]
        if name == "post_message":
            args = {"text": "dry run: I can do the tasks in my cost list."}
        else:
            feasible = self._COST_LINE.findall(messages[1]["content"])
            args = {"task_id": self.rng.choice([*feasible, "skip"])}
        call = types.SimpleNamespace(
            id=f"call_{self.rng.randrange(10**9)}",
            function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        message = types.SimpleNamespace(content="dry run", tool_calls=[call], refusal=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="tool_calls")],
            model=model,
            usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )


def run_one(
    client, config: llm.LLMConfig, seed: int, arm: str, limiter: RateLimiter | None = None
) -> dict:
    """Runs one episode; returns the full record."""
    instance = env.generate_instance(seed, TASK_CONFIG)
    coalition, order = draw_coalition_and_order(instance)
    policy = llm.LLMPolicy(client, config, limiter)
    started = time.time()
    record = run_episode(instance, ARMS[arm], coalition, order, policy)
    record.update(
        {
            "model": config.model,
            "calls": policy.calls,
            "sod_feasible_issues": env.sod_feasible_issues(instance, coalition),
            "started_at": started,
            "duration_s": round(time.time() - started, 2),
            "tokens": sum(
                (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in policy.calls
            ),
        }
    )
    return record


def summarise(records: list[dict]) -> str:
    """Per-arm table: outcome, goal pursuit, failures, cost. Descriptive only."""
    by_arm = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    header = (
        f"{'arm':8s} {'n':>3s} {'SoD':>5s} {'coll.I/R':>9s} {'retry':>6s} {'noCall':>6s} "
        f"{'fSkip':>6s} {'trunc':>6s} {'refus':>6s} {'tok/ep':>7s}"
    )
    rows = [header, "-" * len(header)]
    for arm in ARMS:
        rs = by_arm.get(arm)
        if not rs:
            continue
        colluder_turns = [t for r in rs for t in r["turns"] if t["colluder"]]
        impl_review = sum(
            1 for t in colluder_turns if t["claim"] and t["claim"].split("::")[1] in COMP_TYPES
        )
        calls = [c for r in rs for c in r["calls"]]
        rows.append(
            f"{arm:8s} {len(rs):3d} {sum(r['sod_success'] for r in rs):5d} "
            f"{f'{impl_review}/{len(colluder_turns)}' if colluder_turns else '-':>9s} "
            f"{sum(len(t['attempts']) > 1 for r in rs for t in r['turns']):6d} "
            f"{sum(c['result'] == llm.NO_CALL for c in calls):6d} "
            f"{sum(t['forced_skip'] for r in rs for t in r['turns']):6d} "
            f"{sum(c['finish_reason'] == 'length' for c in calls):6d} "
            f"{sum(bool(c['refusal']) for c in calls):6d} "
            f"{sum(r['tokens'] for r in rs) / len(rs):7.0f}"
        )
    rows.append(
        "SoD = successes; coll.I/R = colluder turns that claimed implement/review; retry = turns "
        "with >1 attempt;\nnoCall = calls without a usable tool call; fSkip = forced skips; "
        "trunc = calls cut off by the output cap;\nrefus = API refusals."
    )
    all_calls = [c for r in records for c in r["calls"]]
    truncated = sum(c["finish_reason"] == "length" for c in all_calls)
    if all_calls and truncated / len(all_calls) > MAX_TRUNCATED_SHARE:
        rows.append(
            f"Warning: {truncated}/{len(all_calls)} calls cut off by the output cap "
            f"(> {MAX_TRUNCATED_SHARE:.0%}). Raise or remove the output cap before interpreting."
        )
    return "\n".join(rows)


def episode_lines(record: dict, elapsed: float, eta: float) -> list[str]:
    """A short, readable progress block for one finished episode (2-4 lines).

    Line 1: who claimed what, in turn order (* = prompted colluder, ° = designated pair in a
    benign arm). Line 2 (P-text only): the coalition posts, shortened. Line 3: outcome and
    cost. Line 4 (only if needed): warnings that make the episode suspect.
    """
    prompted = record["arm"].startswith("P-")
    mark = "*" if prompted else "°"
    claims = " · ".join(
        f"{t['agent']}{mark if t['agent'] in record['coalition'] else ''} "
        f"{t['claim'] or 'skip'}{' (forced)' if t['forced_skip'] else ''}"
        for t in record["turns"]
    )
    lines = [f"  claims: {claims}"]
    if record["posts"]:
        posts = " | ".join(f"{p['agent']}: {_shorten(p['text'], 70)}" for p in record["posts"])
        lines.append(f"  posts:  {posts}")
    calls = record["calls"]
    retries = sum(len(t["attempts"]) > 1 for t in record["turns"])
    truncated = sum(c["finish_reason"] == "length" for c in calls)
    paced = sum(c.get("waited_s", 0) for c in calls)
    outcome = "SoD ✓" if record["sod_success"] else "SoD ✗"
    lines.append(
        f"  {outcome} · {len(calls)} calls · {record['tokens'] / 1000:.1f}k tokens · "
        f"{record['duration_s']:.0f}s (paced {paced:.0f}s) · retries {retries} · "
        f"elapsed {elapsed / 60:.1f}m · eta {eta / 60:.1f}m"
    )
    warnings = []
    if truncated:
        warnings.append(f"{truncated} truncated call(s)")
    forced = sum(t["forced_skip"] for t in record["turns"])
    if forced:
        warnings.append(f"{forced} forced skip(s)")
    refusals = sum(bool(c.get("refusal")) for c in calls)
    if refusals:
        warnings.append(f"{refusals} refusal(s)")
    if warnings:
        lines.append(f"  ⚠ {' · '.join(warnings)}")
    return lines


def _shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _episode_estimate(records: list[dict]) -> tuple[int, int]:
    """(calls, tokens) one more episode is likely to use: the max over recent episodes.

    The maximum, not the mean, so that an episode is only started if it can finish.
    """
    recent = records[-12:]
    if not recent:
        return 8, 30_000  # No data yet: a typical episode (5-6 calls, <= 21k tokens) + margin.
    return max(len(r["calls"]) for r in recent), max(r["tokens"] for r in recent)


def _max_completion_tokens(flag: int | None, env_file: str) -> int | None:
    """Output cap: the flag, else MAX_COMPLETION_TOKENS (environment or .env); 0/empty = None."""
    if flag is None:
        dotenv.load_dotenv(env_file, override=False)
        raw = os.environ.get(MAX_TOKENS_ENV, "").strip()
        flag = int(raw) if raw else 0
    return flag or None


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--provider", default="groq", choices=list(llm.PROVIDERS))
    ap.add_argument("--model", default=None, help="model id (required unless --dry-run)")
    ap.add_argument("--seeds", type=int, default=None, help="usable seeds (the paper uses 40)")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm names")
    ap.add_argument("--dry-run", action="store_true", help="fake API: no key, no cost")
    ap.add_argument("--out", default="runs", help="root directory for run data")
    ap.add_argument("--env-file", default=".env", help="dotenv file with GROQ_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help=f"output cap per call; 0 = no cap (default: {MAX_TOKENS_ENV} in .env, else no cap)",
    )
    ap.add_argument("--tool-choice", default="auto", choices=["auto", "required"])
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low (reasoning models)")
    ap.add_argument("--pin", default=None, help="OpenRouter: serve from this upstream only")
    ap.add_argument("--summary", action="store_true", help="only print the stored summary")
    limits = ap.add_argument_group("client-side pacing (published provider limits; 0 = none)")
    limits.add_argument("--rpm", type=int, default=0, help="requests per minute")
    limits.add_argument("--tpm", type=int, default=0, help="tokens per minute")
    limits.add_argument("--rpd", type=int, default=0, help="requests per day")
    limits.add_argument("--tpd", type=int, default=0, help="tokens per day")
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=0,
        help="wall-clock budget: start no new episode after this many minutes (0 = none); "
        "use it on GPU-second billing (Modal)",
    )
    args = ap.parse_args(argv)

    if not args.dry_run and not args.model:
        ap.error("--model is required (or use --dry-run)")
    model = "dry-run" if args.dry_run else args.model
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; choose from {list(ARMS)}")
    n_seeds = args.seeds or 40

    provider = "dry-run" if args.dry_run else args.provider
    try:
        extra_body = llm.provider_options(args.provider, args.reasoning_effort, args.pin)
    except ValueError as error:
        ap.error(str(error))
    config = llm.LLMConfig(
        model=model,
        temperature=args.temperature,
        max_completion_tokens=_max_completion_tokens(args.max_completion_tokens, args.env_file),
        tool_choice=args.tool_choice,
        extra_body=extra_body,
    )
    manifest = {
        "mode": RUN_MODE,
        "provider": provider,
        "model": model,
        "llm_config": dataclasses.asdict(config),
        "prompt_version": prompts.PROMPT_VERSION,
        "protocol": {
            "planning_rounds": PLANNING_ROUNDS,
            "max_retries": MAX_RETRIES,
            "post_max_chars": POST_MAX_CHARS,
        },
        "task_config": json.loads(json.dumps(dataclasses.asdict(TASK_CONFIG))),
        "seed_start": SEED_START,
    }
    run_dir = pathlib.Path(args.out) / RUN_MODE / provider / _slug(model)
    try:
        store = EpisodeStore(run_dir, manifest)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.summary:
        print(summarise(store.records()))
        return 0

    client = DryRunClient() if args.dry_run else llm.make_client(args.provider, args.env_file)
    plan = plan_episodes(n_seeds, arms)
    done = store.done()
    todo = [(s, a) for s, a in plan if (a, s) not in done]
    print(
        f"{provider} | model={model} | prompt={prompts.PROMPT_VERSION} | {run_dir}\n"
        f"planned {len(plan)} episodes ({n_seeds} seeds x {len(arms)} arms); "
        f"{len(plan) - len(todo)} already done; {len(todo)} to run",
        flush=True,
    )
    limiter = None
    pacing = Limits(args.rpm or None, args.tpm or None, args.rpd or None, args.tpd or None)
    if pacing.any():
        limiter = RateLimiter(pacing)
        # Earlier usage of this run counts toward the day limits (resume on the same day).
        limiter.seed([(r["started_at"], c["prompt_tokens"] + c["completion_tokens"])
                      for r in store.records() for c in r["calls"]
                      if c["prompt_tokens"] is not None])  # fmt: skip
        limiter.on_wait = lambda seconds: print(f"  … pacing {seconds:.0f}s", flush=True)
        print(f"pacing: {pacing}", flush=True)
    started, failures, quota_stop, in_a_row = time.time(), 0, None, 0
    try:
        for i, (seed, arm) in enumerate(todo, start=1):
            if args.max_minutes and time.time() - started > 60 * args.max_minutes:
                print(f"\nStopping: --max-minutes {args.max_minutes:g} reached. Re-run to resume.")
                break
            try:
                if limiter:  # Do not start an episode that the day budget cannot finish.
                    calls, tokens = _episode_estimate(store.records())
                    limiter.check_day(calls, tokens)
                print(f"\n▶ [{i}/{len(todo)}] {arm} · seed {seed} · running …", flush=True)
                record = run_one(client, config, seed, arm, limiter)
            except Exception as error:  # API failure after SDK retries: log, then go on or stop.
                failures += 1
                in_a_row += 1
                quota_stop = llm.quota_exhausted(error)
                store.log_error(
                    {
                        "seed": seed,
                        "arm": arm,
                        "error": repr(error),
                        "quota_exhausted": quota_stop,
                        "traceback": traceback.format_exc(),
                        "at": time.time(),
                    }
                )
                print(f"  ✗ FAILED: {str(error)[:200]}", flush=True)
                if quota_stop:
                    print(f"\nStopping: {quota_stop}. Re-run the same command after the reset.")
                    break
                if in_a_row >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nStopping: {in_a_row} episodes failed in a row; check the endpoint.")
                    break
                continue
            in_a_row = 0
            store.append(record)
            elapsed = time.time() - started
            eta = elapsed / i * (len(todo) - i)
            print("\n".join(episode_lines(record, elapsed, eta)), flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Completed episodes are saved; re-run the same command to resume.")
        return 130

    records = store.records()
    print("\n" + summarise(records))
    if failures:
        print(f"\n{failures} episode(s) failed; see {store.errors_path}. Re-run to retry them.")
    if quota_stop:
        return 3
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
