r"""Manage the Modal Dedicated Endpoint that serves the LLM agents (billed per GPU-second).

Every subcommand is run by hand; nothing here starts on its own. Order of use:

    uv run modal setup                                   # once: log in (browser)
    uv run python scripts/modal_endpoint.py create       # deploy gpt-oss-120b (asks first)
    uv run python scripts/modal_endpoint.py configure    # write URL + proxy token to .env
    uv run python scripts/modal_endpoint.py smoke        # 3 test calls; checks tool calling
    uv run python scripts/modal_endpoint.py watchdog --max-dollars 10   # 2nd terminal, during runs
    uv run python scripts/modal_endpoint.py status       # containers now (idle = 0)
    uv run python scripts/modal_endpoint.py kill         # KILL SWITCH: stop running containers
    uv run python scripts/modal_endpoint.py kill --permanent   # delete the endpoint for good

Cost model (Modal docs): a Dedicated Endpoint bills the GPU, CPU and memory of its running
containers; scaled to zero it costs nothing. Plan credits cover Dedicated Endpoints, NOT Shared
Endpoints (those bill tokens from the first request). Any request, including a retry from a
client, can wake a container. So: stop the client first, then use the kill switch.

Secrets: the proxy-token secret is written to .env (mode 600) and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ENDPOINT = "gpt-oss-120b"  # Endpoint name; Modal also uses it as the request `model` name.
MODEL = "openai/gpt-oss-120b"  # Default Modal Library model (open weights, MXFP4, one 80 GB GPU).
ENV_FILE = pathlib.Path(".env")
TOKEN_NAME = "collusive-agents"
# Upper bound per running container for the watchdog's estimate: H200 $4.54/h is the dearest
# single 80 GB+ GPU that fits the model (H100 $3.95/h), plus CPU and memory (modal.com/pricing).
DOLLARS_PER_CONTAINER_HOUR = 5.0

_URL = re.compile(r"https://[^\s\"']+\.modal\.(?:run|direct)[^\s\"']*")
_APP_ID = re.compile(r"\bap-[A-Za-z0-9]+\b")
_CONTAINER_ID = re.compile(r"\bta-[A-Za-z0-9]+\b")
_TOKEN_ID = re.compile(r"\bwk-[\w-]+")
_TOKEN_SECRET = re.compile(r"\bws-[\w-]+")


def modal(*args: str, capture: bool = True, check: bool = True, stdin: str | None = None) -> str:
    """Runs the Modal CLI of this environment (no TTY, so output is plain text).

    Args:
        *args: CLI arguments after ``modal``.
        capture: Return stdout instead of printing it.
        check: Exit on a non-zero status.
        stdin: Text sent to the command's standard input.

    Returns:
        The captured stdout ("" if not captured).
    """
    result = subprocess.run(
        [sys.executable, "-m", "modal", *args],
        capture_output=capture,
        text=True,
        input=stdin,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip()[-500:] if capture else ""
        sys.exit(f"modal {args[0]} {args[1] if len(args) > 1 else ''} failed. {detail}")
    return result.stdout if capture else ""


def confirm(prompt: str, answer: str) -> None:
    """Exits unless the user types ``answer`` exactly."""
    if input(f"{prompt}\nType '{answer}' to continue: ").strip() != answer:
        sys.exit("Cancelled.")


def endpoint_exists(name: str) -> bool:
    """Whether an Endpoint of this name is provisioning or running."""
    return f'"{name}"' in modal("endpoint", "list", "--json")


def endpoint_info(name: str) -> str:
    """Raw ``modal endpoint info --json`` text (holds the URL and the App id; no secrets)."""
    return modal("endpoint", "info", name, "--json")


def running_containers(name: str) -> tuple[list[str], bool]:
    """Running container ids of the Endpoint's App.

    Returns:
        (container ids, scoped). If the App id cannot be read, ``scoped`` is False and the
        ids are those of the whole workspace (an over-count: safe for the watchdog, but
        ``kill`` asks before stopping them).
    """
    app = _APP_ID.search(endpoint_info(name))
    args = ["container", "list", "--json"] + (["--app-id", app.group(0)] if app else [])
    return sorted(set(_CONTAINER_ID.findall(modal(*args)))), app is not None


def set_env(key: str, value: str) -> None:
    """Sets KEY=value in .env (replaces an existing line), file mode 600."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    lines = [line for line in lines if not line.startswith(f"{key}=")] + [f"{key}={value}"]
    ENV_FILE.write_text("\n".join(lines) + "\n")
    os.chmod(ENV_FILE, 0o600)


def get_env(key: str) -> str | None:
    """Value of KEY in the shell environment or .env."""
    import dotenv

    dotenv.load_dotenv(ENV_FILE, override=False)
    return os.environ.get(key) or None


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_create(args) -> None:
    """Deploys the Dedicated Endpoint after a cost checklist."""
    modal("profile", "current")  # Fails early if `modal setup` was not run.
    if endpoint_exists(args.name):
        sys.exit(f"Endpoint '{args.name}' already exists. Use `status` or `configure`.")
    confirm(
        f"About to create a DEDICATED Endpoint '{args.name}' for {args.model} (billed per "
        "GPU-second; scales to zero when idle).\nBefore you continue, in the dashboard:\n"
        "  1. Usage & Billing: set the SPEND LIMIT to $0 (usage beyond credits is refused).\n"
        "  2. Check that your credits are on this workspace.\n"
        "No region is pinned (pinning costs 1.15-1.75x).",
        "create",
    )
    modal("endpoint", "create", "--name", args.name, "--model", args.model, capture=False)
    print(
        "\nNext, in the dashboard (Endpoints > this endpoint > autoscaling):\n"
        "  - max containers = 1, min containers = 0, buffer = 0\n"
        "Then wait until it shows as ready, and run `configure`."
    )


def cmd_configure(args) -> None:
    """Writes MODAL_BASE_URL and a new MODAL_PROXY_TOKEN into .env (secret never printed)."""
    urls = _URL.findall(endpoint_info(args.name))
    if not urls:
        sys.exit("No URL in `modal endpoint info` yet. Wait until the Endpoint is ready.")
    base = urls[0].rstrip("/")
    base = base if base.endswith("/v1") else base + "/v1"
    set_env("MODAL_BASE_URL", base)
    print(f"MODAL_BASE_URL={base}")
    if get_env("MODAL_PROXY_TOKEN") and not args.new_token:
        print("MODAL_PROXY_TOKEN already set; kept (use --new-token to create another).")
        return
    raw = modal("workspace", "proxy-tokens", "create", "--name", TOKEN_NAME, "--json")
    token_id, secret = _TOKEN_ID.search(raw), _TOKEN_SECRET.search(raw)
    if not (token_id and secret):  # Never print `raw`: it holds the secret.
        sys.exit(
            "Could not read the new token. Create one in the dashboard (Settings > Proxy "
            "Auth Tokens) and set MODAL_PROXY_TOKEN=wk-<id>.ws-<secret> in .env by hand."
        )
    set_env("MODAL_PROXY_TOKEN", f"{token_id.group(0)}.{secret.group(0)}")
    print(f"MODAL_PROXY_TOKEN written to {ENV_FILE} (id {token_id.group(0)}; secret not shown).")


def cmd_smoke(args) -> None:
    """Three real calls: waits for the model, then a claim turn (twice) and a plan turn."""
    import openai

    from action_carriers import prompts

    base, key = get_env("MODAL_BASE_URL"), get_env("MODAL_PROXY_TOKEN")
    if not (base and key):
        sys.exit("Run `configure` first (MODAL_BASE_URL / MODAL_PROXY_TOKEN missing).")
    client = openai.OpenAI(api_key=key, base_url=base, max_retries=0, timeout=60)

    print(f"Waiting for {base} (a cold start loads the model; up to {args.wait_minutes} min) …")
    deadline, started = time.time() + 60 * args.wait_minutes, time.time()
    while True:
        try:
            served = [m.id for m in client.models.list().data]
            break
        except Exception as error:  # Not ready yet: connection error, 5xx or timeout.
            if time.time() > deadline:
                sys.exit(f"Not ready after {args.wait_minutes} min ({error!r}). Run `kill`?")
            print(f"  … not ready ({type(error).__name__}); retry in 20 s", flush=True)
            time.sleep(20)
    print(f"Ready after {time.time() - started:.0f} s. Served models: {served}")
    if not (args.model or served):
        sys.exit("The endpoint lists no models; pass --model.")
    model = args.model or served[0]
    client = client.with_options(timeout=600)  # The first generation may still be warming up.

    examples = prompts.canonical_examples()
    cases = [
        ("claim (cold cache)", "colluder_claim_action_only", prompts.TOOLS["claim"]),
        ("claim (warm)", "colluder_claim_action_only", prompts.TOOLS["claim"]),
        ("plan post", "colluder_plan_text", prompts.TOOLS["plan"]),
    ]
    failures = []
    for label, example, tools in cases:
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": examples[example]},
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                extra_body={"reasoning_effort": args.reasoning_effort},
            )
        except Exception as error:
            failures.append(f"{label}: request failed: {str(error)[:300]}")
            print(f"\n[{label}] FAILED: {str(error)[:300]}")
            continue
        message = response.choices[0].message
        field = next((f for f in ("reasoning", "reasoning_content") if getattr(message, f, None)),
                     None)  # fmt: skip
        calls = message.tool_calls or []
        wanted = tools[0]["function"]["name"]
        parsed = None
        if calls and calls[0].function.name == wanted:
            try:
                parsed = json.loads(calls[0].function.arguments)
            except json.JSONDecodeError:
                parsed = None
        print(
            f"\n[{label}] {time.time() - start:.1f} s · model={response.model} · "
            f"finish={response.choices[0].finish_reason} · tokens in/out="
            f"{response.usage.prompt_tokens}/{response.usage.completion_tokens}\n"
            f"  tool call: {calls[0].function.name + calls[0].function.arguments if calls else '-'}"
            f"\n  reasoning field: {field or '-'}"
            + (f" ({len(getattr(message, field))} chars)" if field else "")
            + f"\n  text: {(message.content or '-')[:160]!r}"
        )
        if parsed is None:
            failures.append(f"{label}: no parsable {wanted} call")
        if field is None:
            failures.append(f"{label}: no reasoning text (M3/H4 need it)")
    print("\nSMOKE " + ("PASSED" if not failures else "FAILED:\n  - " + "\n  - ".join(failures)))
    print("The container now idles until the Endpoint's scale-down window ends (or run `kill`).")


def cmd_status(args) -> None:
    """Endpoint details and the containers running now."""
    print(modal("endpoint", "info", args.name))
    containers, scoped = running_containers(args.name)
    where = "of this Endpoint" if scoped else "in the WHOLE workspace (App id not found)"
    print(f"Running containers {where}: {len(containers)} {containers or ''}")
    print("Exact spend: dashboard > Usage & Billing.")


def cmd_kill(args) -> None:
    """KILL SWITCH. Default: stop running containers (Endpoint kept; next request wakes it)."""
    print("Stop the experiment runner first (Ctrl-C): its next request would wake a container.")
    if args.permanent:
        confirm(
            f"PERMANENT: deletes Endpoint '{args.name}' and its model cache. It cannot be "
            "restarted; `create` makes a new one (new URL; run `configure` again).",
            args.name,
        )
        modal("endpoint", "stop", args.name, capture=False, stdin="y\n")
        print("Endpoint stopped. No further compute charges from it.")
        return
    containers, scoped = running_containers(args.name)
    if not containers:
        print("No running containers: nothing is billing.")
        return
    if not scoped:
        confirm(
            f"Could not identify this Endpoint's App; {len(containers)} running container(s) "
            "in the WHOLE workspace would be stopped.",
            "stop",
        )
    for container in containers:
        modal("container", "stop", "--yes", container, capture=False)
    print(f"Stopped {len(containers)} container(s). Scaled to zero: no compute charges.")


def cmd_watchdog(args) -> None:
    """Estimates spend from running containers; at the cap, stops the Endpoint for good.

    The estimate counts only time while this watchdog runs, at an upper-bound hourly rate, so
    it errs on the side of stopping early. Start it before a run; Ctrl-C ends it.
    """
    print(
        f"Watchdog: cap ${args.max_dollars:.2f} at ≤ ${args.rate:.2f}/container-hour, poll "
        f"every {args.interval} s. At the cap it PERMANENTLY stops '{args.name}'.",
        flush=True,
    )
    spent, last, last_print, last_n = 0.0, time.time(), 0.0, -1
    try:
        while True:
            try:
                n = len(running_containers(args.name)[0])
            except SystemExit as error:  # CLI or network failure: keep the last count.
                n = max(last_n, 0)
                print(f"  ! poll failed ({error}); assuming {n} container(s)", flush=True)
            now = time.time()
            spent += n * args.rate * (now - last) / 3600
            last = now
            if n != last_n or now - last_print > 300:
                stamp = time.strftime("%H:%M:%S")
                print(f"{stamp}  containers={n}  est. spend=${spent:.2f}", flush=True)
                last_print, last_n = now, n
            if spent >= args.max_dollars:
                print(f"CAP REACHED (${spent:.2f}). Stopping Endpoint '{args.name}'.", flush=True)
                modal("endpoint", "stop", args.name, capture=False, stdin="y\n")
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nWatchdog ended. Estimated spend while watching: ${spent:.2f}.")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", default=ENDPOINT, help=f"Endpoint name (default {ENDPOINT})")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create", help="deploy the Dedicated Endpoint (asks first)")
    p.add_argument("--model", default=MODEL, help=f"Modal Library model id (default {MODEL})")
    p = sub.add_parser("configure", help="write MODAL_BASE_URL and MODAL_PROXY_TOKEN to .env")
    p.add_argument("--new-token", action="store_true", help="create a token even if one is set")
    p = sub.add_parser("smoke", help="3 test calls: readiness, tool calling, reasoning field")
    p.add_argument("--model", default=None, help="request model name (default: first served)")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--wait-minutes", type=float, default=20)
    sub.add_parser("status", help="endpoint details and running containers")
    p = sub.add_parser("kill", help="KILL SWITCH: stop running containers")
    p.add_argument("--permanent", action="store_true", help="delete the Endpoint for good")
    p = sub.add_parser("watchdog", help="stop the Endpoint when estimated spend hits a cap")
    p.add_argument("--max-dollars", type=float, required=True)
    p.add_argument("--rate", type=float, default=DOLLARS_PER_CONTAINER_HOUR,
                   help="assumed $ per running container-hour (upper bound)")  # fmt: skip
    p.add_argument("--interval", type=int, default=30, help="poll interval, seconds")
    args = ap.parse_args()
    {
        "create": cmd_create,
        "configure": cmd_configure,
        "smoke": cmd_smoke,
        "status": cmd_status,
        "kill": cmd_kill,
        "watchdog": cmd_watchdog,
    }[args.command](args)


if __name__ == "__main__":
    main()
