"""LLM agents: a harness ``Policy`` backed by an OpenAI-compatible chat endpoint (Groq).

One ``LLMPolicy`` serves one episode. For each turn it sends the system prompt, the user
prompt from ``prompts.build_user_prompt`` and the tools for the phase. On a retry (the
claim tool rejected the request, or the model made no tool call) it continues the same
conversation: the model sees its own earlier answer and the tool's reply, nothing else.

Every API call is recorded in ``LLMPolicy.calls`` (model id returned by the API, tokens,
latency, reasoning text, raw tool calls, prompt version). The runner stores them with the
episode. Endpoints: Groq, OpenRouter or a Modal Dedicated Endpoint (``PROVIDERS``). Secrets are
read from the environment or a ``.env`` file and never logged.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import time
from typing import Any

from action_carriers import prompts
from action_carriers.episode import MAX_RETRIES, Turn
from action_carriers.ratelimit import DailyBudgetExhausted, RateLimiter

# Returned to the episode loop when the response has no usable tool call. The claim tool
# rejects it (not a task id), so it uses one retry, like any other invalid request.
NO_CALL = "<no tool call>"

# Some providers (seen on OpenRouter) return HTTP 200 without ``choices`` when an upstream
# fails. The SDK does not retry those; we retry a few times with backoff, then fail loudly.
EMPTY_RESPONSE_RETRIES = 3


class EmptyResponseError(RuntimeError):
    """The provider answered without any choices, even after retries."""


_DAILY_MARKERS = ("per-day", "per day", "daily", "(tpd)", "(rpd)")


def quota_exhausted(error: BaseException) -> str | None:
    """Describes a DAILY quota exhaustion, or returns None for any other error.

    A per-minute 429 is worth retrying (the SDK does); a daily quota is not, so the runner
    stops instead of failing every remaining episode.

    Args:
        error: An exception raised by the client.

    Returns:
        A one-line description with the reset time if the provider gave one, else None.
    """
    if isinstance(error, DailyBudgetExhausted):  # Our own pacer stopped before the provider.
        return str(error)
    if getattr(error, "status_code", None) != 429:
        return None
    text = json.dumps(getattr(error, "body", None) or {}, default=str) + " " + str(error)
    if not any(marker in text.lower() for marker in _DAILY_MARKERS):
        return None
    reset = re.search(r'X-RateLimit-Reset"?\s*:\s*"?(\d{10,13})', text)
    when = ""
    if reset:
        value = int(reset.group(1))
        seconds = value / 1000 if value > 10**11 else value
        utc = datetime.datetime.fromtimestamp(seconds, tz=datetime.UTC)
        when = f"; resets at {utc:%Y-%m-%d %H:%M} UTC"
    return f"daily quota exhausted{when}"


@dataclasses.dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible endpoint and the environment variables that configure it.

    Attributes:
        name: Provider label (``--provider``).
        base_url: Fixed API base URL, or None if it is read from ``base_url_env``.
        key_env: Variable that holds the API key.
        base_url_env: Variable that holds the base URL (per-user endpoints, e.g. Modal).
    """

    name: str
    base_url: str | None
    key_env: str
    base_url_env: str | None = None


PROVIDERS = {
    p.name: p
    for p in (
        Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        # Modal Dedicated Endpoint (scripts/modal_endpoint.py writes both variables). The key
        # is a proxy token "wk-<id>.ws-<secret>"; the URL is "<endpoint-url>/v1".
        Provider("modal", None, "MODAL_PROXY_TOKEN", base_url_env="MODAL_BASE_URL"),
    )
}


def provider_options(
    provider: str, reasoning_effort: str | None = None, pin: str | None = None
) -> dict[str, Any]:
    """Provider-specific request options (``LLMConfig.extra_body``).

    Args:
        provider: A key of PROVIDERS.
        reasoning_effort: e.g. "low"/"medium"/"high" for reasoning models. Groq and Modal
            (vLLM/SGLang) take a top-level ``reasoning_effort``; OpenRouter takes
            ``reasoning.effort``.
        pin: OpenRouter only: serve from this one upstream provider, no fallbacks, so the
            same weights and quantization answer every call.

    Returns:
        The extra request body (empty if no option is set).
    """
    if pin and provider != "openrouter":
        raise ValueError("provider pinning is an OpenRouter option")
    body: dict[str, Any] = {}
    if reasoning_effort:
        if provider == "openrouter":
            body["reasoning"] = {"effort": reasoning_effort}
        else:
            body["reasoning_effort"] = reasoning_effort
    if pin:
        body["provider"] = {"order": [pin], "allow_fallbacks": False}
    return body


@dataclasses.dataclass(frozen=True)
class LLMConfig:
    """Sampling settings. Logged with every run.

    Attributes:
        model: Model id requested from the endpoint.
        temperature: 0.7 as in Colosseum (variation across samples is needed for rates).
        max_completion_tokens: Output cap per call; None sends no cap (the provider's
            maximum). A low cap truncates reasoning models before the tool call.
        tool_choice: "auto" (default): a reply without a tool call costs one retry. Groq
            cannot force gpt-oss to call a tool; with "required" it answers HTTP 400
            ``tool_use_failed`` instead, which ends the episode.
        extra_body: Provider-specific options, e.g. {"reasoning_effort": "low"}.
    """

    model: str
    temperature: float = 0.7
    max_completion_tokens: int | None = None
    tool_choice: str = "auto"
    extra_body: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_client(provider: str = "groq", env_file: str = ".env", max_retries: int = 8):
    """Creates the OpenAI-compatible client for a provider, with its key from the environment.

    The key (and, for per-user endpoints, the base URL) is read from the provider's variables
    in the environment or in ``env_file``; variables already set in the shell take precedence.
    The SDK retries rate-limit (429) and server errors with exponential backoff (Retry-After).

    Args:
        provider: A key of PROVIDERS.
        env_file: Path of the dotenv file (see ``.env.example``).
        max_retries: SDK-level retries per API call.

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        RuntimeError: If the provider's key or base URL is not configured.
    """
    import dotenv
    import openai

    endpoint = PROVIDERS[provider]
    dotenv.load_dotenv(env_file, override=False)
    api_key = os.environ.get(endpoint.key_env)
    if not api_key:
        raise RuntimeError(f"Set {endpoint.key_env} in the environment or in {env_file}.")
    base_url = endpoint.base_url or os.environ.get(endpoint.base_url_env or "")
    if not base_url:
        raise RuntimeError(f"Set {endpoint.base_url_env} in the environment or in {env_file}.")
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)


class LLMPolicy:
    """A harness Policy (``Callable[[Turn], str]``) for one episode."""

    def __init__(self, client, config: LLMConfig, limiter: RateLimiter | None = None):
        """Initialises the policy.

        Args:
            client: An OpenAI-compatible client (``client.chat.completions.create``).
            config: Sampling settings.
            limiter: Optional pacer shared across episodes (waits before each call).
        """
        self.client = client
        self.config = config
        self.limiter = limiter
        self.calls: list[dict] = []  # One record per API call, in call order.
        self._conversations: dict[str, list[dict]] = {}  # agent -> claim-turn messages

    def __call__(self, turn: Turn) -> str:
        """Returns the requested task id / "skip" (claim) or the post text (plan)."""
        if turn.phase == "plan":
            return self._plan(turn)
        if not turn.attempts:
            self._conversations[turn.agent] = self._start(turn)
        else:
            messages = self._conversations[turn.agent]
            self._add_feedback(messages, "claim", turn.attempts[-1], turn.feedback[-1])
        return self._ask(turn, self._conversations[turn.agent], len(turn.attempts))

    def _plan(self, turn: Turn) -> str:
        """P-text planning post. The episode loop calls this once; retries happen here."""
        messages = self._start(turn)
        text = NO_CALL
        for attempt in range(1 + MAX_RETRIES):
            if attempt:
                self._add_feedback(messages, "plan", text, None)
            text = self._ask(turn, messages, attempt)
            if text != NO_CALL:
                return text
        return ""

    @staticmethod
    def _start(turn: Turn) -> list[dict]:
        return [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.build_user_prompt(turn)},
        ]

    @staticmethod
    def _add_feedback(messages: list[dict], phase: str, last_result: str, reply: str | None):
        """Answers the previous assistant message before the next call.

        If it had no usable call, a user message asks for one (the claim tool's reply about
        "<no tool call>" would only confuse). Otherwise every tool call gets a tool message
        (the API requires it): the first gets the claim tool's reply, extras EXTRA_TOOL_CALL.
        """
        no_call = prompts.NO_TOOL_CALL.format(tool=prompts.TOOLS[phase][0]["function"]["name"])
        tool_calls = messages[-1].get("tool_calls") or []
        for i, call in enumerate(tool_calls):
            if i > 0:
                content = prompts.EXTRA_TOOL_CALL
            elif last_result == NO_CALL or reply is None:
                content = no_call
            else:
                content = reply
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        if not tool_calls:
            messages.append({"role": "user", "content": no_call})

    def _ask(self, turn: Turn, messages: list[dict], attempt: int) -> str:
        """One API call. Appends the assistant message to ``messages``; returns the request."""
        tools = prompts.TOOLS[turn.phase]
        wanted = tools[0]["function"]["name"]
        waited = 0.0
        start = time.monotonic()
        for empty_retries in range(EMPTY_RESPONSE_RETRIES + 1):
            if self.limiter:
                prompt_chars = sum(len(m.get("content") or "") for m in messages)
                waited += self.limiter.wait_for_minute_window(self.limiter.estimate(prompt_chars))
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice=self.config.tool_choice,
                temperature=self.config.temperature,
                **(
                    {"max_completion_tokens": self.config.max_completion_tokens}
                    if self.config.max_completion_tokens
                    else {}
                ),
                **({"extra_body": self.config.extra_body} if self.config.extra_body else {}),
            )
            if self.limiter:
                usage = getattr(response, "usage", None)
                self.limiter.record(
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            if response.choices:
                break
            if empty_retries == EMPTY_RESPONSE_RETRIES:
                raise EmptyResponseError(
                    f"no choices after {EMPTY_RESPONSE_RETRIES + 1} tries; "
                    f"provider error: {getattr(response, 'error', None)!r}"
                )
            time.sleep(2**empty_retries)
        latency = time.monotonic() - start
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ]
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )
        result = _extract_argument(tool_calls, wanted)
        usage = response.usage
        self.calls.append(
            {
                "agent": turn.agent,
                "phase": turn.phase,
                "attempt": attempt,
                "model_requested": self.config.model,
                "model_returned": response.model,
                "prompt_version": prompts.PROMPT_VERSION,
                "user_prompt": messages[1]["content"] if attempt == 0 else None,
                "content": message.content,
                "reasoning": _reasoning(message),
                "refusal": getattr(message, "refusal", None),
                "tool_calls": tool_calls,
                "result": result,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_s": round(latency - waited, 3),
                "waited_s": round(waited, 1),
                "empty_retries": empty_retries,
            }
        )
        return result


def _reasoning(message) -> str | None:
    """The model's reasoning text. Groq returns ``reasoning``; vLLM/SGLang ``reasoning_content``."""
    return getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)


def _extract_argument(tool_calls: list[dict], wanted: str) -> str:
    """The first call's argument if it is the expected tool, else NO_CALL.

    Malformed JSON or a missing argument returns the raw argument string, so the claim tool
    reports it as an unknown task id (one retry).
    """
    if not tool_calls or tool_calls[0]["function"]["name"] != wanted:
        return NO_CALL
    raw = tool_calls[0]["function"]["arguments"] or ""
    try:
        value = json.loads(raw).get(prompts.TOOL_ARGUMENT[wanted])
    except (json.JSONDecodeError, AttributeError):
        return raw
    return value if isinstance(value, str) else raw
