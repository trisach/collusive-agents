"""Client-side pacing to stay inside a provider's published rate limits.

Waiting BEFORE a call is cheaper and more predictable than hitting a 429 and backing off.
Minute limits (requests and tokens per minute) are met by sleeping until the last 60 s of
usage leaves room. Day limits are met by refusing to START an episode that would not fit,
so no half-finished episode wastes budget; the runner then stops and can resume later.

Usage is measured from the API's own token counts. The next call's size is estimated
from its prompt length and the average output of earlier calls.
"""

from __future__ import annotations

import collections
import dataclasses
import time

MINUTE = 60.0
DAY = 86_400.0  # Rolling 24 h window (conservative for providers with a fixed daily reset).
NOTIFY_AFTER = 5.0  # Only waits this long are reported (keeps the terminal quiet).


class DailyBudgetExhausted(RuntimeError):
    """Starting another episode would exceed a per-day limit."""


@dataclasses.dataclass(frozen=True)
class Limits:
    """Published limits; None = unlimited."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None

    def any(self) -> bool:
        """Whether any limit is set."""
        return any(dataclasses.astuple(self))


class RateLimiter:
    """Tracks (time, tokens) of every call and waits or refuses to stay within ``Limits``."""

    def __init__(self, limits: Limits):
        """Initialises the limiter.

        Args:
            limits: The limits to respect.
        """
        self.limits = limits
        self.events: collections.deque[tuple[float, int]] = collections.deque()
        self.total_waited = 0.0
        self.on_wait = None  # Optional callback(seconds) for waits of at least NOTIFY_AFTER s.
        self.outputs: collections.deque[int] = collections.deque(maxlen=50)

    def estimate(self, prompt_chars: int) -> int:
        """Expected tokens of the next call: prompt (~3.5 chars/token) + mean recent output."""
        expected_output = sum(self.outputs) / len(self.outputs) if self.outputs else 2000
        return int(prompt_chars / 3.5 + expected_output)

    def seed(self, events: list[tuple[float, int]]) -> None:
        """Adds earlier usage (e.g. from stored episodes) so day limits survive a resume."""
        for event in sorted(events):
            self.events.append(event)

    def _window(self, seconds: float) -> tuple[int, int]:
        """(requests, tokens) in the last ``seconds``."""
        now = time.time()
        while self.events and now - self.events[0][0] > DAY:
            self.events.popleft()
        recent = [tokens for at, tokens in self.events if now - at <= seconds]
        return len(recent), sum(recent)

    def check_day(self, calls: int, tokens: int) -> None:
        """Raises DailyBudgetExhausted if ``calls``/``tokens`` more would break a day limit."""
        used_calls, used_tokens = self._window(DAY)
        lim = self.limits
        if lim.requests_per_day and used_calls + calls > lim.requests_per_day:
            raise DailyBudgetExhausted(
                f"daily request budget: {used_calls}/{lim.requests_per_day} used in the last 24 h, "
                f"next episode needs ~{calls}"
            )
        if lim.tokens_per_day and used_tokens + tokens > lim.tokens_per_day:
            raise DailyBudgetExhausted(
                f"daily token budget: {used_tokens}/{lim.tokens_per_day} used in the last 24 h, "
                f"next episode needs ~{tokens}"
            )

    def wait_for_minute_window(self, estimated_tokens: int) -> float:
        """Sleeps until one more call of ``estimated_tokens`` fits the minute limits.

        A call larger than the whole token-per-minute limit is sent once the window is
        empty (it cannot fit otherwise; the provider decides).

        Returns:
            Seconds waited.
        """
        lim = self.limits
        waited = 0.0
        while True:
            calls, tokens = self._window(MINUTE)
            calls_ok = not lim.requests_per_minute or calls < lim.requests_per_minute
            tokens_ok = (
                not lim.tokens_per_minute
                or tokens + estimated_tokens <= lim.tokens_per_minute
                or calls == 0
            )
            if calls_ok and tokens_ok:
                self.total_waited += waited
                return waited
            now = time.time()
            oldest = min(at for at, _ in self.events if now - at <= MINUTE)
            delay = max(0.5, MINUTE - (now - oldest) + 0.1)
            if self.on_wait and delay >= NOTIFY_AFTER:
                self.on_wait(delay)
            time.sleep(delay)
            waited += delay

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Records a finished call's actual token usage (from the API's usage field)."""
        self.events.append((time.time(), prompt_tokens + completion_tokens))
        self.outputs.append(completion_tokens)
