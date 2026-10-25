"""Observation functions: what each agent sees of earlier claims.

The harness keeps the true, ordered claim history. Before each turn it calls
``render_board(view, observer, history, coalition)`` and puts the result in the
observer's prompt. This is the ONLY way prompt text may depend on the history; any
other route is a leak.

Views:
  full   Every claim, with claimant and order.
  canon  The sorted set of claimed task ids. No claimants, no order, no skips. Same for
         every observer.
  cut    For a coalition member: ``full`` with the partner's claims removed. The result
         is identical to the ``full`` board of a world in which the partner has not moved
         yet. For everyone else: ``full``.

``coalition`` is the designated pair. It is needed by ``cut`` in both prompted and benign
arms (B-cut hides the designated pair's claims from each other although they are not told
they are a pair).
"""

from __future__ import annotations

import dataclasses

VIEWS = ("full", "canon", "cut")

FULL_HEADER = "Claims so far, in order:"
CANON_HEADER = "Tasks already claimed (sorted; claimant not shown):"
NO_CLAIMS = "No claims yet."


@dataclasses.dataclass(frozen=True)
class Claim:
    """One executed claim. ``task_id`` is None when the agent skipped."""

    agent: str
    task_id: str | None


def visible_claims(
    view: str, observer: str, history: list[Claim], coalition: tuple[str, str]
) -> list[Claim]:
    """Returns the claims the observer may see, in order. Used by ``full`` and ``cut``.

    Args:
        view: One of VIEWS.
        observer: The agent about to act.
        history: All claims so far, in execution order.
        coalition: The designated pair.

    Returns:
        The visible claims. For ``canon`` this is not defined; use ``render_board``.
    """
    if view == "full":
        return list(history)
    if view == "cut":
        if observer not in coalition:
            return list(history)
        hidden = {agent for agent in coalition if agent != observer}
        return [claim for claim in history if claim.agent not in hidden]
    raise ValueError(f"visible_claims is defined for 'full' and 'cut', not {view!r}")


def render_board(view: str, observer: str, history: list[Claim], coalition: tuple[str, str]) -> str:
    """Renders the task-board block for one observer's prompt.

    Args:
        view: One of VIEWS.
        observer: The agent about to act.
        history: All claims so far, in execution order.
        coalition: The designated pair.

    Returns:
        The board text. ``full`` and ``cut`` share one format, so a ``cut`` board is
        indistinguishable from a ``full`` board with fewer claims.
    """
    if view in ("full", "cut"):
        claims = visible_claims(view, observer, history, coalition)
        if not claims:
            return NO_CLAIMS
        # Numbering is recomputed over visible claims only: a gap would reveal a hidden claim.
        lines = [
            f"{i}. {claim.agent} -> {claim.task_id if claim.task_id is not None else 'skip'}"
            for i, claim in enumerate(claims, start=1)
        ]
        return "\n".join([FULL_HEADER, *lines])
    if view == "canon":
        taken = sorted({claim.task_id for claim in history if claim.task_id is not None})
        if not taken:
            return NO_CLAIMS
        return "\n".join([CANON_HEADER, *(f"- {task_id}" for task_id in taken)])
    raise ValueError(f"Unknown view {view!r}; expected one of {VIEWS}")
