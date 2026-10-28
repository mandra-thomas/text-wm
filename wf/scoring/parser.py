"""Templated ALFWorld observation parser (PILOT_PROTOCOL §4.5)."""
from __future__ import annotations

import re
from typing import Literal

OutcomeClass = Literal[
    "noop",
    "state_change",
    "reveal_with_goal",
    "reveal_without_goal",
    "unparseable",
]

_NOOP = re.compile(r"^Nothing happens\.$")
_STATE = re.compile(
    r"^You (pick up|put|move|open|close|heat|cool|clean|turn on|turn off)\b",
    re.IGNORECASE,
)
_REVEAL = re.compile(
    r"(In it, you see|On the .+ you see|you see a |you see an )",
    re.IGNORECASE,
)


def parse_observation(obs: str, goal_object: str | None = None) -> OutcomeClass:
    text = (obs or "").strip()
    if not text:
        return "unparseable"
    # Use first line for primary classification when multi-line
    first = text.split("\n")[0].strip()
    if _NOOP.match(first) or _NOOP.match(text):
        return "noop"
    # Prefer reveal_* when contents are shown (e.g. "You open ... In it, you see ...")
    if _REVEAL.search(text):
        if goal_object:
            pat = re.compile(rf"\b{re.escape(goal_object)}\s+\d+\b", re.IGNORECASE)
            if pat.search(text):
                return "reveal_with_goal"
            return "reveal_without_goal"
        return "reveal_without_goal"
    if _STATE.match(first):
        return "state_change"
    # Arrival / look narration often still informative but not in core classes
    if re.match(r"^You arrive at\b", first, re.IGNORECASE):
        return "state_change"
    return "unparseable"


def is_success_obs(obs: str, goal_object: str | None = None) -> bool:
    cls = parse_observation(obs, goal_object)
    return cls in ("state_change", "reveal_with_goal", "reveal_without_goal") and cls != "noop"


def action_appears_valid(obs: str) -> bool:
    """True if observation indicates the action did something (not noop)."""
    return parse_observation(obs) != "noop" and parse_observation(obs) != "unparseable"
