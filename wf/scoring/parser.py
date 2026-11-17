"""Templated ALFWorld observation parser (PILOT_PROTOCOL §4.5)."""
from __future__ import annotations

import re
from typing import Literal

OutcomeClass = Literal[
    "noop",
    "state_change",
    "reveal_with_goal",
    "reveal_without_goal",
    "info",
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
# Valid but informational: inventory, look with nothing nearby, examine, help.
_INFO = re.compile(
    r"^(You are carrying|You are not carrying|You are facing|You are in the middle of a room"
    r"|Available commands|This is a|There's nothing|The .+ is (?:closed|open)\.)",
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
    if _INFO.match(first):
        return "info"
    return "unparseable"


def is_success_obs(obs: str, goal_object: str | None = None) -> bool:
    cls = parse_observation(obs, goal_object)
    return cls in ("state_change", "reveal_with_goal", "reveal_without_goal", "info")


def action_appears_valid(obs: str) -> bool:
    """True if observation indicates the action did something (not noop)."""
    return parse_observation(obs) != "noop" and parse_observation(obs) != "unparseable"


# --- model-written predictions ---------------------------------------------------------------------
# The environment always says "Nothing happens." when an action fails, but a model predicting a failure
# often writes it in its own words ("You can't find a tomato 1 in the fridge 1."). Those used to come out
# `unparseable` and were dropped from A, which removed failure predictions selectively (2026-11-17).
# `parse_prediction` is for model text only; environment observations keep `parse_observation`.
PredictionClass = Literal[
    "noop",
    "noop_paraphrase",
    "state_change",
    "reveal_with_goal",
    "reveal_without_goal",
    "info",
    "success_paraphrase",
    "unparseable",
]

_FAIL_ANYWHERE = re.compile(
    r"\b(?:can't|can not|cannot|couldn't|could not|unable to|not able to|fails? to|failed to|nothing happens"
    r"|not possible|impossible"
    r"|(?:do not|don't|did not|didn't) (?:see|have|find|notice|hold|carry)"
    r"|there (?:is|are|'s) no"
    r"|(?:is|are) not (?:in|on) your inventory|not in your inventory|you have no"
    r"|(?:is|are) not there|need to .+ first)\b",
    re.IGNORECASE,
)
_FAIL_START = re.compile(
    r"^(?:there (?:is|are) not\b"
    r"|you (?:are|'re) not (?:holding|carrying) (?!anything)"
    r"|you (?:are|'re) already (?:holding|carrying)\b"
    r"|you already (?:have|hold|carry)\b"
    r"|you try to\b.*\bbut\b)",
    re.IGNORECASE,
)
# Says the action did something, in words the environment would not use ("You place the candle 1 on ...").
_SUCCESS_START = re.compile(
    r"^you (?:place|wash|rinse|wet|fill|scrub|submerge|soak|warm|chill|freeze|microwave|slice|take|grab|pick"
    r"|set|switch|turn|move|open|close|put|heat|cool|clean|use)\b",
    re.IGNORECASE,
)
_RESULT = re.compile(r"\bis now (?:hot|heated|warm|cool|cooled|cold|chilled|clean|cleaned|washed|on|off|open|closed)\b",
                     re.IGNORECASE)
_MOVING = re.compile(r"^you (?:arrive at|go to|walk to|walk over to|head to|move to)\b", re.IGNORECASE)
_ACTION = re.compile(r"^(go to|take|move|open|close|heat|cool|clean|examine|use|slice|toggle)\s+(.+?)"
                     r"(?:\s+(?:from|to|with|in|on)\s+.+)?$", re.IGNORECASE)


def _action_parts(action: str | None) -> tuple[str | None, str | None]:
    """("heat", "potato") for "heat potato 1 with microwave 1"; (None, None) when unknown."""
    m = _ACTION.match((action or "").strip())
    if not m:
        return None, None
    return m[1].lower(), re.sub(r"\s+\d+$", "", m[2].strip()).lower()


def parse_prediction(text: str, goal_object: str | None = None, action: str | None = None) -> PredictionClass:
    """Classify a model's written prediction of what `action` does.

    Failure wording wins over everything but the exact "Nothing happens.". With the action known,
    a reply that describes a different action (walking somewhere, another object) is `unparseable`,
    so it is never counted as a success of the action asked about.
    """
    body = (text or "").strip()
    if not body:
        return "unparseable"
    first = body.split("\n")[0].strip()
    if _NOOP.match(first) or _NOOP.match(body):
        return "noop"
    if _FAIL_ANYWHERE.search(first) or _FAIL_START.match(first):
        return "noop_paraphrase"
    verb, obj = _action_parts(action)
    if verb and verb != "go to":
        if _MOVING.match(first):
            # walks somewhere first; counts only if a later sentence does the action to the object
            later = [x.strip() for x in re.split(r"(?<=\.)\s+", body)[1:]]
            done = [x for x in later if (_SUCCESS_START.match(x) or _RESULT.search(x))
                    and (not obj or re.search(rf"\b{re.escape(obj)}\b", x, re.IGNORECASE))]
            return "success_paraphrase" if done else "unparseable"
        if obj and not re.search(rf"\b{re.escape(obj)}\b", body, re.IGNORECASE):
            return "unparseable"  # about some other object
        if verb == "examine":
            return "info"  # any description of the object, with no failure wording, says examining worked
    templated = parse_observation(body, goal_object)
    if templated == "info" and verb in ("take", "move", "heat", "cool", "clean", "use", "slice"):
        templated = "unparseable"  # a state description is not the action's result
    if templated != "unparseable":
        return templated
    if _RESULT.search(body) or _SUCCESS_START.match(first):
        return "success_paraphrase"
    return "unparseable"
