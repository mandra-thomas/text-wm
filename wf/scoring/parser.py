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
    "noop",                 # no effect: "Nothing happens."
    "noop_paraphrase",      # no effect, in other words ("You can't ...", "already clean", "is closed")
    "state_change",         # requested success, in the environment's own wording
    "reveal_with_goal",
    "reveal_without_goal",
    "info",
    "success_paraphrase",   # requested success, in other words ("You place ...", "is now cool")
    "process_only",         # an effect, but the action's result is not stated ("You submerge the pot 1 ...")
    "different_effect",     # an effect of another action or on another object ("You arrive at cabinet 3.")
    "unparseable",          # commits to nothing
]
# RQ1 primary definition (notes/deviations.md, 2026-11-17): failure if and only if no effect is predicted.
NO_EFFECT_CLASSES = ("noop", "noop_paraphrase")
PARTIAL_EFFECT_CLASSES = ("process_only", "different_effect")

_FAIL_ANYWHERE = re.compile(
    r"\b(?:can't|can not|cannot|couldn't|could not|unable to|not able to|fails? to|failed to|nothing happens"
    r"|not possible|impossible"
    r"|(?:do not|don't|did not|didn't) (?:see|have|find|notice|hold|carry)"
    r"|there (?:is|are|'s) no"
    r"|(?:is|are) not (?:in|on) your inventory|not in your inventory|you have no"
    r"|(?:is|are) not there|need to .+ first|must .+ (?:before|first)"
    r"|(?:is|are) already (?:clean|cleaned|cool|cooled|cold|chilled|hot|heated|warm)"
    r"|(?:have|has) already|you already (?:moved|picked|took|put|placed|opened|closed|heated|cooled|cleaned)"
    r"|(?:are|'re) already (?:at|in|on|there)|(?:do not|don't) know where)\b",
    re.IGNORECASE,
)
_FAIL_START = re.compile(
    r"^(?:there (?:is|are) not\b"
    r"|you (?:are|'re) not (?:holding|carrying) (?!anything)"
    r"|you (?:are|'re) already (?:holding|carrying)\b"
    r"|you already (?:have|hold|carry)\b"
    r"|you (?:try|attempt) to\b.*\bbut\b)",
    re.IGNORECASE,
)
# A tried action that meets a closed receptacle: "You attempt to heat ... The microwave 1 is closed."
_TRIED_CLOSED = re.compile(r"^you (?:try|attempt) to\b.*\b(?:is|are) (?:closed|locked)\b", re.IGNORECASE | re.DOTALL)
_CLOSED = re.compile(r"\b(?:is|are) (?:closed|locked)\b", re.IGNORECASE)
# Says the action did something, in words the environment would not use ("You place the candle 1 on ...").
_SUCCESS_START = re.compile(
    r"^you (?:place|wash|rinse|wet|fill|scrub|submerge|soak|warm|chill|freeze|microwave|slice|take|grab|pick"
    r"|set|switch|turn|move|open|close|put|heat|cool|clean|use)\b",
    re.IGNORECASE,
)
_RESULT = re.compile(r"\bis now (?:hot|heated|warm|cool|cooled|cold|chilled|clean|cleaned|washed|on|off|open|closed)\b",
                     re.IGNORECASE)
# The result a heat, cool or clean action must state to count as the requested success.
_STATED_RESULT = {
    "heat": re.compile(r"\b(?:heat|heats|heated|hot|warm|warmed)\b", re.IGNORECASE),
    "cool": re.compile(r"\b(?:cool|cools|cooled|cold|chill|chilled)\b", re.IGNORECASE),
    "clean": re.compile(r"\b(?:clean|cleans|cleaned|wash|washes|washed|rinse|rinses|rinsed)\b", re.IGNORECASE),
}
# "You try to heat ..." names the verb without claiming its result.
_ATTEMPT = re.compile(r"\b(?:try|tries|tried|attempt|attempts|attempted) to \w+", re.IGNORECASE)
# Acting on the object in some way: what a process-only reply has to describe.
_ACTING = re.compile(
    r"^you (?:place|put|set|wet|fill|submerge|soak|rinse|scrub|wash|dip|run|turn|use|open|close|start|begin|try|attempt)\b",
    re.IGNORECASE,
)
_MOVING = re.compile(r"^you (?:arrive at|go to|walk to|walk over to|head to|move to)\b", re.IGNORECASE)
_ACTION = re.compile(r"^(go to|take|move|open|close|heat|cool|clean|examine|use|slice|toggle)\s+(.+?)"
                     r"(?:\s+(?:from|to|with|in|on)\s+.+)?$", re.IGNORECASE)


def _action_parts(action: str | None) -> tuple[str | None, str | None]:
    """("heat", "potato") for "heat potato 1 with microwave 1"; (None, None) when unknown."""
    m = _ACTION.match((action or "").strip())
    if not m:
        return None, None
    return m[1].lower(), re.sub(r"\s+\d+$", "", m[2].strip()).lower()


def _names(obj: str | None, text: str) -> bool:
    return not obj or re.search(rf"\b{re.escape(obj)}\b", text, re.IGNORECASE) is not None


# --- prediction parsing, v4 (after the adversarial review of 2026-11-17) ---------------------------
_EFFECT_VERBS = ("go to", "open", "close", "examine", "use", "toggle", "inventory", "look")
_CONTRACTIONS = [
    (r"[’‘]", "'"), (r"[“”]", '"'), (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"), (r"n't\b", " not"),
    (r"'re\b", " are"), (r"\b(there|it|that|here)'s\b", r"\1 is"), (r"\bI'm\b", "I am"),
]
_NOOP_ANY = re.compile(r"^nothing (?:happens|happened|will happen)\.?$", re.IGNORECASE)
_MORE_FAIL = re.compile(
    r"\b(?:nothing (?:happened|will happen)"
    r"|(?:is|are) not in the world|(?:is|are) not (?:here|there)|(?:is|are) not (?:in|on|inside) (?:the|your|any)"
    r"|there is not\b|you see no\b|hands are full"
    r"|first,? you (?:need|have|must)|(?:open|go to|pick up|take) .+ first\b"
    # third audit (2026-11-18): refusals "The egg 1 is not dirty.", "... already at room temperature.",
    # "The action '...' is not valid."
    r"|(?:is|are) not dirty|already at room temperature|(?:is|are) not (?:a )?valid|invalid (?:action|command)"
    # fourth audit and thinking-mode replies (2026-11-19): "The plate 1 is not heatable.", "The cup 1 is not hot.",
    # "The fridge 1 is still closed.", "The mug 1 is already on desk 1.", "You still hold the egg 2.",
    # "The pan is not found.", "The tomato 1 is not present here.", "You are not at shelf 2.",
    # "The fridge 1 does not contain a tomato 1."
    r"|(?:is|are) not (?:heatable|coolable|cleanable)"
    r"|still (?:closed|locked)|(?:is|are) already (?:in|on|at|inside)\b|you still (?:hold|have|carry)"
    r"|(?:is|are) still (?:holding|carrying)"
    r"|(?:is|are) not (?:found|present|available|visible|located)|you are not at\b"
    r"|(?:does|do) not (?:contain|have)\b)",
    re.IGNORECASE,
)
_NO_SUCH = re.compile(r"^no [a-z]+(?: \d+)? (?:is|are)\b", re.IGNORECASE)
_ALREADY_STATE = re.compile(r"\balready (?:open|opened|closed|on|off|at|here|there|standing)\b", re.IGNORECASE)
_EMPTY = re.compile(r"\b(?:is empty|there is nothing|you see nothing|find nothing|nothing (?:in|on) (?:it|the))\b",
                    re.IGNORECASE)
# Purpose and attempt phrases name a verb without claiming its result.
_NOT_A_RESULT = re.compile(
    r"\b(?:try|tries|tried|attempt\w*|start\w*|begin\w*|begun) to \w+(?: the [a-z]+(?: \d+)?| it| up)*"
    r"|\bto (?:heat|warm|cool|chill|clean|wash|rinse)\b(?: it| up| them| the [a-z]+(?: \d+)?)*",
    re.IGNORECASE,
)
_RECEPTACLE_AFTER = re.compile(r"\b(?:to|in|into|on|onto|using|with|from) (?:the )?([a-z]+ \d+)", re.IGNORECASE)
_SUCCESS_VERBS = {
    # "You are carrying the cd 2." states the take worked (third audit, 2026-11-18)
    "take": r"(?:pick up|picked up|take|took|grab|grabbed|get|got|now have|are (?:now )?(?:holding|carrying))",
    "move": r"(?:move|moved|put|place|placed|set|drop|dropped)",
    "slice": r"(?:slice|sliced|cut)",
}
_RESULT_WORDS = {
    "heat": (r"heat|heats|heated|warm|warms|warmed", r"hot|heated|warm|warmed"),
    "cool": (r"cool|cools|cooled|chill|chills|chilled", r"cool|cooled|cold|chilled"),
    "clean": (r"clean|cleans|cleaned|wash|washes|washed|rinse|rinses|rinsed", r"clean|cleaned|washed|rinsed"),
}


def _normalise(text: str) -> str:
    body = (text or "").strip()
    for pat, rep in _CONTRACTIONS:
        body = re.sub(pat, rep, body, flags=re.IGNORECASE)
    body = re.sub(r"^\s*(?:\*\*)?(?:observation|next observation)\s*:\s*(?:\*\*)?\s*", "", body, flags=re.IGNORECASE)
    body = body.strip().strip('"').strip("*").strip()
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    while lines and lines[0].startswith(">"):  # an echoed action line before the observation
        lines = lines[1:]
    return "\n".join(lines)


def _action_full(action: str | None) -> dict:
    """verb, object name, object instance, and the receptacle the action names."""
    a = (action or "").strip().lower()
    out = {"verb": None, "obj": None, "inst": None, "recep": None}
    if a in ("inventory", "look"):
        out["verb"] = a
        return out
    m = re.match(r"^(go to|open|close|examine|use|toggle) (.+)$", a)
    if m:
        out.update(verb=m[1], inst=m[2], obj=re.sub(r"\s+\d+$", "", m[2]))
        if m[1] in ("go to", "open", "close"):
            out["recep"] = m[2]
        return out
    m = re.match(r"^(take|move|heat|cool|clean|slice) (.+?) (?:from|to|with|in|on) (.+)$", a)
    if m:
        out.update(verb=m[1], inst=m[2], obj=re.sub(r"\s+\d+$", "", m[2]), recep=m[3])
    return out


def _other_instance(obj: str | None, inst: str | None, text: str) -> bool:
    """The reply names the object only under other instance numbers ("mug 2" for "mug 1")."""
    if not obj or not inst or inst == obj:
        return False
    found = re.findall(rf"\b{re.escape(obj)} (\d+)\b", text, re.IGNORECASE)
    return bool(found) and inst.rsplit(" ", 1)[-1] not in found


def _no_effect_words(text: str) -> bool:
    return bool(_FAIL_ANYWHERE.search(text) or _FAIL_START.match(text) or _TRIED_CLOSED.match(text)
                or _MORE_FAIL.search(text) or _NO_SUCH.match(text))


def parse_prediction(text: str, goal_object: str | None = None, action: str | None = None) -> PredictionClass:
    """Classify a model's written prediction of what `action` does (RQ1, decided 2026-11-17).

    No effect: "Nothing happens." or an equivalent. With the action known, an effect is the requested
    success, a process with no stated result (heat, cool, clean), or a different effect (walking
    somewhere, another action, another object or instance, another receptacle).
    """
    body = _normalise(text)
    if not body:
        return "unparseable"
    first_line = body.split("\n")[0]
    if _NOOP.match(first_line) or _NOOP_ANY.match(first_line):
        return "noop"
    a = _action_full(action)
    verb, obj = a["verb"], a["obj"]
    sentences = [x.strip() for x in re.split(r"(?<=[.!])\s+|\n", body) if x.strip()]
    first = sentences[0]

    # Actions whose effect is an observation (arrive, open, examine): the opening sentence decides,
    # so "You arrive at countertop 1. There is no mug 1 here." is an arrival.
    if verb in ("inventory", "look"):
        templated = parse_observation(body, goal_object)
        return "info" if templated == "unparseable" else templated  # any description is the action working
    if verb in _EFFECT_VERBS:
        if _no_effect_words(first) or _ALREADY_STATE.search(first):
            return "noop_paraphrase"
        if verb == "use" and _names(obj, first) and re.search(
                r"\b(?:is|are) (?:now )?(?:on|lit|turned on|switched on)\b", first, re.IGNORECASE):
            return "success_paraphrase"  # "The desklamp 1 is on." (fourth audit, 2026-11-19)
        if _other_instance(obj, a["inst"], first):
            return "different_effect"
        if verb == "examine":
            return "info" if _names(obj, body) and not _MOVING.match(first) else "different_effect"
        templated = parse_observation(body, goal_object)
        if verb == "go to" and templated == "unparseable" and _MOVING.match(first) and _names(obj, first):
            return "success_paraphrase"  # "You walk to the cabinet 1."
        if verb == "go to" and not re.match(r"^you (?:arrive at|walk to|go to|are now at|reach)\b", first, re.IGNORECASE):
            return "different_effect" if _SUCCESS_START.match(first) else ("unparseable" if templated == "unparseable"
                                                                            else templated)
        if templated != "unparseable":
            return templated
        return "success_paraphrase" if _SUCCESS_START.match(first) or _RESULT.search(first) else "unparseable"

    # Other actions (take, move, heat, cool, clean): no-effect wording anywhere decides, unless the
    # reply opens by stating that this action worked.
    if verb and verb != "go to" and _MOVING.match(first):
        later = [x for x in sentences[1:] if (_SUCCESS_START.match(x) or _RESULT.search(x)) and _names(obj, x)]
        if not later:
            return "different_effect"  # walks there instead of doing the action
        sentences = later
        first = later[0]
        body = " ".join(later)
    claims = _NOT_A_RESULT.sub(" ", body)
    opens_with_success = False
    if verb in _SUCCESS_VERBS:
        # preparatory steps (opening, arriving, picking up before a move) may come first
        claim = next((x for x in sentences
                      if re.match(rf"^you\s+(?:[^.]*\band\s+)?(?:then\s+)?{_SUCCESS_VERBS[verb]}\b",
                                  _NOT_A_RESULT.sub(" ", x).strip(), re.IGNORECASE) and _names(obj, x)), None)
        before = sentences[:sentences.index(claim)] if claim else []
        if claim and not any(_no_effect_words(x) for x in before):
            opens_with_success, first = True, claim
    elif verb in _RESULT_WORDS:
        act, state = _RESULT_WORDS[verb]
        opens_with_success = bool(
            re.match(rf"^you (?:have |has )?(?:{act})\b", _NOT_A_RESULT.sub(" ", first).strip(), re.IGNORECASE)) and _names(obj, first)
    if not opens_with_success:
        if _no_effect_words(body):
            return "noop_paraphrase"
        if verb in ("move", "heat", "cool", "clean", "slice") and re.search(
                r"\bnot (?:carrying|holding) anything\b", body, re.IGNORECASE):
            return "noop_paraphrase"  # the action needs the object in hand
        if verb in _RESULT_WORDS and re.search(
                rf"\b{re.escape(obj)}(?: \d+)? (?:is|are) not (?:{_RESULT_WORDS[verb][1]})\b", body, re.IGNORECASE):
            return "noop_paraphrase"  # "The cup 1 is not hot." for a heat: the action's own result is denied
        if verb == "move" and re.match(r"^you are (?:still )?(?:carrying|holding) (?:the )?[a-z]+(?: \d+)?\.?$", body,
                                       re.IGNORECASE):
            return "noop_paraphrase"  # "You are carrying the book 1." for a move: it was not put down (fourth audit)
        if re.match(r"^you see nothing\.?$", body, re.IGNORECASE):
            return "noop_paraphrase"
        if verb == "take" and a["recep"] and re.search(
                rf"\b{re.escape(a['recep'])}\b[^.]*\b(?:is empty|nothing)\b|\bnothing (?:in|on) (?:it|the {re.escape(a['recep'])})\b",
                body, re.IGNORECASE):
            return "noop_paraphrase"  # the source is empty, so the object cannot be taken from it
        if _CLOSED.search(body) and not any(_SUCCESS_START.match(x) or _RESULT.search(x) for x in sentences):
            return "noop_paraphrase"  # only a closed receptacle: the action has no effect
    if not verb:
        templated = parse_observation(body, goal_object)
        if templated != "unparseable":
            return templated
        return "success_paraphrase" if _RESULT.search(body) or _SUCCESS_START.match(first) else "unparseable"
    if not _names(obj, body) or _other_instance(obj, a["inst"], body):
        return "different_effect"  # another object, or another instance of it
    recep_named = _RECEPTACLE_AFTER.findall(first)
    if a["recep"] and recep_named and a["recep"] not in [r.lower() for r in recep_named] and opens_with_success:
        return "different_effect"  # the right verb and object, but another receptacle
    if verb in _RESULT_WORDS:
        act, state = _RESULT_WORDS[verb]
        o = re.escape(obj)
        stated = (re.search(rf"\byou (?:have |has )?(?:{act})\b[^.]*\b{o}\b", claims, re.IGNORECASE)
                  or re.search(rf"\b{o}(?: \d+)? (?:is|becomes|gets|was|has been) (?:now )?(?:{state})\b", claims, re.IGNORECASE)
                  or re.search(rf"\b{o}\b[^.]*\bis now (?:{state})\b", claims, re.IGNORECASE))
        if stated:
            return "state_change" if parse_observation(body, goal_object) == "state_change" else "success_paraphrase"
        return "process_only" if _ACTING.match(first) else "different_effect"
    if opens_with_success:
        return "state_change" if parse_observation(body, goal_object) == "state_change" else "success_paraphrase"
    if verb == "take" and re.search(rf"\b{re.escape(obj)}(?: \d+)? is now in your inventory\b", body, re.IGNORECASE):
        return "success_paraphrase"
    if verb == "move" and a["recep"] and re.search(
            rf"\b{re.escape(obj)}(?: \d+)? is now (?:in|on) (?:the )?{re.escape(a['recep'])}\b", body, re.IGNORECASE):
        return "success_paraphrase"
    if _SUCCESS_START.match(first) or _MOVING.match(first):
        return "different_effect"  # a different action on the object (picking it up when asked to move it)
    templated = parse_observation(body, goal_object)
    return "unparseable" if templated in ("unparseable", "info") else templated
