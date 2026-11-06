"""Goal-agnostic history labeling + scene-matched swapped goals (§4.3)."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

_INTERACT = re.compile(r"^> (?:take|move|heat|cool|clean|examine)\s+([a-zA-Z_]+)", re.IGNORECASE)


def interacted_objects(history_text: str) -> set[str]:
    """Object types named by take/move/heat/cool/clean/examine actions in the history.

    Only "> action" lines count: task lines like "put a clean mug" are not interactions.
    """
    found = (_INTERACT.match(line) for line in (history_text or "").splitlines())
    return {m.group(1).lower() for m in found if m}


def history_goal_agnostic(
    history_text: str,
    goal_object: str,
    swapped_goal_object: str,
) -> bool:
    """True iff prefix has no take/move/heat/cool/clean/examine naming either object.

    Navigation and open/close are allowed.
    """
    return not interacted_objects(history_text) & {goal_object.lower(), swapped_goal_object.lower()}


def swapped_goal(
    scene_objects: list[str],
    goal_object: str,
    receptacle_in_question: str | None = None,
    rng: random.Random | None = None,
    locations: dict[str, str] | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    """A scene object other than the goal and not in the receptacle the action concerns.

    Returns None when no such object exists; the item is then excluded, never substituted.
    """
    rng = rng or random.Random(0)
    goal = goal_object.lower()
    barred = {goal} | {o.lower() for o in (exclude or ())}
    for instance, receptacle in (locations or {}).items():
        if receptacle == receptacle_in_question:
            barred.add(instance.rsplit(" ", 1)[0].lower())
    candidates = sorted(o for o in scene_objects if o.lower() not in barred)
    return rng.choice(candidates) if candidates else None


@dataclass(frozen=True)
class TaskSpec:
    goal_object: str
    receptacle: str


# Every task template in the ALFWorld data; article and modifier never change per object.
_TASK_TEMPLATES = (
    re.compile(r"^(?:put|find) (?:a|an|some|two) (?:(?:clean|hot|cool) )?(?P<obj>[a-z]+)"
               r"(?: and put (?:it|them))? (?:in|on) (?:the )?(?P<rec>[a-z]+)\.?$"),
    re.compile(r"^(?:clean|heat|cool) some (?P<obj>[a-z]+) and put it (?:in|on) (?:the )?(?P<rec>[a-z]+)\.?$"),
    re.compile(r"^look at (?P<obj>[a-z]+) under the (?P<rec>[a-z]+)\.?$"),
    re.compile(r"^examine the (?P<obj>[a-z]+) with the (?P<rec>[a-z]+)\.?$"),
)


def parse_task_line(task_line: str) -> TaskSpec:
    """Goal object and target from the task line. Raises on an unknown template."""
    text = (task_line or "").split("Your task is to:", 1)[-1].strip().lower()
    for pattern in _TASK_TEMPLATES:
        m = pattern.match(text)
        if m:
            return TaskSpec(m["obj"], m["rec"])
    raise ValueError(f"unrecognized task line: {task_line!r}")


def extract_goal_object(task_line: str) -> str:
    return parse_task_line(task_line).goal_object


def names_object(action: str, object_name: str) -> bool:
    """True if the action names this object type (whole word, so pen is not pencil)."""
    return re.search(rf"\b{re.escape(object_name)}\b", action or "", re.IGNORECASE) is not None


def make_swapped_task_line(task_line: str, goal_object: str, swapped_object: str) -> str:
    """The real task line with only the goal object replaced, so the type is unchanged."""
    return re.sub(rf"\b{re.escape(goal_object)}\b", swapped_object, task_line, count=1)
