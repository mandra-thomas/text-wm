"""Goal-agnostic history labeling + scene-matched swapped goals (§4.3)."""
from __future__ import annotations

import random
import re

from wf.scoring.prompts import task_line_for

_INTERACT = re.compile(
    r"\b(take|move|heat|cool|clean|examine)\s+([a-zA-Z_]+)",
    re.IGNORECASE,
)


def history_goal_agnostic(
    history_text: str,
    goal_object: str,
    swapped_goal_object: str,
) -> bool:
    """True iff prefix has no take/move/heat/cool/clean/examine naming either object.

    Navigation and open/close are allowed.
    """
    goals = {goal_object.lower(), swapped_goal_object.lower()}
    for m in _INTERACT.finditer(history_text or ""):
        obj = m.group(2).lower()
        if obj in goals:
            return False
    return True


def swapped_goal(
    scene_objects: list[str],
    goal_object: str,
    receptacle_in_question: str | None = None,
    rng: random.Random | None = None,
    task_type: str = "pick_and_place_simple",
) -> str:
    """Choose an object present in the scene but not the goal; prefer same task type."""
    rng = rng or random.Random(0)
    goal = goal_object.lower()
    candidates = [
        o for o in scene_objects if o.lower() != goal and o.lower() not in ("agent", "floor")
    ]
    if not candidates:
        # last resort — still return something distinct
        return "spraybottle" if goal != "spraybottle" else "apple"
    return rng.choice(candidates)


def extract_goal_object(task_line: str) -> str:
    """Best-effort parse of goal object from task line."""
    # Your task is to: put a mug in coffeemachine.
    m = re.search(
        r"Your task is to:\s*(?:put|move|look at)\s+(?:a|an|the)\s+([a-zA-Z_]+)",
        task_line or "",
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([a-zA-Z_]+)\b", task_line or "")
    return (m.group(1).lower() if m else "mug")


def make_swapped_task_line(swapped_object: str, task_type: str = "pick_and_place_simple") -> str:
    return task_line_for(swapped_object, task_type)
