"""Frozen ReAct + world-model few-shot prompts (freeze before Pilot B)."""
from __future__ import annotations

from typing import Any

# --- Few-shot world-model exemplars (set 0; additional sets for SD later) ---

WM_FEWSHOT_SET0: list[dict[str, str]] = [
    {
        "history": (
            "Your task is to: put a mug in coffeemachine.\n"
            "> go to drawer 1\n"
            "You arrive at drawer 1.\n"
            "> open drawer 1\n"
            "You open the drawer 1. In it, you see a mug 1."
        ),
        "action": "take mug 1 from drawer 1",
        "observation": "You pick up the mug 1 from the drawer 1.",
    },
    {
        "history": (
            "Your task is to: put a mug in coffeemachine.\n"
            "> go to countertop 1\n"
            "You arrive at countertop 1."
        ),
        "action": "take mug 1 from drawer 1",
        "observation": "Nothing happens.",
    },
    {
        "history": (
            "Your task is to: put an apple in fridge.\n"
            "> go to fridge 1\n"
            "You arrive at fridge 1.\n"
            "> open fridge 1\n"
            "You open the fridge 1. In it, you see a egg 1."
        ),
        "action": "take apple 1 from fridge 1",
        "observation": "Nothing happens.",
    },
]

WM_SYSTEM = (
    "You are a text world model for household embodied tasks. "
    "Given the interaction history and the next action, predict the environment's "
    "next observation exactly in ALFWorld style. Output only the observation text."
)


def world_model_prompt(
    history_text: str,
    action: str,
    fewshot: list[dict[str, str]] | None = None,
    task_line_override: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages for next-observation prediction / scoring."""
    shots = fewshot if fewshot is not None else WM_FEWSHOT_SET0
    hist = history_text
    if task_line_override:
        hist = _replace_task_line(hist, task_line_override)

    messages: list[dict[str, str]] = [{"role": "system", "content": WM_SYSTEM}]
    for ex in shots:
        user = f"History:\n{ex['history']}\n\nAction: {ex['action']}\n\nNext observation:"
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ex["observation"]})
    user = f"History:\n{hist}\n\nAction: {action}\n\nNext observation:"
    messages.append({"role": "user", "content": user})
    return messages


def _replace_task_line(history_text: str, task_line: str) -> str:
    lines = history_text.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("Your task is to:"):
            out.append(task_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, task_line)
    return "\n".join(out)


REACT_SYSTEM = (
    "You are a household robot agent playing ALFWorld. "
    "At each step, reason briefly then output a single admissible action. "
    "Format:\nThought: <brief>\nAction: <one action>\n"
    "Use only actions that appear in the admissible list when provided. "
    "Do not invent observations."
)

REACT_FEWSHOT = (
    "Example:\n"
    "Observation: You arrive at drawer 1. Your task is to: put a mug in coffeemachine.\n"
    "Thought: I should open the drawer to look for the mug.\n"
    "Action: open drawer 1\n"
)


def agent_react_messages(
    observation: str,
    history: list[dict[str, str]] | None = None,
    foresight_prediction: str | None = None,
    admissible: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build ReAct chat messages. Foresight shown as Expected outcome before decision."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": REACT_FEWSHOT},
    ]
    body_parts: list[str] = []
    if history:
        for turn in history:
            if "observation" in turn:
                body_parts.append(f"Observation: {turn['observation']}")
            if turn.get("expected_outcome"):
                body_parts.append(f"Expected outcome: {turn['expected_outcome']}")
            if "action" in turn:
                body_parts.append(f"Action: {turn['action']}")
    body_parts.append(f"Observation: {observation}")
    if foresight_prediction is not None:
        body_parts.append(f"Expected outcome: {foresight_prediction}")
    if admissible:
        body_parts.append("Admissible actions:\n- " + "\n- ".join(admissible[:40]))
    body_parts.append("Respond with Thought and Action.")
    messages.append({"role": "user", "content": "\n".join(body_parts)})
    return messages


def parse_react_action(text: str) -> str:
    """Extract Action: line from model output."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            return line.split(":", 1)[1].strip()
    # fallback: last non-empty line
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else "look"


def task_line_for(goal_object: str, task_type: str = "pick_and_place_simple") -> str:
    """Reconstruct first-observation task line in stack template."""
    # Word2World renames put -> move in actions, but task text often still says put
    receptacle = {
        "pick_and_place_simple": "coffeemachine",
        "pick_clean_then_place_in_recep": "shelvingunit",
        "pick_heat_then_place_in_recep": "countertop",
        "pick_cool_then_place_in_recep": "countertop",
        "look_at_obj_in_light": "desklamp",
        "pick_two_obj_and_place": "cabinet",
    }.get(task_type, "coffeemachine")
    article = "an" if goal_object[:1].lower() in "aeiou" else "a"
    if task_type == "look_at_obj_in_light":
        return f"Your task is to: look at {article} {goal_object} under the {receptacle}."
    return f"Your task is to: put {article} {goal_object} in {receptacle}."


def condition_task_line(
    condition: str,
    aligned_task: str,
    swapped_task: str | None = None,
) -> str | None:
    """Return task line override for scoring condition, or None to keep history."""
    if condition == "aligned":
        return aligned_task
    if condition == "swapped":
        return swapped_task
    if condition == "absent":
        return "Your task is to: explore the room."
    if condition in ("committed", "candidate"):
        return aligned_task
    return aligned_task


def world_model_ft_prompt(history_text: str, action: str) -> Any:
    """Prompt protocol aligned with Word2World interact_with_world_model/run.py."""
    # Keep a simple string prompt used by FT checkpoints
    return (
        f"<<SYS>>\nYou are a world model. Predict the next observation.\n<</SYS>>\n"
        f"History:\n{history_text}\nAction: {action}\nObservation:"
    )
