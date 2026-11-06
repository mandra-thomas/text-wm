"""Frozen ReAct + world-model few-shot prompts (freeze before Pilot B)."""
from __future__ import annotations

import re

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
    """Swap the task line in place, leaving every other byte of the history unchanged."""
    replaced, n = re.subn(r"^Your task is to:.*$", lambda _: task_line, history_text, flags=re.MULTILINE)
    return replaced if n else f"{task_line}\n{history_text}"


REACT_SYSTEM = (
    "You are a household robot agent playing ALFWorld. "
    "At each step, reason briefly then output a single admissible action. "
    "Format:\nThought: <brief>\nAction: <one action>\n"
    "Use only actions that appear in the admissible list when provided. "
    "Do not invent observations. "
    "An 'Expected outcome:' line predicts what would happen if the action on the "
    "'Proposed action:' line above it were taken."
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
    proposal: str | None = None,
) -> list[dict[str, str]]:
    """Build ReAct chat messages. Foresight shown as Expected outcome before decision."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": REACT_FEWSHOT},
    ]
    body_parts: list[str] = []
    for turn in history or []:
        body_parts.append(f"Observation: {turn['observation']}")
        if turn.get("expected_outcome"):
            body_parts.append(f"Proposed action: {turn['proposal']}")
            body_parts.append(f"Expected outcome: {turn['expected_outcome']}")
        body_parts.append(f"Action: {turn['action']}")
    body_parts.append(f"Observation: {observation}")
    if foresight_prediction is not None:
        body_parts.append(f"Proposed action: {proposal}")
        body_parts.append(f"Expected outcome: {foresight_prediction}")
    if admissible:
        body_parts.append("Admissible actions:\n- " + "\n- ".join(admissible))
    body_parts.append("Respond with Thought and Action.")
    messages.append({"role": "user", "content": "\n".join(body_parts)})
    return messages


def parse_react_action(text: str) -> str:
    """Extract the action from model output; it may sit on the line after "Action:"."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    action = lines[-1] if lines else "look"
    for i, line in enumerate(lines):
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
            if not action and i + 1 < len(lines):
                action = lines[i + 1]
            break
    return action.strip("`'\" ").rstrip(".") or "look"


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


def world_model_ft_messages(
    system_context: str,
    history_pairs: list[list[str]],
    action: str,
) -> list[dict[str, str]]:
    """Chat format the Word2World checkpoints were trained on (interact_with_world_model/run.py).

    The system message carries the hidden state and first observation; each action is a user
    turn and each observation an assistant turn.
    """
    messages = [{"role": "system", "content": system_context}]
    for past_action, observation in history_pairs:
        messages.append({"role": "user", "content": past_action})
        messages.append({"role": "assistant", "content": observation})
    messages.append({"role": "user", "content": action})
    return messages
