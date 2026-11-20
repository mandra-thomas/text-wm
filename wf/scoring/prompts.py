"""Prompts: the ReAct agent's prompt and the world model's few-shot prompt, with the three frozen example sets.

Set 1 (WM_FEWSHOT_SET0) was frozen before Pilot B and is the preregistered primary set; sets 2 and 3 were frozen
on 2026-11-18 (notes/deviations.md, "Example sets 2 and 3 frozen"). Only the task line differs between a
prompt's goal conditions (`_replace_task_line`).
"""
from __future__ import annotations

import re

# --- Few-shot world-model exemplars: set 1 (named WM_FEWSHOT_SET0 for history), the primary set ---
# Every string is logged server output from t0_smoke on train games (job 6832744), so no
# observation shown to a model is hand-written. Games, in order: train/700, train/270, train/2332.
# Histories are the steps just before the action. One success and two failures, as before.

WM_FEWSHOT_SET0: list[dict[str, str]] = [
    {
        "history": (
            "Your task is to: put a soapbottle in toilet.\n"
            "> go to cabinet 1\n"
            "You arrive at cabinet 1. The cabinet 1 is closed.\n"
            "> open cabinet 1\n"
            "You open the cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 1."
        ),
        "action": "take spraybottle 1 from cabinet 1",
        "observation": "You pick up the spraybottle 1 from the cabinet 1.",
    },
    {
        "history": (
            "Your task is to: put some candle on cabinet.\n"
            "> go to toilet 1\n"
            "You arrive at toilet 1. On the toilet 1, you see a candle 2, and a candle 1.\n"
            "> take candle 1 from toilet 1\n"
            "You pick up the candle 1 from the toilet 1."
        ),
        "action": "move candle 1 to cabinet 1",
        "observation": "Nothing happens.",
    },
    {
        "history": (
            "Your task is to: put two soapbottle in garbagecan.\n"
            "> move soapbottle 2 to garbagecan 1\n"
            "You move the soapbottle 2 to the garbagecan 1.\n"
            "> go to countertop 1\n"
            "You arrive at countertop 1. On the countertop 1, you see a handtowel 1, a soapbar 1, "
            "a spraybottle 3, and a spraybottle 1."
        ),
        "action": "take soapbottle 1 from countertop 1",
        "observation": "Nothing happens.",
    },
]

# Set 2, redrawn 2026-11-18 by rule and seed (scripts/draw_fewshot_sets.py); games train/1213, train/1183, train/403, train/2128.
# Two successes (one a take) and two failures, one goal-relevant of each; see notes/fewshot_sets_review.md.
WM_FEWSHOT_SET2: list[dict[str, str]] = [
    {
        "history": "Your task is to: clean some soapbar and put it in cabinet.\n> go to sinkbasin 2\nYou arrive at sinkbasin 2. On the sinkbasin 2, you see nothing.\n> go to countertop 1\nYou arrive at countertop 1. On the countertop 1, you see a candle 1, and a soapbar 2.",
        "action": "take soapbar 2 from countertop 1",
        "observation": "You pick up the soapbar 2 from the countertop 1.",
    },
    {
        "history": "Your task is to: clean some potato and put it in microwave.\n> go to fridge 1\nYou arrive at fridge 1. The fridge 1 is closed.\n> open fridge 1\nYou open the fridge 1. The fridge 1 is open. In it, you see a cup 1, a egg 1, and a potato 1.",
        "action": "take apple 1 from fridge 1",
        "observation": "Nothing happens.",
    },
    {
        "history": "Your task is to: put some creditcard on armchair.\n> go to sidetable 2\nYou arrive at sidetable 2. On the sidetable 2, you see a houseplant 1, a keychain 1, and a newspaper 2.\n> go to drawer 1\nYou arrive at drawer 1. The drawer 1 is closed.",
        "action": "take creditcard 1 from sofa 1",
        "observation": "Nothing happens.",
    },
    {
        "history": "Your task is to: find two creditcard and put them in coffeetable.\n> open cabinet 4\nYou open the cabinet 4. The cabinet 4 is open. In it, you see nothing.\n> go to drawer 1\nYou arrive at drawer 1. The drawer 1 is closed.",
        "action": "open drawer 1",
        "observation": "You open the drawer 1. The drawer 1 is open. In it, you see a watch 1.",
    },
]

# Set 3, drawn 2026-11-17 and approved as drafted; games train/49, train/2165, train/336, train/1670.
WM_FEWSHOT_SET3: list[dict[str, str]] = [
    {
        "history": "Your task is to: examine the cd with the desklamp.\n> go to desk 1\nYou arrive at desk 1. On the desk 1, you see a alarmclock 1, a book 1, a bowl 2, a cd 3, a cd 2, a cd 1, a keychain 3, a keychain 2, a pencil 3, and a pencil 2.\n> take desklamp from desk 1\nNothing happens.",
        "action": "take cd 1 from desk 1",
        "observation": "You pick up the cd 1 from the desk 1.",
    },
    {
        "history": "Your task is to: find two handtowel and put them in toilet.\n> go to handtowelholder 2\nYou arrive at handtowelholder 2. On the handtowelholder 2, you see a handtowel 2.\n> go to toilet 1\nYou arrive at toilet 1. On the toilet 1, you see a candle 2, a dishsponge 2, and a soapbottle 1.",
        "action": "take handtowel 1 from toilet 1",
        "observation": "Nothing happens.",
    },
    {
        "history": "Your task is to: put a cellphone in armchair.\n> open drawer 1\nYou open the drawer 1. The drawer 1 is open. In it, you see nothing.\n> go to drawer 2\nYou arrive at drawer 2. The drawer 2 is closed.",
        "action": "go to drawer 11",
        "observation": "You arrive at drawer 11. The drawer 11 is closed.",
    },
    {
        "history": "Your task is to: cool some tomato and put it in countertop.\n> go to fridge 1\nYou arrive at fridge 1. The fridge 1 is closed.\n> open fridge 1\nYou open the fridge 1. The fridge 1 is open. In it, you see a apple 2, a apple 1, a bread 2, a egg 1, a mug 1, a pan 1, a plate 3, and a potato 2.",
        "action": "take potato 2 from cabinet 12",
        "observation": "Nothing happens.",
    },
]

# The three prompt-example sets; set 1 (WM_FEWSHOT_SET0) is primary, as filed.
WM_FEWSHOT_SETS = {1: WM_FEWSHOT_SET0, 2: WM_FEWSHOT_SET2, 3: WM_FEWSHOT_SET3}
# Each example's game, in order, so the full-state prompt can show that game's hidden state too.
WM_FEWSHOT_GAMES = {
    1: ("train/700", "train/270", "train/2332"),
    2: ("train/1213", "train/1183", "train/403", "train/2128"),
    3: ("train/49", "train/2165", "train/336", "train/1670"),
}

_OBJECTS_HEADER = "=== Objects on Receptacles ==="


def hidden_block(context: str) -> str:
    """The starting layout from a Word2World context, without the first observation or the task line.

    The context also carries the task line (under "User Environment Information"), so passing it whole
    would show the true goal in every condition, the swapped one included.
    """
    start = context.find(_OBJECTS_HEADER)
    if start < 0:
        raise ValueError("hidden state has no objects section")
    block = context[start:].split("\n# ", 1)[0].strip()
    if "Your task is to" in block:
        raise ValueError("hidden state block carries a task line")
    return block

WM_SYSTEM = (
    "You are a text world model for household embodied tasks. "
    "Given the interaction history and the next action, predict the environment's "
    "next observation exactly in ALFWorld style. Output only the observation text."
)


HIDDEN_INTRO = "Hidden state (where every object is at the start of the episode; the agent cannot see this):"


def world_model_prompt(
    history_text: str,
    action: str,
    fewshot: list[dict[str, str]] | None = None,
    task_line_override: str | None = None,
    hidden: str | None = None,
    fewshot_hidden: list[str] | None = None,
) -> list[dict[str, str]]:
    """Chat messages for next-observation prediction / scoring.

    With `hidden` (the full-state condition) every turn, the examples included, opens with its game's
    starting layout, so the examples show the format the query uses.
    """
    shots = fewshot if fewshot is not None else WM_FEWSHOT_SET0
    if hidden is not None and (fewshot_hidden is None or len(fewshot_hidden) != len(shots)):
        raise ValueError("the full-state prompt needs a hidden state for every example")
    hist = history_text
    if task_line_override:
        hist = _replace_task_line(hist, task_line_override)

    def turn(h: str, act: str, layout: str | None) -> str:
        head = f"{HIDDEN_INTRO}\n{layout}\n\n" if layout is not None else ""
        return f"{head}History:\n{h}\n\nAction: {act}\n\nNext observation:"

    messages: list[dict[str, str]] = [{"role": "system", "content": WM_SYSTEM}]
    for i, ex in enumerate(shots):
        messages.append({"role": "user", "content": turn(ex["history"], ex["action"],
                                                         fewshot_hidden[i] if hidden is not None else None)})
        messages.append({"role": "assistant", "content": ex["observation"]})
    messages.append({"role": "user", "content": turn(hist, action, hidden)})
    return messages


def _replace_task_line(history_text: str, task_line: str) -> str:
    """Swap the task line in place, leaving every other byte of the history unchanged.

    Exactly one task line must be replaced. Until 2026-11-20 a text without one got the new line prepended and
    kept nothing to replace, and a text with two had both replaced; either would show a prompt other than the
    one intended (in the first case the swapped prompt could still carry the real task), so both now raise.
    """
    replaced, n = re.subn(r"^Your task is to:.*$", lambda _: task_line, history_text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"expected exactly one 'Your task is to:' line to replace, found {n}")
    return replaced


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


def parse_react_action(text: str) -> str | None:
    """The action on the agent's explicit "Action:" line (or the line after a bare "Action:").

    No such line is a format error, returned as None, never a guess: taking the last line once
    turned "Expected outcome: ..." copied from retained history into a bogus invalid action.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        bare = line.lstrip("*#> ").replace("**", "")
        if bare.lower().startswith("action:"):
            action = bare.split(":", 1)[1].strip()
            if not action and i + 1 < len(lines):
                action = lines[i + 1]
            action = action.strip("`'\" *").rstrip(".")
            return action or None
    return None


def condition_task_line(
    condition: str,
    aligned_task: str,
    swapped_task: str | None = None,
) -> str | None:
    """Return task line override for scoring condition, or None to keep history."""
    if condition == "aligned":
        return aligned_task
    if condition == "swapped":
        if not swapped_task:
            raise ValueError("swapped condition needs a swapped goal")  # would quietly score the aligned goal
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
