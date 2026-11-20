"""World-model prompts for TextCraft items (notes/textcraft_design.md, section 5).

The format mirrors wf/scoring/prompts.py: a system message, then one user turn per example ("History: ...",
"Action: ...", "Next observation:") with the true observation as the assistant turn, then the query turn. The
history is the transcript as the environment shows it: its first observation (the recipe list and the goal line),
then "> action" and the reply for each step. Conditions change the goal line and nothing else.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FEWSHOT = Path(__file__).resolve().parent / "data" / "fewshot.json"

# AgentGym's TextCraft instruction (agentenv/envs/textcraft.py, TextCraftEnvClient.conversation_start, 3ef9235).
AGENTGYM_INSTRUCTION = (
    'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format '
    '"craft [target object] using [input ingredients]".\nEvery round I will give you an observation, you have to '
    'respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory '
    'or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting '
    'commands.\nYour output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next '
    'action"\n\nReminder: \n1. Always specify the quantity when using "get" and "craft" commands. - Example of get: '
    'get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 '
    'golden carrot using 8 gold nugget, 1 carrot\n2. When using "get" command, do not specify whether the item '
    'comes from the inventory or the environment.\n3. You can use ONLY crafting commands provided, do not use your '
    'own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use '
    'special types of the same ingredient e.g. "dark oak planks" in the command instead.\n\n'
)
# The world model does not act, so the sentence fixing the agent's output format is left out; nothing else changes.
AGENT_FORMAT_SENTENCE = 'Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"\n'
INSTRUCTION = AGENTGYM_INSTRUCTION.replace(AGENT_FORMAT_SENTENCE, "").strip()

SYSTEM = (
    "You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next "
    "action, predict the environment's next observation exactly in TextCraft style. Output only the observation "
    "text.\n\nThe agent was given these instructions:\n" + INSTRUCTION
)
# The exploratory empty-start arm (design 13) appends this rule to the system text; the default prompt never states it.
EMPTY_START = "\n\nThe inventory is empty when the game starts; it holds only what the history's get and craft steps put in it."
YN_QUESTION = "Question: Will this action succeed? Answer Yes or No.\nAnswer:"
NEXT = "Next observation:"
CONDITIONS = ("aligned", "swapped", "neutral")
YN_ARMS = tuple(f"yn_{c}" for c in CONDITIONS)
# Prefix-only control (design section 11, A4): the same prompt as the condition, the two replies without the object
# name ("Crafted K" against "Could not find enough items to craft"; "Got 1" against "Could not find").
PREFIX_ARMS = ("prefix_aligned", "prefix_swapped")
ARMS = CONDITIONS + YN_ARMS + PREFIX_ARMS


def first_observation(recipe_lines: list[str], goal_line: str) -> str:
    """The environment's reset observation: "Crafting commands:\\n<lines>\\n\\nGoal: craft X."."""
    return "Crafting commands:\n" + "\n".join(recipe_lines) + "\n\n" + goal_line


def history_text(recipe_lines: list[str], history: list, goal_line: str) -> str:
    return first_observation(recipe_lines, goal_line) + "".join(f"\n> {a}\n{o}" for a, o in history)


def turn(recipe_lines: list[str], history: list, action: str, goal_line: str) -> str:
    return f"History:\n{history_text(recipe_lines, history, goal_line)}\n\nAction: {action}\n\n{NEXT}"


def task_line(item: dict, condition: str) -> str:
    return {"aligned": item["task_line"], "swapped": item["swapped_task_line"],
            "neutral": item["neutral_task_line"]}[condition]


@lru_cache(maxsize=4)
def _load_fewshot(path: str) -> tuple[dict, ...]:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")))


def fewshot_examples(path: Path = FEWSHOT) -> tuple[dict, ...]:
    return _load_fewshot(str(path))


def messages(item: dict, arm: str, fewshot: tuple[dict, ...] | list[dict] | None = None,
             system: str | None = None) -> list[dict[str, str]]:
    """The chat for one item under one arm: a condition or prefix_<condition> (two-candidate prediction; the prefix
    arm shares the condition's prompt) or yn_<condition> (question). `system` replaces the system text (only the
    exploratory variants of design 13 pass it)."""
    question = arm.startswith("yn_")
    condition = arm[3:] if question else arm.removeprefix("prefix_") if arm in PREFIX_ARMS else arm
    if condition not in CONDITIONS:
        raise ValueError(arm)
    shots = fewshot_examples() if fewshot is None else fewshot
    out = [{"role": "system", "content": SYSTEM if system is None else system}]
    for ex in shots:
        out.append({"role": "user", "content": turn(ex["recipe_lines"], ex["history"], ex["action"],
                                                     f"Goal: craft {ex['goal']}.")})
        out.append({"role": "assistant", "content": ex["observation"]})
    query = turn(item["recipe_lines"], item["history"], item["action"], task_line(item, condition))
    if question:
        query = query[: -len(NEXT)] + YN_QUESTION
    out.append({"role": "user", "content": query})
    return out


def candidates(item: dict, arm: str) -> tuple[str, str]:
    """(o+, o-): the success twin and the failure reply, those without the object name for a prefix arm, or "Yes"
    and "No" for a question."""
    if arm.startswith("yn_"):
        return "Yes", "No"
    if arm in PREFIX_ARMS:
        return item["prefix_candidates"]["o_plus"], item["prefix_candidates"]["o_minus"]
    return item["candidates"]["o_plus"], item["candidates"]["o_minus"]
