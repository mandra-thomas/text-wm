"""TextCraft round 2: prompts, arms, models and the item hash (notes/textcraft_r2_design.md).

Round 2 scores, under the clean world-model system text (prompts.CLEAN_SYSTEM), five models on three kinds of record
in wf/textcraft/data/r2_items.jsonl (built by wf/textcraft/r2_items.py):

- *main*: the frozen main (failure) and success items of the confirmatory build, unchanged, plus the two-item goal
  line "Goal: craft A and B." (the item's goal pair, sorted). Arms: the aligned, swapped, neutral and two-item lines,
  each as the two-candidate score and as the yes/no question. Main failures at 1 to 8 history steps also carry the
  padding arms: the history with 1 or 8 `inventory` look-ups ("Inventory: You are not carrying anything.") put
  before its first step, under the aligned and swapped lines.
- *parent*: new goal-role items (an action crafting the goal G) from the frozen builder's pool, with a parent goal Z
  whose recipe consumes G. Arms: the aligned line on the item's own list, and the aligned ("craft G"), parent
  ("craft Z"), swapped ("craft P") and two-item lines on the extended list (the item's list with Z's tree lines
  added); and the yes/no question under the neutral line on the item's own list, for goal-crafting competence.

An arm name is `[yn_][padK_|ext_]<line>`; every arm of a record changes only what its name says (tests and
r2_items.prompt_problems check this).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from wf.families import MODELS as FAMILY_MODELS
from wf.families import REFERENCE, ModelSpec, fold_system
from wf.textcraft import prompts as P

SYSTEM = P.CLEAN_SYSTEM
# The round-2 yes/no question (amendment 14.Q, 2026-11-21, before any round-2 score). Round 1's question ("Will this
# action succeed?", prompts.YN_QUESTION) is not tied to the environment's reply and can be read as "will this action
# help the task succeed". This one names TextCraft's own reply forms, as ALFWorld's question names "Nothing happens."
# (wf/rq1v.py): a success replies "Crafted K minecraft:..." (a craft) or "Got N ..." (a get); a failure replies
# "Could not find ..." ("Could not find enough items to craft ...", or "Could not find X" for a get). Round-2 yes/no
# scores are therefore a different quantity from round 1's.
YN_QUESTION = ('Question: will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that '
               'it failed ("Could not find ...")? Answer Yes if it succeeds and No if it fails.\nAnswer:')
LINES = ("aligned", "swapped", "neutral", "both", "parent")
LINE_FIELDS = {"aligned": "task_line", "swapped": "swapped_task_line", "neutral": "neutral_task_line",
               "both": "both_task_line", "parent": "parent_task_line"}
PAD_STEP = ("inventory", "Inventory: You are not carrying anything.")  # the environment's reply on an empty inventory
PAD_LEVELS = (1, 8)
PAD_BINS = ("1-3", "4-8")
MAIN_ARMS = ("aligned", "swapped", "neutral", "both", "yn_aligned", "yn_swapped", "yn_neutral", "yn_both")
BOTH_ARMS = ("both", "yn_both")
PAD_ARMS = tuple(f"pad{k}_{line}" for k in PAD_LEVELS for line in ("aligned", "swapped"))
# Amendments before freeze (2026-11-21, design section 14): `yn_neutral` on the item's own list gives the parent
# population its own competence data (one forward pass); `ext_both`, the two-item line on the extended list, splits the
# within-item penalty exactly into ext_aligned - ext_both and ext_both - ext_parent (descriptive). A parent record of
# a pair with no two-item line has no ext_both arm.
PARENT_ARMS = ("aligned", "ext_aligned", "ext_parent", "ext_swapped", "ext_both", "yn_neutral")

# The five models, each with its own chat template at its pinned revision (wf.constants.REVISIONS). Gemma 2 has no
# system role: the system text opens the first user turn (families.fold_system). Mistral's template places the system
# text in the last user turn, as shipped. Llama 3.2 gets the fixed date its template falls back to.
MODELS: dict[str, ModelSpec] = {
    "qwen3-8b": REFERENCE["qwen3-8b"],
    "llama3.1-8b": REFERENCE["llama3.1-8b"],
    "gemma2-9b-it": FAMILY_MODELS["gemma2-9b-it"],
    "mistral-7b-v0.3": FAMILY_MODELS["mistral-7b-v0.3"],
    "llama3.2-3b": FAMILY_MODELS["llama3.2-3b"],
}
TAGS = tuple(MODELS)


def two_item_line(goal: str, partner: str) -> str | None:
    """The two-item goal line: both goals of the pair, in sorted order, so the two members of a couple share it.
    None when a goal's name contains the word "and" ("flint and steel"): the line could not be read as two items."""
    if " and " in f" {goal} " or " and " in f" {partner} ":
        return None
    a, b = sorted((goal, partner))
    return f"Goal: craft {a} and {b}."


def two_item_partner_done(rec: dict) -> bool:
    """User point 1 (2026-11-21): under "craft A and B", crafting one item would complete the whole objective if the
    other named item were already made or held. True when either named goal appears as a history step's product or
    is held after the history (text only). Such records leave the naming and completion contrasts; by construction
    there are none (the walk never crafts a goal, and goals cannot be fetched), which verify checks."""
    from wf.textcraft import checker as K

    held = K.account(rec["history"]).held
    made = {K.step_object(o) for _a, o in rec["history"]}
    return any(g in made or held.get(g, 0) > 0 for g in (rec["goal"], rec["partner"]))


def parse_arm(arm: str) -> tuple[bool, int, bool, str]:
    """(question, padding steps, extended list, line) of an arm name."""
    question = arm.startswith("yn_")
    rest = arm[3:] if question else arm
    pad, ext = 0, False
    if rest.startswith("pad"):
        head, rest = rest.split("_", 1)
        pad = int(head[3:])
        if pad not in PAD_LEVELS:
            raise ValueError(arm)
    elif rest.startswith("ext_"):
        ext, rest = True, rest[4:]
    if rest not in LINES:
        raise ValueError(arm)
    return question, pad, ext, rest


def shown(rec: dict, arm: str) -> tuple[list[str], list, str]:
    """(recipe lines, history, goal line) the query of `arm` shows."""
    _q, pad, ext, line = parse_arm(arm)
    lines = rec["ext_recipe_lines"] if ext else rec["recipe_lines"]
    history = [list(PAD_STEP) for _ in range(pad)] + [list(h) for h in rec["history"]]
    return lines, history, rec[LINE_FIELDS[line]]


def messages(rec: dict, arm: str, shots, system: str = SYSTEM) -> list[dict[str, str]]:
    """The chat for one record and arm: the clean system text, the three verified prompt examples, the query."""
    question = parse_arm(arm)[0]
    lines, history, goal_line = shown(rec, arm)
    out = [{"role": "system", "content": system}]
    for ex in shots:
        out.append({"role": "user", "content": P.turn(ex["recipe_lines"], ex["history"], ex["action"],
                                                       f"Goal: craft {ex['goal']}.")})
        out.append({"role": "assistant", "content": ex["observation"]})
    query = P.turn(lines, history, rec["action"], goal_line)
    if question:
        query = query[: -len(P.NEXT)] + YN_QUESTION
    out.append({"role": "user", "content": query})
    return out


def model_messages(spec: ModelSpec, rec: dict, arm: str, shots) -> list[dict[str, str]]:
    """messages() as the model reads them: folded into the first user turn for a template with no system role."""
    m = messages(rec, arm, shots)
    return fold_system(m) if spec.system == "fold" else m


def candidates(rec: dict, arm: str) -> tuple[str, str]:
    if parse_arm(arm)[0]:
        return "Yes", "No"
    return rec["candidates"]["o_plus"], rec["candidates"]["o_minus"]


HASHED = ("kind", "recipe_lines", "ext_recipe_lines", "history", "action", "task_line", "swapped_task_line",
          "neutral_task_line", "both_task_line", "parent_task_line", "candidates", "arms")


def item_hash(rec: dict, shots) -> str:
    """Everything any arm of the record shows or scores, and the prompt context (system text, question, examples,
    padding step), so a saved score can never attach to a changed record or prompt."""
    fields = {k: rec.get(k) for k in HASHED}
    context = {"system": SYSTEM, "question": YN_QUESTION, "next": P.NEXT, "fewshot": list(shots),
               "pad_step": list(PAD_STEP)}
    return hashlib.sha256(json.dumps({"item": fields, "context": context}, sort_keys=True).encode()).hexdigest()[:16]


def jobs(records: list[dict]) -> list[tuple[dict, str]]:
    """Every (record, arm) a model scores, in file order."""
    return [(r, a) for r in records for a in r["arms"]]


def spec_record(tag: str) -> dict[str, Any]:
    return MODELS[tag].record()
