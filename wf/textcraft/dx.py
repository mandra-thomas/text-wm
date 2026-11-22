"""Exploratory diagnostics of the TextCraft Llama result (notes/textcraft_llama_dx.md). Not part of the frozen design.

`textcraft_dx_zeroshot`: the verified prompt without its in-context examples (system text, then the item's turn),
both models, the aligned, swapped and neutral lines, two-candidate scores, on seeded whole couples of the main and
success populations. Records have the fields of textcraft_score's, with `variant` "zeroshot".
`textcraft_dx_objectgoal`: under the verified prompt, the aligned prompt with its goal line replaced by one naming the
action's own object ("Goal: craft Y."), as the two-candidate score (`object_goal`) and the yes/no question
(`yn_object_goal`), both models, on seeded whole ingredient couples. Read against the verified aligned and swapped scores.
`textcraft_dx_generate`: under the verified prompt (examples included), greedy free generation and the most probable
first tokens, Llama only by default, on seeded whole couples of given roles. One record per item and line.

`textcraft_dx_prompts`: three system-text and history variants (PROMPT_VARIANTS: a world-model-only system text, an
explicit inventory step before the action, both), aligned, swapped and neutral lines, both models; conditions are
"<variant>|<line>".

All save every record as it is made (scoring.jsonl, generations.jsonl) and, with --resume, skip what is saved.
"""
from __future__ import annotations

import math
import random
from typing import Any

from wf import constants as C
from wf.runs import _init_run
from wf.scoring.logodds import LogoddsScorer
from wf.textcraft import prompts as P
from wf.textcraft.checker import account
from wf.textcraft.score import MODELS, item_hash, load_items, verified_files
from wf.util import append_jsonl, read_jsonl, require_filed, write_status

CONDITIONS = ("aligned", "swapped", "neutral")


def sample_couples(items: list[dict], population: str, n: int, tag: str, role: str | None = None) -> list[dict]:
    """n seeded whole couples of a population (optionally of one role; both members of a couple share it), in the
    items file's order."""
    couples = sorted({t["couple_id"] for t in items if t["population"] == population
                      and (role is None or t["role"] == role)})
    keep = set(random.Random(f"{C.RNG_SEED}|textcraft|llama_dx|{tag}").sample(couples, min(n, len(couples))))
    return [t for t in items if t["population"] == population and t["couple_id"] in keep]


def selection(cfg: dict, items: list[dict]) -> list[dict]:
    """The items of every block in cfg['blocks'] ({population, couples, role?}), each block seeded by its own tag."""
    out: list[dict] = []
    for b in cfg["blocks"]:
        tag = f"{cfg['mode']}|{b['population']}|{b.get('role') or 'any'}"
        out += sample_couples(items, b["population"], int(b["couples"]), tag, b.get("role"))
    seen, uniq = set(), []
    for t in out:
        if t["id"] not in seen:
            seen.add(t["id"])
            uniq.append(t)
    return uniq


OBJECT_GOAL_ARMS = ("object_goal", "yn_object_goal")
# Run 4 (notes/textcraft_llama_dx.md): a world-model-only system text. It keeps the framing and states only the
# environment's rules and the output format; every agent-directed sentence of AgentGym's instruction is gone, and so
# is "from the inventory or the environment". It does not state that the inventory starts empty.
CLEAN_SYSTEM = (
    "You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next "
    "action, predict the environment's next observation exactly in TextCraft style. Output only the observation "
    "text.\n\nThe environment's rules: crafting commands are of the format \"craft [target object] using [input "
    "ingredients]\". Each action is \"get\" with a quantity and an object, \"inventory\" to look up the inventory, or "
    "\"craft\" with a quantity and one of the crafting commands shown. If a crafting command uses a generic ingredient "
    "like \"planks\", a special type of the same ingredient, e.g. \"dark oak planks\", can be used instead."
)
PROMPT_VARIANTS = ("clean", "inventory", "clean_inventory")


def inventory_reply(history: list) -> str:
    """The environment's reply to "inventory" after the history (design section 2, rule 3; section 11, A1 rule 5),
    from the text-only checker's accounting, whose held map keeps the environment's listing order."""
    held = {n: c for n, c in account(history).held.items() if c}
    if not held:
        return "Inventory: You are not carrying anything."
    return "Inventory: " + "".join(f"[{n}] ({c}) " for n, c in held.items())


def variant_item(item: dict, variant: str) -> dict:
    """The item as a run-4 variant shows it: with an inventory step appended to the history, or unchanged."""
    if "inventory" in variant:
        return {**item, "history": list(item["history"]) + [["inventory", inventory_reply(item["history"])]]}
    return item


def variant_system(variant: str) -> str:
    return CLEAN_SYSTEM if variant.startswith("clean") else P.SYSTEM


def object_goal_item(item: dict) -> dict:
    """The item with its aligned goal line replaced by one naming the action's own object ("Goal: craft Y.")."""
    return {**item, "task_line": f"Goal: craft {item['object']}."}


def arms_of(mode: str) -> tuple[str, ...]:
    if mode == "prompts":
        return tuple(f"{v}|{c}" for v in PROMPT_VARIANTS for c in CONDITIONS)
    return OBJECT_GOAL_ARMS if mode == "objectgoal" else CONDITIONS


def answer_of(text: str) -> str:
    """A free yes/no answer: Yes, No or other, from its first word."""
    w = text.strip().split()[0].strip(".,:;!").lower() if text.strip() else ""
    return {"yes": "Yes", "no": "No"}.get(w, "other")


def zero_shot_messages(item: dict, condition: str) -> list[dict[str, str]]:
    """The verified prompt for a condition with no in-context examples."""
    return P.messages(item, condition, fewshot=())


def outcome_of(text: str) -> str:
    """How a generated observation reads: success, the failure reply, another failure wording, or other."""
    s = text.strip()
    if s.startswith(("Crafted ", "Got ")):
        return "success"
    if s.startswith("Could not find enough items"):
        return "failure"
    if s.startswith("Could not"):
        return "other_failure"
    return "other"


def top_first_tokens(scorer: LogoddsScorer, messages: list[dict[str, str]], k: int = 10) -> list[list[Any]]:
    """[[token text, probability], ...] for the k most probable first tokens after the prompt."""
    if scorer.mock:
        return [["Could", 0.6], ["Craft", 0.3], ["Got", 0.1]][:k]
    import torch

    ids = scorer._encode(messages)
    device = next(scorer._model.parameters()).device
    with torch.no_grad():
        logits = scorer._model(input_ids=ids.to(device)).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p, idx = probs.topk(k)
    return [[scorer._tok.decode([int(i)]), float(v)] for v, i in zip(p.tolist(), idx.tolist())]


def _models(cfg: dict) -> list[tuple[str, str]]:
    return [m for m in MODELS if m[1] in cfg.get("models", [tag for _n, tag in MODELS])]


def run_textcraft_dx(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"items not verified: {files}", ok=False)
        return 1
    items_path, shots_path, manifest_path = files
    d = _init_run(run_id, cfg, inputs=(items_path, shots_path, manifest_path))
    items = selection(cfg, load_items(items_path))
    shots = P.fewshot_examples(shots_path)
    mode = cfg["mode"]
    if mode == "zeroshot":
        hashes = {t["id"]: item_hash(t, (), P.SYSTEM) for t in items}
    elif mode == "generate":  # the observation prompt's hash; the yes/no prompt differs only in its last line
        hashes = {t["id"]: item_hash(t, shots, P.SYSTEM) for t in items}
    elif mode == "objectgoal":
        hashes = {t["id"]: item_hash(object_goal_item(t), shots, P.SYSTEM) for t in items}
    elif mode == "prompts":  # one hash per item and variant; a record keeps its variant's
        hashes = {(t["id"], v): item_hash(variant_item(t, v), shots, variant_system(v)) for t in items
                  for v in PROMPT_VARIANTS}
    else:
        raise ValueError(mode)
    out = d / ("generations.jsonl" if mode == "generate" else "scoring.jsonl")  # both in RUN_OUTPUTS: wiped or repaired
    saved = read_jsonl(out) if cfg.get("resume") and out.exists() else []
    key = (lambda tid, cond: (tid, cond.split("|")[0])) if mode == "prompts" else (lambda tid, _c: tid)
    if any(hashes.get(key(s["transition_id"], s["condition"])) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items or prompts; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for model_name, tag in _models(cfg):
        todo = [(t, c) for t in items for c in arms_of(mode) if (tag, t["id"], c) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, c in todo:
            base = {"game_id": t["pair_id"], "pair_id": t["pair_id"], "population": t["population"],
                    "role": t["role"], "item_hash": hashes[key(t["id"], c)], "variant": mode}
            if mode == "zeroshot":
                plus, minus = P.candidates(t, c)
                sc = scorer.score_pair(transition_id=t["id"], condition=c, messages=zero_shot_messages(t, c),
                                       o_plus=plus, o_minus=minus)
                rec = {**sc.to_dict(), **base}
            elif mode == "objectgoal":  # the aligned prompt with the object-goal line; yn_ arms ask the question
                arm = "yn_aligned" if c == "yn_object_goal" else "aligned"
                obj = object_goal_item(t)
                plus, minus = P.candidates(obj, arm)
                sc = scorer.score_pair(transition_id=t["id"], condition=c, messages=P.messages(obj, arm, shots),
                                       o_plus=plus, o_minus=minus)
                rec = {**sc.to_dict(), **base, "task_line": obj["task_line"]}
            elif mode == "prompts":
                v, cond = c.split("|")
                shown = variant_item(t, v)
                plus, minus = P.candidates(shown, cond)
                sc = scorer.score_pair(transition_id=t["id"], condition=c,
                                       messages=P.messages(shown, cond, shots, variant_system(v)),
                                       o_plus=plus, o_minus=minus)
                rec = {**sc.to_dict(), **base, "prompt_variant": v, "line": cond}
            else:
                msgs = P.messages(t, c, shots)
                text = scorer.free_generate(msgs, max_new_tokens=int(cfg.get("max_new_tokens", 40)))
                top = top_first_tokens(scorer, msgs, int(cfg.get("top_k", 10)))
                yn = scorer.free_generate(P.messages(t, f"yn_{c}", shots), max_new_tokens=int(cfg.get("yn_tokens", 4)))
                rec = {**base, "model": tag, "model_revision": scorer.revision, "transition_id": t["id"],
                       "condition": c, "generation": text, "outcome": outcome_of(text), "top_first": top,
                       "yn_generation": yn, "yn_answer": answer_of(yn), "truth": "failure" if t["population"] == "main" else "success",
                       "o_plus": t["candidates"]["o_plus"], "o_minus": t["candidates"]["o_minus"],
                       "exact_plus": text.strip() == t["candidates"]["o_plus"],
                       "exact_minus": text.strip() == t["candidates"]["o_minus"]}
            if any(isinstance(v, float) and math.isnan(v) for v in rec.values()):
                rec["nan"] = True
            append_jsonl(out, [rec])
        scorer.close()
    write_status(run_id, f"{mode}: {len(items)} items, {len(_models(cfg))} model(s), {len(arms_of(mode))} arms",
                 ok=False, reported="dx complete")
    return 0
