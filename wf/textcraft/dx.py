"""Exploratory diagnostics of the TextCraft Llama result (notes/textcraft_llama_dx.md). Not part of the frozen design.

`textcraft_dx_zeroshot`: the verified prompt without its in-context examples (system text, then the item's turn),
both models, the aligned, swapped and neutral lines, two-candidate scores, on seeded whole couples of the main and
success populations. Records have the fields of textcraft_score's, with `variant` "zeroshot".
`textcraft_dx_generate`: under the verified prompt (examples included), greedy free generation and the most probable
first tokens, Llama only by default, on seeded whole couples of given roles. One record per item and line.

Both save every record as it is made (scoring.jsonl, generations.jsonl) and, with --resume, skip what is saved.
"""
from __future__ import annotations

import math
import random
from typing import Any

from wf import constants as C
from wf.runs import _init_run
from wf.scoring.logodds import LogoddsScorer
from wf.textcraft import prompts as P
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
    elif mode == "generate":
        hashes = {t["id"]: item_hash(t, shots, P.SYSTEM) for t in items}
    else:
        raise ValueError(mode)
    out = d / ("scoring.jsonl" if mode == "zeroshot" else "generations.jsonl")  # both in RUN_OUTPUTS: wiped or repaired
    saved = read_jsonl(out) if cfg.get("resume") and out.exists() else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items or prompts; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for model_name, tag in _models(cfg):
        todo = [(t, c) for t in items for c in CONDITIONS if (tag, t["id"], c) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, c in todo:
            base = {"game_id": t["pair_id"], "pair_id": t["pair_id"], "population": t["population"],
                    "role": t["role"], "item_hash": hashes[t["id"]], "variant": mode}
            if mode == "zeroshot":
                plus, minus = P.candidates(t, c)
                sc = scorer.score_pair(transition_id=t["id"], condition=c, messages=zero_shot_messages(t, c),
                                       o_plus=plus, o_minus=minus)
                rec = {**sc.to_dict(), **base}
            else:
                msgs = P.messages(t, c, shots)
                text = scorer.free_generate(msgs, max_new_tokens=int(cfg.get("max_new_tokens", 40)))
                top = top_first_tokens(scorer, msgs, int(cfg.get("top_k", 10)))
                rec = {**base, "model": tag, "model_revision": scorer.revision, "transition_id": t["id"],
                       "condition": c, "generation": text, "outcome": outcome_of(text), "top_first": top,
                       "o_plus": t["candidates"]["o_plus"], "o_minus": t["candidates"]["o_minus"],
                       "exact_plus": text.strip() == t["candidates"]["o_plus"],
                       "exact_minus": text.strip() == t["candidates"]["o_minus"]}
            if any(isinstance(v, float) and math.isnan(v) for v in rec.values()):
                rec["nan"] = True
            append_jsonl(out, [rec])
        scorer.close()
    write_status(run_id, f"{mode}: {len(items)} items, {len(_models(cfg))} model(s), {len(CONDITIONS)} lines",
                 ok=False, reported="dx complete")
    return 0
