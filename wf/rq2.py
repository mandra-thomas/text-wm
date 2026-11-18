"""Confirmatory RQ2: does an aligned goal raise a world model's predicted success on actions that fail?

Scores the main-test items collected on `valid_unseen` (rq12_transitions_s*) under the aligned and swapped
goals, history only, prompt example set 1, for the two co-primary models (preregistration), plus the
exploratory object-mention 2x2 (notes/deviations.md, 2026-11-17). Scoring is sharded by game and saved per
score; `rq2_analysis` combines the shards, checks the preregistered minimums and applies the decision rule.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, gen_success_shift, pair_permutation, prereg_rq2_verdicts
from wf.data.transitions import determination_reason
from wf.runs import (
    PILOT_A_MODELS,
    _effect,
    _finished,
    _init_run,
    _item_hash,
    _pilotA_item_sets,
)
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.scoring.parser import parse_prediction
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, runs_root, write_json, write_status

CO_PRIMARY = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
SOURCES = ("rq12_transitions_s0", "rq12_transitions_s1", "rq12_transitions_s2")
MENTION_FOR = {"heat": "clean", "cool": "clean", "clean": "heat"}  # merely irrelevant, never opposed
_ADJ = {"heat": "hot", "cool": "cool", "clean": "clean"}


# ---------------- items ----------------


def load_items(sources: tuple[str, ...] = SOURCES) -> list[dict]:
    items: list[dict] = []
    for run in sources:
        items += read_jsonl(runs_root() / run / "transitions.jsonl")
    return items


def primary_set(transitions: list[dict]) -> list[dict]:
    """The preregistered population: determined failures, goal named, goal-agnostic history, not shared."""
    return _pilotA_item_sets(transitions)["agnostic"]


def _transform(task: str, old: str, new: str) -> str | None:
    """Swap the task's transformation: "heat some X ..." -> "clean some X ...", "put a hot X" -> "put a clean X"."""
    out, n = re.subn(rf"\b{old} some\b", f"{new} some", task)
    if n:
        return out
    out, n = re.subn(rf"\bput (?:a|some) {_ADJ[old]}\b", f"put a {_ADJ[new]}", task)
    return out if n else None


def mention_lines(t: dict) -> tuple[str, str] | None:
    """(mention, fourth) task lines for the object-mention 2x2, or None where the design cannot be built.

    Mention names the action's object, in the aligned goal's template and position, but the action does
    not advance it; fourth applies the same change to the swapped goal. heat and cool map to clean and
    clean to heat (merely irrelevant tasks, never opposed ones); move changes the destination to another
    receptacle type in the room. take items cannot be built (any task naming an object needs it picked up).
    """
    aligned, swapped, action = t.get("task_line") or "", t.get("swapped_task_line") or "", t.get("action") or ""
    if not swapped:
        return None
    verb = action.split(" ", 1)[0]
    if verb in MENTION_FOR:
        new = MENTION_FOR[verb]
        if not re.search(rf"\b(?:{verb} some|put (?:a|some) {_ADJ[verb]})\b", aligned):
            return None  # the aligned task is not this transformation, so it is not advanced by the action
        mention, fourth = _transform(aligned, verb, new), _transform(swapped, verb, new)
        return (mention, fourth) if mention and fourth else None
    m = re.fullmatch(r"move (.+) to (.+)", action)
    if m:
        dest_type = m[2].rsplit(" ", 1)[0]
        r = re.search(r"\b(?:in|on) ([a-z]+)\.\s*$", aligned)
        if not r or r[1] != dest_type:
            return None  # the action's destination is not the goal's, so the aligned goal is not advanced
        types = sorted({rec.rsplit(" ", 1)[0] for rec in t.get("scene_receptacles") or []} - {dest_type})
        if not types:
            return None
        other = random.Random(f"{C.RNG_SEED}|mention|{t['id']}").choice(types)
        swap = lambda line: re.sub(rf"\b(in|on) {dest_type}\.\s*$", rf"\1 {other}.", line)
        mention, fourth = swap(aligned), swap(swapped)
        return (mention, fourth) if mention != aligned and fourth != swapped else None
    return None


def shard_items(items: list[dict], i: int, n: int) -> list[dict]:
    games = sorted({t["game_id"] for t in items})[i::n]
    keep = set(games)
    return [t for t in items if t["game_id"] in keep]


# ---------------- scoring ----------------


def _score(scorer: LogoddsScorer, t: dict, condition: str, task_line: str | None = None) -> Any:
    """Score o+ against o- with `task_line` as the goal (the aligned slot), recorded under `condition`."""
    item = {**t, "task_line": task_line} if task_line else t
    base = "aligned" if task_line else condition
    return scorer.score_pair(transition_id=t["id"], condition=condition, messages=build_messages(item, base),
                             o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])


def run_rq2_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    if problems:
        write_status(run_id, f"collection not finished cleanly: {problems}", ok=False)
        return 1
    # Only items that a full replay on the real server has verified (wf/tools/verify_items.py).
    unverified = {s: why for s in sources for ok, why in [verified(s)] if not ok}
    if unverified:
        write_status(run_id, f"collection not verified: {unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    inputs = tuple(runs_root() / s / "transitions.jsonl" for s in sources)
    d = _init_run(run_id, cfg, inputs=inputs)
    items = shard_items(primary_set(load_items(sources)), i, n)
    hashes = {t["id"]: _item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    saved_gens = read_jsonl(d / "generations.jsonl") if cfg.get("resume") else []
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    gen_done = {(g["model"], g["transition_id"], g["condition"]) for g in saved_gens}
    mention = bool(cfg.get("mention", True))
    generate = bool(cfg.get("generations", True))
    models = [(name, tag, ft) for name, tag, ft in PILOT_A_MODELS if tag in cfg.get("models", CO_PRIMARY)]
    for model_name, tag, ft in models:
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t in items:
            jobs = [("aligned", None), ("swapped", None)]
            lines = mention_lines(t) if mention else None
            if lines:
                jobs += [("mention", lines[0]), ("fourth", lines[1])]
            for cond, line in jobs:
                if (tag, t["id"], cond) in done:
                    continue
                if line is None:
                    sc = _score(scorer, t, cond)
                else:
                    sc = _score(scorer, t, cond, task_line=line)
                rec = {**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]],
                       "task_line_used": line}
                append_jsonl(d / "scoring.jsonl", [rec])
            if generate:
                for cond in ("aligned", "swapped"):
                    if (tag, t["id"], cond) in gen_done:
                        continue
                    text = scorer.free_generate(build_messages(t, cond, ft_prompt=ft))
                    append_jsonl(d / "generations.jsonl", [{
                        "model": tag, "transition_id": t["id"], "condition": cond, "game_id": t["game_id"],
                        "generated": text, "generated_class": parse_prediction(text, t.get("goal_object"), t["action"]),
                        "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} items, {len({t['game_id'] for t in items})} games scored",
                 ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _paired(scores: list[dict], items: list[dict], a: str, b: str, value: str = "L") -> tuple[list[float], list[str]]:
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    diffs, games = [], []
    for t in items:
        x, y = by.get((t["id"], a)), by.get((t["id"], b))
        if x and y and x.get(value) == x.get(value) and y.get(value) == y.get(value):
            diffs.append(x[value] - y[value])
            games.append(t["game_id"])
    return diffs, games


def _mean_ci_p(diffs: list[float], games: list[str]) -> dict:
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    boot = cluster_bootstrap(mean, diffs, games, seed=C.RNG_SEED)
    return {"estimate": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]], "n_items": len(diffs),
            "n_games": len(set(games)), "perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=games)}


def mention_estimands(scores: list[dict], items: list[dict]) -> dict:
    """Goal-directed effect (aligned - mention) - (swapped - fourth), and the mention effect mention - fourth."""
    by = {(s["transition_id"], s["condition"]): s["L"] for s in scores}
    did, did_g, men, men_g = [], [], [], []
    for t in items:
        v = [by.get((t["id"], c)) for c in ("aligned", "mention", "swapped", "fourth")]
        if any(x is None or x != x for x in v):
            continue
        did.append((v[0] - v[1]) - (v[2] - v[3]))
        did_g.append(t["game_id"])
        men.append(v[1] - v[3])
        men_g.append(t["game_id"])
    return {"goal_directed": _mean_ci_p(did, did_g), "mention": _mean_ci_p(men, men_g)}


def run_rq2_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    source = runs_root() / cfg["source"]
    folders = sorted(p for p in source.glob("shard_*_of_*") if p.is_dir())
    problems = []
    counts = {p.name.rsplit("_of_", 1)[1] for p in folders}
    if not folders or len(counts) != 1:
        problems.append(f"shards missing or from different splits: {[p.name for p in folders]}")
    else:
        n = int(counts.pop())
        if sorted(int(p.name.split("_")[1]) for p in folders) != list(range(n)):
            problems.append(f"expected shards 0..{n - 1}, have {[p.name for p in folders]}")
        for p in folders:
            first = (p / "STATUS.md").read_text().splitlines()[0] if (p / "STATUS.md").exists() else ""
            if "REPORTED (shard complete)" not in first:
                problems.append(f"{p.name} not complete: {first or 'no status'}")
        shas = {(p / "git_sha").read_text().strip() for p in folders if (p / "git_sha").exists()}
        if len(shas) != 1:
            problems.append(f"shards ran at different commits: {sorted(shas)}")
    d = _init_run(run_id, {**cfg, "resume": False},
                  inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    if problems:
        write_status(run_id, "RQ2 waits for every scoring shard:\n" + "\n".join(problems), ok=False)
        return 1
    transitions = load_items(tuple(cfg.get("sources", SOURCES)))
    items = primary_set(transitions)
    states = {f"{t['game_id']}#{t['state_idx']}" for t in items}
    # preregistration: n >= 1,000 items from at least 200 states (configurable only for tests)
    prereg_ok = len(items) >= int(cfg.get("min_items", 1000)) and len(states) >= int(cfg.get("min_states", 200))
    scores = [s for p in folders for s in read_jsonl(p / "scoring.jsonl")]
    gens = [g for p in folders for g in read_jsonl(p / "generations.jsonl")]
    wanted = {(m, t["id"], c) for m in CO_PRIMARY for t in items for c in ("aligned", "swapped")}
    got = {(s["model"], s["transition_id"], s["condition"]) for s in scores}
    missing = wanted - got
    bins = {"0": lambda k: k == 0, "1-3": lambda k: 1 <= k <= 3, "4-7": lambda k: 4 <= k <= 7, "8+": lambda k: k >= 8}
    per_model: dict[str, dict] = {}
    for m in CO_PRIMARY:
        mine = [s for s in scores if s["model"] == m]
        core = _effect(mine, items)
        mg = [g for g in gens if g["model"] == m]
        covered = [t for t in items if mention_lines(t)]
        per_model[m] = {
            **core,
            "by_history_len": {k: _effect(mine, [t for t in items if f(t.get("history_len", 0))]) for k, f in bins.items()},
            "by_reason": {r: _effect(mine, [t for t in items if (t.get("determination_reason") or determination_reason(t)) == r])
                          for r in sorted({t.get("determination_reason") or determination_reason(t) for t in items})},
            "object_mention_2x2": {**mention_estimands(mine, covered), "items_covered": len(covered)},
            "written_shift_provisional": gen_success_shift(mg, items) if mg else None,
        }
    verdicts = prereg_rq2_verdicts(per_model)
    report = {
        "items": {"n_items": len(items), "n_states": len(states), "n_games": len({t["game_id"] for t in items}),
                  "prereg_minimum_met": prereg_ok, "missing_primary_scores": len(missing)},
        "verdicts": verdicts,
        "per_model": per_model,
    }
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "RQ2 (confirmatory)", report)
    lines = [f"{m}: dL={per_model[m]['delta_L_nats']:.3f} CI={per_model[m]['delta_L_ci']} "
             f"holm_p={verdicts['per_model'][m]['holm_p']} -> {verdicts['per_model'][m]['verdict']}" for m in CO_PRIMARY]
    ok = prereg_ok and not missing
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines) + f"\nH2 overall: {verdicts['overall']}",
                 ok=False, reported="RQ2 verdict" if ok else None)
    return 0 if ok else 1
