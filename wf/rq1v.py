"""Exploratory validation runs (decided 2026-11-19, after outside reviews of the draft; notes/deviations.md).

1. Counterfactual aligned task (`cf_*` arms). The aligned task is always the episode's real task, so the history
   was produced by an agent pursuing it and the swapped prompt may read as less coherent. Here the real task's
   destination is replaced by another receptacle type present in the room, giving a task the action still
   advances (take, heat, cool and clean do not depend on the destination) that is not the real task; the swapped
   counterpart replaces the object in that line. Neither task in the pair is real. The extra-mention pair is
   rebuilt on these lines too, so the goal-role part can be measured without the real task. Move actions are
   left out: moving an object advances only tasks with that destination.

2. Yes/no feasibility (`yn_*` arms). L compares an action-specific success text with "Nothing happens.", so its
   level is not comparable across different actions. The same prompt ends instead with a fixed question, and
   the model scores the single tokens "Yes" against "No". Scored on the failures and the determined successes
   under the aligned, swapped and neutral tasks, it gives a goal effect, an AUC and a top-candidate rate on a
   scale that is the same for every action.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from typing import Any

from wf import constants as C
from wf import rq1, rq1x
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap
from wf.data.goals import make_swapped_task_line, parse_task_line
from wf.rq2 import SOURCES, analysis_provenance, complete_shards, load_items, shard_items
from wf.runs import _effect, _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

CF_VERBS = ("take", "heat", "cool", "clean")
# Receptacle types that serve as task destinations in ALFWorld.
DESTINATIONS = ("armchair", "bathtubbasin", "bed", "cabinet", "cart", "coffeemachine", "coffeetable", "countertop",
                "desk", "diningtable", "drawer", "dresser", "fridge", "garbagecan", "microwave", "ottoman", "safe",
                "shelf", "sidetable", "sinkbasin", "sofa", "stoveburner", "toilet")
CF_ARMS = ("cf_aligned", "cf_swapped", "cf_aligned_plus", "cf_swapped_plus")
YN_ARMS = ("yn_aligned", "yn_swapped", "yn_absent")
PLAIN_ARMS = ("aligned", "swapped")
YN_QUESTION = ("Question: will this action succeed, or will the environment reply \"Nothing happens.\"? "
               "Answer Yes if it succeeds and No if nothing happens.\nAnswer:")


def _types(receptacles: list[str]) -> set[str]:
    return {re.sub(r"\s+\d+$", "", r) for r in receptacles or []}


def cf_destination(t: dict) -> str | None:
    """Another destination type present in the room, chosen by a hash of the item, or None if not buildable."""
    if t["action"].split(" ", 1)[0] not in CF_VERBS:
        return None
    line = t["task_line"].split("Your task is to:", 1)[-1].strip()
    if line.startswith(("look at", "examine")):
        return None
    try:
        rec = parse_task_line(t["task_line"]).receptacle
    except ValueError:
        return None
    acted = re.search(r" (?:from|with) ([a-z]+) \d+$", t["action"])
    exclude = {rec, acted[1] if acted else None}
    options = sorted(d for d in _types(t.get("scene_receptacles")) if d in DESTINATIONS and d not in exclude)
    if not options:
        return None
    h = int(hashlib.sha256(f"rq1v|{t['id']}".encode()).hexdigest()[:8], 16)
    return options[h % len(options)]


def cf_lines(t: dict) -> tuple[str, str] | None:
    """(counterfactual aligned line, its swapped counterpart): only the destination differs from the real ones."""
    dest = cf_destination(t)
    if not dest or not t.get("swapped_goal_object"):
        return None
    rec = parse_task_line(t["task_line"]).receptacle
    new, n = re.subn(rf"\b{re.escape(rec)}\b(?=\.?\s*$)", dest, t["task_line"].rstrip())
    if n != 1:
        return None
    return new, make_swapped_task_line(new, t["goal_object"], t["swapped_goal_object"])


def arm_messages(t: dict, arm: str) -> list[dict[str, str]] | None:
    if arm in PLAIN_ARMS:
        return rq1.arm_messages(t, arm)
    if arm in CF_ARMS:
        lines = cf_lines(t)
        if not lines:
            return None
        al, sw = lines
        line = {"cf_aligned": al, "cf_swapped": sw,
                "cf_aligned_plus": f"{al} {rq1.mention_sentence(t['swapped_goal_object'])}",
                "cf_swapped_plus": f"{sw} {rq1.mention_sentence(t['goal_object'])}"}[arm]
        return build_messages(t, "aligned", task_line=line)
    if arm in YN_ARMS:
        base = arm.split("_", 1)[1]
        if base == "swapped" and not t.get("swapped_task_line"):
            return None
        msgs = rq1.arm_messages(t, base)
        last = msgs[-1]
        if not last["content"].endswith("Next observation:"):
            raise ValueError(f"{t['id']}: prompt does not end with 'Next observation:'")
        return msgs[:-1] + [{**last, "content": last["content"][: -len("Next observation:")] + YN_QUESTION}]
    raise ValueError(arm)


def rq1v_jobs(primary: list[dict], successes: list[dict]) -> list[tuple[dict, str]]:
    jobs = [(t, a) for t in primary for a in PLAIN_ARMS + CF_ARMS + YN_ARMS]
    jobs += [(t, a) for t in successes for a in YN_ARMS]
    return [(t, a) for t, a in jobs if arm_messages(t, a) is not None]


def run_rq1v_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    unverified = [s for s in sources if not verified(s)[0]]
    if problems or unverified:
        write_status(run_id, f"collection not finished or not verified: {problems + unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=())
    sets = rq1x.rq1x_sets(load_items(sources))
    jobs = rq1v_jobs(shard_items(sets["primary"], i, n), shard_items(sets["successes"], i, n))
    hashes = {t["id"]: _item_hash(t) for t, _ in jobs}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    for model_name, tag, _ft in [m for m in rq1.MODELS if m[1] in cfg.get("models", rq1.FEWSHOT)]:
        todo = [(t, a) for t, a in jobs if (tag, t["id"], a) not in done]
        if not todo:
            continue
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, arm in todo:
            plus, minus = (("Yes", "No") if arm in YN_ARMS else (t["candidates"]["o_plus"], t["candidates"]["o_minus"]))
            sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm),
                                   o_plus=plus, o_minus=minus)
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} scores", ok=False, reported="shard complete")
    return 0


# ---------------- analysis ----------------


def _top1(by: dict, cands: list[dict], arm: str) -> dict[tuple, float]:
    """Per state with at least one working and one failing candidate: 1.0 if the top-scored one fails."""
    states = defaultdict(list)
    for t in cands:
        states[(t["game_id"], t.get("state_idx"), tuple(t.get("prefix_actions") or []))].append(t)
    out = {}
    for key, ts in states.items():
        sc = [(by[(t["id"], arm)], (t.get("true_obs") or "").strip() != "Nothing happens.") for t in ts if (t["id"], arm) in by]
        if any(w for _, w in sc) and not all(w for _, w in sc):
            out[key] = 0.0 if max(sc)[1] else 1.0
    return out


def _paired_top1(by: dict, cands: list[dict], a: str, b: str) -> dict:
    ra, rb = _top1(by, cands, a), _top1(by, cands, b)
    keys = sorted(k for k in ra if k in rb)
    boot = cluster_bootstrap(lambda xs: sum(xs) / len(xs) if xs else float("nan"), [ra[k] - rb[k] for k in keys],
                             [k[0] for k in keys], seed=C.RNG_SEED)
    mean = lambda r: sum(r[k] for k in keys) / len(keys) if keys else float("nan")
    return {f"rate_{a}": mean(ra), f"rate_{b}": mean(rb), "difference": boot, "n_states": len(keys)}


def run_rq1v_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    sets = rq1x.rq1x_sets(load_items(tuple(cfg.get("sources", SOURCES))))
    primary, successes = sets["primary"], sets["successes"]
    cf_items = [t for t in primary if cf_lines(t)]
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_cf": len(cf_items), "n_successes": len(successes)}}
    for tag in cfg.get("models", rq1.FEWSHOT):
        s = [x for x in scores if x["model"] == tag]
        by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
        h0 = [t for t in cf_items if t.get("history_len", 0) == 0]
        h1 = [t for t in cf_items if t.get("history_len", 0) > 0]
        both = primary + successes
        report[tag] = {
            "counterfactual_task": {
                "real_task_effect": _effect(s, cf_items),
                "counterfactual_task_effect": _effect(s, cf_items, primary="cf_aligned", control="cf_swapped"),
                "cf_goal_role_both_named": _effect(s, cf_items, primary="cf_aligned_plus", control="cf_swapped_plus"),
                "cf_mention": _effect(s, cf_items, primary="cf_swapped_plus", control="cf_swapped"),
                "by_history": {"history_0": {"real": _effect(s, h0), "cf": _effect(s, h0, primary="cf_aligned", control="cf_swapped"),
                                             "cf_goal_role": _effect(s, h0, primary="cf_aligned_plus", control="cf_swapped_plus")},
                               "history_1plus": {"real": _effect(s, h1), "cf": _effect(s, h1, primary="cf_aligned", control="cf_swapped"),
                                                 "cf_goal_role": _effect(s, h1, primary="cf_aligned_plus", control="cf_swapped_plus")}},
            },
            "yes_no": {
                "goal_effect_failures": _effect(s, primary, primary="yn_aligned", control="yn_swapped"),
                "aligned_vs_neutral_failures": _effect(s, primary, primary="yn_aligned", control="yn_absent"),
                # outcome="success": before 2026-11-20 this read the failure filter and came out empty (audit B1)
                "goal_effect_successes": _effect(s, successes, require_agnostic=False, primary="yn_aligned",
                                                 control="yn_swapped", outcome="success"),
                "auc": {a: rq1x._auc_ci(by, both, a) for a in YN_ARMS},
                "auc_aligned_minus_neutral": rq1x._auc_diff(by, both, "yn_aligned", "yn_absent"),
                "top_candidate_fails": _paired_top1(by, both, "yn_aligned", "yn_absent"),
            },
        }
    report["provenance"] = analysis_provenance(cfg["source"], folders, cfg.get("sources", SOURCES))
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Validation runs: counterfactual task and yes/no feasibility (exploratory)", report)
    lines = [f"{tag}: real {report[tag]['counterfactual_task']['real_task_effect']['delta_L_nats']:.2f}, "
             f"cf {report[tag]['counterfactual_task']['counterfactual_task_effect']['delta_L_nats']:.2f}, "
             f"yn {report[tag]['yes_no']['goal_effect_failures']['delta_L_nats']:.2f}" for tag in cfg.get("models", rq1.FEWSHOT)]
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines), ok=False, reported="validation report")
    return 0
