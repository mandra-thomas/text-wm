"""Exploratory dissociation test (decided 2026-11-19, after the thinking-mode results; notes/deviations.md).

Does the goal act as evidence that its object exists (and so that an action on it can work)? Each arm adds
one sentence about the action's object to BOTH the aligned and the swapped task line, so the two prompts of a
pair differ only in the goal, as in the preregistered contrast:

- exist:   "There is a mug somewhere in the room."  (existence stated, location not)
- located: "The mug 1 is at cabinet 3."             (its true location, from the scene; the history never
                                                     touches the goal object, so it has not moved)
- absent:  "There is no mug in this room."          (existence denied; counterfactual)

The plain aligned and swapped prompts are rescored in the same job, so every comparison is paired on one GPU.
If the goal works as evidence of existence, the aligned-vs-swapped effect should shrink sharply once the
sentence settles existence (exist, located) or denies it (absent). If the goal works through what the agent
wants, the effect should persist.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from wf import rq1
from wf.analysis.report import write_run_report
from wf.rq2 import SOURCES, complete_shards, load_items, shard_items
from wf.runs import _effect, _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer, build_messages
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

SENTENCES = ("exist", "located", "absent")
ARMS = ("aligned", "swapped") + tuple(f"{g}_{s}" for s in SENTENCES for g in ("aligned", "swapped"))


def instance(t: dict) -> str | None:
    """The goal object's instance named in the action ("mug 1"), if exactly one."""
    found = set(re.findall(rf"\b{re.escape(t.get('goal_object') or '')} \d+\b", t.get("action") or ""))
    return found.pop() if len(found) == 1 else None


def location(t: dict) -> str | None:
    inst = instance(t)
    locs = (t.get("scene_locations") or {}).get(inst) if inst else None
    return locs[0] if locs and len(locs) == 1 else None


def sentence(t: dict, kind: str) -> str | None:
    obj = t["goal_object"]
    if kind == "exist":
        return rq1.mention_sentence(obj)
    if kind == "located":
        loc = location(t)
        return f"The {instance(t)} is at {loc}." if loc else None
    if kind == "absent":
        return f"There is no {obj} in this room."
    raise ValueError(kind)


def arm_messages(t: dict, arm: str) -> list[dict[str, str]] | None:
    """Messages for an arm, or None when the item cannot carry it (no single known location)."""
    goal, _, kind = arm.partition("_")
    line = t["task_line"] if goal == "aligned" else t["swapped_task_line"]
    if kind:
        s = sentence(t, kind)
        if s is None:
            return None
        line = f"{line} {s}"
    return build_messages(t, "aligned", task_line=line)


def run_rq1d_score(cfg: dict) -> int:
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
    primary = rq1.rq1_sets(load_items(sources))["primary"]
    jobs = [(t, a) for t in shard_items(primary, i, n) for a in ARMS if arm_messages(t, a) is not None]
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
            sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm),
                                   o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} scores", ok=False, reported="shard complete")
    return 0


def _block(s: list[dict], items: list[dict]) -> dict:
    """Plain effect, and for each sentence: the effect with it, and how far it moves the swapped prompt."""
    out: dict[str, Any] = {"n": len(items), "plain": _effect(s, items)}
    for k in SENTENCES:
        out[k] = _effect(s, items, primary=f"aligned_{k}", control=f"swapped_{k}")
        out[f"{k}_moves_swapped"] = _effect(s, items, primary=f"swapped_{k}", control="swapped")
        out[f"{k}_moves_aligned"] = _effect(s, items, primary=f"aligned_{k}", control="aligned")
    return out


def location_matches(t: dict) -> bool:
    """Whether the object's true location is the receptacle the action names (most location failures)."""
    from wf.scoring.parser import _action_full
    return location(t) == _action_full(t.get("action"))["recep"]


def run_rq1d_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    primary = rq1.rq1_sets(load_items(tuple(cfg.get("sources", SOURCES))))["primary"]
    located = [t for t in primary if location(t)]  # every sentence compared on the same items
    scores = [x for p in folders for x in read_jsonl(p / "scoring.jsonl")]
    report: dict[str, Any] = {"items": {"n_primary": len(primary), "n_with_location": len(located)}}
    for tag in cfg.get("models", rq1.FEWSHOT):
        s = [x for x in scores if x["model"] == tag]
        reasons = defaultdict(list)
        for t in located:
            reasons[t.get("determination_reason") or "unknown"].append(t)
        report[tag] = {
            "all": _block(s, located),
            "history_0": _block(s, [t for t in located if t.get("history_len", 0) == 0]),
            "history_1plus": _block(s, [t for t in located if t.get("history_len", 0) > 0]),
            "by_failure_reason": {r: _block(s, its) for r, its in sorted(reasons.items()) if len(its) >= 50},
            # amendment (2026-11-19, before any result was read): the located sentence either agrees with the
            # action's receptacle or contradicts it
            "location_matches_action": _block(s, [t for t in located if location_matches(t)]),
            "location_differs_from_action": _block(s, [t for t in located if not location_matches(t)]),
        }
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Dissociation test (exploratory)", report)
    lines = [f"{tag}: plain {report[tag]['all']['plain']['delta_L_nats']:.2f}, "
             + ", ".join(f"{k} {report[tag]['all'][k]['delta_L_nats']:.2f}" for k in SENTENCES)
             for tag in cfg.get("models", rq1.FEWSHOT)]
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines), ok=False, reported="dissociation report")
    return 0
