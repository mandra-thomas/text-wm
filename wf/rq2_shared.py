"""Shared-spot sensitivity for confirmatory RQ2 (decided 2026-11-20, documentation audit A2; not preregistered).

The decision log (notes/deviations.md, the shared-spot rule) promised that items dropped by the shared-spot rule
would be reported both ways. An item is "at a shared spot" when the action's receptacle shares a spot with the
receptacle the agent is at, so "you are elsewhere" proves nothing (wf.data.transitions.at_shared_spot). Such
items left the main test and were never scored on the confirmatory data.

The items here are those excluded from RQ2's primary population ONLY by that rule: determined failures with a
success twin and a swapped goal, whose action names the goal object and whose history is goal-agnostic, but
flagged `shared_location` (`shared_spot_items`). They come from the same collection files as the primary set
(rq12_transitions_s0..s2).

- `rq2_shared_score` scores them exactly as `rq2_score` scores the primary items: both co-primary models,
  aligned and swapped goals, history only, example set 1, the same prompt builder and scorer. Sharded by game.
  A shard with no items loads no model.
- `rq2_shared_analysis` (CPU) reports, per model, ΔL with its game-clustered CI for (a) the shared-spot items
  alone and (b) the confirmatory items with them included, beside the confirmatory set as scored, with the
  first-token term (ΔL_first) and the shift in s = sigmoid(L) (Δs, points) for each. It also counts the
  shared-spot flag at every stage of the item flow, so an empty set is explained rather than silent.

On the confirmatory collection (checked 2026-11-20) the set is empty: 19 of the 12,987 collected items carry the
flag, only one of them is an eligible failure, and that one has a goal-revealing history, so it is outside the
primary population for a second reason. The scoring run then has nothing to score; the analysis reports n = 0
and (b) equal to the confirmatory set.
"""
from __future__ import annotations

import os
from typing import Any

from wf.analysis.report import write_run_report
from wf.rq2 import CO_PRIMARY, SOURCES, _score, analysis_provenance, complete_shards, load_items, primary_set, shard_items
from wf.runs import PILOT_A_MODELS, _effect, _finished, _init_run, _item_hash, _pilotA_item_sets
from wf.scoring.logodds import LogoddsScorer
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, runs_root, write_json, write_status

CONDITIONS = ("aligned", "swapped")


def shared_spot_items(transitions: list[dict]) -> list[dict]:
    """Items excluded from RQ2's primary set only by the shared-spot rule."""
    return [t for t in _pilotA_item_sets(transitions)["agnostic_all"] if t.get("shared_location")]


def shared_flag_flow(transitions: list[dict]) -> dict[str, int]:
    """How many items carry the shared-spot flag at each stage of the primary set's selection."""
    flagged = [t for t in transitions if t.get("shared_location")]
    fail = [t for t in flagged if (t.get("true_obs") or "").strip() == "Nothing happens."]
    det = [t for t in fail if t.get("context_determined")]
    eligible = [t for t in det if t.get("candidates") is not None and t.get("swapped_goal_object")]
    targeted = [t for t in eligible if t.get("targets_goal_object")]
    agnostic = [t for t in targeted if t.get("history_goal_agnostic")]
    return {"collected": len(flagged), "true_failures": len(fail), "determined": len(det),
            "with_twin_and_swap": len(eligible), "targets_goal_object": len(targeted),
            "goal_agnostic_history": len(agnostic),
            "determined_successes": sum(1 for t in flagged if t.get("context_determined") and t.get("candidates")
                                        and (t.get("true_obs") or "").strip() != "Nothing happens.")}


def run_rq2_shared_score(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    i = int(os.environ.get("WF_SHARD") or cfg.get("shard", 0))
    n = int(os.environ.get("WF_N_SHARDS") or cfg.get("n_shards", 1))
    sources = tuple(cfg.get("sources", SOURCES))
    problems = [s for s in sources if not _finished(s, tuple(cfg.get("accept_commits", {}).get(s, ())))]
    if problems:
        write_status(run_id, f"collection not finished cleanly: {problems}", ok=False)
        return 1
    unverified = {s: why for s in sources for ok, why in [verified(s)] if not ok}
    if unverified:
        write_status(run_id, f"collection not verified: {unverified}", ok=False)
        return 1
    run_id = f"{run_id}/shard_{i:02d}_of_{n:02d}"
    cfg = {**cfg, "run_id": run_id, "shard": i, "n_shards": n}
    d = _init_run(run_id, cfg, inputs=tuple(runs_root() / s / "transitions.jsonl" for s in sources))
    items = shard_items(shared_spot_items(load_items(sources)), i, n)
    hashes = {t["id"]: _item_hash(t) for t in items}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["model"], s["transition_id"], s["condition"]) for s in saved}
    models = [(name, tag, ft) for name, tag, ft in PILOT_A_MODELS if tag in cfg.get("models", CO_PRIMARY)]
    for model_name, tag, ft in models:
        todo = [(t, c) for t in items for c in CONDITIONS if (tag, t["id"], c) not in done]
        if not todo:
            continue  # nothing to score: the model is never loaded
        scorer = LogoddsScorer(model_name, model_tag=tag)
        for t, cond in todo:
            sc = _score(scorer, t, cond, ft=ft)  # rq2_score's own prompt and scorer
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]],
                                                "task_line_used": None}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(items)} shared-spot items, {len({t['game_id'] for t in items})} games",
                 ok=False, reported="shard complete")
    return 0


def _summary(e: dict) -> dict[str, Any]:
    keys = ("n_items", "n_games", "delta_L_nats", "delta_L_ci", "delta_L_perm_p", "delta_L_first_nats",
            "delta_L_first_ci", "delta_p_pp", "delta_p_ci", "delta_c_nats")
    return {k: e[k] for k in keys}


def run_rq2_shared_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    sources = tuple(cfg.get("sources", SOURCES))
    transitions = load_items(sources)
    shared, primary = shared_spot_items(transitions), primary_set(transitions)
    base_folders, problems = complete_shards(cfg["base"])
    folders: list = []
    if shared:  # an empty set has nothing scored to wait for
        folders, more = complete_shards(cfg["source"])
        problems += more
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "waits for every shard:\n" + "\n".join(problems), ok=False)
        return 1
    base = [s for p in base_folders for s in read_jsonl(p / "scoring.jsonl")]
    extra = [s for p in folders for s in read_jsonl(p / "scoring.jsonl")]
    report: dict[str, Any] = {
        "items": {"n_shared_spot_only": len(shared), "n_games_shared_spot_only": len({t["game_id"] for t in shared}),
                  "n_primary": len(primary), "shared_flag_by_stage": shared_flag_flow(transitions),
                  "shared_spot_item_ids": sorted(t["id"] for t in shared)},
        "per_model": {},
    }
    for m in cfg.get("models", CO_PRIMARY):
        mine_base = [s for s in base if s["model"] == m and s["condition"] in CONDITIONS]
        mine_extra = [s for s in extra if s["model"] == m]
        report["per_model"][m] = {
            "confirmatory_as_scored": _summary(_effect(mine_base, primary)),
            "shared_spot_only": _summary(_effect(mine_extra, shared)),
            "confirmatory_with_shared_spot": _summary(_effect(mine_base + mine_extra, primary + shared)),
        }
    report["provenance"] = analysis_provenance(cfg["base"], base_folders, sources,
                                               others={cfg["source"]: folders} if folders else None)
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Shared-spot sensitivity for RQ2 (not preregistered)", report)
    lines = [f"{m}: confirmatory dL={r['confirmatory_as_scored']['delta_L_nats']:.3f}, with shared spot "
             f"dL={r['confirmatory_with_shared_spot']['delta_L_nats']:.3f} (n={r['confirmatory_with_shared_spot']['n_items']}), "
             f"shared spot alone n={r['shared_spot_only']['n_items']}" for m, r in report["per_model"].items()]
    write_status(run_id, f"items={report['items']['n_shared_spot_only']} shared-spot-only items; flag by stage "
                         f"{report['items']['shared_flag_by_stage']}\n" + "\n".join(lines),
                 ok=False, reported="shared-spot sensitivity")
    return 0
