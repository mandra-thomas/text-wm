"""Exploratory model-size series (decided 2026-11-19, after the meeting with Prof. Zhang; notes/deviations.md).

The Qwen3 dense instruct models, 0.6B to 32B, thinking off, scored with the RQ2 prompt (example set 1, history
only) on RQ2's primary items and the determined successes, under the aligned, swapped and neutral tasks, both as
the two-candidate score and as the yes/no feasibility question. One scoring run per model (`size_score_<tag>`),
sharded by game, so each model can run on the GPU it needs; `size_analysis` reads them all.
"""
from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from typing import Any

from wf import constants as C
from wf import rq1, rq1v, rq1x
from wf.analysis.report import write_run_report
from wf.analysis.stats import decision_flip_share
from wf.rq2 import SOURCES, complete_shards, load_items, shard_items
from wf.runs import _effect, _finished, _init_run, _item_hash
from wf.scoring.logodds import LogoddsScorer
from wf.tools.verify_items import verified
from wf.util import append_jsonl, read_jsonl, require_filed, write_json, write_status

# tag: (Hugging Face name, actual parameters in billions, from the checkpoints; the names are nominal)
SIZES = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", 0.75),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", 2.03),
    "qwen3-4b": ("Qwen/Qwen3-4B", 4.02),
    "qwen3-8b": ("Qwen/Qwen3-8B", 8.19),
    "qwen3-14b": ("Qwen/Qwen3-14B", 14.77),
    "qwen3-32b": ("Qwen/Qwen3-32B", 32.76),
}
PRED_ARMS = ("aligned", "swapped", "absent")
YN_ARMS = rq1v.YN_ARMS  # yn_aligned, yn_swapped, yn_absent
SUCCESS_ARMS = ("aligned", "absent", "yn_aligned", "yn_absent")


def arm_messages(t: dict, arm: str) -> list[dict[str, str]] | None:
    if arm in YN_ARMS:
        return rq1v.arm_messages(t, arm)
    if arm == "swapped" and not t.get("swapped_task_line"):
        return None
    return rq1.arm_messages(t, arm)


def size_jobs(primary: list[dict], successes: list[dict]) -> list[tuple[dict, str]]:
    jobs = [(t, a) for t in primary for a in PRED_ARMS + YN_ARMS] + [(t, a) for t in successes for a in SUCCESS_ARMS]
    return [(t, a) for t, a in jobs if arm_messages(t, a) is not None]


def run_size_score(cfg: dict) -> int:
    run_id, tag = cfg["run_id"], cfg["model"]
    if tag not in SIZES:
        raise SystemExit(f"unknown model {tag!r}; known: {sorted(SIZES)}")
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
    jobs = size_jobs(shard_items(sets["primary"], i, n), shard_items(sets["successes"], i, n))
    hashes = {t["id"]: _item_hash(t) for t, _ in jobs}
    saved = read_jsonl(d / "scoring.jsonl") if cfg.get("resume") else []
    if any(hashes.get(s["transition_id"]) != s.get("item_hash") for s in saved):
        raise SystemExit(f"{run_id}: saved records belong to different items; rerun without --resume.")
    done = {(s["transition_id"], s["condition"]) for s in saved}
    todo = [(t, a) for t, a in jobs if (t["id"], a) not in done]
    if todo:
        scorer = LogoddsScorer(SIZES[tag][0], model_tag=tag)
        for t, arm in todo:
            if arm in YN_ARMS:
                sc = scorer.score_one_token_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm),
                                                 o_plus="Yes", o_minus="No")
            else:
                sc = scorer.score_pair(transition_id=t["id"], condition=arm, messages=arm_messages(t, arm),
                                       o_plus=t["candidates"]["o_plus"], o_minus=t["candidates"]["o_minus"])
            append_jsonl(d / "scoring.jsonl", [{**sc.to_dict(), "game_id": t["game_id"], "item_hash": hashes[t["id"]]}])
        scorer.close()
    write_status(run_id, f"shard {i} of {n}: {len(jobs)} scores ({tag})", ok=False, reported="shard complete")
    return 0


def _mean_s(s: list[dict], items: list[dict], arm: str) -> float:
    ids = {t["id"] for t in items}
    v = [x["p_success"] for x in s if x["condition"] == arm and x["transition_id"] in ids]
    return sum(v) / len(v) if v else float("nan")


def _flips(s: list[dict], items: list[dict], a: str = "aligned", b: str = "swapped") -> float:
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s}
    pairs = [(by[(t["id"], a)], by[(t["id"], b)]) for t in items if (t["id"], a) in by and (t["id"], b) in by]
    return sum(1 for x, y in pairs if x > 0 > y) / len(pairs) if pairs else float("nan")


def model_block(s: list[dict], primary: list[dict], successes: list[dict]) -> dict[str, Any]:
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    h0 = [t for t in primary if t.get("history_len", 0) == 0]
    h1 = [t for t in primary if t.get("history_len", 0) > 0]
    seen = [t for t in primary if t.get("determination_reason") == "contents_or_open_state"]
    both = primary + successes
    block = {"goal_effect": _effect(s, primary), "aligned_vs_neutral": _effect(s, primary, primary="aligned", control="absent"),
             "goal_effect_history_0": _effect(s, h0), "goal_effect_history_1plus": _effect(s, h1),
             "goal_effect_contents_or_open": _effect(s, seen),
             "flips_to_success": _flips(s, primary), "flips_history_0": _flips(s, h0), "flips_history_1plus": _flips(s, h1),
             # the confirmatory definition: the sign of the first-token L changes (comparable with RQ2's tables)
             "decision_flip": decision_flip_share(s, primary),
             "mean_s_failures": {a: _mean_s(s, primary, a) for a in PRED_ARMS},
             "mean_s_successes": {a: _mean_s(s, successes, a) for a in ("aligned", "absent")},
             "auc": {a: rq1x._auc_ci(by, both, a) for a in ("aligned", "absent")},
             "auc_aligned_minus_neutral": rq1x._auc_diff(by, both, "aligned", "absent"),
             "yes_no": {"goal_effect": _effect(s, primary, primary="yn_aligned", control="yn_swapped"),
                        "goal_effect_history_0": _effect(s, h0, primary="yn_aligned", control="yn_swapped"),
                        "goal_effect_history_1plus": _effect(s, h1, primary="yn_aligned", control="yn_swapped"),
                        "goal_effect_contents_or_open": _effect(s, seen, primary="yn_aligned", control="yn_swapped"),
                        "yes_share_failures": {a: _mean_s(s, primary, a) for a in YN_ARMS},
                        "yes_share_successes": {a: _mean_s(s, successes, a) for a in ("yn_aligned", "yn_absent")},
                        "auc": {a: rq1x._auc_ci(by, both, a) for a in ("yn_aligned", "yn_absent")},
                        "auc_aligned_minus_neutral": rq1x._auc_diff(by, both, "yn_aligned", "yn_absent")}}
    return block


def _diffs_by_game(s: list[dict], items: list[dict], a: str, b: str) -> dict[str, list[float]]:
    by = {(x["transition_id"], x["condition"]): x["L"] for x in s if x["L"] == x["L"]}
    out: dict[str, list[float]] = defaultdict(list)
    for t in items:
        if (t["id"], a) in by and (t["id"], b) in by:
            out[t["game_id"]].append(by[(t["id"], a)] - by[(t["id"], b)])
    return dict(out)


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def size_slope(per_model: dict[str, dict[str, list[float]]], B: int = 1000) -> dict[str, Any]:
    """Slope of the mean goal effect against log(parameters), nats per e-fold of size. The game bootstrap
    resamples games jointly across sizes, since every size scores the same items."""
    tags = sorted(per_model, key=lambda m: SIZES[m][1])
    xs = [math.log(SIZES[m][1]) for m in tags]

    def slope(games: list[str]) -> float:
        ys = []
        for m in tags:
            v = [d for g in games for d in per_model[m].get(g, [])]
            ys.append(sum(v) / len(v) if v else float("nan"))
        return _ols_slope(xs, ys)

    games = sorted({g for m in tags for g in per_model[m]})
    est = slope(games)
    rng = random.Random(C.RNG_SEED)
    boots = sorted(slope([rng.choice(games) for _ in games]) for _ in range(B))
    return {"estimate": est, "ci": [boots[int(0.025 * B)], boots[int(0.975 * B) - 1]], "models": tags,
            "log_params": xs}


def run_size_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    sets = rq1x.rq1x_sets(load_items(tuple(cfg.get("sources", SOURCES))))
    report: dict[str, Any] = {"items": {"n_primary": len(sets["primary"]), "n_successes": len(sets["successes"])},
                              "models": {}, "waiting": {}}
    per_model: dict[str, dict] = {}
    for tag in cfg["models"]:
        folders, problems = complete_shards(f"size_score_{tag}")
        if problems:
            report["waiting"][tag] = problems
            continue
        s = [x for p in folders for x in read_jsonl(p / "scoring.jsonl") if x["model"] == tag]
        report["models"][tag] = {"params_b": SIZES[tag][1], **model_block(s, sets["primary"], sets["successes"])}
        per_model[tag] = {arms: _diffs_by_game(s, sets["primary"], *arms)
                          for arms in (("aligned", "swapped"), ("yn_aligned", "yn_swapped"))}
    if len(per_model) >= 3:
        report["slope_vs_log_params"] = {arms[0]: size_slope({m: v[arms] for m, v in per_model.items()})
                                         for arms in (("aligned", "swapped"), ("yn_aligned", "yn_swapped"))}
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "Model-size series, Qwen3 dense (exploratory)", report)
    lines = [f"{tag} ({m['params_b']}B): dL {m['goal_effect']['delta_L_nats']:.2f}, flips {m['flips_to_success']:.3f}, "
             f"yn dL {m['yes_no']['goal_effect']['delta_L_nats']:.2f}" for tag, m in report["models"].items()]
    ok = not report["waiting"]
    write_status(run_id, "\n".join(lines) + (f"\nwaiting: {sorted(report['waiting'])}" if not ok else ""), ok=False,
                 reported="size report" if ok else "partial size report")
    return 0
