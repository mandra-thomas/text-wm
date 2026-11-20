"""Analysis of the TextCraft replication (notes/textcraft_design.md, sections 1 and 6).

Primary: ΔL = mean of L(aligned) - L(swapped) on the main population, per model, with a 95% percentile bootstrap
that resamples goal pairs and a sign-flip permutation p that flips whole goal pairs, Holm-adjusted over the two
models. The prediction is confirmed for a model when ΔL > 0, the interval excludes zero and the Holm p < 0.01.
Named secondary readouts, logged before any score (design section 11): ΔL on the main items that need arithmetic
(A2), and the prefix-only ΔL with its name-token remainder (A4). Everything else is exploratory. Scores are joined
to items by id and item hash; a score whose item has changed stops the analysis.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation, prereg_rq2_verdict
from wf.rq2 import complete_shards
from wf.runs import _init_run
from wf.textcraft.score import CO_PRIMARY, item_hash, load_items, verified_files
from wf.util import read_jsonl, require_filed, sigmoid, write_json, write_status

P_MAX = 0.01


def _value(s: dict | None, value: str) -> float | None:
    if not s:
        return None
    v = sigmoid(s["L"]) * 100.0 if value == "p" else s.get(value)
    return None if v is None or v != v else v


def effect(scores: list[dict], items: list[dict], a: str, b: str, value: str = "L") -> dict[str, Any]:
    """Mean of value(a) - value(b) over items with both, clustered by goal pair."""
    def diff(by, tid):
        u, v = _value(by.get((tid, a)), value), _value(by.get((tid, b)), value)
        return None if u is None or v is None else u - v
    return effect_of(scores, items, diff)


def name_remainder(by: dict, tid: str) -> float | None:
    """[L(aligned) - L(prefix_aligned)] - [L(swapped) - L(prefix_swapped)]: the part of ΔL carried by the object
    name's tokens after each reply's prefix (design section 11, A4)."""
    v = [_value(by.get((tid, arm)), "L") for arm in ("aligned", "prefix_aligned", "swapped", "prefix_swapped")]
    return None if None in v else (v[0] - v[1]) - (v[2] - v[3])


def effect_of(scores: list[dict], items: list[dict], diff) -> dict[str, Any]:
    """Mean over items of diff(scores by (item, arm), item id), skipping None, clustered by goal pair."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    diffs, pairs = [], []
    for t in items:
        d = diff(by, t["id"])
        if d is None:
            continue
        diffs.append(d)
        pairs.append(t["pair_id"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    boot = cluster_bootstrap(mean, diffs, pairs, seed=C.RNG_SEED)
    return {"estimate": mean(diffs), "ci": [boot["ci_low"], boot["ci_high"]],
            "perm_p": pair_permutation(diffs, seed=C.RNG_SEED, cluster_ids=pairs) if diffs else float("nan"),
            "n_items": len(diffs), "n_pairs": len(set(pairs))}


def level(scores: list[dict], items: list[dict], arm: str) -> dict[str, float]:
    ids = {t["id"] for t in items}
    vals = [s["L"] for s in scores if s["condition"] == arm and s["transition_id"] in ids and s["L"] == s["L"]]
    return {"mean_L": statistics.fmean(vals) if vals else float("nan"),
            "mean_p_success": statistics.fmean(sigmoid(v) for v in vals) if vals else float("nan"), "n": len(vals)}


def readouts(scores: list[dict], items: list[dict]) -> dict[str, Any]:
    """Every exploratory comparison on one population."""
    return {
        "delta_L": effect(scores, items, "aligned", "swapped"),
        "delta_L_first": effect(scores, items, "aligned", "swapped", "L_first"),
        "delta_p_pp": effect(scores, items, "aligned", "swapped", "p"),
        "delta_L_prefix": effect(scores, items, "prefix_aligned", "prefix_swapped"),
        "delta_L_name_remainder": effect_of(scores, items, name_remainder),
        "aligned_minus_neutral": effect(scores, items, "aligned", "neutral"),
        "swapped_minus_neutral": effect(scores, items, "swapped", "neutral"),
        "yes_no_aligned_minus_swapped": effect(scores, items, "yn_aligned", "yn_swapped"),
        "yes_no_aligned_minus_neutral": effect(scores, items, "yn_aligned", "yn_neutral"),
        "levels": {arm: level(scores, items, arm) for arm in ("aligned", "swapped", "neutral", "yn_aligned", "yn_swapped",
                                                             "yn_neutral", "prefix_aligned", "prefix_swapped")},
    }


STRATA = {  # name -> the item fields whose values define the stratum (design sections 6 and 11)
    "failure_class": ("failure_class",), "length_bin": ("length_bin",), "role": ("role",),
    "role_x_class": ("role", "failure_class"), "history_fit": ("history_fit",),
    "class_x_history_fit": ("failure_class", "history_fit"), "pure_class": ("pure_class",),
    "needs_arithmetic": ("needs_arithmetic",), "role_x_object_goal_overlap": ("role", "object_goal_overlap"),
    "object_partner_overlap": ("object_partner_overlap",),
}


def strata(scores: list[dict], items: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fields in STRATA.items():
        groups = defaultdict(list)
        for t in items:
            groups["|".join(str(t[f]) for f in fields)].append(t)
        out[name] = {k: {"delta_L": effect(scores, v, "aligned", "swapped"),
                         "aligned_minus_neutral": effect(scores, v, "aligned", "neutral"),
                         "yes_no": effect(scores, v, "yn_aligned", "yn_swapped"),
                         "delta_L_prefix": effect(scores, v, "prefix_aligned", "prefix_swapped")}
                     for k, v in sorted(groups.items())}
    return out


def stale_scores(scores: list[dict], items: list[dict]) -> list[str]:
    """Scores that do not belong to the current items: unknown id, or an item hash other than the item's now."""
    hashes = {t["id"]: item_hash(t) for t in items}
    return sorted({f"{s['transition_id']} ({'unknown item' if s['transition_id'] not in hashes else 'item changed'})"
                   for s in scores if hashes.get(s["transition_id"]) != s.get("item_hash")})


def verdicts(primary: dict[str, dict]) -> dict[str, Any]:
    """Holm over both models (a model without scores counts as p = 1); confirmed when ΔL > 0, CI > 0, Holm p < 0.01."""
    adjusted = holm({m: primary[m]["perm_p"] if m in primary and primary[m]["perm_p"] == primary[m]["perm_p"] else 1.0
                     for m in CO_PRIMARY})
    out = {}
    for m in CO_PRIMARY:
        r = primary.get(m, {"estimate": float("nan"), "ci": [float("nan")] * 2})
        ok = r["estimate"] > 0 and r["ci"][0] > 0 and adjusted[m] < P_MAX
        out[m] = {"confirmed": bool(ok), "holm_p": adjusted[m],
                  "rq2_magnitude_rule": prereg_rq2_verdict(r["estimate"], r["ci"], adjusted[m])}
    out["replicates"] = all(out[m]["confirmed"] for m in CO_PRIMARY)
    return out


def run_textcraft_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    require_filed(run_id)
    folders, problems = complete_shards(cfg["source"])
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=tuple(p / "scoring.jsonl" for p in folders if (p / "scoring.jsonl").exists()))
    files = verified_files(cfg)
    if isinstance(files, str):
        problems.append(f"items not verified: {files}")
    if problems:
        write_status(run_id, "waits for every scoring shard and verified items:\n" + "\n".join(problems), ok=False)
        return 1
    items = load_items(files[0])
    scores = [s for p in folders for s in read_jsonl(p / "scoring.jsonl")]
    stale = stale_scores(scores, items)
    if stale:
        write_status(run_id, f"{len(stale)} scored items do not match the verified items (rescore them):\n"
                     + "\n".join(stale[:20]), ok=False)
        return 1
    pops = {pop: [t for t in items if t["population"] == pop] for pop in ("main", "secondary_get", "empty_history")}
    models = [m for m in CO_PRIMARY if m in cfg.get("models", CO_PRIMARY)]
    wanted = {(m, t["id"], a) for m in models for t in pops["main"] for a in ("aligned", "swapped")}
    missing = wanted - {(s["model"], s["transition_id"], s["condition"]) for s in scores}
    per_model, primary = {}, {}
    for m in models:
        mine = [s for s in scores if s["model"] == m]
        primary[m] = effect(mine, pops["main"], "aligned", "swapped")
        per_model[m] = {"primary_delta_L": primary[m],
                        "secondary_delta_L_needs_arithmetic": effect(
                            mine, [t for t in pops["main"] if t["needs_arithmetic"]], "aligned", "swapped"),
                        "main": {**readouts(mine, pops["main"]), "strata": strata(mine, pops["main"])},
                        "secondary_get": readouts(mine, pops["secondary_get"]),
                        "empty_history": readouts(mine, pops["empty_history"])}
    report = {"items": {k: len(v) for k, v in pops.items()} | {"pairs_main": len({t["pair_id"] for t in pops["main"]}),
                                                                 "missing_primary_scores": len(missing)},
              "verdicts": verdicts(primary), "per_model": per_model}
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "TextCraft replication of the RQ2 goal effect", report)
    lines = [f"{m}: dL={primary[m]['estimate']:.3f} CI={primary[m]['ci']} holm_p={report['verdicts'][m]['holm_p']} "
             f"confirmed={report['verdicts'][m]['confirmed']}" for m in models]
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines) + f"\nreplicates: {report['verdicts']['replicates']}",
                 ok=False, reported="TextCraft verdict" if not missing else None)
    return 0 if not missing else 1
