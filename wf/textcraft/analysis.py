"""Analysis of the TextCraft replication (notes/textcraft_design.md, sections 1 and 6).

Primary: ΔL = mean of L(aligned) - L(swapped) on the main population, per model, with a 95% percentile bootstrap
that resamples goal pairs and a sign-flip permutation p that flips whole goal pairs, Holm-adjusted over the two
models. The prediction is confirmed for a model when ΔL > 0, the interval excludes zero and the Holm p < 0.01.
A null is read with equivalence margins (TOST on the 90% pair-bootstrap interval: 1.0 nats for Qwen, 0.3 for
Llama; design section 12, C4). Seven named readouts with written predictions (section 12, C7), Holm-adjusted within
a model: the goal role, ingredients whose name shares a word with the goal, usefulness-only ingredients, the
history-goal agreement contrast, the inventory-shown contrast, the yes/no effect, and the counting readout. A
competence check on matched determined successes (C3). Everything else (strata, sensitivity analyses, class
contrasts, the prefix decomposition) is descriptive: estimates and intervals, no p-values. Scores are joined to items by id and item hash; a score whose item has changed
stops the analysis, and a missing score on any arm of any item withholds the verdict.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from wf import constants as C
from wf.analysis.report import write_run_report
from wf.analysis.stats import cluster_bootstrap, holm, pair_permutation, prereg_rq2_verdict
from wf.rq2 import complete_shards
from wf.runs import _init_run
from wf.textcraft import fit as F
from wf.textcraft.prompts import ARMS
from wf.textcraft.score import CO_PRIMARY, item_hash, load_items, verified_files
from wf.util import read_jsonl, require_filed, sigmoid, write_json, write_status

P_MAX = 0.01
EQUIVALENCE_MARGIN = {"qwen3-8b-fewshot": 1.0, "llama3.1-8b-fewshot": 0.3}  # nats (design 12, C4)


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


def _cluster_boots(stat, rows: list, pairs: list[str], n: int = 2000) -> list[float]:
    """The statistic on n resamples of whole goal pairs (as cluster_bootstrap, same seed), sorted, NaN dropped."""
    import random
    rng = random.Random(C.RNG_SEED)
    by: dict[str, list] = defaultdict(list)
    for r, p in zip(rows, pairs):
        by[p].append(r)
    keys = list(by)
    boots = []
    for _ in range(n if keys else 0):
        boots.append(stat([r for k in rng.choices(keys, k=len(keys)) for r in by[k]]))
    return sorted(b for b in boots if b == b)


def _pct(boots: list[float], q: float) -> float:
    return boots[min(len(boots) - 1, int(q * len(boots)))] if boots else float("nan")


def equivalence(scores: list[dict], items: list[dict], margin: float) -> dict[str, Any]:
    """TOST at the 5% level on each side: ΔL lies within ±margin when its 90% pair-bootstrap interval does."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    rows, pairs = [], []
    for t in items:
        u, v = _value(by.get((t["id"], "aligned")), "L"), _value(by.get((t["id"], "swapped")), "L")
        if u is not None and v is not None:
            rows.append(u - v)
            pairs.append(t["pair_id"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    boots = _cluster_boots(mean, rows, pairs)
    ci90 = [_pct(boots, 0.05), _pct(boots, 0.95)]
    return {"margin": margin, "ci90": ci90, "equivalent": bool(boots) and -margin < ci90[0] and ci90[1] < margin}


def couples_where(items: list[dict], pred) -> list[dict]:
    """The items of the couples whose two members both satisfy pred (design 12, C2: readouts on whole couples)."""
    by: dict[str, list[dict]] = defaultdict(list)
    for t in items:
        by[t["couple_id"]].append(t)
    return [t for its in by.values() if len(its) == 2 and all(pred(x) for x in its) for t in its]


def not_used_up_goal(t: dict) -> bool:
    return not (t["failure_class"] == "used_up" and t["role"] == "goal")


def agreement_contrast(scores: list[dict], main: list[dict]) -> dict[str, Any]:
    """Half the difference between ΔL on items whose history favours their aligned goal and on items whose history
    favours the partner (history_fit, section 11 A3). In a couple the two are mirrored, so if the history adds c to
    L when the goal line agrees with it, this estimates c (design 12, C1)."""
    r = contrast(scores, [t for t in main if t["history_fit"] == "aligned"], [t for t in main if t["history_fit"] == "swapped"])
    return {**r, "estimate": r["estimate"] / 2, "ci": [x / 2 for x in r["ci"]]}


def shows_inventory(t: dict) -> bool:
    return any(o.startswith("Inventory: ") for _a, o in t["history"])


def inventory_contrast(scores: list[dict], main: list[dict]) -> dict[str, Any]:
    """ΔL on items whose history shows an inventory reply minus ΔL on items whose history shows none, within each
    length bin, averaged with the bins' item counts as weights (design 12, C5). The walk looks at the inventory by
    a coin flip, so within a bin the two groups differ by chance only."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    rows, pairs = [], []
    for t in main:
        u, v = _value(by.get((t["id"], "aligned")), "L"), _value(by.get((t["id"], "swapped")), "L")
        if u is not None and v is not None:
            rows.append((t["length_bin"], shows_inventory(t), u - v))
            pairs.append(t["pair_id"])

    def stat(rs):
        total, acc = 0, 0.0
        for b in sorted({x[0] for x in rs}):
            yes = [d for bb, sh, d in rs if bb == b and sh]
            no = [d for bb, sh, d in rs if bb == b and not sh]
            if yes and no:
                w = len(yes) + len(no)
                acc += w * (sum(yes) / len(yes) - sum(no) / len(no))
                total += w
        return acc / total if total else float("nan")
    boots = _cluster_boots(stat, rows, pairs)
    return {"estimate": stat(rows), "ci": [_pct(boots, 0.025), _pct(boots, 0.975)],
            "n_items": [sum(sh for _b, sh, _d in rows), sum(not sh for _b, sh, _d in rows)]}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a success scores above a failure), ties counting half."""
    if not pos or not neg:
        return float("nan")
    neg_sorted = sorted(neg)
    import bisect
    return sum(bisect.bisect_left(neg_sorted, x) + 0.5 * (bisect.bisect_right(neg_sorted, x) - bisect.bisect_left(neg_sorted, x))
               for x in pos) / (len(pos) * len(neg))


def competence(scores: list[dict], successes: list[dict], failures: list[dict], arm: str = "neutral") -> dict[str, Any]:
    """Does the model tell the determined successes from the determined failures (design 12, C3)? AUC of L under
    one goal line, overall and per failure class, with pair-bootstrap intervals; the L gap that standardizes ΔL."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    L = lambda t: _value(by.get((t["id"], arm)), "L")
    out: dict[str, Any] = {}
    for cls in ("all",) + tuple(sorted({t["failure_class"] for t in failures})):
        neg_items = [t for t in failures if cls == "all" or t["failure_class"] == cls]
        rows = [(1, L(t)) for t in successes if L(t) is not None] + [(0, L(t)) for t in neg_items if L(t) is not None]
        pairs = [t["pair_id"] for t in successes if L(t) is not None] + [t["pair_id"] for t in neg_items if L(t) is not None]
        stat = lambda rs: auc([x for y, x in rs if y], [x for y, x in rs if not y])
        boots = _cluster_boots(stat, rows, pairs)
        a = stat(rows)
        out[cls] = {"auc": a, "ci": [_pct(boots, 0.025), _pct(boots, 0.975)],
                    "discriminates": bool(boots) and a >= 0.6 and _pct(boots, 0.025) > 0.5}
    ls = [L(t) for t in successes if L(t) is not None]
    lf = [L(t) for t in failures if L(t) is not None]
    out["mean_L_success"] = statistics.fmean(ls) if ls else float("nan")
    out["mean_L_failure"] = statistics.fmean(lf) if lf else float("nan")
    out["gap"] = out["mean_L_success"] - out["mean_L_failure"]
    return out


def named_readouts(scores: list[dict], main: list[dict], usefulness: list[dict]) -> dict[str, Any]:
    """The seven named readouts (design 12, C7); used-up items on the goal role are left out of all of them."""
    core = [t for t in main if not_used_up_goal(t)]
    ingredient = lambda t: t["role"] == "ingredient"
    out = {
        "goal_role": effect(scores, [t for t in core if t["role"] == "goal"], "aligned", "swapped"),
        "ingredient_with_overlap": effect(scores, couples_where(core, lambda t: ingredient(t) and t["object_goal_overlap"]),
                                          "aligned", "swapped"),
        "usefulness_only": effect(scores, couples_where(core, B_usefulness_only) + usefulness, "aligned", "swapped"),
        "history_agreement": agreement_contrast(scores, core),
        "inventory_shown": inventory_contrast(scores, core),
        "yes_no": effect(scores, core, "yn_aligned", "yn_swapped"),
        "counting": effect(scores, arithmetic_items(core), "aligned", "swapped"),
    }
    with_p = {k: v["perm_p"] for k, v in out.items() if "perm_p" in v}
    for k, p in holm(with_p).items():
        out[k]["holm_p_within_model"] = p
    return out


def B_usefulness_only(t: dict) -> bool:
    """As build.usefulness_only: an ingredient whose name shares no word with either goal."""
    return t["role"] == "ingredient" and not t["object_goal_overlap"] and not t["object_partner_overlap"]


def descriptive(r: Any) -> Any:
    """The same readouts without p-values: strata and sensitivity analyses are descriptive (design 12, C7)."""
    if isinstance(r, dict):
        return {k: descriptive(v) for k, v in r.items() if k != "perm_p"}
    return r


def complete(scores: list[dict], items: list[dict], arms: tuple[str, ...]) -> list[dict]:
    """The items that have a usable score on every one of these arms."""
    have = {(s["transition_id"], s["condition"]) for s in scores if s.get("L") == s.get("L")}
    return [t for t in items if all((t["id"], a) in have for a in arms)]


PREFIX_SET = ("aligned", "swapped", "prefix_aligned", "prefix_swapped")


def ols_intercept(rows: list[tuple[float, tuple[int, ...]]]) -> float:
    """Intercept of the least-squares fit of d on the covariates (minimum-norm when they are collinear)."""
    if not rows:
        return float("nan")
    y = np.array([d for d, _x in rows], dtype=float)
    X = np.array([(1.0, *x) for _d, x in rows], dtype=float)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def fit_regression(scores: list[dict], items: list[dict], a: str = "aligned", b: str = "swapped") -> dict[str, Any]:
    """value(a) - value(b) regressed on the signs of the ten fit measures (design 12, B1). The intercept is the
    effect at zero measured fit; its interval resamples goal pairs."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    rows, pairs = [], []
    for t in items:
        u, v = _value(by.get((t["id"], a)), "L"), _value(by.get((t["id"], b)), "L")
        if u is None or v is None:
            continue
        sg = F.signs(t["fit_measures"])
        rows.append((u - v, tuple(sg[m] for m in F.MEASURES)))
        pairs.append(t["pair_id"])
    boot = cluster_bootstrap(ols_intercept, rows, pairs, seed=C.RNG_SEED)
    coef = (np.linalg.lstsq(np.array([(1.0, *x) for _d, x in rows]), np.array([d for d, _x in rows]), rcond=None)[0]
            if rows else [float("nan")] * (1 + len(F.MEASURES)))
    return {"intercept": float(coef[0]), "ci": [boot["ci_low"], boot["ci_high"]],
            "slopes": {m: float(c) for m, c in zip(F.MEASURES, coef[1:])},
            "mean_covariate": {m: (sum(x[k] for _d, x in rows) / len(rows) if rows else float("nan"))
                               for k, m in enumerate(F.MEASURES)},
            "n_items": len(rows), "n_pairs": len(set(pairs))}


def no_fit_for_aligned(items: list[dict]) -> list[dict]:
    """Items whose history speaks for the aligned goal on no fit measure (each neutral or for the partner)."""
    return [t for t in items if F.none_favours_aligned(t["fit_measures"])]


def contrast(scores: list[dict], first: list[dict], second: list[dict]) -> dict[str, Any]:
    """ΔL on one stratum minus ΔL on another, with an interval that resamples goal pairs."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    rows, pairs = [], []
    for group, its in ((0, first), (1, second)):
        for t in its:
            u, v = _value(by.get((t["id"], "aligned")), "L"), _value(by.get((t["id"], "swapped")), "L")
            if u is not None and v is not None:
                rows.append((group, u - v))
                pairs.append(t["pair_id"])

    def stat(rs):
        g0, g1 = [d for g, d in rs if g == 0], [d for g, d in rs if g == 1]
        return sum(g0) / len(g0) - sum(g1) / len(g1) if g0 and g1 else float("nan")
    boot = cluster_bootstrap(stat, rows, pairs, seed=C.RNG_SEED)
    return {"estimate": stat(rows), "ci": [boot["ci_low"], boot["ci_high"]],
            "n_items": [sum(g == 0 for g, _d in rows), sum(g == 1 for g, _d in rows)]}


def class_contrasts(scores: list[dict], main: list[dict]) -> dict[str, Any]:
    """Pre-logged class contrasts within one role (design 12, B2): used up, too few and never obtained within the
    ingredient role; too few against never obtained within the goal role. Never pooled over roles."""
    cell = lambda c, r: [t for t in main if t["failure_class"] == c and t["role"] == r]
    return {"ingredient|used_up_minus_too_few": contrast(scores, cell("used_up", "ingredient"), cell("too_few", "ingredient")),
            "ingredient|used_up_minus_never_obtained": contrast(scores, cell("used_up", "ingredient"), cell("never_obtained", "ingredient")),
            "ingredient|too_few_minus_never_obtained": contrast(scores, cell("too_few", "ingredient"), cell("never_obtained", "ingredient")),
            "goal|too_few_minus_never_obtained": contrast(scores, cell("too_few", "goal"), cell("never_obtained", "goal"))}


def arithmetic_items(main: list[dict]) -> list[dict]:
    """The named arithmetic readout's items: every short input was obtained at some point, and the history does not
    show the inventory after its last change (design 11, A2, amended in 12, B6)."""
    return [t for t in main if t["needs_arithmetic"] and not t["inventory_shown_after_last_change"]]


def sensitivity(scores: list[dict], items: list[dict]) -> dict[str, Any]:
    sub = no_fit_for_aligned(items)
    return {"regression_intercept": fit_regression(scores, items),
            "delta_L_no_fit_for_aligned": effect(scores, sub, "aligned", "swapped"),
            "swapped_minus_neutral_on_fit": fit_regression(scores, items, "swapped", "neutral"),
            "aligned_minus_neutral_on_fit": fit_regression(scores, items, "aligned", "neutral")}


def level(scores: list[dict], items: list[dict], arm: str) -> dict[str, float]:
    ids = {t["id"] for t in items}
    vals = [s["L"] for s in scores if s["condition"] == arm and s["transition_id"] in ids and s["L"] == s["L"]]
    return {"mean_L": statistics.fmean(vals) if vals else float("nan"),
            "mean_p_success": statistics.fmean(sigmoid(v) for v in vals) if vals else float("nan"), "n": len(vals)}


def readouts(scores: list[dict], items: list[dict]) -> dict[str, Any]:
    """Every exploratory comparison on one population."""
    same = complete(scores, items, PREFIX_SET)  # ΔL, its prefix part and the remainder on one item set (B3)
    return {
        "delta_L": effect(scores, items, "aligned", "swapped"),
        "delta_L_first": effect(scores, items, "aligned", "swapped", "L_first"),
        "delta_p_pp": effect(scores, items, "aligned", "swapped", "p"),
        "delta_L_on_prefix_set": effect(scores, same, "aligned", "swapped"),
        "delta_L_prefix": effect(scores, same, "prefix_aligned", "prefix_swapped"),
        "delta_L_name_remainder": effect_of(scores, same, name_remainder),
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
    "inventory_shown_after_last_change": ("inventory_shown_after_last_change",),
    "role_x_class_x_needs_arithmetic": ("role", "failure_class", "needs_arithmetic"),
}


def per_model_gap(scores: list[dict], successes: list[dict], failures: list[dict]) -> float:
    """Mean L under the neutral line on successes minus on failures: the standardizer of ΔL (design 12, C3)."""
    by = {(s["transition_id"], s["condition"]): s for s in scores}
    ls = [v for t in successes if (v := _value(by.get((t["id"], "neutral")), "L")) is not None]
    lf = [v for t in failures if (v := _value(by.get((t["id"], "neutral")), "L")) is not None]
    gap = statistics.fmean(ls) - statistics.fmean(lf) if ls and lf else float("nan")
    return gap if gap and gap == gap else float("nan")


def strata(scores: list[dict], items: list[dict], names: tuple[str, ...] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fields in STRATA.items():
        if names is not None and name not in names:
            continue
        groups = defaultdict(list)
        for t in items:
            groups["|".join(str(t[f]) for f in fields)].append(t)
        out[name] = {k: {"delta_L": effect(scores, v, "aligned", "swapped"),
                         "aligned_minus_neutral": effect(scores, v, "aligned", "neutral"),
                         "yes_no": effect(scores, v, "yn_aligned", "yn_swapped"),
                         "delta_L_prefix": effect(scores, complete(scores, v, PREFIX_SET), "prefix_aligned", "prefix_swapped")}
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
    pops = {pop: [t for t in items if t["population"] == pop]
            for pop in ("main", "usefulness", "success", "secondary_get", "empty_history")}
    models = [m for m in CO_PRIMARY if m in cfg.get("models", CO_PRIMARY)]
    have = {(s["model"], s["transition_id"], s["condition"]) for s in scores if s.get("L") == s.get("L")}
    missing_primary = {(m, t["id"], a) for m in models for t in pops["main"] for a in ("aligned", "swapped")} - have
    missing = {(m, t["id"], a) for m in models for t in items for a in ARMS} - have  # every arm of every item (B3)
    per_model, primary = {}, {}
    for m in models:
        mine = [s for s in scores if s["model"] == m]
        primary[m] = effect(mine, pops["main"], "aligned", "swapped")
        failures = pops["main"] + pops["usefulness"]
        per_model[m] = {"primary_delta_L": primary[m],
                        "equivalence": equivalence(mine, pops["main"], EQUIVALENCE_MARGIN[m]),
                        "named": named_readouts(mine, pops["main"], pops["usefulness"]),
                        "competence": {arm: competence(mine, pops["success"], failures, arm) for arm in ("neutral", "aligned")},
                        "successes": descriptive({**readouts(mine, pops["success"]), "strata": strata(mine, pops["success"], ("role", "length_bin"))}),
                        "descriptive": descriptive({
                            "primary_without_used_up_on_goal": effect(mine, [t for t in pops["main"] if not_used_up_goal(t)], "aligned", "swapped"),
                            "needs_arithmetic_as_logged_in_A2": effect(mine, [t for t in pops["main"] if t["needs_arithmetic"]], "aligned", "swapped"),
                            "sensitivity_history_fit": sensitivity(mine, pops["main"]),
                            "class_contrasts_within_role": class_contrasts(mine, [t for t in pops["main"] if not_used_up_goal(t)]),
                            "standardized_delta_L": primary[m]["estimate"] / per_model_gap(mine, pops["success"], failures),
                            "main": {**readouts(mine, pops["main"]), "strata": strata(mine, pops["main"])},
                            "usefulness": readouts(mine, pops["usefulness"]),
                            "secondary_get": readouts(mine, pops["secondary_get"]),
                            "empty_history": readouts(mine, pops["empty_history"])})}
    report = {"hierarchy": "primary (Holm over models, confirmed rule, TOST for a null); then named readouts (Holm "
                           "within model); everything under 'descriptive', 'successes' and the strata has no p-values",
              "items": {k: len(v) for k, v in pops.items()} | {"pairs_main": len({t["pair_id"] for t in pops["main"]}),
                                                                 "missing_primary_scores": len(missing_primary),
                                                                 "missing_scores_any_arm": len(missing)},
              "verdicts": verdicts(primary), "per_model": per_model}
    write_json(d / "metrics.json", report)
    write_run_report(run_id, "TextCraft replication of the RQ2 goal effect", report)
    lines = [f"{m}: dL={primary[m]['estimate']:.3f} CI={primary[m]['ci']} holm_p={report['verdicts'][m]['holm_p']} "
             f"confirmed={report['verdicts'][m]['confirmed']} equivalent_within_{EQUIVALENCE_MARGIN[m]}="
             f"{per_model[m]['equivalence']['equivalent']}" for m in models]
    write_status(run_id, f"items={report['items']}\n" + "\n".join(lines) + f"\nreplicates: {report['verdicts']['replicates']}",
                 ok=False, reported="TextCraft verdict" if not missing else None)
    return 0 if not missing else 1
