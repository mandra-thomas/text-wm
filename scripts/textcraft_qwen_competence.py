"""Exploratory, post hoc: Qwen3-8B as a TextCraft world model, with Llama-3.1-8B alongside
(notes/textcraft_qwen_competence.md).

Nothing here scores anything. It reuses, unchanged, every function of scripts/textcraft_llama_competence.py (the
heuristics H1-H10 and H2 and H9 as fixed in 9a83738 and implemented in 6522784, the resampling, AUC, kappa,
regression, calibration and families-rule code) and adds:

- H11 (a repeated craft fails), H2 and H11, H6 and H11, fixed in the note (commit e225010) before any fit;
- the families competence rule per role, per class and on each round-2 contrast's population;
- where each model's errors fall, and which heuristics agree with them;
- the repetition cue with surplus held fixed, and three-term regressions;
- the goal effect against neutral confidence and correctness (log-odds, probability, decisions, discrimination);
- the same on the inventory-shown, clean and clean + inventory prompts (textcraft_dx_prompts).

Intervals: 95% percentile, 2,000 resamples of whole goal pairs, one shared set of weights (as in the Llama script).

Usage:
    python scripts/textcraft_qwen_competence.py --scores <textcraft_score dir> --prompts <textcraft_dx_prompts dir> \
        --out notes/textcraft_qwen_competence_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textcraft_llama_competence as tlc  # noqa: E402
from textcraft_llama_competence import (  # noqa: E402
    BINS, CLASSES, MODELS, ROLES, SHORT, Boot, _ci, _r, auc_ci, calibration, competence_block, competence_vs_effect,
    decisions_block, families_rule, features, heuristic_fits, kappa_ci, load_tc_scores, mean_ci, ols_ci, quantities,
    read_jsonl, sigmoid, spearman_ci, tc_table, wauc)
from wf.textcraft.checker import GENERIC_MEMBERS, parse_craft  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NEW_H = ("H11", "H2andH11", "H6andH11")
CONF = 4.595  # |L| above log(99): the model gives the outcome at least 99%
FAIL_BINS = (("below -15", -np.inf, -15), ("-15 to -10", -15, -10), ("-10 to -4.6", -10, -CONF),
             ("-4.6 to 0", -CONF, 0), ("above 0 (wrong)", 0, np.inf))
SUCC_BINS = (("below 0 (wrong)", -np.inf, 0), ("0 to 4.6", 0, CONF), ("4.6 to 10", CONF, 10), ("above 10", 10, np.inf))
CELLS = [("goal", "never_obtained"), ("goal", "too_few"), ("ingredient", "never_obtained"),
         ("ingredient", "too_few"), ("ingredient", "used_up")]


# ---------------------------------------------------------------- features added for this note

def add_features(item: dict, f: dict) -> dict:
    f = dict(f)
    f["H11"] = 1 - f["posthoc_repeat"]
    f["H2andH11"] = int(f["H2"] and f["H11"])
    f["H6andH11"] = int(f["H6"] and f["H11"])
    _k, y, inputs = parse_craft(item["action"])
    f["output"] = y
    f["n_inputs"] = len(inputs)
    f["generic_input"] = int(any(n in GENERIC_MEMBERS for _c, n in inputs))
    return f


# ---------------------------------------------------------------- small helpers

def sub(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if (r[k] in v if isinstance(v, (tuple, list, set)) else r[k] == v)]
    return out


def share_ci(rows, pred, bt):
    return mean_ci([float(pred(r)) for r in rows], [r["pair"] for r in rows], bt)


def brier_ci(L, y, clusters, bt) -> dict:
    """Brier of sigmoid(L), the base-rate predictor's Brier (its own base rate), and skill 1 - model / base."""
    p, y = sigmoid(np.asarray(L, float)), np.asarray(y, float)
    se = (p - y) ** 2
    W = bt.W(clusters)
    sw = W.sum(1)
    bm = (W @ se) / sw
    yb = (W @ y) / sw
    bb = yb * (1 - yb)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = 1 - bm / bb
    b0, y0 = se.mean(), y.mean()
    n, c = len(y), len(set(clusters))
    return {"brier": _ci(b0, bm, n, c), "brier_base_rate": _ci(y0 * (1 - y0), bb, n, c),
            "skill": _ci(1 - b0 / (y0 * (1 - y0)), sk, n, c), "base_rate": _r(y0)}


# ---------------------------------------------------------------- question 1 additions

def q1_extra(rows, bt, rule) -> dict:
    out = {"yes_by_role_class": {}, "calibration_by_role": {}, "brier": {}, "families_rule_by_population": {}}
    for sc in ("neutral", "yn_neutral"):
        d = {}
        for ro in ROLES:
            for cl in CLASSES:
                F = sub(rows, y=0, role=ro, cls=cl)
                if len(F) >= 20:
                    d[f"failures|role={ro}|class={cl}"] = share_ci(F, lambda r: r[sc] > 0, bt)
            for bn in BINS:
                d[f"successes|role={ro}|bin={bn}"] = share_ci(sub(rows, y=1, role=ro, bin=bn), lambda r: r[sc] > 0, bt)
        out["yes_by_role_class"][sc] = d
        out["calibration_by_role"][sc] = {ro: calibration([r[sc] for r in sub(rows, role=ro)],
                                                          [r["y"] for r in sub(rows, role=ro)]) for ro in ROLES}
        out["brier"][sc] = {"all": brier_ci([r[sc] for r in rows], [r["y"] for r in rows], [r["pair"] for r in rows], bt)}
        for ro in ROLES:
            rs = sub(rows, role=ro)
            out["brier"][sc][f"role={ro}"] = brier_ci([r[sc] for r in rs], [r["y"] for r in rs], [r["pair"] for r in rs], bt)
    S = sub(rows, y=1)
    pops = {
        "all (the round-2 gate)": rows,
        "role=goal": sub(rows, role="goal"),
        "role=ingredient": sub(rows, role="ingredient"),
        "class=used_up vs all successes": sub(rows, cls="used_up") + S,
        "class=too_few vs all successes": sub(rows, cls="too_few") + S,
        "class=never_obtained vs all successes": sub(rows, cls="never_obtained") + S,
        "used_up ingredient vs ingredient successes (role-matched)": sub(rows, role="ingredient", cls=("used_up", "success")),
        "contrast Lambda (intermediate lift): ingredient failures vs ingredient successes": sub(rows, role="ingredient"),
        "contrasts Dn, Dc, goal part: goal failures without used-up vs goal successes":
            [r for r in sub(rows, role="goal") if r["cls"] != "used_up"],
        "contrast T (timing), goal part: goal rows at 1-8 steps": sub(rows, role="goal", bin=("1-3", "4-8")),
        "contrast T (timing), ingredient part: ingredient rows at 1-8 steps": sub(rows, role="ingredient", bin=("1-3", "4-8")),
    }
    for bn in BINS:
        pops[f"bin={bn}"] = sub(rows, bin=bn)
    for name, rs in pops.items():
        if not sub(rs, y=0) or not sub(rs, y=1):
            continue
        fr = families_rule(rs, bt, rule)
        fr["auc_two_candidate"] = auc_ci([r["neutral"] for r in rs], [r["y"] for r in rs], [r["pair"] for r in rs], bt)
        out["families_rule_by_population"][name] = fr
    return out


# ---------------------------------------------------------------- question 2 additions

def q2_errors(rows, feats, bt, heur) -> dict:
    """Where the model's neutral decisions are wrong, and whether a heuristic makes the same error."""
    out = {}
    for sc in ("neutral", "yn_neutral"):
        d = {}
        F, S = sub(rows, y=0), sub(rows, y=1)
        fp = [r for r in F if r[sc] > 0]
        fn = [r for r in S if r[sc] <= 0]
        d["n_false_success"], d["n_false_failure"] = len(fp), len(fn)
        d["false_success_by_cell"] = {}
        for ro in ROLES:
            for cl in CLASSES:
                for bn in BINS + ("all",):
                    c = [r for r in F if r["role"] == ro and r["cls"] == cl and (bn == "all" or r["bin"] == bn)]
                    if len(c) >= 20:
                        d["false_success_by_cell"][f"{ro}|{cl}|{bn}"] = {
                            "n": len(c), "rate": _r(np.mean([r[sc] > 0 for r in c])), "count": int(sum(r[sc] > 0 for r in c))}
        d["false_failure_by_group"] = {}
        for ro in ROLES:
            for rep in (0, 1):
                for sur in ("exactly enough", "spare"):
                    c = [r for r in S if r["role"] == ro and feats[r["id"]]["posthoc_repeat"] == rep
                         and ((feats[r["id"]]["surplus_min"] == 0) == (sur == "exactly enough"))]
                    if c:
                        d["false_failure_by_group"][f"{ro}|repeat={rep}|{sur}"] = {
                            "n": len(c), "rate": _r(np.mean([r[sc] <= 0 for r in c])), "count": int(sum(r[sc] <= 0 for r in c))}
            for bn in BINS:
                c = sub(S, role=ro, bin=bn)
                d["false_failure_by_group"][f"{ro}|bin={bn}"] = {"n": len(c), "rate": _r(np.mean([r[sc] <= 0 for r in c]))}
        for lab, grp in (("generic_input", 1), ("no_generic_input", 0)):
            d[f"error_rate|{lab}"] = {
                "failures": _r(np.mean([r[sc] > 0 for r in F if feats[r["id"]]["generic_input"] == grp])),
                "successes": _r(np.mean([r[sc] <= 0 for r in S if feats[r["id"]]["generic_input"] == grp])),
                "n_failures": sum(feats[r["id"]]["generic_input"] == grp for r in F),
                "n_successes": sum(feats[r["id"]]["generic_input"] == grp for r in S)}
        for k in (1, 2, 3):
            d[f"error_rate|n_inputs={k if k < 3 else '3+'}"] = {
                "failures": _r(np.mean([r[sc] > 0 for r in F if min(feats[r["id"]]["n_inputs"], 3) == k] or [np.nan])),
                "successes": _r(np.mean([r[sc] <= 0 for r in S if min(feats[r["id"]]["n_inputs"], 3) == k] or [np.nan]))}
        # heuristics' agreement with the errors: share of wrong items on which the heuristic makes the same call,
        # against the share among the same population's right items
        agree = {}
        for h in heur:
            if h == "H6":
                continue
            hv = lambda r: feats[r["id"]][h]
            fc = [r for r in F if r[sc] <= 0]
            sc_ok = [r for r in S if r[sc] > 0]
            agree[h] = {"false_success_h_says_success": _r(np.mean([hv(r) for r in fp])) if fp else None,
                        "correct_failures_h_says_success": _r(np.mean([hv(r) for r in fc])),
                        "false_failure_h_says_failure": _r(np.mean([1 - hv(r) for r in fn])) if fn else None,
                        "correct_successes_h_says_failure": _r(np.mean([1 - hv(r) for r in sc_ok]))}
        d["heuristic_agreement_with_errors"] = agree
        # errors by output item and by pair: concentration and over-dispersion
        err = [(r, (r[sc] > 0) != bool(r["y"])) for r in rows]
        by_out = defaultdict(list)
        for r, e in err:
            by_out[feats[r["id"]]["output"]].append(e)
        tops = sorted(((sum(v), len(v), k) for k, v in by_out.items()), reverse=True)[:12]
        d["errors_by_output_item_top12"] = [{"output": k, "errors": int(e), "n": n} for e, n, k in tops]
        n_err = sum(e for _r0, e in err)
        cum = np.cumsum(sorted((sum(v) for v in by_out.values()), reverse=True))
        d["n_output_items"] = len(by_out)
        d["share_of_errors_in_top10pct_outputs"] = _r(cum[max(1, len(by_out) // 10) - 1] / max(n_err, 1))
        d["share_of_items_in_those_outputs"] = _r(sum(n for _e, n, _k in sorted(
            ((sum(v), len(v), k) for k, v in by_out.items()), reverse=True)[:max(1, len(by_out) // 10)]) / len(rows))
        by_pair = defaultdict(list)
        for r, e in err:
            by_pair[r["pair"]].append(e)
        pbar = n_err / len(rows)
        obs = np.var([np.mean(v) for v in by_pair.values()])
        exp_ = np.mean([pbar * (1 - pbar) / len(v) for v in by_pair.values()])
        d["pair_error_dispersion_ratio"] = _r(obs / exp_)
        d["error_rate_all"] = _r(pbar)
        out[sc] = d
    return out


def q2_repeat(rows, feats, bt) -> dict:
    out = {}
    for sc in ("neutral", "yn_neutral"):
        d = {}
        S = sub(rows, y=1)
        for ro in ROLES:
            for rep in (0, 1):
                for sur in ("exactly enough", "spare"):
                    c = [r for r in S if r["role"] == ro and feats[r["id"]]["posthoc_repeat"] == rep
                         and ((feats[r["id"]]["surplus_min"] == 0) == (sur == "exactly enough"))]
                    d[f"successes|{ro}|repeat={rep}|{sur}"] = mean_ci([r[sc] for r in c], [r["pair"] for r in c], bt)
            for cons in (0, 1):
                c = [r for r in S if r["role"] == ro and feats[r["id"]]["posthoc_repeat"] == 0
                     and feats[r["id"]]["any_input_consumed_before"] == cons]
                d[f"successes|{ro}|repeat=0|input_consumed_before={cons}"] = mean_ci([r[sc] for r in c], [r["pair"] for r in c], bt)
        uu = sub(rows, y=0, cls="used_up")
        for rep in (0, 1):
            c = [r for r in uu if feats[r["id"]]["posthoc_repeat"] == rep]
            d[f"used_up|repeat={rep}"] = mean_ci([r[sc] for r in c], [r["pair"] for r in c], bt)
        # repeat-matched and surplus-matched AUCs, ingredient role
        ing = [r for r in sub(rows, role="ingredient") if r["y"] == 1 or r["cls"] == "used_up"]
        for lab, sel in (("both repeats", lambda r: feats[r["id"]]["posthoc_repeat"] == 1),
                         ("both repeats, successes exactly enough",
                          lambda r: feats[r["id"]]["posthoc_repeat"] == 1 and (r["y"] == 0 or feats[r["id"]]["surplus_min"] == 0)),
                         ("non-repeat successes vs all used-up", lambda r: r["y"] == 0 or feats[r["id"]]["posthoc_repeat"] == 0)):
            c = [r for r in ing if sel(r)]
            d[f"auc_used_up_vs_ingredient_successes|{lab}"] = auc_ci([r[sc] for r in c], [r["y"] for r in c], [r["pair"] for r in c], bt)
        # three-term regressions
        for lab, rs in (("all", rows), ("ingredient", sub(rows, role="ingredient")),
                        ("used_up_and_ingredient_successes", ing)):
            y = np.array([r["y"] for r in rs], float)
            L = np.array([r[sc] for r in rs], float)
            pr = [r["pair"] for r in rs]
            for hs in (("H2", "H11"), ("H1", "H11")):
                cols = [np.array([feats[r["id"]][h] for r in rs], float) for h in hs]
                X = np.c_[np.ones_like(y), y, *cols]
                d[f"regression|{lab}|truth+{'+'.join(hs)}"] = ols_ci(L, X, pr, bt, ["intercept", "truth", *hs])
        # paired tracking-AUC differences against the truth (H6) on the used-up subsets, same resamples (the reading
        # fixed in the note for H2 and H11)
        for lab, rs in (("used_up_and_all_successes", [r for r in rows if r["y"] == 1 or r["cls"] == "used_up"]),
                        ("used_up_and_ingredient_successes", ing)):
            Ls = np.array([r[sc] for r in rs], float)
            W = bt.W([r["pair"] for r in rs])
            one = np.ones((1, len(rs)))
            t6 = np.array([feats[r["id"]]["H6"] for r in rs], float)
            a6 = (wauc(Ls, t6, one)[0], wauc(Ls, t6, W))
            d[f"tracking|{lab}|H6"] = _ci(a6[0], a6[1], len(rs), len(set(r["pair"] for r in rs)))
            for h in ("H11", "H2andH11", "H6andH11", "H2", "H1"):
                hv = np.array([feats[r["id"]][h] for r in rs], float)
                if hv.std() == 0:
                    continue
                ah = (wauc(Ls, hv, one)[0], wauc(Ls, hv, W))
                d[f"tracking|{lab}|{h}"] = _ci(ah[0], ah[1], len(rs), len(set(r["pair"] for r in rs)))
                d[f"tracking|{lab}|{h}_minus_H6"] = _ci(ah[0] - a6[0], ah[1] - a6[1], len(rs), len(set(r["pair"] for r in rs)))
                d[f"disagree_with_truth|{lab}|{h}"] = int((hv != t6).sum())
        F_fn = [r for r in S if r[sc] <= 0]
        d["share_of_false_failures_that_are_repeats"] = _r(np.mean([feats[r["id"]]["posthoc_repeat"] for r in F_fn])) if F_fn else None
        d["share_of_successes_that_are_repeats"] = _r(np.mean([feats[r["id"]]["posthoc_repeat"] for r in S]))
        d["share_of_repeat_successes_predicted_failure"] = _r(np.mean(
            [r[sc] <= 0 for r in S if feats[r["id"]]["posthoc_repeat"] == 1]))
        d["share_of_nonrepeat_successes_predicted_failure"] = _r(np.mean(
            [r[sc] <= 0 for r in S if feats[r["id"]]["posthoc_repeat"] == 0]))
        out[sc] = d
    return out


# ---------------------------------------------------------------- question 4 additions

def strata_corr_ci(rows, bt, prefix="") -> dict:
    """Spearman across role x class x bin strata (>= 20 failures) of the stratum AUC with the stratum dL, a - n and
    s - n, recomputed in every resample (pair weights), so the interval reflects item sampling."""
    n, a, s = f"{prefix}neutral", f"{prefix}aligned", f"{prefix}swapped"
    strata = []
    for ro in ROLES:
        Sr = sub(rows, y=1, role=ro)
        for cl in CLASSES:
            for bn in BINS:
                Fb = sub(rows, y=0, role=ro, cls=cl, bin=bn)
                if len(Fb) >= 20:
                    strata.append((Fb, sub(Sr, bin=bn)))
    one = np.ones((1, 1))
    aucs, effs = [], {k: [] for k in ("dL", "a_minus_n", "s_minus_n")}
    for Fb, Sb in strata:
        rs = Fb + Sb
        sc = np.array([r[n] for r in rs])
        yy = np.array([r["y"] for r in rs], float)
        W = bt.W([r["pair"] for r in rs])
        aucs.append(np.r_[wauc(sc, yy, np.ones((1, len(rs))))[0], wauc(sc, yy, W)])
        Wf = bt.W([r["pair"] for r in Fb])
        for k, f in (("dL", lambda r: r[a] - r[s]), ("a_minus_n", lambda r: r[a] - r[n]), ("s_minus_n", lambda r: r[s] - r[n])):
            v = np.array([f(r) for r in Fb])
            with np.errstate(invalid="ignore", divide="ignore"):
                effs[k].append(np.r_[v.mean(), (Wf @ v) / Wf.sum(1)])
    A = np.array(aucs)  # strata x (1 + B)
    out = {"n_strata": len(strata)}
    for k, lst in effs.items():
        E = np.array(lst)
        rho = []
        for j in range(A.shape[1]):
            ok = np.isfinite(A[:, j]) & np.isfinite(E[:, j])
            ra, re_ = rankdata(A[ok, j]), rankdata(E[ok, j])
            rho.append(np.corrcoef(ra, re_)[0, 1])
        rho = np.array(rho)
        out[f"spearman_auc_vs_{k}"] = _ci(rho[0], rho[1:], len(strata), len(set(r["pair"] for r in rows)))
    return out


def by_neutral_bins(rows, bt, prefix, bins, pop) -> dict:
    n, a, s = f"{prefix}neutral", f"{prefix}aligned", f"{prefix}swapped"
    out = {}
    for lab, lo, hi in bins:
        c = [r for r in rows if lo <= r[n] < hi]
        if not c:
            continue
        pr = [r["pair"] for r in c]
        out[lab] = {"n": len(c),
                    "dL": mean_ci([r[a] - r[s] for r in c], pr, bt),
                    "a_minus_n": mean_ci([r[a] - r[n] for r in c], pr, bt),
                    "s_minus_n": mean_ci([r[s] - r[n] for r in c], pr, bt),
                    "dp_points": mean_ci([100 * (sigmoid(r[a]) - sigmoid(r[s])) for r in c], pr, bt),
                    "aligned_says_success": _r(np.mean([r[a] > 0 for r in c])),
                    "swapped_says_success": _r(np.mean([r[s] > 0 for r in c])),
                    "composition": dict(Counter(f"{r['role']}|{r['cls']}" for r in c))}
    return out


def within_cell_quintiles(rows, bt, prefix) -> dict:
    """Failures: quintiles of the neutral score within each role x class cell, pooled across cells (each cell
    contributes its q-th fifth to pooled fifth q), so the comparison across fifths holds the cell mix fixed."""
    n, a, s = f"{prefix}neutral", f"{prefix}aligned", f"{prefix}swapped"
    pooled = defaultdict(list)
    per_cell = {}
    for ro, cl in CELLS:
        c = sorted(sub(rows, y=0, role=ro, cls=cl), key=lambda r: r[n])
        q = np.array_split(np.arange(len(c)), 5)
        per_cell[f"{ro}|{cl}"] = []
        for j, ix in enumerate(q):
            cc = [c[i] for i in ix]
            pooled[j].extend(cc)
            per_cell[f"{ro}|{cl}"].append({"fifth": j + 1, "n": len(cc), "mean_neutral": _r(np.mean([r[n] for r in cc])),
                                           "dL": _r(np.mean([r[a] - r[s] for r in cc]))})
    out = {"pooled": [], "per_cell": per_cell}
    for j in range(5):
        cc = pooled[j]
        pr = [r["pair"] for r in cc]
        out["pooled"].append({"fifth (1 = most confident of failure)": j + 1, "n": len(cc),
                              "mean_neutral": _r(np.mean([r[n] for r in cc])),
                              "neutral_correct_share": _r(np.mean([r[n] < 0 for r in cc])),
                              "dL": mean_ci([r[a] - r[s] for r in cc], pr, bt),
                              "dp_points": mean_ci([100 * (sigmoid(r[a]) - sigmoid(r[s])) for r in cc], pr, bt)})
    # paired: least minus most confident fifth, same resamples
    lo, hi = pooled[0], pooled[4]
    W_lo, W_hi = bt.W([r["pair"] for r in lo]), bt.W([r["pair"] for r in hi])
    v_lo, v_hi = np.array([r[a] - r[s] for r in lo]), np.array([r[a] - r[s] for r in hi])
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = (W_hi @ v_hi) / W_hi.sum(1) - (W_lo @ v_lo) / W_lo.sum(1)
    out["fifth5_minus_fifth1"] = _ci(v_hi.mean() - v_lo.mean(), dd, len(lo) + len(hi), len(set(r["pair"] for r in lo + hi)))
    # regression of dL on the neutral margin (correct side positive) with cell fixed effects
    F = [r for r in rows if r["y"] == 0 and (r["role"], r["cls"]) in CELLS]
    X = np.c_[[[float((r["role"], r["cls"]) == c) for c in CELLS] for r in F], [-r[n] for r in F]]
    out["regression_dL_on_margin_cell_fe"] = ols_ci([r[a] - r[s] for r in F], X, [r["pair"] for r in F], bt,
                                                    [f"{ro}|{cl}" for ro, cl in CELLS] + ["margin_slope"])["margin_slope"]
    return out


def decisions_by_line(rows, bt, prefix) -> dict:
    """How the goal line changes decisions and discrimination."""
    n, a, s = f"{prefix}neutral", f"{prefix}aligned", f"{prefix}swapped"
    F, S = sub(rows, y=0), sub(rows, y=1)
    out = {}
    for lab, pop in (("failures", F), ("successes", S)):
        for line in (n, a, s):
            out[f"says_success|{lab}|{line}"] = share_ci(pop, lambda r, line=line: r[line] > 0, bt)
    for line in (n, a, s):
        acc = lambda r, line=line: (r[line] > 0) == bool(r["y"])
        out[f"balanced_accuracy|{line}"] = _r(0.5 * (np.mean([acc(r) for r in F]) + np.mean([acc(r) for r in S])))
        out[f"auc|{line}"] = auc_ci([r[line] for r in rows], [r["y"] for r in rows], [r["pair"] for r in rows], bt)
    # paired AUC differences
    pr = [r["pair"] for r in rows]
    yy = np.array([r["y"] for r in rows], float)
    W = bt.W(pr)
    one = np.ones((1, len(rows)))
    au = {line: (wauc(np.array([r[line] for r in rows]), yy, one)[0], wauc(np.array([r[line] for r in rows]), yy, W))
          for line in (n, a, s)}
    out["auc_aligned_minus_swapped"] = _ci(au[a][0] - au[s][0], au[a][1] - au[s][1], len(rows), len(set(pr)))
    out["auc_aligned_minus_neutral"] = _ci(au[a][0] - au[n][0], au[a][1] - au[n][1], len(rows), len(set(pr)))
    out["auc_swapped_minus_neutral"] = _ci(au[s][0] - au[n][0], au[s][1] - au[n][1], len(rows), len(set(pr)))
    # flips of correct neutral failure predictions, by neutral confidence
    for lab, sel in (("correct, confident (neutral L < -4.6)", lambda r: r[n] < -CONF),
                     ("correct, unsure (-4.6 <= neutral L < 0)", lambda r: -CONF <= r[n] < 0),
                     ("wrong (neutral L >= 0)", lambda r: r[n] >= 0)):
        c = [r for r in F if sel(r)]
        out[f"failures|{lab}"] = {
            "n": len(c), "share": _r(len(c) / len(F)),
            "aligned_says_success": share_ci(c, lambda r: r[a] > 0, bt),
            "swapped_says_success": share_ci(c, lambda r: r[s] > 0, bt),
            "aligned_minus_swapped_says_success": mean_ci([float(r[a] > 0) - float(r[s] > 0) for r in c], [r["pair"] for r in c], bt),
            "dL": mean_ci([r[a] - r[s] for r in c], [r["pair"] for r in c], bt)}
    for lab, sel in (("correct, confident (neutral L > 4.6)", lambda r: r[n] > CONF),
                     ("correct, unsure (0 < neutral L <= 4.6)", lambda r: 0 < r[n] <= CONF),
                     ("wrong (neutral L <= 0)", lambda r: r[n] <= 0)):
        c = [r for r in S if sel(r)]
        out[f"successes|{lab}"] = {
            "n": len(c), "share": _r(len(c) / len(S)),
            "aligned_says_success": share_ci(c, lambda r: r[a] > 0, bt),
            "swapped_says_success": share_ci(c, lambda r: r[s] > 0, bt),
            "dL": mean_ci([r[a] - r[s] for r in c], [r["pair"] for r in c], bt)}
    # flips between the two goal lines, failures and successes
    for lab, pop in (("failures", F), ("successes", S)):
        out[f"flip_to_success_under_aligned|{lab}"] = share_ci(pop, lambda r: r[a] > 0 >= r[s], bt)
        out[f"flip_to_success_under_swapped|{lab}"] = share_ci(pop, lambda r: r[s] > 0 >= r[a], bt)
    return out


def q4(rows, bt) -> dict:
    out = {}
    for lab, pre in (("two_candidate", ""), ("yes_no", "yn_")):
        d = {"strata_correlations_with_ci": strata_corr_ci(rows, bt, pre),
             "failures_by_neutral_bin": by_neutral_bins(sub(rows, y=0), bt, pre, FAIL_BINS, "failures"),
             "successes_by_neutral_bin": by_neutral_bins(sub(rows, y=1), bt, pre, SUCC_BINS, "successes"),
             "within_cell_fifths": within_cell_quintiles(rows, bt, pre),
             "decisions": decisions_by_line(rows, bt, pre)}
        for ro in ROLES:
            d[f"within_role={ro}"] = {"failures_by_neutral_bin": by_neutral_bins(sub(rows, y=0, role=ro), bt, pre, FAIL_BINS, "failures")}
        d["overall"] = {"dL_failures": mean_ci([r[f"{pre}aligned"] - r[f"{pre}swapped"] for r in sub(rows, y=0)],
                                               [r["pair"] for r in sub(rows, y=0)], bt),
                        "dL_successes": mean_ci([r[f"{pre}aligned"] - r[f"{pre}swapped"] for r in sub(rows, y=1)],
                                                [r["pair"] for r in sub(rows, y=1)], bt)}
        out[lab] = d
    return out


def q4_quantities(rows, feats, bt) -> dict:
    """The goal effect against the quantities that decide the outcome (near the threshold or far from it)."""
    out = {}
    for lab, pre in (("two_candidate", ""), ("yes_no", "yn_")):
        a, s = f"{pre}aligned", f"{pre}swapped"
        d = {}
        groups = {
            "too_few|shortfall": (sub(rows, y=0, cls="too_few"), "shortfall_total",
                                  [("1", lambda v: v == 1), ("2-3", lambda v: 2 <= v <= 3), ("4+", lambda v: v >= 4)]),
            "used_up|steps_since_fall": (sub(rows, y=0, cls="used_up"), "steps_since_fall",
                                         [("0", lambda v: v == 0), ("1-2", lambda v: 1 <= v <= 2), ("3+", lambda v: v >= 3)]),
            "successes|surplus": (sub(rows, y=1), "surplus_min",
                                  [("0", lambda v: v == 0), ("1-2", lambda v: 1 <= v <= 2), ("3+", lambda v: v >= 3)]),
        }
        for g, (rs, key, levels) in groups.items():
            for ro in ROLES + ("all",):
                rr = [r for r in rs if ro == "all" or r["role"] == ro]
                for nm, f in levels:
                    c = [r for r in rr if key in feats[r["id"]] and f(feats[r["id"]][key])]
                    if len(c) >= 10:
                        d[f"{g}={nm}|role={ro}"] = {"dL": mean_ci([r[a] - r[s] for r in c], [r["pair"] for r in c], bt),
                                                     "mean_neutral": _r(np.mean([r[f"{pre}neutral"] for r in c]))}
        out[lab] = d
    return out


# ---------------------------------------------------------------- question 5: the prompt variants

def q5(core, scores, pv, model, bt) -> dict:
    ids = {i for i in pv[model]}
    items = [t for t in core if t["id"] in ids]
    base = {r["id"]: r for r in tc_table(items, scores, model)}
    out = {"n_items": len(items), "n_failures": sum(r["y"] == 0 for r in base.values()),
           "pairs": len({r["pair"] for r in base.values()})}
    variants = {"verified": None, "clean": "clean", "inventory": "inventory", "clean_inventory": "clean_inventory"}
    rows_by = {}
    for v, key in variants.items():
        rows = []
        for i, r in base.items():
            if key is None:
                rows.append({**r})
            else:
                s = pv[model][i]
                rows.append({**r, "neutral": s[f"{key}|neutral"]["L"], "aligned": s[f"{key}|aligned"]["L"],
                             "swapped": s[f"{key}|swapped"]["L"]})
        rows_by[v] = rows
        F = sub(rows, y=0)
        d = {"auc_neutral": auc_ci([r["neutral"] for r in rows], [r["y"] for r in rows], [r["pair"] for r in rows], bt),
             "dL_failures": mean_ci([r["aligned"] - r["swapped"] for r in F], [r["pair"] for r in F], bt),
             "dL_successes": mean_ci([r["aligned"] - r["swapped"] for r in sub(rows, y=1)], [r["pair"] for r in sub(rows, y=1)], bt),
             "mean_neutral_failures": mean_ci([r["neutral"] for r in F], [r["pair"] for r in F], bt)}
        for ro in ROLES:
            Fr = sub(F, role=ro)
            d[f"dL_failures|role={ro}"] = mean_ci([r["aligned"] - r["swapped"] for r in Fr], [r["pair"] for r in Fr], bt)
            d[f"auc_neutral|role={ro}"] = auc_ci([r["neutral"] for r in sub(rows, role=ro)], [r["y"] for r in sub(rows, role=ro)],
                                                 [r["pair"] for r in sub(rows, role=ro)], bt)
        d["failures_by_neutral_bin"] = by_neutral_bins(F, bt, "", FAIL_BINS, "failures")
        d["successes_by_neutral_bin"] = by_neutral_bins(sub(rows, y=1), bt, "", SUCC_BINS, "successes")
        d["decisions"] = decisions_by_line(rows, bt, "")
        # margin regression on the failures, cell fixed effects (cells with >= 10 failures here)
        cells = [c for c in CELLS if len(sub(F, role=c[0], cls=c[1])) >= 10]
        Fc = [r for r in F if (r["role"], r["cls"]) in cells]
        X = np.c_[[[float((r["role"], r["cls"]) == c) for c in cells] for r in Fc], [-r["neutral"] for r in Fc]]
        d["regression_dL_on_margin_cell_fe"] = ols_ci([r["aligned"] - r["swapped"] for r in Fc], X, [r["pair"] for r in Fc], bt,
                                                      [f"{a}|{b}" for a, b in cells] + ["margin_slope"])["margin_slope"]
        out[v] = d
    # paired against the verified prompt, same items
    vf = {r["id"]: r for r in rows_by["verified"]}
    for v in ("clean", "inventory", "clean_inventory"):
        F = sub(rows_by[v], y=0)
        out[f"{v}_minus_verified"] = {
            "dL_failures": mean_ci([(r["aligned"] - r["swapped"]) - (vf[r["id"]]["aligned"] - vf[r["id"]]["swapped"]) for r in F],
                                   [r["pair"] for r in F], bt)}
        # items that become confident and correct with this prompt but were not under the verified prompt
        newly = [r for r in F if r["neutral"] < -CONF and not vf[r["id"]]["neutral"] < -CONF]
        stay = [r for r in F if r["neutral"] < -CONF and vf[r["id"]]["neutral"] < -CONF]
        for lab, c in (("newly_confident_correct", newly), ("confident_correct_under_both", stay)):
            out[f"{v}_minus_verified"][lab] = {
                "n": len(c),
                "dL_variant": mean_ci([r["aligned"] - r["swapped"] for r in c], [r["pair"] for r in c], bt),
                "dL_verified": mean_ci([vf[r["id"]]["aligned"] - vf[r["id"]]["swapped"] for r in c], [r["pair"] for r in c], bt)}
    return out


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="qwen,llama")
    ap.add_argument("--b", type=int, default=tlc.B, help="resamples (2,000 for the note; fewer only for a smoke test)")
    args = ap.parse_args()

    items = read_jsonl(tlc.ITEMS)
    rule = json.loads((REPO / "notes" / "families_metrics.json").read_text())["rules"]["competence"]
    scores = load_tc_scores(args.scores)
    n_records = sum(len(c) for m in scores.values() for c in m.values())
    core = [t for t in items if t["population"] in ("main", "success")]
    feats = {t["id"]: add_features(t, features(t)) for t in core}
    assert all(feats[t["id"]]["truth"] == int(t["population"] == "success") for t in core), "accounting disagrees"
    bt = Boot([t["pair_id"] for t in items], b=args.b)
    tlc.HEURISTICS = tlc.HEURISTICS + NEW_H  # the reused fit code iterates over this module-level tuple
    heur = tlc.HEURISTICS

    out = {"note": "Exploratory, post hoc (notes/textcraft_qwen_competence.md). Neutral goal line unless stated. "
                   "95% intervals: 2,000 resamples of whole goal pairs.",
           "inputs": {"textcraft_records": n_records, "items_main": 1500, "items_success": 600},
           "new_heuristics": list(NEW_H), "confident_threshold_abs_L": CONF, "textcraft": {}}
    pv = load_tc_scores(args.prompts)
    want = args.models.split(",")
    for m in MODELS:
        if SHORT[m] not in want:
            continue
        rows = tc_table(core, scores, m)
        for r in rows:
            r["cls"] = r["cls"] if r["y"] == 0 else "success"
        res = {"n_items": len(rows)}
        res["competence"] = {sc: competence_block(rows, bt, sc) for sc in ("neutral", "yn_neutral")}
        res["decisions"] = {sc: decisions_block(rows, bt, sc) for sc in ("neutral", "yn_neutral")}
        res["calibration"] = {sc: calibration([r[sc] for r in rows], [r["y"] for r in rows]) for sc in ("neutral", "yn_neutral")}
        res["families_rule"] = families_rule(rows, bt, rule)
        res["q1_extra"] = q1_extra(rows, bt, rule)
        print(m, "q1", file=sys.stderr, flush=True)
        res["heuristics"] = {sc: heuristic_fits(rows, feats, bt, sc) for sc in ("neutral", "yn_neutral")}
        print(m, "heuristics", file=sys.stderr, flush=True)
        res["errors"] = q2_errors(rows, feats, bt, heur)
        res["repeat"] = q2_repeat(rows, feats, bt)
        res["quantities"] = {sc: quantities(rows, feats, bt, sc) for sc in ("neutral", "yn_neutral")}
        res["competence_vs_effect"] = {"two_candidate": competence_vs_effect(rows, bt),
                                       "yes_no": competence_vs_effect(rows, bt, prefix="yn_")}
        res["q4"] = q4(rows, bt)
        res["q4_quantities"] = q4_quantities(rows, feats, bt)
        print(m, "q4", file=sys.stderr, flush=True)
        res["prompt_variants"] = q5(core, scores, pv, m, bt)
        out["textcraft"][SHORT[m]] = res
        print(m, "done", file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
