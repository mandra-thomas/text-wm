#!/usr/bin/env python3
"""Exploratory, post hoc: why is the goal effect smaller in TextCraft than in ALFWorld? (notes/textcraft_vs_alfworld.md)

Reads saved scores only (no model is run). Every interval is a 95% percentile interval from 2,000 cluster-bootstrap
resamples: games for ALFWorld, goal pairs for TextCraft (seed 20260924). Strata are bootstrapped by resampling all
clusters and keeping the stratum's items in each resample. Cross-environment ratios and differences pair independent
draws of the two environments' bootstraps.

Usage (from the repository root, with the ALFWorld code importable):
  python scripts/xenv/compare.py --archive <text-wm-runs-archive> --tc-items <items.jsonl> \
      --tc-scores <shard scoring.jsonl ...> --out notes/textcraft_vs_alfworld_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import data as D  # noqa: E402

SEED, B = 20260924, 2000


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Boot:
    """Index resamples of whole clusters, shared by every statistic of one table."""

    def __init__(self, clusters, seed=SEED, b=B):
        uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        rng = np.random.default_rng(seed)
        self.n = len(inv)
        self.draws = [np.concatenate([groups[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(b)]

    def run(self, fn):
        """fn(index array) -> float, on the full sample and on every resample."""
        point = fn(np.arange(self.n))
        boots = np.array([fn(ix) for ix in self.draws], dtype=float)
        return point, boots


def summ(point, boots):
    ok = boots[~np.isnan(boots)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    out = {"est": float(point), "lo": float(lo), "hi": float(hi)}
    if len(ok) < len(boots):
        out["dropped_draws"] = int(len(boots) - len(ok))
    return out


def mean_fn(x, mask=None):
    def fn(ix):
        if mask is not None:
            ix = ix[mask[ix]]
        v = x[ix]
        return float(np.mean(v)) if len(v) else np.nan
    return fn


def std_fn(x, mask=None):
    def fn(ix):
        if mask is not None:
            ix = ix[mask[ix]]
        v = x[ix]
        return float(np.mean(v) / np.std(v, ddof=1)) if len(v) > 2 else np.nan
    return fn


def est(boot, fn, keep=False):
    p, b = boot.run(fn)
    s = summ(p, b)
    return (s, b) if keep else s


def ratio(a, b):
    """a, b: (summary, boots). Ratio a/b with independent-draw interval."""
    r = a[1] / b[1]
    return summ(a[0]["est"] / b[0]["est"], r)


def diff(a, b):
    return summ(a[0]["est"] - b[0]["est"], a[1] - b[1])


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- the scale panel

def scales(t, boot, mask=None, keep=False):
    """Every common scale of the aligned-minus-swapped goal effect on one failure table (optionally a stratum)."""
    al, sw, neu = t["al.L"], t["sw.L"], t["neu.L"]
    alf, swf, neuf = t["al.L_first"], t["sw.L_first"], t["neu.L_first"]
    yal, ysw, yneu = t["yal.L"], t["ysw.L"], t["yneu.L"]
    d, dfirst, dy = al - sw, alf - swf, yal - ysw
    out = {
        "n": int(len(al) if mask is None else mask.sum()),
        "dL": est(boot, mean_fn(d, mask), keep),
        "dL_first": est(boot, mean_fn(dfirst, mask), keep),
        "dL_remainder": est(boot, mean_fn(d - dfirst, mask), keep),
        # the remainder split into the success reply's tail and the failure reply's tail
        "remainder_success_tail": est(boot, mean_fn((t["al.logp_plus"] - t["al.logp_first_plus"])
                                                    - (t["sw.logp_plus"] - t["sw.logp_first_plus"]), mask)),
        "remainder_failure_tail": est(boot, mean_fn(-((t["al.logp_minus"] - t["al.logp_first_minus"])
                                                      - (t["sw.logp_minus"] - t["sw.logp_first_minus"])), mask)),
        "dL_yes_no": est(boot, mean_fn(dy, mask), keep),
        "standardized_sd": est(boot, std_fn(d, mask), keep),
        "standardized_sd_first": est(boot, std_fn(dfirst, mask)),
        "standardized_sd_yes_no": est(boot, std_fn(dy, mask), keep),
        "dp_pp": est(boot, mean_fn(100 * (sig(al) - sig(sw)), mask), keep),
        "dp_yes_no_pp": est(boot, mean_fn(100 * (sig(yal) - sig(ysw)), mask), keep),
        "flip_to_success_pct": est(boot, mean_fn(100.0 * ((sw < 0) & (al > 0)), mask)),
        "flip_to_failure_pct": est(boot, mean_fn(100.0 * ((sw > 0) & (al < 0)), mask)),
        "sign_change_full_pct": est(boot, mean_fn(100.0 * (np.sign(al) != np.sign(sw)), mask)),
        "sign_change_first_pct": est(boot, mean_fn(100.0 * (np.sign(alf) != np.sign(swf)), mask)),
        "yes_no_flip_to_yes_pct": est(boot, mean_fn(100.0 * ((ysw < 0) & (yal > 0)), mask)),
        "yes_no_sign_change_pct": est(boot, mean_fn(100.0 * (np.sign(yal) != np.sign(ysw)), mask)),
        "aligned_minus_neutral": est(boot, mean_fn(al - neu, mask)),
        "swapped_minus_neutral": est(boot, mean_fn(sw - neu, mask)),
        "aligned_minus_neutral_first": est(boot, mean_fn(alf - neuf, mask)),
        "swapped_minus_neutral_first": est(boot, mean_fn(swf - neuf, mask)),
        "yes_no_aligned_minus_neutral": est(boot, mean_fn(yal - yneu, mask)),
        "yes_no_swapped_minus_neutral": est(boot, mean_fn(ysw - yneu, mask)),
    }
    m = np.ones(len(al), bool) if mask is None else mask
    out["levels"] = {
        arm: {"mean_L": float(np.mean(t[f"{arm}.L"][m])), "median_L": float(np.median(t[f"{arm}.L"][m])),
              "mean_L_first": float(np.mean(t[f"{arm}.L_first"][m])),
              "mean_success_tail": float(np.mean((t[f"{arm}.logp_plus"] - t[f"{arm}.logp_first_plus"])[m])),
              "mean_failure_tail": float(np.mean((t[f"{arm}.logp_minus"] - t[f"{arm}.logp_first_minus"])[m])),
              "mean_s_pct": float(100 * np.mean(sig(t[f"{arm}.L"][m]))),
              "tokens_success": float(np.mean(t[f"{arm}.n_tokens_plus"][m])),
              "tokens_failure": float(np.mean(t[f"{arm}.n_tokens_minus"][m])),
              "median_first_mass": float(np.median(t[f"{arm}.first_mass"][m]))}
        for arm in ("al", "sw", "neu")}
    out["yes_no_levels"] = {
        arm: {"mean_L": float(np.mean(t[f"{arm}.L"][m])), "median_L": float(np.median(t[f"{arm}.L"][m])),
              "answered_yes_pct": est(boot, mean_fn(100.0 * (t[f"{arm}.L"] > 0), mask)),
              "mean_sigma_pct": float(100 * np.mean(sig(t[f"{arm}.L"][m])))}
        for arm in ("yal", "ysw", "yneu")}
    if "pal.L" in t and not np.all(np.isnan(t["pal.L"])):
        out["dL_prefix_only"] = est(boot, mean_fn(t["pal.L"] - t["psw.L"], mask))
    return out


def core(s):
    return {k: v[0] if isinstance(v, tuple) else v for k, v in s.items()}


# ---------------------------------------------------------------- competence (AUC) and baselines

def competence(fail, succ, boot_pair):
    """Neutral-task levels on failures and successes and the AUC on three readouts, clustered jointly."""
    out = {}
    for arm, label in (("neu.L", "two_candidate"), ("neu.L_first", "first_token"), ("yneu.L", "yes_no")):
        x = np.concatenate([succ[arm], fail[arm]])
        is_s = np.concatenate([np.ones(len(succ[arm]), bool), np.zeros(len(fail[arm]), bool)])
        out[f"auc_neutral_{label}"] = est(boot_pair, lambda ix, x=x, s=is_s: auc(x[ix][s[ix]], x[ix][~s[ix]]))
        out[f"mean_neutral_{label}"] = {"failures": float(np.mean(fail[arm])), "successes": float(np.mean(succ[arm])),
                                        "gap": float(np.mean(succ[arm]) - np.mean(fail[arm]))}
    for arm, label in (("al.L", "two_candidate"), ("yal.L", "yes_no")):
        x = np.concatenate([succ[arm], fail[arm]])
        is_s = np.concatenate([np.ones(len(succ[arm]), bool), np.zeros(len(fail[arm]), bool)])
        out[f"auc_aligned_{label}"] = est(boot_pair, lambda ix, x=x, s=is_s: auc(x[ix][s[ix]], x[ix][~s[ix]]))
    return out


def by_bins(t, boot, base, edges, measures):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (base > lo) & (base <= hi)
        if mask.sum() < 20:
            out[f"({lo},{hi}]"] = {"n": int(mask.sum())}
            continue
        out[f"({lo},{hi}]"] = {"n": int(mask.sum()), "base_mean": float(np.mean(base[mask])),
                               **{name: est(boot, mean_fn(x, mask)) for name, x in measures.items()}}
    return out


def quintiles(t, boot, base, measures):
    qs = np.quantile(base, [0, .2, .4, .6, .8, 1.0])
    qs[0] -= 1e-9
    return by_bins(t, boot, base, list(qs), measures)


def reweighted(t, boot, x, strata, target_weights):
    """Mean of x with strata reweighted to target weights {stratum: w} (sum 1); a draw lacking a stratum is NaN."""
    keys = list(target_weights)
    w = np.array([target_weights[k] for k in keys])

    def fn(ix):
        vals = []
        for k in keys:
            sub = ix[strata[ix] == k]
            if not len(sub):
                return np.nan
            vals.append(np.mean(x[sub]))
        return float(np.dot(w, vals))
    return est(boot, fn, keep=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--tc-items", required=True)
    ap.add_argument("--tc-scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = D.load_alfworld(a.archive)
    T = D.load_textcraft(a.tc_items, a.tc_scores)
    R: dict = {"note": "exploratory, post hoc; see notes/textcraft_vs_alfworld.md",
               "bootstrap": {"resamples": B, "seed": SEED, "clusters": {"alfworld": "game", "textcraft": "goal pair"}}}
    for m in D.MODELS:
        af, asu = A[m]["fail"], A[m]["succ"]
        tm, ts, te, tu = (T[m][p] for p in ("main", "success", "empty_history", "usefulness"))
        ba, bt, be = Boot(af["cluster"]), Boot(tm["cluster"]), Boot(te["cluster"])
        r: dict = {}

        # Q1: every common scale, overall
        sa, st = scales(af, ba, keep=True), scales(tm, bt, keep=True)
        r["scales"] = {"alfworld": core(sa), "textcraft_main": core(st)}
        r["scales"]["textcraft_over_alfworld"] = {
            k: ratio(st[k], sa[k]) for k in ("dL", "dL_first", "dL_remainder", "dL_yes_no", "standardized_sd",
                                              "standardized_sd_yes_no", "dp_pp", "dp_yes_no_pp")}
        r["scales"]["textcraft_minus_alfworld"] = {
            k: diff(st[k], sa[k]) for k in ("dL", "dL_first", "dL_yes_no", "dp_pp", "dp_yes_no_pp")}

        # Q3: competence and baselines; standardized by the neutral success-failure gap
        both_a = np.concatenate([asu["cluster"], af["cluster"]])
        tfail = {k: np.concatenate([tm[k], tu[k]]) for k in tm if k in tu and tm[k].dtype.kind == "f"}
        both_t = np.concatenate([ts["cluster"], tm["cluster"], tu["cluster"]])
        r["competence"] = {"alfworld": competence(af, asu, Boot(both_a)),
                           "textcraft": competence(tfail, ts, Boot(both_t)),
                           "textcraft_main_failures_only": competence(tm, ts, Boot(np.concatenate([ts["cluster"], tm["cluster"]])))}
        r["standardized_by_neutral_gap"] = {
            "alfworld_two_candidate": sa["dL"][0]["est"] / r["competence"]["alfworld"]["mean_neutral_two_candidate"]["gap"],
            "alfworld_first_token": sa["dL_first"][0]["est"] / r["competence"]["alfworld"]["mean_neutral_first_token"]["gap"],
            "alfworld_yes_no": sa["dL_yes_no"][0]["est"] / r["competence"]["alfworld"]["mean_neutral_yes_no"]["gap"],
            "textcraft_two_candidate": st["dL"][0]["est"] / r["competence"]["textcraft"]["mean_neutral_two_candidate"]["gap"],
            "textcraft_first_token": st["dL_first"][0]["est"] / r["competence"]["textcraft"]["mean_neutral_first_token"]["gap"],
            "textcraft_yes_no": st["dL_yes_no"][0]["est"] / r["competence"]["textcraft"]["mean_neutral_yes_no"]["gap"]}
        r["successes"] = {
            "alfworld": {k: est(Boot(asu["cluster"]), mean_fn(asu[f"{p}al.L"] - asu[f"{p}sw.L"]))
                         for k, p in (("dL", ""), ("dL_yes_no", "y"))},
            "textcraft": {k: est(Boot(ts["cluster"]), mean_fn(ts[f"{p}al.L"] - ts[f"{p}sw.L"]))
                          for k, p in (("dL", ""), ("dL_yes_no", "y"))}}

        # Q3: does the effect scale with baseline confidence within each environment?
        meas = lambda t: {"dL": t["al.L"] - t["sw.L"], "dL_first": t["al.L_first"] - t["sw.L_first"],
                          "dL_yes_no": t["yal.L"] - t["ysw.L"]}
        r["by_neutral_quintile"] = {
            env: {base: quintiles(t, b, t[base], meas(t)) for base in ("neu.L_first", "yneu.L")}
            for env, t, b in (("alfworld", af, ba), ("textcraft", tm, bt))}
        # fixed bins on the two scales the environments share (first-token and yes/no log-odds under neutral)
        fixed = {"neu.L_first": [-np.inf, -15, -10, -5, 0, np.inf], "yneu.L": [-np.inf, -25, -20, -15, -10, -5, np.inf]}
        r["by_neutral_fixed_bins"] = {
            env: {base: by_bins(t, b, t[base], edges, meas(t)) for base, edges in fixed.items()}
            for env, t, b in (("alfworld", af, ba), ("textcraft", tm, bt))}
        # ALFWorld reweighted to TextCraft's distribution of neutral first-token log-odds (bins with >=20 ALFWorld items)
        rw = {}
        for base, edges in fixed.items():
            sa_bin = np.digitize(af[base], edges[1:-1], right=True)
            st_bin = np.digitize(tm[base], edges[1:-1], right=True)
            ok = [k for k in range(len(edges) - 1) if (sa_bin == k).sum() >= 20 and (st_bin == k).sum() > 0]
            wt = {k: float(np.mean(st_bin[np.isin(st_bin, ok)] == k)) for k in ok}
            cover = float(np.mean(np.isin(st_bin, ok)))
            rw[base] = {"textcraft_share_covered": cover, "weights": {str(k): v for k, v in wt.items()},
                        **{name: reweighted(af, ba, x, sa_bin, wt)[0] for name, x in meas(af).items()},
                        "textcraft_same_bins": {name: reweighted(tm, bt, x, st_bin, wt)[0] for name, x in meas(tm).items()}}
        r["alfworld_reweighted_to_textcraft_baseline"] = rw

        # Q2: population matching
        hl = af["history_len"]
        alf_bins = {"0": hl == 0, "1-3": (hl >= 1) & (hl <= 3), "4-7": (hl >= 4) & (hl <= 7), "8-11": hl >= 8,
                    "1+": hl >= 1}
        r["alfworld_by_history_length"] = {k: core(scales(af, ba, v)) for k, v in alf_bins.items()}
        th = tm["history_len"]
        tc_bins = {"1-3": th <= 3, "4-7": (th >= 4) & (th <= 7), "8-12": th >= 8}
        r["textcraft_by_history_length"] = {k: core(scales(tm, bt, v)) for k, v in tc_bins.items()}
        r["textcraft_empty_history"] = core(scales(te, be))
        r["textcraft_empty_history_by_role"] = {role: core(scales(te, be, te["role"] == role)) for role in ("goal", "ingredient")}
        s0, se = scales(af, ba, hl == 0, keep=True), scales(te, be, keep=True)
        r["step0_vs_empty"] = {k: {"alfworld_step0": s0[k][0], "textcraft_empty": se[k][0], "ratio": ratio(se[k], s0[k])}
                               for k in ("dL", "dL_first", "dL_yes_no", "dp_pp", "dp_yes_no_pp", "standardized_sd")}
        # matched history lengths: ALFWorld 1+ reweighted to TextCraft's length distribution (TextCraft's 12 merged into 11)
        th11 = np.minimum(th, 11)
        wt = {k: float(np.mean(th11 == k)) for k in range(1, 12)}
        r["alfworld_later_reweighted_to_textcraft_lengths"] = {
            name: reweighted(af, ba, x, hl, wt)[0] for name, x in meas(af).items()}
        r["alfworld_later_reweighted_to_textcraft_lengths"]["dp_pp"] = reweighted(af, ba, 100 * (sig(af["al.L"]) - sig(af["sw.L"])), hl, wt)[0]
        # failure types and roles
        r["alfworld_by_reason"] = {k: core(scales(af, ba, af["reason"] == k)) for k in np.unique(af["reason"])}
        r["alfworld_by_action"] = {k: core(scales(af, ba, af["action_type"] == k)) for k in np.unique(af["action_type"])}
        r["alfworld_never_obtained"] = core(scales(af, ba, af["action_type"] != "take"))
        r["textcraft_by_role_class"] = {f"{ro}|{fc}": core(scales(tm, bt, (tm["role"] == ro) & (tm["failure_class"] == fc)))
                                        for ro in ("goal", "ingredient") for fc in ("never_obtained", "too_few", "used_up")
                                        if ((tm["role"] == ro) & (tm["failure_class"] == fc)).sum() >= 20}
        r["textcraft_by_role"] = {ro: core(scales(tm, bt, tm["role"] == ro)) for ro in ("goal", "ingredient")}
        r["textcraft_by_class"] = {fc: core(scales(tm, bt, tm["failure_class"] == fc)) for fc in ("never_obtained", "too_few", "used_up")}

        # Q4: history-goal agreement
        r["textcraft_by_history_fit"] = {k: core(scales(tm, bt, tm["history_fit"] == k)) for k in ("aligned", "swapped", "tie")}
        later = hl >= 1
        g, s = af["goal_seen_in_history"].astype(bool), af["swap_seen_in_history"].astype(bool)
        r["alfworld_by_history_mentions"] = {
            "no_history_step0": core(scales(af, ba, hl == 0)),
            "neither_named": core(scales(af, ba, later & ~g & ~s)),
            "goal_named_only": core(scales(af, ba, later & g & ~s)),
            "swap_named_only": core(scales(af, ba, later & ~g & s)),
            "both_named": core(scales(af, ba, later & g & s))}
        cf = ~np.isnan(af["cfal.L"])
        r["alfworld_counterfactual_aligned"] = {
            "n": int(cf.sum()),
            "cf_effect": est(ba, mean_fn(af["cfal.L"] - af["cfsw.L"], cf)),
            "real_effect_same_run": est(ba, mean_fn(af["val.L"] - af["vsw.L"], cf)),
            "cf_minus_real": est(ba, mean_fn((af["cfal.L"] - af["cfsw.L"]) - (af["val.L"] - af["vsw.L"]), cf)),
            "cf_effect_first": est(ba, mean_fn(af["cfal.L_first"] - af["cfsw.L_first"], cf)),
            "real_effect_first_same_run": est(ba, mean_fn(af["val.L_first"] - af["vsw.L_first"], cf))}

        # Q5: the completing act with a missing precondition
        mv_t = af["move_to_target"].astype(bool)
        tk = af["task_kind"]
        r["alfworld_completing_act"] = {
            "move_goal_to_target_type_all": core(scales(af, ba, mv_t)),
            "move_goal_to_target_type_place_tasks": core(scales(af, ba, mv_t & (tk == "place"))),
            "move_goal_to_target_type_transform_tasks": core(scales(af, ba, mv_t & np.isin(tk, ["heat", "cool", "clean"]))),
            "move_goal_to_target_type_two_tasks": core(scales(af, ba, mv_t & (tk == "two"))),
            "move_goal_elsewhere": core(scales(af, ba, (af["action_type"] == "move") & ~mv_t)),
            "heat_cool_clean_goal": core(scales(af, ba, np.isin(af["action_type"], ["heat", "cool", "clean"]))),
            "move_to_target_step0": core(scales(af, ba, mv_t & (hl == 0))),
            "move_to_target_later": core(scales(af, ba, mv_t & (hl >= 1)))}
        r["textcraft_goal_never_obtained"] = {
            "main": core(scales(tm, bt, (tm["role"] == "goal") & (tm["failure_class"] == "never_obtained"))),
            "empty_history": core(scales(te, be, te["role"] == "goal"))}
        R[m] = r
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
