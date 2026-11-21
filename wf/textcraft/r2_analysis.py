"""TextCraft round 2 analysis: `textcraft_r2_analysis` (notes/textcraft_r2_design.md, sections 3 and 6 to 9).

Reads every model's `textcraft_r2_score_<tag>` shards and writes metrics.json: per model the competence gate (wave A's
rule, on the yes/no question under the neutral line), the five confirmatory contrasts (intermediate lift, within-item
parent penalty, naming without completion, completion with naming held, timing), each with a test of difference and a
two one-sided equivalence test, all 50 in one Holm family of fixed size; the verdicts and readings of the design's
tables; and the descriptive readouts. Scores are joined to records by id and record hash; a stale or unknown score
stops the analysis, and with any score missing the verdicts are withheld.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf.analysis.stats import holm
from wf.runs import _init_run
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft.r2_score import load_records, verified_files
from wf.util import read_jsonl, runs_root, write_json, write_status

B_DRAWS = 4000
N_FLIPS = 4000
DESC_DRAWS = 2000  # descriptive intervals
AUC_DRAWS = 1000
ALPHA = 0.05
MARGIN_SD = 0.3  # equivalence margin: 0.3 x the item SD of ΔL (aligned - swapped) on the main failures
CONTRASTS = ("lift", "parent_penalty", "naming", "completion", "timing")
TESTS = ("differs", "equivalent")
COMPETENCE = {"auc_neutral_min": 0.70, "yes_share_neutral_failures_max": 0.95, "yes_share_neutral_successes_min": 0.05,
              "median_yes_no_mass_neutral_min": 0.5}  # wave A's rule (wf/families.py COMPETENCE), unchanged


# ---------------- loading ----------------


def score_run(tag: str) -> str:
    return f"textcraft_r2_score_{tag}"


def load_scores(tag: str, root: Path | None = None) -> list[dict]:
    root = root or runs_root()
    out = []
    for p in sorted((root / score_run(tag)).glob("shard_*/scoring.jsonl")):
        out += read_jsonl(p)
    return out


class StaleScore(RuntimeError):
    pass


def index_scores(scores: list[dict], records: list[dict], shots, tag: str) -> tuple[dict, list[tuple[str, str]]]:
    """{(record id, arm): score} and the (record id, arm) pairs with no score. A score whose record is unknown, whose
    record hash differs, whose model is another, or that is duplicated, stops the analysis."""
    hashes = {r["id"]: R.item_hash(r, shots) for r in records}
    by: dict[tuple[str, str], dict] = {}
    for s in scores:
        if s.get("model") != tag:
            raise StaleScore(f"{tag}: a score of model {s.get('model')!r}")
        if hashes.get(s["transition_id"]) != s.get("item_hash"):
            raise StaleScore(f"{tag}: score for {s['transition_id']} {s['condition']} does not match its record")
        key = (s["transition_id"], s["condition"])
        if key in by:
            raise StaleScore(f"{tag}: duplicate score {key}")
        by[key] = s
    missing = [(r["id"], a) for r, a in R.jobs(records) if (r["id"], a) not in by or by[(r["id"], a)]["L"] != by[(r["id"], a)]["L"]]
    return by, missing


# ---------------- statistics ----------------


def _clusters(rows: list[Any], clusters: list[str]) -> tuple[list[str], dict[str, list]]:
    by: dict[str, list] = defaultdict(list)
    for r, c in zip(rows, clusters):
        by[c].append(r)
    return sorted(by), by


def draws(stat: Callable[[list], float], rows: list, clusters: list[str], B: int | None = None, seed: int = C.RNG_SEED) -> list[float]:
    B = B_DRAWS if B is None else B
    keys, by = _clusters(rows, clusters)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        out.append(stat([r for k in rng.choices(keys, k=len(keys)) for r in by[k]]))
    return [x for x in out if x == x]


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def bootstrap_p(d: list[float]) -> float:
    """Two-sided bootstrap p: twice the smaller share of draws on either side of zero, smoothed."""
    if not d:
        return float("nan")
    return min(1.0, (2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) + 1) / (len(d) + 1))


def tost_p(d: list[float], margin: float) -> float:
    """The larger of the two one-sided bootstrap ps against -margin and +margin."""
    if not d or not margin == margin or margin <= 0:
        return float("nan")
    low = (sum(x <= -margin for x in d) + 1) / (len(d) + 1)
    high = (sum(x >= margin for x in d) + 1) / (len(d) + 1)
    return max(low, high)


def sign_flip_p(values: list[float], clusters: list[str], n: int | None = None, seed: int = C.RNG_SEED) -> float:
    """Two-sided sign-flip p for the mean, whole clusters flipped together."""
    n = N_FLIPS if n is None else n
    if not values:
        return float("nan")
    tot: dict[str, float] = defaultdict(float)
    for v, c in zip(values, clusters):
        tot[c] += v
    groups = [tot[k] for k in sorted(tot)]
    obs = abs(sum(groups)) / len(values)
    rng = random.Random(seed)
    hits = sum(abs(sum(g if rng.random() < 0.5 else -g for g in groups)) / len(values) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def within(values: list[float], clusters: list[str], margin: float) -> dict[str, Any]:
    """A mean of per-item differences: estimate, 95% interval, sign-flip p, TOST p."""
    d = draws(mean, values, clusters)
    return {"estimate": mean(values), "ci95": [pct(d, 0.025), pct(d, 0.975)], "ci90": [pct(d, 0.05), pct(d, 0.95)],
            "n_items": len(values), "n_pairs": len(set(clusters)), "p_differs": sign_flip_p(values, clusters),
            "p_equivalent": tost_p(d, margin), "margin": margin, "p_method": "sign-flip of whole pairs"}


def _gap(rows: list[tuple[str, float]]) -> float:
    return mean([x for r, x in rows if r == "goal"]) - mean([x for r, x in rows if r == "ingredient"])


def between(rows: list[tuple[str, float]], clusters: list[str], margin: float) -> dict[str, Any]:
    """Goal-role mean minus ingredient mean: estimate, 95% interval, bootstrap p, TOST p."""
    d = draws(_gap, rows, clusters)
    return {"estimate": _gap(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "ci90": [pct(d, 0.05), pct(d, 0.95)],
            "goal_mean": mean([x for r, x in rows if r == "goal"]), "ingredient_mean": mean([x for r, x in rows if r == "ingredient"]),
            "n_goal": sum(r == "goal" for r, _ in rows), "n_ingredient": sum(r == "ingredient" for r, _ in rows),
            "n_pairs": len(set(clusters)), "p_differs": bootstrap_p(d), "p_equivalent": tost_p(d, margin),
            "margin": margin, "p_method": "pair bootstrap"}


def describe(values: list[float], clusters: list[str]) -> dict[str, Any]:
    d = draws(mean, values, clusters, B=DESC_DRAWS)
    return {"estimate": mean(values), "ci95": [pct(d, 0.025), pct(d, 0.975)], "n_items": len(values),
            "n_pairs": len(set(clusters))}


def describe_gap(rows: list[tuple[str, float]], clusters: list[str]) -> dict[str, Any]:
    d = draws(_gap, rows, clusters, B=DESC_DRAWS)
    return {"estimate": _gap(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "n_items": len(rows), "n_pairs": len(set(clusters))}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a success scores above a failure), ties half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rank_sum += sum(lab for _v, lab in allv[i:j]) * (i + j + 1) / 2
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos: list[tuple[str, float]], neg: list[tuple[str, float]]) -> dict[str, Any]:
    rows = [("s", v) for _c, v in pos] + [("f", v) for _c, v in neg]
    clusters = [c for c, _v in pos] + [c for c, _v in neg]
    stat = lambda rs: auc([v for k, v in rs if k == "s"], [v for k, v in rs if k == "f"])
    d = draws(stat, rows, clusters, B=AUC_DRAWS)
    return {"auc": stat(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "n_successes": len(pos), "n_failures": len(neg)}


# ---------------- populations ----------------


def populations(records: list[dict]) -> dict[str, list[dict]]:
    main = [r for r in records if r["kind"] == "main"]
    return {"main_failures": [r for r in main if r["population"] == "main"],
            "successes": [r for r in main if r["population"] == "success"],
            "goal": [r for r in main if r["population"] == "main" and r["role"] == "goal" and r["failure_class"] != "used_up"],
            "ingredient": [r for r in main if r["population"] == "main" and r["role"] == "ingredient"],
            "padded": [r for r in main if "pad8_aligned" in r["arms"]],
            "parent_failures": [r for r in records if r["kind"] == "parent" and r["population"] == "main"],
            "parent_successes": [r for r in records if r["kind"] == "parent" and r["population"] == "success"]}


def L(by: dict, rid: str, arm: str) -> float:
    s = by.get((rid, arm))
    return s["L"] if s is not None else float("nan")


def ok(*xs: float) -> bool:
    return all(x == x for x in xs)


# ---------------- competence ----------------


def _mass(s: dict) -> float:
    return math.exp(s["logp_plus"]) + math.exp(s["logp_minus"])


def _first_mass(s: dict) -> float:
    return math.exp(s["logp_first_plus"]) + math.exp(s["logp_first_minus"])


def competence(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    """Wave A's gate on the yes/no question under the neutral line (main failures against the 600 successes), and
    the descriptive role-, class- and parent-level AUCs on both scales."""
    f, s = pops["main_failures"], pops["successes"]
    yn = lambda rs: [by[(r["id"], "yn_neutral")] for r in rs if (r["id"], "yn_neutral") in by]
    fs, ss = yn(f), yn(s)
    a = auc([x["L"] for x in ss], [x["L"] for x in fs])
    yes_f = mean([x["L"] > 0 for x in fs])
    yes_s = mean([x["L"] > 0 for x in ss])
    masses = [_mass(x) for x in fs + ss]
    m = statistics.median(masses) if masses else float("nan")
    checks = {"auc_neutral": {"value": a, "pass": a >= COMPETENCE["auc_neutral_min"]},
              "yes_share_neutral_failures": {"value": yes_f, "pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]},
              "yes_share_neutral_successes": {"value": yes_s, "pass": yes_s >= COMPETENCE["yes_share_neutral_successes_min"]},
              "median_yes_no_mass_neutral": {"value": m, "pass": m >= COMPETENCE["median_yes_no_mass_neutral_min"]}}
    detail: dict[str, Any] = {}
    for scale, arm in (("yes_no", "yn_neutral"), ("two_candidate", "neutral")):
        val = lambda rs: [(r["pair_id"], by[(r["id"], arm)]["L"]) for r in rs if (r["id"], arm) in by]
        detail[scale] = {"all": auc_ci(val(s), val(f)),
                         **{f"role_{role}": auc_ci(val([r for r in s if r["role"] == role]), val([r for r in f if r["role"] == role]))
                            for role in ("goal", "ingredient")},
                         **{f"class_{c}": auc_ci(val(s), val([r for r in f if r["failure_class"] == c]))
                            for c in ("used_up", "too_few", "never_obtained")},
                         **{f"bin_{b}": auc_ci(val([r for r in s if r["length_bin"] == b]), val([r for r in f if r["length_bin"] == b]))
                            for b in ("1-3", "4-8", "9-12")},
                         # role-matched: used-up failures on ingredients against ingredient successes (the confirmatory
                         # used-up figure set mostly-ingredient failures against successes of both roles)
                         "used_up_vs_ingredient_successes": auc_ci(
                             val([r for r in s if r["role"] == "ingredient"]),
                             val([r for r in f if r["failure_class"] == "used_up" and r["role"] == "ingredient"]))}
    pf, ps = pops["parent_failures"], pops["parent_successes"]
    pv = lambda rs, arm: [(r["pair_id"], by[(r["id"], arm)]["L"]) for r in rs if (r["id"], arm) in by]
    detail["parent_two_candidate_aligned"] = auc_ci(pv(ps, "aligned"), pv(pf, "aligned"))
    fm = [_first_mass(by[(r["id"], "neutral")]) for r in f if (r["id"], "neutral") in by]
    detail["two_candidate_first_token_mass_neutral_median"] = statistics.median(fm) if fm else float("nan")
    detail["heuristics"] = heuristic_fits(by, pops)
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks, "rule": COMPETENCE,
            "descriptive": detail}


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa of two binary series."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def heuristic_fits(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    """Descriptive (design section 9): how closely each model's neutral-line scores follow the competence note's
    heuristics H1, H2, H6 (the truth) and H2 and H9 (definitions fixed at 9a83738; wf/textcraft/r2_heuristics.py),
    on the main failures and the 600 successes, on both scales: the tracking AUC (the model's L between the items a
    heuristic calls a success and those it calls a failure), the same within the true failures and within the true
    successes, and the agreement and Cohen's kappa between the model's decision (L above 0) and the heuristic."""
    from wf.textcraft.r2_heuristics import HEURISTICS, features

    items = pops["main_failures"] + pops["successes"]
    feats = {r["id"]: features(r) for r in items}
    out: dict[str, Any] = {}
    for scale, arm in (("two_candidate", "neutral"), ("yes_no", "yn_neutral")):
        rows = [(r, by[(r["id"], arm)]["L"]) for r in items if (r["id"], arm) in by]
        block = {}
        for h in HEURISTICS:
            pred = [(r, L, feats[r["id"]][h]) for r, L in rows]
            pos = [(r["pair_id"], L) for r, L, x in pred if x]
            neg = [(r["pair_id"], L) for r, L, x in pred if not x]
            d = {"share_predicting_success": mean([x for _r, _L, x in pred]), "tracking_auc": auc_ci(pos, neg)}
            for lab, want in (("failures", "main"), ("successes", "success")):
                sub = [(r, L, x) for r, L, x in pred if r["population"] == want]
                d[f"tracking_auc_within_{lab}"] = auc_ci([(r["pair_id"], L) for r, L, x in sub if x],
                                                         [(r["pair_id"], L) for r, L, x in sub if not x])
            dec = [int(L > 0) for _r, L, _x in pred]
            hv = [x for _r, _L, x in pred]
            d["decision_agreement"] = mean([a == b for a, b in zip(dec, hv)])
            d["kappa"] = kappa(dec, hv)
            block[h] = d
        out[scale] = block
    return out


# ---------------- the five contrasts ----------------


def margin_of(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    d = [L(by, r["id"], "aligned") - L(by, r["id"], "swapped") for r in pops["main_failures"]]
    d = [x for x in d if x == x]
    sd = statistics.stdev(d) if len(d) > 1 else float("nan")
    return {"sd_delta_L_main": sd, "margin": MARGIN_SD * sd, "rule": f"{MARGIN_SD} x item SD of ΔL on the main failures"}


def diff_rows(by: dict, recs: list[dict], a: str, b: str) -> tuple[list[float], list[str]]:
    vals, cl = [], []
    for r in recs:
        x = L(by, r["id"], a) - L(by, r["id"], b)
        if x == x:
            vals.append(x)
            cl.append(r["pair_id"])
    return vals, cl


def gap_rows(by: dict, recs: list[dict], value: Callable[[str], float]) -> tuple[list[tuple[str, float]], list[str]]:
    rows, cl = [], []
    for r in recs:
        x = value(r["id"])
        if x == x:
            rows.append((r["role"], x))
            cl.append(r["pair_id"])
    return rows, cl


def contrast_inputs(by: dict, pops: dict[str, list[dict]]) -> dict[str, tuple]:
    roles = pops["goal"] + pops["ingredient"]
    padded = [r for r in pops["padded"] if not (r["role"] == "goal" and r["failure_class"] == "used_up")]
    delta = lambda rid, k: L(by, rid, f"pad{k}_aligned") - L(by, rid, f"pad{k}_swapped")
    return {
        "lift": ("within", *diff_rows(by, pops["ingredient"], "aligned", "swapped")),
        "parent_penalty": ("within", *diff_rows(by, pops["parent_failures"], "ext_aligned", "ext_parent")),
        "naming": ("between", *gap_rows(by, roles, lambda i: L(by, i, "both") - L(by, i, "swapped"))),
        "completion": ("between", *gap_rows(by, roles, lambda i: L(by, i, "aligned") - L(by, i, "both"))),
        "timing": ("between", *gap_rows(by, padded, lambda i: delta(i, 8) - delta(i, 1))),
        "timing_role_gap_at_one": ("between", *gap_rows(by, padded, lambda i: delta(i, 1))),
    }


def contrasts(by: dict, pops: dict[str, list[dict]], margin: float) -> dict[str, Any]:
    out = {}
    for name, (kind, rows, cl) in contrast_inputs(by, pops).items():
        out[name] = within(rows, cl, margin) if kind == "within" else between(rows, cl, margin)
    gap1 = out.pop("timing_role_gap_at_one")
    out["timing"]["role_gap_at_one_lookup"] = {k: gap1[k] for k in ("estimate", "ci95", "n_goal", "n_ingredient")}
    out["timing"]["applicable"] = bool(gap1["ci95"][1] < 0)
    return out


# ---------------- family, verdicts, readings ----------------


def family(per_model: dict[str, dict], expected: list[str]) -> dict[str, dict]:
    """The 50 tests, Holm-adjusted together; a missing model or NaN p enters as 1."""
    p, info = {}, {}
    for tag in expected:
        for c in CONTRASTS:
            rec = per_model.get(tag, {}).get("contrasts", {}).get(c)
            for t in TESTS:
                key = f"{tag}:{c}:{t}"
                pv = rec[f"p_{t}"] if rec else float("nan")
                p[key] = pv if pv == pv else 1.0
                info[key] = {"raw_p": pv, "padded": pv != pv}
    adj = holm(p)
    return {k: {**info[k], "p_in_family": p[k], "p_holm": adj[k]} for k in p}


def verdict(rec: dict | None, fam: dict, tag: str, contrast: str, interpretable: bool | None) -> str:
    if rec is None:
        return "waiting"
    if not interpretable:
        return "not interpretable"
    if contrast == "timing" and not rec.get("applicable", False):
        return "not applicable"
    lo, hi = rec["ci95"]
    if fam[f"{tag}:{contrast}:differs"]["p_holm"] < ALPHA:
        if rec["estimate"] > 0 and lo > 0:
            return "confirmed above zero"
        if rec["estimate"] < 0 and hi < 0:
            return "confirmed below zero"
    if fam[f"{tag}:{contrast}:equivalent"]["p_holm"] < ALPHA:
        return "equivalent to zero"
    return "inconclusive"


def reading(v: dict[str, str]) -> dict[str, str]:
    """The design's per-model reading tables (section 8)."""
    if "not interpretable" in v.values():
        return {"penalty": "not interpretable", "timing": "not interpretable", "lift": "not interpretable"}
    p, dn, dc, t = v["parent_penalty"], v["naming"], v["completion"], v["timing"]
    if p == "confirmed below zero":
        if dn == "confirmed below zero":
            pen = "naming penalty (a completion part cannot be separated from the dilution of naming)"
        elif dc == "confirmed below zero":
            pen = "completion penalty"
        elif dn == "inconclusive" and dc == "inconclusive":
            pen = "penalty present, mechanism unresolved"
        else:
            pen = "penalty present; pattern of naming and completion not in the table (inconclusive)"
    elif p == "equivalent to zero":
        pen = "no penalty under the clean prompt"
    elif p == "confirmed above zero":
        pen = "completing a named goal raises predicted success (no penalty)"
    else:
        pen = "inconclusive on the penalty"
    tim = {"confirmed above zero": "timing", "equivalent to zero": "content", "not applicable": "not applicable",
           "confirmed below zero": "padding deepens the penalty (F4)"}.get(t, "inconclusive")
    lift = {"confirmed above zero": "lift present", "equivalent to zero": "no lift", "confirmed below zero": "negative lift"}.get(v["lift"], "inconclusive")
    return {"penalty": pen, "timing": tim, "lift": lift}


def across(verdicts: dict[str, dict], gates: dict[str, bool | None]) -> dict[str, Any]:
    passing = [t for t, ok_ in gates.items() if ok_]
    below = [t for t in passing if verdicts[t]["parent_penalty"] == "confirmed below zero"]
    equiv = [t for t in passing if verdicts[t]["parent_penalty"] == "equivalent to zero"]
    if len(passing) < 2:
        pattern = "fewer than two models pass the gate: no statement across families"
    elif below and len(below) == len(passing):
        pattern = "penalty general across the models that pass"
    elif below == ["llama3.1-8b"]:
        pattern = "penalty below zero for Llama-3.1 only (F3)"
    elif below and equiv:
        pattern = "penalty model-specific"
    else:
        pattern = "no clear pattern"
    return {"models_passing_gate": passing, "penalty_below_zero": below, "penalty_equivalent_to_zero": equiv,
            "penalty_pattern": pattern,
            "F1_llama_no_penalty": ("not testable (Llama-3.1 fails the gate)" if "llama3.1-8b" not in passing else
                                    verdicts["llama3.1-8b"]["parent_penalty"] in ("equivalent to zero", "confirmed above zero")),
            "F3_penalty_not_shared": bool(len(passing) >= 2 and not [t for t in below if t != "llama3.1-8b"])}


# ---------------- descriptive readouts ----------------


def descriptive(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    d = lambda recs, a, b: describe(*diff_rows(by, recs, a, b))
    mf = pops["main_failures"]
    out["delta_L_main"] = {"all": d(mf, "aligned", "swapped"),
                           **{f"role_{x}": d([r for r in mf if r["role"] == x], "aligned", "swapped") for x in ("goal", "ingredient")},
                           **{f"class_{x}": d([r for r in mf if r["failure_class"] == x], "aligned", "swapped")
                              for x in ("used_up", "too_few", "never_obtained")},
                           **{f"bin_{x}": d([r for r in mf if r["length_bin"] == x], "aligned", "swapped") for x in ("1-3", "4-8", "9-12")}}
    out["neutral_contrasts"] = {f"{a}_minus_neutral": d(mf, a, "neutral") for a in ("aligned", "swapped", "both")}
    out["successes_delta_L"] = d(pops["successes"], "aligned", "swapped")
    roles = pops["goal"] + pops["ingredient"]
    yn = lambda f: describe_gap(*gap_rows(by, roles, f))
    out["yes_no"] = {"lift": d(pops["ingredient"], "yn_aligned", "yn_swapped"),
                     "naming": yn(lambda i: L(by, i, "yn_both") - L(by, i, "yn_swapped")),
                     "completion": yn(lambda i: L(by, i, "yn_aligned") - L(by, i, "yn_both")),
                     "delta_L_main": d(mf, "yn_aligned", "yn_swapped")}
    pf = pops["parent_failures"]
    held_none = lambda r: (r["fit_measures"]["completion"][0] if isinstance(r["fit_measures"], dict) else 0) == 0
    out["parent"] = {
        "penalty_by_class": {c: d([r for r in pf if r["failure_class"] == c], "ext_aligned", "ext_parent")
                             for c in ("too_few", "never_obtained")},
        "penalty_by_bin": {b: d([r for r in pf if r["length_bin"] == b], "ext_aligned", "ext_parent") for b in ("1-3", "4-8", "9-12")},
        "penalty_by_inputs_held": {"none": d([r for r in pf if held_none(r)], "ext_aligned", "ext_parent"),
                                   "some": d([r for r in pf if not held_none(r)], "ext_aligned", "ext_parent")},
        "penalty_by_rule": {x: d([r for r in pf if r["parent_rule"] == x], "ext_aligned", "ext_parent") for x in ("strict", "word_overlap")},
        "penalty_successes": d(pops["parent_successes"], "ext_aligned", "ext_parent"),
        "lift_as_intermediate": d(pf, "ext_parent", "ext_swapped"),
        "goal_effect_extended": d(pf, "ext_aligned", "ext_swapped"),
        "list_control": d(pf, "ext_aligned", "aligned")}
    first = lambda r: sorted((r["goal"], r["partner"]))[0] == r["goal"]
    pos = {}
    for name, f in (("naming", lambda i: L(by, i, "both") - L(by, i, "swapped")), ("completion", lambda i: L(by, i, "aligned") - L(by, i, "both"))):
        for side, keep in (("object_named_first", True), ("object_named_second", False)):
            sub = [r for r in roles if first(r) == keep]
            pos[f"{name}|{side}"] = describe_gap(*gap_rows(by, sub, f))
    out["two_item_by_position"] = pos
    out["lift_by_overlap"] = {f"overlap_{x}": d([r for r in pops["ingredient"] if r["object_goal_overlap"] is x], "aligned", "swapped")
                              for x in (True, False)}
    padded = pops["padded"]
    delta = lambda i, k: L(by, i, f"pad{k}_aligned") - L(by, i, f"pad{k}_swapped")
    pad_parts = {}
    for role in ("goal", "ingredient"):
        sub = [r for r in padded if r["role"] == role]
        pad_parts[f"{role}_eight_minus_one"] = describe(*_vals(sub, lambda i: delta(i, 8) - delta(i, 1)))
        pad_parts[f"{role}_one_minus_none"] = describe(*_vals(sub, lambda i: delta(i, 1) - (L(by, i, "aligned") - L(by, i, "swapped"))))
    out["padding"] = pad_parts
    return out


def _vals(recs: list[dict], f: Callable[[str], float]) -> tuple[list[float], list[str]]:
    v, c = [], []
    for r in recs:
        x = f(r["id"])
        if x == x:
            v.append(x)
            c.append(r["pair_id"])
    return v, c


# ---------------- the run ----------------


def model_block(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    gate = competence(by, pops)
    m = margin_of(by, pops)
    return {"competence": gate, "equivalence_margin": m, "contrasts": contrasts(by, pops, m["margin"]),
            "descriptive": descriptive(by, pops)}


def analyse(records: list[dict], shots, scores_by_tag: dict[str, list[dict]], expected: list[str]) -> dict[str, Any]:
    pops = populations(records)
    for r in records:  # fit_measures may be stored as a JSON string
        if isinstance(r.get("fit_measures"), str):
            import json
            r["fit_measures"] = json.loads(r["fit_measures"])
    per_model, missing = {}, {}
    for tag in expected:
        if tag not in scores_by_tag:
            continue
        by, miss = index_scores(scores_by_tag[tag], records, shots, tag)
        missing[tag] = len(miss)
        per_model[tag] = model_block(by, pops)
    fam = family(per_model, expected)
    complete = len(per_model) == len(expected) and not any(missing.values())
    out: dict[str, Any] = {"models_expected": expected, "missing_scores": missing, "family": fam,
                           "family_size": len(fam), "per_model": per_model}
    if not complete:
        out["verdicts"] = "withheld: " + ("models not scored: " + str([t for t in expected if t not in per_model])
                                         if len(per_model) < len(expected) else f"missing scores {missing}")
        return out
    gates = {t: per_model[t]["competence"]["interpretable"] for t in expected}
    verdicts = {t: {c: verdict(per_model[t]["contrasts"][c], fam, t, c, gates[t]) for c in CONTRASTS} for t in expected}
    out["verdicts"] = {t: {"contrasts": verdicts[t], "reading": reading(verdicts[t]), "competence": gates[t]} for t in expected}
    out["across_models"] = across(verdicts, gates)
    return out


def run_textcraft_r2_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    d = _init_run(run_id, cfg, inputs=(rpath, spath, mpath))
    records = load_records(rpath)
    shots = P.fewshot_examples(spath)
    expected = list(cfg.get("models", R.TAGS))
    scores = {t: s for t in expected if (s := load_scores(t))}
    try:
        metrics = analyse(records, shots, scores, expected)
    except StaleScore as e:
        write_status(run_id, f"stale score: {e}", ok=False)
        return 1
    write_json(d / "metrics.json", metrics)
    if isinstance(metrics["verdicts"], str):
        write_status(run_id, f"{metrics['verdicts']}", ok=False)
        return 1
    lines = [f"{t}: gate {'pass' if v['competence'] else 'fail'}; " + "; ".join(f"{c} {x}" for c, x in v["contrasts"].items())
             for t, v in metrics["verdicts"].items()]
    write_status(run_id, "\n".join(lines + [f"across: {metrics['across_models']['penalty_pattern']}"]), ok=True)
    return 0
