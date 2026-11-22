"""TextCraft round 2 analysis: `textcraft_r2_analysis` (notes/textcraft_r2_design.md, sections 3 and 6 to 9).

Reads every model's `textcraft_r2_score_<tag>` shards and writes metrics.json: per model the overall competence gate
(wave A's rule, on round 2's anchored yes/no question under the neutral line) and the contrast-specific competence, the
five confirmatory contrasts (intermediate lift, within-item parent penalty, naming / partial-completion, completion
with naming held, timing), each with a test of difference and a two one-sided equivalence test, all 50 in one Holm
family of fixed size; the verdicts, interpretations and readings of the design's tables; the coded rejection list and
the descriptive central-question mapping (design 14.W); and the descriptive readouts. Scores are joined to records by id and record hash; a stale or unknown score
stops the analysis, and with any score missing the verdicts are withheld.

Amended before the freeze (user decisions, 2026-11-21; design section 14): each contrast is read against the
competence of the model in its own action categories (`local_competence`: goal crafting for the parent penalty,
ingredient crafting for the lift, both for naming, completion and timing), with wave A's criteria applied within each
population; the overall gate stays reported. A failed local gate no longer hides a verdict: the statistical verdict
stands and its interpretation is limited ("a measurable goal effect", not "a distortion in a competent predictor").
The equivalence margin is min(0.3 x SD, 0.4 nats), with the power of each equivalence test reported; the mechanism
readings need the naming contribution to be constrained before any completion-only reading.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf.analysis.stats import holm
from wf.runs import _init_run
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft.r2_score import load_records, verified_files
from wf.util import read_jsonl, runs_root, write_json, write_status

B_DRAWS = 20000  # review 1: at least 20,000 for the 50 family tests
N_FLIPS = 20000
DESC_DRAWS = 2000  # descriptive intervals
AUC_DRAWS = 1000
ALPHA = 0.05
MARGIN_SD = 0.3  # equivalence margin: min(0.3 x the item SD of ΔL (aligned - swapped) on the main failures, 0.4 nats)
MARGIN_CAP = 0.4  # amendment A3 (2026-11-21), fixed before the freeze; two-candidate nats, every contrast
Z_HOLM_FIRST = 3.090232  # one-sided normal quantile at 0.05 / 50, the most stringent Holm step
Z_ALPHA = 1.644854  # one-sided normal quantile at 0.05, the least stringent
POWER_MIN = 0.8  # an equivalence test with less power at a true zero (at the first Holm step) is "underpowered"
# Contrast-specific competence (amendment A1; design section 14): the yes/no neutral-line AUC, point estimate, at
# least LOCAL_AUC_MIN within every population a contrast is read in; the other three wave-A criteria stay on the
# full population (overall gate). No fallback to another scale.
LOCAL_AUC_MIN = 0.70
CONTRAST_POPULATIONS = {"lift": ("ingredient_all",), "parent_penalty": ("parent", "goal_main"),
                        "naming": ("goal_main", "ingredient_matched"), "completion": ("goal_main", "ingredient_matched"),
                        "timing": ("goal_padded", "ingredient_padded_matched")}
MATCHED_CLASSES = ("too_few", "never_obtained")  # review 1: the ingredient control of Dn, Dc and T, class-matched
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
    arms = {r["id"]: set(r["arms"]) for r in records}
    by: dict[tuple[str, str], dict] = {}
    for s in scores:
        if s.get("model") != tag:
            raise StaleScore(f"{tag}: a score of model {s.get('model')!r}")
        if hashes.get(s["transition_id"]) != s.get("item_hash"):
            raise StaleScore(f"{tag}: score for {s['transition_id']} {s['condition']} does not match its record")
        if s["condition"] not in arms[s["transition_id"]]:
            raise StaleScore(f"{tag}: score for {s['transition_id']} under {s['condition']}, not an arm of the record")
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
    if stat is mean or stat is _gap:  # the same resamples as the general path, from per-pair sums
        return _fast_draws(stat is _gap, rows, clusters, B, seed)
    keys, by = _clusters(rows, clusters)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        out.append(stat([r for k in rng.choices(keys, k=len(keys)) for r in by[k]]))
    return [x for x in out if x == x]


def _fast_draws(gap: bool, rows: list, clusters: list[str], B: int, seed: int) -> list[float]:
    """mean or _gap over pair resamples, from each pair's sums and counts; identical draws to the general path."""
    keys, by = _clusters(rows, clusters)
    agg = {}
    for k in keys:
        if gap:
            g = [x for r, x in by[k] if r == "goal"]
            i = [x for r, x in by[k] if r == "ingredient"]
            agg[k] = (sum(g), len(g), sum(i), len(i))
        else:
            agg[k] = (sum(by[k]), len(by[k]), 0.0, 0)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        sg = ng = si = ni = 0
        for k in rng.choices(keys, k=len(keys)):
            a = agg[k]
            sg += a[0]; ng += a[1]; si += a[2]; ni += a[3]
        if gap:
            v = (sg / ng if ng else float("nan")) - (si / ni if ni else float("nan"))
        else:
            v = sg / ng if ng else float("nan")
        if v == v:
            out.append(v)
    return out


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


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def equivalence_power(se: float, margin: float, z: float) -> float:
    """Approximate power of the two one-sided tests at a true effect of zero: P(|estimate| + z se < margin)."""
    if not (se == se and margin == margin) or se <= 0:
        return float("nan")
    return max(0.0, 2 * _phi((margin - z * se) / se) - 1)


def _power_block(d: list[float], margin: float) -> dict[str, Any]:
    se = statistics.stdev(d) if len(d) > 1 else float("nan")
    first, last = equivalence_power(se, margin, Z_HOLM_FIRST), equivalence_power(se, margin, Z_ALPHA)
    return {"bootstrap_se": se, "equivalence_power_at_zero_holm_first_step": first,
            "equivalence_power_at_zero_alpha_05": last, "equivalence_underpowered": not (first >= POWER_MIN)}


def mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def within(values: list[float], clusters: list[str], margin: float) -> dict[str, Any]:
    """A mean of per-item differences: estimate, 95% interval, sign-flip p, TOST p."""
    d = draws(mean, values, clusters)
    return {"estimate": mean(values), "ci95": [pct(d, 0.025), pct(d, 0.975)], "ci90": [pct(d, 0.05), pct(d, 0.95)],
            "n_items": len(values), "n_pairs": len(set(clusters)), "p_differs": sign_flip_p(values, clusters),
            "p_equivalent": tost_p(d, margin), "margin": margin, "p_method": "sign-flip of whole pairs",
            **_power_block(d, margin)}


def _gap(rows: list[tuple[str, float]]) -> float:
    return mean([x for r, x in rows if r == "goal"]) - mean([x for r, x in rows if r == "ingredient"])


def between(rows: list[tuple[str, float]], clusters: list[str], margin: float) -> dict[str, Any]:
    """Goal-role mean minus ingredient mean: estimate, 95% interval, bootstrap p, TOST p."""
    d = draws(_gap, rows, clusters)
    return {"estimate": _gap(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "ci90": [pct(d, 0.05), pct(d, 0.95)],
            "goal_mean": mean([x for r, x in rows if r == "goal"]), "ingredient_mean": mean([x for r, x in rows if r == "ingredient"]),
            "n_goal": sum(r == "goal" for r, _ in rows), "n_ingredient": sum(r == "ingredient" for r, _ in rows),
            "n_pairs": len(set(clusters)), "p_differs": bootstrap_p(d), "p_equivalent": tost_p(d, margin),
            "margin": margin, "p_method": "pair bootstrap", **_power_block(d, margin)}


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


def word_contained(a: str, b: str) -> bool:
    """a's words appear, in order and contiguous, among b's words (so "stone stairs" is not inside "smooth sandstone
    stairs", although it is a character substring of it)."""
    x, y = a.split(), b.split()
    return any(y[i:i + len(x)] == x for i in range(len(y) - len(x) + 1))


def overlap_kind(obj: str, goal: str) -> str:
    """full_name: the object's whole name inside the goal's; word_only: a shared word but not the whole name; none."""
    if word_contained(obj, goal):
        return "full_name"
    return "word_only" if set(obj.split()) & set(goal.split()) else "none"


def parent_subset(r: dict) -> str:
    """Review 1: strict (the parent shares no word with G), full_name (G's whole name inside Z's), word_only."""
    if r["parent_rule"] == "strict":
        return "strict"
    return "full_name" if word_contained(r["goal"], r["parent"]) else "word_only"


def populations(records: list[dict]) -> dict[str, list[dict]]:
    main = [r for r in records if r["kind"] == "main"]
    mf = [r for r in main if r["population"] == "main"]
    ms = [r for r in main if r["population"] == "success"]
    goal = [r for r in mf if r["role"] == "goal" and r["failure_class"] != "used_up"]
    ingredient = [r for r in mf if r["role"] == "ingredient"]
    padded = [r for r in mf if "pad8_aligned" in r["arms"]]
    pad_bins = set(R.PAD_BINS)
    return {"main_failures": mf, "successes": ms, "goal": goal, "ingredient": ingredient,
            "ingredient_matched": [r for r in ingredient if r["failure_class"] in MATCHED_CLASSES],
            "padded": padded,
            "padded_goal": [r for r in padded if r["role"] == "goal" and r["failure_class"] != "used_up"],
            "padded_ingredient_matched": [r for r in padded if r["role"] == "ingredient" and r["failure_class"] in MATCHED_CLASSES],
            "parent_failures": [r for r in records if r["kind"] == "parent" and r["population"] == "main"],
            "parent_successes": [r for r in records if r["kind"] == "parent" and r["population"] == "success"],
            "goal_successes": [r for r in ms if r["role"] == "goal"],
            "ingredient_successes": [r for r in ms if r["role"] == "ingredient"],
            "goal_successes_padded_bins": [r for r in ms if r["role"] == "goal" and r["length_bin"] in pad_bins],
            "ingredient_successes_padded_bins": [r for r in ms if r["role"] == "ingredient" and r["length_bin"] in pad_bins]}


# The competence populations (amendment A1): name -> (failures key, successes key)
COMPETENCE_POPULATIONS = {"ingredient_all": ("ingredient", "ingredient_successes"),
                          "ingredient_matched": ("ingredient_matched", "ingredient_successes"),
                          "goal_main": ("goal", "goal_successes"),
                          "parent": ("parent_failures", "parent_successes"),
                          "goal_padded": ("padded_goal", "goal_successes_padded_bins"),
                          "ingredient_padded_matched": ("padded_ingredient_matched", "ingredient_successes_padded_bins")}


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


def gate_checks(by: dict, failures: list[dict], successes: list[dict]) -> dict[str, dict]:
    """Wave A's four criteria (families.COMPETENCE) on the yes/no question under the neutral line, within the given
    failures and successes."""
    yn = lambda rs: [by[(r["id"], "yn_neutral")] for r in rs if (r["id"], "yn_neutral") in by]
    fs, ss = yn(failures), yn(successes)
    a = auc([x["L"] for x in ss], [x["L"] for x in fs])
    yes_f = mean([x["L"] > 0 for x in fs])
    yes_s = mean([x["L"] > 0 for x in ss])
    masses = [_mass(x) for x in fs + ss]
    m = statistics.median(masses) if masses else float("nan")
    return {"auc_neutral": {"value": a, "pass": a >= COMPETENCE["auc_neutral_min"]},
            "yes_share_neutral_failures": {"value": yes_f, "pass": yes_f <= COMPETENCE["yes_share_neutral_failures_max"]},
            "yes_share_neutral_successes": {"value": yes_s, "pass": yes_s >= COMPETENCE["yes_share_neutral_successes_min"]},
            "median_yes_no_mass_neutral": {"value": m, "pass": m >= COMPETENCE["median_yes_no_mass_neutral_min"]}}


def classification(by: dict, failures: list[dict], successes: list[dict]) -> dict[str, Any]:
    """Under the neutral line, on each scale the records have: the share of successes predicted to succeed and of
    failures predicted to fail (L above or below zero), and the AUC with a pair-bootstrap interval."""
    out = {}
    for scale, arm in (("yes_no", "yn_neutral"), ("two_candidate", "neutral")):
        val = lambda rs: [(r["pair_id"], by[(r["id"], arm)]["L"]) for r in rs if (r["id"], arm) in by]
        sv, fv = val(successes), val(failures)
        if not sv and not fv:
            continue  # the parent records have the neutral line on the yes/no scale only
        out[scale] = {"success_classification_rate": mean([L > 0 for _p, L in sv]),
                      "failure_classification_rate": mean([L < 0 for _p, L in fv]), "auc": auc_ci(sv, fv)}
    return out


def local_competence(by: dict, pops: dict[str, list[dict]], overall_checks: dict[str, dict]) -> dict[str, Any]:
    """Amendment A1 (user decision and review 1, 2026-11-21). For each competence population: the yes/no
    neutral-line AUC (point estimate; passes at LOCAL_AUC_MIN or above), the classification rates and AUCs on both
    scales, and the heuristic tracking within it. Per contrast: competent when every one of its populations passes
    and the other three wave-A criteria pass on the full population (the overall gate's yes shares and mass)."""
    pop_out: dict[str, Any] = {}
    for name, (fk, sk) in COMPETENCE_POPULATIONS.items():
        f, s = pops[fk], pops[sk]
        a = gate_checks(by, f, s)["auc_neutral"]["value"]
        pop_out[name] = {"yes_no_auc": a, "passes": bool(a >= LOCAL_AUC_MIN), "n_failures": len(f), "n_successes": len(s),
                         "classification": classification(by, f, s), "heuristics": heuristic_fits(by, f + s)}
        if name == "goal_main":  # review 2: the central question's paired comparison of tracking AUCs
            pop_out[name]["tracking_differences"] = tracking_differences(by, f + s)
    others = all(overall_checks[k]["pass"] for k in ("yes_share_neutral_failures", "yes_share_neutral_successes",
                                                       "median_yes_no_mass_neutral"))
    per_contrast = {}
    for c, names in CONTRAST_POPULATIONS.items():
        aucs = {n: pop_out[n]["yes_no_auc"] for n in names}
        per_contrast[c] = {"populations": aucs, "smallest_auc": min(aucs.values()),
                           "competent": bool(all(pop_out[n]["passes"] for n in names) and others)}
    return {"populations": pop_out, "other_criteria_on_full_population_pass": others, "per_contrast": per_contrast,
            "rule": f"yes/no neutral-line AUC >= {LOCAL_AUC_MIN} in every population of the contrast (point estimate), "
                    "and the other three wave-A criteria on the full population"}


def competence(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    """Wave A's overall gate on the yes/no question under the neutral line (main failures against the 600
    successes), kept visible for every model (amendment A1), and the descriptive role-, class- and parent-level AUCs
    on both scales."""
    f, s = pops["main_failures"], pops["successes"]
    checks = gate_checks(by, f, s)
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
    for scale, arm in (("yes_no", "yn_neutral"), ("two_candidate", "neutral")):
        val = lambda rs: [(r["pair_id"], by[(r["id"], arm)]["L"]) for r in rs if (r["id"], arm) in by]
        detail[scale]["ingredient_class_" + "_".join(MATCHED_CLASSES)] = auc_ci(val(pops["ingredient_successes"]),
                                                                                val(pops["ingredient_matched"]))
        for c in ("used_up", "too_few", "never_obtained"):
            detail[scale][f"ingredient_class_{c}"] = auc_ci(val(pops["ingredient_successes"]),
                                                            val([r for r in pops["ingredient"] if r["failure_class"] == c]))
    fm = [_first_mass(by[(r["id"], "neutral")]) for r in f if (r["id"], "neutral") in by]
    detail["two_candidate_first_token_mass_neutral_median"] = statistics.median(fm) if fm else float("nan")
    detail["heuristics"] = heuristic_fits(by, f + s)
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks, "rule": COMPETENCE,
            "local": local_competence(by, pops, checks), "descriptive": detail}


def tracking_differences(by: dict, items: list[dict], arm: str = "yn_neutral") -> dict[str, Any]:
    """Tracking AUC of H1 and of H2 and H9 minus that of H6 (the truth), on the yes/no question under the neutral
    line, with a paired pair-bootstrap 95% interval (the same resamples for both AUCs; DESC_DRAWS draws)."""
    from wf.textcraft.r2_heuristics import features

    rows = []
    for r in items:
        if (r["id"], arm) in by:
            f = features(r)
            rows.append((r["pair_id"], by[(r["id"], arm)]["L"], f["H1"], f["H2andH9"], f["H6"]))

    def track(rs: list, k: int) -> float:
        return auc([x[1] for x in rs if x[k]], [x[1] for x in rs if not x[k]])

    out = {}
    for name, k in (("H1_minus_H6", 2), ("H2andH9_minus_H6", 3)):
        stat = lambda rs, k=k: track(rs, k) - track(rs, 4)
        d = draws(stat, rows, [x[0] for x in rows], B=DESC_DRAWS)
        out[name] = {"estimate": stat(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "n_items": len(rows)}
    return out


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa of two binary series."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def heuristic_fits(by: dict, items: list[dict]) -> dict[str, Any]:
    """Descriptive (design section 9): how closely each model's neutral-line scores follow the competence note's
    heuristics H1, H2, H6 (the truth) and H2 and H9 (definitions fixed at 9a83738; wf/textcraft/r2_heuristics.py),
    on the given failures and successes, on both scales: the tracking AUC (the model's L between the items a
    heuristic calls a success and those it calls a failure), the same within the true failures and within the true
    successes, and the agreement and Cohen's kappa between the model's decision (L above 0) and the heuristic."""
    from wf.textcraft.r2_heuristics import HEURISTICS, features

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
    return {"sd_delta_L_main": sd, "margin": min(MARGIN_SD * sd, MARGIN_CAP), "margin_sd_rule": MARGIN_SD * sd,
            "cap_nats": MARGIN_CAP, "rule": f"min({MARGIN_SD} x item SD of ΔL on the main failures, {MARGIN_CAP} nats)"}


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


def two_item_rows(pops: dict[str, list[dict]]) -> list[dict]:
    """The rows of the two-item contrasts and of the role gap: goal-role failures (without used up) and class-matched
    ingredient failures that have a two-item line and whose other named goal is not already made or held (14.V1)."""
    return [r for r in pops["goal"] + pops["ingredient_matched"] if r["both_task_line"] and not R.two_item_partner_done(r)]


def contrast_inputs(by: dict, pops: dict[str, list[dict]]) -> dict[str, tuple]:
    """Rows of the five contrasts. Dn, Dc and T compare goal-role failures with class-matched ingredient failures
    (too few and never obtained; review 1), so the difference in differences is not also a difference in failure
    class; Λ keeps all 990 ingredient failures."""
    # user point 1: a record whose other named goal is already made or held leaves naming and completion (none exist)
    roles = two_item_rows(pops)
    padded = pops["padded_goal"] + pops["padded_ingredient_matched"]
    delta = lambda rid, k: (L(by, rid, f"pad{k}_aligned") - L(by, rid, f"pad{k}_swapped")) if k else \
        (L(by, rid, "aligned") - L(by, rid, "swapped"))
    return {
        "lift": ("within", *diff_rows(by, pops["ingredient"], "aligned", "swapped")),
        "parent_penalty": ("within", *diff_rows(by, pops["parent_failures"], "ext_aligned", "ext_parent")),
        "naming": ("between", *gap_rows(by, roles, lambda i: L(by, i, "both") - L(by, i, "swapped"))),
        "completion": ("between", *gap_rows(by, roles, lambda i: L(by, i, "aligned") - L(by, i, "both"))),
        "timing": ("between", *gap_rows(by, padded, lambda i: delta(i, 8) - delta(i, 1))),
        "timing_role_gap_at_one": ("between", *gap_rows(by, padded, lambda i: delta(i, 1))),
        "timing_one_vs_none": ("between", *gap_rows(by, padded, lambda i: delta(i, 1) - delta(i, 0))),
        # review 2: the role gap (= naming + completion on these rows), with the family's resamples, for rejection
        # criterion R3; outside the Holm family
        "role_gap": ("between", *gap_rows(by, roles, lambda i: L(by, i, "aligned") - L(by, i, "swapped"))),
    }


def contrasts(by: dict, pops: dict[str, list[dict]], margin: float) -> dict[str, Any]:
    out = {}
    for name, (kind, rows, cl) in contrast_inputs(by, pops).items():
        out[name] = within(rows, cl, margin) if kind == "within" else between(rows, cl, margin)
    rg = out.pop("role_gap")
    out["role_gap"] = {k: rg[k] for k in ("estimate", "ci95", "goal_mean", "ingredient_mean", "n_goal", "n_ingredient",
                                         "n_pairs")}
    out["role_gap"]["matched"] = matched_gap(by, two_item_rows(pops), lambda i: L(by, i, "aligned") - L(by, i, "swapped"),
                                             B=B_DRAWS)
    gap1 = out.pop("timing_role_gap_at_one")
    one = out.pop("timing_one_vs_none")
    keep = ("estimate", "ci95", "goal_mean", "ingredient_mean", "n_goal", "n_ingredient")
    out["timing"]["role_gap_at_one_lookup"] = {k: gap1[k] for k in keep}
    out["timing"]["applicable"] = bool(gap1["ci95"][1] < 0)
    # Review 1: the change from no look-up to one, goal minus ingredient, beside T. A T whose 95% interval holds this
    # change is comparable to it and is not read as timing.
    out["timing"]["one_lookup_vs_none"] = {k: one[k] for k in keep}
    lo, hi = out["timing"]["ci95"]
    out["timing"]["comparable_to_one_lookup_change"] = bool(lo <= one["estimate"] <= hi)
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


def verdict(rec: dict | None, fam: dict, tag: str, contrast: str) -> str:
    """The statistical verdict (amendment A1: never withheld for competence; read with `interpretation`)."""
    if rec is None:
        return "waiting"
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


def interpretation(v: str, competent: bool, rec: dict | None) -> str:
    """How far a verdict can be read (design section 14): a statistically measurable goal effect, or a distortion in
    a competent predictor; an equivalence is "no effect larger than the margin", never "no effect"."""
    if v in ("confirmed above zero", "confirmed below zero"):
        return ("distortion in a competent predictor" if competent else
                "statistically measurable goal effect; the model is not competent in this contrast's populations, so "
                "it is not read as a distortion of a working state judgement")
    if v == "equivalent to zero":
        return ("no effect larger than the margin, in a competent predictor" if competent else
                "no effect larger than the margin; limited interpretation (not competent in these populations)")
    if v == "inconclusive":
        return "inconclusive" + (" (equivalence underpowered)" if rec and rec.get("equivalence_underpowered") else "")
    return v


READING_CONTRASTS = {"penalty": ("parent_penalty", "naming", "completion"), "timing": ("timing",), "lift": ("lift",)}


def reading(v: dict[str, str], timing_comparable: bool = False, competent: dict[str, bool] | None = None) -> dict[str, str]:
    """The design's per-model reading rules (section 8, amended in sections 14 and 15). Any combination of P, Dn and
    Dc not in the table reads "penalty present, mechanism unresolved" when P is confirmed below zero. When a contrast a
    reading uses is not competent, the reading says so (review 2): "measurable effect only; not competent for ...".
    """
    p, dn, dc, t = v["parent_penalty"], v["naming"], v["completion"], v["timing"]
    naming_constrained = dn in ("equivalent to zero", "confirmed above zero")
    if p == "confirmed below zero":
        if dn == "confirmed below zero":
            pen = ("naming / partial-completion penalty (naming the crafted item, or completing one item of a named "
                   "two-item goal, lowers predicted success; a completion part cannot be separated from the dilution of "
                   "naming)")
        elif dc == "confirmed below zero" and naming_constrained:
            pen = "completion penalty, naming constrained"
        elif dc == "confirmed below zero":
            pen = "completion contributes; naming not constrained (no completion-only conclusion)"
        else:
            pen = "penalty present, mechanism unresolved"
    elif p == "equivalent to zero":
        pen = "no penalty larger than the margin under the clean prompt"
    elif p == "confirmed above zero":
        pen = "completing a named goal raises predicted success (no penalty)"
    else:
        pen = "inconclusive on the penalty"
    if t == "confirmed above zero":
        tim = ("sensitivity to added empty-inventory steps (elapsed turns or repeated emphasis; not separable from "
               "prompt length, repetition or attention)")
        if timing_comparable:
            tim += "; comparable to the one-look-up change, so not read as timing"
    elif t == "equivalent to zero":
        tim = "no sensitivity to added empty-inventory steps larger than the margin"
    elif t == "confirmed below zero":
        tim = "added steps deepen the penalty (R5)"
    else:
        tim = t if t in ("not applicable", "waiting") else "inconclusive"
    lift = {"confirmed above zero": "lift present", "equivalent to zero": "no lift larger than the margin",
            "confirmed below zero": "negative lift"}.get(v["lift"], "inconclusive")
    out = {"penalty": pen, "timing": tim, "lift": lift}
    if competent is not None:
        for key, cs in READING_CONTRASTS.items():
            missing = [c for c in cs if not competent.get(c)]
            if missing:
                out[key] += "; measurable effect only; not competent for " + ", ".join(missing)
    return out


def _sign(lo: float, hi: float) -> int:
    """+1 when a 95% interval lies above zero, -1 below, 0 otherwise."""
    return 1 if lo > 0 else -1 if hi < 0 else 0


def rejections_for_model(v: dict[str, str], comp: dict[str, bool], block: dict[str, Any]) -> dict[str, dict]:
    """The coded per-model part of the rejection list (design 15.R): R1, R3 and R5. Each entry: whether it can be
    tested (the competence it needs), whether it fires, and a note."""
    out: dict[str, dict] = {}
    p = v["parent_penalty"]
    out["R1_penalty_absent"] = {"needs": "competent for P", "testable": bool(comp["parent_penalty"]),
                                "fires": bool(comp["parent_penalty"] and p in ("equivalent to zero", "confirmed above zero")),
                                "rejects": "the penalty component of the decomposition, for this model"}
    rg = block["contrasts"]["role_gap"]
    m = rg["matched"]
    p_sign = 1 if p == "confirmed above zero" else -1 if p == "confirmed below zero" else 0
    rg_sign = _sign(*rg["ci95"])
    m_sign = _sign(*m["ci95"]) if m["ci95"][0] == m["ci95"][0] else 0
    testable = bool(comp["parent_penalty"] and comp["naming"] and comp["completion"])
    disagree = bool(p_sign and rg_sign and p_sign != rg_sign)
    fires = bool(testable and disagree and m_sign == rg_sign)
    note = ("" if not (testable and disagree) else "rejects" if fires else
            "P and the unmatched role gap disagree; the matched role gap does not confirm (item-structure confound)")
    out["R3_penalty_and_role_gap_disagree"] = {"needs": "competent for P, Dn and Dc", "testable": testable, "fires": fires,
                                               "note": note, "rejects": "the decomposition, for this model"}
    t = v["timing"]
    out["R5_added_steps_deepen_penalty"] = {"needs": "competent for T", "testable": bool(comp["timing"]),
                                            "fires": bool(comp["timing"] and t == "confirmed below zero"),
                                            "rejects": "the claim that the penalty is largest early (in either reading)"}
    return out


def central_question(v: dict[str, str], comp: dict[str, bool], local: dict[str, Any]) -> dict[str, Any]:
    """The descriptive mapping of design 15.C (outside the Holm family), mutually exclusive by construction."""
    tr = local["populations"]["goal_main"]["tracking_differences"]
    # A comparison that is undefined counts for neither side: H2 and H9 calls every goal-role craft a failure, so its
    # tracking AUC (and its difference from H6) is undefined in the goal-role population (design 15.C).
    defined = [k for k in ("H1_minus_H6", "H2andH9_minus_H6") if tr[k]["estimate"] == tr[k]["estimate"]]
    shortcut = any(tr[k]["ci95"][0] > 0 for k in defined)
    h6_best = bool(defined) and all(tr[k]["estimate"] <= 0 for k in defined)
    p_conf = v["parent_penalty"] == "confirmed below zero"
    p_comp = bool(comp["parent_penalty"])
    if p_conf and p_comp and h6_best:
        answer = "mechanism"
    elif p_conf and shortcut:
        answer = "combination"
    elif not p_conf and (not p_comp or shortcut):
        answer = "poor state reasoning"
    else:
        answer = "undecided"
    return {"answer": answer, "P_confirmed_below_zero": p_conf, "competent_for_P": p_comp,
            "H6_tracks_at_least_as_well_as_H1_and_H2andH9": h6_best, "a_shortcut_tracks_better_than_H6": shortcut,
            "lift_null": v["lift"] == "equivalent to zero", "descriptive": True}


def across(verdicts: dict[str, dict], competent_for_penalty: dict[str, bool | None],
           lift: dict[str, dict] | None = None) -> dict[str, Any]:
    """Across models, among the models competent for the penalty contrast (its two goal-crafting populations); the
    across-model rejection criteria R2 (F3) and R4 (F2)."""
    passing = [t for t, ok_ in competent_for_penalty.items() if ok_]
    below = [t for t in passing if verdicts[t]["parent_penalty"] == "confirmed below zero"]
    equiv = [t for t in passing if verdicts[t]["parent_penalty"] == "equivalent to zero"]
    above = [t for t in passing if verdicts[t]["parent_penalty"] == "confirmed above zero"]
    if len(passing) < 2:
        pattern = "fewer than two models are competent for the penalty: no statement across families"
    elif below and len(below) == len(passing):
        pattern = "penalty general across the competent models"
    elif below == ["llama3.1-8b"]:
        pattern = "penalty below zero for Llama-3.1 only"
    elif below and (equiv or above):
        pattern = "penalty model-specific"
    else:
        pattern = "no clear pattern"
    others = [t for t in passing if t != "llama3.1-8b"]
    if [t for t in others if t in below]:
        f3 = False
    elif [t for t in others if t in equiv or t in above]:
        f3 = True
    else:
        f3 = "undecided"
    out = {"models_competent_for_penalty": passing, "penalty_below_zero": below, "penalty_equivalent_to_zero": equiv,
           "penalty_above_zero": above, "penalty_pattern": pattern,
           "F1_llama_no_penalty": ("not testable (Llama-3.1 is not competent for the penalty contrast)"
                                   if "llama3.1-8b" not in passing else
                                   verdicts["llama3.1-8b"]["parent_penalty"] in ("equivalent to zero", "confirmed above zero")),
           "F1_note": "if F1 holds, top-item status of the round-1 goal crafts (recipe structure) is a candidate explanation",
           "F3_penalty_not_shared": f3,
           "R2_penalty_not_shared": {"needs": "competent for P (each model counted)", "fires": f3 is True, "value": f3,
                                     "rejects": "the claim that the penalty is shared across families"}}
    if lift is not None:  # R4 (F2): {tag: {"verdict", "estimate", "competent"}}
        ll, qq = lift.get("llama3.1-8b"), lift.get("qwen3-8b")
        testable = bool(ll and qq and ll["competent"] and qq["competent"])
        fires = bool(testable and ll["verdict"] == "confirmed above zero" and ll["estimate"] >= qq["estimate"])
        out["R4_lift_does_not_explain_the_difference"] = {
            "needs": "Llama-3.1 and Qwen3 both competent for Λ", "testable": testable, "fires": fires,
            "rejects": "the claim that the lift carries the Qwen-Llama difference"}
    return out


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
    roles = two_item_rows(pops)  # review 2: the same rows as the confirmatory two-item contrasts
    yn = lambda f: describe_gap(*gap_rows(by, roles, f))
    # yes/no arms exist for the main records only: the lift, naming and completion contrasts have yes/no versions,
    # the parent penalty and timing do not (design section 9, corrected by review 1)
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
        # review 1: the three pre-registered subsets and their differences (the pooled P is not free of partial naming)
        "penalty_by_subset": {x: d([r for r in pf if parent_subset(r) == x], "ext_aligned", "ext_parent")
                              for x in ("strict", "word_only", "full_name")},
        "penalty_subset_differences": {f"{a}_minus_{b}": _subset_diff(by, pf, a, b)
                                       for a, b in (("strict", "word_only"), ("strict", "full_name"), ("word_only", "full_name"))},
        # user point 4: the direct comparison of strict parents with overlapping ones (word only and whole name)
        "penalty_strict_minus_overlapping": _subset_diff(by, pf, "strict", "overlapping"),
        # each line on its own faithful display: "craft G" on the item's list against "craft Z" on the extended list
        "penalty_on_own_displays": d(pf, "aligned", "ext_parent"),
        # the two-item line on the extended list splits P exactly (descriptive)
        "split_completion_ext_aligned_minus_ext_both": d([r for r in pf if "ext_both" in r["arms"]], "ext_aligned", "ext_both"),
        "split_naming_ext_both_minus_ext_parent": d([r for r in pf if "ext_both" in r["arms"]], "ext_both", "ext_parent"),
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
    # the round-2 role gap on the rows both two-item contrasts use (equals naming plus completion exactly; 14.R3)
    # (the rows already all have a two-item line, so no further guard is needed; review 2's official mutation run
    # showed the earlier guard had become redundant)
    out["role_gap_main"] = describe_gap(*gap_rows(by, roles, lambda i: L(by, i, "aligned") - L(by, i, "swapped")))
    # user point 2: matched sensitivity analyses of the role comparisons (not primary)
    roles_all = two_item_rows(pops)  # review 2: the same rows as role_gap_main
    pad_rows = pops["padded_goal"] + pops["padded_ingredient_matched"]
    dl = lambda i, k: (L(by, i, f"pad{k}_aligned") - L(by, i, f"pad{k}_swapped"))
    out["sensitivity_matched"] = {
        "naming": matched_gap(by, roles_all, lambda i: L(by, i, "both") - L(by, i, "swapped")),
        "completion": matched_gap(by, roles_all, lambda i: L(by, i, "aligned") - L(by, i, "both")),
        "timing": matched_gap(by, pad_rows, lambda i: dl(i, 8) - dl(i, 1)),
        "role_gap": matched_gap(by, roles_all, lambda i: L(by, i, "aligned") - L(by, i, "swapped"))}
    ing = pops["ingredient"]
    out["lift_by_class"] = {c: d([r for r in ing if r["failure_class"] == c], "aligned", "swapped")
                            for c in ("used_up", "too_few", "never_obtained")}
    # review 1: the ingredient's name against the goal's: whole name inside it, a shared word only, or no shared word
    kinds = ("full_name", "word_only", "none")
    out["lift_by_overlap"] = {k: d([r for r in ing if overlap_kind(r["object"], r["goal"]) == k], "aligned", "swapped") for k in kinds}
    im = pops["ingredient_matched"]
    out["two_item_ingredient_controls_by_overlap"] = {
        f"{name}|{k}": describe(*_vals([r for r in im if overlap_kind(r["object"], r["goal"]) == k], f))
        for name, f in (("naming", lambda i: L(by, i, "both") - L(by, i, "swapped")),
                        ("completion", lambda i: L(by, i, "aligned") - L(by, i, "both"))) for k in kinds}
    padded = pops["padded_goal"] + pops["padded_ingredient_matched"]
    delta = lambda i, k: L(by, i, f"pad{k}_aligned") - L(by, i, f"pad{k}_swapped")
    pad_parts = {}
    for role in ("goal", "ingredient"):
        sub = [r for r in padded if r["role"] == role]
        pad_parts[f"{role}_eight_minus_one"] = describe(*_vals(sub, lambda i: delta(i, 8) - delta(i, 1)))
        pad_parts[f"{role}_one_minus_none"] = describe(*_vals(sub, lambda i: delta(i, 1) - (L(by, i, "aligned") - L(by, i, "swapped"))))
    out["padding"] = pad_parts
    return out


def short_profile(r: dict) -> tuple[str, str, str]:
    """User point 2: (length bin, whether the missing inputs are craftable, raw or mixed, one or several missing),
    read from the prompt text (an input is craftable when a displayed line outputs it)."""
    from wf.textcraft.checker import displayed_outputs

    outs = displayed_outputs(r["recipe_lines"])
    kinds = {"craftable" if s["name"] in outs else "raw" for s in r["short_inputs"]}
    kind = kinds.pop() if len(kinds) == 1 else "mixed"
    return r["length_bin"], kind, "one" if len(r["short_inputs"]) == 1 else "several"


MATCH_MIN = 5  # items of each role a cell needs


def matched_gap(by: dict, recs: list[dict], value: Callable[[str], float], profile: Callable[[dict], tuple] = short_profile,
                B: int | None = None) -> dict[str, Any]:
    """Sensitivity (not primary): goal-role minus ingredient mean of `value` within cells of equal profile (length
    bin, craftable or raw missing inputs, number of missing inputs), averaged over cells holding at least MATCH_MIN
    items of each role, each cell weighted by its smaller count; as the round-1 matched comparison. Pair bootstrap."""
    rows = []
    for r in recs:
        x = value(r["id"])
        if x == x:
            rows.append((r["pair_id"], r["role"], profile(r), x))

    def stat(rs: list) -> float:
        cells: dict[tuple, dict[str, list]] = defaultdict(lambda: {"goal": [], "ingredient": []})
        for _p, role, prof, x in rs:
            cells[prof][role].append(x)
        num = den = 0.0
        for c in cells.values():
            w = min(len(c["goal"]), len(c["ingredient"]))
            if w >= MATCH_MIN:
                num += w * (mean(c["goal"]) - mean(c["ingredient"]))
                den += w
        return num / den if den else float("nan")

    d = draws(stat, rows, [p for p, *_ in rows], B=DESC_DRAWS if B is None else B)
    cells = defaultdict(Counter)
    for _p, role, prof, _x in rows:
        cells[prof][role] += 1
    used = [c for c in cells.values() if min(c["goal"], c["ingredient"]) >= MATCH_MIN]
    return {"estimate": stat(rows), "ci95": [pct(d, 0.025), pct(d, 0.975)], "n_cells_used": len(used),
            "n_items_in_used_cells": sum(c["goal"] + c["ingredient"] for c in used), "min_items_per_role": MATCH_MIN}


def _subset_diff(by: dict, recs: list[dict], a: str, b: str) -> dict[str, Any]:
    """Mean P in parent subset a minus subset b, pair bootstrap (the subsets play the goal and ingredient slots)."""
    rows, cl = [], []
    for r in recs:
        k = parent_subset(r)
        k = "overlapping" if (b == "overlapping" and k != "strict") else k
        x = L(by, r["id"], "ext_aligned") - L(by, r["id"], "ext_parent")
        if k in (a, b) and x == x:
            rows.append(("goal" if k == a else "ingredient", x))
            cl.append(r["pair_id"])
    return describe_gap(rows, cl)


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


def model_verdicts(block: dict[str, Any], fam: dict, tag: str) -> dict[str, Any]:
    v = {c: verdict(block["contrasts"][c], fam, tag, c) for c in CONTRASTS}
    comp = {c: block["competence"]["local"]["per_contrast"][c]["competent"] for c in CONTRASTS}
    timing = block["contrasts"]["timing"]
    return {"contrasts": v, "competent": comp,
            "interpretation": {c: interpretation(v[c], comp[c], block["contrasts"][c]) for c in CONTRASTS},
            "reading": reading(v, bool(timing.get("comparable_to_one_lookup_change")), comp),
            "rejections": rejections_for_model(v, comp, block),
            "central_question": central_question(v, comp, block["competence"]["local"]),
            "overall_gate_interpretable": block["competence"]["interpretable"]}


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
    out["verdicts"] = {t: model_verdicts(per_model[t], fam, t) for t in expected}
    out["across_models"] = across({t: out["verdicts"][t]["contrasts"] for t in expected},
                                  {t: out["verdicts"][t]["competent"]["parent_penalty"] for t in expected},
                                  {t: {"verdict": out["verdicts"][t]["contrasts"]["lift"],
                                       "estimate": per_model[t]["contrasts"]["lift"]["estimate"],
                                       "competent": out["verdicts"][t]["competent"]["lift"]} for t in expected})
    return out


def provenance(tags: list[str], root: Path | None = None) -> list[str]:
    """Review 1 (as families.py): every shard of every model scored at one git commit, and every score carries its
    model's pinned revision. Returns the problems."""
    root = root or runs_root()
    shas = {(root / score_run(t) / p.name / "git_sha").read_text(encoding="utf-8").strip()
            for t in tags for p in sorted((root / score_run(t)).glob("shard_*"))
            if (root / score_run(t) / p.name / "git_sha").exists()}
    problems = [f"shards were scored at {len(shas)} commits: {sorted(shas)}"] if len(shas) > 1 else []
    for t in tags:
        revs = {s.get("model_revision") for s in load_scores(t, root)}
        if revs and revs != {R.MODELS[t].revision}:
            problems.append(f"{t}: model revisions {sorted(map(str, revs))}, pinned {R.MODELS[t].revision}")
    return problems


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
    bad = provenance([t for t in expected if t in scores])
    if bad:
        write_status(run_id, f"provenance: {bad}", ok=False)
        return 1
    try:
        metrics = analyse(records, shots, scores, expected)
    except StaleScore as e:
        write_status(run_id, f"stale score: {e}", ok=False)
        return 1
    write_json(d / "metrics.json", metrics)
    if isinstance(metrics["verdicts"], str):
        write_status(run_id, f"{metrics['verdicts']}", ok=False)
        return 1
    lines = [f"{t}: overall gate {'pass' if v['overall_gate_interpretable'] else 'fail'}; "
             + "; ".join(f"{c} {x} ({'competent' if v['competent'][c] else 'not competent'})" for c, x in v["contrasts"].items())
             for t, v in metrics["verdicts"].items()]
    write_status(run_id, "\n".join(lines + [f"across: {metrics['across_models']['penalty_pattern']}"]), ok=True)
    return 0
