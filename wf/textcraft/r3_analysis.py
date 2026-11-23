"""TextCraft round 3 analysis: `textcraft_r3_analysis` (notes/textcraft_r3_design.md, sections 6 to 8).

Round 2's frozen analysis functions (wf/textcraft/r2_analysis.py) are called on round-3 scores, unchanged: the
competence gate and contrast-specific competence, the margin, the five contrasts with their tests, the verdicts,
interpretations, readings, R1, R3 and R5 and the central-question mapping. Added here:

- the fixed Holm families A3 (core, 30 tests), A3x (extension, 30) and B3 (pairs, 30);
- H3-P per model and overall; R2 restated for round 3; round 2's general finding per contrast, read mechanically from
  round 2's filed analysis, and the replication rule (partial replications labelled "partial", never counted);
- analysis (b): the standardized contrast of each model (mean over the SD of the item values; for the role-gap
  contrasts, over the pooled within-role SD), the paired goal-pair bootstrap of its difference within each pair, the
  equivalence test against B3_MARGIN, the readings; raw nats, item shares and the first-token / tail split beside it;
- the environment check (round-2 scores against serve2 rescores), the format checks (one-fifth rule) and the
  thinking arm's readouts, all outside every family.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from wf import constants as C
from wf import families as FAM
from wf.analysis.stats import holm
from wf.runs import _init_run
from wf.textcraft import prompts as P
from wf.textcraft import r2 as R
from wf.textcraft import r2_analysis as A
from wf.textcraft import r3 as R3
from wf.textcraft import r3_gates as G
from wf.textcraft.r2_score import load_records, verified_files
from wf.util import read_jsonl, runs_root, write_json, write_status

ALPHA = 0.05
B_DRAWS = A.B_DRAWS  # 20,000 for the family tests
DESC_DRAWS = A.DESC_DRAWS  # 2,000 for descriptive intervals
B3_MARGIN = 0.20  # equivalence margin on the standardized difference (a fifth of a standard deviation)
STRICT_AUC = 0.80  # descriptive stricter competence floor
MIN_COMPETENT = 2  # models competent for a contrast needed for a general finding (both rounds)
CONTRASTS = A.CONTRASTS
WITHIN = ("lift", "parent_penalty")
CONFIRMED = ("confirmed above zero", "confirmed below zero")
NOT_INTERPRETABLE_ENV = "not interpretable (environment check)"
def env_rule() -> dict[str, float]:
    """The environment check's pass rule: round 3's own constant (r3.ENVCHECK_RULE), wave B's original criteria,
    unamended (design 14.M4). Never read from wf/families.py, so a later change in wave B does not change round 3."""
    return dict(R3.ENVCHECK_RULE)


def per_score_check(fmt: str, diffs: list[float]) -> dict[str, Any]:
    """The per-score criterion: the mean absolute difference per score against round 3's constant (yes/no 0.07 nats,
    two-candidate 0.35)."""
    m = sum(abs(x) for x in diffs) / len(diffs) if diffs else float("nan")
    limit = env_rule()[f"{fmt}_mean_abs_max"]
    return {"ok": m <= limit, "mean_abs_diff": m, "limit": limit, "source": "r3.ENVCHECK_RULE"}
ONE_FIFTH = 0.2

# ---------------- loading ----------------


def run_folder(kind: str, tag: str) -> str:
    return f"textcraft_r3_{kind}_{tag}"


def load_run(run: str, root: Path | None = None) -> list[tuple[Path, list[dict]]]:
    """(shard folder, its scores) for every shard of a run."""
    root = root or runs_root()
    return [(p.parent, read_jsonl(p)) for p in sorted((root / run).glob("shard_*/scoring.jsonl"))]


def scores_of(run: str, tag: str, root: Path | None = None) -> list[dict]:
    return [s for _p, ss in load_run(run, root) for s in ss if s.get("model") == tag]


# ---------------- per-record contrast values (the same definitions as r2_analysis.contrast_inputs) ----------------


def record_values(by: dict, pops: dict[str, list[dict]], contrast: str) -> dict[str, tuple[str | None, float, str]]:
    """{record id: (role or None, value, goal pair)} of one contrast's rows, defined exactly as round 2's
    contrast_inputs (a test checks that the means agree). For Λ and P the value is the per-item difference; for Dn,
    Dc and T it is the item's own term and the role says which side of the gap it is on."""
    L = A.L
    delta = lambda i, k: (L(by, i, f"pad{k}_aligned") - L(by, i, f"pad{k}_swapped"))
    spec: dict[str, tuple[list[dict], Callable[[str], float], bool]] = {
        "lift": (pops["ingredient"], lambda i: L(by, i, "aligned") - L(by, i, "swapped"), False),
        "parent_penalty": (pops["parent_failures"], lambda i: L(by, i, "ext_aligned") - L(by, i, "ext_parent"), False),
        "naming": (A.two_item_rows(pops), lambda i: L(by, i, "both") - L(by, i, "swapped"), True),
        "completion": (A.two_item_rows(pops), lambda i: L(by, i, "aligned") - L(by, i, "both"), True),
        "timing": (pops["padded_goal"] + pops["padded_ingredient_matched"], lambda i: delta(i, 8) - delta(i, 1), True),
    }
    recs, f, gap = spec[contrast]
    out = {}
    for r in recs:
        x = f(r["id"])
        if x == x:
            out[r["id"]] = (r["role"] if gap else None, x, r["pair_id"])
    return out


def _stats(sums: tuple[float, ...], gap: bool) -> tuple[float, float]:
    """(raw contrast, standardized contrast) from summed sufficient statistics (n, sum, sum of squares per side)."""
    if gap:
        ng, sg, qg, ni, si, qi = sums
        if ng < 2 or ni < 2:
            return float("nan"), float("nan")
        mg, mi = sg / ng, si / ni
        vg, vi = (qg - ng * mg * mg) / (ng - 1), (qi - ni * mi * mi) / (ni - 1)
        pooled = math.sqrt(max(((ng - 1) * vg + (ni - 1) * vi) / (ng + ni - 2), 0.0))
        raw = mg - mi
    else:
        n, s, q = sums[:3]
        if n < 2:
            return float("nan"), float("nan")
        m = s / n
        pooled = math.sqrt(max((q - n * m * m) / (n - 1), 0.0))
        raw = m
    return raw, (raw / pooled if pooled > 0 else float("nan"))


def _cluster_sums(values: dict[str, tuple[str | None, float, str]], keep: set[str] | None, gap: bool) -> dict[str, list[float]]:
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0] * 6)
    for rid, (role, x, pair) in values.items():
        if keep is not None and rid not in keep:
            continue
        a = agg[pair]
        off = 3 if gap and role == "ingredient" else 0
        a[off] += 1
        a[off + 1] += x
        a[off + 2] += x * x
    return dict(agg)


def _total(agg: dict[str, list[float]], keys) -> tuple[float, ...]:
    t = [0.0] * 6
    for k in keys:
        a = agg.get(k)
        if a is None:
            continue
        for j in range(6):
            t[j] += a[j]
    return tuple(t)


def standardized(values: dict[str, tuple[str | None, float, str]], gap: bool, B: int | None = None,
                 seed: int = C.RNG_SEED) -> dict[str, Any]:
    """One model's standardized contrast with a goal-pair bootstrap interval (descriptive)."""
    B = DESC_DRAWS if B is None else B
    agg = _cluster_sums(values, None, gap)
    keys = sorted(agg)
    raw, z = _stats(_total(agg, keys), gap)
    rng = random.Random(seed)
    d = []
    for _ in range(B):
        v = _stats(_total(agg, rng.choices(keys, k=len(keys))), gap)[1]
        if v == v:
            d.append(v)
    return {"estimate": z, "ci95": [A.pct(d, 0.025), A.pct(d, 0.975)], "raw_nats": raw, "n_pairs": len(keys),
            "n_items": len(values), "scale": "mean over the SD of item values" if not gap else
            "goal-minus-ingredient mean over the pooled within-role SD"}


def paired_difference(old: dict, new: dict, gap: bool, B: int | None = None, seed: int = C.RNG_SEED,
                      margin: float = B3_MARGIN, fixed_sd: tuple[float, float] | None = None) -> dict[str, Any]:
    """The newer model's standardized contrast minus the older model's on the records both scored, with one goal-pair
    resample serving both models (design 7.4): estimate, 95% interval, two-sided bootstrap p, TOST p against
    `margin`, the equivalence power at a true zero (Holm's first step in a family of 30, and 0.05; 14.S3); the raw
    difference in nats with its interval, and, with `fixed_sd` (each model's SD of ΔL over the main failures), the
    fixed-denominator difference (both descriptive)."""
    B = B_DRAWS if B is None else B
    common = set(old) & set(new)
    ao, an = _cluster_sums(old, common, gap), _cluster_sums(new, common, gap)
    keys = sorted(set(ao) | set(an))
    ro, zo = _stats(_total(ao, keys), gap)
    rn, zn = _stats(_total(an, keys), gap)
    rng = random.Random(seed)
    dz, dr, df = [], [], []
    so, sn = fixed_sd if fixed_sd else (float("nan"), float("nan"))
    for _ in range(B):
        ks = rng.choices(keys, k=len(keys))
        r1, z1 = _stats(_total(ao, ks), gap)
        r2, z2 = _stats(_total(an, ks), gap)
        if z1 == z1 and z2 == z2:
            dz.append(z2 - z1)
        if r1 == r1 and r2 == r2:
            dr.append(r2 - r1)
            if fixed_sd and so > 0 and sn > 0:
                df.append(r2 / sn - r1 / so)
    se = statistics.stdev(dz) if len(dz) > 1 else float("nan")
    first = A.equivalence_power(se, margin, R3.Z_FIRST_30)
    out = {"standardized": {"old": zo, "new": zn, "estimate": zn - zo, "ci95": [A.pct(dz, 0.025), A.pct(dz, 0.975)],
                            "p_differs": A.bootstrap_p(dz), "p_equivalent": A.tost_p(dz, margin), "margin": margin,
                            "bootstrap_se": se, "n_draws": B, "equivalence_power_at_zero_holm_first_step": first,
                            "equivalence_power_at_zero_alpha_05": A.equivalence_power(se, margin, R3.Z_ALPHA),
                            "equivalence_underpowered": not (first >= A.POWER_MIN)},
           "raw_nats_descriptive": {"old": ro, "new": rn, "estimate": rn - ro,
                                    "ci95": [A.pct(dr, 0.025), A.pct(dr, 0.975)]},
           "n_items": len(common), "n_pairs": len(keys), "resampling": "paired, whole goal pairs"}
    if fixed_sd:
        out["fixed_denominator_descriptive"] = {
            "old": ro / so if so > 0 else float("nan"), "new": rn / sn if sn > 0 else float("nan"),
            "estimate": (rn / sn - ro / so) if so > 0 and sn > 0 else float("nan"),
            "ci95": [A.pct(df, 0.025), A.pct(df, 0.975)],
            "denominator": "each model's SD of ΔL over the 1,500 main failures (the SD that sets its margin)"}
    return out


def item_share(values: dict[str, tuple[str | None, float, str]]) -> float:
    v = [x for _r, x, _p in values.values()]
    return sum(x > 0 for x in v) / len(v) if v else float("nan")


def share_difference(old: dict, new: dict, B: int | None = None, seed: int = C.RNG_SEED) -> dict[str, Any]:
    """For Λ and P: each model's share of items with a positive per-item difference, and the paired difference."""
    B = DESC_DRAWS if B is None else B
    common = sorted(set(old) & set(new))
    by_pair: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rid in common:
        by_pair[old[rid][2]].append((int(old[rid][1] > 0), int(new[rid][1] > 0)))
    keys = sorted(by_pair)
    rng = random.Random(seed)

    def diff(ks) -> float:
        rows = [x for k in ks for x in by_pair[k]]
        return (sum(b for _a, b in rows) - sum(a for a, _b in rows)) / len(rows) if rows else float("nan")

    d = [diff(rng.choices(keys, k=len(keys))) for _ in range(B)]
    return {"old": item_share({k: old[k] for k in common}), "new": item_share({k: new[k] for k in common}),
            "difference": diff(keys), "ci95": [A.pct(d, 0.025), A.pct(d, 0.975)]}


def first_tail(by: dict, pops: dict[str, list[dict]]) -> dict[str, Any]:
    """Each contrast computed on the first-token log-odds and on the tail (L minus the first-token log-odds), with
    descriptive goal-pair intervals; two-candidate arms only (the yes/no arms are single tokens)."""
    split = {}
    for part in ("first_token", "tail"):
        b2 = {}
        for k, s in by.items():
            if R.parse_arm(k[1])[0]:
                continue
            first = s["logp_first_plus"] - s["logp_first_minus"]
            b2[k] = {**s, "L": first if part == "first_token" else s["L"] - first}
        inputs = A.contrast_inputs(b2, pops)
        split[part] = {c: (A.describe(rows, cl) if kind == "within" else A.describe_gap(rows, cl))
                       for c, (kind, rows, cl) in inputs.items() if c in CONTRASTS}
    return split


def r3_extras(by: dict, pops: dict[str, list[dict]], block: dict[str, Any]) -> dict[str, Any]:
    """Round 3's per-model additions: the standardized contrasts, item shares, the first-token / tail split and the
    competence at the stricter floor (all descriptive)."""
    vals = {c: record_values(by, pops, c) for c in CONTRASTS}
    pc = block["competence"]["local"]["per_contrast"]
    pop_auc = {n: p["yes_no_auc"] for n, p in block["competence"]["local"]["populations"].items()}
    strict = {c: bool(all(pop_auc[n] >= STRICT_AUC for n in A.CONTRAST_POPULATIONS[c])
                      and block["competence"]["local"]["other_criteria_on_full_population_pass"]) for c in CONTRASTS}
    return {"standardized": {c: standardized(vals[c], c not in WITHIN) for c in CONTRASTS},
            "item_share": {c: item_share(vals[c]) for c in WITHIN},
            "first_token_tail": first_tail(by, pops),
            "competent_at_strict_floor": strict, "strict_floor": STRICT_AUC,
            "competent": {c: pc[c]["competent"] for c in CONTRASTS}}


# ---------------- analysis (a): H3-P, R2, round 2's findings, replication ----------------


def r2_contemporary(verdicts: dict[str, dict], competent: dict[str, bool]) -> dict[str, Any]:
    """R2 restated for round 3 (design 7.3): among the core models competent for P, any confirmed below zero: does
    not fire; else any equivalent or confirmed above: fires; else undecided."""
    passing = [t for t, ok in competent.items() if ok]
    below = [t for t in passing if verdicts[t]["parent_penalty"] == "confirmed below zero"]
    against = [t for t in passing if verdicts[t]["parent_penalty"] in ("equivalent to zero", "confirmed above zero")]
    value = False if below else True if against else "undecided"
    return {"needs": "competent for P (each core model counted)", "models_competent": passing, "below": below,
            "equivalent_or_above": against, "fires": value is True, "value": value,
            "rejects": "the claim that the penalty is shared among these contemporary models"}


def h3p_reading(verdicts: dict[str, dict], competent: dict[str, bool], r1: dict[str, bool], r2: dict[str, Any]) -> dict[str, Any]:
    """H3-P per core model and overall (design 7.3)."""
    per = {}
    for t, v in verdicts.items():
        p = v["parent_penalty"]
        if p == NOT_SCORED:  # review 2, S3
            per[t] = NOT_SCORED
        elif competent[t] and p == "confirmed below zero":
            per[t] = "supported"
        elif r1[t]:
            per[t] = "rejected"
        elif p == "confirmed below zero":
            per[t] = "measurable only (not competent for P)"
        else:
            per[t] = "inconclusive"
    comp = [t for t in verdicts if competent[t]]
    sup = [t for t in comp if per[t] == "supported"]
    rej = [t for t in comp if per[t] == "rejected"]
    if r2.get("fires") or (rej and not sup):
        overall = "rejected"
    elif len(comp) < MIN_COMPETENT:  # review 1, S1: one competent model, even supported, is too few
        overall = "inconclusive"
    elif len(sup) == len(comp):
        overall = "supported"
    elif sup:
        overall = "partial"
    else:
        overall = "inconclusive"
    reason = None
    if overall == "inconclusive":
        reason = (f"fewer than {MIN_COMPETENT} core models competent for P" if len(comp) < MIN_COMPETENT else
                  "no competent core model has P confirmed below zero or rejected")
    return {"per_model": per, "overall": overall, "reason": reason, "competent": comp,
            "note": "'partial' is reported per model and never counted as support"}


def round2_finding(verdicts: dict[str, str], competent: dict[str, bool]) -> dict[str, Any]:
    """Round 2's general finding for one contrast from its filed verdicts and per-contrast competence (design 7.3)."""
    comp = [t for t, ok in competent.items() if ok]
    vs = [verdicts[t] for t in comp]
    if len(comp) >= MIN_COMPETENT and all(v == "confirmed below zero" for v in vs):
        kind, sign = "general", -1
    elif len(comp) >= MIN_COMPETENT and all(v == "confirmed above zero" for v in vs):
        kind, sign = "general", 1
    elif len(comp) >= MIN_COMPETENT and all(v == "equivalent to zero" for v in vs):
        kind, sign = "general null", 0
    elif any(v in CONFIRMED for v in vs) and ("equivalent to zero" in vs or len({v for v in vs if v in CONFIRMED}) > 1):
        kind, sign = "model-specific", None
    else:
        kind, sign = "not established", None
    return {"kind": kind, "sign": sign, "models_competent": comp, "verdicts": {t: verdicts[t] for t in comp}}


def replication(finding: dict[str, Any], verdicts: dict[str, str], competent: dict[str, bool]) -> dict[str, Any]:
    """Round 3's reading of one contrast against round 2's finding (design 7.3), among the round-3 models competent
    for it; models not competent are listed and not counted."""
    comp = [t for t, ok in competent.items() if ok]
    per = {t: verdicts[t] for t in comp}
    out = {"round2_finding": finding["kind"], "models_competent": comp, "per_model": per,
           "not_competent": {t: verdicts[t] for t, ok in competent.items() if not ok}}
    if finding["kind"] not in ("general", "general null"):
        return {**out, "reading": "no replication target"}
    if len(comp) < MIN_COMPETENT:
        return {**out, "reading": "inconclusive", "reason": f"fewer than {MIN_COMPETENT} competent round-3 models"}
    if finding["kind"] == "general":
        target = "confirmed below zero" if finding["sign"] < 0 else "confirmed above zero"
        opposite = "confirmed above zero" if finding["sign"] < 0 else "confirmed below zero"
        hit = [t for t in comp if per[t] == target]
        against = [t for t in comp if per[t] in ("equivalent to zero", opposite)]
        if len(hit) == len(comp):
            reading = "replicated"
        elif hit:
            reading = "partial"
        elif against:
            reading = "not replicated"
        else:
            reading = "inconclusive"
    else:
        eq = [t for t in comp if per[t] == "equivalent to zero"]
        conf = [t for t in comp if per[t] in CONFIRMED]
        if conf:
            reading = "not replicated"
        elif len(eq) == len(comp):
            reading = "replicated"
        elif eq:
            reading = "partial"
        else:
            reading = "inconclusive"
    return {**out, "reading": reading, "counted_as_replication": reading == "replicated"}


def read_round2(path: Path) -> dict[str, Any] | None:
    """Round 2's filed analysis: per model, the five verdicts and the per-contrast competence (None if not filed or
    its verdicts are withheld)."""
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(m.get("verdicts"), dict):
        return None
    return {t: {"contrasts": v["contrasts"], "competent": v["competent"]} for t, v in m["verdicts"].items()}


def replication_block(r2: dict[str, Any] | None, verdicts: dict[str, dict], competent: dict[str, dict]) -> dict[str, Any]:
    if r2 is None:
        return {c: {"reading": "waiting (round 2's analysis not filed)"} for c in CONTRASTS}
    out = {}
    for c in CONTRASTS:
        f = round2_finding({t: v["contrasts"][c] for t, v in r2.items()}, {t: v["competent"][c] for t, v in r2.items()})
        out[c] = replication(f, {t: verdicts[t][c] for t in verdicts}, {t: competent[t][c] for t in competent})
        out[c]["round2"] = f
    return out


# ---------------- analysis (b): pairs ----------------


def b3_family(diffs: dict[str, dict[str, dict]], expected: list[tuple[str, str]]) -> dict[str, dict]:
    """Family B3: the differs and equivalence tests of every pair and contrast on the standardized difference, Holm
    together; a missing (or not scored) pair or NaN p enters as 1."""
    p, info = {}, {}
    for old, new in expected:
        for c in CONTRASTS:
            rec = diffs.get(f"{old}->{new}", {}).get(c)
            for t in A.TESTS:
                key = f"{old}->{new}:{c}:{t}"
                pv = rec["standardized"][f"p_{t}"] if rec else float("nan")
                p[key] = pv if pv == pv else 1.0
                info[key] = {"raw_p": pv, "padded": pv != pv}
    adj = holm(p)
    return {k: {**info[k], "p_in_family": p[k], "p_holm": adj[k]} for k in p}


def direction_words(contrast: str, old: float, new: float) -> str:
    """The plain-words direction of a pair's difference (review 1, S4): "newer higher" on a negative P means a smaller
    penalty."""
    if not (old == old and new == new):
        return ""
    if old < 0 and new < 0:
        size = "smaller in size" if new > old else "larger in size"
        return f"{size}{' (a smaller penalty)' if contrast == 'parent_penalty' and new > old else ' (a larger penalty)' if contrast == 'parent_penalty' else ''}"
    if old > 0 and new > 0:
        return "larger in size" if new > old else "smaller in size"
    return "opposite signs" if (old < 0) != (new < 0) else ""


def pair_reading(rec: dict | None, fam: dict, key: str) -> str:
    """A pair's reading for one contrast (design 7.4 and 14.S3-S4): differs (newer higher / lower, with the direction
    in plain words), similar within the margin, or not distinguishable (flagged "equivalence underpowered" when its
    power at a true zero at Holm's first step is below 0.8)."""
    if rec is None:
        return "waiting"
    s = rec["standardized"]
    lo, hi = s["ci95"]
    words = direction_words(key.split(":")[-1], s.get("old", float("nan")), s.get("new", float("nan")))
    if fam[f"{key}:differs"]["p_holm"] < ALPHA:
        if s["estimate"] > 0 and lo > 0:
            return "differs, newer higher" + (f" ({words})" if words else "")
        if s["estimate"] < 0 and hi < 0:
            return "differs, newer lower" + (f" ({words})" if words else "")
    if fam[f"{key}:equivalent"]["p_holm"] < ALPHA:
        return "similar within the margin"
    return "not distinguishable" + (" (equivalence underpowered)" if s.get("equivalence_underpowered") else "")


PAIR_CAVEAT = ("Each pair changes developer data, post-training, size, tokenizer, template, architecture and software at "
               "once: a 'differs' describes two released models, not a generation effect; three purposive pairs say "
               "nothing about model generations in general.")
PAIR_CHANGES = {
    "qwen3-8b->qwen3.5-9b": "data and post-training; 8.2 B to 9.4 B parameters (Qwen3.5-9B: 9.41 B in the architecture, "
                            "8.95 B in the language model; 9.65 B tensors stored, with multi-token-prediction heads); "
                            "tokenizer; the same empty think block before the reply; Qwen3.5's linear-attention layers; "
                            "transformers 4.51 to 5.23",
    "mistral-7b-v0.3->ministral3-8b-it": "data and post-training; 7.3 B to 8.9 B; tokenizer (Ministral 3 splits "
                                         "'Crafted' into 'C' + 'rafted'); system text in the last user turn and a "
                                         "space before the reply (Mistral 7B) against system text first and no "
                                         "space (Ministral 3); transformers 4.51 to 5.23",
    "gemma2-9b-it->gemma4-12b-it": "data and post-training; 9.2 B to 12.0 B; system text folded into the first user "
                                   "turn (Gemma 2) against a system turn and an empty thought channel (Gemma 4); "
                                   "sliding windows of 4,096 against 1,024 tokens (exceeded by 99% of inputs); eager "
                                   "against SDPA attention; transformers 4.51 to 5.23",
    "gemma2-9b-it->gemma4-e4b-it": "as Gemma 2 to Gemma 4 12B, and the fallback has about 4 B effective parameters "
                                   "against Gemma 2's 9 B (size confound); sliding window 512",
}


# ---------------- the environment check (design 14.M4) ----------------


def envcheck_block(old: list[tuple[Path, list[dict]]], new: list[tuple[Path, list[dict]]], tag: str,
                   records: list[dict], gpu_of: Callable[[Path], list[str] | None],
                   unobtainable: bool = False) -> dict[str, Any]:
    """Round 2's scores of `tag` against its serve2 rescores, record by record: a record is comparable when its
    round-2 shard and the environment-check shard that rescored it each ran on one GPU list throughout (typed) and the
    two lists are identical. Wave B's pass rule on the union of comparable records, with at least 200 comparable.
    Fewer than 200 comparable: "not run (GPU types)" when the plan could not reach 200 or a planned GPU type was
    filed as unobtainable, else "waiting"."""
    from wf.textcraft.r3_score import envcheck_subset

    old_gpu, new_gpu, oldmap, newmap = {}, {}, {}, {}
    for p, ss in old:
        g = gpu_of(p)
        for s in ss:
            if s.get("model") == tag:
                oldmap[(s["transition_id"], s["condition"])] = s["L"]
                old_gpu[s["transition_id"]] = g
    for p, ss in new:
        g = gpu_of(p)
        for s in ss:
            if s.get("model") == tag + R3.ENVCHECK_SUFFIX:
                newmap[(s["transition_id"], s["condition"])] = s["L"]
                new_gpu[s["transition_id"]] = g
    seeded = envcheck_subset(records)
    chosen = [r for r in seeded if old_gpu.get(r["id"]) is not None and new_gpu.get(r["id"]) is not None
              and old_gpu[r["id"]] == new_gpu[r["id"]]]
    expected = {(r["id"], a) for r in chosen for a in r["arms"]}
    pairs = {k: (oldmap[k], newmap[k]) for k in expected if k in oldmap and k in newmap}
    rule = env_rule()
    out: dict[str, Any] = {"n_comparable_records": len(chosen), "n_expected": len(expected), "n_paired": len(pairs),
                           "gpus": sorted({str(old_gpu[r["id"]]) for r in chosen}), "rule": rule,
                           "min_comparable": R3.ENVCHECK_MIN_COMPARABLE}
    if len(chosen) < R3.ENVCHECK_MIN_COMPARABLE:
        out["status"] = "not run (GPU types)" if unobtainable else "waiting"
        out["pass"] = None
        return out
    checks: dict[str, bool] = {"complete": expected <= set(pairs)}
    mf = [r for r in chosen if r["kind"] == "main" and r["population"] == "main"]
    for fmt, keep, (al, sw) in (("yes_no", lambda a: a.startswith("yn_"), ("yn_aligned", "yn_swapped")),
                                ("two_candidate", lambda a: not a.startswith("yn_"), ("aligned", "swapped"))):
        d = [b - a for (i, c), (a, b) in pairs.items() if keep(c)]
        effects = {}
        for which, idx in (("old", 0), ("new", 1)):
            g = [pairs[(r["id"], al)][idx] - pairs[(r["id"], sw)][idx] for r in mf
                 if (r["id"], al) in pairs and (r["id"], sw) in pairs]
            effects[which] = sum(g) / len(g) if g else float("nan")
            effects[f"share_{which}"] = sum(x > 0 for x in g) / len(g) if g else float("nan")
        rec = {"n": len(d), "mean_abs_diff": sum(abs(x) for x in d) / len(d) if d else float("nan"),
               "max_abs_diff": max((abs(x) for x in d), default=float("nan")),
               "goal_effect_old": effects["old"], "goal_effect_new": effects["new"],
               "goal_effect_diff": effects["new"] - effects["old"], "n_main_failures": len(mf)}
        rec["per_score"] = per_score_check(fmt, d)
        checks[f"{fmt}_per_score"] = rec["per_score"]["ok"]
        checks[f"{fmt}_goal_effect"] = abs(rec["goal_effect_diff"]) <= rule[f"{fmt}_goal_effect_diff_max"]
        if fmt == "yes_no":
            rec["share_diff"] = effects["share_new"] - effects["share_old"]
            checks["yes_no_share"] = abs(rec["share_diff"]) <= rule["yes_no_share_diff_max"]
        out[fmt] = rec
    out["checks"] = checks
    out["pass"] = all(checks.values())  # NaN fails every comparison
    out["status"] = "pass" if out["pass"] else "fail"
    return out


def golden_tokens_check(r3_golden: dict, r2_golden: dict, tag: str) -> dict[str, Any]:
    """Part 1 of the environment check: the earlier model's golden renderings in serve2 (`<tag>-serve2` in round 3's
    golden file) have the same ids as round 2's (transformers 4.51.3)."""
    new = r3_golden.get("models", {}).get(tag + R3.ENVCHECK_SUFFIX)
    old = r2_golden.get("models", {}).get(tag)
    if not new or not old:
        return {"ok": False, "problem": "golden renderings missing"}
    keys = [k for k in old if k != "revision"]
    diff = [k for k in keys if (old[k]["n_ids"], old[k]["ids_sha256"]) != (new.get(k, {}).get("n_ids"), new.get(k, {}).get("ids_sha256"))]
    return {"ok": not diff and bool(keys), "n_cases": len(keys), "different": diff}


def environment_status(models: dict[str, dict], tokens: dict[str, dict]) -> str:
    """The environment check over its three models (design 14.M4): "pass" when every model passed and its golden
    tokens agree; "fail" when any model failed or its tokens differ; "not run (GPU types)" when none failed and any
    could not be run; "waiting" otherwise."""
    states = [models.get(t, {}).get("status", "waiting") for t in R3.ENVCHECK_MODELS]
    tok_ok = [bool(tokens.get(t, {}).get("ok")) for t in R3.ENVCHECK_MODELS]
    if any(s == "fail" for s in states) or not all(tok_ok):
        return "fail"
    if all(s == "pass" for s in states):
        return "pass"
    if any(s == "not run (GPU types)" for s in states) and not any(s == "waiting" for s in states):
        return "not run (GPU types)"
    return "waiting"


def b3_source(env_status: str, cfg: dict) -> str:
    """Which older scores B3 reads (review 1, S5): round 2's if the environment check passed; the declared serve2
    rescoring if it failed or could not be run; none while it waits. A config value that disagrees is refused."""
    want = {"pass": "round2", "fail": "rescore", "not run (GPU types)": "rescore"}.get(env_status, "waiting")
    given = cfg.get("b3_old_scores")
    if given not in (None, want):
        raise ValueError(f"b3_old_scores is {given!r}, but the environment check ({env_status}) gives {want!r}")
    return want


# ---------------- format checks and the thinking arm (exploratory) ----------------


def format_block(native: dict, variant: dict, records: list[dict]) -> dict[str, Any]:
    """A format check against its model's native scores on the same records (design 8.3), both formats: the paired
    difference of the goal effect (variant minus native), its interval, and wave A's one-fifth rule."""
    from wf.textcraft.r3_score import format_subset

    recs = format_subset(records)
    out = {}
    for fmt, (al, sw) in (("two_candidate", ("aligned", "swapped")), ("yes_no", ("yn_aligned", "yn_swapped"))):
        rows, cl, nat = [], [], []
        for r in recs:
            keys = [(r["id"], al), (r["id"], sw)]
            if all(k in native and k in variant for k in keys):
                n_eff = native[keys[0]]["L"] - native[keys[1]]["L"]
                v_eff = variant[keys[0]]["L"] - variant[keys[1]]["L"]
                rows.append(v_eff - n_eff)
                nat.append(n_eff)
                cl.append(r["pair_id"])
        diff = A.describe(rows, cl)
        native_eff = sum(nat) / len(nat) if nat else float("nan")
        lo, hi = diff["ci95"]
        excludes = bool(lo == lo and hi == hi and (lo > 0 or hi < 0))
        large = bool(diff["estimate"] == diff["estimate"] and native_eff == native_eff
                     and abs(diff["estimate"]) > ONE_FIFTH * abs(native_eff))
        out[fmt] = {"variant_minus_native": diff, "native_effect": native_eff, "ci_excludes_zero": excludes,
                    "above_one_fifth": large, "format_dependent": excludes and large}
    return out


def think_block(think: list[dict], off_by: dict, records: list[dict]) -> dict[str, Any]:
    """The thinking arm's readouts (design 8.2), all descriptive and without p-values (review 1, S11): P and Λ after
    reasoning, their standardized values, their paired differences from thinking off on the same records, the yes/no
    AUC after reasoning, the shares closed at the cap and ended unclosed, the reasoning length and the mention rates;
    on all generations and on the unforced ones (closed by the model itself)."""
    pops = A.populations(records)
    out: dict[str, Any] = {}
    for subset in ("all", "unforced"):
        by = {(s["transition_id"], s["condition"]): s for s in think if subset == "all" or not s.get("forced")}
        block: dict[str, Any] = {}
        for c, recs, (a, b) in (("parent_penalty", pops["parent_failures"], ("ext_aligned", "ext_parent")),
                                ("lift", pops["ingredient"], ("aligned", "swapped"))):
            on = {r["id"]: (None, A.L(by, r["id"], a) - A.L(by, r["id"], b), r["pair_id"]) for r in recs
                  if (r["id"], a) in by and (r["id"], b) in by}
            off = {r["id"]: (None, A.L(off_by, r["id"], a) - A.L(off_by, r["id"], b), r["pair_id"]) for r in recs
                   if (r["id"], a) in off_by and (r["id"], b) in off_by}
            block[c] = {"thinking_on": A.describe([x for _r, x, _p in on.values()], [p for _r, _x, p in on.values()]),
                        "standardized_on": standardized(on, False),
                        "on_minus_off": descriptive_only(paired_difference(off, on, False, B=DESC_DRAWS))}
        succ = [by[(r["id"], "yn_neutral")]["L"] for r in pops["successes"] if (r["id"], "yn_neutral") in by]
        fail = [by[(r["id"], "yn_neutral")]["L"] for r in pops["main_failures"] if (r["id"], "yn_neutral") in by]
        block["yes_no_auc_after_reasoning"] = {"auc": A.auc(succ, fail), "n_successes": len(succ), "n_failures": len(fail)}
        block["n_scores"] = len(by)
        out[subset] = block
    n = len(think)
    share = lambda f: sum(bool(f(s)) for s in think) / n if n else float("nan")  # noqa: E731
    out["forced_share"] = share(lambda s: s.get("forced"))
    out["cap_share"] = share(lambda s: s.get("forced_reason") == "cap")
    out["ended_unclosed_share"] = share(lambda s: s.get("forced_reason") == "ended unclosed")
    out["median_new_tokens"] = statistics.median([s["n_new_tokens"] for s in think]) if n else float("nan")
    out["mentions_goal_word"] = share(lambda s: s.get("mentions_goal_word"))
    out["mentions_crafted_item"] = share(lambda s: s.get("mentions_crafted_item"))
    out["boundary_problems"] = sum(not s.get("boundary_ok", True) for s in think)
    out["kind"] = "exploratory; outside every family; never gated or pooled with the instruction runs"
    return out


def descriptive_only(rec: dict) -> dict:
    """An exploratory comparison without p-values (review 1, S11): estimates and intervals only."""
    out = json.loads(json.dumps(rec))
    for k in ("p_differs", "p_equivalent", "margin", "equivalence_power_at_zero_holm_first_step",
              "equivalence_power_at_zero_alpha_05", "equivalence_underpowered"):
        out.get("standardized", {}).pop(k, None)
    out["inference"] = "descriptive (exploratory): no test"
    return out


# ---------------- provenance (M1, S12) ----------------


def provenance_problems(folders: dict[str, list[Path]], scores: dict[str, list[dict]], tag_commit: str | None,
                        mock_run: bool = False, run_git: Callable[[list[str]], str] | None = None,
                        gpus_of: Callable[[Path], list[list[str] | None]] | None = None,
                        commits_of: Callable[[Path], list[str]] | None = None,
                        dirty_of: Callable[[Path], bool] | None = None) -> dict[str, list[str]]:
    """{tag: problems} for every round-3 run read: every job of every shard at the freeze tag's commit (or a
    code-identical one; S12); every job of every shard of a model on the identical GPU list, of its pinned type (each
    job's own list from jobs.jsonl; M1); every score with its model's pinned revision, and no other. A mock run checks
    the revisions only."""
    gpus_of = gpus_of or G.job_gpus
    commits_of = commits_of or G.shard_commits
    dirty_of = dirty_of or G.shard_dirty
    out: dict[str, list[str]] = defaultdict(list)
    if not mock_run:
        for tag, ps in folders.items():  # review 2, S2: a shard scored from a checkout with uncommitted changes
            bad = [p.name for p in ps if dirty_of(p)]
            if bad:
                out[tag].append(f"{tag}: shards {bad} ran with uncommitted changes (git_dirty)")
        cp = G.commit_problems({t: [c for p in ps for c in commits_of(p)] for t, ps in folders.items()}, tag_commit,
                               **({"run_git": run_git} if run_git else {}))
        for t, probs in cp.items():
            out[t] += probs
        for tag, ps in folders.items():
            lists = [g for p in ps for g in gpus_of(p)]
            base = R3.base_tag(tag)
            if not lists or any(g is None for g in lists) or any(g != lists[0] for g in lists):
                out[tag].append(f"{tag}: jobs ran on different or unrecorded GPUs {sorted(map(str, lists))}")
            elif base in R3.GPU_PIN and not R3.gpu_type_ok(base, lists[0]):
                out[tag].append(f"{tag}: GPU {lists[0]} is not the pinned {R3.GPU_PIN[base]}")
    for tag, ss in scores.items():
        revs = {s.get("model_revision") for s in ss}
        want = R3.spec_for(tag).revision
        if revs and revs != {want}:
            out[tag].append(f"{tag}: model revisions {sorted(map(str, revs))}, pinned {want}")
    return dict(out)


def other_commit_problems(other: dict[str, list[Path]], tag_commit: str | None, dev: dict[str, Any],
                          commits_of: Callable[[Path], list[str]] | None = None,
                          run_git: Callable[[list[str]], str] | None = None) -> dict[str, list[str]]:
    """S12 for the runs outside the families (environment check, rescoring, thinking arm): every job at the freeze tag's
    commit; the thinking arm's (key "think") also at D-T1's deviation commit `dev["commit"]`, and only those."""
    commits_of = commits_of or G.shard_commits
    kw = {"run_git": run_git} if run_git else {}
    cs = {k: [c for p in ps for c in commits_of(p)] for k, ps in other.items()}
    cp = G.commit_problems({k: v for k, v in cs.items() if k != "think"}, tag_commit, **kw)
    if "think" in cs:  # D-T1
        cp.update(G.commit_problems({"think": cs["think"]}, tag_commit, also=dev.get("commit"), **kw))
        if dev.get("problems"):
            cp.setdefault("think", []).extend(dev["problems"])
    return cp


def think_shard_commits(folders: list[Path], commits_of: Callable[[Path], list[str]] | None = None) -> dict[str, list[str]]:
    """D-T1: the commits each thinking-arm shard's jobs ran at, recorded in the metrics."""
    commits_of = commits_of or G.shard_commits
    return {f"{p.parent.name}/{p.name}": sorted(set(commits_of(p))) for p in folders}


# ---------------- the analysis ----------------


def holm30_power(block: dict[str, Any]) -> None:
    """Review 1, S2: round 2's power fields are computed at Holm's first step in its family of 50; round 3's families
    have 30 tests. The equivalence power and the underpowered flag are recomputed at 0.05/30 (round 2's values are
    kept under *_round2_family)."""
    for c in CONTRASTS:
        rec = block["contrasts"][c]
        se, margin = rec.get("bootstrap_se", float("nan")), rec.get("margin", float("nan"))
        rec["equivalence_power_at_zero_holm_first_step_round2_family"] = rec.get("equivalence_power_at_zero_holm_first_step")
        rec["equivalence_underpowered_round2_family"] = rec.get("equivalence_underpowered")
        first = A.equivalence_power(se, margin, R3.Z_FIRST_30)
        rec["equivalence_power_at_zero_holm_first_step"] = first
        rec["equivalence_underpowered"] = not (first >= A.POWER_MIN)
        rec["holm_first_step_z"] = R3.Z_FIRST_30


def model_analysis(records: list[dict], shots, scores: dict[str, list[dict]], tags: list[str]) -> dict[str, Any]:
    """Round 2's per-model block for each tag, with round 3's additions; missing scores withhold that model."""
    pops = A.populations(records)
    out, bys, missing = {}, {}, {}
    for tag in tags:
        if tag not in scores:
            continue
        by, miss = A.index_scores(scores[tag], records, shots, tag)
        missing[tag] = len(miss)
        if miss:
            continue
        bys[tag] = by
        block = A.model_block(by, pops)
        holm30_power(block)
        block["round3"] = r3_extras(by, pops, block)
        out[tag] = block
    return {"blocks": out, "bys": bys, "missing": missing, "pops": pops}


NOT_SCORED = "not scored"


def family_a(name: str, tags: list[str], ma: dict[str, Any], r2: dict[str, Any] | None,
             states: dict[str, dict] | None = None) -> dict[str, Any]:
    """A3 or A3x: round 2's family function over these tags (30 tests), verdicts, H3-P, R2 and replication. A model
    "not scored" (its filed gate record failed; design 14.M2) enters with p = 1, verdict "not scored" and competence
    false for every contrast, and the readings proceed; "waiting" only for a model whose run is not filed yet."""
    states = states or {}
    blocks = {t: b for t, b in ma["blocks"].items() if states.get(t, {}).get("state") != NOT_SCORED}
    fam = A.family(blocks, tags)
    out: dict[str, Any] = {"family": name, "tags": tags, "family_size": len(fam), "tests": fam}
    if len(fam) != R3.FAMILY_SIZES[name]:
        raise ValueError(f"family {name} has {len(fam)} tests, fixed at {R3.FAMILY_SIZES[name]}")
    done = [t for t in tags if t in blocks]
    not_scored = [t for t in tags if states.get(t, {}).get("state") == NOT_SCORED]
    out["verdicts"] = {t: A.model_verdicts(blocks[t], fam, t) for t in done}
    for t in not_scored:
        out["verdicts"][t] = {"contrasts": {c: NOT_SCORED for c in CONTRASTS}, "competent": {c: False for c in CONTRASTS},
                              "state": states[t], "rejections": {"R1_penalty_absent": {"fires": False}}}
    out["not_scored"] = not_scored
    out["waiting"] = [t for t in tags if t not in blocks and t not in not_scored]
    if not out["waiting"]:
        v = {t: out["verdicts"][t]["contrasts"] for t in tags}
        comp = {t: out["verdicts"][t]["competent"] for t in tags}
        r2c = r2_contemporary(v, {t: comp[t]["parent_penalty"] for t in tags})
        r1 = {t: out["verdicts"][t]["rejections"]["R1_penalty_absent"]["fires"] for t in tags}
        out["R2_contemporary"] = r2c
        out["H3_P"] = h3p_reading(v, {t: comp[t]["parent_penalty"] for t in tags}, r1, r2c)
        out["replication"] = replication_block(r2, v, comp)
    return out


def analyse(records: list[dict], shots, scores: dict[str, list[dict]], cfg: dict[str, Any],
            r2: dict[str, Any] | None = None, old_scores: dict[str, list[dict]] | None = None,
            environment: dict[str, Any] | None = None, think: list[dict] | None = None,
            formats: dict[str, tuple[str, str]] | None = None, states: dict[str, dict] | None = None,
            gemma: str | None = None, refused: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The analysis. `gemma` is the Gemma slot from the filed smoke record (r3_gates.gemma_slot); `states` the model
    states (r3_gates.model_state); `refused` the families refused for provenance."""
    if cfg.get("b_draws") not in (None, B_DRAWS) and not G.mock():  # review 1, S7
        raise ValueError(f"b_draws {cfg.get('b_draws')} outside a mock run: B3 uses {B_DRAWS} resamples")
    gemma = gemma or R3.GEMMA_CORE
    refused = refused or {}
    states = states or {}
    core, ext = list(R3.core_tags(gemma)), list(R3.EXTENSION)
    ma = model_analysis(records, shots, {t: s for t, s in scores.items() if states.get(t, {}).get("state") != NOT_SCORED},
                        core + ext)
    out: dict[str, Any] = {"gemma_core": gemma, "missing_scores": ma["missing"], "fallback_used": gemma != R3.GEMMA_CORE,
                           "model_states": states, "per_model": ma["blocks"]}
    for name, tags in (("A3", core), ("A3x", ext)):
        out[name] = ({"family": name, "refused": refused[name], "family_size": R3.FAMILY_SIZES[name]} if name in refused
                     else family_a(name, tags, ma, r2, states))
    # analysis (b)
    environment = environment or {"status": "waiting"}
    out["environment_check"] = environment
    source = b3_source(environment.get("status", "waiting"), cfg)
    pairs = list(R3.pairs(gemma))
    diffs: dict[str, dict] = {}
    readings: dict[str, dict] = {}
    olds = model_analysis(records, shots, old_scores or {}, [o for o, _n in pairs]) if source != "waiting" else None
    for old, new in pairs:
        key = f"{old}->{new}"
        if states.get(new, {}).get("state") == NOT_SCORED:
            readings[key] = {c: NOT_SCORED for c in CONTRASTS}
        elif source == "waiting" or "B3" in refused:
            readings[key] = {c: ("refused (provenance)" if "B3" in refused else "waiting for the environment check")
                             for c in CONTRASTS}
        elif old in olds["bys"] and new in ma["bys"]:
            diffs[key] = {}
            sd_old = olds["blocks"][old]["equivalence_margin"]["sd_delta_L_main"]
            sd_new = ma["blocks"][new]["equivalence_margin"]["sd_delta_L_main"]
            for c in CONTRASTS:
                vo = record_values(olds["bys"][old], ma["pops"], c)
                vn = record_values(ma["bys"][new], ma["pops"], c)
                rec = paired_difference(vo, vn, c not in WITHIN, B=int(cfg.get("b_draws", B_DRAWS)),
                                        fixed_sd=(sd_old, sd_new))
                if c in WITHIN:
                    rec["item_share_descriptive"] = share_difference(vo, vn)
                rec["first_token_tail"] = {"old": olds["blocks"][old]["round3"]["first_token_tail"],
                                           "new": ma["blocks"][new]["round3"]["first_token_tail"]}
                rec["competent"] = {"old": olds["blocks"][old]["round3"]["competent"][c],
                                    "new": ma["blocks"][new]["round3"]["competent"][c]}
                diffs[key][c] = rec
    fam_b = b3_family(diffs, pairs)
    for old, new in pairs:
        key = f"{old}->{new}"
        if key not in readings:
            readings[key] = {c: pair_reading(diffs.get(key, {}).get(c), fam_b, f"{key}:{c}") for c in CONTRASTS}
        readings[key]["what_changes"] = PAIR_CHANGES.get(key, "")
    out["B3"] = {"family": fam_b, "family_size": len(fam_b), "differences": diffs, "readings": readings,
                 "margin": B3_MARGIN, "scale": "standardized (primary); raw nats and the fixed-denominator column "
                 "descriptive only", "old_scores": source, "caveat": PAIR_CAVEAT,
                 "equivalence_note": "equivalence within 0.20 SD is practically unreachable for Dn, Dc and T at these "
                                     "sample sizes (design 14.S3); 'not distinguishable' is never 'the same'"}
    if len(fam_b) != R3.FAMILY_SIZES["B3"]:
        raise ValueError(f"family B3 has {len(fam_b)} tests, fixed at {R3.FAMILY_SIZES['B3']}")
    # analysis (c), exploratory: estimates and intervals only (S11)
    expl: dict[str, Any] = {}
    for small, large in (("qwen3.5-9b", "qwen3.8-27b"), (gemma, "gemma4-31b-it")):
        if small in ma["bys"] and large in ma["bys"]:
            expl[f"{small}->{large}"] = {c: descriptive_only(paired_difference(
                record_values(ma["bys"][small], ma["pops"], c), record_values(ma["bys"][large], ma["pops"], c),
                c not in WITHIN, B=DESC_DRAWS)) for c in CONTRASTS}
    expl["capability"] = {t: {"overall_yes_no_auc": b["competence"]["checks"]["auc_neutral"]["value"],
                              "standardized": {c: b["round3"]["standardized"][c]["estimate"] for c in CONTRASTS}}
                          for t, b in ma["blocks"].items()}
    if formats:
        expl["format_checks"] = {}
        for name, (native_tag, variant_tag) in formats.items():
            if native_tag in ma["bys"] and variant_tag in scores:
                vby = {(s["transition_id"], s["condition"]): s for s in scores[variant_tag]}
                expl["format_checks"][name] = format_block(ma["bys"][native_tag], vby, records)
    if think is not None and "qwen3.5-9b" in ma["bys"]:
        expl["thinking_arm"] = think_block(think, ma["bys"]["qwen3.5-9b"], records)
    expl["kind"] = "exploratory: estimates and intervals only; no test, no family"
    out["exploratory"] = expl
    out["inference_map"] = INFERENCE_MAP
    return out


INFERENCE_MAP = {
    "confirmatory": {"A3": "3 core models x 5 contrasts x (differs, equivalent), Holm, 30 tests",
                     "A3x": "3 extension models x 5 x 2, Holm, 30 tests",
                     "B3": "3 pairs x 5 contrasts x (differs, equivalent) on the standardized difference, Holm, 30 tests"},
    "decision rules from verdicts (no new test)": ["H3-P", "R1", "R2 (round 3)", "R5", "round 2's finding",
                                                   "replication", "pair readings", "readings", "interpretations"],
    "descriptive": ["R3 diagnostic", "central question", "standardized contrasts (analysis a)", "raw nats in B3",
                    "the fixed-denominator column in B3", "item shares", "first-token / tail split",
                    "competence at the 0.80 floor", "the smoke runs' format diagnostic", "every other readout"],
    "exploratory": ["size and recency within a line", "format checks", "thinking arm", "capability table"],
    "gates": ["the environment check (B3 reads round 2's scores only if it passes, else the serve2 rescores)",
              "the Gemma fallback rule (r3_gates.gemma_slot, on the filed smoke record)",
              "model states: 'not scored' from a filed gate failure (r3_gates.model_state)"],
}


def newer_model_refusals(probs: dict[str, list[str]], pairs) -> list[str]:
    """B3 is refused when a pair's newer model (the Gemma slot's model included) has a provenance problem (review 2,
    S1), as A3 or A3x is for the same model."""
    return [f"a pair's newer model: {x}" for _o, n in pairs for x in probs.get(n, [])]


def _sha(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run_textcraft_r3_analysis(cfg: dict) -> int:
    run_id = cfg["run_id"]
    files = verified_files(cfg)
    if isinstance(files, str):
        write_status(run_id, f"records not verified: {files}", ok=False)
        return 1
    rpath, spath, mpath = files
    d = _init_run(run_id, cfg, inputs=(rpath, spath, mpath))
    records = load_records(rpath)
    shots = P.fewshot_examples(spath)
    root = runs_root()
    mock_run = G.mock()
    # the Gemma slot from the filed smoke record (design 14.M3); a config value must agree
    slot = G.gemma_slot(root)
    if slot["slot"] == "waiting":
        gemma = R3.GEMMA_CORE
    else:
        gemma = slot["slot"]
    if cfg.get("gemma_core") not in (None, gemma):
        write_status(run_id, f"gemma_core is {cfg.get('gemma_core')!r}, but the fallback rule gives {gemma!r}", ok=False)
        return 1
    tags = list(R3.core_tags(gemma)) + list(R3.EXTENSION)
    rule_bad = G.rule_problems()  # round 3's own rule, as pinned (design 14.M4)
    if rule_bad:
        write_status(run_id, f"environment-check rule: {rule_bad}", ok=False)
        return 1
    states = {t: G.model_state(root, t) for t in tags}
    first_core_file = root / G.FIRST_CORE
    first_core = json.loads(first_core_file.read_text())["time"] if first_core_file.exists() else None
    declared = {}
    for t in tags:  # review 2, S6: an extension model declared not run counts as not scored
        dec = G.not_run_declaration(root, t, first_core)
        if dec["declared"]:
            declared[t] = dec
            if dec["valid"] and states[t]["state"] == "waiting":
                states[t] = {"state": NOT_SCORED, "reasons": [f"declared not run ({dec.get('reason')}, {dec['date']})"]}
    scores, folders = {}, {}
    for t in tags:
        shards = load_run(run_folder("score", t), root)
        if shards:
            folders[t] = [p for p, _s in shards]
            scores[t] = [s for _p, ss in shards for s in ss]
    formats = {}
    for name, (model, suffix) in (("placement", ("gemma4-12b-it", FAM.FIRST_TURN)),
                                  ("nopreserve", ("qwen3.8-27b", FAM.NO_PRESERVE))):
        shards = load_run(run_folder(name, model), root)
        if shards:
            scores[model + suffix] = [s for _p, ss in shards for s in ss]
            folders[model + suffix] = [p for p, _s in shards]
            formats[name] = (model, model + suffix)
    # every other round-3 run is checked for its commit too (S12): environment check, rescoring, thinking arm
    other = {}
    for t in R3.ENVCHECK_MODELS:
        for kind in ("envcheck", "rescore"):
            ps = [p for p, _s in load_run(run_folder(kind, t), root)]
            if ps:
                other[f"{kind}:{t}"] = ps
    tps = [p for p, _s in load_run(run_folder("think", "qwen3.5-9b"), root)]
    if tps:
        other["think"] = tps
    tag_commit = None if mock_run else G.freeze_commit()
    # D-T1: the thinking arm's shards may also run at the deviation commit; every other run only at the freeze tag's
    dev = {"commit": None, "problems": []} if mock_run else G.deviation_commit(tag_commit=tag_commit)
    probs = provenance_problems(folders, scores, tag_commit, mock_run)
    for k, ps in (other.items() if not mock_run else ()):  # review 2, S2
        bad = [p.name for p in ps if G.shard_dirty(p)]
        if bad:
            probs.setdefault(k, []).append(f"{k}: shards {bad} ran with uncommitted changes (git_dirty)")
    for t, dec in declared.items():
        if not dec["valid"]:
            probs.setdefault(t, []).extend(dec["problems"])
    if not mock_run:
        cp = other_commit_problems(other, tag_commit, dev)
        for k, v in cp.items():
            probs.setdefault(k, []).extend(v)
        for k, ps in other.items():  # the fallback rescoring runs on exactly one L40S per job (design 14.M4)
            if k.startswith("rescore:") and any(g != [R3.RESCORE_GPU] for p in ps for g in G.job_gpus(p)):
                probs.setdefault(k, []).append(f"{k}: jobs not on exactly [{R3.RESCORE_GPU!r}]")
    refused: dict[str, list[str]] = {}
    for name, ts in (("A3", list(R3.core_tags(gemma))), ("A3x", list(R3.EXTENSION))):
        bad = [x for t in ts for x in probs.get(t, [])]
        if bad:
            refused[name] = bad
    r2_root = Path(cfg.get("r2_runs_root") or root)
    r2_metrics = r2_root / "textcraft_r2_analysis" / "metrics.json"
    r2 = read_round2(r2_metrics)
    r2_status = (r2_root / "textcraft_r2_analysis" / "STATUS.md")
    # the environment check (design 14.M4)
    environment: dict[str, Any] = {"models": {}, "tokens": {}}
    gpu_of = (lambda p: ["mock"]) if mock_run else G.shard_gpu
    gdir = Path(__file__).resolve().parent / "data"
    load = lambda f: json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}  # noqa: E731
    for t in R3.ENVCHECK_MODELS:
        old = [(p.parent, read_jsonl(p)) for p in sorted((r2_root / A.score_run(t)).glob("shard_*/scoring.jsonl"))]
        new = load_run(run_folder("envcheck", t), root)
        unob = G.unobtainable_check(root / run_folder("envcheck", t))  # review 2, S4
        un = unob["valid"]
        if unob["filed"] and not unob["valid"]:
            probs.setdefault(f"envcheck:{t}", []).append(f"gpu_unobtainable.json: {unob.get('problem')}")
        plan = (G.read_plan() or {}).get("models", {}).get(t, {})
        if plan and not plan.get("runnable", True):
            un = True
        if old:
            environment["models"][t] = envcheck_block(old, new, t, records, gpu_of, unobtainable=un)
        environment["tokens"][t] = golden_tokens_check(load(gdir / "r3_chat_golden.json"), load(gdir / "r2_chat_golden.json"), t)
    environment["status"] = environment_status(environment["models"], environment["tokens"])
    environment["pass"] = environment["status"] == "pass"
    # B3's older scores (S5, S6): from the environment check's outcome; round 2's only after round 2's provenance check
    gemma_pairs = R3.pairs(gemma)
    try:
        source = b3_source(environment["status"], cfg)
    except ValueError as e:
        write_status(run_id, str(e), ok=False)
        return 1
    old_scores: dict[str, list[dict]] = {}
    newer_bad = newer_model_refusals(probs, gemma_pairs)  # review 2, S1
    if newer_bad:
        refused["B3"] = newer_bad
    if source == "round2":
        bad = A.provenance([o for o, _n in gemma_pairs], r2_root)
        if bad:
            refused["B3"] = refused.get("B3", []) + [f"round 2's provenance: {x}" for x in bad]
        else:
            for old, _new in gemma_pairs:
                ss = A.load_scores(old, r2_root)
                if ss:
                    old_scores[old] = ss
    elif source == "rescore":
        for old, _new in gemma_pairs:
            ss = [s for _p, x in load_run(run_folder("rescore", old), root) for s in x]
            if ss:
                old_scores[old] = [{**s, "model": old} for s in ss]  # saved under <tag>-serve2
        bad = [x for o, _n in gemma_pairs for x in probs.get(f"rescore:{o}", [])]
        if bad:
            refused["B3"] = refused.get("B3", []) + bad
    think = [s for _p, ss in load_run(run_folder("think", "qwen3.5-9b"), root) for s in ss] or None
    try:
        metrics = analyse(records, shots, scores, cfg, r2, old_scores, environment, think, formats, states, gemma, refused)
    except A.StaleScore as e:
        write_status(run_id, f"stale score: {e}", ok=False)
        return 1
    metrics["gemma_slot"] = slot
    metrics["declared_not_run"] = declared
    metrics["environment_rule"] = G.rule_fingerprint()
    if metrics["A3"].get("H3_P") is not None and not first_core_file.exists():  # the first core reading, once
        import datetime as dt

        write_json(first_core_file, {"time": dt.datetime.now().isoformat(timespec="seconds"), "run": run_id})
    metrics["provenance_problems"] = probs
    metrics["round2_analysis_read"] = {"metrics_sha256": _sha(r2_metrics), "status_line":
                                       r2_status.read_text(encoding="utf-8").splitlines()[0] if r2_status.exists() else None}
    metrics["freeze_commit"] = tag_commit
    metrics["deviation_t1"] = {"commit": dev["commit"], "problems": dev["problems"],  # which commit each thinking shard ran at
                               "think_shard_commits": think_shard_commits(other.get("think", []))}
    write_json(d / "metrics.json", metrics)
    lines = [f"Gemma slot: {gemma} (triggers {slot.get('triggers')})"]
    for fam in ("A3", "A3x"):
        f = metrics[fam]
        if "refused" in f:
            lines.append(f"{fam}: refused (provenance) {f['refused'][:3]}")
        elif f["waiting"]:
            lines.append(f"{fam}: waiting {f['waiting']}")
        else:
            lines.append(f"{fam}: H3-P {f['H3_P']['overall']}; not scored {f['not_scored']}; replication "
                         + "; ".join(f"{c} {r['reading']}" for c, r in f["replication"].items()))
    lines.append(f"environment check: {environment['status']}; B3 reads {source}")
    lines.append("B3: " + "; ".join(f"{k} " + ", ".join(f"{c} {v}" for c, v in r.items() if c in CONTRASTS)
                                    for k, r in metrics["B3"]["readings"].items()))
    write_status(run_id, "\n".join(lines), ok=True)
    return 0
