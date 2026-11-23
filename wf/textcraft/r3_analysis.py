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
ENV_RULE = FAM.ENVCHECK_RULE  # wave B's pass rule, unchanged
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
                      margin: float = B3_MARGIN) -> dict[str, Any]:
    """The newer model's standardized contrast minus the older model's on the records both scored, with one goal-pair
    resample serving both models (design 7.4): estimate, 95% interval, two-sided bootstrap p, TOST p against
    `margin`; and the raw difference in nats with its interval (descriptive)."""
    B = B_DRAWS if B is None else B
    common = set(old) & set(new)
    ao, an = _cluster_sums(old, common, gap), _cluster_sums(new, common, gap)
    keys = sorted(set(ao) | set(an))
    ro, zo = _stats(_total(ao, keys), gap)
    rn, zn = _stats(_total(an, keys), gap)
    rng = random.Random(seed)
    dz, dr = [], []
    for _ in range(B):
        ks = rng.choices(keys, k=len(keys))
        r1, z1 = _stats(_total(ao, ks), gap)
        r2, z2 = _stats(_total(an, ks), gap)
        if z1 == z1 and z2 == z2:
            dz.append(z2 - z1)
        if r1 == r1 and r2 == r2:
            dr.append(r2 - r1)
    return {"standardized": {"old": zo, "new": zn, "estimate": zn - zo, "ci95": [A.pct(dz, 0.025), A.pct(dz, 0.975)],
                             "p_differs": A.bootstrap_p(dz), "p_equivalent": A.tost_p(dz, margin), "margin": margin,
                             "bootstrap_se": statistics.stdev(dz) if len(dz) > 1 else float("nan")},
            "raw_nats_descriptive": {"old": ro, "new": rn, "estimate": rn - ro,
                                     "ci95": [A.pct(dr, 0.025), A.pct(dr, 0.975)]},
            "n_items": len(common), "n_pairs": len(keys), "resampling": "paired, whole goal pairs"}


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
        if competent[t] and p == "confirmed below zero":
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
    elif len(comp) >= MIN_COMPETENT and len(sup) == len(comp):
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
    together; a missing pair or NaN p enters as 1."""
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


def pair_reading(rec: dict | None, fam: dict, key: str) -> str:
    """A pair's reading for one contrast (design 7.4): differs (newer higher / lower), similar within the margin, or
    not distinguishable."""
    if rec is None:
        return "waiting"
    s = rec["standardized"]
    lo, hi = s["ci95"]
    if fam[f"{key}:differs"]["p_holm"] < ALPHA:
        if s["estimate"] > 0 and lo > 0:
            return "differs, newer higher"
        if s["estimate"] < 0 and hi < 0:
            return "differs, newer lower"
    if fam[f"{key}:equivalent"]["p_holm"] < ALPHA:
        return "similar within the margin"
    return "not distinguishable"


PAIR_CAVEAT = ("Each pair changes developer data, post-training, size, tokenizer, template, architecture and software at "
               "once: a 'differs' describes two released models, not a generation effect; three purposive pairs say "
               "nothing about model generations in general.")
PAIR_CHANGES = {
    "qwen3-8b->qwen3.5-9b": "data and post-training; 8.2 B to 9.7 B parameters; tokenizer; the same empty think block "
                            "before the reply; Qwen3.5's linear-attention layers; transformers 4.51 to 5.23",
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


# ---------------- the environment check ----------------


def envcheck_block(old: list[tuple[Path, list[dict]]], new: list[tuple[Path, list[dict]]], tag: str,
                   records: list[dict], gpu_of: Callable[[Path], list[str] | None]) -> dict[str, Any]:
    """Round 2's scores of `tag` against its serve2 rescores (design 8.1), on the records whose round-2 shard ran on
    exactly the rescoring's GPU type; wave B's pass rule."""
    new_gpus = {str(gpu_of(p)) for p, _s in new}
    want = gpu_of(new[0][0]) if new and len(new_gpus) == 1 else None
    oldmap, newmap, comparable = {}, {}, set()
    for p, ss in old:
        same = gpu_of(p) == want and want is not None
        for s in ss:
            if s.get("model") == tag:
                oldmap[(s["transition_id"], s["condition"])] = s["L"]
                if same:
                    comparable.add(s["transition_id"])
    for _p, ss in new:
        for s in ss:
            if s.get("model") == tag + R3.ENVCHECK_SUFFIX:
                newmap[(s["transition_id"], s["condition"])] = s["L"]
    from wf.textcraft.r3_score import envcheck_subset

    chosen = [r for r in envcheck_subset(records) if r["id"] in comparable]
    expected = {(r["id"], a) for r in chosen for a in r["arms"]}
    pairs = {k: (oldmap[k], newmap[k]) for k in expected if k in oldmap and k in newmap}
    checks: dict[str, bool] = {"gpu_type_single": want is not None,
                               "enough_comparable_records": len(chosen) >= R3.ENVCHECK_MIN_COMPARABLE,
                               "complete": bool(expected) and expected <= set(pairs)}
    out: dict[str, Any] = {"gpu": want, "n_comparable_records": len(chosen), "n_expected": len(expected),
                           "n_paired": len(pairs), "rule": ENV_RULE}
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
               "goal_effect_diff": effects["new"] - effects["old"]}
        checks[f"{fmt}_mean_abs"] = rec["mean_abs_diff"] <= ENV_RULE[f"{fmt}_mean_abs_max"]
        checks[f"{fmt}_goal_effect"] = abs(rec["goal_effect_diff"]) <= ENV_RULE[f"{fmt}_goal_effect_diff_max"]
        if fmt == "yes_no":
            rec["share_diff"] = effects["share_new"] - effects["share_old"]
            checks["yes_no_share"] = abs(rec["share_diff"]) <= ENV_RULE["yes_no_share_diff_max"]
        out[fmt] = rec
    out["checks"] = checks
    out["pass"] = all(checks.values())  # NaN fails every comparison
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
    """The thinking arm's readouts (design 8.2), all descriptive: P and Λ after reasoning, their standardized values,
    their paired differences from thinking off on the same records, the yes/no AUC after reasoning, the forced share,
    the reasoning length and the mention rates; on all generations and on unforced ones."""
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
                        "on_minus_off": paired_difference(off, on, False, B=DESC_DRAWS)}
        succ = [by[(r["id"], "yn_neutral")]["L"] for r in pops["successes"] if (r["id"], "yn_neutral") in by]
        fail = [by[(r["id"], "yn_neutral")]["L"] for r in pops["main_failures"] if (r["id"], "yn_neutral") in by]
        block["yes_no_auc_after_reasoning"] = {"auc": A.auc(succ, fail), "n_successes": len(succ), "n_failures": len(fail)}
        out[subset] = block
    n = len(think)
    out["forced_share"] = sum(bool(s.get("forced")) for s in think) / n if n else float("nan")
    out["median_new_tokens"] = statistics.median([s["n_new_tokens"] for s in think]) if n else float("nan")
    out["mentions_goal_word"] = sum(bool(s.get("mentions_goal_word")) for s in think) / n if n else float("nan")
    out["mentions_crafted_item"] = sum(bool(s.get("mentions_crafted_item")) for s in think) / n if n else float("nan")
    out["boundary_problems"] = sum(not s.get("boundary_ok", True) for s in think)
    out["kind"] = "exploratory; outside every family; never gated or pooled with the instruction runs"
    return out


# ---------------- provenance ----------------


def provenance_problems(folders: dict[str, list[Path]], scores: dict[str, list[dict]],
                        record: Callable[[Path], dict[str, Any]] | None = None) -> list[str]:
    """Every shard of the round-3 scoring runs at one commit; every score with its model's pinned revision; every
    shard of a model on one GPU list, of its pinned type (design section 10)."""
    if record is None:
        from wf.provenance import run_record as record
    problems = []
    recs = {tag: [record(p) for p in ps] for tag, ps in folders.items()}
    commits = {r["commit"] for rs in recs.values() for r in rs}
    if len(commits) > 1:
        problems.append(f"shards were scored at {len(commits)} commits: {sorted(map(str, commits))}")
    for tag, rs in recs.items():
        gpus = {str(r["gpu"]) for r in rs}
        if len(gpus) > 1:
            problems.append(f"{tag}: shards ran on different GPUs {sorted(gpus)}")
        base = R3.base_tag(tag)
        if base in R3.GPU_PIN and rs and not R3.gpu_type_ok(base, rs[0]["gpu"] if isinstance(rs[0]["gpu"], list) else None):
            problems.append(f"{tag}: GPU {rs[0]['gpu']} is not the pinned {R3.GPU_PIN[base]}")
    for tag, ss in scores.items():
        revs = {s.get("model_revision") for s in ss}
        want = R3.spec_for(tag).revision
        if revs and revs != {want}:
            problems.append(f"{tag}: model revisions {sorted(map(str, revs))}, pinned {want}")
    return problems


# ---------------- the analysis ----------------


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
        block["round3"] = r3_extras(by, pops, block)
        out[tag] = block
    return {"blocks": out, "bys": bys, "missing": missing, "pops": pops}


def family_a(name: str, tags: list[str], ma: dict[str, Any], r2: dict[str, Any] | None) -> dict[str, Any]:
    """A3 or A3x: round 2's family function over these tags (30 tests), verdicts, H3-P, R2 and replication."""
    blocks = ma["blocks"]
    fam = A.family(blocks, tags)
    out: dict[str, Any] = {"family": name, "tags": tags, "family_size": len(fam), "tests": fam}
    if len(fam) != R3.FAMILY_SIZES[name]:
        raise ValueError(f"family {name} has {len(fam)} tests, fixed at {R3.FAMILY_SIZES[name]}")
    done = [t for t in tags if t in blocks]
    out["verdicts"] = {t: A.model_verdicts(blocks[t], fam, t) for t in done}
    out["waiting"] = [t for t in tags if t not in blocks]
    if len(done) == len(tags):
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
            formats: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    gemma = cfg.get("gemma_core", R3.GEMMA_CORE)
    core, ext = list(R3.core_tags(gemma)), list(R3.EXTENSION)
    tags = core + ext
    ma = model_analysis(records, shots, scores, tags)
    out: dict[str, Any] = {"gemma_core": gemma, "missing_scores": ma["missing"],
                           "fallback_used": gemma != R3.GEMMA_CORE,
                           "per_model": ma["blocks"]}
    out["A3"] = family_a("A3", core, ma, r2)
    out["A3x"] = family_a("A3x", ext, ma, r2)
    # analysis (b)
    env_ok = bool(environment and environment.get("pass"))
    out["environment_check"] = environment or {"pass": None, "status": "waiting"}
    pairs = list(R3.pairs(gemma))
    diffs: dict[str, dict] = {}
    if old_scores is not None and (env_ok or cfg.get("b3_old_scores") == "rescore"):
        olds = model_analysis(records, shots, old_scores, [o for o, _n in pairs])
        for old, new in pairs:
            if old in olds["bys"] and new in ma["bys"]:
                key = f"{old}->{new}"
                diffs[key] = {}
                for c in CONTRASTS:
                    vo = record_values(olds["bys"][old], ma["pops"], c)
                    vn = record_values(ma["bys"][new], ma["pops"], c)
                    rec = paired_difference(vo, vn, c not in WITHIN, B=int(cfg.get("b_draws", B_DRAWS)))
                    if c in WITHIN:
                        rec["item_share_descriptive"] = share_difference(vo, vn)
                    rec["first_token_tail"] = {"old": olds["blocks"][old]["round3"]["first_token_tail"],
                                               "new": ma["blocks"][new]["round3"]["first_token_tail"]}
                    rec["competent"] = {"old": olds["blocks"][old]["round3"]["competent"][c],
                                        "new": ma["blocks"][new]["round3"]["competent"][c]}
                    diffs[key][c] = rec
    fam_b = b3_family(diffs, pairs)
    readings = {}
    for old, new in pairs:
        key = f"{old}->{new}"
        readings[key] = {c: (pair_reading(diffs.get(key, {}).get(c), fam_b, f"{key}:{c}") if env_ok or
                             cfg.get("b3_old_scores") == "rescore" else NOT_INTERPRETABLE_ENV) for c in CONTRASTS}
        readings[key]["what_changes"] = PAIR_CHANGES.get(key, "")
    out["B3"] = {"family": fam_b, "family_size": len(fam_b), "differences": diffs, "readings": readings,
                 "margin": B3_MARGIN, "scale": "standardized (primary); raw nats descriptive only",
                 "old_scores": cfg.get("b3_old_scores", "round2"), "caveat": PAIR_CAVEAT}
    if len(fam_b) != R3.FAMILY_SIZES["B3"]:
        raise ValueError(f"family B3 has {len(fam_b)} tests, fixed at {R3.FAMILY_SIZES['B3']}")
    # analysis (c), exploratory
    expl: dict[str, Any] = {}
    for small, large in (("qwen3.5-9b", "qwen3.8-27b"), (gemma, "gemma4-31b-it")):
        if small in ma["bys"] and large in ma["bys"]:
            expl[f"{small}->{large}"] = {c: paired_difference(record_values(ma["bys"][small], ma["pops"], c),
                                                              record_values(ma["bys"][large], ma["pops"], c),
                                                              c not in WITHIN, B=DESC_DRAWS) for c in CONTRASTS}
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
    expl["kind"] = "exploratory: estimates and intervals only; no family"
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
                    "item shares", "first-token / tail split", "competence at the 0.80 floor", "every other readout"],
    "exploratory": ["size and recency within a line", "format checks", "thinking arm", "capability table"],
    "gates": ["the environment check (B3 is read only if it passes, or on the serve2 rescores)",
              "the Gemma fallback rule (applied at the smoke run, before any contrast)"],
}


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
    gemma = cfg.get("gemma_core", R3.GEMMA_CORE)
    tags = list(R3.core_tags(gemma)) + list(R3.EXTENSION)
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
    bad = provenance_problems(folders, scores)
    if bad:
        write_status(run_id, f"provenance: {bad}", ok=False)
        return 1
    r2_root = Path(cfg.get("r2_runs_root") or root)
    r2 = read_round2(r2_root / "textcraft_r2_analysis" / "metrics.json")
    # the environment check (needs round 2's scores of the two models and their serve2 rescores)
    environment: dict[str, Any] = {"models": {}, "tokens": {}}
    from wf.provenance import run_record

    gpu_of = lambda p: (lambda g: g if isinstance(g, list) else None)(run_record(p)["gpu"])  # noqa: E731
    r3_golden_path = Path(__file__).resolve().parent / "data" / "r3_chat_golden.json"
    r2_golden_path = Path(__file__).resolve().parent / "data" / "r2_chat_golden.json"
    for t in R3.ENVCHECK_MODELS:
        old = [(p.parent, read_jsonl(p)) for p in sorted((r2_root / A.score_run(t)).glob("shard_*/scoring.jsonl"))]
        new = load_run(run_folder("envcheck", t), root)
        if old and new:
            environment["models"][t] = envcheck_block(old, new, t, records, gpu_of)
        load = lambda f: json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}  # noqa: E731
        environment["tokens"][t] = golden_tokens_check(load(r3_golden_path), load(r2_golden_path), t)
    environment["pass"] = bool(set(environment["models"]) == set(R3.ENVCHECK_MODELS)
                               and all(m["pass"] for m in environment["models"].values())
                               and set(environment["tokens"]) == set(R3.ENVCHECK_MODELS)
                               and all(x["ok"] for x in environment["tokens"].values()))
    old_scores: dict[str, list[dict]] = {}
    for old, _new in R3.pairs(gemma):
        if cfg.get("b3_old_scores") == "rescore":
            ss = [s for _p, x in load_run(run_folder("rescore", old), root) for s in x]
            old_scores[old] = [{**s, "model": old} for s in ss]  # saved under <tag>-serve2
        else:
            ss = A.load_scores(old, r2_root)
            if ss:
                old_scores[old] = ss
    think = [s for _p, ss in load_run(run_folder("think", "qwen3.5-9b"), root) for s in ss] or None
    try:
        metrics = analyse(records, shots, scores, cfg, r2, old_scores, environment, think, formats)
    except A.StaleScore as e:
        write_status(run_id, f"stale score: {e}", ok=False)
        return 1
    write_json(d / "metrics.json", metrics)
    lines = []
    for fam in ("A3", "A3x"):
        f = metrics[fam]
        lines.append(f"{fam}: waiting {f['waiting']}" if f["waiting"] else
                     f"{fam}: H3-P {f['H3_P']['overall']}; replication "
                     + "; ".join(f"{c} {r['reading']}" for c, r in f["replication"].items()))
    lines.append("B3: " + "; ".join(f"{k} " + ", ".join(f"{c} {v}" for c, v in r.items() if c in CONTRASTS)
                                    for k, r in metrics["B3"]["readings"].items()))
    write_status(run_id, "\n".join(lines), ok=True)
    return 0
