"""Independent pre-freeze audit of the TextCraft round-2 statistics (notes/textcraft_r2_audit.md).

Runs on synthetic data and on the real round-2 records with random scores; it never reads a model score.
Usage: python scripts/audit_textcraft_r2_stats.py
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wf.textcraft import r2 as R  # noqa: E402
from wf.textcraft import r2_analysis as A  # noqa: E402
from wf.textcraft import prompts as P  # noqa: E402
from wf.textcraft.r2_score import load_records  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---------------- 1. Holm at a fixed 50 with NaN padding ----------------

def my_holm(ps: list[float]) -> list[float]:
    """Textbook Holm step-down on exactly the list given (no entries dropped)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def audit_holm() -> None:
    rng = random.Random(1)
    tags = list(R.TAGS)
    for trial in range(200):
        per = {}
        for t in tags:
            if rng.random() < 0.2:
                continue  # model not scored
            per[t] = {"contrasts": {}}
            for c in A.CONTRASTS:
                if rng.random() < 0.1:
                    continue  # contrast missing
                per[t]["contrasts"][c] = {
                    "p_differs": float("nan") if rng.random() < 0.1 else rng.random() ** 3,
                    "p_equivalent": float("nan") if rng.random() < 0.1 else rng.random() ** 3}
        fam = A.family(per, tags)
        keys = [f"{t}:{c}:{x}" for t in tags for c in A.CONTRASTS for x in A.TESTS]
        raw = []
        for k in keys:
            t, c, x = k.split(":")
            rec = per.get(t, {}).get("contrasts", {}).get(c)
            p = rec[f"p_{x}"] if rec else float("nan")
            raw.append(p if p == p else 1.0)
        want = my_holm(raw)
        ok = len(fam) == 50 and all(abs(fam[k]["p_holm"] - w) < 1e-15 for k, w in zip(keys, want))
        if not ok:
            check("Holm equals textbook Holm over a fixed 50 (NaN/missing -> 1)", False, f"trial {trial}")
            return
    check("Holm equals textbook Holm over a fixed 50 (NaN/missing -> 1), 200 random families", True)
    # smallest p is multiplied by 50 even when 45 are padded
    per = {"qwen3-8b": {"contrasts": {"lift": {"p_differs": 0.0005, "p_equivalent": float("nan")}}}}
    fam = A.family(per, tags)
    check("single real p of 0.0005 among 49 padded -> Holm 0.025", abs(fam["qwen3-8b:lift:differs"]["p_holm"] - 0.025) < 1e-12)
    # The family size follows the `expected` list; it is 50 only because the config lists five models.
    fam4 = A.family({}, tags[:4])
    check("family size depends on the expected list (4 models -> 40): no hard guard on 50 [finding]", len(fam4) == 40,
          f"len={len(fam4)}")


# ---------------- 2. TOST and the two-sided bootstrap p ----------------

def audit_tost() -> None:
    rng = random.Random(2)
    for _ in range(500):
        B = rng.choice([10, 99, 1000])
        mu, sd, m = rng.uniform(-1, 1), rng.uniform(0.05, 1), rng.uniform(0.01, 0.6)
        d = [rng.gauss(mu, sd) for _ in range(B)]
        lo = (sum(x <= -m for x in d) + 1) / (B + 1)
        hi = (sum(x >= m for x in d) + 1) / (B + 1)
        if A.tost_p(d, m) != max(lo, hi):
            check("TOST p = max of the smoothed one-sided shares", False)
            return
        two = min(1.0, (2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) + 1) / (B + 1))
        if A.bootstrap_p(d) != two:
            check("two-sided bootstrap p per design section 6", False)
            return
    check("TOST p = max((#<=-δ + 1)/(B+1), (#>=δ + 1)/(B+1)) on 500 random draw sets", True)
    check("two-sided bootstrap p = min(1, (2 min(#<=0, #>=0) + 1)/(B+1)) on 500 random draw sets", True)
    # TOST at level a rejects iff the (1-2a) percentile interval lies inside (-δ, δ), up to the +1 smoothing
    r3 = random.Random(3)
    agree = 0
    for mu in [x / 100 for x in range(-50, 51, 2)]:
        d = sorted(r3.gauss(mu, 0.12) for _ in range(20000))
        m = 0.4
        p = A.tost_p(d, m)
        lo90, hi90 = A.pct(d, 0.05), A.pct(d, 0.95)
        agree += (p < 0.05) == (-m < lo90 and hi90 < m)
    check("TOST p < 0.05 agrees with the 90% percentile interval inside (-δ, δ), 51 true means from -0.5 to 0.5", agree == 51,
          f"{agree} of 51 agree")
    check("TOST undefined margin -> NaN (enters the family as 1)",
          all(A.tost_p([0.0, 0.1], x) != A.tost_p([0.0, 0.1], x) for x in (float("nan"), 0.0, -1.0)))
    check("TOST smoothed minimum is 1/(B+1), never 0", A.tost_p([0.0] * 99, 0.4) == 0.01)
    check("sign-flip p smoothed minimum is 1/(n+1)", A.sign_flip_p([1.0] * 30, [str(i) for i in range(30)], n=999) == 1 / 1000)


# ---------------- records and random scores ----------------

def load():
    m = json.loads((ROOT / "wf/textcraft/data/r2_manifest.json").read_text())
    records = load_records(ROOT / m["records_file"])
    shots = P.fewshot_examples(ROOT / m["fewshot_file"])
    for r in records:
        if isinstance(r.get("fit_measures"), str):
            r["fit_measures"] = json.loads(r["fit_measures"])
    return records, shots


def random_scores(records, tag: str, seed: int, shots=None) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for r, a in R.jobs(records):
        L = rng.gauss(0.3 if a.startswith("ext_aligned") else 0.0, 2.0)
        lp = -math.log1p(math.exp(-L))
        lm = -math.log1p(math.exp(L))
        s = {"model": tag, "transition_id": r["id"], "condition": a, "L": L, "logp_plus": lp, "logp_minus": lm,
             "logp_first_plus": lp, "logp_first_minus": lm}
        if shots is not None:
            s["item_hash"] = R.item_hash(r, shots)
        out.append(s)
    return out


def by_of(scores):
    return {(s["transition_id"], s["condition"]): s for s in scores}


# ---------------- 3. δ and its timing ----------------

def audit_margin(records) -> None:
    pops = A.populations(records)
    by = by_of(random_scores(records, "x", 5))
    d = [by[(r["id"], "aligned")]["L"] - by[(r["id"], "swapped")]["L"] for r in pops["main_failures"]]
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    m = A.margin_of(by, pops)
    check("margin SD is the n-1 item SD of ΔL over the 1,500 main failures", n == 1500 and abs(m["sd_delta_L_main"] - sd) < 1e-12,
          f"n={n}, sd={sd:.4f}")
    check("δ = min(0.3 SD, 0.4)", m["margin"] == min(0.3 * sd, 0.4))
    # small-SD case: the SD rule binds
    by2 = {k: {**v, "L": v["L"] * 0.1} for k, v in by.items()}
    m2 = A.margin_of(by2, pops)
    check("δ follows 0.3 SD when that is below 0.4", abs(m2["margin"] - 0.3 * m2["sd_delta_L_main"]) < 1e-15 and m2["margin"] < 0.4,
          f"δ={m2['margin']:.4f}")
    # timing: margin_of runs before any tost_p, once per model, and every TOST uses its value
    events = []
    o_m, o_t = A.margin_of, A.tost_p
    A.margin_of = lambda *a: (events.append(("margin", None)), o_m(*a))[1]
    A.tost_p = lambda dd, mm: (events.append(("tost", mm)), o_t(dd, mm))[1]
    try:
        blk = A.model_block(by, pops)
    finally:
        A.margin_of, A.tost_p = o_m, o_t
    first_tost = next(i for i, e in enumerate(events) if e[0] == "tost")
    margins = [e for e in events if e[0] == "margin"]
    tost_margins = {e[1] for e in events if e[0] == "tost"}
    check("margin computed once per model, before the first TOST, and every TOST uses it",
          len(margins) == 1 and events.index(margins[0]) < first_tost and tost_margins == {blk["equivalence_margin"]["margin"]},
          f"{len(margins)} margin call(s), {sum(e[0] == 'tost' for e in events)} TOST calls")
    # δ does not depend on any contrast population: changing scores outside main failures leaves it unchanged
    mf_ids = {r["id"] for r in pops["main_failures"]}
    by3 = {k: ({**v, "L": v["L"] + 5} if k[0] not in mf_ids else v) for k, v in by.items()}
    check("δ depends only on the main failures' aligned and swapped scores", A.margin_of(by3, pops)["margin"] == m["margin"])


# ---------------- 4. every resample is over goal pairs ----------------

def audit_clusters(records, shots) -> None:
    A.B_DRAWS, A.N_FLIPS, A.DESC_DRAWS, A.AUC_DRAWS = 30, 30, 20, 10
    id2pair = {r["id"]: r["pair_id"] for r in records}
    pairs = set(id2pair.values())
    calls = []
    o_d, o_s = A.draws, A.sign_flip_p
    orig_choices = random.Random.choices
    rng_calls = Counter()

    def spy_d(stat, rows, clusters, *a, **k):
        calls.append(("draws", len(rows), list(clusters)))
        return o_d(stat, rows, clusters, *a, **k)

    def spy_s(values, clusters, *a, **k):
        calls.append(("flip", len(values), list(clusters)))
        return o_s(values, clusters, *a, **k)

    def spy_choices(self, population, *a, **k):
        rng_calls["choices"] += 1
        pop = list(population)
        if not set(pop) <= pairs:
            rng_calls["non_pair_population"] += 1
        return orig_choices(self, population, *a, **k)

    A.draws, A.sign_flip_p = spy_d, spy_s
    random.Random.choices = spy_choices
    try:
        tags = list(R.TAGS)
        out = A.analyse(copy.deepcopy(records), shots, {t: random_scores(records, t, 10 + i, shots) for i, t in enumerate(tags)}, tags)
    finally:
        A.draws, A.sign_flip_p = o_d, o_s
        random.Random.choices = orig_choices
    bad = [c for c in calls if not set(c[2]) <= pairs]
    check("every draws()/sign_flip_p() call in a full analyse() gets goal-pair ids as clusters", not bad and len(calls) > 50,
          f"{len(calls)} calls, {len(bad)} bad")
    check("every rng.choices population (the resampled units) is a set of goal-pair ids",
          rng_calls["choices"] > 0 and rng_calls["non_pair_population"] == 0,
          f"{rng_calls['choices']} choices calls, {rng_calls['non_pair_population']} non-pair")
    check("clusters are coarser than items in every call (no item-level resampling)",
          all(len(set(c[2])) < c[1] for c in calls if c[1] > 300), "")
    # fast path equals general path
    rng = random.Random(4)
    rows = [rng.gauss(0, 1) for _ in range(400)]
    cl = [f"p{rng.randrange(60)}" for _ in rows]
    gen = A.draws(lambda v: sum(v) / len(v), rows, cl, B=200)  # a lambda forces the general path
    fast = A.draws(A.mean, rows, cl, B=200)
    check("fast mean path gives the same draws as the general path", max(abs(a - b) for a, b in zip(gen, fast)) < 1e-9)
    grows = [("goal" if rng.random() < 0.5 else "ingredient", rng.gauss(0, 1)) for _ in range(400)]
    gen = A.draws(lambda v: A._gap(v), grows, cl, B=200)
    fast = A.draws(A._gap, grows, cl, B=200)
    check("fast gap path gives the same draws as the general path",
          len(gen) == len(fast) and max(abs(a - b) for a, b in zip(gen, fast)) < 1e-9)
    # sign flip: flipping item-level would differ from pair-level when pairs hold several items
    vals = [1.0] * 10 + [-0.5] * 10
    p_pair = A.sign_flip_p(vals, ["a"] * 10 + ["b"] * 10, n=999)
    check("sign flip flips whole pairs (two clusters -> p can only take the 2-cluster values)", p_pair > 0.4, f"p={p_pair:.3f}")
    return out


# ---------------- 5. Dn + Dc identity on the 996 rows ----------------

def audit_identity(records) -> None:
    pops = A.populations(records)
    rows = A.two_item_rows(pops)
    cnt = Counter(r["role"] for r in rows)
    check("two_item_rows: 996 rows, 500 goal-role and 496 class-matched ingredient", len(rows) == 996 and cnt == {"goal": 500, "ingredient": 496},
          str(dict(cnt)))
    check("two-item rows: goal rows exclude used up; ingredient rows are too few / never obtained only",
          all(r["failure_class"] != "used_up" for r in rows) and
          all(r["failure_class"] in A.MATCHED_CLASSES for r in rows if r["role"] == "ingredient"))
    check("two-item rows all have a two-item line; none has its partner goal already done",
          all(r["both_task_line"] for r in rows) and not any(R.two_item_partner_done(r) for r in rows))
    worst_est = worst_draw = 0.0
    A.B_DRAWS = 300
    for seed in range(5):
        by = by_of(random_scores(records, "x", 100 + seed))
        ci = A.contrast_inputs(by, pops)
        _, n_rows, n_cl = ci["naming"]
        _, c_rows, c_cl = ci["completion"]
        _, g_rows, g_cl = ci["role_gap"]
        same_rows = n_cl == c_cl == g_cl and [x[0] for x in n_rows] == [x[0] for x in g_rows] and len(g_rows) == 996
        e = abs(A._gap(n_rows) + A._gap(c_rows) - A._gap(g_rows))
        dn, dc, dg = (A.draws(A._gap, r, cl) for r, cl in ((n_rows, n_cl), (c_rows, c_cl), (g_rows, g_cl)))
        dd = max(abs(a + b - c) for a, b, c in zip(dn, dc, dg))
        worst_est, worst_draw = max(worst_est, e), max(worst_draw, dd)
        if not same_rows:
            check("Dn, Dc and the role gap use the same 996 rows in the same order", False)
            return
    check("Dn, Dc and the role gap use the same 996 rows and clusters (contrast_inputs)", True)
    check("Dn + Dc = role gap (item-weighted), 5 random score sets", worst_est < 1e-12, f"max |err| {worst_est:.2e}")
    check("Dn + Dc = role gap in every bootstrap draw (same resamples)", worst_draw < 1e-12, f"max |err| {worst_draw:.2e}")
    # the descriptive role_gap_main uses the same rows
    check("role_gap and matched role gap read two_item_rows (same row set as Dn, Dc)",
          "two_item_rows(pops)" in Path(A.__file__).read_text().split("def contrasts(")[1].split("def family(")[0])
    # matched role gap: cells and weights
    prof = Counter((r["role"], A.short_profile(r)) for r in rows)
    cells = {p for _r, p in prof}
    used = [p for p in cells if min(prof[("goal", p)], prof[("ingredient", p)]) >= A.MATCH_MIN]
    n_used = sum(prof[("goal", p)] + prof[("ingredient", p)] for p in used)
    check("matched role gap: 13 of 15 cells used, 981 items (as 15.F1 states)", len(cells) == 15 and len(used) == 13 and n_used == 981,
          f"{len(used)} of {len(cells)} cells, {n_used} items")
    check("every two-item row has at least one missing input (profile well defined)", all(r["short_inputs"] for r in rows))
    by = by_of(random_scores(records, "x", 7))
    f = lambda i: by[(i, "aligned")]["L"] - by[(i, "swapped")]["L"]
    num = den = 0.0
    for p in used:
        g = [f(r["id"]) for r in rows if r["role"] == "goal" and A.short_profile(r) == p]
        i = [f(r["id"]) for r in rows if r["role"] == "ingredient" and A.short_profile(r) == p]
        w = min(len(g), len(i))
        num += w * (sum(g) / len(g) - sum(i) / len(i))
        den += w
    mg = A.matched_gap(by, rows, f, B=20)
    check("matched role gap = Σ min(n_g, n_i)(mean_g - mean_i) / Σ min(n_g, n_i) over cells with ≥5 per role",
          abs(mg["estimate"] - num / den) < 1e-12 and mg["n_cells_used"] == 13 and mg["n_items_in_used_cells"] == 981)


# ---------------- 6. R2 ----------------

def r2_spec(verdicts, comp):
    """15.R / 14.R6 / F3, written independently: among competent models other than Llama-3.1."""
    others = [t for t in comp if comp[t] and t != "llama3.1-8b"]
    if any(verdicts[t] == "confirmed below zero" for t in others):
        return False
    if any(verdicts[t] in ("equivalent to zero", "confirmed above zero") for t in others):
        return True
    return "undecided"


def audit_r2() -> None:
    tags = list(R.TAGS)
    vs = ("confirmed below zero", "equivalent to zero", "confirmed above zero", "inconclusive", "not applicable")
    n = bad = 0
    for vv in itertools.product(vs, repeat=5):
        for cc in itertools.product((True, False, None), repeat=5):
            verdicts = {t: {"parent_penalty": v} for t, v in zip(tags, vv)}
            comp = dict(zip(tags, cc))
            got = A.across(verdicts, comp)
            want = r2_spec({t: v for t, v in zip(tags, vv)}, comp)
            n += 1
            if got["F3_penalty_not_shared"] != want or got["R2_penalty_not_shared"]["fires"] != (want is True):
                bad += 1
    check("R2 three-way logic matches the spec on all verdict x competence combinations", bad == 0, f"{n} cases, {bad} mismatches")


# ---------------- 7. R3 ----------------

def audit_r3() -> None:
    sides = {1: (0.2, 1.0), -1: (-1.0, -0.2), 0: (-0.2, 0.2), "nan": (float("nan"), float("nan"))}
    n = bad = 0
    for p in ("confirmed below zero", "confirmed above zero", "equivalent to zero", "inconclusive"):
        for rg in (1, -1, 0):
            for m in (1, -1, 0, "nan"):
                for cp, cn, cc in itertools.product((True, False), repeat=3):
                    v = {"parent_penalty": p, "timing": "inconclusive"}
                    comp = {"parent_penalty": cp, "naming": cn, "completion": cc, "timing": True}
                    block = {"contrasts": {"role_gap": {"ci95": list(sides[rg]), "matched": {"ci95": list(sides[m])}}}}
                    r3 = A.rejections_for_model(v, comp, block)["R3_penalty_and_role_gap_disagree"]
                    psign = {"confirmed below zero": -1, "confirmed above zero": 1}.get(p, 0)
                    want = bool(cp and cn and cc and psign != 0 and rg != 0 and psign != rg and m == rg)
                    n += 1
                    if r3["fires"] != want or r3["kind"] != A.DIAGNOSTIC:
                        bad += 1
    check("R3 fires iff competent for P, Dn, Dc; P confirmed; unmatched gap CI on the other side; matched CI on the same side as the unmatched",
          bad == 0, f"{n} cases, {bad} mismatches")
    # boundary: an interval touching zero is not "on a side"
    block = {"contrasts": {"role_gap": {"ci95": [0.0, 1.0], "matched": {"ci95": [0.1, 1.0]}}}}
    r3 = A.rejections_for_model({"parent_penalty": "confirmed below zero", "timing": "x"},
                                {"parent_penalty": True, "naming": True, "completion": True, "timing": True}, block)
    check("R3: an unmatched interval with lower bound exactly 0 does not count as above zero", not r3["R3_penalty_and_role_gap_disagree"]["fires"])


# ---------------- 8. no p-value outside the family ----------------

def audit_no_p(out) -> None:
    found = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if (k.startswith("p_") and k != "p_method") or k in ("p", "pvalue", "p_value"):
                    found.append(path + "/" + k)
                walk(v, path + "/" + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk({k: v for k, v in out.items() if k != "family"}, "")
    allowed = {f"/per_model/{t}/contrasts/{c}/p_{x}" for t in R.TAGS for c in A.CONTRASTS for x in A.TESTS}
    extra = sorted(set(found) - allowed)
    missing = allowed - set(found)
    check("the only p-values in the output (outside `family`) are the 50 family tests", not extra and not missing,
          f"extra={extra[:5]}, missing={len(missing)}")
    check("family has 50 entries and verdicts exist", len(out["family"]) == 50 and isinstance(out["verdicts"], dict))
    r4 = out["across_models"]["R4_lift_comparison"]
    check("R4 output has no 'fires' and no p; it is labelled descriptive",
          "fires" not in r4 and r4["kind"].startswith("descriptive")
          and not any(k.startswith("p_") for k in (r4.get("difference_llama_minus_qwen") or {})))
    for t in R.TAGS:
        r3 = out["verdicts"][t]["rejections"]["R3_penalty_and_role_gap_disagree"]
        if r3["kind"] != A.DIAGNOSTIC:
            check("R3 labelled a diagnostic in every model's output", False)
            return
    check("R3 labelled a diagnostic in every model's output", True)
    # lift difference uses the 990 ingredient failures
    d = r4["difference_llama_minus_qwen"]
    check("R4 lift difference on the 990 ingredient failures, 162 pairs", d["n_items"] == 990 and d["n_pairs"] == 162,
          f"{d['n_items']} items, {d['n_pairs']} pairs")


def audit_hashes(records, shots) -> None:
    h = hashlib.sha256("".join(R.item_hash(r, shots) for r in records).encode()).hexdigest()
    q = hashlib.sha256(R.YN_QUESTION.encode()).hexdigest()
    print(f"INFO item-hash digest over {len(records)} records: {h}")
    print(f"INFO sha256(R.YN_QUESTION): {q}")
    pops = A.populations(records)
    sizes = {k: len(v) for k, v in pops.items()}
    print("INFO population sizes:", json.dumps(sizes))
    check("population sizes as designed (1500 main failures, 600 successes, 675 parent failures, 207 parent successes, 990 ingredient, 500+6 goal)",
          sizes["main_failures"] == 1500 and sizes["successes"] == 600 and sizes["parent_failures"] == 675
          and sizes["parent_successes"] == 207 and sizes["ingredient"] == 990 and sizes["ingredient_matched"] == 496,
          f"goal={sizes['goal']}")


if __name__ == "__main__":
    audit_holm()
    audit_tost()
    recs, shots = load()
    audit_hashes(recs, shots)
    audit_margin(recs)
    audit_identity(recs)
    audit_r2()
    audit_r3()
    out = audit_clusters(recs, shots)
    audit_no_p(out)
    n_fail = sum(not ok for _n, ok, _d in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
