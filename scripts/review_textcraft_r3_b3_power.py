"""Reviewer (TextCraft round 3, review 1): rough power of family B3 on the real record and goal-pair structure, with synthetic item values only.

Each model's item values are pair effect + item noise (intraclass correlation ICC), the two models correlated rho on
both levels, true standardized difference zero. The bootstrap SE of the paired standardized difference is read from
r3_analysis.paired_difference (2,000 resamples); equivalence power and the minimum detectable difference follow.
"""
import math
import random
from pathlib import Path

from wf.textcraft import r2_analysis as A
from wf.textcraft import r3_analysis as Z
from wf.textcraft.r2_score import load_records

recs = load_records(Path("wf/textcraft/data/r2_items.jsonl"))
pops = A.populations(recs)
rows = {"lift": [(None, r) for r in pops["ingredient"]],
        "parent_penalty": [(None, r) for r in pops["parent_failures"]],
        "naming / completion": [(r["role"], r) for r in A.two_item_rows(pops)],
        "timing": [(r["role"], r) for r in pops["padded_goal"] + pops["padded_ingredient_matched"]]}
Z_FIRST = 2.935199  # one-sided normal quantile at 0.05 / 30
Z_LAST = 1.644854
Z_DIFF_FIRST = 3.143908  # two-sided at 0.05 / 30
for icc in (0.10,):
    for rho in (0.0, 0.5, 0.8):
        for name, rs in rows.items():
            rng = random.Random(7)
            gap = name not in ("lift", "parent_penalty")
            pairs = sorted({r["pair_id"] for _ro, r in rs})
            ses = []
            for rep in range(3):
                uo = {p: rng.gauss(0, 1) for p in pairs}
                un = {p: rho * uo[p] + math.sqrt(1 - rho * rho) * rng.gauss(0, 1) for p in pairs}
                old, new = {}, {}
                for role, r in rs:
                    eo = rng.gauss(0, 1)
                    en = rho * eo + math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
                    old[r["id"]] = (role, math.sqrt(icc) * uo[r["pair_id"]] + math.sqrt(1 - icc) * eo, r["pair_id"])
                    new[r["id"]] = (role, math.sqrt(icc) * un[r["pair_id"]] + math.sqrt(1 - icc) * en, r["pair_id"])
                out = Z.paired_difference(old, new, gap, B=2000, seed=rep)
                ses.append(out["standardized"]["bootstrap_se"])
            se = sum(ses) / len(ses)
            print(f"ICC {icc} rho {rho} {name:20s} n_items {len(rs):4d} pairs {len(pairs):3d} SE(dz) {se:.3f} "
                  f"equiv power first step {A.equivalence_power(se, 0.2, Z_FIRST):.2f} at 0.05 "
                  f"{A.equivalence_power(se, 0.2, Z_LAST):.2f}; reachable bound first step {max(0, 0.2 - Z_FIRST * se):.3f}; "
                  f"MDE (80%, first step) {(Z_DIFF_FIRST + 0.8416) * se:.2f}", flush=True)
