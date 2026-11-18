"""Are the RQ2 statistics calibrated? Simulate item-level ΔL with the real data's clustering (2026-11-17).

Each replicate draws 134 games with a pilot-like number of items per game, a game effect and an item
effect whose variances match a pilot model's ICC and SD, and a true mean of zero (or a planted effect).
It then runs the code the confirmatory analysis uses: `pair_permutation` with game clusters, and
`cluster_bootstrap` (percentile, games resampled). Under the null the permutation test should reject
at about its nominal rate and the 95% interval should cover the true mean about 95% of the time.

    python scripts/calibration_check.py --reps 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wf.analysis.stats import cluster_bootstrap, pair_permutation  # noqa: E402

# From Pilot A (history only, primary items): ICC and per-item SD of ΔL.
SCENARIOS = {
    "qwen3_null": {"icc": 0.30, "sd": 6.66, "effect": 0.0, "tails": "normal"},
    "llama_null": {"icc": 0.19, "sd": 1.90, "effect": 0.0, "tails": "normal"},
    "qwen3_null_heavy_tails": {"icc": 0.30, "sd": 6.66, "effect": 0.0, "tails": "t3"},
    "llama_effect_0.5": {"icc": 0.19, "sd": 1.90, "effect": 0.5, "tails": "normal"},
}


def _noise(rng: random.Random, sd: float, tails: str) -> float:
    if tails == "t3":  # Student t with 3 df, scaled to the same SD
        z = rng.gauss(0, 1)
        chi = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
        return z / math.sqrt(chi / 3) * sd / math.sqrt(3.0)
    return rng.gauss(0, sd)


def replicate(rng: random.Random, icc: float, sd: float, effect: float, tails: str, games: int = 134):
    tau, sigma = math.sqrt(icc) * sd, math.sqrt(1 - icc) * sd
    diffs, ids = [], []
    for g in range(games):
        n = max(1, int(rng.expovariate(1 / 20)))  # skewed items per game, mean about 20
        u = _noise(rng, tau, tails)
        for _ in range(n):
            diffs.append(effect + u + _noise(rng, sigma, tails))
            ids.append(f"g{g}")
    return diffs, ids


def run(reps: int, seed: int) -> dict:
    out = {}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for name, sc in SCENARIOS.items():
        rng = random.Random(f"{seed}|{name}")
        p01 = p05 = cover = 0
        for r in range(reps):
            diffs, ids = replicate(rng, sc["icc"], sc["sd"], sc["effect"], sc["tails"])
            p = pair_permutation(diffs, seed=r, cluster_ids=ids)
            ci = cluster_bootstrap(mean, diffs, ids, seed=r)
            p01 += p < 0.01
            p05 += p < 0.05
            cover += ci["ci_low"] <= sc["effect"] <= ci["ci_high"]
        se = lambda k: math.sqrt(max(k, 1) * (reps - k) / reps) / reps
        out[name] = {"reps": reps, "reject_at_0.01": p01 / reps, "reject_at_0.05": p05 / reps,
                     "ci95_coverage": cover / reps, "se_0.05": se(p05), "se_coverage": se(cover)}
        print(name, json.dumps(out[name]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    result = run(args.reps, args.seed)
    nulls = {k: v for k, v in result.items() if k.endswith("null") or "null" in k}
    ok = all(abs(v["reject_at_0.05"] - 0.05) <= 3 * v["se_0.05"] + 0.01 and abs(v["ci95_coverage"] - 0.95)
             <= 3 * v["se_coverage"] + 0.01 for v in nulls.values())
    print("CALIBRATED" if ok else "NOT CALIBRATED", flush=True)
    sys.exit(0 if ok else 1)
