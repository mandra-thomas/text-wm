#!/usr/bin/env python3
"""Option C precision from archived RQ3 episodes only (no Option C data): how well a closed-loop comparison of two
verifier arms over the same 134 games can detect a difference in task success, and in invalid actions per episode.

Inputs: the RQ3 episodes (`text-wm-fix11/runs/rq3`, commit ddeabc2), every condition with dropped predictions (none,
perfect foresight, valid-or-not, and the six corrupted episodes per game). Nothing from Stage 1a, Stage 1b or Option C.

Model. Game g has a success propensity p_g; an episode of arm a succeeds with probability p_g (task-blind arm) or
p_g * (1 - delta / mean p) (task-aware arm, a proportional loss whose mean over games is delta). Episodes are drawn
independently given p_g, which ignores the shared prefix of the two arms (common random numbers), so the powers are
conservative in that respect, and ignores between-game variation in the true effect, which makes them optimistic in
another. Two noise scenarios for p_g:

  empirical: p_g = the game's success share over the 9 dropped RQ3 conditions (two-thirds of games always or never
             succeed; the discordance of two independent episodes of one game is 2 E[p(1-p)], about 9%);
  chaotic:   the empirical p_g shrunk toward 1/2 until that discordance is 20%, close to the 19-24% seen between RQ3's
             no-foresight and perfect-foresight episodes (which also carry a real effect).

S seeds per (game, arm) are averaged within the game. A comparison "detects" when the whole-game sign-flip p (1,000
flips) is below 0.05 and the 95% game-bootstrap interval (1,000 resamples) excludes zero, the rule of the design.

Invalid actions per episode use a normal approximation from the paired per-game standard deviation of the difference
in invalid actions between two RQ3 conditions that diverge in most games (no foresight against perfect foresight:
about 5 per episode); seeds are assumed to cut the within-game part of the variance by S, with half of the variance
between games (an assumption, stated in the output).

Usage: python scripts/optionc_precision.py /path/to/text-wm-fix11/runs/rq3 [reps]
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict

import numpy as np

SEED = 20260924
N_PERM = 1000
N_BOOT = 1000


def load(root: str) -> tuple[dict, dict]:
    eps = [json.loads(line) for f in sorted(glob.glob(f"{root}/shard_*/episodes.jsonl")) for line in open(f)]
    by: dict[tuple, dict] = defaultdict(dict)
    for e in eps:
        by[(e["condition"], e.get("arm"), e["retention"], e.get("schedule_seed"))][e["game_id"]] = e
    return by, {k: v for k, v in by.items() if k[2] == "discarded"}


def propensities(dropped: dict) -> np.ndarray:
    per: dict[str, list[bool]] = defaultdict(list)
    for v in dropped.values():
        for g, e in v.items():
            per[g].append(bool(e["success"]))
    return np.array([st.mean(per[g]) for g in sorted(per)], dtype=float)


def shrink_to(p: np.ndarray, discordance: float) -> np.ndarray:
    """p shrunk toward 1/2 until 2 E[p(1-p)] equals the target discordance (bisection on the shrink factor)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2
        q = 0.5 + lam * (p - 0.5)
        if 2 * np.mean(q * (1 - q)) > discordance:
            lo = lam
        else:
            hi = lam
    return 0.5 + lo * (p - 0.5)


def detects(d: np.ndarray, rng: np.random.Generator) -> bool:
    n = len(d)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    obs = abs(d.mean())
    p = (1 + np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12)) / (N_PERM + 1)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return p < 0.05 and (lo > 0 or hi < 0)


def power(p: np.ndarray, delta: float, seeds: int, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    q = np.clip(p * (1 - delta / p.mean()), 0, 1)
    hits, widths = 0, []
    for _ in range(reps):
        a = rng.random((seeds, len(p))) < p
        b = rng.random((seeds, len(p))) < q
        d = a.mean(axis=0) - b.mean(axis=0)
        hits += detects(d, rng)
        widths.append(1.96 * d.std(ddof=1) / math.sqrt(len(d)))
    return hits / reps, float(np.mean(widths))


def invalid_sd(by: dict) -> float:
    a, b = by[("none", None, "discarded", None)], by[("oracle", None, "discarded", None)]
    inv = lambda e: sum(s["real_obs"].strip() == "Nothing happens." for s in e["trace"])
    return float(np.std([inv(a[g]) - inv(b[g]) for g in sorted(a)]))


def main() -> None:
    root = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rng = np.random.default_rng(SEED)
    by, dropped = load(root)
    p_emp = propensities(dropped)
    scen = {"empirical": p_emp, "chaotic": shrink_to(p_emp, 0.20)}
    print(f"games {len(p_emp)}; dropped conditions {len(dropped)}; mean p {p_emp.mean():.3f}; "
          f"always or never succeed {np.mean((p_emp == 0) | (p_emp == 1)):.2f}")
    for name, p in scen.items():
        print(f"\nscenario {name}: discordance of two independent episodes {2 * np.mean(p * (1 - p)):.3f}")
        print("delta (points) | seeds | power | mean 95% half-width (points)")
        for seeds in (1, 2, 3):
            for delta in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
                pw, hw = power(p, delta, seeds, reps, rng)
                print(f"{100 * delta:5.0f} | {seeds} | {pw:.2f} | {100 * hw:.1f}")
    sd = invalid_sd(by)
    print(f"\ninvalid actions per episode: paired SD of the per-game difference, no foresight vs perfect foresight "
          f"= {sd:.2f}")
    for seeds in (1, 2, 3):
        sd_s = sd * math.sqrt(0.5 + 0.5 / seeds)  # assumption: half the variance between games, half within
        se = sd_s / math.sqrt(len(p_emp))
        mde = (1.96 + 0.84) * se
        print(f"seeds {seeds}: SE {se:.2f}, 95% half-width {1.96 * se:.2f}, difference detectable with power 0.8 "
              f"{mde:.2f} invalid actions per episode")


if __name__ == "__main__":
    main()
