"""The world, its generators and histories, and the exact posterior (notes/bayesenv_design.md, sections 2 and 3).

World. Six boxes numbered 1 to 6; the queried action is always "take the key from box 3". Each box is locked with
probability 0.2, independently. A generator (pi_other, pi_named) hides every object that the task line or the house
tag refers to with probability pi_named, and every other object with probability pi_other; a hidden object goes into
one of the six boxes, chosen uniformly. The take succeeds if and only if the key is in box 3 and box 3 is unlocked.

Evidence (conditionally independent given the state): looks into other boxes, each reported empty, missing a present
object with a stated rate; a close look into box 3 that never misses (reported empty); a lock check of box 3 reported
"locked", wrong with a stated rate; a glance at box 3 that suggested a key (right 90% of the time when a key is there,
wrongly 10% of the time otherwise); a certain statement that a key is in the house; a caretaker's note "There is a key
in this house." whose writer looks with probability rho and otherwise writes either sentence by a fair coin; and who
chose the queried action (a tester at random, a helper who saw only the history, or a guide who sees inside the
boxes). Every actor that chose an earlier action is a tester at random, so earlier choices carry no information.

The posterior is computed by enumerating the key's location (none, or one of six boxes) and all 64 lock patterns.
`simulate` draws the stated story forward and conditions by rejection; tests compare the two.
"""
from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

K = 6                      # boxes
TARGET = 3                 # the queried box (1-based)
LOCK_P = 0.2               # each box locked independently
KAPPA = 0.9                # the seeing guide's competence
N_TAKE_ACTIONS = 12        # take the key or the coin, from any of six boxes
MISS_NOISY = 0.3           # a noisy look misses a present object
LOCK_REPORT_ERROR = 0.02   # a lock check is wrong
GLIMPSE_HIT, GLIMPSE_FALSE = 0.9, 0.1

# (pi_other, pi_named): the hiding probability of an object the task line or tag does not / does refer to.
GENERATORS: dict[str, tuple[float, float]] = {
    "A": (0.5, 0.5),       # likelihood ratio 1: the cue carries no information
    "R3": (0.5, 0.75),     # likelihood ratio 3
    "C": (0.5, 0.9),       # likelihood ratio 9
    "D": (0.2, 0.907),     # likelihood ratio 39.0 (0.907/0.093 over 0.2/0.8), lower base rate
}
GENERATOR_ORDER = ("A", "R3", "C", "D")
RHOS = (0.0, 0.5, 0.8, 0.95)


def likelihood_ratio(gen: str) -> float:
    lo, hi = GENERATORS[gen]
    return (hi / (1 - hi)) / (lo / (1 - lo))


def note_ratio(rho: float) -> float:
    """Likelihood ratio on existence of a note written by a source that checks with probability rho."""
    return (1 + rho) / (1 - rho)


@dataclass(frozen=True)
class Evidence:
    """What the prompt states, as it bears on the key. `looks` counts other boxes looked into (all reported empty)."""
    looks: int = 0
    look_miss: float = 0.0
    close_look_target: bool = False     # box 3 looked into closely and reported empty (never misses)
    lock_report: float | None = None    # box 3 reported locked, by a check wrong with this probability
    glimpse: bool = False               # a glance at box 3 suggested a key
    exists: bool = False                # it is known for certain that a key is somewhere in the house
    note: float | None = None           # caretaker's affirming note, reliability rho
    provenance: str = "P0"              # P0 tester; P1 history-only helper; P2 seeing guide (key task only)
    lock_p: float = LOCK_P              # 0 for the ALFWorld-surface rendering (drawers have no locks)
    outcome: str = "take"               # "take": key in box 3 and box 3 unlocked; "in_target": key in box 3


# The seven histories of the reduced design (codes kept from the planning documents; H1, H5 and H8 are not used).
HISTORIES: dict[str, Evidence] = {
    "H0": Evidence(),                                                              # nothing looked at yet
    "H2": Evidence(looks=4, look_miss=MISS_NOISY),                                 # four other boxes, noisy looks
    "H3": Evidence(looks=4, look_miss=0.0),                                        # four other boxes, sure looks
    "H4": Evidence(looks=2, look_miss=0.0, exists=True),                           # existence settled, outcome open
    "H6": Evidence(looks=2, look_miss=MISS_NOISY, lock_report=LOCK_REPORT_ERROR),  # box 3 reported locked (2% error)
    "H7": Evidence(looks=2, look_miss=MISS_NOISY, close_look_target=True),         # box 3 seen empty: failure settled
    "H9": Evidence(glimpse=True),                                                  # a glance suggested a key in box 3
}
HISTORY_ORDER = ("H0", "H2", "H3", "H4", "H6", "H7", "H9")
GRADED = ("H0", "H2", "H3", "H4", "H6", "H9")   # exact success probability strictly between 0 and 1
SETTLED = ("H7",)                               # exact success probability 0


def prior(gen: str, names_key: bool, announced: bool = False) -> float:
    """Probability that a key is hidden. Observational: set by whether the shown task line or tag refers to the key.
    Announced replacement: the shown task carries no information, and the house was set up for a task drawn uniformly
    from find the key, find the coin and explore the house, with no tag."""
    lo, hi = GENERATORS[gen]
    if announced:
        return (hi + 2 * lo) / 3
    return hi if names_key else lo


_LOCKS = np.array(list(itertools.product((0, 1), repeat=K)), dtype=np.int8)   # (64, 6), 1 = locked


@functools.lru_cache(maxsize=None)
def posterior(pi: float, ev: Evidence) -> float:
    """Exact probability of the outcome given the evidence, by enumeration (key location x lock pattern)."""
    if ev.provenance == "P1" and (ev.close_look_target or ev.glimpse):
        raise ValueError("the history-only helper never picks a box it has looked into; no reference exists")
    if ev.provenance not in ("P0", "P1", "P2"):
        raise ValueError(f"unknown provenance {ev.provenance!r}")
    locks = _LOCKS if ev.lock_p > 0 else _LOCKS[:1]
    p_locks = np.prod(np.where(locks == 1, ev.lock_p, 1 - ev.lock_p), axis=1)               # (L,)
    loc = np.arange(-1, K)                                                                 # -1 none, 0..5 = box 1..6
    p_loc = np.where(loc < 0, 1 - pi, pi / K)
    w = p_loc[:, None] * p_locks[None, :]                                                  # (7, L)
    t = TARGET - 1
    in_t = (loc == t)[:, None]
    has = (loc >= 0)[:, None]
    locked_t = (locks[:, t] == 1)[None, :]
    # looks into `looks` other boxes, each reported empty; which boxes does not matter by symmetry (they exclude 3)
    others = [b for b in range(K) if b != t][: ev.looks]
    for b in others:
        w = w * np.where((loc == b)[:, None], ev.look_miss, 1.0)
    if ev.close_look_target:
        w = w * np.where(in_t, 0.0, 1.0)
    if ev.lock_report is not None:
        w = w * np.where(locked_t, 1 - ev.lock_report, ev.lock_report)
    if ev.glimpse:
        w = w * np.where(in_t, GLIMPSE_HIT, GLIMPSE_FALSE)
    if ev.exists:
        w = w * np.where(has, 1.0, 0.0)
    if ev.note is not None:
        rho = ev.note
        w = w * np.where(has, rho + (1 - rho) / 2, (1 - rho) / 2)
    if ev.provenance == "P2":
        accessible_t = in_t & ~locked_t
        accessible_any = np.zeros_like(w, dtype=bool)
        for b in range(K):
            accessible_any |= (loc == b)[:, None] & (locks[:, b] == 0)[None, :]
        w = w * (KAPPA * accessible_t + (1 - KAPPA * accessible_any) / N_TAKE_ACTIONS)
    # P0 and P1: the action's probability does not depend on the state given the history, so it cancels
    succ = in_t & ~locked_t if ev.outcome == "take" else np.broadcast_to(in_t, w.shape)
    z = w.sum()
    if z <= 0:
        raise ValueError("the stated evidence has probability zero")
    return float((w * succ).sum() / z)


def logit(p: float) -> float:
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    return math.log(p) - math.log1p(-p)


@functools.lru_cache(maxsize=None)
def exact(gen: str, hist: str, names_key: bool, note: float | None = None, provenance: str = "P0",
          announced: bool = False) -> float:
    """The exact success probability of one prompt of the toy design."""
    ev = replace(HISTORIES[hist], note=note, provenance=provenance)
    return posterior(prior(gen, names_key, announced), ev)


# ----------------------------------------------------------------------------- forward simulation of the story


def simulate(pi: float, ev: Evidence, n: int, rng: np.random.Generator) -> tuple[float, int]:
    """Draw the stated story forward n times and keep the draws whose observations match the prompt.
    Returns (share of kept draws in which the outcome holds, number kept). Written from the story, not from
    `posterior`, so that the two check each other."""
    t = TARGET - 1
    has = rng.random(n) < pi
    box = rng.integers(0, K, n)
    loc = np.where(has, box, -1)
    locked = rng.random((n, K)) < ev.lock_p
    keep = np.ones(n, dtype=bool)
    others = [b for b in range(K) if b != t][: ev.looks]
    for b in others:                       # each look reports "empty": the key is not there, or it was missed
        missed = rng.random(n) < ev.look_miss
        keep &= (loc != b) | missed
    if ev.close_look_target:
        keep &= loc != t
    if ev.lock_report is not None:         # the check reports "locked"
        wrong = rng.random(n) < ev.lock_report
        reported_locked = np.where(wrong, ~locked[:, t], locked[:, t])
        keep &= reported_locked
    if ev.glimpse:                         # the glance suggests a key
        u = rng.random(n)
        keep &= np.where(loc == t, u < GLIMPSE_HIT, u < GLIMPSE_FALSE)
    if ev.exists:
        keep &= has
    if ev.note is not None:                # the note says "there is a key"
        checks = rng.random(n) < ev.note
        coin = rng.random(n) < 0.5
        keep &= np.where(checks, has, coin)
    if ev.provenance == "P2":              # the guide's choice is "take the key from box 3"
        accessible = (loc >= 0) & ~locked[np.arange(n), np.clip(loc, 0, K - 1)]
        pursue = accessible & (rng.random(n) < KAPPA)
        chosen_box = np.where(pursue, loc, -2)
        rand_action = rng.integers(0, N_TAKE_ACTIONS, n)    # 0..5 key from box 1..6, 6..11 coin
        picks_it = np.where(pursue, chosen_box == t, rand_action == t)
        keep &= picks_it
    elif ev.provenance == "P1":            # the helper picks uniformly among unlooked boxes: independent of the state
        unlooked = [b for b in range(K) if b not in others]
        pick = rng.integers(0, len(unlooked), n)
        keep &= np.array(unlooked)[pick] == t
    if ev.outcome == "take":
        success = (loc == t) & ~locked[:, t]
    else:
        success = loc == t
    kept = int(keep.sum())
    return (float(success[keep].mean()) if kept else float("nan")), kept
