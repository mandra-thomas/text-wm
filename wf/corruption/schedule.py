"""Seeded per-game corruption schedules (§4.8)."""
from __future__ import annotations

import hashlib
import random
from typing import Callable


def schedule(
    game_id: str,
    seed: int,
    arm: str,
    p_arm: float,
) -> Callable[[int, bool], bool]:
    """Return (t, proposal_valid) -> corrupt_now, deterministic in (game_id, seed).

    Eligibility: favorable corrupts only invalid proposals; unfavorable only valid.
    """

    def _rng_at(t: int) -> random.Random:
        h = hashlib.sha256(f"{game_id}|{seed}|{arm}|{t}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def corrupt_now(t: int, proposal_valid: bool) -> bool:
        if arm == "favorable" and proposal_valid:
            return False
        if arm == "unfavorable" and not proposal_valid:
            return False
        if arm not in ("favorable", "unfavorable"):
            return False
        return _rng_at(t).random() < p_arm

    return corrupt_now


def calibrate_p_arm(
    eligibility_rate: float,
    target_corruptions: float,
    expected_steps: float,
) -> float:
    """p_arm so expected realized corruptions ≈ target.

    E[corruptions] ≈ eligibility_rate * expected_steps * p_arm
    """
    denom = max(eligibility_rate * expected_steps, 1e-6)
    p = target_corruptions / denom
    return float(min(max(p, 0.0), 1.0))
