"""The task direction at the answer position and its controls (design §7)."""
from __future__ import annotations

from typing import Any


def diff_of_means(H_aligned: Any, H_swapped: Any) -> dict:
    """Unit direction of the mean paired difference, its raw norm, and the midpoint of the mean projections.

    `H_aligned`, `H_swapped`: (items, width), the same items in the same order. The midpoint is the value a mean
    ablation sets the component to.
    """
    import numpy as np

    A, S = np.asarray(H_aligned, dtype=np.float64), np.asarray(H_swapped, dtype=np.float64)
    if A.shape != S.shape:
        raise ValueError("aligned and swapped activations must be paired")
    d = (A - S).mean(axis=0)
    norm = float(np.linalg.norm(d))
    if norm == 0:
        raise ValueError("zero mean difference")
    u = d / norm
    mid = float(((A @ u).mean() + (S @ u).mean()) / 2)
    return {"u": u, "norm": norm, "midpoint": mid}


def random_units(width: int, k: int, seed: int) -> Any:
    """k random unit directions (Gaussian, normalised), fixed by `seed`."""
    import numpy as np

    rng = np.random.default_rng(seed)
    v = rng.standard_normal((k, width))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def midpoint_for(u: Any, H_a: Any, H_b: Any) -> float:
    import numpy as np

    return float(((np.asarray(H_a) @ u).mean() + (np.asarray(H_b) @ u).mean()) / 2)


def cosine(u: Any, v: Any) -> float:
    import numpy as np

    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a positive scores above a negative), ties counting one half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def removed_share(effect_before: float, effect_after: float) -> float:
    return float("nan") if effect_before == 0 else 1 - effect_after / effect_before
