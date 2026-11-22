"""Exploratory, post hoc diagnostics of the TextCraft Llama result (notes/textcraft_llama_dx.md).

Nothing here is part of the frozen TextCraft analysis (wf/textcraft/analysis.py). It reads the scoring records of
`textcraft_score` (and, when given, the exploratory prompt variants) next to the verified items, and estimates
per-item differences with a percentile bootstrap that resamples whole goal pairs, as the frozen analysis does. No
p-values: every number is descriptive.

Usage:
    python scripts/textcraft_llama_dx.py <scores dir> [<variant name>=<variant dir> ...] > out.json

<scores dir> is searched recursively for scoring.jsonl files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ITEMS = REPO / "wf" / "textcraft" / "data" / "items.jsonl"
MODELS = ("llama3.1-8b-fewshot", "qwen3-8b-fewshot")
SEED = 20260924
B = 2000


def read_scores(root: Path) -> list[dict]:
    out = []
    for p in sorted(Path(root).rglob("scoring.jsonl")):
        out.extend(json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


def index(scores: Iterable[dict]) -> dict[str, dict[tuple[str, str], dict]]:
    """model -> (item id, condition) -> score record."""
    by: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for s in scores:
        by[s["model"]][(s["transition_id"], s["condition"])] = s
    return by


def pair_boot(values: list[float], pairs: list[str], b: int = B, seed: int = SEED) -> dict:
    """Mean of values with a 95% percentile interval from resampling whole goal pairs (vectorized: each resample
    weights each pair by how often it was drawn)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"est": float("nan"), "ci": [float("nan"), float("nan")], "n": 0, "pairs": 0}
    keys = sorted(set(pairs))
    k = {p: i for i, p in enumerate(keys)}
    idx = np.array([k[p] for p in pairs])
    sums = np.bincount(idx, weights=v, minlength=len(keys))
    cnts = np.bincount(idx, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(keys), [1.0 / len(keys)] * len(keys), size=b)
    boots = (w @ sums) / (w @ cnts)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"est": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(v.size), "pairs": len(keys)}


def diff_boot(by: dict, items: list[dict], f: Callable[[dict, dict], float | None]) -> dict:
    """pair_boot of f(scores of the item by condition, item) over items where f is defined."""
    vals, pairs = [], []
    for t in items:
        d = f(_view(by, t["id"]), t)
        if d is not None and d == d:
            vals.append(d)
            pairs.append(t["pair_id"])
    return pair_boot(vals, pairs)


class _view(dict):
    """The scores of one item, keyed by condition (lazy lookup into the model's index)."""
    def __init__(self, by: dict, tid: str):
        super().__init__()
        self._by, self._tid = by, tid

    def __missing__(self, cond):
        return self._by.get((self._tid, cond))

    def get(self, cond, default=None):
        s = self._by.get((self._tid, cond))
        return default if s is None else s


def field_diff(a: str, b: str, field: str = "L") -> Callable[[dict, dict], float | None]:
    """f for diff_boot: field under condition a minus field under condition b."""
    def f(v, _t):
        sa, sb = v.get(a), v.get(b)
        return None if sa is None or sb is None else sa[field] - sb[field]
    return f


def level(a: str, field: str = "L") -> Callable[[dict, dict], float | None]:
    def f(v, _t):
        s = v.get(a)
        return None if s is None else s[field]
    return f


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def load_items(path: Path = ITEMS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def leave_one_pair_out(values: list[float], pairs: list[str]) -> dict:
    """The mean with each goal pair left out in turn: its range, and the pairs whose removal moves it most."""
    v = np.asarray(values, float)
    keys = sorted(set(pairs))
    pa = np.asarray(pairs)
    out = {}
    for k in keys:
        m = pa != k
        out[k] = float(v[m].mean())
    s = sorted(out.items(), key=lambda kv: kv[1])
    return {"min": s[0], "max": s[-1], "full": float(v.mean())}


if __name__ == "__main__":  # pragma: no cover - the exploratory driver lives in the note's scratch
    print(__doc__)
