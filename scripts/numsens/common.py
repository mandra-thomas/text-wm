"""Shared helpers for the numerical-sensitivity analysis (notes/numerical_sensitivity.md).

All inputs are copies of archived run folders from ICE (read-only copies; see fetch.sh). Nothing here scores a model.
Set NUMSENS_DATA to the folder that holds the copied `text-wm-fix*` trees (default: ./numsens_data).
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("NUMSENS_DATA", "numsens_data"))
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261125)
EXACT = 1e-6  # two scores count as identical when they differ by less than this (all identical pairs are bit-equal)


# ---------------------------------------------------------------- loading

def load_scores(run_glob: str, shard_gpu: dict[int, str] | None = None, model_filter: str | None = None,
                default_gpu: str | None = None) -> dict[tuple[str, str], dict]:
    """(transition_id, condition) -> record, with the record's shard index and GPU type added.
    shard_gpu maps shard index -> GPU type for runs whose job logs (not jobs.jsonl) record the GPU."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted(glob.glob(str(DATA / run_glob))):
        shard = None
        for part in Path(f).parts:
            if part.startswith("shard_"):
                shard = int(part.split("_")[1])
        gpu = default_gpu
        jobs = Path(f).parent / "jobs.jsonl"
        if jobs.exists():
            gpus = set()
            for line in jobs.read_text().splitlines():
                g = json.loads(line).get("gpu")
                if isinstance(g, list):
                    gpus.update(g)
                elif g:
                    gpus.add(g)
            if gpus:
                gpu = "+".join(sorted(gpus))
        if shard_gpu is not None and shard in shard_gpu:
            gpu = shard_gpu[shard]
        for line in open(f):
            r = json.loads(line)
            if model_filter and not r.get("model", "").startswith(model_filter):
                continue
            r["_shard"], r["_gpu"] = shard, short_gpu(gpu)
            out[(r["transition_id"], r["condition"])] = r
    return out


def short_gpu(g: str | None) -> str | None:
    if g is None:
        return None
    g = g.replace("NVIDIA ", "")
    if "A100" in g and "80" in g:
        return "A100-80GB"
    if "A100" in g:
        return "A100-40GB"
    return g.strip()


def transitions_meta() -> dict[str, dict]:
    """transition id -> history length (steps) and characters of history text (prompt-length proxy) for ALFWorld."""
    meta = {}
    for f in sorted(glob.glob(str(DATA / "trans" / "s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            meta[r["id"]] = {"history_len": int(r.get("history_len") or 0),
                             "history_chars": len(r.get("history_text") or "") + len(r.get("hidden_state") or ""),
                             "game_id": r["game_id"]}
    return meta


# ---------------------------------------------------------------- statistics

def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 4 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3.0)


def describe(d: np.ndarray) -> dict:
    """Per-score (or per-contrast) differences between two environments."""
    d = np.asarray(d, float)
    a = np.abs(d)
    n = len(d)
    if n == 0:
        return {"n": 0}
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    mad_sd = float(1.4826 * np.median(np.abs(d - np.median(d))))
    return {
        "n": n,
        "share_identical": float((a < EXACT).mean()),
        "mean_signed": float(d.mean()),
        "mean_abs": float(a.mean()),
        "sd": sd,
        "robust_sd_mad": mad_sd,
        "p50_abs": float(np.percentile(a, 50)),
        "p90_abs": float(np.percentile(a, 90)),
        "p99_abs": float(np.percentile(a, 99)),
        "max_abs": float(a.max()),
        "share_abs_gt_0.125": float((a > 0.125 + 1e-6).mean()),
        "share_abs_gt_0.5": float((a > 0.5 + 1e-6).mean()),
        "share_abs_gt_1": float((a > 1.0 + 1e-6).mean()),
        "share_beyond_3sd": float((a > 3 * sd).mean()) if sd and sd > 0 else float("nan"),
        "excess_kurtosis": excess_kurtosis(d),
        # nonzero part only: tail shape of the differences that are not exact repeats
        "nonzero": None if (a >= EXACT).sum() < 4 else {
            "n": int((a >= EXACT).sum()), "sd": float(d[a >= EXACT].std(ddof=1)),
            "excess_kurtosis": excess_kurtosis(d[a >= EXACT])},
    }


def cluster_boot_mean(values: np.ndarray, clusters: list[str], B: int = 4000) -> dict:
    """Game-clustered bootstrap of a mean: estimate, SE, 95% percentile interval, design effect."""
    values = np.asarray(values, float)
    groups = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    keys = list(groups)
    sums = np.array([np.sum(groups[k]) for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    idx = RNG.integers(0, len(keys), size=(B, len(keys)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    se = float(means.std(ddof=1))
    naive = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
    return {"estimate": float(values.mean()), "se": se, "ci95": [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))], "n": int(len(values)), "n_clusters": len(keys),
            "naive_se": naive, "design_effect": (se / naive) ** 2 if naive and naive > 0 else float("nan")}


def independence(dA: np.ndarray, dB: np.ndarray) -> dict:
    """Do the environment differences of the two arms of a contrast cancel, add, or anticorrelate?
    ratio = Var(dA - dB) / (Var dA + Var dB): 1 if independent, below 1 if they cancel, above 1 if they add."""
    dA, dB = np.asarray(dA, float), np.asarray(dB, float)
    vA, vB = dA.var(ddof=1), dB.var(ddof=1)
    vC = (dA - dB).var(ddof=1)
    cov = float(np.cov(dA, dB)[0, 1])
    corr = cov / math.sqrt(vA * vB) if vA > 0 and vB > 0 else float("nan")
    # also on the subset where at least one arm changed (exact repeats dilute the correlation)
    m = (np.abs(dA) >= EXACT) | (np.abs(dB) >= EXACT)
    corr_nz = float(np.corrcoef(dA[m], dB[m])[0, 1]) if m.sum() > 3 and dA[m].std() > 0 and dB[m].std() > 0 else float("nan")
    both_zero = float(((np.abs(dA) < EXACT) & (np.abs(dB) < EXACT)).mean())
    one_zero = float(((np.abs(dA) < EXACT) ^ (np.abs(dB) < EXACT)).mean())
    return {"var_arm_a": float(vA), "var_arm_b": float(vB), "var_contrast": float(vC), "cov": cov, "corr": corr,
            "corr_where_any_changed": corr_nz,
            "ratio_var_contrast_to_sum": float(vC / (vA + vB)) if vA + vB > 0 else float("nan"),
            "share_both_arms_identical": both_zero, "share_exactly_one_arm_identical": one_zero}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def binned(abs_d: np.ndarray, x: np.ndarray, edges: list[float]) -> list[dict]:
    """Mean |difference| and share identical within bins of x."""
    out = []
    abs_d, x = np.asarray(abs_d, float), np.asarray(x, float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        out.append({"bin": [lo, hi], "n": int(m.sum()), "mean_abs": float(abs_d[m].mean()),
                    "share_identical": float((abs_d[m] < EXACT).mean()),
                    "p90_abs": float(np.percentile(abs_d[m], 90))})
    return out


def grid_of(d: np.ndarray) -> dict:
    """Share of nonzero differences that are whole multiples of 1/8 and of 1/4 nat (bf16 logit spacing)."""
    d = np.asarray(d, float)
    nz = d[np.abs(d) >= EXACT]
    if len(nz) == 0:
        return {"n_nonzero": 0}
    def mult(g):
        return float((np.abs(nz / g - np.round(nz / g)) < 1e-3).mean())
    return {"n_nonzero": int(len(nz)), "share_multiple_of_1_8": mult(0.125), "share_multiple_of_1_4": mult(0.25),
            "share_multiple_of_1_64": mult(1 / 64)}


def dump(name: str, obj) -> Path:
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    return p
