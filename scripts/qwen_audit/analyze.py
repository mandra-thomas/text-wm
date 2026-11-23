"""Forensic audit of Qwen3-8B's ALFWorld goal effect from the stored scores (exploratory, CPU only).

Reads the scores copied by scripts/qwen_audit/fetch.sh (default root runs/_qwen_audit) and the collection through
wf.rq2 (RUNS_DIR must point at the folder holding rq12_transitions_s0..2). Writes one metrics JSON. No model is run.

Usage, from the repository root:
  RUNS_DIR=runs/_qwen_audit/fix9 PYTHONPATH=. python scripts/qwen_audit/analyze.py \
      --root runs/_qwen_audit --out notes/qwen_effect_audit_forensic_metrics.json

The analysis list is fixed in notes/qwen_effect_audit_forensic.md, section 0. Section letters below follow it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from scipy import stats

from wf.rq2 import load_items, primary_set
from wf.runs import _item_hash

SEED = 20261122
B = 2000


# ---------------------------------------------------------------- loading

def load_run(root: str, run: str) -> tuple[dict, dict]:
    """{(model, condition): {transition_id: record}} with each record's shard folder, and a problems report."""
    out: dict = collections.defaultdict(dict)
    dup = collections.Counter()
    dup_disagree = collections.Counter()
    nonfinite = collections.Counter()
    for f in sorted(glob.glob(os.path.join(root, run, "*", "scoring.jsonl"))):
        shard = Path(f).parent.name
        for line in open(f):
            r = json.loads(line)
            r["_shard"] = shard
            key = (r["model"], r["condition"])
            for k in ("L", "logp_plus", "logp_minus", "logp_first_plus", "logp_first_minus"):
                v = r.get(k)
                if v is None or not math.isfinite(v):
                    nonfinite[key] += 1
                    break
            prev = out[key].get(r["transition_id"])
            if prev is not None:
                dup[key] += 1
                if any(prev.get(k) != r.get(k) for k in ("L", "logp_plus", "logp_minus", "item_hash")):
                    dup_disagree[key] += 1
            out[key][r["transition_id"]] = r
    problems = {"duplicates": {f"{m}|{c}": n for (m, c), n in dup.items()},
                "duplicates_disagreeing": {f"{m}|{c}": n for (m, c), n in dup_disagree.items()},
                "nonfinite": {f"{m}|{c}": n for (m, c), n in nonfinite.items()}}
    return dict(out), problems


def task_type(task_line: str) -> str:
    t = task_line.lower()
    if " two " in t:
        return "pick_two"
    if "examine" in t or "look at" in t:
        return "look_at"
    if "heat" in t or " hot " in t:
        return "heat"
    if "cool" in t:
        return "cool"
    if "clean" in t:
        return "clean"
    return "put"


# ---------------------------------------------------------------- statistics

class Games:
    """Game-clustered bootstrap over a fixed item order."""

    def __init__(self, games: list[str]):
        self.names, self.idx = np.unique(np.asarray(games), return_inverse=True)
        self.G = len(self.names)
        rng = np.random.default_rng(SEED)
        self.W = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=B).astype(float)  # B x G
        self.count = np.bincount(self.idx, minlength=self.G).astype(float)

    def sums(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.idx, weights=x, minlength=self.G)

    def mean_ci(self, x: np.ndarray, mask: np.ndarray | None = None) -> dict:
        x = np.asarray(x, float)
        m = np.ones_like(x, bool) if mask is None else mask
        s = self.sums(np.where(m, x, 0.0)); c = self.sums(m.astype(float))
        bs = (self.W @ s) / np.maximum(self.W @ c, 1e-12)
        return {"mean": float(x[m].mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "n": int(m.sum())}

    def ratio_ci(self, num: np.ndarray, den: np.ndarray, mask: np.ndarray | None = None) -> dict:
        m = np.ones(len(num), bool) if mask is None else mask
        sn = self.sums(np.where(m, num, 0.0)); sd = self.sums(np.where(m, den, 0.0))
        bs = (self.W @ sn) / (self.W @ sd)
        return {"ratio": float(num[m].sum() / den[m].sum()), "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5))}

    def stat_ci(self, fn, *arrays, nb: int = 500) -> dict:
        """Any statistic of item arrays, bootstrapped by repeating each game's items by its draw count."""
        val = float(fn(*arrays))
        members = [np.flatnonzero(self.idx == g) for g in range(self.G)]
        bs = []
        for b in range(nb):
            take = np.concatenate([np.repeat(members[g], int(k)) for g, k in enumerate(self.W[b]) if k])
            bs.append(fn(*[a[take] for a in arrays]))
        return {"value": val, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def q(x, ps=(1, 10, 25, 50, 75, 90, 99)) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), **{f"p{p}": float(np.percentile(x, p)) for p in ps}}


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def pooled_auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b) + half the ties, pooled across items (Mann-Whitney)."""
    r = stats.rankdata(np.concatenate([a, b]))
    ra = r[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


# ---------------------------------------------------------------- per-model block

def arrays(recs_a: dict, recs_b: dict, ids: list[str]) -> dict:
    """Per-item arrays for two conditions over ids (both must be present)."""
    def col(recs, k):
        return np.array([recs[i][k] for i in ids], float)
    out = {}
    for tag, recs in (("a", recs_a), ("b", recs_b)):
        L = col(recs, "L"); lp = col(recs, "logp_plus"); lm = col(recs, "logp_minus")
        fp = col(recs, "logp_first_plus"); fm = col(recs, "logp_first_minus")
        out[tag] = {"L": L, "lp": lp, "lm": lm, "fp": fp, "fm": fm, "F": fp - fm, "Tp": lp - fp, "Tm": lm - fm,
                    "ntp": col(recs, "n_tokens_plus"), "ntm": col(recs, "n_tokens_minus")}
    return out


def two_cand_block(G: Games, A: dict, Bc: dict, N: dict | None, meta: dict, ids: list[str]) -> dict:
    """Questions B, C1, D and E for one model's two-candidate scores (A aligned, Bc swapped, N neutral or None)."""
    X = arrays(A, Bc, ids)
    a, b = X["a"], X["b"]
    d = {k: a[k] - b[k] for k in ("L", "lp", "lm", "F", "Tp", "Tm")}
    step0 = np.array([meta[i]["hl"] == 0 for i in ids])
    res: dict = {"n": len(ids)}
    # C1: decomposition
    res["effect"] = {k: G.mean_ci(d[k]) for k in ("L", "F", "Tp", "Tm", "lp", "lm")}
    res["effect_later"] = {k: G.mean_ci(d[k], ~step0) for k in ("L", "F", "Tp", "Tm", "lp", "lm")}
    res["effect_step0"] = {k: G.mean_ci(d[k], step0) for k in ("L", "F", "Tp", "Tm", "lp", "lm")}
    res["tail_share"] = G.ratio_ci(d["Tp"], d["L"])
    res["tail_share_later"] = G.ratio_ci(d["Tp"], d["L"], ~step0)
    res["share_items_lp_up"] = float((d["lp"] > 0).mean())
    res["share_items_lm_down"] = float((d["lm"] < 0).mean())
    # B1/B2: distributions and probabilities
    dist = {}
    for tag, x in (("aligned", a), ("swapped", b)):
        dist[tag] = {k: q(x[k]) for k in ("L", "lp", "lm", "F", "Tp", "Tm")}
        p_sig = sig(x["L"]); p_plus = np.exp(x["lp"]); p_you = np.exp(x["fp"]); p_nh = np.exp(x["lm"])
        dist[tag]["prob"] = {
            "sigma_L_gt": {str(t): float((p_sig > t).mean()) for t in (0.01, 0.1, 0.5)},
            "p_success_reply_gt": {str(t): float((p_plus > t).mean()) for t in (0.01, 0.1, 0.5)},
            "p_first_you_gt": {str(t): float((p_you > t).mean()) for t in (0.01, 0.1, 0.5)},
            "median_p_success_reply": float(np.median(p_plus)), "mean_p_success_reply": float(p_plus.mean()),
            "median_p_nothing": float(np.median(p_nh)), "mean_p_nothing": float(p_nh.mean()),
            "median_p_you": float(np.median(p_you)), "mean_p_you": float(p_you.mean()),
            "mean_sigma_L": float(p_sig.mean()),
            "mean_sigma_F": float(sig(x["F"]).mean()),
        }
    if N is not None:
        n = arrays(N, N, ids)["a"]
        dist["neutral"] = {k: q(n[k]) for k in ("L", "lp", "lm", "F", "Tp", "Tm")}
        res["vs_neutral"] = {
            "aligned_minus_neutral": {k: G.mean_ci(a[k] - n[k]) for k in ("L", "F", "Tp", "Tm")},
            "swapped_minus_neutral": {k: G.mean_ci(b[k] - n[k]) for k in ("L", "F", "Tp", "Tm")},
        }
    res["dist"] = dist
    # B4: by decile of swapped L
    dec = np.quantile(b["L"], np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(dec, b["L"], side="right") - 1, 0, 9)
    res["by_swapped_decile"] = [{"decile": k + 1, "L_swapped_range": [float(dec[k]), float(dec[k + 1])],
                                 "dL": float(d["L"][bins == k].mean()), "dF": float(d["F"][bins == k].mean()),
                                 "dTp": float(d["Tp"][bins == k].mean()), "n": int((bins == k).sum())}
                                for k in range(10)]
    # D1/D2: heavy tails, concentration
    x = d["L"]
    srt = np.sort(x)[::-1]
    gs = G.sums(x)
    gsort = np.sort(gs)[::-1]
    loo = [(x.sum() - gs[g]) / (len(x) - G.count[g]) - x.mean() for g in range(G.G)]
    res["heavy"] = {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "median_ci": G.stat_ci(np.median, x),
        "trim10": float(stats.trim_mean(x, 0.10)), "trim20": float(stats.trim_mean(x, 0.20)),
        "trim20_ci": G.stat_ci(lambda v: stats.trim_mean(v, 0.20), x),
        "share_gt10": float((x > 10).mean()), "share_gt20": float((x > 20).mean()),
        "share_lt0": float((x < 0).mean()), "share_eq0": float((x == 0).mean()),
        "sd": float(x.std(ddof=1)), "skew": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x)),
        "top1pct_share_of_sum": float(srt[: max(1, len(x) // 100)].sum() / x.sum()),
        "top5pct_share_of_sum": float(srt[: len(x) // 20].sum() / x.sum()),
        "top10pct_share_of_sum": float(srt[: len(x) // 10].sum() / x.sum()),
        "top5_games_share_of_sum": float(gsort[:5].sum() / x.sum()),
        "top10_games_share_of_sum": float(gsort[:10].sum() / x.sum()),
        "top5_games_share_of_items": float(np.sort(G.count)[::-1][:5].sum() / len(x)),
        "loo_game_max_abs_change": float(np.max(np.abs(loo))),
        "game_means": {"median": float(np.median(gs / G.count)), "p10": float(np.percentile(gs / G.count, 10)),
                       "p90": float(np.percentile(gs / G.count, 90)), "share_positive": float((gs > 0).mean())},
        "mean_without_top1pct": float(np.sort(x)[: len(x) - max(1, len(x) // 100)].mean()),
        "first_token": {"mean": float(d["F"].mean()), "median": float(np.median(d["F"])),
                        "trim20": float(stats.trim_mean(d["F"], 0.2)), "share_gt10": float((d["F"] > 10).mean())},
    }
    # D3: groups
    groups = {}
    for key in ("step0", "verb", "reason", "task_type"):
        g = collections.defaultdict(list)
        for j, i in enumerate(ids):
            g[meta[i][key]].append(j)
        groups[key] = {str(k): {"n": len(v), "dL": float(x[v].mean()), "dF": float(d["F"][v].mean()),
                                "dTp": float(d["Tp"][v].mean()), "median_dL": float(np.median(x[v])),
                                "contribution": float(x[v].sum() / len(x))}
                       for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    res["groups"] = groups
    # E: scale
    def scale(xa, xb, xn, dx):
        out = {"sd_aligned": float(xa.std(ddof=1)), "sd_swapped": float(xb.std(ddof=1)),
               "mean_abs_swapped": float(np.abs(xb).mean()), "median_swapped": float(np.median(xb)),
               "sd_delta": float(dx.std(ddof=1)), "dz": float(dx.mean() / dx.std(ddof=1)),
               "std_by_sd_swapped": float(dx.mean() / xb.std(ddof=1)),
               "share_delta_pos": float((dx > 0).mean()), "share_delta_neg": float((dx < 0).mean()),
               "pooled_auc_aligned_over_swapped": pooled_auc(xa, xb),
               "sign_up": float(((xb < 0) & (xa > 0)).mean()), "sign_down": float(((xb > 0) & (xa < 0)).mean()),
               "sign_either": float((np.sign(xa) != np.sign(xb)).mean()),
               "dsigma_pp": float(100 * (sig(xa) - sig(xb)).mean())}
        if xn is not None:
            out["sd_neutral"] = float(xn.std(ddof=1)); out["std_by_sd_neutral"] = float(dx.mean() / xn.std(ddof=1))
            out["mean_abs_neutral"] = float(np.abs(xn).mean()); out["median_neutral"] = float(np.median(xn))
        return out
    nn = arrays(N, N, ids)["a"] if N is not None else None
    res["scale"] = {"L": scale(a["L"], b["L"], None if nn is None else nn["L"], d["L"]),
                    "F": scale(a["F"], b["F"], None if nn is None else nn["F"], d["F"])}
    res["scale"]["L"]["dz_ci"] = G.stat_ci(lambda v: v.mean() / v.std(ddof=1), d["L"])
    res["scale"]["F"]["dz_ci"] = G.stat_ci(lambda v: v.mean() / v.std(ddof=1), d["F"])
    # C2 proxy: tail shift against reply length within verb
    verbs = np.array([meta[i]["verb"] for i in ids])
    ntp = a["ntp"]
    by = {}
    for v in sorted(set(verbs)):
        m = verbs == v
        if m.sum() > 30 and np.std(ntp[m]) > 0:
            rho = stats.spearmanr(ntp[m], d["Tp"][m]).statistic
            by[v] = {"n": int(m.sum()), "spearman_len_vs_dTp": float(rho), "mean_len": float(ntp[m].mean())}
    res["tail_vs_length"] = by
    res["_arrays"] = {"dL": d["L"], "dF": d["F"], "dTp": d["Tp"], "La": a["L"], "Lb": b["L"], "Fa": a["F"], "Fb": b["F"]}
    return res


def contrast(G: Games, R: dict, model: str, c1: str, c0: str, ids: list[str]) -> dict:
    """Mean of c1 - c0 on L, first token and both tails, over the ids scored in both."""
    A, Bc = R.get((model, c1)), R.get((model, c0))
    if not A or not Bc:
        return {}
    ids = [i for i in ids if i in A and i in Bc]
    gm = Games([A[i]["game_id"] for i in ids])
    X = arrays(A, Bc, ids)
    out = {"n": len(ids)}
    for k in ("L", "F", "Tp", "Tm"):
        out[k] = gm.mean_ci(X["a"][k] - X["b"][k])
    out["tail_share"] = float((X["a"]["Tp"] - X["b"]["Tp"]).mean() / (X["a"]["L"] - X["b"]["L"]).mean())
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/_qwen_audit")
    ap.add_argument("--out", default="notes/qwen_effect_audit_forensic_metrics.json")
    ap.add_argument("--tok", default="runs/_qwen_audit/prompt_check.json", help="output of prompt_check.py (optional)")
    args = ap.parse_args()
    root = args.root

    items = load_items()
    P = primary_set(items)
    ids = [t["id"] for t in P]
    byid = {t["id"]: t for t in P}
    meta = {}
    for t in P:
        meta[t["id"]] = {"hl": t["history_len"], "step0": t["history_len"] == 0, "verb": t["action_type"],
                         "reason": t["determination_reason"], "task_type": task_type(t["task_line"]),
                         "game": t["game_id"], "obj": t["goal_object"], "hash": _item_hash(t)}
    G = Games([t["game_id"] for t in P])
    M: dict = {"items": len(ids), "games": G.G}

    runs = {"rq2": "fix11/rq2_score", "rq1": "fix12/rq1_score", "rq1x": "fix13/rq1x_score", "rq1v": "fix17/rq1v_score",
            "mech": "fix19/mech_score", "mech2": "fix36/mech2_score"}
    for s in ("0.6b", "1.7b", "4b", "8b", "14b", "32b"):
        runs[f"size_{s}"] = f"fix18/size_score_qwen3-{s}"
    for f in ("gemma2-9b-it", "llama3.2-1b", "llama3.2-3b", "mistral-7b-v0.3"):
        runs[f"fam_{f}"] = f"fix21/family_score_{f}"
    R, probs = {}, {}
    for k, r in runs.items():
        R[k], probs[k] = load_run(root, r)

    # ---------------- A: sanity checks
    A: dict = {"run_problems": probs}
    stale, missing, extra = collections.Counter(), collections.Counter(), collections.Counter()
    cand_mismatch = collections.Counter()
    for k, rr in R.items():
        for (model, cond), recs in rr.items():
            for tid, r in recs.items():
                if tid in meta and r.get("item_hash") and r["item_hash"] != meta[tid]["hash"]:
                    stale[f"{k}|{model}|{cond}"] += 1
    for k in ("rq2", "rq1", "rq1v"):
        for (model, cond), recs in R[k].items():
            if cond in ("aligned", "swapped", "absent", "yn_aligned", "yn_swapped", "yn_absent"):
                missing[f"{k}|{model}|{cond}"] = sum(1 for i in ids if i not in recs)
    # candidate tokenization across conditions, within each run and model
    for k, rr in R.items():
        models = {m for m, _ in rr}
        for model in models:
            conds = [c for m, c in rr if m == model]
            for fmt in ("yn", "cand"):
                cs = [c for c in conds if (c.startswith("yn_") if fmt == "yn" else not c.startswith("yn_"))]
                if k == "mech":
                    cs = [c for c in cs if c.startswith("pred_")]
                for i in ids:
                    sig_ = {(rr[(model, c)][i]["n_tokens_plus"], rr[(model, c)][i]["n_tokens_minus"],
                             rr[(model, c)][i]["first_token_plus"], rr[(model, c)][i]["first_token_minus"])
                            for c in cs if i in rr[(model, c)]}
                    if len(sig_) > 1:
                        cand_mismatch[f"{k}|{model}|{fmt}"] += 1
    A["stale_item_hash"] = dict(stale)
    A["missing_primary"] = {k: v for k, v in missing.items()}
    A["candidate_signature_mismatch_items"] = dict(cand_mismatch)
    shas = {}
    for k, r in runs.items():
        shas[k] = sorted({open(p).read().strip()[:7] for p in glob.glob(os.path.join(root, r, "*", "git_sha"))})
    A["commits"] = shas
    # A4: GPU / shard
    gpu = {}
    tsv = Path(root) / "gpu_by_shard.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 4:
                gpu.setdefault(parts[2], []).append(parts[3])
    A["gpu_by_job_start_line"] = gpu
    rq2q_a, rq2q_s = R["rq2"][("qwen3-8b-fewshot", "aligned")], R["rq2"][("qwen3-8b-fewshot", "swapped")]
    sz_a, sz_s = R["size_8b"][("qwen3-8b", "aligned")], R["size_8b"][("qwen3-8b", "swapped")]
    by_shard = {}
    for sh in sorted({rq2q_a[i]["_shard"] for i in ids}):
        m = np.array([rq2q_a[i]["_shard"] == sh for i in ids])
        dl_rq2 = np.array([rq2q_a[i]["L"] - rq2q_s[i]["L"] for i in ids])
        dl_sz = np.array([sz_a[i]["L"] - sz_s[i]["L"] for i in ids])
        diff = dl_rq2 - dl_sz
        exact = np.array([(rq2q_a[i]["L"] == sz_a[i]["L"]) and (rq2q_s[i]["L"] == sz_s[i]["L"]) for i in ids])
        size_shards = collections.Counter(sz_a[i]["_shard"] for i, mm in zip(ids, m) if mm)
        by_shard[sh] = {"n": int(m.sum()), "games": len({meta[i]["game"] for i, mm in zip(ids, m) if mm}),
                        "dL_rq2": float(dl_rq2[m].mean()), "dL_size_rescore": float(dl_sz[m].mean()),
                        "mean_diff_rq2_minus_rescore": float(diff[m].mean()), "sd_diff": float(diff[m].std(ddof=1)),
                        "share_exact_both_conditions": float(exact[m].mean()), "size_shards": dict(size_shards)}
    A["qwen_rq2_vs_rescore_by_rq2_shard"] = by_shard
    for model, mk, run_k in (("llama3.1-8b-fewshot", "llama", "rq1"), ("qwen3-8b-fewshot", "qwen", "rq1")):
        a2, s2 = R["rq2"][(model, "aligned")], R["rq2"][(model, "swapped")]
        a1, s1 = R[run_k][(model, "aligned")], R[run_k][(model, "swapped")]
        d2 = np.array([a2[i]["L"] - s2[i]["L"] for i in ids]); d1 = np.array([a1[i]["L"] - s1[i]["L"] for i in ids])
        A[f"{mk}_rq2_vs_rq1_score"] = {"mean_rq2": float(d2.mean()), "mean_rq1": float(d1.mean()),
                                       "sd_item_diff": float((d2 - d1).std(ddof=1)),
                                       "pearson": float(np.corrcoef(d2, d1)[0, 1]),
                                       "share_exact": float((d2 == d1).mean())}
    # task_line_used in rq2 records: aligned should equal the item's task line, swapped the swapped line
    tl_bad = collections.Counter()
    for (model, cond), recs in R["rq2"].items():
        for i in ids:
            r = recs.get(i)
            if r is None or cond not in ("aligned", "swapped"):
                continue
            want = byid[i]["task_line"] if cond == "aligned" else byid[i]["swapped_task_line"]
            if r.get("task_line_used") not in (None, want):
                tl_bad[f"{model}|{cond}"] += 1
    A["rq2_task_line_used_mismatch"] = dict(tl_bad)
    M["A"] = A

    # ---------------- per-model two-candidate and yes/no blocks
    models = {
        "qwen3-8b": ("rq2", "qwen3-8b-fewshot", "rq1", "rq1v"),
        "llama3.1-8b": ("rq2", "llama3.1-8b-fewshot", "rq1", "rq1v"),
        "gemma2-9b": ("fam_gemma2-9b-it", "gemma2-9b-it", None, None),
        "mistral-7b": ("fam_mistral-7b-v0.3", "mistral-7b-v0.3", None, None),
        "llama3.2-3b": ("fam_llama3.2-3b", "llama3.2-3b", None, None),
        "llama3.2-1b": ("fam_llama3.2-1b", "llama3.2-1b", None, None),
    }
    for s in ("0.6b", "1.7b", "4b", "8b", "14b", "32b"):
        models[f"qwen3-{s} (size run)"] = (f"size_{s}", f"qwen3-{s}", None, None)
    per = {}
    keep_arrays = {}
    for name, (rk, mtag, neutral_run, yn_run) in models.items():
        rr = R[rk]
        Aa, Ss = rr[(mtag, "aligned")], rr[(mtag, "swapped")]
        Nn = (R[neutral_run] if neutral_run else rr).get((mtag, "absent"))
        blk = two_cand_block(G, Aa, Ss, Nn, meta, ids)
        yr = R[yn_run] if yn_run else rr
        ya, ys, yn_ = yr[(mtag, "yn_aligned")], yr[(mtag, "yn_swapped")], yr.get((mtag, "yn_absent"))
        Y = arrays(ya, ys, ids)
        dy = Y["a"]["L"] - Y["b"]["L"]
        step0 = np.array([meta[i]["step0"] for i in ids])
        yb = {"effect": G.mean_ci(dy), "effect_later": G.mean_ci(dy, ~step0), "effect_step0": G.mean_ci(dy, step0),
              "median": float(np.median(dy)), "trim20": float(stats.trim_mean(dy, 0.2)),
              "share_gt10": float((dy > 10).mean())}
        ynn = arrays(yn_, yn_, ids)["a"]["L"] if yn_ else None
        sc = {"sd_aligned": float(Y["a"]["L"].std(ddof=1)), "sd_swapped": float(Y["b"]["L"].std(ddof=1)),
              "median_swapped": float(np.median(Y["b"]["L"])), "mean_abs_swapped": float(np.abs(Y["b"]["L"]).mean()),
              "sd_delta": float(dy.std(ddof=1)), "dz": float(dy.mean() / dy.std(ddof=1)),
              "std_by_sd_swapped": float(dy.mean() / Y["b"]["L"].std(ddof=1)),
              "share_delta_pos": float((dy > 0).mean()), "share_delta_neg": float((dy < 0).mean()),
              "pooled_auc_aligned_over_swapped": pooled_auc(Y["a"]["L"], Y["b"]["L"]),
              "answered_yes_swapped": float((Y["b"]["L"] > 0).mean()), "answered_yes_aligned": float((Y["a"]["L"] > 0).mean()),
              "flip_no_to_yes": float(((Y["b"]["L"] <= 0) & (Y["a"]["L"] > 0)).mean()),
              "dsigma_pp": float(100 * (sig(Y["a"]["L"]) - sig(Y["b"]["L"])).mean()),
              "share_swapped_within_mean_shift_of_boundary": float((Y["b"]["L"] > -dy.mean()).mean()),
              "share_swapped_within_5nats_of_boundary": float((Y["b"]["L"] > -5).mean())}
        sc["dz_ci"] = G.stat_ci(lambda v: v.mean() / v.std(ddof=1), dy)
        if ynn is not None:
            sc["sd_neutral"] = float(ynn.std(ddof=1)); sc["median_neutral"] = float(np.median(ynn))
            sc["std_by_sd_neutral"] = float(dy.mean() / ynn.std(ddof=1))
        yb["scale"] = sc
        dL = blk["_arrays"]["dL"]; dF = blk["_arrays"]["dF"]
        yb["spearman_item_dL2c_vs_dLyn"] = float(stats.spearmanr(dL, dy).statistic)
        yb["spearman_item_dF_vs_dLyn"] = float(stats.spearmanr(dF, dy).statistic)
        blk["yes_no"] = yb
        keep_arrays[name] = {**blk.pop("_arrays"), "dY": dy, "Ya": Y["a"]["L"], "Yb": Y["b"]["L"]}
        per[name] = blk
    M["models"] = per

    # ---------------- F: format gap and cross-model gap, Qwen
    q8 = keep_arrays["qwen3-8b"]
    step0 = np.array([meta[i]["step0"] for i in ids])
    F = {}
    for tag, mask in (("all", np.ones(len(ids), bool)), ("later", ~step0), ("step0", step0)):
        F[tag] = {
            "two_cand": float(q8["dL"][mask].mean()), "first_token": float(q8["dF"][mask].mean()),
            "success_tail": float(q8["dTp"][mask].mean()), "yes_no": float(q8["dY"][mask].mean()),
            "gap_2c_minus_yn": G.mean_ci(q8["dL"] - q8["dY"], mask),
            "gap_first_minus_yn": G.mean_ci(q8["dF"] - q8["dY"], mask),
        }
    M["F_qwen_format_gap"] = F
    # standardized (dz) per model and format, all and later-only
    std = {}
    for name, arr in keep_arrays.items():
        row = {}
        for tag, mask in (("all", np.ones(len(ids), bool)), ("later", ~step0)):
            for k in ("dL", "dF", "dY"):
                v = arr[k][mask]
                row[f"{k}_{tag}_mean"] = float(v.mean())
                row[f"{k}_{tag}_dz"] = float(v.mean() / v.std(ddof=1))
                row[f"{k}_{tag}_share_pos"] = float((v > 0).mean())
                row[f"{k}_{tag}_median"] = float(np.median(v))
            for base, k in (("Lb", "dL"), ("Fb", "dF"), ("Yb", "dY")):
                row[f"{k}_{tag}_by_sd_swapped"] = float(arr[k][mask].mean() / arr[base][mask].std(ddof=1))
        std[name] = row
    M["F_standardized"] = std

    # ---------------- C3: copying / priming contrasts with the decomposition
    C = {}
    for model in ("qwen3-8b-fewshot", "llama3.1-8b-fewshot"):
        c = {}
        c["rq1_aligned_minus_swapped"] = contrast(G, R["rq1"], model, "aligned", "swapped", ids)
        c["rq1_mention_added_to_swapped"] = contrast(G, R["rq1"], model, "swapped_plus", "swapped", ids)
        c["rq1_goal_role_both_named"] = contrast(G, R["rq1"], model, "aligned_plus", "swapped_plus", ids)
        c["rq1_aligned_minus_neutral"] = contrast(G, R["rq1"], model, "aligned", "absent", ids)
        c["rq1_swapped_minus_neutral"] = contrast(G, R["rq1"], model, "swapped", "absent", ids)
        c["rq1x_described_aligned_minus_described_swapped"] = contrast(G, R["rq1x"], model, "aligned_desc", "swapped_desc", ids)
        c["rq2_2x2_mention_part"] = contrast(G, R["rq2"], model, "mention", "fourth", ids)
        c["rq2_2x2_aligned_minus_mention"] = contrast(G, R["rq2"], model, "aligned", "mention", ids)
        c["rq2_2x2_items_aligned_minus_swapped"] = contrast(G, R["rq2"], model, "aligned", "swapped",
                                                            [i for i in ids if i in R["rq2"][(model, "mention")]])
        c["mech_facts_aligned_minus_swapped"] = contrast(G, R["mech"], model, "pred_facts_aligned", "pred_facts_swapped", ids)
        c["mech_rules_aligned_minus_swapped"] = contrast(G, R["mech"], model, "pred_rules_aligned", "pred_rules_swapped", ids)
        m2 = R["mech2"]
        samp = [i for i in ids if (model, "aligned") in m2 and i in m2[(model, "aligned")] and m2[(model, "aligned")][i].get("in_sample")]
        for nm, c1, c0 in (("aligned_minus_swapped", "aligned", "swapped"), ("P3_mention_ignore", "mention_ignore", "swapped"),
                           ("P4_described", "described", "swapped"), ("mention_exists", "mention_exists", "swapped"),
                           ("P7_negated", "negated", "swapped"), ("P8_negated_minus_affirmed", "negated", "affirmed"),
                           ("affirmed_minus_swapped", "affirmed", "swapped"),
                           ("placebo_negated_minus_affirmed", "placebo_negated", "placebo_affirmed")):
            c[f"mech2_{nm}"] = contrast(G, m2, model, c1, c0, samp)
        # the reference effect on the take items of P7/P8
        take_s = [i for i in samp if (model, "negated") in m2 and i in m2[(model, "negated")]]
        c["mech2_aligned_minus_swapped_on_take_items"] = contrast(G, m2, model, "aligned", "swapped", take_s)
        C[model] = c
    M["C3_contrasts"] = C

    # ---------------- C5: what Qwen writes after "You" (greedy predictions)
    gens = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(root, runs["rq2"], "*", "generations.jsonl"))):
        for line in open(f):
            g = json.loads(line)
            if g["transition_id"] not in meta:
                continue
            txt = g["generated"].strip()
            first = txt.split(" ")[0] if txt else ""
            if first == "You":
                o_plus = byid[g["transition_id"]]["candidates"]["o_plus"]
                kind = "You: the success reply exactly" if txt == o_plus else (
                    "You: same verb, other details" if txt.split(" ")[:2] == o_plus.split(" ")[:2] else
                    "You: other wording (" + " ".join(txt.split(" ")[:3]) + ")")
            else:
                kind = "Nothing happens." if txt.startswith("Nothing happens") else "other: " + " ".join(txt.split(" ")[:3])
            gens[f"{g['model']}|{g['condition']}"][kind] += 1
    M["C5_generations"] = {k: dict(v.most_common(15)) for k, v in gens.items()}

    # ---------------- tokenizer-side checks from prompt_check.py (run on ICE), if present
    if Path(args.tok).exists():
        tok = json.loads(Path(args.tok).read_text())
        M["A2_prompt_check"] = {k: v for k, v in tok.items() if k != "per_item"}
        per_item = tok.get("per_item", {})
        for name in ("qwen3-8b", "llama3.1-8b"):
            arr = keep_arrays[name]
            key = "qwen" if name.startswith("qwen") else "llama"
            obj_tok = np.array([per_item.get(i, {}).get(f"{key}_obj_tokens", np.nan) for i in ids], float)
            ok = np.isfinite(obj_tok)
            out = {}
            verbs = np.array([meta[i]["verb"] for i in ids])
            for v in sorted(set(verbs)):
                m = ok & (verbs == v)
                if m.sum() > 30 and np.std(obj_tok[m]) > 0:
                    out[v] = {"n": int(m.sum()), "spearman_obj_tokens_vs_dTp": float(stats.spearmanr(obj_tok[m], arr["dTp"][m]).statistic),
                              "dTp_by_obj_tokens": {str(int(k)): float(arr["dTp"][m & (obj_tok == k)].mean())
                                                    for k in sorted(set(obj_tok[m])) if (m & (obj_tok == k)).sum() >= 20}}
            per[name]["tail_vs_object_tokens"] = out
    Path(args.out).write_text(json.dumps(M, indent=1, sort_keys=False, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
