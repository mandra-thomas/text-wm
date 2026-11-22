"""Exploratory, post hoc diagnostics of the τ-bench retail replication (notes/tau_dx.md).

Nothing here scores anything, and nothing imports wf/tau/pipeline.py or wf/analysis/stats.py: the per-item scores
are re-read from the raw `tau_score` records and the cluster bootstrap is a separate implementation (numpy,
multinomial counts of whole users, one shared set of draws per population). Heuristics H1-H6 and the status 2 x 2
were fixed in notes/tau_dx.md (commit bc7acf6) before any item-level score was read.

Usage:
    python scripts/tau_dx.py --scores <tau_score dir> --build <tau_build dir> \
        [--natural-mentions <natural_mentions.json>] [--reference notes/tau_metrics.json] \
        --out notes/tau_dx_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

SEED = 20260924
B = 2000
MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
SHORT = {"qwen3-8b-fewshot": "qwen", "llama3.1-8b-fewshot": "llama"}
CONDS = ("aligned", "swapped", "neutral")
PINS = {"items.jsonl": "2c8ffeac7dedd4635df5974be955d6ceff84d3f8184f88a73eca579282a578b6",
        "natural_items.jsonl": "9bd6c3f7fda464f7cc32307e9a570da564b1bed7d4e3ab0d75dbfd00f6b83502",
        "fewshot.json": "231607f79c92580375ada0ce4267fb954e7e0bc0a743ccb9b75813610eb13119"}
STATUS_RULE = {"modify_pending_order_items": lambda s: s == "pending",
               "cancel_pending_order": lambda s: s == "pending",
               "exchange_delivered_order_items": lambda s: s == "delivered",
               "return_delivered_order_items": lambda s: s == "delivered",
               "modify_pending_order_payment": lambda s: "pending" in s,
               "modify_pending_order_address": lambda s: "pending" in s}
CONF = math.log(99)  # |L| above this: the model gives the outcome at least 99%


def r(x, n=3):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fewshot_reversed(item_id: str) -> bool:
    return int(hashlib.sha256(f"{SEED}|{item_id}|fewshot-order".encode()).hexdigest()[:16], 16) % 2 == 1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


# ---------------------------------------------------------------- bootstrap (whole users, multinomial counts)

class Boot:
    """One shared set of 2,000 user resamples for a population of rows; each row carries its user."""

    def __init__(self, users: list[str], seed: int = SEED, b: int = B):
        self.users = sorted(set(users))
        ix = {u: i for i, u in enumerate(self.users)}
        self.u = np.array([ix[x] for x in users])
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.users), size=(b, len(self.users)))
        counts = np.zeros((b, len(self.users)))
        np.add.at(counts, (np.repeat(np.arange(b), len(self.users)), draws.ravel()), 1)
        self.W = counts[:, self.u]  # b x rows

    def mean(self, v, mask=None) -> dict:
        v = np.asarray(v, float)
        m = np.isfinite(v) if mask is None else (np.asarray(mask, bool) & np.isfinite(v))
        if not m.any():
            return {"est": None, "ci": None, "n": 0, "users": 0}
        W = self.W[:, m]
        num, den = W @ v[m], W.sum(1)
        ok = den > 0
        bs = num[ok] / den[ok]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": r(v[m].mean()), "ci": [r(lo), r(hi)], "n": int(m.sum()), "users": int(len(set(self.u[m])))}

    def stat(self, fn, *arrays, mask=None) -> dict:
        """Any statistic by replication: rows are repeated by their user's count in each resample."""
        arrays = [np.asarray(a) for a in arrays]
        m = np.ones(len(arrays[0]), bool) if mask is None else np.asarray(mask, bool)
        for a in arrays:
            if a.dtype.kind == "f":
                m &= np.isfinite(a)
        sub = [a[m] for a in arrays]
        est = fn(*sub)
        Wm = self.W[:, m].astype(int)
        bs = []
        for b in range(Wm.shape[0]):
            idx = np.repeat(np.arange(m.sum()), Wm[b])
            if len(idx) == 0:
                continue
            try:
                val = fn(*[a[idx] for a in sub])
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                continue
            if val is not None and np.all(np.isfinite(val)):
                bs.append(val)
        bs = np.array(bs)
        if np.ndim(est) == 0:
            lo, hi = np.percentile(bs, [2.5, 97.5])
            return {"est": r(est), "ci": [r(lo), r(hi)], "n": int(m.sum()), "users": int(len(set(self.u[m])))}
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        return {"est": [r(x) for x in est], "ci": [[r(a), r(b_)] for a, b_ in zip(lo, hi)], "n": int(m.sum())}


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = rankdata(s)
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def trimmed(v, a):
    v = np.sort(np.asarray(v, float))
    k = int(math.floor(a * len(v)))
    return v[k:len(v) - k].mean() if len(v) > 2 * k else float("nan")


def kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ---------------------------------------------------------------- data

class Data:
    def __init__(self, scores_dir: Path, build_dir: Path):
        self.hashes = {name: sha(build_dir / name) for name in PINS}
        assert self.hashes == PINS, f"build files differ from the pins: {self.hashes}"
        gen = read_jsonl(build_dir / "items.jsonl")
        nat = read_jsonl(build_dir / "natural_items.jsonl")
        self.items = {t["id"]: t for t in gen + nat}
        self.primary = [t for t in gen if t["source"] == "generated" and t["main_pool"] and t["status_matched"]]
        self.main = [t for t in gen if t["main_pool"]]
        self.natural = nat
        self.by = {}
        dup = 0
        shas = set()
        for shard in sorted(scores_dir.glob("shard_*_of_16")):
            shas.add((shard / "git_sha").read_text().strip())
            for s in read_jsonl(shard / "scoring.jsonl"):
                k = (s["model"], s["transition_id"], s.get("arm", "main"), s.get("action", "fail"), s["readout"], s["condition"])
                if k in self.by:
                    dup += 1
                self.by[k] = s
        self.n_records, self.duplicates, self.shard_shas = len(self.by), dup, sorted(shas)

    def get(self, model, ids, arm="main", action="fail", readout="first", cond="aligned", kind="L") -> np.ndarray:
        out = np.full(len(ids), np.nan)
        for i, t in enumerate(ids):
            s = self.by.get((model, t, arm, action, readout, cond))
            if s is None:
                continue
            if kind == "L":
                out[i] = s["L"]
            elif kind == "norm":
                out[i] = s["logp_plus"] / s["n_tokens_plus"] - s["logp_minus"] / s["n_tokens_minus"]
            elif kind == "first_of_full":
                out[i] = s["logp_first_plus"] - s["logp_first_minus"]
            elif kind == "mass":
                out[i] = math.exp(s["logp_plus"]) + math.exp(s["logp_minus"])
            elif kind == "tail":  # full-reply L minus its first-token term
                out[i] = s["L"] - (s["logp_first_plus"] - s["logp_first_minus"])
            elif kind == "tail_plus":  # log P(success reply after its first token)
                out[i] = s["logp_plus"] - s["logp_first_plus"]
            elif kind == "tail_minus":  # log P(failure reply after its first token)
                out[i] = s["logp_minus"] - s["logp_first_minus"]
        return out


# ---------------------------------------------------------------- per-call features (visible text only)

def parse_outputs(history):
    orders, users, products = {}, {}, {}
    for h in history:
        try:
            o = json.loads(h["output"])
        except (ValueError, TypeError):
            continue
        if not isinstance(o, dict):
            continue
        if {"order_id", "items", "status"} <= set(o):
            orders[o["order_id"]] = o
        if {"user_id", "payment_methods"} <= set(o):
            users[o["user_id"]] = o
        if {"product_id", "variants"} <= set(o):
            products[o["product_id"]] = o
    return orders, users, products


def call_features(t: dict, call: dict) -> dict:
    orders, users, products = parse_outputs(t["history"])
    a = call["arguments"]
    o = orders.get(a.get("order_id"), {})
    status = o.get("status", "")
    name = call["name"]
    h1 = int(STATUS_RULE.get(name, lambda s: True)(status))
    unav = 0
    for nid in a.get("new_item_ids", []) or []:
        for p in products.values():
            v = p["variants"].get(nid)
            if v is not None and v.get("available") is False:
                unav = 1
    pm = a.get("payment_method_id", "")
    user = users.get(o.get("user_id", t["user_id"]), {})
    src = (user.get("payment_methods", {}).get(pm, {}) or {}).get("source", "")
    crowd = None
    if t.get("target_product_id") in products:
        vs = products[t["target_product_id"]]["variants"]
        crowd = sum(1 for v in vs.values() if v.get("available") is False) / max(1, len(vs))
    return {"H1": h1, "H2": 1 - unav, "H1andH2": int(h1 and not unav), "H3": int(status != "delivered"),
            "H4": (0 if name == "exchange_delivered_order_items" else 1 if name == "modify_pending_order_items" else None),
            "H5": int(src != "gift_card"), "H6": crowd, "delivered": int(status == "delivered"),
            "exchange": int(name == "exchange_delivered_order_items"), "tool": name, "status": status, "gift_card": int(src == "gift_card")}


def item_features(t: dict) -> dict:
    hist_chars = sum(len(json.dumps({"name": h["name"], "arguments": h["arguments"]})) + len(h["output"]) for h in t["history"])
    orders, _, _ = parse_outputs(t["history"])
    target_products = {it["name"] for it in orders.get(t["order_id"], {}).get("items", [])}
    comp_products = {it["name"] for it in orders.get(t.get("comparison_order_id"), {}).get("items", [])}
    return {"cls": t["cls"], "tool": t["action"]["name"], "status": t["target_status"],
            "cell": f"{t['cls']}:{t['action']['name'].split('_')[0]}-on-{t['target_status']}",
            "n_calls": len(t["history"]), "hist_chars": hist_chars, "target_first": t.get("target_first"),
            "fewshot_reversed": fewshot_reversed(t["id"]),
            "same_product_name": t.get("product") is not None and t.get("product") == t.get("comparison_product"),
            "swapped_product_in_target_order": t.get("comparison_product") in target_products,
            "target_product_in_comparison_order": t.get("product") in comp_products,
            "witness_same_pm": t["aligned_witness"]["arguments"].get("payment_method_id") == t["action"]["arguments"].get("payment_method_id")
            if t.get("aligned_witness") else None,
            "witness_checker": t.get("witness_checker")}


# ---------------------------------------------------------------- analyses

def effects_block(bt: Boot, D: Data, model, ids, arm="main", action="fail", readout="first", kind="L", mask=None) -> dict:
    v = {c: D.get(model, ids, arm, action, readout, c, kind) for c in CONDS}
    return {"aligned_minus_swapped": bt.mean(v["aligned"] - v["swapped"], mask),
            "aligned_minus_neutral": bt.mean(v["aligned"] - v["neutral"], mask),
            "swapped_minus_neutral": bt.mean(v["swapped"] - v["neutral"], mask)}


def q1_recompute(D: Data, ref: dict | None) -> dict:
    P = [t["id"] for t in D.primary]
    users = [t["user_id"] for t in D.primary]
    bt = Boot(users)
    out = {}
    for m in MODELS:
        o = {}
        for ro, kind in (("first", "L"), ("yn", "L"), ("full", "L"), ("full", "norm")):
            o[f"{ro}_{kind}"] = effects_block(bt, D, m, P, readout=ro, kind=kind)
        o["witness_first"] = effects_block(bt, D, m, P, action="witness")
        o["witness_yn"] = effects_block(bt, D, m, P, action="witness", readout="yn")
        d = D.get(m, P, cond="aligned") - D.get(m, P, cond="swapped")
        u = np.array(users)
        per_user = np.array([d[u == x].mean() for x in sorted(set(users))])
        ucount = np.array([(u == x).sum() for x in sorted(set(users))])
        o["distribution"] = {
            "item_quartiles": [r(x) for x in np.percentile(d, [25, 50, 75])],
            "item_p5_p95": [r(x) for x in np.percentile(d, [5, 95])],
            "item_sd": r(d.std(ddof=1)),
            "share_items_positive": bt.mean((d > 0).astype(float)),
            "share_items_zero": r(float((d == 0).mean())),
            "user_quartiles": [r(x) for x in np.percentile(per_user, [25, 50, 75])],
            "share_users_positive": r(float((per_user > 0).mean())),
            "median": bt.stat(np.median, d),
            "trimmed_10": bt.stat(lambda x: trimmed(x, 0.10), d),
            "trimmed_20": bt.stat(lambda x: trimmed(x, 0.20), d),
            "mean_of_user_means": r(per_user.mean()),
        }
        tot = d.sum()
        loo = [(tot - d[u == x].sum()) / (len(d) - (u == x).sum()) for x in sorted(set(users))]
        o["leave_one_user_out"] = [r(min(loo)), r(max(loo))]
        # influence: user's contribution to the mean relative to the leave-one-out mean
        infl = np.array([d.mean() - l_ for l_ in loo])
        order = np.argsort(-infl)
        topk = {}
        for k in (1, 5, 10, 20):
            drop = set(np.array(sorted(set(users)))[order[:k]])
            keep = np.array([x not in drop for x in users])
            topk[str(k)] = bt.mean(d, keep)
        o["drop_top_k_influential_users"] = topk
        pos_share = np.sort(per_user * ucount)[::-1]
        k10 = int(math.ceil(0.1 * len(per_user)))
        o["share_of_total_from_top_10pct_users"] = r(float(pos_share[:k10].sum() / tot))
        o["n_items"], o["n_users"] = len(P), len(set(users))
        if ref:
            rm = ref["models"][m]["primary"]
            o["reference_check"] = {
                "first": [o["first_L"]["aligned_minus_swapped"]["est"], r(rm["first"]["aligned_minus_swapped"]["estimate"])],
                "yn": [o["yn_L"]["aligned_minus_swapped"]["est"], r(rm["yn"]["aligned_minus_swapped"]["estimate"])],
                "full": [o["full_L"]["aligned_minus_swapped"]["est"], r(rm["full"]["aligned_minus_swapped"]["estimate"])],
                "full_norm": [o["full_norm"]["aligned_minus_swapped"]["est"], r(rm["full_norm"]["aligned_minus_swapped"]["estimate"])],
                "first_ci_ref": [r(x) for x in rm["first"]["aligned_minus_swapped"]["ci"]],
            }
        out[m] = o
    return out


def cut_block(bt, D, m, ids, mask) -> dict:
    f = {c: D.get(m, ids, cond=c) for c in CONDS}
    w = {c: D.get(m, ids, action="witness", cond=c) for c in CONDS}
    y = {c: D.get(m, ids, readout="yn", cond=c) for c in CONDS}
    return {"fail_as": bt.mean(f["aligned"] - f["swapped"], mask),
            "fail_an": bt.mean(f["aligned"] - f["neutral"], mask),
            "fail_sn": bt.mean(f["swapped"] - f["neutral"], mask),
            "witness_as": bt.mean(w["aligned"] - w["swapped"], mask),
            "witness_an": bt.mean(w["aligned"] - w["neutral"], mask),
            "witness_sn": bt.mean(w["swapped"] - w["neutral"], mask),
            "fail_minus_witness": bt.mean((f["aligned"] - f["swapped"]) - (w["aligned"] - w["swapped"]), mask),
            "yn_as": bt.mean(y["aligned"] - y["swapped"], mask),
            "neutral_fail_mean": r(float(np.nanmean(f["neutral"][mask]))) if mask.any() else None,
            "neutral_witness_mean": r(float(np.nanmean(w["neutral"][mask]))) if mask.any() else None}


def q2_heterogeneity(D: Data, nat_mentions: dict | None) -> dict:
    P = D.primary
    ids = [t["id"] for t in P]
    bt = Boot([t["user_id"] for t in P])
    F = [item_features(t) for t in P]
    chars = np.array([f["hist_chars"] for f in F])
    terc = np.percentile(chars, [100 / 3, 200 / 3])
    cuts = {}
    for c in sorted({f["cls"] for f in F}):
        cuts[f"class={c}"] = np.array([f["cls"] == c for f in F])
    for c in sorted({f["cell"] for f in F}):
        cuts[f"cell={c}"] = np.array([f["cell"] == c for f in F])
    for n in sorted({f["n_calls"] for f in F}):
        cuts[f"n_calls={n}"] = np.array([f["n_calls"] == n for f in F])
    for i, (lo, hi) in enumerate(((-1, terc[0]), (terc[0], terc[1]), (terc[1], 1e12))):
        cuts[f"chars_third={i + 1}"] = (chars > lo) & (chars <= hi)
    for cls in ("status", "unavailable"):
        cc = np.array([f["cls"] == cls for f in F])
        med = np.median(chars[cc])
        cuts[f"{cls}:chars<=median"] = cc & (chars <= med)
        cuts[f"{cls}:chars>median"] = cc & (chars > med)
    for k in ("target_first", "fewshot_reversed", "swapped_product_in_target_order", "target_product_in_comparison_order",
              "same_product_name", "witness_same_pm"):
        for val in (True, False):
            cuts[f"{k}={val}"] = np.array([f[k] == val for f in F])
    for cls in ("status", "unavailable"):
        cc = np.array([f["cls"] == cls for f in F])
        for val in (True, False):
            cuts[f"{cls}:target_first={val}"] = cc & np.array([f["target_first"] == val for f in F])
            cuts[f"{cls}:fewshot_reversed={val}"] = cc & np.array([f["fewshot_reversed"] == val for f in F])
    out = {"tercile_cutpoints_chars": [r(x, 0) for x in terc], "cuts": {}}
    for name, mask in cuts.items():
        if mask.sum() == 0:
            continue
        out["cuts"][name] = {"n": int(mask.sum()), **{SHORT[m]: cut_block(bt, D, m, ids, mask) for m in MODELS}}
    # within-class correlation of the item effect with history length
    out["spearman_effect_vs_chars"] = {}
    for m in MODELS:
        d = D.get(m, ids, cond="aligned") - D.get(m, ids, cond="swapped")
        for cls in ("status", "unavailable"):
            cc = np.array([f["cls"] == cls for f in F])
            out["spearman_effect_vs_chars"][f"{SHORT[m]}:{cls}"] = r(spearmanr(d[cc], chars[cc]).statistic)
    # main pool (unmatched included): by tool and status matching, for the tool/action view
    M = D.main
    mids = [t["id"] for t in M]
    btm = Boot([t["user_id"] for t in M])
    mf = [item_features(t) for t in M]
    mcuts = {}
    for c in sorted({(t["cls"], t["action"]["name"], t["target_status"], t["status_matched"]) for t in M}):
        mask = np.array([(t["cls"], t["action"]["name"], t["target_status"], t["status_matched"]) == c for t in M])
        if mask.sum() >= 5:
            mcuts[":".join(map(str, c))] = {"n": int(mask.sum()), **{SHORT[m]: {
                "fail_as": btm.mean(D.get(m, mids, cond="aligned") - D.get(m, mids, cond="swapped"), mask),
                "witness_as": btm.mean(D.get(m, mids, action="witness", cond="aligned") - D.get(m, mids, action="witness", cond="swapped"), mask)}
                for m in MODELS}}
    out["main_pool_cells"] = mcuts
    # natural: position and user mentions
    out["natural"] = q9_natural(D, nat_mentions)
    return out


def q3_specificity(D: Data) -> dict:
    P = D.primary
    ids = [t["id"] for t in P]
    bt = Boot([t["user_id"] for t in P])
    F = [item_features(t) for t in P]
    masks = {"all": np.ones(len(P), bool)}
    for c in ("status", "unavailable", "wrong_return_pm", "gc_balance"):
        masks[c] = np.array([f["cls"] == c for f in F])
    for c in sorted({f["cell"] for f in F}):
        masks[c] = np.array([f["cell"] == c for f in F])
    out = {}
    for m in MODELS:
        o = {}
        for name, mask in masks.items():
            f = {c: D.get(m, ids, cond=c) for c in CONDS}
            w = {c: D.get(m, ids, action="witness", cond=c) for c in CONDS}
            ma = D.get(m, ids, cond="mention_aligned")
            ms = D.get(m, ids, cond="mention_swapped")
            yf = {c: D.get(m, ids, readout="yn", cond=c) for c in CONDS + ("mention_aligned", "mention_swapped")}
            yw = {c: D.get(m, ids, action="witness", readout="yn", cond=c) for c in CONDS}
            fmw = (f["aligned"] - f["swapped"]) - (w["aligned"] - w["swapped"])
            o[name] = {
                "n": int(mask.sum()),
                "first": {
                    "fail": {"as": bt.mean(f["aligned"] - f["swapped"], mask), "an": bt.mean(f["aligned"] - f["neutral"], mask),
                             "sn": bt.mean(f["swapped"] - f["neutral"], mask)},
                    "witness": {"as": bt.mean(w["aligned"] - w["swapped"], mask), "an": bt.mean(w["aligned"] - w["neutral"], mask),
                                "sn": bt.mean(w["swapped"] - w["neutral"], mask)},
                    "fail_minus_witness": bt.mean(fmw, mask),
                    "fail_minus_witness_share_positive": r(float((fmw[mask] > 0).mean())) if mask.any() else None,
                    "ratio_fail_to_witness_as": r(float(np.nanmean((f["aligned"] - f["swapped"])[mask]) / np.nanmean((w["aligned"] - w["swapped"])[mask]))) if mask.any() else None,
                    "mention": {"ma_ms": bt.mean(ma - ms, mask), "ma_n": bt.mean(ma - f["neutral"], mask),
                                "ms_n": bt.mean(ms - f["neutral"], mask), "a_ma": bt.mean(f["aligned"] - ma, mask),
                                "s_ms": bt.mean(f["swapped"] - ms, mask),
                                "write_minus_mention": bt.mean((f["aligned"] - f["swapped"]) - (ma - ms), mask)},
                    "levels": {c: r(float(np.nanmean(f[c][mask]))) for c in CONDS} | {"mention_aligned": r(float(np.nanmean(ma[mask]))),
                               "mention_swapped": r(float(np.nanmean(ms[mask])))} if mask.any() else None,
                    "witness_levels": {c: r(float(np.nanmean(w[c][mask]))) for c in CONDS} if mask.any() else None,
                },
                "yn": {
                    "fail": {"as": bt.mean(yf["aligned"] - yf["swapped"], mask), "an": bt.mean(yf["aligned"] - yf["neutral"], mask),
                             "sn": bt.mean(yf["swapped"] - yf["neutral"], mask)},
                    "witness": {"as": bt.mean(yw["aligned"] - yw["swapped"], mask), "an": bt.mean(yw["aligned"] - yw["neutral"], mask),
                                "sn": bt.mean(yw["swapped"] - yw["neutral"], mask)},
                    "fail_minus_witness": bt.mean((yf["aligned"] - yf["swapped"]) - (yw["aligned"] - yw["swapped"]), mask),
                    "mention": {"ma_ms": bt.mean(yf["mention_aligned"] - yf["mention_swapped"], mask),
                                "ma_n": bt.mean(yf["mention_aligned"] - yf["neutral"], mask),
                                "ms_n": bt.mean(yf["mention_swapped"] - yf["neutral"], mask)},
                },
            }
        out[m] = o
    return out


BASE_BINS = ((-np.inf, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, np.inf))


def q3b_baseline_and_binding(D: Data) -> dict:
    """(1) Fail and witness effects at a matched neutral baseline: is the witness's larger shift a matter of where its
    neutral score sits? (2) The unmatched pool's status items, where the swapped line names an order of the other
    status: does the swapped line act through the status of the order it names?"""
    P = D.primary
    ids = [t["id"] for t in P]
    users = [t["user_id"] for t in P]
    out = {}
    for m in MODELS:
        o = {}
        for ro in ("first", "yn"):
            rows_d, rows_n, rows_t, rows_u = [], [], [], []
            for action, T in (("fail", 0), ("witness", 1)):
                a, s, n = (D.get(m, ids, action=action, readout=ro, cond=c) for c in CONDS)
                rows_d.append(a - s)
                rows_n.append(n)
                rows_t.append(np.full(len(ids), T))
                rows_u += users
            d, n, T = np.concatenate(rows_d), np.concatenate(rows_n), np.concatenate(rows_t)
            bt = Boot(rows_u)
            bins = {}
            for lo, hi in BASE_BINS:
                mk = (n > lo) & (n <= hi)
                bins[f"{lo}..{hi}"] = {"fail": bt.mean(d, mk & (T == 0)), "witness": bt.mean(d, mk & (T == 1))}
            o[ro] = bins
            # one number: witness minus fail, reweighting the witness to the failing calls' distribution over the bins
            w_d = []
            for lo, hi in BASE_BINS:
                mf, mw = (n > lo) & (n <= hi) & (T == 0), (n > lo) & (n <= hi) & (T == 1)
                if mf.sum() >= 10 and mw.sum() >= 10:
                    w_d.append((mf.sum(), d[mw].mean() - d[mf].mean()))
            o[f"{ro}_witness_minus_fail_reweighted"] = r(float(sum(k * v for k, v in w_d) / sum(k for k, _ in w_d))) if w_d else None
            o[f"{ro}_reweighted_coverage"] = int(sum(k for k, _ in w_d))
        out[m] = o
    # binding: main-pool status items, matched against unmatched comparison orders
    M = [t for t in D.main if t["cls"] == "status"]
    mids = [t["id"] for t in M]
    btm = Boot([t["user_id"] for t in M])
    cell = np.array([f"{t['action']['name'].split('_')[0]}-on-{t['target_status']}" for t in M])
    match = np.array([t["status_matched"] for t in M])
    # does the tool's status rule pass for the status of the order the swapped line names?
    swapped_ok = np.array([STATUS_RULE[t["action"]["name"]](t["comparison_status"]) for t in M])
    b = {}
    for m in MODELS:
        a, s, n = (D.get(m, mids, cond=c) for c in CONDS)
        wa, ws = D.get(m, mids, action="witness", cond="aligned"), D.get(m, mids, action="witness", cond="swapped")
        for c in sorted(set(cell)):
            for lab, mk in (("matched", match), ("unmatched, rule passes for the named order", ~match & swapped_ok),
                            ("unmatched, rule fails for the named order", ~match & ~swapped_ok)):
                mm = (cell == c) & mk
                if mm.sum() >= 5:
                    b[f"{SHORT[m]}:{c}:{lab}"] = {"n": int(mm.sum()), "fail_as": btm.mean(a - s, mm), "fail_sn": btm.mean(s - n, mm),
                                                  "witness_as": btm.mean(wa - ws, mm)}
    out["binding_status_main_pool"] = b
    return out


def calls_table(D: Data, model: str, readout: str, cond: str = "neutral"):
    """The primary population as calls: each item's failing call (T=0) and witness (T=1)."""
    rows = []
    for t in D.primary:
        for action, T in (("fail", 0), ("witness", 1)):
            call = t["action"] if action == "fail" else t["aligned_witness"]
            s = D.by.get((model, t["id"], "main", action, readout, cond))
            if s is None:
                continue
            f = call_features(t, call)
            rows.append({"id": t["id"], "user": t["user_id"], "cls": t["cls"], "T": T, "L": s["L"],
                         "mass": math.exp(s["logp_plus"]) + math.exp(s["logp_minus"]),
                         "determined": action == "fail" or t.get("witness_checker") == "PASS", **f})
    return rows


def families_rule(rows, bt_rows) -> dict:
    L = np.array([x["L"] for x in rows])
    T = np.array([x["T"] for x in rows])
    a = auc(L, T)
    return {"auc": a, "yes_on_failures": float((L[T == 0] > 0).mean()), "yes_on_successes": float((L[T == 1] > 0).mean()),
            "median_mass": float(np.median([x["mass"] for x in rows]))}


def q4_competence(D: Data) -> dict:
    out = {}
    for m in MODELS:
        o = {}
        for ro in ("first", "yn"):
            rows = calls_table(D, m, ro)
            users = [x["user"] for x in rows]
            bt = Boot(users)
            L = np.array([x["L"] for x in rows])
            T = np.array([x["T"] for x in rows])
            det = np.array([x["determined"] for x in rows])
            cls = np.array([x["cls"] for x in rows])
            cellname = np.array([f"{x['cls']}:{x['tool'].split('_')[0]}-on-{x['status']}" for x in rows])
            blk = {"auc_all": bt.stat(auc, L, T), "auc_determined": bt.stat(auc, L, T, mask=det)}
            for c in ("status", "unavailable", "wrong_return_pm", "gc_balance"):
                mk = cls == c
                blk[f"auc_{c}"] = bt.stat(auc, L, T, mask=mk) if mk.sum() > 10 else {"est": r(auc(L[mk], T[mk])), "n": int(mk.sum())}
            # paired discrimination per item: witness minus failing call
            byid = defaultdict(dict)
            for x in rows:
                byid[x["id"]][x["T"]] = x["L"]
            pid = [i for i in byid if 0 in byid[i] and 1 in byid[i]]
            pd_ = np.array([byid[i][1] - byid[i][0] for i in pid])
            pcls = np.array([D.items[i]["cls"] for i in pid])
            btp = Boot([D.items[i]["user_id"] for i in pid])
            blk["paired_witness_minus_fail"] = {c: btp.mean(pd_, pcls == c) for c in ("status", "unavailable", "wrong_return_pm", "gc_balance")}
            blk["paired_share_witness_above_fail"] = {c: r(float((pd_[pcls == c] > 0).mean())) for c in ("status", "unavailable", "wrong_return_pm", "gc_balance")}
            # predicted success rates by class and outcome
            blk["share_predicted_success"] = {f"{c}:{'witness' if t_ else 'fail'}": r(float((L[(cls == c) & (T == t_)] > 0).mean()))
                                              for c in ("status", "unavailable", "wrong_return_pm", "gc_balance") for t_ in (0, 1)}
            blk["share_predicted_success_by_cell"] = {f"{c}:{'witness' if t_ else 'fail'}": r(float((L[(cellname == c) & (T == t_)] > 0).mean()))
                                                      for c in sorted(set(cellname)) for t_ in (0, 1)}
            blk["mean_L_by_cell"] = {f"{c}:{'witness' if t_ else 'fail'}": r(float(L[(cellname == c) & (T == t_)].mean()))
                                     for c in sorted(set(cellname)) for t_ in (0, 1)}
            blk["balanced_accuracy"] = r(float(((L[T == 1] > 0).mean() + (L[T == 0] <= 0).mean()) / 2))
            # calibration and Brier
            p = sigmoid(L)
            bins = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0001]
            cal = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mk = (p >= lo) & (p < hi)
                if mk.sum():
                    cal.append({"bin": [lo, min(hi, 1.0)], "n": int(mk.sum()), "mean_p": r(float(p[mk].mean())), "share_success": r(float(T[mk].mean()))})
            blk["calibration"] = cal
            brier = float(((p - T) ** 2).mean())
            base = float(((T.mean() - T) ** 2).mean())
            blk["brier"] = {"model": r(brier), "base_rate": r(base), "skill": r(1 - brier / base)}
            blk["brier_determined"] = {"model": r(float(((p[det] - T[det]) ** 2).mean())),
                                       "base_rate": r(float(((T[det].mean() - T[det]) ** 2).mean()))}
            if ro == "yn":
                fr_all = families_rule(rows, bt)
                fr_det = families_rule([x for x in rows if x["determined"]], bt)
                blk["families_rule"] = {
                    "all_calls": {k: r(v) for k, v in fr_all.items()},
                    "determined_calls": {k: r(v) for k, v in fr_det.items()},
                    "rule": "yes/no, neutral line: AUC >= 0.70, Yes on <= 95% of failures and >= 5% of successes, median mass >= 0.5"}
                for nm, fr in (("all_calls", fr_all), ("determined_calls", fr_det)):
                    blk["families_rule"][nm]["passes"] = bool(fr["auc"] >= 0.70 and fr["yes_on_failures"] <= 0.95
                                                              and fr["yes_on_successes"] >= 0.05 and fr["median_mass"] >= 0.5)
                for c in ("status", "unavailable"):
                    sub = [x for x in rows if x["cls"] == c]
                    fr = families_rule(sub, bt)
                    blk["families_rule"][c] = {k: r(v) for k, v in fr.items()} | {"passes": bool(
                        fr["auc"] >= 0.70 and fr["yes_on_failures"] <= 0.95 and fr["yes_on_successes"] >= 0.05 and fr["median_mass"] >= 0.5)}
            # heuristics
            hf = {}
            for h in ("H1", "H2", "H1andH2", "H3", "H4", "H5"):
                hv = np.array([np.nan if x[h] is None else x[h] for x in rows], float)
                ok = np.isfinite(hv)
                hb = hv.astype(float)
                hf[h] = {"n": int(ok.sum()), "validity": r(float((hb[ok] == T[ok]).mean())),
                         "tracking_auc": bt.stat(lambda s, y: auc(s, y), L, np.where(ok, hb, 0).astype(int), mask=ok),
                         "truth_auc_same_calls": r(auc(L[ok], T[ok])),
                         "kappa_with_model_sign": r(kappa(hb[ok], (L[ok] > 0).astype(int)))}
                # within-truth tracking: among failing calls only, and among witnesses only
                for t_, nm in ((0, "within_fail"), (1, "within_witness")):
                    mk = ok & (T == t_)
                    if len(set(hb[mk])) == 2:
                        hf[h][f"tracking_auc_{nm}"] = r(auc(L[mk], hb[mk].astype(int)))
            o_h6 = {}
            crowd = np.array([np.nan if x["H6"] is None else x["H6"] for x in rows], float)
            for t_, nm in ((0, "fail"), (1, "witness")):
                mk = (cls == "unavailable") & (T == t_) & np.isfinite(crowd)
                o_h6[f"spearman_L_vs_crowd_{nm}"] = r(spearmanr(L[mk], crowd[mk]).statistic)
            mk = (cls == "unavailable") & np.isfinite(crowd)
            X = np.column_stack([np.ones(mk.sum()), T[mk], crowd[mk]])
            o_h6["ols_L_on_T_crowd"] = bt.stat(lambda y, a, c_: ols(y, np.column_stack([np.ones(len(y)), a, c_])), L, T.astype(float), crowd, mask=mk)
            o_h6["crowd_quartiles"] = [r(x) for x in np.percentile(crowd[mk], [25, 50, 75])]
            hf["H6"] = o_h6
            # joint regression on all calls: T and the heuristics
            def joint(y, Tt, h1, h2, h3, h5):
                return ols(y, np.column_stack([np.ones(len(y)), Tt, h3, h5]))
            blk["heuristics"] = hf
            # the status 2 x 2 regression (status and unavailable classes)
            su = np.isin(cls, ["status", "unavailable"])
            dv = np.array([x["delivered"] for x in rows], float)
            ex = np.array([x["exchange"] for x in rows], float)
            h1 = np.array([x["H1"] for x in rows], float)
            un = np.array([1 - x["H2"] for x in rows], float)
            sc = (cls == "status").astype(float)
            gc = np.array([x["gift_card"] for x in rows], float)
            blk["status_2x2_as_planned"] = {
                "terms": ["intercept", "delivered", "exchange", "status_rule_passes", "unavailable_shown"],
                **bt.stat(lambda y, a, b_, c_, d_: ols(y, np.column_stack([np.ones(len(y)), a, b_, c_, d_])), L, dv, ex, h1, un, mask=su)}
            blk["status_2x2_with_history_type"] = {
                "terms": ["intercept", "delivered", "exchange", "status_rule_passes", "unavailable_shown", "status_class_history", "gift_card"],
                **bt.stat(lambda y, a, b_, c_, d_, e_, g_: ols(y, np.column_stack([np.ones(len(y)), a, b_, c_, d_, e_, g_])), L, dv, ex, h1, un, sc, gc, mask=su)}
            # within the status class only: the four cells
            st = cls == "status"
            blk["status_class_cells"] = {
                "terms": ["intercept", "delivered", "exchange", "status_rule_passes"],
                **bt.stat(lambda y, a, b_, c_: ols(y, np.column_stack([np.ones(len(y)), a, b_, c_])), L, dv, ex, h1, mask=st)}
            blk["gift_card_effect_unavailable_fail"] = bt.mean(np.where(gc == 1, L, np.nan), (cls == "unavailable") & (T == 0))
            blk["non_gift_card_unavailable_fail"] = bt.mean(np.where(gc == 0, L, np.nan), (cls == "unavailable") & (T == 0))
            o[ro] = blk
        out[m] = o
    return out


def q5_readouts(D: Data) -> dict:
    P = D.primary
    ids = [t["id"] for t in P]
    users = [t["user_id"] for t in P]
    bt = Boot(users)
    cls = np.array([t["cls"] for t in P])
    out = {}
    for m in MODELS:
        o = {}
        for action in ("fail", "witness"):
            F = {c: D.get(m, ids, action=action, cond=c) for c in CONDS}
            Y = {c: D.get(m, ids, action=action, readout="yn", cond=c) for c in CONDS}
            N = {c: D.get(m, ids, action=action, readout="full", cond=c, kind="norm") for c in CONDS}
            FL = {c: D.get(m, ids, action=action, readout="full", cond=c) for c in CONDS}
            TL = {c: D.get(m, ids, action=action, readout="full", cond=c, kind="tail") for c in CONDS}
            dF, dY, dN, dFL, dT = (F["aligned"] - F["swapped"], Y["aligned"] - Y["swapped"], N["aligned"] - N["swapped"],
                                   FL["aligned"] - FL["swapped"], TL["aligned"] - TL["swapped"])
            TP = {c: D.get(m, ids, action=action, readout="full", cond=c, kind="tail_plus") for c in CONDS}
            TM = {c: D.get(m, ids, action=action, readout="full", cond=c, kind="tail_minus") for c in CONDS}
            blk = {}
            for name, mask in (("all", np.ones(len(P), bool)), ("status", cls == "status"), ("unavailable", cls == "unavailable")):
                blk[name] = {
                    "first": bt.mean(dF, mask), "yn": bt.mean(dY, mask), "full_norm": bt.mean(dN, mask), "full": bt.mean(dFL, mask),
                    "full_tail": bt.mean(dT, mask),
                    "success_tail_aligned_minus_swapped": bt.mean(TP["aligned"] - TP["swapped"], mask),
                    "failure_tail_aligned_minus_swapped": bt.mean(TM["aligned"] - TM["swapped"], mask),
                    "yn_minus_first": bt.mean(dY - dF, mask),
                    "spearman_first_yn": r(spearmanr(dF[mask], dY[mask]).statistic),
                    "spearman_first_fullnorm": r(spearmanr(dF[mask], dN[mask]).statistic),
                    "sign_agree_first_yn": r(float((np.sign(dF[mask]) == np.sign(dY[mask])).mean())),
                    "share_yn_up_first_not_up": r(float(((dY[mask] > 0) & (dF[mask] <= 0)).mean())),
                    "share_first_up_yn_not_up": r(float(((dF[mask] > 0) & (dY[mask] <= 0)).mean())),
                    "sd_first": r(float(np.nanstd(dF[mask], ddof=1))), "sd_yn": r(float(np.nanstd(dY[mask], ddof=1))),
                    "level_sd_yn_neutral": r(float(np.nanstd(Y["neutral"][mask], ddof=1))),
                    "level_sd_first_neutral": r(float(np.nanstd(F["neutral"][mask], ddof=1))),
                    "mean_levels_first": {c: r(float(np.nanmean(F[c][mask]))) for c in CONDS},
                    "mean_levels_yn": {c: r(float(np.nanmean(Y[c][mask]))) for c in CONDS},
                }
                # matched-first-token regression: Y_ig = c_g + b F_ig, pooled over lines
                rows_y = np.concatenate([Y[c][mask] for c in CONDS])
                rows_f = np.concatenate([F[c][mask] for c in CONDS])
                ga = np.concatenate([np.full(mask.sum(), c == "aligned", float) for c in CONDS])
                gs = np.concatenate([np.full(mask.sum(), c == "swapped", float) for c in CONDS])
                uu = np.concatenate([np.array(users)[mask]] * 3)
                bt3 = Boot(list(uu))
                res = bt3.stat(lambda y, f_, a_, s_: ols(y, np.column_stack([np.ones(len(y)), f_, a_, s_])), rows_y, rows_f, ga, gs)
                est, ci = res["est"], res["ci"]
                blk[name]["matched_first_regression"] = {"slope": [est[1], ci[1]], "aligned_minus_neutral": [est[2], ci[2]],
                                                         "swapped_minus_neutral": [est[3], ci[3]]}
                # aligned minus swapped at matched first token: c_a - c_s, from the same fit
                res2 = bt3.stat(lambda y, f_, a_, s_: (lambda b_: b_[2] - b_[3])(ols(y, np.column_stack([np.ones(len(y)), f_, a_, s_]))),
                                rows_y, rows_f, ga, gs)
                blk[name]["matched_first_regression"]["aligned_minus_swapped"] = [res2["est"], res2["ci"]]
                # dissociation: share answered No among calls whose first token confidently predicts success (>= +3)
                blk[name]["share_no_given_first_ge3"] = {c: [r(float((Y[c][mask & (F[c] >= 3)] < 0).mean())) if (mask & (F[c] >= 3)).any() else None,
                                                             int((mask & (F[c] >= 3)).sum())] for c in CONDS}
                blk[name]["share_yes_given_first_le_minus3"] = {c: [r(float((Y[c][mask & (F[c] <= -3)] > 0).mean())) if (mask & (F[c] <= -3)).any() else None,
                                                                    int((mask & (F[c] <= -3)).sum())] for c in CONDS}
            o[action] = blk
        out[m] = o
    return out


def q6_practical(D: Data) -> dict:
    P = D.primary
    ids = [t["id"] for t in P]
    bt = Boot([t["user_id"] for t in P])
    cls = np.array([t["cls"] for t in P])
    out = {}
    for m in MODELS:
        o = {}
        for ro in ("first", "yn"):
            for action in ("fail", "witness"):
                V = {c: D.get(m, ids, action=action, readout=ro, cond=c) for c in CONDS}
                blk = {}
                for name, mask in (("all", np.ones(len(P), bool)), ("status", cls == "status"), ("unavailable", cls == "unavailable")):
                    a, s, n = V["aligned"], V["swapped"], V["neutral"]
                    blk[name] = {
                        "share_success_predicted": {c: bt.mean((V[c] > 0).astype(float), mask) for c in CONDS},
                        "flip_to_success_s_to_a": bt.mean(((s < 0) & (a > 0)).astype(float), mask),
                        "flip_to_failure_s_to_a": bt.mean(((s > 0) & (a < 0)).astype(float), mask),
                        "neutral_to_aligned_to_success": bt.mean(((n < 0) & (a > 0)).astype(float), mask),
                        "neutral_to_aligned_to_failure": bt.mean(((n > 0) & (a < 0)).astype(float), mask),
                        "neutral_to_swapped_to_success": bt.mean(((n < 0) & (s > 0)).astype(float), mask),
                        "neutral_to_swapped_to_failure": bt.mean(((n > 0) & (s < 0)).astype(float), mask),
                        "probability_shift_points": bt.mean(100 * (sigmoid(a) - sigmoid(s)), mask),
                        "mean_p_success": {c: r(float(sigmoid(V[c][mask]).mean())) for c in CONDS},
                        "exact_zero_L_share": {c: r(float((V[c][mask] == 0).mean())) for c in CONDS},
                    }
                    # decision-relevant: correct and confident under neutral (|L| >= log 99 on the right side)
                    right = (n < -CONF) if action == "fail" else (n > CONF)
                    mk = mask & right
                    blk[name]["confident_correct_under_neutral"] = {
                        "n": int(mk.sum()),
                        "turned_wrong_aligned": r(float(((a > 0) if action == "fail" else (a < 0))[mk].mean())) if mk.any() else None,
                        "turned_wrong_swapped": r(float(((s > 0) if action == "fail" else (s < 0))[mk].mean())) if mk.any() else None}
                o[f"{ro}:{action}"] = blk
        out[m] = o
    return out


def q7_arms(D: Data) -> dict:
    out = {}
    for m in MODELS:
        arm_ids = sorted({k[1] for k in D.by if k[0] == m and k[2] == "matched"})
        items = [D.items[i] for i in arm_ids]
        bt = Boot([t["user_id"] for t in items])
        cls = np.array([t["cls"] for t in items])
        o = {"n_items": len(items), "n_users": len({t["user_id"] for t in items}),
             "same_items_in_rules_only": sorted({k[1] for k in D.by if k[0] == m and k[2] == "rules_only"}) == arm_ids}
        for name, mask in (("all", np.ones(len(items), bool)), ("status", cls == "status"), ("unavailable", cls == "unavailable")):
            blk = {}
            for ro in ("first", "yn"):
                for arm in ("main", "matched", "rules_only"):
                    V = {c: D.get(m, arm_ids, arm=arm, readout=ro, cond=c) for c in CONDS}
                    blk[f"{ro}:{arm}:fail"] = {"as": bt.mean(V["aligned"] - V["swapped"], mask), "an": bt.mean(V["aligned"] - V["neutral"], mask),
                                               "sn": bt.mean(V["swapped"] - V["neutral"], mask),
                                               "levels": {c: r(float(np.nanmean(V[c][mask]))) for c in CONDS}}
                for arm in ("main", "matched"):
                    if ro == "yn" and arm == "matched":
                        continue
                    W = {c: D.get(m, arm_ids, arm=arm, action="witness", readout=ro, cond=c) for c in CONDS}
                    if np.isfinite(W["aligned"]).any():
                        blk[f"{ro}:{arm}:witness"] = {"as": bt.mean(W["aligned"] - W["swapped"], mask), "an": bt.mean(W["aligned"] - W["neutral"], mask),
                                                      "sn": bt.mean(W["swapped"] - W["neutral"], mask),
                                                      "levels": {c: r(float(np.nanmean(W[c][mask]))) for c in CONDS}}
                # arm differences per condition level (neutral included): does the statement move the neutral level?
                for arm in ("matched", "rules_only"):
                    for c in CONDS:
                        blk[f"{ro}:{arm}_minus_main:{c}"] = bt.mean(D.get(m, arm_ids, arm=arm, readout=ro, cond=c) - D.get(m, arm_ids, readout=ro, cond=c), mask)
            # discrimination in the matched arm vs main (first token, neutral line)
            fm = D.get(m, arm_ids, arm="matched", cond="neutral")
            wm = D.get(m, arm_ids, arm="matched", action="witness", cond="neutral")
            f0 = D.get(m, arm_ids, cond="neutral")
            w0 = D.get(m, arm_ids, action="witness", cond="neutral")
            det = np.array([t.get("witness_checker") == "PASS" for t in items])
            blk["discrimination_first_neutral"] = {"main": bt.mean(w0 - f0, mask & det), "matched": bt.mean(wm - fm, mask & det)}
            o[name] = blk
        out[m] = o
    return out


def q9_natural(D: Data, mentions: dict | None) -> dict:
    N = D.natural
    ids = [t["id"] for t in N]
    bt = Boot([t["user_id"] for t in N])
    men = {x["id"]: x for x in (mentions or [])}
    out = {"n": len(N), "n_users": len({t["user_id"] for t in N}),
           "by_class_tool": dict(Counter(f"{t['cls']}:{t['action']['name']}:{t['target_status']}" for t in N)),
           "aligned_achievable": sum(bool(t.get("aligned_achievable")) for t in N),
           "with_swapped": sum(bool(t.get("task_swapped")) for t in N),
           "status_matched": sum(bool(t.get("status_matched")) for t in N)}
    if mentions:
        out["mentions_counts"] = {k: sum(bool(men[i][k]) for i in ids) for k in men[ids[0]] if k.startswith(("user_", "scenario_"))}
        out["reward_counts"] = dict(Counter(str(men[i]["reward"]) for i in ids))
    step = np.array([t["provenance"]["step"] for t in N], float)
    out["step_quartiles"] = [r(x) for x in np.percentile(step, [25, 50, 75])]
    for m in MODELS:
        o = {}
        V = {c: D.get(m, ids, cond=c) for c in CONDS}
        Y = {c: D.get(m, ids, readout="yn", cond=c) for c in CONDS}
        W = {c: D.get(m, ids, action="witness", cond=c) for c in CONDS}
        allm = np.ones(len(N), bool)
        o["first"] = {"as": bt.mean(V["aligned"] - V["swapped"]), "an": bt.mean(V["aligned"] - V["neutral"]), "sn": bt.mean(V["swapped"] - V["neutral"])}
        o["yn"] = {"as": bt.mean(Y["aligned"] - Y["swapped"]), "an": bt.mean(Y["aligned"] - Y["neutral"])}
        o["witness_first"] = {"as": bt.mean(W["aligned"] - W["swapped"]), "an": bt.mean(W["aligned"] - W["neutral"])}
        o["share_predicted_success_neutral_first"] = r(float((V["neutral"] > 0).mean()))
        o["share_predicted_success_aligned_first"] = r(float((V["aligned"] > 0).mean()))
        o["share_predicted_success_neutral_yn"] = r(float((Y["neutral"] > 0).mean()))
        o["witness_share_success_neutral_first"] = r(float((W["neutral"][np.isfinite(W["neutral"])] > 0).mean())) if np.isfinite(W["neutral"]).any() else None
        o["neutral_mean_first_fail"] = r(float(np.nanmean(V["neutral"])))
        o["neutral_mean_first_witness"] = r(float(np.nanmean(W["neutral"]))) if np.isfinite(W["neutral"]).any() else None
        # natural status/exchange-on-pending items, beside the generated status cell of the same kind
        exc = np.array([t["cls"] == "status" and t["action"]["name"] == "exchange_delivered_order_items" for t in N])
        o["status_exchange_items"] = {"n": int(exc.sum()), "an": bt.mean(V["aligned"] - V["neutral"], exc), "as": bt.mean(V["aligned"] - V["swapped"], exc)}
        if mentions:
            um = np.array([bool(men[i]["user_mentions_target_order"]) for i in ids])
            up = np.array([bool(men[i]["user_mentions_target_product"]) for i in ids])
            o["by_user_mentions_target_order"] = {str(v): {"n": int((um == v).sum()), "an": r(float(np.nanmean((V["aligned"] - V["neutral"])[um == v]))),
                                                           "as": r(float(np.nanmean((V["aligned"] - V["swapped"])[um == v])))} for v in (True, False)}
            o["by_user_mentions_target_product"] = {str(v): {"n": int((up == v).sum()), "an": r(float(np.nanmean((V["aligned"] - V["neutral"])[up == v]))),
                                                             "as": r(float(np.nanmean((V["aligned"] - V["swapped"])[up == v])))} for v in (True, False)}
        o["spearman_an_vs_step"] = r(spearmanr(V["aligned"] - V["neutral"], step).statistic)
        o["items"] = [{"id": t["id"], "cls": t["cls"], "tool": t["action"]["name"], "status": t["target_status"], "step": t["provenance"]["step"],
                       "neutral": r(V["neutral"][i], 2), "aligned": r(V["aligned"][i], 2), "swapped": r(V["swapped"][i], 2),
                       "yn_neutral": r(Y["neutral"][i], 2)} for i, t in enumerate(N)]
        out[m] = o
    return out


def q8_crossenv(xenv: dict | None, q1: dict, q3: dict) -> dict:
    if not xenv:
        return {}
    out = {}
    for m in MODELS:
        s = xenv[m]["scales"]
        g = lambda env, k: s[env][k]["est"] if isinstance(s[env].get(k), dict) else s[env].get(k)
        out[m] = {
            "alfworld": {"first_as": g("alfworld", "dL_first"), "first_an": g("alfworld", "aligned_minus_neutral_first"),
                         "first_sn": g("alfworld", "swapped_minus_neutral_first"), "yn_as": g("alfworld", "dL_yes_no"),
                         "yn_an": g("alfworld", "yes_no_aligned_minus_neutral"), "yn_sn": g("alfworld", "yes_no_swapped_minus_neutral"),
                         "success_two_candidate_as": xenv[m]["successes"]["alfworld"]["dL"]["est"],
                         "success_yn_as": xenv[m]["successes"]["alfworld"]["dL_yes_no"]["est"],
                         "failure_two_candidate_as": g("alfworld", "dL")},
            "textcraft": {"first_as": g("textcraft_main", "dL_first"), "first_an": g("textcraft_main", "aligned_minus_neutral_first"),
                          "first_sn": g("textcraft_main", "swapped_minus_neutral_first"), "yn_as": g("textcraft_main", "dL_yes_no"),
                          "yn_an": g("textcraft_main", "yes_no_aligned_minus_neutral"), "yn_sn": g("textcraft_main", "yes_no_swapped_minus_neutral"),
                          "success_two_candidate_as": xenv[m]["successes"]["textcraft"]["dL"]["est"],
                          "success_yn_as": xenv[m]["successes"]["textcraft"]["dL_yes_no"]["est"],
                          "failure_two_candidate_as": g("textcraft_main", "dL")},
            "tau": {"first_as": q1[m]["first_L"]["aligned_minus_swapped"]["est"], "first_an": q1[m]["first_L"]["aligned_minus_neutral"]["est"],
                    "first_sn": q1[m]["first_L"]["swapped_minus_neutral"]["est"], "yn_as": q1[m]["yn_L"]["aligned_minus_swapped"]["est"],
                    "yn_an": q1[m]["yn_L"]["aligned_minus_neutral"]["est"], "yn_sn": q1[m]["yn_L"]["swapped_minus_neutral"]["est"],
                    "witness_first_as": q1[m]["witness_first"]["aligned_minus_swapped"]["est"],
                    "witness_yn_as": q1[m]["witness_yn"]["aligned_minus_swapped"]["est"]},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--natural-mentions", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--xenv", type=Path, help="notes/textcraft_vs_alfworld_metrics.json from origin/xenv-compare")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    D = Data(a.scores, a.build)
    ref = json.loads(a.reference.read_text()) if a.reference else None
    men = json.loads(a.natural_mentions.read_text()) if a.natural_mentions else None
    xenv = json.loads(a.xenv.read_text()) if a.xenv else None
    only = set(a.only.split(",")) if a.only else None
    res = {"note": "exploratory, post hoc (notes/tau_dx.md); 95% percentile intervals, 2,000 resamples of whole users, seed 20260924",
           "inputs": {"build_sha256": D.hashes, "scoring_records": D.n_records, "duplicate_keys": D.duplicates, "shard_git_shas": D.shard_shas,
                      "primary_items": len(D.primary), "primary_users": len({t['user_id'] for t in D.primary})}}
    if a.out.exists() and only:
        res = json.loads(a.out.read_text()) | {"inputs": res["inputs"]}
    run = lambda k: (only is None) or (k in only)
    if run("q1"):
        res["q1_recompute"] = q1_recompute(D, ref)
    if run("q2"):
        res["q2_heterogeneity"] = q2_heterogeneity(D, men)
    if run("q3"):
        res["q3_specificity"] = q3_specificity(D)
    if run("q3b"):
        res["q3b_baseline_and_binding"] = q3b_baseline_and_binding(D)
    if run("q4"):
        res["q4_competence"] = q4_competence(D)
    if run("q5"):
        res["q5_readouts"] = q5_readouts(D)
    if run("q6"):
        res["q6_practical"] = q6_practical(D)
    if run("q7"):
        res["q7_arms"] = q7_arms(D)
    if run("q8") and "q1_recompute" in res:
        res["q8_crossenv"] = q8_crossenv(xenv, res["q1_recompute"], res.get("q3_specificity", {}))
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
