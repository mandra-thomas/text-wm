"""Exploratory, post hoc robustness audit of Qwen3-8B's positive TextCraft goal effect (notes/textcraft_qwen_robustness.md).

Nothing here is part of the frozen analysis (wf/textcraft/analysis.py) and nothing here scores anything. It re-reads the
confirmatory scoring records of `textcraft_score` and the exploratory prompt-variant records (examples on goal, empty
start, zero-shot, object goal, clean / inventory / clean + inventory), checks every record's item hash against the
prompt it claims, and computes descriptive estimates. Llama-3.1-8B is computed alongside on every cut.

Every interval is a 95% percentile interval from 2,000 resamples of whole goal pairs (one shared set of resampling
weights per pair set, seed 20260924), or of whole recipe families where stated. Robust statistics (trimmed and
winsorized means, medians) are recomputed inside each resample with the resample's weights. No p-values.

Usage (from the repository root, with the score copies verified by sha256 against ICE):
    python scripts/textcraft_qwen_robust.py --scores <dir with the textcraft_score shards> \
        --variants <dir holding examples_on_goal/ empty_start/ zeroshot/ objectgoal/ prompts/ [generate_qwen/]> \
        --out notes/textcraft_qwen_robustness_metrics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from textcraft_llama_dx import read_scores  # noqa: E402
from wf.textcraft import dx as X  # noqa: E402
from wf.textcraft import prompts as P  # noqa: E402
from wf.textcraft.checker import parse_craft  # noqa: E402
from wf.textcraft.score import item_hash, variant_context  # noqa: E402

ITEMS = REPO / "wf" / "textcraft" / "data" / "items.jsonl"
MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
SHORT = {"qwen3-8b-fewshot": "qwen", "llama3.1-8b-fewshot": "llama"}
SEED = 20260924
B = 2000
CLASSES = ("never_obtained", "too_few", "used_up")
CELLS = (("goal", "never_obtained"), ("goal", "too_few"), ("ingredient", "never_obtained"),
         ("ingredient", "too_few"), ("ingredient", "used_up"))
BINS = ("1-3", "4-8", "9-12")


# ------------------------------------------------------------------ resampling and weighted statistics

class Boot:
    """Shared cluster-resampling weights: W(clusters) is a (B, n) matrix of item weights."""

    def __init__(self, clusters, b: int = B, seed: int = SEED):
        self.keys = sorted(set(clusters))
        self.k = {c: i for i, c in enumerate(self.keys)}
        rng = np.random.default_rng(seed)
        self.C = rng.multinomial(len(self.keys), [1.0 / len(self.keys)] * len(self.keys), size=b).astype(float)

    def W(self, clusters) -> np.ndarray:
        return self.C[:, [self.k[c] for c in clusters]]


def _r(x, k: int = 4):
    x = float(x)
    return None if x != x else round(x, k)


def _pack(point, draws, n, clusters, **extra):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan)
    out = {"est": _r(point), "ci": [_r(lo), _r(hi)], "n": int(n), "clusters": int(clusters)}
    out.update(extra)
    return out


def _sorted_w(v, W):
    o = np.argsort(v, kind="mergesort")
    return v[o], W[:, o]


def w_mean(v, W):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ v) / W.sum(1)


def w_trim(v, W, a):
    """Weighted a-trimmed mean (a from each tail), fractional weights at the cut points."""
    s, Ws = _sorted_w(v, W)
    cw = np.cumsum(Ws, 1)
    tot = cw[:, -1:]
    lo, hi = a * tot, (1 - a) * tot
    part = np.clip(np.minimum(cw, hi) - np.maximum(cw - Ws, lo), 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (part @ s) / part.sum(1)


def w_quant(v, W, q):
    s, Ws = _sorted_w(v, W)
    cw = np.cumsum(Ws, 1)
    tot = cw[:, -1]
    idx = np.array([np.searchsorted(cw[i], q * tot[i]) for i in range(cw.shape[0])])
    idx = np.clip(idx, 0, len(s) - 1)
    return s[idx]


def w_median(v, W):
    return w_quant(v, W, 0.5)


def w_wins(v, W, a):
    lo, hi = w_quant(v, W, a), w_quant(v, W, 1 - a)
    cl = np.clip(v[None, :], lo[:, None], hi[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * cl).sum(1) / W.sum(1)


def w_share_pos(v, W):
    return w_mean((v > 0).astype(float), W)


def stat_ci(v, clusters, bt: Boot, fn=w_mean, **extra) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return {"est": None, "ci": [None, None], "n": 0, "clusters": 0}
    point = fn(v, np.ones((1, v.size)))[0]
    return _pack(point, fn(v, bt.W(clusters)), v.size, len(set(clusters)), **extra)


def mean_ci(v, clusters, bt):
    return stat_ci(v, clusters, bt)


def robust(v, clusters, bt) -> dict:
    """Mean, median, 10% and 20% trimmed means, 5% and 10% winsorized means, share above zero, and the item SD and
    standardized mean (both point estimates only)."""
    v = np.asarray(v, float)
    out = {"mean": stat_ci(v, clusters, bt), "median": stat_ci(v, clusters, bt, w_median),
           "trim10": stat_ci(v, clusters, bt, lambda x, W: w_trim(x, W, 0.10)),
           "trim20": stat_ci(v, clusters, bt, lambda x, W: w_trim(x, W, 0.20)),
           "wins05": stat_ci(v, clusters, bt, lambda x, W: w_wins(x, W, 0.05)),
           "wins10": stat_ci(v, clusters, bt, lambda x, W: w_wins(x, W, 0.10)),
           "share_pos": stat_ci(v, clusters, bt, w_share_pos),
           "share_neg": stat_ci(-v, clusters, bt, w_share_pos)}
    if v.size > 1:
        out["sd"] = _r(v.std(ddof=1))
        out["d"] = _r(v.mean() / v.std(ddof=1)) if v.std() > 0 else None
    return out


def diff_ci(v1, c1, v2, c2, bt) -> dict:
    """Mean of v1 minus mean of v2 (two item sets resampled with the same cluster weights)."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    point = v1.mean() - v2.mean()
    draws = w_mean(v1, bt.W(c1)) - w_mean(v2, bt.W(c2))
    return _pack(point, draws, v1.size + v2.size, len(set(c1) | set(c2)))


def wauc(s, y, W):
    """Weighted AUC P(score of a positive > score of a negative), ties one half, for each row of W."""
    order = np.argsort(s, kind="mergesort")
    s, y, W = s[order], y[order], W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    Wp = np.add.reduceat(W * y, starts, axis=1)
    Wn = np.add.reduceat(W * (1 - y), starts, axis=1)
    below = np.cumsum(Wn, axis=1) - Wn
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wp * (below + 0.5 * Wn)).sum(1) / (Wp.sum(1) * Wn.sum(1))


def auc_ci(s, y, clusters, bt) -> dict:
    s, y = np.asarray(s, float), np.asarray(y, float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"est": None, "ci": [None, None], "n": int(len(y)), "clusters": len(set(clusters))}
    return _pack(wauc(s, y, np.ones((1, len(s))))[0], wauc(s, y, bt.W(clusters)), len(s), len(set(clusters)),
                 n_pos=int(y.sum()), n_neg=int((1 - y).sum()))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ items, families, overlap

def words(s: str) -> list[str]:
    return s.lower().replace("_", " ").split()


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _contig(a: list[str], b: list[str]) -> bool:
    n = len(a)
    return any(b[i:i + n] == a for i in range(len(b) - n + 1))


def overlap_category(obj: str, goal: str, stem: bool = False) -> str:
    """'identical' (the goal itself), 'contained' (the object's full name is a run of words of the goal's name),
    'word_only' (they share a word, not contained), 'none' (no shared word). With stem=True a final plural s is
    dropped from every word first (so "nether bricks" is contained in "nether brick fence")."""
    o, g = words(obj), words(goal)
    if stem:
        o, g = [_stem(w) for w in o], [_stem(w) for w in g]
    if o == g:
        return "identical"
    if _contig(o, g):
        return "contained"
    if set(o) & set(g):
        return "word_only"
    return "none"


def families(items: list[dict]) -> tuple[dict, dict, dict]:
    """The two recipe-family definitions fixed in notes/textcraft_llama_dx.md (plan 7) before any effect was computed
    on them: connected components of pairs sharing a craftable, and the primary material (the non-craftable input on
    the most lines of the pair's displayed list, ties alphabetical)."""
    lists = {}
    for t in items:
        lists.setdefault(t["pair_id"], t["recipe_lines"])
    outs = {p: {parse_craft(l)[1] for l in ls} for p, ls in lists.items()}
    par = {p: p for p in lists}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    ps = sorted(lists)
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            if outs[a] & outs[b]:
                par[find(a)] = find(b)
    comp = {p: find(p) for p in ps}

    def primary(ls):
        o = {parse_craft(l)[1] for l in ls}
        cnt = Counter(n for l in ls for _c, n in parse_craft(l)[2] if n not in o)
        return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    fam = {p: primary(ls) for p, ls in lists.items()}
    return comp, fam, outs


# ------------------------------------------------------------------ loading and provenance

def load_variant(root: Path, name: str, items: list[dict], shots) -> tuple[list[dict], dict]:
    """Records of one exploratory run with each record's item hash recomputed for the prompt it claims."""
    recs = read_scores(root / name) if name != "generate_qwen" else [
        json.loads(l) for l in (root / name / "generations.jsonl").read_text().splitlines() if l.strip()]
    I = {t["id"]: t for t in items}
    if name in ("examples_on_goal", "empty_start"):
        vs, vsys = variant_context(name, shots)
        h = lambda t, c: item_hash(t, vs, vsys)  # noqa: E731
    elif name == "zeroshot":
        h = lambda t, c: item_hash(t, (), P.SYSTEM)  # noqa: E731
    elif name == "objectgoal":
        h = lambda t, c: item_hash(X.object_goal_item(t), shots, P.SYSTEM)  # noqa: E731
    elif name == "prompts":
        h = lambda t, c: item_hash(X.variant_item(t, c.split("|")[0]), shots, X.variant_system(c.split("|")[0]))  # noqa: E731
    else:  # generation: the observation prompt's hash (verified prompt)
        h = lambda t, c: item_hash(t, shots, P.SYSTEM)  # noqa: E731
    cache = {}
    stale = 0
    for s in recs:
        key = (s["transition_id"], s["condition"].split("|")[0] if name == "prompts" else "")
        if key not in cache:
            cache[key] = h(I[s["transition_id"]], s["condition"])
        stale += cache[key] != s["item_hash"]
    return recs, {"records": len(recs), "stale": int(stale), "items": len({s["transition_id"] for s in recs})}


def table(items, by: dict, model: str, fam: dict, comp: dict) -> list[dict]:
    """One row per main / success / usefulness item with every arm the confirmatory run scored."""
    rows = []
    for t in items:
        if t["population"] not in ("main", "success", "usefulness"):
            continue
        g = lambda c: by[(model, t["id"], c)]  # noqa: E731
        r = {"id": t["id"], "pair": t["pair_id"], "couple": t["couple_id"], "pop": t["population"],
             "role": t["role"], "cls": t["failure_class"], "bin": t["length_bin"], "hl": t["history_len"],
             "depth": t["depth"], "fit": t["history_fit"], "fam": fam[t["pair_id"]], "comp": comp[t["pair_id"]],
             "ov": overlap_category(t["object"], t["goal"]), "ov_stem": overlap_category(t["object"], t["goal"], True),
             "pov": bool(t["object_partner_overlap"]), "completion": t["fit_measures"]["completion"][0],
             "n_short": len(t["short_inputs"]),
             "repeat": any(o.startswith("Crafted") and o.endswith("minecraft:" + t["object"].replace(" ", "_"))
                           for _a, o in t["history"]),
             "short_craftable": (all(x["name"] in {parse_craft(l)[1] for l in t["recipe_lines"]} for x in t["short_inputs"])
                                 if t["short_inputs"] else None)}
        for c, k in (("aligned", "a"), ("swapped", "s"), ("neutral", "n")):
            r["L" + k] = g(c)["L"]
            r["F" + k] = g(c)["L_first"]
            r["Y" + k] = g("yn_" + c)["L"]
            r["M" + k] = g(c)["first_mass"]
        r["Pa"], r["Ps"] = g("prefix_aligned")["L"], g("prefix_swapped")["L"]
        rows.append(r)
    return rows


def arr(rows, f):
    return np.array([f(r) for r in rows], float)


def pairs_of(rows):
    return [r["pair"] for r in rows]


# ------------------------------------------------------------------ check 1: recompute and concentration

def check_recompute(rows, bt, frozen):
    main = [r for r in rows if r["pop"] == "main"]
    d = arr(main, lambda r: r["La"] - r["Ls"])
    cl = pairs_of(main)
    out = {
        "delta_L": mean_ci(d, cl, bt),
        "yes_no": mean_ci(arr(main, lambda r: r["Ya"] - r["Ys"]), cl, bt),
        "first_token": mean_ci(arr(main, lambda r: r["Fa"] - r["Fs"]), cl, bt),
        "prefix": mean_ci(arr(main, lambda r: r["Pa"] - r["Ps"]), cl, bt),
        "name_remainder": mean_ci(arr(main, lambda r: (r["La"] - r["Pa"]) - (r["Ls"] - r["Ps"])), cl, bt),
        "aligned_minus_neutral": mean_ci(arr(main, lambda r: r["La"] - r["Ln"]), cl, bt),
        "swapped_minus_neutral": mean_ci(arr(main, lambda r: r["Ls"] - r["Ln"]), cl, bt),
        "delta_p_pp": mean_ci(100 * (sig(arr(main, lambda r: r["La"])) - sig(arr(main, lambda r: r["Ls"]))), cl, bt),
        "goal_role": mean_ci(d[[r["role"] == "goal" for r in main]], [r["pair"] for r in main if r["role"] == "goal"], bt),
        "first_mass_mean": {c: _r(np.mean([r["M" + c] for r in main])) for c in "asn"},
    }
    out["frozen"] = {k: {"est": _r(frozen["descriptive"]["main"][k]["estimate"]),
                         "ci": [_r(x) for x in frozen["descriptive"]["main"][k]["ci"]]}
                     for k in ("delta_L", "yes_no_aligned_minus_swapped", "delta_L_first", "delta_L_prefix",
                               "delta_L_name_remainder", "aligned_minus_neutral", "swapped_minus_neutral", "delta_p_pp")}
    out["robust"] = robust(d, cl, bt)
    out["robust_yes_no"] = robust(arr(main, lambda r: r["Ya"] - r["Ys"]), cl, bt)
    out["robust_first_token"] = robust(arr(main, lambda r: r["Fa"] - r["Fs"]), cl, bt)
    # shards: the frozen run split goal pairs over four jobs ([i::4] of the sorted pair ids)
    ps = sorted({r["pair"] for r in main})
    shard = {p: i % 4 for i, p in enumerate(ps)}
    out["by_shard"] = {str(k): _r(np.mean([r["La"] - r["Ls"] for r in main if shard[r["pair"]] == k])) for k in range(4)}
    return out


def check_concentration(rows, bt_main):
    main = [r for r in rows if r["pop"] == "main"]
    d = arr(main, lambda r: r["La"] - r["Ls"])
    cl = np.array(pairs_of(main))
    n = d.size
    out = {"n_items": n, "item_sd": _r(d.std(ddof=1)), "item_quantiles": {str(q): _r(np.percentile(d, q))
                                                                          for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
           "share_items_pos": _r((d > 0).mean()), "share_items_neg": _r((d < 0).mean()),
           "share_items_zero": _r((d == 0).mean()),
           "share_abs_below_0_5": _r((np.abs(d) < 0.5).mean())}
    # pair means
    pm = {p: d[cl == p].mean() for p in sorted(set(cl))}
    psum = {p: d[cl == p].sum() for p in sorted(set(cl))}
    v = np.array(list(pm.values()))
    out["pairs"] = {"n": len(v), "share_pos": _r((v > 0).mean()), "quartiles": [_r(x) for x in np.percentile(v, [25, 50, 75])],
                    "min": _r(v.min()), "max": _r(v.max())}
    # couples
    cpl = defaultdict(list)
    for r, x in zip(main, d):
        cpl[r["couple"]].append(x)
    cm = np.array([np.mean(x) for x in cpl.values()])
    out["couples"] = {"n": len(cm), "complete": int(sum(len(x) == 2 for x in cpl.values())),
                      "share_pos": _r((cm > 0).mean()), "median": _r(np.median(cm))}
    # leave one pair out
    loo = {p: (d.sum() - psum[p]) / (n - (cl == p).sum()) for p in pm}
    s = sorted(loo.items(), key=lambda kv: kv[1])
    out["leave_one_pair_out"] = {"min": [s[0][0], _r(s[0][1])], "max": [s[-1][0], _r(s[-1][1])]}
    # leave the top-k pairs out (by contribution to the sum), with intervals over the remaining pairs
    order = sorted(psum, key=lambda p: -psum[p])
    out["top_pairs_by_contribution"] = [[p, _r(psum[p] / n), int((cl == p).sum()), _r(pm[p])] for p in order[:10]]
    lk = {}
    for k in (1, 3, 5, 10, 20, 30):
        keep = ~np.isin(cl, order[:k])
        bt = Boot(list(cl[keep]))
        lk[str(k)] = mean_ci(d[keep], list(cl[keep]), bt)
    out["leave_top_k_pairs_out"] = lk
    # how many top pairs must go for the mean, and for the interval's lower end, to reach zero
    cum = 0.0
    k0 = None
    for k, p in enumerate(order, 1):
        cum += psum[p]
        rest = n - sum((cl == q).sum() for q in order[:k])
        if (d.sum() - cum) / rest <= 0:
            k0 = k
            break
    out["pairs_removed_for_mean_zero"] = k0
    klo = None
    for k in range(1, 60):
        keep = ~np.isin(cl, order[:k])
        r = mean_ci(d[keep], list(cl[keep]), Boot(list(cl[keep])))
        if r["ci"][0] <= 0:
            klo = k
            break
    out["pairs_removed_for_ci_to_reach_zero"] = klo
    # the same from the bottom, for symmetry (pairs with the most negative contribution)
    out["bottom_pairs_by_contribution"] = [[p, _r(psum[p] / n), int((cl == p).sum()), _r(pm[p])] for p in order[::-1][:5]]
    # items: share of the total sum carried by the top items, and how many top items must go for the mean to reach 0
    srt = np.sort(d)[::-1]
    out["items_top_share_of_sum"] = {f"top{q}pct": _r(srt[: int(round(n * q / 100))].sum() / d.sum()) for q in (1, 5, 10, 20)}
    out["items_bottom_share_of_sum"] = {f"bottom{q}pct": _r(np.sort(d)[: int(round(n * q / 100))].sum() / d.sum())
                                        for q in (1, 5, 10, 20)}
    c = np.cumsum(srt)
    out["items_removed_for_mean_zero"] = int(np.argmax((d.sum() - c) <= 0) + 1)
    # what the extreme items are: top and bottom 5% by ΔL
    o = np.argsort(-d, kind="mergesort")
    k5 = int(round(0.05 * n))
    lab = lambda r: f"{r['role']}|{r['cls']}|{'repeat' if r['repeat'] else 'first'}|{'overlap' if r['ov'] != 'none' else 'none'}"  # noqa: E731
    out["top5pct_makeup"] = dict(Counter(lab(main[i]) for i in o[:k5]).most_common())
    out["bottom5pct_makeup"] = dict(Counter(lab(main[i]) for i in o[-k5:]).most_common())
    out["top5pct_pairs"] = len({main[i]["pair"] for i in o[:k5]})
    # contribution of disjoint subsets to the primary ΔL (sum over the subset / 1500), with each subset's mean
    subs = {"goal|too_few": lambda r: r["role"] == "goal" and r["cls"] == "too_few",
            "goal|never_obtained": lambda r: r["role"] == "goal" and r["cls"] == "never_obtained",
            "goal|used_up": lambda r: r["role"] == "goal" and r["cls"] == "used_up",
            "ingredient|first craft|overlap": lambda r: r["role"] == "ingredient" and not r["repeat"] and r["ov"] != "none",
            "ingredient|first craft|no shared word": lambda r: r["role"] == "ingredient" and not r["repeat"] and r["ov"] == "none",
            "ingredient|repeat craft": lambda r: r["role"] == "ingredient" and r["repeat"]}
    out["contribution_by_subset"] = {k: {"items": int(sum(f(r) for r in main)),
                                         "contribution": _r(sum(x for r, x in zip(main, d) if f(r)) / n),
                                         "mean": mean_ci(np.array([x for r, x in zip(main, d) if f(r)]),
                                                         [r["pair"] for r in main if f(r)], bt_main)}
                                     for k, f in subs.items()}
    # symmetric removal (k largest and k smallest): this is a trimmed mean, reported in check 1's robust block
    # tail regions: where the shift happens on the log-odds scale
    La, Ls = arr(main, lambda r: r["La"]), arr(main, lambda r: r["Ls"])
    reg = {}
    for T in (3.0, 5.0):
        both_fail = (La <= -T) & (Ls <= -T)
        both_succ = (La >= T) & (Ls >= T)
        mid = ~(both_fail | both_succ)
        reg[f"T{int(T)}"] = {lab: {"n": int(m.sum()), "share_items": _r(m.mean()), "contribution": _r(d[m].sum() / n),
                                   "mean_within": _r(d[m].mean()) if m.any() else None}
                             for lab, m in (("both_confident_failure", both_fail), ("both_confident_success", both_succ),
                                            ("other", mid))}
    out["tail_regions"] = reg
    # clipped log-odds: L clipped to [-T, T] before the difference
    out["clipped"] = {f"T{T}": mean_ci(np.clip(La, -T, T) - np.clip(Ls, -T, T), list(cl), bt_main) for T in (2, 3, 5, 8)}
    # competent subset: failures the model calls failures under the neutral line
    Ln = arr(main, lambda r: r["Ln"])
    m = Ln < 0
    out["neutral_correct"] = {"share": _r(m.mean()), "delta_L": mean_ci(d[m], list(cl[m]), bt_main),
                              "delta_L_neutral_wrong": mean_ci(d[~m], list(cl[~m]), bt_main)}
    return out


# ------------------------------------------------------------------ check 2: families

def check_families(rows, comp, fam):
    main = [r for r in rows if r["pop"] == "main"]
    cc = Counter(comp[r["pair"]] for r in main)
    fc = Counter(fam[p] for p in {r["pair"] for r in main})
    out = {"components": {"n": len(set(comp[r["pair"]] for r in main)),
                          "largest_pairs": max(Counter(comp[p] for p in {r["pair"] for r in main}).values()),
                          "pairs": len({r["pair"] for r in main}), "usable": False},
           "primary_material": {"n": len(fc), "largest": fc.most_common(5)}}
    bf = Boot([r["fam"] for r in main])
    bp = Boot(pairs_of(main))
    res = {}
    for lab, sel in (("all", lambda r: True), ("goal", lambda r: r["role"] == "goal"),
                     ("ingredient", lambda r: r["role"] == "ingredient")):
        rs = [r for r in main if sel(r)]
        for scale, f in (("two_candidate", lambda r: r["La"] - r["Ls"]), ("yes_no", lambda r: r["Ya"] - r["Ys"])):
            v = arr(rs, f)
            res[f"{lab}|{scale}"] = {"pair_clustered": mean_ci(v, pairs_of(rs), bp),
                                     "family_clustered": mean_ci(v, [r["fam"] for r in rs], bf),
                                     "family_median": stat_ci(v, [r["fam"] for r in rs], bf, w_median),
                                     "family_trim10": stat_ci(v, [r["fam"] for r in rs], bf, lambda x, W: w_trim(x, W, 0.1))}
    out["estimates"] = res
    d = arr(main, lambda r: r["La"] - r["Ls"])
    fa = np.array([r["fam"] for r in main])
    lofo = {f: d[fa != f].mean() for f in set(fa)}
    s = sorted(lofo.items(), key=lambda kv: kv[1])
    out["leave_one_family_out"] = {"min": [s[0][0], _r(s[0][1])], "max": [s[-1][0], _r(s[-1][1])]}
    out["per_family"] = {f: {"items": int((fa == f).sum()), "pairs": int(len({r["pair"] for r in main if r["fam"] == f})),
                             "delta_L": _r(d[fa == f].mean())} for f, _ in fc.most_common(8)}
    fm = np.array([d[fa == f].mean() for f in sorted(set(fa))])
    out["share_families_pos"] = _r((fm > 0).mean())
    return out


# ------------------------------------------------------------------ check 3: word overlap

def check_overlap(rows, bt):
    main = [r for r in rows if r["pop"] == "main"]
    ing = [r for r in main if r["role"] == "ingredient"]
    out = {"counts": {k: Counter(r[k] for r in ing) for k in ("ov", "ov_stem")}}
    out["counts_by_class"] = {cat: dict(Counter(r["cls"] for r in ing if r["ov"] == cat)) for cat in ("contained", "word_only", "none")}
    out["partner_overlap_by_cat"] = {cat: int(sum(r["pov"] for r in ing if r["ov"] == cat)) for cat in ("contained", "word_only", "none")}
    scales = {"two_candidate": lambda r: r["La"] - r["Ls"], "yes_no": lambda r: r["Ya"] - r["Ys"],
              "first_token": lambda r: r["Fa"] - r["Fs"], "aligned_minus_neutral": lambda r: r["La"] - r["Ln"],
              "swapped_minus_neutral": lambda r: r["Ls"] - r["Ln"]}
    cats = {"goal_identical": [r for r in main if r["role"] == "goal"],
            "ing_contained": [r for r in ing if r["ov"] == "contained"],
            "ing_word_only": [r for r in ing if r["ov"] == "word_only"],
            "ing_none": [r for r in ing if r["ov"] == "none"],
            "ing_any_overlap": [r for r in ing if r["ov"] != "none"],
            "ing_stem_contained": [r for r in ing if r["ov_stem"] == "contained"],
            "ing_stem_word_only": [r for r in ing if r["ov_stem"] == "word_only"],
            "ing_none_no_partner_overlap": [r for r in ing if r["ov"] == "none" and not r["pov"]]}
    res = {}
    for cname, rs in cats.items():
        res[cname] = {s: mean_ci(arr(rs, f), pairs_of(rs), bt) for s, f in scales.items()}
        res[cname]["two_candidate_median"] = stat_ci(arr(rs, scales["two_candidate"]), pairs_of(rs), bt, w_median)
        res[cname]["two_candidate_trim10"] = stat_ci(arr(rs, scales["two_candidate"]), pairs_of(rs), bt, lambda x, W: w_trim(x, W, 0.1))
        res[cname]["by_class"] = {c: mean_ci(arr([r for r in rs if r["cls"] == c], scales["two_candidate"]),
                                             pairs_of([r for r in rs if r["cls"] == c]), bt)
                                  for c in CLASSES if any(r["cls"] == c for r in rs)}
        res[cname]["by_class_yes_no"] = {c: mean_ci(arr([r for r in rs if r["cls"] == c], scales["yes_no"]),
                                                    pairs_of([r for r in rs if r["cls"] == c]), bt)
                                         for c in CLASSES if any(r["cls"] == c for r in rs)}
    out["by_category"] = res
    # class-standardized contrasts: mean over classes of the within-class difference, weighted by the class's share of
    # all ingredient items, on the classes present in both categories; one shared set of pair weights
    def std_contrast(a_rows, b_rows, f):
        cls = [c for c in CLASSES if any(r["cls"] == c for r in a_rows) and any(r["cls"] == c for r in b_rows)]
        wts = np.array([sum(r["cls"] == c for r in ing) for c in cls], float)
        wts /= wts.sum()
        pt, dr = 0.0, 0.0
        for w, c in zip(wts, cls):
            A = [r for r in a_rows if r["cls"] == c]
            Bq = [r for r in b_rows if r["cls"] == c]
            va, vb = arr(A, f), arr(Bq, f)
            pt += w * (va.mean() - vb.mean())
            dr = dr + w * (w_mean(va, bt.W(pairs_of(A))) - w_mean(vb, bt.W(pairs_of(Bq))))
        return _pack(pt, dr, len(a_rows) + len(b_rows), len({r["pair"] for r in a_rows + b_rows}), classes=cls)

    con = {}
    for a, b in (("ing_contained", "ing_none"), ("ing_word_only", "ing_none"), ("ing_contained", "ing_word_only"),
                 ("ing_any_overlap", "ing_none")):
        for s in ("two_candidate", "yes_no"):
            con[f"{a}-{b}|{s}|raw"] = diff_ci(arr(cats[a], scales[s]), pairs_of(cats[a]),
                                             arr(cats[b], scales[s]), pairs_of(cats[b]), bt)
            con[f"{a}-{b}|{s}|class_standardized"] = std_contrast(cats[a], cats[b], scales[s])
    # class-standardized level of each ingredient category (weights: all ingredient items' class shares)
    lev = {}
    for cname in ("ing_contained", "ing_word_only", "ing_none"):
        for s in ("two_candidate", "yes_no"):
            rs = cats[cname]
            cls = [c for c in CLASSES if any(r["cls"] == c for r in rs)]
            wts = np.array([sum(r["cls"] == c for r in ing) for c in cls], float)
            wts /= wts.sum()
            pt, dr = 0.0, 0.0
            for w, c in zip(wts, cls):
                A = [r for r in rs if r["cls"] == c]
                v = arr(A, scales[s])
                pt += w * v.mean()
                dr = dr + w * w_mean(v, bt.W(pairs_of(A)))
            lev[f"{cname}|{s}"] = _pack(pt, dr, len(rs), len({r["pair"] for r in rs}), classes=cls)
    out["class_standardized_levels"] = lev
    out["contrasts"] = con
    # contribution of each category to the primary ΔL (sum over its items / 1500)
    d_all = len(main)
    out["contribution_to_primary"] = {c: _r(sum(r["La"] - r["Ls"] for r in cats[c]) / d_all)
                                      for c in ("goal_identical", "ing_contained", "ing_word_only", "ing_none")}
    # history fit within each category: the category subsets break the couple balance, so the fit strata are shown
    out["by_fit"] = {cat: {fit: mean_ci(arr([r for r in ing if r["ov"] == cat and r["fit"] == fit], scales["two_candidate"]),
                                        [r["pair"] for r in ing if r["ov"] == cat and r["fit"] == fit], bt)
                           for fit in ("aligned", "swapped", "tie")} for cat in ("contained", "word_only", "none")}
    out["fit_counts"] = {cat: dict(Counter(r["fit"] for r in ing if r["ov"] == cat)) for cat in ("contained", "word_only", "none")}
    # the cleanest contrast: couples with one overlapping and one non-overlapping ingredient member (same pair, same
    # history, same class); the per-couple difference of the two members' ΔL
    cp = defaultdict(list)
    for r in main:
        cp[r["couple"]].append(r)
    mixed = [(a, b) if a["ov"] != "none" else (b, a) for a, b in (v for v in cp.values() if len(v) == 2)
             if a["role"] == b["role"] == "ingredient" and (a["ov"] == "none") != (b["ov"] == "none")]
    def within(sub, f=scales["two_candidate"]):
        if len(sub) < 5:
            return {"n": len(sub)}
        return mean_ci(np.array([f(a) - f(b) for a, b in sub]), [a["pair"] for a, b in sub], bt)
    mx = {"n_couples": len(mixed), "two_candidate": within(mixed), "yes_no": within(mixed, scales["yes_no"]),
          "overlap_member": mean_ci(arr([a for a, b in mixed], scales["two_candidate"]), [a["pair"] for a, b in mixed], bt),
          "none_member": mean_ci(arr([b for a, b in mixed], scales["two_candidate"]), [a["pair"] for a, b in mixed], bt)}
    for fit in ("aligned", "swapped", "tie"):
        mx[f"fit_of_overlap_member={fit}"] = within([(a, b) for a, b in mixed if a["fit"] == fit])
    for c in CLASSES:
        mx[f"class={c}"] = within([(a, b) for a, b in mixed if a["cls"] == c])
    for cat in ("contained", "word_only"):
        mx[f"overlap_member={cat}"] = within([(a, b) for a, b in mixed if a["ov"] == cat])
    for side, ix in (("overlap_member", 0), ("none_member", 1)):
        mx[f"{side}_aligned_minus_neutral"] = mean_ci(np.array([p[ix]["La"] - p[ix]["Ln"] for p in mixed]), [p[0]["pair"] for p in mixed], bt)
        mx[f"{side}_swapped_minus_neutral"] = mean_ci(np.array([p[ix]["Ls"] - p[ix]["Ln"] for p in mixed]), [p[0]["pair"] for p in mixed], bt)
    out["mixed_couples"] = mx
    # couples whose two members are in the same category
    same = {}
    for lab, f in (("both_contained", lambda a, b: a["ov"] == b["ov"] == "contained"),
                   ("both_word_only", lambda a, b: a["ov"] == b["ov"] == "word_only"),
                   ("both_none", lambda a, b: a["ov"] == b["ov"] == "none"),
                   ("both_any_overlap", lambda a, b: a["role"] == b["role"] == "ingredient" and a["ov"] != "none" and b["ov"] != "none")):
        rs = [r for v in cp.values() if len(v) == 2 and v[0]["role"] == v[1]["role"] == "ingredient" and f(v[0], v[1]) for r in v]
        if rs:
            same[lab] = {s: mean_ci(arr(rs, scales[s]), pairs_of(rs), bt) for s in ("two_candidate", "yes_no")}
            same[lab]["couples"] = len(rs) // 2
    out["same_category_couples"] = same
    # the usefulness population (never obtained ingredients sharing no word with either goal; partly sign-mirrored,
    # goal identity not balanced: design D1) as an outside check of the no-overlap level
    use = [r for r in rows if r["pop"] == "usefulness"]
    out["usefulness_population"] = {s: mean_ci(arr(use, scales[s]), pairs_of(use), Boot(pairs_of(use)))
                                    for s in ("two_candidate", "yes_no")}
    return out


# ------------------------------------------------------------------ check 4: scales and the yes/no anomaly

SCALES = {"two_candidate": lambda r: r["La"] - r["Ls"], "first_token": lambda r: r["Fa"] - r["Fs"],
          "prefix": lambda r: r["Pa"] - r["Ps"], "name_remainder": lambda r: (r["La"] - r["Pa"]) - (r["Ls"] - r["Ps"]),
          "yes_no": lambda r: r["Ya"] - r["Ys"]}


def subsets(main):
    out = {"all": main, "goal": [r for r in main if r["role"] == "goal"],
           "ingredient": [r for r in main if r["role"] == "ingredient"]}
    for role, c in CELLS:
        out[f"{role}|{c}"] = [r for r in main if r["role"] == role and r["cls"] == c]
    return out


def check_scales(rows, bt):
    main = [r for r in rows if r["pop"] == "main"]
    out = {}
    for lab, rs in subsets(main).items():
        cl = pairs_of(rs)
        out[lab] = {}
        for s, f in SCALES.items():
            v = arr(rs, f)
            out[lab][s] = {"mean": mean_ci(v, cl, bt), "median": stat_ci(v, cl, bt, w_median),
                           "trim10": stat_ci(v, cl, bt, lambda x, W: w_trim(x, W, 0.1)),
                           "share_pos": _r((v > 0).mean()), "d": _r(v.mean() / v.std(ddof=1)) if len(v) > 1 else None}
        out[lab]["delta_p_two_candidate_pp"] = mean_ci(100 * (sig(arr(rs, lambda r: r["La"])) - sig(arr(rs, lambda r: r["Ls"]))), cl, bt)
        out[lab]["delta_p_yes_no_pp"] = mean_ci(100 * (sig(arr(rs, lambda r: r["Ya"])) - sig(arr(rs, lambda r: r["Ys"]))), cl, bt)
    # item-level agreement of the two-candidate and yes/no effects
    from scipy.stats import spearmanr
    a, b = arr(main, SCALES["two_candidate"]), arr(main, SCALES["yes_no"])
    out["item_agreement"] = {"spearman": _r(spearmanr(a, b).correlation),
                             "same_sign_share": _r(np.mean(np.sign(a) == np.sign(b))),
                             "two_cand_pos_yes_no_neg": _r(np.mean((a > 0) & (b < 0))),
                             "two_cand_neg_yes_no_pos": _r(np.mean((a < 0) & (b > 0)))}
    return out


def levels(rs, prefix):
    """Mean L, mean probability (sigmoid of L) and share with L > 0, under each line."""
    out = {}
    for c, k in (("aligned", "a"), ("swapped", "s"), ("neutral", "n")):
        v = arr(rs, lambda r: r[prefix + k])
        out[c] = {"mean_L": _r(v.mean()), "mean_p": _r(sig(v).mean()), "share_pos": _r((v > 0).mean()),
                  "median_L": _r(np.median(v))}
    return out


def check_yes_no(rows, bt):
    main = [r for r in rows if r["pop"] == "main"]
    succ = [r for r in rows if r["pop"] == "success"]
    out = {}
    groups = dict(subsets(main))
    groups["success|all"] = succ
    groups["success|goal"] = [r for r in succ if r["role"] == "goal"]
    groups["success|ingredient"] = [r for r in succ if r["role"] == "ingredient"]
    for lab, rs in groups.items():
        cl = pairs_of(rs)
        out[lab] = {"n": len(rs),
                    "yes_no_levels": levels(rs, "Y"), "two_candidate_levels": levels(rs, "L"),
                    "yes_no_aligned_minus_neutral": mean_ci(arr(rs, lambda r: r["Ya"] - r["Yn"]), cl, bt),
                    "yes_no_swapped_minus_neutral": mean_ci(arr(rs, lambda r: r["Ys"] - r["Yn"]), cl, bt),
                    "two_cand_aligned_minus_neutral": mean_ci(arr(rs, lambda r: r["La"] - r["Ln"]), cl, bt),
                    "two_cand_swapped_minus_neutral": mean_ci(arr(rs, lambda r: r["Ls"] - r["Ln"]), cl, bt),
                    "yes_no_delta": mean_ci(arr(rs, lambda r: r["Ya"] - r["Ys"]), cl, bt),
                    "two_cand_delta": mean_ci(arr(rs, lambda r: r["La"] - r["Ls"]), cl, bt)}
    # discrimination under each line: failures (main) against successes, overall and per role, both scales
    disc = {}
    for role in ("all", "goal", "ingredient"):
        F = [r for r in main if role == "all" or r["role"] == role]
        S = [r for r in succ if role == "all" or r["role"] == role]
        rs = F + S
        y = np.array([1.0] * 0 + [0.0] * len(F) + [1.0] * len(S))
        for c in "asn":
            for sc in ("L", "Y"):
                disc[f"{role}|{sc}|{c}"] = auc_ci(arr(rs, lambda r: r[sc + c]), y, pairs_of(rs), bt)
    out["auc_by_line"] = disc
    return out


# ------------------------------------------------------------------ check 6: decisions

def check_decisions(rows, bt):
    main = [r for r in rows if r["pop"] == "main"]
    succ = [r for r in rows if r["pop"] == "success"]
    out = {}
    groups = {"fail|all": main, "fail|goal": [r for r in main if r["role"] == "goal"],
              "fail|ingredient": [r for r in main if r["role"] == "ingredient"],
              "succ|all": succ, "succ|goal": [r for r in succ if r["role"] == "goal"],
              "succ|ingredient": [r for r in succ if r["role"] == "ingredient"]}
    for role, c in CELLS:
        groups[f"fail|{role}|{c}"] = [r for r in main if r["role"] == role and r["cls"] == c]
    for b in BINS:
        groups[f"succ|goal|{b}"] = [r for r in succ if r["role"] == "goal" and r["bin"] == b]
        groups[f"succ|ingredient|{b}"] = [r for r in succ if r["role"] == "ingredient" and r["bin"] == b]
        groups[f"fail|goal|{b}"] = [r for r in main if r["role"] == "goal" and r["bin"] == b]
    for lab, rs in groups.items():
        if not rs:
            continue
        cl = pairs_of(rs)
        g = {"n": len(rs), "pairs": len(set(cl))}
        for sc, nm in (("L", "two_candidate"), ("Y", "yes_no")):
            a, s, n = (arr(rs, lambda r, k=k: r[sc + k]) for k in "asn")
            g[nm] = {"share_pos": {"aligned": stat_ci((a > 0).astype(float), cl, bt),
                                   "swapped": stat_ci((s > 0).astype(float), cl, bt),
                                   "neutral": _r((n > 0).mean())},
                     "mean_p": {"aligned": _r(sig(a).mean()), "swapped": _r(sig(s).mean()), "neutral": _r(sig(n).mean())},
                     "share_pos_aligned_minus_swapped": mean_ci((a > 0).astype(float) - (s > 0).astype(float), cl, bt),
                     "flip_up": _r(np.mean((a > 0) & (s <= 0))), "flip_down": _r(np.mean((a <= 0) & (s > 0))),
                     "n_flip_up": int(np.sum((a > 0) & (s <= 0))), "n_flip_down": int(np.sum((a <= 0) & (s > 0))),
                     "delta_p_pp": mean_ci(100 * (sig(a) - sig(s)), cl, bt)}
        out[lab] = g
    # balanced accuracy under each line (failures: main; successes: success population); threshold L > 0
    ba = {}
    for role in ("all", "goal", "ingredient"):
        F = [r for r in main if role == "all" or r["role"] == role]
        S = [r for r in succ if role == "all" or r["role"] == role]
        for sc in ("L", "Y"):
            for c in "asn":
                tnr = np.mean([r[sc + c] <= 0 for r in F])
                tpr = np.mean([r[sc + c] > 0 for r in S])
                ba[f"{role}|{'two_candidate' if sc == 'L' else 'yes_no'}|{c}"] = {
                    "tpr": _r(tpr), "tnr": _r(tnr), "balanced": _r((tpr + tnr) / 2)}
    out["balanced_accuracy"] = ba
    # the false-failure effect: goal-role successes answered Yes under aligned but No under swapped, and where they sit
    gs = [r for r in succ if r["role"] == "goal"]
    ff = [r for r in gs if r["Ya"] > 0 and r["Ys"] <= 0]
    out["false_failure_goal_successes"] = {
        "n_goal_successes": len(gs), "n_yes_aligned_no_swapped": len(ff),
        "n_no_aligned_yes_swapped": int(sum(r["Ya"] <= 0 and r["Ys"] > 0 for r in gs)),
        "pairs_holding_them": len({r["pair"] for r in ff}),
        "by_bin": dict(Counter(r["bin"] for r in ff)), "goal_successes_by_bin": dict(Counter(r["bin"] for r in gs)),
        "by_depth": dict(Counter(r["depth"] for r in ff)), "goal_successes_by_depth": dict(Counter(r["depth"] for r in gs)),
        "by_fit": dict(Counter(r["fit"] for r in ff)), "goal_successes_by_fit": dict(Counter(r["fit"] for r in gs)),
        "neutral_yes_share_among_them": _r(np.mean([r["Yn"] > 0 for r in ff])) if ff else None,
        "two_candidate_pos_aligned_among_them": _r(np.mean([r["La"] > 0 for r in ff])) if ff else None,
        "two_candidate_pos_swapped_among_them": _r(np.mean([r["Ls"] > 0 for r in ff])) if ff else None,
    }
    return out


# ------------------------------------------------------------------ check 7: heterogeneity

def check_heterogeneity(rows, bt):
    main = [r for r in rows if r["pop"] == "main"]
    succ = [r for r in rows if r["pop"] == "success"]
    out = {}
    for role, c in CELLS:
        rs = [r for r in main if r["role"] == role and r["cls"] == c]
        S = [r for r in succ if r["role"] == role]
        cl = pairs_of(rs)
        d = arr(rs, lambda r: r["La"] - r["Ls"])
        Ln = arr(rs, lambda r: r["Ln"])
        cell = {"n": len(rs), "delta_L": mean_ci(d, cl, bt), "median": stat_ci(d, cl, bt, w_median),
                "levels": levels(rs, "L"),
                "auc_neutral_vs_same_role_successes": auc_ci(np.r_[Ln, arr(S, lambda r: r["Ln"])],
                                                             np.r_[np.zeros(len(rs)), np.ones(len(S))],
                                                             cl + pairs_of(S), bt),
                "share_neutral_confident_failure_le_minus5": _r((Ln <= -5).mean()),
                "share_neutral_uncertain_abs_lt_3": _r((np.abs(Ln) < 3).mean())}
        bins = {}
        for lab, lo, hi in (("Ln<-10", -np.inf, -10), ("-10<=Ln<-5", -10, -5), ("-5<=Ln<0", -5, 0), ("Ln>=0", 0, np.inf)):
            m = (Ln >= lo) & (Ln < hi)
            if m.sum() >= 5:
                bins[lab] = mean_ci(d[m], list(np.array(cl)[m]), bt)
        cell["by_neutral_L"] = bins
        cell["by_bin"] = {b: mean_ci(d[[r["bin"] == b for r in rs]], [r["pair"] for r in rs if r["bin"] == b], bt)
                          for b in BINS if any(r["bin"] == b for r in rs)}
        out[f"{role}|{c}"] = cell
    # ΔL against the neutral-line log-odds over all failures (the goal line acting where the model is unsure?)
    d = arr(main, lambda r: r["La"] - r["Ls"])
    Ln = arr(main, lambda r: r["Ln"])
    cl = np.array(pairs_of(main))
    allb = {}
    for lab, lo, hi in (("Ln<-15", -np.inf, -15), ("-15<=Ln<-10", -15, -10), ("-10<=Ln<-5", -10, -5),
                        ("-5<=Ln<-2", -5, -2), ("-2<=Ln<2", -2, 2), ("Ln>=2", 2, np.inf)):
        m = (Ln >= lo) & (Ln < hi)
        if m.sum() >= 5:
            allb[lab] = mean_ci(d[m], list(cl[m]), bt)
    out["all_by_neutral_L"] = allb
    # goal x never obtained: by completion (share of the goal's inputs held) and by missing-input kind
    gn = [r for r in main if r["role"] == "goal" and r["cls"] == "never_obtained"]
    sub = {}
    for lab, f in (("completion=0", lambda r: r["completion"] == 0), ("0<completion<0.5", lambda r: 0 < r["completion"] < 0.5),
                   ("completion>=0.5", lambda r: r["completion"] >= 0.5),
                   ("missing_all_craftable", lambda r: r["short_craftable"] is True),
                   ("missing_not_all_craftable", lambda r: r["short_craftable"] is False)):
        rs = [r for r in gn if f(r)]
        if len(rs) >= 5:
            cl2 = pairs_of(rs)
            sub[lab] = {"n": len(rs),
                        "two_candidate": mean_ci(arr(rs, lambda r: r["La"] - r["Ls"]), cl2, bt),
                        "yes_no": mean_ci(arr(rs, lambda r: r["Ya"] - r["Ys"]), cl2, bt),
                        "two_cand_a_minus_n": mean_ci(arr(rs, lambda r: r["La"] - r["Ln"]), cl2, bt),
                        "two_cand_s_minus_n": mean_ci(arr(rs, lambda r: r["Ls"] - r["Ln"]), cl2, bt),
                        "yes_no_a_minus_n": mean_ci(arr(rs, lambda r: r["Ya"] - r["Yn"]), cl2, bt),
                        "yes_no_s_minus_n": mean_ci(arr(rs, lambda r: r["Ys"] - r["Yn"]), cl2, bt)}
    out["goal_never_obtained_split"] = sub
    a, b = arr(gn, lambda r: r["La"] - r["Ls"]), arr(gn, lambda r: r["Ya"] - r["Ys"])
    out["goal_never_obtained_agreement"] = {"two_cand_neg_yes_no_pos": _r(np.mean((a < 0) & (b > 0))),
                                            "both_pos": _r(np.mean((a > 0) & (b > 0))),
                                            "both_neg": _r(np.mean((a < 0) & (b < 0)))}
    # ingredient crafts that repeat a craft the history already shows ("Crafted K minecraft:<object>" earlier)
    rep = {}
    for c in CLASSES:
        for lab, flag in (("repeat", True), ("first", False)):
            rs = [r for r in main if r["role"] == "ingredient" and r["cls"] == c and r["repeat"] == flag]
            rep[f"{c}|{lab}"] = ({"delta_L": mean_ci(arr(rs, lambda r: r["La"] - r["Ls"]), pairs_of(rs), bt),
                                  "yes_no": mean_ci(arr(rs, lambda r: r["Ya"] - r["Ys"]), pairs_of(rs), bt),
                                  "mean_L_neutral": _r(np.mean([r["Ln"] for r in rs]))} if len(rs) >= 5 else {"n": len(rs)})
    rs = [r for r in main if r["role"] == "ingredient"]
    rep["all_ingredient|repeat"] = mean_ci(arr([r for r in rs if r["repeat"]], lambda r: r["La"] - r["Ls"]), [r["pair"] for r in rs if r["repeat"]], bt)
    rep["all_ingredient|first"] = mean_ci(arr([r for r in rs if not r["repeat"]], lambda r: r["La"] - r["Ls"]), [r["pair"] for r in rs if not r["repeat"]], bt)
    rep["counts"] = {c: dict(Counter(r["repeat"] for r in rs if r["cls"] == c)) for c in CLASSES}
    rep["success_ingredient_repeat"] = int(sum(r["repeat"] for r in succ if r["role"] == "ingredient"))
    for lab, flag in (("repeat", True), ("first", False)):
        rs = [r for r in succ if r["role"] == "ingredient" and r["repeat"] == flag]
        rep[f"success_ingredient|{lab}"] = {"delta_L": mean_ci(arr(rs, lambda r: r["La"] - r["Ls"]), pairs_of(rs), bt),
                                            "mean_L_neutral": _r(np.mean([r["Ln"] for r in rs])),
                                            "yes_share": {c: _r(np.mean([r["Y" + c[0]] > 0 for r in rs])) for c in ("aligned", "swapped", "neutral")}}
    # repeat crossed with word overlap and class (ingredient failures)
    cross = {}
    for flag in (False, True):
        for ov in (True, False):
            base = [r for r in main if r["role"] == "ingredient" and r["repeat"] == flag and (r["ov"] != "none") == ov]
            key = f"{'repeat' if flag else 'first'}|{'overlap' if ov else 'no shared word'}"
            if len(base) >= 5:
                cross[key] = {"two_candidate": mean_ci(arr(base, lambda r: r["La"] - r["Ls"]), pairs_of(base), bt),
                              "yes_no": mean_ci(arr(base, lambda r: r["Ya"] - r["Ys"]), pairs_of(base), bt),
                              "classes": dict(Counter(r["cls"] for r in base))}
            for c in CLASSES:
                rs = [r for r in base if r["cls"] == c]
                if len(rs) >= 10:
                    cross[f"{key}|{c}"] = mean_ci(arr(rs, lambda r: r["La"] - r["Ls"]), pairs_of(rs), bt)
    rep["cross_with_overlap"] = cross
    out["repeat_craft"] = rep
    # by length bin, per role
    out["by_bin_role"] = {f"{role}|{b}": mean_ci(arr([r for r in main if r["role"] == role and r["bin"] == b], lambda r: r["La"] - r["Ls"]),
                                                  [r["pair"] for r in main if r["role"] == role and r["bin"] == b], bt)
                          for role in ("goal", "ingredient") for b in BINS}
    return out


# ------------------------------------------------------------------ check 5: prompt variants

def check_prompts(variants: dict, rows_by_model: dict, items):
    """Per variant: the variant's ΔL against the verified prompt's ΔL on the same items (pair-clustered)."""
    I = {t["id"]: t for t in items}
    out = {}
    for vname, recs in variants.items():
        if vname == "generate_qwen":
            continue
        by = defaultdict(dict)
        for s in recs:
            by[s["model"]][(s["transition_id"], s["condition"])] = s
        sub_variants = X.PROMPT_VARIANTS if vname == "prompts" else (None,)
        for sv in sub_variants:
            key = sv if sv else vname
            res = {}
            for m in MODELS:
                ver = {r["id"]: r for r in rows_by_model[m]}
                cond = (lambda c: f"{sv}|{c}") if sv else (lambda c: c)
                if vname == "objectgoal":
                    continue
                ids = sorted({tid for (tid, c) in by[m] if c == cond("aligned")})
                rs = [ver[i] for i in ids]
                main = [r for r in rs if r["pop"] == "main"]
                succ = [r for r in rs if r["pop"] == "success"]
                bt = Boot([r["pair"] for r in rs])
                has_neutral = any(c == cond("neutral") for (_t, c) in by[m])
                g = lambda r, c: by[m][(r["id"], cond(c))]["L"]  # noqa: E731
                block = {"items": len(main), "pairs": len({r["pair"] for r in main})}
                for lab, sel in (("all", lambda r: True), ("goal", lambda r: r["role"] == "goal"),
                                 ("ingredient", lambda r: r["role"] == "ingredient")):
                    R = [r for r in main if sel(r)]
                    if not R:
                        continue
                    cl = pairs_of(R)
                    vv = arr(R, lambda r: g(r, "aligned") - g(r, "swapped"))
                    vr = arr(R, lambda r: r["La"] - r["Ls"])
                    gf = lambda r, c: by[m][(r["id"], cond(c))]["L_first"]  # noqa: E731
                    vf = arr(R, lambda r: gf(r, "aligned") - gf(r, "swapped"))
                    e = {"variant": mean_ci(vv, cl, bt), "verified": mean_ci(vr, cl, bt),
                         "variant_minus_verified": mean_ci(vv - vr, cl, bt),
                         "variant_median": stat_ci(vv, cl, bt, w_median),
                         "verified_median": stat_ci(vr, cl, bt, w_median),
                         "variant_trim10": stat_ci(vv, cl, bt, lambda x, W: w_trim(x, W, 0.1)),
                         "verified_trim10": stat_ci(vr, cl, bt, lambda x, W: w_trim(x, W, 0.1)),
                         "variant_first_token": mean_ci(vf, cl, bt),
                         "variant_share_pos": _r((vv > 0).mean()), "verified_share_pos": _r((vr > 0).mean()),
                         "variant_mean_L": {c: _r(np.mean([g(r, c) for r in R])) for c in ("aligned", "swapped")},
                         "variant_first_mass": _r(np.mean([by[m][(r["id"], cond("aligned"))]["first_mass"] for r in R]))}
                    if has_neutral:
                        e["variant_aligned_minus_neutral"] = mean_ci(arr(R, lambda r: g(r, "aligned") - g(r, "neutral")), cl, bt)
                        e["variant_swapped_minus_neutral"] = mean_ci(arr(R, lambda r: g(r, "swapped") - g(r, "neutral")), cl, bt)
                    block[lab] = e
                cells = {}
                for role, c in CELLS:
                    R = [r for r in main if r["role"] == role and r["cls"] == c]
                    if len(R) >= 10:
                        cl = pairs_of(R)
                        cells[f"{role}|{c}"] = {"variant": mean_ci(arr(R, lambda r: g(r, "aligned") - g(r, "swapped")), cl, bt),
                                                "verified": mean_ci(arr(R, lambda r: r["La"] - r["Ls"]), cl, bt)}
                block["cells"] = cells
                if succ:
                    cl = pairs_of(succ)
                    block["successes"] = {"variant": mean_ci(arr(succ, lambda r: g(r, "aligned") - g(r, "swapped")), cl, bt),
                                          "verified": mean_ci(arr(succ, lambda r: r["La"] - r["Ls"]), cl, bt)}
                    gs = [r for r in succ if r["role"] == "goal"]
                    block["goal_successes_share_pos"] = {
                        c: _r(np.mean([g(r, c) > 0 for r in gs])) for c in ("aligned", "swapped", "neutral")} if has_neutral else None
                    block["goal_successes_share_pos_verified"] = {
                        c: _r(np.mean([r["L" + c[0]] > 0 for r in gs])) for c in ("aligned", "swapped", "neutral")}
                    if has_neutral:
                        y = np.r_[np.zeros(len(main)), np.ones(len(succ))]
                        cl2 = pairs_of(main) + pairs_of(succ)
                        block["auc_neutral_variant"] = auc_ci(np.r_[arr(main, lambda r: g(r, "neutral")), arr(succ, lambda r: g(r, "neutral"))], y, cl2, bt)
                        block["auc_neutral_verified"] = auc_ci(np.r_[arr(main, lambda r: r["Ln"]), arr(succ, lambda r: r["Ln"])], y, cl2, bt)
                res[SHORT[m]] = block
            if res:
                out[key] = res
    # object goal: the ingredient action's own object made the goal, against aligned and swapped (verified)
    if "objectgoal" in variants:
        by = defaultdict(dict)
        for s in variants["objectgoal"]:
            by[s["model"]][(s["transition_id"], s["condition"])] = s
        og = {}
        for m in MODELS:
            ver = {r["id"]: r for r in rows_by_model[m]}
            ids = sorted({tid for (tid, c) in by[m] if c == "object_goal"})
            rs = [ver[i] for i in ids]
            bt = Boot([r["pair"] for r in rs])
            blk = {}
            for pop in ("main", "success"):
                R = [r for r in rs if r["pop"] == pop]
                cl = pairs_of(R)
                o = arr(R, lambda r: by[m][(r["id"], "object_goal")]["L"])
                yo = arr(R, lambda r: by[m][(r["id"], "yn_object_goal")]["L"])
                blk[pop] = {"n": len(R), "object_minus_swapped": mean_ci(o - arr(R, lambda r: r["Ls"]), cl, bt),
                            "object_minus_aligned": mean_ci(o - arr(R, lambda r: r["La"]), cl, bt),
                            "aligned_minus_swapped": mean_ci(arr(R, lambda r: r["La"] - r["Ls"]), cl, bt),
                            "yn_object_minus_swapped": mean_ci(yo - arr(R, lambda r: r["Ys"]), cl, bt),
                            "yn_aligned_minus_swapped": mean_ci(arr(R, lambda r: r["Ya"] - r["Ys"]), cl, bt),
                            "share_pos_object": _r((o > 0).mean()), "share_pos_aligned": _r(np.mean([r["La"] > 0 for r in R])),
                            "share_pos_swapped": _r(np.mean([r["Ls"] > 0 for r in R])),
                            "yes_share_object": _r((yo > 0).mean()), "yes_share_aligned": _r(np.mean([r["Ya"] > 0 for r in R])),
                            "yes_share_swapped": _r(np.mean([r["Ys"] > 0 for r in R]))}
            og[SHORT[m]] = blk
        out["object_goal"] = og
    return out


# ------------------------------------------------------------------ check 8: free generation (Qwen), if run

def check_generation(recs, rows_by_model):
    if not recs:
        return None
    out = {}
    ver = {r["id"]: r for r in rows_by_model["qwen3-8b-fewshot"]}
    groups = defaultdict(lambda: defaultdict(dict))
    for s in recs:
        groups[(s["population"], s["role"])][s["transition_id"]][s["condition"]] = s
    for (pop, role), byi in sorted(groups.items()):
        ids = sorted(byi)
        rs = [ver[i] for i in ids]
        bt = Boot([r["pair"] for r in rs])
        cl = pairs_of(rs)
        g = {"n": len(ids)}
        for c in ("aligned", "swapped", "neutral"):
            g[c] = {"observation": dict(Counter(byi[i][c]["outcome"] for i in ids)),
                    "answer": dict(Counter(byi[i][c]["yn_answer"] for i in ids))}
        su = lambda i, c: float(byi[i][c]["outcome"] == "success")  # noqa: E731
        ye = lambda i, c: float(byi[i][c]["yn_answer"] == "Yes")  # noqa: E731
        g["obs_success_aligned_minus_swapped"] = mean_ci(np.array([su(i, "aligned") - su(i, "swapped") for i in ids]), cl, bt)
        g["yes_aligned_minus_swapped"] = mean_ci(np.array([ye(i, "aligned") - ye(i, "swapped") for i in ids]), cl, bt)
        g["examples_other"] = sorted({byi[i][c]["generation"][:60] for i in ids for c in ("aligned", "swapped", "neutral")
                                      if byi[i][c]["outcome"] == "other"})[:8]
        # agreement of the free observation with the two-candidate sign and of the free answer with the yes/no sign
        g["agree_two_candidate_sign"] = _r(np.mean([(byi[i][c]["outcome"] == "success") == (ver[i]["L" + c[0]] > 0)
                                                    for i in ids for c in ("aligned", "swapped", "neutral")]))
        g["agree_yes_no_sign"] = _r(np.mean([(byi[i][c]["yn_answer"] == "Yes") == (ver[i]["Y" + c[0]] > 0)
                                             for i in ids for c in ("aligned", "swapped", "neutral")]))
        out[f"{pop}|{role}"] = g
    return out


def wls_line_shift(rows, bt, lines=("a", "s", "n")) -> dict:
    """Yes/no L regressed on first-token L with one intercept per goal line (pooled item x line observations):
    Y = c_line + b * F. The intercept differences are the yes/no shifts between lines that the observation readout's
    own shift does not account for. Pair-bootstrap intervals (weighted least squares per resample)."""
    Ys, Fs, D, cl = [], [], [], []
    for r in rows:
        for j, k in enumerate(lines):
            Ys.append(r["Y" + k])
            Fs.append(r["F" + k])
            d = [0.0] * len(lines)
            d[j] = 1.0
            D.append(d)
            cl.append(r["pair"])
    Y, F, D = np.array(Ys), np.array(Fs), np.array(D)
    X = np.column_stack([D, F])
    W = bt.W(cl)

    def fit(w):
        Xw = X * w[:, None]
        return np.linalg.lstsq(Xw.T @ X, Xw.T @ Y, rcond=None)[0]

    pt = fit(np.ones(len(Y)))
    draws = np.array([fit(w) for w in W])
    ia, is_, in_ = lines.index("a"), lines.index("s"), lines.index("n")
    out = {"slope": _pack(pt[-1], draws[:, -1], len(rows), len(set(cl))),
           "aligned_minus_swapped": _pack(pt[ia] - pt[is_], draws[:, ia] - draws[:, is_], len(rows), len(set(cl))),
           "aligned_minus_neutral": _pack(pt[ia] - pt[in_], draws[:, ia] - draws[:, in_], len(rows), len(set(cl))),
           "swapped_minus_neutral": _pack(pt[is_] - pt[in_], draws[:, is_] - draws[:, in_], len(rows), len(set(cl)))}
    return out


def check_task_reading(rows, bt) -> dict:
    """How much of the yes/no effect is plausibly a task-success reading ("does this action succeed at my goal?")
    rather than a prediction of the environment's reply. TextCraft's question ("Will this action succeed? Answer Yes
    or No.") is not anchored to the reply, unlike ALFWorld's. Compared item by item with the first-token readout of
    the observation (which is the observation decision, first-token mass 1 for Qwen), on failures and successes, by
    role, and by line: under the aligned line the action always serves the named goal (a goal craft completes it, an
    ingredient craft is a step toward it); under the swapped line it never does (the object lies in the aligned tree
    only); the neutral line names no goal."""
    main = [r for r in rows if r["pop"] == "main"]
    succ = [r for r in rows if r["pop"] == "success"]
    groups = {"fail|all": main, "fail|goal": [r for r in main if r["role"] == "goal"],
              "fail|ingredient": [r for r in main if r["role"] == "ingredient"],
              "succ|all": succ, "succ|goal": [r for r in succ if r["role"] == "goal"],
              "succ|ingredient": [r for r in succ if r["role"] == "ingredient"]}
    out = {}
    for lab, rs in groups.items():
        cl = pairs_of(rs)
        yd, fd, ld = (arr(rs, lambda r: r["Ya"] - r["Ys"]), arr(rs, lambda r: r["Fa"] - r["Fs"]),
                      arr(rs, lambda r: r["La"] - r["Ls"]))
        g = {"n": len(rs),
             "yes_no": mean_ci(yd, cl, bt), "first_token": mean_ci(fd, cl, bt), "two_candidate": mean_ci(ld, cl, bt),
             "yes_no_minus_first_token": mean_ci(yd - fd, cl, bt),
             "yes_no_aligned_minus_neutral": mean_ci(arr(rs, lambda r: r["Ya"] - r["Yn"]), cl, bt),
             "yes_no_swapped_minus_neutral": mean_ci(arr(rs, lambda r: r["Ys"] - r["Yn"]), cl, bt),
             "first_aligned_minus_neutral": mean_ci(arr(rs, lambda r: r["Fa"] - r["Fn"]), cl, bt),
             "first_swapped_minus_neutral": mean_ci(arr(rs, lambda r: r["Fs"] - r["Fn"]), cl, bt),
             "share_yes_no_up_first_not_up": _r(np.mean((yd > 0) & (fd <= 0))),
             "share_first_up_yes_no_not_up": _r(np.mean((fd > 0) & (yd <= 0)))}
        if len(rs) > 5:
            from scipy.stats import spearmanr
            g["spearman_item_effects"] = _r(spearmanr(yd, fd).correlation)
        # the observation readout says success (first-token L at least +3) but the answer is No, and the reverse
        dis = {}
        for c, k in (("aligned", "a"), ("swapped", "s"), ("neutral", "n")):
            F, Y = arr(rs, lambda r: r["F" + k]), arr(rs, lambda r: r["Y" + k])
            pos, neg = F >= 3, F <= -3
            dis[c] = {"n_obs_success": int(pos.sum()),
                      "answer_no_given_obs_success": _r(np.mean(Y[pos] <= 0)) if pos.any() else None,
                      "n_obs_failure": int(neg.sum()),
                      "answer_yes_given_obs_failure": _r(np.mean(Y[neg] > 0)) if neg.any() else None}
        g["dissociation"] = dis
        g["line_shift_given_first_token"] = wls_line_shift(rs, bt)
        out[lab] = g
    # on successes, the yes/no effect should be about zero for a world model that knows the action works; compare it
    # with the observation readout on the same items, per length bin
    out["succ_by_bin"] = {b: {"yes_no": mean_ci(arr([r for r in succ if r["bin"] == b], lambda r: r["Ya"] - r["Ys"]),
                                                [r["pair"] for r in succ if r["bin"] == b], bt),
                              "first_token": mean_ci(arr([r for r in succ if r["bin"] == b], lambda r: r["Fa"] - r["Fs"]),
                                                     [r["pair"] for r in succ if r["bin"] == b], bt)} for b in BINS}
    return out


def check_format(scores, gen_llama) -> dict:
    """What free generation could add: the probability mass on the two candidates' first tokens per arm (main and
    success items), how often the first-token sign equals the full two-candidate sign, and, for Llama's existing
    generation run (textcraft_dx_generate), how often the generated outcome follows the sign of the first-token score."""
    out = {}
    for m in MODELS:
        blk = {}
        for c in ("aligned", "swapped", "neutral", "yn_aligned", "yn_swapped", "yn_neutral"):
            fm = np.array([s["first_mass"] for s in scores if s["model"] == m and s["condition"] == c
                           and s["population"] in ("main", "success")])
            blk[c] = {"mean": _r(fm.mean(), 5), "min": _r(fm.min()), "share_below_0_99": _r((fm < 0.99).mean()),
                      "share_below_0_5": _r((fm < 0.5).mean())}
        x = [s for s in scores if s["model"] == m and s["condition"] in ("aligned", "swapped", "neutral")
             and s["population"] in ("main", "success")]
        blk["first_sign_equals_full_sign"] = _r(np.mean([(s["L_first"] > 0) == (s["L"] > 0) for s in x]))
        out[SHORT[m]] = blk
    if gen_llama:
        lf = {(s["transition_id"], s["condition"]): s for s in scores if s["model"] == "llama3.1-8b-fewshot"}
        ok = [(g["outcome"] == "success") == (lf[(g["transition_id"], g["condition"])]["L_first"] > 0)
              for g in gen_llama if g["outcome"] in ("success", "failure")]
        yn = {(s["transition_id"], s["condition"][3:]): s for s in scores
              if s["model"] == "llama3.1-8b-fewshot" and s["condition"].startswith("yn_")}
        ok2 = [(g["yn_answer"] == "Yes") == (yn[(g["transition_id"], g["condition"])]["L"] > 0)
               for g in gen_llama if g["yn_answer"] in ("Yes", "No")]
        out["llama_generation_check"] = {"generations": len(gen_llama), "success_or_failure": len(ok),
                                         "outcome_follows_first_token_sign": _r(np.mean(ok)),
                                         "yes_no_answers": len(ok2), "answer_follows_yes_no_sign": _r(np.mean(ok2))}
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()]
    shots = P.fewshot_examples()
    frozen = json.loads((REPO / "notes" / "textcraft_metrics.json").read_text())["per_model"]
    scores = read_scores(Path(a.scores))
    I = {t["id"]: t for t in items}
    vh = {t["id"]: item_hash(t) for t in items}
    prov = {"items_sha256": sha256(ITEMS),
            "score_files": {str(p.relative_to(a.scores)): sha256(p) for p in sorted(Path(a.scores).rglob("scoring.jsonl"))},
            "main_records": len(scores), "main_stale": int(sum(vh[s["transition_id"]] != s["item_hash"] for s in scores))}
    by = {(s["model"], s["transition_id"], s["condition"]): s for s in scores}
    comp, fam, _outs = families(items)
    rows_by_model = {m: table(items, by, m, fam, comp) for m in MODELS}
    vroot = Path(a.variants)
    variants, vprov = {}, {}
    for name in ("examples_on_goal", "empty_start", "zeroshot", "objectgoal", "prompts", "generate_qwen"):
        if (vroot / name).exists():
            recs, info = load_variant(vroot, name, items, shots)
            files = sorted((vroot / name).rglob("scoring.jsonl")) + sorted((vroot / name).rglob("generations.jsonl"))
            info["sha256"] = {str(p.relative_to(vroot)): sha256(p) for p in files}
            variants[name], vprov[name] = recs, info
    prov["variants"] = vprov
    out = {"provenance": prov, "per_model": {}}
    for m in MODELS:
        rows = rows_by_model[m]
        main = [r for r in rows if r["pop"] == "main"]
        bt = Boot([r["pair"] for r in rows])  # every pair of the three populations; main-only cuts use their rows
        bt_main = Boot([r["pair"] for r in main])
        print(f"[{SHORT[m]}] recompute", file=sys.stderr)
        res = {"recompute": check_recompute(rows, bt_main, frozen[m]),
               "concentration": check_concentration(rows, bt_main)}
        print(f"[{SHORT[m]}] families, overlap", file=sys.stderr)
        res["families"] = check_families(rows, comp, fam)
        res["overlap"] = check_overlap(rows, bt_main)
        print(f"[{SHORT[m]}] scales, yes/no", file=sys.stderr)
        res["scales"] = check_scales(rows, bt_main)
        res["yes_no"] = check_yes_no(rows, bt)
        print(f"[{SHORT[m]}] decisions, heterogeneity", file=sys.stderr)
        res["decisions"] = check_decisions(rows, bt)
        res["heterogeneity"] = check_heterogeneity(rows, bt)
        print(f"[{SHORT[m]}] task-success reading of the yes/no question", file=sys.stderr)
        res["task_reading"] = check_task_reading(rows, bt)
        out["per_model"][SHORT[m]] = res
    print("[both] prompt variants", file=sys.stderr)
    out["prompts"] = check_prompts(variants, rows_by_model, items)
    out["generation_qwen"] = check_generation(variants.get("generate_qwen"), rows_by_model)
    gl = vroot / "generate" / "generations.jsonl"
    gen_llama = [json.loads(l) for l in gl.read_text().splitlines() if l.strip()] if gl.exists() else None
    if gen_llama:
        prov["llama_generation_sha256"] = sha256(gl)
    out["format"] = check_format(scores, gen_llama)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=False, default=lambda o: o if not isinstance(o, Counter) else dict(o)) + "\n")


if __name__ == "__main__":
    main()
