"""Exploratory, post hoc: how competent Llama-3.1-8B (and Qwen3-8B, the reference) is as a TextCraft world model,
which simple text heuristic its scores follow, and how competence relates to its goal effect
(notes/textcraft_llama_competence.md; the heuristics H1-H10 were fixed in that note before any fit was computed).

Nothing here scores anything. It reads the confirmatory scoring records of `textcraft_score` (51,200 records), the
exploratory prompt-variant records of `textcraft_dx_prompts` (inventory shown), and, for the ALFWorld comparison,
the archived ALFWorld runs (rq1_score, rq1x_score at text-wm-fix12/13; rq1v_score at text-wm-fix17).

Every interval is a 95% percentile interval from 2,000 resamples of whole clusters (TextCraft: goal pairs;
ALFWorld: games), one shared set of resampling weights per environment. No p-values.

Usage:
    python scripts/textcraft_llama_competence.py --scores <dir with textcraft_score scoring.jsonl files> \
        --prompts <textcraft_dx_prompts dir> --archive <text-wm-runs-archive> --out notes/textcraft_llama_competence_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wf.textcraft.checker import GENERIC_MEMBERS, account, parse_craft, from_id  # noqa: E402

ITEMS = REPO / "wf" / "textcraft" / "data" / "items.jsonl"
MODELS = ("llama3.1-8b-fewshot", "qwen3-8b-fewshot")
SHORT = {"llama3.1-8b-fewshot": "llama", "qwen3-8b-fewshot": "qwen"}
SEED = 20260924
B = 2000
CLASSES = ("used_up", "too_few", "never_obtained")
ROLES = ("goal", "ingredient")
BINS = ("1-3", "4-8", "9-12")
CAL_EDGES = np.linspace(0, 1, 11)


# ---------------------------------------------------------------- resampling

class Boot:
    """One shared set of cluster-resampling weights: W(clusters) gives a (B, n) matrix of item weights."""

    def __init__(self, clusters: list[str], b: int = B, seed: int = SEED):
        self.keys = sorted(set(clusters))
        self.k = {c: i for i, c in enumerate(self.keys)}
        rng = np.random.default_rng(seed)
        self.C = rng.multinomial(len(self.keys), [1.0 / len(self.keys)] * len(self.keys), size=b).astype(float)

    def W(self, clusters) -> np.ndarray:
        return self.C[:, [self.k[c] for c in clusters]]


def _ci(point: float, draws: np.ndarray, n: int, clusters: int) -> dict:
    d = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
    return {"est": _r(point), "ci": [_r(lo), _r(hi)], "n": int(n), "clusters": int(clusters)}


def _r(x, k: int = 4):
    x = float(x)
    return None if x != x else round(x, k)


def wauc(s: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted AUC P(score of a positive > score of a negative), ties one half, for each row of weights W."""
    order = np.argsort(s, kind="mergesort")
    s, y, W = s[order], y[order], W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    Wp = np.add.reduceat(W * y, starts, axis=1)
    Wn = np.add.reduceat(W * (1 - y), starts, axis=1)
    below = np.cumsum(Wn, axis=1) - Wn
    num = (Wp * (below + 0.5 * Wn)).sum(1)
    den = Wp.sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def auc_ci(s, y, clusters, bt: Boot) -> dict:
    s, y = np.asarray(s, float), np.asarray(y, float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"est": None, "ci": [None, None], "n": int(len(y)), "clusters": len(set(clusters))}
    point = wauc(s, y, np.ones((1, len(s))))[0]
    out = _ci(point, wauc(s, y, bt.W(clusters)), len(s), len(set(clusters)))
    out["n_pos"], out["n_neg"] = int(y.sum()), int((1 - y).sum())
    return out


def mean_ci(v, clusters, bt: Boot) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return {"est": None, "ci": [None, None], "n": 0, "clusters": 0}
    W = bt.W(clusters)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (W @ v) / W.sum(1)
    return _ci(v.mean(), draws, len(v), len(set(clusters)))


def spearman_ci(x, y, clusters, bt: Boot) -> dict:
    """Spearman correlation; the interval reuses the full-sample ranks (weighted Pearson of fixed ranks)."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"est": None, "ci": [None, None], "n": int(len(x)), "clusters": len(set(clusters))}
    rx, ry = rankdata(x), rankdata(y)

    def wcorr(W):
        sw = W.sum(1, keepdims=True)
        mx, my = (W @ rx)[:, None] / sw, (W @ ry)[:, None] / sw
        dx, dy = rx[None] - mx, ry[None] - my
        cov = (W * dx * dy).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return cov / np.sqrt((W * dx * dx).sum(1) * (W * dy * dy).sum(1))

    return _ci(wcorr(np.ones((1, len(x))))[0], wcorr(bt.W(clusters)), len(x), len(set(clusters)))


def ols_ci(y, X, clusters, bt: Boot, names: list[str]) -> dict:
    """Least squares of y on the columns of X (intercept included by the caller), pair-bootstrap per slope."""
    y, X = np.asarray(y, float), np.asarray(X, float)

    def fit(W):
        XtW = np.einsum("bi,ij->bij", W, X)
        A = np.einsum("bij,ik->bjk", XtW, X)
        c = np.einsum("bij,i->bj", XtW, y)
        return np.linalg.solve(A + 1e-9 * np.eye(X.shape[1]), c[..., None])[..., 0]

    point = fit(np.ones((1, len(y))))[0]
    draws = fit(bt.W(clusters))
    return {nm: _ci(point[j], draws[:, j], len(y), len(set(clusters))) for j, nm in enumerate(names)}


def kappa_ci(a, b, clusters, bt: Boot) -> dict:
    """Agreement share and Cohen's kappa between two binary vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def stats(W):
        sw = W.sum(1)
        po = (W @ (a == b).astype(float)) / sw
        pa, pb = (W @ a) / sw, (W @ b) / sw
        pe = pa * pb + (1 - pa) * (1 - pb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return po, (po - pe) / (1 - pe)

    p0, k0 = stats(np.ones((1, len(a))))
    pd, kd = stats(bt.W(clusters))
    return {"agreement": _ci(p0[0], pd, len(a), len(set(clusters))), "kappa": _ci(k0[0], kd, len(a), len(set(clusters)))}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def calibration(L, y) -> dict:
    """Binned predicted P(success) = sigmoid(L) against the observed success rate."""
    p, y = sigmoid(np.asarray(L, float)), np.asarray(y, float)
    idx = np.clip(np.digitize(p, CAL_EDGES[1:-1]), 0, len(CAL_EDGES) - 2)
    bins, ece = [], 0.0
    for k in range(len(CAL_EDGES) - 1):
        m = idx == k
        if m.sum():
            bins.append({"bin": [round(CAL_EDGES[k], 2), round(CAL_EDGES[k + 1], 2)], "n": int(m.sum()),
                         "mean_pred": _r(p[m].mean()), "observed": _r(y[m].mean()),
                         "n_success": int(y[m].sum())})
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"bins": bins, "ece": _r(ece), "brier": _r(((p - y) ** 2).mean()), "base_rate": _r(y.mean()),
            "mean_pred_failures": _r(p[y == 0].mean()), "mean_pred_successes": _r(p[y == 1].mean()),
            "n": int(len(p))}


# ---------------------------------------------------------------- data

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tc_scores(root: str) -> dict:
    """model -> item id -> condition -> record."""
    by: dict = defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(str(Path(root) / "**" / "scoring.jsonl"), recursive=True)):
        for s in read_jsonl(p):
            by[s["model"]][s["transition_id"]][s["condition"]] = s
    return by


# ---------------------------------------------------------------- the heuristics (fixed in the note before any fit)

def _mentions(action: str, reply: str) -> set[str]:
    out: set[str] = set()
    if action.startswith("get "):
        out.add(action.split(" ", 2)[2])
    elif action.startswith("craft "):
        _k, y, inputs = parse_craft(action)
        out.add(y)
        out |= {n for _c, n in inputs}
    if reply.startswith("Got "):
        out.add(reply.split(" ", 2)[2])
    elif reply.startswith("Crafted "):
        out.add(from_id(reply.split(" ", 2)[2]))
    elif reply.startswith("Inventory: ") and "not carrying" not in reply:
        import re
        out |= set(re.findall(r"\[([^\]]+)\] \(", reply))
    return out


def features(item: dict) -> dict:
    """The heuristics' predictions and graded scores, and the quantities of question 3, from the prompt text."""
    hist = [tuple(h) for h in item["history"]]
    T = len(hist)
    _k, y, inputs = parse_craft(item["action"])
    ledger = account(hist)
    gained = defaultdict(int)  # gross units obtained
    last_gain, last_consume, last_mention = {}, {}, {}
    # running held, to find when each input last fell from >= need to < need, and inventory displays after it
    held = defaultdict(int)
    need = {n: c for c, n in inputs}
    fell_at: dict[str, int] = {}
    shown_after_fall: dict[str, bool] = {}
    consumed_ever = defaultdict(bool)
    for t, (a, r) in enumerate(hist, 1):
        for n in _mentions(a, r):
            last_mention[n] = t
        if r.startswith("Got "):
            nn, x = r.split(" ", 2)[1:]
            gained[x] += int(nn)
            held[x] += int(nn)
            last_gain[x] = t
        elif r.startswith("Crafted "):
            _kk, yy, ins = parse_craft(a)
            for c, n in ins:
                before = held[n]
                held[n] -= c
                last_consume[n] = t
                consumed_ever[n] = True
                if n in need and before >= need[n] > held[n]:
                    fell_at[n], shown_after_fall[n] = t, False
            kk, x = r.split(" ", 2)[1:]
            x = from_id(x)
            gained[x] += int(kk)
            held[x] += int(kk)
            last_gain[x] = t
        elif r.startswith("Inventory: "):
            for n in fell_at:
                shown_after_fall[n] = True
    h = {n: ledger.held.get(n, 0) for _c, n in inputs}
    assert all(h[n] == max(held[n], 0) for n in h)
    truth = all(h[n] >= c for c, n in inputs)
    lines = [parse_craft(line) for line in item["recipe_lines"]]
    consumed_names = set()
    for _kk, _out, ins in lines:
        for _c, n in ins:
            consumed_names.add(n)
            consumed_names |= set(GENERIC_MEMBERS.get(n, ()))
    top = y not in consumed_names

    def recent_gain(k):
        return all(last_gain.get(n, -10 ** 6) > T - k for _c, n in inputs)

    since_mention = max((T - last_mention[n]) if n in last_mention else T + 1 for _c, n in inputs)
    f = {
        "truth": int(truth),
        "H1": int(all(n in last_mention for _c, n in inputs)),
        "H2": int(all(gained[n] >= c for c, n in inputs)),
        "H2_g": min(min(1.0, gained[n] / c) for c, n in inputs),
        "H3": int(recent_gain(3)), "H3_k1": int(recent_gain(1)), "H3_k6": int(recent_gain(6)),
        "H4": int(since_mention <= 2), "H4_g": -since_mention,
        "H5": int(T >= 4), "H5_g": T,
        "H6": int(truth), "H6_g": min(min(1.0, h[n] / c) for c, n in inputs),
        "H7": int(all(h[n] >= 1 for _c, n in inputs)),
        "H8": int(all(last_gain.get(n, -1) > last_consume.get(n, -1) and n in last_gain for _c, n in inputs)),
        "H9": int(not top),
        "H10": 0,
    }
    f["H2andH9"] = int(f["H2"] and f["H9"])
    # Post hoc, added after the fits of H1-H10 had been read (not one of the heuristics fixed in advance): the
    # identical action appears earlier in the history and succeeded ("Crafted ...").
    f["posthoc_repeat"] = int(any(a == item["action"] and r.startswith("Crafted ") for a, r in hist))
    # question 3 quantities
    short = [(c, n) for c, n in inputs if h[n] < c]
    f["shortfall_total"] = sum(c - h[n] for c, n in short)
    f["shortfall_max"] = max((c - h[n] for c, n in short), default=0)
    f["n_short"] = len(short)
    f["surplus_min"] = min(h[n] - c for c, n in inputs)
    f["surplus_ratio_min"] = min(h[n] / c for c, n in inputs)
    uu = [n for c, n in short if n in fell_at]
    if uu:  # used-up inputs: steps since the most recent fall below need, and whether an inventory showed it after
        f["steps_since_fall"] = min(T - fell_at[n] for n in uu)
        f["inventory_after_fall"] = int(any(shown_after_fall[n] for n in uu))
        f["held_zero"] = int(all(h[n] == 0 for n in uu))
    f["any_input_consumed_before"] = int(any(consumed_ever[n] for _c, n in inputs))
    f["is_top"] = int(top)
    return f


# ---------------------------------------------------------------- analyses

def tc_table(items, scores, model):
    """Per item: neutral L (two-candidate, first token, yes/no), the aligned/swapped/neutral arms, decisions."""
    rows = []
    for t in items:
        s = scores[model].get(t["id"], {})
        if "neutral" not in s or "yn_neutral" not in s:
            continue
        r = {"id": t["id"], "pair": t["pair_id"], "pop": t["population"], "role": t["role"],
             "cls": t["failure_class"], "bin": t["length_bin"], "y": int(t["population"] == "success")}
        for c in ("neutral", "aligned", "swapped", "yn_neutral", "yn_aligned", "yn_swapped"):
            r[c] = s[c]["L"] if c in s else np.nan
        r["first_neutral"] = s["neutral"]["L_first"]
        r["mass_yn"] = s["yn_neutral"]["first_mass"]
        rows.append(r)
    return rows


def competence_block(rows, bt, scale):
    """AUC of the neutral score between successes and failures, overall and by class, role, bin."""
    out = {}
    succ = [r for r in rows if r["y"] == 1]
    fails = [r for r in rows if r["y"] == 0]

    def a(F, S):
        rs = F + S
        return auc_ci([r[scale] for r in rs], [r["y"] for r in rs], [r["pair"] for r in rs], bt)

    out["all"] = a(fails, succ)
    for cl in CLASSES:
        out[f"class={cl}"] = a([r for r in fails if r["cls"] == cl], succ)
    for ro in ROLES:
        S = [r for r in succ if r["role"] == ro]
        out[f"role={ro}"] = a([r for r in fails if r["role"] == ro], S)
        for cl in CLASSES:
            F = [r for r in fails if r["role"] == ro and r["cls"] == cl]
            if len(F) >= 20:
                out[f"role={ro}|class={cl}"] = a(F, S)
                for bn in BINS:
                    Fb = [r for r in F if r["bin"] == bn]
                    Sb = [r for r in S if r["bin"] == bn]
                    if len(Fb) >= 20:
                        out[f"role={ro}|class={cl}|bin={bn}"] = a(Fb, Sb)
    for bn in BINS:
        out[f"bin={bn}"] = a([r for r in fails if r["bin"] == bn], [r for r in succ if r["bin"] == bn])
    return out


def decisions_block(rows, bt, scale):
    """Share predicted success (score > 0) on failures (by class and role) and on successes (by role); accuracy."""
    out = {}

    def share(rs):
        return mean_ci([float(r[scale] > 0) for r in rs], [r["pair"] for r in rs], bt)

    F = [r for r in rows if r["y"] == 0]
    S = [r for r in rows if r["y"] == 1]
    out["yes_share_failures"] = share(F)
    out["yes_share_successes"] = share(S)
    for cl in CLASSES:
        out[f"yes_share_failures|class={cl}"] = share([r for r in F if r["cls"] == cl])
    for ro in ROLES:
        out[f"yes_share_failures|role={ro}"] = share([r for r in F if r["role"] == ro])
        out[f"yes_share_successes|role={ro}"] = share([r for r in S if r["role"] == ro])
    acc_f = 1 - (out["yes_share_failures"]["est"] or 0)
    acc_s = out["yes_share_successes"]["est"] or 0
    out["accuracy_failures"], out["accuracy_successes"] = _r(acc_f), _r(acc_s)
    out["balanced_accuracy"] = _r((acc_f + acc_s) / 2)
    out["accuracy_all"] = _r(np.mean([float((r[scale] > 0) == bool(r["y"])) for r in rows]))
    out["mean_L_failures"] = mean_ci([r[scale] for r in F], [r["pair"] for r in F], bt)
    out["mean_L_successes"] = mean_ci([r[scale] for r in S], [r["pair"] for r in S], bt)
    return out


def families_rule(rows, bt, rule) -> dict:
    """The families competence rule (notes/families_metrics.json, rules.competence) on the yes/no neutral readout."""
    a = auc_ci([r["yn_neutral"] for r in rows], [r["y"] for r in rows], [r["pair"] for r in rows], bt)
    F = [r for r in rows if r["y"] == 0]
    S = [r for r in rows if r["y"] == 1]
    yf = float(np.mean([r["yn_neutral"] > 0 for r in F]))
    ys = float(np.mean([r["yn_neutral"] > 0 for r in S]))
    mass = float(np.median([r["mass_yn"] for r in rows]))
    checks = {
        "auc_neutral": {"value": a["est"], "ci": a["ci"], "threshold": f">= {rule['auc_neutral_min']}",
                        "pass": a["est"] >= rule["auc_neutral_min"]},
        "yes_share_neutral_failures": {"value": _r(yf), "threshold": f"<= {rule['yes_share_neutral_failures_max']}",
                                       "pass": yf <= rule["yes_share_neutral_failures_max"]},
        "yes_share_neutral_successes": {"value": _r(ys), "threshold": f">= {rule['yes_share_neutral_successes_min']}",
                                        "pass": ys >= rule["yes_share_neutral_successes_min"]},
        "median_yes_no_mass_neutral": {"value": _r(mass), "threshold": f">= {rule['median_yes_no_mass_neutral_min']}",
                                       "pass": mass >= rule["median_yes_no_mass_neutral_min"]},
    }
    return {"interpretable": all(c["pass"] for c in checks.values()), "checks": checks,
            "n_failures": len(F), "n_successes": len(S)}


HEURISTICS = ("H1", "H2", "H3", "H3_k1", "H3_k6", "H4", "H5", "H6", "H7", "H8", "H9", "H2andH9", "H10")
GRADED = {"H2": "H2_g", "H4": "H4_g", "H5": "H5_g", "H6": "H6_g"}


def heuristic_fits(rows, feats, bt, scale="neutral") -> dict:
    """Validity and fit to the model of each heuristic (measures 1-6 of the note)."""
    out = {}
    y = np.array([r["y"] for r in rows], float)
    L = np.array([r[scale] for r in rows], float)
    pairs = [r["pair"] for r in rows]
    dec = (L > 0).astype(float)
    for h in HEURISTICS:
        hv = np.array([feats[r["id"]][h] for r in rows], float)
        d = {"share_predicting_success": _r(hv.mean())}
        d["validity_auc"] = auc_ci(hv, y, pairs, bt) if hv.std() > 0 else {"est": 0.5}
        for cl in CLASSES:
            m = np.array([(r["y"] == 1) or (r["cls"] == cl) for r in rows])
            if hv[m].std() > 0:
                d[f"validity_auc|class={cl}"] = auc_ci(hv[m], y[m], [p for p, k in zip(pairs, m) if k], bt)["est"]
            else:
                d[f"validity_auc|class={cl}"] = 0.5
        if hv.std() > 0:
            d["tracking_auc"] = auc_ci(L, hv, pairs, bt)
            for lab, mm in (("failures", y == 0), ("successes", y == 1)):
                if hv[mm].std() > 0:
                    d[f"tracking_auc|{lab}"] = auc_ci(L[mm], hv[mm], [p for p, k in zip(pairs, mm) if k], bt)
            if h != "H6" and np.corrcoef(hv, y)[0, 1] < 0.999:
                X = np.c_[np.ones_like(y), y, hv]
                d["joint_regression"] = ols_ci(L, X, pairs, bt, ["intercept", "truth", "heuristic"])
            # used-up subset: used-up failures and all successes
            mu = np.array([(r["y"] == 1) or (r["cls"] == "used_up") for r in rows])
            if hv[mu].std() > 0:
                d["tracking_auc|used_up_and_successes"] = auc_ci(L[mu], hv[mu], [p for p, k in zip(pairs, mu) if k], bt)
        d.update(kappa_ci(dec, hv, pairs, bt))
        if h in GRADED:
            g = np.array([feats[r["id"]][GRADED[h]] for r in rows], float)
            d["graded_validity_auc"] = auc_ci(g, y, pairs, bt)
            d["spearman_graded_vs_L"] = spearman_ci(g, L, pairs, bt)
            mu = np.array([(r["y"] == 1) or (r["cls"] == "used_up") for r in rows])
            d["graded_validity_auc|used_up_vs_successes"] = auc_ci(g[mu], y[mu], [p for p, k in zip(pairs, mu) if k], bt)["est"]
        out[h] = d
    # paired differences of tracking AUC on the same resamples: each heuristic minus the truth (H6), and the two
    # best-fitting heuristics against each other
    Wb = bt.W(pairs)
    one = np.ones((1, len(L)))
    tr = {h: (wauc(L, np.array([feats[r["id"]][h] for r in rows], float), one)[0],
              wauc(L, np.array([feats[r["id"]][h] for r in rows], float), Wb))
          for h in HEURISTICS if h != "H10"}
    for h in tr:
        if h != "H6":
            out[h]["tracking_auc_minus_truth"] = _ci(tr[h][0] - tr["H6"][0], tr[h][1] - tr["H6"][1], len(L), len(set(pairs)))
    out["paired_H1_minus_H2andH9"] = _ci(tr["H1"][0] - tr["H2andH9"][0], tr["H1"][1] - tr["H2andH9"][1], len(L), len(set(pairs)))
    out["paired_H2andH9_minus_H2"] = _ci(tr["H2andH9"][0] - tr["H2"][0], tr["H2andH9"][1] - tr["H2"][1], len(L), len(set(pairs)))
    # post hoc (see features): mean L by repeat x outcome on ingredient crafts whose inputs were all named before
    ph = {}
    for lab, sel in (("used_up_failures", lambda r: r["y"] == 0 and r["cls"] == "used_up"),
                     ("ingredient_successes", lambda r: r["y"] == 1 and r["role"] == "ingredient")):
        for rep in (0, 1):
            sub = [r for r in rows if sel(r) and r["role"] == "ingredient" and feats[r["id"]]["posthoc_repeat"] == rep]
            ph[f"{lab}|repeat={rep}"] = mean_ci([r[scale] for r in sub], [r["pair"] for r in sub], bt)
    ing = [r for r in rows if r["role"] == "ingredient" and (r["y"] == 1 or r["cls"] == "used_up")
           and feats[r["id"]]["posthoc_repeat"] == 1]
    ph["auc_used_up_vs_ingredient_successes|both_repeats"] = auc_ci([r[scale] for r in ing], [r["y"] for r in ing],
                                                                   [r["pair"] for r in ing], bt)
    ph["share_repeat"] = {lab: _r(np.mean([feats[r["id"]]["posthoc_repeat"] for r in rows if sel(r)]))
                          for lab, sel in (("used_up", lambda r: r["y"] == 0 and r["cls"] == "used_up"),
                                           ("too_few", lambda r: r["y"] == 0 and r["cls"] == "too_few"),
                                           ("never_obtained", lambda r: r["y"] == 0 and r["cls"] == "never_obtained"),
                                           ("successes", lambda r: r["y"] == 1))}
    out["posthoc_repeat"] = ph
    return out


def quantities(rows, feats, bt, scale="neutral") -> dict:
    """Question 3: within a class, does L follow the deciding quantities?"""
    out = {}

    def sp(rs, key):
        rs = [r for r in rs if key in feats[r["id"]]]
        return spearman_ci([feats[r["id"]][key] for r in rs], [r[scale] for r in rs], [r["pair"] for r in rs], bt)

    def by_level(rs, key, levels):
        res = {}
        for name, f in levels:
            sub = [r for r in rs if key in feats[r["id"]] and f(feats[r["id"]][key])]
            res[name] = mean_ci([r[scale] for r in sub], [r["pair"] for r in sub], bt)
        return res

    tf = [r for r in rows if r["y"] == 0 and r["cls"] == "too_few"]
    uu = [r for r in rows if r["y"] == 0 and r["cls"] == "used_up"]
    su = [r for r in rows if r["y"] == 1]
    out["too_few"] = {
        "spearman_L_vs_shortfall_total": sp(tf, "shortfall_total"),
        "spearman_L_vs_coverage": sp(tf, "H6_g"),
        "L_by_shortfall": by_level(tf, "shortfall_total", [("1", lambda v: v == 1), ("2-3", lambda v: 2 <= v <= 3),
                                                            ("4-7", lambda v: 4 <= v <= 7), ("8+", lambda v: v >= 8)]),
    }
    for ro in ROLES:
        out["too_few"][f"spearman_L_vs_shortfall_total|role={ro}"] = sp([r for r in tf if r["role"] == ro], "shortfall_total")
    out["used_up"] = {
        "spearman_L_vs_steps_since_fall": sp(uu, "steps_since_fall"),
        "L_by_steps_since_fall": by_level(uu, "steps_since_fall", [("0 (fell at the last step)", lambda v: v == 0),
                                                                    ("1-2", lambda v: 1 <= v <= 2),
                                                                    ("3-5", lambda v: 3 <= v <= 5),
                                                                    ("6+", lambda v: v >= 6)]),
        "L_by_inventory_shown_after_fall": by_level(uu, "inventory_after_fall", [("no", lambda v: v == 0),
                                                                                   ("yes", lambda v: v == 1)]),
        "L_by_held_now": by_level(uu, "held_zero", [("none left", lambda v: v == 1), ("some left", lambda v: v == 0)]),
        "spearman_L_vs_shortfall_total": sp(uu, "shortfall_total"),
        "spearman_L_vs_coverage": sp(uu, "H6_g"),
        "spearman_L_vs_history_length": sp(uu, "H5_g"),
    }
    out["successes"] = {
        "spearman_L_vs_surplus_min": sp(su, "surplus_min"),
        "spearman_L_vs_surplus_ratio_min": sp(su, "surplus_ratio_min"),
        "L_by_surplus": by_level(su, "surplus_min", [("0 (exactly enough)", lambda v: v == 0), ("1-2", lambda v: 1 <= v <= 2),
                                                     ("3-7", lambda v: 3 <= v <= 7), ("8+", lambda v: v >= 8)]),
        "L_by_input_consumed_before": by_level(su, "any_input_consumed_before", [("no", lambda v: v == 0),
                                                                                ("yes", lambda v: v == 1)]),
        "spearman_L_vs_history_length": sp(su, "H5_g"),
    }
    for ro in ROLES:
        out["successes"][f"spearman_L_vs_surplus_min|role={ro}"] = sp([r for r in su if r["role"] == ro], "surplus_min")
    # matched comparison: used-up failures against successes whose inputs were consumed earlier too
    out["used_up_vs_successes_with_prior_consumption"] = auc_ci(
        [r[scale] for r in su + uu if r["y"] == 0 or feats[r["id"]]["any_input_consumed_before"]],
        [r["y"] for r in su + uu if r["y"] == 0 or feats[r["id"]]["any_input_consumed_before"]],
        [r["pair"] for r in su + uu if r["y"] == 0 or feats[r["id"]]["any_input_consumed_before"]], bt)
    out["used_up_vs_successes_by_bin"] = {
        bn: auc_ci([r[scale] for r in su + uu if r["bin"] == bn], [r["y"] for r in su + uu if r["bin"] == bn],
                   [r["pair"] for r in su + uu if r["bin"] == bn], bt) for bn in BINS}
    out["mean_history_length"] = {"used_up": _r(np.mean([feats[r["id"]]["H5_g"] for r in uu])),
                                  "successes": _r(np.mean([feats[r["id"]]["H5_g"] for r in su])),
                                  "too_few": _r(np.mean([feats[r["id"]]["H5_g"] for r in tf]))}
    return out


def competence_vs_effect(rows, bt, prefix="") -> dict:
    """Question 4: per-item and per-stratum relation between neutral correctness / discrimination and the effects."""
    n, a, s = f"{prefix}neutral", f"{prefix}aligned", f"{prefix}swapped"
    out = {"per_item": {}, "strata": [], "strata_correlations": {}}
    F = [r for r in rows if r["y"] == 0]
    S = [r for r in rows if r["y"] == 1]

    def eff(rs):
        p = [r["pair"] for r in rs]
        return {"dL": mean_ci([r[a] - r[s] for r in rs], p, bt), "a_minus_n": mean_ci([r[a] - r[n] for r in rs], p, bt),
                "s_minus_n": mean_ci([r[s] - r[n] for r in rs], p, bt)}

    for ro in ROLES + ("all",):
        Fr = [r for r in F if ro == "all" or r["role"] == ro]
        for ok in (True, False):
            sub = [r for r in Fr if (r[n] < 0) == ok]
            out["per_item"][f"role={ro}|neutral_correct={ok}"] = {"n": len(sub), **eff(sub)}
        margin = [-r[n] for r in Fr]
        out["per_item"][f"role={ro}|spearman_margin_vs_dL"] = spearman_ci(margin, [r[a] - r[s] for r in Fr],
                                                                         [r["pair"] for r in Fr], bt)
    for ro in ROLES:
        Sr = [r for r in S if r["role"] == ro]
        for cl in CLASSES:
            Fc = [r for r in F if r["role"] == ro and r["cls"] == cl]
            if len(Fc) < 20:
                continue
            for bn in BINS + ("all",):
                Fb = [r for r in Fc if bn == "all" or r["bin"] == bn]
                Sb = [r for r in Sr if bn == "all" or r["bin"] == bn]
                if len(Fb) < 20:
                    continue
                rs = Fb + Sb
                au = auc_ci([r[n] for r in rs], [r["y"] for r in rs], [r["pair"] for r in rs], bt)
                out["strata"].append({"role": ro, "class": cl, "bin": bn, "n": len(Fb), "auc": au,
                                      "neutral_correct_share": _r(np.mean([r[n] < 0 for r in Fb])), **eff(Fb)})
    from scipy.stats import spearmanr
    st = [x for x in out["strata"] if x["bin"] != "all"]
    for k in ("dL", "a_minus_n", "s_minus_n"):
        rho = spearmanr([x["auc"]["est"] for x in st], [x[k]["est"] for x in st])
        out["strata_correlations"][f"spearman_auc_vs_{k}"] = {"rho": _r(rho.statistic), "n_strata": len(st)}
    for ro in ROLES:
        Sr = [r for r in S if r["role"] == ro]
        out[f"successes|role={ro}"] = eff(Sr)
    return out


# ---------------------------------------------------------------- ALFWorld

def alfworld(archive: str, rule: dict) -> dict:
    A = Path(archive)
    rd = lambda pat: [s for p in sorted(glob.glob(str(A / pat))) for s in read_jsonl(p)]
    rq1 = rd("text-wm-fix12/runs/rq1_score/shard_*/scoring.jsonl")
    rq1x = rd("text-wm-fix13/runs/rq1x_score/shard_*/scoring.jsonl")
    rq1v = rd("text-wm-fix17/runs/rq1v_score/shard_*/scoring.jsonl")
    out = {}
    for m in MODELS:
        two_f = {s["transition_id"]: s for s in rq1 if s["model"] == m and s["condition"] == "absent"}
        two_s = {s["transition_id"]: s for s in rq1x if s["model"] == m and s["condition"] == "absent"}
        yn = {s["transition_id"]: s for s in rq1v if s["model"] == m and s["condition"] == "yn_absent"}
        rows = []
        for tid, s in list(two_f.items()) + list(two_s.items()):
            if tid in yn:
                rows.append({"id": tid, "pair": s["game_id"], "y": int(tid in two_s), "neutral": s["L"],
                             "yn_neutral": yn[tid]["L"], "mass_yn": yn[tid]["first_mass"]})
        bt = Boot([r["pair"] for r in rows])
        y = [r["y"] for r in rows]
        res = {"n_failures": y.count(0), "n_successes": y.count(1), "games": len({r["pair"] for r in rows})}
        for sc in ("neutral", "yn_neutral"):
            res[sc] = {"auc": auc_ci([r[sc] for r in rows], y, [r["pair"] for r in rows], bt),
                       "decisions": _dec_simple(rows, bt, sc),
                       "calibration": calibration([r[sc] for r in rows], y)}
        res["families_rule"] = families_rule(rows, bt, rule)
        out[SHORT[m]] = res
    return out


def _dec_simple(rows, bt, scale):
    F = [r for r in rows if r["y"] == 0]
    S = [r for r in rows if r["y"] == 1]
    sh = lambda rs: mean_ci([float(r[scale] > 0) for r in rs], [r["pair"] for r in rs], bt)
    out = {"yes_share_failures": sh(F), "yes_share_successes": sh(S),
           "mean_L_failures": mean_ci([r[scale] for r in F], [r["pair"] for r in F], bt),
           "mean_L_successes": mean_ci([r[scale] for r in S], [r["pair"] for r in S], bt)}
    out["balanced_accuracy"] = _r((1 - out["yes_share_failures"]["est"] + out["yes_share_successes"]["est"]) / 2)
    return out


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = read_jsonl(ITEMS)
    rule = json.loads((REPO / "notes" / "families_metrics.json").read_text())["rules"]["competence"]
    scores = load_tc_scores(args.scores)
    n_records = sum(len(c) for m in scores.values() for c in m.values())
    core = [t for t in items if t["population"] in ("main", "success")]
    withuse = [t for t in items if t["population"] in ("main", "success", "usefulness")]
    feats = {t["id"]: features(t) for t in core}
    assert all(feats[t["id"]]["truth"] == int(t["population"] == "success") for t in core), "accounting disagrees"
    h9_goal = all(feats[t["id"]]["is_top"] == int(t["role"] == "goal") for t in core)
    bt = Boot([t["pair_id"] for t in items])

    out = {"note": "Exploratory, post hoc (notes/textcraft_llama_competence.md). Neutral goal line unless stated. "
                   "95% intervals: 2,000 resamples of whole goal pairs (TextCraft) or games (ALFWorld).",
           "inputs": {"textcraft_records": n_records, "items_main": 1500, "items_success": 600,
                      "items_usefulness": 600, "top_item_equals_goal_role": h9_goal},
           "families_rule": rule, "textcraft": {}, "alfworld": None}
    for m in MODELS:
        rows = tc_table(core, scores, m)
        rows_u = tc_table(withuse, scores, m)
        res = {"n_items": len(rows)}
        res["competence"] = {sc: competence_block(rows, bt, sc) for sc in ("neutral", "yn_neutral", "first_neutral")}
        res["competence_with_usefulness"] = {sc: competence_block(rows_u, bt, sc)["all"] for sc in ("neutral", "yn_neutral")}
        res["decisions"] = {sc: decisions_block(rows, bt, sc) for sc in ("neutral", "yn_neutral")}
        res["calibration"] = {sc: calibration([r[sc] for r in rows], [r["y"] for r in rows]) for sc in ("neutral", "yn_neutral")}
        res["families_rule"] = families_rule(rows, bt, rule)
        res["families_rule_with_usefulness"] = families_rule(rows_u, bt, rule)
        res["heuristics"] = {"neutral": heuristic_fits(rows, feats, bt, "neutral"),
                             "yn_neutral": heuristic_fits(rows, feats, bt, "yn_neutral")}
        res["quantities"] = {"neutral": quantities(rows, feats, bt, "neutral"),
                             "yn_neutral": quantities(rows, feats, bt, "yn_neutral")}
        res["competence_vs_effect"] = {"two_candidate": competence_vs_effect(rows, bt),
                                       "yes_no": competence_vs_effect(rows, bt, prefix="yn_")}
        out["textcraft"][SHORT[m]] = res
        print(m, "done", file=sys.stderr)

    # the inventory-shown prompt variant: competence and the H2 / H6 fits on its 500 items, against the verified prompt
    pv = load_tc_scores(args.prompts)
    inv = {}
    for m in MODELS:
        ids = set(pv[m])
        sub = [t for t in core if t["id"] in ids]
        rows_v = tc_table(sub, scores, m)
        rows_i = []
        for r in rows_v:
            s = pv[m][r["id"]]
            rows_i.append({**r, "neutral": s["inventory|neutral"]["L"]})
        d = {"n": len(rows_i)}
        for lab, rs in (("verified", rows_v), ("inventory_shown", rows_i)):
            hf = heuristic_fits(rs, feats, bt, "neutral")
            d[lab] = {"auc": competence_block(rs, bt, "neutral")["all"],
                      "auc_used_up": competence_block(rs, bt, "neutral")["class=used_up"],
                      **{h: {k: hf[h].get(k) for k in ("tracking_auc", "tracking_auc|failures", "kappa",
                                                       "tracking_auc|used_up_and_successes", "spearman_graded_vs_L")}
                         for h in ("H1", "H2", "H4", "H6", "H7", "H8", "H9")}}
        inv[SHORT[m]] = d
    out["inventory_shown_variant"] = inv
    out["alfworld"] = alfworld(args.archive, rule)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
