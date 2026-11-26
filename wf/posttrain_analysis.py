"""Post-training pilot: the frozen analysis (notes/posttraining_design.md, sections 8-10).

    python -m wf.posttrain_analysis --runs runs/posttrain --lexsem <lexsem runs dir> --out notes/posttraining_metrics.json

Reads every arm's scores (runs/posttrain/<arm>/scoring.jsonl), merges the cells lexsem scored in the identical
environment (after the bit-identity check on the reuse-check items), applies the gates, computes the estimands per arm,
stratum and readout with one game-clustered bootstrap per stratum shared by every arm, and applies the reading rules
to the pre-specified stage comparisons. It refuses a dirty tree, mixed environments within a comparison, and
mock or unfinished runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from wf import posttrain as P

SEED = 20261126
B_DEFAULT = 2000
STRATA = ("later", "step0")
READOUTS = ("first", "cont", "full", "yn")
PRIMARY_STRATUM = "later"
# gates (design section 5)
AUC_MIN, DEGENERATE_HI, DEGENERATE_LO, MASS_FIRST, MASS_YN = 0.70, 0.95, 0.05, 0.7, 0.9
# reading margins (design section 8)
STD_MARGIN, ZERO_MARGIN, SHARE_MARGIN, RANK_MARGIN = 0.15, 0.10, 0.25, 0.05
FLOOR_MULT, FRAGILE_MULT = 5.0, 2.0
REF_FRACTION = 0.5
HOLM_ALPHA = 0.05
MENTIONS = tuple(P.PAIRS)  # name_assert, name_deny, name_use, name_ignore

# Pre-specified stage comparisons, primary (plain format, Holm family): (from, to, label)
PRIMARY_PAIRS = (
    ("llama-base_plain", "tulu-sft_plain", "base -> Tulu SFT"),
    ("tulu-sft_plain", "tulu-dpo_plain", "Tulu SFT -> DPO"),
    ("tulu-dpo_plain", "tulu-rl_plain", "Tulu DPO -> RLVR (PPO)"),
    ("llama-base_plain", "tulu-rl_plain", "base -> Tulu final"),
    ("llama-base_plain", "llama-instruct_plain", "base -> Meta Instruct"),
    ("tulu-rl_plain", "llama-instruct_plain", "Tulu final vs Meta Instruct (sibling recipes)"),
    ("qwen-base_plain", "qwen_plain", "Qwen3-8B base -> post-trained"),
)
SECONDARY_PAIRS = (
    ("tulu-dpo_plain", "tulu31-grpo_plain", "Tulu DPO -> RLVR (GRPO)"),
    ("tulu-rl_plain", "tulu31-grpo_plain", "PPO vs GRPO from the same DPO checkpoint"),
    ("tulu-sft_native", "tulu-dpo_native", "Tulu SFT -> DPO, native template"),
    ("tulu-dpo_native", "tulu-rl_native", "Tulu DPO -> RLVR (PPO), native template"),
    ("tulu-dpo_native", "tulu31-grpo_native", "Tulu DPO -> RLVR (GRPO), native template"),
    ("tulu-rl_native", "llama-instruct_native", "Tulu final vs Meta Instruct, native templates"),
)
FORMAT_PAIRS = tuple((f"{ck}_plain", f"{ck}_native") for ck in P.NATIVE)
LINEAGE_PLAIN = {"llama": ("llama-base_plain", "tulu-sft_plain", "tulu-dpo_plain", "tulu-rl_plain",
                           "tulu31-grpo_plain", "llama-instruct_plain"),
                 "qwen": ("qwen-base_plain", "qwen_plain")}
SEVEN = {a: a + "7" for arms_ in LINEAGE_PLAIN.values() for a in arms_}


# ---------------- loading ----------------

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _cell_equal(a: dict, b: dict) -> bool:
    keys = ("lp_a", "lp_b", "tok_lp")
    return all(a.get(k) == b.get(k) for k in keys if k in a or k in b)


def merge_lexsem(arm: str, recs: list[dict], lexsem_dir: Path | None, env: dict) -> tuple[list[dict], dict]:
    """Add the cells lexsem scored for this checkpoint and format. Allowed only if lexsem's environment record
    matches this arm's (GPU type, stack, revision, dtype, unbatched, mode) and every reuse-check cell (the first five
    failures) is bit-identical; otherwise the arm is marked for rescoring and not read."""
    lx = P.LEXSEM_REUSE.get(arm)
    if not lx:
        return recs, {"reused": False}
    status: dict = {"lexsem_arm": lx}
    if lexsem_dir is None:
        return recs, {**status, "ok": False, "why": "no lexsem directory given"}
    lenv_p = lexsem_dir / f"{lx}_env.json"
    lrecs = read_jsonl(lexsem_dir / f"{lx}.jsonl")
    lsucc = read_jsonl(lexsem_dir / f"{lx}_successes.jsonl")
    if not lenv_p.exists() or not lrecs:
        return recs, {**status, "ok": False, "why": "lexsem scores missing"}
    lenv = json.loads(lenv_p.read_text())
    gpu_file = lexsem_dir / f"{lx}_gpu.txt"
    lgpu = gpu_file.read_text().strip() if gpu_file.exists() else lenv.get("gpu", "")
    keys = ("torch", "transformers", "python", "revision", "dtype", "unbatched", "mode")
    diffs = {k: (env.get(k), lenv.get(k)) for k in keys if env.get(k) != lenv.get(k)}
    if "L40S" not in str(lgpu).replace("_", " ") or "L40S" not in str(env.get("gpu")):
        diffs["gpu"] = (env.get("gpu"), lgpu)
    if diffs:
        return recs, {**status, "ok": False, "why": f"environment differs: {diffs}"}
    by = {r["id"]: r for r in lrecs}
    sby = {r["id"]: r for r in lsucc}
    checked, mismatched = 0, []
    for r in recs:
        for c, fm in (r.get("reuse_check") or {}).items():
            for f, cell in fm.items():
                other = by.get(r["id"], {}).get("scores", {}).get(c, {}).get(f)
                checked += 1
                if other is None or not _cell_equal(cell, other):
                    mismatched.append(f"{r['id']}|{c}|{f}")
    if checked == 0 or mismatched:
        return recs, {**status, "ok": False, "why": f"bit-identity check: {len(mismatched)} of {checked} differ",
                      "mismatched": mismatched[:10]}
    cells = P.lexsem_cells(lx)
    out = []
    for r in recs:
        src = by if r["item_set"] == "failure" else sby
        lr = src.get(r["id"])
        sc = {c: dict(v) for c, v in r["scores"].items()}
        for c, f in cells[r["item_set"]]:
            if lr and c in lr["scores"] and f in lr["scores"][c]:
                sc.setdefault(c, {}).setdefault(f, lr["scores"][c][f])
        out.append({**r, "scores": sc})
    return out, {**status, "ok": True, "reused": True, "n_checked_cells": checked}


def load_arm(runs: Path, arm: str, lexsem_dir: Path | None) -> dict | None:
    d = runs / arm
    recs = read_jsonl(d / "scoring.jsonl")
    if not recs:
        return None
    env = json.loads((d / "env.json").read_text())
    recs, reuse = merge_lexsem(arm, recs, lexsem_dir, env)
    sha = (d / "git_sha").read_text().strip() if (d / "git_sha").exists() else None
    status = (d / "STATUS.md").read_text().splitlines()[0] if (d / "STATUS.md").exists() else ""
    recog = read_jsonl(d / "recognition.jsonl")
    jobs = read_jsonl(d / "jobs.jsonl")
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    return {"arm": arm, "config": config, "env": env, "commit": sha, "status": status, "reuse": reuse, "jobs": jobs,
            "fail": {r["id"]: r for r in recs if r["item_set"] == "failure"},
            "succ": {r["id"]: r for r in recs if r["item_set"] == "success"},
            "recognition": {(x["object"], x["control"]): x["passed"] for x in recog}}


# ---------------- readouts ----------------

def readout(rec: dict, cond: str, ro: str) -> float | None:
    """first: the decision at the reply's first position, log p(first token of the success text) - log p("Nothing");
    full: log p(success text) - log p("Nothing happens."), each from its own sequence; cont = full - first;
    yn: log p("Yes") - log p("No")."""
    s = rec["scores"].get(cond, {})
    if ro == "yn":
        y = s.get("yn")
        return None if y is None else y["lp_a"] - y["lp_b"]
    o = s.get("obs")
    if o is None:
        return None
    first = o["tok_lp"][0] - o["lp_b"]
    if ro == "first":
        return first
    f = s.get("fail")
    if f is None:
        return None
    full = sum(o["tok_lp"]) - sum(f["tok_lp"])
    return full if ro == "full" else full - first


def mass(rec: dict, cond: str, ro: str) -> float | None:
    s = rec["scores"].get(cond, {})
    if ro == "yn":
        y = s.get("yn")
        return None if y is None else math.exp(y["lp_a"]) + math.exp(y["lp_b"])
    o = s.get("obs")
    return None if o is None else math.exp(o["tok_lp"][0]) + math.exp(o["lp_b"])


# ---------------- bootstrap ----------------

class Boot:
    """Game-clustered bootstrap by draw counts, one per stratum, shared by every arm (all arms score the same
    items), so paired differences between arms use the same draws."""

    def __init__(self, games: list[str], B: int, seed: int):
        self.games = sorted(set(games))
        self.index = {g: i for i, g in enumerate(self.games)}
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.games), size=(B, len(self.games)))
        self.counts = np.zeros((B, len(self.games)))
        np.add.at(self.counts, (np.repeat(np.arange(B), len(self.games)), draws.ravel()), 1)

    def sums(self, games: list[str], x: np.ndarray) -> tuple[float, np.ndarray]:
        per = np.zeros(len(self.games))
        np.add.at(per, [self.index[g] for g in games], x)
        return float(per.sum()), self.counts @ per


def iv(point: float, draws: np.ndarray, level: float = 0.95) -> dict:
    d = draws[np.isfinite(draws)]
    a = (1 - level) / 2
    lo, hi = np.quantile(d, [a, 1 - a]) if len(d) else (math.nan, math.nan)
    lo90, hi90 = np.quantile(d, [0.05, 0.95]) if len(d) else (math.nan, math.nan)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean()) if len(d) else math.nan
    return {"estimate": float(point), "ci": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)],
            "p_two_sided": float(min(1.0, p)), "n_draws_undefined": int(len(draws) - len(d))}


class Stat:
    """Per-item vectors on one stratum's items, evaluated on the shared draws."""

    def __init__(self, boot: Boot, games: list[str]):
        self.boot, self.games = boot, games
        self.k, self.kd = boot.sums(games, np.ones(len(games)))

    def mean(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        s, d = self.boot.sums(self.games, x)
        return s / self.k, d / self.kd

    def std(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        m, md = self.mean(x)
        q, qd = self.mean(x * x)
        n, nd = self.k, self.kd
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = math.sqrt(max(q - m * m, 0) * n / max(n - 1, 1))
            sdd = np.sqrt(np.maximum(qd - md * md, 0) * nd / np.maximum(nd - 1, 1))
            return (m / sd if sd else math.nan), md / sdd

    def ratio(self, num: np.ndarray, den: np.ndarray) -> tuple[float, np.ndarray]:
        a, ad = self.boot.sums(self.games, num)
        b, bd = self.boot.sums(self.games, den)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a / b if b else math.nan), ad / bd


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return math.nan
    neg_s = np.sort(np.asarray(neg))
    pos_a = np.asarray(pos)
    lt = np.searchsorted(neg_s, pos_a, side="left")
    le = np.searchsorted(neg_s, pos_a, side="right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(pos) * len(neg)))


# ---------------- gates ----------------

def competence(arm: dict, ro: str) -> dict:
    """Under the neutral task only (no goal contrast is read): AUC of successes against failures, degenerate
    answering, median readout mass. Failures of both strata are pooled."""
    f = [readout(r, P.NEUTRAL, ro) for r in arm["fail"].values()]
    s = [readout(r, P.NEUTRAL, ro) for r in arm["succ"].values()]
    f, s = [x for x in f if x is not None], [x for x in s if x is not None]
    if not f or not s:
        return {"ok": False, "why": "neutral task not scored"}
    m_ro = "yn" if ro == "yn" else "first"
    masses = [mass(r, P.NEUTRAL, m_ro) for r in list(arm["fail"].values()) + list(arm["succ"].values())]
    masses = [x for x in masses if x is not None]
    out = {"auc": auc(s, f), "share_failures_positive": float(np.mean([x > 0 for x in f])),
           "share_successes_positive": float(np.mean([x > 0 for x in s])),
           "median_mass": float(np.median(masses)) if masses else math.nan, "n_failures": len(f), "n_successes": len(s)}
    out["ok_auc"] = out["auc"] >= AUC_MIN
    out["ok_degenerate"] = out["share_failures_positive"] <= DEGENERATE_HI and out["share_successes_positive"] >= DEGENERATE_LO
    out["ok_mass"] = out["median_mass"] >= (MASS_YN if ro == "yn" else MASS_FIRST)
    out["ok"] = out["ok_auc"] and out["ok_degenerate"] and out["ok_mass"]
    return out


def arm_gates(arm: dict) -> dict:
    """An arm is interpretable if its full two-candidate readout passes discrimination and degeneracy and its
    first-token mass passes (design section 5). The decision token (first) and the continuation are then read; the
    first token also needs its own discrimination and degeneracy to be read as a decision readout. Yes/no is gated
    on its own (native arms only)."""
    g = {ro: competence(arm, ro) for ro in ("first", "full", "cont", "yn")}
    full_ok = g["full"].get("ok_auc", False) and g["full"].get("ok_degenerate", False) and g["first"].get("ok_mass", False)
    return {"by_readout": g, "arm_ok": bool(full_ok),
            "read": {"full": bool(full_ok), "cont": bool(full_ok),
                     "first": bool(full_ok and g["first"].get("ok_auc") and g["first"].get("ok_degenerate")),
                     "yn": bool(g["yn"].get("ok", False))}}


# ---------------- estimands ----------------

def item_ids(arm: dict, stratum: str) -> list[str]:
    return sorted(i for i, r in arm["fail"].items() if r["stratum"] == stratum)


def vec(arm: dict, ids: list[str], cond: str, ro: str) -> np.ndarray:
    out = [readout(arm["fail"][i], cond, ro) for i in ids]
    return np.array([np.nan if x is None else x for x in out], dtype=float)


def rank_gain(arm: dict, ids: list[str], ro: str, cond_hi: str, cond_lo: str) -> np.ndarray:
    """Within-item rank of a condition among the item's 12 failure lines (average ranks for ties), minus the
    swapped line's rank, over 11: invariant to any monotone rescaling of the item's scores."""
    out = []
    for i in ids:
        vals = {c: readout(arm["fail"][i], c, ro) for c in P.FAIL_CONDS}
        if any(v is None for v in vals.values()):
            out.append(np.nan)
            continue
        xs = np.array(list(vals.values()))
        order = {c: (xs < v).sum() + 0.5 * ((xs == v).sum() - 1) for c, v in vals.items()}
        out.append((order[cond_hi] - order[cond_lo]) / (len(xs) - 1))
    return np.array(out)


def recognised_ids(arms: dict, lineage_arms: list[str], ids: list[str]) -> list[str]:
    """Items whose (object, control) pair every checkpoint of the lineage recognises (plain-arm check)."""
    out = []
    for i in ids:
        ok = True
        for a in lineage_arms:
            rec = arms[a]
            pair = rec.get("_pair_of", {}).get(i)
            if pair is None or not rec["recognition"].get(pair, False):
                ok = False
                break
        if ok:
            out.append(i)
    return out


def ref_ok(g: dict) -> bool:
    lo = g["ci"][0]
    return bool(lo > 0 and lo >= REF_FRACTION * g["estimate"])


def estimands(arm: dict, ids: list[str], games: list[str], boot: Boot, ro: str, desc_ids: list[str] | None) -> dict:
    """Every per-arm quantity on one stratum and readout, with its bootstrap draws kept for paired differences."""
    st = Stat(boot, games)
    al, sw = vec(arm, ids, "aligned", ro), vec(arm, ids, "swapped", ro)
    d = al - sw
    if np.isnan(d).any():
        return {"missing": int(np.isnan(d).sum())}
    out: dict = {"n_items": len(ids)}
    draws: dict = {}

    def put(name, pair):
        out[name] = iv(*pair)
        draws[name] = pair[1]

    put("G", st.mean(d))
    put("G_std", st.std(d))
    put("net_share_moved", st.mean(np.sign(d)))
    rg = rank_gain(arm, ids, ro, "aligned", "swapped")
    put("rank_gain", st.mean(rg))
    out["G_reference_ok"] = ref_ok(out["G"])
    for c in MENTIONS:
        k = P.PAIRS[c]
        net = vec(arm, ids, c, ro) - vec(arm, ids, k, ro)
        if np.isnan(net).any():
            continue
        put(f"delta|{c}", st.mean(net))
        put(f"delta_std|{c}", st.std(net))
        put(f"share|{c}", st.ratio(net, d))
        put(f"rank_share|{c}", st.ratio(rank_gain(arm, ids, ro, c, k), rg))
    if "share|name_assert" in draws and "share|name_deny" in draws:
        put("C", (out["share|name_assert"]["estimate"] - out["share|name_deny"]["estimate"],
                  draws["share|name_assert"] - draws["share|name_deny"]))
    if "share|name_use" in draws:
        out["B"] = out["share|name_use"]
        draws["B"] = draws["share|name_use"]
    neu = vec(arm, ids, P.NEUTRAL, ro)
    put("neutral_minus_swapped", st.mean(neu - sw))
    if desc_ids:
        pos = [ids.index(i) for i in desc_ids if i in ids]
        if pos:
            sub_games = [games[p] for p in pos]
            st2 = Stat(boot, sub_games)
            dd = vec(arm, [ids[p] for p in pos], P.DESC, ro) - sw[pos]
            if not np.isnan(dd).any():
                put("delta|described_goal", st2.mean(dd))
                put("share|described_goal", st2.ratio(dd, d[pos]))
                out["described_goal_n_items"] = len(pos)
    out["_draws"] = draws
    return out


def certainty_quartiles(arm: dict, ids: list[str], ro: str) -> dict:
    """Shares by quartile of the neutral task's score (a prompt in neither the contrast nor the goal effect);
    descriptive, point estimates only."""
    neu = vec(arm, ids, P.NEUTRAL, ro)
    if np.isnan(neu).any():
        return {}
    q = np.quantile(neu, [0.25, 0.5, 0.75])
    bins = np.digitize(neu, q)
    d = vec(arm, ids, "aligned", ro) - vec(arm, ids, "swapped", ro)
    out = {}
    for b in range(4):
        m = bins == b
        row = {"neutral_range": [float(neu[m].min()), float(neu[m].max())] if m.any() else None,
               "G": float(d[m].mean()) if m.any() else math.nan}
        for c in MENTIONS:
            net = vec(arm, ids, c, ro) - vec(arm, ids, P.PAIRS[c], ro)
            row[c] = float(net[m].sum() / d[m].sum()) if m.any() and d[m].sum() else math.nan
        out[f"q{b + 1}"] = row
    return out


def pi_share(e: dict, games: list[str], boot: Boot, first: dict, full: dict) -> dict | None:
    """The decision token's share of the goal effect, G_first / G_full (full = first + continuation)."""
    if "_draws" not in first or "_draws" not in full:
        return None
    gf, gF = first["_draws"]["G"], full["_draws"]["G"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return iv(first["G"]["estimate"] / full["G"]["estimate"], gf / gF)


# ---------------- floors ----------------

def floor_of(main: dict, floor: dict, ro: str, ids: list[str], games: list[str], boot: Boot) -> dict | None:
    """The shift of the mean paired effect between the main environment (L40S) and the floor rescoring (A40) on
    the 300 later items: floor = sqrt(shift^2 + its bootstrap SD^2), for the goal effect and the name assertion's net
    effect, the larger; also in standardized units (over the item SD of the main goal effect)."""
    ids = [i for i in ids if i in floor["fail"]]
    if len(ids) < 100:
        return None
    gs = [main["fail"][i]["game_id"] for i in ids]
    st = Stat(boot, gs)
    res = {}
    for name, (a, b) in (("G", ("aligned", "swapped")), ("delta_assert", ("name_assert", "ctrl_assert"))):
        m = vec(main, ids, a, ro) - vec(main, ids, b, ro)
        f = vec(floor, ids, a, ro) - vec(floor, ids, b, ro)
        if np.isnan(m).any() or np.isnan(f).any():
            return None
        sh, shd = st.mean(f - m)
        res[name] = {"shift": sh, "floor": float(math.sqrt(sh ** 2 + np.nanstd(shd) ** 2)),
                     "item_sd": float(np.std(m, ddof=1))}
    fl = max(v["floor"] for v in res.values())
    return {"floor_nats": fl, "floor_std": fl / res["G"]["item_sd"] if res["G"]["item_sd"] else math.nan,
            "detail": res, "n_items": len(ids)}


# ---------------- readings ----------------

def holm(ps: dict[str, float]) -> dict[str, float]:
    keys = sorted(ps, key=lambda k: ps[k])
    m, out, run = len(keys), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * ps[k]))
        out[k] = run
    return out


def three_way(ci: list[float], margin: float) -> str:
    lo, hi = ci
    if hi < -margin:
        return "lower"
    if lo >= -margin and hi <= margin:
        return "the same"
    if lo > margin:
        return "higher"
    return "inconclusive"


def fragile(ci: list[float], floor: float, bounds: tuple[float, ...]) -> bool:
    return bool(floor and any(abs(e - b) < FRAGILE_MULT * floor for e in ci for b in bounds))


def std_g_reading(d_std: dict, d_nats: dict, d_net: dict, d_rank: dict, floor_std: float | None,
                  p_holm: float | None) -> dict:
    """A stage difference in the standardized goal effect (design section 8). "higher"/"lower": the paired 95%
    interval excludes zero, the estimate exceeds 0.15 and five floors, the Holm-adjusted p is below 0.05, and nats,
    standardized effect, net share moved and rank gain agree in sign with the rank gain's interval excluding zero and
    its estimate at least 0.05. If the standardized criteria hold but the scales disagree: "scale-dependent".
    "the same": the 90% interval within +-0.15. Otherwise "inconclusive"."""
    e, (lo, hi) = d_std["estimate"], d_std["ci"]
    fl = floor_std if floor_std is not None and math.isfinite(floor_std) else 0.0
    sig = (lo > 0 or hi < 0) and abs(e) > STD_MARGIN and abs(e) > FLOOR_MULT * fl and \
        (p_holm is None or p_holm < HOLM_ALPHA)
    signs = {np.sign(x["estimate"]) for x in (d_std, d_nats, d_net, d_rank)}
    rank_ok = (d_rank["ci"][0] > 0 or d_rank["ci"][1] < 0) and abs(d_rank["estimate"]) >= RANK_MARGIN
    if sig and len(signs) == 1 and rank_ok:
        reading = "higher" if e > 0 else "lower"
    elif sig:
        reading = "scale-dependent"
    elif -STD_MARGIN <= d_std["ci90"][0] and d_std["ci90"][1] <= STD_MARGIN:
        reading = "the same"
    else:
        reading = "inconclusive"
    return {"reading": reading, "numerically_fragile": fragile(d_std["ci"], fl, (-STD_MARGIN, 0.0, STD_MARGIN)),
            "floor_std_used": fl}


def suppressed_reading(g_std: dict, floor_std: float | None) -> str:
    lo, hi = g_std["ci90"]
    if -ZERO_MARGIN <= lo and hi <= ZERO_MARGIN:
        return "suppressed to about zero"
    if g_std["ci"][0] > 0:
        return "above zero"
    if g_std["ci"][1] < 0:
        return "below zero"
    return "not shown to be about zero"


def paired(a: dict, b: dict, key: str) -> dict | None:
    """b minus a on the same draws."""
    if key not in a.get("_draws", {}) or key not in b.get("_draws", {}):
        return None
    return iv(b[key]["estimate"] - a[key]["estimate"], b["_draws"][key] - a["_draws"][key])


# ---------------- driver ----------------

def analyse(arms: dict[str, dict], floors: dict[str, dict], B: int = B_DEFAULT, seed: int = SEED) -> dict:
    out: dict = {"arms": {}, "comparisons": {}, "secondary": {}, "format": {}, "interactions": {}, "bridge": {}}
    gates = {a: arm_gates(x) for a, x in arms.items() if P.ARMS[a]["kind"] == "main"}
    # the seven-example fallback (design section 5): if a lineage's base fails, only its seven-example arms are read
    fallback = {}
    for lin, members in LINEAGE_PLAIN.items():
        if members[0] in gates and not gates[members[0]]["arm_ok"]:
            fallback[lin] = "base failed the competence gate: the lineage's seven-example arms are read instead"
    out["fallback"] = fallback
    out["gates"] = gates
    # shared bootstrap per stratum
    ref = next(iter(arms.values()))
    est: dict = {}
    for stratum in STRATA:
        ids = item_ids(ref, stratum)
        if not ids:
            continue
        games = [ref["fail"][i]["game_id"] for i in ids]
        boot = Boot(games, B, seed + (0 if stratum == "later" else 1))
        # described-goal items: recognised by every checkpoint of the lineage (plain arms)
        lin_arms = {lin: [a for a in members if a in arms] for lin, members in LINEAGE_PLAIN.items()}
        desc_ids = {lin: recognised_ids(arms, lin_arms[lin], ids) for lin in lin_arms}
        for a, x in arms.items():
            if P.ARMS[a]["kind"] == "floor" or any(i not in x["fail"] for i in ids):
                continue
            lin = "qwen" if P.ARMS[a]["ckpt"].startswith("qwen") else "llama"
            for ro in READOUTS:
                if ro == "yn" and "yn" not in P.ARMS[a]["formats"]:
                    continue
                e = estimands(x, ids, games, boot, ro, desc_ids[lin] if P.ARMS[a]["kind"] == "main" else None)
                est[(a, stratum, ro)] = e
            first, full = est.get((a, stratum, "first")), est.get((a, stratum, "full"))
            if first and full and "G" in first and "G" in full:
                est[(a, stratum, "pi")] = {"pi": pi_share(None, games, boot, first, full)}
            if P.ARMS[a]["kind"] == "main":
                out["arms"].setdefault(a, {}).setdefault(stratum, {})["certainty_quartiles"] = {
                    ro: certainty_quartiles(x, ids, ro) for ro in ("first", "full")}
        out.setdefault("described_goal_items", {})[stratum] = {k: len(v) for k, v in desc_ids.items()}
        # floors (later stratum only: the floor rescoring covers the 300 later items)
        if stratum == "later":
            for a, f in floors.items():
                for ro in ("first", "cont", "full"):
                    fl = floor_of(arms[a], f, ro, ids, games, boot) if a in arms else None
                    out.setdefault("floors", {}).setdefault(a, {})[ro] = fl
    # per-arm summaries
    for (a, stratum, ro), e in est.items():
        if ro == "pi":
            out["arms"].setdefault(a, {}).setdefault(stratum, {})["pi"] = e["pi"]
            continue
        clean = {k: v for k, v in e.items() if k != "_draws"}
        if "G_std" in e:
            fl = ((out.get("floors", {}).get(a) or {}).get(ro) or {}).get("floor_std")
            clean["zero_reading"] = suppressed_reading(e["G_std"], fl)
        out["arms"].setdefault(a, {}).setdefault(stratum, {})[ro] = clean

    def compare(a: str, b: str, stratum: str) -> dict:
        res: dict = {}
        for ro in ("first", "cont", "full", "yn"):
            ea, eb = est.get((a, stratum, ro)), est.get((b, stratum, ro))
            if not ea or not eb or "_draws" not in ea or "_draws" not in eb:
                continue
            r: dict = {k: paired(ea, eb, k) for k in ("G", "G_std", "net_share_moved", "rank_gain", "C", "B")}
            for c in MENTIONS:
                r[f"share|{c}"] = paired(ea, eb, f"share|{c}")
                r[f"delta_std|{c}"] = paired(ea, eb, f"delta_std|{c}")
                if r[f"share|{c}"] and c != "name_assert" and r.get("share|name_assert"):
                    da = (eb["_draws"][f"share|{c}"] - ea["_draws"][f"share|{c}"]) - \
                         (eb["_draws"]["share|name_assert"] - ea["_draws"]["share|name_assert"])
                    r[f"stage_by_meaning|{c}"] = iv(r[f"share|{c}"]["estimate"] - r["share|name_assert"]["estimate"], da)
            res[ro] = {k: v for k, v in r.items() if v is not None}
        pa, pb = est.get((a, stratum, "pi")), est.get((b, stratum, "pi"))
        if pa and pb and pa["pi"] and pb["pi"]:
            fa, fA = est[(a, stratum, "first")]["_draws"]["G"], est[(a, stratum, "full")]["_draws"]["G"]
            fb, fB = est[(b, stratum, "first")]["_draws"]["G"], est[(b, stratum, "full")]["_draws"]["G"]
            with np.errstate(divide="ignore", invalid="ignore"):
                res["pi"] = iv(pb["pi"]["estimate"] - pa["pi"]["estimate"], fb / fB - fa / fA)
        return res

    # primary comparisons with Holm over the family (later stratum)
    def resolve(a: str) -> str:
        lin = "qwen" if a.startswith("qwen") else "llama"
        return SEVEN[a] if lin in fallback and a in SEVEN and SEVEN[a] in arms else a

    prim = {}
    for a, b, label in PRIMARY_PAIRS:
        a2, b2 = resolve(a), resolve(b)
        if a2 not in arms or b2 not in arms:
            continue
        prim[label] = {"from": a2, "to": b2, "strata": {s: compare(a2, b2, s) for s in STRATA}}
        prim[label]["strata"] = {s: v for s, v in prim[label]["strata"].items() if v}
    pvals = {}
    for label, c in prim.items():
        later = c["strata"].get("later", {})
        for q, (ro, key) in {"G_std_first": ("first", "G_std"), "G_std_full": ("full", "G_std"),
                             "C": ("first", "C"), "B": ("first", "B")}.items():
            if ro in later and key in later[ro]:
                pvals[f"{label}|{q}"] = later[ro][key]["p_two_sided"]
        if "pi" in later:
            pvals[f"{label}|pi"] = later["pi"]["p_two_sided"]
    adj = holm(pvals)
    out["holm"] = {"n_readings": len(pvals), "adjusted": adj}
    for label, c in prim.items():
        a, b = c["from"], c["to"]
        ga, gb = gates.get(a, {}), gates.get(b, {})
        readings = {}
        for s in c["strata"]:
            r = c["strata"][s]
            rd = {}
            for ro in ("first", "full", "cont"):
                if ro not in r or not (ga.get("read", {}).get(ro) and gb.get("read", {}).get(ro)):
                    rd[ro] = {"reading": "not interpretable (competence gate)"} if ro in r else {"reading": "missing"}
                    continue
                fa = ((out.get("floors", {}).get(a) or {}).get(ro) or {}).get("floor_std") or 0.0
                fb = ((out.get("floors", {}).get(b) or {}).get(ro) or {}).get("floor_std") or 0.0
                q = f"G_std_{ro}"
                p = adj.get(f"{label}|{q}") if s == "later" else None
                g = std_g_reading(r[ro]["G_std"], r[ro]["G"], r[ro]["net_share_moved"], r[ro]["rank_gain"],
                                  math.sqrt(fa ** 2 + fb ** 2), p)
                g["holm_p"] = p
                # shares, read only when both arms' goal effects pass the reference rule
                ea, eb = est[(a, s, ro)], est[(b, s, ro)]
                shares_ok = ea.get("G_reference_ok") and eb.get("G_reference_ok")
                sh = {}
                for k in ("C", "B") + tuple(f"share|{m}" for m in MENTIONS) + \
                        tuple(f"stage_by_meaning|{m}" for m in MENTIONS if m != "name_assert"):
                    if k in r[ro]:
                        sh[k] = three_way(r[ro][k]["ci"], SHARE_MARGIN) if shares_ok else \
                            "not read (a goal effect fails the reference rule)"
                g["shares"] = sh
                rd[ro] = g
            if "pi" in r:
                ok = all(est[(x, s, "full")].get("G_reference_ok") for x in (a, b))
                rd["pi"] = three_way(r["pi"]["ci"], SHARE_MARGIN) if ok else "not read (reference rule)"
            readings[s] = rd
        c["readings"] = readings
        c["strata"] = _strip(c["strata"])
    out["comparisons"] = prim
    for a, b, label in SECONDARY_PAIRS:
        if a in arms and b in arms:
            out["secondary"][label] = {"from": a, "to": b, "strata": _strip({s: compare(a, b, s) for s in STRATA})}
    for a, b in FORMAT_PAIRS:
        if a in arms and b in arms:
            out["format"][f"{a} -> {b}"] = _strip({s: compare(a, b, s) for s in STRATA})
    # stage-by-format interaction: the change of the format effect between successive post-trained stages
    for x, y in (("tulu-sft", "tulu-dpo"), ("tulu-dpo", "tulu-rl"), ("tulu-rl", "llama-instruct")):
        keys = [f"{x}_plain", f"{x}_native", f"{y}_plain", f"{y}_native"]
        if all(k in arms for k in keys):
            res = {}
            for ro in ("first", "full"):
                es = [est.get((k, "later", ro)) for k in keys]
                if all(e and "_draws" in e for e in es):
                    for q in ("G_std",) + tuple(f"share|{m}" for m in MENTIONS):
                        if all(q in e["_draws"] for e in es):
                            d = (es[3]["_draws"][q] - es[2]["_draws"][q]) - (es[1]["_draws"][q] - es[0]["_draws"][q])
                            pt = (es[3][q]["estimate"] - es[2][q]["estimate"]) - (es[1][q]["estimate"] - es[0][q]["estimate"])
                            res[f"{ro}|{q}"] = iv(pt, d)
            out["interactions"][f"stage x format {x} -> {y}"] = res
    # bridge (descriptive): goal effect in the Word2World format, with its own discrimination (swapped task)
    for a, x in arms.items():
        if P.ARMS[a]["kind"] != "bridge":
            continue
        row = {}
        for ro in ("first", "full"):
            f = [readout(r, "swapped", ro) for r in x["fail"].values()]
            s = [readout(r, "swapped", ro) for r in x["succ"].values()]
            row[f"auc_swapped|{ro}"] = auc([v for v in s if v is not None], [v for v in f if v is not None])
        for stratum in STRATA:
            e = {ro: est.get((a, stratum, ro)) for ro in ("first", "cont", "full")}
            row[stratum] = {ro: {k: v for k, v in (e[ro] or {}).items() if k in ("G", "G_std", "net_share_moved", "n_items")}
                            for ro in e}
        out["bridge"][a] = row
    out["reuse"] = {a: x["reuse"] for a, x in arms.items()}
    return out


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "_draws"}
    return x


def check_environment(arms: dict[str, dict]) -> list[str]:
    """Every main and bridge arm on an L40S in one stack, every floor arm on an A40 in the same stack; one revision
    per checkpoint; bf16; unbatched; every job of a run on the same GPU type."""
    problems = []
    stacks = {(x["env"]["torch"], x["env"]["transformers"], x["env"]["python"]) for x in arms.values()}
    if len(stacks) != 1:
        problems.append(f"more than one software stack: {stacks}")
    for a, x in arms.items():
        want = "A40" if P.ARMS[a]["kind"] == "floor" else "L40S"
        gpus = {g for j in x["jobs"] for g in (j.get("gpu") or [])} | {x["env"].get("gpu", "")}
        if not gpus or any(want not in g for g in gpus):
            problems.append(f"{a}: GPU {gpus}, expected {want}")
        if x["env"].get("revision") != P.CHECKPOINTS[P.ARMS[a]["ckpt"]]["revision"]:
            problems.append(f"{a}: revision {x['env'].get('revision')}")
        if x["env"].get("dtype") != "bfloat16" or not x["env"].get("unbatched"):
            problems.append(f"{a}: dtype or batching")
        if x.get("config", {}).get("limit"):
            problems.append(f"{a}: scored with a testing limit")
        if x["reuse"].get("lexsem_arm") and not x["reuse"].get("ok"):
            problems.append(f"{a}: reuse of lexsem cells refused ({x['reuse'].get('why')}); rescore with --no-reuse")
    commits = {x["commit"] for x in arms.values()}
    if len(commits) != 1:
        problems.append(f"scores from more than one commit: {commits}")
    return problems


def attach_pairs(arms: dict[str, dict]) -> None:
    """Each failure item's (object, control) pair, for the recognition filter (from the items record)."""
    from wf import lexsem
    from wf.rq2 import load_items

    its = P.items(load_items())
    pairs = {t["id"]: (t["goal_object"], lexsem.control_object(t)) for t in its["failure"]}
    for x in arms.values():
        x["_pair_of"] = pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/posttrain")
    ap.add_argument("--lexsem", default=None, help="lexsem's run directory (runs/lexsem of its checkout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--allow-partial", action="store_true", help="smoke/testing only: analyse the arms present")
    args = ap.parse_args()
    from wf.util import git_dirty, git_sha

    if git_dirty() and not args.allow_partial:
        raise SystemExit("uncommitted changes to tracked files: refused")
    runs, lx = Path(args.runs), (Path(args.lexsem) if args.lexsem else None)
    arms, floors = {}, {}
    for a in P.ARMS:
        x = load_arm(runs, a, lx)
        if x is None:
            continue
        if not x["status"].startswith("# STATUS") or "PASS" not in x["status"]:
            if not args.allow_partial:
                raise SystemExit(f"{a}: run not finished ({x['status']!r})")
        if "MOCK" in x["status"]:
            raise SystemExit(f"{a}: mock scores refused")
        (floors if P.ARMS[a]["kind"] == "floor" else arms)[a] = x
    problems = check_environment({**arms, **floors})
    if problems and not args.allow_partial:
        raise SystemExit("environment check failed:\n" + "\n".join(problems))
    attach_pairs(arms)
    res = analyse(arms, {a.removesuffix("_floor"): f for a, f in floors.items()}, B=args.B)
    res["environment_problems"] = problems
    res["analysis_commit"] = git_sha()
    res["arms_present"] = sorted(arms)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {args.out}: {len(arms)} arms, {len(floors)} floor arms", flush=True)


if __name__ == "__main__":
    main()
