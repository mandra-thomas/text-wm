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
B_DEFAULT = 10000  # resamples; fine enough for Holm's smallest threshold (0.05/35)
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


# files whose content decides a score: an arm scored at another commit is accepted only if these are identical
SCORER_FILES = ("wf/posttrain.py", "wf/posttrain_score.py", "wf/lexsem.py", "scripts/lexsem/score.py",
                "scripts/qwen_audit/followup_score.py", "wf/scoring/logodds.py", "wf/scoring/prompts.py", "wf/rq1.py",
                "wf/rq1x.py", "wf/rq1v.py", "wf/rq2.py", "wf/data", "wf/constants.py")
# the files that decide a lexsem score (lexsem's commit has no pilot files and its own pin table; the pinned revision
# of the reused checkpoint is compared through the environment records instead) (deviation D-PT1)
LEXSEM_FILES = ("wf/lexsem.py", "scripts/lexsem/score.py", "scripts/qwen_audit/followup_score.py", "wf/scoring",
                "wf/rq1.py", "wf/rq1x.py", "wf/rq1v.py", "wf/rq2.py", "wf/data")


# Deviation D-PT1 (2026-11-26, design note section 16): exactly these file versions are treated as identical.
# Lexsem commit 5aabb99 changed one line inside scripts/lexsem/score.py's smoke(), which no full-run scoring path calls;
# the pair is (path, git blob at one commit, git blob at the other), in either order. Any other difference still fails.
ACCEPTED_BLOB_PAIRS = {("scripts/lexsem/score.py", "fd6dabab20a831fe846146384b461ed0db7645a1",
                        "fcbf7c58c2233ad8102e475f060be51f495bf536")}


def same_code(commit: str | None, ref: str = "HEAD", files: tuple[str, ...] = SCORER_FILES) -> bool:
    """Is every scorer file at `commit` identical to `ref`, apart from the exact file versions listed in
    ACCEPTED_BLOB_PAIRS (deviation D-PT1)?"""
    import subprocess

    if not commit:
        return False
    r = subprocess.run(["git", "diff", "--quiet", commit, ref, "--", *files], capture_output=True)
    if r.returncode == 0:
        return True
    names = subprocess.run(["git", "diff", "--name-only", commit, ref, "--", *files], capture_output=True,
                           text=True)
    if names.returncode != 0:
        return False

    def blob(c: str, path: str) -> str | None:
        out = subprocess.run(["git", "rev-parse", f"{c}:{path}"], capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    for path in [x for x in names.stdout.split("\n") if x]:
        pair = (blob(commit, path), blob(ref, path))
        if not any(p == path and set(pair) == {a, b} for p, a, b in ACCEPTED_BLOB_PAIRS):
            return False
    return True


def merge_lexsem(arm: str, recs: list[dict], lexsem_dir: Path | None, env: dict, config: dict) -> tuple[list[dict], dict]:
    """Add the failure cells lexsem scored for this checkpoint and format (never success cells). Allowed only if
    (i) lexsem's environment record matches this arm's (GPU type, stack, revision, dtype, unbatched, mode); (ii)
    lexsem's run commit is recorded and its scorer files are identical to this analysis commit's; (iii) every
    reuse-check cell (the first five failures of each stratum) is bit-identical; (iv) every reused cell exists for
    every failure. Otherwise the arm is marked for rescoring with --no-reuse and not read."""
    lx = P.LEXSEM_REUSE.get(arm)
    if not lx or not config.get("reuse", True):
        return recs, {"reused": False, "ok": True}
    status: dict = {"lexsem_arm": lx}
    if lexsem_dir is None:
        return recs, {**status, "ok": False, "why": "no lexsem directory given"}
    lenv_p = lexsem_dir / f"{lx}_env.json"
    lrecs = read_jsonl(lexsem_dir / f"{lx}.jsonl")
    if not lenv_p.exists() or not lrecs:
        return recs, {**status, "ok": False, "why": "lexsem scores missing"}
    lenv = json.loads(lenv_p.read_text())
    keys = ("torch", "transformers", "python", "revision", "dtype", "unbatched", "mode")
    diffs = {k: (env.get(k), lenv.get(k)) for k in keys if env.get(k) != lenv.get(k)}
    gpu_file = lexsem_dir / f"{lx}_gpu.txt"
    lgpus = {str(lenv.get("gpu", ""))} | ({gpu_file.read_text().strip().replace("_", " ")} if gpu_file.exists() else set())
    if any("L40S" not in g for g in lgpus) or "L40S" not in str(env.get("gpu")):
        diffs["gpu"] = (env.get("gpu"), sorted(lgpus))
    if diffs:
        return recs, {**status, "ok": False, "why": f"environment differs: {diffs}"}
    cfile = lexsem_dir / f"{lx}_commit.txt"
    lcommit = cfile.read_text().strip() if cfile.exists() else None
    if not same_code(lcommit, files=LEXSEM_FILES):
        return recs, {**status, "ok": False, "why": f"lexsem commit {lcommit} missing or its scorer files differ"}
    by = {r["id"]: r for r in lrecs}
    checked, mismatched, strata = 0, [], set()
    for r in recs:
        for c, fm in (r.get("reuse_check") or {}).items():
            strata.add(r["stratum"])
            for f, cell in fm.items():
                other = by.get(r["id"], {}).get("scores", {}).get(c, {}).get(f)
                checked += 1
                if other is None or not _cell_equal(cell, other):
                    mismatched.append(f"{r['id']}|{c}|{f}")
    if checked == 0 or mismatched or strata != {"later", "step0"}:
        return recs, {**status, "ok": False, "why": f"bit-identity check: {len(mismatched)} of {checked} differ "
                      f"(strata checked {sorted(strata)})", "mismatched": mismatched[:10]}
    cells = sorted(P.lexsem_cells(lx)["failure"] & set(P.arm_cells(arm)["failure"]))
    out, missing = [], 0
    for r in recs:
        sc = {c: dict(v) for c, v in r["scores"].items()}
        if r["item_set"] == "failure":
            lr = by.get(r["id"])
            for c, f in cells:
                cell = (lr or {}).get("scores", {}).get(c, {}).get(f)
                if cell is None:
                    missing += 1
                    continue
                sc.setdefault(c, {}).setdefault(f, cell)
        out.append({**r, "scores": sc})
    n_fail = sum(r["item_set"] == "failure" for r in recs)
    if missing or n_fail != 450:
        return recs, {**status, "ok": False, "why": f"{missing} reused cells missing ({n_fail} failures)"}
    return out, {**status, "ok": True, "reused": True, "n_checked_cells": checked, "lexsem_commit": lcommit}


def load_arm(runs: Path, arm: str, lexsem_dir: Path | None) -> dict | None:
    d = runs / arm
    recs = read_jsonl(d / "scoring.jsonl")
    if not recs:
        return None
    env = json.loads((d / "env.json").read_text())
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    recs, reuse = merge_lexsem(arm, recs, lexsem_dir, env, config)
    sha = (d / "git_sha").read_text().strip() if (d / "git_sha").exists() else None
    status = (d / "STATUS.md").read_text().splitlines()[0] if (d / "STATUS.md").exists() else ""
    recog = read_jsonl(d / "recognition.jsonl")
    jobs = read_jsonl(d / "jobs.jsonl")
    return {"arm": arm, "config": config, "env": env, "commit": sha, "status": status, "reuse": reuse, "jobs": jobs,
            "fail": {r["id"]: r for r in recs if r["item_set"] == "failure"},
            "succ": {r["id"]: r for r in recs if r["item_set"] == "success"},
            "recognition": {x["object"]: x["passed"] for x in recog}}


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
    answering, median readout mass. RQ2's determined successes all have a history, so they are compared with the
    later-step failures, the stratum the primary readings use; step 0 gets a descriptive AUC (arm_gates)."""
    f = [readout(r, P.NEUTRAL, ro) for r in arm["fail"].values() if r["stratum"] == "later"]
    s = [readout(r, P.NEUTRAL, ro) for r in arm["succ"].values()]
    f, s = [x for x in f if x is not None], [x for x in s if x is not None]
    if not f or not s:
        return {"ok": False, "why": "neutral task not scored"}
    m_ro = "yn" if ro == "yn" else "first"
    masses = [mass(r, P.NEUTRAL, m_ro) for r in list(arm["fail"].values()) + list(arm["succ"].values())
              if r["stratum"] != "step0"]
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
    g = {ro: competence(arm, ro) for ro in ("first", "full", "yn")}
    for ro in ("first", "full"):  # descriptive: step-0 failures against the successes
        f0 = [readout(r, P.NEUTRAL, ro) for r in arm["fail"].values() if r["stratum"] == "step0"]
        s0 = [readout(r, P.NEUTRAL, ro) for r in arm["succ"].values()]
        g[ro]["auc_step0_descriptive"] = auc([x for x in s0 if x is not None], [x for x in f0 if x is not None])
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


TASK_LINES = ("aligned", "swapped", P.NEUTRAL)


def rank_gain(arm: dict, ids: list[str], ro: str, cond_hi: str, cond_lo: str,
              among: tuple[str, ...] = P.FAIL_CONDS) -> np.ndarray:
    """Within-item rank of a condition among the item's lines `among` (average ranks for ties), minus the other
    condition's rank, over (number of lines - 1): invariant to any monotone rescaling of the item's scores. The
    robustness rule uses the three task lines (real, swapped, neutral), so mention sentences cannot move it; the
    twelve-line version is descriptive."""
    out = []
    for i in ids:
        vals = {c: readout(arm["fail"][i], c, ro) for c in among}
        if any(v is None for v in vals.values()):
            out.append(np.nan)
            continue
        xs = np.array(list(vals.values()))
        order = {c: (xs < v).sum() + 0.5 * ((xs == v).sum() - 1) for c, v in vals.items()}
        out.append((order[cond_hi] - order[cond_lo]) / (len(xs) - 1))
    return np.array(out)


def recognised_ids(arms: dict, lineage_arms: list[str], ids: list[str]) -> list[str]:
    """Items whose goal object every plain-format checkpoint of the lineage recognises from its description."""
    if not lineage_arms:
        return []
    ref = arms[lineage_arms[0]]
    return [i for i in ids if all(arms[a]["recognition"].get(ref["fail"][i].get("goal_object"), False)
                                  for a in lineage_arms)]


def ref_ok(g: dict) -> bool:
    """lexsem's reference rule, made two-sided (no direction is assumed): the interval excludes zero and its end
    nearest zero is at least half the estimate in absolute value."""
    lo, hi = g["ci"]
    near = lo if g["estimate"] > 0 else -hi
    return bool((lo > 0 or hi < 0) and near >= REF_FRACTION * abs(g["estimate"]))


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
    out["G_item_sd"] = float(np.std(d, ddof=1))
    put("net_share_moved", st.mean(np.sign(d)))
    put("rank_gain", st.mean(rank_gain(arm, ids, ro, "aligned", "swapped", TASK_LINES)))
    rg = rank_gain(arm, ids, ro, "aligned", "swapped")
    put("rank_gain_12", st.mean(rg))
    out["G_reference_ok"] = ref_ok(out["G"])
    sv = [readout(r, P.NEUTRAL, ro) for r in arm["succ"].values()]
    fv = vec(arm, ids, P.NEUTRAL, ro)
    if sv and None not in sv and not np.isnan(fv).any():
        e = float(np.mean(sv) - fv.mean())
        out["E_discrimination"] = e
        out["R_goal_over_discrimination"] = out["G"]["estimate"] / e if e else math.nan
    keep = [k for k, i in enumerate(ids) if not arm["fail"][i].get("fewshot_object")]
    if len(keep) < len(ids):  # the items whose object is in the few-shot examples, left out (descriptive)
        st3 = Stat(boot, [games[k] for k in keep])
        out["without_fewshot_objects"] = {"n_items": len(keep), "G": iv(*st3.mean(d[keep])),
                                          "G_std": iv(*st3.std(d[keep]))}
    for c in MENTIONS:
        k = P.PAIRS[c]
        net = vec(arm, ids, c, ro) - vec(arm, ids, k, ro)
        if np.isnan(net).any():
            continue
        put(f"delta|{c}", st.mean(net))
        put(f"delta_std|{c}", st.std(net))
        put(f"share|{c}", st.ratio(net, d))
        put(f"rank_share|{c}", st.ratio(rank_gain(arm, ids, ro, c, k), rg))
        if "without_fewshot_objects" in out:
            out["without_fewshot_objects"][f"share|{c}"] = iv(*st3.ratio(net[keep], d[keep]))
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
                put("delta_std|described_goal", st2.std(dd))
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
    pairs = (("G", ("aligned", "swapped")),) + tuple((f"delta|{c}", (c, P.PAIRS[c])) for c in MENTIONS)
    for name, (a, b) in pairs:
        m = vec(main, ids, a, ro) - vec(main, ids, b, ro)
        f = vec(floor, ids, a, ro) - vec(floor, ids, b, ro)
        if np.isnan(f).all() and name != "G":
            continue  # pair not in this arm's floor rescoring
        if np.isnan(m).any() or np.isnan(f).any():
            return None
        sh, shd = st.mean(f - m)
        sd = float(np.std(m, ddof=1))
        res[name] = {"shift": sh, "floor": float(math.sqrt(sh ** 2 + np.nanstd(shd) ** 2)), "item_sd": sd,
                     "shift_std": sh / sd if sd else math.nan}
        if name == "G":  # the standardized goal effect itself on both GPU types (Rule 4)
            res[name]["G_std_shift"] = st.std(f)[0] - st.std(m)[0]
    fl = max(v["floor"] for v in res.values())
    sd_g = res["G"]["item_sd"]
    return {"floor_nats": fl, "floor_std": fl / sd_g if sd_g else math.nan,
            "G_std_shift": res["G"]["G_std_shift"], "detail": res, "n_items": len(ids)}


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
                  p_holm: float | None, holm_applies: bool = True) -> dict:
    """A stage difference in the standardized goal effect (design section 8). "higher"/"lower": the paired 95%
    interval excludes zero, the estimate exceeds 0.15 and five floors, the Holm-adjusted p is below 0.05, and nats,
    standardized effect, net share moved and rank gain agree in sign with the rank gain's interval excluding zero and
    its estimate at least 0.05. If the standardized criteria hold but the scales disagree: "scale-dependent".
    "the same": the 90% interval within +-0.15, which must be at least five floors. Otherwise "inconclusive". A
    missing floor makes the reading "not readable (floor not measured)". Where no Holm correction applies (the
    continuation, step 0, secondary comparisons) the same rule gives a descriptive label."""
    if floor_std is None or not math.isfinite(floor_std):
        return {"reading": "not readable (floor not measured)", "numerically_fragile": None, "floor_std_used": None}
    e, (lo, hi) = d_std["estimate"], d_std["ci"]
    fl = floor_std
    p_ok = (p_holm is not None and p_holm < HOLM_ALPHA) if holm_applies else True
    sig = (lo > 0 or hi < 0) and abs(e) > STD_MARGIN and abs(e) > FLOOR_MULT * fl and p_ok
    signs = {np.sign(x["estimate"]) for x in (d_std, d_nats, d_net, d_rank)}
    rank_ok = (d_rank["ci"][0] > 0 or d_rank["ci"][1] < 0) and abs(d_rank["estimate"]) >= RANK_MARGIN
    if sig and len(signs) == 1 and rank_ok:
        reading = "higher" if e > 0 else "lower"
    elif sig:
        reading = "scale-dependent"
    elif -STD_MARGIN <= d_std["ci90"][0] and d_std["ci90"][1] <= STD_MARGIN:
        reading = "the same" if STD_MARGIN >= FLOOR_MULT * fl else "inconclusive (margin under five floors)"
    else:
        reading = "inconclusive"
    if not holm_applies:
        reading += " (descriptive)"
    return {"reading": reading, "numerically_fragile": fragile(d_std["ci"], fl, (-STD_MARGIN, 0.0, STD_MARGIN)),
            "floor_std_used": fl}


def share_reading(ci: list[float], floor_share: float | None, p_holm: float | None, holm_applies: bool) -> str:
    """lexsem's three-way rule with the 0.25 margin; a "higher"/"lower" in the Holm family also needs the
    adjusted p below 0.05; "the same" needs 0.25 to be at least five floors (in share units)."""
    if floor_share is None or not math.isfinite(floor_share):
        return "not readable (floor not measured)"
    r = three_way(ci, SHARE_MARGIN)
    if r in ("higher", "lower") and holm_applies and not (p_holm is not None and p_holm < HOLM_ALPHA):
        r = "inconclusive (Holm)"
    if r == "the same" and SHARE_MARGIN < FLOOR_MULT * floor_share:
        r = "inconclusive (margin under five floors)"
    return r


def suppressed_reading(g_std: dict, floor_std: float | None, shift_std: float | None = None) -> str:
    """A checkpoint's own standardized goal effect: "suppressed to about zero" if its 90% interval lies within
    +-0.10 (at least five floors; "environment-dependent" if the A40 rescoring moves it by more than a fifth of the
    margin, Rule 4)."""
    lo, hi = g_std["ci90"]
    if -ZERO_MARGIN <= lo and hi <= ZERO_MARGIN:
        if floor_std is None or not math.isfinite(floor_std):
            return "within +-0.10, not readable (floor not measured)"
        if ZERO_MARGIN < FLOOR_MULT * floor_std:
            return "within +-0.10, inconclusive (margin under five floors)"
        if shift_std is not None and abs(shift_std) > ZERO_MARGIN / 5:
            return "suppressed to about zero (environment-dependent)"
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
            fx = (out.get("floors", {}).get(a) or {}).get(ro) or {}
            clean["zero_reading"] = suppressed_reading(e["G_std"], fx.get("floor_std"), fx.get("G_std_shift"))
            if stratum == "step0":
                clean["zero_reading"] += " (floor measured on the later items)"
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
            if ea.get("G_item_sd"):
                res[ro]["item_sd_ratio"] = eb["G_item_sd"] / ea["G_item_sd"]
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
    # the Holm family is fixed at 7 comparisons x 5 quantities = 35; a missing or ungated reading counts as p = 1
    qmap = {"G_std_first": ("first", "G_std"), "G_std_full": ("full", "G_std"), "C": ("first", "C"),
            "B": ("first", "B"), "pi": ("pi", None)}
    pvals = {}
    for a0, b0, label in PRIMARY_PAIRS:
        later = prim.get(label, {}).get("strata", {}).get("later", {})
        a2, b2 = resolve(a0), resolve(b0)
        for q, (ro, key) in qmap.items():
            gate_ro = ("first", "full") if ro == "pi" else (ro,)
            gated = all(gates.get(x, {}).get("read", {}).get(g_) for x in (a2, b2) for g_ in gate_ro)
            if ro == "pi":
                pvals[f"{label}|{q}"] = later["pi"]["p_two_sided"] if "pi" in later and gated else 1.0
            else:
                ok = ro in later and key in later[ro] and gated
                pvals[f"{label}|{q}"] = later[ro][key]["p_two_sided"] if ok else 1.0
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
                fxa = (out.get("floors", {}).get(a) or {}).get(ro) or {}
                fxb = (out.get("floors", {}).get(b) or {}).get(ro) or {}
                fa, fb = fxa.get("floor_std"), fxb.get("floor_std")
                pair_floor = math.sqrt(fa ** 2 + fb ** 2) if fa is not None and fb is not None else None
                q = f"G_std_{ro}"
                in_family = s == "later" and ro in ("first", "full")
                p = adj.get(f"{label}|{q}") if in_family else None
                g = std_g_reading(r[ro]["G_std"], r[ro]["G"], r[ro]["net_share_moved"], r[ro]["rank_gain"],
                                  pair_floor, p, holm_applies=in_family)
                g["holm_p"] = p
                shifts = [fx.get("G_std_shift") for fx in (fxa, fxb) if fx.get("G_std_shift") is not None]
                if any(abs(x) > STD_MARGIN / 5 for x in shifts):
                    g["environment_dependent"] = True
                # shares, read only when both arms' goal effects pass the reference rule
                ea, eb = est[(a, s, ro)], est[(b, s, ro)]
                shares_ok = ea.get("G_reference_ok") and eb.get("G_reference_ok")
                fna, fnb = fxa.get("floor_nats"), fxb.get("floor_nats")
                g_min = min(abs(ea["G"]["estimate"]), abs(eb["G"]["estimate"]))
                floor_share = (math.sqrt(fna ** 2 + fnb ** 2) / g_min
                               if fna is not None and fnb is not None and g_min else None)
                sh = {}
                for k in ("C", "B") + tuple(f"share|{m}" for m in MENTIONS) + \
                        tuple(f"stage_by_meaning|{m}" for m in MENTIONS if m != "name_assert"):
                    if k in r[ro]:
                        fam = in_family and ro == "first" and k in ("C", "B")
                        if not shares_ok:
                            sh[k] = "not read (a goal effect fails the reference rule)"
                            continue
                        sh[k] = share_reading(r[ro][k]["ci"], floor_share, adj.get(f"{label}|{k}"), fam)
                        if not fam:
                            sh[k] += " (descriptive)"
                g["shares"] = sh
                rd[ro] = g
            if "pi" in r:
                gated = all(g_.get("read", {}).get(x) for g_ in (ga, gb) for x in ("first", "full"))
                ok = all(est[(x, s, "full")].get("G_reference_ok") for x in (a, b))
                fl_pi = [((out.get("floors", {}).get(x) or {}).get("full") or {}).get("floor_nats") for x in (a, b)]
                g_min = min(abs(est[(x, s, "full")]["G"]["estimate"]) for x in (a, b))
                fs = math.sqrt(sum(f_ ** 2 for f_ in fl_pi)) / g_min if None not in fl_pi and g_min else None
                if not gated:
                    rd["pi"] = "not interpretable (competence gate)"
                elif not ok:
                    rd["pi"] = "not read (reference rule)"
                else:
                    rd["pi"] = share_reading(r["pi"]["ci"], fs, adj.get(f"{label}|pi"), s == "later")
                    if s != "later":
                        rd["pi"] += " (descriptive)"
            readings[s] = rd
        c["readings"] = readings
        c["strata"] = _strip(c["strata"])
    out["comparisons"] = prim
    for a, b, label in SECONDARY_PAIRS:
        a, b = resolve(a), resolve(b)
        if a in arms and b in arms:
            out["secondary"][label] = {"from": a, "to": b, "strata": _strip({s: compare(a, b, s) for s in STRATA})}
    for a, b in FORMAT_PAIRS:
        if ("qwen" if a.startswith("qwen") else "llama") in fallback:
            out["format"][f"{a} -> {b}"] = "not read (the lineage's three-example plain arms are replaced)"
        elif a in arms and b in arms:
            out["format"][f"{a} -> {b}"] = _strip({s: compare(a, b, s) for s in STRATA})
    # stage-by-format interaction: the change of the format effect between successive post-trained stages
    for x, y in (("tulu-sft", "tulu-dpo"), ("tulu-dpo", "tulu-rl"), ("tulu-rl", "llama-instruct")):
        keys = [f"{x}_plain", f"{x}_native", f"{y}_plain", f"{y}_native"]
        if "llama" in fallback:
            out["interactions"][f"stage x format {x} -> {y}"] = "not read (three-example plain arms replaced)"
        elif all(k in arms for k in keys):
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
    for c in sorted({str(x["commit"]) for x in arms.values()}):
        if not same_code(None if c == "None" else c):
            problems.append(f"scores from commit {c}, whose scorer files differ from this analysis commit's")
    problems += token_identity(arms)
    return problems


def token_identity(arms: dict[str, dict]) -> list[str]:
    """The tokenizer gate across a lineage, on every scored sequence: for each item, condition and format, the
    token ids (their hash) are identical in every plain-format arm of the lineage with the same examples, and in
    their floor rescorings."""
    problems = []
    groups: dict = {}
    for a, x in arms.items():
        k = P.ARMS[a]
        if k["mode"] != "plain" or k["kind"] == "bridge":
            continue
        lin = "qwen" if k["ckpt"].startswith("qwen") else "llama"
        groups.setdefault((lin, k.get("fewshot")), []).append(a)
    for members in groups.values():
        ref: dict = {}
        for a in members:
            for i, r in arms[a]["fail"].items():
                for c, fm in r["scores"].items():
                    for f, cell in fm.items():
                        h = cell.get("ids_sha")
                        if h is None:
                            continue  # a reused lexsem cell: checked bit for bit instead
                        prev = ref.setdefault((i, c, f), (a, h))
                        if prev[1] != h:
                            problems.append(f"token ids differ: {i}|{c}|{f} in {prev[0]} and {a}")
    return problems[:20]


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
    res = analyse(arms, {a.removesuffix("_floor"): f for a, f in floors.items()}, B=args.B)
    res["environment_problems"] = problems
    res["analysis_commit"] = git_sha()
    res["arms_present"] = sorted(arms)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {args.out}: {len(arms)} arms, {len(floors)} floor arms", flush=True)


if __name__ == "__main__":
    main()
