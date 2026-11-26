"""Frozen analysis of the natural ALFWorld bridge (notes/bridge_design.md §§5-8).

Reads the scoring runs of wf/bridge.py once every shard is complete, checks the environment rules (one commit, the
freeze tag, one GPU type per run, a clean checkout), and writes every estimand, gate and category to metrics.json.
Intervals are game-clustered percentile bootstraps (2,000 resamples, one fixed set of resamples per item list);
the primary tests are game-level sign-flip permutation tests, Holm-adjusted over a fixed family of 18.
"""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from wf import constants as C
from wf.bridge import (ARMS, COMP_ARMS, GATED, STATED, TASKS, WORDINGS, history_bin, item_sets, stratum)
from wf.rq2 import SOURCES, analysis_provenance, complete_shards, load_items, primary_set
from wf.runs import _init_run
from wf.util import git_dirty, git_sha, is_mock, read_jsonl, require_filed, write_json, write_status

MODELS = ("qwen3-8b", "llama3.1-8b")
PRIMARY_FORMAT = {"qwen3-8b": "plain", "llama3.1-8b": "chat"}
READOUTS = ("cand", "first", "yn")
BINS = ("0", "1-3", "4-7", "8+")
# Rule 3 of notes/numerical_sensitivity.md: margins at least about 0.3-0.4 nats for Qwen-sized logits and 0.1 for
# the others; "numerically fragile" within about twice the numerical SD of a mean (about 0.05 Qwen, 0.015 others)
FLOOR_MARGIN = {"qwen3-8b": 0.4, "llama3.1-8b": 0.1}
NUM_SD = {"qwen3-8b": 0.05, "llama3.1-8b": 0.015}
DZ_MARGIN = 0.15
PC_MIN = 1.0
GATE_SHARE = 0.8
MIN_EXIST_SHARE = 0.2  # 300 of the 1,500 sample items; the same share on the 300-item subset (review 1, finding 1)
MASS_MIN = 0.5
N_BOOT = C.BOOTSTRAP_RESAMPLES
N_PERM_PRIMARY = 10000
ALPHA = 0.05


class Boot:
    """Game-clustered bootstrap with one fixed set of resamples for every statistic on one item list, so that
    statistics on the same items (and the cross-GPU comparison) share resamples. A mask restricts a statistic to
    some items while keeping the resampled games' weights."""

    def __init__(self, games: list[str], seed: int = C.RNG_SEED, B: int = N_BOOT):
        import numpy as np

        uniq = sorted(set(games))
        pos = {g: k for k, g in enumerate(uniq)}
        self.gi = np.array([pos[g] for g in games])
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(uniq), [1 / len(uniq)] * len(uniq), size=B).astype(float)
        self.Wi = self.W[:, self.gi]
        self.n_games = len(uniq)
        self.seed = seed

    def _prep(self, x, mask):
        import numpy as np

        x = np.asarray(x, dtype=float)
        m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
        return x, m, self.Wi * m[None, :]

    @staticmethod
    def _ci(b, level):
        import numpy as np

        q = (1 - level) / 2
        lo, hi = np.nanpercentile(b, [100 * q, 100 * (1 - q)])
        return float(lo), float(hi)

    def mean(self, x, mask=None, level: float = 0.95) -> dict:
        import numpy as np

        x, m, w = self._prep(x, mask)
        if not m.any():
            return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (w @ np.where(m, x, 0.0)) / w.sum(1)
        lo, hi = self._ci(b, level)
        return {"estimate": float(x[m].mean()), "ci_low": lo, "ci_high": hi, "n": int(m.sum())}

    def ratio(self, a, b, mask=None, level: float = 0.95) -> dict:
        import numpy as np

        a, m, w = self._prep(a, mask)
        b = np.asarray(b, float)
        if not m.any():
            return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (w @ np.where(m, a, 0.0)) / (w @ np.where(m, b, 0.0))
        lo, hi = self._ci(r, level)
        return {"estimate": float(a[m].sum() / b[m].sum()), "ci_low": lo, "ci_high": hi}

    def _dzb(self, x, m, w):
        import numpy as np

        xs = np.where(m, x, 0.0)
        sw = w.sum(1)
        mu = (w @ xs) / sw
        return mu / np.sqrt(np.maximum((w @ (xs * xs)) / sw - mu ** 2, 1e-12))

    def dz(self, x, mask=None, level: float = 0.95) -> dict:
        x, m, w = self._prep(x, mask)
        lo, hi = self._ci(self._dzb(x, m, w), level)
        return {"estimate": float(x[m].mean() / x[m].std()), "ci_low": lo, "ci_high": hi}

    def std_removal(self, r, ref, mask=None, level: float = 0.90) -> dict:
        """mean(r) / SD(ref): a removal in units of the item-level spread of the reference effect, same resamples.
        (Comparing standardized effects of a single arm and of a two-wording average would favour the average,
        whose noise is smaller.)"""
        import numpy as np

        r, m, w = self._prep(r, mask)
        ref = np.where(m, np.asarray(ref, float), 0.0)
        sw = w.sum(1)
        mu_ref = (w @ ref) / sw
        sd_ref = np.sqrt(np.maximum((w @ (ref * ref)) / sw - mu_ref ** 2, 1e-12))
        b = ((w @ np.where(m, r, 0.0)) / sw) / sd_ref
        lo, hi = self._ci(b, level)
        rr = np.asarray(ref)[m]
        return {"estimate": float(r[m].mean() / rr.std()), "ci_low": lo, "ci_high": hi}

    def perm_p(self, x, mask=None, n: int = N_PERM_PRIMARY) -> float:
        """Two-sided sign-flip test of the mean, all of a game's items flipping together."""
        import numpy as np

        x, m, _ = self._prep(x, mask)
        if not m.any():
            return 1.0
        tot = np.zeros(self.n_games)
        np.add.at(tot, self.gi[m], x[m])
        rng = np.random.default_rng(self.seed + 1)
        signs = rng.choice([-1.0, 1.0], size=(n, self.n_games))
        null = np.abs(signs @ tot)
        return float((1 + (null >= abs(tot.sum()) - 1e-9).sum()) / (1 + n))


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=lambda k: (ps[k], k))
    out, running, m = {}, 0.0, len(keys)
    for r, k in enumerate(keys):
        running = max(running, min(1.0, (m - r) * ps[k]))
        out[k] = running
    return out


def load_values(records: list[dict]) -> dict:
    """{(readout, task, arm): {item id: value}} for one model and format. Readouts: cand (L), first (L_first), yn,
    c1, c2, exist_check (log P(Yes) - log P(No)); medians of the two-way masses under mass_first and mass_yn."""
    v: dict = defaultdict(dict)
    for r in records:
        task, arm, ro = r["job"]
        key = (ro, task, arm)
        if r["transition_id"] in v[key]:
            raise ValueError(f"duplicate record {r['transition_id']} {key}")
        v[key][r["transition_id"]] = r["L"]
        if ro == "cand":
            v[("first", task, arm)][r["transition_id"]] = r["L_first"]
            v[("mass_first", task, arm)][r["transition_id"]] = r["first_mass"]
        elif ro == "yn":
            v[("mass_yn", task, arm)][r["transition_id"]] = r["mass"]
    return v


class Values:
    def __init__(self, v: dict, ids: list[str]):
        self.v, self.ids = v, ids

    def get(self, ro: str, task: str, arm: str):
        import numpy as np

        d = self.v.get((ro, task, arm), {})
        missing = [i for i in self.ids if i not in d]
        if missing:
            raise ValueError(f"missing {len(missing)} scores for {(ro, task, arm)}, e.g. {missing[:3]}")
        return np.array([d[i] for i in self.ids], float)

    def G(self, ro: str, arm: str):
        """The goal effect under an arm: aligned minus swapped, per item."""
        return self.get(ro, "aligned", arm) - self.get(ro, "swapped", arm)


def comprehension(V: Values) -> dict:
    """On the subset (aligned task line): per statement and question, the share of items whose answer favours the
    stated one. A tester or agent wording passes (review 1, finding 2) when, on each question, at least 80% of items
    favour its stated answer AND at least 80% of items answer it on the stated side of the positive control's answer
    to the same question on the same item (the positive control's stated answers are the opposite ones), so a model
    that answers the same whatever the statement says cannot pass."""
    out: dict[str, Any] = {}
    pos = {q: V.get(q, "aligned", "positive") for q in ("c1", "c2")}
    none = {q: V.get(q, "aligned", "none") for q in ("c1", "c2")}
    for arm in COMP_ARMS:
        row: dict[str, Any] = {}
        for q in ("c1", "c2"):
            L = V.get(q, "aligned", arm)
            row[q] = {"share_yes": float((L > 0).mean()), "mean_L": float(L.mean())}
            if arm in STATED:
                want = STATED[arm][q]
                row[q]["stated"] = want
                row[q]["share_favouring_stated"] = float(((L > 0) if want == "Yes" else (L < 0)).mean())
            if arm in GATED:
                toward = (L > pos[q]) if STATED[arm][q] == "Yes" else (L < pos[q])
                row[q]["share_beyond_positive_control"] = float(toward.mean())
                row[q]["mean_difference_from_no_statement"] = float((L - none[q]).mean())
        if arm in GATED:
            row["gate_pass"] = all(row[q]["share_favouring_stated"] >= GATE_SHARE
                                   and row[q]["share_beyond_positive_control"] >= GATE_SHARE for q in ("c1", "c2"))
        out[arm] = row
    out["n_items"] = len(V.ids)
    return out


def category(model: str, blk: dict, p_holm: float) -> tuple[str, bool]:
    """Pre-specified categories for a removal R (positive: the statement removes part of the effect), and whether
    the reading is numerically fragile (an interval end within twice the numerical SD of its boundary).

    removes most: R's 95% interval above 0, Holm p < 0.05, R at least the margin, the share of the reference effect
      with its 95% interval at or above 0.5 and the share of the no-statement effect at least 0.5;
    removes a part: interval above 0, Holm p < 0.05, R at least the margin (review 1, finding 3);
    increases the effect: interval below 0, Holm p < 0.05, R at most minus the margin;
    removes nothing: 90% interval inside (-margin, margin), and for Qwen the standardized removal's 90% interval
      inside (-0.15, 0.15) (a removal that is detectable but inside the margin is flagged, not a separate category);
    removes a small amount: interval above 0, Holm p < 0.05, below the margin, not shown equivalent;
    otherwise undetermined."""
    est, share, eq, delta = blk["removal"], blk["share_removed"], blk["removal_ci90"], blk["margin"]
    share_none = blk["share_of_no_statement_effect"]["estimate"]
    dzd = blk["standardized_removal_ci90"] if model == "qwen3-8b" else None
    sd2 = 2 * NUM_SD[model]
    sig_pos = est["ci_low"] > 0 and p_holm < ALPHA
    sig_neg = est["ci_high"] < 0 and p_holm < ALPHA
    if sig_pos and est["estimate"] >= delta and share["ci_low"] >= 0.5 and share_none >= 0.5:
        return "removes most", est["ci_low"] < sd2
    if sig_pos and est["estimate"] >= delta:
        return "removes a part", est["ci_low"] < sd2
    if sig_neg and est["estimate"] <= -delta:
        return "increases the effect", -est["ci_high"] < sd2
    if -delta < eq["ci_low"] and eq["ci_high"] < delta and (
            dzd is None or (-DZ_MARGIN < dzd["ci_low"] and dzd["ci_high"] < DZ_MARGIN)):
        return ("removes nothing beyond the margin (equivalent to zero within it)",
                min(eq["ci_low"] + delta, delta - eq["ci_high"]) < sd2)
    if sig_pos:
        return "removes a small amount (below the margin, not shown equivalent)", est["ci_low"] < sd2
    if sig_neg:
        return "increases the effect by a small amount (below the margin, not shown equivalent)", -est["ci_high"] < sd2
    return "undetermined", False


def removal_block(model: str, boot: Boot, G_ref, G_arm, G_none, mask=None) -> dict:
    """Removal R = G_ref - G_arm against the reference arm (the content-free sentence in the statement's position):
    its mean, its share of the reference arm's effect (review 1, finding 4) and of the no-statement effect, the
    equivalence interval (90%), the standardized removal and the permutation p (category set after Holm)."""
    import numpy as np

    R = np.asarray(G_ref) - np.asarray(G_arm)
    ref = boot.mean(G_ref, mask)
    return {"removal": boot.mean(R, mask), "removal_ci90": boot.mean(R, mask, level=0.90),
            "share_removed": boot.ratio(R, G_ref, mask), "reference_effect": ref,
            "share_of_no_statement_effect": boot.ratio(np.asarray(G_none) - np.asarray(G_arm), G_none, mask),
            "no_statement_effect": boot.mean(G_none, mask),
            "effect_under_statement": boot.mean(G_arm, mask),
            "margin": max(0.25 * abs(ref["estimate"]), FLOOR_MARGIN[model]),
            "dz_reference": boot.dz(G_ref, mask), "dz_under_statement": boot.dz(G_arm, mask),
            "standardized_removal_ci90": boot.std_removal(R, G_ref, mask), "perm_p": boot.perm_p(R, mask)}


def describe_arms(boot: Boot, V: Values, ro: str) -> dict:
    """Per arm: mean goal effect, standardized effect, share of items moved toward success, the probability-scale
    effect (first token and yes/no) and each task line's level shift against no statement."""
    import numpy as np

    sig = lambda x: 1 / (1 + np.exp(-x))
    out = {}
    for arm in ARMS:
        G = V.G(ro, arm)
        row = {"effect": boot.mean(G), "dz": boot.dz(G), "share_moved": float((G > 0).mean())}
        if ro in ("first", "yn"):
            row["prob_effect"] = boot.mean(sig(V.get(ro, "aligned", arm)) - sig(V.get(ro, "swapped", arm)))
        for task in TASKS:
            row[f"level_shift_{task}"] = boot.mean(V.get(ro, task, arm) - V.get(ro, task, "none"))
        out[arm] = row
    return out


def masses(V: Values) -> dict:
    import numpy as np

    out = {}
    for kind in ("mass_first", "mass_yn"):
        for arm in ARMS:
            for task in TASKS:
                d = V.v.get((kind, task, arm), {})
                vals = [d[i] for i in V.ids if i in d]
                if vals:
                    out[f"{kind}/{task}/{arm}"] = float(np.median(vals))
    return out


def names_object(t: dict) -> bool:
    """Whether the history after its first two lines (room description, task line) names the action's object."""
    import re

    rest = "\n".join((t.get("history_text") or "").splitlines()[2:])
    return re.search(rf"\b{re.escape(t.get('goal_object') or '')}\b", rest) is not None


def model_block(model: str, V: Values, boot: Boot, comp: dict, items: list[dict], tests: dict,
                a1prime: dict | None = None) -> dict:
    """Every estimand for one model in one format on one item list; `tests` collects the primary p-values."""
    import numpy as np

    out: dict[str, Any] = {"n_items": len(V.ids), "n_games": boot.n_games}
    passing = {s: [w for w in WORDINGS[s] if comp.get(w, {}).get("gate_pass")] for s in WORDINGS}
    out["passing_wordings"] = passing
    ec = {(task, arm): V.get("exist_check", task, arm) for task in TASKS for arm in ("none", "exist")}
    F = (ec[("aligned", "exist")] > 0) & (ec[("swapped", "exist")] > 0)
    changed = F & (ec[("swapped", "none")] < 0)  # the statement turned the swapped prompt's check to "present"
    named_hist = np.array([names_object(t) for t in items])
    min_exist = int(math.ceil(MIN_EXIST_SHARE * len(V.ids)))
    out["existence_check"] = {"n_favour_present_both": int(F.sum()), "gate_minimum": min_exist,
                              "n_changed_by_statement": int(changed.sum()),
                              **{f"share_yes/{task}/{arm}": float((ec[(task, arm)] > 0).mean())
                                 for task in TASKS for arm in ("none", "exist")}}
    out["masses_median"] = masses(V)
    low = sorted(k for k, v in out["masses_median"].items() if k.startswith("mass_first") and v < MASS_MIN)
    out["first_token_low_mass_arms"] = low  # design §8 gate 4 (review 1, finding 8)
    hb = np.array([history_bin(int(t.get("history_len") or 0)) for t in items])
    for ro in READOUTS:
        r: dict[str, Any] = {"arms": describe_arms(boot, V, ro)}
        G0, Gp = V.G(ro, "none"), V.G(ro, "placebo")
        pc = (V.get(ro, "aligned", "positive") - V.get(ro, "aligned", "placebo")
              + V.get(ro, "swapped", "positive") - V.get(ro, "swapped", "placebo")) / 2
        pc_b = boot.mean(pc)
        pc_none = (V.get(ro, "aligned", "positive") - V.get(ro, "aligned", "none")
                   + V.get(ro, "swapped", "positive") - V.get(ro, "swapped", "none")) / 2
        r["positive_control"] = {**pc_b, "vs_no_statement": boot.mean(pc_none),
                                 "pass": pc_b["estimate"] >= PC_MIN and pc_b["ci_low"] > 0}
        # reported, not gated: a statement that should increase the task effect if provenance is read
        r["expert_statement_increase"] = boot.mean(V.G(ro, "expert") - Gp)
        r["placebo_removal"] = boot.mean(G0 - Gp)
        for s in WORDINGS:
            ws = passing[s] or list(WORDINGS[s])  # the p-value is always computed, so the Holm family keeps 18 tests
            Gs = sum(V.G(ro, w) for w in ws) / len(ws)
            blk = removal_block(model, boot, Gp, Gs, G0)
            blk["against_no_statement"] = boot.mean(G0 - Gs)
            blk["wordings_used"] = ws
            blk["wording_spread"] = boot.mean(V.G(ro, WORDINGS[s][0]) - V.G(ro, WORDINGS[s][1]))
            blk["gate"] = ("comprehension gate failed for both wordings" if not passing[s]
                           else "positive control failed on this readout" if not r["positive_control"]["pass"]
                           else "pass")
            blk["by_history_bin"] = {b: boot.mean(Gp - Gs, hb == b) for b in BINS}
            if ro == "first" and low:
                blk["low_first_token_mass"] = low
            r[s] = blk
            tests[(model, ro, s)] = blk["perm_p"]
        Ge, Gn, Gq = V.G(ro, "exist"), V.G(ro, "named"), V.G(ro, "exist_placebo")
        mask = F if F.any() else None
        ex = removal_block(model, boot, Gq, Ge, G0, mask=mask)
        ex["named_on_same_items"] = boot.mean(Gq - Gn, mask)
        ex["named_share_on_same_items"] = boot.ratio(Gq - Gn, Gq, mask)
        ex["named_all_items"] = boot.mean(Gq - Gn)
        ex["unnamed_all_items"] = boot.mean(Gq - Ge)
        ex["content_free_sentence_removal"] = boot.mean(G0 - Gq, mask)
        ex["changed_by_statement_items"] = boot.mean(Gq - Ge, changed if changed.any() else None)
        ex["history_names_object"] = boot.mean(Gq - Ge, F & named_hist)
        ex["history_does_not_name_object"] = boot.mean(Gq - Ge, F & ~named_hist)
        ex["gate"] = "pass" if F.sum() >= min_exist else f"fewer than {min_exist} items pass the existence check"
        ex["by_history_bin"] = {b: boot.mean(Gq - Ge, F & (hb == b)) for b in BINS}
        if a1prime:
            idx = np.array([a1prime.get(t["id"], np.nan) for t in items])
            take = F & ~np.isnan(idx)
            ex["a1prime_take_items"] = {"n": int(take.sum()),
                                        "index_mean": float(np.nanmean(np.where(take, idx, np.nan))) if take.any() else None,
                                        "unnamed_effect_remaining": boot.mean(Ge, take if take.any() else None),
                                        "removal": boot.mean(Gq - Ge, take if take.any() else None)}
        if ro == "first" and low:
            ex["low_first_token_mass"] = low
        r["existence"] = ex
        tests[(model, ro, "existence")] = ex["perm_p"] if F.any() else 1.0
        r["no_statement_by_history_bin"] = {b: boot.mean(G0, hb == b) for b in BINS}
        out[ro] = r
    return out


def apply_holm(blocks: dict, tests: dict) -> dict:
    """Holm over the given primary tests, then each removal's category (a failed gate overrides it)."""
    adj = holm({"|".join(k): p for k, p in tests.items()})
    for (model, ro, s) in tests:
        blk = blocks[model][ro][s]
        p = adj["|".join((model, ro, s))]
        blk["holm_p"] = p
        blk["category"], blk["numerically_fragile"] = category(model, blk, p)
        if s in WORDINGS and blk["gate"] != "pass":
            blk["category"] = f"not readable: {blk['gate']}"
        if s == "existence" and blk["gate"] != "pass":
            blk["category"] = f"not assessable: {blk['gate']}"
    return adj




def reading(model_blocks: dict) -> dict:
    """One plain sentence per model, readout and question, from the categories (design §10). Each says what the
    effect does when the statement is added, not why."""
    lines: dict[str, Any] = {}
    for model, mb in model_blocks.items():
        for ro in READOUTS:
            r = mb[ro]
            for s, who in (("tester", "a random tester"), ("agent", "an agent that saw only the history")):
                sh = r[s]["share_removed"]
                lines[f"{model}/{ro}/{s}"] = (f"Saying {who} chose the action, against a content-free sentence in the "
                                              f"same place: {r[s]['category']} (share of the effect removed "
                                              f"{sh['estimate']:.2f} [{sh['ci_low']:.2f}, {sh['ci_high']:.2f}]).")
            sh = r["existence"]["share_removed"]
            lines[f"{model}/{ro}/existence"] = (f"Existence settled without naming the object, against a content-free "
                                                f"sentence in the task line: {r['existence']['category']} (share "
                                                f"removed {sh['estimate']:.2f} [{sh['ci_low']:.2f}, {sh['ci_high']:.2f}]).")
        cats = {s: {mb[ro][s]["category"] for ro in READOUTS} for s in ("tester", "agent", "existence")}
        lines[f"{model}/readouts_agree"] = {s: len(v) == 1 for s, v in cats.items()}
    return lines


def rescore_comparison(model: str, main_v: dict, re_v: dict, items: list[dict]) -> dict:
    """Rule 4: every decision statistic on the 300-item subset, main GPU type against the rescore GPU type, with the
    same resamples; each run's own comprehension gate; Holm within this model's nine tests on the subset. A
    statistic is environment-dependent when its category or gate changes or it moves by more than a fifth of its
    margin (review 1, finding 1)."""
    ids = [t["id"] for t in items]
    boot = Boot([t["game_id"] for t in items])
    out, comp = {}, {}
    for name, v in (("main", main_v), ("rescore", re_v)):
        tests: dict = {}
        comp[name] = comprehension(Values(v, ids))
        blk = model_block(model, Values(v, ids), boot, comp[name], items, tests)
        apply_holm({model: blk}, tests)
        out[name] = blk
    cmp: dict[str, Any] = {}
    for w in GATED:
        cmp[f"comprehension/{w}"] = {"main": comp["main"][w]["gate_pass"], "rescore": comp["rescore"][w]["gate_pass"],
                                     "environment_dependent": comp["main"][w]["gate_pass"] != comp["rescore"][w]["gate_pass"]}
    a, b = out["main"]["existence_check"], out["rescore"]["existence_check"]
    cmp["existence_check"] = {"main": a["n_favour_present_both"], "rescore": b["n_favour_present_both"],
                              "environment_dependent": (a["n_favour_present_both"] >= a["gate_minimum"])
                              != (b["n_favour_present_both"] >= b["gate_minimum"])}
    for ro in READOUTS:
        for s in ("tester", "agent", "existence"):
            a, b = out["main"][ro][s], out["rescore"][ro][s]
            shift = b["removal"]["estimate"] - a["removal"]["estimate"]
            cmp[f"{ro}/{s}"] = {"main": a["removal"]["estimate"], "rescore": b["removal"]["estimate"],
                               "shift": shift, "margin": a["margin"], "category_main": a["category"],
                               "category_rescore": b["category"],
                               "environment_dependent": a["category"] != b["category"] or abs(shift) > a["margin"] / 5}
        margin = out["main"][ro]["tester"]["margin"]
        a, b = out["main"][ro]["positive_control"], out["rescore"][ro]["positive_control"]
        cmp[f"{ro}/positive_control"] = {"main": a["estimate"], "rescore": b["estimate"],
                                         "shift": b["estimate"] - a["estimate"], "pass_main": a["pass"],
                                         "pass_rescore": b["pass"], "margin": margin,
                                         "environment_dependent": a["pass"] != b["pass"]
                                         or abs(b["estimate"] - a["estimate"]) > margin / 5}
        e0a = out["main"][ro]["arms"]["none"]["effect"]["estimate"]
        e0b = out["rescore"][ro]["arms"]["none"]["effect"]["estimate"]
        cmp[f"{ro}/no_statement_effect"] = {"main": e0a, "rescore": e0b, "shift": e0b - e0a, "margin": margin,
                                            "environment_dependent": abs(e0b - e0a) > margin / 5}
    return {"comparison": cmp, "any_environment_dependent": any(x.get("environment_dependent") for x in cmp.values())}


def _shard_gpus(folders: list[Path]) -> set[str]:
    from wf.provenance import run_record

    out: set[str] = set()
    for p in folders:
        g = run_record(p)["gpu"]
        out |= set(g) if isinstance(g, list) else {str(g)}
    return out


STACK = {"transformers": "5.23.0", "torch": "2.14.1"}


def check_runs(cfg: dict) -> tuple[dict, list[str]]:
    """Shard folders per (kind, model), and every reason the analysis must not run."""
    problems, folders, commits = [], {}, set()
    for kind in ("score_runs", "rescore_runs"):
        for model, run in cfg[kind].items():
            fs, pr = complete_shards(run)
            problems += [f"{run}: {x}" for x in pr]
            folders[(kind, model)] = fs
            commits |= {(p / "git_sha").read_text().strip() for p in fs if (p / "git_sha").exists()}
            gpus = _shard_gpus(fs)
            want = cfg["main_gpu"] if kind == "score_runs" else cfg["rescore_gpu"]
            if not is_mock() and (not gpus or not all(want.lower() in g.lower() for g in gpus)):
                problems.append(f"{run}: GPUs {sorted(gpus)}, expected {want} only")
            for p in fs:
                st = json.loads((p / "stack.json").read_text()) if (p / "stack.json").exists() else {}
                if not is_mock() and (st.get("transformers") != STACK["transformers"]
                                      or not str(st.get("torch", "")).startswith(STACK["torch"])):
                    problems.append(f"{p}: software stack {st or 'not recorded'}, expected {STACK}")
    if cfg["main_gpu"].lower() == cfg["rescore_gpu"].lower():
        problems.append("the rescore must run on another GPU type than the main run")
    if len(commits) > 1:
        problems.append(f"scoring runs at different commits: {sorted(commits)}")
    if cfg.get("freeze_tag") and not is_mock():
        tag = subprocess.run(["git", "rev-parse", f"{cfg['freeze_tag']}^{{commit}}"], capture_output=True,
                             text=True).stdout.strip()
        if git_sha() != tag:
            problems.append(f"the analysis checkout is not at {cfg['freeze_tag']} ({tag or 'tag not found'})")
        if commits and commits != {tag}:
            problems.append(f"scoring commits {sorted(commits)} are not the freeze tag's {tag}")
    return folders, problems


def check_records(recs: list[dict], tag: str, hashes: dict) -> list[str]:
    """Every record is the expected model's and carries the hash of the item the analysis rebuilt (review 1, 13)."""
    bad_model = sum(r.get("model") != tag for r in recs)
    bad_hash = sum(hashes.get(r["transition_id"]) not in (None, r.get("item_hash")) for r in recs)
    unknown = sum(r["transition_id"] not in hashes for r in recs)
    return [f"{tag}: {n} records {what}" for n, what in ((bad_model, "of another model"),
                                                         (bad_hash, "whose item hash differs"),
                                                         (unknown, "for items outside the sample")) if n]


def load_a1prime(cfg: dict) -> dict | None:
    """Per-item A1' index (location plus instance number, given existence) from scripts/a1prime_check.py."""
    from wf.util import REPO_ROOT

    path = REPO_ROOT / cfg["a1prime_items"] if cfg.get("a1prime_items") else None
    if not path or not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["id"]] = r["location_given_existence"] + r["instance_given_existence"]
    return out


def run_bridge_analysis(cfg: dict) -> int:
    """Config: score_runs and rescore_runs ({model tag: run id}), main_gpu, rescore_gpu, freeze_tag, a1prime_items."""
    from wf.runs import _item_hash

    run_id = cfg["run_id"]
    require_filed(run_id)
    if not is_mock() and git_dirty():
        write_status(run_id, "refused: the analysis checkout has uncommitted changes", ok=False)
        return 1
    folders, problems = check_runs(cfg)
    d = _init_run(run_id, {**cfg, "resume": False}, inputs=())
    if problems:
        write_status(run_id, "refused:\n" + "\n".join(problems), ok=False)
        return 1
    sets = item_sets(primary_set(load_items(tuple(cfg.get("sources", SOURCES)))))
    items, sub = sets["sample"], sets["subset"]
    ids, sub_ids = [t["id"] for t in items], [t["id"] for t in sub]
    hashes = {t["id"]: _item_hash(t) for t in items}
    a1 = load_a1prime(cfg)
    report: dict[str, Any] = {
        "items": {"n_sample": len(items), "n_subset": len(sub), "n_games": len({t["game_id"] for t in items}),
                  "strata": {"|".join(k): v for k, v in sorted(Counter(stratum(t) for t in items).items())}},
        "a1prime_index_loaded": a1 is not None}
    boot = Boot([t["game_id"] for t in items])
    tests: dict = {}
    blocks, comps, raw, rec_problems = {}, {}, {}, []
    for model in cfg["score_runs"]:
        recs = [x for p in folders[("score_runs", model)] for x in read_jsonl(p / "scoring.jsonl")]
        rec_problems += check_records(recs, model, hashes)
        byfmt = defaultdict(list)
        for x in recs:
            byfmt[x["format"]].append(x)
        raw[model] = {f: load_values(r) for f, r in byfmt.items()}
        prim = PRIMARY_FORMAT[model]
        comps[model] = comprehension(Values(raw[model][prim], sub_ids))
        blocks[model] = model_block(model, Values(raw[model][prim], ids), boot, comps[model], items, tests, a1)
    if rec_problems:
        write_status(run_id, "refused:\n" + "\n".join(rec_problems), ok=False)
        return 1
    if len(tests) != 18:
        write_status(run_id, f"refused: the primary family has {len(tests)} tests, not 18", ok=False)
        return 1
    report["holm_adjusted_p"] = apply_holm(blocks, tests)
    report["comprehension"] = comps
    report["models"] = blocks
    report["reading"] = reading(blocks)
    if "block" in raw.get("qwen3-8b", {}):  # RQ2's chat format on the subset: a descriptive link to RQ2
        sboot = Boot([t["game_id"] for t in sub])
        link = {}
        for fmt in ("plain", "block"):
            t_: dict = {}
            vfmt = Values(raw["qwen3-8b"][fmt], sub_ids)
            cfmt = comprehension(vfmt)  # each format's own comprehension (review 1, finding 15)
            blk = model_block("qwen3-8b", vfmt, sboot, cfmt, sub, t_)
            apply_holm({"qwen3-8b": blk}, t_)
            link[fmt] = {"comprehension_gate": {w: cfmt[w]["gate_pass"] for w in GATED},
                         **{ro: {"no_statement": blk[ro]["arms"]["none"]["effect"],
                                 "dz_no_statement": blk[ro]["arms"]["none"]["dz"],
                                 "positive_control": blk[ro]["positive_control"],
                                 **{s: {"share_removed": blk[ro][s]["share_removed"], "removal": blk[ro][s]["removal"],
                                        "category_on_subset": blk[ro][s]["category"]}
                                    for s in ("tester", "agent", "existence")}} for ro in READOUTS}}
        report["format_link_qwen_subset"] = link
    report["cross_gpu"] = {}
    for model in cfg["rescore_runs"]:
        recs = [x for p in folders[("rescore_runs", model)] for x in read_jsonl(p / "scoring.jsonl")]
        bad = check_records(recs, model, hashes)
        if bad:
            write_status(run_id, "refused:\n" + "\n".join(bad), ok=False)
            return 1
        rv = load_values([x for x in recs if x["format"] == PRIMARY_FORMAT[model]])
        report["cross_gpu"][model] = rescore_comparison(model, raw[model][PRIMARY_FORMAT[model]], rv, sub)
    others = {cfg[k][m]: fs for (k, m), fs in folders.items()}
    report["provenance"] = analysis_provenance(None, [], tuple(cfg.get("sources", SOURCES)), others=others)
    write_json(d / "metrics.json", report)
    write_status(run_id, json.dumps(report["reading"], indent=1), ok=False, reported="bridge analysis")
    return 0
