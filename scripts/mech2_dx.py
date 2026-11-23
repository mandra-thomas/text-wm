#!/usr/bin/env python
"""Exploratory, post hoc deep-dive of the mechanism battery (notes/mechanism2_dx.md, whose list of analyses was
committed before this script was run).

Reads the saved scores of `mech2_score` (8 shards) and rebuilds the battery's items with the frozen item functions of
`wf/mech2.py` (sample, step-0 items, populations, the receptacle-shown split, the 87 swapped-target items). Everything
else (contrasts, bootstrap, ratios, splits) is this script's own code and shares nothing with the frozen analysis.
It writes `notes/mechanism2_dx_metrics.json` and a plain-text table dump; it never writes into `runs/`.

    RUNS_DIR=data_local/rr python scripts/mech2_dx.py \
        --scores data_local/runs/mech2_score --metrics data_local/runs/mech2_analysis/metrics.json \
        --out notes/mechanism2_dx_metrics.json --tables data_local/mech2_dx_tables.txt

`RUNS_DIR` must hold the three `rq12_transitions_s*` folders (the item files); the scores folder holds the shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 20261122
MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot", "mistral-7b-v0.3")
FORMATS = ("cand", "yn")
REF = ((1.0, "aligned"), (-1.0, "swapped"))
# the key contrasts of the deep-dive and the population each reads ("sample" or "take" = P7's 503 items)
KEY = {
    "P3": (((1.0, "mention_ignore"), (-1.0, "swapped")), "sample"),
    "P4": (((1.0, "described"), (-1.0, "swapped")), "sample"),
    "P7": (((1.0, "negated"), (-1.0, "swapped")), "take"),
    "P8": (((1.0, "negated"), (-1.0, "affirmed")), "take"),
    "M": (((0.5, "negated"), (0.5, "affirmed"), (-1.0, "swapped")), "take"),
}
EXTRA = {
    "M_half_affirmed_minus_swapped": (((1.0, "affirmed"), (-1.0, "swapped")), "take"),
    "M_half_swapped_minus_negated": (((1.0, "swapped"), (-1.0, "negated")), "take"),
    "S2_placebo": (((1.0, "placebo_negated"), (-1.0, "placebo_affirmed")), "take"),
    "S3_P8_net": (((1.0, "negated"), (-1.0, "affirmed"), (-1.0, "placebo_negated"), (1.0, "placebo_affirmed")), "take"),
}
STOP = {"a", "an", "the", "some", "in", "on", "from", "to", "with", "and", "it", "your", "task", "is", "two", "then",
        "of", "at", "by", "for", "under"}


# ---------------- data ----------------

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scores(folder: Path) -> tuple[dict, dict]:
    """{(model, item, arm, fmt): (L, L_first)} and the files' hashes."""
    out, hashes = {}, {}
    for p in sorted(folder.glob("shard_*/scoring.jsonl")):
        hashes[str(p.relative_to(folder.parent))] = sha256(p)
        with open(p) as f:
            for line in f:
                x = json.loads(line)
                key = (x["model"], x["transition_id"], x["arm"], x["format"])
                if key in out:
                    raise SystemExit(f"duplicate score {key}")
                out[key] = (x["L"], x.get("L_first"))
    return out, hashes


def build_items():
    from wf import mech2, rq1
    from wf.rq2 import SOURCES, load_items
    primary = rq1.rq1_sets(load_items(SOURCES))["primary"]
    sets = mech2.battery_items(primary)
    pops = mech2.populations(sets["union"], sets["sample"], sets["step0"])
    take_ids = {t["id"] for t in pops["sample_take"]}
    step0_take = [t for t in pops["step0"] if mech2.p7_eligible(t)]
    info = {}
    for t in sets["union"]:
        p = mech2.parse_action(t["action"])
        info[t["id"]] = {
            "game": t["game_id"], "task_type": mech2.task_type(t["task_line"]), "step0": t.get("history_len", 0) == 0,
            "reason": t.get("determination_reason"), "verb": t["action"].split()[0],
            "shown": (mech2.negated_status(t)[1] if mech2.is_take(t) else None),
            "swapped_target_at_rec": (mech2.swapped_target_at_rec(t) if t["id"] in take_ids else None),
            "action": t["action"], "task_line": t["task_line"], "swapped_task_line": t["swapped_task_line"],
            "goal_object": t["goal_object"], "history_text": t["history_text"], "rec": p[2] if p else None,
        }
    return {"sample": [t["id"] for t in pops["sample"]], "take": [t["id"] for t in pops["sample_take"]],
            "p5": [t["id"] for t in pops["p5"]], "step0": [t["id"] for t in pops["step0"]],
            "step0_take": [t["id"] for t in step0_take]}, info


# ---------------- statistics (own code) ----------------

class Stats:
    def __init__(self, scores: dict, info: dict, B: int):
        self.s, self.info, self.B = scores, info, B

    def values(self, model, ids, terms, fmt, field=0):
        """Per item: the contrast and the aligned effect (items lacking a score are dropped)."""
        d, r, g, keep = [], [], [], []
        for i in ids:
            vs = [self.s.get((model, i, arm, fmt)) for _, arm in terms]
            rs = [self.s.get((model, i, arm, fmt)) for _, arm in REF]
            if any(v is None for v in vs + rs):
                continue
            d.append(sum(c * v[field] for (c, _), v in zip(terms, vs)))
            r.append(sum(c * v[0] for (c, _), v in zip(REF, rs)))
            g.append(self.info[i]["game"])
            keep.append(i)
        return np.array(d), np.array(r), g, keep

    def _weights(self, games):
        uniq = sorted(set(games))
        idx = {gm: k for k, gm in enumerate(uniq)}
        gi = np.array([idx[gm] for gm in games])
        rng = np.random.default_rng(SEED)
        W = rng.multinomial(len(uniq), [1.0 / len(uniq)] * len(uniq), size=self.B)
        return uniq, gi, W

    def est(self, model, ids, terms, fmt, field=0):
        d, r, g, keep = self.values(model, ids, terms, fmt, field)
        if len(d) == 0:
            return {"n_items": 0}
        uniq, gi, W = self._weights(g)
        G = len(uniq)
        sd, sr, n = (np.bincount(gi, weights=x, minlength=G) for x in (d, r, np.ones_like(d)))
        wd, wr, wn = W @ sd, W @ sr, W @ n
        mean_b = wd / wn
        ok = wr > 0
        ratio_b = np.where(ok, wd / np.where(ok, wr, 1), np.nan)
        out = {"n_items": int(len(d)), "n_games": G, "estimate": float(d.mean()),
               "ci": [float(np.percentile(mean_b, 2.5)), float(np.percentile(mean_b, 97.5))],
               "aligned_effect": float(r.mean())}
        if r.mean() > 0:
            rb = ratio_b[ok]
            out["r"] = float(d.mean() / r.mean())
            out["r_ci"] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
            out["r_dropped"] = int((~ok).sum())
        return out

    def group_diff(self, model, a_ids, b_ids, terms, fmt):
        da, _, ga, _ = self.values(model, a_ids, terms, fmt)
        db, _, gb, _ = self.values(model, b_ids, terms, fmt)
        if not len(da) or not len(db):
            return None
        games = ga + gb
        uniq, gi, W = self._weights(games)
        G = len(uniq)
        grp = np.array([1] * len(da) + [0] * len(db))
        d = np.concatenate([da, db])
        sa = np.bincount(gi, weights=d * grp, minlength=G)
        na = np.bincount(gi, weights=grp.astype(float), minlength=G)
        sb = np.bincount(gi, weights=d * (1 - grp), minlength=G)
        nb = np.bincount(gi, weights=(1 - grp).astype(float), minlength=G)
        wa, wna, wb, wnb = W @ sa, W @ na, W @ sb, W @ nb
        ok = (wna > 0) & (wnb > 0)
        diff = wa[ok] / wna[ok] - wb[ok] / wnb[ok]
        return {"n_a": int(len(da)), "n_b": int(len(db)), "estimate": float(da.mean() - db.mean()),
                "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "n_dropped": int((~ok).sum())}


def tails(d: np.ndarray) -> dict:
    n = len(d)
    k = max(1, math.ceil(0.1 * n))
    sign = 1.0 if d.mean() >= 0 else -1.0
    dd = sign * d  # in the direction of the mean
    pos = np.sort(dd[dd > 0])[::-1]
    lo, hi = np.percentile(d, [10, 90])
    trimmed = d[(d >= lo) & (d <= hi)]
    return {"n": n, "mean": float(d.mean()), "median": float(np.median(d)), "trimmed_mean_10": float(trimmed.mean()),
            "share_above_0": float((d > 0).mean()), "share_above_plus1": float((d > 1).mean()),
            "share_below_minus1": float((d < -1).mean()), "direction_of_mean": "positive" if sign > 0 else "negative",
            "top10pct_share_of_contributions_in_direction_of_mean": float(pos[:k].sum() / pos.sum()) if pos.sum() > 0 else None}


def per_game(d: np.ndarray, r: np.ndarray, games: list[str]) -> dict:
    uniq = sorted(set(games))
    gi = np.array([uniq.index(g) for g in games])
    sd = np.bincount(gi, weights=d, minlength=len(uniq))
    sr = np.bincount(gi, weights=r, minlength=len(uniq))
    n = np.bincount(gi, minlength=len(uniq)).astype(float)
    tot_d, tot_r, tot_n = sd.sum(), sr.sum(), n.sum()
    loo_mean = (tot_d - sd) / (tot_n - n)
    loo_r = (tot_d - sd) / (tot_r - sr)
    big = n >= 3
    sign = 1.0 if tot_d >= 0 else -1.0
    contrib = np.sort(sign * sd)[::-1]
    return {"n_games": len(uniq), "loo_estimate_range": [float(loo_mean.min()), float(loo_mean.max())],
            "loo_r_range": [float(loo_r.min()), float(loo_r.max())],
            "n_games_3plus_items": int(big.sum()),
            "share_games_3plus_mean_above_0": float(((sd / n)[big] > 0).mean()),
            "largest_game_share_of_total": float(contrib[0] / (sign * tot_d)),
            "top5_games_share_of_total": float(contrib[:5].sum() / (sign * tot_d))}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


# ---------------- analyses ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()

    scores, hashes = load_scores(a.scores)
    pops, info = build_items()
    frozen = json.loads(a.metrics.read_text())
    st = Stats(scores, info, a.B)
    pop_ids = {"sample": pops["sample"], "take": pops["take"]}
    res: dict = {"meta": {"scores_sha256": hashes, "metrics_sha256": sha256(a.metrics), "bootstrap_resamples": a.B,
                          "seed": SEED, "n": {k: len(v) for k, v in pops.items()}}}
    lines: list[str] = []
    P = lines.append

    def fmt_e(e):
        if not e or not e.get("n_items"):
            return "n/a"
        s = f"{e['estimate']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"
        if "r" in e:
            s += f" r {e['r']:+.2f} [{e['r_ci'][0]:+.2f}, {e['r_ci'][1]:+.2f}]"
        return s + f" n={e['n_items']}"

    # 0. independent recompute against metrics.json
    from wf import mech2
    rec = {}
    P("== 0. recompute (max |difference| from metrics.json)")
    for m in MODELS:
        diffs = []
        for name, spec in mech2.PRIMARY.items():
            ids = {"sample": pops["sample"], "p5": pops["p5"], "sample_take": pops["take"], "step0": pops["step0"]}[spec["population"]]
            for fmt, fam in (("cand", "primary"), ("yn", "primary_yes_no")):
                mine = st.values(m, ids, spec["terms"], fmt)[0]
                theirs = frozen[m][fam]["contrasts"][name]["estimate"]
                if len(mine):
                    diffs.append((abs(mine.mean() - theirs), name, fmt))
        mr = frozen[m]["mixture_readout"]
        for name, terms in (("M_mention_residual", KEY["M"][0]),
                            ("M_half_affirmed_minus_swapped", EXTRA["M_half_affirmed_minus_swapped"][0]),
                            ("M_half_swapped_minus_negated", EXTRA["M_half_swapped_minus_negated"][0])):
            for fmt in FORMATS:
                mine = st.values(m, pops["take"], terms, fmt)[0]
                diffs.append((abs(mine.mean() - mr[name][fmt]["estimate"]), name, fmt))
        if m != "mistral-7b-v0.3":
            sn = frozen[m]["secondary_named"]
            for name, key in (("S2_negation_placebo", "S2_placebo"), ("S3_P8_net_of_placebo", "S3_P8_net")):
                for fmt in FORMATS:
                    mine = st.values(m, pops["take"], EXTRA[key][0], fmt)[0]
                    diffs.append((abs(mine.mean() - sn[name]["by_format"][fmt]["estimate"]), name, fmt))
        worst = max(diffs)
        rec[m] = {"n_compared": len(diffs), "max_abs_difference": worst[0], "where": f"{worst[1]} {worst[2]}"}
        P(f"{m}: {len(diffs)} estimates, max |diff| {worst[0]:.2e} at {worst[1]} {worst[2]}")
    res["recompute"] = rec

    # the key contrasts on their populations, overall, with r
    res["overall"] = {}
    P("\n== key contrasts overall")
    for m in MODELS:
        for c, (terms, pop) in {**KEY, **EXTRA}.items():
            for fmt in FORMATS:
                e = st.est(m, pop_ids[pop], terms, fmt)
                res["overall"].setdefault(m, {}).setdefault(c, {})[fmt] = e
                if e.get("n_items"):
                    P(f"{m:22s} {c:32s} {fmt:4s} {fmt_e(e)}")

    # 1. per game
    res["per_game"] = {}
    P("\n== 1. per game")
    for m in MODELS:
        for c, (terms, pop) in KEY.items():
            for fmt in FORMATS:
                d, r, g, _ = st.values(m, pop_ids[pop], terms, fmt)
                pg = per_game(d, r, g)
                res["per_game"].setdefault(m, {}).setdefault(c, {})[fmt] = pg
                P(f"{m:22s} {c:3s} {fmt:4s} LOO est [{pg['loo_estimate_range'][0]:+.2f}, {pg['loo_estimate_range'][1]:+.2f}] "
                  f"LOO r [{pg['loo_r_range'][0]:+.2f}, {pg['loo_r_range'][1]:+.2f}] games>0 {pg['share_games_3plus_mean_above_0']:.2f} "
                  f"(of {pg['n_games_3plus_items']}) largest game {pg['largest_game_share_of_total']:.3f} top5 {pg['top5_games_share_of_total']:.3f}")

    # 2. per task type
    res["per_task_type"] = {}
    P("\n== 2. per task type")
    types = ("pick", "pick_two", "clean", "heat", "cool", "examine")
    for m in MODELS:
        for c, (terms, pop) in KEY.items():
            for fmt in FORMATS:
                pooled = res["overall"][m][c][fmt]["estimate"]
                row = {}
                for ty in types:
                    ids = [i for i in pop_ids[pop] if info[i]["task_type"] == ty]
                    e = st.est(m, ids, terms, fmt)
                    e["sign_differs_from_pooled"] = bool(e.get("n_items") and np.sign(e["estimate"]) != np.sign(pooled))
                    row[ty] = e
                res["per_task_type"].setdefault(m, {}).setdefault(c, {})[fmt] = row
                P(f"{m:22s} {c:3s} {fmt:4s} " + " | ".join(f"{ty} {row[ty]['estimate']:+.2f} r {row[ty].get('r', float('nan')):+.2f} n{row[ty]['n_items']}"
                                                           + (" *SIGN*" if row[ty]["sign_differs_from_pooled"] else "") for ty in types))

    # 3. heavy tails
    res["tails"] = {}
    P("\n== 3. tails")
    for m in MODELS:
        for c, (terms, pop) in {**KEY, "aligned_effect": (REF, "sample")}.items():
            for fmt in FORMATS:
                d = st.values(m, pop_ids[pop], terms, fmt)[0]
                tl = tails(d)
                res["tails"].setdefault(m, {}).setdefault(c, {})[fmt] = tl
                P(f"{m:22s} {c:14s} {fmt:4s} mean {tl['mean']:+.2f} median {tl['median']:+.2f} trim {tl['trimmed_mean_10']:+.2f} "
                  f">0 {tl['share_above_0']:.3f} >+1 {tl['share_above_plus1']:.3f} <-1 {tl['share_below_minus1']:.3f} "
                  f"top10% {tl['top10pct_share_of_contributions_in_direction_of_mean']:.3f}")

    # 4. step 0 against later steps
    res["by_step"] = {}
    P("\n== 4. step 0 vs later")
    for m in MODELS:
        for c, (terms, pop) in {**KEY, "aligned_effect": (REF, "sample")}.items():
            base = pop_ids[pop]
            s0 = [i for i in base if info[i]["step0"]]
            later = [i for i in base if not info[i]["step0"]]
            all0 = pops["step0"] if pop == "sample" else pops["step0_take"]
            for fmt in FORMATS:
                row = {"sample_step0": st.est(m, s0, terms, fmt), "sample_later": st.est(m, later, terms, fmt),
                       "all_step0": st.est(m, all0, terms, fmt)}
                d_all = st.values(m, base, terms, fmt)[0]
                d_s0 = st.values(m, s0, terms, fmt)[0]
                row["step0_item_share"] = len(d_s0) / len(d_all)
                row["step0_share_of_total"] = float(d_s0.sum() / d_all.sum()) if d_all.sum() != 0 else None
                res["by_step"].setdefault(m, {}).setdefault(c, {})[fmt] = row
                P(f"{m:22s} {c:14s} {fmt:4s} s0 {fmt_e(row['sample_step0'])} | later {fmt_e(row['sample_later'])} | all514 "
                  f"{fmt_e(row['all_step0'])} | s0 items {row['step0_item_share']:.3f} share of total {row['step0_share_of_total']:.3f}")

    # 5. agreement between formats
    res["format_agreement"] = {}
    P("\n== 5. format agreement")
    for m in MODELS:
        for c, (terms, pop) in {**KEY, "aligned_effect": (REF, "sample")}.items():
            dc, _, g, kc = st.values(m, pop_ids[pop], terms, "cand")
            dy, _, _, ky = st.values(m, pop_ids[pop], terms, "yn")
            assert kc == ky
            rho = spearmanr(dc, dy).statistic
            same = float((np.sign(dc) == np.sign(dy)).mean())
            uniq = sorted(set(g))
            gi = np.array([uniq.index(x) for x in g])
            n = np.bincount(gi).astype(float)
            gc, gy = np.bincount(gi, weights=dc) / n, np.bincount(gi, weights=dy) / n
            pr = float(np.corrcoef(gc, gy)[0, 1])
            res["format_agreement"].setdefault(m, {})[c] = {"spearman_items": float(rho), "same_sign_share": same,
                                                            "pearson_game_means": pr, "n_items": len(dc), "n_games": len(uniq)}
            P(f"{m:22s} {c:14s} spearman {rho:+.2f} same sign {same:.3f} game-mean pearson {pr:+.2f}")

    # 6. 293 not shown against 210 shown without the object
    res["receptacle_shown"] = {}
    P("\n== 6. receptacle not shown (293) vs shown without it (210)")
    ns = [i for i in pops["take"] if info[i]["shown"] == "contents not shown"]
    sh = [i for i in pops["take"] if info[i]["shown"] == "shown, without it"]
    for m in MODELS:
        for c, (terms, _) in {**{k: KEY[k] for k in ("P7", "P8", "M")}, **EXTRA}.items():
            for fmt in FORMATS:
                row = {"not_shown": st.est(m, ns, terms, fmt), "shown_without": st.est(m, sh, terms, fmt),
                       "not_shown_minus_shown": st.group_diff(m, ns, sh, terms, fmt)}
                if not row["not_shown"].get("n_items"):
                    continue
                res["receptacle_shown"].setdefault(m, {}).setdefault(c, {})[fmt] = row
                gd = row["not_shown_minus_shown"]
                P(f"{m:22s} {c:32s} {fmt:4s} not shown {fmt_e(row['not_shown'])} | shown {fmt_e(row['shown_without'])} | "
                  f"diff {gd['estimate']:+.2f} [{gd['ci'][0]:+.2f}, {gd['ci'][1]:+.2f}]")
        within = [i for i in pops["take"] if info[i]["reason"] == "contents_or_open_state"]
        for c in ("P8", "M"):
            for fmt in FORMATS:
                a_ids = [i for i in within if info[i]["shown"] == "contents not shown"]
                b_ids = [i for i in within if info[i]["shown"] == "shown, without it"]
                row = {"not_shown": st.est(m, a_ids, KEY[c][0], fmt), "shown_without": st.est(m, b_ids, KEY[c][0], fmt),
                       "not_shown_minus_shown": st.group_diff(m, a_ids, b_ids, KEY[c][0], fmt)}
                res["receptacle_shown"][m][f"{c}_within_contents_or_open_state"] = {
                    **res["receptacle_shown"][m].get(f"{c}_within_contents_or_open_state", {}), fmt: row}
                gd = row["not_shown_minus_shown"]
                P(f"{m:22s} {c} within contents/open {fmt:4s} not shown {fmt_e(row['not_shown'])} | shown {fmt_e(row['shown_without'])} | "
                  f"diff {gd['estimate']:+.2f} [{gd['ci'][0]:+.2f}, {gd['ci'][1]:+.2f}]")

    # 7. the 87 swapped-target items, for the placebo
    res["swapped_target_at_rec"] = {}
    P("\n== 7. the 87 swapped-target items vs the other 416")
    on = [i for i in pops["take"] if info[i]["swapped_target_at_rec"]]
    off = [i for i in pops["take"] if not info[i]["swapped_target_at_rec"]]
    for m in MODELS[:2]:
        for c, terms in (("S2_placebo", EXTRA["S2_placebo"][0]), ("P8", KEY["P8"][0]), ("S3_P8_net", EXTRA["S3_P8_net"][0])):
            for fmt in FORMATS:
                row = {"the_87": st.est(m, on, terms, fmt), "other_416": st.est(m, off, terms, fmt),
                       "difference": st.group_diff(m, on, off, terms, fmt)}
                res["swapped_target_at_rec"].setdefault(m, {}).setdefault(c, {})[fmt] = row
                gd = row["difference"]
                P(f"{m:22s} {c:10s} {fmt:4s} 87: {fmt_e(row['the_87'])} | 416: {fmt_e(row['other_416'])} | diff {gd['estimate']:+.2f} [{gd['ci'][0]:+.2f}, {gd['ci'][1]:+.2f}]")

    # 8. P3 and word overlap
    feats = {}
    for i in pops["sample"]:
        x = info[i]
        act = words(x["action"])
        sw = words(x["swapped_task_line"])
        real = words(x["task_line"])
        rec_type = re.sub(r"\s+\d+$", "", x["rec"]) if x["rec"] else None
        n_sw = len(act & sw)
        n_real = len(act & real)
        feats[i] = {"swapped_names_rec_type": bool(rec_type and rec_type in sw),
                    "swapped_has_verb": x["verb"] in sw,
                    "swapped_shared_words": "0" if n_sw == 0 else "1" if n_sw == 1 else "2+",
                    "real_shared_words": "1" if n_real <= 1 else "2" if n_real == 2 else "3+",
                    "object_in_history": bool(re.search(rf"\b{re.escape(x['goal_object'])}\b", "\n".join(
                        ln for ln in x["history_text"].lower().split("\n") if not ln.startswith("your task is to:")))),
                    "verb": x["verb"]}
    res["overlap_feature_counts"] = {k: dict(Counter(str(f[k]) for f in feats.values())) for k in next(iter(feats.values()))}
    res["overlap_feature_counts"]["verbs_where_swapped_has_verb"] = dict(Counter(f["verb"] for f in feats.values() if f["swapped_has_verb"]))
    res["overlap_feature_counts"]["task_types_where_swapped_has_verb"] = dict(Counter(info[i]["task_type"] for i, f in feats.items() if f["swapped_has_verb"]))
    res["overlap_feature_counts"]["verb_by_task_type_clean_heat_cool"] = dict(Counter(
        f"{f['verb']}/{info[i]['task_type']}/{f['swapped_has_verb']}" for i, f in feats.items() if f["verb"] in ("clean", "heat", "cool")))
    P("\n== 8. P3 by overlap; counts " + json.dumps(res["overlap_feature_counts"]))
    res["overlap"] = {}
    for m in MODELS:
        for feat in ("swapped_names_rec_type", "swapped_has_verb", "swapped_shared_words", "object_in_history", "verb", "real_shared_words"):
            for c, terms in (("P3", KEY["P3"][0]), ("P4", KEY["P4"][0]), ("aligned_effect", REF)):
                for fmt in FORMATS:
                    levels = sorted({str(f[feat]) for f in feats.values()})
                    row = {lv: st.est(m, [i for i in pops["sample"] if str(feats[i][feat]) == lv], terms, fmt) for lv in levels}
                    if len(levels) == 2:
                        a_ids = [i for i in pops["sample"] if str(feats[i][feat]) == levels[1]]
                        b_ids = [i for i in pops["sample"] if str(feats[i][feat]) == levels[0]]
                        row["difference_" + levels[1] + "_minus_" + levels[0]] = st.group_diff(m, a_ids, b_ids, terms, fmt)
                    res["overlap"].setdefault(m, {}).setdefault(feat, {}).setdefault(c, {})[fmt] = row
                    P(f"{m:22s} {feat:22s} {c:14s} {fmt:4s} " + " | ".join(
                        f"{lv}: {row[lv]['estimate']:+.2f} r {row[lv].get('r', float('nan')):+.2f} n{row[lv]['n_items']}" for lv in levels)
                      + "".join(f" | diff {v['estimate']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
                                for k, v in row.items() if k.startswith("difference") and v))

    # 9. practical size in the yes/no format
    res["yes_share"] = {}
    P("\n== 9. share answered Yes (yes/no L > 0), sample")
    arms = ("aligned", "swapped", "neutral", "mention_ignore", "mention_exists", "described", "desire_keep_sw",
            "desire_keep", "desire_check", "desire_check_can", "done", "cancelled", "negated", "affirmed",
            "placebo_negated", "placebo_affirmed")
    for m in MODELS:
        row = {}
        for arm in arms:
            ids = pops["take"] if arm in ("negated", "affirmed", "placebo_negated", "placebo_affirmed") else pops["sample"]
            v = [(scores.get((m, i, arm, "yn")), scores.get((m, i, "swapped", "yn"))) for i in ids]
            v = [(x[0], y[0]) for x, y in v if x is not None and y is not None]
            if not v:
                continue
            row[arm] = {"population": "take" if ids is pops["take"] else "sample", "n": len(v),
                        "share_yes": float(np.mean([x > 0 for x, _ in v])),
                        "share_swapped_yes_same_items": float(np.mean([y > 0 for _, y in v])),
                        "share_no_to_yes": float(np.mean([(y <= 0) and (x > 0) for x, y in v])),
                        "share_yes_to_no": float(np.mean([(y > 0) and (x <= 0) for x, y in v]))}
        res["yes_share"][m] = row
        P(m + ": " + " | ".join(f"{k} {v['share_yes']:.3f} (sw {v['share_swapped_yes_same_items']:.3f}, No->Yes {v['share_no_to_yes']:.3f})"
                                for k, v in row.items()))

    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    a.tables.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
