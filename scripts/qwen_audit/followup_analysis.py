"""Exploratory follow-up analysis (notes/qwen_effect_audit_pipeline.md, section 7). CPU only; run on ICE, where the
collection and the stored runs are (read-only).

A. Qwen3-8B on all 4,666 primary items: the stored scores (with the block) against the rescoring without it.
B. Qwen3 sizes on the 300-item subsample, both formats from one job each (8B: stored and rescored values).
   B_plain: the same items in plain text (no chat template) against the block, with the first-token mass.
C. The decision simulation's step 0: yes/no acceptance, scores and the verification rule's chosen failures, per arm
   and format; the with-block values checked against the stored decision-simulation scores. C_plain: the same in
   plain text (" Yes" against " No"), with the first-token mass.

Intervals: 95% percentile bootstrap over games, 2,000 resamples, seed 20260924. Writes followup_summary.json.

    python scripts/qwen_audit/followup_analysis.py --dir runs/qwen_audit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

SEED, B = 20260924, 2000
TAG8 = "qwen3-8b-fewshot"


def boot(rows, games, stat, n=B, seed=SEED):
    """Percentile interval of stat(rows) resampling games; rows and games are parallel lists."""
    by = defaultdict(list)
    for r, g in zip(rows, games):
        by[g].append(r)
    keys = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        v = stat(pick)
        if v == v:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def describe(pairs, games):
    """pairs: (effect with block, effect without block) per item."""
    w = lambda rs: [a for a, _ in rs]
    o = lambda rs: [b for _, b in rs]
    stats = {
        "mean_with": lambda rs: mean(w(rs)), "mean_without": lambda rs: mean(o(rs)),
        "paired_difference": lambda rs: mean([a - b for a, b in rs]),
        "ratio_without_over_with": lambda rs: mean(o(rs)) / mean(w(rs)),
        "standardized_with": lambda rs: mean(w(rs)) / sd(w(rs)), "standardized_without": lambda rs: mean(o(rs)) / sd(o(rs)),
        "share_positive_with": lambda rs: mean([float(a > 0) for a in w(rs)]),
        "share_positive_without": lambda rs: mean([float(b > 0) for b in o(rs)]),
        "sd_with": lambda rs: sd(w(rs)), "sd_without": lambda rs: sd(o(rs)),
    }
    out = {"n_items": len(pairs), "n_games": len(set(games))}
    for k, f in stats.items():
        out[k] = f(pairs)
        if not k.startswith("sd_"):
            out[k + "_ci"] = boot(pairs, games, f)
    return out


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.open()] if p.exists() else []


def part_a(d: Path, items: dict) -> dict:
    res = {}
    for f in sorted(d.glob("fu_8b_shard*.jsonl")):
        if f.name.endswith("_gpu.txt"):
            continue
        for r in load_jsonl(f):
            res[r["id"]] = r
    # Check: the first items of each job, rescored with the block, against the stored values.
    diffs = [abs(r["scores"][f"block/{c}"]["L"] - items[i]["stored"][TAG8][c]["L"])
             for i, r in res.items() if "block/aligned" in r["scores"] for c in ("aligned", "swapped")]
    pairs, games, h0, h0g, first = [], [], [], [], []
    bins = defaultdict(lambda: ([], []))
    for i, it in items.items():
        if i not in res:
            continue
        s, n = it["stored"][TAG8], res[i]["scores"]
        e_with = s["aligned"]["L"] - s["swapped"]["L"]
        e_wo = n["noblock/aligned"]["L"] - n["noblock/swapped"]["L"]
        g, h = it["transition"]["game_id"], it["transition"]["history_len"]
        pairs.append((e_with, e_wo)); games.append(g)
        first.append((s["aligned"]["L_first"] - s["swapped"]["L_first"], n["noblock/aligned"]["L_first"] - n["noblock/swapped"]["L_first"]))
        if h == 0:
            h0.append((e_with, e_wo)); h0g.append(g)
        b = "0" if h == 0 else "1-3" if h <= 3 else "4-7" if h <= 7 else "8+"
        bins[b][0].append((e_with, e_wo)); bins[b][1].append(g)
    return {
        "n_rescored": len(res), "check_block_rescore": {"n_scores": len(diffs), "n_exact": sum(x == 0 for x in diffs),
                                                         "max_abs_diff": max(diffs) if diffs else None},
        "all": describe(pairs, games), "step0": describe(h0, h0g),
        "first_token": {"mean_with": mean([a for a, _ in first]), "mean_without": mean([b for _, b in first]),
                        "mean_with_ci": boot(first, games, lambda rs: mean([a for a, _ in rs])),
                        "mean_without_ci": boot(first, games, lambda rs: mean([b for _, b in rs]))},
        "by_history_length": {b: {"n": len(v[0]), "mean_with": mean([a for a, _ in v[0]]),
                                  "mean_without": mean([x for _, x in v[0]])} for b, v in sorted(bins.items())},
        "mean_L": {"with_aligned": mean([items[i]["stored"][TAG8]["aligned"]["L"] for i in res]),
                   "with_swapped": mean([items[i]["stored"][TAG8]["swapped"]["L"] for i in res]),
                   "without_aligned": mean([r["scores"]["noblock/aligned"]["L"] for r in res.values()]),
                   "without_swapped": mean([r["scores"]["noblock/swapped"]["L"] for r in res.values()])},
    }


def part_b(d: Path, items_all: dict, sizes_items: list[str]) -> dict:
    out = {}
    a_res = {}
    for f in sorted(d.glob("fu_8b_shard*.jsonl")):
        for r in load_jsonl(f):
            a_res[r["id"]] = r
    src = {"qwen3-4b": d / "fu_sizes_qwen3-4b.jsonl", "qwen3-14b": d / "fu_sizes_qwen3-14b.jsonl",
           "qwen3-32b": d / "fu_sizes_qwen3-32b.jsonl"}
    for tag, p in src.items():
        res = {r["id"]: r for r in load_jsonl(p)}
        ids = [i for i in sizes_items if i in res]
        if not ids:
            continue
        pairs = [(res[i]["scores"]["block/aligned"]["L"] - res[i]["scores"]["block/swapped"]["L"],
                  res[i]["scores"]["noblock/aligned"]["L"] - res[i]["scores"]["noblock/swapped"]["L"]) for i in ids]
        out[tag] = describe(pairs, [items_all[i]["transition"]["game_id"] for i in ids])
    ids = [i for i in sizes_items if i in a_res]
    if ids:
        s = lambda i, c: items_all[i]["stored"][TAG8][c]["L"]
        n = lambda i, c: a_res[i]["scores"][f"noblock/{c}"]["L"]
        pairs = [(s(i, "aligned") - s(i, "swapped"), n(i, "aligned") - n(i, "swapped")) for i in ids]
        out["qwen3-8b"] = describe(pairs, [items_all[i]["transition"]["game_id"] for i in ids])
    return out


def part_b_plain(d: Path, items_all: dict, sizes_items: list[str]) -> dict:
    """Plain text against the chat template with the block, per size on the 300 items. The block values come from
    the first round's same-job rescoring (4B, 14B) or the stored scores (8B); the plain values from the plain round
    (another job, so a few tenths of a nat of GPU noise per score are possible). Also the first-token mass."""
    out = {}
    for tag in ("qwen3-4b", "qwen3-8b", "qwen3-14b"):
        plain = {r["id"]: r for r in load_jsonl(d / f"fu_sizes_{tag}_plain.jsonl")}
        if tag == "qwen3-8b":
            block = {i: (items_all[i]["stored"][TAG8]["aligned"]["L"] - items_all[i]["stored"][TAG8]["swapped"]["L"])
                     for i in sizes_items}
        else:
            block = {r["id"]: r["scores"]["block/aligned"]["L"] - r["scores"]["block/swapped"]["L"]
                     for r in load_jsonl(d / f"fu_sizes_{tag}.jsonl")}
        ids = [i for i in sizes_items if i in plain and i in block]
        if not ids:
            continue
        pairs = [(block[i], plain[i]["scores"]["plain/aligned"]["L"] - plain[i]["scores"]["plain/swapped"]["L"])
                 for i in ids]
        r = describe(pairs, [items_all[i]["transition"]["game_id"] for i in ids])
        masses = [plain[i]["scores"][f"plain/{c}"]["first_mass"] for i in ids for c in ("aligned", "swapped")]
        r["plain_first_mass_mean"], r["plain_first_mass_median"] = mean(masses), st.median(masses)
        r["plain_mean_L_aligned"] = mean([plain[i]["scores"]["plain/aligned"]["L"] for i in ids])
        r["plain_mean_L_swapped"] = mean([plain[i]["scores"]["plain/swapped"]["L"] for i in ids])
        out[tag] = r
    return out


def part_c(d: Path, src: Path, stored_dsim: Path, fname: str = "fu_dsim_step0.jsonl",
           formats: tuple[str, ...] = ("block", "noblock")) -> dict:
    from wf import decisionsim as D

    recs = {(r["game_id"], r["state_idx"]): r for r in load_jsonl(d / fname)}
    if not recs:
        return {}
    transitions = [json.loads(x) for k in range(3) for x in (src / f"rq12_transitions_s{k}" / "transitions.jsonl").open()]
    states = [s for s in D.build_states(transitions) if s["state_idx"] == 0 and (s["game_id"], 0) in recs]
    stored = {}
    for x in map(json.loads, stored_dsim.open()):
        if x.get("state_idx") == 0:
            stored[(x["transition_id"], x["condition"])] = x["L"]
    chk = [abs(D.quantize(recs[(s["game_id"], 0)]["scores"][t["id"]][f"block/{a}"]) - D.quantize(stored[(t["id"], a)]))
           for s in states for t in s["candidates"] for a in D.DEFAULT_ARMS
           if f"block/{a}" in recs[(s["game_id"], 0)]["scores"].get(t["id"], {}) and (t["id"], a) in stored]
    if "block" not in formats:
        chk = []
    out = {"n_states": len(states), "check_block_vs_stored": {"n": len(chk), "n_exact_at_1_64": sum(x == 0 for x in chk),
                                                              "mean_abs_diff": mean(chk), "max_abs_diff": max(chk) if chk else None},
           "proposal_fails_share": mean([float(not D.succeeded(D._proposal(s))) for s in states]), "arms": {}}
    if "plain" in formats:
        m = [v for r in recs.values() for per in r.get("first_mass", {}).values() for v in per.values()]
        out["plain_first_mass_mean"], out["plain_first_mass_median"] = mean(m), st.median(m)
    for fmt in formats:
        for arm in D.DEFAULT_ARMS:
            acc, fail, work_L, goalfail_L, games = [], [], [], [], []
            for s in states:
                sc = recs[(s["game_id"], 0)]["scores"]
                L = {t["id"]: D.quantize(sc[t["id"]][f"{fmt}/{arm}"]) for t in s["candidates"]
                     if f"{fmt}/{arm}" in sc.get(t["id"], {})}
                if len(L) != len(s["candidates"]):
                    continue  # the swapped arm has no swap object at this state
                p = D._proposal(s)
                acc.append(float(D.accepts(L, p)))
                fail.append(float(not D.succeeded(D.rule_choice("rule2_verification", s, L))))
                work_L += [L[t["id"]] for t in s["candidates"] if D.succeeded(t)]
                goalfail_L += [L[t["id"]] for t in s["candidates"] if not D.succeeded(t) and t.get("targets_goal_object")]
                games.append(s["game_id"])
            out["arms"][f"{fmt}/{arm}"] = {
                "n_states": len(acc), "proposal_accept_rate": mean(acc), "rule2_chosen_failure_rate": mean(fail),
                "rule2_chosen_failure_ci": boot(fail, games, mean), "mean_L_working": mean(work_L),
                "mean_L_goal_object_failures": mean(goalfail_L)}
    # Paired aligned-minus-blind difference in chosen failures, per format, over states scored in both arms.
    for fmt in formats:
        diffs, games = [], []
        for s in states:
            sc = recs[(s["game_id"], 0)]["scores"]
            Ls = {a: {t["id"]: D.quantize(sc[t["id"]][f"{fmt}/{a}"]) for t in s["candidates"]} for a in ("yn_aligned", "yn_blind")}
            f = {a: float(not D.succeeded(D.rule_choice("rule2_verification", s, Ls[a]))) for a in Ls}
            diffs.append(f["yn_aligned"] - f["yn_blind"]); games.append(s["game_id"])
        out[f"{fmt}_aligned_minus_blind_failure"] = {"mean": mean(diffs), "ci": boot(diffs, games, mean)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/qwen_audit")
    ap.add_argument("--src", default="/storage/ice1/2/1/gmat3/text-wm-fix11/runs")
    ap.add_argument("--stored-dsim", default="/storage/ice1/2/1/gmat3/text-wm-fix30/runs/decisionsim_score_qwen3-8b/"
                                             "shard_00_of_01/scoring.jsonl")
    args = ap.parse_args()
    d = Path(args.dir)
    items_all = {r["id"]: r for r in load_jsonl(d / "items_all.jsonl")}
    sizes_items = [r["id"] for r in load_jsonl(d / "items_sizes.jsonl")]
    out = {"A_qwen3_8b_all_items": part_a(d, items_all), "B_sizes_300": part_b(d, items_all, sizes_items),
           "C_decisionsim_step0": part_c(d, Path(args.src), Path(args.stored_dsim)),
           "B_sizes_300_plain": part_b_plain(d, items_all, sizes_items),
           "C_decisionsim_step0_plain": part_c(d, Path(args.src), Path(args.stored_dsim), "fu_dsim_step0_plain.jsonl",
                                               ("plain",))}
    out["gpus"] = {g.stem: g.read_text().strip() for g in d.glob("fu_*_gpu.txt")}
    (d / "followup_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
