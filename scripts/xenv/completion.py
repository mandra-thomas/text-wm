#!/usr/bin/env python3
"""Exploratory, post hoc: naming the object against completing the task, in ALFWorld (notes/textcraft_vs_alfworld.md,
section 10). Reads saved scores only.

Every RQ2 action acts on the goal object, and no RQ2 history acts on it, so every move, heat, cool and clean item fails
because the object was never picked up. The actions fall into groups by their place in the task's plan:
  complete     move the goal object to a receptacle of the task's target type: the last step of a placement, pick-two
               or heat/cool/clean task (it would complete a placement task outright)
  hcc_adv      heat, cool or clean the goal object when the task asks for that transformation: an intermediate step
  take         take the goal object from a receptacle: the first step of every task
  move_other   move the goal object to a receptacle that is not of the target type: off the plan
  hcc_nonadv   heat, cool or clean when the task asks for another transformation (or none): off the plan
Completing is compared with each other group, raw and matched. Matching reweights the comparison group to the
completing group's distribution over strata of failure reason, step (0 or later) and, where both groups have it, task
kind; strata the comparison group lacks are dropped from both sides, and the share of completing items kept is
reported. Clusters are games; both groups come from the same resamples, so the difference's interval is paired.

Arms: rq2_score aligned/swapped and the 2x2 mention/fourth arms; rq1_score aligned/swapped and the extra-mention
aligned_plus/swapped_plus arms; rq1x_score object-free aligned_desc/swapped_desc arms; rq1v_score yes/no.

Adds the key "completion_vs_intermediate" to the metrics JSON given by --metrics.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import data as D  # noqa: E402
from compare import Boot, sig, summ  # noqa: E402


def groups(f: dict) -> np.ndarray:
    at, tk, mt = f["action_type"], f["task_kind"], f["move_to_target"].astype(bool)
    hcc = np.isin(at, ["heat", "cool", "clean"])
    return np.where(mt, "complete", np.where(at == "take", "take", np.where(hcc & (at == tk), "hcc_adv",
                    np.where(hcc, "hcc_nonadv", "move_other"))))


def measures(f: dict) -> dict:
    out = {
        "dL": f["al.L"] - f["sw.L"],
        "dL_first": f["al.L_first"] - f["sw.L_first"],
        "dL_yes_no": f["yal.L"] - f["ysw.L"],
        "dp_pp": 100 * (sig(f["al.L"]) - sig(f["sw.L"])),
        "dp_yes_no_pp": 100 * (sig(f["yal.L"]) - sig(f["ysw.L"])),
        "yes_no_aligned_minus_neutral": f["yal.L"] - f["yneu.L"],
        "first_aligned_minus_neutral": f["al.L_first"] - f["neu.L_first"],
        # rq1_score's extra-mention design: goal role with both objects named, and the swapped-prompt mention part
        "goal_role_both_named": f["r1alp.L"] - f["r1swp.L"],
        "mention_sentence_on_swapped": f["r1swp.L"] - f["r1sw.L"],
        # object-free (described) task, and how much of the named effect on the same run it keeps
        "dL_described": f["dal.L"] - f["dsw.L"],
        # rq2_score's 2x2 (only move-to-target and matching heat/cool/clean items have it)
        "two_by_two_goal_directed": (f["al.L"] - f["men.L"]) - (f["sw.L"] - f["fou.L"]),
        "two_by_two_mention": f["men.L"] - f["fou.L"],
    }
    return out


def strata_key(f: dict, with_task: bool) -> np.ndarray:
    step = np.where(f["history_len"] == 0, "s0", "later")
    base = np.char.add(np.char.add(f["reason"].astype(str), "|"), step)
    return np.char.add(np.char.add(base, "|"), f["task_kind"].astype(str)) if with_task else base


def compare_groups(x: np.ndarray, g: np.ndarray, key: np.ndarray, boot: Boot, a: str, b: str) -> dict:
    ok = ~np.isnan(x)
    ka = set(key[(g == a) & ok])
    shared = sorted(k for k in ka if ((g == b) & ok & (key == k)).any())
    n_a = int(((g == a) & ok).sum())
    kept_a = (g == a) & ok & np.isin(key, shared)
    kept_b = (g == b) & ok & np.isin(key, shared)
    if not kept_a.any():
        return {"n_a": n_a, "n_b": int(((g == b) & ok).sum()), "shared_strata": 0}
    w = {k: float(np.mean(key[kept_a] == k)) for k in shared}

    def raw(ix):
        ia, ib = ix[((g == a) & ok)[ix]], ix[((g == b) & ok)[ix]]
        return np.mean(x[ia]) - np.mean(x[ib]) if len(ia) and len(ib) else np.nan

    def side(ix, mask):
        sub = ix[mask[ix]]
        vals = []
        for k in shared:
            s = sub[key[sub] == k]
            if not len(s):
                return np.nan
            vals.append(np.mean(x[s]))
        return float(np.dot([w[k] for k in shared], vals))

    def matched(ix):
        return side(ix, kept_a) - side(ix, kept_b)

    pr, br = boot.run(raw)
    pa, ba = boot.run(lambda ix: side(ix, kept_a))
    pb, bb = boot.run(lambda ix: side(ix, kept_b))
    return {"n_a": n_a, "n_b": int(((g == b) & ok).sum()), "shared_strata": len(shared),
            "share_of_a_kept": float(kept_a.sum() / n_a), "n_a_kept": int(kept_a.sum()), "n_b_kept": int(kept_b.sum()),
            "mean_a": float(np.mean(x[(g == a) & ok])), "mean_b": float(np.mean(x[(g == b) & ok])),
            "raw_difference": summ(pr, br),
            "matched_a": summ(pa, ba), "matched_b": summ(pb, bb), "matched_difference": summ(pa - pb, ba - bb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    A = D.load_alfworld(a.archive)
    g_ = lambda run: sorted(glob.glob(os.path.join(a.archive, run, "shard_*", "scoring.jsonl")))
    rq2s = D._scores(g_("text-wm-fix11/runs/rq2_score"))
    rq1s = D._scores(g_("text-wm-fix12/runs/rq1_score"))
    rq1x = D._scores(g_("text-wm-fix13/runs/rq1x_score"))
    res: dict = {"note": "exploratory, post hoc; section 10 of notes/textcraft_vs_alfworld.md"}
    for m in D.MODELS:
        f = dict(A[m]["fail"])
        ids = f["id"]
        for prefix, (sc, cond) in {"men": (rq2s, "mention"), "fou": (rq2s, "fourth"), "r1al": (rq1s, "aligned"),
                                   "r1sw": (rq1s, "swapped"), "r1alp": (rq1s, "aligned_plus"),
                                   "r1swp": (rq1s, "swapped_plus"), "dal": (rq1x, "aligned_desc"),
                                   "dsw": (rq1x, "swapped_desc")}.items():
            f[f"{prefix}.L"] = np.array([sc.get((m, i, cond), {}).get("L", np.nan) for i in ids], dtype=float)
        g = groups(f)
        boot = Boot(f["cluster"])
        meas = measures(f)
        r: dict = {"counts": {k: int((g == k).sum()) for k in np.unique(g)},
                   "group_means": {k: {name: (float(np.nanmean(x[g == k])) if (~np.isnan(x[g == k])).any() else None)
                                       for name, x in meas.items()} for k in np.unique(g)}}
        # named-effect share kept by the described task, per group, on the rq1_score named effect (same scoring family)
        named = f["r1al.L"] - f["r1sw.L"]
        r["described_share_kept"] = {}
        for k in np.unique(g):
            mk = (g == k) & ~np.isnan(meas["dL_described"])
            pk, bk = boot.run(lambda ix, mk=mk: (np.mean(meas["dL_described"][ix[mk[ix]]]) / np.mean(named[ix[mk[ix]]]))
                              if mk[ix].any() else np.nan)
            r["described_share_kept"][k] = summ(pk, bk)
        r["comparisons"] = {}
        for other, with_task in (("hcc_adv", True), ("take", False), ("move_other", False), ("hcc_nonadv", False)):
            key = strata_key(f, with_task)
            r["comparisons"][f"complete_vs_{other}"] = {
                "strata": "reason x step" + (" x task kind" if with_task else ""),
                **{name: compare_groups(x, g, key, boot, "complete", other) for name, x in meas.items()}}
        # within heat/cool/clean tasks only: the final placement against the transformation step of the same task kind
        tk_mask = np.isin(f["task_kind"], ["heat", "cool", "clean"])
        g2 = np.where(tk_mask, g, "other")
        r["comparisons"]["complete_vs_hcc_adv_transform_tasks_only"] = {
            "strata": "reason x step x task kind, heat/cool/clean tasks only",
            **{name: compare_groups(x, g2, strata_key(f, True), boot, "complete", "hcc_adv") for name, x in meas.items()}}
        res[m] = r
    with open(a.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["completion_vs_intermediate"] = res
    with open(a.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("added completion_vs_intermediate to", a.metrics)


if __name__ == "__main__":
    main()
