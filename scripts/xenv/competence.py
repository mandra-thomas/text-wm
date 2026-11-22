#!/usr/bin/env python3
"""Exploratory, post hoc: does the goal effect track competence or baseline confidence across models?
(notes/textcraft_vs_alfworld.md, section 9). Reads saved scores only.

(a) Qwen3-8B and Llama-3.1-8B in ALFWorld and TextCraft: competence (neutral-task AUC, mean L on failures and
    successes) next to every effect scale, including the item share toward Yes and answers flipped No -> Yes.
(b) ALFWorld only, across the wave-A family models (families_score runs, ICE text-wm-fix23) and the Qwen3 size series
    (size_score_qwen3-*, archive text-wm-fix18): the same per-model numbers, and Spearman rank correlations across
    models between each effect and each competence or baseline measure. With about a dozen models that differ in
    many ways at once, the correlations are descriptive; the permutation p treats models as exchangeable, which they
    are not, and the intervals on per-model numbers reflect only the sampling of games.

Adds the key "competence_vs_effect" to the metrics JSON given by --metrics (written by compare.py).

Usage:
  python scripts/xenv/competence.py --archive <text-wm-runs-archive> --tc-items <items.jsonl> \
      --tc-scores <shards> --families-dir <dir with <model>.jsonl family scores> \
      --metrics notes/textcraft_vs_alfworld_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import data as D  # noqa: E402
from compare import Boot, auc, est, mean_fn, sig  # noqa: E402

FAMILY = ("gemma2-9b-it", "mistral-7b-v0.3", "llama3.2-3b", "llama3.2-1b")
SIZES = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b")
ARMS = {"al": "aligned", "sw": "swapped", "neu": "absent", "yal": "yn_aligned", "ysw": "yn_swapped", "yneu": "yn_absent"}


def table_from(rows: dict, ids, clusters) -> dict:
    t = {"cluster": np.asarray(clusters)}
    for p, cond in ARMS.items():
        for f in ("L", "L_first"):
            t[f"{p}.{f}"] = np.array([rows.get((i, cond), {}).get(f, np.nan) for i in ids], dtype=float)
    return t


def read_rows(paths) -> dict:
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["transition_id"], r["condition"])] = r
    return out


def profile(fail: dict, succ: dict, boot: Boot | None = None) -> dict:
    """Competence and every effect scale for one model in one environment (failures, successes)."""
    al, sw = fail["al.L"], fail["sw.L"]
    yal, ysw, yneu = fail["yal.L"], fail["ysw.L"], fail["yneu.L"]
    d, df, dy = al - sw, fail["al.L_first"] - fail["sw.L_first"], yal - ysw
    out = {
        "auc_neutral_two_candidate": auc(succ["neu.L"], fail["neu.L"]),
        "auc_neutral_yes_no": auc(succ["yneu.L"], fail["yneu.L"]),
        "mean_neutral_L_failures": float(np.mean(fail["neu.L"])),
        "mean_neutral_L_successes": float(np.mean(succ["neu.L"])),
        "mean_neutral_yes_no_L_failures": float(np.mean(yneu)),
        "mean_neutral_yes_no_L_successes": float(np.mean(succ["yneu.L"])),
        "answered_yes_neutral_failures_pct": float(100 * np.mean(yneu > 0)),
        "answered_yes_neutral_successes_pct": float(100 * np.mean(succ["yneu.L"] > 0)),
        "answered_yes_failures_swapped_pct": float(100 * np.mean(ysw > 0)),
        "answered_yes_failures_aligned_pct": float(100 * np.mean(yal > 0)),
        "dL": float(np.mean(d)), "dL_first": float(np.mean(df)), "dL_yes_no": float(np.mean(dy)),
        "standardized_two_candidate": float(np.mean(d) / np.std(d, ddof=1)),
        "standardized_yes_no": float(np.mean(dy) / np.std(dy, ddof=1)),
        "dp_pp": float(100 * np.mean(sig(al) - sig(sw))),
        "dp_yes_no_pp": float(100 * np.mean(sig(yal) - sig(ysw))),
        "share_toward_yes": float(np.mean(yal > ysw)),
        "flip_no_to_yes_pct": float(100 * np.mean((ysw < 0) & (yal > 0))),
        "flip_to_success_two_candidate_pct": float(100 * np.mean((sw < 0) & (al > 0))),
        "n_failures": int(len(al)), "n_successes": int(len(succ["neu.L"])),
    }
    if boot is not None:
        out["ci"] = {"dL": est(boot, mean_fn(d)), "dL_yes_no": est(boot, mean_fn(dy)),
                     "share_toward_yes": est(boot, mean_fn((yal > ysw).astype(float))),
                     "flip_no_to_yes_pct": est(boot, mean_fn(100.0 * ((ysw < 0) & (yal > 0))))}
    return out


EFFECTS = ("dL", "dL_first", "dL_yes_no", "standardized_two_candidate", "standardized_yes_no", "dp_pp",
           "dp_yes_no_pp", "share_toward_yes", "flip_no_to_yes_pct")
COMPETENCE = ("auc_neutral_yes_no", "auc_neutral_two_candidate", "mean_neutral_yes_no_L_failures",
              "answered_yes_neutral_failures_pct")


def correlations(models: dict, names) -> dict:
    rng = np.random.default_rng(20260924)
    out = {}
    for e in EFFECTS:
        for c in COMPETENCE:
            x = np.array([models[n][c] for n in names])
            y = np.array([models[n][e] for n in names])
            rho = float(spearmanr(x, y).statistic)
            perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(5000)])
            out[f"{e}~{c}"] = {"spearman": rho, "perm_p": float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / 5001),
                               "n_models": len(names)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--tc-items", required=True)
    ap.add_argument("--tc-scores", nargs="+", required=True)
    ap.add_argument("--families-dir", required=True)
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()

    A = D.load_alfworld(a.archive)
    T = D.load_textcraft(a.tc_items, a.tc_scores)
    res: dict = {"note": "exploratory, post hoc; section 9 of notes/textcraft_vs_alfworld.md"}

    # (a) the two reference models in both environments
    ref = {}
    for m in D.MODELS:
        af, asu = A[m]["fail"], A[m]["succ"]
        tm, ts, tu = T[m]["main"], T[m]["success"], T[m]["usefulness"]
        tfail = {k: np.concatenate([tm[k], tu[k]]) for k in ("neu.L", "yneu.L")}
        p_alf = profile(af, asu, Boot(af["cluster"]))
        p_tc = profile(tm, ts, Boot(tm["cluster"]))
        # TextCraft's competence as its design defines it: main and usefulness failures against successes
        p_tc["auc_neutral_two_candidate_with_usefulness"] = auc(ts["neu.L"], tfail["neu.L"])
        p_tc["auc_neutral_yes_no_with_usefulness"] = auc(ts["yneu.L"], tfail["yneu.L"])
        ref[m] = {"alfworld": p_alf, "textcraft": p_tc}
    res["reference_models_both_environments"] = ref

    # (b) ALFWorld across models: the two references (their confirmatory-run tables), the family models, the sizes
    m0 = D.MODELS[0]
    fail_ids, succ_ids = list(A[m0]["fail"]["id"]), list(A[m0]["succ"]["id"])
    fail_cl, succ_cl = A[m0]["fail"]["cluster"], A[m0]["succ"]["cluster"]
    models = {f"{m} (reference)": ref[m]["alfworld"] for m in D.MODELS}
    for name in FAMILY:
        rows = read_rows([os.path.join(a.families_dir, f"{name}.jsonl")])
        f, s = table_from(rows, fail_ids, fail_cl), table_from(rows, succ_ids, succ_cl)
        models[name] = profile(f, s, Boot(f["cluster"]))
    for name in SIZES:
        rows = read_rows(sorted(glob.glob(os.path.join(a.archive, "text-wm-fix18", "runs", f"size_score_{name}",
                                                       "shard_*", "scoring.jsonl"))))
        f, s = table_from(rows, fail_ids, fail_cl), table_from(rows, succ_ids, succ_cl)
        models[f"{name} (size series)"] = profile(f, s, Boot(f["cluster"]))
    res["alfworld_models"] = models
    # one entry per model: the size series' 8B is a rescore of the reference Qwen3-8B, so it is left out of the pooled set
    pooled = [n for n in models if n != "qwen3-8b (size series)"]
    competent = [n for n in pooled if models[n]["auc_neutral_yes_no"] >= 0.70
                 and models[n]["answered_yes_neutral_failures_pct"] <= 95
                 and models[n]["answered_yes_neutral_successes_pct"] >= 5]
    res["alfworld_correlations"] = {"all_models": {"models": pooled, **correlations(models, pooled)},
                                    "competent_models": {"models": competent, **correlations(models, competent)},
                                    "qwen3_sizes": {"models": [f"{s} (size series)" for s in SIZES],
                                                    **correlations(models, [f"{s} (size series)" for s in SIZES])}}

    with open(a.metrics, encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["competence_vs_effect"] = res
    with open(a.metrics, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("added competence_vs_effect to", a.metrics)


if __name__ == "__main__":
    main()
