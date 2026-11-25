"""ALFWorld paired rescorings: the same prompts scored twice under different numerics.

For every pair of runs and every GPU-type stratum, computes
  * per-score differences (B minus A) for each condition and pooled,
  * per-item contrast differences for the contrasts the project analyses (aligned minus swapped, aligned minus absent),
  * whether the two arms' differences cancel, add or correlate,
  * dependence on score magnitude and on prompt length (history characters, history steps),
  * condition-specific bias (mean signed per-score difference by condition, game-clustered),
  * the shift of the aggregate effect, item-share and sign flips, standardized effect.
Output: out/alfworld_pairs.json
"""
from __future__ import annotations

import numpy as np

from common import (binned, cluster_boot_mean, describe, dump, grid_of, independence, load_scores, spearman,
                    transitions_meta)

RQ2_GPU = {0: "A40", 1: "A40", 2: "A40", 3: "L40S"}
RQ1_GPU = {0: "L40S", 1: "A40", 2: "L40S", 3: "L40S", 4: "A100-80GB", 5: "A100-80GB", 6: "A100-40GB", 7: "L40S"}
RQ1V_GPU = {0: "L40S", 1: "L40S", 2: "L40S", 3: "L40S", 4: "L40S", 5: "A40", 6: "L40S", 7: "L40S"}

TC = [("aligned", "swapped"), ("aligned", "absent")]
YN = [("yn_aligned", "yn_swapped"), ("yn_aligned", "yn_absent")]


def runs():
    q = lambda g, **k: load_scores(g, model_filter="qwen3-8b", **k)
    l = lambda g, **k: load_scores(g, model_filter="llama3.1-8b", **k)
    R = {
        "rq2_qwen": q("text-wm-fix11/runs/rq2_score/*/scoring.jsonl", shard_gpu=RQ2_GPU),
        "rq2_llama": l("text-wm-fix11/runs/rq2_score/*/scoring.jsonl", shard_gpu=RQ2_GPU),
        "rq1_qwen": q("text-wm-fix12/runs/rq1_score/*/scoring.jsonl", shard_gpu=RQ1_GPU),
        "rq1_llama": l("text-wm-fix12/runs/rq1_score/*/scoring.jsonl", shard_gpu=RQ1_GPU),
        "rq1v_qwen": q("text-wm-fix17/runs/rq1v_score/*/scoring.jsonl", shard_gpu=RQ1V_GPU),
        "rq1v_llama": l("text-wm-fix17/runs/rq1v_score/*/scoring.jsonl", shard_gpu=RQ1V_GPU),
        "size_qwen8": q("text-wm-fix18/runs/size_score_qwen3-8b/*/scoring.jsonl", default_gpu="L40S"),
        "size_qwen32": load_scores("text-wm-fix18/runs/size_score_qwen3-32b/*/scoring.jsonl", default_gpu="A100-80GB"),
        "s2_qwen8": load_scores("text-wm-fix52/runs/family_rescore_qwen3-8b/*/scoring.jsonl"),
        "s2_llama": load_scores("text-wm-fix52/runs/family_rescore_llama3.1-8b/*/scoring.jsonl"),
        "s2_mistral": load_scores("text-wm-fix52/runs/family_rescore_mistral-7b-v0.3/*/scoring.jsonl"),
        "s2_gemma": load_scores("text-wm-fix52/runs/family_rescore_gemma2-9b-it/*/scoring.jsonl"),
        "s2_qwen32": load_scores("text-wm-fix52/runs/family_rescore_qwen3-32b/*/scoring.jsonl"),
        "old_mistral": load_scores("text-wm-fix21/runs/family_score_mistral-7b-v0.3/*/scoring.jsonl", default_gpu="A40"),
        "old_gemma": load_scores("text-wm-fix21/runs/family_score_gemma2-9b-it/*/scoring.jsonl", default_gpu="L40S"),
        "envchk_qwen8": load_scores("text-wm-fix39/runs/family_envcheck_qwen3-8b/*/scoring.jsonl"),
        "envchk_mistral": load_scores("text-wm-fix39/runs/family_envcheck_mistral-7b-v0.3/*/scoring.jsonl"),
    }
    return R


# (name, what differs, run A, run B, contrasts)
PAIRS = [
    ("gpu_qwen8_rq2_vs_size", "GPU type (A40 vs L40S) or same-type rerun; old stack; same code path", "rq2_qwen", "size_qwen8", TC[:1]),
    ("gpu_qwen8_rq1_vs_size", "GPU type (L40S/A40/A100 vs L40S) or same-type rerun; old stack", "rq1_qwen", "size_qwen8", TC),
    ("gpu_qwen8_rq1v_vs_size", "GPU type (A40 vs L40S) or same-type rerun; old stack; yes/no and two-candidate", "rq1v_qwen", "size_qwen8", TC[:1] + YN[:1] + [("yn_aligned", "yn_absent")]),
    ("gpu_llama_rq2_vs_rq1v", "GPU type or same-type rerun; old stack", "rq2_llama", "rq1v_llama", TC[:1]),
    ("gpu_llama_rq1_vs_rq1v", "GPU type or same-type rerun; old stack", "rq1_llama", "rq1v_llama", TC[:1]),
    ("stack_qwen8_L40S", "software stack (torch 2.6/transformers 4.51 -> torch 2.14/transformers 5.23); L40S both", "size_qwen8", "s2_qwen8", TC + YN),
    ("stack_llama_vs_rq1v", "software stack (+ GPU type in the rq1v A40 shard)", "rq1v_llama", "s2_llama", TC[:1] + YN),
    ("stack_mistral_A40", "software stack; A40 both", "old_mistral", "s2_mistral", TC + YN),
    ("stack_gemma2_L40S", "software stack; L40S both; eager attention", "old_gemma", "s2_gemma", TC + YN),
    ("stack_qwen32_A100", "software stack; A100-80GB both", "size_qwen32", "s2_qwen32", TC + YN),
    ("rerun_serve2_qwen8_L40S", "same stack (serve2), same GPU type (L40S), different job and day", "envchk_qwen8", "s2_qwen8", TC + YN),
    ("rerun_serve2_mistral_A40", "same stack (serve2), same GPU type (A40), different job and day", "envchk_mistral", "s2_mistral", TC + YN),
]

ABS_EDGES = [0, 1, 2, 4, 8, 16, 32, 1e9]
LEN_EDGES = [0, 1, 3, 6, 9, 13, 1e9]


def analyse_pair(A: dict, B: dict, contrasts, meta) -> dict:
    strata: dict[tuple, list] = {}
    for key in A.keys() & B.keys():
        strata.setdefault((A[key]["_gpu"], B[key]["_gpu"]), []).append(key)
    res = {}
    for (ga, gb), keys in sorted(strata.items(), key=lambda kv: str(kv[0])):
        keys = set(keys)
        S = {"gpu_a": ga, "gpu_b": gb, "same_gpu_type": ga == gb}
        conds = sorted({c for (_t, c) in keys})
        # ---- per score
        per_cond = {}
        all_d, all_abs_level, all_len, all_steps, all_ntok = [], [], [], [], []
        for c in conds:
            ks = [k for k in keys if k[1] == c]
            d = np.array([B[k]["L"] - A[k]["L"] for k in ks])
            games = [A[k]["game_id"] for k in ks]
            per_cond[c] = {"describe": describe(d), "bias": cluster_boot_mean(d, games, B=2000),
                           "grid": grid_of(d) if c.startswith("yn_") else None}
            all_d += list(d)
            all_abs_level += [abs(0.5 * (B[k]["L"] + A[k]["L"])) for k in ks]
            all_len += [meta.get(k[0], {}).get("history_chars", np.nan) for k in ks]
            all_steps += [meta.get(k[0], {}).get("history_len", np.nan) for k in ks]
            all_ntok += [A[k].get("n_tokens_plus", np.nan) for k in ks]
        all_d = np.array(all_d)
        ad = np.abs(all_d)
        lv, ln, st = np.array(all_abs_level), np.array(all_len, float), np.array(all_steps, float)
        ok = ~np.isnan(ln)
        S["per_score_pooled"] = describe(all_d)
        S["per_score_by_condition"] = per_cond
        S["magnitude"] = {"spearman_absdiff_vs_abs_score": spearman(ad, lv), "bins_abs_score": binned(ad, lv, ABS_EDGES)}
        S["prompt_length"] = {"spearman_absdiff_vs_history_chars": spearman(ad[ok], ln[ok]),
                              "spearman_absdiff_vs_history_steps": spearman(ad[ok], st[ok]),
                              "bins_history_steps": binned(ad[ok], st[ok], LEN_EDGES)}
        # yes/no and two-candidate separately for the magnitude reading
        for tag, pref in (("yes_no", True), ("two_candidate", False)):
            ks = [k for k in keys if k[1].startswith("yn_") == pref]
            if not ks:
                continue
            d = np.abs(np.array([B[k]["L"] - A[k]["L"] for k in ks]))
            lvl = np.array([abs(0.5 * (B[k]["L"] + A[k]["L"])) for k in ks])
            lp = np.array([abs(A[k]["logp_first_plus"]) + abs(A[k]["logp_first_minus"]) for k in ks])
            S[f"magnitude_{tag}"] = {"spearman_absdiff_vs_abs_score": spearman(d, lvl),
                                     "spearman_absdiff_vs_first_token_logprob_size": spearman(d, lp),
                                     "bins_abs_score": binned(d, lvl, ABS_EDGES)}
            lnk = np.array([meta.get(k[0], {}).get("history_chars", np.nan) for k in ks], float)
            m = ~np.isnan(lnk)
            S[f"prompt_length_{tag}"] = {"spearman_absdiff_vs_history_chars": spearman(d[m], lnk[m])}
        # ---- per contrast
        C = {}
        for c1, c2 in contrasts:
            tids = sorted({t for (t, c) in keys if c == c1} & {t for (t, c) in keys if c == c2})
            if len(tids) < 10:
                continue
            eA = np.array([A[(t, c1)]["L"] - A[(t, c2)]["L"] for t in tids])
            eB = np.array([B[(t, c1)]["L"] - B[(t, c2)]["L"] for t in tids])
            d1 = np.array([B[(t, c1)]["L"] - A[(t, c1)]["L"] for t in tids])
            d2 = np.array([B[(t, c2)]["L"] - A[(t, c2)]["L"] for t in tids])
            dE = eB - eA
            games = [A[(t, c1)]["game_id"] for t in tids]
            sdA = eA.std(ddof=1)
            sdB = eB.std(ddof=1)
            nzA, nzB = eA != 0, eB != 0
            flips = int(((np.sign(eA) != np.sign(eB)) & nzA & nzB).sum())
            C[f"{c1}-{c2}"] = {
                "n_items": len(tids),
                "effect_a": cluster_boot_mean(eA, games, B=2000),
                "effect_b": float(eB.mean()),
                "shift_b_minus_a": cluster_boot_mean(dE, games),
                "per_item_contrast_diff": describe(dE),
                "item_effect_sd_a": float(sdA),
                "noise_share_of_item_variance": float(dE.var(ddof=1) / (2 * eA.var(ddof=1))),
                "item_r": float(np.corrcoef(eA, eB)[0, 1]),
                "independence_of_arms": independence(d1, d2),
                "sign_flips": flips, "share_sign_flips": flips / len(tids),
                "item_share_positive_a": float((eA > 0).mean()), "item_share_positive_b": float((eB > 0).mean()),
                "standardized_a": float(eA.mean() / sdA), "standardized_b": float(eB.mean() / sdB),
                "contrast_noise_vs_abs_effect": spearman(np.abs(dE), np.abs(0.5 * (eA + eB))),
                "contrast_noise_vs_history_chars": spearman(np.abs(dE), np.array(
                    [meta.get(t, {}).get("history_chars", 0) for t in tids], float)),
            }
        S["contrasts"] = C
        # same-type strata: which shards hold the non-identical scores
        if ga == gb:
            by_shard = {}
            for k in keys:
                s = (A[k]["_shard"], B[k]["_shard"])
                by_shard.setdefault(str(s), []).append(abs(B[k]["L"] - A[k]["L"]) < 1e-6)
            S["share_identical_by_shard_pair"] = {s: {"n": len(v), "share": float(np.mean(v))} for s, v in sorted(by_shard.items())}
        res[f"{ga}|{gb}"] = S
    return res


def main():
    R = runs()
    meta = transitions_meta()
    out = {}
    for name, what, a, b, contrasts in PAIRS:
        print("pair", name, flush=True)
        out[name] = {"what_differs": what, "run_a": a, "run_b": b, "strata": analyse_pair(R[a], R[b], contrasts, meta)}
    dump("alfworld_pairs", out)


if __name__ == "__main__":
    main()
