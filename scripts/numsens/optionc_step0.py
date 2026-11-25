"""Option C step-0 yes/no scores against Stage 1a's (same prompts, same stack, same code path, different GPU type).

Stage 1a: decisionsim_score_<model> (Qwen3-8B on an A100-PCIE-40GB, Llama-3.1-8B on an A40).
Option C step-0: optionc_step0_<model> (both on L40S). The follow-up rescoring on Stage 1a's GPU types
(optionc_fu/rescore_<model>.jsonl) is also compared, to confirm same-type reproduction.
Contrasts: aware minus swapped (the RQ2-style goal effect) and aware minus neutral, per candidate action.
Decision statistics: acceptance (rounded score above zero), acceptance flips, and the aware-minus-blind gap in
accepting working first moves is not recomputed here (blind was not part of the Stage 1a comparison).
Prompt length: n_tokens from optionc_fu/prompts_oc_<model>.jsonl.
Output: out/optionc_step0.json
"""
from __future__ import annotations

import json

import numpy as np

from common import DATA, binned, cluster_boot_mean, describe, dump, grid_of, independence, spearman

ARM2COND = {"aware": "yn_aligned", "swapped": "yn_swapped", "neutral": "yn_absent"}


def load(model: str):
    s1a = {}
    for line in open(next(DATA.glob(f"text-wm-fix30/runs/decisionsim_score_{model}/*/scoring.jsonl"))):
        r = json.loads(line)
        s1a[(r["transition_id"], r["condition"])] = r
    oc = {}
    for line in open(DATA / f"text-wm-fix40/runs/optionc_step0_{model}/scoring.jsonl"):
        r = json.loads(line)
        if r["readout"] == "yn" and r["arm"] in ARM2COND:
            oc[(r["transition_id"], ARM2COND[r["arm"]])] = r
    fu = {}
    for line in open(DATA / f"text-wm-fix50/runs/optionc_fu/rescore_{model}.jsonl"):
        st = json.loads(line)
        for r in st["records"]:
            if r["readout"] == "yn" and r["arm"] in ARM2COND:
                fu[(r["transition_id"], ARM2COND[r["arm"]])] = dict(r, game_id=st["game_id"])
    ntok = {}
    for line in open(DATA / f"text-wm-fix50/runs/optionc_fu/prompts_oc_{model}.jsonl"):
        r = json.loads(line)
        ntok[(r["transition_id"], r["condition"])] = r["n_tokens"]
    return s1a, oc, fu, ntok


def rnd(x):
    return round(x * 64) / 64


def compare(A, B, ntok, games_from):
    keys = sorted(A.keys() & B.keys())
    d = np.array([B[k]["L_raw" if "L_raw" in B[k] else "L"] - (A[k]["L_raw"] if "L_raw" in A[k] else A[k]["L"]) for k in keys])
    LA = np.array([A[k]["L_raw"] if "L_raw" in A[k] else A[k]["L"] for k in keys])
    LB = LA + d
    acc_flip = int(((np.array([rnd(x) for x in LA]) > 0) != (np.array([rnd(x) for x in LB]) > 0)).sum())
    nt = np.array([ntok.get(k, np.nan) for k in keys], float)
    games = [games_from[k]["game_id"] for k in keys]
    out = {"per_score": describe(d), "grid": grid_of(d), "acceptance_flips": acc_flip,
           "corr_scores": float(np.corrcoef(LA, LB)[0, 1]),
           "magnitude": {"spearman_absdiff_vs_abs_score": spearman(np.abs(d), np.abs(0.5 * (LA + LB))),
                         "bins_abs_score": binned(np.abs(d), np.abs(0.5 * (LA + LB)), [0, 1, 2, 4, 8, 16, 24, 32, 1e9])},
           "prompt_length": {"spearman_absdiff_vs_n_tokens": spearman(np.abs(d)[~np.isnan(nt)], nt[~np.isnan(nt)]),
                             "bins_n_tokens": binned(np.abs(d)[~np.isnan(nt)], nt[~np.isnan(nt)], [0, 450, 500, 550, 600, 700, 1e9])},
           "bias_by_condition": {}}
    for c in sorted({k[1] for k in keys}):
        m = np.array([k[1] == c for k in keys])
        out["bias_by_condition"][c] = cluster_boot_mean(d[m], [g for g, mm in zip(games, m) if mm], B=2000)
    contrasts = {}
    for c1, c2 in (("yn_aligned", "yn_swapped"), ("yn_aligned", "yn_absent")):
        tids = sorted({t for t, c in keys if c == c1} & {t for t, c in keys if c == c2})
        get = lambda R, t, c: R[(t, c)]["L_raw"] if "L_raw" in R[(t, c)] else R[(t, c)]["L"]
        eA = np.array([get(A, t, c1) - get(A, t, c2) for t in tids])
        eB = np.array([get(B, t, c1) - get(B, t, c2) for t in tids])
        d1 = np.array([get(B, t, c1) - get(A, t, c1) for t in tids])
        d2 = np.array([get(B, t, c2) - get(A, t, c2) for t in tids])
        g = [games_from[(t, c1)]["game_id"] for t in tids]
        flips = int(((np.sign(eA) != np.sign(eB)) & (eA != 0) & (eB != 0)).sum())
        contrasts[f"{c1}-{c2}"] = {"n": len(tids), "effect_a": float(eA.mean()), "effect_b": float(eB.mean()),
                                    "shift": cluster_boot_mean(eB - eA, g), "per_item": describe(eB - eA),
                                    "item_r": float(np.corrcoef(eA, eB)[0, 1]), "sign_flips": flips,
                                    "independence_of_arms": independence(d1, d2),
                                    "share_positive_a": float((eA > 0).mean()), "share_positive_b": float((eB > 0).mean())}
    out["contrasts"] = contrasts
    return out


def main():
    res = {}
    for model in ("qwen3-8b", "llama3.1-8b"):
        s1a, oc, fu, ntok = load(model)
        res[model] = {
            "stage1a_vs_optionc_L40S": compare(s1a, oc, ntok, oc),
            "rescore_same_type_vs_stage1a": compare(s1a, fu, ntok, oc),
        }
    dump("optionc_step0", res)
    for m, R in res.items():
        for k, v in R.items():
            p = v["per_score"]
            print(m, k, "n", p["n"], "ident", round(p["share_identical"], 3), "mean|d|", round(p["mean_abs"], 3),
                  "sd", round(p["sd"], 3), "max", p["max_abs"], "kurt", p["excess_kurtosis"], "acc flips", v["acceptance_flips"])
            print("   magnitude", v["magnitude"]["spearman_absdiff_vs_abs_score"],
                  [(b["bin"][0], round(b["mean_abs"], 3), b["n"]) for b in v["magnitude"]["bins_abs_score"]])
            print("   length", v["prompt_length"]["spearman_absdiff_vs_n_tokens"],
                  [(b["bin"][0], round(b["mean_abs"], 3), b["n"]) for b in v["prompt_length"]["bins_n_tokens"]])
            print("   bias", {c: (round(b["estimate"], 3), [round(x, 3) for x in b["ci95"]]) for c, b in v["bias_by_condition"].items()})
            for c, C in v["contrasts"].items():
                print("  ", c, "n", C["n"], "effA", round(C["effect_a"], 3), "effB", round(C["effect_b"], 3), "shift",
                      round(C["shift"]["estimate"], 3), [round(x, 3) for x in C["shift"]["ci95"]], "sd", round(C["per_item"]["sd"], 3),
                      "r", round(C["item_r"], 4), "flips", C["sign_flips"], "ratio",
                      round(C["independence_of_arms"]["ratio_var_contrast_to_sum"], 2), "corr", round(C["independence_of_arms"]["corr"], 2),
                      "share+", round(C["share_positive_a"], 3), round(C["share_positive_b"], 3), "kurt", C["per_item"]["excess_kurtosis"])


if __name__ == "__main__":
    main()
