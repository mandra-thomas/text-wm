"""Print the audit's tables from the metrics JSON written by analyze.py (for reading; the note quotes these).

Usage: python scripts/qwen_audit/report.py notes/qwen_effect_audit_forensic_metrics.json [section ...]
Sections: A B C D E F C3 C5 all (default all).
"""
import json
import sys


def ci(d, k="mean", f="{:.2f}"):
    return f"{f.format(d[k])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def main():
    M = json.load(open(sys.argv[1]))
    want = set(sys.argv[2:]) or {"all"}
    on = lambda s: "all" in want or s in want
    models = M["models"]
    if on("A"):
        A = M["A"]
        print("== A sanity")
        for k in ("run_problems", "stale_item_hash", "missing_primary", "candidate_signature_mismatch_items", "commits",
                  "rq2_task_line_used_mismatch", "qwen_rq2_vs_rescore_by_rq2_shard", "llama_rq2_vs_rq1_score",
                  "qwen_rq2_vs_rq1_score", "A2_prompt_check"):
            print(k, json.dumps(A.get(k, M.get(k)), indent=None)[:3000])
    if on("C"):
        print("== C1 decomposition (aligned - swapped): L, first token, success tail, failure tail; tail share")
        for n, b in models.items():
            e = b["effect"]
            print(f"{n:24s} L {ci(e['L'])}  F {ci(e['F'])}  Tp {ci(e['Tp'])}  Tm {e['Tm']['mean']:.2f}  "
                  f"share {b['tail_share']['ratio']:.2f} [{b['tail_share']['lo']:.2f},{b['tail_share']['hi']:.2f}]  "
                  f"later: L {b['effect_later']['L']['mean']:.2f} F {b['effect_later']['F']['mean']:.2f} "
                  f"Tp {b['effect_later']['Tp']['mean']:.2f} share {b['tail_share_later']['ratio']:.2f}  "
                  f"step0: L {b['effect_step0']['L']['mean']:.2f} F {b['effect_step0']['F']['mean']:.2f} Tp {b['effect_step0']['Tp']['mean']:.2f}")
        for n in ("qwen3-8b", "llama3.1-8b"):
            print(n, "tail vs length", models[n]["tail_vs_length"])
            print(n, "tail vs object tokens", models[n].get("tail_vs_object_tokens"))
    if on("B"):
        print("== B where on the scale")
        for n in ("qwen3-8b", "llama3.1-8b", "mistral-7b", "gemma2-9b"):
            b = models[n]
            print("--", n, "dlp", ci(b["effect"]["lp"]), "dlm", ci(b["effect"]["lm"]),
                  "share lp up", round(b["share_items_lp_up"], 3), "share lm down", round(b["share_items_lm_down"], 3))
            for c in ("aligned", "swapped", "neutral"):
                if c not in b["dist"]:
                    continue
                d = b["dist"][c]
                for k in ("L", "lp", "lm", "F", "Tp"):
                    x = d[k]
                    print(f"   {c:8s} {k:3s} mean {x['mean']:7.2f} sd {x['sd']:6.2f} p1 {x['p1']:7.2f} p10 {x['p10']:7.2f} "
                          f"p25 {x['p25']:7.2f} p50 {x['p50']:7.2f} p75 {x['p75']:7.2f} p90 {x['p90']:7.2f} p99 {x['p99']:7.2f}")
                if "prob" in d:
                    print("   ", c, json.dumps(d["prob"]))
            print("   by swapped decile:", [(x["decile"], round(x["L_swapped_range"][0], 1), round(x["dL"], 2), round(x["dF"], 2), round(x["dTp"], 2)) for x in b["by_swapped_decile"]])
            if "vs_neutral" in b:
                for k, v in b["vs_neutral"].items():
                    print("   ", k, {kk: round(vv["mean"], 2) for kk, vv in v.items()})
    if on("D"):
        print("== D heavy tails")
        for n, b in models.items():
            h = b["heavy"]
            print(f"{n:24s} mean {h['mean']:.2f} median {h['median']:.2f} [{h['median_ci']['lo']:.2f},{h['median_ci']['hi']:.2f}] "
                  f"trim10 {h['trim10']:.2f} trim20 {h['trim20']:.2f} [{h['trim20_ci']['lo']:.2f},{h['trim20_ci']['hi']:.2f}] "
                  f">10 {h['share_gt10']:.3f} >20 {h['share_gt20']:.3f} <0 {h['share_lt0']:.3f} =0 {h['share_eq0']:.3f} "
                  f"top1% {h['top1pct_share_of_sum']:.3f} top5% {h['top5pct_share_of_sum']:.3f} top10% {h['top10pct_share_of_sum']:.3f} "
                  f"top5g {h['top5_games_share_of_sum']:.3f} (items {h['top5_games_share_of_items']:.3f}) top10g {h['top10_games_share_of_sum']:.3f} "
                  f"loo {h['loo_game_max_abs_change']:.2f} w/o top1% {h['mean_without_top1pct']:.2f} games {h['game_means']} F {h['first_token']}")
        for n in ("qwen3-8b", "llama3.1-8b"):
            for key, g in models[n]["groups"].items():
                print("  ", n, key, {k: (v["n"], round(v["dL"], 2), round(v["dF"], 2), round(v["dTp"], 2), round(v["median_dL"], 2), round(v["contribution"], 2)) for k, v in g.items()})
    if on("E"):
        print("== E scale (two-candidate L, first token F, yes/no)")
        for n, b in models.items():
            for k in ("L", "F"):
                s = b["scale"][k]
                print(f"{n:24s} {k} sdA {s['sd_aligned']:.2f} sdS {s['sd_swapped']:.2f} sdN {s.get('sd_neutral', float('nan')):.2f} "
                      f"|S| {s['mean_abs_swapped']:.2f} medS {s['median_swapped']:.2f} sdD {s['sd_delta']:.2f} dz {s['dz']:.2f} "
                      f"[{s['dz_ci']['lo']:.2f},{s['dz_ci']['hi']:.2f}] /sdS {s['std_by_sd_swapped']:.2f} /sdN {s.get('std_by_sd_neutral', float('nan')):.2f} "
                      f"pos {s['share_delta_pos']:.3f} neg {s['share_delta_neg']:.3f} AUC {s['pooled_auc_aligned_over_swapped']:.3f} "
                      f"up {s['sign_up']:.3f} either {s['sign_either']:.3f} dsig {s['dsigma_pp']:.1f}")
            y = b["yes_no"]; s = y["scale"]
            print(f"{n:24s} YN eff {ci(y['effect'])} later {y['effect_later']['mean']:.2f} step0 {y['effect_step0']['mean']:.2f} "
                  f"med {y['median']:.2f} trim20 {y['trim20']:.2f} sdS {s['sd_swapped']:.2f} sdN {s.get('sd_neutral', float('nan')):.2f} "
                  f"medS {s['median_swapped']:.2f} dz {s['dz']:.2f} [{s['dz_ci']['lo']:.2f},{s['dz_ci']['hi']:.2f}] /sdS {s['std_by_sd_swapped']:.2f} "
                  f"pos {s['share_delta_pos']:.3f} AUC {s['pooled_auc_aligned_over_swapped']:.3f} yesS {s['answered_yes_swapped']:.3f} "
                  f"yesA {s['answered_yes_aligned']:.3f} flip {s['flip_no_to_yes']:.3f} dsig {s['dsigma_pp']:.1f} "
                  f"nearB {s['share_swapped_within_mean_shift_of_boundary']:.3f} near5 {s['share_swapped_within_5nats_of_boundary']:.3f} "
                  f"rho(2c,yn) {y['spearman_item_dL2c_vs_dLyn']:.2f} rho(F,yn) {y['spearman_item_dF_vs_dLyn']:.2f}")
    if on("E"):
        print("== below decision (added during analysis)")
        for n, b in models.items():
            y = b["yes_no"]["scale"]
            print(f"{n:24s} L {json.dumps({k: round(v, 3) for k, v in b['below_decision']['L'].items()})} "
                  f"F {json.dumps({k: round(v, 3) for k, v in b['below_decision']['F'].items()})} "
                  f"YN clip5 {y['clipped_pm5_effect']:.2f} clip10 {y['clipped_pm10_effect']:.2f} "
                  f"below10 {y['share_of_summed_effect_both_below_minus10']:.2f} rho(dF,dTp) {b['spearman_dF_vs_dTp']:.2f}")
    if on("F"):
        print("== F format gap, Qwen3-8B")
        for k, v in M["F_qwen_format_gap"].items():
            print(k, {kk: (round(vv, 2) if isinstance(vv, float) else ci(vv)) for kk, vv in v.items()})
        print("== F standardized")
        for n, r in M["F_standardized"].items():
            print(f"{n:24s}", " ".join(f"{k}={v:.2f}" for k, v in r.items() if "dz" in k or "sd_swapped" in k or "share_pos" in k))
    if on("C3"):
        print("== C3 contrasts")
        for model, c in M["C3_contrasts"].items():
            for k, v in c.items():
                if v:
                    print(f"{model[:5]} {k:52s} n {v['n']:5d} L {ci(v['L'])} F {ci(v['F'])} Tp {ci(v['Tp'])} Tm {v['Tm']['mean']:.2f} share {v['tail_share']:.2f}")
    if on("C5"):
        print("== C5 generations")
        for k, v in M["C5_generations"].items():
            print(k, v)


if __name__ == "__main__":
    main()
