"""Print compact tables from out/*.json (used to write the report)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def f(x, n=3):
    return "nan" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def alfworld():
    d = json.loads((OUT / "alfworld_pairs.json").read_text())
    print("pair | stratum | n scores | identical | mean|d| | sd | p99 | max | kurt(nonzero) | yn grid1/8")
    for name, P in d.items():
        for s, S in P["strata"].items():
            ps = S["per_score_pooled"]
            nz = ps.get("nonzero") or {}
            print(f"{name} | {s} | {ps['n']} | {f(ps['share_identical'])} | {f(ps['mean_abs'])} | {f(ps['sd'])} | "
                  f"{f(ps['p99_abs'])} | {f(ps['max_abs'])} | {f(nz.get('excess_kurtosis'),1)}")
    print()
    print("pair | stratum | contrast | n | effA | shift [95%] | deff | sd(dE) | p99|dE| | max | r | flips | shareA->B | zA->zB | ratio var | corr arms | corr nz")
    for name, P in d.items():
        for s, S in P["strata"].items():
            for c, C in S["contrasts"].items():
                sh = C["shift_b_minus_a"]
                pc = C["per_item_contrast_diff"]
                ind = C["independence_of_arms"]
                print(f"{name} | {s} | {c} | {C['n_items']} | {f(C['effect_a']['estimate'],2)} | {f(sh['estimate'])} "
                      f"[{f(sh['ci95'][0])}, {f(sh['ci95'][1])}] | {f(sh['design_effect'],1)} | {f(pc['sd'])} | {f(pc['p99_abs'])} | {f(pc['max_abs'])} | "
                      f"{f(C['item_r'],4)} | {C['sign_flips']} | {f(C['item_share_positive_a'])}->{f(C['item_share_positive_b'])} | "
                      f"{f(C['standardized_a'])}->{f(C['standardized_b'])} | {f(ind['ratio_var_contrast_to_sum'],2)} | {f(ind['corr'],2)} | {f(ind['corr_where_any_changed'],2)}")
    print()
    print("pair | stratum | condition | n | identical | mean signed [95%] | mean|d| | grid 1/8")
    for name, P in d.items():
        for s, S in P["strata"].items():
            for c, C in S["per_score_by_condition"].items():
                b = C["bias"]
                g = C.get("grid") or {}
                print(f"{name} | {s} | {c} | {b['n']} | {f(C['describe']['share_identical'])} | {f(b['estimate'])} [{f(b['ci95'][0])}, {f(b['ci95'][1])}] | "
                      f"{f(C['describe']['mean_abs'])} | {f(g.get('share_multiple_of_1_8'))}")
    print()
    print("pair | stratum | rho(|d|,|score|) yn / 2c | rho(|d|,first-token size) 2c | rho(|d|,hist chars) yn / 2c | bins |score| 2c")
    for name, P in d.items():
        for s, S in P["strata"].items():
            my, m2 = S.get("magnitude_yes_no", {}), S.get("magnitude_two_candidate", {})
            ly, l2 = S.get("prompt_length_yes_no", {}), S.get("prompt_length_two_candidate", {})
            bins = " ".join(f"{b['bin'][0]}:{f(b['mean_abs'],2)}" for b in m2.get("bins_abs_score", []))
            binsy = " ".join(f"{b['bin'][0]}:{f(b['mean_abs'],2)}" for b in my.get("bins_abs_score", []))
            print(f"{name} | {s} | {f(my.get('spearman_absdiff_vs_abs_score'),2)} / {f(m2.get('spearman_absdiff_vs_abs_score'),2)} | "
                  f"{f(m2.get('spearman_absdiff_vs_first_token_logprob_size'),2)} | {f(ly.get('spearman_absdiff_vs_history_chars'),2)} / "
                  f"{f(l2.get('spearman_absdiff_vs_history_chars'),2)} | 2c {bins} | yn {binsy}")
    print()
    for name, P in d.items():
        for s, S in P["strata"].items():
            if "share_identical_by_shard_pair" in S:
                print(name, s, {k: (v['n'], round(v['share'], 3)) for k, v in S["share_identical_by_shard_pair"].items()})


if __name__ == "__main__":
    {"alfworld": alfworld}[sys.argv[1] if len(sys.argv) > 1 else "alfworld"]()
