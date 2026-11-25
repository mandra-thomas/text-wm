"""Pool every cross-environment comparison of a mean contrast and split its shift into item-level noise and a
systematic (bias) component.

For each comparison k the mean contrast moves by s_k, and the item-level numerical noise alone would give it a
standard error se_k (game-clustered bootstrap of the per-item contrast differences). If the noise were independent and
unbiased, s_k / se_k would be standard normal. The method-of-moments estimate of the systematic spread is
tau^2 = mean(s_k^2 - se_k^2), floored at zero.
Also reports per-item contrast-noise SD and per-score noise per model family and kind of environment change.
Output: out/floor.json
"""
from __future__ import annotations

import json

import numpy as np

from common import OUT, dump


def rows():
    out = []
    A = json.loads((OUT / "alfworld_pairs.json").read_text())
    for name, P in A.items():
        model = ("qwen3-32b" if "qwen32" in name else "qwen3-8b" if "qwen" in name else
                 "llama3.1-8b" if "llama" in name else "mistral-7b" if "mistral" in name else "gemma2-9b")
        for s, S in P["strata"].items():
            if S["same_gpu_type"] and not name.startswith("stack"):
                continue  # same environment: identical (reported separately)
            kind = "stack" if name.startswith("stack") else "gpu"
            for c, C in S["contrasts"].items():
                sh = C["shift_b_minus_a"]
                out.append({"source": f"alfworld/{name}/{s}", "model": model, "change": kind,
                            "readout": "yes/no" if c.startswith("yn_") else "two-candidate", "contrast": c,
                            "n": C["n_items"], "shift": sh["estimate"], "se": sh["se"], "item_sd": C["per_item_contrast_diff"]["sd"],
                            "design_effect": sh["design_effect"]})
    O = json.loads((OUT / "optionc_step0.json").read_text())
    for model, R in O.items():
        for c, C in R["stage1a_vs_optionc_L40S"]["contrasts"].items():
            out.append({"source": f"optionc_step0/{model}", "model": model, "change": "gpu", "readout": "yes/no (step 0)",
                        "contrast": c, "n": C["n"], "shift": C["shift"]["estimate"], "se": C["shift"]["se"],
                        "item_sd": C["per_item"]["sd"], "design_effect": C["shift"]["design_effect"]})
    T = json.loads((OUT / "textcraft.json").read_text())
    for model, M in T["per_model"].items():
        for c, C in M["contrasts"].items():
            if c.startswith("timing_gate"):
                continue
            ns = C["numerical_shift"]
            out.append({"source": f"textcraft/{model}", "model": model.replace("-v0.3", "").replace("-it", ""), "change": "stack",
                        "readout": "two-candidate (TextCraft)", "contrast": c, "n": None, "shift": ns["estimate"],
                        "se": ns["bootstrap_se"], "item_sd": None, "design_effect": None})
    return out


def tau(rs):
    s = np.array([r["shift"] for r in rs])
    se = np.array([r["se"] for r in rs])
    t2 = float(np.mean(s ** 2 - se ** 2))
    return {"k": len(rs), "rms_shift": float(np.sqrt(np.mean(s ** 2))), "rms_se": float(np.sqrt(np.mean(se ** 2))),
            "max_abs_shift": float(np.max(np.abs(s))), "tau": float(np.sqrt(max(t2, 0.0))),
            "share_abs_z_gt_2": float(np.mean(np.abs(s / se) > 2)), "mean_abs_z": float(np.mean(np.abs(s / se)))}


def main():
    R = rows()
    groups = {
        "qwen3-8b, GPU type": [r for r in R if r["model"] == "qwen3-8b" and r["change"] == "gpu"],
        "qwen3-8b, stack": [r for r in R if r["model"] == "qwen3-8b" and r["change"] == "stack"],
        "qwen3-32b, stack": [r for r in R if r["model"] == "qwen3-32b"],
        "llama3.1-8b, GPU type": [r for r in R if r["model"] == "llama3.1-8b" and r["change"] == "gpu"],
        "llama/mistral/gemma, stack": [r for r in R if r["model"] in ("llama3.1-8b", "mistral-7b", "gemma2-9b") and r["change"] == "stack"],
        "all non-Qwen": [r for r in R if not r["model"].startswith("qwen")],
        "all Qwen": [r for r in R if r["model"].startswith("qwen")],
    }
    res = {"rows": R, "groups": {g: tau(v) for g, v in groups.items() if v}}
    dump("floor", res)
    for r in R:
        print(f"{r['source']:55s} {r['readout']:26s} {r['contrast']:28s} shift {r['shift']:+.3f} se {r['se']:.3f} z {r['shift']/r['se']:+.1f}")
    for g, t in res["groups"].items():
        print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()})


if __name__ == "__main__":
    main()
