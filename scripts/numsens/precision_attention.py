"""Smaller comparisons: precision (bf16 against float32 weights), engine (vLLM against the study scorer),
attention implementation (eager against SDPA), and batch shape (from the archived smoke records).

* Qwen pipeline audit (text-wm-qwenaudit2/runs/qwen_audit, 200 items, aligned and swapped, two formats):
  the independent bf16 scorer (SDPA), float32 weights with eager attention, and vLLM 0.8.5, against each other.
  The float32 run is the closest thing to an exact reference: its difference from bf16 is the bf16 error itself.
* attention_agreement blocks in tokenization_checks.json (Gemma 4 jobs: wave B and TextCraft round 3).
* lexsem batched smoke (max differences only).
Output: out/precision_attention.json
"""
from __future__ import annotations

import json

import numpy as np

from common import DATA, describe, dump, independence

QA = DATA / "text-wm-qwenaudit2/runs/qwen_audit"


def load_audit(name):
    out = {}
    for line in open(QA / f"{name}.jsonl"):
        r = json.loads(line)
        for k, v in r["scores"].items():
            out[(r["id"], k)] = v
    return out


def compare(A, B, fmt):
    ids = sorted({i for (i, k) in A if k == f"{fmt}/aligned"} & {i for (i, k) in B if k == f"{fmt}/aligned"}
                 & {i for (i, k) in A if k == f"{fmt}/swapped"} & {i for (i, k) in B if k == f"{fmt}/swapped"})
    if not ids:
        return None
    res = {}
    for field in ("L", "L_first"):
        da = np.array([B[(i, f"{fmt}/aligned")][field] - A[(i, f"{fmt}/aligned")][field] for i in ids])
        ds = np.array([B[(i, f"{fmt}/swapped")][field] - A[(i, f"{fmt}/swapped")][field] for i in ids])
        eA = np.array([A[(i, f"{fmt}/aligned")][field] - A[(i, f"{fmt}/swapped")][field] for i in ids])
        eB = eA + da - ds
        res[field] = {"n_items": len(ids), "per_score": describe(np.concatenate([da, ds])),
                      "bias_aligned": float(da.mean()), "bias_swapped": float(ds.mean()),
                      "effect_a": float(eA.mean()), "effect_b": float(eB.mean()), "shift": float((eB - eA).mean()),
                      "shift_se": float((eB - eA).std(ddof=1) / np.sqrt(len(ids))),
                      "per_item_contrast": describe(eB - eA), "independence_of_arms": independence(da, ds),
                      "item_r": float(np.corrcoef(eA, eB)[0, 1])}
    return res


def attention_blocks():
    out = {}
    for f in sorted(DATA.glob("text-wm-fix*/runs/*/shard_*/tokenization_checks.json")):
        d = json.loads(f.read_text())
        a = d.get("attention_agreement")
        if not a:
            continue
        sc = a.get("scores") or []
        diffs = [{"condition": s.get("condition"), "diff_sdpa_minus_eager": s["L_sdpa"] - s["L_eager"]} for s in sc
                 if "L_sdpa" in s and "L_eager" in s]
        out[str(f.relative_to(DATA))] = {k: a.get(k) for k in ("ok", "yes_no_mean_abs_diff", "two_candidate_mean_abs_diff",
                                                                "max_abs_diff", "problems", "n_items")} | {"diffs": diffs}
    return out


def main():
    gpu = {p.stem: p.read_text().strip() for p in QA.glob("*gpu.txt")}
    Q = {n: load_audit(n) for n in ("qwen_bf16", "qwen_fp32", "qwen_vllm", "qwen_repo")}
    Lm = {n: load_audit(n) for n in ("llama_bf16", "llama_vllm")}
    res = {"gpus": gpu, "qwen": {}, "llama": {}}
    for fmt in ("native", "nothink"):
        res["qwen"][f"bf16_vs_fp32_{fmt}"] = compare(Q["qwen_bf16"], Q["qwen_fp32"], fmt)
        res["qwen"][f"bf16_vs_vllm_{fmt}"] = compare(Q["qwen_bf16"], Q["qwen_vllm"], fmt)
        res["qwen"][f"fp32_vs_vllm_{fmt}"] = compare(Q["qwen_fp32"], Q["qwen_vllm"], fmt)
    res["llama"]["bf16_vs_vllm_native"] = compare(Lm["llama_bf16"], Lm["llama_vllm"], "native")
    res["attention_agreement"] = attention_blocks()
    res["lexsem_batched_smoke"] = {p.name: json.loads(p.read_text()).get("max_abs_logodds_diff_vs_smallest_batch")
                                   for p in (DATA / "text-wm-lexsem/runs/lexsem/smoke1_batched").glob("*.json")}
    dump("precision_attention", res)
    print("gpus", gpu)
    for m in ("qwen", "llama"):
        for k, v in res[m].items():
            if not v:
                print(m, k, "none"); continue
            for field, x in v.items():
                p, c = x["per_score"], x["per_item_contrast"]
                print(f"{m} {k} {field}: n {x['n_items']} per-score mean|d| {p['mean_abs']:.3f} sd {p['sd']:.3f} max {p['max_abs']:.2f} "
                      f"bias al {x['bias_aligned']:+.3f} sw {x['bias_swapped']:+.3f} | effect {x['effect_a']:.3f}->{x['effect_b']:.3f} "
                      f"shift {x['shift']:+.3f} (se {x['shift_se']:.3f}) contrast sd {c['sd']:.3f} r {x['item_r']:.4f} "
                      f"ratio {x['independence_of_arms']['ratio_var_contrast_to_sum']:.2f}")
    for k, v in res["attention_agreement"].items():
        print(k, {kk: v[kk] for kk in ("ok", "yes_no_mean_abs_diff", "two_candidate_mean_abs_diff", "max_abs_diff")},
              [round(x["diff_sdpa_minus_eager"], 3) for x in v["diffs"]])
    print(res["lexsem_batched_smoke"])


if __name__ == "__main__":
    main()
