"""Round 3, deviation D-G1: Gemma 4 E4B eager against SDPA attention on the smoke run's five attention-agreement items,
in bf16 (as the gate) and in float32, with the yes/no readout logits. Run from a checkout at the freeze tag with
node-local weights. Writes one JSON to argv[1]. No study quantity: only eager-minus-SDPA differences of single scores.
"""
import json
import math
import sys

import torch

from wf import families as FAM
from wf.scoring.logodds import LogoddsScorer
from wf.textcraft import prompts as P
from wf.textcraft import r3 as R3
from wf.textcraft.r2_score import load_records, verified_files
from wf.textcraft.r3_score import attention_agreement, score_one, smoke_subset

OUT = sys.argv[1]
TAG = "gemma4-e4b-it"
spec = R3.MODELS[TAG]
rpath, spath, _ = verified_files({"manifest": "wf/textcraft/data/r2_manifest.json"})
records = smoke_subset(load_records(rpath))  # the smoke run's one shard, in its order
shots = P.fewshot_examples(spath)
items = [r for r in records if r["kind"] == "main" and r["population"] == "main"][: FAM.ATTN_AGREEMENT["n_items"]]
ARMS = ("aligned", "yn_aligned")
res = {"items": [r["id"] for r in items], "criterion": {"yes_no_mean_abs_max": 0.01, "yes_no_any_abs_max": 0.05}}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)


def spacing(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


def open_scorer(dtype):
    if dtype == "bfloat16":
        return FAM.open_scorer(spec, TAG)
    return LogoddsScorer(spec.hf_id, model_tag=TAG, dtype=dtype, attn_implementation=spec.attn_implementation,
                         template_kwargs=spec.template_kwargs, model_class=spec.model_class,
                         tokenizer_kwargs=spec.tokenizer_kwargs, prompt_format=spec.prompt_format,
                         strict_load=spec.strict_load, unexpected_keys_allowed=spec.unexpected_keys_allowed)


def readout(sc):
    out = []
    dev = next(sc._model.parameters()).device
    y, n = [sc._encode_candidate(c)[0].tolist()[0] for c in ("Yes", "No")]
    for r in items:
        ids = sc._encode(R3.model_messages(spec, r, "yn_aligned", shots)).to(dev)
        with torch.no_grad():
            lg = sc._model(input_ids=ids).logits[0, -1]
        f = lg.float()
        out.append({"id": r["id"], "logits_dtype": str(lg.dtype), "logit_yes": float(f[y]), "logit_no": float(f[n]),
                    "bf16_step_at_yes": spacing(float(f[y])), "bf16_step_at_no": spacing(float(f[n])),
                    "top3": [round(v, 4) for v in torch.topk(f, 3).values.tolist()],
                    "max_abs_logit": float(f.abs().max())})
    return out


for dtype in ("bfloat16", "float32"):
    sc = open_scorer(dtype)
    block = {"dtype_loaded": str(sc._model.dtype), "attn_loaded": FAM._attn_loaded(sc._model)}
    block["gate_function"] = attention_agreement(sc, spec, records, shots)  # the gate's own computation
    per = {}
    ro = {}
    for attn in ("sdpa", "eager"):
        FAM._set_text_attn(sc._model, attn)
        sc._cache.clear()
        per[attn] = {f"{r['id']}|{a}": score_one(sc, spec, r, a, shots).L for r in items for a in ARMS}
        ro[attn] = readout(sc)
    FAM._set_text_attn(sc._model, "sdpa")
    sc._cache.clear()
    diffs = {k: per["eager"][k] - per["sdpa"][k] for k in per["sdpa"]}
    for fmt, keep in (("yes_no", lambda k: k.endswith("|yn_aligned")), ("two_candidate", lambda k: k.endswith("|aligned"))):
        v = [abs(x) for k, x in diffs.items() if keep(k)]
        block[f"{fmt}_mean_abs"] = sum(v) / len(v)
        block[f"{fmt}_max_abs"] = max(v)
    block["per_score_eager_minus_sdpa"] = diffs
    block["yes_no_readout"] = ro
    res[dtype] = block
    print(dtype, {k: block[k] for k in ("yes_no_mean_abs", "yes_no_max_abs", "two_candidate_mean_abs",
                                         "two_candidate_max_abs")}, flush=True)
    save()
    sc.close()
    del sc
    torch.cuda.empty_cache()

f32 = res["float32"]
res["criterion_met"] = f32["yes_no_mean_abs"] <= 0.01 and f32["yes_no_max_abs"] <= 0.05
print("criterion met (float32 yes/no mean <= 0.01, max <= 0.05):", res["criterion_met"], flush=True)
save()
