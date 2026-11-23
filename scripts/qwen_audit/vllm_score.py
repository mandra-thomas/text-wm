"""Exploratory pipeline audit: the same scores through a second engine, vLLM's prompt_logprobs.

The prompt and candidate token ids are built exactly as in independent_score.py (rendered text, candidate appended,
tokenized once, prompt-prefix checked). vLLM then runs its own model code, attention kernels and batching, and
returns the log-probability of every prompt token; a candidate's score is the sum over its positions. Prefix caching
is off. Results are written once at the end (the whole run takes minutes).

    python scripts/qwen_audit/vllm_score.py --items runs/qwen_audit/items.jsonl --model Qwen/Qwen3-8B \
        --formats native,nothink --out runs/qwen_audit/qwen_vllm.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_score import QWEN_ONLY, render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--formats", default="native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from wf.constants import REVISIONS
    from wf.scoring.prompts import WM_FEWSHOT_SET0

    rev = REVISIONS[args.model]
    is_qwen = args.model.startswith("Qwen/")
    formats = [f for f in args.formats.split(",") if f and (is_qwen or f not in QWEN_ONLY)]
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    items = [json.loads(x) for x in Path(args.items).open()][: args.limit]

    jobs = []  # (item id, fmt, cond, which, full ids, prompt length)
    for it in items:
        t = it["transition"]
        for cond in ("aligned", "swapped"):
            for fmt in formats:
                prompt, sep = render(tok, fmt, t, cond, WM_FEWSHOT_SET0, is_qwen)
                p = ids(prompt)
                for which, cand in (("plus", t["candidates"]["o_plus"]), ("minus", t["candidates"]["o_minus"])):
                    f = ids(prompt + sep + cand)
                    if f[: len(p)] != p:
                        f = p + ids(sep + cand)
                    jobs.append((it["id"], fmt, cond, which, f, len(p)))

    llm = LLM(model=args.model, revision=rev, tokenizer_revision=rev, dtype="bfloat16", enable_prefix_caching=False,
              enforce_eager=True, max_model_len=8192, gpu_memory_utilization=0.85, seed=0)
    print(f"vllm {vllm.__version__}; {len(jobs)} sequences", flush=True)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": j[4]} for j in jobs],
                        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1), use_tqdm=False)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    res: dict[str, dict] = {}
    for (tid, fmt, cond, which, f, p), o in zip(jobs, outs):
        lps = [o.prompt_logprobs[i][f[i]].logprob for i in range(p, len(f))]
        s = res.setdefault(tid, {"id": tid, "model": args.model, "engine": f"vllm {vllm.__version__}", "scores": {}})
        d = s["scores"].setdefault(f"{fmt}/{cond}", {})
        d[f"logp_{which}"] = math.fsum(lps)
        d[f"first_{which}"] = lps[0]
        d[f"n_{which}"] = len(lps)
    with Path(args.out).open("w") as fh:
        for s in res.values():
            for d in s["scores"].values():
                d["L"] = d["logp_plus"] - d["logp_minus"]
                d["L_first"] = d["first_plus"] - d["first_minus"]
            fh.write(json.dumps(s) + "\n")
    print(f"wrote {len(res)} items to {args.out}", flush=True)


if __name__ == "__main__":
    main()
