"""Tokenizer-only check of the controlled environment's prompts (notes/bayesenv_design.md, section 6), run before
the freeze: every prompt type, framing, readout and space of the smoke and full designs is encoded with the scorer's
`encode` for Qwen3-8B (plain text) and Llama-3.1-8B (chat template), so that a candidate that is not one token, or a
joint tokenization that does not extend the prompt by exactly that token, is found before any score exists. Writes
the candidate tokens and prompt-length statistics. No model weights are loaded.

    python scripts/bayesenv_tokcheck.py --out runs/bayesenv_tokcheck.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from wf.bayesenv import design as D
from wf.bayesenv.score import encode

MODELS = (("Qwen/Qwen3-8B", "plain"), ("meta-llama/Llama-3.1-8B-Instruct", "chat"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=7, help="check every n-th unit of the full design (all smoke units)")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from wf.constants import revision_for

    units = D.smoke_units() + D.full_units(D.Design(w_goal=8, w_tag=8))[:: args.every]
    report = {}
    for model, mode in MODELS:
        tok = AutoTokenizer.from_pretrained(model, revision=revision_for(model), trust_remote_code=True)
        lengths: dict = defaultdict(list)
        cands: dict = {}
        failures = []
        for u in units:
            for t, fr, r in u.sequences():
                try:
                    ids, c = encode(tok, D.prompt_for(u, t, fr, r), mode)
                except SystemExit as e:
                    failures.append(f"{u.uid} {t} {fr} {r}: {e}")
                    continue
                key = f"{u.space}|{fr}|{r}"
                lengths[key].append(len(ids))
                cands.setdefault(key, [tok.decode([c[0]]), tok.decode([c[1]]), c])
        report[model] = {"mode": mode, "failures": failures[:20], "n_failures": len(failures),
                         "candidates": cands,
                         "length": {k: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v), "n": len(v)}
                                    for k, v in lengths.items()}}
        print(model, "failures", len(failures), json.dumps(report[model]["length"]), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
