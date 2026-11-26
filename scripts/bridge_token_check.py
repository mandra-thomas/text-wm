"""CPU check of every prompt the natural bridge scores (notes/bridge_design.md §12), with the real tokenizers.

Runs wf.bridge.token_checks on the whole sample for each model and format (the same checks every scoring run repeats
before it scores), and records prompt lengths and the shared-prefix length per group, for the cost estimate. Loads
tokenizers only, at the pinned revisions.

    python scripts/bridge_token_check.py --out runs/bridge_token_check.json
"""
from __future__ import annotations

import argparse
import json
import statistics

from wf import constants as C
from wf.bridge import Renderer, build_sequences, item_sets, jobs_for, token_checks
from wf.rq2 import load_items, primary_set
from wf.scoring.logodds import LogoddsScorer
from wf.scoring.prefix import common_prefix_len

FORMATS = {"Qwen/Qwen3-8B": ("plain", "block"), "meta-llama/Llama-3.1-8B-Instruct": ("chat",)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    sets = item_sets(primary_set(load_items()))
    sub = {t["id"] for t in sets["subset"]}
    report = {"n_sample": len(sets["sample"]), "n_subset": len(sub)}
    for model, fmts in FORMATS.items():
        tok = AutoTokenizer.from_pretrained(model, revision=C.REVISIONS[model])
        sc = LogoddsScorer.for_tokenizer(model, tok)
        for fmt in fmts:
            items = sets["sample"] if fmt != "block" else sets["subset"]
            r = Renderer(sc, fmt)
            chk = token_checks(items, fmt, r, sub)
            stems, rest, n_tok = [], [], []
            for t in items[:200]:
                for _, entries in build_sequences(t, jobs_for(t, t["id"] in sub), r).items():
                    flat = [s for _, seqs in entries for s in seqs]
                    s = common_prefix_len([p for p, _, _ in flat])
                    stems.append(s)
                    rest += [len(p) + len(c) - s + 1 for p, c, _ in flat]
                    n_tok += [len(p) for p, _, _ in flat]
            chk["first_200_items"] = {"stem_tokens_median": statistics.median(stems), "stem_tokens_max": max(stems),
                                      "remainder_tokens_median": statistics.median(rest),
                                      "remainder_tokens_max": max(rest), "prompt_tokens_median": statistics.median(n_tok),
                                      "prompt_tokens_max": max(n_tok), "sequences_per_item": len(rest) / 200}
            yes, no = r.yes_no()
            chk["yes_no_ids"] = [yes, no, tok.decode([yes]), tok.decode([no])]
            report[f"{model}/{fmt}"] = chk
            print(model, fmt, json.dumps({k: v for k, v in chk.items() if k != "problems"}), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
