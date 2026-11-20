"""Token checks for the τ-bench readouts (notes/tau_design.md §6.1); tokenizers only, no model weights.

    python -m wf.tau.token_check <items.jsonl> <fewshot.json> [--models Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct]

For each model it reports, and passes only if all hold:
- '{"' and 'Error' (and 'Yes', 'No') are single tokens without a leading space;
- they are the first tokens of the tokenised success twins and failure replies of a sample of items;
- the chat template's generation prompt ends in a newline (Qwen3, thinking off: '</think>\\n\\n'), so the answer's
  first token carries no leading space;
- at the prompt/answer boundary, tokenising prompt and answer separately (as the scorer does) gives the same ids
  as tokenising the concatenated text, so no token merges across the boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wf.tau.prompts import build_messages

MODELS = ("Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct")


def check_model(name: str, items: list[dict], fewshot: list[dict], cache_dir: str | None) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    singles = {s: enc(s) for s in ('{"', "Error", "Yes", "No", ' {"', " Error", " Yes", " No")}
    gen_tail = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)[-24:]
    first_ok, boundary_ok, n = 0, 0, 0
    examples = []
    for it in items:
        twin, fail = enc(it["success_twin"]), enc(it["failure_reply"])
        first_ok += twin[:1] == singles['{"'] and fail[:1] == singles["Error"]
        prompt = tok.apply_chat_template(build_messages(it, "aligned", fewshot), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        joint_twin = tok(prompt + it["success_twin"], add_special_tokens=False).input_ids
        joint_fail = tok(prompt + it["failure_reply"], add_special_tokens=False).input_ids
        boundary_ok += joint_twin == p_ids + twin and joint_fail == p_ids + fail
        n += 1
        if len(examples) < 2:
            examples.append({"id": it["id"], "prompt_tokens": len(p_ids), "twin_tokens": len(twin), "failure_tokens": len(fail),
                             "twin_first": tok.convert_ids_to_tokens(twin[:3]), "failure_first": tok.convert_ids_to_tokens(fail[:3])})
    single_ok = all(len(singles[s]) == 1 for s in ('{"', "Error", "Yes", "No"))
    passed = single_ok and first_ok == n and boundary_ok == n and gen_tail.endswith("\n")
    return {"model": name, "single_token_ids": singles,
            "single_tokens": {s: tok.convert_ids_to_tokens(v) for s, v in singles.items()},
            "generation_prompt_tail": gen_tail, "items": n, "first_tokens_ok": first_ok, "boundary_ok": boundary_ok,
            "examples": examples, "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path)
    ap.add_argument("fewshot", type=Path)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cache-dir")
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("main_pool")][:: max(1, len([r for r in rows if r.get("main_pool")]) // args.n)][: args.n]
    fewshot = json.loads(args.fewshot.read_text(encoding="utf-8"))
    out = [check_model(m, rows, fewshot, args.cache_dir) for m in args.models]
    print(json.dumps(out, indent=1))
    return 0 if all(r["passed"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
