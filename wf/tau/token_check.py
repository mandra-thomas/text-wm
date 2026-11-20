"""Token checks for the τ-bench readouts (notes/tau_design.md §6.1, amended 2026-11-20); tokenizers only, no weights.

    python -m wf.tau.token_check <items.jsonl> <fewshot.json> [--natural natural_items.jsonl] [--models <name or dir> ...]

Run on every scored item (the main population and the natural set). For each model it reports, and passes only if
all hold:
- '{"', 'Error', 'Yes' and 'No' are single tokens without a leading space;
- '{"' opens every tokenised success twin and witness output, and 'Error' every failure reply and witness failure;
- the chat template's generation prompt ends in a newline (Qwen3, thinking off: '</think>\\n\\n');
- at the prompt/answer boundary, tokenising prompt and answer separately (as the scorer does) gives the same ids as
  tokenising the concatenated text, for the failing call's and the witness's replies;
- the rendered prompts of every task line of an item differ only in the task line, for every scored variant
  (failing call and witness; output cue, yes/no and manipulation question; main, rules-only and ALFWorld-matched
  arm), each item with its own few-shot order (reversed for about half of the items; amendment after review 2).
It also reports prompt lengths in tokens (median and maximum) of the output-cue prompts per arm and action, and the
longest prompt over every rendered variant (with its item, arm, action, cue and task line) and that prompt plus the
longer of its two candidate replies.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from wf.tau.prompts import ALL_CONDITIONS, ARMS, CUES, build_messages

MODELS = ("Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct")


def check_model(name: str, items: list[dict], fewshot: list[dict], cache_dir: str | None) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    render = lambda ms: tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    singles = {s: enc(s) for s in ('{"', "Error", "Yes", "No", ' {"', " Error", " Yes", " No")}
    gen_tail = render([{"role": "user", "content": "x"}])[-24:]
    bad = {"first_token": [], "boundary": [], "only_task_line": []}
    lengths: dict[str, list[int]] = {}
    longest, longest_total = [0], [0]
    for it in items:
        actions = ["fail"] + (["witness"] if it.get("aligned_witness") else [])
        conds = [c for c in ALL_CONDITIONS if it.get(f"task_{c}")]
        for action in actions:
            plus, minus = ((it["success_twin"], it["failure_reply"]) if action == "fail" else (it["witness_output"], it["witness_failure"]))
            p_ids, m_ids = enc(plus), enc(minus)
            if p_ids[:1] != singles['{"'] or m_ids[:1] != singles["Error"]:
                bad["first_token"].append((it["id"], action))
            for arm in ARMS:
                for cue in CUES:
                    rendered = {c: render(build_messages(it, c, fewshot, cue=cue, action=action, arm=arm)) for c in conds}
                    for c in conds:
                        n_tok = len(enc(rendered[c]))
                        if n_tok > longest[0]:
                            longest[:] = [n_tok, it["id"], arm, action, cue, c]
                        if n_tok + max(len(p_ids), len(m_ids)) > longest_total[0]:
                            longest_total[:] = [n_tok + max(len(p_ids), len(m_ids)), it["id"], arm, action, cue, c]
                    base = rendered["aligned"].replace(f"Task: {it['task_aligned']}\n", "Task: <X>\n", 1)
                    for c in conds[1:]:
                        if rendered[c].replace(f"Task: {it[f'task_{c}']}\n", "Task: <X>\n", 1) != base:
                            bad["only_task_line"].append((it["id"], arm, action, cue, c))
                    if cue != "output":  # the questions add a fixed number of tokens to the output-cue prompt
                        continue
                    ids = enc(rendered["aligned"])
                    lengths.setdefault(f"{arm}/{action}/{cue}", []).append(len(ids))
                    if arm == "main":
                        if enc(rendered["aligned"] + plus) != ids + p_ids or enc(rendered["aligned"] + minus) != ids + m_ids:
                            bad["boundary"].append((it["id"], action))
    single_ok = all(len(singles[s]) == 1 for s in ('{"', "Error", "Yes", "No"))
    passed = single_ok and gen_tail.endswith("\n") and not any(bad.values())
    return {"model": name, "single_token_ids": singles,
            "single_tokens": {s: tok.convert_ids_to_tokens(v) for s, v in singles.items()},
            "generation_prompt_tail": gen_tail, "items": len(items),
            "problems": {k: len(v) for k, v in bad.items()}, "examples": {k: v[:5] for k, v in bad.items()},
            "prompt_tokens": {k: {"median": statistics.median(v), "max": max(v), "n": len(v)} for k, v in sorted(lengths.items())},
            "longest_prompt": longest, "longest_prompt_plus_reply": longest_total,
            "model_max_length": getattr(tok, "model_max_length", None),
            "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path)
    ap.add_argument("fewshot", type=Path)
    ap.add_argument("--natural", type=Path)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--n", type=int, default=0, help="check only every k-th item so that about n are checked (0: all)")
    ap.add_argument("--cache-dir")
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("main_pool")]
    if args.natural:
        rows += [json.loads(x) for x in args.natural.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.n:
        rows = rows[:: max(1, len(rows) // args.n)][: args.n]
    fewshot = json.loads(args.fewshot.read_text(encoding="utf-8"))
    out = [check_model(m, rows, fewshot, args.cache_dir) for m in args.models]
    print(json.dumps(out, indent=1))
    return 0 if all(r["passed"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
