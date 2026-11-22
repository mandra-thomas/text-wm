"""Token check for the τ-bench authorization experiment (notes/tau_auth_design.md §8); tokenizers only, no weights.

    python -m wf.tau.auth_token_check <items.jsonl> <fewshot.json> [--models ...] [--config tau_auth_score] [--out f]

It renders every prompt that the scoring config scores (every item, arm, action, readout, task line and
comprehension question) with each model's chat template at the pinned revision, and passes a model only if:
- '{"', 'Error', 'Yes' and 'No' are single tokens;
- '{"' opens every success twin and witness output, and 'Error' every failure reply, witness failure and the
  authorization error;
- the chat template's generation prompt ends in a newline;
- at the prompt/answer boundary, tokenising prompt and answer separately (as the scorer does) gives the same ids as
  tokenising the concatenated text: the first-token prompts with the action's success and failure replies and the
  authorization error, and the yes/no and comprehension prompts with "Yes" and "No";
- within an arm, the prompts of a job's task lines differ only in the task line;
- the scoring run's own prompt checks (wf.tau.auth.check_prompts) report no problem.
It also reports prompt lengths in tokens (median and maximum) per arm, action and readout kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from wf.constants import REVISIONS
from wf.runs import load_config
from wf.tau import auth as A
from wf.tau import prompts as PR

MODELS = ("Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct")


def chosen(items_path: Path, fewshot_path: Path, cfg: dict) -> tuple[list[dict], set[str], list[dict]]:
    rows = [json.loads(x) for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    items = A.auth_items(rows, cfg["items_per_class"])
    return items, A.comprehension_ids(items, cfg["comprehension_per_class"]), json.loads(fewshot_path.read_text(encoding="utf-8"))


def check_model(name: str, items: list[dict], comp: set[str], fewshot: list[dict], cache_dir: str | None = None) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=REVISIONS.get(name), cache_dir=cache_dir)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    render = lambda ms: tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    singles = {s: enc(s) for s in ('{"', "Error", "Yes", "No", ' {"', " Error", " Yes", " No")}
    gen_tail = render([{"role": "user", "content": "x"}])[-24:]
    bad: dict[str, list] = {"first_token": [], "boundary": [], "only_task_line": []}
    lengths: dict[str, list[int]] = {}
    longest = [0]
    if enc(A.AUTH_ERROR)[:1] != singles["Error"]:
        bad["first_token"].append(("auth_error", A.AUTH_ERROR))
    n_prompts = 0
    for it in items:
        for action in A.ACTIONS:
            plus, minus = ((it["success_twin"], it["failure_reply"]) if action == "fail" else (it["witness_output"], it["witness_failure"]))
            if enc(plus)[:1] != singles['{"'] or enc(minus)[:1] != singles["Error"]:
                bad["first_token"].append((it["id"], action))
            readouts = ["first", "yn"] + (list(A.COMPREHENSION) if action == "fail" and it["id"] in comp else [])
            for readout in readouts:
                lines = A.FIRST_LINES if readout == "first" else A.YN_LINES if readout == "yn" else ("neutral",)
                for arm in A.ARMS:
                    rendered = {c: render(A.build_messages(it, c, fewshot, readout=readout, action=action, arm=arm)) for c in lines}
                    n_prompts += len(rendered)
                    base = rendered[lines[0]].replace(f"Task: {PR.task_for(it, lines[0])}\n", "Task: <X>\n", 1)
                    for c in lines[1:]:
                        if rendered[c].replace(f"Task: {PR.task_for(it, c)}\n", "Task: <X>\n", 1) != base:
                            bad["only_task_line"].append((it["id"], arm, action, readout, c))
                    ids = enc(rendered[lines[0]])
                    kind = "comp" if readout in A.COMPREHENSION else readout
                    lengths.setdefault(f"{arm}/{action}/{kind}", []).append(len(ids))
                    if len(ids) > longest[0]:
                        longest[:] = [len(ids), it["id"], arm, action, readout]
                    answers = (plus, minus, A.AUTH_ERROR) if readout == "first" else ("Yes", "No")
                    for ans in answers:
                        if enc(rendered[lines[0]] + ans) != ids + enc(ans):
                            bad["boundary"].append((it["id"], arm, action, readout, ans[:12]))
    single_ok = all(len(singles[s]) == 1 for s in ('{"', "Error", "Yes", "No"))
    passed = single_ok and gen_tail.endswith("\n") and not any(bad.values())
    return {"model": name, "revision": REVISIONS.get(name), "single_token_ids": singles,
            "single_tokens": {s: tok.convert_ids_to_tokens(v) for s, v in singles.items()},
            "auth_error_ids": enc(A.AUTH_ERROR)[:4], "generation_prompt_tail": gen_tail, "items": len(items),
            "comprehension_items": len(comp), "prompts_rendered": n_prompts,
            "problems": {k: len(v) for k, v in bad.items()}, "examples": {k: v[:5] for k, v in bad.items()},
            "prompt_tokens": {k: {"median": statistics.median(v), "max": max(v), "n": len(v)} for k, v in sorted(lengths.items())},
            "longest_prompt": longest, "model_max_length": getattr(tok, "model_max_length", None), "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path)
    ap.add_argument("fewshot", type=Path)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--config", default="tau_auth_score")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    items, comp, fewshot = chosen(args.items, args.fewshot, cfg)
    prompt_problems = A.check_prompts(items, comp, fewshot)
    out = {"config": args.config, "items": len(items), "comprehension_items": len(comp),
           "prompt_check_problems": prompt_problems[:20], "n_prompt_check_problems": len(prompt_problems),
           "system_message_sha256": {arm: hashlib.sha256(A.system_message(arm).encode()).hexdigest() for arm in A.ARMS},
           "models": [check_model(m, items, comp, fewshot, args.cache_dir) for m in args.models]}
    out["passed"] = not prompt_problems and all(r["passed"] for r in out["models"])
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
