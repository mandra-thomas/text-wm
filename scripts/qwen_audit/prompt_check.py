"""Prompt and tokenization checks for RQ2's aligned and swapped prompts (CPU; question 6 of the forensic audit).

Two modes:

  render  --out F.jsonl     Render every primary item's aligned and swapped messages with whatever `wf` is on
                            PYTHONPATH and write their SHA-256 per item. Run once with the code at the scoring commit
                            (ddeabc2, extracted with git archive) and once with the current code; `tokens` compares.
  tokens  --out F.json      With the current code: check on the strings that the two prompts differ only in the task
                            line; tokenize both with each pinned tokenizer (local snapshot folders, no download) and
                            check that the token difference is one contiguous span lying inside the task line;
                            tokenize the candidates as the scorer does; count the object name's tokens in the success
                            reply. With --old-render, also compare the two renders.

The collection is read through wf.rq2 (RUNS_DIR must hold rq12_transitions_s0..2). Nothing is written to any run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys


def _sha(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def render(out: str) -> None:
    from wf.rq2 import load_items, primary_set
    from wf.scoring.logodds import build_messages
    P = primary_set(load_items())
    with open(out, "w") as f:
        for t in P:
            f.write(json.dumps({"id": t["id"], "aligned": _sha(build_messages(t, "aligned")),
                                "swapped": _sha(build_messages(t, "swapped"))}) + "\n")
    print("rendered", len(P), "items to", out)


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def tokens(out: str, snapshots: dict[str, str], old_render: str | None, new_render: str | None) -> None:
    from transformers import AutoTokenizer
    from wf.rq2 import load_items, primary_set
    from wf.scoring.logodds import build_messages

    P = primary_set(load_items())
    res: dict = {"items": len(P), "snapshots": snapshots}
    # string level
    str_bad = collections.Counter()
    task_line_counts = collections.Counter()
    msgs = {}
    for t in P:
        a, s = build_messages(t, "aligned"), build_messages(t, "swapped")
        msgs[t["id"]] = (a, s)
        if len(a) != len(s) or any(x["role"] != y["role"] for x, y in zip(a, s)):
            str_bad["structure"] += 1
            continue
        diff = [k for k, (x, y) in enumerate(zip(a, s)) if x["content"] != y["content"]]
        if len(diff) != 1:
            str_bad[f"messages_differing={len(diff)}"] += 1
            continue
        xa, xs = a[diff[0]]["content"], s[diff[0]]["content"]
        n_task = xa.count(t["task_line"])
        task_line_counts[n_task] += 1
        if xa.replace(t["task_line"], t["swapped_task_line"]) != xs:
            str_bad["not_a_task_line_swap"] += 1
        if t["task_line"] == t["swapped_task_line"]:
            str_bad["identical_task_lines"] += 1
    res["string_check"] = {"problems": dict(str_bad), "aligned_task_line_occurrences_in_prompt": dict(task_line_counts)}
    per_item: dict = collections.defaultdict(dict)
    for key, path in snapshots.items():
        tok = AutoTokenizer.from_pretrained(path)
        bad = collections.Counter()
        span_len = collections.Counter()
        cand_first = collections.Counter()
        n_prompt = []
        for t in P:
            a, s = msgs[t["id"]]
            ia = tok.apply_chat_template(a, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            is_ = tok.apply_chat_template(s, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if hasattr(ia, "input_ids"):
                ia, is_ = ia["input_ids"], is_["input_ids"]
            n_prompt.append(len(ia))
            pre = _common_prefix(ia, is_)
            suf = _common_prefix(ia[::-1], is_[::-1])
            suf = min(suf, len(ia) - pre, len(is_) - pre)
            da, ds = tok.decode(ia[pre:len(ia) - suf]), tok.decode(is_[pre:len(is_) - suf])
            span_len[(len(ia) - pre - suf, len(is_) - pre - suf)] += 1
            if da.strip() and da.strip() not in t["task_line"]:
                bad["aligned_diff_outside_task_line"] += 1
            if ds.strip() and ds.strip() not in t["swapped_task_line"]:
                bad["swapped_diff_outside_task_line"] += 1
            if t["goal_object"] not in da and t["swapped_goal_object"] not in ds:
                bad["diff_misses_both_objects"] += 1
            # candidates, tokenized alone as the scorer does
            cp = tok(t["candidates"]["o_plus"], add_special_tokens=False)
            cm = tok(t["candidates"]["o_minus"], add_special_tokens=False)
            cand_first[(tok.decode(cp["input_ids"][:1]), tok.decode(cm["input_ids"][:1]), len(cm["input_ids"]))] += 1
            # object name tokens in the success reply: tokens overlapping the first " <obj> " after the first word
            text = t["candidates"]["o_plus"]
            k = text.find(" " + t["goal_object"] + " ")
            n_obj = None
            if k >= 0 and hasattr(tok, "__call__"):
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                lo, hi = k + 1, k + 1 + len(t["goal_object"])
                n_obj = sum(1 for (x, y) in enc["offset_mapping"] if x < hi and y > lo)
            per_item[t["id"]][f"{key}_obj_tokens"] = n_obj
            per_item[t["id"]][f"{key}_plus_tokens"] = len(cp["input_ids"])
            # does the candidate tokenize the same at the prompt boundary as alone? (prompt text + reply as one string)
            joint = tok(tok.decode(ia) + text, add_special_tokens=False)["input_ids"]
            if joint[len(ia):] != cp["input_ids"] or joint[: len(ia)] != list(ia):
                bad["candidate_retokenizes_differently_in_context"] += 1
        res[key] = {"problems": dict(bad),
                    "diff_span_token_lengths_aligned_swapped": {f"{x}/{y}": n for (x, y), n in span_len.most_common(12)},
                    "candidate_first_tokens_and_failure_length": {f"{a}|{b}|{c}": n for (a, b, c), n in cand_first.items()},
                    "prompt_tokens_median": sorted(n_prompt)[len(n_prompt) // 2]}
    if old_render and new_render:
        old = {json.loads(l)["id"]: json.loads(l) for l in open(old_render)}
        new = {json.loads(l)["id"]: json.loads(l) for l in open(new_render)}
        res["render_old_vs_new"] = {"items_old": len(old), "items_new": len(new),
                                    "aligned_identical": sum(old[i]["aligned"] == new[i]["aligned"] for i in new if i in old),
                                    "swapped_identical": sum(old[i]["swapped"] == new[i]["swapped"] for i in new if i in old)}
    res["per_item"] = per_item
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_item"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["render", "tokens"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--qwen", help="local snapshot folder of Qwen/Qwen3-8B at the pinned revision")
    ap.add_argument("--llama", help="local snapshot folder of meta-llama/Llama-3.1-8B-Instruct at the pinned revision")
    ap.add_argument("--old-render"); ap.add_argument("--new-render")
    a = ap.parse_args()
    if a.mode == "render":
        render(a.out)
    else:
        snaps = {k: v for k, v in (("qwen", a.qwen), ("llama", a.llama)) if v}
        tokens(a.out, snaps, a.old_render, a.new_render)


if __name__ == "__main__":
    sys.exit(main())
