"""Tokenization checks on the real prompts (notes/textcraft_design.md, section 12, B7; validation standard, step 7).

LogoddsScorer tokenizes the chat (apply_chat_template, enable_thinking=False) and each candidate on its own, then
joins the two id lists. For each item and arm this checks, with the model's own tokenizer:

1. the chat template's ids are the plain tokenization of its rendered text;
2. joined equals alone: tokenizing the rendered prompt followed by the candidate gives the prompt's ids followed by
   the candidate's ids, so the scored ids are the ones the text would have;
3. for the prefix-only arms, each prefix's ids are a leading part of its full reply's ids, alone and after the
   prompt, so that L splits exactly into the prefix part and the name-token remainder;
4. "Yes" and "No" are single tokens.

Needs only a Hugging Face tokenizer; the smoke run calls it before scoring (wf/textcraft/score.py), and on ICE it
runs without a GPU:  python -m wf.textcraft.tokcheck <tokenizer dir> [items.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wf.textcraft import prompts as P


def render(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ids(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False).input_ids)


def chat_ids(tok, messages: list[dict]) -> list[int]:
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    out = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
    return list(out[0] if out and isinstance(out[0], list) else out)


def item_problems(tok, item: dict, shots, arms: tuple[str, ...] = P.ARMS, encode=None) -> list[str]:
    """Every check above for one item; `encode` is the scorer's own chat encoder (messages -> ids) when given."""
    problems = []
    for arm in arms:
        messages = P.messages(item, arm, shots)
        text = render(tok, messages)
        prompt = ids(tok, text)
        if (encode(messages) if encode else chat_ids(tok, messages)) != prompt:
            problems.append(f"{arm}: chat-template ids differ from the rendered text's ids")
        for cand in P.candidates(item, arm):
            alone = ids(tok, cand)
            if ids(tok, text + cand) != prompt + alone:
                problems.append(f"{arm}: {cand!r} tokenizes differently after the prompt than alone")
        if arm in P.PREFIX_ARMS:
            for pre, full in zip(P.candidates(item, arm), P.candidates(item, arm.removeprefix("prefix_"))):
                a, b = ids(tok, pre), ids(tok, full)
                if b[: len(a)] != a:
                    problems.append(f"{arm}: {pre!r} is not a token prefix of {full!r}")
                ca, cb = ids(tok, text + pre)[len(prompt):], ids(tok, text + full)[len(prompt):]
                if cb[: len(ca)] != ca:
                    problems.append(f"{arm}: {pre!r} is not a token prefix of {full!r} after the prompt")
    for word in ("Yes", "No") if any(a in P.YN_ARMS for a in arms) else ():
        if len(ids(tok, word)) != 1:
            problems.append(f"{word!r} is not a single token")
    return problems


def check(tok, items: list[dict], shots, arms: tuple[str, ...] = P.ARMS, encode=None) -> dict:
    bad = {t["id"]: p for t in items if (p := item_problems(tok, t, shots, arms, encode))}
    return {"n_items": len(items), "arms": list(arms), "n_items_with_problems": len(bad),
            "examples": dict(list(bad.items())[:5]), "ok": not bad}


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    argv = sys.argv[1:] if argv is None else argv
    tok = AutoTokenizer.from_pretrained(argv[0])
    path = Path(argv[1]) if len(argv) > 1 else P.FEWSHOT.parent / "items.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = check(tok, items, P.fewshot_examples())
    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
