"""Golden rendered prompts: one real item through each model's own chat template (design section 12, B7).

    python -m wf.textcraft.golden_chat make <qwen tokenizer dir> <llama tokenizer dir>   # on ICE; writes the file
    python -m wf.textcraft.golden_chat check <tag> <tokenizer dir>                        # compares a tokenizer with it

The golden item is the first main item of the items file. For it, the aligned and the yes/no aligned prompts are
rendered with apply_chat_template(add_generation_prompt=True, enable_thinking=False), as LogoddsScorer does, and the
text, the number of ids and the sha256 of the ids are stored with the tokenizer's snapshot. The prefix arms share
the aligned prompt. tests/test_textcraft_review2.py checks the stored messages and the structure of the text
without a tokenizer; the smoke run (wf/textcraft/score.py) re-renders with the real tokenizer and compares.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from wf.textcraft import prompts as P
from wf.textcraft import tokcheck as T

GOLDEN = Path(__file__).resolve().parent / "data" / "chat_golden.json"
ARMS = ("aligned", "yn_aligned")
TAGS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")


def golden_item(items: list[dict]) -> dict:
    return next(t for t in items if t["population"] == "main")


def render_one(tok, item: dict, shots) -> dict:
    out = {}
    for arm in ARMS:
        text = T.render(tok, P.messages(item, arm, shots))
        ids = T.chat_ids(tok, P.messages(item, arm, shots))
        out[arm] = {"text": text, "n_ids": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    return out


def problems(tok, tag: str, items: list[dict], shots, golden: dict | None = None) -> list[str]:
    """Where this tokenizer's rendering of the golden item differs from the stored one."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if golden is None else golden
    item = next((t for t in items if t["id"] == golden["item_id"]), None)
    if item is None:
        return [f"golden item {golden['item_id']} is not in the items file"]
    now = render_one(tok, item, shots)
    return [f"{tag} {arm}: rendering differs from the golden" for arm in ARMS if now[arm] != golden["models"][tag][arm]]


def _load(path: Path | None = None) -> list[dict]:
    path = path or P.FEWSHOT.parent / "items.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    argv = sys.argv[1:] if argv is None else argv
    items, shots = _load(), P.fewshot_examples()
    if argv[0] == "make":
        item = golden_item(items)
        models = {}
        for tag, path in zip(TAGS, argv[1:3]):
            models[tag] = {"snapshot": Path(path).name, **render_one(AutoTokenizer.from_pretrained(path), item, shots)}
        GOLDEN.write_text(json.dumps({"item_id": item["id"], "arms": list(ARMS), "models": models}, indent=1,
                                     ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    found = problems(AutoTokenizer.from_pretrained(argv[2]), argv[1], items, shots)
    print(json.dumps({"problems": found}))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
