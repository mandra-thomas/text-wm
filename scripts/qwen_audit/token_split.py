"""Exploratory pipeline audit: where in the two candidate texts the aligned-minus-swapped difference sits.

For each result file given, splits each item's effect into: the success text's first token ("You"), its second
token (the verb: " move", " pick", " clean", ...), its later tokens that spell the action's object (any token whose
stripped text is a substring of the object name, two characters or more), its other later tokens, and the
"Nothing happens." text (entering with a minus sign). The five parts sum to the item's effect. Only the independent scorer's files carry per-token log-probs.

    python scripts/qwen_audit/token_split.py runs/qwen_audit/items.jsonl runs/qwen_audit/qwen_fp32.jsonl
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys


def action_object(action: str) -> str:
    m = re.match(r"(?:take|move|heat|cool|clean|use|examine) (.+?) \d+", action)
    return m[1] if m else ""


def main() -> None:
    items = {json.loads(x)["id"]: json.loads(x) for x in open(sys.argv[1])}
    for path in sys.argv[2:]:
        fmts = sorted({k.split("/")[0] for r in map(json.loads, open(path)) for k in r["scores"]})
        for fmt in fmts:
            parts = {"first": [], "second": [], "object_tokens": [], "other_later": [], "nothing_text": []}
            for r in map(json.loads, open(path)):
                a, s = r["scores"].get(f"{fmt}/aligned"), r["scores"].get(f"{fmt}/swapped")
                if not a or "tok_lp_plus" not in a:
                    continue
                obj = action_object(items[r["id"]]["transition"]["action"])
                d = [x - y for x, y in zip(a["tok_lp_plus"], s["tok_lp_plus"])]
                is_obj = [k > 1 and len(tk.strip()) >= 2 and tk.strip() in obj for k, tk in enumerate(a["toks_plus"])]
                parts["first"].append(d[0])
                parts["second"].append(d[1] if len(d) > 1 else 0.0)
                parts["object_tokens"].append(sum(x for x, o in zip(d, is_obj) if o))
                parts["other_later"].append(sum(x for k, (x, o) in enumerate(zip(d, is_obj)) if k > 1 and not o))
                parts["nothing_text"].append(-(a["logp_minus"] - s["logp_minus"]))
            if parts["first"]:
                total = st.fmean(sum(v) for v in zip(*parts.values()))
                print(f"{path} {fmt}: n={len(parts['first'])} effect={total:.3f} "
                      + " ".join(f"{k}={st.fmean(v):.3f}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
