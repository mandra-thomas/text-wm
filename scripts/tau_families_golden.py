#!/usr/bin/env python3
"""Golden rendered prompts of the τ-bench wave-A families extension (tests/golden/tau_families_rendered.json).

The golden file holds one real primary item of the pinned build, its four few-shot examples, and the sha256 of the
messages of every scored job under each model's handling (tokenizer-free; tests/test_tau_families.py checks them
everywhere). This script adds, or checks, the sha256 and token count of each job's rendered prompt under each pinned
tokenizer (Gemma 2 for the record of its length). It needs transformers and the cached tokenizers (ICE, offline):

    python scripts/tau_families_golden.py --write   # record
    python scripts/tau_families_golden.py           # check; exit 1 on any difference
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "tests" / "golden" / "tau_families_rendered.json"
TAGS = ("mistral-7b-v0.3", "llama3.2-3b", "mistral-7b-v0.3-firstturn", "gemma2-9b-it")


def rendered(tag: str, item: dict, fewshot: list[dict]) -> dict[str, dict]:
    from transformers import AutoTokenizer

    from wf.tau import families as T

    spec = T.spec_for(tag, allow_excluded=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)
    kw = {"enable_thinking": False, **spec.template_kwargs}
    jobs = T.PLACEMENT_JOBS if tag.endswith(T.FIRST_TURN) else T.PROTOCOL_JOBS
    out = {}
    for _, action, readout, cond in jobs:
        text = tok.apply_chat_template(T.messages(spec, item, cond, fewshot, readout, action), tokenize=False,
                                       add_generation_prompt=True, **kw)
        out[f"{action}/{readout}/{cond}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                             "tokens": len(tok(text, add_special_tokens=False).input_ids)}
    return out


def main() -> int:
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = {tag: rendered(tag, g["item"], g["fewshot"]) for tag in TAGS}
    if "--write" in sys.argv:
        g["rendered"] = now
        GOLDEN.write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({t: {k: v["tokens"] for k, v in r.items()} for t, r in now.items()}, indent=1))
        return 0
    diff = [(t, k) for t in TAGS for k in now[t] if g.get("rendered", {}).get(t, {}).get(k) != now[t][k]]
    print("golden rendered prompts:", "identical" if not diff else f"{len(diff)} differ: {diff[:5]}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
