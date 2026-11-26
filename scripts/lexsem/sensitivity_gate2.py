"""Lexsem, post-freeze sensitivity requested on 2026-11-26 (not part of the frozen readings): the frozen analysis
rerun unchanged except that gate 2's first-token mass criterion is switched off, so every condition is read on the
normalized first token instead of falling back to log p(" You"). Comparing its output with the frozen
`analysis.json` shows whether any first-token reading depends on the fallback. Same data, bootstrap size and seed.

    python scripts/lexsem/sensitivity_gate2.py --dir runs/lexsem --out runs/lexsem/analysis_gate2_off.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    from wf import lexsem
    from wf import lexsem_analysis as LA

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-commit")
    a = ap.parse_args()
    LA.MASS_FIRST, LA.MASS_FIRST_GAP = 0.0, math.inf  # gate 2 off for the first token; yes/no gate unchanged
    d = Path(a.dir)
    extra = {n: str(d / f"{n}.jsonl") for n in lexsem.ARMS if n not in ("qwen_plain", "qwen_block", "llama_chat")}
    floors = {n: str(d / f"{n}_floor.jsonl") for n in lexsem.ARMS}
    rep = LA.run(str(d / "qwen_plain.jsonl"), a.out, block_path=str(d / "qwen_block.jsonl"),
                 llama_path=str(d / "llama_chat.jsonl"), B=LA.B_DEFAULT, expect_commit=a.expect_commit,
                 extra_arms=extra, floor_paths=floors)
    print(json.dumps(rep["qwen_plain"]["readings_by_readout"], indent=1))


if __name__ == "__main__":
    main()
