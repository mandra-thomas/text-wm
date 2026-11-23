"""Exploratory follow-up: the effect with and without the empty think block on the thinking probe's own items.

The thinking-mode result (4.01 nats with reasoning against 6.06 with thinking off, on 1,000 items) compares an answer
read after the model's reasoning with one read after the empty think block. This script rebuilds that probe's item
sample exactly as wf.thinking does (rq1x's primary set, wf.thinking.sample_items, 1,000 items) and reports the stored
with-block effect and the follow-up's no-block effect on those items, with game-clustered intervals. CPU only.

    python scripts/qwen_audit/thinking_sample.py --dir runs/qwen_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from followup_analysis import TAG8, describe, load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/qwen_audit")
    args = ap.parse_args()
    from wf import rq1x
    from wf.thinking import sample_items

    d = Path(args.dir)
    items = {r["id"]: r for r in load_jsonl(d / "items_all.jsonl")}
    fu = {r["id"]: r for f in sorted(d.glob("fu_8b_shard*.jsonl")) for r in load_jsonl(f)}
    primary = [r["transition"] for r in items.values() if rq1x.desc_ok(r["transition"])]
    sample = sample_items(primary, 1000)
    pairs = [(items[t["id"]]["stored"][TAG8]["aligned"]["L"] - items[t["id"]]["stored"][TAG8]["swapped"]["L"],
              fu[t["id"]]["scores"]["noblock/aligned"]["L"] - fu[t["id"]]["scores"]["noblock/swapped"]["L"]) for t in sample]
    out = {"n_rq1x_primary": len(primary), **describe(pairs, [t["game_id"] for t in sample])}
    (d / "followup_thinking_sample.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
