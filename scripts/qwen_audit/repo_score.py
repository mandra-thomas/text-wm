"""Exploratory pipeline audit: the study's own scorer on the audit sample, on the same GPU as the independent one.

Scores each sampled item with wf.scoring.logodds.LogoddsScorer.score_pair and build_messages, exactly as rq2_score
does (aligned and swapped, example set 1). Run in the same job as independent_score.py, it separates two causes of
a gap to the stored scores: a difference between the two implementations (which would show here, on one GPU) and
the GPU type (stored shards 0-2 ran on A40s, shard 3 on an L40S). Appends per item; a rerun skips items done.

    python scripts/qwen_audit/repo_score.py --items runs/qwen_audit/items.jsonl --model Qwen/Qwen3-8B \
        --out runs/qwen_audit/qwen_repo.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAGS = {"Qwen/Qwen3-8B": "qwen3-8b-fewshot", "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b-fewshot"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from wf.scoring.logodds import LogoddsScorer, build_messages

    items = [json.loads(x) for x in Path(args.items).open()][: args.limit]
    out = Path(args.out)
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()
    scorer = LogoddsScorer(args.model, model_tag=TAGS[args.model])
    for it in items:
        if it["id"] in done:
            continue
        t = it["transition"]
        rec = {"id": it["id"], "model": args.model, "scores": {}}
        for cond in ("aligned", "swapped"):
            r = scorer.score_pair(t["id"], cond, build_messages(t, cond), t["candidates"]["o_plus"],
                                  t["candidates"]["o_minus"])
            rec["scores"][f"native/{cond}"] = r.to_dict()
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    scorer.close()
    print(f"done: {len(items)} items", flush=True)


if __name__ == "__main__":
    main()
