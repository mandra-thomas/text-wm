"""Exploratory pipeline audit of Qwen3-8B's goal effect: draw the seeded item sample (CPU, read-only on the source).

Reads the stored confirmatory scores (`rq2_score`, 4 shards) and the collected items they were scored on, from the
checkout that produced them (notes/reproducing.md: fix11, commit ddeabc2), and writes one line per sampled item:
the item as collected plus every stored score for it. Nothing in the source checkout is written.

The population is the set of items that carry aligned and swapped scores for both co-primary models, i.e. RQ2's
4,666 primary items; the script stops if the count differs. The sample is drawn with Python's random.Random,
seeded with the study seed and a label, from the sorted item ids, so it does not depend on file order.

    python scripts/qwen_audit/sample_items.py --src /storage/ice1/2/1/gmat3/text-wm-fix11/runs \
        --out runs/qwen_audit/items.jsonl --n 200
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260924  # wf.constants.RNG_SEED
MODELS = ("qwen3-8b-fewshot", "llama3.1-8b-fewshot")
EXPECTED_PRIMARY = 4666


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="runs/ folder of the checkout that produced rq2_score")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    src = Path(args.src)

    stored: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    shard_of: dict[str, int] = {}
    shas = set()
    for i in range(4):
        d = src / "rq2_score" / f"shard_{i:02d}_of_04"
        shas.add((d / "git_sha").read_text().strip())
        for line in (d / "scoring.jsonl").open():
            s = json.loads(line)
            stored[s["transition_id"]][s["model"]][s["condition"]] = s
            shard_of[s["transition_id"]] = i
    primary = sorted(t for t, by in stored.items()
                     if all(c in by.get(m, {}) for m in MODELS for c in ("aligned", "swapped")))
    if len(primary) != EXPECTED_PRIMARY:
        raise SystemExit(f"expected {EXPECTED_PRIMARY} primary items, found {len(primary)}")
    sample = random.Random(f"{SEED}|qwen_audit_pipeline").sample(primary, args.n)
    want = set(sample)

    items: dict[str, dict] = {}
    for k in range(3):
        for line in (src / f"rq12_transitions_s{k}" / "transitions.jsonl").open():
            t = json.loads(line)
            if t["id"] in want:
                items[t["id"]] = t
    missing = want - set(items)
    if missing:
        raise SystemExit(f"items not found in the collection: {sorted(missing)[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for tid in sample:  # sample order; the first 50 are the Llama control's items
            f.write(json.dumps({"id": tid, "shard": shard_of[tid], "transition": items[tid],
                                "stored": {m: dict(stored[tid][m]) for m in MODELS}}) + "\n")

    # The full-population effect from the stored scores, as a check against the reported 6.12 and 1.65.
    for m in MODELS:
        d = [stored[t][m]["aligned"]["L"] - stored[t][m]["swapped"]["L"] for t in primary]
        ds = [stored[t][m]["aligned"]["L"] - stored[t][m]["swapped"]["L"] for t in sample]
        print(f"{m}: stored mean dL over {len(d)} primary items = {sum(d) / len(d):.3f}; "
              f"over the {len(ds)}-item sample = {sum(ds) / len(ds):.3f}")
    print(f"source commits: {sorted(shas)}; wrote {len(sample)} items to {out}")


if __name__ == "__main__":
    main()
