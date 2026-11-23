"""Reviewer (TextCraft round 3, review 1): where the 300 seeded environment-check records fall among round 2's four
scoring shards (records only; no score is read).

    PYTHONPATH=. python scripts/review_textcraft_r3_envcheck_shards.py
"""
from itertools import combinations
from pathlib import Path

from wf.textcraft.r2_score import load_records, shard_records
from wf.textcraft.r3_score import envcheck_subset

recs = load_records(Path("wf/textcraft/data/r2_items.jsonl"))
print("records", len(recs))
ids = {r["id"] for r in envcheck_subset(recs)}
c = []
for i in range(4):
    sh = shard_records(recs, i, 4)
    n = sum(r["id"] in ids for r in sh)
    mf = sum(r["id"] in ids and r["kind"] == "main" and r["population"] == "main" for r in sh)
    c.append(n)
    print("round-2 shard", i, "envcheck records", n, "main failures", mf)
print("best single shard", max(c), "best two", max(a + b for a, b in combinations(c, 2)),
      "every three:", {k: sum(c[j] for j in k) for k in combinations(range(4), 3)})
