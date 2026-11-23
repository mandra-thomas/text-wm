"""Reviewer (TextCraft round 3, review 2): the environment-check plan on synthetic round-2 job records (GPU names only;
no score): all four shards on L40S, and a 2-2 split.

    PYTHONPATH=. python scripts/review2_textcraft_r3_plan.py
"""
import json
import tempfile
from pathlib import Path

from wf.textcraft import r3_gates as G
from wf.textcraft.r2_score import load_records

records = load_records(Path("wf/textcraft/data/r2_items.jsonl"))
for name, gpus in (("all L40S", ["NVIDIA L40S"] * 4), ("2-2 split", ["NVIDIA L40S", "NVIDIA A40"] * 2),
                   ("one shard resumed on another type", ["NVIDIA L40S"] * 3 + [None])):
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "textcraft_r2_score_qwen3-8b"
        for i, g in enumerate(gpus):
            d = run / f"shard_{i:02d}_of_04"
            d.mkdir(parents=True)
            jobs = [{"gpu": [g]}] if g else [{"gpu": ["NVIDIA L40S"]}, {"gpu": ["NVIDIA A40"]}]
            (d / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
        p = G.envcheck_plan(Path(tmp), "qwen3-8b", records)
        print(name, "->", {k: p[k] for k in ("single_type", "n_comparable_planned", "runnable")},
              [(j["gpu"], j["r2_shards"]) for j in p["jobs"]])
