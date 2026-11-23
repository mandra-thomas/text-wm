"""TextCraft round 3: the mocked dry run, end to end, on round 2's full records (notes/textcraft_r3_design.md, 13).

    WF_MOCK_SCORE=1 python scripts/textcraft_r3_dryrun.py <empty folder>

No model, no tokenizer, no GPU: every scorer is the mock scorer. In a fresh runs folder it scores round 2's five
models (round 2's own run, unchanged) and files round 2's analysis, then runs every round-3 run in its config's shards
(the core and extension models, both environment-check rescorings, both format checks and the thinking arm), then
the round-3 analysis, and prints the counts and the analysis's family sizes. The mock scores carry no effect; the run
shows that every block is written, every family has its fixed size and nothing is missing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    if os.environ.get("WF_MOCK_SCORE") != "1":
        raise SystemExit("set WF_MOCK_SCORE=1")
    out = Path(sys.argv[1]).resolve()
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=False)
    (runs / "FILED.txt").write_text("filed_date: dry run\n")
    os.environ["RUNS_DIR"] = str(runs)
    from wf.runs import load_config
    from wf.textcraft import r2 as R
    from wf.textcraft import r2_analysis as A
    from wf.textcraft import r2_score as S2
    from wf.textcraft import r3 as R3
    from wf.textcraft import r3_analysis as Z
    from wf.textcraft import r3_score as S
    from wf.textcraft import r3_think as K

    t0 = time.time()
    counts = {}
    for tag in R.TAGS:
        cfg = load_config(f"textcraft_r2_score_{tag}")
        for i in range(int(cfg.get("n_shards", 1))):
            assert S2.run_textcraft_r2_score({**cfg, "shard": i}) == 0
        counts[f"r2 {tag}"] = len(A.load_scores(tag, runs))
    A.run_textcraft_r2_analysis(load_config("textcraft_r2_analysis"))

    def shards(run: str, fn) -> None:
        cfg = load_config(run)
        for i in range(int(cfg.get("n_shards", 1))):
            assert fn({**cfg, "shard": i}) == 0, run

    for tag in list(R3.CORE) + list(R3.EXTENSION):
        shards(f"textcraft_r3_score_{tag}", S.run_textcraft_r3_score)
        counts[tag] = len(Z.scores_of(f"textcraft_r3_score_{tag}", tag, runs))
    for tag in R3.ENVCHECK_MODELS:
        shards(f"textcraft_r3_envcheck_{tag}", S.run_textcraft_r3_envcheck)
        counts[f"envcheck {tag}"] = len(Z.scores_of(f"textcraft_r3_envcheck_{tag}", tag + R3.ENVCHECK_SUFFIX, runs))
    shards("textcraft_r3_placement_gemma4-12b-it", S.run_textcraft_r3_format)
    shards("textcraft_r3_nopreserve_qwen3.8-27b", S.run_textcraft_r3_format)
    shards("textcraft_r3_think_qwen3.5-9b", K.run_textcraft_r3_think)
    counts["think"] = len(Z.scores_of("textcraft_r3_think_qwen3.5-9b", R3.THINK_TAG, runs))
    rc = Z.run_textcraft_r3_analysis(load_config("textcraft_r3_analysis"))
    m = json.loads((runs / "textcraft_r3_analysis" / "metrics.json").read_text())
    summary = {"analysis_exit": rc, "scores": counts, "missing": m["missing_scores"],
               "family_sizes": {f: m[f]["family_size"] for f in ("A3", "A3x", "B3")},
               "waiting": {f: m[f]["waiting"] for f in ("A3", "A3x")},
               "H3_P": m["A3"].get("H3_P", {}).get("overall"),
               "replication": {c: r["reading"] for c, r in m["A3"].get("replication", {}).items()},
               "environment_pass": m["environment_check"].get("pass"),
               "B3_pairs": sorted(m["B3"]["differences"]), "exploratory": sorted(m["exploratory"]),
               "minutes": round((time.time() - t0) / 60, 1)}
    (out / "dryrun_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
