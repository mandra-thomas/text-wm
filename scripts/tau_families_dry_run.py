#!/usr/bin/env python3
"""Dry run of the τ-bench wave-A families chain without any model (notes/tau_families_design.md §10).

Mock scoring (WF_MOCK_SCORE=1) in a temporary runs folder, on the full item build (the natural trajectories are not
needed: the generated items and few-shot examples reproduce their pins without them; the natural file is empty here,
so the dry run pins its own hash for it). It runs every stage with the real configs and real shard counts:
build, verify, token check (skipped in mock: no tokenizer), smoke scoring and its analysis, both models' full scoring
in 4 shards, the placement check, a mock reference τ run (first-token and yes/no readouts only) with its τ analysis,
and the families analysis. It prints the job counts and the time of each stage. Usage:

    python scripts/tau_families_dry_run.py [--out summary.json]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    out_path = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    tmp = Path(tempfile.mkdtemp(prefix="tau_family_dry_"))
    for k in ("WF_MOCK_ENV", "WF_MOCK_AGENT", "WF_MOCK_SCORE", "WF_MOCK_CONTINUE"):
        os.environ[k] = "1"
    os.environ["RUNS_DIR"] = str(tmp)
    (tmp / "FILED.txt").write_text("filed_date: dry run\n")
    from wf.runs import load_config
    from wf.tau import families as T
    from wf.tau import pipeline as P
    from wf.util import file_sha, read_jsonl

    summary: dict = {"runs_dir": str(tmp), "stages": {}}

    def stage(name, fn):
        t0 = time.perf_counter()
        code = fn()
        summary["stages"][name] = {"exit": code, "seconds": round(time.perf_counter() - t0, 1)}
        print(f"{name}: exit {code} in {summary['stages'][name]['seconds']} s", flush=True)
        return code

    stage("tau_build", lambda: P.run_tau_build({"run_id": "tau_build", "natural_required": False}))
    pins = {n: file_sha(tmp / "tau_build" / n) for n in P.OUTPUT_FILES}
    real = load_config("tau_score")["expected_outputs_sha256"]
    summary["pins_equal_to_the_real_build"] = {n: pins[n] == real[n] for n in P.OUTPUT_FILES}
    stage("tau_verify", lambda: P.run_tau_verify({"run_id": "tau_verify", "fresh_sample": 10}))
    cfg = lambda name, **kw: {**load_config(name), "expected_outputs_sha256": pins, **kw}
    stage("tau_family_token_check", lambda: T.run_tau_family_token_check(cfg("tau_family_token_check")))
    stage("tau_family_score_smoke", lambda: T.run_tau_family_score(cfg("tau_family_score_smoke")))
    stage("tau_family_smoke_analysis", lambda: T.run_tau_family_analysis(load_config("tau_family_smoke_analysis")))
    for tag in T.SCORED:
        for i in range(4):
            os.environ["WF_SHARD"] = str(i)
            stage(f"tau_family_score_{tag} shard {i}", lambda: T.run_tau_family_score(cfg(f"tau_family_score_{tag}")))
    os.environ.pop("WF_SHARD")
    os.environ["WF_SHARD"] = "0"
    stage("resume of a finished shard", lambda: T.run_tau_family_score(cfg("tau_family_score_llama3.2-3b", resume=True)))
    os.environ.pop("WF_SHARD")
    stage("tau_family_placement", lambda: T.run_tau_family_score(cfg("tau_family_placement_mistral-7b-v0.3")))
    ref = {**load_config("tau_score"), "run_id": "tau_score_reference", "expected_outputs_sha256": pins, "n_shards": 4,
           "readouts": ["first", "yn"], "rules_arm_items": 0, "matched_arm_items": 0, "score_natural": False}
    for i in range(4):
        os.environ["WF_SHARD"] = str(i)
        stage(f"reference tau_score shard {i}", lambda: P.run_tau_score(ref))
    os.environ.pop("WF_SHARD")
    stage("reference tau_analysis (full replies are missing by design here)",
          lambda: P.run_tau_analysis({**load_config("tau_analysis"), "source": "tau_score_reference"}))
    stage("tau_family_analysis", lambda: T.run_tau_family_analysis(
        {**load_config("tau_family_analysis"), "reference_commits": [], "published_metrics": str(tmp / "tau_analysis" / "metrics.json")}))
    prim = T.primary_items({"build_run": "tau_build"})
    counts = {}
    for tag in T.SCORED:
        shards = sorted((tmp / f"tau_family_score_{tag}").glob("shard_*"))
        per = [len(read_jsonl(p / "scoring.jsonl")) for p in shards]
        counts[tag] = {"passes_per_shard": per, "passes": sum(per), "expected": 16 * len(prim)}
    place = read_jsonl(tmp / "tau_family_placement_mistral-7b-v0.3" / "shard_00_of_01" / "scoring.jsonl")
    smoke = read_jsonl(tmp / "tau_family_score_smoke" / "shard_00_of_01" / "scoring.jsonl")
    m = json.loads((tmp / "tau_family_analysis" / "metrics.json").read_text())
    sm = json.loads((tmp / "tau_family_smoke_analysis" / "metrics.json").read_text())
    summary.update({
        "primary_items": len(prim), "primary_users": len({t["user_id"] for t in prim}),
        "passes": counts, "placement_passes": len(place), "smoke_items": len({r["transition_id"] for r in smoke}),
        "smoke_passes": len(smoke),
        "analysis": {"missing": m["missing_scores"], "stale": m["stale_scores"], "reference_missing": m["reference_run"]["missing"],
                     "reference_check_all_equal": m["reference_check"].get("all_equal"),
                     "verdict_keys": sorted(m["verdicts"]["per_model"]), "comparisons": sorted(m["comparisons"]),
                     "placement_items": m["placement"]["n_items"],
                     "status": (tmp / "tau_family_analysis" / "STATUS.md").read_text().splitlines()[0]},
        "smoke_projection_keys": sorted(sm["projection"]["models"]),
    })
    print(json.dumps(summary, indent=1))
    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=1))
    ok = (all(v["passes"] == v["expected"] for v in counts.values()) and len(place) == 600
          and m["missing_scores"] == {t: 0 for t in T.SCORED} and m["reference_check"].get("all_equal"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
