"""Option C follow-up, task 2 (2026-11-24; spec in notes/optionc_followups_2026-11-24.md), CPU: the step-0 check's
classification in plain text, computed by the frozen code (wf.optionc.step0_report: the same cells, gaps, interval
rule, competence rule and classification as optionc_step0_analysis), beside the same computation on the logged
with-block scores (which must reproduce the frozen metrics' classification and gaps).

    python scripts/optionc_followup/plain_step0_analysis.py --plain runs/optionc_fu/plain_step0_qwen3-8b.jsonl \
        --block runs/optionc_step0_qwen3-8b/scoring.jsonl --frozen runs/optionc_step0_analysis/metrics.json \
        --out runs/optionc_fu/plain_step0_summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def q(x: float) -> float:
    return round(x * 64) / 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from wf import optionc as O
    from wf.rq2 import SOURCES

    states = O.step0_states(list(SOURCES))
    lines = [json.loads(x) for x in Path(args.plain).open()]
    plain = [r for st in lines for r in st["records"]]
    checks = [r for st in lines for r in st.get("block_check", [])]
    block = [json.loads(x) for x in Path(args.block).open()]
    frozen = json.load(open(args.frozen))["models"][O.PRIMARY["model"]]

    n_jobs = len(O.step0_jobs(states, tuple(O.VERIFIER_ARMS), O.READOUTS))
    rep_plain = O.step0_report(states, plain)
    rep_block = O.step0_report(states, block)

    logged = {(r["transition_id"], r["arm"], r["readout"]): r for r in block}
    hash_ok = sum(logged.get((r["transition_id"], r["arm"], r["readout"]), {}).get("score_hash") == r["score_hash"]
                  for r in plain)
    chk = [(r, logged[(r["transition_id"], r["arm"], r["readout"])]) for r in checks]
    block_check = {"n": len(chk), "identical_rounded": sum(r["L"] == l["L"] for r, l in chk),
                   "max_abs_diff_raw": max((abs(r["L_raw"] - l["L_raw"]) for r, l in chk), default=None)}

    def mass(readout: str) -> dict:
        m = [r["first_mass"] for r in plain if r["readout"] == readout]
        return {"n": len(m), "mean": statistics.mean(m), "median": statistics.median(m), "min": min(m),
                "share_below_0.5": sum(x < 0.5 for x in m) / len(m)}

    def brief(rep: dict) -> dict:
        return {"classification": rep["classification"], "gap_states": rep["gap_states"],
                "proposal_gaps": rep["proposal_gaps"], "blind_competence": rep["blind_competence"],
                "working_proposals_accepted": {rd: {a: c["working_proposals_accepted"] for a, c in cells.items()}
                                               for rd, cells in rep["cells"].items()},
                "failures_accepted": {rd: {a: c["yes_failures"] for a, c in cells.items()} for rd, cells in rep["cells"].items()},
                "auc": {rd: {a: c["auc"] for a, c in cells.items()} for rd, cells in rep["cells"].items()},
                "median_L_working_proposal": {rd: {a: c["median_L_working_proposal"] for a, c in cells.items()}
                                              for rd, cells in rep["cells"].items()},
                "shifts": rep["shifts"]}

    out = {"n_states": len(states), "n_jobs_expected": n_jobs, "n_plain": len(plain), "n_states_done": len(lines),
           "gpus": sorted({st["gpu"] for st in lines}), "revisions": sorted({st["revision"] for st in lines}),
           "plain_prompt_hash_equals_logged": hash_ok,
           "fail_tokens": sorted({r["fail_token"] for r in plain if r["readout"] == "obs"}),
           "you_tokens": sorted({r["you_token"] for r in plain if r["readout"] == "obs"}),
           "first_token_mass": {"yn": mass("yn"), "obs": mass("obs")},
           "block_reproduction_check": block_check,
           "with_block_recomputed_equals_frozen": {
               "classification": rep_block["classification"] == frozen["classification"],
               "proposal_gaps": rep_block["proposal_gaps"] == frozen["proposal_gaps"],
               "blind_competence": rep_block["blind_competence"] == frozen["blind_competence"]},
           "with_block": brief(rep_block), "plain": brief(rep_plain)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in ("n_plain", "n_jobs_expected", "gpus", "plain_prompt_hash_equals_logged",
                                          "fail_tokens", "first_token_mass", "block_reproduction_check",
                                          "with_block_recomputed_equals_frozen")}, indent=1))
    print("with block:", out["with_block"]["classification"], out["with_block"]["gap_states"])
    print("plain text:", out["plain"]["classification"], out["plain"]["gap_states"])
    print(json.dumps(out["plain"]["proposal_gaps"], indent=1))


if __name__ == "__main__":
    main()
