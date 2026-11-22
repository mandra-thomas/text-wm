"""Run the frozen Stage 1a analysis functions (wf.decisionsim, unchanged) on local score copies of any model, outside the
run machinery, and print each model's Option C category against the task-blind and neutral verifiers. Used in
notes/decisionsim_dx.md to report the secondary models, whose scores finished after the frozen analysis ran; it writes
nothing into runs/ and never touches runs/decisionsim_analysis.

Usage: .venv/bin/python scripts/decisionsim_dx_frozen.py <transitions dir with s0..s2.jsonl> <scores dir> <model> ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wf import decisionsim as DS  # noqa: E402
from wf.util import read_jsonl  # noqa: E402

tdir, sdir = Path(sys.argv[1]), Path(sys.argv[2])
items = [json.loads(line) for s in (0, 1, 2) for line in open(tdir / f"s{s}.jsonl")]
states = DS.build_states(items)
for m in sys.argv[3:]:
    rep, _ = DS.analyse_model(states, read_jsonl(sdir / m / "scoring.jsonl"))
    for base in (DS.BLIND, DS.NEUTRAL):
        oc = DS.option_c_model(rep, base)
        print(json.dumps({"model": m, "comparison": base, **{k: oc[k] for k in (
            "category", "fail_rate_difference", "fail_rate_difference_ci", "p", "net_states", "fallback_net_states",
            "aligned_minus_swapped", "aligned_minus_comparison_same_states", "comparison_verifier_beats_no_verifier",
            "worth_doing_inside_a_verifier_that_does_not_pay", "n_states_incomplete")}}))
