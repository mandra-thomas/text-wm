#!/usr/bin/env bash
# Local dry-run of the same run_ids the ICE sbatch scripts execute.
# Uses WF_MOCK_*=1 (no GPU / no ALFWorld server).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WF_MOCK_ENV=1 WF_MOCK_AGENT=1 WF_MOCK_SCORE=1
export RUNS_DIR="${RUNS_DIR:-$ROOT/runs_mock}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

mkdir -p "$RUNS_DIR"

echo "=== Tier 0 (mock) ==="
for rid in t0_smoke t0_replay t0_agent_determinism t0_parser t0_ckpt_accuracy t0_scoring_agreement t0_expert_labels; do
  echo "--- $rid ---"
  "$PY" -m wf.runs "$rid" --mock
done
"$PY" -m wf.tools.write_env_json

# Filing gate for Pilot A/B
if [[ ! -f "$RUNS_DIR/FILED.txt" ]]; then
  echo "note: mock filing for pipeline dry-run" >"$RUNS_DIR/FILED.txt"
fi

# A dry run walks past gates and pilot rules so every stage still runs; a crash still stops it.
export WF_MOCK_CONTINUE=1
gated() {
  rm -f "$RUNS_DIR/$1/STATUS.md"
  "$PY" -m wf.runs "$1" --mock || {
    [[ -f "$RUNS_DIR/$1/STATUS.md" ]] || exit 1
    echo "note: $1 stopped at its gate or rule; mock dry run continues"
  }
}

echo "=== Pilot A (mock) ==="
gated pilotA_transitions
gated pilotA_score

echo "=== Pilot B (mock) ==="
gated pilotB_headroom
gated pilotB_headroom_recheck
gated pilotB_consumption
gated pilotB_eligibility
gated pilotB_retention_minirun

"$PY" -m wf.tools.assemble_pilot_report
echo "=== mock pipeline done ==="
