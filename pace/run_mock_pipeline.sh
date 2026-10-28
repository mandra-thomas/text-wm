#!/usr/bin/env bash
# Local dry-run of the same run_ids the ICE sbatch scripts execute.
# Uses WF_MOCK_*=1 (no GPU / no ALFWorld server).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export WF_MOCK_ENV=1 WF_MOCK_AGENT=1 WF_MOCK_SCORE=1
export RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"
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
  SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  cat >"$RUNS_DIR/FILED.txt" <<EOF
filed_date: $(date -u +%Y-%m-%d)
constants_commit: ${SHA}
RETENTION_MARGIN: 1.0
note: mock filing for pipeline dry-run; replace with real filing date before ICE Pilot A/B
EOF
fi

echo "=== Pilot A (mock) ==="
"$PY" -m wf.runs pilotA_transitions --mock
"$PY" -m wf.runs pilotA_score --mock

echo "=== Pilot B (mock) ==="
"$PY" -m wf.runs pilotB_headroom --mock
"$PY" -m wf.runs pilotB_consumption --mock
"$PY" -m wf.runs pilotB_eligibility --mock
"$PY" -m wf.runs pilotB_retention_minirun --mock

"$PY" -m wf.tools.assemble_pilot_report
echo "=== mock pipeline done ==="
