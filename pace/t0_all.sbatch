#!/bin/bash
#SBATCH --job-name=wf-t0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/t0_%j.out
#SBATCH --error=logs/t0_%j.err
#
# Override account/qos via: sbatch --account=... --qos=... pace/t0_all.sbatch
# Or set defaults in pace/pace.env and wrap with a submit helper.

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

echo "=== Tier 0 START $(date) host=$(hostname) ==="
nvidia-smi || true

cleanup() {
  stop_vllm || true
  stop_alfworld_server || true
}
trap cleanup EXIT

start_alfworld_server
start_vllm

TIER0_RUNS=(
  t0_smoke
  t0_replay
  t0_agent_determinism
  t0_parser
  t0_ckpt_accuracy
  t0_scoring_agreement
  t0_expert_labels
)

failed=0
for rid in "${TIER0_RUNS[@]}"; do
  echo "=== RUN ${rid} ==="
  if ! run_wf "${rid}"; then
    echo "FAIL ${rid} — see runs/${rid}/STATUS.md"
    failed=1
    break
  fi
done

python -m wf.tools.write_env_json || true

echo "=== Tier 0 END $(date) failed=${failed} ==="
exit "${failed}"
