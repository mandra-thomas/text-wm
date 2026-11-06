#!/bin/bash
#SBATCH --job-name=wf-pilotA
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/pilotA_%j.out
#SBATCH --error=logs/pilotA_%j.err
#
# GPU ownership (1×A100): vLLM for transitions (agent), then stop vLLM
# so HF scoring owns the GPU for pilotA_score.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — Pilot A refuses to run (protocol §6)." >&2
  exit 2
fi

echo "=== Pilot A START $(date) host=$(hostname) ==="
nvidia-smi || true

cleanup() {
  stop_vllm || true
  stop_alfworld_server || true
}
trap cleanup EXIT

start_alfworld_server

echo "=== Phase A: vLLM for pilotA_transitions ==="
start_vllm
run_wf pilotA_transitions

echo "=== Phase B: stop vLLM before pilotA_score (HF) ==="
stop_vllm
run_wf pilotA_score

echo "=== Pilot A END $(date) ==="
