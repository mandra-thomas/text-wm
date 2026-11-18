#!/bin/bash
#SBATCH --job-name=wf-pilotA
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
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

# Tier 0's last stage never ran (Tier 0 stopped at its scoring gate). Minutes long, server only;
# its result is a covariate label, so a failure here must not hold up Pilot A.
run_wf t0_expert_labels || echo "WARN: t0_expert_labels failed; see runs/t0_expert_labels/STATUS.md"

echo "=== Phase A: vLLM for pilotA_transitions ==="
start_vllm
run_wf pilotA_transitions

echo "=== Phase B: stop vLLM before pilotA_score (HF) ==="
stop_vllm
run_wf pilotA_score

echo "=== Pilot A END $(date) ==="
