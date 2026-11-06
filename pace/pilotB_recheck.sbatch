#!/bin/bash
#SBATCH --job-name=wf-recheck
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/pilotB_recheck_%j.out
#SBATCH --error=logs/pilotB_recheck_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — Pilot B refuses to run (protocol §6)." >&2
  exit 2
fi

echo "=== Pilot B recheck START $(date) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT

start_alfworld_server
start_vllm

# Only after pilotB_headroom came back borderline (exit 3). The later stages run only if this clears it.
run_wf pilotB_headroom_recheck
run_wf pilotB_consumption
run_wf pilotB_eligibility
run_wf pilotB_retention_minirun

echo "=== Pilot B recheck END $(date) ==="
