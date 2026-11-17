#!/bin/bash
#SBATCH --job-name=wf-rq12col
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/rq12col_%j.out
#SBATCH --error=logs/rq12col_%j.err
#
# Confirmatory RQ1/RQ2 item collection on valid_unseen: agent rollouts (vLLM) and counterfactual
# replays only. No world model is scored here. Resume after a timeout with WF_RESUME=1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi

echo "=== RQ1/RQ2 collection START $(date) host=$(hostname) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT
start_alfworld_server
start_vllm
run_wf "${RQ12_RUN:-rq12_transitions}"  # RQ12_RUN=rq12_transitions_smoke for the train-game check
echo "=== RQ1/RQ2 collection END $(date) ==="
