#!/bin/bash
#SBATCH --job-name=wf-pilotB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/pilotB_%j.out
#SBATCH --error=logs/pilotB_%j.err

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — Pilot B refuses to run (protocol §6)." >&2
  exit 2
fi

echo "=== Pilot B START $(date) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT

start_alfworld_server
start_vllm

run_wf pilotB_headroom
run_wf pilotB_consumption
run_wf pilotB_eligibility
run_wf pilotB_retention_minirun

echo "=== Pilot B END $(date) ==="
