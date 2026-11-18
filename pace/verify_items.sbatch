#!/bin/bash
#SBATCH --job-name=wf-vitems
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/verify_items_%j.out
#SBATCH --error=logs/verify_items_%j.err
#
# Full verification of collected items on the real server, CPU only (no GPU, outside the GPU cap):
#   sbatch ... --export=ALL,VERIFY_RUNS="rq12_transitions_s0" pace/verify_items.sbatch
# Writes runs/<run>/VERIFY.json; rq2_score refuses a shard without a full pass on its exact file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
RUNS="${VERIFY_RUNS:?set VERIFY_RUNS to one or more collection runs}"
cleanup() { stop_alfworld_server || true; }
trap cleanup EXIT
start_alfworld_server
activate_serve
# shellcheck disable=SC2086
python -m wf.tools.verify_items ${RUNS} --server
