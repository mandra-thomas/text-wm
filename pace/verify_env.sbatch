#!/bin/bash
#SBATCH --job-name=wf-verify
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#
# Real AgentGym server, no LLM and no GPU: checks the client, scene scans, success twins,
# context_determined soundness and real timing before any GPU time is spent.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

cleanup() { stop_alfworld_server || true; }
trap cleanup EXIT

start_alfworld_server
activate_serve
python -m wf.tools.verify_real_env --games "${VERIFY_GAMES:-30}" --win-games "${VERIFY_WIN_GAMES:-15}" ${VERIFY_ONLY:+--only "$VERIFY_ONLY"}
