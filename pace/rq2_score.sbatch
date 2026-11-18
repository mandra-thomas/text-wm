#!/bin/bash
#SBATCH --job-name=wf-rq2
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/rq2_%j.out
#SBATCH --error=logs/rq2_%j.err
#
# Confirmatory RQ2 scoring, one shard of the games per job (HF transformers only; no vLLM, no game server):
#   sbatch ... --export=ALL,RQ2_RUN=rq2_score,WF_SHARD=0,WF_N_SHARDS=4 pace/rq2_score.sbatch
# Every score is saved as it finishes; resubmit with WF_RESUME=1 after a timeout. When every shard says
# "shard complete": python -m wf.runs rq2_analysis  (no GPU needed).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${RQ2_RUN:?set RQ2_RUN, e.g. rq2_score or rq2_score_smoke}"
echo "=== RQ2 ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== RQ2 ${RUN} END $(date) ==="
