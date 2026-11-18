#!/bin/bash
#SBATCH --job-name=wf-rq2
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sized from the real-data smoke (2026-11-18): 65 items, both models, 6 min, 11.7 GB peak, under 1 core.
# A quarter of the confirmatory items is about 1,100: roughly an hour, so 2 h; scores resume if cut off.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/rq2_%j.out
#SBATCH --error=logs/rq2_%j.err
#
# Confirmatory RQ2 scoring, one shard of the games per job (HF transformers only; no vLLM, no game server):
#   sbatch ... --export=ALL,RQ2_RUN=rq2_score,WF_SHARD=0,WF_N_SHARDS=4 pace/rq2_score.sbatch
# Every score is saved as it finishes; resubmit with WF_RESUME=1 after a timeout. When every shard says
# "shard complete": python -m wf.runs rq2_analysis  (no GPU needed).
# The RQ1 run uses this script too, with a longer limit (about 2.5 GPU-hours per eighth of the games):
#   sbatch --time=4:00:00 ... --export=ALL,RQ2_RUN=rq1_score,WF_SHARD=0,WF_N_SHARDS=8 pace/rq2_score.sbatch
# and a resume job queued behind each with --dependency=afternotok:<job> and WF_RESUME=1.

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
