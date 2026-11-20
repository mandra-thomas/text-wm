#!/bin/bash
#SBATCH --job-name=wf-size32b
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# A guess with headroom (the 2026-11-19 run's timing is in the private tracker only); scores resume if cut off.
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
# Qwen3-32B in bf16 is about 66 GB of weights, so only an 80 GB A100 holds it (ICE feature A100-80GB; brackets
# because ICE appends &cuda13). submit.sh never overrides this constraint.
#SBATCH --constraint=(A100-80GB)
#SBATCH --output=logs/size32b_%j.out
#SBATCH --error=logs/size32b_%j.err
#
# The size series' 32B scoring, one shard per job (wf/sizes.py). Added 2026-11-20 so the requirement is in the
# repo; the 2026-11-19 run had no job script here (its submission is recorded in the private tracker only):
#   sbatch --export=ALL,WF_SHARD=0,WF_N_SHARDS=2 pace/size_score_32b.sbatch
# The smaller sizes run through pace/rq2_score.sbatch (any L40S, A40 or A100):
#   sbatch --export=ALL,RQ2_RUN=size_score_qwen3-14b,WF_SHARD=0,WF_N_SHARDS=2 pace/rq2_score.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
echo "=== size_score_qwen3-32b shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf size_score_qwen3-32b
echo "=== size_score_qwen3-32b END $(date) ==="
