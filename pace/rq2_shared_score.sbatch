#!/bin/bash
#SBATCH --job-name=wf-rq2-shared
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# As pace/rq2_score.sbatch, sized for a handful of items: rq2_score scored about 1,100 items per hour with both
# models, so even a few hundred shared-spot items fit in 30 minutes. On the confirmatory collection the set is
# empty (2026-11-20), and the job then exits within a minute without loading a model.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/rq2_shared_%j.out
#SBATCH --error=logs/rq2_shared_%j.err
#
# Shared-spot sensitivity scoring (wf/rq2_shared.py; documentation audit A2, 2026-11-20), one shard per job:
#   sbatch --export=ALL,WF_SHARD=0,WF_N_SHARDS=1 pace/rq2_shared_score.sbatch
# Then, on CPU: sbatch --export=ALL,ANALYSIS_RUNS=rq2_shared_analysis pace/analysis_cpu.sbatch
# Resubmit with WF_RESUME=1 after a timeout.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
echo "=== rq2_shared_score shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf rq2_shared_score
echo "=== rq2_shared_score END $(date) ==="
