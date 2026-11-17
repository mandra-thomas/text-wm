#!/bin/bash
#SBATCH --job-name=wf-rq3
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/rq3_%j.out
#SBATCH --error=logs/rq3_%j.err
#
# Confirmatory RQ3, one shard of the games per job (games are dealt round-robin to shards):
#   WF_SHARD=0 WF_N_SHARDS=4 bash pace/submit.sh rq3.sbatch     (then WF_SHARD=1, 2, 3)
# Each shard saves to runs/rq3/shard_<i>_of_<n>/ after every episode and stops cleanly after
# max_episodes_per_job new episodes (the ALFWorld server leaks memory on every reset), so a
# shard takes several jobs: resubmit with WF_RESUME=1 and the same WF_SHARD / WF_N_SHARDS.
# Once every shard says "shard complete", run the analysis (no GPU needed):
#   python -m wf.runs rq3_analysis      or      RQ3_ANALYSE=1 on the last shard job
# Train-game smoke test, one job with its analysis:
#   RQ3_RUN=rq3_smoke RQ3_ANALYSE=1 bash pace/submit.sh rq3.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi

echo "=== RQ3 ${RQ3_RUN:-rq3} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT
start_alfworld_server
start_vllm
run_wf "${RQ3_RUN:-rq3}"
if [[ "${RQ3_ANALYSE:-}" == "1" ]]; then
  run_wf "${RQ3_RUN:-rq3}_analysis"
fi
echo "=== RQ3 END $(date) ==="
