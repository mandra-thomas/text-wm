#!/bin/bash
#SBATCH --job-name=wf-rq3
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s|gpu-a40|gpu-a100
#SBATCH --output=logs/rq3_%j.out
#SBATCH --error=logs/rq3_%j.err
#
# Confirmatory RQ3, one shard of the games per job (games are dealt round-robin to shards):
#   WF_SHARD=0 WF_N_SHARDS=4 bash pace/submit.sh rq3.sbatch     (then WF_SHARD=1, 2, 3)
# Each shard saves to runs/rq3/shard_<i>_of_<n>/ after every episode. The ALFWorld server leaks
# memory on every reset, so the run stops with PARTIAL after max_episodes_per_job new episodes, the
# server restarts and the run resumes, while vLLM stays up. Jobs are 4 h, inside the per-user GPU
# cap; if one ends before its shard is complete, resubmit with WF_RESUME=1 and the same shard.
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
start_vllm
nvidia-smi --query-gpu=name,memory.total --format=csv || true  # which GPU this job got
RUN="${RQ3_RUN:-rq3}"
rc=0
while true; do
  start_alfworld_server
  set +e
  run_wf "${RUN}"
  rc=$?
  set -e
  stop_alfworld_server || true
  # the shard folder is the only one under runs/<RUN>/ that this job's WF_SHARD names
  status=$(ls "${RUNS_DIR}/${RUN}"/shard_*/STATUS.md 2>/dev/null | grep "shard_$(printf %02d "${WF_SHARD:-0}")_of_" | head -1)
  if [[ -n "${status}" ]] && grep -q "PARTIAL" "${status}"; then
    export WF_RESUME=1
    continue
  fi
  break
done
if [[ "${RQ3_ANALYSE:-}" == "1" ]]; then
  run_wf "${RUN}_analysis"
fi
echo "=== RQ3 ${RUN} END $(date) rc=${rc} ==="
exit "${rc}"
