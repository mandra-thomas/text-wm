#!/bin/bash
#SBATCH --job-name=wf-rq12col
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
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

RUN="${RQ12_RUN:?set RQ12_RUN to a shard, e.g. --export=ALL,RQ12_RUN=rq12_transitions_s0}"
echo "=== RQ1/RQ2 collection ${RUN} START $(date) host=$(hostname) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT
start_vllm
nvidia-smi --query-gpu=name,memory.total --format=csv || true  # which GPU this job got
# The ALFWorld server leaks memory on every reset, so it is restarted after every max_games_per_job
# games (the run stops with a PARTIAL status) while vLLM stays up; the run then resumes.
while true; do
  start_alfworld_server
  set +e
  run_wf "${RUN}"
  rc=$?
  set -e
  stop_alfworld_server || true
  if [[ -f "${RUNS_DIR}/${RUN}/STATUS.md" ]] && grep -q "PARTIAL" "${RUNS_DIR}/${RUN}/STATUS.md"; then
    export WF_RESUME=1
    continue
  fi
  break
done
echo "=== RQ1/RQ2 collection ${RUN} END $(date) rc=${rc} ==="
exit "${rc}"
