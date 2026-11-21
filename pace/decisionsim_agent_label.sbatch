#!/bin/bash
#SBATCH --job-name=wf-dsa-label
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# About 1 busy core; the server grows about 36 MB per reset (verify_items.sbatch), one reset per candidate.
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
# Stage 1b labelling, CPU only: one fresh replay per candidate on the real server, one shard of the games per job
# (8 shards; 9,000-18,000 replays in all, about 1,300 an hour). A job stops after max_replays_per_job replays
# (PARTIAL), the server restarts and the run resumes, so memory stays bounded.
#SBATCH --time=4:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/dsa_label_%j.out
#SBATCH --error=logs/dsa_label_%j.err
#
# After every sampling shard says "shard complete":
#   for i in $(seq 0 7); do sbatch --export=ALL,DSA_RUN=decisionsim_agent_label,WF_SHARD=$i,WF_N_SHARDS=8 pace/decisionsim_agent_label.sbatch; done
# Smoke: sbatch --export=ALL,DSA_RUN=decisionsim_agent_label_smoke pace/decisionsim_agent_label.sbatch
# A shard cut off by the time limit: the same command with WF_RESUME=1 in --export.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${DSA_RUN:?set DSA_RUN to decisionsim_agent_label or decisionsim_agent_label_smoke}"
case "${RUN}" in
  decisionsim_agent_label|decisionsim_agent_label_smoke) ;;
  *) echo "ERROR: DSA_RUN must be a decisionsim_agent_label run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
cleanup() { stop_alfworld_server || true; }
trap cleanup EXIT
rc=0
while true; do
  start_alfworld_server
  set +e
  run_wf "${RUN}"
  rc=$?
  set -e
  stop_alfworld_server || true
  status=$(ls "${RUNS_DIR}/${RUN}"/shard_*/STATUS.md 2>/dev/null | grep "shard_$(printf %02d "${WF_SHARD:-0}")_of_" | head -1)
  if [[ -n "${status}" ]] && grep -q "PARTIAL" "${status}"; then
    export WF_RESUME=1
    continue
  fi
  break
done
echo "=== ${RUN} END $(date) rc=${rc} ==="
exit "${rc}"
