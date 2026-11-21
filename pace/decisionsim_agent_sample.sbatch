#!/bin/bash
#SBATCH --job-name=wf-dsa-sample
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Stage 1b sampling (wf/decisionsim_agent.py, notes/decisionsim_agent_design.md): one shard of the 134 games per job.
# About 750 states per shard, one vLLM request of 20 completions per state and one replay walk per game; the design
# note estimates 2-3 GPU-hours for all four shards. 3 h per job, about three times a shard's share; a cut-off shard
# resumes from its saved lists (WF_RESUME=1), never resampling a saved state.
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/dsa_sample_%j.out
#SBATCH --error=logs/dsa_sample_%j.err
#
# The RQ3 episodes (runs/rq3, commit ddeabc2) must be in this checkout's runs/ or named by an absolute path in
# wf/config/decisionsim_agent_sample.yaml (episodes_source). Smoke first (two train games, rq3_smoke), and check its
# gate (format errors under 25% of samples, no replay mismatch: runs/decisionsim_agent_sample_smoke/shard_00_of_01/
# metrics.json) before the census:
#   DSA_RUN=decisionsim_agent_sample_smoke bash pace/submit.sh decisionsim_agent_sample.sbatch
#   for i in 0 1 2 3; do DSA_RUN=decisionsim_agent_sample WF_SHARD=$i WF_N_SHARDS=4 bash pace/submit.sh decisionsim_agent_sample.sbatch; done
# Then label on CPU (pace/decisionsim_agent_label.sbatch).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${DSA_RUN:?set DSA_RUN to decisionsim_agent_sample or decisionsim_agent_sample_smoke}"
case "${RUN}" in
  decisionsim_agent_sample|decisionsim_agent_sample_smoke) ;;
  *) echo "ERROR: DSA_RUN must be a decisionsim_agent_sample run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT
start_vllm
nvidia-smi --query-gpu=name,memory.total --format=csv || true
start_alfworld_server
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
