#!/bin/bash
#SBATCH --job-name=wf-dsa-score
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Stage 1b verifier scores, one model and one shard per job (2 shards per model). Every distinct candidate of the two
# lists (one shared cache) under the task-aware, task-blind and, where a swap object exists, swapped prompts, plus the
# neutral prompt for the priority models; single-candidate states get the blind arm only. 45,000-100,000 single-token
# passes per priority model, 35,000-80,000 for Qwen, at about 0.15-0.25 s (histories to 30 steps): 2-7 GPU-hours per
# model. 6 h per half-model job; a cut-off job resumes from its saved scores (WF_RESUME=1).
# Submit only after the model-free analysis has been read (the futility rule, notes/decisionsim_agent_design.md).
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100 (every model here fits in 40 GB); brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/dsa_score_%j.out
#SBATCH --error=logs/dsa_score_%j.err
#
# After every labelling shard says "shard complete" (priority models first):
#   for t in llama3.1-8b mistral-7b-v0.3 qwen3-8b; do for i in 0 1; do
#     bash pace/submit.sh decisionsim_agent_score.sbatch --export=ALL,DSA_RUN=decisionsim_agent_score_$t,WF_SHARD=$i,WF_N_SHARDS=2
#   done; done
# Mistral needs the wave A weights in the cache (pace/families_prefetch.sbatch) and stops before scoring if the Yes/No
# gate fails (runs/<run>/shard_*/tokenization_checks.json). When every model's shards are complete:
#   sbatch --export=ALL,ANALYSIS_RUNS=decisionsim_agent_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${DSA_RUN:?set DSA_RUN, e.g. decisionsim_agent_score_llama3.1-8b}"
case "${RUN}" in
  decisionsim_agent_score_*) ;;
  *) echo "ERROR: DSA_RUN must be a decisionsim_agent_score_<tag> run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
