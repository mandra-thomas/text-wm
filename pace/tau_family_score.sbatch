#!/bin/bash
#SBATCH --job-name=wf-tau-family
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# τ-bench, wave-A model families (notes/tau_families_design.md §4.4). One model per job, 4 shards per model by user:
# 856 primary items x 14 single-token passes = 11,984 per model, about 3,000 per shard. Estimated at about 1.5 s a pass
# for Mistral 7B v0.3 (prompts median 11,870 tokens, at most 14,642) and about 0.65 s for Llama 3.2 3B (9,767 / 11,907):
# about 1.25 h and 0.55 h per shard. The limit is 3 h (tau_family_smoke_analysis.yaml time_limit_hours must equal it;
# a test checks). The smoke run's projection decides before the full run is submitted. Every score is saved as it
# finishes, so a shard cut off resumes: the same command with WF_RESUME=1, on the GPU type it started on (--constraint).
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
# L40S, A100 or H100 (A40s are left out: about half the speed); brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s|gpu-a100|gpu-h100)
#SBATCH --output=logs/tau_family_%j.out
#SBATCH --error=logs/tau_family_%j.err
#
# After tau_family_token_check has passed at this commit (pace/tau_family_items.sbatch):
#   sbatch --account=coc --qos=coc-ice --export=ALL,TAU_FAMILY_RUN=tau_family_score_smoke pace/tau_family_score.sbatch
#   sbatch --account=coc --qos=coc-ice --export=ALL,TAU_FAMILY_RUN=tau_family_score_mistral-7b-v0.3,WF_SHARD=0,WF_N_SHARDS=4 pace/tau_family_score.sbatch
#   sbatch --account=coc --qos=coc-ice --export=ALL,TAU_FAMILY_RUN=tau_family_placement_mistral-7b-v0.3 pace/tau_family_score.sbatch
# The run refuses unverified or unpinned items, a model whose token check has not passed, Gemma 2 (excluded), and a
# prompt at or above the model's context. When every shard says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=tau_family_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${TAU_FAMILY_RUN:?set TAU_FAMILY_RUN, e.g. tau_family_score_mistral-7b-v0.3}"
case "${RUN}" in
  tau_family_score_*|tau_family_placement_*) ;;
  *) echo "ERROR: TAU_FAMILY_RUN must be a tau_family_score_* or tau_family_placement_* run, not ${RUN}" >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
