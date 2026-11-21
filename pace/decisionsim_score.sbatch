#!/bin/bash
#SBATCH --job-name=wf-decisionsim
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Gemma 2 9B in bf16 is about 18.5 GB of weights; loading goes through CPU memory first.
#SBATCH --mem=48G
# One model per job. The census (all 761 eligible states, 9,930 candidates; aligned and neutral everywhere, swapped
# at the 709 states where it is defined) is 29,056 single-token forward passes. Wave A ran about 43,700 passes of a
# 7-9B model in 103-107 minutes on an A40 or L40S (about 0.15 s a pass), so expect about 73 minutes for an 8B model
# plus a few minutes of loading, about 25 for Llama 3.2 3B; 4.5-5 GPU-hours for all five. Estimates, not timings.
# 4 h per job (about three times the estimate) so a slow node cannot cut a job off; scores resume if it does
# (WF_RESUME=1).
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100 (every model here fits in 40 GB); brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/decisionsim_%j.out
#SBATCH --error=logs/decisionsim_%j.err
#
# Option A decision simulation (wf/decisionsim.py, notes/decisionsim_design.md), one model per job:
#   bash pace/submit.sh decisionsim_score.sbatch --export=ALL,DECISIONSIM_RUN=decisionsim_score_qwen3-8b
#   bash pace/submit.sh decisionsim_score.sbatch --export=ALL,DECISIONSIM_RUN=decisionsim_score_llama3.1-8b
# and, secondary, decisionsim_score_mistral-7b-v0.3, _gemma2-9b-it, _llama3.2-3b. The family models need the wave A
# weights in the cache (pace/families_prefetch.sbatch) and stop before scoring if the Yes/No gate fails
# (runs/<run>/shard_*/tokenization_checks.json). After a timeout: the same command with WF_RESUME=1 in --export.
# When every model's shard says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=decisionsim_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${DECISIONSIM_RUN:?set DECISIONSIM_RUN, e.g. decisionsim_score_qwen3-8b}"
case "${RUN}" in
  decisionsim_score_*) ;;
  *) echo "ERROR: DECISIONSIM_RUN must be a decisionsim_score_<tag> run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
