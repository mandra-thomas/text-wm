#!/bin/bash
#SBATCH --job-name=wf-family
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Gemma 2 9B in bf16 is about 18.5 GB of weights; loading goes through CPU memory first.
#SBATCH --mem=48G
# A whole model in one shard is about 43,700 forward passes (4,666 primary items x 9 + 292 successes x 6). From the
# RQ2 sizing (about 0.4 s a pass for an 8B model; prompts here are up to about 1,160 tokens), expect roughly 5 h for Mistral 7B, 6 h
# for Gemma 2 9B (eager attention is slower than SDPA), 2 h for Llama 3.2 3B and 1 h for 1B. These are estimates,
# not timings. 10 h by default; scores resume if cut off (WF_RESUME=1).
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100 (every wave A model fits in 40 GB); brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/family_%j.out
#SBATCH --error=logs/family_%j.err
#
# Model families, wave A (wf/families.py, notes/families_design.md): one model per job, one shard by default.
#   bash pace/submit.sh family_score.sbatch --export=ALL,FAMILY_RUN=family_score_gemma2-9b-it
#   bash pace/submit.sh family_score.sbatch --time=3:00:00 --export=ALL,FAMILY_RUN=family_score_llama3.2-1b
# Run pace/families_prefetch.sbatch first, so the job does not download weights on GPU time. The job stops before
# scoring anything if the tokenization gate fails (runs/<run>/shard_*/tokenization_checks.json).
# After a timeout: the same command with WF_RESUME=1 in --export. When every shard says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=families_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${FAMILY_RUN:?set FAMILY_RUN, e.g. family_score_gemma2-9b-it}"
case "${RUN}" in
  family_score_*) ;;
  *) echo "ERROR: FAMILY_RUN must be a family_score_<tag> run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
