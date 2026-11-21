#!/bin/bash
#SBATCH --job-name=wf-tc2
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Gemma 2 9B in bf16 is about 18.5 GB of weights; loading goes through CPU memory first.
#SBATCH --mem=48G
# One shard of one model (24,008 / 4 records-and-arms per model, design section 11): about 25 minutes for the 7-8B
# models, 15 for Llama 3.2 3B, 50 for Gemma 2 with eager attention. Estimates, not timings; 2 h leaves room for the
# gate, which renders every prompt of the shard before any score.
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/tc2_%j.out
#SBATCH --error=logs/tc2_%j.err
#
# TextCraft round 2 scoring (wf/textcraft/r2_score.py, notes/textcraft_r2_design.md). ONLY AFTER the design is frozen
# at its tag. One model and one shard per job:
#   sbatch --account=coc --qos=coc-ice --partition=coc-gpu \
#     --export=ALL,TC2_RUN=textcraft_r2_score_qwen3-8b,WF_SHARD=0,WF_N_SHARDS=4,SERVE_ENV=<env>,HF_HOME=<cache>,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
#     pace/textcraft_r2_score.sbatch
# Smoke first (textcraft_r2_smoke_<tag>, 20 records, one shard). The job stops before any score if the tokenization
# gate or the golden comparison fails (runs/<run>/shard_*/tokenization_checks.json). After a timeout: WF_RESUME=1.
# When every shard of every model says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=textcraft_r2_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${TC2_RUN:?set TC2_RUN, e.g. textcraft_r2_score_qwen3-8b}"
case "${RUN}" in
  textcraft_r2_score_*|textcraft_r2_smoke_*) ;;
  *) echo "ERROR: TC2_RUN must be a textcraft_r2_score_<tag> or textcraft_r2_smoke_<tag> run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
