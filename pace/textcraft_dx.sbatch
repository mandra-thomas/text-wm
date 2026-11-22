#!/bin/bash
#SBATCH --job-name=wf-tcdx
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Exploratory TextCraft Llama diagnostics (notes/textcraft_llama_dx.md). Each run is well under an hour on one GPU
# (the examples-on-goal variant, 1,600 scores with both models, took about 10 minutes).
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/tcdx_%j.out
#SBATCH --error=logs/tcdx_%j.err
#
#   sbatch --account=coc --qos=coc-ice --partition=coc-gpu \
#     --export=ALL,DX_RUN=textcraft_dx_zeroshot,SERVE_ENV=<env>,HF_HOME=<cache>,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
#     pace/textcraft_dx.sbatch
# Every record is saved as it is made; resubmit with WF_RESUME=1 after a timeout.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${DX_RUN:?set DX_RUN to textcraft_dx_zeroshot or textcraft_dx_generate}"
echo "=== TextCraft dx ${RUN} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== TextCraft dx ${RUN} END $(date) ==="
