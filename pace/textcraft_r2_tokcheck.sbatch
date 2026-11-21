#!/bin/bash
#SBATCH --job-name=wf-tc2tok
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tc2tok_%j.out
#SBATCH --error=logs/tc2tok_%j.err
#
# TextCraft round 2 (notes/textcraft_r2_design.md, section 3): the tokenization gate and the golden renderings for one
# model, on every record and arm, with the pinned tokenizer from the HF cache, offline, on a CPU node. No GPU and no
# model weights. One job per model:
#   for t in qwen3-8b llama3.1-8b gemma2-9b-it mistral-7b-v0.3 llama3.2-3b; do
#     sbatch --account=coc --qos=coc-ice --partition=coc-cpu \
#       --export=ALL,TC2_TAG=$t,SERVE_ENV=<serve env>,HF_HOME=<hf cache> pace/textcraft_r2_tokcheck.sbatch
#   done
# The serve environment is only read: bytecode is never written (PYTHONDONTWRITEBYTECODE), and nothing is installed.
# Writes runs/textcraft_r2_tokcheck/<tag>.json; then, on any machine with the five reports:
#   python -m wf.textcraft.r2_tokcheck golden runs/textcraft_r2_tokcheck/*.json

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${TC2_TAG:?set TC2_TAG to a model tag of wf/textcraft/r2.py}"
: "${SERVE_ENV:?set SERVE_ENV to the serve environment's prefix}"
: "${HF_HOME:?set HF_HOME to the HF cache holding the pinned snapshots}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" XDG_CACHE_HOME="$(pwd)/.cache"
unset WF_MOCK_SCORE WF_MOCK_ENV WF_MOCK_AGENT
mkdir -p runs/textcraft_r2_tokcheck logs "${XDG_CACHE_HOME}"
echo "=== r2 tokcheck ${TC2_TAG} START $(date) host=$(hostname) commit=$(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) ==="
"${SERVE_ENV}/bin/python" -m wf.textcraft.r2_tokcheck check "${TC2_TAG}" "runs/textcraft_r2_tokcheck/${TC2_TAG}.json"
echo "=== r2 tokcheck ${TC2_TAG} END $(date) ==="
