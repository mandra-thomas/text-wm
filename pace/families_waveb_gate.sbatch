#!/bin/bash
#SBATCH --job-name=wf-gateB
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/gateB_%j.out
#SBATCH --error=logs/gateB_%j.err
#
# Wave B tokenization and template gate on every item, tokenizers only (wf/tools/family_gate_cpu.py; design section 7).
# Needs only the tokenizer and config files at the pinned revisions in the shared cache; loads no model.
#   bash pace/submit.sh families_waveb_gate.sbatch
#   bash pace/submit.sh families_waveb_gate.sbatch --export=ALL,GATE_TAGS="gemma4-31b-it"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
export HF_HUB_OFFLINE=1
activate_serve
TAGS="${GATE_TAGS:-ministral3-8b-it qwen3.5-9b qwen3.5-9b-plain qwen3.5-9b-base-plain gemma4-e4b-it qwen3.8-27b gemma4-31b-it}"
echo "=== wave B gate START $(date) host=$(hostname) tags=${TAGS} ==="
# shellcheck disable=SC2086
python -m wf.tools.family_gate_cpu ${TAGS} --out "logs/gateB_${SLURM_JOB_ID:-local}.json" \
  --golden "logs/gateB_golden_${SLURM_JOB_ID:-local}.json"
echo "=== wave B gate END $(date) ==="
