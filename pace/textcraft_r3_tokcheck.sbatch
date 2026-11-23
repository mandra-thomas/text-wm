#!/bin/bash
#SBATCH --job-name=wf-tc3tok
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=logs/tc3tok_%j.out
#SBATCH --error=logs/tc3tok_%j.err
#
# TextCraft round 3 (notes/textcraft_r3_design.md, section 4): the round-3 gate and the golden renderings, on every
# record and arm, for every round-3 tag (the models, the format variants, and the earlier models as serve2 loads them
# for the environment check), with the pinned tokenizers from the HF cache, offline, in serve2, on a CPU node. One
# process per tag, in parallel. No GPU, no weights, no score.
#   sbatch --account=coc --qos=coc-ice --partition=coc-cpu \
#     --export=ALL,SERVE_ENV=<serve2 prefix>,HF_HOME=<hf cache with the pinned tokenizers> pace/textcraft_r3_tokcheck.sbatch
# The serve environment is only read (PYTHONDONTWRITEBYTECODE; nothing installed). Writes runs/textcraft_r3_tokcheck/;
# then: python -m wf.textcraft.r3_tokcheck golden runs/textcraft_r3_tokcheck/*.json

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SERVE_ENV:?set SERVE_ENV to the serve2 prefix}"
: "${HF_HOME:?set HF_HOME to the HF cache holding the pinned tokenizers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" XDG_CACHE_HOME="$(pwd)/.cache"
unset WF_MOCK_SCORE WF_MOCK_ENV WF_MOCK_AGENT
mkdir -p runs/textcraft_r3_tokcheck logs "${XDG_CACHE_HOME}"
TAGS="${TC3_TAGS:-qwen3.5-9b ministral3-8b-it gemma4-12b-it gemma4-e4b-it qwen3.8-27b gemma4-31b-it olmo3-7b-it gemma4-12b-it-firstturn qwen3.8-27b-nopreserve qwen3-8b-serve2 mistral-7b-v0.3-serve2 gemma2-9b-it-serve2}"
echo "=== r3 tokcheck START $(date) host=$(hostname) commit=$(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD) ==="
status=0
for t in ${TAGS}; do
  "${SERVE_ENV}/bin/python" -m wf.textcraft.r3_tokcheck check "${t}" "runs/textcraft_r3_tokcheck/${t}.json" \
    > "logs/tc3tok_${SLURM_JOB_ID:-local}_${t}.log" 2>&1 &
done
for job in $(jobs -p); do wait "${job}" || status=1; done
for t in ${TAGS}; do tail -n 1 "logs/tc3tok_${SLURM_JOB_ID:-local}_${t}.log"; done
echo "=== r3 tokcheck END $(date) status=${status} ==="
