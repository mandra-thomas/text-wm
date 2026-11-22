#!/bin/bash
#SBATCH --job-name=wf-tc3compat
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/tc3compat_%j.out
#SBATCH --error=logs/tc3compat_%j.err
#
# TextCraft round 3, pre-design compatibility check (notes/textcraft_r3_proposal.md, section 5): round 2's records and
# prompts through each candidate model's pinned tokenizer and chat template, on a CPU node, offline. No GPU, no model
# weights, no score. Tokenizer and config files only, in TC3_TOK (folders <org>__<name>@<sha>).
#   sbatch --account=coc --qos=coc-ice --partition=coc-cpu \
#     --export=ALL,SERVE_ENV=<serve2 prefix>,TC3_TOK=<tokenizer root> pace/textcraft_r3_compat.sbatch
# The serve environment is only read (PYTHONDONTWRITEBYTECODE; nothing installed). Writes runs/textcraft_r3_compat/.
# This script does not source pace/_common.sh (and so never reads pace/pace.env).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${SERVE_ENV:?set SERVE_ENV to the serve2 prefix}"
: "${TC3_TOK:?set TC3_TOK to the tokenizer root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$(pwd)/.hf_unused" XDG_CACHE_HOME="$(pwd)/.cache"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
unset WF_MOCK_SCORE WF_MOCK_ENV WF_MOCK_AGENT
mkdir -p runs/textcraft_r3_compat logs "${XDG_CACHE_HOME}"
TAGS="${TC3_TAGS:-qwen3.5-9b qwen3.6-27b qwen3.8-27b gemma4-12b-it gemma4-e4b-it gemma4-31b-it ministral3-8b-it ministral3-14b-it olmo3-7b-it granite4.2-8b qwen3.5-9b-think ministral3-8b-reasoning olmo3-7b-think gpt-oss-20b}"
echo "=== r3 compat START $(date) host=$(hostname) commit=$(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD) ==="
status=0
for t in ${TAGS}; do
  "${SERVE_ENV}/bin/python" scripts/textcraft_r3_compat.py "${TC3_TOK}" "${t}" "runs/textcraft_r3_compat/${t}.json" \
    > "logs/tc3compat_${SLURM_JOB_ID:-local}_${t}.log" 2>&1 &
done
for job in $(jobs -p); do wait "${job}" || status=1; done
for t in ${TAGS}; do tail -n 1 "logs/tc3compat_${SLURM_JOB_ID:-local}_${t}.log"; done
echo "=== r3 compat END $(date) status=${status} ==="
