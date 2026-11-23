#!/bin/bash
#SBATCH --job-name=wf-qfollow
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/qfollow_%j.out
#SBATCH --error=logs/qfollow_%j.err
#
# Exploratory follow-up to the Qwen pipeline audit (notes/qwen_effect_audit_pipeline.md, section 7): the study's own
# scorer with and without Qwen3's empty think block. FU_TASK picks the work:
#   rq2    Qwen3-8B without the block on the items of stored rq2_score shards FU_SHARDS (pin the GPU type those shards
#          ran on: 0-2 A40, 3 L40S), the first 20 also with the block. Reads runs/qwen_audit/items_all.jsonl.
#   sizes  FU_MODELS (colon-separated Hugging Face ids; sbatch --export splits on commas) on runs/qwen_audit/items_sizes.jsonl, both formats.
#   dsim   Qwen3-8B, the decision simulation's step-0 states, four arms, both formats.
# Example:
#   bash pace/submit.sh qwen_followup.sbatch --export=ALL,FU_TASK=rq2,FU_SHARDS=0 --constraint=gpu-a40

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK="${FU_TASK:?set FU_TASK to rq2, sizes or dsim}"
D="${RUNS_DIR}/qwen_audit"
S=scripts/qwen_audit/followup_score.py
echo "=== qwen follow-up ${TASK} ${FU_SHARDS:-}${FU_MODELS:-} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

case "${TASK}" in
  rq2)
    SH="${FU_SHARDS:?set FU_SHARDS}"
    echo "${GPU}" >"${D}/fu_8b_shard${SH//,/_}_gpu.txt"
    python "${S}" rq2 --items "${D}/items_all.jsonl" --model Qwen/Qwen3-8B --stored-shards "${SH}" \
      --check-block 20 --out "${D}/fu_8b_shard${SH//,/_}.jsonl"
    ;;
  sizes)
    for M in ${FU_MODELS//:/ }; do
      tag="$(basename "${M}" | tr 'A-Z' 'a-z')"
      echo "${GPU}" >"${D}/fu_sizes_${tag}_gpu.txt"
      python "${S}" rq2 --items "${D}/items_sizes.jsonl" --model "${M}" --formats block,noblock \
        --out "${D}/fu_sizes_${tag}.jsonl"
    done
    ;;
  dsim)
    echo "${GPU}" >"${D}/fu_dsim_gpu.txt"
    python "${S}" dsim --out "${D}/fu_dsim_step0.jsonl"
    ;;
  *) echo "unknown FU_TASK=${TASK}" >&2; exit 2 ;;
esac
echo "=== qwen follow-up ${TASK} END $(date) ==="
