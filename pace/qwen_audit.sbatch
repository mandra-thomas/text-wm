#!/bin/bash
#SBATCH --job-name=wf-qaudit
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/qaudit_%j.out
#SBATCH --error=logs/qaudit_%j.err
#
# Exploratory pipeline audit of Qwen3-8B's goal effect (notes/qwen_effect_audit_pipeline.md). Reads
# runs/qwen_audit/items.jsonl (scripts/qwen_audit/sample_items.py) and writes runs/qwen_audit/<step>*.jsonl.
# AUDIT_STEP picks the work:
#   qwen_hf   Qwen3-8B bf16: the independent scorer in every format, then the study's own scorer (same GPU)
#   other_hf  Llama-3.1-8B bf16 in every format; then Qwen3-8B with float32 weights and eager attention (native)
#   vllm      Qwen3-8B and Llama-3.1-8B through vLLM prompt_logprobs
# Every step resumes per item except vllm, which takes minutes. Submit (a GPU type can be pinned with --constraint):
#   bash pace/submit.sh qwen_audit.sbatch --export=ALL,AUDIT_STEP=qwen_hf --constraint=gpu-a40

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STEP="${AUDIT_STEP:?set AUDIT_STEP to qwen_hf, other_hf or vllm}"
D="${RUNS_DIR}/qwen_audit"
ITEMS="${D}/items.jsonl"
[[ -f "${ITEMS}" ]] || { echo "ERROR: ${ITEMS} missing; run scripts/qwen_audit/sample_items.py first" >&2; exit 2; }
echo "=== qwen audit ${STEP} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
QWEN=Qwen/Qwen3-8B
LLAMA=meta-llama/Llama-3.1-8B-Instruct
S=scripts/qwen_audit
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
echo "${GPU}" >"${D}/${STEP}_gpu.txt"

case "${STEP}" in
  qwen_hf)
    python "${S}/independent_score.py" --items "${ITEMS}" --model "${QWEN}" \
      --formats native,nothink,think_all,sysfold,zeroshot,plain --out "${D}/qwen_bf16.jsonl"
    python "${S}/repo_score.py" --items "${ITEMS}" --model "${QWEN}" --out "${D}/qwen_repo.jsonl"
    ;;
  other_hf)
    python "${S}/independent_score.py" --items "${ITEMS}" --model "${LLAMA}" \
      --formats native,sysfold,zeroshot,plain --out "${D}/llama_bf16.jsonl"
    python "${S}/independent_score.py" --items "${ITEMS}" --model "${QWEN}" --dtype float32 --attn eager \
      --formats native --out "${D}/qwen_fp32.jsonl"
    ;;
  vllm)
    python "${S}/vllm_score.py" --items "${ITEMS}" --model "${QWEN}" --formats native,nothink --out "${D}/qwen_vllm.jsonl"
    python "${S}/vllm_score.py" --items "${ITEMS}" --model "${LLAMA}" --formats native --out "${D}/llama_vllm.jsonl"
    ;;
  *) echo "unknown AUDIT_STEP=${STEP}" >&2; exit 2 ;;
esac
echo "=== qwen audit ${STEP} END $(date) ==="
