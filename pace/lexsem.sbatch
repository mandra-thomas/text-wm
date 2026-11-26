#!/bin/bash
#SBATCH --job-name=wf-lexsem
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/lexsem_%j.out
#SBATCH --error=logs/lexsem_%j.err
#
# Lexsem diagnostic (notes/lexsem_design.md). Submit only from the commit of the design's freeze tag, in the serve2
# stack (decided by the user, 2026-11-25). LX_ARM is one arm of wf.lexsem.ARMS (qwen_plain, qwen_block, llama_chat,
# qwen4b_plain, qwen4b_block, qwen14b_plain, qwen14b_block, llama3b_chat, qwen8b_base_plain, llama8b_base_plain), scored on an L40S, or its floor rescoring
# "<arm>_floor", which must be submitted with --constraint=gpu-a40 (a second environment, design section 7).
# Example: sbatch ... --export=ALL,LX_ARM=qwen_plain,SERVE_ENV=...,HF_HOME=...,HF_HUB_CACHE=... pace/lexsem.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D="${RUNS_DIR}/lexsem"
mkdir -p "${D}"
ARM="${LX_ARM:?set LX_ARM to an arm of wf.lexsem.ARMS, or <arm>_floor}"
echo "=== lexsem ${ARM} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' >"${D}/${ARM}_gpu.txt"
git rev-parse HEAD >"${D}/${ARM}_commit.txt"
python scripts/lexsem/score.py --arm "${ARM}" --out "${D}/${ARM}.jsonl"
echo "=== lexsem ${ARM} END $(date) ==="
