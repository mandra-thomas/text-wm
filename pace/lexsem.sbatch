#!/bin/bash
#SBATCH --job-name=wf-lexsem
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/lexsem_%j.out
#SBATCH --error=logs/lexsem_%j.err
#
# Lexsem diagnostic (notes/lexsem_design.md). Submit only from the commit of the design's freeze tag, in the serve2
# stack (decided by the user, 2026-11-25). LX_ARM picks one of the three runs:
#   qwen_plain  Qwen3-8B, plain text, core + ignore + position + reliability, obs and yes/no, failure text on core
#   qwen_block  Qwen3-8B, thinking-off chat format (empty think block), block8, yes/no
#   llama_chat  Llama-3.1-8B, its chat template, names_only, yes/no
# Example: sbatch ... --export=ALL,LX_ARM=qwen_plain,SERVE_ENV=...,HF_HOME=...,HF_HUB_CACHE=... pace/lexsem.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D="${RUNS_DIR}/lexsem"
mkdir -p "${D}"
ARM="${LX_ARM:?set LX_ARM to qwen_plain, qwen_block or llama_chat}"
echo "=== lexsem ${ARM} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' >"${D}/${ARM}_gpu.txt"
git rev-parse HEAD >"${D}/${ARM}_commit.txt"
S=scripts/lexsem/score.py
case "${ARM}" in
  qwen_plain) python "${S}" --mode plain --groups core,ignore,position,reliability --formats obs,yn \
                --fail-groups core --out "${D}/${ARM}.jsonl" ;;
  qwen_block) python "${S}" --mode block --groups block8 --formats yn --out "${D}/${ARM}.jsonl" ;;
  llama_chat) python "${S}" --model meta-llama/Llama-3.1-8B-Instruct --mode chat --groups names_only --formats yn \
                --out "${D}/${ARM}.jsonl" ;;
  *) echo "unknown LX_ARM=${ARM}" >&2; exit 2 ;;
esac
echo "=== lexsem ${ARM} END $(date) ==="
