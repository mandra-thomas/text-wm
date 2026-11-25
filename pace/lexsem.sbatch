#!/bin/bash
#SBATCH --job-name=wf-lexsem
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/lexsem_%j.out
#SBATCH --error=logs/lexsem_%j.err
#
# Lexsem diagnostic (notes/lexsem_design.md). Submit only from the commit of the design's freeze tag.
# The cost estimate (about 19 GPU-minutes for the Qwen core) assumes an L40S; an A40 takes about twice as long.
#   LX_MODEL  Hugging Face id (default Qwen/Qwen3-8B)
#   LX_MODE   plain (default; Qwen without the empty think block) or chat (the optional Llama arm)
#   LX_CONDS  core (default) or a comma list of condition names (quote it in --export)
#   LX_NAME   output name under runs/lexsem/ (default qwen3-8b_plain)
# Example:
#   bash pace/submit.sh lexsem.sbatch --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D="${RUNS_DIR}/lexsem"
mkdir -p "${D}"
NAME="${LX_NAME:-qwen3-8b_plain}"
echo "=== lexsem ${NAME} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' >"${D}/${NAME}_gpu.txt"
git rev-parse HEAD >"${D}/${NAME}_commit.txt"
python scripts/lexsem/score.py --model "${LX_MODEL:-Qwen/Qwen3-8B}" --mode "${LX_MODE:-plain}" \
  --conditions "${LX_CONDS:-core}" --out "${D}/${NAME}.jsonl"
echo "=== lexsem ${NAME} END $(date) ==="
