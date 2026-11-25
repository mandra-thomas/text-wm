#!/bin/bash
#SBATCH --job-name=wf-lexsem-smoke
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/lexsem_smoke_%j.out
#SBATCH --error=logs/lexsem_smoke_%j.err
#
# Lexsem smoke run (notes/lexsem_design.md section 11): timing and validity figures only, on 5 items. No effect is
# computed or written. The three runs of pace/lexsem.sbatch, scored exactly as there (one unpadded sequence at a time),
# on 5 items each.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D="${RUNS_DIR}/lexsem"
mkdir -p "${D}"
echo "=== lexsem smoke START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
S=scripts/lexsem/score.py
python "${S}" --mode plain --groups core,ignore,position,reliability --formats obs,yn --fail-groups core --smoke 5 \
  --out "${D}/smoke_qwen_plain.json"
python "${S}" --mode block --groups block8 --formats yn --smoke 5 --out "${D}/smoke_qwen_block.json"
python "${S}" --model meta-llama/Llama-3.1-8B-Instruct --mode chat --groups names_only --formats yn --smoke 5 \
  --out "${D}/smoke_llama_chat.json"
echo "=== lexsem smoke END $(date) ==="
