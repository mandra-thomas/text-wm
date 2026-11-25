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
# computed or written. Three passes: Qwen3-8B in plain text with every condition group and both formats at batch
# sizes 1, 8, 16 and 32; Qwen3-8B in its thinking-off chat format (the empty think block), the reduced condition
# set, yes/no; Llama-3.1-8B in its chat template, the reduced set, yes/no.

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
python "${S}" --mode plain --groups core,ignore,position,reliability --smoke 5 --batch-sizes 1,8,16,32 \
  --out "${D}/smoke_qwen_plain.json"
python "${S}" --mode block --groups names_only --formats yn --smoke 5 --batch-sizes 1,16 \
  --out "${D}/smoke_qwen_block.json"
python "${S}" --model meta-llama/Llama-3.1-8B-Instruct --mode chat --groups names_only --formats yn --smoke 5 \
  --batch-sizes 1,16 --out "${D}/smoke_llama_chat.json"
echo "=== lexsem smoke END $(date) ==="
