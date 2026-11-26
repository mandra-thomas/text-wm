#!/bin/bash
#SBATCH --job-name=wf-lexsem-smoke
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/lexsem_smoke_%j.out
#SBATCH --error=logs/lexsem_smoke_%j.err
#
# Lexsem smoke run (notes/lexsem_design.md section 13): timing and validity figures only, on 5 items, for every
# arm of wf.lexsem.ARMS (LX_ARMS, space-separated, to choose some). No effect is computed or written. Each arm is
# scored exactly as in the full run: one unpadded sequence at a time.

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
ARMS="${LX_ARMS:-qwen_plain qwen_block llama_chat qwen4b_plain qwen4b_block qwen14b_plain qwen14b_block llama3b_chat qwen8b_base_plain llama8b_base_plain}"
for ARM in ${ARMS}; do
  python scripts/lexsem/score.py --arm "${ARM}" --smoke 5 --out "${D}/smoke_${ARM}.json"
done
echo "=== lexsem smoke END $(date) ==="
