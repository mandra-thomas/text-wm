#!/bin/bash
#SBATCH --job-name=pt-cpu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/posttrain_cpu_%j.out
#SBATCH --error=logs/posttrain_cpu_%j.err
# Post-training pilot, CPU work in the serve2 stack (dry runs, the analysis): PT_CMD is the command. Example:
#   sbatch --export=ALL,PT_CMD="python -m wf.posttrain_analysis --out x.json" pace/posttrain_cpu.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export SERVE_ENV=/storage/ice1/2/1/gmat3/envs/serve2
export WF_INIT_CONTEXTS=/storage/ice1/2/1/gmat3/hfcache/hub/datasets--X1AOX1A--LLMasWorldModels/snapshots/ff6ae2b924d1a49e4b89825913887f2ea96cb282/init_contexts/alfworld
activate_serve
echo "=== posttrain cpu START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
echo "cmd: ${PT_CMD:?set PT_CMD}"
bash -c "${PT_CMD}"
echo "=== posttrain cpu END $(date) ==="
