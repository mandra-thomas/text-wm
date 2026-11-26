#!/bin/bash
#SBATCH --job-name=wf-bayesenv
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/bayesenv_%j.out
#SBATCH --error=logs/bayesenv_%j.err
#
# Controlled Bayesian environment (notes/bayesenv_design.md). BE_RUN names the run (wf/config/<BE_RUN>.yaml):
#   bayesenv_smoke_<model>, bayesenv_checks_<model>, bayesenv_score_<model> (with WF_SHARD and WF_N_SHARDS),
#   bayesenv_xgpu_<model> (submit with --constraint=gpu-a40). Every score is one unpadded sequence at a time.
# Example: bash pace/submit.sh bayesenv.sbatch --export=ALL,BE_RUN=bayesenv_score_qwen3-8b,WF_SHARD=0,WF_N_SHARDS=4
# A resumed shard (WF_RESUME=1) must keep the shard's GPU type: the run refuses a different GPU or environment.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN="${BE_RUN:?set BE_RUN}"
echo "=== bayesenv ${RUN} shard=${WF_SHARD:-0}/${WF_N_SHARDS:-1} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
run_wf "${RUN}"
echo "=== bayesenv ${RUN} END $(date) ==="
