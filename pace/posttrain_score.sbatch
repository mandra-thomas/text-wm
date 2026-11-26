#!/bin/bash
#SBATCH --job-name=pt-score
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --constraint=gpu-l40s
#SBATCH --output=logs/posttrain_%j.out
#SBATCH --error=logs/posttrain_%j.err
#
# Post-training pilot (notes/posttraining_design.md). One checkpoint per job: PT_CKPT (a key of
# wf.posttrain.CHECKPOINTS) and PT_ARMS (its arms separated by colons: sbatch --export splits values at commas). Main and bridge arms on an L40S; floor arms
# ("<arm>_floor") must be submitted with --constraint=gpu-a40. PT_SMOKE=5 runs the smoke (timing and validity
# only); PT_RESUME=1 continues a timed-out job on the same GPU type. The serve2 stack is named here. Weights are
# downloaded at the pinned revision into node-local disk (the shared scratch quota is nearly full) and removed at
# the end; the token is HF_TOKEN from pace/pace.env and is never printed.
#   sbatch --export=ALL,PT_CKPT=tulu-sft,PT_ARMS=tulu-sft_plain:tulu-sft_native:tulu-sft_w2w pace/posttrain_score.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export SERVE_ENV=/storage/ice1/2/1/gmat3/envs/serve2
LOCAL="/tmp/wf_pt_${SLURM_JOB_ID:-local}"
mkdir -p "${LOCAL}"
trap 'rm -rf "${LOCAL}"' EXIT
# PT_HF_CACHE names a cache that already holds the checkpoint (the Llama-3.1-8B base sits in the shared cache)
export HF_HUB_CACHE="${PT_HF_CACHE:-${LOCAL}/hub}" TRANSFORMERS_CACHE="${PT_HF_CACHE:-${LOCAL}/hub}"
# Word2World's full-state contexts (bridge arms): the LLMasWorldModels dataset snapshot in the shared cache
export WF_INIT_CONTEXTS=/storage/ice1/2/1/gmat3/hfcache/hub/datasets--X1AOX1A--LLMasWorldModels/snapshots/ff6ae2b924d1a49e4b89825913887f2ea96cb282/init_contexts/alfworld
CKPT="${PT_CKPT:?set PT_CKPT}"
ARMS="${PT_ARMS:?set PT_ARMS}"
ARMS="${ARMS//:/,}"
echo "=== posttrain ${CKPT} [${ARMS}] START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
python -c 'import torch, transformers; v=(torch.__version__, transformers.__version__); print("stack", v); assert v == ("2.14.1+cu126", "5.23.0"), v'
EXTRA=()
[[ -n "${PT_SMOKE:-}" ]] && EXTRA+=(--smoke "${PT_SMOKE}")
[[ "${PT_RESUME:-}" == "1" ]] && EXTRA+=(--resume)
[[ "${PT_NO_REUSE:-}" == "1" ]] && EXTRA+=(--no-reuse)
python -m wf.posttrain_score --ckpt "${CKPT}" --arms "${ARMS}" ${EXTRA[@]+"${EXTRA[@]}"}
echo "=== posttrain ${CKPT} END $(date) ==="
