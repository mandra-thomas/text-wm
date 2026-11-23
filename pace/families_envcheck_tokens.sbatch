#!/bin/bash
#SBATCH --job-name=wf-envtok
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/envtok_%j.out
#SBATCH --error=logs/envtok_%j.err
#
# Wave B environment check, part 1 (design section 8; wf/tools/envcheck_tokens.py): token ids of every prompt and
# candidate for Mistral 7B v0.3 and Qwen3-8B, in a copy of the study environment's tokenizer stack (built here, in
# this node's local temporary folder, and gone when the job ends; the study environment itself is not touched)
# and in serve2. Writes runs/envcheck_tokens/compare_<tag>.json, which families_envcheck reads.
#   bash pace/submit.sh families_envcheck_tokens.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
SERVE2="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
OUT="${RUNS_DIR}/envcheck_tokens"
mkdir -p "${OUT}"
WORK="${TMPDIR:-/tmp}/wf_envtok_${SLURM_JOB_ID:-local}"
cleanup() { [[ -n "${WORK}" && -d "${WORK}" ]] && rm -rf "${WORK:?}"; }
trap cleanup EXIT
mkdir -p "${WORK}"
export UV_CACHE_DIR="${WORK}/uvcache"
module load python/3.11.9 2>/dev/null || true
UV="${UV:-${HOME}/.local/bin/uv}"
# The study environment's versions (notes/reproducing.md): Python 3.11, torch 2.6.0, transformers 4.51.3, tokenizers 0.21.4.
"${UV}" venv --python 3.11 "${WORK}/old"
"${UV}" pip install --python "${WORK}/old/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
"${UV}" pip install --python "${WORK}/old/bin/python" "transformers==4.51.3" "tokenizers==0.21.4" jinja2 sentencepiece \
  protobuf numpy scipy pyyaml requests tqdm
"${UV}" pip freeze --python "${WORK}/old/bin/python" > "${OUT}/old_stack_freeze.txt"
for TAG in mistral-7b-v0.3 qwen3-8b; do
  echo "=== ${TAG} $(date) ==="
  HF_HUB_OFFLINE=1 TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" "${WORK}/old/bin/python" -m wf.tools.envcheck_tokens dump "${TAG}" \
    --cache "${TRANSFORMERS_CACHE}" --out "${OUT}/old_${TAG}.json"
  HF_HUB_OFFLINE=1 HF_HUB_CACHE="${TRANSFORMERS_CACHE}" "${SERVE2}/bin/python" -m wf.tools.envcheck_tokens dump "${TAG}" \
    --cache "${TRANSFORMERS_CACHE}" --out "${OUT}/serve2_${TAG}.json"
  "${SERVE2}/bin/python" -m wf.tools.envcheck_tokens compare "${OUT}/old_${TAG}.json" "${OUT}/serve2_${TAG}.json" \
    --out "${OUT}/compare_${TAG}.json" || true
done
echo "=== envcheck tokens END $(date) ==="
