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
# Section 8 amendment (2026-11-23): the same comparison for every older model of the serve2 rescoring, into its own
# folder (never the environment check's), with the rescoring's tokenizer settings:
#   bash pace/submit.sh families_envcheck_tokens.sbatch --export=ALL,ENVTOK_RESCORE=1
# Qwen3-32B's files are in no ICE cache: its tokenizer and config files (no weights) are downloaded at the pinned
# revision into this job's node-local folder and read from there by both stacks.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
SERVE2="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
if [[ "${ENVTOK_RESCORE:-0}" == "1" ]]; then
  OUT="${RUNS_DIR}/rescore_tokens"
  TAGS="mistral-7b-v0.3 qwen3-8b gemma2-9b-it llama3.1-8b qwen3-32b"
  RESCORE=(--rescore)
else
  OUT="${RUNS_DIR}/envcheck_tokens"
  TAGS="mistral-7b-v0.3 qwen3-8b"
  RESCORE=()
fi
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
for TAG in ${TAGS}; do
  echo "=== ${TAG} $(date) ==="
  CACHE="${TRANSFORMERS_CACHE}"
  if [[ "${TAG}" == "qwen3-32b" ]]; then
    CACHE="${WORK}/hf"
    "${SERVE2}/bin/python" - "${CACHE}" <<'PY'
import sys
from huggingface_hub import snapshot_download
from wf.constants import REVISIONS
p = snapshot_download("Qwen/Qwen3-32B", revision=REVISIONS["Qwen/Qwen3-32B"], cache_dir=sys.argv[1],
                      allow_patterns=["*.json", "merges.txt", "*.jinja"], ignore_patterns=["*.safetensors*"])
print("tokenizer files:", p)
PY
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_CACHE="${CACHE}" "${WORK}/old/bin/python" -m wf.tools.envcheck_tokens dump "${TAG}" \
    ${RESCORE[@]+"${RESCORE[@]}"} --cache "${CACHE}" --out "${OUT}/old_${TAG}.json"
  HF_HUB_OFFLINE=1 HF_HUB_CACHE="${CACHE}" "${SERVE2}/bin/python" -m wf.tools.envcheck_tokens dump "${TAG}" \
    ${RESCORE[@]+"${RESCORE[@]}"} --cache "${CACHE}" --out "${OUT}/serve2_${TAG}.json"
  "${SERVE2}/bin/python" -m wf.tools.envcheck_tokens compare "${OUT}/old_${TAG}.json" "${OUT}/serve2_${TAG}.json" \
    --out "${OUT}/compare_${TAG}.json" || true
done
echo "=== envcheck tokens END $(date) ==="
