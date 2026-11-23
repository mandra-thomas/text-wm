#!/bin/bash
#SBATCH --job-name=wf-envprec
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Two model loads (about 2 min each) and 450 forward passes per environment; well under 30 minutes.
#SBATCH --time=0:40:00
#SBATCH --gres=gpu:1
# Qwen3-8B's earlier run and its serve2 rescoring both ran on an L40S (wf.families.ENVCHECK_GPU). Brackets because
# ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s)
#SBATCH --output=logs/envprec_%j.out
#SBATCH --error=logs/envprec_%j.err
#
# Precision diagnostic for the wave B environment check (scripts/families_envcheck_precision.py; design note
# section 8, amendment): the same 50 items scored in the study environment (transformers 4.51.3, read-only use of its
# Python) and in serve2, keeping bf16 and float64-head logits. No wave B model is loaded.
#   bash pace/submit.sh families_envcheck_precision.sbatch   (account, QOS and partition from pace.env)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

OUT="${RUNS_DIR}/families_envcheck_precision"
mkdir -p "${OUT}"
OLD_ENV="${STUDY_SERVE_ENV:-/storage/ice1/2/1/gmat3/text-wm/.envs/serve}"
NEW_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
# one cache for both environments, as the scoring jobs read it; offline, so nothing downloads
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
export HF_HUB_OFFLINE=1
# read-only use of the study environment: no bytecode written into its site-packages
export PYTHONDONTWRITEBYTECODE=1

nvidia-smi || true
echo "=== old environment (${OLD_ENV}) $(date) ==="
( export SERVE_ENV="${OLD_ENV}"; activate_serve
  python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
  python scripts/families_envcheck_precision.py score --label old --out "${OUT}/old.jsonl" )
echo "=== serve2 (${NEW_ENV}) $(date) ==="
( export SERVE_ENV="${NEW_ENV}"; activate_serve
  python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
  python scripts/families_envcheck_precision.py score --label new --out "${OUT}/new.jsonl"
  python scripts/families_envcheck_precision.py compare --old "${OUT}/old.jsonl" --new "${OUT}/new.jsonl" \
    --archived-old "${RUNS_DIR}/size_score_qwen3-8b" --archived-new "${RUNS_DIR}/family_envcheck_qwen3-8b" \
    --out "${OUT}/report.json" )
echo "=== END $(date) ==="
