#!/bin/bash
#SBATCH --job-name=tc3-paddiag
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s)
#SBATCH --output=diag/paddiag_%j.out
#SBATCH --error=diag/paddiag_%j.err
# Round 3 thinking-arm padding diagnosis (not a study run): node-local Qwen3.5-9B by route A, then diag/think_pad_diag.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source pace/_common.sh
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
activate_serve
LOCAL_CACHE="${TMPDIR:-/tmp}/wf_hf_${SLURM_JOB_ID}"
trap 'rm -rf "${LOCAL_CACHE:?}"' EXIT
mkdir -p "${LOCAL_CACHE}"
export HF_HUB_CACHE="${LOCAL_CACHE}" HF_XET_CACHE="${LOCAL_CACHE}/xet" WF_WEIGHTS=local
export WF_WEIGHTS_RECORD="diag/prefetch_${SLURM_JOB_ID}.json"
echo "=== download $(date) host=$(hostname) ==="
python -m wf.textcraft.r3_prefetch qwen3.5-9b --cache "${LOCAL_CACHE}" --out "${WF_WEIGHTS_RECORD}"
export HF_HUB_OFFLINE=1
echo "=== diag start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
python diag/think_pad_diag.py "diag/paddiag_${SLURM_JOB_ID}.json"
echo "=== diag end $(date) ==="
