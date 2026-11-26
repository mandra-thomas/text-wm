#!/bin/bash
#SBATCH --job-name=pt-access
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU only (post-training pilot, step 1; notes/posttraining_design.md): checks that HF_TOKEN reaches every pilot
# checkpoint at its pinned revision (HEAD requests on the weight files, no weight download) and checks the official
# tokenizer files. Downloads only tokenizer and config files, into tokcache/ inside this checkout.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/posttrain_access_%j.out
#SBATCH --error=logs/posttrain_access_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
activate_serve
echo "=== posttrain access START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
python scripts/posttrain/access_check.py --cache "${PROJECT_ROOT}/tokcache" --out "logs/posttrain_access_${SLURM_JOB_ID:-local}.json"
echo "=== posttrain access END $(date) ==="
