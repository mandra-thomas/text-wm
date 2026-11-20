#!/bin/bash
#SBATCH --job-name=wf-prefetch
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU only: downloads about 42 GB (Gemma 2 9B 18.5, Mistral 7B 14.5, Llama 3.2 3B 6.4, 1B 2.5) and checks each
# tokenizer offline.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/prefetch_%j.out
#SBATCH --error=logs/prefetch_%j.err
#
# Downloads exactly the pinned revisions (wf.constants.REVISIONS) of the wave A models into the cache transformers
# reads (TRANSFORMERS_CACHE, set in pace/_common.sh), using HF_TOKEN from pace/pace.env, which is never printed.
# Prints one line per model: OK, or the HTTP status of the failure (403: access not granted yet). A refused model
# does not stop the others; the job exits 1 if any failed. Then --check runs the tokenization gate on the golden
# item with each downloaded tokenizer.
#   bash pace/submit.sh families_prefetch.sbatch
#   bash pace/submit.sh families_prefetch.sbatch --export=ALL,PREFETCH_TAGS="llama3.2-1b llama3.2-3b"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
activate_serve

echo "=== prefetch START $(date) host=$(hostname) ==="
# shellcheck disable=SC2086
python -m wf.tools.prefetch_models ${PREFETCH_TAGS:-} --check --out "logs/prefetch_${SLURM_JOB_ID:-local}.json"
echo "=== prefetch END $(date) ==="
