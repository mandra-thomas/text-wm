#!/bin/bash
#SBATCH --job-name=wf-tau-family-items
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU only (notes/tau_families_design.md §6.2): rebuild and verify the τ-bench items at this commit (they must
# reproduce the pins), then the per-model token check on every scored prompt, one process per model (3 workers:
# Mistral, Llama 3.2 3B, and Gemma 2's lengths only).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/tau_family_items_%j.out
#SBATCH --error=logs/tau_family_items_%j.err
#
#   sbatch --account=coc --qos=coc-ice --export=ALL,TAU_NATURAL_DIR=<tau2-bench>/data/tau2/results/final pace/tau_family_items.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
activate_serve
export TAU_NATURAL_DIR="${TAU_NATURAL_DIR:-${PROJECT_ROOT}/third_party/tau2-bench/data/tau2/results/final}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== tau family items START $(date) host=$(hostname) ==="
python -m wf.runs tau_build
python -m wf.runs tau_verify
python -m wf.runs tau_family_token_check
echo "=== tau family items END $(date) ==="
