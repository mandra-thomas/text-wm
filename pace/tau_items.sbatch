#!/bin/bash
#SBATCH --job-name=wf-tau-items
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU only. Locally the build takes about 20 s, the full verification about 10 min and the token check on every
# scored item about an hour per model, under 2 GB.
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tau_items_%j.out
#SBATCH --error=logs/tau_items_%j.err
#
# τ-bench retail items (notes/tau_design.md): build, full verification, and the tokenizer checks:
#   sbatch --account=coc --qos=coc-ice --export=ALL,TAU_NATURAL_DIR=<tau2-bench>/data/tau2/results/final pace/tau_items.sbatch
# Then scoring, one shard of the users per GPU job, with the generic HF scoring script:
#   sbatch ... --export=ALL,RQ2_RUN=tau_score,WF_SHARD=0,WF_N_SHARDS=16 pace/tau_score.sbatch (after the smoke run)
# and, when every shard says "shard complete": python -m wf.runs tau_analysis (no GPU).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
activate_serve
export TAU_NATURAL_DIR="${TAU_NATURAL_DIR:-${PROJECT_ROOT}/third_party/tau2-bench/data/tau2/results/final}"
export HF_HUB_OFFLINE=1
echo "=== tau items START $(date) host=$(hostname) ==="
python -m wf.runs tau_build
python -m wf.runs tau_verify
python -m wf.tau.token_check "${RUNS_DIR}/tau_build/items.jsonl" "${RUNS_DIR}/tau_build/fewshot.json" \
  --natural "${RUNS_DIR}/tau_build/natural_items.jsonl" --cache-dir "${TRANSFORMERS_CACHE}" > "${RUNS_DIR}/tau_build/token_check.json"
echo "=== tau items END $(date) ==="
