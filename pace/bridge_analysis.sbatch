#!/bin/bash
#SBATCH --job-name=wf-bridge-cpu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/bridge_cpu_%j.out
#SBATCH --error=logs/bridge_cpu_%j.err
#
# CPU work of the natural bridge (notes/bridge_design.md), in serve2:
#   BRIDGE_CPU=tests        tests/test_bridge.py with torch (the cached routine against a full pass)
#   BRIDGE_CPU=token_check  every prompt the bridge scores, tokenized with the real tokenizers (before the freeze)
#   BRIDGE_CPU=analysis     the frozen analysis, once, from a clean checkout at the freeze tag
#   sbatch --account=coc --qos=coc-ice --export=ALL,BRIDGE_CPU=analysis pace/bridge_analysis.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
activate_serve
echo "=== bridge ${BRIDGE_CPU:?set BRIDGE_CPU} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
case "${BRIDGE_CPU}" in
  tests) PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_bridge.py ;;
  token_check) python scripts/bridge_token_check.py --out "${RUNS_DIR}/bridge_token_check.json" ;;
  analysis) run_wf bridge_analysis ;;
  *) echo "unknown BRIDGE_CPU=${BRIDGE_CPU}" >&2; exit 2 ;;
esac
echo "=== bridge ${BRIDGE_CPU} END $(date) ==="
