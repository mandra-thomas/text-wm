#!/bin/bash
#SBATCH --job-name=wf-bridge
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
# The main runs and the smoke run are pinned to L40S, the cross-GPU rescore to A40 (the run itself also refuses any
# other GPU type, wf/bridge.py). Brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s)
#SBATCH --output=logs/bridge_%j.out
#SBATCH --error=logs/bridge_%j.err
#
# Natural ALFWorld bridge (wf/bridge.py, notes/bridge_design.md), in the serve2 environment (transformers 5), from a
# clean checkout at the design-freeze tag. One job per shard:
#   bash pace/submit.sh bridge.sbatch --export=ALL,BRIDGE_RUN=bridge_smoke_qwen3-8b
#   bash pace/submit.sh bridge.sbatch --export=ALL,BRIDGE_RUN=bridge_score_llama3.1-8b,WF_SHARD=0,WF_N_SHARDS=3
#   bash pace/submit.sh bridge.sbatch --export=ALL,BRIDGE_RUN=bridge_rescore_qwen3-8b --constraint='(gpu-a40)'
# After a timeout: the same command with WF_RESUME=1 in --export, on the same GPU type.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${BRIDGE_RUN:?set BRIDGE_RUN, e.g. bridge_smoke_qwen3-8b}"
case "${RUN}" in
  bridge_score_*|bridge_smoke_*|bridge_rescore_*) ;;
  *) echo "ERROR: BRIDGE_RUN must be bridge_score_<tag>, bridge_smoke_<tag> or bridge_rescore_<tag>, not ${RUN}" >&2; exit 2 ;;
esac

# serve2, whatever pace.env says SERVE_ENV is; transformers 5 reads HF_HUB_CACHE, pointed at the shared cache.
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
activate_serve
python - <<'PY'
import importlib.util, sys
import torch, transformers
if transformers.__version__ != "5.23.0" or not torch.__version__.startswith("2.14.1"):
    sys.exit(f"serve2 expected (transformers 5.23.0, torch 2.14.1), found {transformers.__version__}, {torch.__version__}")
if importlib.util.find_spec("kernels") is not None:
    sys.exit("the kernels package is importable in serve2; refusing")
print("transformers", transformers.__version__, "torch", torch.__version__)
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
