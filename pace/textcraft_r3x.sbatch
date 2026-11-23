#!/bin/bash
#SBATCH --job-name=wf-tc3x
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Loading goes through CPU memory first; the 27B and 31B models need --mem=128G.
#SBATCH --mem=64G
# About 21,000 forward passes per shard of a core model (two shards); estimates in notes/textcraft_r3_design.md
# section 11. Scores resume if cut off (WF_RESUME=1, same GPU type).
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
# Pinned GPU types (wf/textcraft/r3.py GPU_PIN, design section 10): L40S for the 7-12B models and the thinking arm;
# an A100 80 GB for the 27B and 31B models (--constraint=A100-80GB --mem=128G), or two L40S on one node as the
# declared alternative (--gres=gpu:l40s:2 --constraint=gpu-l40s), the same for every shard of a model. The
# environment check runs one job per job of the committed plan (wf/config/textcraft_r3_envcheck_plan.json,
# WF_ENVCHECK_JOB), on exactly that job's GPU name (design 14.M4); the fallback rescoring on exactly one L40S.
# Brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s)
#SBATCH --output=logs/tc3x_%j.out
#SBATCH --error=logs/tc3x_%j.err
#
# TextCraft round 3, deviation D-G1 (notes/textcraft_r3_deviation_gemma_e4b.md): the exploratory run
# textcraft_r3x_gemma4-e4b-it, outside every family, scored as a core model is (serve2, bf16, SDPA, node-local weights,
# the round-3 gate); its attention gate is the committed float32 diagnostic. It runs from a checkout at the deviation
# tag textcraft-r3-deviation-t1-2026-11-23 (the job checks this at start). Two shards, one L40S each:
#   for s in 0 1; do bash pace/submit.sh textcraft_r3x.sbatch --account=coc --qos=coc-ice \
#     --export=ALL,TC3_RUN=textcraft_r3x_gemma4-e4b-it,WF_WEIGHTS=local,WF_SHARD=$s,WF_N_SHARDS=2; done
# Weights (route A, decision D5): WF_WEIGHTS=local downloads the pinned snapshot to this node's local disk, hashes
# every file against the committed manifest (wf/config/weights_manifests) and stops on any difference; the copy is
# removed when the job ends.


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${TC3_RUN:?set TC3_RUN=textcraft_r3x_gemma4-e4b-it}"
case "${RUN}" in
  textcraft_r3x_*) TAG="${RUN#textcraft_r3x_}" ;;
  *) echo "ERROR: this script runs only textcraft_r3x_ runs (D-G1), not ${RUN}" >&2; exit 2 ;;
esac

# serve2, whatever pace.env says SERVE_ENV is (that one is the study's environment, transformers 4.51.3).
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
activate_serve
python - <<'PY'
import importlib.util, sys
import transformers
if int(transformers.__version__.split(".")[0]) < 5:
    sys.exit(f"serve2 expected, found transformers {transformers.__version__}")
if importlib.util.find_spec("kernels") is not None:
    sys.exit("the kernels package is importable in serve2; refusing")
print("transformers", transformers.__version__)
PY

LOCAL_CACHE=""
cleanup() {
  if [[ -n "${LOCAL_CACHE}" && -d "${LOCAL_CACHE}" ]]; then
    rm -rf "${LOCAL_CACHE:?}"
  fi
}
trap cleanup EXIT
if [[ "${WF_WEIGHTS:-shared}" == "local" ]]; then
  [[ -n "${TAG}" ]] || { echo "ERROR: WF_WEIGHTS=local is for round-3 models" >&2; exit 2; }
  LOCAL_CACHE="${TMPDIR:-/tmp}/wf_hf_${SLURM_JOB_ID:-local}"
  mkdir -p "${LOCAL_CACHE}"
  df -h "${LOCAL_CACHE}" || true
  export HF_HUB_CACHE="${LOCAL_CACHE}"
  export HF_XET_CACHE="${LOCAL_CACHE}/xet"
  export WF_WEIGHTS=local
  export WF_WEIGHTS_RECORD="logs/tc3_prefetch_${SLURM_JOB_ID:-local}.json"
  echo "=== downloading ${TAG} into ${LOCAL_CACHE} $(date) ==="
  python -m wf.textcraft.r3_prefetch "${TAG}" --cache "${LOCAL_CACHE}" --out "${WF_WEIGHTS_RECORD}"
  echo "=== download and hash check done $(date) ==="
fi
export HF_HUB_OFFLINE=1

echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) cache=${HF_HUB_CACHE} ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
