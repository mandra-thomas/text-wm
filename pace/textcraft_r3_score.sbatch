#!/bin/bash
#SBATCH --job-name=wf-tc3
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
#SBATCH --output=logs/tc3_%j.out
#SBATCH --error=logs/tc3_%j.err
#
# TextCraft round 3 (notes/textcraft_r3_design.md) in the serve2 environment (transformers 5). Run order
# (wf.textcraft.r3.RUN_ORDER): the environment check, the core smoke runs (the Gemma fallback rule), the core
# models; then the extension, the format checks and the thinking arm as GPUs allow. Nothing is scored before the
# round-3 design-freeze tag exists, and every job runs from a checkout at the tag's commit (detached, never
# pulled); the job checks this at start (design 14.S12).
#   bash pace/submit.sh textcraft_r3_score.sbatch --export=ALL,TC3_RUN=textcraft_r3_envcheck_qwen3-8b,WF_ENVCHECK_JOB=0 \
#        --constraint=<the plan's GPU for job 0, e.g. gpu-l40s>
#   bash pace/submit.sh textcraft_r3_score.sbatch --export=ALL,TC3_RUN=textcraft_r3_smoke_gemma4-12b-it,WF_WEIGHTS=local
#   for s in 0 1; do bash pace/submit.sh textcraft_r3_score.sbatch \
#     --export=ALL,TC3_RUN=textcraft_r3_score_qwen3.5-9b,WF_WEIGHTS=local,WF_SHARD=$s,WF_N_SHARDS=2; done
#   bash pace/submit.sh textcraft_r3_score.sbatch --export=ALL,TC3_RUN=textcraft_r3_score_qwen3.8-27b,WF_WEIGHTS=local,WF_SHARD=0,WF_N_SHARDS=4 \
#        --constraint=A100-80GB --mem=128G
#   bash pace/submit.sh textcraft_r3_score.sbatch --export=ALL,TC3_RUN=textcraft_r3_think_qwen3.5-9b,WF_WEIGHTS=local,WF_SHARD=0,WF_N_SHARDS=2 --time=8:00:00
# Weights (route A, decision D5): WF_WEIGHTS=local downloads the pinned snapshot to this node's local disk, hashes
# every file against the committed manifest (wf/config/weights_manifests) and stops on any difference; the copy is
# removed when the job ends. The earlier models (envcheck, rescore) are read from the shared cache.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${TC3_RUN:?set TC3_RUN, e.g. textcraft_r3_score_qwen3.5-9b}"
case "${RUN}" in
  textcraft_r3_score_*) TAG="${RUN#textcraft_r3_score_}" ;;
  textcraft_r3_smoke_*) TAG="${RUN#textcraft_r3_smoke_}" ;;
  textcraft_r3_placement_*) TAG="${RUN#textcraft_r3_placement_}" ;;
  textcraft_r3_nopreserve_*) TAG="${RUN#textcraft_r3_nopreserve_}" ;;
  textcraft_r3_think_smoke_*|textcraft_r3_think_*) TAG="qwen3.5-9b" ;;
  textcraft_r3_envcheck_*|textcraft_r3_rescore_*) TAG="" ;;  # earlier models, already in the shared cache
  *) echo "ERROR: TC3_RUN must be a textcraft_r3_ scoring run, not ${RUN}" >&2; exit 2 ;;
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
