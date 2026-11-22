#!/bin/bash
#SBATCH --job-name=wf-familyB
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Loading goes through CPU memory first; the 27B and 31B models need --mem=128G.
#SBATCH --mem=64G
# About 43,700 forward passes per model (one shard). Estimates (notes/families_waveb_design.md section 12): 1.5-3 h
# for the 8-9B models, 4-6 h for the 27B and 31B models; scores resume if cut off (WF_RESUME=1).
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
# The small models fit any of these; the 27B and 31B models need --constraint=A100-80GB (or --gres=gpu:l40s:2 with
# --constraint=gpu-l40s to shard over two cards). The environment check runs on its model's earlier GPU type:
# --constraint=gpu-a40 for Mistral 7B, --constraint=gpu-l40s for Qwen3-8B. Brackets because ICE appends &cuda13.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/familyB_%j.out
#SBATCH --error=logs/familyB_%j.err
#
# Model families, wave B (wf/families.py, notes/families_waveb_design.md), in the serve2 environment (transformers 5):
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_score_gemma4-e4b-it
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_envcheck_qwen3-8b --constraint=gpu-l40s
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_score_qwen3.8-27b,WF_WEIGHTS=local \
#        --constraint=A100-80GB --mem=128G --time=10:00:00
# Weights: by default from the shared cache (prefetched first; scoring runs offline and stops if a file is missing).
# WF_WEIGHTS=local downloads the pinned snapshot into this node's local disk first and scores from there; the copy is
# removed when the job ends, so no quota is used (section 13). Nothing is scored before the design-freeze tag.
# After a timeout: the same command with WF_RESUME=1 in --export (same GPU type).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${FAMILY_RUN:?set FAMILY_RUN, e.g. family_score_gemma4-e4b-it}"
case "${RUN}" in
  family_score_*) TAG="${RUN#family_score_}" ;;
  family_placement_*) TAG="${RUN#family_placement_}" ;;
  family_envcheck_*) TAG="" ;;  # earlier models, already in the shared cache
  *) echo "ERROR: FAMILY_RUN must be family_score_<tag>, family_placement_<tag> or family_envcheck_<tag>, not ${RUN}" >&2; exit 2 ;;
esac

# serve2, whatever pace.env says SERVE_ENV is (that one is the study's environment, transformers 4.51.3).
export SERVE_ENV="${SERVE2_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
# transformers 5 reads HF_HUB_CACHE, not TRANSFORMERS_CACHE: point it at the folder transformers 4 reads, so both
# environments use one cache and nothing already cached is downloaded again (design section 4).
export HF_HUB_CACHE="${TRANSFORMERS_CACHE}"
activate_serve
python - <<'PY'
import importlib.util, sys
import transformers
if int(transformers.__version__.split(".")[0]) < 5:
    sys.exit(f"serve2 expected, found transformers {transformers.__version__}")
if importlib.util.find_spec("kernels") is not None:
    sys.exit("the kernels package is importable in serve2; refusing (design section 4)")
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
  [[ -n "${TAG}" ]] || { echo "ERROR: WF_WEIGHTS=local is for wave B models" >&2; exit 2; }
  LOCAL_CACHE="${TMPDIR:-/tmp}/wf_hf_${SLURM_JOB_ID:-local}"
  mkdir -p "${LOCAL_CACHE}"
  df -h "${LOCAL_CACHE}" || true
  export HF_HUB_CACHE="${LOCAL_CACHE}"
  echo "=== downloading ${TAG} into ${LOCAL_CACHE} $(date) ==="
  python -m wf.tools.prefetch_models "${TAG}" --out "logs/familyB_prefetch_${SLURM_JOB_ID:-local}.json"
  echo "=== download done $(date) ==="
fi
export HF_HUB_OFFLINE=1

echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) cache=${HF_HUB_CACHE} ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
