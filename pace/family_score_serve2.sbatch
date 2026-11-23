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
# Model families, wave B (wf/families.py, notes/families_waveb_design.md), in the serve2 environment (transformers 5).
# Run order (pre-registered after review 1): the environment check, the four small models, Qwen3.8 27B and its
# preserve_thinking check, Gemma 4 31B last (wf.families.RUN_ORDER_B).
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_envcheck_mistral-7b-v0.3 --constraint=gpu-a40
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_envcheck_qwen3-8b --constraint=gpu-l40s
# Section 8 amendment (2026-11-23), the older comparison models rescored in serve2 on their pinned types:
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_rescore_mistral-7b-v0.3 --constraint='(gpu-a40)'
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_rescore_qwen3-8b --constraint='(gpu-l40s)'
#   (the same for family_rescore_gemma2-9b-it and family_rescore_llama3.1-8b on gpu-l40s)
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_rescore_qwen3-32b,WF_WEIGHTS=local,WF_SHARD=0,WF_N_SHARDS=2 \
#        --constraint=A100-80GB --mem=128G --time=8:00:00   (and WF_SHARD=1)
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_score_gemma4-e4b-it,WF_WEIGHTS=local
#   bash pace/submit.sh family_score_serve2.sbatch --export=ALL,FAMILY_RUN=family_score_qwen3.8-27b,WF_WEIGHTS=local \
#        --constraint=A100-80GB --mem=128G --time=10:00:00
# The environment check's GPU type is pinned (wf.families.ENVCHECK_GPU) and checked below before anything loads.
# Each scoring run refuses to load if the visible GPUs hold less than the model's gpu_min_gb or, for the 27B and 31B
# models, if the job has less than --mem=128G (wf.families.require_resources).
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
ENVTAG=""
RESCORETAG=""
case "${RUN}" in
  family_score_*) TAG="${RUN#family_score_}" ;;
  family_placement_*) TAG="${RUN#family_placement_}" ;;
  family_nopreserve_*) TAG="${RUN#family_nopreserve_}" ;;  # review 1, S8: Qwen3.8 with preserve_thinking=False
  family_envcheck_*) TAG=""; ENVTAG="${RUN#family_envcheck_}" ;;  # earlier models, already in the shared cache
  # section 8 amendment (2026-11-23): an older comparison model rescored in serve2 on every item and condition
  family_rescore_*) TAG=""; RESCORETAG="${RUN#family_rescore_}" ;;
  *) echo "ERROR: FAMILY_RUN must be family_score_<tag>, family_placement_<tag>, family_nopreserve_<tag>, family_envcheck_<tag> or family_rescore_<tag>, not ${RUN}" >&2; exit 2 ;;
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
# Review 1 (S4): the environment check rescoring runs only on its model's earlier GPU type (wf.families.ENVCHECK_GPU:
# Mistral 7B on an A40, Qwen3-8B on an L40S); a job that landed elsewhere stops here, and families_envcheck fails a
# shard whose recorded GPU differs.
if [[ -n "${ENVTAG}" ]]; then
  python - "${ENVTAG}" <<'PY'
import sys
from wf.families import ENVCHECK_GPU
from wf.provenance import gpu_names
want, have = ENVCHECK_GPU[sys.argv[1]], gpu_names()
if have != [want]:
    sys.exit(f"the environment check of {sys.argv[1]} runs on one {want} only; this job has {have}")
print("GPU", have)
PY
fi
# Section 8 amendment: each serve2 rescoring runs on its pinned GPU type (wf.families.RESCORE_GPU), and a model that is
# in no ICE cache (RESCORE_NODE_LOCAL: Qwen3-32B) only with node-local weights checked against its manifest.
if [[ -n "${RESCORETAG}" ]]; then
  python - "${RESCORETAG}" "${WF_WEIGHTS:-shared}" <<'PY'
import sys
from wf.families import RESCORE_GPU, RESCORE_NODE_LOCAL
from wf.provenance import gpu_names
tag, weights = sys.argv[1], sys.argv[2]
want, have = RESCORE_GPU[tag], gpu_names()
if have != [want]:
    sys.exit(f"the serve2 rescoring of {tag} runs on one {want} only; this job has {have}")
if tag in RESCORE_NODE_LOCAL and weights != "local":
    sys.exit(f"{tag} is in no ICE cache: rescore it with WF_WEIGHTS=local (route A)")
print("GPU", have)
PY
fi

LOCAL_CACHE=""
cleanup() {
  if [[ -n "${LOCAL_CACHE}" && -d "${LOCAL_CACHE}" ]]; then
    rm -rf "${LOCAL_CACHE:?}"
  fi
}
trap cleanup EXIT
if [[ "${WF_WEIGHTS:-shared}" == "local" ]]; then
  [[ -n "${TAG}" || -n "${RESCORETAG}" ]] || { echo "ERROR: WF_WEIGHTS=local is for wave B models and rescorings" >&2; exit 2; }
  TAG="${TAG:-${RESCORETAG}}"
  LOCAL_CACHE="${TMPDIR:-/tmp}/wf_hf_${SLURM_JOB_ID:-local}"
  mkdir -p "${LOCAL_CACHE}"
  df -h "${LOCAL_CACHE}" || true
  # scoring reads the weights from here; the download goes here by name (--cache), never to /storage (review 1, S1)
  export HF_HUB_CACHE="${LOCAL_CACHE}"
  export HF_XET_CACHE="${LOCAL_CACHE}/xet"  # review 1, S3: hf-xet's chunk cache stays node-local too
  export WF_WEIGHTS=local
  # review 1, M3: free space against the manifest first; after the download every file is hashed against the
  # committed manifest (wf/config/weights_manifests) and the job stops on any difference. The scoring job copies
  # this record into the shard (wf/families.py, weights_check).
  export WF_WEIGHTS_RECORD="logs/familyB_prefetch_${SLURM_JOB_ID:-local}.json"
  echo "=== downloading ${TAG} into ${LOCAL_CACHE} $(date) ==="
  python -m wf.tools.prefetch_models "${TAG}" --cache "${LOCAL_CACHE}" --node-local --out "${WF_WEIGHTS_RECORD}"
  echo "=== download and hash check done $(date) ==="
fi
export HF_HUB_OFFLINE=1

echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) cache=${HF_HUB_CACHE} ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
