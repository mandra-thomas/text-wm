#!/bin/bash
#SBATCH --job-name=wf-bayesenv-cpu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/bayesenv_cpu_%j.out
#SBATCH --error=logs/bayesenv_cpu_%j.err
#
# CPU work of the controlled environment (notes/bayesenv_design.md): BE_CPU=tokcheck runs the tokenizer-only check
# (scripts/bayesenv_tokcheck.py); otherwise BE_CPU names a CPU run (bayesenv_smoke_readout_<model>,
# bayesenv_analysis). Example: bash pace/submit.sh bayesenv_cpu.sbatch --export=ALL,BE_CPU=tokcheck

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SERVE_ENV="${BE_SERVE_ENV:-/storage/ice1/2/1/gmat3/envs/serve2}"
# The shared cache holding both pinned snapshots (transformers 5 reads HF_HUB_CACHE).
export HF_HUB_CACHE="${BE_HF_CACHE:-/storage/ice1/2/1/gmat3/hfcache}" TRANSFORMERS_CACHE="${BE_HF_CACHE:-/storage/ice1/2/1/gmat3/hfcache}"
WHAT="${BE_CPU:?set BE_CPU}"
echo "=== bayesenv cpu ${WHAT} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
activate_serve
python -c 'import transformers; print("transformers", transformers.__version__)'
if [[ "${WHAT}" == "prefetch" ]]; then
  # The pinned Llama-3.1-8B-Instruct snapshot into the cache transformers reads (HF_TOKEN from pace.env, never printed)
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  python - <<'PY'
from huggingface_hub import snapshot_download
from wf.constants import revision_for
from wf.tools.prefetch_models import ALLOW, IGNORE, transformers_cache
m = "meta-llama/Llama-3.1-8B-Instruct"
p = snapshot_download(m, revision=revision_for(m), allow_patterns=ALLOW, ignore_patterns=IGNORE, cache_dir=transformers_cache())
print("OK", m, p)
PY
elif [[ "${WHAT}" == "tokcheck" ]]; then
  python scripts/bayesenv_tokcheck.py --out "${RUNS_DIR}/bayesenv_tokcheck.json"
else
  run_wf "${WHAT}"
fi
echo "=== bayesenv cpu ${WHAT} END $(date) ==="
