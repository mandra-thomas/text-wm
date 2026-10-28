#!/usr/bin/env bash
# Shared preamble for Wishful Foresight sbatch scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"

if [[ -f "${PROJECT_ROOT}/pace/pace.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/pace/pace.env"
  set +a
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
export RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
export WORD2WORLD_ROOT="${WORD2WORLD_ROOT:-${PROJECT_ROOT}/third_party/Word2World}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hfcache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export ALFWORLD_PORT="${ALFWORLD_PORT:-36001}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${RUNS_DIR}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

activate_serve() {
  if [[ -n "${SERVE_ENV:-}" && -f "${SERVE_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SERVE_ENV}/bin/activate"
  elif [[ -n "${SERVE_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${SERVE_ENV}"
  fi
  # Prefer pip-bundled CUDA libs over older module CUDA (nvJitLink clash)
  if command -v python >/dev/null 2>&1; then
    local _nvlibs
    _nvlibs="$(python - <<'PY' 2>/dev/null || true
import os, glob
roots=[]
try:
    import nvidia.nvjitlink as m
    roots.append(os.path.join(os.path.dirname(m.__file__), "lib"))
except Exception:
    pass
pref = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or ""
if pref:
    roots.extend(glob.glob(os.path.join(pref, "lib/python*/site-packages/nvidia/*/lib")))
print(":".join(dict.fromkeys([r for r in roots if r])))
PY
)"
    if [[ -n "${_nvlibs}" ]]; then
      export LD_LIBRARY_PATH="${_nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}

start_alfworld_server() {
  local log="${PROJECT_ROOT}/logs/alfworld_${SLURM_JOB_ID:-local}.log"
  if [[ -x "${WORD2WORLD_ROOT}/scripts/env_server/start_alfworld.sh" ]]; then
    (
      cd "${WORD2WORLD_ROOT}"
      bash scripts/env_server/start_alfworld.sh
    ) >"${log}" 2>&1 &
    echo $! >"${PROJECT_ROOT}/logs/alfworld.pid"
    echo "ALFWorld server starting (log=${log})"
    sleep 15
  else
    echo "WARN: Word2World ALFWorld server script missing at ${WORD2WORLD_ROOT}; runs needing env will fail."
  fi
}

stop_alfworld_server() {
  if [[ -f "${PROJECT_ROOT}/logs/alfworld.pid" ]]; then
    kill "$(cat "${PROJECT_ROOT}/logs/alfworld.pid")" 2>/dev/null || true
    rm -f "${PROJECT_ROOT}/logs/alfworld.pid"
  fi
}

start_vllm() {
  local model="${VLLM_MODEL:-Qwen/Qwen3-8B}"
  local log="${PROJECT_ROOT}/logs/vllm_${SLURM_JOB_ID:-local}.log"
  activate_serve
  python -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --enable-prefix-caching \
    --port "${VLLM_PORT}" \
    >"${log}" 2>&1 &
  echo $! >"${PROJECT_ROOT}/logs/vllm.pid"
  echo "vLLM starting model=${model} port=${VLLM_PORT} (log=${log})"
  # Wait until healthy or timeout
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 \
      || curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM ready"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: vLLM failed to become ready" >&2
  return 1
}

stop_vllm() {
  if [[ -f "${PROJECT_ROOT}/logs/vllm.pid" ]]; then
    kill -2 "$(cat "${PROJECT_ROOT}/logs/vllm.pid")" 2>/dev/null || true
    rm -f "${PROJECT_ROOT}/logs/vllm.pid"
  fi
}

run_wf() {
  local run_id="$1"
  activate_serve
  python -m wf.runs "${run_id}"
}
