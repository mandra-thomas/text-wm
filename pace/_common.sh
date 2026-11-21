#!/usr/bin/env bash
# Shared preamble for Wishful Foresight sbatch scripts.
set -euo pipefail

for mock_var in WF_MOCK_ENV WF_MOCK_AGENT WF_MOCK_SCORE; do
  if [[ -n "${!mock_var:-}" ]]; then
    echo "ERROR: ${mock_var} is set; cluster jobs must not run in mock mode" >&2
    exit 1
  fi
done

# The checkout this script sits in is the project, whatever the shell has left in PROJECT_ROOT.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
export PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"

if [[ -f "${PROJECT_ROOT}/pace/pace.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/pace/pace.env"
  set +a
fi

if [[ "$(cd "${PROJECT_ROOT}" 2>/dev/null && pwd -P)" != "${PROJECT_ROOT_DEFAULT}" ]]; then
  echo "ERROR: pace.env says PROJECT_ROOT=${PROJECT_ROOT}, but this script is in ${PROJECT_ROOT_DEFAULT}" >&2
  exit 1
fi

export RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
export WORD2WORLD_ROOT="${WORD2WORLD_ROOT:-${PROJECT_ROOT}/third_party/Word2World}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hfcache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export ALFWORLD_PORT="${ALFWORLD_PORT:-36001}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Home is a 30 GB quota that other work shares; a full home kills vLLM's compile cache and
# anything else that writes there, so every tool cache lives in the checkout on /storage.
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"

# Per-job pid files, so concurrent jobs sharing this checkout cannot stop each other's servers.
export VLLM_PID_FILE="${PROJECT_ROOT}/logs/vllm_${SLURM_JOB_ID:-local}.pid"
export ALFWORLD_PID_FILE="${PROJECT_ROOT}/logs/alfworld_${SLURM_JOB_ID:-local}.pid"

mkdir -p "${RUNS_DIR}" "${PROJECT_ROOT}/logs" "${XDG_CACHE_HOME}"

cd "${PROJECT_ROOT}"

activate_serve() {
  # Conda prefix envs often lack bin/activate — put env bin first on PATH.
  module load anaconda3/2023.03 2>/dev/null || true
  if [[ -n "${SERVE_ENV:-}" && -x "${SERVE_ENV}/bin/python" ]]; then
    export PATH="${SERVE_ENV}/bin:${PATH}"
    export CONDA_PREFIX="${SERVE_ENV}"
  elif [[ -n "${SERVE_ENV:-}" && -f "${SERVE_ENV}/bin/activate" ]]; then
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
  hash -r 2>/dev/null || true
  echo "python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo missing))"
}

# --- ports (2026-11-18) -------------------------------------------------------------------------------
# Jobs share nodes. Checking that a port is free and then starting a server is a race: a server takes
# seconds (ALFWorld) or minutes (vLLM) to bind, and meanwhile another job's server can answer on that port.
# On 2026-11-18 three verification jobs on one node all used a fourth job's ALFWorld server this way.
# So each job starts from a port derived from its job id, skips ports in use, and a server counts as ready
# only when the process listening on the port is our own (or a child of it).
_port_in_use() {
  ss -ltnH "sport = :$1" 2>/dev/null | grep -q . || curl -s --max-time 2 "http://127.0.0.1:$1/" >/dev/null 2>&1
}

_pick_port() {  # base span -> a free port in [base, base+span), starting from this job's offset
  local base="$1" span="$2" start i q
  start=$(( ${SLURM_JOB_ID:-0} % span ))
  for i in $(seq 0 $((span - 1))); do
    q=$(( base + (start + i) % span ))
    if ! _port_in_use "${q}"; then
      echo "${q}"
      return 0
    fi
  done
  return 1
}

_descendants() {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do
    echo "${c}"
    _descendants "${c}"
  done
}

_owns_port() {  # pid port -> is the listener on port this pid or one of its descendants?
  local pid="$1" port="$2" mine listeners l
  mine=" ${pid} $(_descendants "${pid}" | tr '\n' ' ') "
  listeners=$(ss -ltnpH "sport = :${port}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2)
  for l in ${listeners}; do
    [[ "${mine}" == *" ${l} "* ]] && return 0
  done
  return 1
}

start_alfworld_server() {
  local log="${PROJECT_ROOT}/logs/alfworld_${SLURM_JOB_ID:-local}.log"
  local alfworld="${WORD2WORLD_ROOT}/uv_alfworld/bin/alfworld"
  if [[ ! -x "${alfworld}" ]]; then
    echo "ERROR: ALFWorld server missing at ${alfworld}" >&2
    return 1
  fi
  local attempt pid
  for attempt in 1 2 3; do
    ALFWORLD_PORT=$(_pick_port 36100 800) || { echo "ERROR: no free ALFWorld port" >&2; return 1; }
    export ALFWORLD_PORT
    export ALFWORLD_BASE_URL="http://127.0.0.1:${ALFWORLD_PORT}"
    (
      cd "${WORD2WORLD_ROOT}"
      # shellcheck disable=SC1091
      source uv_alfworld/bin/activate
      exec alfworld --host 127.0.0.1 --port "${ALFWORLD_PORT}"
    ) >"${log}" 2>&1 &
    pid=$!
    echo "${pid}" >"${ALFWORLD_PID_FILE}"
    echo "ALFWorld server starting port=${ALFWORLD_PORT} pid=${pid} attempt=${attempt} (log=${log})"
    for _ in $(seq 1 60); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break  # exited: most likely another job took the port first; try another
      fi
      # Ready only when the listener is our own process, not another job's server on the same port.
      if curl -sf "${ALFWORLD_BASE_URL}/" >/dev/null 2>&1 && _owns_port "${pid}" "${ALFWORLD_PORT}"; then
        echo "ALFWorld ready at ${ALFWORLD_BASE_URL} (pid ${pid})"
        return 0
      fi
      sleep 5
    done
    kill "${pid}" 2>/dev/null || true
    echo "WARN: ALFWorld server on port ${ALFWORLD_PORT} did not come up as ours; see ${log}" >&2
  done
  echo "ERROR: ALFWorld server failed to become ready" >&2
  return 1
}

stop_alfworld_server() {
  if [[ -f "${ALFWORLD_PID_FILE}" ]]; then
    local pid kids
    pid="$(cat "${ALFWORLD_PID_FILE}")"
    kids="$(_descendants "${pid}")"  # the listener can be a child; collect before the parent goes
    # shellcheck disable=SC2086
    kill "${pid}" ${kids} 2>/dev/null || true
    # Wait until they are gone: a dying server still answers on its port, and the next start must not
    # take that answer for its own (shard 2's collection job crashed this way on 2026-11-18).
    local p _
    for _ in $(seq 1 30); do
      local alive=0
      for p in ${pid} ${kids}; do kill -0 "${p}" 2>/dev/null && alive=1; done
      [[ "${alive}" == 0 ]] && break
      sleep 1
    done
    # shellcheck disable=SC2086
    kill -KILL "${pid}" ${kids} 2>/dev/null || true
    rm -f "${ALFWORLD_PID_FILE}"
  fi
}

start_vllm() {
  local model="${VLLM_MODEL:-Qwen/Qwen3-8B}"
  local log="${PROJECT_ROOT}/logs/vllm_${SLURM_JOB_ID:-local}.log"
  activate_serve
  # The pinned snapshot from wf.constants.REVISIONS (2026-11-20); an unpinned model stops the job here unless
  # WF_ALLOW_UNPINNED=1, and VLLM_REVISION overrides the map.
  local revision="${VLLM_REVISION:-}"
  if [[ -z "${revision}" ]]; then
    revision="$(python -c 'import sys; from wf.constants import revision_for; print(revision_for(sys.argv[1]) or "")' "${model}")" \
      || { echo "ERROR: no pinned revision for ${model} (wf.constants.REVISIONS)" >&2; return 1; }
  fi
  local revision_args=()
  [[ -n "${revision}" ]] && revision_args=(--revision "${revision}" --tokenizer-revision "${revision}")
  VLLM_PORT=$(_pick_port 8000 1000) || { echo "ERROR: no free vLLM port" >&2; return 1; }
  export VLLM_PORT
  export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --model "${model}" \
    ${revision_args[@]+"${revision_args[@]}"} \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --no-enable-prefix-caching \
    ${VLLM_EXTRA_ARGS:-} \
    --port "${VLLM_PORT}" \
    >"${log}" 2>&1 &
  echo $! >"${VLLM_PID_FILE}"
  echo "vLLM starting model=${model} revision=${revision:-unpinned} port=${VLLM_PORT} extra=${VLLM_EXTRA_ARGS:-none} (log=${log})"
  # Wait until healthy or timeout
  for _ in $(seq 1 120); do
    # Fail fast if process died; an answer on this port could come from another job's server.
    if [[ -f "${VLLM_PID_FILE}" ]] && ! kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
      echo "ERROR: vLLM process exited early; see ${log}" >&2
      return 1
    fi
    if { curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 \
      || curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; } \
      && _owns_port "$(cat "${VLLM_PID_FILE}")" "${VLLM_PORT}"; then
      echo "vLLM ready at ${VLLM_BASE_URL}"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: vLLM failed to become ready" >&2
  return 1
}

stop_vllm() {
  # Tear down the OpenAI server so HF scoring can own the GPU (no CPU offload).
  local pid=""
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    pid="$(cat "${VLLM_PID_FILE}")"
    rm -f "${VLLM_PID_FILE}"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping vLLM pid=${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
    # Also stop any orphaned vLLM workers bound to our port.
    if [[ -n "${VLLM_PORT:-}" ]]; then
      pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}" 2>/dev/null || true
    fi
    for _ in $(seq 1 60); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  # Give the driver a moment to release VRAM before HF loads.
  sleep 5
  nvidia-smi || true
}

ensure_vllm() {
  if [[ -f "${VLLM_PID_FILE}" ]] && kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
  fi
  start_vllm
}

run_wf() {
  local run_id="$1"
  activate_serve
  # WF_RESUME=1 keeps saved work after a timeout: WF_RESUME=1 bash pace/submit.sh <job>.sbatch
  if [[ "${WF_RESUME:-}" == "1" ]]; then
    python -m wf.runs "${run_id}" --resume
  else
    python -m wf.runs "${run_id}"
  fi
}
