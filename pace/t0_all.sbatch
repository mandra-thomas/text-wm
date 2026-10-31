#!/bin/bash
#SBATCH --job-name=wf-t0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/t0_%j.out
#SBATCH --error=logs/t0_%j.err
#
# GPU ownership (1×A100):
#   Phase A — vLLM up for agent runs (smoke / determinism).
#   Phase B — vLLM stopped; HF transformers owns the GPU for WM scoring
#             (ckpt_accuracy / scoring_agreement). Avoids CPU-offload thrash.
# Override account/qos via: sbatch --account=... --qos=... pace/t0_all.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

echo "=== Tier 0 START $(date) host=$(hostname) ==="
nvidia-smi || true

cleanup() {
  stop_vllm || true
  stop_alfworld_server || true
}
trap cleanup EXIT

start_alfworld_server

run_phase() {
  local rid="$1"
  echo "=== RUN ${rid} ==="
  if ! run_wf "${rid}"; then
    echo "FAIL ${rid} — see runs/${rid}/STATUS.md"
    return 1
  fi
  return 0
}

failed=0

# --- Phase A: agent needs vLLM ---
echo "=== Phase A: start vLLM (agent) ==="
start_vllm
for rid in t0_smoke t0_replay t0_agent_determinism t0_parser; do
  if ! run_phase "${rid}"; then
    failed=1
    break
  fi
done

# --- Phase B: HF scoring needs exclusive GPU ---
if [[ "${failed}" -eq 0 ]]; then
  echo "=== Phase B: stop vLLM before HF scoring ==="
  stop_vllm
  for rid in t0_ckpt_accuracy t0_scoring_agreement t0_expert_labels; do
    if ! run_phase "${rid}"; then
      failed=1
      break
    fi
  done
fi

python -m wf.tools.write_env_json || true

echo "=== Tier 0 END $(date) failed=${failed} ==="
exit "${failed}"
