#!/bin/bash
#SBATCH --job-name=wf-models
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/models_%j.out
#SBATCH --error=logs/models_%j.err
#
# Known-answer checks on real models. HF scoring first, then vLLM, since each needs the GPU
# alone. A failing part does not stop the next: every result is useful on its own.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
set +e

cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT

start_alfworld_server || exit 1
activate_serve

# Run everything even if one part fails, then report failure: a COMPLETED job must mean all passed.
failed=0
echo "=== human review file ==="
python -m wf.tools.export_review || failed=1
echo "=== memory after close/create ==="
python -m wf.tools.verify_real_env --only memory_recreate || failed=1
echo "=== scoring checks ==="
python -m wf.tools.verify_models scoring || failed=1
echo "=== agent checks ==="
{ start_vllm && python -m wf.tools.verify_models agent; } || failed=1
echo "=== done failed=${failed} ==="
exit "${failed}"
