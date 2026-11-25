#!/bin/bash
#SBATCH --job-name=wf-ocfu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ocfu_%j.out
#SBATCH --error=logs/ocfu_%j.err
#
# Option C follow-ups of 2026-11-24 (notes/optionc_followups_2026-11-24.md). FU_TASK picks the work; the GPU type is
# given at submission (--constraint), since each task needs a particular one:
#   rescore  FU_MODEL (qwen3-8b on A100-40GB, llama3.1-8b on gpu-a40): Option C's step-0 yes/no path on Stage 1a's
#            GPU type, for the stage1a_agreement diagnosis.
#   plain    Qwen3-8B, the step-0 check in plain text (both readouts, four arms), on gpu-l40s; the first 3 states
#            also rescored with the block.
# Examples:
#   bash pace/submit.sh optionc_followup.sbatch --export=ALL,FU_TASK=rescore,FU_MODEL=qwen3-8b --constraint=A100-40GB
#   bash pace/submit.sh optionc_followup.sbatch --export=ALL,FU_TASK=plain --constraint=gpu-l40s

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK="${FU_TASK:?set FU_TASK to rescore or plain}"
D="${RUNS_DIR}/optionc_fu"
mkdir -p "${D}"
echo "=== optionc follow-up ${TASK} ${FU_MODEL:-} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
activate_serve
echo "python=$(command -v python)"
case "${TASK}" in
  rescore)
    M="${FU_MODEL:?set FU_MODEL}"
    python scripts/optionc_followup/rescore_stage1a.py score --model "${M}" --out "${D}/rescore_${M}.jsonl"
    ;;
  plain)
    python scripts/optionc_followup/plain_step0.py --out "${D}/plain_step0_qwen3-8b.jsonl" --check-block 3
    ;;
  *) echo "unknown FU_TASK=${TASK}" >&2; exit 2 ;;
esac
echo "=== optionc follow-up ${TASK} END $(date) ==="
