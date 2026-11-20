#!/bin/bash
#SBATCH --job-name=wf-placement
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# 300 primary items x (aligned, swapped) x two candidates + 300 x (yn_aligned, yn_swapped) = 1,800 forward passes,
# about 12 minutes at the RQ2 sizing (0.4 s a pass at 8B), plus loading and the gate on the 300 items. Expect
# 20 to 30 GPU-minutes; 1 h allowed.
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13. For the cleanest comparison with
# family_score_mistral-7b-v0.3, pass the GPU type that run used (its shard's jobs.jsonl), e.g.
# --constraint=gpu-a100: GPU types differ in numerics (notes/families_design.md).
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/placement_%j.out
#SBATCH --error=logs/placement_%j.err
#
# Model families, wave A: Mistral 7B v0.3's system-text placement check (exploratory, logged 2026-11-20 before any
# wave A score was read; notes/families_design.md). The system text goes into the first user turn instead of the
# template's last user turn; scores are saved under the tag mistral-7b-v0.3-firstturn.
#   bash pace/submit.sh family_placement.sbatch
# The tokenization gate runs first, on the 300 items, and stops the job before any score if it fails. After a
# timeout: the same command with --export=ALL,WF_RESUME=1. families_analysis reads this run with the family runs.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="family_placement_mistral-7b-v0.3"
echo "=== ${RUN} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
