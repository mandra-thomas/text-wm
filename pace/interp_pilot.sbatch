#!/bin/bash
#SBATCH --job-name=wf-interp
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Minimal interpretability pilot (notes/interp_design.md, amendment 1, section M), one model (Qwen3-8B) on one
# A100 or L40S. Estimated 1 + 3 + 1.5 A100-hours (about 1.5 times that on an L40S); 8 h covers the A100 case with
# room, and every stage resumes per item, so a cut-off job is resubmitted with WF_RESUME=1.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
# Brackets because ICE appends &cuda13 to the constraint. One GPU type for the whole pilot: the aligned and
# swapped runs of an item, and stage 0 and the later stages, should not mix GPU types (deviations.md,
# "Scores across GPU types"); pin it with --constraint=gpu-a100 (or gpu-l40s) on resubmission.
#SBATCH --constraint=(gpu-a100|gpu-l40s)
#SBATCH --output=logs/interp_%j.out
#SBATCH --error=logs/interp_%j.err
#
# Runs the stages in order; a stage that fails stops the job (stage 0 fails if the reproduction gate fails, and
# nothing else runs until that is explained):
#   bash pace/submit.sh interp_pilot.sbatch
#   bash pace/submit.sh interp_pilot.sbatch --export=ALL,INTERP_STAGES="interp_pilot_patch interp_pilot_ablate"
# Before the full patching run, a 10-item timing check is advisable (design section 10):
#   python -m wf.runs interp_pilot_patch  with n_items: 10 in a copy of the config.
# Then on a CPU node: sbatch --export=ALL,ANALYSIS_RUNS=interp_pilot_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
df -h "${RUNS_DIR}" | tail -1
STAGES="${INTERP_STAGES:-interp_pilot_stage0 interp_pilot_patch interp_pilot_ablate}"
nvidia-smi || true
for run in ${STAGES}; do
  echo "=== ${run} START $(date) host=$(hostname) ==="
  run_wf "${run}"
  echo "=== ${run} END $(date) ==="
done
