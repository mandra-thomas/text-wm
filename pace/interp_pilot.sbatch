#!/bin/bash
#SBATCH --job-name=wf-interp
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Minimal interpretability pilot (notes/interp_design.md, amendment 1, section M; amendment 2), one model
# (Qwen3-8B) on one A100 or L40S. Estimated forward rows (amendment 2, B8): stage 0 28,900 (yes/no prompts with
# residuals and first-token prompts without), patching at most 144,000 (12 groups by 36 layers, noising and
# denoising, 100 items and 50 placebo pairs), ablation 24,000 (calibration fold only, 55 edits per prompt in 4
# batches of 16); about 197,000 rows, 5.5 A100-hours at 0.1 s a row, 8.2 L40S-hours at 0.15 s. 12 h covers the
# L40S case with about 45% to spare. Every stage resumes per item, so a cut-off job is resubmitted with WF_RESUME=1.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
# Brackets because ICE appends &cuda13 to the constraint. One GPU type for the whole pilot: the aligned and
# swapped runs of an item, and stage 0 and the later stages, must not mix GPU types (deviations.md,
# "Scores across GPU types"). The first job may take either type; every later stage, and every resumed stage,
# refuses to start on a GPU type other than the one in stage 0's jobs.jsonl (wf.interp.pilot.check_gpu), so
# resubmit with --constraint=gpu-a100 or --constraint=gpu-l40s, whichever stage 0 ran on. The check compares GPU
# names exactly (amendment 3, C7): --constraint=gpu-a100 can land on another A100 model (40 GB against 80 GB, PCIe
# against SXM) with a different name, which is refused; resubmit until the job lands on stage 0's exact GPU (the
# refusal message names it). Ablation: 55 edits per prompt, still 4 batches of 16, so the estimate is unchanged.
#SBATCH --constraint=(gpu-a100|gpu-l40s)
#SBATCH --output=logs/interp_%j.out
#SBATCH --error=logs/interp_%j.err
#
# Runs the stages in order; a stage that fails stops the job (stage 0 fails if the reproduction gate fails, and
# nothing else runs until that is explained):
#   bash pace/submit.sh interp_pilot.sbatch
#   bash pace/submit.sh interp_pilot.sbatch --export=ALL,INTERP_STAGES="interp_pilot_patch interp_pilot_ablate"
# Before the full patching run, a 10-item timing check is advisable (design section 10):
#   python -m wf.runs interp_pilot_patch  with n_items: 10 in a copy of the config. Its stratum minimums are
#   scaled to 10/100 of the configured ones (amendment 3, C6): 3 step-0 items and 3 with an object mention.
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
