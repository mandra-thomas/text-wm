#!/bin/bash
#SBATCH --job-name=wf-analysis
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU only: analyses read saved scores and run game bootstraps in one Python process. The inputs are small (the
# largest, rq1_score, is about 90 MB); a laptop ran audit_analysis in about 2 minutes and rq1x_analysis_v2 in
# well under an hour (2026-11-20).
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#
# Runs one or more analysis runs in order, each with its own run id under runs/ (added 2026-11-20):
#   sbatch --export=ALL,ANALYSIS_RUNS="audit_analysis rq2_shared_analysis" pace/analysis_cpu.sbatch
# Every input run must be under this checkout's runs/ (a symlink to another checkout's run folder is fine), or be
# named by absolute path in the run's config (audit_analysis's `inputs:`).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUNS="${ANALYSIS_RUNS:?set ANALYSIS_RUNS to one or more analysis run ids}"
for run in ${RUNS}; do
  echo "=== ${run} START $(date) host=$(hostname) ==="
  run_wf "${run}"
  echo "=== ${run} END $(date) ==="
done
