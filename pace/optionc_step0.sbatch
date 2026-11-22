#!/bin/bash
#SBATCH --job-name=wf-optionc-step0
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Option C step-0 check (notes/optionc_design.md): Stage 1a's 134 step-0 states, every candidate under four arms and
# two readouts, about 14,000 single forward passes per model (about 1 GPU-hour); no agent and no ALFWorld server. 3 h
# per job; a cut-off job resumes from its saved scores (WF_RESUME=1). The collection runs (rq12_transitions_s0..s2)
# must be in this checkout's runs/.
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/optionc_step0_%j.out
#SBATCH --error=logs/optionc_step0_%j.err
#
#   for t in qwen3-8b llama3.1-8b; do bash pace/submit.sh optionc_step0.sbatch --export=ALL,OC_RUN=optionc_step0_$t; done
# then (CPU):   sbatch --export=ALL,ANALYSIS_RUNS=optionc_step0_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${OC_RUN:?set OC_RUN, e.g. optionc_step0_qwen3-8b}"
case "${RUN}" in
  optionc_step0_qwen3-8b|optionc_step0_llama3.1-8b|optionc_step0_gemma2-9b-it) ;;
  *) echo "ERROR: OC_RUN must be an Option C step-0 run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} START $(date) host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
