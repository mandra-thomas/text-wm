#!/bin/bash
#SBATCH --job-name=wf-mech2
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# One shard of eight is about 2,590 two-candidate and 2,590 yes/no prompts for each primary model (Qwen3-8B,
# Llama-3.1-8B; 20,720 of each over the run, 18 arms) and about 1,485 of each for Mistral-7B-Instruct-v0.3, the
# secondary model (11,878 over the run: the arms the primary contrasts read, on the sample, and P6's arms on every
# step-0 item; amendments after reviews 1 to 3): about 20,000 forward passes (159,954 over the run). At 0.26 to 0.4 s
# a pass (rq2_score's rate and the families' sizing) that is 1.4 to 2.2 hours, plus three model loads and the
# tokenization gate (CPU, minutes). Estimates, not timings; the smoke run measures them.
# 4 hours by default; scores resume if cut off (WF_RESUME=1).
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40 or A100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100)
#SBATCH --output=logs/mech2_%j.out
#SBATCH --error=logs/mech2_%j.err
#
# Mechanism battery (wf/mech2.py, notes/mechanism2_design.md), one game shard per job:
#   bash pace/submit.sh mech2_score.sbatch --export=ALL,MECH2_RUN=mech2_score_smoke,WF_SHARD=0,WF_N_SHARDS=125
#   bash pace/submit.sh mech2_score.sbatch --export=ALL,MECH2_RUN=mech2_score,WF_SHARD=0,WF_N_SHARDS=8   (0..7)
# Read the smoke run for format, gate and timing only. The job stops before scoring a model if its loaded revision is
# not the pinned one or the tokenization gate fails (runs/<run>/shard_*/tokenization_checks_<model>.json).
# After a timeout: the same command with WF_RESUME=1 in --export. When every shard says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=mech2_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${MECH2_RUN:?set MECH2_RUN to mech2_score or mech2_score_smoke}"
case "${RUN}" in
  mech2_score|mech2_score_smoke) ;;
  *) echo "ERROR: MECH2_RUN must be mech2_score or mech2_score_smoke, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== ${RUN} END $(date) ==="
