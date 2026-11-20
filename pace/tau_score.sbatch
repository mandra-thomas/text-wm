#!/bin/bash
#SBATCH --job-name=wf-tau-score
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sizing (notes/tau_design.md, amendments of 2026-11-20). Per item and model the full run scores about 30 forward
# passes of a ~10.5k-token prompt (Qwen3; ~9.6k for Llama): the failing call under 3 task lines x (first token,
# yes/no, 2 full replies) = 12, the mention pair 4, the manipulation check 2, the witness 12; plus 6 for the 500
# rules-only items. About 45k passes per model, 90k in all; with 16 shards about 5.7k passes per shard. At
# ~1.5 s a pass (A100/L40S, 8B in bf16 at ~10k tokens) that is ~2.5 h a shard, ~5 h on an A40. The limit is 8 h,
# and every score is saved as it finishes, so a shard cut off by the limit resumes where it stopped:
#   sbatch --dependency=afternotok:<job> ... --export=ALL,WF_RESUME=1,RQ2_RUN=tau_score,WF_SHARD=<i>,WF_N_SHARDS=16 pace/tau_score.sbatch
# The smoke run (tau_score_smoke, then tau_smoke_analysis) measures seconds per job and projects the hours per
# shard (metrics.json, "projection"); check it against this limit before submitting the full run.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40, A100 or H100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100|gpu-h100)
#SBATCH --output=logs/tau_score_%j.out
#SBATCH --error=logs/tau_score_%j.err
#
#   sbatch --account=coc --qos=coc-ice --export=ALL,RQ2_RUN=tau_score_smoke pace/tau_score.sbatch
#   sbatch --account=coc --qos=coc-ice --export=ALL,RQ2_RUN=tau_score,WF_SHARD=0,WF_N_SHARDS=16 pace/tau_score.sbatch
# When every shard says "shard complete": python -m wf.runs tau_analysis (no GPU).
# A resume refuses to mix commits, settings or item files (wf.runs._check_resume) and records whose item hash
# differs (run_tau_score), and a second job on the same shard refuses to start (run lock).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${RQ2_RUN:?set RQ2_RUN to tau_score or tau_score_smoke}"
export HF_HUB_OFFLINE=1
echo "=== tau ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== tau ${RUN} END $(date) ==="
