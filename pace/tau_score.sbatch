#!/bin/bash
#SBATCH --job-name=wf-tau-score
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sizing (notes/tau_design.md, amendments of 2026-11-20 after reviews 1 and 2). Per item and model the full run
# scores about 30 forward passes of a ~10.5k-token prompt (Qwen3; ~9.6k for Llama): the failing call under 3 task
# lines x (first token, yes/no, 2 full replies) = 12, the mention pair 4, the manipulation check 2, the witness 12;
# plus 6 each for the 500 items of the rules-only and the ALFWorld-matched arm (the same items), and 3 more for the
# matched arm's witness (first token; fix round 3). About 49.7k passes per model, 99k in all; with 16 shards about
# 6.2k passes per shard on average. At ~1.5 s a pass (A100/L40S, 8B in bf16 at ~10k
# tokens) that is ~2.5 h for an average shard and ~3.4 h for the largest (4,053 passes per model; ~6.8 h on an A40). The limit is 8 h (tau_smoke_analysis.yaml time_limit_hours must
# equal it; a test checks), and every score is saved as it finishes, so a shard cut off by the limit resumes:
#   sbatch --dependency=afternotok:<job> ... --export=ALL,WF_RESUME=1,RQ2_RUN=tau_score,WF_SHARD=<i>,WF_N_SHARDS=16 pace/tau_score.sbatch
# The smoke run (tau_score_smoke, then tau_smoke_analysis) measures the mean seconds per job and projects the hours
# per shard against the limit ("projection" in metrics.json, "fits_time_limit"), with the projected CI half-widths;
# both are recorded in the design note and committed before the full run is submitted.
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
