#!/bin/bash
#SBATCH --job-name=wf-tau-auth
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sizing (notes/tau_auth_design.md §9.1). Per item and model: 3 arms x (failing call and witness) x (5 task lines first
# token + 3 yes/no) = 48 passes; plus 3 arms x 4 comprehension questions on 24 items. 9,504 passes per model. At about
# 1.5 s (Qwen3, arm (i), ~10.7k tokens) and 1.73 s (arms (ii) and (iii), ~12.3k tokens), and 1.2 / 1.38 s for Llama,
# the full run is about 7.9 GPU-hours; with 4 shards by user about 2 hours per shard for both models. The limit is 4 h
# (tau_auth_smoke_analysis.yaml time_limit_hours must equal it; a test checks), and every score is saved as it
# finishes, so a shard cut off by the limit resumes:
#   sbatch --dependency=afternotok:<job> ... --export=ALL,WF_RESUME=1,RQ2_RUN=tau_auth_score,WF_SHARD=<i>,WF_N_SHARDS=4 pace/tau_auth_score.sbatch
# Nothing is scored before the design is frozen at an annotated tag after independent review (design §10).
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
# Any free L40S, A40, A100 or H100; brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40|gpu-a100|gpu-h100)
#SBATCH --output=logs/tau_auth_score_%j.out
#SBATCH --error=logs/tau_auth_score_%j.err
#
#   sbatch --account=coc --qos=coc-ice --export=ALL,RQ2_RUN=tau_auth_score_smoke pace/tau_auth_score.sbatch
#   sbatch --account=coc --qos=coc-ice --export=ALL,RQ2_RUN=tau_auth_score,WF_SHARD=0,WF_N_SHARDS=4 pace/tau_auth_score.sbatch
# When every shard says "shard complete": ANALYSIS_RUNS=tau_auth_analysis pace/analysis_cpu.sbatch (no GPU).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${RQ2_RUN:?set RQ2_RUN to tau_auth_score or tau_auth_score_smoke}"
export HF_HUB_OFFLINE=1
echo "=== tau-auth ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
nvidia-smi || true
run_wf "${RUN}"
echo "=== tau-auth ${RUN} END $(date) ==="
