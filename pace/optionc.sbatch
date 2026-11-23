#!/bin/bash
#SBATCH --job-name=wf-optionc
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Option C episodes (wf/optionc.py, notes/optionc_design.md): one shard of one run per job. The agent (Qwen3-8B, vLLM)
# and the verifier (Hugging Face, bf16) share one GPU, so vLLM is held to half of it and the GPU must have 48 GB
# (L40S or A40; an 80 GB A100 also works). The design note estimates about 6 GPU-hours per verifier arm per seed over
# 134 games (Tier 1, four shards: about 11 h per shard); jobs are 6 h and a cut-off shard resumes from its saved
# episodes and cache (WF_RESUME=1). The ALFWorld server leaks memory on every reset, so the run stops with PARTIAL
# after max_episodes_per_job new episodes, the server restarts and the run resumes, while vLLM stays up.
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
# Brackets because ICE appends &cuda13 to the constraint.
#SBATCH --constraint=(gpu-l40s|gpu-a40)
#SBATCH --output=logs/optionc_%j.out
#SBATCH --error=logs/optionc_%j.err
#
# Smoke first (two train games; read smoke_gate and seconds_per_step in runs/optionc_smoke/shard_00_of_01/metrics.json):
#   OC_RUN=optionc_smoke bash pace/submit.sh optionc.sbatch
# Every run but the smoke refuses to start without a passed smoke, a reported step-0 analysis and the cost rule
# (optionc.py, order_gate_check); Gemma also needs OC_RUN=optionc_memsmoke_gemma2-9b-it first. vLLM keeps
# --no-enable-prefix-caching (pace/_common.sh, for greedy determinism); the smoke's seconds_sampling shows how much the
# 20-sample requests queue for it (review 1, S8).
# Then the step-0 check (pace/optionc_step0.sbatch) and, once the design is frozen, Tier 1:
#   for i in 0 1 2 3; do bash pace/submit.sh optionc.sbatch --export=ALL,OC_RUN=optionc_qwen3-8b_yn,WF_SHARD=$i,WF_N_SHARDS=4; done
# Resume a shard that ran out of time with the same command plus WF_RESUME=1. When every shard says "shard complete":
#   sbatch --export=ALL,ANALYSIS_RUNS=optionc_analysis pace/analysis_cpu.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"

if [[ ! -f "${RUNS_DIR}/FILED.txt" ]]; then
  echo "ERROR: ${RUNS_DIR}/FILED.txt missing — refusing to run (protocol §6)." >&2
  exit 2
fi
RUN="${OC_RUN:?set OC_RUN, e.g. optionc_smoke or optionc_qwen3-8b_yn}"
case "${RUN}" in
  optionc_smoke|optionc_qwen3-8b_yn|optionc_qwen3-8b_yn_argmax|optionc_llama3.1-8b_yn|optionc_qwen3-8b_obs|optionc_gemma2-9b-it_yn|optionc_memsmoke_gemma2-9b-it) ;;
  *) echo "ERROR: OC_RUN must be an Option C episode run, not ${RUN}" >&2; exit 2 ;;
esac
echo "=== ${RUN} shard ${WF_SHARD:-config} of ${WF_N_SHARDS:-config} START $(date) host=$(hostname) ==="
cleanup() { stop_vllm || true; stop_alfworld_server || true; }
trap cleanup EXIT
# The agent is greedy as in RQ3; its sampled alternatives send every parameter explicitly, and vLLM's own defaults
# replace the model's generation_config.json (Stage 1b). Half the GPU is left for the verifier.
export VLLM_EXTRA_ARGS="--generation-config vllm --gpu-memory-utilization 0.50"
start_vllm
python -c "import vllm; print('vllm', vllm.__version__)" || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
rc=0
while true; do
  start_alfworld_server
  set +e
  run_wf "${RUN}"
  rc=$?
  set -e
  stop_alfworld_server || true
  # `|| true`: with no STATUS.md yet, grep finds nothing, which must not end the job under pipefail (review 1)
  status=$(ls "${RUNS_DIR}/${RUN}"/shard_*/STATUS.md 2>/dev/null | grep "shard_$(printf %02d "${WF_SHARD:-0}")_of_" | head -1 || true)
  if [[ -n "${status}" ]] && grep -q "PARTIAL" "${status}"; then
    export WF_RESUME=1
    continue
  fi
  break
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
echo "=== ${RUN} END $(date) rc=${rc} ==="
exit "${rc}"
