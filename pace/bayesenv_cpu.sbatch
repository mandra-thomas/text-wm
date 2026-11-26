#!/bin/bash
#SBATCH --job-name=wf-bayesenv-cpu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/bayesenv_cpu_%j.out
#SBATCH --error=logs/bayesenv_cpu_%j.err
#
# CPU work of the controlled environment (notes/bayesenv_design.md): BE_CPU=tokcheck runs the tokenizer-only check
# (scripts/bayesenv_tokcheck.py); otherwise BE_CPU names a CPU run (bayesenv_smoke_readout_<model>,
# bayesenv_analysis). Example: bash pace/submit.sh bayesenv_cpu.sbatch --export=ALL,BE_CPU=tokcheck

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WHAT="${BE_CPU:?set BE_CPU}"
echo "=== bayesenv cpu ${WHAT} START $(date) host=$(hostname) commit=$(git rev-parse HEAD) ==="
activate_serve
python -c 'import transformers; print("transformers", transformers.__version__)'
if [[ "${WHAT}" == "tokcheck" ]]; then
  python scripts/bayesenv_tokcheck.py --out "${RUNS_DIR}/bayesenv_tokcheck.json"
else
  run_wf "${WHAT}"
fi
echo "=== bayesenv cpu ${WHAT} END $(date) ==="
