#!/bin/bash
#SBATCH --job-name=wf-tau-auth-tok
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU, tokenizers only (notes/tau_auth_design.md §8): every prompt tau_auth_score scores, for both models. Needs no
# pace.env: the caller names the Python and the Hugging Face cache, which are only read.
#   sbatch --account=coc --qos=coc-ice --export=ALL,WF_PYTHON=<python>,HF_HOME=<hf cache>,TOK_CACHE_DIR=<dir> pace/tau_auth_tokcheck.sbatch
# TOK_CACHE_DIR (optional) is the folder holding models--<org>--<name>/snapshots/<revision>; on ICE the Llama snapshot
# is only in the top level of the shared hfcache, not in hfcache/hub.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/tau_auth_tokcheck_%j.out
#SBATCH --error=logs/tau_auth_tokcheck_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PY="${WF_PYTHON:?set WF_PYTHON}"
export HF_HOME="${HF_HOME:?set HF_HOME}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$(pwd)"
mkdir -p runs/tau_auth_tokcheck
echo "=== tau-auth token check START $(date) host=$(hostname) commit=$(git rev-parse HEAD 2>/dev/null || cat GIT_SHA) ==="
"${PY}" -m wf.tau.auth_token_check runs/tau_build/items.jsonl runs/tau_build/fewshot.json \
  --out runs/tau_auth_tokcheck/report.json ${TOK_CACHE_DIR:+--cache-dir "${TOK_CACHE_DIR}"}
echo "=== tau-auth token check END $(date) ==="
