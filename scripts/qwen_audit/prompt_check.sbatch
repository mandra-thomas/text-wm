#!/bin/bash
#SBATCH -J qa-prompt-check
#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/qa_prompt_check_%j.out
# CPU job for the forensic audit's prompt and tokenizer checks. Runs in its own checkout
# (/storage/ice1/2/1/gmat3/text-wm-qwenaudit); reads the collection from text-wm-fix9 and the pinned tokenizer
# snapshots from the shared model cache, and writes only under this checkout's runs/_qwen_audit/.
# Does not source pace/_common.sh (no account settings, no secrets are needed for a tokenizer check).
set -euo pipefail
B=/storage/ice1/2/1/gmat3
ROOT=$B/text-wm-qwenaudit
PY=$B/text-wm/.envs/serve/bin/python
OUT=$ROOT/runs/_qwen_audit
mkdir -p "$OUT"
export RUNS_DIR=$B/text-wm-fix9/runs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.cache/hf
export PYTHONDONTWRITEBYTECODE=1
QWEN=$B/hfcache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LLAMA=$B/hfcache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
cd "$ROOT"
PYTHONPATH=$ROOT/old_ddeabc2 $PY scripts/qwen_audit/prompt_check.py render --out "$OUT/render_ddeabc2.jsonl"
PYTHONPATH=$ROOT $PY scripts/qwen_audit/prompt_check.py render --out "$OUT/render_current.jsonl"
PYTHONPATH=$ROOT $PY scripts/qwen_audit/prompt_check.py tokens --out "$OUT/prompt_check.json" \
  --qwen "$QWEN" --llama "$LLAMA" --old-render "$OUT/render_ddeabc2.jsonl" --new-render "$OUT/render_current.jsonl"
echo "prompt check done"
