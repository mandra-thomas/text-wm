#!/usr/bin/env bash
# Copy the stored scores (read-only) from ICE into runs/_qwen_audit/<checkout>/<run>/ (git-ignored).
# Copies scoring.jsonl, transitions.jsonl and each run's provenance files; never writes to the ICE checkouts.
# Usage (from the repository root): bash scripts/qwen_audit/fetch.sh [dest]
set -euo pipefail
DEST=${1:-runs/_qwen_audit}
B=/storage/ice1/2/1/gmat3
SPECS="fix11:rq2_score fix12:rq1_score fix13:rq1x_score fix17:rq1v_score
fix18:size_score_qwen3-0.6b fix18:size_score_qwen3-1.7b fix18:size_score_qwen3-4b fix18:size_score_qwen3-8b
fix18:size_score_qwen3-14b fix18:size_score_qwen3-32b
fix21:family_score_gemma2-9b-it fix21:family_score_llama3.2-1b fix21:family_score_llama3.2-3b fix21:family_score_mistral-7b-v0.3
fix19:mech_score fix36:mech2_score fix9:rq12_transitions_s0 fix9:rq12_transitions_s1 fix9:rq12_transitions_s2"
for spec in $SPECS; do
  c=${spec%%:*}; r=${spec#*:}
  mkdir -p "$DEST/$c/$r"
  rsync -a --include='*/' --include='scoring.jsonl' --include='transitions.jsonl' --include='config.json' \
    --include='git_sha' --include='STATUS.md' --include='jobs.jsonl' --include='tokenization_checks.json' \
    --include='inputs.json' --exclude='*' "ice:$B/text-wm-$c/runs/$r/" "$DEST/$c/$r/"
done
# The confirmatory run's greedy written predictions (what Qwen writes after "You"; section 4.2 of the note)
rsync -a --include='*/' --include='generations.jsonl' --exclude='*' \
  "ice:$B/text-wm-fix11/runs/rq2_score/" "$DEST/fix11/rq2_score/"
