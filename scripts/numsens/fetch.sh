#!/usr/bin/env bash
# Copy (read-only) the archived run folders this analysis reads from ICE into $NUMSENS_DATA, then run the analyses.
# Nothing on the cluster is written; no job is submitted. pace/pace.env and any .env are never read.
# Usage: NUMSENS_DATA=/some/scratch/dir bash fetch.sh && python alfworld_pairs.py && python optionc_step0.py \
#        && python textcraft.py && python precision_attention.py && python floor.py
set -euo pipefail
D=${NUMSENS_DATA:?set NUMSENS_DATA}
mkdir -p "$D/trans"
B=/storage/ice1/2/1/gmat3
ssh -o BatchMode=yes ice "cd $B; tar chf - --exclude='*.npy' --exclude=acts \
 text-wm-fix11/runs/rq2_score text-wm-fix18/runs/size_score_qwen3-8b text-wm-fix18/runs/size_score_qwen3-32b \
 text-wm-fix17/runs/rq1v_score text-wm-fix12/runs/rq1_score text-wm-fix13/runs/rq1x_score \
 text-wm-fix21/runs/family_score_mistral-7b-v0.3 text-wm-fix21/runs/family_score_gemma2-9b-it \
 text-wm-fix52/runs/family_rescore_qwen3-8b text-wm-fix52/runs/family_rescore_llama3.1-8b \
 text-wm-fix52/runs/family_rescore_mistral-7b-v0.3 text-wm-fix52/runs/family_rescore_gemma2-9b-it \
 text-wm-fix52/runs/family_rescore_qwen3-32b \
 text-wm-fix39/runs/family_envcheck_qwen3-8b text-wm-fix39/runs/family_envcheck_mistral-7b-v0.3 \
 text-wm-fix39/runs/families_envcheck text-wm-fix45/runs/families_envcheck_precision \
 text-wm-fix30/runs/decisionsim_score_qwen3-8b text-wm-fix30/runs/decisionsim_score_llama3.1-8b \
 text-wm-fix40/runs/optionc_step0_qwen3-8b text-wm-fix40/runs/optionc_step0_llama3.1-8b text-wm-fix50/runs/optionc_fu \
 text-wm-fix42/runs/textcraft_r2_score_qwen3-8b text-wm-fix42/runs/textcraft_r2_score_mistral-7b-v0.3 \
 text-wm-fix42/runs/textcraft_r2_score_gemma2-9b-it \
 text-wm-fix47/runs/textcraft_r3_envcheck_qwen3-8b text-wm-fix47/runs/textcraft_r3_envcheck_mistral-7b-v0.3 \
 text-wm-fix47/runs/textcraft_r3_envcheck_gemma2-9b-it text-wm-fix47/runs/textcraft_r3_rescore_qwen3-8b \
 text-wm-fix47/runs/textcraft_r3_rescore_mistral-7b-v0.3 text-wm-fix47/runs/textcraft_r3_rescore_gemma2-9b-it \
 text-wm-lexsem/runs/lexsem \
 text-wm-qwenaudit2/runs/qwen_audit/qwen_bf16.jsonl text-wm-qwenaudit2/runs/qwen_audit/qwen_fp32.jsonl \
 text-wm-qwenaudit2/runs/qwen_audit/qwen_vllm.jsonl text-wm-qwenaudit2/runs/qwen_audit/qwen_repo.jsonl \
 text-wm-qwenaudit2/runs/qwen_audit/llama_bf16.jsonl text-wm-qwenaudit2/runs/qwen_audit/llama_vllm.jsonl \
 text-wm-qwenaudit2/runs/qwen_audit/summary.json \
 \$(ls -d text-wm-qwenaudit2/runs/qwen_audit/*gpu.txt) \
 \$(ls text-wm-fix47/runs/textcraft_r3_smoke_*/shard_00_of_01/tokenization_checks.json) \
 text-wm-fix52/runs/family_score_gemma4-e4b-it/shard_00_of_01/tokenization_checks.json \
 text-wm-fix52/runs/family_placement_gemma4-e4b-it/shard_00_of_01/tokenization_checks.json" | tar xf - -C "$D"
for s in 0 1 2; do
  ssh -o BatchMode=yes ice "cat $B/text-wm-fix17/runs/rq12_transitions_s$s/transitions.jsonl" > "$D/trans/s$s.jsonl"
done
# round-2 TextCraft records (failure classes, roles) from the frozen tag
git -C "$(dirname "$0")/../../../.." show textcraft-r2-design-frozen-2026-11-23:wf/textcraft/data/r2_items.jsonl > "$D/r2_items.jsonl"
# GPU per job for runs without jobs.jsonl: read from the job logs (nvidia-smi banner); the shard maps in
# alfworld_pairs.py (RQ2_GPU, RQ1_GPU, RQ1V_GPU) were taken from this listing.
ssh -o BatchMode=yes ice "for d in $B/text-wm-fix*; do for f in \$d/logs/*.out; do [ -f \"\$f\" ] || continue; \
  printf '%s\t%s\t%s\t%s\n' \$(basename \$d) \$(basename \$f .out) \"\$(grep -m1 -E '=== .* START' \$f | sed -E 's/ START.*//; s/=== //')\" \
  \"\$(grep -m1 -oE 'NVIDIA (A100[^|]*|A40|L40S|H100[^|]*)' \$f)\"; done; done" > "$D/gpu_by_job.tsv"
