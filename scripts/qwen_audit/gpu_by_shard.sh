#!/usr/bin/env bash
# Run on ICE (read-only): for every job log of the scoring runs this audit reads, print the job id, the START line
# (run and shard), the GPU name from nvidia-smi, and whether the job ended with "shard complete".
# Usage: ssh ice 'bash -s' < scripts/qwen_audit/gpu_by_shard.sh > runs/_qwen_audit/gpu_by_shard.tsv
B=/storage/ice1/2/1/gmat3
for c in fix11 fix12 fix13 fix17 fix18 fix19 fix21 fix36; do
  for f in "$B/text-wm-$c"/logs/*.out; do
    [ -f "$f" ] || continue
    start=$(grep -m1 -E "=== .* START" "$f" | sed -E 's/ START.*//; s/=== //')
    [ -n "$start" ] || continue
    gpu=$(grep -m1 -oE "NVIDIA (A100[^|]*|A40|L40S|H100[^|]*|H200[^|]*|RTX[^|]*|V100[^|]*)" "$f" | sed -E 's/ +$//')
    done_=$(grep -c -iE "shard complete|END" "$f")
    printf "%s\t%s\t%s\t%s\t%s\n" "$c" "$(basename "$f" .out)" "$start" "$gpu" "$done_"
  done
done
