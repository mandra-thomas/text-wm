# Model families, wave B: review 1 (2026-11-22, before any wave B score)

Independent review of the wave B design ([families_waveb_design.md](families_waveb_design.md)) and its code at dbfbdad (branch `families-waveb`). Condensed from the reviewer's report; recorded verbatim. The fixes are listed in the design note, section "Amendments after review 1, before any score", and in [deviations.md](deviations.md).

## MUST-FIX

M1. Every chat-format score crashes in serve2 (`wf/scoring/logodds.py:258`, used at 293–295 and 372). In transformers 5.x `apply_chat_template(tokenize=True)` defaults to `return_dict=True`, which returns a BatchEncoding. `_teacher_force_logprob` passes it to torch.cat and `score_one_token_pair` passes it as input_ids, and both raise. The gate does not catch this because it unwraps through `_ids()`. Fix `_encode` (pass return_dict=False on transformers ≥5, or unwrap input_ids) without changing behaviour on transformers 4. Add a serve2-only CPU test that runs `score_pair` and `score_one_token_pair` end to end. Use each pinned real wave B tokenizer and a tiny random-weight model built from each wave B config (2 layers), so that the text-only forward passes of Gemma4ForConditionalGeneration, Qwen3_5ForConditionalGeneration and Mistral3ForConditionalGeneration are exercised. It is acceptable that this test can only run on ICE in serve2. Write it, have it skip cleanly elsewhere, and state in your report that it must be run there (I will run it). Also add a local unit test with a fake tokenizer returning a BatchEncoding, which fails on the unfixed code.

M2. B1's basis (`notes/families_waveb_design.md` §10 line ~156, §11 line ~169, `PREDICTION_B` in `wf/families.py`) says "every interpretable model scored so far shows it". That is true for the log-odds effect but not for the share toward Yes. Qwen3-4B is competent (AUC 0.94, Yes on 2.2% of failures) yet its share is 0.49 with a yes/no effect of 2.27 nats; Qwen3-14B and Qwen3-32B have shares of 0.70 and 0.71. State the Qwen3-4B precedent. Change the outcome-table row: if the share is not supported but the yes/no effect's CI is above zero, read "effect carried by a minority of items (as Qwen3-4B)". Read "no detectable effect" only when both are null. Implement this in code and test it.

M3. Route A has no file-hash check. Before the freeze, commit a manifest per model: file name, size and LFS SHA-256 at the pinned revision, from `HfApi().model_info(id, revision=sha, files_metadata=True)`, for exactly the files ALLOW/IGNORE selects. Also cross-check against `snapshot_download(dry_run=True)` under the same patterns if available. Building the manifest needs only Hub metadata, not downloads. Use the ICE serve2 python or a local venv with huggingface_hub, and do not print or use any token from pace/pace.env; public metadata calls should not need one. If a model is gated and metadata needs auth, stop and report rather than reading the token. After download in the job, hash every file and compare it with the manifest. Stop on any mismatch, missing file or extra file, and save the hashes in the shard's record. Test the comparison logic.

## SHOULD-FIX (apply all)

S1. Route A has no test that the download goes to node-local disk; a mutant deleting `export HF_HUB_CACHE="${LOCAL_CACHE}"` (sbatch line ~75) survived. Pass the cache folder to the prefetch tool explicitly (`--cache "${LOCAL_CACHE}"`), refuse any path under /storage, and test both.

S2. Add a disk-space check before download against the manifest's total bytes plus 10%.

S3. In local mode also set `HF_XET_CACHE="${LOCAL_CACHE}/xet"`.

S4. The environment check must enforce the GPU type. In `run_families_envcheck`, read the new shard's provenance GPU name and fail unless it matches a pinned constant per model: "NVIDIA L40S" for Qwen3-8B (earlier jobs 7174094–7174099 ran on L40S per their logs) and "NVIDIA A40" for Mistral 7B (per the design). Pin the sbatch constraint accordingly.

S5. Close the environment-gate leaks. (a) `waveb_predictions` (`wf/families.py:~1329`) sets B1 `holds` for `qwen3.5-9b-plain` even when the environment check failed; make it None unless the check passed. (b) The status lines (~1442 onward) print shares, R and effects even when the check failed; make `families_waveb_analysis` suppress all wave B score readings unless the check passed. Test both.

S6. Enforce `gpu_min_gb` at job start, and for the 27B/31B tags require SLURM_MEM_PER_NODE ≥ 128G.

S7. Add first-turn support to `family_gate_cpu` so `gemma4-e4b-it-firstturn` can be gated on CPU. Run what can run locally; note what must run on ICE.

S8. Qwen3.8's template writes an empty think block in every few-shot reply, and Qwen3's does not. Pre-register a 300-item check with `preserve_thinking=False` (aligned and swapped, both formats, about 1,800 passes, about 0.3 A100-h), read by the one-fifth rule as wave A's format checks are. Implement it as a config plus analysis.

S9. At the start of each Gemma 4 job, score about 5 items under both eager and SDPA attention and require agreement within a small stated tolerance, saved to the record; stop on disagreement.

S10. Fix the `first_turn_spec` note wording ("not the last" is Mistral's; it is wrong for Gemma). Add a test that `family_gate_cpu` exits non-zero on failure. Add a plain-format gate test where the two candidates share a first token (the mutant removing the distinct-first-token check in `plain_checks` survived).

## Decisions on the science wording (the study's author, recorded with the review)

- Generation pairs are confounded by size. Within the Qwen3 line the share runs 0.49 / 0.89 / 0.70 / 0.71 across 4B / 8B / 14B / 32B. State this spread beside every pair reading, and say a "differs" is not a generation effect.
- Gemma 4 E4B against Gemma 2 9B is the weakest pair; the reviewer suggests the size-matched Gemma-4-12B-it. Decision: swap E4B for Gemma-4-12B-it only if it exists on the Hub at a pinnable revision, uses the same chat template family and first-turn handling as E4B, and fits the same GPU tier. If any of those fails or is unclear, keep E4B and state the size confound explicitly.
- B2 is hard to fail ("inconclusive" counts as holding; equivalence within ±0.05 is nearly unreachable at wave A's CI widths). Say plainly that B2 can fail only on a clear negative difference.
- B3 (base vs instruct in plain format): pre-register now the two-candidate effect as the B3 readout if the base model fails the yes/no coverage criterion.
- The outcome table must say "goal-congruent shift", not "bias", and must not claim the share separates wishful prediction from task relevance.
- Pre-register the run order: environment check, then the four small models, then Qwen3.8 (with the S8 check), then Gemma 4 31B last.
