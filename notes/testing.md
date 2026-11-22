# Testing strategy and open testing work

The harness talks to a stateful text environment (ALFWorld through the AgentGym server) and to language models. Its worst bugs so far were silent: jobs finished and wrote plausible results, but the results were artefacts of the implementation. The worst was a simulator whose step was a pure function, while the real server keeps one live game per id (found 2026-11-05). Others were found on 2026-11-17: a parser that dropped failure predictions written in the model's own words, and a collector that skipped open/closed failures and ran out of memory at scale. Each layer below catches bugs the layers above cannot see.

## Layers in place

1. **Unit tests on real strings.** Parser, history reading, determination reasons, statistics. Cases come from real server text and real model replies wherever possible.
2. **Simulator runs** (`wf/env/sim.py` behind the real client). The full pipeline runs locally in seconds. This proves only that the logic is consistent with the simulator, not with the real server.
3. **Crash and resume tests.** Kill a run mid-way and check that the resumed result matches an uninterrupted run exactly.
4. **Bug checks (mutation testing, `scripts/mutation_check.py`).** Break the code on purpose; the tests must notice.
5. **Known-answer checks on the real server** (`wf/tools/verify_real_env.py`, `t0_*`). Expert solutions win, replay reproduces state, and sandbox replays never touch the live game.
6. **Smoke runs on train games** before any confirmatory run: fields, timing, memory.
7. **End-of-run replay** (collector, 2026-11-17). Each collection shard replays a seeded sample of its saved items on the real server. Any history or outcome that does not reproduce exactly fails the run. The collector also fails the run on a missing hidden state, or on a hidden state that does not list every object the live scan found.
8. **Independent review.** Fresh reviewers get the code and the spec, but not the author's reasoning. Blind raters handle anything that needs judgment (the parser audits).

## Added 2026-11-17 and 18, before the confirmatory runs (all in place)

These were planned on 2026-11-17 and were all in use before any confirmatory item was scored.

- **Full replay of every collected item** on CPU nodes (no GPU), not just a sample. History and true outcome must reproduce exactly. The attached success text must be what the server prints when the action works. No two items may be the same state and action. RQ2 scoring uses only items that pass.
- **Failure labels checked against the hidden state.** A failure labelled as decided by contents is checked against Word2World's hidden state, an independent source. Where they disagree, one of the two is wrong.
- **Rule and template conformance against the real server.** The simulator uses fixed made-up rooms, so real traces cannot be replayed through it. Instead, every collected item is checked against what the real server did. No item that the history-reading rules call a sure failure may have succeeded (shared-spot items are exempt, because there the rules are known not to prove failure). Every real reply must match a template the code knows. The live scene scan must place every object where Word2World's hidden state does. Every contents-decided failure must agree with the hidden state. On the 11 unseen games collected before the restart, the scan and the hidden state agreed on every object (14 to 56 per game), 88 contents labels agreed, and each planted error was flagged. (`wf/tools/verify_items.py`; RQ2 scoring refuses a collection shard without a full pass on its exact file.)
- **Result on the confirmatory collection.** All three shards passed, each verified in four parts (`runs/rq12_transitions_s<k>/VERIFY_part_<i>_of_4.json` on ICE; not in this repository). Over the 12,987 collected items (4,350, 4,473 and 4,164 per shard) from 134 games:
  - every item was replayed on the real server, and every history and true outcome reproduced;
  - the success texts of 7,207 items were regenerated, all exactly;
  - no item that the rules call a sure failure succeeded;
  - no reply fell outside the known templates;
  - in every game, the scene scan matched Word2World's hidden state for every object;
  - no contents-decided failure disagreed with the hidden state;
  - there were no duplicate ids and no duplicate state-and-action pairs.

  These checks show the labels are consistent with the real server and with an independent record of the layout. They do not show that a model reading only the history could know everything the rules use; see the shared-spot rule in `rq2_results.md`.
- **Calibration of the statistics** (`scripts/calibration_check.py`, run 2026-11-17, 500 simulated datasets per scenario, 134 games, about 20 items per game, the pilot's ICC and SD). Under no true effect, the game-level permutation test rejected 1.0% (Qwen3-like), 0.8% (Llama-like) and 0.6% (heavy-tailed) of datasets at p < 0.01, and 6.4%, 5.8% and 3.8% at p < 0.05, all within simulation noise of nominal. The percentile game-bootstrap 95% interval covered the true value 93.2%, 93.4% and 94.8% of the time, slightly narrow: a known small-sample property. Every bootstrap interval in the notes should be read with this in mind. It barely affects the verdicts, because the preregistered rule also requires the exactly calibrated permutation test at p < 0.01 after Holm. With a true effect of 0.5 nats and Llama-like noise, the test rejected 98% of datasets at p < 0.01.
- **Honest bug checks.** Each deliberate bug is run against the tests that cover its module. A timeout is reported as inconclusive, not as caught. The list in `scripts/mutation_check.py` holds 189 deliberate bugs at commit e7ca63b. They cover the collector, scene scan, goals and swaps, success twins, prompts, scorer, parser, statistics, the run gates and resume, RQ3, and the verification tools. The newer runners have no deliberate bugs yet: `wf/rq2.py`, `wf/rq1.py`, `wf/rq1x.py`, `wf/rq1d.py`, `wf/rq1v.py`, `wf/mech.py`, `wf/sizes.py` and `wf/thinking.py`. Mutants for them are being added (2026-11-20). The last recorded result is from 2026-11-17 and 18, when the list had 62 bugs: all were caught except these.
  - Two, rerun alone with a longer time limit, went uncaught by the full suite. The first was the collector building every alternative at every step, not only at sampled steps; a test now catches it. The second was Pilot A's advisory ΔL futility rule reading the wrong end of its interval. That rule served only the finished pilot, so it stays untested.
  - The scorer rounding to bfloat16 needs torch, which the laptop lacks. On ICE its test passes with the fix and fails without it.
- **Real-data smoke of the whole RQ2 pipeline** (2026-11-18, two train games, 231 items). Full verification passed: every item replayed, and all 77 success texts regenerated exactly. Scoring ran on an A100 in 6 minutes, and the analysis produced verdicts and every exploratory table. The smoke found three problems, all fixed before the confirmatory run:
  - the RQ2 configs named output folders the runner ignores, so the analysis could not find the scores;
  - the scorer rounded log-probabilities to bfloat16;
  - ICE's automatic `&cuda13` made the any-GPU constraint invalid.
- **Part-wise verification.** A shard's items can be replayed in parts, each a separate job with a fresh server, because the server leaks memory per reset. The RQ2 gate accepts a complete set of passing parts made on the same file, with every item replayed once.

## TODO (after 2026-11-19)

- [ ] **Randomised (property-based) tests of the parser and history reading.** Generate many plausible observations and replies from templates. Assert properties that must always hold:
  - every real success template is read as an effect;
  - "Nothing happens." is always no effect;
  - normalising text twice changes nothing;
  - classification is deterministic;
  - history facts never claim a location the agent never reached.
- [ ] **Scoring cross-check against a second implementation.** On a sample, compare the teacher-forced log-probabilities from `LogoddsScorer` with vLLM's prompt log-probabilities for the same text. Check that scoring the same text twice gives the same number, and that padding or batching does not change it. This targets tokenisation errors at the prompt–answer boundary.
- [ ] **Golden end-to-end snapshot.** A small fixed dataset runs through collection analysis, scoring analysis and the RQ1/RQ2/RQ3 reports, with every output frozen. Any unintended change to any stage shows up as a diff.
- [ ] **Automatic red flags in every report.** Flag:
  - outlying L values;
  - median first-token mass below a threshold;
  - empty or truncated generations;
  - games with unusual item counts;
  - large shifts in the mix of determination reasons or action types against the pilot;
  - any episode longer than the step cap.
- [ ] **Line-by-line simulator comparison.** Give `wf/env/sim.py` the layouts of a few real games, replay real traces through it, and compare every reply with the real server's. Any difference marks a place where a test passing in simulation means nothing.
- [ ] **Human spot-checks per run.** Twenty random items or episodes from each confirmatory run, read by a person, using the existing `HUMAN_REVIEW.md` export.

## Model families, wave A (2026-11-20)

`tests/test_families.py` covers the family runs: golden renders of one real item from each pinned chat template, the tokenization gate on stand-in tokenizers built to fail in one item or one prompt (a late success's candidate, one item's swapped yes/no prompt, a template that writes "Yes" as another token than the scorer reads), the gate's scope in the shard runner (every item, on a resume too, with the scorer's template settings), the analysis (flip definitions by name, the yes/no arms in the standardised effect, records filtered by model and by population), the pre-data primary metrics (the item share toward Yes and the goal-to-evidence ratio R, including their invariance to rescaled logits), the competence gate at each of its bounds, the TOST equivalence of R on paired items, the Holm-corrected primary family, and the Mistral placement run. Fix round 2 (third amendment) adds: the R comparison's fallback to the paired difference when the ratio is not estimable (review 2's synthetic models with R near zero, slightly positive and negative), the 2% cap on dropped draws, the item-share comparison (difference by game sign flips, TOST within ±0.05 as the larger one-sided p), the 90% and 95% interval rules for both comparisons on hand-set bootstrap draws, pairing across noisy games, the primary family padded with p = 1 for missing or NaN tests, the planned comparisons including Gemma 2 against Mistral, the two-sided noise floor, ties at log-odds zero, NaN scores, and game-clustered p-values on games that move together. With transformers, torch and the tokenizers cached (`WF_TEST_HF_CACHE`), the real templates run too. `scripts/mutation_check.py` holds 75 families mutants (`MUTANTS_MATCHING=families`), including the ten a review found surviving, seventeen for the primary metrics, gate, equivalence and family, review 2's M1 to M14 (two each for M6 and M7, R and share) and nine for fix round 2. All 75 are caught (2026-11-20). Two survived the first run and were killed by added tests: the TOST p taken from the smaller side (the test's draws had been symmetric) and R kept when the evidence effect is negative (the test had used an evidence effect of exactly zero). The refactored gate was also run against the earlier one on every real item with the pinned tokenizers, with identical results (`notes/families_design.md`).

## Model families, wave B, review 1 (2026-11-22, before any wave B score)

Review 1 of the wave B design ([families_waveb_review1.md](families_waveb_review1.md)) found that every chat-format score would have crashed in serve2, because transformers 5 returns a BatchEncoding from `apply_chat_template(tokenize=True)` and no test ran a real forward pass through the scorer. Three test layers were added.

- **`tests/test_families_waveb_serve2.py` (serve2 only).** Tiny random-weight models built from each pinned wave B config (two decoder layers, four for Gemma 4 E4B so that key-value sharing stays in; small widths; the real vocabulary), with each pinned tokenizer, run `score_pair` and `score_one_token_pair` end to end, and the one-pass yes/no score must equal the teacher-forced one. They also run the eager-against-SDPA switch on the real Gemma 4 class and the gate of the two check variants with the real templates. It skips everywhere but serve2. On ICE (2026-11-22, login node, read-only) all 11 tests passed; with the fix reverted, all five chat-format tags failed and the two plain-format tags passed, as they never call the template.
- **`tests/test_families_waveb_review1.py` (local).** A stand-in tokenizer that returns a BatchEncoding and a stand-in torch and model run the scorer's two scoring paths (fails on the unfixed scorer); the weights manifest and the hash check (missing, extra, size, LFS hash, Git blob hash, disk margin, `/storage` refusal, also behind a link, the folder passed by name); the environment check's GPU type; resources at job start; the CPU gate tool's variants and exit code; the plain gate with a shared first token; the first-turn note; the preserve_thinking variant and the one-fifth rule; the eager-SDPA agreement check and its tolerances; the run order.
- **`tests/test_families_waveb.py` (extended).** B1's readings (minority, no effect, reversed), the size caveat on generation pairs, B3's two-candidate fallback, the predictions read nothing unless the environment check passed, and the dry run: with a failed environment check no wave B number reaches the metrics, the report or the status lines.

**Mutation check, strict rule.** From this round a mutant counts as killed only when pytest exits 1 and names a failing test (`scripts/mutation_check.py`); any other non-zero exit is reported as "suite unrunnable", not as a kill. The 63 mutants named "families wave B review1" are listed below with the test that killed each; R1 to R8 put back the review's findings (R1 and R2 are the two survivors the reviewer named; R3 to R8 are the other findings written as mutants), the rest break each fix. Run with `MUTANTS_MATCHING="families wave B review1" MUTATION_TESTS_FIRST=tests/test_families_waveb_review1.py`, four shards.

| Mutant (families wave B review1 …) | Killed by |
|---|---|
| R1 node-local cache not exported to scoring | `tests/test_families_waveb_review1.py::test_serve2_job_downloads_node_local_and_checks_hashes_before_scoring` |
| R2 plain gate allows a shared first token | `tests/test_families_waveb_review1.py::test_plain_gate_fails_when_the_candidates_share_their_first_token` |
| R3 BatchEncoding passed to torch.cat and the model | `tests/test_families_waveb_review1.py::test_scoring_reads_the_ids_out_of_a_batch_encoding` |
| R4 B1 holds although the environment check failed | `tests/test_families_waveb.py::test_predictions_read_nothing_unless_the_environment_check_passed` |
| R5 status lines print wave B numbers when blocked | `tests/test_families_waveb.py::test_dry_run_scores_every_wave_b_model_and_reads_the_family` |
| R6 CPU gate tool exits 0 when a gate fails | `tests/test_families_waveb_review1.py::test_gate_tool_exits_non_zero_when_a_gate_fails[oks1-1]` |
| R7 Gemma first-turn note says not the last | `tests/test_families_waveb_review1.py::test_first_turn_note_is_right_for_each_template` |
| R8 envcheck GPU type not enforced | `tests/test_families_waveb_review1.py::test_envcheck_fails_on_the_wrong_gpu_type` |
| B3 fallback for any competence failure | `tests/test_families_waveb.py::test_b3_falls_back_to_the_two_candidate_share_only_when_the_base_fails_only_coverage` |
| B3 fallback holds on a negative difference | `tests/test_families_waveb.py::test_b3_falls_back_to_the_two_candidate_share_only_when_the_base_fails_only_coverage` |
| B3 fallback ignores the base two-candidate AUC | `tests/test_families_waveb.py::test_b3_falls_back_to_the_two_candidate_share_only_when_the_base_fails_only_coverage` |
| B3 fallback on the yes/no rows | `tests/test_families_waveb.py::test_b3_falls_back_to_the_two_candidate_share_only_when_the_base_fails_only_coverage` |
| M1 template ids not unwrapped | `tests/test_families_waveb_review1.py::test_scoring_reads_the_ids_out_of_a_batch_encoding` |
| M2 CI touching zero read as above it | `tests/test_families_waveb.py::test_b1_reading_separates_a_minority_effect_from_no_effect[not` |
| M2 minority effect read as no effect | `tests/test_families_waveb.py::test_b1_reading_separates_a_minority_effect_from_no_effect[not` |
| M2 reversed effect read as none | `tests/test_families_waveb.py::test_b1_reading_separates_a_minority_effect_from_no_effect[not` |
| M3 Git blob hash without its header | `tests/test_families_waveb_review1.py::test_file_digest_matches_git_and_lfs` |
| M3 extra file not reported | `tests/test_families_waveb_review1.py::test_verify_snapshot_fails_on_any_difference[change3-extra:` |
| M3 hash failure does not fail the download | `tests/test_families_waveb_review1.py::test_prefetch_checks_disk_before_and_hashes_after` |
| M3 hash not compared | `tests/test_families_waveb_review1.py::test_verify_snapshot_fails_on_any_difference[change0-hash:` |
| M3 hashes not saved in the shard | `tests/test_families_waveb_review1.py::test_the_gate_saves_the_weights_hashes_in_the_shard` |
| M3 manifest revision not checked | `tests/test_families_waveb_review1.py::test_committed_manifests_cover_every_wave_b_model_at_its_pinned_revision` |
| M3 missing file not reported | `tests/test_families_waveb_review1.py::test_verify_snapshot_fails_on_any_difference[change4-missing:` |
| M3 scoring accepts a failed hash check | `tests/test_families_waveb_review1.py::test_weights_check_requires_a_passed_record_for_this_model_and_revision` |
| M3 scoring accepts another revision | `tests/test_families_waveb_review1.py::test_weights_check_requires_a_passed_record_for_this_model_and_revision` |
| M3 size not compared | `tests/test_families_waveb_review1.py::test_verify_snapshot_fails_on_any_difference[change2-size:` |
| M3 weights check skipped for node-local weights | `tests/test_families_waveb_review1.py::test_weights_check_requires_a_passed_record_for_this_model_and_revision` |
| S1 job prefetch without --node-local | `tests/test_families_waveb_review1.py::test_serve2_job_downloads_node_local_and_checks_hashes_before_scoring` |
| S1 links not resolved | `tests/test_families_waveb_review1.py::test_node_local_cache_behind_a_link_to_storage_is_refused` |
| S1 node-local folder not checked | `tests/test_families_waveb_review1.py::test_prefetch_node_local_downloads_into_the_named_cache` |
| S1 prefetch ignores --cache | `tests/test_families_waveb_review1.py::test_prefetch_node_local_downloads_into_the_named_cache` |
| S1 shared storage below /storage accepted | `tests/test_families_waveb_review1.py::test_node_local_cache_is_never_on_the_shared_file_system[/storage/ice1/2/1/gmat3/hfcache-True]` |
| S2 download despite too little space | `tests/test_families_waveb_review1.py::test_prefetch_checks_disk_before_and_hashes_after` |
| S2 no disk margin | `tests/test_families_waveb_review1.py::test_check_disk_needs_the_manifest_total_plus_ten_percent` |
| S3 xet cache on shared storage | `tests/test_families_waveb_review1.py::test_serve2_job_downloads_node_local_and_checks_hashes_before_scoring` |
| S4 Qwen3-8B pinned to the wrong GPU | `tests/test_families_waveb_review1.py::test_envcheck_gpu_must_be_the_pinned_type[jobs0-True]` |
| S4 any GPU type accepted | `tests/test_families_waveb_review1.py::test_envcheck_gpu_must_be_the_pinned_type[jobs2-False]` |
| S4 job does not stop on the wrong GPU | `tests/test_families_waveb_review1.py::test_serve2_job_stops_an_envcheck_on_another_gpu_type` |
| S5 B2 estimate shown when blocked | `tests/test_families_waveb.py::test_predictions_read_nothing_unless_the_environment_check_passed` |
| S5 comparisons kept when blocked | `tests/test_families_waveb.py::test_dry_run_scores_every_wave_b_model_and_reads_the_family` |
| S5 competence verdict read when blocked | `tests/test_families_waveb.py::test_environment_gate_blocks_every_read_verdict_unless_passed` |
| S5 estimates kept when blocked | `tests/test_families_waveb.py::test_environment_gate_blocks_every_read_verdict_unless_passed` |
| S5 wave B blocks written when blocked | `tests/test_families_waveb.py::test_dry_run_scores_every_wave_b_model_and_reads_the_family` |
| S6 GPU memory not enforced | `tests/test_families_waveb_review1.py::test_resources_are_checked_before_loading[gemma4-31b-it-47.7-131072-False]` |
| S6 Qwen3.8 job memory not required | `tests/test_families_waveb_review1.py::test_resources_are_checked_before_loading[qwen3.8-27b-85.1-131071-False]` |
| S6 job memory compared in GB not MB | `tests/test_families_waveb_review1.py::test_resources_are_checked_before_loading[gemma4-31b-it-85.1-65536-False]` |
| S6 resources not checked at job start | `tests/test_families_waveb.py::test_dry_run_scores_every_wave_b_model_and_reads_the_family` |
| S7 first-turn variant not resolved | `tests/test_families_waveb_review1.py::test_gate_tool_resolves_check_variants_and_gates_their_shards_items` |
| S7 variants gated on every item | `tests/test_families_waveb_review1.py::test_gate_tool_resolves_check_variants_and_gates_their_shards_items` |
| S8 one-fifth rule at one half | `tests/test_families_waveb_review1.py::test_one_fifth_rule[0.5-ci0-2.0-True]` |
| S8 one-fifth rule ignores the CI | `tests/test_families_waveb_review1.py::test_one_fifth_rule[0.5-ci3-2.0-False]` |
| S8 template check reads the first-turn tag | `tests/test_families_waveb.py::test_dry_run_scores_every_wave_b_model_and_reads_the_family` |
| S8 think blocks before the ending not checked | `tests/test_families_waveb_review1.py::test_the_gate_finds_think_blocks_in_the_examples` |
| S8 variant keeps the think blocks | `tests/test_families_waveb_review1.py::test_gate_tool_resolves_check_variants_and_gates_their_shards_items` |
| S9 Gemma 4 31B without the attention check | `tests/test_families_waveb_review1.py::test_the_gate_runs_the_attention_check_for_gemma4_and_stops_on_disagreement` |
| S9 disagreement does not stop the job | `tests/test_families_waveb_review1.py::test_the_gate_runs_the_attention_check_for_gemma4_and_stops_on_disagreement` |
| S9 failed switch to eager not reported | `tests/test_families_waveb_review1.py::test_attention_agreement_fails_on_disagreement_or_no_switch[model_kw3-scorer_kw3-could` |
| S9 gate skips the attention check | `tests/test_families_waveb_review1.py::test_the_gate_runs_the_attention_check_for_gemma4_and_stops_on_disagreement` |
| S9 no switch back to SDPA | `tests/test_families_waveb_review1.py::test_attention_agreement_passes_within_the_tolerance` |
| S9 scores read from the cache | `tests/test_families_waveb_review1.py::test_attention_agreement_passes_within_the_tolerance` |
| S9 whole-model attention switch | `tests/test_families_waveb_review1.py::test_attention_agreement_passes_within_the_tolerance` |
| S9 yes/no tolerance loosened | `tests/test_families_waveb_review1.py::test_attention_agreement_passes_within_the_tolerance` |
| pair readings without the size caveat | `tests/test_families_waveb.py::test_b2_generation_pairs_carry_the_size_caveat` |

All 63 killed under the strict rule, by 31 different tests; none survived, timed out or left the suite unrunnable. The 72 earlier wave B mutants were rerun under the strict rule on the same code: 72 of 72 killed (both lists, with the killing test of each, are in `notes/families_waveb_mutants.txt`).
