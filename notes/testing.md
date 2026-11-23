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

## Decision simulation, Option A (2026-11-20)

`tests/test_decisionsim.py` covers `wf/decisionsim.py` on synthetic states: the agent's action read from longer prefixes (and a conflict refused), the eligibility rules (proposal among the candidates, a success, a determined failure; the last step left out), the seeded order-free sample, one swap object per state never named by a candidate, the jobs (a state's arms together, undefined swaps skipped, hashes that change with the arm and the swap), the prompts (the earlier yes/no prompts; Gemma's fold), each rule (alphabetical tie-break, verification accepting only above zero and never falling back to the rejected proposal, the generated-only set, rule 3 only where a goal-relevant candidate works), the contrast counts and the harm-minus-help identity, game-level bootstrap and sign-flip permutation, rule 2's error rates, the baseline, incomplete states dropped, the Option C rule (including that a secondary model never decides it), the Yes/No gate on stand-in tokenizers, and mock runs end to end (resume, refusal of another sample's scores, scores saved one at a time, a failed gate scoring nothing). `scripts/mutation_check.py` holds 18 decision-simulation mutants (`MUTANTS_MATCHING=decisionsim`); all 18 are caught (2026-11-20).

Fix round 1 (after review 1, before any score) adds tests for: odd-step states left out of every state-level rate (contrast, calibration, lure rate, baseline) and false acceptance reported by parity; scores rounded to 1/64 nat, with decisions, ties and the Yes threshold reading the rounded scores and the mean scores the raw ones; the seeded per-state tie hash; the harm rate, the change rate split by outcome, the proposal share and the failure rate by source; rule 2 by acceptance cell (contributions add up); tie counts, mixed-outcome ties and the even-split sensitivity; the baseline and its shifts by source and goal relevance; the calibration margins; the lure rate; the swapped decomposition; every Option C category at its bounds (harm exactly 5%, half exactly, p exactly 0.05, an interval that reaches zero), rule 1 and secondary models never deciding, waiting for the swapped arm, and the rule on a computed report; each condition's own prompt in a recorded scoring run; and review 1's surviving mutants (false rejection at a score of zero, the swap draw seeded by the state, a state without a swap kept when swapped is scored, the random-choice rate over the rule's own set, shards split by game, duplicate candidates refused, ties counted, the determined share's denominator, missing arms refused, the aligned baseline on every state, the gate on a resume checking every prompt). 

`scripts/mutation_check.py` now holds 72 decision-simulation mutants: the first 18 adapted to the new code, review 1's 17 (adapted; M1 to M3, M6 to M8, M10, M11 and M13 to M15 had survived the first tests) and 37 for the new code. All 72 are caught (`MUTANTS_MATCHING=decisionsim`, six shards, 2026-11-20). The table gives, for each, the first test in `tests/test_decisionsim.py` that fails with the mutant in place (run with `-x`, so a later test may catch it too).

| Mutant | First test that fails |
|---|---|
| review1 M1 a succeeding proposal at zero is not a false rejection | `test_decisions_ties_and_the_threshold_read_rounded_scores` |
| review1 M2 swap draw ignores the state index | `test_one_swap_object_per_state_never_named_by_a_candidate` |
| review1 M3 complete_states needs swapped scores where no swap exists | `test_complete_states_keep_a_state_without_a_swap_when_swapped_is_scored` |
| review1 M4 every arm scored with the aligned prompt | `test_each_condition_gets_its_own_prompt` |
| review1 M5 neutral scored with the aligned prompt | `test_each_condition_gets_its_own_prompt` |
| review1 M6 random-choice rate over all candidates for every rule | `test_random_choice_rate_is_over_the_rules_own_set` |
| review1 M7 shard ignored | `test_shards_split_the_states_by_game` |
| review1 M8 duplicate-candidate check removed | `test_duplicate_candidate_actions_are_refused` |
| review1 M9 Option C reads the interval's upper bound | `test_option_c_categories (interval reaching zero)` |
| review1 M10 tie count reports nothing | `test_decisions_ties_and_the_threshold_read_rounded_scores` |
| review1 M11 determined share over all chosen actions | `test_contrast_counts_harm_help_and_the_identity` |
| review1 M12 rejected proposal chosen when it scores highest | `test_rule2_accepts_a_proposal_only_above_zero_and_falls_back_to_the_others` |
| review1 M13 analysis reads a model without both arms | `test_decisions_ties_and_the_threshold_read_rounded_scores` |
| review1 M14 every baseline only on states with a swap | `test_model_report_adds_swapped_contrasts_only_where_scored` |
| review1 M15 gate checks only the prompts still to score | `test_the_gate_on_a_resume_checks_every_prompt_not_only_the_unscored` |
| review1 M16 eligibility drops the success requirement | `test_eligible_states_need_the_proposal_a_success_and_a_determined_failure` |
| review1 M17 changes toward success never reported | `test_option_c_categories (toward success)` |
| fix1 Option C harm at the bound fails | `test_option_c_categories (harm exactly 5%)` |
| fix1 Option C harm threshold 10% | `test_option_c_categories (default case, worth doing)` |
| fix1 Option C ignores p | `test_option_c_categories (p exactly 0.05)` |
| fix1 Option C p at the bound passes | `test_option_c_categories (p exactly 0.05)` |
| fix1 Option C reads rule 1 | `test_option_c_categories (default case, worth doing)` |
| fix1 Option C without the swapped arm | `test_option_c_waits_for_both_primary_models_and_their_swapped_arm` |
| fix1 a secondary model decides Option C | `test_option_c_reads_rule2_only_and_primary_models_only` |
| fix1 acceptance cells swapped | `test_rule2_difference_splits_by_acceptance_cell` |
| fix1 acceptance contributions per cell | `test_rule2_difference_splits_by_acceptance_cell` |
| fix1 attribution ignores the half | `test_option_c_categories (swapped part below half)` |
| fix1 attribution ignores the sign | `test_option_c_categories (swapped part negative)` |
| fix1 baseline on every state | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| fix1 broad generator every step | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| fix1 calibration at odd steps | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| fix1 calibration margin under the aligned task | `test_calibration_margin_and_the_aligned_shift_on_the_best_failure` |
| fix1 decisions on raw scores | `test_decisions_ties_and_the_threshold_read_rounded_scores` |
| fix1 decomposition at odd steps | `test_decomposition_adds_up_on_the_states_with_a_swap` |
| fix1 decomposition where no swap exists | `test_decomposition_adds_up_on_the_states_with_a_swap` |
| fix1 false rejection at odd steps | `test_verification_error_rates_by_parity` |
| fix1 groups by goal instead of source | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| fix1 harm counts help too | `test_contrast_counts_harm_help_and_the_identity` |
| fix1 lure from a failing proposal | `test_lure_rate_pairs_a_working_proposal_with_each_goal_object_failure` |
| fix1 lure from any failure | `test_lure_rate_pairs_a_working_proposal_with_each_goal_object_failure` |
| fix1 mixed-outcome ties are any ties | `test_ties_are_counted_and_split_evenly_in_the_sensitivity_readout` |
| fix1 odd-step states kept in contrasts | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| fix1 pooled false acceptance drops odd steps | `test_verification_error_rates_by_parity` |
| fix1 proposal share inverted | `test_contrast_counts_harm_help_and_the_identity` |
| fix1 rounded to a 16th | `test_scores_are_rounded_to_a_64th_of_a_nat_and_the_raw_ones_kept` |
| fix1 scores not rounded | `test_scores_are_rounded_to_a_64th_of_a_nat_and_the_raw_ones_kept` |
| fix1 shifts in rounded scores | `test_baseline_shifts_by_source_and_goal` |
| fix1 success-to-success changes counted as failure-to-failure | `test_contrast_counts_harm_help_and_the_identity` |
| fix1 swapped arm not scored by default | `test_every_model_is_registered_and_configured` |
| fix1 tie hash ignores the seed | `test_rule1_takes_the_largest_score_and_breaks_ties_by_a_seeded_state_hash` |
| fix1 tie hash ignores the state | `test_rule1_takes_the_largest_score_and_breaks_ties_by_a_seeded_state_hash` |
| fix1 tie split reads the chosen action | `test_ties_are_counted_and_split_evenly_in_the_sensitivity_readout` |
| fix1 toward success read from the lower bound | `test_option_c_categories (p exactly 0.05)` |
| fix1 verdict takes the last category | `test_option_c_categories (default case, worth doing)` |
| Option C ignores the direction | `test_option_c_categories (interval reaching zero)` |
| analysis reads another sample's scores | `test_end_to_end_score_resume_and_analysis` |
| baseline Yes read from raw scores | `test_baseline_reads_yes_as_a_positive_score_and_splits_by_source_and_goal` |
| bootstrap resamples states | `test_game_bootstrap_resamples_games_not_states` |
| family gate skipped | `test_a_failed_gate_scores_nothing` |
| generated-only set keeps the proposal | `test_rule1_takes_the_largest_score_and_breaks_ties_by_a_seeded_state_hash` |
| incomplete states kept | `test_complete_states_drop_a_state_with_any_unscored_candidate` |
| permutation flips states | `test_permutation_flips_whole_games` |
| resume trusts saved scores | `test_resume_refuses_scores_of_another_sample` |
| rule 3 read where it must fail | `test_rule3_reads_goal_relevant_candidates_where_one_of_them_works` |
| sample depends on input order | `test_sample_is_seeded_sorted_and_all_means_all` |
| scores saved at the end | `test_end_to_end_score_resume_and_analysis` |
| states need no determined failure | `test_eligible_states_need_the_proposal_a_success_and_a_determined_failure` |
| swap object may be a candidate's object | `test_one_swap_object_per_state_never_named_by_a_candidate` |
| swapped line from each candidate's own draw | `test_one_swap_object_per_state_never_named_by_a_candidate` |
| tie goes to the last action | `test_rule1_takes_the_largest_score_and_breaks_ties_by_a_seeded_state_hash` |
| verification accepts a score of zero | `test_rule2_accepts_a_proposal_only_above_zero_and_falls_back_to_the_others` |
| verification falls back to the rejected proposal | `test_rule2_accepts_a_proposal_only_above_zero_and_falls_back_to_the_others` |

## Model families, wave A (2026-11-20)

`tests/test_families.py` covers the family runs: golden renders of one real item from each pinned chat template, the tokenization gate on stand-in tokenizers built to fail in one item or one prompt (a late success's candidate, one item's swapped yes/no prompt, a template that writes "Yes" as another token than the scorer reads), the gate's scope in the shard runner (every item, on a resume too, with the scorer's template settings), the analysis (flip definitions by name, the yes/no arms in the standardised effect, records filtered by model and by population), the pre-data primary metrics (the item share toward Yes and the goal-to-evidence ratio R, including their invariance to rescaled logits), the competence gate at each of its bounds, the TOST equivalence of R on paired items, the Holm-corrected primary family, and the Mistral placement run. Fix round 2 (third amendment) adds: the R comparison's fallback to the paired difference when the ratio is not estimable (review 2's synthetic models with R near zero, slightly positive and negative), the 2% cap on dropped draws, the item-share comparison (difference by game sign flips, TOST within ±0.05 as the larger one-sided p), the 90% and 95% interval rules for both comparisons on hand-set bootstrap draws, pairing across noisy games, the primary family padded with p = 1 for missing or NaN tests, the planned comparisons including Gemma 2 against Mistral, the two-sided noise floor, ties at log-odds zero, NaN scores, and game-clustered p-values on games that move together. With transformers, torch and the tokenizers cached (`WF_TEST_HF_CACHE`), the real templates run too. `scripts/mutation_check.py` holds 75 families mutants (`MUTANTS_MATCHING=families`), including the ten a review found surviving, seventeen for the primary metrics, gate, equivalence and family, review 2's M1 to M14 (two each for M6 and M7, R and share) and nine for fix round 2. All 75 are caught (2026-11-20). Two survived the first run and were killed by added tests: the TOST p taken from the smaller side (the test's draws had been symmetric) and R kept when the evidence effect is negative (the test had used an evidence effect of exactly zero). The refactored gate was also run against the earlier one on every real item with the pinned tokenizers, with identical results (`notes/families_design.md`).




## Stage 1b: the agent's own ranked candidates (2026-11-20; rebuilt after review 1 and amended after review 2, 2026-11-21)

`tests/test_decisionsim_agent.py` (93 tests) covers `wf/decisionsim_agent.py`. On synthetic data: the candidate lists (rank 1 is the recorded greedy action even when never sampled, its sampled variants merged into it, the rest by count with the seeded hash for ties and independent of reply order, format errors counted and never turned into `look`, the cut at `k_max`, normalisation and its per-list count, the action key shared by both lists), the sampling settings of both lists, the greedy diagnostic and its comparison of normalised actions, a pinned seed value, bookkeeping (identical replies, truncation, greedy agreement and the 90% drift rule at its boundary), the seed check (primary settings, sequences not multisets, the same-seed repeat) and the smoke gate (format errors, replay mismatches, a non-degenerate seed-check state, the required truncation and agreement readouts); the verifier prompts (for each model the aware, blind, swapped and neutral prompts differ only in the query's task line, examples keep theirs, the aware, swapped and neutral prompts equal the earlier `yn_*` prompts), the blind deletion's guards, the blind-only rule at single-candidate states; list views (each list's order over shared labels, runnable candidates only, the sampled-twice and first-occurrence filters); the selection rules; regime, encounter, strict encounter, reachability, step bands (encounters, not reachable ones); the futility rule after review 2 (games with an encounter state that also lists a working candidate, the 6- and 7-game boundary, review 2's 8-game counterexample that is significant with no reachable encounter); harm and help with the identity and the kinds of harm counted on harm states only; the sign-flip floor; leave-one-game-out; the decomposition and the attribution rule; whole-game bootstrap and permutation; Holm; the verdict reading the Holm p; the verdict boundaries (a lower bound of exactly zero; a regime share of exactly one half carries no suffix, below it the "not evidence for the main finding" suffix); a failed gate entering Holm with p = 1; the overall and conditional blocks each reading their own p, decomposition and states; the AUC and the competence thresholds; the model report and every sensitivity, the temperature-1 readouts reading the temperature-1 lists; shards with different settings refused; the run-order check (nothing written, another label run, an analysis with models, an unfinished one, a logged override); progress events; the swap object, which excludes every candidate of both lists. On the simulator, with the mock agent and scorer: everything listed for review 1, plus scoring refusing to start before the model-free analysis of the same labels, the logged override, the model-free analysis reporting the temperature-1 encounters from the temperature-1 lists, the labelling budget counting the next state, and the request settings and normalisation counts recorded by a sampling shard.

`scripts/mutants_decisionsim_agent.py` holds 157 Stage 1b mutants (`MUTANTS_MATCHING="dsa "`): the review-1 list (two adapted to the code changed after review 2), review 2's 42 mutants (adapted where the code had changed; named "review2 ..."), and 13 mutants of the code added after review 2. All 157 are caught, by the per-mutant runs below and by `scripts/mutation_check.py` in six shards (2026-11-21). Review 2's A6, A9 and B4 and two new run-order mutants first survived and were killed by tests added for them. The table gives, for each, the first test in `tests/test_decisionsim_agent.py` that fails with the mutant in place (run with `-x`).

| Mutant | First test that fails |
|---|---|
| greedy action also listed again below rank 1 | `test_greedy_samples_merge_into_rank1_and_keep_the_recorded_string` |
| ties by order of appearance | `test_ranked_by_count_then_seeded_hash_not_order_of_appearance` |
| rarest action ranked first | `test_ranked_by_count_then_seeded_hash_not_order_of_appearance` |
| list one longer than k_max | `test_list_is_cut_at_k_max_and_the_cut_is_recorded` |
| actions not normalised | `test_normalize_collapses_space_case_and_final_stop` |
| thinking on | `test_real_sampler_sends_the_designed_payload` |
| agent prompt without admissible actions | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| history turn shows the next observation | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| replay mismatch ignored | `test_replay_mismatch_stops_the_game_there` |
| resume trusts lists from another prompt | `test_sample_resume_draws_nothing_new_and_refuses_another_prompt` |
| fresh sample run appends to old lists | `test_a_fresh_sample_run_does_not_append_to_old_lists` |
| lists saved at the end of each game | `test_sample_resume_draws_nothing_new_and_refuses_another_prompt` |
| rank 1 not checked against the episode | `test_rank1_mismatch_makes_the_state_unusable` |
| every failure determined | `test_label_state_marks_determination_and_goal_agnostic_history` |
| fresh label run appends to old labels | `test_label_stops_partial_and_resumes_to_the_same_labels` |
| label does not wait for sampling | `test_label_waits_for_every_sampling_shard` |
| swap may be named by a candidate | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| swap may be acted on in the history | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| swap may sit at a candidate's receptacle | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| progress from any goal mention | `test_progress_events` |
| progress from a move anywhere | `test_progress_events` |
| blind prompt strips the examples too | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| swapped arm shows the real task | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| a score of zero accepted | `test_selection_rules` |
| fallback to the last candidate | `test_selection_rules` |
| oracle picks a failure | `test_selection_rules` |
| scores not rounded | `test_scores_are_rounded_to_a_64th_before_the_threshold` |
| coarser grid | `test_scores_are_rounded_to_a_64th_before_the_threshold` |
| server-error candidates kept | `test_arms_at_a_single_candidate_state_is_blind_only` |
| regime ignores shared spots | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| regime without determination | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| regime without the goal object | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| strict encounter ignores the history | `test_encounter_strict_and_reachable` |
| reachable past a success | `test_encounter_strict_and_reachable` |
| bootstrap resamples states | `test_inference_resamples_and_flips_whole_games` |
| permutation flips states | `test_inference_resamples_and_flips_whole_games` |
| Holm not monotone | `test_holm_and_verdicts` |
| missing p counts as zero | `test_holm_and_verdicts` |
| Holm over the models that are in | `test_a_failed_gate_enters_holm_with_p_one` |
| toward failure read from the upper bound | `test_holm_and_verdicts` |
| analysis reads another label run's scores | `test_analysis_waits_and_refuses_scores_of_another_label_run` |
| harm and help swapped | `test_reviewer_counterexample_eight_unreachable_games_can_be_significant` |
| decomposition part reversed | `test_decomposition_on_states_with_a_swap_and_its_identity` |
| score resume trusts saved scores | `test_score_resume_and_refusal_of_other_prompts` |
| family gate skipped | `test_a_failed_gate_scores_nothing_and_enters_holm_with_p_one` |
| shards overlap | `test_end_to_end_every_decision_point_once` |
| archived history without action marks | `test_sampling_history_text_is_the_world_model_history` |
| tempered t1 list | `test_sampling_settings_after_review_1` |
| primary list untruncated | `test_sampling_settings_after_review_1` |
| one seed for every state | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| label replay budget ignored | `test_label_stops_partial_and_resumes_to_the_same_labels` |
| blind prompt keeps the task line | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| incompetent verifier interpreted | `test_incompetent_models_stay_in_holm_and_harm_from_loops_is_flagged` |
| swapped scores not required | `test_complete_labs_need_every_arm_where_defined` |
| swapped arm never scored | `test_arms_at_a_single_candidate_state_is_blind_only` |
| scores saved at the end | `test_scores_are_saved_one_at_a_time` |
| R3 first 12 steps includes step 12 | `test_first_12_steps_means_steps_0_to_11` |
| R4 by_encounter uses strict | `test_by_encounter_uses_the_main_not_the_strict_label` |
| R5 goal-agnostic inverted | `test_label_state_marks_determination_and_goal_agnostic_history` |
| R10 verifier value sign flipped | `test_verifier_value_sign_and_progress_direction` |
| R11 game weighted sums states | `test_game_weighted_delta_weighs_games_equally` |
| R13 progress transforms any object | `test_progress_needs_the_task_transformation_no_repeats_and_the_object_in_hand_for_the_lamp` |
| R15 analysis keeps unusable states | `test_rank1_mismatch_makes_the_state_unusable` |
| R17 k cut before dropping server errors | `test_server_error_candidates_are_skipped_before_the_k_cut` |
| R23 step bands off by one | `test_encounter_bands_and_the_futility_rule` |
| R24 competence needs more than the threshold | `test_auc_and_competence` |
| R26 acceptance counts zero as Yes | `test_acceptance_counts_zero_as_no` |
| R28 progress difference reversed | `test_verifier_value_sign_and_progress_direction` |
| R29 replayed history not checked | `test_a_history_that_does_not_replay_makes_the_state_unusable` |
| R8 verdict reads the raw p | `test_the_verdict_reads_the_holm_p_not_the_raw_p` |
| gate failure not p = 1 | `test_a_failed_gate_enters_holm_with_p_one` |
| gate failure ignored by the analysis | `test_a_failed_gate_scores_nothing_and_enters_holm_with_p_one` |
| attribution ignores the half | `test_attribution_rule` |
| attribution ignores the sign | `test_attribution_rule` |
| conditional on all states | `test_model_report_parts_add_up_and_readouts` |
| conditional verdict without its own Holm | `test_end_to_end_every_decision_point_once` |
| blind guard: any task line | `test_blind_history_deletes_only_the_real_task_line_in_the_first_observation` |
| sampling history not saved | `test_end_to_end_every_decision_point_once` |
| history mismatch still usable | `test_a_history_that_does_not_replay_makes_the_state_unusable` |
| union labels only the primary list | `test_label_state_marks_determination_and_goal_agnostic_history` |
| views key by rank not action | `test_views_follow_each_list_and_share_labels` |
| single-candidate states score every arm | `test_arms_at_a_single_candidate_state_is_blind_only` |
| single-candidate states need aware scores | `test_complete_labs_need_every_arm_where_defined` |
| twice keeps singletons | `test_sampled_twice_and_first_occurrence_filters` |
| first occurrence keeps repeats | `test_sampled_twice_and_first_occurrence_filters` |
| sign-flip floor off by one | `test_contrast_counts_harm_help_and_the_identity` |
| futility threshold six games | `test_encounter_bands_and_the_futility_rule` |
| futility counts encounters not reach | `test_encounter_bands_and_the_futility_rule` |
| leave-one-out keeps the game | `test_leave_one_game_out` |
| progress repeats count | `test_progress_needs_the_task_transformation_no_repeats_and_the_object_in_hand_for_the_lamp` |
| progress any transformation | `test_progress_needs_the_task_transformation_no_repeats_and_the_object_in_hand_for_the_lamp` |
| lamp counts without the object | `test_progress_needs_the_task_transformation_no_repeats_and_the_object_in_hand_for_the_lamp` |
| sampler drops the fixed penalties | `test_real_sampler_sends_the_designed_payload` |
| greedy diagnostic not greedy | `test_sampling_settings_after_review_1` |
| drift threshold | `test_sampling_settings_after_review_1` |
| smoke gate ignores the seed check | `test_bookkeeping_flags_identical_replies_drift_and_seed_check` |
| neutral arm never scored | `test_arms_at_a_single_candidate_state_is_blind_only` |
| neutral arm shows the real task | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| inert rank-1 share counts fallbacks only | `test_model_report_parts_add_up_and_readouts` |
| harm share from all failures | `test_harm_kinds_count_harm_states_only` |
| review2 S1 seed check uses t1 params | `test_seed_check_uses_the_primary_settings_and_compares_sequences` |
| review2 S2 seed check compares sorted multiset | `test_seed_check_uses_the_primary_settings_and_compares_sequences` |
| review2 S3 smoke gate drops nondegenerate>0 requirement | `test_smoke_gate_needs_a_non_degenerate_seed_check_state_and_reports_readouts` |
| review2 S4 smoke gate format <= 0.25 | `test_bookkeeping_flags_identical_replies_drift_and_seed_check` |
| review2 S5 fresh greedy agrees raw compare | `test_fresh_greedy_agreement_compares_normalised_actions_and_the_drift_boundary` |
| review2 S6 sample seed 7 hex digits | `test_sample_seed_value_is_pinned` |
| review2 S7 request_settings omit FIXED | `test_request_settings_and_normalisation_counts_are_recorded` |
| review2 S8 server_info wrong path | `test_real_sampler_sends_the_designed_payload` |
| review2 S9 saved history differs from served history | `test_end_to_end_every_decision_point_once` |
| review2 S10 drift rule <= | `test_fresh_greedy_agreement_compares_normalised_actions_and_the_drift_boundary` |
| review2 L1 union T1 first | `test_label_state_marks_determination_and_goal_agnostic_history` |
| review2 L2 history check only first replay | `test_every_replay_of_a_state_must_reproduce_its_history` |
| review2 L3 determination on replayed history | `test_labels_match_a_fresh_replay_on_the_simulator` |
| review2 L4 admissible case-insensitive | `test_end_to_end_every_decision_point_once` |
| review2 L5 swap excludes only primary candidates | `test_the_swap_object_excludes_every_candidate_of_both_lists` |
| review2 L6 budget ignores current state | `test_label_budget_counts_the_next_state` |
| review2 L7 akey not normalised | `test_normalize_collapses_space_case_and_final_stop` |
| review2 F1 futility on encounter games | `test_encounter_bands_and_the_futility_rule` |
| review2 F2 futility <= | `test_encounter_bands_and_the_futility_rule` |
| review2 F3 analysis encounter on all labelled states | `test_rank1_mismatch_makes_the_state_unusable` |
| review2 F4 t1 encounter uses primary | `test_t1_encounter_report_in_the_analysis_reads_the_t1_lists` |
| review2 A1 conditional on reachable | `test_each_block_reads_its_own_estimand_p_decomposition_and_states` |
| review2 A2 harm_mostly_regime strict > | `test_verdict_boundaries` |
| review2 A3 attribution strict > | `test_attribution_rule` |
| review2 A4 verdict lo >= 0 | `test_verdict_boundaries` |
| review2 A5 t1 conditional uses all t1 | `test_t1_list_readouts_read_the_t1_list` |
| review2 A6 complete_labs ignores neutral | `test_complete_labs_need_every_arm_where_defined` |
| review2 A7 sampled_twice keeps count>=1 | `test_sampled_twice_and_first_occurrence_filters` |
| review2 A8 g_nz counts all games | `test_contrast_counts_harm_help_and_the_identity` |
| review2 A9 view ignores server errors | `test_views_count_only_runnable_candidates` |
| review2 A10 score uses all labs | `test_rank1_mismatch_makes_the_state_unusable` |
| review2 A11 conditional decomposition on all | `test_each_block_reads_its_own_estimand_p_decomposition_and_states` |
| review2 A12 mistral gate p=0 | `test_a_failed_gate_enters_holm_with_p_one` |
| review2 A13 reachable counts rank1 regime only | `test_encounter_strict_and_reachable` |
| review2 A14 encounter by step band uses reachable | `test_encounter_bands_count_encounters_not_reachable_ones` |
| review2 A15 regime ignores determination | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| review2 B4 conditional verdict reads overall decomposition | `test_each_block_reads_its_own_estimand_p_decomposition_and_states` |
| review2 B6 t1 list drawn with the primary seed | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| review2 B7 shards with different settings accepted | `test_shards_with_different_settings_are_refused` |
| review2 B8 conditional Holm uses overall p | `test_each_block_reads_its_own_estimand_p_decomposition_and_states` |
| review2 B9 seed check same seed as primary | `test_seed_check_on_the_smoke_states` |
| review2 B10 encounter strict ignores agnostic | `test_encounter_strict_and_reachable` |
| futility counts reachable games again | `test_encounter_bands_and_the_futility_rule` |
| verdict suffix never added | `test_incompetent_models_stay_in_holm_and_harm_from_loops_is_flagged` |
| verdict suffix at exactly half | `test_verdict_boundaries` |
| scoring ignores the run order | `test_scoring_waits_for_the_model_free_analysis_unless_overridden` |
| run order accepts another label run | `test_run_order_check_cases` |
| run order accepts an analysis with models | `test_scoring_waits_for_the_model_free_analysis_unless_overridden` |
| run order accepts an unfinished analysis | `test_run_order_check_cases` |
| run order override not logged | `test_scoring_waits_for_the_model_free_analysis_unless_overridden` |
| run order not saved | `test_scoring_waits_for_the_model_free_analysis_unless_overridden` |
| same-seed repeat uses another seed | `test_seed_check_uses_the_primary_settings_and_compares_sequences` |
| normalisation count counts every action | `test_request_settings_and_normalisation_counts_are_recorded` |
| smoke gate drops the greedy readout | `test_smoke_gate_needs_a_non_degenerate_seed_check_state_and_reports_readouts` |
| precision floor off by one | `test_precision_script_floor_and_holm_threshold` |
| format error becomes look | `test_format_errors_count_and_add_no_candidate` |
| seed check reads one seed | `test_seed_check_flags_a_seed_that_changes_nothing` |

## Option C: the closed-loop agent with a world-model verifier (2026-11-22; amended after review 1)

`tests/test_optionc.py` (77 tests) covers `wf/optionc.py`. Prompts: for every verifier model and both readouts, the aware, blind, swapped and neutral prompts differ only in the query's task line (blind deletes it, swapped replaces only the object, neutral is "explore the room"), the examples are identical, the yes/no prompts end with the yes/no question and the observation prompts with "Next observation:", and each equals the earlier runs' prompt (`sizes.arm_messages`, `families.family_messages` for Gemma). Readouts: yes/no log-odds and the observation readout's complement of "Nothing" (with "You" against "Nothing" reported), rounding to 1/64 nat before acceptance, a non-finite readout refused, the mock complement consistent, and the first-token gate (passing, split tokens, a template that writes the reply differently, no prompts). Decision rules: an accepted proposal is executed without sampling; accepted-only takes the first accepted alternative in rank order and stops scoring there, keeps the proposal when nothing is accepted or no alternative exists; argmax takes the highest alternative whatever its sign, ties by the seeded hash; the sample seed depends on the seed, game and prompt only. The cache: made once, saved, reloaded per game. Episodes on the simulator with stand-in models: no verifier executes the greedy proposal; an accepting verifier reproduces the agent and never samples; a rejecting one with nothing accepted abstains at every step; sampling uses the agent's own prompt and the designed settings and seed; an override records the proposal's true outcome from a fresh replay; the agent never sees the verifier; the verifier reads the live state's world-model history and its arm's task line; argument checks; common random numbers through the cache (shared prefix, a repeated episode identical); format errors become `look` and are flagged; alternatives cut at `k_max`; repeats and the score of a loss. Runs end to end on the mock agent, scorer and simulator: every episode once; shards split the games; a partial job resumes to the same episodes as an uninterrupted run; a fresh run starts a new cache; the smoke cache check and gate (and a cache that does not reproduce); games without a swap object skip only the swapped arm; bad configs refused; the observation readout's episodes wait for the step-0 analysis of their own model, and an override is logged. Analysis on synthetic episodes: per-game seed means, paired contrasts, the verdict categories with attribution at exactly half and below it, help and no difference, invalid actions read with the reversed sign, the interval and p both needed, the primary run's no-verifier episodes serving other runs, divergence, step rates by the proposal's true outcome, drift against the archived agent, and the analysis refusing incomplete, repeated or unfinished runs. The step-0 check: the four classifications and their thresholds at the boundaries, competence read under the blind arm, the naming lift and any-task drop, agreement with Stage 1a's rounded scores, the scoring run (every candidate, arm and readout; no swapped arm without a swap object; resume; refusing other prompts; waiting for the collection) and its analysis (waiting for every scoring run). Registry and configs.

**Added after review 1** (`notes/optionc_review1.md`). `tests/test_optionc.py` now collects 107 tests (one skips without torch). New: the yes/no readout read through `next_token_logprobs`; matched-state scoring (every aware step scored under the blind and swapped prompts, alternatives included, never acted on; none for other arms; no swapped shadow without a swap object) and its analysis on identical steps; seeds per arm (the swapped arm at one seed; a verifier arm with no seed refused); the obstruction check at its 10-point margin (exactly 10 is clear), not computable, not tested, and the in-loop naming lift; the attribution labels (attributable, not attributable, unresolved when the interval spans half the cost or the cost is not positive on the swap-object games, not identifiable, not tested); override harm minus help as a key secondary where more is worse; the decomposition on the swap-object games only (reviewer R3); the step-0 classification by interval with its boundaries, a 70%/70% observation readout labelled question-specific, an incompetent task-blind yes/no arm labelled not classifiable; failing `go to` kept out of the step-0 go-to readouts (R9); the order gate (no smoke, a smoke that failed, no step-0 report, the cost rule at exactly 1.5 times and above, a missing cost ceiling, Gemma's memory smoke, a logged override) and a run that waits for it; the projection from the smoke; the last-position check (adopted only if no rounded score moves) and the scorer's keyword fallbacks, plus a CPU equality test on a tiny model that runs where torch is installed; the memory smoke run and a recorded out-of-memory failure; the primary-run qualification, including the reallocation rule; a missing RQ3 archive recorded, not fatal; the new drift readouts; and the reviewer's R1 (seeds draw their own alternatives), R2 (a counterfactual classified at its own prefix), R4 (a failed Yes/No gate stops a Gemma run, a failed first-token gate an observation run), R5 (an unreported step-0 analysis), R6 (no alternative counts as an abstention), R7 (goal flags of proposal and chosen action), R8 (no rejections in no-verifier episodes) and R10 (a cache replay that differs while success does not). The test file uses 200 bootstrap resamples and permutations (patched in the tests only).

`scripts/mutants_optionc.py` now holds 131 mutants (`MUTANTS_MATCHING="optionc "`): the first 80 less five retired because review 1 replaced the code they targeted (attribution strict at half and attribution ignoring the sign, now covered by the interval rule and the no-cost case; the two old collapse thresholds and the old competence line, replaced by the interval classification), five adapted to the new code, review 1's ten (R9 split in two, so eleven), and 45 mutants of the code added after review 1. Each was applied to a copy of the repository and `tests/test_optionc.py` run with `-x`, under the strict rule: killed only if pytest exits 1 and names a failing test. All 131 are killed (2026-11-22); the table gives the first failing test. The real (non-mock) path of `next_token_logprobs` needs a model; its last-position branch is covered by the stub and tiny-model tests, and every run checks it on its gate prompts.

| Mutant | First test that fails |
|---|---|
| accept at zero | `test_rounding_decides_acceptance` |
| no rounding | `test_yes_no_readout_is_the_log_odds_rounded_to_a_64th` |
| blind keeps the task line | `test_arms_differ_only_in_the_query_task_line[yn-qwen3-8b]` |
| swapped arm shows the real task | `test_arms_differ_only_in_the_query_task_line[yn-qwen3-8b]` |
| neutral arm shows the real task | `test_arms_differ_only_in_the_query_task_line[yn-qwen3-8b]` |
| observation readout on the yes/no prompt | `test_arms_differ_only_in_the_query_task_line[obs-qwen3-8b]` |
| observation readout is You against Nothing | `test_observation_readout_is_the_complement_of_nothing` |
| observation readout sign flipped | `test_observation_readout_is_the_complement_of_nothing` |
| mock complement wrong | `test_mock_next_token_logprobs_complement_is_consistent` |
| mock yes/no rejects everything | `test_mock_yes_no_accepts_some_and_rejects_some` |
| gate passes split tokens | `test_first_token_gate_passes_and_catches_split_tokens` |
| gate ignores the template | `test_first_token_gate_passes_and_catches_split_tokens` |
| gate passes with no prompts | `test_first_token_gate_passes_and_catches_split_tokens` |
| proposal never accepted | `test_an_accepted_proposal_is_executed_without_sampling[accepted_only]` |
| accepted-only scans from the last rank | `test_accepted_only_takes_the_first_accepted_in_rank_order_not_the_best` |
| abstaining executes an alternative | `test_accepted_only_keeps_the_proposal_when_nothing_is_accepted` |
| no alternative executes look | `test_no_alternative_keeps_the_proposal[accepted_only]` |
| argmax takes the lowest | `test_argmax_takes_the_highest_alternative_whatever_its_sign` |
| argmax ties by order | `test_argmax_ties_go_to_the_smallest_seeded_hash` |
| argmax only among accepted | `test_argmax_takes_the_highest_alternative_whatever_its_sign` |
| sample seed ignores the prompt | `test_sample_seed_depends_on_seed_game_and_prompt_only` |
| sample seed ignores the seed | `test_sample_seed_depends_on_seed_game_and_prompt_only` |
| alternatives at temperature 1 | `test_sampling_uses_the_agent_prompt_and_the_designed_settings` |
| ten samples | `test_sampling_uses_the_agent_prompt_and_the_designed_settings` |
| k_max ignored | `test_alternatives_are_cut_at_k_max` |
| greedy cache ignores the prompt | `test_no_verifier_executes_the_greedy_proposal` |
| score cache ignores the arm's prompt | `test_common_random_numbers_share_the_prefix_through_the_cache` |
| cache never written | `test_cache_makes_once_saves_and_reloads` |
| agent sees the rejected proposal | `test_the_agent_never_sees_the_verifier` |
| verifier sees only the first observation | `test_the_verifier_reads_the_world_model_history_of_the_live_state` |
| proposal executed after an override | `test_override_records_the_proposals_true_outcome` |
| counterfactual on the live env | `test_smoke_cache_check_and_gate` |
| counterfactual skipped | `test_override_records_the_proposals_true_outcome` |
| failure never recorded | `test_no_verifier_executes_the_greedy_proposal` |
| override never flagged | `test_override_records_the_proposals_true_outcome` |
| repeats never flagged | `test_repeats_and_score` |
| format error never flagged | `test_format_error_becomes_look_and_is_flagged` |
| override harm counts both failing | `test_episode_summary_classifies_overrides_and_runs` |
| rejection run never resets | `test_episode_summary_classifies_overrides_and_runs` |
| a loss scores like the cap | `test_repeats_and_score` |
| no-verifier arm never played | `test_episode_jobs_order` |
| arms in config order | `test_episode_jobs_order` |
| episodes saved at the end of each game | `test_partial_job_resumes_to_the_same_episodes` |
| resume replays saved episodes | `test_partial_job_resumes_to_the_same_episodes` |
| fresh run keeps the old cache | `test_a_fresh_run_starts_a_new_cache_and_episodes` |
| partial job never stops | `test_partial_job_resumes_to_the_same_episodes` |
| swapped arm without a swap object | `test_games_without_a_swap_object_skip_only_the_swapped_arm` |
| observation gate ignored | `test_observation_episodes_wait_for_the_step0_check` |
| observation gate reads another model | `test_observation_episodes_wait_for_the_step0_check` |
| observation gate always open | `test_observation_episodes_wait_for_the_step0_check` |
| cache check always passes | `test_cache_check_catches_a_cache_that_does_not_reproduce` |
| smoke gate ignores the cache check | `test_smoke_gate_reads_format_errors_and_the_cache_check` |
| smoke format threshold inclusive | `test_smoke_gate_reads_format_errors_and_the_cache_check` |
| seeds summed, not averaged | `test_per_game_means_over_seeds` |
| per game mixes rules | `test_per_game_means_over_seeds` |
| attribution ignores the half | `test_attribution_categories` |
| invalid actions read like success | `test_invalid_actions_verdict_has_the_reversed_sign` |
| harm without the permutation p | `test_verdict_needs_both_interval_and_p` |
| harm read from the lower bound | `test_verdict_needs_both_interval_and_p` |
| help without the interval | `test_verdict_needs_both_interval_and_p` |
| analysis accepts a run that does not match the design | `test_analysis_end_to_end_and_its_refusals` |
| analysis accepts repeated episodes | `test_analysis_end_to_end_and_its_refusals` |
| analysis ignores the primary run's no-verifier episodes | `test_the_primary_runs_no_verifier_episodes_serve_other_runs` |
| diverging lengths counted as never diverging | `test_divergence` |
| acceptance of working proposals reads failing ones | `test_step_rates_by_the_proposals_true_outcome` |
| drift compares every condition | `test_drift_against_the_archived_agent` |
| question-specific without the gap | `test_gap_states_and_classification_at_their_boundaries` |
| observation collapse not checked | `test_step0_other_classifications` |
| competence AUC strict | `test_step0_competence_rule` |
| step-0 swapped arm without a swap | `test_step0_run_scores_every_candidate_arm_and_readout` |
| step-0 resume trusts other prompts | `test_step0_run_scores_every_candidate_arm_and_readout` |
| naming lift sign flipped | `test_step0_shifts_naming_and_any_task` |
| Stage 1a agreement unrounded | `test_stage1a_agreement` |
| step-0 analysis does not wait | `test_step0_run_scores_every_candidate_arm_and_readout` |
| step-0 waits for nothing | `test_step0_waits_for_the_collection` |
| review1 R1 sample cache key drops the seed | `test_two_seeds_draw_their_own_alternatives` |
| review1 R2 counterfactual cache key drops the prefix | `test_counterfactual_is_classified_at_its_own_prefix` |
| review1 R3 decomposition on every game | `test_decomposition_uses_only_the_games_with_a_swap_object` |
| review1 R4 family yes/no gate not enforced | `test_family_yes_no_gate_stops_the_run` |
| review1 R5 observation gate ignores the status | `test_observation_gate_needs_a_reported_status` |
| review1 R6 no_alternative not an abstention | `test_no_alternative_counts_as_an_abstention_and_new_secondaries` |
| review1 R7 goal flag of the chosen action | `test_goal_flags_follow_the_proposal_and_the_chosen_action` |
| review1 R8 step-0 rejection for every no-verifier episode | `test_no_verifier_episodes_never_count_rejections` |
| review1 R9 any-task drop includes failing go to | `test_step0_go_to_readouts_count_only_working_go_to` |
| review1 R9b go-to acceptance includes failing go to | `test_step0_go_to_readouts_count_only_working_go_to` |
| review1 R10 cache check compares only success | `test_cache_check_flags_a_replay_that_differs_while_success_does_not` |
| obstruction at exactly the margin | `test_obstruction_check_margin` |
| obstruction ignored by attribution | `test_verdict_not_identifiable_when_the_swapped_verifier_obstructs` |
| obstruction not computable read as clear | `test_obstruction_check_margin` |
| obstruction reads failing proposals | `test_verdict_harm_attributable_when_the_swapped_verifier_recovers_the_cost` |
| attribution decided on the point estimate | `test_attribution_categories` |
| attribution with no cost on the subset | `test_attribution_categories` |
| attribution threshold the whole cost | `test_verdict_harm_attributable_when_the_swapped_verifier_recovers_the_cost` |
| net harm read with the wrong sign | `test_override_net_harm_is_a_key_secondary_where_more_is_worse` |
| matched states read the aware score twice | `test_matched_states_compare_prompts_on_the_same_steps` |
| matched states from every arm | `test_matched_states_compare_prompts_on_the_same_steps` |
| matched states reject reads accept | `test_matched_states_compare_prompts_on_the_same_steps` |
| shadow skips the alternatives | `test_shadow_scores_every_aware_step_and_never_acts` |
| shadow on every arm | `test_shadow_scores_every_aware_step_and_never_acts` |
| shadow swapped without a swap object | `test_games_without_a_swap_object_skip_only_the_swapped_arm` |
| seeds_by_arm ignored | `test_seeds_by_arm_and_the_swapped_arm_at_one_seed` |
| every arm at every seed | `test_seeds_by_arm_and_the_swapped_arm_at_one_seed` |
| order gate skipped | `test_a_run_waits_for_its_gates_and_logs_them` |
| order gate ignores the smoke gate | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| cost rule inclusive | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| cost rule ignored | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| cost ceiling optional | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| step-0 analysis not required | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| memory smoke not required | `test_gemma_needs_a_passed_memory_smoke` |
| memory smoke result not read | `test_gemma_needs_a_passed_memory_smoke` |
| order override not logged | `test_order_gate_refuses_until_smoke_step0_and_cost_pass` |
| projection ignores unplayed arms | `test_projection_from_the_smoke` |
| last-only adopted regardless | `test_last_only_is_adopted_only_if_no_rounded_score_changes` |
| last-only compares raw scores | `test_last_only_is_adopted_only_if_no_rounded_score_changes` |
| last logits ignore the flag | `test_scorer_asks_for_the_last_position_only_when_told` |
| last logits without the older keyword | `test_scorer_asks_for_the_last_position_only_when_told` |
| memory smoke crashes on a failure | `test_memory_smoke_records_a_failure` |
| yes/no readout sign flipped | `test_yes_no_readout_is_the_log_odds_rounded_to_a_64th` |
| first-ten invalid counts every step | `test_no_alternative_counts_as_an_abstention_and_new_secondaries` |
| net harm adds help | `test_no_alternative_counts_as_an_abstention_and_new_secondaries` |
| drift first action read at the end | `test_drift_against_the_archived_agent` |
| drift success compared with itself | `test_drift_against_the_archived_agent` |
| missing archive fatal | `test_analysis_refuses_a_primary_run_that_does_not_qualify_and_records_a_missing_archive` |
| observation primary without the step-0 condition | `test_primary_run_qualification` |
| primary run not checked | `test_analysis_refuses_a_primary_run_that_does_not_qualify_and_records_a_missing_archive` |
| primary run of any rule | `test_primary_run_qualification` |
| collapse bound inclusive | `test_gap_states_and_classification_at_their_boundaries` |
| no-gap bounds strict | `test_gap_states_and_classification_at_their_boundaries` |
| observation competence not checked in the classification | `test_gap_states_and_classification_at_their_boundaries` |
| yes/no competence not checked in the classification | `test_step0_other_classifications` |
| step-0 competence read under the aware arm | `test_step0_question_specific` |
