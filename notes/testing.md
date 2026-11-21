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


## Stage 1b: the agent's own ranked candidates (2026-11-20)

`tests/test_decisionsim_agent.py` covers `wf/decisionsim_agent.py`. On synthetic data: the candidate list (rank 1 is the recorded greedy action even when never sampled, its sampled variants merged into it, the rest by count with the seeded hash for ties and independent of the order of the replies, format errors counted and never turned into `look`, the cut at `k_max`, normalisation), per-state seeds, the verifier prompts (for each of the three models the aware, blind and swapped prompts differ only in the query's task line, the examples keep theirs, every prompt ends with the yes/no question, and the aware and swapped prompts equal the earlier `yn_*` prompts), the selection rules (none, oracle, verifiers accepting only above zero after rounding to 1/64 nat, the rank-1 fallback and the highest-L sensitivity, the cut lists, candidates the server could not run), the regime, encounter, strict-encounter and reachability definitions, harm and help with the identity, the decomposition, whole-game bootstrap and permutation, Holm (monotone, a missing p as 1, waiting for every priority model), the verdicts, the AUC (against a brute-force count) and the competence rule, the progress events and the swap object. On the simulator, with the mock agent and scorer: mock RQ3 "none" episodes run end to end through sampling, labelling, scoring for the three models and the analysis (every decision point once, rank 1 the episode's action); the sampling prompt is exactly the prompt the RQ3 agent saw, with the designed request settings; resume draws nothing new and refuses lists sampled on another prompt; a fresh run resamples; a replay mismatch stops the game there; labelling waits for every sampling shard, stops after its replay budget and resumes to identical labels, and its labels match a fresh replay; a rank-1 mismatch makes a state unusable; each arm sends its own prompt once per candidate; scores, lists and labels are saved one at a time; scoring resumes and refuses other prompts; the Mistral gate stops scoring; the analysis refuses scores of another label run; the model-free dry run, and its history rebuilt from an archived episode equals the replayed history.

The dry run on the archived data (`decisionsim_agent_dryrun`, run locally on 2026-11-20 with the archive's `text-wm-fix11/runs`) passed: 2,973 states in 134 games, no history without exactly one task line, the rank-1 failure rate 22.57% (671 states), and the known-answer check of the analysis code (one candidate per state, so the difference is exactly zero) held.

`scripts/mutants_decisionsim_agent.py` holds 57 Stage 1b mutants, added to `scripts/mutation_check.py`'s list (`MUTANTS_MATCHING="dsa "`). All 57 are caught. Two had survived the first tests (a fresh sampling run reading old lists back; every failure labelled determined) and were killed by the two tests named below. The table gives, for each, the first test in `tests/test_decisionsim_agent.py` that fails with the mutant in place (run with `-x`).

| Mutant | First test that fails |
|---|---|
| greedy action also listed again below rank 1 | `test_greedy_samples_merge_into_rank1_and_keep_the_recorded_string` |
| ties by order of appearance | `test_ranked_by_count_then_seeded_hash_not_order_of_appearance` |
| rarest action ranked first | `test_ranked_by_count_then_seeded_hash_not_order_of_appearance` |
| format error becomes look | `test_format_errors_count_and_add_no_candidate` |
| list one longer than k_max | `test_list_is_cut_at_k_max_and_the_cut_is_recorded` |
| actions not normalised | `test_normalize_collapses_space_case_and_final_stop` |
| tempered sampling | `test_list_is_cut_at_k_max_and_the_cut_is_recorded` |
| top-k truncation | `test_real_sampler_sends_the_designed_payload` |
| thinking on | `test_real_sampler_sends_the_designed_payload` |
| one seed for every state | `test_end_to_end_every_decision_point_once` |
| agent prompt without admissible actions | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| history turn shows the next observation | `test_sampling_prompt_is_the_rq3_agent_prompt_and_request_settings` |
| replay mismatch ignored | `test_replay_mismatch_stops_the_game_there` |
| resume trusts lists from another prompt | `test_sample_resume_draws_nothing_new_and_refuses_another_prompt` |
| fresh sample run appends to old lists | `test_a_fresh_sample_run_does_not_append_to_old_lists` |
| lists saved at the end of each game | `test_sample_resume_draws_nothing_new_and_refuses_another_prompt` |
| rank 1 not checked against the episode | `test_rank1_mismatch_makes_the_state_unusable` |
| every failure determined | `test_label_state_marks_determination_by_the_rq2_rules` |
| label replay budget ignored | `test_label_stops_partial_and_resumes_to_the_same_labels` |
| fresh label run appends to old labels | `test_label_stops_partial_and_resumes_to_the_same_labels` |
| label does not wait for sampling | `test_label_waits_for_every_sampling_shard` |
| swap may be named by a candidate | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| swap may be acted on in the history | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| swap may sit at a candidate's receptacle | `test_swap_object_excludes_named_interacted_and_co_located_types` |
| progress from any goal mention | `test_progress_events` |
| progress from a move anywhere | `test_progress_events` |
| blind prompt keeps the task line | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| blind prompt strips the examples too | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| swapped arm shows the real task | `test_each_arm_differs_only_in_the_query_task_line[llama3.1-8b]` |
| a score of zero accepted | `test_selection_rules` |
| fallback to the last candidate | `test_selection_rules` |
| oracle picks a failure | `test_selection_rules` |
| scores not rounded | `test_scores_are_rounded_to_a_64th_before_the_threshold` |
| coarser grid | `test_scores_are_rounded_to_a_64th_before_the_threshold` |
| server-error candidates kept | `test_server_error_candidates_are_skipped` |
| regime ignores shared spots | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| regime without determination | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| regime without the goal object | `test_regime_needs_failure_goal_object_determination_and_no_shared_spot` |
| strict encounter ignores the history | `test_encounter_strict_and_reachable` |
| reachable past a success | `test_encounter_strict_and_reachable` |
| bootstrap resamples states | `test_inference_resamples_and_flips_whole_games` |
| permutation flips states | `test_inference_resamples_and_flips_whole_games` |
| Holm not monotone | `test_holm_and_verdicts` |
| missing p counts as zero | `test_holm_and_verdicts` |
| Holm over the models that are in | `test_holm_waits_for_every_priority_model` |
| toward failure read from the upper bound | `test_holm_and_verdicts` |
| incompetent verifier interpreted | `test_end_to_end_every_decision_point_once` |
| analysis reads another label run's scores | `test_analysis_waits_and_refuses_scores_of_another_label_run` |
| swapped scores not required | `test_complete_states_need_every_arm_where_defined` |
| harm and help swapped | `test_contrast_counts_harm_help_and_the_identity` |
| decomposition part reversed | `test_decomposition_on_states_with_a_swap_and_its_identity` |
| swapped arm never scored | `test_end_to_end_every_decision_point_once` |
| score resume trusts saved scores | `test_score_resume_and_refusal_of_other_prompts` |
| scores saved at the end | `test_scores_are_saved_one_at_a_time` |
| family gate skipped | `test_a_failed_gate_scores_nothing` |
| shards overlap | `test_end_to_end_every_decision_point_once` |
| archived history without action marks | `test_dry_run_history_is_the_replayed_history` |
