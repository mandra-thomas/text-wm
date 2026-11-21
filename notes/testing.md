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
| fix1 Option C harm at the bound fails (review 2: net condition at the bound) | `test_option_c_categories` (net exactly half the headroom) |
| fix1 Option C harm threshold 10% (review 2: net share 0.75) | `test_option_c_categories` (default case, worth doing) |
| fix1 Option C ignores p | `test_option_c_categories (p exactly 0.05)` |
| fix1 Option C p at the bound passes | `test_option_c_categories (p exactly 0.05)` |
| fix1 Option C reads rule 1 | `test_option_c_categories (default case, worth doing)` |
| fix1 Option C without the swapped arm | `test_option_c_waits_for_the_task_blind_arm` (from Stage 1a) |
| fix1 a secondary model decides Option C | `test_option_c_reads_rule2_only_and_primary_models_only` |
| fix1 acceptance cells swapped | `test_rule2_difference_splits_by_acceptance_cell` |
| fix1 acceptance contributions per cell | `test_rule2_difference_splits_by_acceptance_cell` |
| fix1 attribution ignores the half | `test_option_c_categories (swapped part below half)` |
| fix1 attribution ignores the sign | retired after review 2 as equivalent (see below) |
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

### Stage 1a additions (2026-11-20, before any score)

New tests in `tests/test_decisionsim.py` (67 collected, from 53; 54 test functions, from 44): the task-blind history (the first observation's task sentence and the line break before it removed and nothing else, at step 0 and later; refused with no task sentence, two, one outside the first observation, or another game's task); the task-blind prompt for each of the five models (identical to the aligned prompt except for that one deletion, with the system text and the few-shot examples, task lines included, unchanged; the other arms built as before) and its refusal of a prompt that does not hold the query's history; the recorded prompts of a mock run (every candidate scored once under each of the four arms, the blind prompt equal to that candidate's aligned prompt minus its task sentence); the contrasts with the blind arm, the decomposition aligned − blind = (aligned − swapped) + (swapped − blind) on synthetic scores, the blind arm in the model report (baseline, shifts, calibration against blind, reference points); the Option C rule against the task-blind verifier with the neutral version as a secondary verdict (each category under both bases, a case where they disagree, the attribution check reading aligned minus the same base, waiting when the blind arm or its decomposition is missing, a real report where only the comparison differs); and the no-verifier and simulator-oracle reference points on synthetic scores (rule 2's proposal executed as is, random choice over each rule's own set, the oracle's choices, headroom, each verifier minus no verifier, the share of headroom realised, the two half-oracles for rule 2, odd-step states and rule-3 states left out, the swapped arm only where a swap exists). The configuration test now requires the four arms.

`scripts/mutation_check.py` holds 37 new Stage 1a mutants, and three earlier ones were adapted to the changed code (fix1 "attribution ignores the half", "Option C without the swapped arm", "swapped arm not scored by default"); 109 decision-simulation mutants in all. All 37 new mutants are caught (`MUTANTS_MATCHING="decisionsim stage1a"`, six shards, 2026-11-20); the three adapted ones are caught too (first failing test checked for each), and a rerun of all 109 decision-simulation mutants on the final code caught every one (`MUTANTS_MATCHING=decisionsim`, six shards, 2026-11-20). The table gives the first test that fails with each new mutant in place (run with `-x`).

| Mutant | First test that fails |
|---|---|
| stage1a Option C attribution read against neutral | `test_option_c_categories` (default case, worth doing) |
| stage1a Option C decides against neutral | `test_option_c_categories` (default case, worth doing) |
| stage1a Option C does not wait for the blind arm | `test_option_c_waits_for_the_task_blind_arm` |
| stage1a Option C harm and difference read against neutral | `test_option_c_categories` (default case, worth doing) |
| stage1a aligned-vs-blind contrast missing (adapted in review 2) | `test_contrasts_with_the_task_blind_verifier` |
| stage1a aligned-vs-blind contrast reversed (adapted in review 2) | `test_contrasts_with_the_task_blind_verifier` |
| stage1a blind arm is the neutral prompt | `test_task_blind_prompt_differs_from_the_aligned_one_only_by_the_task_sentence` (qwen3-8b) |
| stage1a blind arm keeps the task sentence | `test_task_blind_prompt_differs_from_the_aligned_one_only_by_the_task_sentence` (qwen3-8b) |
| stage1a blind arm not scored by default | `test_every_model_is_registered_and_configured` |
| stage1a blind arm refused by the scoring run (adapted in review 2) | `test_every_arm_reads_rounded_scores` |
| stage1a blind arm strips the examples' task lines | `test_task_blind_prompt_differs_from_the_aligned_one_only_by_the_task_sentence` (qwen3-8b) |
| stage1a blind history accepts a task sentence after an action | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| stage1a blind history accepts another game's task | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| stage1a blind history accepts two task sentences | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| stage1a blind history drops the room description | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| stage1a blind history leaves an empty line | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| stage1a blind prompt not checked for the query's history | `test_blind_prompt_refuses_a_prompt_that_does_not_hold_the_querys_history` |
| stage1a blind shifts only where a swap exists | `test_model_report_with_the_task_blind_arm` |
| stage1a calibration against blind uses neutral (adapted in review 2) | `test_model_report_with_the_task_blind_arm` |
| stage1a decomposition against blind uses neutral (adapted in review 2) | `test_model_report_with_both_task_blind_prompts` |
| stage1a decomposition labels every base neutral | `test_decomposition_against_the_task_blind_verifier_adds_up` |
| stage1a every verdict marked deciding | `test_option_c_categories` (default case, worth doing) |
| stage1a half-oracle accept/reject reads the model | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a half-oracle fallback reads the model | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a headroom share inverted | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a neutral-vs-blind contrast missing (adapted in review 2) | `test_contrasts_with_the_task_blind_verifier` |
| stage1a no verifier for rule 2 is a random choice | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a oracle knows nothing | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a pass count without the blind arm (adapted in review 2) | `test_end_to_end_score_resume_and_analysis` |
| stage1a random choice over all candidates | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a reference points at odd steps | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a reference points ignore where rule 3 applies | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |
| stage1a reference points left out of the report | `test_model_report_with_the_task_blind_arm` |
| stage1a secondary verdict decides | `test_end_to_end_score_resume_and_analysis` |
| stage1a swapped reference points where no swap exists | `test_reference_points_for_the_swapped_arm_read_only_states_with_a_swap` |
| stage1a swapped-vs-blind contrast missing (adapted in review 2) | `test_contrasts_with_the_task_blind_verifier` |
| stage1a verifier minus no verifier reversed | `test_no_verifier_and_oracle_reference_points_on_synthetic_scores` |

### Review 2 (2026-11-21, before any score)

New tests in `tests/test_decisionsim.py` (96 collected, from 67; 68 test functions, from 54):
- **Option C rule.** The rule rewritten around the net condition, the fallback-menu condition and the sign of aligned minus the comparison. There are twenty category cases, each checked under all three comparisons: integer bounds at exactly half the headroom, 21 of 42 against 21.5 of 43, the fallback share at exactly half, attribution checked before the fallback, and an interval touching zero from below. Further tests check the category order, a report whose numbers against the task-blind and neutral comparisons differ (the harm, p and counts are read against the same base), the pays-for-itself flag, and a real report in which the harm runs through a false rejection, through the fallback menu, or neither.
- **Counts.** The net, harm and help counts, the per-cell nets and rule 2's headroom in states (odd-step states left out).
- **`yn_blind_all`.** Its prompt for each of the five templates differs from `yn_blind` only by the examples' task lines; it refuses other example sets; recorded prompts of a mock run cover all five arms; the scoring run refuses it for a secondary model. The model report with both task-blind prompts and the prompt-format flag are tested too.
- **The step split.**
- **The reviewer's six gaps:** every arm reads rounded scores; rounding rounds; a state missing one task-blind score is dropped; a non-determined failing proposal counts for the no-verifier rate; the swapped headroom share is read on its own states; the rule 3 count is on even steps.

`scripts/mutation_check.py` now holds 154 decision-simulation mutants:
- 22 of the reviewer's 24 mutants, adapted to the amended code (R1 to R23 below, without R19).
- 24 for review 2's new code (N1 to N24).
- 10 earlier mutants adapted because the code they target changed: fix1 "harm threshold 10%" and "harm at the bound fails" now mutate the net condition, plus eight Stage 1a mutants.
- 3 left out or retired as equivalent:
  - the reviewer's "decomposition against blind takes aligned minus swapped from neutral": aligned minus swapped is identical in every decomposition;
  - the reviewer's R19, "half-oracle fallback may return the rejected proposal": the oracle picks a true success either way;
  - fix1 "attribution ignores the sign": a positive aligned-minus-base difference and the half condition already force it.

All 154 are caught (`MUTANTS_MATCHING=decisionsim`, six shards, 2026-11-21). The table gives the first test in `tests/test_decisionsim.py` that fails with each review 2 mutant in place (run with `-x`).

| Mutant | First test that fails |
|---|---|
| review2 N1 net condition ignored | `test_option_c_categories` (net one below half) |
| review2 N2 headroom counts odd-step states | `test_rule2_headroom_and_net_counts_in_the_contrast` |
| review2 N3 net counts harm only | `test_contrast_counts_harm_help_and_the_identity` |
| review2 N4 cell net counts harm only | `test_rule2_difference_splits_by_acceptance_cell` |
| review2 N5 fallback-menu condition ignored | `test_option_c_categories` (fallback above half) |
| review2 N6 fallback at exactly half counts as through it | `test_option_c_categories` (fallback exactly half) |
| review2 N7 fallback read from another cell | `test_option_c_categories` (default case, worth doing) |
| review2 N8 fallback checked before attribution | `test_option_c_categories` (not attributable with a large fallback share) |
| review2 N9 fallback category after not attributable | `test_option_c_category_order` |
| review2 N10 attribution ignores the sign of aligned minus base | `test_option_c_categories` (aligned minus base negative) |
| review2 N11 pays-for-itself strict | `test_option_c_notes_a_comparison_verifier_that_does_not_pay_for_itself` |
| review2 N12 yn_blind_all allowed for a secondary model | `test_arms_must_include_aligned_and_neutral` |
| review2 N13 example task lines not checked | `test_strip_example_tasks_refuses_a_prompt_without_the_examples` |
| review2 N14 yn_blind_all keeps the query's task | `test_blind_all_prompt_also_strips_only_the_examples_task_lines` (qwen3-8b) |
| review2 N15 yn_blind_all not scored for the primary models | `test_every_model_is_registered_and_configured` |
| review2 N16 format verdict against yn_blind | `test_prompt_format_sensitivity_is_flagged_when_the_verdicts_differ` |
| review2 N17 format sensitivity inverted | `test_prompt_format_sensitivity_is_flagged_when_the_verdicts_differ` |
| review2 N18 calibration keys collide | `test_model_report_with_both_task_blind_prompts` |
| review2 N19 no decomposition against yn_blind_all | `test_model_report_adds_swapped_contrasts_only_where_scored` |
| review2 N20 the two task-blind prompts not compared | `test_model_report_with_both_task_blind_prompts` |
| review2 N21 step 0 split takes step 2 too | `test_rule2_by_step_splits_step_zero_from_later_steps` |
| review2 N22 primary pass count without yn_blind_all | `test_end_to_end_score_resume_and_analysis` |
| review2 N23 net share a quarter | `test_option_c_categories` (net one below half) |
| review2 N24 fallback share three quarters | `test_option_c_categories` (fallback above half) |
| review2 R1 blind task sentence may be the first line | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| review2 R2 complete_states ignores the blind arm | `test_complete_states_drop_a_state_missing_one_task_blind_score` (yn_blind) |
| review2 R3 aligned-vs-blind contrast only on swap states | `test_model_report_with_the_task_blind_arm` |
| review2 R4 blind scores not rounded | `test_every_arm_reads_rounded_scores` |
| review2 R5 secondary verdict computed against blind | `test_prompt_format_sensitivity_is_flagged_when_the_verdicts_differ` |
| review2 R6 harm rate over changed states only | `test_contrast_counts_harm_help_and_the_identity` |
| review2 R7 paired difference reversed | `test_contrast_counts_harm_help_and_the_identity` |
| review2 R8 toward success includes an interval touching zero | `test_option_c_categories` (interval touching zero from below) |
| review2 R9 calibration vs blind arguments reversed | `test_model_report_with_the_task_blind_arm` |
| review2 R10 decomposition aligned-minus-swapped sign | `test_decomposition_adds_up_on_the_states_with_a_swap` |
| review2 R11 rounding truncates | `test_quantize_rounds_rather_than_truncates` |
| review2 R12 Option C harm read against neutral only | `test_option_c_categories` (default case, worth doing) |
| review2 R13 Option C p from rule 1 | `test_option_c_categories` (default case, worth doing) |
| review2 R14 gate skips blind prompts | `test_the_gate_on_a_resume_checks_every_prompt_not_only_the_unscored` |
| review2 R15 blind prompt drops the examples | `test_task_blind_prompt_differs_from_the_aligned_one_only_by_the_task_sentence` (qwen3-8b) |
| review2 R16 lure counts pairs at odd steps | `test_odd_step_states_are_left_out_of_every_state_level_rate` |
| review2 R17 swapped headroom read on all even states | `test_swapped_reference_headroom_is_read_on_its_own_states` |
| review2 R18 attribution read on the contrast's states | `test_option_c_categories` (aligned minus base negative) |
| review2 R20 no-verifier rule 2 counts only determined failures | `test_no_verifier_counts_a_failing_proposal_the_history_does_not_determine` |
| review2 R21 tie split for b reads La | `test_ties_are_counted_and_split_evenly_in_the_sensitivity_readout` |
| review2 R22 blind history accepts two task sentences (count >= 1) | `test_blind_history_removes_the_first_observations_task_sentence_and_nothing_else` |
| review2 R23 rule 3 states counted over all states | `test_rule3_state_count_is_on_even_steps` |

## Model families, wave A (2026-11-20)

`tests/test_families.py` covers the family runs: golden renders of one real item from each pinned chat template, the tokenization gate on stand-in tokenizers built to fail in one item or one prompt (a late success's candidate, one item's swapped yes/no prompt, a template that writes "Yes" as another token than the scorer reads), the gate's scope in the shard runner (every item, on a resume too, with the scorer's template settings), the analysis (flip definitions by name, the yes/no arms in the standardised effect, records filtered by model and by population), the pre-data primary metrics (the item share toward Yes and the goal-to-evidence ratio R, including their invariance to rescaled logits), the competence gate at each of its bounds, the TOST equivalence of R on paired items, the Holm-corrected primary family, and the Mistral placement run. Fix round 2 (third amendment) adds: the R comparison's fallback to the paired difference when the ratio is not estimable (review 2's synthetic models with R near zero, slightly positive and negative), the 2% cap on dropped draws, the item-share comparison (difference by game sign flips, TOST within ±0.05 as the larger one-sided p), the 90% and 95% interval rules for both comparisons on hand-set bootstrap draws, pairing across noisy games, the primary family padded with p = 1 for missing or NaN tests, the planned comparisons including Gemma 2 against Mistral, the two-sided noise floor, ties at log-odds zero, NaN scores, and game-clustered p-values on games that move together. With transformers, torch and the tokenizers cached (`WF_TEST_HF_CACHE`), the real templates run too. `scripts/mutation_check.py` holds 75 families mutants (`MUTANTS_MATCHING=families`), including the ten a review found surviving, seventeen for the primary metrics, gate, equivalence and family, review 2's M1 to M14 (two each for M6 and M7, R and share) and nine for fix round 2. All 75 are caught (2026-11-20). Two survived the first run and were killed by added tests: the TOST p taken from the smaller side (the test's draws had been symmetric) and R kept when the evidence effect is negative (the test had used an evidence effect of exactly zero). The refactored gate was also run against the earlier one on every real item with the pinned tokenizers, with identical results (`notes/families_design.md`).
