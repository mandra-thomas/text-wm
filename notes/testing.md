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

## Interpretability pilot, review 2 (2026-11-20, branch `interp`)

Review 2 put 35 bugs into the interpretability code (N1 to N35); 21 survived the tests, mostly because the end-to-end test of the minimal pilot checked the structure of its outputs and not their values. Fix round 2 added value checks on synthetic data with known answers, and 42 mutants of its own code (J1 to J42). All 109 interpretability mutants in `scripts/mutation_check.py` (review 1's R1 to R20 and I1 to I12, and these) are run with a CPU torch environment (`MUTANTS_MATCHING=interp`). The test that kills each of the review's survivors and each new mutant (files `tests/test_interp.py`, `tests/test_interp_pilot.py`):

| Mutant | Killed by |
|---|---|
| N1 layer chosen on the item's own fold; N2 item read at the other fold's layer | `test_ablation_summary_chooses_each_folds_layer_on_the_other_fold` (folds with different best layers; exact shares) |
| N4 success direction from the other fold | `test_fold_directions_use_each_folds_own_successes` |
| N5 placebo target is the aligned prompt | `test_placebo_patching_targets_the_swapped_prompt` |
| N7 effect sign flipped | `test_effects_sign_and_strata` |
| N8 task_from only at its own layer; N25 random directions ablated to 0 | `test_ablation_edits_layers_and_targets` |
| N10 maximum-over-layers interval with the layer fixed | `test_maximum_over_layers_rechooses_the_layer_in_every_resample` |
| N11 signed placebo floor; N12 above-floor read on the upper bound; N13 early-span sign | `test_placebo_floor_bonferroni_and_early_span` |
| N14 one-step histories counted as step 0 | `test_effects_sign_and_strata` |
| N16 gate's median check dropped | `test_gate_median_check_hashes_and_first_token` |
| N19 history-share verdict from the point estimate | `test_verdict_rows_read_interval_bounds_onset_and_h4` |
| N21 answer projection without the success direction | `test_answer_projection_removes_readout_and_success` |
| N24 lens direction without the norm weight | `test_lens_direction_carries_the_norm_weight` (random norm weights; the tiny model's are all 1) |
| N26 gap on aligned prompts; N27 steering read on aligned prompts | `test_ablation_summary_chooses_each_folds_layer_on_the_other_fold` |
| N29 identical-prefix check skipped | `test_patching_runs_the_identical_prefix_check` |
| N32 norm epsilon dropped from the readout | `test_readout_of_a_zero_residual_is_finite` |
| N33 between-fold cosine of task with success | `test_minimal_pilot_end_to_end` (value of the cosine) |
| J1 profile includes the last layer; J6, J7 H4 layers | `test_profile_caveat_and_h4_layers` |
| J2 onset on the point estimate; J3 onset need not stay; J4 inverted, J5 unrounded comparison with arrival | `test_onset_layer_needs_the_lower_bound_to_stay_above` |
| J8 success direction pooled over verbs; J9 verbs weighted equally | `test_success_direction_is_estimated_within_verbs` |
| J10 H4 cosine projected on one side; J11, J12 perp steering from the raw success direction | `test_h4_cosine_is_read_after_projecting_out_the_readout` |
| J13 quotas ignored; J14 weights inverted; J40 mention stratum ignored | `test_selection_quotas_and_weights` |
| J15, J16 weights ignored in curves and removed shares | `test_weights_enter_pooled_estimates` |
| J17, J18 the analysis run drops the weights; J30 ablation step-0 quota ignored; J36 cross-fold cosine within one fold; J19 both folds edited; J25 first-token prompts keep activations (also `test_first_token_prompts_can_be_stored_without_activations`); J42 stage 0 gate without hashes | `test_minimal_pilot_end_to_end` |
| J20 GPU check accepts any GPU | `test_gpu_check_refuses_a_different_gpu` |
| J21, J41 patching or ablation run without the GPU check | `test_gpu_runs_refuse_before_any_work` |
| J22 item-hash check dropped; J23 first-token gate on the yes/no arms | `test_gate_median_check_hashes_and_first_token` |
| J24 stage 0 passes on the yes/no gate alone | `test_stage0_gate_fails_on_a_shifted_log` |
| J26 realized edit read before the edit; J27 realized steering without the clean row; J39 logged on every prompt | `test_realized_edits_are_read_after_the_edit` |
| J28 Bonferroni at alpha 0.05 | `test_placebo_floor_bonferroni_and_early_span` |
| J29 onset reads the caveat layer | `test_ablation_summary_reports_caveat_and_last_layers_apart` |
| J31 sample size not squared | `test_sample_size_for_a_half_width` |
| J32 mention flag reads the whole history | `test_mention_flag_agrees_with_the_object_mention_group` |
| J33 action object found from the receptacle; J34 grid without `hist_swap` | `test_action_line_splits_into_its_object_and_the_rest` |
| J35 step-0 failures in the success direction | `test_fold_directions_use_each_folds_own_successes` |
| J37 gap defined without successes | `test_ablation_summary_chooses_each_folds_layer_on_the_other_fold` |
| J38 steering share over the aligned readout | `test_ablation_summary_chooses_each_folds_layer_on_the_other_fold` (swapped readout not zero) |

The review's other 14 mutants (N3, N6, N9, N15, N17, N18, N20, N22, N23, N28, N30, N31, N34, N35) were already killed and stay in the list. Review 2's simulation of the last-layer tautology is a test (`test_last_layer_ablation_is_tautological_and_the_depth_profile_is_not`).

## Interpretability pilot, review 3 (2026-11-21, branch `interp`)

Review 3 put 40 bugs into the code (K1 to K40); 19 survived, most of them in the weighted estimators, whose tests checked point estimates with equal weights. K19 (weights computed over n instead of the number chosen) is equivalent and left out: with more eligible items than n the selection always takes exactly n. Fix round 3 added value tests and 28 mutants of its own code (L1 to L28). All 176 interpretability mutants in `scripts/mutation_check.py` are run with a CPU torch environment (`MUTANTS_MATCHING=interp`). Killing tests (files `tests/test_interp_pilot.py`, `tests/test_interp.py`):

| Mutant | Killed by |
|---|---|
| K2 H4 layers include the arrival layer; K23 discrimination at 0.8; K27 cross-fold cosine of fold 0 twice; K4, K5 arrival index or arrival kind in the table | `test_table_rows_read_interval_bounds_onset_and_h4` |
| K6, K7 weights on the point estimate only; K8 steering share, K9 mean shift, K10 gap without weights; K11 mean shift over the count | `test_weighted_bootstrap_rows_carry_the_weights` (every bootstrap draw recomputed by hand) |
| K12 verbs weighted by successes only; K33 mention rows without a mention prompt; K35 first-token gate reads the neutral arm; K20 a minimum above its stratum; K15 GPU overlap accepted; K18 two GPU types on record; K38 caveat layers keep profile layers | `test_review_3_survivors_on_small_cases` |
| K36 successes without weights; K37 random directions at the first layer | `test_minimal_pilot_end_to_end` |
| K26 perp over task inverted (now in `analysis.perp_over_task`) | `test_steering_relative_to_the_task_direction_and_the_perp_over_task_rule` |
| K32 sample size at a fixed n | `test_table_rows_read_interval_bounds_onset_and_h4` |
| L1 arrival not rounded; L2 arrival beyond the profile read anyway; L3 no single direction without the upper bound | `test_onset_layer_needs_the_lower_bound_to_stay_above` |
| L4 arrival from the final group | `test_patch_arrival_on_the_answer_position_under_the_onset_rule` |
| L5 direct path without the swapped prompt; L8 at the first layer's scale | `test_direct_path_report_is_the_readout_lens_at_the_final_scale` |
| L6 direct-path rule on the lower bound; L7 rule strict at 0.3 | `test_direct_path_share_tracks_the_removal_and_the_rule_excludes_it` |
| L9 onset ignores the direct path | `test_ablation_summary_reads_the_onset_on_layers_free_of_the_direct_path` |
| L10 steering relative inverted; L11 perp over task always assessable | `test_steering_relative_to_the_task_direction_and_the_perp_over_task_rule` |
| L12 per-stratum steering from every row; L13 edits ignore the stratum; L14 steer_task along the success direction | `test_steering_amounts_per_stratum_and_the_task_reference` |
| L15 ablation run steers with pooled amounts; L23, L24 minimums not scaled; L25 disattenuated by the raw between-fold cosine; L26 analysis without the direct path; L27 no answer direction | `test_minimal_pilot_end_to_end` |
| L16 cosine bootstrap projects nothing; L17 answer direction not centred; L18 disattenuation without the square root | `test_answer_direction_projections_and_the_cosine_bootstrap` |
| L19 H4 layers cut at the final-group arrival; L20 perp-over-task median over unassessable layers; L28 every layer excluded falls back to the profile | `test_table_rows_read_interval_bounds_onset_and_h4` (L28 by a case with every layer excluded) |
| L21 sample size without the margin | `test_sample_size_for_a_half_width` |
| L22 scaled minimum truncated | `test_minimums_scale_with_a_smaller_draw` |

The other 21 of review 3's mutants were already killed by earlier tests and stay in the list. From 2026-11-21 the mutation check counts a mutant as killed only when pytest exits with code 1 (a named test failed); a collection error, an internal error or a run stopped from outside is reported as inconclusive, never as a kill. All 176 interpretability mutants were rerun under that rule. Review 3's direct-path simulation is a test (`test_direct_path_share_tracks_the_removal_and_the_rule_excludes_it`).

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
