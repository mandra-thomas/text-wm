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

## TextCraft round 2 (2026-11-21)

`tests/test_textcraft_r2.py` covers round 2 (`notes/textcraft_r2_design.md`) without a model or a real tokenizer:

- *the clean prompt*: its text pinned by sha256, identical to the diagnostics' text, with no agent sentence;
- *the arms*: parsing, each arm's query against an independently built expected query, the padding before the first step, Gemma's folded system text, the item hash over every field and the prompt context;
- *the records*: counts, main records equal to the frozen items, the two-item line (sorted, shared within a couple, absent for the pair whose goal name contains "and"), the parent rules (the partner-disjoint rule, the strict preference, the rule's choice for all 118 goals), the seeded insertion of the added lines, a byte-for-byte rebuild of two pairs' parent records, replay of parent and padded records in the environment under goal, partner, parent and no goal, the text-only checks (with planted bad parents, a wrong padding reply, a wrong padded twin and class);
- *the tokenization gate* on a word-level stand-in tokenizer that passes, then fails in one way each (a reply prefix the candidate lacks, "Yes" as two tokens, candidates sharing their first token, a system message for a template with no system role), the golden comparison, the gate's refusal without the golden or with the wrong attention;
- *the scoring run* in mock mode: refusal of unverified or changed records, every record and arm scored once, resume, refusal of saved scores of a changed record, the gate stopping before any score, yes/no arms in one forward pass, the seeded smoke subset;
- *the analysis* on planted scores (L built from a lift, a naming term, a completion term, a timing change and a list effect): each contrast recovers its planted term exactly, the naming reading when naming carries the penalty, the Holm family fixed at 50 with missing tests padded at p = 1, withheld verdicts with a missing score, stale, unknown, duplicated and foreign scores refused, the competence gate (wave A's rule, read on the yes/no question, with role and class AUCs), every contrast "not interpretable" for a model that fails it, the verdict and reading tables row by row, timing "not applicable" without a role gap, the cross-model patterns and falsifiers, the equivalence margin, the bootstrap, TOST and sign-flip helpers on whole pairs, and a mock end-to-end run of five models.

`scripts/mutation_check.py` holds 54 round-2 mutants (`MUTANTS_MATCHING='textcraft r2'`), and the script now logs the first failing test of every killed mutant. In the first campaign three mutants were not killed by the round-2 tests. Two were killed by tests added for them: the gate's candidate check switched off (the stand-in tokenizer's reply prefix also broke the yes/no check, which hid it; a test now runs the gate on two-candidate arms only), and the naming contrast read against the neutral line instead of the swapped one (the planted scores gave both lines the same value; the neutral line now sits apart by role, so the swap changes the estimate). A third, H1 ignoring names listed in "Inventory:" replies, is equivalent and was removed: a name can only be listed in an inventory after a get or craft step has already mentioned it. The final run of all 54 against the final tests (2026-11-21, four shards) killed every one:

| Mutant | First test that failed |
|---|---|
| two-item line kept for names with and | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| hash misses the system text | `test_item_hash_covers_fields_and_context` |
| eight look-ups become four | `test_parse_arm` |
| padding on 9-12 steps too | `test_check_records_reports_and_passes` |
| padded twin not checked | `test_padded_twin_and_class_checks` |
| gate failure ignored | `test_scores_every_record_and_arm_then_resumes` |
| shards by record not pair | `test_scores_every_record_and_arm_then_resumes` |
| gate unfolded for Gemma | `test_gate_folds_for_gemma` |
| completion lines reversed | `test_planted_effects_are_recovered_exactly` |
| family without Holm | `test_family_is_holm_adjusted` |
| competence AUC 0.6 | `test_competence_gate_uses_wave_a_rule` |
| naming reading only without completion | `test_reading_table[confirmed below zero-confirmed below zero-confirmed below zero-naming penalty]` |
| bootstrap resamples items | `test_statistics_helpers` |
| padding after the history | `test_every_arm_changes_only_what_its_name_says` |
| hash misses the extended list | `test_item_hash_covers_fields_and_context` |
| parent may share craftables with the partner | `test_a_parent_sharing_a_craftable_with_the_partner_does_not_qualify` |
| frozen items reused as parents | `test_rebuilding_two_pairs_gives_the_committed_records` |
| padded class not checked | `test_padded_twin_and_class_checks` |
| resume ignores record hashes | `test_resume_refuses_scores_of_changed_records` |
| gate candidate check off | `test_gate_catches_a_candidate_written_differently_by_the_template` |
| lift on the goal role | `test_planted_effects_are_recovered_exactly` |
| naming against neutral | `test_planted_effects_are_recovered_exactly` |
| family shrinks with missing p | `test_family_is_fixed_at_fifty_and_pads_missing` |
| competence on the two-candidate score | `test_competence_gate_uses_wave_a_rule` |
| stale scores accepted | `test_stale_unknown_duplicate_or_foreign_scores_stop` |
| sign flips single items | `test_sign_flip_flips_whole_pairs` |
| used-up AUC not role-matched | `test_used_up_auc_is_role_matched` |
| ext arm shows the item's own list | `test_every_arm_changes_only_what_its_name_says` |
| Gemma not folded | `test_gemma_folds_the_system_text_and_the_others_do_not` |
| strict parent not preferred | `test_every_parent_is_the_rules_choice` |
| parent draw seed drops the bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| text-only partner check off | `test_text_parent_check_catches_bad_parents` |
| golden problems ignored | `test_run_gate_requires_the_golden_and_the_attention` |
| gate single-token check off | `test_gate_fails_on_each_tokenization_problem[tok1-not one token]` |
| penalty against the swapped line | `test_planted_effects_are_recovered_exactly` |
| timing reversed | `test_planted_effects_are_recovered_exactly` |
| margin half an SD | `test_margin_is_three_tenths_of_the_item_sd` |
| confirmed without the interval | `test_verdict_rules` |
| verdicts with missing scores | `test_missing_scores_withhold_verdicts` |
| heuristic H9 always true | `test_heuristics_agree_with_the_competence_script_on_every_item` |
| two-item line not sorted | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| yes/no query keeps Next observation | `test_every_arm_changes_only_what_its_name_says` |
| yes/no candidates reversed | `test_candidates` |
| added lines appended | `test_structure_check_catches_changes` |
| three failures per bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| yes/no scored with two passes | `test_yes_no_arms_use_one_forward_pass` |
| attention not checked | `test_run_gate_requires_the_golden_and_the_attention` |
| gate first-token check off | `test_gate_fails_on_each_tokenization_problem[tok2-share their first token]` |
| gap without the ingredient term | `test_planted_effects_are_recovered_exactly` |
| timing always applicable | `test_timing_not_applicable_without_a_role_gap` |
| TOST from the smaller side | `test_statistics_helpers` |
| failed gate still read | `test_failed_gate_makes_every_contrast_not_interpretable` |
| used-up goal items kept | `test_populations_on_the_full_records` |
| heuristic H2 counts crafted units twice | `test_heuristics_agree_with_the_competence_script_on_every_item` |

### TextCraft round 2, amendments before freeze (2026-11-21)

The amendments (design section 14) added tests for:

- the statistical verdicts, now never withheld;
- the interpretation table (a distortion in a competent predictor against a measurable goal effect; "inconclusive (equivalence underpowered)", never "no effect");
- every row of the amended mechanism readings, the partial-naming and one-look-up qualifiers, and the corrected F3 (true, false and undecided);
- the margin cap and that every contrast uses `margin_of`'s value;
- equivalence power and the 20,000-draw floor;
- the per-pair bootstrap against the item-level one;
- the contrast-specific competence rules on the full records: each population, the parent population alone failing, used-up ingredients kept out of the class-matched populations, the other three criteria on the full population, and the 0.70 boundary (0.70 passes, 0.69 fails);
- class-matched controls (shifting used-up ingredients leaves naming, completion and timing unchanged);
- timing's applicability read from the interval, and the one-look-up change beside it;
- the parent subsets, overlap kinds (with the "stone stairs" case) and the exact extended-list split of P;
- review 1's surviving mutants: every hashed field and context key; the system-text prompt check; excluded goals as parents; the strict-word text check; the list-is-pair-list check; the padded verdict and no-goal replay; NaN scores as missing; the mass criterion on failures and successes; resume across models; sorted shard slices; the smoothed sign-flip p;
- provenance (one commit, pinned revisions);
- the role gap equalling naming plus completion.

**Mutants after the amendments (2026-11-21).** `scripts/mutation_check.py` now holds 102 round-2 mutants: the 54 of the first campaign (two rewritten for the amended code, one dropped because its target, the "not interpretable" override, no longer exists), review 1's 18 non-equivalent survivors adapted to the amended code (R1-M…), 26 for the amended rules (A …), two for the anchored question (Q …) and six for the owner's final points (U …). M52 (a padded-prompt check that only repeated the expected-query check) was removed with its check instead of being tested. All 102 are killed:

- 55 in the official run (four shards), stopped part-way when three of review 1's hash mutants and the padded-goal population mutant survived the covering tests because the hash test looped over the mutated field list and the population test did not count the padded goal; both tests were strengthened;
- the other 46 in a screen against the round-2 tests (`MUTATION_FAST_ONLY=1`, which only skips the whole-suite rerun of survivors, so a kill is the same as in the official run), which left one survivor, the role-gap readout taking rows without a two-item line;
- that one in the official mode, after its test was strengthened.

The script now logs the first failing test of every kill:

| Mutant | First test that failed |
|---|---|
| two-item line not sorted | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| two-item line kept for names with and | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| padding after the history | `test_every_arm_changes_only_what_its_name_says` |
| ext arm shows the item's own list | `test_every_arm_changes_only_what_its_name_says` |
| yes/no query keeps Next observation | `test_every_arm_changes_only_what_its_name_says` |
| hash misses the system text | `test_item_hash_covers_fields_and_context` |
| hash misses the extended list | `test_item_hash_covers_fields_and_context` |
| Gemma not folded | `test_gemma_folds_the_system_text_and_the_others_do_not` |
| yes/no candidates reversed | `test_candidates` |
| eight look-ups become four | `test_parse_arm` |
| parent may share craftables with the partner | `test_a_parent_sharing_a_craftable_with_the_partner_does_not_qualify` |
| strict parent not preferred | `test_every_parent_is_the_rules_choice` |
| added lines appended | `test_structure_check_catches_changes` |
| padding on 9-12 steps too | `test_check_records_reports_and_passes` |
| frozen items reused as parents | `test_rebuilding_two_pairs_gives_the_committed_records` |
| parent draw seed drops the bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| three failures per bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| padded twin not checked | `test_padded_twin_and_class_checks` |
| padded class not checked | `test_padded_twin_and_class_checks` |
| text-only partner check off | `test_text_parent_check_catches_bad_parents` |
| yes/no scored with two passes | `test_yes_no_arms_use_one_forward_pass` |
| gate failure ignored | `test_scores_every_record_and_arm_then_resumes` |
| resume ignores record hashes | `test_resume_refuses_scores_of_changed_records` |
| golden problems ignored | `test_run_gate_requires_the_golden_and_the_attention` |
| attention not checked | `test_run_gate_requires_the_golden_and_the_attention` |
| shards by record not pair | `test_scores_every_record_and_arm_then_resumes` |
| gate candidate check off | `test_gate_catches_a_candidate_written_differently_by_the_template` |
| gate single-token check off | `test_gate_fails_on_each_tokenization_problem[tok1-not one token]` |
| gate first-token check off | `test_gate_fails_on_each_tokenization_problem[tok2-share their first token]` |
| gate unfolded for Gemma | `test_gate_folds_for_gemma` |
| lift on the goal role | `test_planted_effects_are_recovered_exactly` |
| penalty against the swapped line | `test_planted_effects_are_recovered_exactly` |
| gap without the ingredient term | `test_planted_effects_are_recovered_exactly` |
| completion lines reversed | `test_planted_effects_are_recovered_exactly` |
| naming against neutral | `test_planted_effects_are_recovered_exactly` |
| timing reversed | `test_planted_effects_are_recovered_exactly` |
| timing always applicable | `test_timing_not_applicable_without_a_role_gap` |
| family without Holm | `test_family_is_holm_adjusted` |
| family shrinks with missing p | `test_family_is_fixed_at_fifty_and_pads_missing` |
| margin half an SD | `test_margin_is_capped_at_four_tenths` |
| TOST from the smaller side | `test_statistics_helpers` |
| competence AUC 0.6 | `test_competence_gate_uses_wave_a_rule` |
| competence on the two-candidate score | `test_competence_gate_uses_wave_a_rule` |
| confirmed without the interval | `test_verdicts_are_statistical_and_never_withheld` |
| naming reading only without completion | `test_reading_table[confirmed below zero-confirmed below zero-confirmed below zero-naming / partial-completion penalty]` |
| stale scores accepted | `test_stale_unknown_duplicate_or_foreign_scores_stop` |
| verdicts with missing scores | `test_missing_scores_withhold_verdicts` |
| used-up goal items kept | `test_populations_on_the_full_records` |
| bootstrap resamples items | `test_fast_draws_equal_the_general_path` |
| sign flips single items | `test_sign_flip_flips_whole_pairs` |
| heuristic H9 always true | `test_heuristics_agree_with_the_competence_script_on_every_item` |
| heuristic H2 counts crafted units twice | `test_heuristics_agree_with_the_competence_script_on_every_item` |
| used-up AUC not role-matched | `test_used_up_auc_is_role_matched` |
| R1-M1 hash misses action | `test_hash_covers_every_field_and_context_key` |
| R1-M2 hash misses swapped line | `test_hash_covers_every_field_and_context_key` |
| R1-M3 hash misses recipe_lines | `test_hash_covers_every_field_and_context_key` |
| R1-M4 hash context misses NEXT | `test_hash_covers_every_field_and_context_key` |
| R1-M6 prompt check of system and examples off | `test_prompt_check_catches_a_changed_system_text` |
| R1-M9 excluded goals allowed as parents | `test_excluded_goals_never_qualify_as_parents` |
| R1-M10 text strict-word check off | `test_text_checks_catch_a_named_strict_parent_and_a_changed_list` |
| R1-M11 padded verdict check off | `test_padded_reply_check` |
| R1-M13 list-is-pair-list check off | `test_text_checks_catch_a_named_strict_parent_and_a_changed_list` |
| R1-M14 NaN L not counted missing | `test_nan_scores_count_as_missing` |
| R1-M15 timing applicable on the estimate | `test_timing_applicability_reads_the_interval` |
| R1-M16 padded goal keeps used up | `test_populations_on_the_full_records` |
| R1-M21 mass on failures only | `test_mass_criterion_uses_failures_and_successes` |
| R1-M35 contrasts get twice the margin | `test_margin_is_capped_at_four_tenths` |
| R1-M38 resume ignores the model | `test_resume_refuses_scores_of_another_model` |
| R1-M40 shards from an unsorted pair set | `test_shards_are_sorted_pair_slices` |
| R1-M48 sign-flip p unsmoothed | `test_sign_flip_p_is_smoothed` |
| R1-M51 padded replay without the no-goal game | `test_padded_replay_covers_no_goal` |
| A local AUC 0.6 | `test_local_auc_boundary_is_inclusive[4-False]` |
| A local AUC strict boundary | `test_local_auc_boundary_is_inclusive[3-True]` |
| A penalty competence without the parent population | `test_local_competence_rules_and_the_boundary` |
| A naming competence on all ingredients | `test_local_competence_rules_and_the_boundary` |
| A contrast competence ignores the other criteria | `test_local_competence_rules_and_the_boundary` |
| A two-item rows use all ingredients | `test_contrasts_use_class_matched_ingredients` |
| A padded rows use all ingredients | `test_contrasts_use_class_matched_ingredients` |
| A margin not capped | `test_margin_is_capped_at_four_tenths` |
| A interpretation ignores competence | `test_interpretation_separates_measurable_effects_from_distortions` |
| A inconclusive naming counts as constrained | `test_reading_table[confirmed below zero-inconclusive-confirmed below zero-completion contributes; naming not constrained]` |
| A timing comparison ignored | `test_timing_qualifier_and_no_partial_naming_rule` |
| A F3 true when another model is below zero | `test_across_models_patterns_and_f3` |
| A F3 never undecided | `test_across_models_patterns_and_f3` |
| A provenance commit check off | `test_provenance_requires_one_commit_and_pinned_revisions` |
| A provenance revision check off | `test_provenance_requires_one_commit_and_pinned_revisions` |
| A family draws 4000 | `test_family_tests_use_at_least_20000_draws` |
| A parent whole-name subset reversed | `test_parent_subsets_overlap_kinds_and_splits` |
| A overlap kind by characters | `test_parent_subsets_overlap_kinds_and_splits` |
| A power at the 0.05 level only | `test_equivalence_power` |
| A parent records without the neutral question | `test_record_counts_and_arms` |
| A fast bootstrap ignores the ingredient side | `test_fast_draws_equal_the_general_path` |
| A fast bootstrap resamples items | `test_statistics_helpers` |
| A role gap on all rows | `test_role_gap_descriptive_equals_naming_plus_completion` |
| Q round-1 question in the prompts | `test_every_arm_changes_only_what_its_name_says` |
| Q hash covers round 1's question | `test_hash_covers_every_field_and_context_key` |
| U1 completed two-item records kept | `test_a_made_partner_is_detected_and_leaves_naming_and_completion` |
| U1 partner-made check off | `test_a_made_partner_is_detected_and_leaves_naming_and_completion` |
| U2 matched cells unweighted | `test_matched_sensitivity_removes_a_profile_confound` |
| U2 matched cells of one item | `test_matched_sensitivity_removes_a_profile_confound` |
| U2 craftable and raw swapped | `test_matched_sensitivity_removes_a_profile_confound` |
| U4 strict against overlapping not pooled | `test_sensitivity_and_strict_comparison_are_reported` |

### TextCraft round 2, after review 2 (2026-11-22)

New tests cover:

- review 2's verdict mutants: a confirmed effect above zero needs its interval (N1); the bootstrap p is two-sided (N10); difference is checked before equivalence (N19); the sign flip compares absolute values (N24);
- review 2's surviving competence and contrast mutants: the competence populations exactly (N7–N9, N23); applicability read at one look-up (N5); the comparison with the one-look-up change (N6); the underpowered flag at Holm's first step (N12); the classification rates (N14); heuristics in the gated population (N15); the penalty competence passed across models (N16); the lift on every ingredient failure (N17); the Yes share on successes (N22); the TOST lower side (N25); the 95% interval in `between` (N11); verify's count of completed two-item records (N13);
- the coded rejection list R1–R5, each firing, not firing and not testable, including R3's matched confirmation and its item-structure note;
- the central question's four answers, which are mutually exclusive, with undefined tracking differences counting for neither side;
- the paired tracking differences;
- readings that say when a contrast is not competent;
- scores refused for arms a record does not have;
- the shared row set of the two-item contrasts and role gaps.

The official run's log, `notes/textcraft_r2_mutation_log.txt`, is committed with the results below.

**Official run (2026-11-22, at ef73d40, four shards; log `notes/textcraft_r2_mutation_log.txt`).** 137 round-2 mutants: 136 killed, 1 survivor, no timeouts, none not applied.

The survivor, "A role gap on all rows", had become equivalent. After review 2 the role-gap readout reads `two_item_rows`, which already keeps only rows with a two-item line, so the guard that mutant removed did nothing. The guard was deleted, and the mutant with it. The row filter itself is covered by "W two-item rows keep records without the line", which was killed. The list now holds 136 round-2 mutants, all killed in the official run:

| Mutant | First test that failed |
|---|---|
| two-item line kept for names with and | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| hash misses the system text | `test_item_hash_covers_fields_and_context` |
| eight look-ups become four | `test_parse_arm` |
| padding on 9-12 steps too | `test_check_records_reports_and_passes` |
| padded twin not checked | `test_padded_twin_and_class_checks` |
| gate failure ignored | `test_scores_every_record_and_arm_then_resumes` |
| shards by record not pair | `test_scores_every_record_and_arm_then_resumes` |
| gate unfolded for Gemma | `test_gate_folds_for_gemma` |
| completion lines reversed | `test_planted_effects_are_recovered_exactly` |
| family without Holm | `test_family_is_holm_adjusted` |
| competence AUC 0.6 | `test_competence_gate_uses_wave_a_rule` |
| stale scores accepted | `test_stale_unknown_duplicate_or_foreign_scores_stop` |
| sign flips single items | `test_sign_flip_flips_whole_pairs` |
| R1-M1 hash misses action | `test_hash_covers_every_field_and_context_key` |
| R1-M6 prompt check of system and examples off | `test_prompt_check_catches_a_changed_system_text` |
| R1-M13 list-is-pair-list check off | `test_text_checks_catch_a_named_strict_parent_and_a_changed_list` |
| R1-M21 mass on failures only | `test_mass_criterion_uses_failures_and_successes` |
| R1-M48 sign-flip p unsmoothed | `test_sign_flip_p_is_smoothed` |
| A penalty competence without the parent population | `test_local_competence_rules_and_the_boundary` |
| A padded rows use all ingredients | `test_contrasts_use_class_matched_ingredients` |
| A timing comparison ignored | `test_timing_qualifier_and_no_partial_naming_rule` |
| A provenance revision check off | `test_provenance_requires_one_commit_and_pinned_revisions` |
| A power at the 0.05 level only | `test_equivalence_power` |
| U1 partner-made check off | `test_a_made_partner_is_detected_and_leaves_naming_and_completion` |
| U4 strict against overlapping not pooled | `test_sensitivity_and_strict_comparison_are_reported` |
| R2-N6 comparable uses role gap not one-lookup change | `test_n6_comparable_reads_the_one_lookup_change` |
| R2-N10 bootstrap p one-sided | `test_n10_bootstrap_p_is_two_sided` |
| R2-N14 failure classification rate inverted | `test_n14_classification_rates` |
| R2-N18 lift competence on matched ingredients | `test_local_competence_rules_and_the_boundary` |
| R2-N22 yes share successes check off | `test_n22_yes_share_on_successes` |
| W R1 ignores the penalty competence | `test_rejection_list_per_model` |
| W R4 without the size comparison | `test_rejection_r4_lift_and_r2` |
| W tracking difference unpaired | `test_tracking_differences_are_paired` |
| padding after the history | `test_every_arm_changes_only_what_its_name_says` |
| hash misses the extended list | `test_item_hash_covers_fields_and_context` |
| parent may share craftables with the partner | `test_a_parent_sharing_a_craftable_with_the_partner_does_not_qualify` |
| frozen items reused as parents | `test_rebuilding_two_pairs_gives_the_committed_records` |
| padded class not checked | `test_padded_twin_and_class_checks` |
| resume ignores record hashes | `test_resume_refuses_scores_of_changed_records` |
| gate candidate check off | `test_gate_catches_a_candidate_written_differently_by_the_template` |
| lift on the goal role | `test_planted_effects_are_recovered_exactly` |
| naming against neutral | `test_planted_effects_are_recovered_exactly` |
| family shrinks with missing p | `test_family_is_fixed_at_fifty_and_pads_missing` |
| competence on the two-candidate score | `test_competence_gate_uses_wave_a_rule` |
| verdicts with missing scores | `test_missing_scores_withhold_verdicts` |
| heuristic H9 always true | `test_heuristics_agree_with_the_competence_script_on_every_item` |
| R1-M2 hash misses swapped line | `test_hash_covers_every_field_and_context_key` |
| R1-M9 excluded goals allowed as parents | `test_excluded_goals_never_qualify_as_parents` |
| R1-M14 NaN L not counted missing | `test_nan_scores_count_as_missing` |
| R1-M35 contrasts get twice the margin | `test_margin_is_capped_at_four_tenths` |
| R1-M51 padded replay without the no-goal game | `test_padded_replay_covers_no_goal` |
| A naming competence on all ingredients | `test_local_competence_rules_and_the_boundary` |
| A margin not capped | `test_margin_is_capped_at_four_tenths` |
| A F3 true when another model is below zero | `test_across_models_patterns_and_f3` |
| A family draws 4000 | `test_family_tests_use_at_least_20000_draws` |
| A parent records without the neutral question | `test_record_counts_and_arms` |
| Q round-1 question in the prompts | `test_every_arm_changes_only_what_its_name_says` |
| U2 matched cells unweighted | `test_matched_sensitivity_removes_a_profile_confound` |
| R2-N1 confirmed above without interval | `test_n1_confirmed_above_needs_the_interval` |
| R2-N7 goal_main competence vs all successes | `test_n7_n8_n9_n23_competence_populations_are_exact` |
| R2-N11 ci95 is a 90% interval in between() | `test_n11_between_reports_the_95_percent_interval` |
| R2-N15 heuristics on the full population, not the gated one | `test_n15_heuristics_are_read_in_the_gated_population` |
| R2-N19 equivalence checked before differs | `test_n19_difference_is_checked_before_equivalence` |
| R2-N23 parent competence vs main goal successes | `test_n7_n8_n9_n23_competence_populations_are_exact` |
| W R3 without the matched confirmation | `test_rejection_list_per_model` |
| W central mechanism without P competence | `test_central_question_is_exclusive[p_ci4-False-tracking4-undecided]` |
| W reading competence qualifier off | `test_reading_says_when_a_contrast_is_not_competent` |
| ext arm shows the item's own list | `test_every_arm_changes_only_what_its_name_says` |
| Gemma not folded | `test_gemma_folds_the_system_text_and_the_others_do_not` |
| strict parent not preferred | `test_every_parent_is_the_rules_choice` |
| parent draw seed drops the bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| text-only partner check off | `test_text_parent_check_catches_bad_parents` |
| golden problems ignored | `test_run_gate_requires_the_golden_and_the_attention` |
| gate single-token check off | `test_gate_fails_on_each_tokenization_problem[tok1-not one token]` |
| penalty against the swapped line | `test_planted_effects_are_recovered_exactly` |
| timing reversed | `test_planted_effects_are_recovered_exactly` |
| margin half an SD | `test_margin_is_capped_at_four_tenths` |
| confirmed without the interval | `test_verdicts_are_statistical_and_never_withheld` |
| used-up goal items kept | `test_populations_on_the_full_records` |
| heuristic H2 counts crafted units twice | `test_heuristics_agree_with_the_competence_script_on_every_item` |
| R1-M3 hash misses recipe_lines | `test_hash_covers_every_field_and_context_key` |
| R1-M10 text strict-word check off | `test_text_checks_catch_a_named_strict_parent_and_a_changed_list` |
| R1-M15 timing applicable on the estimate | `test_timing_applicability_reads_the_interval` |
| R1-M38 resume ignores the model | `test_resume_refuses_scores_of_another_model` |
| A local AUC 0.6 | `test_local_auc_boundary_is_inclusive[4-False]` |
| A contrast competence ignores the other criteria | `test_local_competence_rules_and_the_boundary` |
| A interpretation ignores competence | `test_interpretation_separates_measurable_effects_from_distortions` |
| A F3 never undecided | `test_across_models_patterns_and_f3` |
| A parent whole-name subset reversed | `test_parent_subsets_overlap_kinds_and_splits` |
| A fast bootstrap ignores the ingredient side | `test_fast_draws_equal_the_general_path` |
| Q hash covers round 1's question | `test_hash_covers_every_field_and_context_key` |
| U2 matched cells of one item | `test_role_gap_descriptive_equals_naming_plus_completion` |
| R2-N3 completion confirmatory reversed only | `test_planted_effects_are_recovered_exactly` |
| R2-N8 goal_padded competence vs all goal successes | `test_n7_n8_n9_n23_competence_populations_are_exact` |
| R2-N12 underpowered flag at the 0.05 level | `test_n12_underpowered_at_the_first_holm_step` |
| R2-N16 across uses lift competence | `test_n16_across_reads_the_penalty_competence` |
| R2-N20 margin from all main records incl successes | `test_margin_is_capped_at_four_tenths` |
| R2-N24 sign flip obs not absolute | `test_n24_sign_flip_uses_the_absolute_value` |
| W R3 testable on P competence only | `test_rejection_list_per_model` |
| W central poor reasoning needs incompetence only | `test_central_question_is_exclusive[p_ci6-True-tracking6-poor state reasoning]` |
| W two-item rows keep records without the line | `test_role_gap_descriptive_equals_naming_plus_completion` |
| two-item line not sorted | `test_two_item_line_is_sorted_and_shared_by_a_couple` |
| yes/no query keeps Next observation | `test_every_arm_changes_only_what_its_name_says` |
| yes/no candidates reversed | `test_candidates` |
| added lines appended | `test_structure_check_catches_changes` |
| three failures per bin | `test_rebuilding_two_pairs_gives_the_committed_records` |
| yes/no scored with two passes | `test_yes_no_arms_use_one_forward_pass` |
| attention not checked | `test_run_gate_requires_the_golden_and_the_attention` |
| gate first-token check off | `test_gate_fails_on_each_tokenization_problem[tok2-share their first token]` |
| gap without the ingredient term | `test_planted_effects_are_recovered_exactly` |
| timing always applicable | `test_timing_not_applicable_without_a_role_gap` |
| TOST from the smaller side | `test_statistics_helpers` |
| naming reading only without completion | `test_reading_table[confirmed below zero-confirmed below zero-confirmed below zero-naming / partial-completion penalty]` |
| bootstrap resamples items | `test_fast_draws_equal_the_general_path` |
| used-up AUC not role-matched | `test_used_up_auc_is_role_matched` |
| R1-M4 hash context misses NEXT | `test_hash_covers_every_field_and_context_key` |
| R1-M11 padded verdict check off | `test_padded_reply_check` |
| R1-M16 padded goal keeps used up | `test_populations_on_the_full_records` |
| R1-M40 shards from an unsorted pair set | `test_shards_are_sorted_pair_slices` |
| A local AUC strict boundary | `test_local_auc_boundary_is_inclusive[3-True]` |
| A two-item rows use all ingredients | `test_contrasts_use_class_matched_ingredients` |
| A inconclusive naming counts as constrained | `test_reading_table[confirmed below zero-inconclusive-confirmed below zero-completion contributes; naming not constrained]` |
| A provenance commit check off | `test_provenance_requires_one_commit_and_pinned_revisions` |
| A overlap kind by characters | `test_parent_subsets_overlap_kinds_and_splits` |
| A fast bootstrap resamples items | `test_statistics_helpers` |
| U1 completed two-item records kept | `test_a_made_partner_is_detected_and_leaves_naming_and_completion` |
| U2 craftable and raw swapped | `test_matched_sensitivity_removes_a_profile_confound` |
| R2-N5 applicability on the unpadded gap | `test_n5_applicability_reads_one_lookup` |
| R2-N9 ingredient padded competence vs all ingredient successes | `test_n7_n8_n9_n23_competence_populations_are_exact` |
| R2-N13 verify partner-done count off | `test_n13_verify_counts_two_item_records_with_a_made_goal` |
| R2-N17 lift on class-matched ingredients | `test_n17_lift_uses_every_ingredient_failure` |
| R2-N21 F3 counts Llama | `test_across_models_patterns_and_f3` |
| R2-N25 tost low side at twice the margin | `test_n25_tost_low_side_uses_the_margin` |
| W R5 ignores the timing competence | `test_rejection_list_per_model` |
| W central shortcut on the point estimate | `test_central_question_is_exclusive[p_ci3-True-tracking3-undecided]` |
| W scores of other arms accepted | `test_scores_must_be_arms_of_their_record` |

### TextCraft round 2, sign-off corrections (2026-11-22)

The sign-off corrections (design 15.F) and the statistical audit (15.A) added tests for:

- R3 labelled a descriptive diagnostic;
- R4 descriptive, with a paired lift difference resampled over goal pairs;
- R5's wording;
- interpretation and reading strings that never call an AUC gate competence or understanding;
- the neutral central-question labels;
- the inference map, with no p-value outside the 50 family tests;
- Holm with padding to 50;
- TOST from one-sided bootstrap ps;
- the margin computed once and shared by all contrasts;
- goal-pair clusters in every resample of a model block;
- the exact Dn + Dc identity on the 996 shared rows.

The mutation list was updated: two stale mutants were replaced or removed, and ten mutants were added for the corrections. 145 round-2 mutants are now listed. The 22 new or changed ones (the 15.F mutants, the review-2 "W" mutants whose code changed, and the interpretation mutant) were run in official mode at 170db45, and all 22 were killed by named tests. Their log is appended to `notes/textcraft_r2_mutation_log.txt`. The other 123 were killed in the official run at ef73d40 and their code is unchanged.
