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

### TextCraft round 2, pre-freeze audit (2026-11-22)

The independent audit (`notes/textcraft_r2_audit.md`) left three of its 12 mutants alive: X1, X4 and X12. Four tests were added:

- `test_x1_r3_does_not_fire_when_both_gaps_cross_zero`;
- `test_x4_role_gap_is_naming_plus_completion_through_contrasts`, which uses noisy scores through `contrasts` with agreement to 1e-12;
- `test_x12_lift_difference_is_llama_minus_qwen`;
- `test_x10_central_question_exact_tie_counts_as_inventory_consistent`.

A further test, `test_analysis_requires_the_five_models`, covers the model-set check and the timing gates in the inference map.

Rerun at ae8f005 with the auditor's strict script, all 12 audit mutants are killed by named tests. The two new mutants (the model-set check, the gates entry) are killed in official mode. The 12 audit mutants and the 2 new ones are now also in `scripts/mutation_check.py`, which holds 159 round-2 mutants. The log is appended to `notes/textcraft_r2_mutation_log.txt`.
Full suite at the pre-freeze audit (run once at 0db1f7c, before the deviations entry): 1,008 passed, 7 skipped.
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

## TextCraft round 3 (2026-11-23, before any round-3 score; amended after review 1)

`tests/test_textcraft_r3.py` (78 tests) covers round 3 (`notes/textcraft_r3_design.md`, sections 1 to 14) without a
model or a real tokenizer: the model specs and pins, the GPU classes, the seeded subsets, the round-3 gate on a
stand-in chat tokenizer with a thinking switch, the scoring runs in mock mode, the attention-agreement rule, the
thinking arm (its jobs, per-batch seeds, record-and-arm order, budget forcing split into cap and ended unclosed, the
prompt and padding checks at job start, resume), and every decision rule of the analysis, including the review-1
amendments: provenance per job with two-L40S shards and the freeze-tag commit (M1, S12), "not scored" from filed
gate records (M2), the Gemma fallback rule on the filed smoke record with its format-mismatch trigger (M3, S9), the
environment-check plan mirroring round 2's shards, the record-by-record GPU match, the outcomes and B3's older scores
(M4, S5, S8), H3-P with one competent model (S1), power at 0.05/30 (S2), B3's power, flag and fixed-denominator
column (S3), plain-words directions (S4), round 2's provenance check (S6), the resample count (S7), no p-values in
exploratory blocks (S11), and two end-to-end mock runs. `tests/test_textcraft_r3_serve2.py` runs only in serve2
(ICE).

Mutation testing, official run under the strict named-kill rule (log `notes/textcraft_r3_mutation_log.txt`): **109
round-3 mutants, all killed, by 57 different tests**: the 67 earlier mutants (9 rewritten for the amended code), the
reviewer's 11 (adapted to the amended code; 8 had survived the whole suite before), and 31 for the review-1 fixes. A
screen run (MUTATION_FAST_ONLY=1) found one survivor (the environment-check run accepting any model, which an older
test no longer reached once Gemma 2 joined the check); its test was changed before the official run.

| Mutant | Killed by |
|---|---|
| B3 NaN dropped | `tests/test_textcraft_r3.py::test_b3_family_and_pair_readings` |
| B3 margin 0.5 | `tests/test_textcraft_r3.py::test_b3_family_and_pair_readings` |
| B3 read when the environment fails | `tests/test_textcraft_r3.py::test_environment_status_and_b3_source` |
| Gemma 12B allows unexpected tensors | `tests/test_textcraft_r3.py::test_new_specs_take_the_31b_template_handling_and_strict_loading` |
| Gemma 12B without the attention check | `tests/test_textcraft_r3.py::test_new_specs_take_the_31b_template_handling_and_strict_loading` |
| Gemma 12B without the generation suffix | `tests/test_textcraft_r3.py::test_new_specs_take_the_31b_template_handling_and_strict_loading` |
| H3-P ignores R2 | `tests/test_textcraft_r3.py::test_h3p_reading` |
| H3-P partial counted as support | `tests/test_textcraft_r3.py::test_h3p_reading` |
| H3-P supported with one competent model | `tests/test_textcraft_r3.py::test_h3p_reading` |
| M1 job GPUs collapsed to a set | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| M1 provenance accepts jobs on different GPU lists | `tests/test_textcraft_r3.py::test_provenance_accepts_two_l40s_per_job_and_checks_each_job` |
| M2 B3 pair with a model not scored | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| M2 gate failures ignored | `tests/test_textcraft_r3.py::test_model_state_from_filed_gate_records` |
| M2 loading failure not filed | `tests/test_textcraft_r3.py::test_a_loading_failure_is_filed_and_stops_the_model` |
| M2 not scored counted as waiting | `tests/test_textcraft_r3.py::test_a_core_model_not_scored_enters_at_p_one_and_the_readings_proceed` |
| M3 analysis accepts any gemma_core | `tests/test_textcraft_r3.py::test_the_analysis_refuses_a_gemma_core_the_rule_does_not_give` |
| M3 fallback runs without the rule | `tests/test_textcraft_r3.py::test_the_fallback_model_runs_only_when_the_rule_fired` |
| M3 the rule never fires | `tests/test_textcraft_r3.py::test_gemma_slot_rule` |
| M4 envcheck re-partitions the seeded records | `tests/test_textcraft_r3.py::test_envcheck_jobs_score_round_two_shards_on_the_exact_gpu` |
| M4 exact GPU compares the first name only | `tests/test_textcraft_r3.py::test_envcheck_jobs_score_round_two_shards_on_the_exact_gpu` |
| M4 no single-type shortcut | `tests/test_textcraft_r3.py::test_envcheck_plan_from_round_two_gpu_names` |
| M4 per-score limit copied | `tests/test_textcraft_r3.py::test_environment_check_per_score_criterion_is_wave_bs_live_rule` |
| M4 plan always runnable | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| M4 token disagreement not a failure | `tests/test_textcraft_r3.py::test_environment_status_and_b3_source` |
| M4 unobtainable still waits | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| OLMo default prompt allowed | `tests/test_textcraft_r3.py::test_new_specs_take_the_31b_template_handling_and_strict_loading` |
| OLMo not strict | `tests/test_textcraft_r3.py::test_new_specs_take_the_31b_template_handling_and_strict_loading` |
| R2 counts incompetent models | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| R2 fires despite a confirmed penalty | `tests/test_textcraft_r3.py::test_r2_for_round_three` |
| S10 cap not separated from ended | `tests/test_textcraft_r3.py::test_budget_forcing_and_the_readout_point` |
| S10 padding tolerance off | `tests/test_textcraft_r3.py::test_thinking_start_checks` |
| S10 thinking prompt ending unchecked | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| S11 exploratory p-values kept | `tests/test_textcraft_r3.py::test_exploratory_blocks_carry_no_p_values` |
| S12 any change counts as code-identical | `tests/test_textcraft_r3.py::test_provenance_requires_the_freeze_commit_and_one_revision` |
| S12 job start at any commit | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| S2 power at the round-2 step | `tests/test_textcraft_r3.py::test_round_three_power_uses_holm_first_step_of_thirty` |
| S3 underpowered flag dropped | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| S4 size words reversed | `tests/test_textcraft_r3.py::test_pair_readings_say_the_direction_in_plain_words` |
| S5 b3 source accepts any config | `tests/test_textcraft_r3.py::test_environment_status_and_b3_source` |
| S6 round-2 provenance skipped | `tests/test_textcraft_r3.py::test_mock_end_to_end` |
| S7 b_draws accepted outside mock | `tests/test_textcraft_r3.py::test_b3_uses_twenty_thousand_resamples_outside_mock` |
| S9 mass alone fires the fallback | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| S9 only special tokens are structural | `tests/test_textcraft_r3.py::test_structural_tokens` |
| S9 smoke diagnostic not recorded | `tests/test_textcraft_r3.py::test_smoke_runs_record_the_format_diagnostic` |
| an A100 of 40 GB counts as 80 | `tests/test_textcraft_r3.py::test_gpu_classes_and_pins` |
| attention disagreement ignored | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| attention tolerance loosened | `tests/test_textcraft_r3.py::test_attention_agreement_rule` |
| attention without SDPA loaded | `tests/test_textcraft_r3.py::test_attention_agreement_rule` |
| base tag keeps the suffix | `tests/test_textcraft_r3.py::test_spec_for_variants_and_base_tags` |
| envcheck completeness off | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| envcheck of any model | `tests/test_textcraft_r3.py::test_runs_refuse_unknown_models_and_unverified_records` |
| envcheck on any GPU | `tests/test_textcraft_r3.py::test_environment_check_matches_gpus_record_by_record` |
| envcheck tolerance loosened | `tests/test_textcraft_r3.py::test_environment_check_criteria` |
| fallback slot unchecked | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| family B3 size 20 | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| finding counts incompetent models | `tests/test_textcraft_r3.py::test_round2_finding` |
| finding needs one competent model | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| first-token split on the whole L | `tests/test_textcraft_r3.py::test_first_token_tail_split_adds_up` |
| format check on all records | `tests/test_textcraft_r3.py::test_seeded_subsets_are_fixed` |
| gate candidate check off | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| gate ending check off | `tests/test_textcraft_r3.py::test_gate_catches_a_wrong_ending_a_forbidden_text_and_an_unread_switch` |
| gate failure ignored | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| gate first-token check off | `tests/test_textcraft_r3.py::test_gate_catches_a_multi_token_yes_and_shared_first_tokens` |
| gate forbidden text off | `tests/test_textcraft_r3.py::test_gate_catches_a_wrong_ending_a_forbidden_text_and_an_unread_switch` |
| gate ignores the suffix | `tests/test_textcraft_r3.py::test_gate_passes_and_counts_every_record_and_arm` |
| gate isolation check off | `tests/test_textcraft_r3.py::test_gate_catches_tokens_changed_outside_the_goal_line` |
| gate mistral-common optional | `tests/test_textcraft_r3.py::test_gate_requires_mistral_common_when_declared` |
| gate single-token check off | `tests/test_textcraft_r3.py::test_gate_catches_a_multi_token_yes_and_shared_first_tokens` |
| gate switch check off | `tests/test_textcraft_r3.py::test_gate_catches_a_template_that_ignores_the_thinking_switch` |
| golden problems ignored | `tests/test_textcraft_r3.py::test_run_gate_requires_the_golden_and_records_the_loading` |
| golden tokens ignore the id hash | `tests/test_textcraft_r3.py::test_golden_tokens_check` |
| gpu check off | `tests/test_textcraft_r3.py::test_gpu_classes_and_pins` |
| isolation ignores the right side | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| large models on any card | `tests/test_textcraft_r3.py::test_gpu_classes_and_pins` |
| lift values from the class-matched rows | `tests/test_textcraft_r3.py::test_record_values_agree_with_round_twos_contrast_inputs` |
| null replication tolerates a confirmed effect | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| one-fifth rule halved | `tests/test_textcraft_r3.py::test_format_check_one_fifth_rule` |
| pair difference reversed | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| pair differs without the interval | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| pair equivalence unadjusted | `tests/test_textcraft_r3.py::test_b3_family_and_pair_readings` |
| pair of the wrong Gemma | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| pair on records of one model only | `tests/test_textcraft_r3.py::test_paired_difference_detects_a_real_difference_and_uses_common_records_only` |
| pair resamples differ between models | `tests/test_textcraft_r3.py::test_paired_difference_is_scale_free_and_zero_for_the_same_model` |
| pooled SD unweighted | `tests/test_textcraft_r3.py::test_stats_match_the_definitions` |
| provenance GPU check off | `tests/test_textcraft_r3.py::test_provenance_accepts_two_l40s_per_job_and_checks_each_job` |
| provenance commit check off | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| replication counts incompetent models | `tests/test_textcraft_r3.py::test_replication_rule` |
| replication ignores the opposite sign | `tests/test_textcraft_r3.py::test_replication_rule` |
| replication partial counted | `tests/test_textcraft_r3.py::test_replication_rule` |
| resume ignores record hashes | `tests/test_textcraft_r3.py::test_score_run_scores_every_record_and_arm_then_resumes` |
| review1 rev1 standardized SD with n instead of n-1 | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| review1 rev10 thinking keeps text after the closing marker | `tests/test_textcraft_r3.py::test_budget_forcing_and_the_readout_point` |
| review1 rev11 thinking jobs not in record-and-arm order | `tests/test_textcraft_r3.py::test_thinking_batches_have_their_own_seeds_in_record_and_arm_order` |
| review1 rev2 envcheck goal effect over every chosen record | `tests/test_textcraft_r3.py::test_environment_check_criteria` |
| review1 rev3 envcheck yes/no share criterion off | `tests/test_textcraft_r3.py::test_environment_check_share_criterion` |
| review1 rev4 thinking unforced subset keeps forced traces | `tests/test_textcraft_r3.py::test_thinking_unforced_subset_and_forced_reasons` |
| review1 rev5 thinking batch seed constant | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| review1 rev6 provenance accepts mixed revisions | `tests/test_textcraft_r3.py::test_provenance_requires_the_freeze_commit_and_one_revision` |
| review1 rev7 B3 family on 2,000 resamples | `tests/test_textcraft_r3.py::test_b3_uses_twenty_thousand_resamples_outside_mock` |
| review1 rev8 attention agreement on one item | `tests/test_textcraft_r3.py::test_attention_agreement_rule` |
| review1 rev9 envcheck accepts rescoring shards on mixed GPUs | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| standardized without the SD | `tests/test_textcraft_r3.py::test_stats_match_the_definitions` |
| strict floor 0.7 | `tests/test_textcraft_r3.py::test_round3_extras_are_descriptive_and_include_the_strict_floor` |
| think failures not sampled | `tests/test_textcraft_r3.py::test_thinking_jobs_and_procedure` |
| think forced not flagged | `tests/test_textcraft_r3.py::test_thinking_run_mock_generates_scores_and_resumes` |
| think readout before the blank line | `tests/test_textcraft_r3.py::test_budget_forcing_and_the_readout_point` |
| think reasoning cap 4096 | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| think resume ignores hashes | `tests/test_textcraft_r3.py::test_core_extension_fallback_and_thinking_arm` |
| thinking arm with thinking off | `tests/test_textcraft_r3.py::test_every_model_is_pinned_at_the_design_revision` |
