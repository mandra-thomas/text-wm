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

## Mechanism battery (added 2026-11-20, before any of its scores)

`tests/test_mech2.py` covers `wf/mech2.py` (design: `mechanism2_design.md`):

- **Templates.** Every arm builds for every ALFWorld task template (two wordings per type where the data have two) and every verb (take, move, heat, cool, clean), at step 0 and after; the exact text of each template; existence presupposed only by desire_keep and mention_ignore; no instance named except the action's receptacle in desire_check; the described arms name no object.
- **Prompts differ only where intended.** Every content arm's prompt equals the aligned prompt with only the task line replaced, in both formats; only the aligned, done and cancelled arms show the real task; a position arm changes only the task line's position, and the line is first (start) or last (end); the step-0 `_end` arms repeat the plain prompt and are not scored; the yes/no prompt equals the two-candidate prompt up to rq1v's question; the reused arms equal `rq1_score`'s, `rq1x_score`'s and `rq1v_score`'s prompts message for message. `scripts/check_mech2_prompts.py` runs the same checks on every real item of the battery (no problems found on 2026-11-20; 20,017 prompts per format per model).
- **Items.** The sample is proportional by (game, failure reason), seeded and independent of input order; every step-0 item is added; the "done" contradiction flag.
- **Analysis on synthetic data.** Each account's fingerprint (H2, H3, H4, H5) is planted in synthetic scores (40 games, 12 items each); the readout finds the planted account with no contradicted contrast and every other account contradicted at least once. A planted position effect is read as H6 and contradicts H2, H4 and H5 on P6. Also: contrast signs and the size ratio, Holm within the family, the category rules, P5 restricted to placement tasks, the item share counting ties against Yes, the alias lookup, the few-shot overlap and knowledge-split checks.
- **Gate and runner.** The tokenization gate passes a clean template, stops on a space before the reply or a multi-token "Yes", and refuses an unpinned revision; a failed gate stops the run before any score; two shards then the analysis end to end on mock data (every job scored once, pinned revision on every record); a resume scores nothing twice.

**Deliberate bugs.** 28 mutants of `wf/mech2.py` were added to `scripts/mutation_check.py`, which now prints the first failing test for each. Run on 2026-11-20 (`MUTANTS_MATCHING=mech2`): all 28 killed, none timed out.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| desire_keep not reversed | `test_templates_read_as_written` |
| desire_check presupposes the object | `test_templates_read_as_written` |
| ignore sentence on the real task | `test_templates_read_as_written` |
| done quotes the swapped task | `test_templates_read_as_written` |
| described arm keeps the name | `test_described_arms_name_no_object` |
| start puts the task line last | `test_position_arms_move_only_the_task_line` |
| end arms skipped at every step | `test_jobs_cover_every_arm_and_skip_aliases` |
| alias reads the aligned score | `test_usual_position_and_aliases` |
| alias fallback at every step | `test_alias_reads_the_plain_score` |
| yes/no keeps the observation cue | `test_yes_no_differs_only_in_the_question` |
| sample allocated equally | `test_sample_is_stratified_seeded_and_order_free` |
| sample follows input order | `test_sample_is_stratified_seeded_and_order_free` |
| step-0 items not added | `test_battery_items_add_every_step0_item` |
| contrast sign flipped | `test_each_planted_fingerprint_is_read_as_its_own_account[H2_evidence]` |
| ratio over its own size | `test_a_planted_position_effect_is_read_as_an_artefact` |
| Holm skipped | `test_holm_is_applied_within_the_family` |
| null band widened | `test_categories_follow_the_rules` |
| large threshold lowered | `test_categories_follow_the_rules` |
| P5 against swapped | `test_each_planted_fingerprint_is_read_as_its_own_account[H3_salience]` |
| P5 includes examine tasks | `test_p5_reads_placement_tasks_only` |
| item share counts ties toward Yes | `test_item_share_counts_ties_against` |
| gate failure ignored | `test_gate_stops_on_a_space_before_the_reply_or_a_split_word` |
| gate accepts any revision | `test_gate_refuses_an_unpinned_revision` |
| gate accepts a multi-token answer | `test_gate_stops_on_a_space_before_the_reply_or_a_split_word` |
| resume rescores saved work | `test_resume_scores_nothing_twice` |
| done contradicted with a target unseen | `test_done_contradicted_only_when_every_target_was_seen_without_the_object` |
| overlap read against set 1 only | `test_fewshot_overlap_finds_a_planted_overlap_boost` |
| knowledge split reversed | `test_knowledge_split_reads_mech_state_questions` |

The gate's tests use a stand-in tokenizer and template: the laptop has no transformers, so the gate meets the real Qwen3 and Llama 3.1 templates first in `mech2_score_smoke` on ICE.

**Review 1 (2026-11-20, before any score).** The review's twenty mutants (R1 to R20; ten had survived: R2, R5, R6, R7, R10, R12, R13, R15, R16, R17) and fourteen mutants of the amendment were added to `scripts/mutation_check.py`; five of the review's mutants were rewritten where the amended code moved their text (R1 now removes the rule that a non-trivial interval needs Holm's p below 0.05; R11 reads any overlap of zones as consistent; R14, R15 match the gate's prompt transform; R20 drops P6 from the seven-contrast family). The existing 28 were updated where their text moved (null band, large threshold, the ratio, P5's population). New tests: P6 reads the step-0 items only and counts only for the position account (a trivial position effect contradicts it); the yes/no family reads yes/no scores with its own denominator; σ through the sigmoid; the permutation flips whole games; the dropped-draw rule; size zones and the TRIVIAL label; a trivial effect never contradicts a ≈0 prediction; verdicts (consistent, contradicted, untested) and the two-format rule; the mixture summary; the largest-remainder allocation; P5's and P7's items; the negated arm's template and claim status; the gate on a resume left with yes/no prompts only; the resume guard on item hashes; Mistral's registry pin and template handling, in the runner too. Run on 2026-11-20 (`MUTANTS_MATCHING=mech2`, four shards): all 62 mech2 mutants killed, none timed out or failed to apply. `tests/test_mech2.py`: 57 tests pass.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| R1 non-trivial zones read without significance | `test_categories_follow_the_zones` |
| R2 P6 on the sample | `test_position_counts_only_for_the_position_account` |
| R3 desire participle always taken | `test_templates_read_as_written` |
| R4 move preposition from | `test_templates_read_as_written` |
| R5 yes/no family read on cand | `test_yes_no_family_reads_yes_no_scores_with_its_own_denominator` |
| R6 r denominator always cand | `test_yes_no_family_reads_yes_no_scores_with_its_own_denominator` |
| R7 sigma read as L | `test_sigma_is_read_through_the_sigmoid` |
| R8 knowledge: either answer right | `test_knowledge_split_reads_mech_state_questions` |
| R9 done contradicted with one target seen | `test_done_contradicted_only_when_every_target_was_seen_without_the_object` |
| R10 smallest remainder gets the extra seat | `test_largest_remainders_get_the_extra_seats` |
| R11 any overlap read as consistent | `test_a_trivial_effect_never_contradicts_a_null_prediction` |
| R12 permutation flips items, not games | `test_permutation_flips_whole_games` |
| R13 later includes step 0 | `test_later_holds_no_step0_item` |
| R14 gate checks only the first item | `test_gate_passes_a_clean_template` |
| R15 gate skips yes/no prompts | `test_gate_checks_every_yes_no_prompt_it_is_given` |
| R16 resume ignores item hashes | `test_resume_refuses_records_of_other_items` |
| R17 ratio accepts negative denominators | `test_ratio_needs_a_positive_reference_and_few_dropped_draws` |
| R18 mention_ignore on neutral line | `test_templates_read_as_written` |
| R19 cancelled same as done | `test_templates_read_as_written` |
| R20 Holm over six (P6 dropped) | `test_holm_is_applied_within_the_family` |
| trivial band read as a contradiction of 0 | `test_each_planted_fingerprint_is_read_as_its_own_account[H3_salience]` |
| dropped draws ignored | `test_ratio_needs_a_positive_reference_and_few_dropped_draws` |
| P6 counts for evidence | `test_each_planted_fingerprint_is_read_as_its_own_account[H2_evidence]` |
| untested read as consistent | `test_readout_verdicts_and_the_joint_rule` |
| one format decides | `test_readout_verdicts_and_the_joint_rule` |
| mixture never named | `test_joint_summary_names_a_mixture_when_every_content_account_is_contradicted` |
| P5 keeps takes from a target-type receptacle | `test_p5_leaves_out_takes_from_a_target_type_receptacle_and_contradicted_items` |
| P5 keeps contradicted items | `test_p5_leaves_out_takes_from_a_target_type_receptacle_and_contradicted_items` |
| P7 keeps contradicted claims | `test_negated_claim_status_and_p7_items` |
| negated built for every verb | `test_every_arm_builds_for_every_task_type_and_verb` |
| negated drops the user | `test_templates_read_as_written` |
| Mistral without its template settings | `test_mistral_is_scored_with_the_families_handling` |
| folding model not folded | `test_a_folding_model_gets_the_system_text_in_its_first_turn` |
| scoring skips the template handling | `test_the_runner_scores_through_the_model_template_handling` |

The real tokenizers (pinned Qwen3-8B, Llama-3.1-8B and Mistral-7B-Instruct-v0.3, ICE login node, CPU) passed the battery's gate on every prompt: both candidates of all 1,403 items and all 20,731 yes/no prompts per model, no problem; "Yes" and "No" single tokens for each; every item's first tokens "You" against "Nothing".

**Review 2 (2026-11-20, before any score).** The review's 22 mutants (Q1 to Q22; fifteen had survived: Q1, Q2, Q4 to Q11, Q13, Q15, Q16, Q20, Q21) were added to `scripts/mutation_check.py`, adapted where the amended code moved their text: Q4 and Q10 now remove a zone the amended cells allow (H2 on P1 without L; H2 on P2 allowing S), since the amendment made the original mutants' additions part of the design; Q5 to Q8 follow the one-line-per-account prediction table; Q13 targets the two-format rule's helper `_joint`; Q19 the template shared by desire_check and desire_check_can. Twenty-six mutants of the amendment (N1 to N26: the minimal pairs and P8, the new templates, Mistral's arms, the rival rule for "consistent", the summary lists and wording, the integrity check and the item-file hashes, the secondary readouts, the first-token flag, the mixture readout, the separation check) and three for the step-0 restriction of the position arms were added. The `_end` arms and the alias reading were removed with the amendment, so their three mutants ("end arms skipped at every step", "alias reads the aligned score", "alias fallback at every step") were replaced by the three step-0 mutants below, and nine existing mutants were updated where their text moved (desire_keep and desire_check templates, made specific now that two templates share their words; the described arm; resume; P6 for evidence; untested; one format decides; the mixture wording; negated for every verb). New tests: the minimal pairs differ only in the designed words; the design note's amended prediction table, parsed cell by cell ("+" as S or L), equals `PREDICTIONS`; P6 counts every step-0 item of the union when the sample leaves some out; contradicted in one format and undecided in the other is undecided, and the account untested; "consistent" needs a rival contradicted on the same contrast (for P6, the rival "no position effect"); no two accounts share a row, and the two pairs separated by one contrast only are known; Holm at a fixed family of eight with a missing p as 1; the dropped-draw rule counts resamples, not items; the secondary readouts read the arms they name (all-placement P5, P5 against cancelled's lift, the old P1 and P2, the desire gap, P7 and P8 on items whose receptacle was not seen, the mixture readout); the first-token sign split is flagged; the by-history P7 readout reads takes of the goal object only; the knowledge split reads the content contrasts only; the analysis reports Mistral by default; the scoring run records the item files' hashes; the analysis refuses changed items, missing scores and doubled scores; the integrity check counts per model, arm and format; Mistral's arms are the primary contrasts' arms. Run on 2026-11-20 (`MUTANTS_MATCHING=mech2`, four shards): all 110 mech2 mutants killed, none timed out or failed to apply. `tests/test_mech2.py`: 77 tests pass.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| position arms after step 0 | `test_every_arm_builds_for_every_task_type_and_verb` |
| position arms dropped at step 0 too | `test_every_arm_builds_for_every_task_type_and_verb` |
| missing score read from another arm | `test_a_missing_score_reads_as_none` |
| Q1 NaN p not replaced by 1 (family shrinks) | `test_holm_family_of_eight_counts_a_missing_p_as_one` |
| Q2 P6 on sample step-0 only | `test_p6_counts_every_step0_item_of_the_union_even_outside_the_sample` |
| Q3 strata by game only | `test_sample_is_stratified_seeded_and_order_free` |
| Q4 H2 P1 without L | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q5 H3 P5 allows N | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q6 H4 P7 allows N | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q7 H5 P3 only Z | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q8 H6_lexical P4 allows S | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q9 H6_position allows only N | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q10 H2 P2 allows S | `test_the_prediction_table_in_the_note_is_the_code_s` |
| Q11 dropped-draw rule uses 2% of n_items | `test_dropped_draw_rule_counts_resamples_not_items` |
| Q12 r interval from mean bootstrap | `test_yes_no_family_reads_yes_no_scores_with_its_own_denominator` |
| Q13 contradicted only needs one format | `test_two_format_rule_contradicted_in_one_format_and_undecided_in_the_other_is_undecided` |
| Q14 negated_status reads exact instances | `test_negated_claim_status_and_p7_items` |
| Q15 all-placement done against swapped | `test_secondary_readouts_read_the_arms_they_name` |
| Q16 analysis default models omit mistral | `test_end_to_end_two_shards_then_analysis` |
| Q17 zones_of returns only endpoints | `test_categories_follow_the_zones` |
| Q18 categorize ignores estimable | `test_ratio_needs_a_positive_reference_and_few_dropped_draws` |
| Q19 desire_check article always a | `test_templates_read_as_written` |
| Q20 knowledge split on step-0 terms too | `test_knowledge_split_reads_the_content_contrasts_only` |
| Q21 P7 by history on takes of any object | `test_negated_by_history_reads_takes_of_the_goal_object_only` |
| Q22 TRIVIAL label replaced by POS_SMALL when significant | `test_position_counts_only_for_the_position_account` |
| N1 P1 against swapped | `test_each_planted_fingerprint_is_read_as_its_own_account[H3_salience]` |
| N2 P2 against swapped | `test_each_planted_fingerprint_is_read_as_its_own_account[H2_evidence]` |
| N3 P8 against swapped | `test_each_planted_fingerprint_is_read_as_its_own_account[H3_salience]` |
| N4 desire_check_can hopes it cannot | `test_templates_read_as_written` |
| N5 desire_keep_sw without the swapped task | `test_templates_read_as_written` |
| N6 affirmed denies too | `test_templates_read_as_written` |
| N7 Mistral scored on every arm | `test_the_secondary_model_is_scored_and_reported_descriptively` |
| N8 rival readings left out | `test_position_counts_only_for_the_position_account` |
| N9 consistent without a rival contradicted | `test_consistent_needs_a_rival_contradicted_on_the_same_contrast` |
| N10 untested listed as nothing | `test_two_format_rule_contradicted_in_one_format_and_undecided_in_the_other_is_undecided` |
| N11 contradicted read as rejected | `test_readout_verdicts_and_the_joint_rule` |
| N12 integrity ignores item hashes | `test_analysis_refuses_a_score_of_a_changed_item` |
| N13 integrity mismatch never refuses | `test_analysis_refuses_a_score_of_a_changed_item` |
| N14 analysis ignores the integrity check | `test_analysis_refuses_a_score_of_a_changed_item` |
| N15 item files not recorded | `test_scoring_records_the_item_files_hashes` |
| N16 duplicates not counted | `test_analysis_refuses_doubled_scores` |
| N17 missing scores not counted | `test_analysis_refuses_missing_or_doubled_scores` |
| N18 P5 against the aligned effect, not cancelled | `test_secondary_readouts_read_the_arms_they_name` |
| N19 contents-not-shown reads every take item | `test_contents_not_shown_reads_only_the_takes_whose_receptacle_was_not_seen` |
| N20 first-token split flagged on agreement | `test_secondary_readouts_read_the_arms_they_name` |
| N21 mixture readout reads cand for yes/no | `test_secondary_readouts_read_the_arms_they_name` |
| N22 status line without yes/no categories | `test_end_to_end_two_shards_then_analysis` |
| N23 old P1 reads the new keep line | `test_secondary_readouts_read_the_arms_they_name` |
| N24 separations read overlap as separation | `test_no_two_accounts_share_a_row_and_the_thin_pairs_are_known` |
| N25 affirmed with "a" for every object | `test_templates_read_as_written` |
| N26 H4 P2 allows Z | `test_each_planted_fingerprint_is_read_as_its_own_account[H5_policy_blend]` |

The gate was rerun on the real tokenizers after the amendment (ICE, `coc-cpu` partition, CPU job 7205732, the job's environment, 2026-11-20): both candidates of all 1,403 items for each model, and every yes/no prompt each model is scored on (19,292 for Qwen3-8B and Llama-3.1-8B, 15,083 for Mistral-7B-Instruct-v0.3, among them 3,520 of the three new arms), no problem for any model; "Yes" and "No" are the same single tokens as before.

**Review 3 (2026-11-21, before any score).** The review's 22 mutants (A1 to A22; fourteen had survived: A1 to A9, A11, A15, A18, A19, A21) were added to `scripts/mutation_check.py`, adapted where the amended code moved their text (A4: the refusal list now ends with the primary-arm NaN count; A13: the mixture readout is now followed by M; A16: the affirmed line reads the swapped line from `sw`; A18: the NaN count goes through `_undefined`), except two that are equivalent and left out with the reason in the mutation list: A15 (the analysis reads a mismatched secondary model; a mismatch of any model refuses the analysis before the list of scored models is read) and A19 (H6_position consistent without a rival; an H6_position consistent on P6, zones within N, S and L, always contradicts the rival "no position effect", Z). A5 was equivalent as the code stood (a score of an unknown item was also counted as unexpected); the amendment counts the two apart, which makes A5 killable. Thirty-eight mutants of the amendment (T1 to T38: M and the two summary sentences, the risky rule and the labels, H2 on P3, the placebo pair and the named secondaries, Mistral's jobs, the integrity changes, H7_check's readouts and the knowledge split's known-minus-unknown) were added, and five existing mutants were updated where their text moved (untested read as consistent, negated built for every verb, N25, Q20, and the P5 take-from-target mutant, now anchored to its own function since `completes_task` repeats its last line). New tests (27; `tests/test_mech2.py` now has 104, all passing): discriminating contrasts need the rival to fail on the same contrast; consistent needs a risky prediction, with the untested reason and the labels; H2 allows P3 about zero; categories use Holm's p, not the raw p; P8 (and M and the placebo) read P7's items only; the integrity check refuses unknown items and unexpected conditions, counted apart; undefined scores counted, refused on primary arms, reported elsewhere (unit and end to end, status line); untested accounts are not read as a mixture; the joint position verdict needs both formats and reads the rival's own row; the first-token split is read against the two-candidate full L; every primary contrast's r is over aligned minus swapped; P5 against cancelled's lift reads P5's items; the two summary sentences and when they are withheld; M is half negated plus half affirmed minus swapped; the named secondaries in both formats against their predictions; secondary zones need an estimable r; Mistral's jobs (unit, runner and integrity); completing actions; the group difference and its r; the H7 readouts; the knowledge split's known-minus-unknown; the note's secondary prediction table equals `SECONDARY_PREDICTIONS`; H7_check stays out of the verdicts; the status line lists each account with its contrasts. Run on 2026-11-21 (`MUTANTS_MATCHING=mech2`): all 168 mech2 mutants killed, none timed out or failed to apply (the first run was stopped from outside after 137 mutants, all killed by a named test; the other 31 were run by name). A mutant counts as killed only when a named test fails: T4's first record named none (its pytest was probably ended by another process's `pkill`), so the summary test was made to check that case directly and T4 was rerun alone, killed by that test. Every one of the 168 has a named killing test below.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| A1 rival contradicted on any contrast counts | `test_discriminating_contrasts_are_those_where_a_rival_fails_on_the_same_contrast` |
| A2 category from raw p, not Holm | `test_categories_use_the_holm_p_not_the_raw_p` |
| A3 P8 on the whole sample | `test_p8_reads_p7_items_only` |
| A4 integrity ignores unexpected scores | `test_integrity_refuses_unknown_items_and_unexpected_conditions` |
| A5 integrity ignores unknown items | `test_integrity_refuses_unknown_items_and_unexpected_conditions` |
| A6 untested accounts not counted as fitting | `test_untested_accounts_are_not_read_as_a_mixture` |
| A7 rival reading taken from H6_position row | `test_joint_position_verdict_reads_both_formats_and_the_rival` |
| A8 first-token split against yes/no full L | `test_first_token_split_is_read_against_the_two_candidate_full_L` |
| A9 P1 reference is aligned minus mention_ignore | `test_every_primary_contrast_reads_r_over_the_aligned_minus_swapped_effect` |
| A10 rival P6 band includes N | `test_position_counts_only_for_the_position_account` |
| A11 joint position verdict from cand only | `test_joint_position_verdict_reads_both_formats_and_the_rival` |
| A12 contents-not-shown keeps shown-without | `test_contents_not_shown_reads_only_the_takes_whose_receptacle_was_not_seen` |
| A13 mixture readout drops P5 | `test_end_to_end_two_shards_then_analysis` |
| A14 secondary not-scored refuses | `test_end_to_end_two_shards_then_analysis` |
| A16 negated/affirmed receptacle without its number | `test_templates_read_as_written` |
| A17 desire_keep_sw lower-case sentence | `test_templates_read_as_written` |
| A18 undefined L never counted | `test_undefined_scores_count_and_refuse_on_primary_arms` |
| A20 old P2 secondary against can | `test_secondary_readouts_read_the_arms_they_name` |
| A21 P5 lift ratio on all placement items | `test_p5_lift_reads_p5_items_only` |
| A22 mixture text when any account is contradicted | `test_readout_verdicts_and_the_joint_rule` |
| T1 M is P7 | `test_mention_residual_is_half_negated_plus_half_affirmed_minus_swapped` |
| T2 M above the band in one format suffices | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T3 M read above zero, not above the band | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T4 salience sentence without the claims-only condition | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T5 claims-only allows other contrasts | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T6 claims sentence without H2 consistent | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T7 risky rule dropped | `test_consistent_needs_a_risky_prediction` |
| T8 risky read as excluding N | `test_position_counts_only_for_the_position_account` |
| T9 label hides the contrasts that allow zero | `test_consistent_needs_a_risky_prediction` |
| T10 untested reason from risky contrasts | `test_consistent_needs_a_risky_prediction` |
| T11 H2 P3 back to positive only | `test_the_prediction_table_in_the_note_is_the_code_s` |
| T12 placebo denies the action object | `test_templates_read_as_written` |
| T13 placebo article of the action object | `test_templates_read_as_written` |
| T14 placebo readout reversed | `test_named_secondaries_read_their_arms_in_both_formats` |
| T15 P8 net adds the placebo | `test_named_secondaries_read_their_arms_in_both_formats` |
| T16 S1 read in the two-candidate format only | `test_named_secondaries_read_their_arms_in_both_formats` |
| T17 Mistral on every item | `test_mistral_is_scored_on_the_sample_and_p6_arms_on_every_step0_item` |
| T18 Mistral without P6 arms outside the sample | `test_mistral_is_scored_on_the_sample_and_p6_arms_on_every_step0_item` |
| T19 P6 arms are the position arms only | `test_mistral_is_scored_on_the_sample_and_p6_arms_on_every_step0_item` |
| T20 integrity expects every job of the arms | `test_the_secondary_model_is_scored_and_reported_descriptively` |
| T21 analysis checks integrity without the sample | `test_the_secondary_model_is_scored_and_reported_descriptively` |
| T22 runner gives Mistral every job | `test_the_secondary_model_is_scored_and_reported_descriptively` |
| T23 undefined L on a primary arm does not refuse | `test_undefined_scores_count_and_refuse_on_primary_arms` |
| T24 undefined L on any arm counted as primary | `test_undefined_scores_count_and_refuse_on_primary_arms` |
| T25 unknown items counted as unexpected too | `test_integrity_refuses_unknown_items_and_unexpected_conditions` |
| T26 status line hides undefined scores | `test_status_line_reports_undefined_scores_of_secondary_arms` |
| T27 every move completes the task | `test_completing_actions_move_the_goal_object_into_a_target_type_receptacle` |
| T28 group difference reversed | `test_group_difference_is_a_difference_in_differences_with_r` |
| T29 group r over group a reference only | `test_group_difference_is_a_difference_in_differences_with_r` |
| T30 H7a groups swapped | `test_h7_readouts_split_p7_items_by_what_the_history_shows_and_actions_by_role` |
| T31 H7c against every item | `test_h7_readouts_split_p7_items_by_what_the_history_shows_and_actions_by_role` |
| T32 known minus unknown reversed | `test_knowledge_split_reports_known_minus_unknown` |
| T33 knowledge split without desire_check | `test_knowledge_split_reads_the_content_contrasts_only` |
| T34 secondary joint reading from one format | `test_secondary_zones_need_an_estimable_r` |
| T35 secondary zones ignore estimable | `test_secondary_zones_need_an_estimable_r` |
| T36 placebo predicted below zero under H2 | `test_named_secondaries_read_their_arms_in_both_formats` |
| T37 summary not given M | `test_mention_residual_is_half_negated_plus_half_affirmed_minus_swapped` |
| T38 M on every sample item | `test_p8_reads_p7_items_only` |
| N25 affirmed with "a" for every object | `test_templates_read_as_written` |
| negated built for every verb | `test_every_arm_builds_for_every_task_type_and_verb` |
| Q20 knowledge split on step-0 terms too | `test_knowledge_split_reads_the_content_contrasts_only` |
| P5 keeps takes from a target-type receptacle | `test_p5_leaves_out_takes_from_a_target_type_receptacle_and_contradicted_items` |
| untested read as consistent | `test_readout_verdicts_and_the_joint_rule` |

The gate was rerun on the real tokenizers after the amendment (ICE, `coc-cpu` partition, CPU job 7219082, the job's environment read-only, 2026-11-21): both candidates of all 1,403 items for each model and every yes/no prompt each model is scored on (20,720, 20,720 and 11,878, the placebo pair's 1,428 per primary model among them), no problem for any model.

**Review 4 and the adversarial check of M (2026-11-21, before any score).** The review's 20 mutants (R1 to R20 of review 4, not to be confused with review 1's R1 to R20; eleven had survived: R1, R2, R3, R6, R7, R9, R10, R13, R15, R16, R18) were added to `scripts/mutation_check.py`, adapted where the amended code moved their text, except two that are equivalent and left out with the reason given there: R10 (the risky rule reads `PREDICTIONS` only; `_risky` is only called for accounts, whose cells are `PREDICTIONS`' own) and R19 (adds a comment). Twelve mutants of the review-4 amendment (U1 to U12: M's halves, the no-residual sentence, the placebo sentence and the verdicts resting on P8, the placebo without the swapped-target items, the S1 line) and two of the adversarial check (W1, W2: the summary attributing the residual, or its absence, to salience) were added; eight round-3 mutants were adapted where the code moved (T2, T3, T4, T14, T15, T27, T37, T38). New tests (18 collected, `tests/test_mech2.py` now 122): one per reviewer survivor (R1, R2, R3, R6, R7, R9, R13, R15, R16, R18, listed in the table), the placebo sentence and its conditions and the unchanged verdicts, the no-residual sentence, M's halves, the placebo without the swapped-target items, the S1 status line, the placebo reaching the summary through `model_block`, and the adversarial regression tests: `test_pure_task_as_evidence_with_asymmetric_claims_is_never_read_as_salience` (four planted pure-task-as-evidence patterns with asymmetric claim responses) and `test_the_m_sentences_never_attribute_the_effect_to_salience`; the summary-sentence test pins the new wording. The end-to-end test now checks the status line's per-contrast pattern ("P1 … (CATEGORY; yes/no CATEGORY)") rather than the word "yes/no", which the S1 line also contains: with the word alone N22 (status line without the yes/no categories) was no longer caught by its own test and timed out on the full suite; rerun alone after the change, it is killed by that test. Run on 2026-11-22 (`MUTANTS_MATCHING=mech2`, four shards, after the change): all 200 mech2 mutants killed in one run, each by a named failing test, none timed out or failed to apply.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| R1 salience contradictions read from the two-candidate format | `test_mention_sentence_reads_h3_from_the_joint_rows_not_one_format` |
| R2 claims sentence from the two-candidate H2 verdict | `test_claims_sentence_needs_task_as_evidence_consistent_jointly` |
| R3 claims-only true when salience is not contradicted | `test_no_claim_sentences_when_salience_is_not_contradicted` |
| R4 claim contrasts include P3 | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| R5 claim contrasts drop P7 | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| R6 M above the band from its edge | `test_mention_band_edges` |
| R7 placebo on every sample item | `test_secondary_readouts_per_format_and_population` |
| R8 M over the affirmed lift | `test_mention_residual_is_half_negated_plus_half_affirmed_minus_swapped` |
| R9 M zones from the two-candidate format only | `test_mention_halves_and_per_format_zones` |
| R11 discriminating only against content rivals | `test_position_counts_only_for_the_position_account` |
| R12 Mistral outside the sample on the start arms only | `test_mistral_is_scored_on_the_sample_and_p6_arms_on_every_step0_item` |
| R13 placebo on takes of another object | `test_placebo_is_built_only_on_takes_of_the_goal_object` |
| R14 undefined L on yes/no primary arms not counted | `test_undefined_scores_count_and_refuse_on_primary_arms` |
| R15 P8 net read in the two-candidate format only | `test_secondary_readouts_per_format_and_population` |
| R16 H7a within groups reversed | `test_h7a_within_contents_or_open_state_keeps_the_sign` |
| R17 mention sentence needs H2 consistent too | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| R18 hash check skipped for the secondary model | `test_integrity_catches_a_stale_hash_of_the_secondary_model` |
| R20 estimable ignored in the interval tests | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| U1 affirmed half reads the denial | `test_mention_halves_and_per_format_zones` |
| U2 denied half reversed | `test_mention_halves_and_per_format_zones` |
| U3 absent M checks the upper edge only | `test_mention_band_edges` |
| U4 absent M without the claims-only condition | `test_mention_band_edges` |
| U5 placebo read below zero, not below the band | `test_placebo_below_the_band_names_the_verdicts_resting_on_p8` |
| U6 verdicts on P8 omit consistent accounts | `test_placebo_below_the_band_names_the_verdicts_resting_on_p8` |
| U7 placebo sentence names the verdicts on P5 | `test_placebo_below_the_band_names_the_verdicts_resting_on_p8` |
| U8 summary not given the placebo | `test_model_block_passes_the_placebo_and_p8_net_to_the_summary` |
| U9 placebo subset keeps only the left-out items | `test_placebo_also_read_without_swapped_targets_at_the_action_receptacle` |
| U10 swapped target read from the real task | `test_placebo_also_read_without_swapped_targets_at_the_action_receptacle` |
| U11 S1 line whatever the verdicts | `test_status_line_gives_s1_when_wishful_and_blending_both_stand` |
| U12 status line drops S1 | `test_status_line_gives_s1_when_wishful_and_blending_both_stand` |
| W1 residual sentence attributes the effect to salience | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| W2 absent residual read as no salience | `test_mention_band_edges` |
| T2 M above the band in one format suffices | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T3 M read above zero, not above the band | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T4 salience sentence without the claims-only condition | `test_summary_sentences_on_salience_contradicted_only_by_claims` |
| T14 placebo readout reversed | `test_named_secondaries_read_their_arms_in_both_formats` |
| T15 P8 net adds the placebo | `test_named_secondaries_read_their_arms_in_both_formats` |
| T27 every move completes the task | `test_completing_actions_move_the_goal_object_into_a_target_type_receptacle` |
| T37 summary not given M | `test_mention_residual_is_half_negated_plus_half_affirmed_minus_swapped` |
| T38 M on every sample item | `test_p8_reads_p7_items_only` |
| N22 status line without yes/no categories | `test_end_to_end_two_shards_then_analysis` |

**Review 5 (2026-11-22, before any score).** The review's surviving mutants X1 and X2 (a sentence attributing the residual to salience appended to the summary), X8 (the S1 line only when both wishful prediction and policy blending are consistent) and X9 (P8 net of the placebo quoted without an interval in both formats) were added to `scripts/mutation_check.py`, with two mutants of the new M-below-band sentence (Y1, Y2); U5 was anchored to the placebo's own comparison, which the new below-band test repeats. New tests: the whole summary text for M above, inside and below the band; the below-band sentence's conditions; the S1 line with one of the two accounts untested; P8 net quoted only with an interval in both formats. `tests/test_mech2.py`: 126 pass. Run on 2026-11-22 (these seven mutants by name): all killed by a named test; 206 mech2 mutants in all.

| Mutant | Killed by (`tests/test_mech2.py::`) |
| --- | --- |
| X1 attribution appended after the residual sentence | `test_the_whole_summary_is_exactly_as_written` |
| X2 claims sentence says M indicates salience | `test_the_whole_summary_is_exactly_as_written` |
| X8 S1 line only when both accounts are consistent | `test_s1_line_when_one_of_wishful_and_blending_is_only_untested` |
| X9 P8 net quoted without an interval in both formats | `test_p8_net_text_needs_an_interval_in_both_formats` |
| Y1 below-band sentence without the claims-only condition | `test_the_below_band_sentence_needs_both_formats_and_the_claims_only_condition` |
| Y2 below-band read below zero, not below the band | `test_the_below_band_sentence_needs_both_formats_and_the_claims_only_condition` |
| U5 placebo read below zero, not below the band (re-anchored) | `test_placebo_below_the_band_names_the_verdicts_resting_on_p8` |

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
