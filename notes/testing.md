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
