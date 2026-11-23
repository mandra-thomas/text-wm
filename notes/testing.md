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
- **Mutation testing, state on 2026-11-23** (this updates the bullet above, whose count of 189 was the list at commit e7ca63b and whose list of runners without mutants is out of date). Counted on `main` at 56c3619 by importing each runner's list:

  | Runner | Study | Mutants |
  |---|---|---|
  | `scripts/mutation_check.py` | Harness, pilots, RQ1 to RQ3, verification tools and the 2026-11-20 audit fixes (including one or two mutants each of `wf/rq2.py`, `wf/rq1x.py`, `wf/rq1v.py`, `wf/mech.py`, `wf/thinking.py`, `wf/rq2_shared.py` and `wf/audit.py`) | 208 |
  | `scripts/mutation_check.py` | Mechanism battery (`wf/mech2.py`; names start with `mech2`) | 206 |
  | `scripts/mutation_check.py` | Decision simulation Stage 1a (`wf/decisionsim.py`; `decisionsim`) | 163 |
  | `scripts/mutation_check.py` | Model families wave A and the size series (`wf/families.py`, `wf/sizes.py`, `wf/scoring/logodds.py`, `wf/tools/prefetch_models.py`; `families`) | 75 |
  | `scripts/mutation_check.py`, list in `scripts/mutants_decisionsim_agent.py` | Decision simulation Stage 1b (`wf/decisionsim_agent.py` and its precision script; `dsa`) | 175 |
  | `wf/textcraft/mutation_check.py` | TextCraft round 1 (the 193 run before the freeze, plus two tie-couple mutants added at 7495cb8 and killed by their test) | 195 |
  | `scripts/tau_mutation_check.py` | τ-bench retail | 155 |

  The main runner holds 827 mutants in all. `wf/rq1.py` and `wf/rq1d.py` still have no mutants of their own.

  **Which kill rule each runner applies.** On `main` all three runners count any non-zero pytest exit as a kill. `scripts/mutation_check.py` runs the test files that name the mutated module, then the whole suite if the mutant survived; it prints the first failed test but does not first check that the unmutated suite passes, and it records a mutant whose correct text is missing as "not applied". `wf/textcraft/mutation_check.py` and `scripts/tau_mutation_check.py` first require the unmutated tests to pass and log the first failing (or erroring) test of every kill; the τ-bench runner labels a kill with no test id "no test id (collection or import error)". The strict named-kill rule (a kill only when pytest exits 1 and names a failing test; any other non-zero exit is "broken", not a kill) is in the code of three runners on study branches, not on `main`: families wave B's `scripts/mutation_check.py` (from its first review, 2026-11-22), TextCraft round 2's `scripts/audit_textcraft_r2_mutants.py`, and, as an exit-code-1 rule without the named-test condition, the interpretability pilot's runner (from 2026-11-21). The newer studies' freeze tags (Stage 1b, Option C, the τ-bench authorization experiment, wave B, the interpretability pilot, TextCraft round 2) report every mutant killed by a named test; where the runner does not enforce that rule (Stage 1b and Option C on the main runner, the authorization experiment on the τ-bench runner), the claim rests on the killing test recorded per mutant, as in the Stage 1b table below. (Superseded for `scripts/mutation_check.py` by the next bullet, 2026-11-23.)
- **Mutation runner: strict kill rule (2026-11-23).** `scripts/mutation_check.py` now applies the strict rule by default. A mutant is **killed** only when pytest exits with code 1 and at least one named test FAILED. It is **error-only**, and not counted as killed, when pytest exits with any other non-zero code (a collection or import error, a usage or internal error, no tests collected) or when the only named problems are ERRORs in a fixture's setup or teardown; in that last case the run is repeated without `-x`, and a named failure further on still counts as a kill. A timeout stays inconclusive. This is families wave B's rule (its `scripts/mutation_check.py`, also on `textcraft-r3`) with one tightening: wave B also accepted a named `ERROR path::test` line as a kill. Before any mutant is judged, the unmutated copy must pass the same tests (the covering test files, and the whole suite before the first whole-suite rerun); otherwise the run stops. The old rule, any non-zero exit is a kill, is kept behind `--legacy-any-exit` only to reproduce logs written before 2026-11-23, and it skips the baseline check as they did. Other changes: `--results FILE.jsonl` appends one record per mutant as it finishes (outcome, killing test or error, exit code, test targets, the root's commit) and a rerun skips mutants already recorded under the same rule, so an interrupted run resumes; `--root CHECKOUT` and `--mutants-from FILE` run another checkout's mutants against that checkout's code and tests on a temporary copy, which is how frozen studies are rechecked without touching their tags; `--require-unique` applies the τ-bench runner's rule that the correct text must occur exactly once. The pytest subprocess runs with `PYTHONDONTWRITEBYTECODE=1`: a `.pyc` records its source's mtime in whole seconds and its size, so a same-length mutant written within a second of an earlier run of the same file could be read from the stale cache and look like a survivor (this can only hide kills, never create them; the runner's own test found it). `tests/test_mutation_check.py` runs the runner on a small repository of its own: a mutant that breaks the import is error-only (pytest exit 2), a mutant caught by a named test is killed, an untested change survives the covering tests and the whole suite, the legacy flag counts the import-breaking mutant as killed, resume, a failing baseline, setup errors, and loading another checkout's mutant list with its own sibling imports. `wf/textcraft/mutation_check.py` and `scripts/tau_mutation_check.py` on `main` still use the old rule.

  **Three frozen studies rechecked under the strict rule (2026-11-23).** Each freeze tag was checked out into its own temporary worktree, and the ported runner was run from outside it (`--root` on the worktree, `--mutants-from` on the tag's own mutant list), so every mutant was applied to a temporary copy of the code exactly as frozen. The worktrees were clean at the tagged commit before and after. First, the unmutated whole suite at each tag reproduced the count recorded at the freeze: Stage 1b 853 passed and 7 skipped, Option C 953 and 8, the τ-bench authorization experiment 1000 and 7.

  | Study (tag) | Mutants | Recorded at the freeze | Strict rule, 2026-11-23 |
  |---|---|---|---|
  | Stage 1b (`decisionsim-stage1b-frozen-2026-11-22`, list `scripts/mutants_decisionsim_agent.py`) | 175 | all killed, a killing test per mutant in the table below | 175 killed by a named failing test; none error-only, survived, timed out or not applied |
  | Option C (`optionc-design-frozen-2026-11-22`, list `scripts/mutants_optionc.py`) | 131 | all killed "under the strict rule" (exit 1 and a named failing test), first failing test per mutant in its table | 131 killed by a named failing test; none otherwise |
  | τ-bench authorization (`tau-auth-design-frozen-2026-11-22`, the `tau-auth` mutants in `scripts/mutation_check.py`, with `--require-unique` as its runner did) | 87 | all killed, each kill's test logged (log not committed) | 87 killed by a named failing test; none otherwise |

  All 393 runs exited with code 1 and named a FAILED test. No run had an ERROR line, so wave B's rule and this one agree on every mutant. For Stage 1b and Option C, the first failing test is the one in the freeze table for every mutant; the only differences are in how parametrised test ids are written. Option C's runner at its tag counts any non-zero exit. The note's "strict rule" was therefore applied outside the runner's code; its definition is wave B's, and the recheck confirms its tally. The τ-bench authorization count is 87 (58 of the build, review 1's 12 and 17 for the amendments), not 75. One Stage 1b mutant ("dsa analysis reads another label run's scores") changes two occurrences of its text, as the main runner's replace-all did when it was first run. Records: one JSON line per mutant (tag, commit, outcome, killing test, exit code, test files) in [mutation_strict_recheck_2026-11-23.jsonl](mutation_strict_recheck_2026-11-23.jsonl).
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
| fix1 attribution ignores the half (adapted in review 3) | `test_option_c_categories` (swapped part below half) |
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
| review2 N10 attribution ignores the sign of aligned minus base (adapted in review 3) | `test_option_c_categories` (aligned minus base negative) |
| review2 N11 pays-for-itself strict (adapted in review 3) | `test_option_c_notes_a_comparison_verifier_that_does_not_pay_for_itself` |
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

### Review 3 (2026-11-21, before any score)

New tests in `tests/test_decisionsim.py` (100 collected, from 96; 72 test functions, from 68):
- the attribution check at exactly half (20, 8 and 14 of 676 states), plus a sweep of every exactly-half case up to 80 failing states (one state less fails); the old float comparison failed 387 of those 1,560 cases;
- the pays-for-itself flag reads its own comparison arm and stays false when the reference points are missing;
- a helpful fallback menu (a negative rejected-both net) does not count against;
- rule 2's headroom counts a failing proposal the history does not determine;
- the step split reads the task-blind comparison, not the neutral one;
- the prompt-format flag waits for the deciding verdict;
- each verdict reports its states and incomplete states.

`scripts/mutation_check.py` now holds 163 decision-simulation mutants:
- 6 of the reviewer's survivors: S1 to S4, S6 and S10. Two others are left out as equivalent: S9, the same as the retired fix1 "attribution ignores the sign", and S11, since the task-blind query has no task line left to strip.
- 3 for the fixes (T1, T3, T4).
- 3 adapted to the changed code: fix1 "attribution ignores the half", review2 N10 and N11.

All 163 are caught (`MUTANTS_MATCHING=decisionsim`, six shards, 2026-11-21).

| Mutant | First test that fails |
|---|---|
| review3 S1 pays flag read from the deciding base for every verdict | `test_option_c_flags_read_their_own_comparison_and_stay_false_without_information` |
| review3 S2 worth-doing-inside flag set when pays is unknown | `test_option_c_flags_read_their_own_comparison_and_stay_false_without_information` |
| review3 S3 fallback share read as an absolute value | `test_a_fallback_menu_that_helps_does_not_count_against` |
| review3 S4 headroom counts only determined failing proposals | `test_rule2_headroom_and_net_counts_in_the_contrast` |
| review3 S6 step split against neutral | `test_rule2_by_step_splits_step_zero_from_later_steps` |
| review3 S10 format sensitivity ignores a waiting deciding verdict | `test_prompt_format_sensitivity_waits_for_the_deciding_verdict` |
| review3 T1 half compared without tolerance | `test_attribution_at_exactly_half_passes_whatever_the_floating_point` |
| review3 T3 verdict does not report its states | `test_option_c_on_a_real_report` |
| review3 T4 verdict does not report incomplete states | `test_option_c_on_a_real_report` |

## Model families, wave A (2026-11-20)

`tests/test_families.py` covers the family runs: golden renders of one real item from each pinned chat template, the tokenization gate on stand-in tokenizers built to fail in one item or one prompt (a late success's candidate, one item's swapped yes/no prompt, a template that writes "Yes" as another token than the scorer reads), the gate's scope in the shard runner (every item, on a resume too, with the scorer's template settings), the analysis (flip definitions by name, the yes/no arms in the standardised effect, records filtered by model and by population), the pre-data primary metrics (the item share toward Yes and the goal-to-evidence ratio R, including their invariance to rescaled logits), the competence gate at each of its bounds, the TOST equivalence of R on paired items, the Holm-corrected primary family, and the Mistral placement run. Fix round 2 (third amendment) adds: the R comparison's fallback to the paired difference when the ratio is not estimable (review 2's synthetic models with R near zero, slightly positive and negative), the 2% cap on dropped draws, the item-share comparison (difference by game sign flips, TOST within ±0.05 as the larger one-sided p), the 90% and 95% interval rules for both comparisons on hand-set bootstrap draws, pairing across noisy games, the primary family padded with p = 1 for missing or NaN tests, the planned comparisons including Gemma 2 against Mistral, the two-sided noise floor, ties at log-odds zero, NaN scores, and game-clustered p-values on games that move together. With transformers, torch and the tokenizers cached (`WF_TEST_HF_CACHE`), the real templates run too. `scripts/mutation_check.py` holds 75 families mutants (`MUTANTS_MATCHING=families`), including the ten a review found surviving, seventeen for the primary metrics, gate, equivalence and family, review 2's M1 to M14 (two each for M6 and M7, R and share) and nine for fix round 2. All 75 are caught (2026-11-20). Two survived the first run and were killed by added tests: the TOST p taken from the smaller side (the test's draws had been symmetric) and R kept when the evidence effect is negative (the test had used an evidence effect of exactly zero). The refactored gate was also run against the earlier one on every real item with the pinned tokenizers, with identical results (`notes/families_design.md`).




## Stage 1b: the agent's own ranked candidates (2026-11-20; rebuilt after review 1 and amended after review 2, 2026-11-21)

`tests/test_decisionsim_agent.py` (93 tests) covers `wf/decisionsim_agent.py`. On synthetic data: the candidate lists (rank 1 is the recorded greedy action even when never sampled, its sampled variants merged into it, the rest by count with the seeded hash for ties and independent of reply order, format errors counted and never turned into `look`, the cut at `k_max`, normalisation and its per-list count, the action key shared by both lists), the sampling settings of both lists, the greedy diagnostic and its comparison of normalised actions, a pinned seed value, bookkeeping (identical replies, truncation, greedy agreement and the 90% drift rule at its boundary), the seed check (primary settings, sequences not multisets, the same-seed repeat) and the smoke gate (format errors, replay mismatches, a non-degenerate seed-check state, the required truncation and agreement readouts); the verifier prompts (for each model the aware, blind, swapped and neutral prompts differ only in the query's task line, examples keep theirs, the aware, swapped and neutral prompts equal the earlier `yn_*` prompts), the blind deletion's guards, the blind-only rule at single-candidate states; list views (each list's order over shared labels, runnable candidates only, the sampled-twice and first-occurrence filters); the selection rules; regime, encounter, strict encounter, reachability, step bands (encounters, not reachable ones); the futility rule after review 2 (games with an encounter state that also lists a working candidate, the 6- and 7-game boundary, review 2's 8-game counterexample that is significant with no reachable encounter); harm and help with the identity and the kinds of harm counted on harm states only; the sign-flip floor; leave-one-game-out; the decomposition and the attribution rule; whole-game bootstrap and permutation; Holm; the verdict reading the Holm p; the verdict boundaries (a lower bound of exactly zero; a regime share of exactly one half carries no suffix, below it the "not evidence for the main finding" suffix); a failed gate entering Holm with p = 1; the overall and conditional blocks each reading their own p, decomposition and states; the AUC and the competence thresholds; the model report and every sensitivity, the temperature-1 readouts reading the temperature-1 lists; shards with different settings refused; the run-order check (nothing written, another label run, an analysis with models, an unfinished one, a logged override); progress events; the swap object, which excludes every candidate of both lists. On the simulator, with the mock agent and scorer: everything listed for review 1, plus scoring refusing to start before the model-free analysis of the same labels, the logged override, the model-free analysis reporting the temperature-1 encounters from the temperature-1 lists, the labelling budget counting the next state, and the request settings and normalisation counts recorded by a sampling shard.


**After review 3 (2026-11-22).** Eight more tests (101 in all) cover the additions: one changed state per path (a abstained, b abstained, a accepted what b rejected, a rejected what b chose) counted by outcome cell and step band, with unchanged states in no path; each decomposition part with its own paths; the hybrid verifiers (their arms, choices, fallbacks and differences against blind); acceptance by rank 1 against alternatives, outcome and step band, the aware-minus-swapped and swapped-minus-blind shifts and the step-0 headline; inert readouts by step band (step 1 counts as later); a "not attributable" toward-failure verdict with mostly non-regime harm getting the suffix; and the run-order check comparing the label hashes the model-free analysis read with those the scorer reads, also end to end (labels edited after the analysis are refused).

`scripts/mutants_decisionsim_agent.py` holds 175 Stage 1b mutants (`MUTANTS_MATCHING="dsa "`): the review-1 list, review 2's 42 (adapted), the mutants of the code added after review 2, and 19 mutants of the code added after review 3 (two earlier mutants re-adapted to the changed `choose` and `inert`). All 175 are caught; the table below is from the per-mutant run on the final code (2026-11-22). Four review-3 mutants first survived: three were killed by sharpened tests (step 1 in the later band; the swapped-minus-blind paths; the swapped-minus-blind shift), and one ("acceptance rank 1 by its greedy flag") is equivalent, because in a usable state the first runnable candidate is always the agent's own rank-1 action, so it was left out. R30 (review 1) is also equivalent. The table gives, for each, the first test in `tests/test_decisionsim_agent.py` that fails with the mutant in place (run with `-x`).

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
| paths: abstention read from the other side | `test_change_paths_by_path_outcome_and_step_band` |
| paths: rank order reversed | `test_change_paths_by_path_outcome_and_step_band` |
| paths: abstention ignored | `test_change_paths_by_path_outcome_and_step_band` |
| step band includes step 1 | `test_inert_split_by_step_band` |
| paths: harm and help swapped | `test_change_paths_by_path_outcome_and_step_band` |
| paths not in the contrast | `test_change_paths_by_path_outcome_and_step_band` |
| decomposition part paths reversed | `test_decomposition_parts_get_their_own_paths` |
| decomposition swapped-blind paths from aware | `test_decomposition_parts_get_their_own_paths` |
| hybrid arms reversed | `test_hybrid_verifiers` |
| hybrid judges every candidate by its rank-1 arm | `test_hybrid_verifiers` |
| hybrid difference against aware | `test_hybrid_verifiers` |
| acceptance ignores the step band | `test_acceptance_by_position_and_step_band_and_every_decomposition_shift` |
| headline reads later steps | `test_acceptance_by_position_and_step_band_and_every_decomposition_shift` |
| aware-swapped shift reversed | `test_acceptance_by_position_and_step_band_and_every_decomposition_shift` |
| swapped-blind shift from aware | `test_acceptance_by_position_and_step_band_and_every_decomposition_shift` |
| inert bands not split | `test_inert_split_by_step_band` |
| run order skips the label hashes | `test_run_order_check_compares_the_label_hashes` |
| scorer passes no label files | `test_scoring_refuses_labels_changed_after_the_model_free_analysis` |
| inert rank-1 share counts fallbacks only | `test_model_report_parts_add_up_and_readouts` |
