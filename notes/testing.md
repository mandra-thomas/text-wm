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
