# Deviations after filing (filed 2026-11-05)

Changes made on `fix/audit-bugs` after the filing marker. No preregistered threshold was changed. Report these in the final writeup.

## Still open

- **Decision 6, remaining parts.** Agreed, not built: a generator quota of 150 to 200 failures decided by observed contents or holding state, H1 run on those items (with the pooled number also reported, and the change of carrying items disclosed), and false-success versus false-failure comparisons matched on the deciding fact. H1 also needs the full-state condition for the few-shot models, which the code does not have yet. None of this blocks the pilots.
- **Agent stuck loops (observed 2026-11-17, nothing changed).** With greedy decoding, Qwen3-8B sometimes repeats an action that does nothing ("go to drawer 1" while already there) until the step limit. A replay of `train/791` (trace job 7136513) showed the prompts are built as intended: the oracle prediction for the repeated action correctly said "Nothing happens", and the agent took it anyway. In the first 85 Pilot B headroom episodes, loops of 5 or more steps appear in 4 of 22 episodes with no prediction, 4 of 21 with oracle predictions kept in the history, 0 of 21 with oracle predictions dropped, and 0 of 21 with the validity verdict only. Success is 10 of 21 in both retention arms. A loop guard or sampling would change the agent, so that is a team decision, not a fix.
- **Llama scoring stop rule fired (Pilot A, 2026-11-17).** For Llama-3.1-8B-Instruct the sign of L agrees with its written prediction on 741 of 1,138 determined failures (65%, below the 90% line of decision 13); Qwen3-8B agrees on 91%. Llama's ΔL is 1.36 nats (CI 1.08 to 1.68, p = 0.001). Its first-token mass is lower (median 0.86 against >0.99), and some of its failure sentences also start with "You", so ΔL_first is a weaker check for Llama. **Decided 2026-11-17:** Llama stays co-primary in the preregistered analysis, with the failed check disclosed as a measurement issue. Its written predictions, re-read with the fixed parser, also shift toward success under the aligned goal (+17.5 points, CI 12.0 to 23.6), so they are reported as the behavioural check on its ΔL; that reading waits for the parser audit.
- **Parser drops free-form failure predictions (Pilot A, 2026-11-17).** Written predictions such as "You can't take the mug 1 from cabinet 3 because it is empty." or "There is no tomato 1 in the environment." are classed `unparseable` and left out of A. That removes failure predictions selectively: 364 of Llama's 475 unparseable replies and 99 of Qwen3's 151 are worded as failures. Counting them as failures (a rough regex, not the fix) moves Llama's A from 0.99 to about 0.87 and Qwen3's from 0.22 to about 0.20; Word2World has none. Fix before the confirmatory RQ1 run and check it against the 100-item parser hand check. **Fixed in code, awaiting the audit:** model text now goes through `parse_prediction` (environment observations keep `parse_observation`). Failure wording (can't, cannot, there is no, you don't see/have, you have no, already holding, "you try to … but") is `noop_paraphrase` and counts as a failure; success described in other words ("You place …", "You wash …") is `success_paraphrase` and counts as a success; anything else stays `unparseable`. `pilotA_reparse` recomputes every Ŷ_gen measure from Pilot A's saved text, before and after, without running a model. The blind audit (`python -m wf.tools.parser_audit`) samples 100 few-shot predictions, over-sampling the hard cases, and hides the parser's label and the true outcome. It passes if the parser matches the reader on at least 95% of rows the reader labels S or F (the parser's "?" counts as a miss) and no failure-worded reply is read as a success. These criteria were fixed before the audit was read. The parser is frozen once it passes, before any confirmatory data are scored.
- **Injected errors per episode are not matched in the retained cells (Pilot B mini-run, 2026-11-17).** The proposal matches the two error types on errors per episode. Calibration (`pilotB_eligibility`) set one rate per arm from oracle episodes, and the discarded cells came out close (2.9 false successes, 3.2 false failures per episode), but the retained cells did not (4.5 against 2.1). Retained false-success episodes offer more eligible steps, most likely because the agent repeats failing actions. **Decided 2026-11-17:** cap and yoke, see "Decided 2026-11-17, after the pilots".
- **ScienceWorld fallback.** No longer triggered: `PILOT_RULE_HEADROOM` is advisory (see the headroom decision), and RQ3 stays on ALFWorld.

## Decided 2026-11-17, after the pilots (before any confirmatory data)

These were decided after Pilot A and Pilot B results existed, on train games only. No confirmatory (`valid_unseen`) data had been collected or scored. The preregistered primary test (ΔL per co-primary model, game bootstrap, game-level permutation, Holm) is unchanged.

**RQ3 size.** RQ3 runs on the 134 solvable `valid_unseen` games instead of 274 games. This is the proposal's own contingency, applied as written: "if RQ3 lags at the milestone it shrinks to the 134 unseen games". RQ3 lagged at the milestone (2026-10-29) and still lags. All cells use the same games, so every comparison is paired by game.

**RQ3 error matching: cap and yoke.** The proposal matches the two error types "on errors per episode and step type". The pilot calibration (one rate per arm) matched the counts when predictions were dropped but not when they were kept (4.5 false successes against 2.1 false failures per episode), because false successes make the agent retry failing actions, and each retry was another chance to inject. A fixed count per episode would not work either: under perfect foresight with predictions dropped, the median episode offers one chance for a false success and 38% offer none.

- **False-success arm.** A chance is a step where the agent proposes an action that will fail and the environment has a success text for it. At each chance, inject with probability 0.75, decided by a seed per (game, schedule), until 3 have been delivered. The cap of 3 is `CORRUPTIONS_PER_EPISODE_TARGET`, set in the pilot code's first commit (`e2c1f1f`, 2026-10-28); it is not in the proposal, which specifies three error *schedules*, not three errors. Checked against the pilot's delivered distribution: among false-success episodes that received any error, 67% (dropped) and 79% (kept) received 3 or more, and the most was 17. So the cap binds for most episodes and stops a stuck agent from attracting unlimited errors.
- **Schedules.** Three seeds per game, shared across both retention modes. They differ in which chances are injected. With probability 1 the three schedules would be identical, because runs repeat exactly with the prefix cache off.
- **False-failure arm, yoked.** For the same game, schedule and retention mode, the false-failure episode gets exactly d errors, where d is the number its false-success partner received. The i-th false failure goes at the first working proposal with the same verb, at or after the step of the i-th false success. If none exists, the first working proposal of any verb at or after that step is used, and the fallback is logged.
- **Records.** Every episode logs planned and delivered errors and each fallback. Each cell reports the number of episodes that fell short. Cost per delivered error and the delivered count are secondary analyses.

**RQ3 primary analysis set.** The primary set is the (game, schedule) units whose false-success episode received at least one error in both retention modes. With d = 0, an episode is identical to perfect foresight (and its yoked partner gets none), so membership depends only on behaviour before any injection, which makes it a pre-treatment selection. The interaction I is computed on this set with a game-clustered CI. All units are reported as a secondary analysis. From the pilot's perfect-foresight episodes, about 48% of games qualify at probability 0.75 (52% at 1.0, 41% at 0.5), which is about 64 of 134 games. The pilot's per-game SD of I in qualifying games is 5.05 steps (13 games). Counting one schedule per game, the SE is about 0.63 steps and the 80%-power minimum detectable effect is about 1.8 steps (two-sided, 0.05); the three schedules can only lower this. The pilot's estimate in that set is +3.0 steps (H3 predicts a negative I).

**RQ3 pilot interaction, recorded before confirmatory data.** The mini-run gave I = +1.7 steps (95% CI −0.3 to +3.6), the opposite sign to H3, on 30 games. H3 is not revised. Two notes for the write-up:

1. In the kept cells the false-success arm received more than twice as many errors as the false-failure arm and still did as well, so the imbalance probably biased the pilot toward H3, making its positive I conservative. This is not clean, because episodes going badly also offer more chances to inject, so error count and difficulty are entangled.
2. **Exploratory explanation, written now so it cannot be fitted afterwards.** In the kept format, the real observation sits directly after the wrong prediction in the history ("Expected outcome: You pick up the mug 1." then "Observation: Nothing happens."). The agent can see the world model was wrong and may discount it. H3 assumed kept predictions make false beliefs persist; in this format the contradiction is kept too.

**Rule R margin.** `PILOT_RULE_RETAINED_FORMAT` did not fire (retention moved the false-success arm by 1.3 steps against a margin of 1.0, with an SE of about 1). The rule is followed, so the trace-line kept format stays. It was close, and the format should not be described as validated.

**Labels for conditions added to RQ1 and RQ2.**

- **Full-state condition for the few-shot models.** Pre-specified in the proposal ("each item is scored history-only (primary) and full-state", and H1's predictions), not in the filed preregistration. It had not been built when the pilots ran. Hidden state means Word2World's starting hidden state for the game (its trained input); the current state follows from it plus the history.
- **Object-mention control.** Exploratory, added after the pilot RQ2 results. It asks whether the aligned goal's effect is goal-directed or comes from the action's object being named. ALFWorld has no task that names an object without requiring the agent to pick it up, so the control keeps the swapped goal and adds one neutral sentence naming the action's object ("There is a mug somewhere in this room."). Comparing it with the plain swapped goal isolates mention; comparing the aligned goal with it isolates goal relevance. Proposed design, open to team veto before scoring.

**Confirmatory RQ1 and RQ2 item collection.** 134 `valid_unseen` games with the Pilot A collector, extended so that each item saves the game's hidden state (the Word2World context text and its SHA-256), the scene's object locations and receptacles, and its `determination_reason`. Every failure decided by observed contents, open state or holding state is generated at every visited step, with no subsampling (decision 6). Other alternatives keep Pilot A's sampling (every second step plus every invalid proposal).

## Decided 2026-11-17

Made before any real Pilot A or Pilot B result existed. Pilot A on the old code produced no results (its folders hold only config files).

| Decision | Outcome |
| --- | --- |
| 4. Goal-channel rule | Advisory. Evaluated as filed in Δp and also in nats (fires when every scored model's ΔL upper CI bound is below 0.5). The preregistered RQ2 test runs and is reported whichever way it falls. Δp stays, because the preregistration requires it. |
| 5. RQ1 measure | A uses Ŷ_gen only, the parsed free generation, as the proposal says. The first-token log-odds, logit(q_first), is reported alongside ΔL as a secondary measure. |
| 6. `context_determined` | Kept, with disclosure. Every Pilot A result is also split by the fact that decides each failure (`by_reason`, `false_success_by_reason`). See "Still open" for the rest. |
| 10. Pilot rule provenance | "Pre-specified, not preregistered". See below. |
| 13. Scoring gate | Continue with disclosure. Tier 0 on `d4fa809` failed at 87/100 and stays FAIL on record. The stop rule, written before resubmitting: Pilot A stops if sign(L) and Ŷ_gen agree on fewer than 90% of its determined true failures, for any few-shot model. Pilot B never waited on this gate in code; the queue dependency is removed. |
| 14. vLLM prefix cache | Off (`--no-enable-prefix-caching`) for every agent run. Scoring runs in HF transformers, not vLLM, and already repeats exactly. |
| Pilot A size | 50 games, up from 30, with chained resume jobs so a time limit loses nothing. |
| Llama-3.1-8B-Instruct | Scored in Pilot A itself (replaces decision 7's later run). The model is gated; if its weights cannot be loaded, Pilot A records it as not scored and stays unfinished, and a resume scores only Llama. |
| Headroom rule (H) | Advisory, decided while `pilotB_headroom` was running (27 of 50 games in, success about flat). Reported in success points and in steps to completion; never stops the job; no 100-game recheck. The gate is the proposal's check, `pilotB_consumption`. See below. |

**Decision 5 in detail.** Write q for the model's probability that the action succeeds in any wording. Since a failure always reads "Nothing happens.", L ≈ logit(q) − c, where c = −log P(o+ | success) ≥ 0 is a wording penalty that grows with every detail of the success text the model cannot guess. So the sign of L reads as "predicts success" only when c ≈ 0, which fails for `go to` texts that list unseen contents: on Tier 0's 100 train items, all 12 successes where the sign of L disagreed with the model's own text were such `go to` items. On context-determined items, the population A uses, A came out 0.24 from Ŷ_gen and 0.22 from the sign of L, but with only 9 determined successes. Each score now also saves the first token of each text, its decoded form (to confirm what follows the chat template), their combined probability `first_mass`, and `c_hat = L_first − L`; Pilot A reports the mean change in c_hat between goals, since ΔL = Δlogit(q_first) − Δc.

**Headroom rule in detail.** The proposal's pilot check is that "perfect foresight changes the agent's actions; if not, RQ3 moves to ScienceWorld". The pilot code (`e2c1f1f`, 2026-10-28) instead required perfect foresight to raise ALFWorld success by at least 5 points over no foresight. That requirement is in neither the proposal nor the preregistration, and it conflicts with the proposal's premise that foresight often fails to help, even when correct, because agents misuse it [2]. We now follow the filed check. `pilotB_consumption` gates the later stages: the share of steps where the perfect prediction changes the agent's action must reach `CONSUMPTION_MIN` (0.10). With 27 games of the headroom run done, it was 23% (oracle predictions dropped from the history after each step), 34% (oracle predictions kept in the history) and 19% (valid / not valid verdict only). The success gain stays in every report as an advisory number, together with the paired gain in steps to completion (`oracle_steps_gain`). The 100-game recheck is skipped because the main RQ3 runs measure headroom on 274 games. The headroom episodes were collected at `28a396d`; later code accepts them through `accept_commits` in the consumption and eligibility configs. The retained-format parser fix (explicit "Action:" line) is in `28a396d`, so it covers every headroom episode and the retention mini-run.

**Decision 10 in detail.** All three rules and their thresholds (`FUTILITY_EST_PP`, `FUTILITY_UPPER_PP`, `HEADROOM_MIN_PP`, `RETENTION_MARGIN`) are in commit `e2c1f1f`, made 2026-10-28 12:48:36 and pushed to GitHub at 12:48:43 by the local reflog (confirm on GitHub's activity page). The first ICE job of any kind started at 13:02 that day. `docs/milestone_checkin.tex` names all three rules without thresholds. Suggested wording: "G, H and R were fixed in the project repository (commit `e2c1f1f`, pushed 2026-10-28) before any pilot data existed. They were not part of the filed preregistration. H and R are feasibility gates that select designs; G governs emphasis only, and the preregistered RQ2 test is reported regardless." Rule R reads the retention effect within the false-success (favorable) arm, in steps averaged per game; if it fires, the retained format changes to state persistence, and RQ3 is not dropped. Pilot games come from `train`, so none enter the confirmatory tests on the validation splits.

**Agent action parser.** The parser took the last line of the agent's reply whenever no line began with "Action:". In the retained condition the history shows "Proposed action:" and "Expected outcome:" lines, so a reply copying that pattern could turn "Expected outcome: ..." into the action and count as an invalid proposal. This fits, but has not been shown to cause, the doubled invalid-proposal rate under retained perfect foresight (0.37 vs 0.19 on 20 games). The parser now requires an explicit "Action:" line. A reply without one is a format error: the agent looks around (`look`), the step is flagged (`proposal_format_error`, `executed_format_error`, `under_true_format_error`), it never counts as an invalid proposal or as decision-critical, and Pilot B reports the format-error rate per cell. Every step now saves the agent's raw text (`proposal_raw`, `executed_raw`). Fixed before Pilot B, because rule R reads the retained arm.

**Tool caches.** The ICE home quota (30 GB) was full, mostly a 20 GB uv download cache written by this project's setup on 2026-10-28; it was cleared after confirming no environment links into it. Jobs now keep every tool cache (vLLM, triton, torch inductor, uv, pip, XDG) under the checkout on `/storage`; Hugging Face already used `/storage`.

**Placebo spread, for sample size.** The Qwen3 placebo spread of 4.56 nats is the per-item SD of L under one irrelevant goal minus L under another, over 137 true failures from 20 train games, unclustered. The per-item values were not saved, so ρ cannot be estimated from it. Pilot A estimates ρ on the paired ΔL (`icc_L`) and reports the items needed for 80% power at true effects of 0.5, 0.75 and 1.0 (`n_for_80pct_power`, two-sided α = 0.005, 134 games). Plan on 0.75: the preregistered criterion needs the point estimate itself to reach 0.5, so a true effect of exactly 0.5 meets it about half the time at any n.

## Decided 2026-11-06

| Decision | Outcome |
| --- | --- |
| 1. Mini-run | Held until the calibration fixes below are in |
| 2. Pilot B memory | 128 GB, because the server may reach the old 64 GB limit |
| 3. vLLM prefix cache | A test with the cache off ran alongside; see decision 14 on 2026-11-17 |
| 7. Llama-3.1-8B-Instruct | Superseded on 2026-11-17: Llama is scored in Pilot A itself |
| 8. `STEP_CAP` | Stays 30. A cap of 20 would have lost 2 of the 6 real Qwen3 wins seen, at steps 24 and 25 |
| 9. Admissible list | Stays for now; revisit with the invalid rate under oracle runs |
| 11. `ELIGIBLE_SET_TARGET` | No fallback. If no size up to 5,000 reaches SE ≤ 1.5 it reports null, and the team decides then |
| 12. Pilot A | Held until the fixes below are in, so nothing is scored twice |

## Read before analysing or writing up: changes on 2026-11-06

All of these came from four independent code reviews and were made before any real Pilot A or Pilot B result existed.

**Swapped goals are now achievable.** The protocol asks for a swapped goal of the same task type, but the code took any object in the room, so 53% of main-test failures in a sample of 204 real `valid_unseen` items had an impossible swap, such as `put a hot pot in cabinet`. RQ2 compared an achievable goal against an impossible one about half the time. A swap must now pass the game's own flags in that game's `initial_state.pddl`: pickupable, heatable, coolable or cleanable as the task needs, accepted by the target receptacle (`canContain`), two instances for pick-two tasks, and not the target itself. The rule accepted 717 of 721 real goals in their own games; the other 4 are pick-two tasks whose hidden-state text lists one instance. When no object qualifies, the item has no swap and leaves Pilot A (`swap_feasible` false, reason `no_swap_object`). No substitute is used.

**Closed arrivals are no longer "determined".** `The drawer 2 is closed.` counted as decidable from history even when the history never said so. These were 22 of 28 determined successes in the same 204 items, so A's false-failure rate was mostly this one case. A `go to` success is now determined only if the history already shows the receptacle closed. Other success labels are unchanged, under decision 6.

**A leaves out excluded successes.** `shared_location` successes had stayed in A while the matching failures were removed, and unparseable successes were counted too. On planted data this moved A from −0.01 to −0.30. Both are now out, exclusions are counted on both sides, and the sensitivity check puts flagged items back on both sides. A's real-data size is unknown.

**Few-shot examples are real strings.** The three examples were hand-written and matched no real text: none of 295 real `go to` observations stop at the name, and all 64 real `open` observations say `The X is open.`. That also broke the protocol's rule that observations come from the environment. They are now logged server text from `t0_smoke` on train games 700, 270 and 2332, still one success and two failures. This edits a frozen prompt. The effect on ΔL should be small, because both conditions share the prompt, but it has not been measured. `t0_scoring_agreement` must be rerun.

**RQ3 error rates are matched between arms.** Rates were calibrated on `none` and oracle episodes pooled, counted invalid proposals that cannot be corrupted (19% of real invalid proposals were `done`), and let each arm aim for 3 errors alone. On simulated data with no true effect, unequal error counts gave a false H and an I whose CI excluded zero. Calibration now uses oracle episodes only, counts only steps with a success twin, and gives both arms one target: the lower of 3 and what each can reach. `corruption_mix` shows which action types get corrupted. Real invalid proposals are 55% `go to`, which may rarely mislead the agent, so read that mix before interpreting H.

**Objects can be in two receptacles.** ALFWorld listed at least one object in two places in 366 of 832 sampled games, and picking one up clears only one listing. `Scene.locations` now holds every listing. No outcome label was wrong because of this; the old scan picked one arbitrarily.

**Success text can no longer contradict the item.** A twin from reset could say `In it, you see a toiletpaper 1.` while the agent held that object. `open` and `go to` items whose receptacle the agent has changed now have no twin (`twin_contradicts_state`). This applies to corruptions too.

**The Pilot A gate counts the main test.** ≥500 items and ≥80 states now refer to goal-agnostic, goal-naming, non-shared items. `eligible_failure_determined` still shows the broader count.

**Headroom can stop a job.** It reports `REPORTED`, never PASS. A borderline or fired rule exits 3, which stops the job, and a resume does not skip it. `pace/pilotB_recheck.sbatch` runs the recheck and then the later stages, which refuse to start until the recheck clears the rule.

**Agent repeatability.** Agreement was 0.850 and 0.757 in two Tier 0 runs, against a 0.95 target. Prompts were byte-identical at every divergence, so this is not our code. The protocol's advice to add seeding cannot work, because the agent picks greedily at temperature 0, and that wording is corrected. In a 3-way comparison, first and second runs of a prompt differed on 7 of 194 prompts and later runs on 0 of 83, which points at vLLM's prefix cache without proving it (p = 0.08). All 10 paired games in the first run ended the same way. Treat this as noise in RQ3, not bias.

**Things the write-up should say.** Foresight had never run with the real agent model before Pilot B. Long success texts can make L look pessimistic, which matters for A more than ΔL. The fine-tuned models learned a goal signal only through a `[SUCCESS]` marker we do not score, so a null result for them may be partly structural. 23 of 117 "goal-agnostic" failures show the goal object in an observation, because the rule reads actions only, as filed.

## Read before analysing or writing up: shared locations

**Decided 2026-11-06, before any real Pilot A scores existed.** Only item counts and label checks had been seen, so the choice could not have been shaped by results. Mention this rule in the methods section and report both versions of the result.

**What we found.** ALFWorld places some receptacles at the same spot. From `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. `context_determined` assumed each receptacle had its own spot, so "the agent is elsewhere" counted as proof of failure. Nearly all failure labels rest on that reasoning.

| Evidence | Result |
| --- | --- |
| Game files with at least one shared spot (sample of 400 train games) | 296 of 400 (74%) |
| Receptacles at a shared spot in that sample | 1,786 of 9,527 (19%) |
| "Certain to fail" claims that were wrong on the real server, before the fix | 6 of 1,053 (0.6%) over 30 games, all fitting this pattern; 3 traced to it directly |
| Failure labels resting on location | 480 of 506 after 15 games |
| Detection test: `go to` is not offered for a receptacle at the same spot | Matched a direct test on all 2,164 ordered pairs in 8 games (20 shared) |
| Scene-scan misses in those 8 games explained by shared spots | 13 of 13 |

**What we changed.**

- The scene scan records which receptacles share a spot, and it now looks inside all of them. Before this, it skipped them and missed 34 of 873 objects (3.9%) over 30 games.
- Every item has a `shared_location` field: true when the action's receptacle shares a spot with the one the agent is at.
- Flagged items are still built and scored, but they are left out of the main test: the RQ2 goal effect, RQ1's A, and the 500-item and 80-state gate.
- `pilotA_score` also reports each model with those items put back, as a sensitivity check.

**Where to find it.**

| What | Where |
| --- | --- |
| Main results | `reports.<model>.goal_agnostic`, `A`, `A_ci` in `runs/pilotA_score/metrics.json` |
| Sensitivity check | `reports.<model>.with_shared_location` in the same file, also summarised in `STATUS.md` |
| Item counts | `n_main_test` and `n_shared_location` in `pilotA_transitions`; `exclusions` lists `shared_location` |
| Per-item flag | `shared_location` in `transitions.jsonl` |

**How to report it.**

- State the main result, then the sensitivity result. If they disagree, say so and treat the finding as depending on this choice.
- RQ1's A is more exposed than RQ2's ΔL. ΔL compares one item under two goals, so a doubtful item affects both sides. A counts a success prediction on such an item as a false success.
- The fix is partial, and the write-up should say so. The history rarely shows which receptacles share a spot (`look` can), so "determined" still means determined for someone who knows the layout. Flagged items were identified using environment knowledge that the model never sees. See decision 6 above.
- Confirmatory items need the same rule. It is not yet applied anywhere beyond Pilot A, because the confirmatory build does not exist.

## Read before analysing or writing up: `valid_unseen` prompts

**Found 2026-11-06 by a crash in `t0_ckpt_accuracy`, and fixed before any confirmatory work.** World-model prompts were looked up by position in the file, but each row's `id` is the game's global index and the rows have gaps. On `train` that made no difference: all 2,420 rows line up. On `valid_unseen` only 3 of 150 rows line up, so a fine-tuned model saw another game's hidden state, with the correct task line written over it. Out-of-range games crashed, and the rest gave no error.

- **Not affected:** Pilot A and Pilot B, which use `train`.
- **Would have been affected:** RQ2 and RQ1 on `valid_unseen`, for the fine-tuned models.
- **Invalid:** every `t0_ckpt_accuracy` figure from before this fix, including the one in `runs/t0_ckpt_accuracy` and in the milestone documents.
- **Now:** a prompt is found by `id`, and a run stops if the prompt is missing or does not contain the game's task line. All 34 `valid_unseen` games from the failed run match with the fix.

## Protocol changes

| Area | Before | After |
| --- | --- | --- |
| Permutation test | Single items flip | Whole games flip, as the prereg requires (§4.9 said "within pairs") |
| Primary Pilot A population | All eligible failures | Only actions naming the goal object, since the aligned goal must be one the action would advance. Others stay scored. |
| Items scored | Failures only | Also true successes under `aligned`, needed for A |
| `absent` etc. on fine-tuned models | Invented prompt | Trained chat format with hidden state, so these are full-state, as the proposal says |
| `committed` / `candidate` wording | After "Next observation:" | Before it, and in the system message for fine-tuned models |
| Scene objects | Fixed list | Read from the environment by visiting every receptacle on a private env id |
| Success twins | Earlier prefixes plus four fixed setups | Planned from reset using that scene |
| Corruption draws | Hash included the arm | Shared across arms (§4.8) |
| `t0_scoring_agreement`, `t0_ckpt_accuracy` | Fixed or first-step items | Items from seeded walks; no LLM needed, so they stay in Phase B |
| `t0_replay` | Scripted actions | 20 logged states × 3 actions, plus replay-vs-live agreement |
| `t0_ckpt_accuracy` match | Exact or same class | Exact only |
| Parser classes | Four plus unparseable | Adds `info` for inventory, look, help, examine |
| Agent prompt | Prediction with no action named | `Proposed action:` above `Expected outcome:`, explained in the system prompt |
| Pilot A main test | All eligible failures naming the goal object | Also leaves out `shared_location` items, which stay scored for a sensitivity check (see above) |
| Scene scan | Not applicable | Restarts the game before ALFWorld's 50-step limit (`ENV_STEP_LIMIT`), which also caps any future `STEP_CAP` at 50 |
| ALFWorld server | Fixed port, no health check | Job-chosen port and a health check, since jobs can share a node |
| Headroom "materially negative" | Any negative difference | Whole CI below zero |
| Saving results | Not specified | Added to the protocol's rules: every long job saves each finished unit and resumes without rework or mixed commits |
| `STATUS.md` | Reported-only runs said PASS | PASS means a target was met; others say `REPORTED`, and results under a target say `BELOW TARGET` |
| `pilotB` job | 6 h, 64 GB | 14 h, 128 GB: about 10–12 h estimated from real episode times |
| `pilotA_transitions` | 120 games, 4 h | 30 games, 6 h: 30 real games gave 818 eligible items against a 500 target, and fewer games halve the time and server memory |

The text of `PILOT_PROTOCOL.md` was brought up to date with these changes on 2026-11-06. No preregistered value changed.

## Bug fixes (no design change intended)

Foresight on a private env id; task and history in every condition; invalid actions executed rather than rewritten; no-twin steps not counted as corruptions; goal parser; swapped task line changes only the object; `context_determined` from history facts; `history_goal_agnostic` reads actions only; `delta_p` filtering; A computable; room description kept in history; no truncation of the admissible list; trace `targets_goal_object`; 100 distinct agreement items; `t0_parser` reads the dataset and fails loudly; receptacle lookup no longer matches `to` inside `potato` or `tomato`; world-model prompts looked up by each row's `id`, and checked against the game's task line, instead of by file position; `use` and `slice` have their own action type; contents are filed under the receptacle the line names; a move no longer creates contents for an unseen receptacle; non-finite scores stop the run; the score cache includes the prompt; a missing swap raises an error; `recovery` needs a positive oracle gain; the retention rule compares the same games; fine-tuned generation stops at every end-of-turn token and allows 256 tokens; every episode, score, generation and built game saved as it finishes, with `--resume` on all long runs; a resume skips runs that finished at the same commit with unchanged inputs, and refuses changed commits, uncommitted code, changed settings or changed input files. Scores carry an item hash. A run refuses a dependency that did not finish, unless its config names the commit under `accept_commits`. Mock runs stay out of real folders, mock records are marked and refused by real runs, and `runs/*/STATUS.md`, `REPORT.md` and `PILOT_REPORT.md` are no longer tracked. The report refuses missing, stale or mock runs. Verification jobs exit nonzero on failure.

## Tier 0 results invalidated

Job 6685206 ran the old code, so the Tier 0 claims in `docs/milestone_checkin.tex` need replacing. Every `t0_ckpt_accuracy` figure before 2026-11-06 is invalid. Tier 0 must be rerun on the commit that holds the changes above: `t0_scoring_agreement` uses the new few-shot examples, and items now use the new swaps and labels. `runs/` in older checkouts still holds old mock output, which real runs refuse.
