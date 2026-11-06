# Deviations after filing (filed 2026-11-05)

Changes made on `fix/audit-bugs` after the filing marker. No preregistered threshold was changed. Report these in the final writeup.

## Still open

Numbers match the team's decision list. Settle 4, 5 and 6 before anyone reads Pilot A results.

- **Decision 4.** **Metric for the goal-channel futility rule.** The prereg PDF defines the headline as ΔL in nats. `PILOT_RULE_GOAL_CHANNEL` in `PILOT_PROTOCOL.md` uses Δp in percentage points, and Δp compresses when L is far from 0: at L = −6, ΔL = 1.0 gives Δp = 0.42pp. The rule still runs as filed. `pilotA_score` now also reports ΔL and lists any model where the rule fired despite ΔL ≥ 0.5 (`rule_conflicts_with_delta_L`).
- **Decision 5. How A is calculated.** The proposal has the model write its prediction for RQ1 and notes that text length cancels for ΔL. The code uses the sign of L, where length does not cancel, so long success texts can push A negative. A stays sign(L) until this is decided.
- **Decision 6.** **How strict `context_determined` should be.** The label now requires a fact in the history, but the agent's location is always known, so `take X from Y` while standing elsewhere counts as determined. Nearly every failure passes on that basis, and shared locations show that basis is not airtight. If the team wants only failures decided by observed contents or holding state, tighten `_failure_determined` in `wf/data/transitions.py`. Successes need no supporting fact, except `go to` arrivals (see below), so the two sides of A are not held to one standard.
- **Decision 10.** **Pilot rule provenance.** The protocol calls the pilot rules verbatim from the preregistration, but the one-page PDF does not contain them.

## Decided 2026-11-06

| Decision | Outcome |
| --- | --- |
| 1. Mini-run | Held until the calibration fixes below are in |
| 2. Pilot B memory | 128 GB, because the server may reach the old 64 GB limit |
| 3. vLLM prefix cache | A test with the cache off runs alongside the pilots; result pending |
| 7. Llama-3.1-8B-Instruct | A small scoring run on Pilot A's saved items, after Pilot A. Not built yet |
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
