# Deviations after filing (filed 2026-11-05)

Changes made on `fix/audit-bugs` after the filing marker. No preregistered threshold was changed. Report these in the final writeup.

## Needs a team decision

1. **Metric for the goal-channel futility rule.** The prereg PDF defines the headline as ΔL in nats. `PILOT_RULE_GOAL_CHANNEL` in `PILOT_PROTOCOL.md` uses Δp in percentage points, and Δp compresses when L is far from 0: at L = −6, ΔL = 1.0 gives Δp = 0.42pp. The rule still runs as filed. `pilotA_score` now also reports ΔL and lists any model where the rule fired despite ΔL ≥ 0.5 (`rule_conflicts_with_delta_L`).
2. **Admissible list.** The agent still sees it and is told to use it, matching upstream. Invalid proposals may therefore be rare, which limits the favorable arm. `pilotB_eligibility` now warns if an arm cannot reach its target.
3. **`STEP_CAP`.** Still 30. The protocol asks to confirm it against the stack, whose `run_vllm.sh` default is 20.
4. **Pilot rule provenance.** The protocol calls the pilot rules verbatim from the preregistration, but the one-page PDF does not contain them.
5. **How strict `context_determined` should be.** The label now requires a fact in the history, but the agent's location is always known, so `take X from Y` while standing elsewhere counts as determined. Nearly every failure passes on that basis, and shared locations show that basis is not airtight. If the team wants only failures decided by observed contents or holding state, tighten `_failure_determined` in `wf/data/transitions.py`.
6. **`ELIGIBLE_SET_TARGET` can be null.** It searches 1,000 to 5,000 items for SE ≤ 1.5 over 134 `valid_unseen` games. If none qualifies, it reports null and does not fall back to 1,000.

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
- The fix is partial, and the write-up should say so. The history never shows which receptacles share a spot, so "determined" still means determined for someone who knows the layout. Flagged items were identified using environment knowledge that the model never sees. See decision 5 above.
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
| `pilotA_transitions` | 120 games, 4 h | 30 games, 6 h: 30 real games gave 818 eligible items against a 500 target, and fewer games halve the time and server memory |

The text of `PILOT_PROTOCOL.md` was brought up to date with these changes on 2026-11-06. No preregistered value changed.

## Bug fixes (no design change intended)

Foresight on a private env id; task and history in every condition; invalid actions executed rather than rewritten; no-twin steps not counted as corruptions; goal parser; swapped task line changes only the object; `context_determined` from history facts; `history_goal_agnostic` reads actions only; `delta_p` filtering; A computable; room description kept in history; no truncation of the admissible list; trace `targets_goal_object`; 100 distinct agreement items; `t0_parser` reads the dataset and fails loudly; receptacle lookup no longer matches `to` inside `potato` or `tomato`; world-model prompts looked up by each row's `id`, and checked against the game's task line, instead of by file position; every episode, score, generation and built game saved as it finishes, with `--resume` on all long runs; a resume skips runs that passed at the same commit and refuses to mix commits.

## Tier 0 results invalidated

Job 6685206 ran the old code, so `runs/t0_*` and the Tier 0 claims in `docs/milestone_checkin.tex` need the rerun. `runs/pilot*` still holds old mock output until real runs replace it.
