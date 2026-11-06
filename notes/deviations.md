# Deviations after filing (filed 2026-11-05)

Changes made on `fix/audit-bugs` after the filing marker. No preregistered threshold was changed. Report these in the final writeup.

## Needs a team decision

1. **Metric for the goal-channel futility rule.** The prereg PDF defines the headline as ΔL in nats. `PILOT_RULE_GOAL_CHANNEL` in `PILOT_PROTOCOL.md` uses Δp in percentage points, and Δp compresses when L is far from 0: at L = −6, ΔL = 1.0 gives Δp = 0.42pp. The rule still runs as filed. `pilotA_score` now also reports ΔL and lists any model where the rule fired despite ΔL ≥ 0.5 (`rule_conflicts_with_delta_L`).
2. **Admissible list.** The agent still sees it and is told to use it, matching upstream. Invalid proposals may therefore be rare, which limits the favorable arm. `pilotB_eligibility` now warns if an arm cannot reach its target.
3. **`STEP_CAP`.** Still 30. The protocol asks to confirm it against the stack, whose `run_vllm.sh` default is 20.
4. **Pilot rule provenance.** The protocol calls the pilot rules verbatim from the preregistration, but the one-page PDF does not contain them.
5. **How strict `context_determined` should be.** The label now requires a fact in the history, but the agent's location is always known, so `take X from Y` while standing elsewhere counts as determined. Nearly every failure passes on that basis. If the team wants only failures decided by observed contents or holding state, tighten `_failure_determined` in `wf/data/transitions.py`.
6. **`ELIGIBLE_SET_TARGET` can be null.** It searches 1,000 to 5,000 items for SE ≤ 1.5 over 134 `valid_unseen` games. If none qualifies, it reports null and does not fall back to 1,000.

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
| Scene scan | Not applicable | Restarts the game before ALFWorld's 50-step limit (`ENV_STEP_LIMIT`), which also caps any future `STEP_CAP` at 50 |
| ALFWorld server | Fixed port, no health check | Job-chosen port and a health check, since jobs can share a node |
| Headroom "materially negative" | Any negative difference | Whole CI below zero |
| `pilotA_transitions` | 120 games, 4 h | 60 games, 6 h: yield per game is higher, cost per game unmeasured on ICE |

## Bug fixes (no design change intended)

Foresight on a private env id; task and history in every condition; invalid actions executed rather than rewritten; no-twin steps not counted as corruptions; goal parser; swapped task line changes only the object; `context_determined` from history facts; `history_goal_agnostic` reads actions only; `delta_p` filtering; A computable; room description kept in history; no truncation of the admissible list; trace `targets_goal_object`; 100 distinct agreement items; `t0_parser` reads the dataset and fails loudly; every episode, score, generation and built game saved as it finishes, with `--resume` on all long runs; a resume skips runs that passed at the same commit and refuses to mix commits.

## Tier 0 results invalidated

Job 6685206 ran the old code, so `runs/t0_*` and the Tier 0 claims in `docs/milestone_checkin.tex` need the rerun. `runs/pilot*` still holds old mock output until real runs replace it.
