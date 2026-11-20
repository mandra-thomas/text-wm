# RQ2 confirmatory result (2026-11-18)

Scoring: run `rq2_score`, 4 shards at ddeabc2. Shards 0–2 ran on A40s and shard 3 on an L40S; each item's aligned and swapped prompts were scored in the same job on the same GPU. Analysis: run `rq2_analysis` at ddeabc2. Full numbers are in `rq2_confirmatory_metrics.json` (a copy of the run's metrics.json).

Items: 4,666 main-test items from 1,033 states in 125 of the 134 unseen games. All passed full replay verification. The preregistered minimum was met, and no primary scores are missing. Every item is a failure that the context determines: the true next observation is "Nothing happens." How the items were built, and which were left out, is in "Methods" below.

A note on the measures. L is the log-odds of the action's success text against "Nothing happens.", and s = σ(L) is the two-candidate score. s is **not** the model's probability that the action succeeds: it compares one particular success wording with the failure reply, so it is lowered by every detail of the success text the model cannot guess (`deviations.md`, "Decision 5 in detail"). Columns below headed "Δp" or "mean P(success)" are means of s, kept under their logged names.

## Preregistered verdict

| model | ΔL (nats) | 95% CI | Holm p | verdict |
|---|---|---|---|---|
| Qwen3-8B, few-shot | 6.12 | 5.48 to 6.75 | 0.002 | supported (strong) |
| Llama-3.1-8B, few-shot | 1.65 | 1.51 to 1.78 | 0.002 | supported (strong) |

The p-values are at the floor set by 1,000 game-level permutations (raw p = 1/1001, doubled by Holm).

## Secondary measures

| model | Δp, i.e. Δs (pp) | ΔL first token (nats) | mean s = σ(L), aligned / swapped | written success shift (pp) |
|---|---|---|---|---|
| Qwen3-8B | 5.9 (5.1 to 6.8) | 2.52 (2.26 to 2.79) | 0.076 / 0.017 | 8.8 (7.5 to 10.3) |
| Llama-3.1-8B | 16.4 (14.6 to 18.3) | 0.68 (0.62 to 0.76) | 0.407 / 0.243 | 27.4 (24.6 to 30.1) |

By history length, Qwen's effect is concentrated where the agent has not acted yet. With history length 0 (514 items), ΔL is 11.0 nats and Δp is 42 pp. At lengths 1–3, 4–7 and 8+, ΔL is 5.2 to 6.0 nats but Δp is only about 1.4 pp. Llama's effect is flat across lengths: ΔL is 1.5 to 1.7 nats and Δp is 13 to 18 pp.

## Object-mention 2×2 (exploratory)

The 2×2 covers 1,076 items whose action is move, heat, cool or clean; take items cannot be built. It has four task lines. Aligned is the true goal, which the action advances. Swapped is the true goal with another object. Mention names the action's object, but in a goal the action does not advance: another transformation, or another destination. Fourth applies the same change to the swapped line.

The two parts sum exactly to aligned minus swapped on these items. The goal-directed part is (aligned − mention) − (swapped − fourth). The mention part is mention − fourth.

| model | goal-directed (nats) | mention (nats) |
|---|---|---|
| Qwen3-8B | 0.72 (0.05 to 1.38), p = 0.04 | 6.93 (6.04 to 7.77), p = 0.001 |
| Llama-3.1-8B | −0.18 (−0.31 to −0.05), p = 0.01 | 1.04 (0.87 to 1.23), p = 0.001 |

On these items, almost all of the aligned-versus-swapped difference comes from naming the action's object in the task line. Very little comes from the action advancing the goal.

**Superseded reading (2026-11-19).** The 2×2 says the difference needs the action's object to be *named* in the task line, not that it comes from the name alone. When the task names the object only by a description ("thick drinking vessel with a handle…"), 61% (Qwen) and 92% (Llama) of the effect remains, so the task acts through the object's meaning, not its name. See `exploratory_results.md`, section 1.

## Methods

### What "ALFWorld" means here

The environment is ALFWorld served by the AgentGym server that ships with Word2World (checkout `e8dc240`), not the original ALFWorld release. It uses Word2World's renamed-action data: `put` is called `move`, and there is a `help` action. The server runs alfworld 0.3.3 and TextWorld 1.7.0 (versions in `reproducing.md`). AgentGym's `valid_unseen` list has 195 games, of which 134 are solvable; the rest crash on reset, and `wf/env/server.py` leaves them out (`PILOT_PROTOCOL.md`, §5.1). ALFWorld itself ends every episode at 50 steps (`ENV_STEP_LIMIT`). To any command that is not admissible at that moment, the server replies "Nothing happens.", so every failing action in this study has that same reply.

### How the histories were produced

Each game was played once from reset by a Qwen3-8B ReAct agent: greedy decoding (temperature 0), thinking off, at most 256 new tokens per reply, served by vLLM with the prefix cache off. The agent saw the real task, its history and the list of admissible actions, and no predictions (`ForesightConfig("none")`). A reply without an "Action:" line executes `look`.

The collector stopped each episode after **12 steps** (`step_cap: 12` in `wf/config/rq12_transitions*.yaml`), so item histories hold at most 11 actions. That cap came from Pilot A's collection config, where it has been since the first commit; it is not in the preregistration, and the agent cap of 30 (`STEP_CAP`) applies only to the RQ3 and Pilot B agents. See `deviations.md`, 2026-11-20.

Each state is rebuilt by replaying the saved action prefix from reset on a separate environment id, so the histories can be replayed exactly from the saved prefixes, but not regenerated: the agent's greedy rollouts are not bit-identical across GPUs.

### How alternative actions were generated

At each state along the episode, two generators propose actions, and each is run on a fresh replay of the state to get its true reply (`wf/data/transitions.py`).

- **The broad generator** runs at every second step (steps 0, 2, 4, …) and at any step where the agent's own action failed. It proposes the agent's own action; taking the goal object from the receptacle that holds it at reset and from two receptacles that do not; moving it to a receptacle of the task's target type; heating, cooling and cleaning it with the fixed tools `microwave 1`, `fridge 1` and `sinkbasin 1` when they are in the room; reopening the most recently opened receptacle; wrong-object and wrong-receptacle versions of the agent's action; and three random admissible actions as likely successes.
- **The exhaustive generator** runs at every step where the agent's location is known, and covers every failure decided by contents, open state or holding at the agent's own receptacle: taking each goal-object instance that the agent can see is not there (or from a closed receptacle), opening what is open, closing what is closed, taking anything while holding something else, and placing, heating, cooling or cleaning the goal object without holding it.

Random choices use one random stream per state, seeded with `20260924|<game>|<state>`. The mix of failure reasons, and so the pooled ΔL, depends on these generators. One consequence: at step 0, "take from a receptacle the agent has not gone to" includes the receptacle that really holds the object.

### When a failure counts as determined

A failure is *determined* when the history alone proves it (`_failure_determined`). The history is read for the agent's location ("You arrive at …"), what it holds ("You pick up …", cleared by "You move …"), which receptacles are open or closed, and each receptacle's contents from the latest observation that lists them. The rules:

- `take X from R` fails if the agent is known to hold something (one object in hand), is known to be somewhere other than R, R is known to be closed, or R's contents were seen and X was not among them;
- `move X to R` and `heat`/`cool`/`clean X with R` fail if the agent is known not to hold X, or is known to be away from R;
- `open R` fails if the agent is away or R is known to be open; `close R` if the agent is away or R is known to be closed;
- `go to R` fails if the agent is already at R, or R is not in the room's opening description.

A success is determined only if its reply reveals nothing new (no "you see"), and a `go to` only if the history already showed the receptacle closed. The determination rules were checked on every collected item against the real server: no item they call a sure failure succeeded (outside shared spots, below). The checks are listed in `testing.md`.

### Shared spots

ALFWorld puts some receptacles at the same spot: from `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. There, "the agent is elsewhere" proves nothing, so an item is flagged `shared_location` when the action's receptacle shares a spot with the one the agent is at (from the scene scan, which the model never sees). Flagged items are left out of RQ2's population and RQ1's items. The rule was decided on 2026-11-06, before any scores (`deviations.md`, "Read before analysing or writing up: shared locations"), and the log promised the result both with and without these items. The confirmatory runs did not score them; a scoring run for them has now been added (`deviations.md`, 2026-11-20).

### How success texts ("twins") were made

The success candidate for a failing action is the real server's reply to the same action in a state where it works, never a hand-written template (`wf/data/candidates.py`). A short plan is built from reset using the scene scan (go to the object's receptacle, open it if closed, take the object, carry it to the action's receptacle), replayed, and the action executed. The reply is cached per (game, action). Limits:

- the twin comes from a state reached from reset, not from the item's own state, so it shows the world as the plan left it;
- `take`, `move`, and `heat`/`cool`/`clean` replies do not depend on that state; `open` and `go to` replies list contents and open state, so those twins are dropped (`twin_contradicts_state`) once the agent has changed that receptacle;
- an action with no setup plan (`slice`, or opening something open from the start) has no twin and leaves the study (`no_success_twin`);
- a plan that wins the game ends it before the action can run, so that item gets no twin either; for tasks that only require putting one goal object at the target, this removes `take <goal> from <target>` when the goal is not already at the target, because the plan to set it up (carry the goal there and put it down) wins the game. How often this happens is not yet counted.

Full verification regenerated every twin exactly; that shows the twins are deterministic, not that they fit each item's state.

### How swapped goals were chosen

The swapped task line replaces the first occurrence of the goal object's name in the real task line, so the template is unchanged. The swap object is drawn uniformly, from the same per-state random stream, among the room's object types that (`wf/data/goals.py`, `pick_swap`):

- are not the goal type, are not named in the action, and are not acted on (take, move, heat, cool, clean, examine) in the history;
- have no instance at the action's receptacle at reset;
- could do the task by the game's own PDDL flags: pickupable, heatable, coolable or cleanable as the task needs, accepted by the target receptacle, two instances for pick-two tasks, and not the target itself.

The draw is per item, so items at one state can have different swaps. The swap object may appear in the history's observations. Items with no qualifying object have no swap and are dropped (`no_swap_object`; nine whole games had none, which is why 125 of 134 games contribute).

### The population

RQ2's preregistered population is every determined failure whose action names the goal object, whose history does not act on the goal or swap object (`history_goal_agnostic`), that has a swap and a success twin, and that is not at a shared spot. All such items were scored.

### Exclusion flow

**Pending (run `audit_analysis`).** A table of how the 12,987 collected items reduce to the 4,666 scored ones, stage by stage: not a failure, not determined, no success twin, twin contradicts the state, no swap object, shared spot, action not on the goal object, history acts on a goal object.

### Shared-spot sensitivity

**Pending (shared-spot scoring run, then `audit_analysis`).** ΔL with the shared-spot items put back, for both models, next to the result above.
