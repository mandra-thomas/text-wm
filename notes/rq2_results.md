# RQ2 confirmatory result (2026-11-18)

*Exploratory follow-up 2026-11-22: a forensic audit of the stored scores ([qwen_effect_audit_forensic.md](qwen_effect_audit_forensic.md)) found no pipeline fault and leaves this result unchanged. It finds that 59% of Qwen3-8B's 6.12 nats (3.60) is the success sentence's tail, that two-candidate nats are not comparable across models, and that, divided by the SD of each model's item-level shifts, Qwen3-8B's effect (1.02) is close to Llama-3.1-8B's (0.85). The confirmatory text below is unchanged.*

*Exploratory follow-up 2026-11-23: an independent pipeline audit ([qwen_effect_audit_pipeline.md](qwen_effect_audit_pipeline.md), sections 3 and 7) reproduces the stored scores exactly on the same GPU type, so no bug produced the number. About half of Qwen3-8B's effect in nats depends on the prompt format: with thinking off, Qwen3's chat template inserts an empty `<think></think>` block before the answer, and rescored without that block on all 4,666 items the effect is 3.31 nats (2.95 to 3.64) instead of 6.12, a ratio of 0.54 (0.51 to 0.57). The render without the block (the template with thinking on) is off the model's usual path, but a plain-text prompt with no chat template, where the two answers keep about 90% of the next-token probability, gives the same value on a 300-item subsample (3.26 against 3.28). The standardized effect (1.02 with the block, 1.05 without) and the share of items moved toward success (87.9% and 88.7%) do not change: the block stretches the scale of Qwen's scores, not its sensitivity to the goal. The verdict and every number below are unchanged.*

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
| Qwen3-8B | 5.9 (5.1 to 6.8) | 2.52 (2.26 to 2.79) | 0.075 / 0.017 | 8.8 (7.5 to 10.3) |
| Llama-3.1-8B | 16.4 (14.6 to 18.3) | 0.68 (0.62 to 0.76) | 0.408 / 0.244 | 27.4 (24.6 to 30.1) |

The mean-s column is not in `rq2_confirmatory_metrics.json`, which records only the differences. It comes from the secondary scoring of the same 4,666 items, run `rq1_score` (`rq1_v2_metrics.json`, `per_model.<model>.rq2_secondary.mean_p_success.aligned` and `.swapped`). That run scored the items in different jobs and on different GPUs, so its difference (5.8 and 16.4 points) is close to, but not exactly, the confirmatory Δp. The values quoted here before 2026-11-20 (0.076 / 0.017 and 0.407 / 0.243) had no source in any metrics file.

By history length, Qwen's effect is concentrated where the agent has not acted yet. With history length 0 (514 items), ΔL is 11.0 nats and Δp is 42 pp. At lengths 1–3, 4–7 and 8+, ΔL is 5.2 to 6.0 nats but Δp is only about 1.4 pp. Llama's effect is flat across lengths: ΔL is 1.5 to 1.7 nats and Δp is 13 to 18 pp.

*(Updated 2026-11-23, exploratory follow-up: Qwen's step-0 concentration depends on the empty `<think></think>` block that the chat template places before the answer with thinking off. Rescored without the block on the same items, scorer and GPU type, the 514 step-0 items give 2.73 nats (2.42 to 3.04) instead of 10.99, and lengths 1–3, 4–7 and 8+ give 3.59, 3.27 and 3.29. Without the block, step 0 is slightly below the later steps rather than far above them, and its standardized effect falls from 1.45 to 0.89. The render without the block is off the model's usual path; on the 22 step-0 items of the earlier 200-item audit, plain text gives the same 2.5 nats as that render. The stored values in this paragraph are unchanged. Source: `qwen_effect_audit_pipeline.md` §4 and §7.2 A, metrics `qwen_effect_audit_followup_metrics.json`.)*

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

The environment is ALFWorld served by the AgentGym server that ships with Word2World (checkout `e8dc240`), not the original ALFWorld release. Its game data came through Word2World's download script but are ALFWorld's own files (byte-identical to ALFWorld's GitHub release files; checked 2026-11-25), with the action wording ALFWorld introduced in version 0.4.0 (pull request #94): `put` is called `move`, and there is a `help` action. The server runs alfworld 0.3.3 and TextWorld 1.7.0 (versions in `reproducing.md`). AgentGym's `valid_unseen` list has 195 games, of which 134 are solvable; the rest crash on reset, and `wf/env/server.py` leaves them out (`PILOT_PROTOCOL.md`, §5.1). ALFWorld itself ends every episode at 50 steps (`ENV_STEP_LIMIT`). To any command that is not admissible at that moment, the server replies "Nothing happens.", so every failing action in this study has that same reply.

### How the histories were produced

Each game was played once from reset by a Qwen3-8B ReAct agent: greedy decoding (temperature 0), thinking off, at most 256 new tokens per reply, served by vLLM with the prefix cache off. The agent saw the real task, its history and the list of admissible actions, and no predictions (`ForesightConfig("none")`). A reply without an "Action:" line executes `look`.

The collector stopped each episode after **12 steps** (`step_cap: 12` in `wf/config/rq12_transitions*.yaml`), so item histories hold at most 11 actions. That cap came from Pilot A's collection config, where it has been since the first commit; it is not in the preregistration, and the agent cap of 30 (`STEP_CAP`) applies only to the RQ3 and Pilot B agents. See `deviations.md`, 2026-11-20.

Each state is rebuilt by replaying the saved action prefix from reset on a separate environment id, so the histories can be replayed exactly from the saved prefixes, but not regenerated: the agent's greedy rollouts are not bit-identical across GPUs.

### How alternative actions were generated

At each state along the episode, two generators propose actions, and each is run on a fresh replay of the state to get its true reply (`wf/data/transitions.py`).

- **The broad generator** runs at every second step (steps 0, 2, 4, …) and at any step where the agent's own action failed. It proposes the agent's own action; taking the goal object from the receptacle that holds it at reset and from two receptacles that do not; moving it to a receptacle of the task's target type; heating, cooling and cleaning it with the fixed tools `microwave 1`, `fridge 1` and `sinkbasin 1` when they are in the room; reopening the most recently opened receptacle; wrong-object and wrong-receptacle versions of the agent's action; and three random admissible actions as likely successes. Because of this schedule, a state at an odd step gets the broad candidate set only when the agent's own action there failed. This does not affect item-level analyses such as RQ2, but state-level rates must use even steps (`exploratory_results.md`, section 7).
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

ALFWorld puts some receptacles at the same spot: from `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. There, "the agent is elsewhere" proves nothing, so an item is flagged `shared_location` when the action's receptacle shares a spot with the one the agent is at (from the scene scan, which the model never sees). Flagged items are left out of RQ2's population and RQ1's items. The rule was decided on 2026-11-06, before any scores (`deviations.md`, "Read before analysing or writing up: shared locations"), and the log promised the result both with and without these items. The confirmatory runs did not score them. The check run on 2026-11-20 found that no item was excluded by this rule alone, so the result with them put back is the same result ("Shared-spot sensitivity" below).

### How success texts ("twins") were made

The success candidate for a failing action is the real server's reply to the same action in a state where it works, never a hand-written template (`wf/data/candidates.py`). A short plan is built from reset using the scene scan (go to the object's receptacle, open it if closed, take the object, carry it to the action's receptacle), replayed, and the action executed. The reply is cached per (game, action). Limits:

- the twin comes from a state reached from reset, not from the item's own state, so it shows the world as the plan left it;
- `take`, `move`, and `heat`/`cool`/`clean` replies do not depend on that state; `open` and `go to` replies list contents and open state, so those twins are dropped (`twin_contradicts_state`) once the agent has changed that receptacle;
- an action with no setup plan (`slice`, or opening something open from the start) has no twin and leaves the study (`no_success_twin`);
- a plan that wins the game ends it before the action can run, so that item gets no twin either; for tasks that only require putting one goal object at the target, this removes `take <goal> from <target>` when the goal is not already at the target, because the plan to set it up (carry the goal there and put it down) wins the game. Counted on 2026-11-20 (`audit_metrics.json`, `twin_plan_wins`): 164 failing `take <goal> from <target>` actions in 22 games lost their twin this way, and 127 of them would otherwise have been in RQ2's population (78 decided by contents or open state, 21 by the agent being elsewhere, 8 by the agent never having moved, 20 by full hands). That is about 2.7% of the population's size, all from one action type. The count is read from the recorded fields (task, action, scene and exclusion reason); the collector did not record that the replay ended the game, so it counts items with exactly this signature, not replayed confirmations. It is a selection effect: RQ2 contains no `take` of the goal object from the task's own target receptacle.

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

How the 12,987 collected items reduce to the 4,666 scored ones, from run `audit_analysis` at 28b4d30 (`audit_metrics.json`, `exclusion_flow`). Each stage applies one rule to the items that passed the stages above it, in this order; the counts would differ in another order, because some items fail several rules. "Games" is the number of games that still contribute items.

| Stage: items kept if… | Dropped | Remaining | Games |
| --- | --- | --- | --- |
| collected (every alternative action proposed by the generators) | | 12,987 | 134 |
| the true reply is "Nothing happens." (a real failure) | 3,120 | 9,867 | 134 |
| the environment's reply could be parsed | 0 | 9,867 | 134 |
| the history determines the failure | 41 | 9,826 | 134 |
| a success twin exists and fits the state (2,132 had no twin; 508 had a twin that contradicts the state) | 2,640 | 7,186 | 134 |
| a swapped goal object exists | 658 | 6,528 | 125 |
| the action is not at a shared spot | 1 | 6,527 | 125 |
| the action names the goal object | 861 | 5,666 | 125 |
| the history does not act on the goal or swap object | 1,000 | 4,666 | 125 |

The final count matches the scored population exactly (`flow_matches_primary_set`). The 3,120 true successes include the 307 determined successes not at a shared spot that RQ1 uses (309 before that rule); the 6,527 items left after the shared-spot stage are RQ1's true failures. The nine games with no swap object at all are `valid_unseen/100`, `101`, `102`, `135`, `136`, `137`, `165`, `166` and `167`.

### Items per game and near-duplicates

The 4,666 items are spread unevenly over the 125 games (`audit_metrics.json`, `items_per_game`): 37.3 per game on average (median 38, range 4 to 90), and the 10% of games with the most items hold 18.9% of them. Every test weighs items equally within the game-clustered bootstrap and permutation, so games with more items count for more; that defines the estimand as the item-weighted mean.

No two items are exact duplicates. Because the agent can revisit a state, some items repeat the same question in practice: 238 groups of items share the game, action, swapped goal and every fact read from the history (location, held object, open and closed receptacles, last-seen contents), covering 542 items, or 304 more than one per group. Items sharing only the game and action are much more common (1,013 groups, 2,876 items beyond one per group), because the same failing action recurs at different states. Clustering by game already accounts for this dependence in the intervals and the permutation test.

### The first-token term

The first-token term ℓ_first compares the first token of the success text with the first token of "Nothing happens.". On every one of the 4,666 failing items, for every model and every prompt scored, the success text begins with the token "You" and the failure with "Nothing" (`audit_metrics.json`, `first_token`). Among the 292 determined successes used for working actions, 236 success texts begin with "You" and 56 with "The", "There" or "This", so the term means something slightly different there.

How much of the model's next-token probability these two tokens cover (`first_mass`) differs by model. In the confirmatory scores (`rq2_score`) the median coverage is above 0.9999 for Qwen3-8B under both tasks. For Llama-3.1-8B it is 0.85 under the aligned task and 0.80 under the swapped task, and coverage falls below one half on 0.8% and 7.2% of items. The Word2World models are above 0.9999, and the Qwen3 size series is at 0.99 or above at every size. So for a typical Llama item 15–20% of the first-token probability goes to other words, and ℓ_first is a coarser reading of its preference than it is for Qwen.

### Shared-spot sensitivity

Run `rq2_shared_analysis` at 28b4d30 (`rq2_shared_metrics.json`). The flag is rare (`items.shared_flag_by_stage`): 19 of the 12,987 collected items carry it, 17 of them are real failures, 4 of those are determined by the history, and 1 has both a twin and a swapped goal; that is the one item the shared-spot stage of the flow above drops, out of 6,528. It would have been dropped anyway two stages later: its action names the goal object, but its history acts on a goal object. So **no item was excluded by the shared-spot rule alone**: `n_shared_spot_only` is 0, and the population with the shared-spot items put back is the same 4,666 items in 125 games. The result with them is therefore identical to the confirmatory result: ΔL is 6.12 nats (5.48 to 6.75) for Qwen3-8B and 1.65 nats (1.51 to 1.78) for Llama-3.1-8B, both with permutation p = 0.001 before Holm. The logged "report both ways" commitment is met, and the confirmatory result does not depend on the rule. No shared-spot scoring job was needed, because there was nothing to score.
