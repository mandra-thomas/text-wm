# Option A: offline decision simulation (design, 2026-11-20)

Exploratory. This note was written and committed before any decision result existed: no candidate set has been scored completely under any condition, so no decision could have been computed. Code: `wf/decisionsim.py`; runs `decisionsim_score_<model>` and `decisionsim_analysis`.

## Question

The confirmatory result (RQ2) and its yes/no replication show that naming the action's object in the task line pushes a world model toward predicting success on actions whose failure the history already settles. That is a statement about scores of single actions. This simulation asks the next question: **does the task-induced bias change which action a planner that consults the world model would choose, and how often does it make that planner choose an action that fails?**

## What this is and what it is not

It is a **decision-level diagnostic**. At real states from the collected agent episodes, a fixed set of candidate actions with known true outcomes is scored by the world model, a fixed decision rule picks one, and we record whether the pick fails. The only thing that varies between conditions is the task line in the world model's prompt.

It is **not** a measure of end-to-end planning harm. Nothing is executed after the choice; there is no recovery, no later step, no task success. A failing choice in ALFWorld costs one step ("Nothing happens.") and may or may not matter for the episode. The rules are stylised planners that use only the world model's yes/no feasibility, not an agent's own preferences or progress toward the goal: rule 1 below is happy to choose `go to cabinet 2` over a useful action, because it maximises predicted feasibility and nothing else. The candidate sets come from our generators (mostly goal-object failures, a few random admissible actions, and the agent's own proposal), not from a planner's proposal distribution. So the rates are properties of this stylised setting, useful for deciding whether the bias reaches decisions often enough to justify an end-to-end test (Option C), not estimates of harm to a real agent.

## Why a new scoring run is needed (coverage)

The existing yes/no scores (`rq1v_score` for Qwen3-8B and Llama-3.1-8B; `size_score_qwen3-8b`; `family_score_<model>` for Gemma-2-9B, Mistral-7B, Llama-3.2-1B and 3B on ICE) all cover the same 4,958 items: RQ2's 4,666 primary failures and the 292 determined successes. A decision needs a score for **every** candidate at a state. Checked on the archived collection (`text-wm-fix17`, identical to the collection every run read):

- The collection has 12,987 candidates at 1,461 states in 134 games; 4,958 of them (38%) are scored.
- Only 179 states have every candidate scored, and **none of them has a candidate that succeeds** (the scored successes are scattered over states whose other candidates are unscored). A decision there is a forced failure, so these states say nothing.
- The 181 states with at least one scored success and one scored failure (the post hoc "top-ranked candidate" numbers in `exploratory_results.md`) are partial sets: on average 47% of the candidates at those states are unscored, and at 113 of them the agent's own action is not a scored candidate (at 108 it is a candidate but unscored).

So no state has a complete candidate set with a working option, for any model. The existing scores cannot answer the question, and the family scores on ICE cover the same items, so they cannot either. A small new scoring run is required; it is specified below and not yet run.

## States

A state is a point (game, step) in one of the 134 collected Qwen3-8B ReAct episodes (at most 12 steps each; `rq2_results.md`, "Methods"). A state is **eligible** when:

- the broad generator ran there, so its candidates include the agent's own next action, and that action is known (it is the next element of a longer saved prefix, so the last saved step of each game is out);
- at least one candidate truly succeeds (the environment's reply is not "Nothing happens.");
- at least one candidate is a failure the history determines (`context_determined`).

761 states in all 134 games are eligible. At every eligible state the candidate set still has a success and a determined failure after the agent's action is removed, so both set variants below are defined at every state. The agent's own action fails at 75 of them (43 of those failures are determined).

**Sample.** 300 eligible states (3,857 candidates in 123 games), a simple random sample drawn with `random.Random(20260924)` from the eligible states sorted by (game, step). A simple random sample keeps every rate an unweighted estimate for the eligible population. Scoring all 761 (`n_states: all` in the config) costs about 2.5 times as much and is still small (see "Cost"); it is the better choice if the budget allows, mainly because the verification rule depends on the agent's failing proposals, of which the 300-state sample holds 37 (75 in the census).

## Candidate sets

The candidate set at a state is every action collected there, from both generators, deduplicated by action string: 8 to 24 candidates, 13 on average, with 3 to 5 successes and 4 to 20 failures. It is the same set under every condition. Each candidate's outcome is the real environment's reply on a fresh replay of the state. The set keeps the 35 failures the history does not determine and the 16 candidates at a shared spot: their outcomes are still real; the analysis reports how many chosen failures are determined. Two variants are analysed:

- **With the proposal:** the full set, which includes the action the agent actually proposed at that state.
- **Generated only:** the same set without the agent's proposal.

Composition matters for reading the results. The successes are mostly `go to` actions (about 85%) drawn as random admissible actions; nearly all failures act on the task's object. At 672 of the 761 eligible states no candidate that acts on the goal object succeeds.

## Conditions

Only the task line in the world model's prompt changes; the history, action, few-shot examples (set 1, history only) and the yes/no question are those of `rq1v`'s `yn_*` arms. The planner is always pursuing the real task: the conditions change what the world model is told, not the decision rule.

- **Aligned:** the episode's real task line.
- **Neutral:** "Your task is to: explore the room."
- **Swapped (optional, secondary):** the real task line with its object replaced by one state-level swap object, chosen with the state's seed among the swap objects already drawn for that state's candidates, excluding any object type an instance of which is named in any candidate action. Defined at 709 of 761 eligible states. The default run scores only aligned and neutral, as the plan asked; adding `yn_swapped` to `arms` costs 50% more.

## Score

Each candidate is scored with the standardised yes/no feasibility score, the log-odds of the single tokens "Yes" and "No" after the fixed question:

$$
L = \log P(\text{Yes}) - \log P(\text{No}).
$$

It is on the same scale for every action, unlike the two-candidate log-likelihoods, whose success texts differ in length and content across actions. "Predicted to succeed" means the model prefers Yes to No, which is the same as

$$
\sigma(L) = \frac{P(\text{Yes})}{P(\text{Yes}) + P(\text{No})} > 0.5, \quad \text{that is,} \quad L > 0 .
$$

## Decision rules

1. **Highest feasibility.** Choose the candidate with the largest L. A tie is broken by the action string in alphabetical order, which is blind to outcomes and identical across conditions; the number of ties is reported.
2. **Verification.** Accept the agent's own proposal if its L is above zero; otherwise choose the candidate with the largest L among the others (the generated-only set). Defined only on the with-proposal sets.
3. **Highest feasibility among goal-relevant candidates (secondary).** Restrict the set to candidates that act on the task's object (`targets_goal_object`), and choose the largest L among them. Because almost no goal-relevant candidate succeeds, rule 3 is reported only at states with at least one succeeding goal-relevant candidate (89 of the 761 eligible states, 33 in the sample). Elsewhere it must pick a failure whatever the scores are, so it would measure nothing. It is descriptive only.

## Measures

For a model, a rule and a set variant, let $F_c$ be the share of states whose chosen action fails under condition $c$. The primary contrast is aligned against neutral:

$$
\Delta F = F_{\text{aligned}} - F_{\text{neutral}} .
$$

Reported for each model, rule and set variant:

- the chosen-failure rate under each condition, and the difference above with its interval and permutation p;
- the share of chosen failures that are determined by the history;
- the **change rate**: the share of states where the chosen action differs between the two conditions;
- among changed states, the share where the aligned choice fails and the neutral choice succeeds (task-induced harm), the share where the aligned choice succeeds and the neutral choice fails (task-induced help), and the shares where both fail or both succeed. Over all states, the harm count minus the help count, divided by the number of states, equals the difference above exactly; the analysis checks this identity;
- for rule 2, separately: the **false-acceptance rate** (share of states with a failing proposal where the proposal is accepted) and the **false-rejection rate** (share of states with a succeeding proposal where it is rejected), under each condition, with the aligned-minus-neutral differences.

**Baseline judgement, separated from the task-induced change.** A model can choose failures often under every task because it is poorly calibrated (Mistral already answers Yes to 45% of failures under the swapped task). So every rate is reported under the neutral task first, as the baseline, and the task-induced effect is always the aligned-minus-neutral difference, never the aligned rate alone. For each model and condition the analysis also reports the per-candidate accuracy (the share of candidates where the sign of L matches the true outcome), the share of failing candidates answered Yes, the share of succeeding candidates answered Yes, and two reference points: the failure rate of a uniformly random choice (the mean share of failures in the sets) and of an oracle (zero).

The swapped condition, where scored, gives the same measures for aligned against swapped and for swapped against neutral, on the states where it is defined.

## Inference

Intervals are game-clustered percentile bootstraps (2,000 resamples of games, seed 20260924) of the per-state values. The p-value of a difference is a two-sided sign-flip permutation of the per-state paired differences (aligned minus neutral, each in $\{-1, 0, 1\}$) in which all states of a game flip together (1,000 permutations, seed 20260924), the test used throughout the study. States from one game share an episode, so the game is the unit of resampling. No multiplicity adjustment is made; apart from the Option C rule below, every number is descriptive.

## Predictions (logged before any score)

1. Under rule 1 with the proposal, the aligned task raises the chosen-failure rate over the neutral task, $\Delta F > 0$, for both primary models. Mechanism: the aligned task raises Yes on actions that act on the task's object, and nearly all failures do, while most successes are unrelated `go to` actions.
2. Under rule 2, the aligned task raises the false-acceptance rate of failing proposals for both primary models.
3. The size of both follows calibration rather than the size of the effect in nats. Llama-3.1-8B, which answers Yes to 21% of determined failures under the aligned task, will change more decisions than Qwen3-8B, which answers Yes to 0.2%; Qwen's change rate under rule 1 may be below 5%. Mistral-7B will change many decisions but from a high baseline failure rate.
4. Because the aligned task also raises Yes on working actions (by 2.44 nats for Qwen and 0.98 for Llama on the determined successes), it will lower rule 2's false-rejection rate of succeeding proposals, a task-induced help that partly offsets prediction 2.

## What each outcome implies for Option C

Option C would branch from real agent runs: at states along real episodes, replace the agent's next action with the planner's choice under each world-model condition and continue the episode, to measure the end-to-end cost. It is expensive (about 30 GPU-minutes per game, from RQ3), so it should run only if the bias reaches decisions often enough to matter. The rule, fixed now:

- **Option C is worth doing** if, for at least one primary model (Qwen3-8B or Llama-3.1-8B), under rule 1 or rule 2 on the with-proposal sets, the change rate between the aligned and neutral tasks is at least 5% of states (point estimate) **and** the chosen-failure difference $\Delta F$ is positive with a 95% interval that excludes zero. The bias then changes a meaningful share of decisions, and toward failure.
- **Weak case** if the change rate reaches 5% for some primary model but $\Delta F$ is not positive with an interval above zero, so the changed decisions are about as often helpful as harmful. Option C would measure choice shifts without a decision-level reason to expect harm; it would be low priority.
- **Not worth doing for these models** if the change rate is below 5% for both primary models under both rules. The bias then moves scores but rarely the argmax or the threshold decision, because calibration absorbs it. A secondary model with a high change rate (for example Mistral) would be reported as the case where it does reach decisions, but it would not by itself justify Option C on the primary models.

Rule 3, the generated-only sets and the swapped condition inform the reading but do not enter the rule.

## The scoring run (not yet run)

`decisionsim_score_<model>`, one run per model, sharded by game (`WF_SHARD`, `WF_N_SHARDS`; one shard by default), yes/no only, using the existing scorer (`LogoddsScorer.score_one_token_pair`) at the pinned revisions in `wf.constants.REVISIONS`. Every condition of a state is scored in the same job, so each paired comparison is on one GPU. Each score is saved as it finishes, and a resume (`WF_RESUME=1`) skips saved scores and refuses saved scores that belong to different items or a different sample. Prompts are exactly `rq1v`'s (`sizes.arm_messages` for the reference models, `families.family_messages` for the family models, so Gemma's system text is folded into the first turn as in wave A). The family models get a Yes/No tokenization gate before scoring, as in wave A.

Models: Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct (primary); Mistral-7B-Instruct-v0.3, Gemma-2-9B-it and Llama-3.2-3B-Instruct (secondary). Llama-3.2-1B is left out: it failed wave A's competence gate.

**Cost.** Wave A's family runs took 103 to 107 minutes on an A40 or L40S for about 43,700 forward passes of 7–9B models (about 0.15 s a pass) and 33 minutes on an A100 for Llama-3.2-3B. The sample has 3,857 candidates, so two conditions are 7,714 single-token forward passes per model: about 20–25 minutes for an 8B model plus loading, under 10 minutes for Llama-3.2-3B. All five models at the default (300 states, aligned and neutral) come to about 1.5–2 GPU-hours; with the swapped arm, about 2.5; the census of 761 states with all three arms, about 4.5–5. The sbatch asks for 3 hours per model so a slow node cannot cut a job off.

## Caveats

- One agent (Qwen3-8B), one collection, the first 12 steps of each episode, one prompt format.
- The decision rules see only feasibility. A real planner weighs progress too; rule 3 is a gesture toward that and is underpowered.
- The candidate mix is set by the generators; the share of goal-object failures among candidates drives rule 1's failure rate under every condition.
- The neutral task is itself unusual for an agent's world model; the aligned-minus-neutral contrast measures what showing the task does, not what a different real task would do (the swapped arm does that).
