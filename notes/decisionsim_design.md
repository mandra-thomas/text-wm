# Option A: offline decision simulation (design, 2026-11-20)

Exploratory. This note was written and committed before any decision result existed: no candidate set has been scored completely under any condition, so no decision could have been computed. It was amended after review 1, still before any score existed; the changes are listed with their reasons in "Amendments after review 1, before any score" at the end, and the text below is the amended design (the original is commit 72aa1fd). Code: `wf/decisionsim.py`; runs `decisionsim_score_<model>` and `decisionsim_analysis`.

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

- the broad generator ran there, so its candidates include the agent's own next action, and that action is known (it is the next element of a longer saved prefix, so the last saved step of each game is out: 134 states, 19 at even steps and 115 at odd steps);
- at least one candidate truly succeeds (the environment's reply is not "Nothing happens.");
- at least one candidate is a failure the history determines (`context_determined`).

761 states in all 134 games are eligible. At every eligible state the candidate set still has a success and a determined failure after the agent's action is removed, so both set variants below are defined at every state. The agent's own action fails at 75 of them (43 of those failures are determined).

**Even and odd steps (amended).** The collector (`wf/data/transitions.py`, `build_transitions_from_rollout`, with k = 2) ran the broad generator, whose candidates include the agent's proposal, at every even step, and at odd steps only when the agent's proposal was invalid. So an odd-step state is eligible only if the agent's action there failed: odd-step states are selected on the outcome of the agent's action, and mixing them in would inflate every rate that depends on the proposal (all 33 odd-step eligible states have a failing proposal, against 42 of the 728 even-step states). Every state-level rate (the decision rules, change rates, harm and help, false rejection, the calibration readouts, the lure rate, the per-candidate baseline) is therefore computed on the **728 even-step states** (in all 134 games), where the generator ran whatever the agent did. The odd-step states are kept only for rule 2's false-acceptance rate, which is conditional on a failing proposal anyway; it is reported at even steps (42 failing proposals), at odd steps (33) and pooled (75). The constant `BROAD_EVERY_K = 2` in the code cites the collector.

**Census (amended).** Every eligible state is scored (`n_states: all`): 761 states, 9,930 candidates in 134 games. The first version scored a 300-state random sample; the census costs about 2.5 times as much and is still small (see "Cost"), and the verification rule depends on the agent's failing proposals, of which the sample held 37. Composition at the even-step states: 686 have a succeeding proposal and 42 a failing one; 676 have a swap object; at 87 a goal-relevant candidate succeeds. Of the 761 eligible states, 666 have exactly 4 succeeding candidates (74 have 3, 21 have 5).

## Candidate sets

The candidate set at a state is every action collected there, from both generators, deduplicated by action string: 8 to 24 candidates, 13 on average, with 3 to 5 successes and 4 to 20 failures. It is the same set under every condition. Each candidate's outcome is the real environment's reply on a fresh replay of the state. The set keeps the 35 failures the history does not determine and the 16 candidates at a shared spot: their outcomes are still real; the analysis reports how many chosen failures are determined. Two variants are analysed:

- **With the proposal:** the full set, which includes the action the agent actually proposed at that state.
- **Generated only:** the same set without the agent's proposal.

Composition matters for reading the results. The successes are mostly `go to` actions (about 85%) drawn as random admissible actions; nearly all failures act on the task's object. At 672 of the 761 eligible states no candidate that acts on the goal object succeeds.

## Conditions

Only the task line in the world model's prompt changes; the history, action, few-shot examples (set 1, history only) and the yes/no question are those of `rq1v`'s `yn_*` arms. The planner is always pursuing the real task: the conditions change what the world model is told, not the decision rule.

- **Aligned:** the episode's real task line.
- **Neutral:** "Your task is to: explore the room."
- **Swapped (scored for every model, amended):** the real task line with its object replaced by one state-level swap object, chosen with the state's seed among the swap objects already drawn for that state's candidates, excluding any object type an instance of which is named in any candidate action. Defined at 709 of 761 eligible states (676 of the 728 even-step states). It is needed to tell what the real task does from what showing any task of that form does (see "Decomposition" and the Option C rule).

## Score

Each candidate is scored with the standardised yes/no feasibility score, the log-odds of the single tokens "Yes" and "No" after the fixed question:

$$
L = \log P(\text{Yes}) - \log P(\text{No}).
$$

It is on the same scale for every action, unlike the two-candidate log-likelihoods, whose success texts differ in length and content across actions. **Scores are rounded to 1/64 nat (the bf16 resolution of the logit difference) before any comparison; ties are then exact.** The raw score is kept for the mean-score readouts only. "Predicted to succeed" means the model prefers Yes to No (after rounding), which is the same as

$$
\sigma(L) = \frac{P(\text{Yes})}{P(\text{Yes}) + P(\text{No})} > 0.5, \quad \text{that is,} \quad L > 0 .
$$

## Decision rules

1. **Highest feasibility.** Choose the candidate with the largest L. A tie is broken by a seeded per-state hash of the action, the smallest of `sha256(f"{seed}|{game}|{step}|{action}")` (seed 20260924), which is blind to outcomes and identical across conditions (amended: the first version used alphabetical order, which favours `examine` and `go to` over `take` at every state alike). The number of ties is reported, with the number whose tied candidates differ in outcome, and a sensitivity readout that splits every tied top set evenly (each state counts the share of failures in its tied set).
2. **Verification.** Accept the agent's own proposal if its L is above zero; otherwise choose the candidate with the largest L among the others (the generated-only set). Defined only on the with-proposal sets.
3. **Highest feasibility among goal-relevant candidates (secondary).** Restrict the set to candidates that act on the task's object (`targets_goal_object`), and choose the largest L among them. Because almost no goal-relevant candidate succeeds, rule 3 is reported only at states with at least one succeeding goal-relevant candidate (87 of the 728 even-step states). Elsewhere it must pick a failure whatever the scores are, so it would measure nothing. It is descriptive only.
4. **Small-menu rule 1, the "lure rate" (descriptive, amended).** At each even-step state whose proposal succeeds, the proposal is paired with one failing goal-object candidate at a time (5,475 pairs at 686 states), and rule 1 picks between the two; a lure is the failure picked. It isolates the goal-object pull from the composition of the full menu.

## Measures

For a model, a rule and a set variant, let $F_c$ be the share of even-step states whose chosen action fails under condition $c$. The primary contrast is aligned against neutral:

$$
\Delta F = F_{\text{aligned}} - F_{\text{neutral}} .
$$

Reported for each model, rule and set variant:

- the chosen-failure rate under each condition, and the difference above with its interval and permutation p;
- the share of chosen failures that are determined by the history;
- the **change rate**: the share of states where the chosen action differs between the two conditions, split into outcome-changing changes (one choice fails and the other succeeds) and changes between two failures or between two successes (a change from one working action to another is not harm);
- the **harm rate** (amended): the share of all states where the aligned choice fails and the neutral choice succeeds (task-induced harm), and likewise the help rate; among changed states, the shares of harm, help, both fail and both succeed. Over all states, the harm rate minus the help rate equals the difference above exactly; the analysis checks this identity;
- the share of choices that are the agent's proposal under each condition, and the failure rate of the choice by source (the proposal or a generated candidate) (amended);
- for rule 2, the difference split by acceptance cell (amended): states where the proposal is accepted under both tasks, rejected under both, accepted only under the aligned task, or accepted only under the neutral one, each with its count and its contribution to the difference (the contributions add up to it);
- for rule 2, separately: the **false-acceptance rate** (share of states with a failing proposal where the proposal is accepted; reported at even steps, odd steps and pooled) and the **false-rejection rate** (share of states with a succeeding proposal where it is rejected; even steps), under each condition, with the aligned-minus-neutral differences;
- calibration readouts (amended): at each even-step state, the neutral margin, the best succeeding candidate's rounded score minus the best failing candidate's under the neutral task, and the aligned shift on that best failure (its aligned minus neutral score), summarised by quartiles, the share of states whose margin is not positive, and the share of positive-margin states where the shift alone would close the margin.

**Baseline judgement, separated from the task-induced change.** A model can choose failures often under every task because it is poorly calibrated (Mistral already answers Yes to 45% of failures under the swapped task). So every rate is reported under the neutral task first, as the baseline, and the task-induced effect is always the aligned-minus-neutral difference, never the aligned rate alone. For each model and condition the analysis also reports the per-candidate accuracy (the share of candidates where the sign of L matches the true outcome), the share of failing candidates answered Yes, the share of succeeding candidates answered Yes, and two reference points: the failure rate of a uniformly random choice (the mean share of failures in the sets) and of an oracle (zero). The Yes rates and mean scores are also split by source (the agent's proposal or a generated candidate) by goal relevance by outcome, per condition and as aligned-minus-neutral and aligned-minus-swapped shifts (amended), so a change in decisions can be traced to the candidates whose scores moved.

The swapped condition gives the same measures for aligned against swapped and for swapped against neutral, on the states where it is defined.

**Decomposition (amended).** On the 676 even-step states with a swap, for every rule, the aligned-minus-neutral difference in chosen-failure rates splits exactly into an aligned-minus-swapped part (what naming the real task's object adds over a task of the same form about another object) and a swapped-minus-neutral part (what showing a task of that form does at all):

$$
F_{\text{aligned}} - F_{\text{neutral}} = (F_{\text{aligned}} - F_{\text{swapped}}) + (F_{\text{swapped}} - F_{\text{neutral}}) .
$$

All three are computed on the same states, so the identity is exact; the analysis checks it.

## Inference

Intervals are game-clustered percentile bootstraps (2,000 resamples of games, seed 20260924) of the per-state values. The p-value of a difference is a two-sided sign-flip permutation of the per-state paired differences (aligned minus neutral, each in $\{-1, 0, 1\}$) in which all states of a game flip together (1,000 permutations, seed 20260924), the test used throughout the study. States from one game share an episode, so the game is the unit of resampling. A percentile bootstrap of sparse differences in $\{-1, 0, 1\}$ can be narrow (when few states change, many resamples look alike), so the Option C rule asks for the permutation p as well as the interval. No multiplicity adjustment is made; apart from the Option C rule below, every number is descriptive.

## Predictions (logged before any score)

1. Under rule 1 with the proposal, the aligned task raises the chosen-failure rate over the neutral task, $\Delta F > 0$, for both primary models. Mechanism (reworded after review 1): two components. An object-naming bias, by which the aligned task raises Yes on actions that act on the task's object, nearly all of which fail here; and a task-relevance penalty, by which a stated task lowers Yes on actions irrelevant to it, which includes most successes (unrelated `go to` actions). The first should show in the aligned-minus-swapped part of the decomposition, the second in the swapped-minus-neutral part.
2. (Revised after review 1.) Only 6 failing proposals act on the goal object; the goal-object mechanism predicts little change in false acceptance; this rate is reported with an interval and has very low power.
3. The size of both follows calibration rather than the size of the effect in nats. Llama-3.1-8B, which answers Yes to 21% of determined failures under the aligned task, will change more decisions than Qwen3-8B, which answers Yes to 0.2%; Qwen's change rate under rule 1 may be below 5%. Mistral-7B will change many decisions but from a high baseline failure rate.
4. Because the aligned task also raises Yes on working actions (by 2.44 nats for Qwen and 0.98 for Llama on the determined successes), it will lower rule 2's false-rejection rate of succeeding proposals, a task-induced help that partly offsets prediction 2.

## What each outcome implies for Option C

Option C would branch from real agent runs: at states along real episodes, replace the agent's next action with the planner's choice under each world-model condition and continue the episode, to measure the end-to-end cost. It is expensive (about 30 GPU-minutes per game, from RQ3), so it should run only if the bias reaches decisions often enough to matter. The rule, as amended after review 1 and before any score (the original rule is quoted in the amendments section):

**Option C is worth doing if, for at least one primary model (Qwen3-8B, Llama-3.1-8B), on even-step states, under the decision rule Option C would itself run (rule 2: verification of the agent's own proposal, with fallback), the task-induced harm rate (aligned choice fails, neutral choice succeeds) is at least 5% of states, ΔF has a 95% game-bootstrap interval above zero and a game-permutation p below 0.05, and the aligned-minus-swapped ΔF under the same rule is positive with a point estimate at least half the aligned-minus-neutral one. Rule 1 on the full sets is reported but cannot by itself trigger Option C: its ΔF follows from the item-level effect and the generator's menu. A change rate is reported, split into outcome-changing and success-to-success changes.**

The aligned-minus-swapped and aligned-minus-neutral differences of the last condition are both taken from the decomposition, on the even-step states with a swap, so they are on the same states. Each primary model falls in exactly one of these categories, checked in this order:

- **Worth doing:** every condition above holds.
- **Toward failure, not attributable to the real task:** the harm rate is at least 5%, the interval is above zero and p is below 0.05, but the aligned-minus-swapped difference is not positive or is less than half the aligned-minus-neutral one. Showing a task moves decisions toward failure, but a task about another object does it nearly as much, so Option C would not test what the real task does.
- **Toward failure, too rare:** the interval is above zero and p is below 0.05, but the harm rate is below 5% of states.
- **Changes toward success:** the 95% interval of ΔF is entirely below zero: the aligned task makes rule 2 choose failures less often than the neutral one.
- **No reliable change in failures:** none of the above (the interval reaches zero, or it is above zero but p is at least 0.05).

The overall verdict waits until both primary models are scored with all three arms; it is then the first category in the order above that some primary model reaches, and only "worth doing" triggers Option C. A secondary model never decides it; one with a high harm rate (for example Mistral) would be reported as the case where the bias does reach decisions. Rule 1, rule 3, the generated-only sets, the lure rate and the calibration readouts inform the reading but do not enter the rule.

## The scoring run (not yet run)

`decisionsim_score_<model>`, one run per model, all 761 eligible states and all three arms, sharded by game (`WF_SHARD`, `WF_N_SHARDS`; one shard by default), yes/no only, using the existing scorer (`LogoddsScorer.score_one_token_pair`) at the pinned revisions in `wf.constants.REVISIONS`. Every condition of a state is scored in the same job, so each paired comparison is on one GPU. Each score is saved as it finishes, and a resume (`WF_RESUME=1`) skips saved scores and refuses saved scores that belong to different items or a different sample. Prompts are exactly `rq1v`'s (`sizes.arm_messages` for the reference models, `families.family_messages` for the family models, so Gemma's system text is folded into the first turn as in wave A). The family models get a Yes/No tokenization gate before scoring, as in wave A.

Models: Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct (primary); Mistral-7B-Instruct-v0.3, Gemma-2-9B-it and Llama-3.2-3B-Instruct (secondary). Llama-3.2-1B is left out: it failed wave A's competence gate.

**Cost (amended for the census).** Wave A's family runs took 103 to 107 minutes on an A40 or L40S for about 43,700 forward passes of 7–9B models (about 0.15 s a pass) and 33 minutes on an A100 for Llama-3.2-3B. The census has 9,930 candidates: aligned and neutral for each and swapped for the 7,196 candidates at the 709 states with a swap, 29,056 single-token forward passes per model (checked by a dry run on the archived collection). That is about 73 minutes for an 8B model plus loading and about 25 minutes for Llama-3.2-3B; all five models come to about 4.5–5 GPU-hours. The sbatch asks for 4 hours per job, about three times the estimate, so a slow node cannot cut a job off; a job that is cut off resumes from its saved scores.

## Caveats

- One agent (Qwen3-8B), one collection, the first 12 steps of each episode, one prompt format.
- The decision rules see only feasibility. A real planner weighs progress too; rule 3 is a gesture toward that and is underpowered.
- The candidate mix is set by the generators; the share of goal-object failures among candidates drives rule 1's failure rate under every condition.
- The neutral task is itself unusual for an agent's world model; the aligned-minus-neutral contrast measures what showing the task does, not what a different real task would do (the swapped arm does that).

## Amendments after review 1, before any score (2026-11-20)

Review 1 passed the engineering with fixes and failed the science as written. Every change below was made before any decision-simulation score existed (no `decisionsim_score_<model>` run had started), so none of them could have been chosen with knowledge of a result. The counts quoted were rechecked by a dry run of the amended code on the archived collection (`text-wm-fix17`), which builds the states and candidate sets but reads no model score.

1. **Odd-step states left out of state-level rates.** The collector ran the broad generator (whose candidates include the agent's proposal) at even steps always and at odd steps only when the proposal was invalid (`wf/data/transitions.py`, `build_transitions_from_rollout`, k = 2). The first version did not know this, so its eligible set mixed 728 even-step states with 33 odd-step states that are eligible only because the agent's action failed. That selection inflates every rate that involves the proposal (rule 2 especially, and the generated-only contrast is no cure, since the set's existence is what is selected). Every state-level rate is now on the 728 even-step states (constant `BROAD_EVERY_K = 2`); the odd-step states enter only the false-acceptance rate, which is conditional on a failing proposal, reported by parity and pooled.
2. **Scores rounded to 1/64 nat.** Scores are rounded to 1/64 nat (the bf16 resolution of the logit difference) before any comparison; ties are then exact. The existing yes/no scores sit on that grid up to float noise, so without rounding two candidates the model cannot tell apart could be ordered by noise, a tie could be missed, and a score of a millionth of a nat would count as Yes. Raw scores are kept for the mean-score readouts.
3. **A new Option C rule.** The original rule was: Option C is worth doing if, for at least one primary model, under rule 1 or rule 2 on the with-proposal sets, the change rate between the aligned and neutral tasks is at least 5% of states and ΔF is positive with a 95% interval that excludes zero; "weak case" if the change rate reached 5% without that; otherwise not worth doing. Three problems: (a) rule 1 on the full sets has a ΔF that follows almost mechanically from the item-level effect and the generator's menu (mostly goal-object failures against unrelated `go to` successes), so it could trigger Option C without saying anything new about planning; Option C would run rule 2, verification of the agent's own proposal, so the gate now uses rule 2. (b) A change rate counts changes from one working action to another, which cost nothing; the gate now uses the harm rate (aligned choice fails, neutral choice succeeds) and the change rate is reported split. (c) A negative ΔF with a 5% change rate was labelled "weak case", which misreads a change toward success; it now has its own category, and every category is defined. The gate also asks for the permutation p (the percentile bootstrap of sparse per-state differences of minus one, zero or one can be narrow) and for the swapped attribution check (next item). The rule is quoted in full in "What each outcome implies for Option C".
4. **Swapped task scored for every model.** The aligned-minus-neutral contrast mixes two things: what naming the real task's object does, and what showing any task of that form does (the neutral line has no object and no goal at all). The swapped arm separates them, so it is now scored for all five models, the decomposition aligned − neutral = (aligned − swapped) + (swapped − neutral) is reported on the even-step states with a swap, and the Option C rule asks that the aligned-minus-swapped part be positive and at least half of the whole. Prediction 1's mechanism was reworded to name both components (object-naming bias and a task-relevance penalty on irrelevant actions).
5. **Source and goal-relevance splits; prediction 2 revised.** Rule 2's behaviour depends on the agent's proposal, which is a different kind of candidate from the generated ones (the proposal is mostly a working `go to`; the generated failures mostly act on the goal object). The per-candidate Yes rates and mean scores are now split by source by goal relevance by outcome, per condition and as shifts, and each contrast reports how often the choice is the proposal and how often a chosen proposal or a chosen generated candidate fails. Prediction 2 was revised before any data: of the 75 failing proposals only 6 act on the goal object (4 at even steps), so the goal-object mechanism predicts little change in false acceptance, and the rate has very low power.
6. **Tests that each condition gets its own prompt.** A test now records every prompt the scoring run sends and checks the task line per arm (the real line; "Your task is to: explore the room."; the line from `make_swapped_task_line` with the state's swap object), the yes/no question at the end, and that each candidate is scored once per arm. Another checks that a state without a swap object is kept when the swapped arm is scored.
7. **Tie order.** Ties now go to a seeded per-state hash of the action instead of alphabetical order, which favoured the same verbs at every state (and `examine` and `go to`, mostly successes, over `take`, mostly failures). The analysis also counts tied top sets whose candidates differ in outcome and reports a sensitivity readout that splits each tied set evenly.
8. **Rule 2 by acceptance cell; calibration readouts; lure rate.** Rule 2's difference is split by whether the proposal was accepted under both tasks, neither, or one of them, so a change in the fallback can be told from a change in verification. Each even-step state's neutral margin and the aligned shift on its best failure show how close rule 1 is to a flip and how far the task moves it. The small-menu lure rate pairs a working proposal with one goal-object failure at a time.
9. **Census.** All 761 eligible states are scored instead of a 300-state sample, with the swapped arm: 29,056 passes per model, about 73 minutes for an 8B model, 4.5–5 GPU-hours for all five; the sbatch limit is 4 hours per job.
10. **Facts added to the text.** The last saved step of each game is excluded (134 states: 19 at even steps, 115 at odd steps); 666 of the 761 eligible states have exactly 4 succeeding candidates; the percentile bootstrap on sparse differences can be narrow, hence the permutation p in the gate.

