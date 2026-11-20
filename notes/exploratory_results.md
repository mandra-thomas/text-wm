# Exploratory follow-ups to RQ2 (2026-11-19)

None of these analyses was preregistered. Each design was logged in `deviations.md` before its data; the dates and wording are there. Numbers marked **post hoc** were computed by one-off scripts after the results were in, and are not yet produced by a pipeline run. All other numbers come from the analysis runs named in each section, whose metrics are copied here as `<run>_metrics.json`.

Common setup, unless stated otherwise:

- **Models:** few-shot Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct.
- **Prompt:** example set 1, history only.
- **Items:** RQ2's 4,666 determined failures.
- **Measures:**
  - L is the log-odds of the success text against "Nothing happens.".
  - ΔL is the mean of L(aligned task) − L(swapped task), in nats.
  - The two-candidate score is s = σ(L), and Δs is its shift in percentage points.
  - The first-word term ℓ_first is the part of L carried by the first word ("You" against "Nothing").
- **Intervals:** game-clustered bootstrap 95% intervals.

## 1. Object-free goals, extra-mention design, working actions (`rq1x_score`, `rq1x_analysis`)

This run uses a secondary scoring of the same items, which carries all the controls. It ran on different GPUs, so individual scores differ slightly: overall ΔL is 6.10 and 1.64, against 6.12 and 1.65 in the confirmatory run.

- **Described (object-free) goal.** Every object name in the task is replaced by a fixed description with no ALFWorld object name ("cool some thick drinking vessel with a handle for hot drinks and put it in cabinet"). The action still names `mug 1`.
  - Most of the effect survives: Qwen 3.71 nats [3.30, 4.11], 61% of the named-goal effect; Llama 1.51 [1.36, 1.67], 92%.
  - On items where both descriptions passed a strict naming test, 61% and 84% survive.
  - The effect therefore goes through meaning, not only through repeating the object's name.
- **Extra-mention design** (from `rq1_score`, in `rq1_metrics.json` under `rq2_secondary.extra_mention`, not from `rq1x`). "There is a {obj} somewhere in the room." is added to the swapped task for the action's object, or to the aligned task for the swapped object. The goal effect then splits exactly into three parts. Each is shown as its contribution to the effect. For the aligned-prompt sentence this is minus its effect on the aligned prompt: the sentence lowers that prompt's L, by 1.09 nats for Qwen and 0.06 for Llama.

  | Part | Qwen | Llama |
  | --- | --- | --- |
  | Sentence added to the swapped prompt | 6.00 | 0.99 |
  | Goal role, with both objects named | −0.99 [−1.42, −0.55] | +0.59 [0.43, 0.74] |
  | Sentence added to the aligned prompt | 1.09 | 0.06 |

  For Qwen, on average, mentioning the object does most of the work. At step 0, where Qwen's effect is 11.0 nats, the sentence accounts for 4.5 and the goal role is +2.8 (both in the metrics, under `history_0`). For Llama about a third of the effect depends on the object being the goal. That part is concentrated in take actions and after step 0.
- **2×2 design** (1,076 move, heat, cool and clean items). The part due to usefulness is 0.72 for Qwen and −0.18 for Llama.
- **Neutral task ("explore the room").** Against it, even the swapped task raises L, by 2.56 nats for Qwen and 1.20 for Llama.
- **Determined successes** (292 items; they are not built like the failures).
  - **Matched successes.** On the 37 that meet all of the failures' criteria (**post hoc**), the goal effect is as large as on failures: 10.3 nats [8.3, 12.8] for Qwen and 1.5 [1.1, 2.1] for Llama.
  - **Discrimination (AUC).** This measures how well L ranks working actions above failing ones. Moving from the neutral to the aligned task, Llama falls from 0.87 to 0.77 (−0.10 [−0.12, −0.08]) and Qwen from 0.93 to 0.91.
  - **Discrimination on the first-word term** (**post hoc**). Llama falls from 0.88 to 0.75 and Qwen from 0.99 to 0.97.
  - **Top-ranked candidate.** This is counted at the 181 states with both a working and a failing candidate. A failing action is ranked first at 19% of states under the aligned task and 9% under the neutral task for Llama, and 8% and 8% for Qwen (**post hoc**, one-off script; not in the metrics file).
- **Ignore-goal instruction.** "(The agent's goal does not change what happens. Predict the observation from the history and the action alone.)" removes 9% of Qwen's effect and none of Llama's.

## 2. Qwen3 thinking mode (`thinking_probe`, `thinking_analysis`)

Qwen3-8B reasoned before answering. The run used 1,000 sampled items and five task lines: aligned, swapped, neutral, and both described goals. Sampling followed Qwen's recommended settings, with the reasoning capped at 3,072 tokens; 2.6–7.5% of replies hit the cap and were forced to answer, and leaving them out changes nothing. L was scored given the model's own reasoning.

- **Smaller gap.** The aligned-vs-swapped effect is 4.01 nats [2.87, 5.12] with reasoning, against 6.06 [5.31, 6.76] with thinking off on the same items.
- **But not less lift.** Relative to the neutral task, the aligned task raises L about equally: 9.0 nats with reasoning and 8.6 without. The swapped task's lift doubles, from 2.5 to 5.0. Reasoning spreads the lift to other tasks; it does not remove it.
- **Answers off target** (**post hoc**, by reading the reasoning). In 14% of answers the reasoning drifted to another object or to a few-shot example. On the answers that stayed on target (**post hoc**), the pattern holds.
- **Written predictions.** Their aligned-vs-swapped shift is 4.7 points [1.6, 8.3] (final parser version). These rates are not used further, because of the off-target answers.

## 3. Dissociation test (`rq1d_score`, `rq1d_analysis`)

This tests whether the task acts as evidence that its object exists. The items are the 4,382 with one known location for the action's object. One sentence about the action's object is added to both the aligned and the swapped prompts, so each pair still differs only in the goal. There are three versions:

- **Exists:** "There is a mug somewhere in the room."
- **Located:** "The mug 1 is at cabinet 3."
- **Denied:** "There is no mug in this room." This is counterfactual.

The logged readouts disagree:

- **Denial.** It lowers the swapped prompt's L for both models, by 3.15 nats for Qwen and 2.21 for Llama, as the evidence account predicts. On the outcome word it lowers Llama's (−0.38) but raises Qwen's (+0.71), as a salience account predicts.
- **Location.** The logged comparison ran over every verb. For move, heat, cool and clean, a location away from the action's receptacle is not a contradiction. Restricted to take actions (**post hoc**):
  - Stating that the object is elsewhere barely moves the score: Qwen −0.5 points, Llama +7.8.
  - Stating that it is at the action's receptacle raises it sharply: Qwen +68 points, Llama +49.
  - For takes, then, the models do use where the object is.
- **Goal effect with the same sentence in both prompts.** On the outcome word the goal effect vanishes whenever both prompts say whether the object exists: Qwen −0.16 and −0.12, Llama 0.07 and 0.05, against 2.42 and 0.66 with no sentence. This fits the evidence account, but salience predicts it as well, as the log said before the data.

The readouts are mixed. They need a confirmatory test with a design that separates the two accounts.

## 4. Counterfactual aligned task and yes/no question (`rq1v_score`, `rq1v_analysis`)

- **Counterfactual aligned task** (2,974 take, heat, cool and clean items). The real task's destination is replaced by another receptacle type in the room. This leaves the action's relation to the task unchanged, but the task is no longer the one the agent pursued.
  - **Effect unchanged.** It is 6.52 [5.83, 7.23] for Qwen and 2.09 [1.88, 2.29] for Llama, against 6.33 and 1.96 with the real task on the same items.
  - **Goal-specific part unchanged.** Llama's part that depends on the object being the goal is also unchanged.
  - **What this rules out:** that the effect needs the exact real task. It does not rule out that the history fits the task's object.
- **Yes/no question.** The final prompt line is replaced by a fixed question: will the action succeed, or will the environment reply "Nothing happens."? "Yes" is scored against "No".
  - **Goal effect.** It replicates in log-odds: 2.80 [2.59, 3.02] for Qwen and 2.21 [2.05, 2.39] for Llama.
  - **Yes-share.** Qwen almost never says yes to these failures (0.2% under the aligned task, **post hoc**). Llama's yes-share shifts by 16 points: 40 at step 0 and 13 afterwards (**post hoc**).
  - **Discrimination.**
    - AUC falls from 0.99 to 0.95 for Qwen and from 0.97 to 0.85 for Llama.
    - A failing action is ranked first at 8% against 2% of states for Qwen, and 20% against 7% for Llama.

## Further post hoc breakdowns of the confirmatory items

- **Practical size depends on the model and the format.**
  - **Two-candidate score, Qwen.** It shifts 42 points at step 0 (514 items) and 1.4 afterwards.
  - **Two-candidate score, Llama.** It shifts 13 and 17 points.
  - **Yes/no question.** The numbers are in section 4.
- **The agent's actual first action.** In 134 ReAct episodes it was always `go to` or `look`. Qwen's largest effect therefore falls on step-0 actions that such an agent does not propose.
- **Failures determined under any rule** (788 items where the receptacle was already seen without the object). The effect is 4.48 [3.57, 5.41] nats for Qwen and 2.82 [2.47, 3.18] for Llama. That is 33 points of score for Llama and 0.1 for Qwen.
- **Effect by reason for failure** (nats, Qwen / Llama):
  - not at the receptacle, agent has not moved: 10.9 / 1.5;
  - not at the receptacle, moved elsewhere: 6.0 / 1.6;
  - receptacle seen without the object: 4.5 / 2.8;
  - receptacle closed: 3.6 / 2.0;
  - hands full: 5.5 / 1.3;
  - object not held: 5.9 / 0.8.

## 5. Mechanism diagnostic: state inference or judgement? (`mech_score`, `mech_analysis`)

Design and readouts were logged in `deviations.md` before any score existed, and corrected after an independent code check, also before any score. Scored at 10a61a7, on the 4,666 primary items with both models. Metrics are in `mech_metrics.json`.

**The two accounts.**
- **(a) State inference:** the model doesn't recover the facts that decide the failure, and falls back on the task.
- **(b) Judgement:** it recovers them, but judges the outcome by the task.

**The design.**
- **Four versions of the prediction,** each under the aligned and the swapped task:
  - plain;
  - with the deciding facts stated, taken from the history (location, held object, open and closed receptacles, contents seen);
  - with the environment's rules stated;
  - with both.
- **Yes/no questions** under the aligned, swapped and neutral tasks:
  - about the deciding fact, each with a twin question whose correct answer is Yes;
  - about whether the action's conditions are met;
  - about the rule (on 1,000 items), with a twin.
- **The agent's own preference for the action:** the ReAct agent's prompt, scoring the action against "look".

| Readout (logged before data) | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Goal effect, plain prompt (nats) | 6.12 [5.51, 6.74] | 1.64 [1.50, 1.79] |
| With the deciding facts stated | 4.60; share removed 25% [21%, 29%] | 1.60; share removed 2% [−3%, 8%] |
| With the rules stated | 10.06 (64% larger) | 2.10 (28% larger) |
| With facts and rules | 8.59 | 2.18 |
| Effect on items whose state question and twin are both answered correctly (neutral task), as a share of the overall effect | 1.11 [1.07, 1.16] | 1.01 [0.92, 1.10] |
| Goal effect on "Yes" to the state question, minus the effect on its twin (nats) | −0.34 [−0.72, 0.05] | +0.38 [0.32, 0.45] |
| Spearman correlation between the shift in the agent's preference and ΔL | −0.08 [−0.14, −0.01] | +0.06 [0.01, 0.11] |

**Readings against the logged thresholds:**
- **Stating the facts.** It removes less than 25% for Llama, which favours (b). For Qwen it removes 25%, right at the threshold.
- **The rules arm.** Stating the rules *increases* the effect. This was not anticipated. The rules make both models much more confident of failure overall, yet the gap between the aligned and swapped tasks grows: Qwen's first-word term rises from 2.52 to 5.41 nats. On the success-score scale, Qwen's shift falls from 5.8 to 3.8 points, because the scores are at the floor. Llama's rises from 16.3 to 17.6.
- **Where the state is known.** The effect is fully present on items where the model answers the state question correctly. This favours (b) for both models.
- **Does the goal change the state belief?** For Qwen, the task raises "Yes" equally on the state question and its twin. That is a general shift toward "Yes", not a specific change in its belief about the state. For Llama there is a small specific shift toward the fact that would let the action work.
- **The agent's own preference.** The task raises it enormously (Qwen 14.4 nats, Llama 6.5). But the correlation with the world-model shift is far below the logged 0.3, so item-by-item variation in the agent's own preference does not explain the effect.
- **Precondition question.** Both models answer No on 97–100% of items under every task, yet the aligned task still shifts its log-odds toward Yes (Qwen 3.5 nats, Llama 1.6).

**Interpretation (exploratory):**
- The leak is not caused by missing information about the state, which goes against (a).
- What remains is best described as a task-relevance bias applied at answer time.
- **Caveat:** the yes/no questions are weak measures of knowledge. Both models lean toward No, and rule knowledge is poor: Qwen answers the rule question and its twin correctly on only 9% of items, under the neutral task.

**Earlier post hoc looks.** Two analyses of the confirmatory scores came before this design (history length, room size, receptacles not yet observed, actions that advance the task). An independent check showed their conclusions depend on the specification, so they are recorded in `deviations.md` as inconclusive.

## 6. Model-size series, Qwen3 dense (`size_score_<model>`, `size_analysis`)

Design and prediction were logged before any score. Scoring ran at 8bfd84c and the analysis at 10a61a7, on the 4,666 primary items and 292 determined successes. Metrics are in `size_metrics.json`. The rescored Qwen3-8B reproduces the confirmatory results exactly: 6.12 nats two-candidate, 2.80 yes/no.

**The columns:**
- the two ΔL columns are the aligned-minus-swapped effect in nats;
- the score shift is in percentage points of the two-candidate success score;
- flips are the share of items whose first-word decision changes;
- the yes-share is on failing actions, under the neutral task;
- the AUC compares working with failing actions on the two-candidate score.

| Size (parameters) | ΔL two-candidate | ΔL yes/no | Score shift | Flips | Yes-share on failures | AUC, neutral → aligned |
|---|---|---|---|---|---|---|
| 0.6B (0.75B) | 0.89 | 0.12 | 2.0 | 0.6% | 99.9% | 0.72 → 0.57 |
| 1.7B (2.03B) | 2.49 | 2.08 | 15.6 | 24.8% | 99.9% | 0.81 → 0.79 |
| 4B (4.02B) | 7.49 | 2.27 | 12.7 | 19.2% | 2.3% | 0.93 → 0.89 |
| 8B (8.19B) | 6.12 | 2.80 | 5.8 | 9.7% | 0.2% | 0.93 → 0.91 |
| 14B (14.77B) | 4.33 | 3.18 | 12.3 | 21.3% | 0.6% | 0.90 → 0.89 |
| 32B (32.76B) | 2.35 | 1.55 | 13.1 | 17.8% | 4.7% | 0.91 → 0.88 |

**Against the logged prediction:**
- **The effect is positive at every size,** as predicted.
- **"The slope against log parameters is not positive" fails.** The slope is +0.52 nats per e-fold of parameters [0.40, 0.63] on the two-candidate score, and +0.46 [0.36, 0.55] on yes/no. The bootstrap resamples games jointly across sizes.

**Post hoc reading.** The two smallest models largely cannot do the task: they answer Yes to almost every yes/no question, working or failing. Among the competent sizes (4B–32B):
- **The two-candidate effect falls steadily** (7.5, 6.1, 4.3, 2.4 nats). The yes/no effect does not fall steadily.
- **The practical size does not fall.** Score shifts are 12.7, 5.8, 12.3 and 13.1 points, and flips 19%, 10%, 21% and 18%. Larger models are less extreme in log-odds, so a smaller effect in nats moves probabilities as much.
- **The largest model is still clearly moved.** Qwen3-32B answers Yes to failing actions 15.0% of the time under the aligned task, against 5.8% under the swapped task.

Scaling within Qwen3 does not remove the leak.
