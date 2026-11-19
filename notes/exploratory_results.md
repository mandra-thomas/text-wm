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
- **Extra-mention design.** "There is a {obj} somewhere in the room." is added to the swapped task for the action's object, or to the aligned task for the swapped object. The goal effect then splits exactly into three parts:

  | Part | Qwen | Llama |
  | --- | --- | --- |
  | Sentence added to the swapped prompt | 6.00 | 0.99 |
  | Goal role, with both objects named | −0.99 [−1.42, −0.55] | +0.59 [0.43, 0.74] |
  | Sentence added to the aligned prompt | 1.09 | 0.06 |

  For Qwen, on average, mentioning the object does most of the work. At step 0, where Qwen's effect is 11.0 nats, the sentence accounts for 4.6 and the goal role is +2.8 (**post hoc**). For Llama about a third of the effect depends on the object being the goal. That part is concentrated in take actions and after step 0.
- **2×2 design** (1,076 move, heat, cool and clean items). The part due to usefulness is 0.72 for Qwen and −0.18 for Llama.
- **Neutral task ("explore the room").** Against it, even the swapped task raises L, by 2.56 nats for Qwen and 1.20 for Llama.
- **Determined successes** (292 items; they are not built like the failures).
  - **Matched successes.** On the 37 that meet all of the failures' criteria (**post hoc**), the goal effect is as large as on failures: 10.3 nats [8.3, 12.8] for Qwen and 1.5 [1.1, 2.1] for Llama.
  - **Discrimination (AUC).** This measures how well L ranks working actions above failing ones. Moving from the neutral to the aligned task, Llama falls from 0.87 to 0.77 (−0.10 [−0.12, −0.08]) and Qwen from 0.93 to 0.91.
  - **Discrimination on the first-word term** (**post hoc**). Llama falls from 0.88 to 0.75 and Qwen from 0.99 to 0.97.
  - **Top-ranked candidate.** This is counted at the 181 states with both a working and a failing candidate. A failing action is ranked first at 19% of states under the aligned task and 9% under the neutral task for Llama, and 8% and 8% for Qwen.
- **Ignore-goal instruction.** "(The agent's goal does not change what happens. Predict the observation from the history and the action alone.)" removes 9% of Qwen's effect and none of Llama's.

## 2. Qwen3 thinking mode (`thinking_probe`, `thinking_analysis`)

Qwen3-8B reasoned before answering. The run used 1,000 sampled items and five task lines: aligned, swapped, neutral, and both described goals. Sampling followed Qwen's recommended settings, with the reasoning capped at 3,072 tokens; 2.6–7.5% of replies hit the cap and were forced to answer, and leaving them out changes nothing. L was scored given the model's own reasoning.

- **Smaller gap.** The aligned-vs-swapped effect is 4.01 nats [2.87, 5.12] with reasoning, against 6.06 [5.31, 6.76] with thinking off on the same items.
- **But not less lift.** Relative to the neutral task, the aligned task raises L about equally: 9.0 nats with reasoning and 8.6 without. The swapped task's lift doubles, from 2.5 to 5.0. Reasoning spreads the lift to other tasks; it does not remove it.
- **Answers off target.** In 14% of answers the reasoning drifted to another object or to a few-shot example. On the answers that stayed on target (**post hoc**), the pattern holds.
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
  - **Yes-share.** Qwen almost never says yes to these failures (0.2% under the aligned task). Llama's yes-share shifts by 16 points: 40 at step 0 and 13 afterwards (**post hoc**).
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
