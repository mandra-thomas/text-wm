# TextCraft replication: results (2026-11-20)

Confirmatory for the primary test, which was frozen before scoring at tag `textcraft-design-frozen-2026-11-20` (commit `5a3702a`; design in [textcraft_design.md](textcraft_design.md), sections 1, 12 and 13). Scoring run `textcraft_score` (4 shards, ICE jobs 7204417–7204420, 51,200 scores, none missing) and analysis run `textcraft_analysis` (ICE job 7206143), both at `5a3702a`. Every number below is from [textcraft_metrics.json](textcraft_metrics.json).

## Question

On ALFWorld, naming the action's object in the task line (aligned) rather than another object (swapped) raises a prompted world model's predicted success for actions whose failure the history already settles. Does the same goal effect appear in TextCraft, where failures follow from inventory arithmetic rather than object location? The estimand is ΔL, the mean over the 1,500 main items (750 same-history couples, 229 goal pairs) of the aligned minus swapped log-odds of the success reply against the failure reply. A world model that reads the history correctly gives ΔL = 0.

## Primary result

The rule fixed in advance: confirmed for a model when ΔL > 0, the 95% interval clustered by goal pair excludes zero, and the goal-pair permutation p, Holm-adjusted over the two models, is below 0.01. The replication holds only if it is confirmed for both models.

| Model | ΔL (nats), 95% CI | Holm p | Competence (neutral AUC, 95% CI) | Reading (section 13, D7) |
|---|---|---|---|---|
| Qwen3-8B (thinking off) | +0.94 [+0.63, +1.24] | 0.002 | 0.87 [0.85, 0.89] | Confirmed: the goal effect replicates |
| Llama-3.1-8B-Instruct | −0.19 [−0.30, −0.09] | 0.002 | 0.68 [0.65, 0.71] | Negative: naming the goal makes the model more careful |

**The replication does not hold across models.** It is confirmed for Qwen3-8B only. Under the pre-registered rule that is reported as "confirmed for one model only", not as a replication.

## Qwen3-8B

- **Robustness.** The coherence-robust estimate is +1.17 [+0.71, +1.64]. On tie couples, where the history favours neither goal, it is +0.99 [+0.12, +2.03]. Equivalence to zero is rejected on all three scales.
- **Size against ALFWorld.**
  - On the yes/no scale the effect is +2.93 [+2.54, +3.30] nats. *(This number should not be compared with ALFWorld's 2.80: TextCraft's yes/no question is not anchored to the environment's reply and partly reads as task relevance; see "The yes/no question" under the follow-up analyses.)*
  - On the two-candidate scale it is about a sixth of ALFWorld's 6.12.
  - Standardized by the item SD of the difference it is 0.20, against about 1.0 on ALFWorld.
  - The two-candidate scale depends on each environment's reply texts (section 13, D5), so the three scales are reported together.
- **Named readouts.** These are Holm-adjusted within the model.
  - Goal role, where the action names the goal: +1.04 [+0.51, +1.60].
  - Ingredient sharing a word with its goal: +1.26 [+0.58, +1.93].
  - Counting, the items whose failure only inventory arithmetic settles: +0.98 [+0.52, +1.42]. Its competence gate passes, with an AUC of 0.81 on its own items.
  - Yes/no: +2.93.
  - All four are confirmed. Usefulness only (+0.56 [+0.10, +1.05], 102 items) is not confirmed after Holm (p = 0.076); this readout can detect only a large effect (D1).
- **Where the shift sits.** The first-token readout carries the effect (+0.97), and the name-dependent remainder of the reply is near zero (−0.01). The goal line moves the predicted outcome, not just the copying of a name.
- **Successes.** On determined successes, ΔL is +2.61 [+1.82, +3.38].

## Llama-3.1-8B-Instruct

- **The effect is small and negative.** It is −0.19 on the two-candidate scale and −0.15 [−0.21, −0.09] on yes/no. It is concentrated in the goal role (−0.63 [−0.80, −0.47]). Ingredient with overlap (+0.05), usefulness only (−0.16) and counting (0.00) are not significant.
- **Equivalence.**
  - On the two-candidate scale the 90% interval lies inside the ±0.3 margin.
  - On the yes/no scale it lies inside ±0.37.
  - On the standardized scale it does not (−0.14 against a ±0.14 margin).
  - The negative row of the outcome table is the reading because its condition is met: the estimate and the 95% interval are below zero and the Holm p is below 0.01.
- **Competence is weak.** Overall neutral AUC is 0.68, which passes the gate because its interval excludes 0.5.
  - The model separates failures where the item was never obtained (AUC 0.86).
  - It barely separates "too few" (0.67).
  - It ranks "used up" failures *above* real successes (AUC 0.31).
  - Its mean L is −2.98 on failures and −0.33 on successes, so it leans toward failure for everything.
  - On the counting items it does not tell successes from failures (AUC 0.42), so the counting readout is withheld by rule.
- **Both task lines raise L over the neutral line.** Aligned is +0.20 over neutral and swapped is +0.39, so the negative ΔL is "swapped raises it more", not "aligned lowers it". In TextCraft the swapped goal is the other displayed task, with its recipe tree on screen.

## Where the effect sits: role × failure class (descriptive strata)

Mean ΔL in nats with 95% intervals clustered by goal pair, from `descriptive.main.strata.role_x_class`. These are descriptive strata of the analysis, with no p-values. "Goal" means the action crafts the task's final item; "ingredient" means it crafts an intermediate item.

| Cell (items) | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Goal, never obtained (252) | −0.45 [−0.88, −0.06] | −0.75 [−0.92, −0.57] |
| Goal, too few (252) | +2.52 [+1.56, +3.54] | −0.52 [−0.78, −0.26] |
| Ingredient, never obtained (248) | +1.64 [+1.06, +2.21] | −0.11 [−0.29, +0.07] |
| Ingredient, too few (248) | +2.28 [+1.26, +3.53] | +0.27 [−0.02, +0.52] |
| Ingredient, used up (494) | −0.20 [−0.58, +0.21] | −0.02 [−0.11, +0.08] |

Goal × used up has only 6 items (one pair) and is omitted.

**Llama's negative effect is entirely in goal-completing crafts.** It is −0.63 [−0.79, −0.47] on the 510 goal-role items and +0.03 [−0.07, +0.12] on the 990 ingredient items; weighted by item counts, these reconstruct the overall −0.19. It is largest where an ingredient was never obtained, which is the class Llama discriminates best (AUC 0.86). Weak inventory accounting therefore cannot by itself explain the reversal.

**Qwen is negative in the same cell.** Crafting the goal item when an ingredient was never obtained gives −0.45 for Qwen as well. Qwen's positive total comes from the "too few" cells and from ingredient crafts. Neither model shows an effect on "used up" ingredient items.

**The reversal is small in practice.** Llama's mean two-candidate score shift is +0.38 percentage points [−0.54, +1.35], against +3.59 [+2.30, +4.85] for Qwen. The 90% intervals of its log-odds effect lie inside the pre-set equivalence margins on the two-candidate (±0.3) and yes/no (±0.37) scales, though not on the standardized scale. What the data support is that Llama's ALFWorld effect does not appear in TextCraft and leaves a small negative residual. They do not support a robust negative task-relevance bias. Whether that residual holds up under other reasonable prompts is under exploratory study. One open question is the world-model system prompt itself: it quotes the original TextCraft agent instructions ("you have to respond an action…"), framed as the agent's rules.

**"More careful" is relative, not absolute.** The pre-registered label for Llama's row reads "naming the goal makes the model more careful". On the two-candidate scale, though, both named task lines raise Llama's L over the neutral line: aligned by +0.20 and swapped by +0.39. On the yes/no question, both lower it: aligned minus neutral is −0.60. The negative ΔL is a difference between two named tasks, not a drop in predicted success when the goal is named.

## Reading and limits

- **The effect is not specific to ALFWorld for Qwen3-8B.** In an environment with no hidden object locations, where failures follow from counting, the matching task line still scores actions the history has already settled as more likely to succeed than the mismatched line does. That includes actions whose failure only arithmetic settles. On readouts anchored to the environment's reply the effect is about a third to a half of ALFWorld's, and it is heavy-tailed. The earlier statement that the yes/no effect equals ALFWorld's is withdrawn (follow-up analyses, "The yes/no question").
- **Llama's ALFWorld effect does not carry over.** Llama-3.1-8B's ALFWorld effect (1.65 nats two-candidate, 2.21 yes/no) is absent in TextCraft, replaced by a small reversed effect. That reversal comes from a model that does not track inventory arithmetic here (used-up AUC 0.31; counting AUC 0.42). The pre-registered reading is "naming the goal makes the model more careful". Given the weak competence, we do not read it as a working state judgement being shifted. This qualification is our interpretation, not a pre-registered reading: D7 only qualifies a *confirmed* effect when competence fails, and the same logic applies to a negative one.
- **Why the ALFWorld–TextCraft difference for Llama cannot be pinned down.** As section 13, D7 lists, TextCraft differs from ALFWorld in four ways a cross-environment difference cannot separate:
  - no hidden state;
  - couples remove history–goal agreement;
  - the swapped goal's recipe tree is on screen;
  - the reply formats differ.
- **Scoring runs.** One scoring run per model (see the multi-seed note in the decision log). Greedy, teacher-forced scoring has little run-to-run variation, but other sources (prompt examples, item draws) are not varied here.
- **Follow-up.** The exploratory prompt variants (section 13, D10: examples on goal; empty start) were scored after this analysis. They, and an exploratory investigation of the goal-completion cell, are reported separately in [textcraft_llama_dx.md](textcraft_llama_dx.md). Nothing exploratory replaces the confirmatory outcome above.

## Follow-up analyses (exploratory, 2026-11-21)

Everything in this section was done after the confirmatory result above had been read. None of it changes that result. Details, dated plans and all numbers are in [textcraft_llama_dx.md](textcraft_llama_dx.md) and [textcraft_llama_competence.md](textcraft_llama_competence.md) (Llama diagnostics and competence), [textcraft_vs_alfworld.md](textcraft_vs_alfworld.md) (comparison with ALFWorld), and [textcraft_qwen_robustness.md](textcraft_qwen_robustness.md) and [textcraft_qwen_competence.md](textcraft_qwen_competence.md) (Qwen robustness and competence), each with its metrics file next to it.

**Qwen's effect in TextCraft is smaller than in ALFWorld, by about half on rulers anchored to the environment's reply.** *(Corrected 2026-11-21. An earlier version of this paragraph said the two environments "agree" on the yes/no question, 2.92 against 2.80. That comparison is withdrawn; see "The yes/no question" below.)*

- *The two-candidate drop is largely about the replies.* The two-candidate ΔL falls from 6.12 nats in ALFWorld to 0.94 here. Of ALFWorld's figure, 59% is the goal raising the probability of the words *after* "You" in ALFWorld's success sentence, and TextCraft's two replies name the same item, so that term is zero by design. ALFWorld's largest effects are also at step 0, which the main set lacks.
- *On the reply's first token, TextCraft is a third to a half of ALFWorld.* The first token is the model's predicted observation, "Craft" against "Could". The ratio is 0.39 overall, about 0.54 at matched history lengths and about 0.46 at a matched baseline ([textcraft_vs_alfworld.md](textcraft_vs_alfworld.md)).
- *The flip rates are similar.* Failures flipped to predicted success are 5.6% here against 5.9% in ALFWorld.

**The yes/no question.** TextCraft's question, "Will this action succeed? Answer Yes or No.", is not anchored to the environment's reply. ALFWorld's is: "…or will the environment reply 'Nothing happens.'?". Qwen's TextCraft yes/no answers partly judge whether the action serves the named task ([textcraft_qwen_robustness.md](textcraft_qwen_robustness.md), section 9).

- *At matched predicted observations, the swapped line shifts the answer.* At the same first-token score, the swapped line shifts the yes/no answer toward "No" by 2.0 nats on failures and 7.2 nats on successes.
- *The clearest case is working goal crafts.* On working goal crafts that Qwen's own predicted observation confidently calls successes, it answers "No" to 72% under the swapped line, against 13% aligned and 1% neutral.
- *About two-thirds of Qwen's yes/no effect on failures is not in its predicted observation.* The share is three-quarters on successes.
- *Llama shows no such component.*

TextCraft's confirmed yes/no readout (+2.93) is therefore partly a task-relevance judgement. It should not be compared with ALFWorld's yes/no effect, which tracks its predicted observation. Later TextCraft runs use a question anchored to the environment's replies.

**The effect has two components with opposite signs.**

- *A lift on intermediate crafts.* For actions that craft an ingredient of the goal, Qwen's ΔL is about +0.9; Llama's is +0.03 [−0.07, +0.13].
- *A penalty on crafting the named goal.* Goal and ingredient items were matched on failure class, history length and recipe structure. Goal minus ingredient is −0.80 [−1.05, −0.53] for Llama and −0.82 [−1.69, +0.09] for Qwen.
  - The penalty is largest when none of the goal's inputs is held and fades as they accumulate: Llama −0.85, −0.75 and −0.27 by share held; Qwen −0.16, +0.27 and +2.93.
  - It follows whichever item is named as the goal. Scoring an ingredient craft of Y under "Goal: craft Y." lowers Llama's L by −1.40 on failures and −1.74 on successes.

Llama's overall negative is this penalty without the lift. Qwen's positive total is the lift outweighing it. In TextCraft, naming an item and making its craft the task-completing step always coincide, so the penalty's cause, completion or naming, is open.

**Explanations not supported.**

- *Copying the prompt examples.* Moving the off-goal example shifts Llama's ΔL by −0.05. The reversal persists with no examples at all.
- *A competence floor.* The reversal sits where Llama discriminates well: goal-role AUC 0.84, and −0.76 on items it predicts correctly under the neutral line.
- *Doubt about the starting inventory.* Stating that it starts empty changes Qwen by −0.04 and Llama by +0.07.
- *Hidden state in general.* Showing the inventory raises competence (Qwen AUC 0.94) and changes neither model's ΔL. Qwen's effect therefore persists when nothing about the state is hidden.
- *Reply format.* The effect is decided at the first token, which is the same pair of tokens in every condition, and it appears on yes/no and in free generation.
- *A few influential pairs.* Clustering by recipe family keeps the interval below zero.

**Llama's practical size is negligible, and its overall negative is not robust.**

- The mean predicted chance of success moves by +0.38 points.
- Llama answers "Yes" to 0% of goal-role failures under both lines, and its free-text predictions say success for 2 of 120 against 5 of 120.
- The overall interval reaches zero when the empty start is stated (−0.11 [−0.23, +0.01]) and under the clean prompt with the inventory shown (−0.09 [−0.24, +0.06]).
- The goal-role negative persists under every prompt tried.

**Llama's competence in TextCraft, and what it implies.** Competence here is the AUC under the neutral goal line, the probability that the model scores a real success above a real failure (0.5 is chance). The full analysis is in [textcraft_llama_competence.md](textcraft_llama_competence.md). That note's heuristics were fixed and committed before they were fitted.

*How well Llama discriminates.*

- *Overall.* On the 1,500 main failures against the 600 successes, Llama's AUC is 0.61 [0.58, 0.64] on the two-candidate scale and 0.61 on yes/no. Qwen's is 0.86 and 0.85.
- *Why the confirmatory 0.68 is higher.* The confirmatory gate's figure of 0.68 includes the easier usefulness failures.
- *By role.*
  - On goal-item crafts Llama is decent: 0.84, and 0.90 where an input was never obtained.
  - On ingredient crafts it is near chance: 0.58, and 0.43 for used-up failures against ingredient successes.
  - The earlier figure of 0.31 for used-up failures compared them with all successes, half of which are goal crafts that Llama scores low whether they work or not. It mostly reflects that mix of roles.
- *By history length.* It discriminates well only at 1–3 steps (0.79), and at about 0.55 at 4–12 steps.

*How well Llama's predictions match outcomes.*

- *Accuracy.* Its balanced accuracy is 0.53 on the two-candidate scale and 0.51 on yes/no; Qwen's are 0.78 and 0.77. On yes/no Llama answers "Yes" to 9.9% of failing and 11.3% of working actions.
- *Calibration.* Above a predicted probability of 0.1, the observed success rate is flat. Its Brier score is 0.34, against 0.20 for always predicting the base rate.

*What Llama does instead of tracking the inventory.* Its scores follow simple cues better than the true outcome:

- "every input's name appears in the history": tracking AUC 0.87;
- "enough units were obtained at some point, ignoring later use, and goal crafts fail": agreement κ = 0.57 with its decisions;
- the true inventory rule: 0.61.

Given the cue, the true outcome adds nothing to Llama's score (slope −0.55 nats), while for Qwen it keeps a large weight (+12.6). Llama's scores fall with the number of missing units, but do not register that an input was used up unless an inventory reply shows it. With the inventory shown in the prompt, Llama still follows the cue (0.89) more than the truth (0.62).

*Against ALFWorld.* The same model discriminates well there (AUC 0.97 yes/no, Brier 0.04) and passes the families rule. What it lacks in TextCraft is inventory arithmetic over a history.

*Implications, all descriptive.*

- **The confirmatory verdict stands.** Llama passes the pre-registered TextCraft gate, which asks that the AUC interval exclude 0.5, so its row reads "negative". Under the stricter rule of the model-families study (AUC at least 0.70, yes/no format), it would be classed as not interpretable. That is a disclosed difference between the two studies' rules, not a change of verdict. The headline would still be "confirmed for Qwen3-8B only".
- **Llama's TextCraft result is not evidence about a working world model.** It should not be read as evidence about how a goal line biases a working world model, in either direction. Nor does it show that the ALFWorld finding fails to generalize, since the same model is competent in ALFWorld.
- **Competence is necessary for a positive effect here, but not sufficient.** No competence comes with no positive effect, which fits the cross-model pattern (`textcraft_vs_alfworld.md`, section 9). But in the one ingredient cell where Llama is competent (never obtained, AUC 0.86), its effect is still null (−0.11), where Qwen's is +1.64.
- **The goal-craft negative is not explained by incompetence.** It sits where Llama discriminates. Part of that discrimination is a prior that goal crafts fail: Llama predicts success for only 24% of goal crafts that work.
- **Round 2 applies the families rule to every model before any data.** Llama-3.1-8B is expected to fail it.

**How robust Qwen's effect is** ([textcraft_qwen_robustness.md](textcraft_qwen_robustness.md) and [textcraft_qwen_competence.md](textcraft_qwen_competence.md)).

- *The direction holds on every check.* An independent recompute gives +0.94 [+0.60, +1.27]. Dropping any one pair or recipe family leaves +0.85 to +1.06. The sign also holds under trimmed and winsorized means, on every reply-anchored readout, and under every prompt variant; the clean prompt raises the effect.
- *It is heavy-tailed.* The median item moves +0.25 [−0.00, +0.48] and the 10% trimmed mean is +0.48 [+0.24, +0.73]. The top 5% of items carry 80% of the total, but the interval reaches zero only after the 29 largest-contributing pairs of 229 are dropped.
- *It sits where Qwen's judgement is open.* That means crafts the history has not shown before, and too-few failures. On repeats of a craft the history already shows it is −0.33. Among ingredient crafts, it rests mostly on names that share a word with the goal: +0.18 [−0.40, +0.74] without any shared word.
- *Against the neutral line, much of it is a penalty under the wrong goal.* On failing actions, predicted success is 19.0% aligned, 16.5% neutral and 15.1% swapped, a difference of +3.9 points [+2.2, +5.5] between aligned and swapped. The penalty also reaches working actions. On working goal crafts, "Yes" on the yes/no question falls from 65% aligned to 14% swapped (156 of 300 flip), partly through the task-relevance reading described above.
- *Competence.* Qwen discriminates well (AUC 0.86) but is overconfident: 32% of items it scores above 0.9 fail. Its scores follow the "input names appear" cue (tracking 0.89) and treat repeated crafts as likely failures, while still giving the true outcome large weight. The goal effect is largest where Qwen is uncertain. On confident, correct failures it shifts log-odds (+0.68) but rarely changes the answer: 3.4% flip under the aligned line, against 0.9% under the swapped line.
- *How to describe it.* Qwen's TextCraft result is best described as a task-relevance bias between the matching and the mismatched task lines, in a competent but overconfident predictor. It is not well described as "naming the goal raises predicted success".

**The prompt matters.** The verified system text quotes the original agent's instructions ("Every round I will give you an observation, you have to respond an action…"). Removing them, with the world-model framing and the environment's rules kept, raises the goal-role effect by +0.39 for Llama and +1.66 for Qwen. Competence is left about unchanged (−0.016 and +0.002 AUC). From now on TextCraft runs use this clean system text, after the tokenization and golden-prompt checks pass. Results under it are a different quantity from the confirmatory one and are reported as such.

**Next.** A pre-registered second round (branch `textcraft-r2`) runs under the clean prompt and freezes predictions from the two-component account before any score:

- **Models.** Gemma-2-9B, Mistral-7B and Llama-3.2-3B, with Qwen and Llama rescored.
- **Separating naming from completion.** Arms test this by making the goal craft an intermediate step, or by naming two items.
- **Timing against content.** Histories are padded with steps that change nothing.
