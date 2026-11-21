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
  - On the yes/no scale the effect is +2.93 [+2.54, +3.30] nats, the same as ALFWorld's 2.80.
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

- **The effect is not specific to ALFWorld for Qwen3-8B.** In an environment with no hidden object locations, where failures follow from counting, naming the goal still raises predicted success for actions the history has already settled. That includes actions whose failure only arithmetic settles. On the yes/no scale it does so by the same amount as on ALFWorld.
- **Llama's ALFWorld effect does not carry over.** Llama-3.1-8B's ALFWorld effect (1.65 nats two-candidate, 2.21 yes/no) is absent in TextCraft, replaced by a small reversed effect. That reversal comes from a model that does not track inventory arithmetic here (used-up AUC 0.31; counting AUC 0.42). The pre-registered reading is "naming the goal makes the model more careful". Given the weak competence, we do not read it as a working state judgement being shifted. This qualification is our interpretation, not a pre-registered reading: D7 only qualifies a *confirmed* effect when competence fails, and the same logic applies to a negative one.
- **Why the ALFWorld–TextCraft difference for Llama cannot be pinned down.** As section 13, D7 lists, TextCraft differs from ALFWorld in four ways a cross-environment difference cannot separate:
  - no hidden state;
  - couples remove history–goal agreement;
  - the swapped goal's recipe tree is on screen;
  - the reply formats differ.
- **Scoring runs.** One scoring run per model (see the multi-seed note in the decision log). Greedy, teacher-forced scoring has little run-to-run variation, but other sources (prompt examples, item draws) are not varied here.
- **Follow-up.** The exploratory prompt variants (section 13, D10: examples on goal; empty start) were scored after this analysis. They, and an exploratory investigation of the goal-completion cell, are reported separately (branch `textcraft-llama-dx`). Nothing exploratory replaces the confirmatory outcome above.

## Follow-up analyses (exploratory, 2026-11-21)

Everything in this section was done after the confirmatory result above had been read. None of it changes that result. Details, dated plans and all numbers are on two branches: `textcraft-llama-dx` (`notes/textcraft_llama_dx.md`) and `xenv-compare` (`notes/textcraft_vs_alfworld.md`).

**Qwen's smaller effect is mostly a matter of scale and items.** The two-candidate ΔL falls from 6.12 nats in ALFWorld to 0.94 here, but 59% of ALFWorld's figure is the goal raising the probability of the words *after* "You" in ALFWorld's success sentence. TextCraft's two replies name the same item, so this term is zero by design. ALFWorld's largest effects are also at step 0, which the main set lacks. On rulers that do not depend on the reply wording the two environments agree: the yes/no effect is 2.92 against 2.80 (ratio 1.04 [0.89, 1.21]), and failures flipped to predicted success are 5.6% against 5.9%. A residual factor of about two remains on the first token at later steps.

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

**Llama's competence in TextCraft, and what it implies.** Competence here is the AUC under the neutral goal line, the probability that the model scores a real success above a real failure (0.5 is chance).

- *The figures.*
  - Overall: 0.68 [0.65, 0.72], against 0.87 for Qwen.
  - By failure type: 0.86 where an input was never obtained, 0.67 where too few are held, and 0.31 where an input was used up earlier. The last is below chance: Llama ranks used-up failures above real successes.
  - By action type: 0.84 on goal-item crafts, 0.58 on ingredient crafts.
  - On counting items: 0.42, so the counting readout was withheld by rule.
- *A lean toward failure.* Its mean L is −2.98 on failures and −0.33 on successes. On the yes/no question it answers "Yes" to 9.9% of failing and 11.3% of working main actions.
- *Its free-text predictions.* These say failure for about 80% of goal crafts that succeed.
- *Without the prompt examples.* Its AUC falls to 0.43.
- *With the inventory shown.* Its AUC rises only modestly on a sample (0.54 to 0.62), and used-up failures stay below chance (0.35).
- *Against ALFWorld.* There the same model discriminates well (AUC 0.97 on yes/no).

Implications, all descriptive:

- **The confirmatory reading stands.** By the pre-registered TextCraft gate, which asks that the AUC interval exclude 0.5, Llama is competent, so its row reads "negative". Under the stricter rule used in the model-families study (AUC at least 0.70), it would be classed as not competent. That is a disclosed difference between the two studies' rules, not a change of verdict.
- **The absent positive effect fits the cross-model pattern.** In ALFWorld, every model that can do the task shows an effect and the only near-zero effects come from models that cannot (`textcraft_vs_alfworld.md`, section 9). Llama's absent positive effect on ingredient crafts, where it barely discriminates, fits that pattern.
- **The goal-craft negative is not explained by incompetence.** It sits where Llama discriminates well.
- **Round 2 fixes one rule in advance.** It applies the families rule to every model before any data.
- **More to come.** A fuller analysis of what Llama does instead of tracking the inventory is in `notes/textcraft_llama_competence.md` on branch `textcraft-llama-dx`.

**The prompt matters.** The verified system text quotes the original agent's instructions ("Every round I will give you an observation, you have to respond an action…"). Removing them, with the world-model framing and the environment's rules kept, raises the goal-role effect by +0.39 for Llama and +1.66 for Qwen. Competence is left about unchanged (−0.016 and +0.002 AUC). From now on TextCraft runs use this clean system text, after the tokenization and golden-prompt checks pass. Results under it are a different quantity from the confirmatory one and are reported as such.

**Next.** A pre-registered second round (branch `textcraft-r2`) runs under the clean prompt and freezes predictions from the two-component account before any score:

- **Models.** Gemma-2-9B, Mistral-7B and Llama-3.2-3B, with Qwen and Llama rescored.
- **Separating naming from completion.** Arms test this by making the goal craft an intermediate step, or by naming two items.
- **Timing against content.** Histories are padded with steps that change nothing.
