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

## Reading and limits

- **The effect is not specific to ALFWorld for Qwen3-8B.** In an environment with no hidden object locations, where failures follow from counting, naming the goal still raises predicted success for actions the history has already settled. That includes actions whose failure only arithmetic settles. On the yes/no scale it does so by the same amount as on ALFWorld.
- **Llama's ALFWorld effect does not carry over.** Llama-3.1-8B's ALFWorld effect (1.65 nats two-candidate, 2.21 yes/no) is absent in TextCraft, replaced by a small reversed effect. That reversal comes from a model that does not track inventory arithmetic here (used-up AUC 0.31; counting AUC 0.42). The pre-registered reading is "naming the goal makes the model more careful". Given the weak competence, we do not read it as a working state judgement being shifted. This qualification is our interpretation, not a pre-registered reading: D7 only qualifies a *confirmed* effect when competence fails, and the same logic applies to a negative one.
- **Why the ALFWorld–TextCraft difference for Llama cannot be pinned down.** As section 13, D7 lists, TextCraft differs from ALFWorld in four ways a cross-environment difference cannot separate:
  - no hidden state;
  - couples remove history–goal agreement;
  - the swapped goal's recipe tree is on screen;
  - the reply formats differ.
- **Scoring runs.** One scoring run per model (see the multi-seed note in the decision log). Greedy, teacher-forced scoring has little run-to-run variation, but other sources (prompt examples, item draws) are not varied here.
- **Not yet scored.** The exploratory prompt variants (section 13, D10: examples on goal; empty start) were scored after this analysis and are read separately. They are exploratory.
