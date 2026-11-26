# Post-training pilot: results (2026-11-26)

**Status.** Exploratory, with a frozen procedure. The design is `notes/posttraining_design.md`: frozen at tag `posttraining-design-frozen-2026-11-26`, with deviations D-PT1 and D-PT2 recorded before any score was read. The analysis ran once (CPU job 7364918), from a clean checkout at tag `posttraining-deviation-dpt2-2026-11-26` (26d5ac1). It wrote `notes/posttraining_metrics.json` (sha256 57bd41ff4f080c0dd26a8cdb1e303139f40e38c5e5635a982331a3899b167252). The environment check found no problems: one stack, L40S for every main and bridge arm, A40 for every floor arm, pinned revisions, identical token ids across each lineage, and identical scorer files at every arm's commit.

**The Llama lineage's base model failed its competence gate.** The frozen rules therefore apply the seven-example fallback to the Llama tree, and the design says to stop and report before running it, so this note reports the first part of the study. The Llama-tree stage comparisons in the shared plain-text format are **not read**: they would have to come from the fallback's seven-example arms, which have not been scored. The numbers the analysis computed for them from the three-example arms are in the metrics file but are not readings. The Qwen pair, the chat-template comparisons among the post-trained Llama-tree checkpoints, and the Word2World-format bridge are reported below. The bridge and the template comparisons are descriptive by design.

## 1. What was done, and what it asks

**Setup.** A language model is asked to predict how ALFWorld responds to an action whose failure the earlier history already settles, for example `take saltshaker 1 from shelf 2` when the history has shown that shelf 2 holds no saltshaker. The prompt also carries the agent's task line. The **goal effect** of a set of $n$ such items is the mean change in the model's log-odds of success when the task line names the action's object (the real task) instead of another object in the room (the swapped task):

$$G = \frac{1}{n}\sum_{i=1}^{n}\big[L_i(\text{real task}) - L_i(\text{swapped task})\big].$$

A simulator that kept the agent's intention out of its prediction of the dynamics would show little or none of this. The question is how post-training changes the effect: whether it introduces it, amplifies it, suppresses it, changes what it is made of, or moves its parts in opposite directions. No direction was assumed.

**Checkpoints.**

- *The Llama-3.1-8B tree:* the pretrained base; AI2's Tülu 3 after supervised fine-tuning (SFT), after preference optimization (DPO), and after reinforcement learning with PPO; Tülu 3.1, a second RL run with GRPO from the same DPO checkpoint; Meta's Llama-3.1-8B-Instruct, a sibling post-training of the same base; and the Word2World ALFWorld model, a domain-specific SFT of the same base, used in the bridge only.
- *The Qwen pair:* Qwen3-8B-Base and the released Qwen3-8B.

**Formats.**

- *Plain text (primary).* The study's few-shot prompt with no chat template, used for every checkpoint.
- *Each post-trained checkpoint's own chat template (secondary).* For Qwen3-8B this is the thinking-off template with its empty think block.
- *The Word2World message format (the bridge).* The room's full state, no few-shot examples, descriptive only.

**Items and task lines.** The items are 300 later-step failures and 150 step-0 failures, plus 292 actions known to succeed. Each failure is scored under twelve task lines:

- the real task, the swapped task, and a neutral task ("explore the room");
- four sentences that name the object without making it the goal, each paired with a twin that names an absent control object instead:
  - an assertion that the object is in the room;
  - a denial;
  - the bare name, listed with no claim;
  - "Ignore the X";
- the real task with the object described instead of named.

**Readouts.** All three are teacher-forced on the environment's reply.

- *First-token decision:* the log-probability of the success text's first token (" You") minus that of " Nothing".
- *Full two-candidate score:* the log-probability of the success text minus that of "Nothing happens.".
- *Continuation:* the full score minus the first-token decision.

Yes/no answers are recorded only for the post-trained checkpoints in their templates.

**Scales.** Every goal effect is reported on four scales: nats; the standardized effect $Z$ (the mean item effect over its item standard deviation); the net share of items moved; and the within-item rank gain. A sentence $c$'s **share** of the goal effect is its net effect over its control, divided by $G$:

$$s(c) = \frac{\Delta(c)}{G}, \qquad \Delta(c) = \frac{1}{n}\sum_{i}\big[L_i(c) - L_i(\text{control of } c)\big].$$

Two summary indices describe the decision token. **Content sensitivity** is the gap between asserting and denying:

$$C = s^{\text{first}}(\text{assertion}) - s^{\text{first}}(\text{denial}).$$

The **content-blind lift** $B$ is the listed word's share. The **decision-token share** is the part of the full goal effect carried by the first token:

$$\pi = G^{\text{first}}/G^{\text{full}}.$$

All intervals are 95% game-clustered bootstrap intervals (10,000 resamples). The pre-specified rules (design section 8) and a fixed Holm family of 35 readings decide the primary readings. Everything else is descriptive.

## 2. Competence gates (neutral task only)

Each checkpoint and format must separate the successes from the later-step failures under the neutral task, which reads no goal contrast. It needs an area under the ROC curve (AUC) of at least 0.70 on the full score, no degenerate answering, and a median first-token mass of at least 0.7.

| Checkpoint, format | AUC, full score | AUC, first token | Median first-token mass | Interpretable |
|---|---|---|---|---|
| Llama-3.1-8B base, plain | 0.54 | 0.50 | 0.92 | **no** |
| Tülu SFT, plain | 0.73 | 0.70 (just below) | 0.91 | full score and continuation only |
| Tülu DPO, plain | 0.71 | 0.70 (just below) | 0.95 | full score and continuation only |
| Tülu RL (PPO), plain | 0.70 | 0.69 | 0.96 | full score and continuation only |
| Tülu 3.1 (GRPO), plain | 0.69 | 0.65 | 0.95 | no |
| Meta Instruct, plain | 0.83 | 0.84 | 0.87 | yes |
| Qwen3-8B-Base, plain | 0.85 | 0.88 | 0.93 | yes |
| Qwen3-8B, plain | 0.93 | 0.98 | 0.98 | yes |
| Tülu SFT, DPO, RL, GRPO, own template | 0.77–0.78 | 0.79–0.82 | 0.93–0.96 | yes (Tülu RL's first token falls on its degeneracy test) |
| Meta Instruct, own template | 0.88 | 0.88 | 0.84 | yes |
| Qwen3-8B, with the think block | 0.93 | 1.00 | 1.00 | yes |

**What the base's failure shows.** In the three-example plain format, the Llama-3.1-8B base puts nearly all its probability on the two reply words (mass 0.92). But it does not tell working actions from failing ones: the AUC is 0.54 on the full score and 0.50 on the first token, and its first-token decision favours success on 86% of the failures. The base therefore cannot be read in this format. This is not evidence that it lacks the goal dependence: a model that does not use the history cannot show whether the task line moves a history-settled prediction.

The Tülu checkpoints pass in plain text only barely, or not at all for GRPO, while every post-trained checkpoint passes comfortably in its own template. In this format, the Llama tree's competence is itself something post-training changes, through the template it trains on as much as through the task.

**The consequence.** The pre-specified fallback rescores every plain-format Llama-tree checkpoint with seven examples and reads only those arms. It is costed in section 6 and has not been run.

## 3. Qwen3-8B-Base against Qwen3-8B (primary, plain text, later steps)

Both checkpoints are interpretable on every readout. The described-goal contrast uses the 297 of 300 later items whose object both checkpoints recognise from its description.

| Quantity | Qwen3-8B-Base | Qwen3-8B | Change |
|---|---|---|---|
| Goal effect, first token, nats | 0.20 [0.14, 0.27] | 1.40 [1.24, 1.56] | +1.20 [1.04, 1.36] |
| Goal effect, first token, standardized | 0.47 [0.30, 0.66] | 1.29 [1.11, 1.51] | +0.82 [0.58, 1.08] |
| Goal effect, full score, nats | 0.44 [0.35, 0.54] | 3.59 [3.17, 4.01] | +3.15 [2.75, 3.54] |
| Goal effect, full score, standardized | 0.67 [0.52, 0.85] | 1.32 [1.12, 1.57] | +0.65 [0.41, 0.91] |
| Net share of items moved (first / full) | 0.42 / 0.57 | 0.82 / 0.90 | +0.40 / +0.33 |
| Rank gain among the three task lines (first / full) | 0.22 / 0.31 | 0.48 / 0.57 | +0.26 / +0.26 |
| Content sensitivity $C$ (first token) | 3.14 [2.38, 4.62] | 1.41 [1.26, 1.60] | −1.72 [−3.17, −1.00] |
| Content-blind lift $B$ (listed word, first token) | 1.36 [1.04, 1.91] | 0.95 [0.87, 1.04] | −0.41 [−0.95, −0.10] |
| Decision-token share $\pi$ | 0.46 [0.38, 0.52] | 0.39 [0.36, 0.42] | −0.07 [−0.14, 0.01] |

**Pre-specified readings** (Holm-corrected within the family of 35):

- *Standardized goal effect, first token:* **higher** after post-training. The change in nats, in standardized units, in the net share moved and in the rank gain all agree in sign, and the change clears the 0.15 margin and five numerical floors; Holm-adjusted p < 0.001.
- *Standardized goal effect, full score:* **higher**, on the same criteria. It is labelled *environment-dependent*: on the A40 rescoring, the base's standardized full-score effect moved by 0.046, more than a fifth of the margin. The direction does not depend on this; the size of the change is less certain.
- *Content sensitivity $C$:* **lower** after post-training (Holm-adjusted p < 0.001).
- *Content-blind lift $B$:* **inconclusive**. Its interval excludes zero, but the Holm-adjusted p is 0.11.
- *Decision-token share $\pi$:* **inconclusive**. The interval reaches the margin, and the margin is under five floors in share units.

**What this means.** In this pair the dependence exists before post-training: the base's standardized goal effect is clearly above zero, at 0.47 on the first token and 0.67 on the full score. Post-training (strong-to-weak distillation, for Qwen3-8B) amplifies it on every scale. It also changes its character.

- *The base's decision token follows content.* Asserting the object lifts it by 0.20 nats, about the goal effect itself. Denying it lowers it by 0.44 nats, so the denial's share is −2.18. But the base also lifts on the bare listed word by 0.28 nats and on "Ignore the X" by 0.15 nats. So the base already has a content-blind lift, beside a strong response to negation.
- *After post-training, every naming sentence lifts the decision token by 1.2 to 1.8 nats,* while the denial moves it by only −0.18 nats [−0.35, −0.01]. The response to the object's name grows several-fold, and the push-down from a denial nearly disappears. On the full score the denial still lowers the reading (−1.69 nats), mostly through the continuation.
- *Relative to the goal effect,* the decision token becomes less content-sensitive ($C$ falls from 3.1 to 1.4). Whether the content-blind lift itself changed is not decided ($B$ inconclusive).

**What this does not mean.** It is one size of one family, and Qwen's intermediate stages are not released, so nothing locates the change at a stage. The base and the released model differ in data and objective together. "Less content-sensitive" describes how the readout responds to these sentences; it is not a claim about the computation. The comparison also does not show what Qwen3-8B does in its own think-block format, which the next subsection describes.

**The think block (descriptive).** Qwen3-8B scored with its empty think block has a smaller standardized goal effect on the first token than in plain text: the difference is −0.44 [−0.62, −0.28], although it is larger in nats. With the block, a denial *raises* the decision token by +1.41 nats [1.07, 1.73], where in plain text it moved it by −0.18. This matches the earlier mention investigation's finding that Qwen's decision readout, with the block, rises on any sentence naming the object, including denials. Here that finding is seen in the same run, beside the plain-text arm, which does not show it. So the denial pattern is tied to the think-block format. The increased lift on mentions in general is already present in plain text.

**Step 0** (150 items, descriptive) shows the same directions: the standardized goal effect is higher after post-training (+0.96 first, +0.88 full), and $C$ is lower.

## 4. Post-trained Llama-tree checkpoints in their own chat templates (secondary, descriptive, later steps)

Every comparison here is between post-trained checkpoints in their own templates, which the fallback does not touch. No Holm correction applies; these are descriptions with intervals.

| Checkpoint (own template) | Goal effect, first token: nats ($Z$) | Continuation: nats ($Z$) | Full score: nats ($Z$) | Denial share, first token | "Ignore the X" share: first / continuation |
|---|---|---|---|---|---|
| Tülu SFT | 0.16 (0.42) | −0.03 (−0.09) | 0.13 (0.27) | −3.93 | −0.62 / not defined |
| Tülu DPO | 0.66 (0.77) | −0.17 (−0.13) | 0.49 (0.35) | −1.85 | −0.55 / not defined |
| Tülu RL (PPO) | not interpretable (gate) | −0.18 (−0.13) | 0.74 (0.46) | not interpretable (gate) | not interpretable / not defined |
| Tülu 3.1 (GRPO) | 0.48 (0.71) | −0.20 (−0.21) | 0.28 (0.24) | −2.42 | −0.34 / not defined |
| Meta Instruct | 0.64 (0.67) | 1.12 (0.69) | 1.76 (0.93) | −0.66 | −0.12 / 0.68 |

"Not defined" marks a share whose denominator, the continuation's goal effect, is about zero. Tülu RL's first token fails its degeneracy test in this template: under the neutral task it favours success on 98% of the failures (AUC 0.82), so its first-token values are not read, although its full score is. The table's numbers come from `notes/posttraining_metrics.json`, under `arms`, `later`.

**What stands out.**

- *Across Tülu's stages, the goal effect grows with SFT to DPO.* On the first token the standardized change is +0.35 [0.26, 0.44]. On the full score, the change is +0.08 [−0.00, 0.16] from SFT to DPO and +0.12 [0.08, 0.15] from DPO to PPO. Tülu RL's first token cannot be compared, because it fails its gate. GRPO from the same DPO checkpoint moves the effect slightly the other way (first token −0.06 [−0.12, −0.00], full −0.10 [−0.16, −0.04]). At every stage whose first token is interpretable, the denial keeps a strong negative share, so Tülu's decision token stays content-sensitive.
- *Tülu's continuation carries no goal effect,* or a slightly negative one. Meta Instruct's continuation carries more than half of its goal effect, and in its continuation "Ignore the X" lifts by 0.68 of that effect, while its decision token does not (−0.12). This is the pattern the mention investigation described for Llama-3.1-8B-Instruct: the continuation lifts on any mention while the decision token tracks content. It appears in Meta's post-training of this base and not in AI2's at any released stage. In these templates, then, that pattern belongs to a recipe, not to post-training of this base in general.
- *Tülu's final model against Meta Instruct:* the full-score standardized goal effect is higher for Meta (+0.47 [0.32, 0.62]), carried by the continuation (+0.82 [0.68, 0.98]). Tülu RL's first token is not interpretable, so the decision tokens are not compared.

**What this does not mean.** The two recipes use different templates (Tülu's role tags with no begin-of-text token; Meta's headers), so the recipe contrast includes a format contrast. None of these are primary readings.

## 5. The Word2World-format bridge (descriptive, later steps)

In the full-state format the Word2World model was trained on, every checkpoint discriminates (AUC 0.76–0.81 under the swapped task; Word2World itself 0.99).

- *First-token standardized goal effect:* about zero for the Llama base (0.03 [−0.11, 0.18]), 0.41–0.57 for Tülu's stages and Meta Instruct, and slightly negative for the Word2World model (−0.19 [−0.29, −0.08]).
- *Continuation:* negative for the base and the Tülu stages (−0.32 to −0.47).
- *Full score:* −0.19 for the base, 0.07–0.21 for Tülu's stages, 0.41 for Meta Instruct, and −0.17 for Word2World.

So, in this one format, the base predicts the dynamics without a goal effect on its decision. General post-training (both AI2's and Meta's) introduces a decision-token goal effect. Domain SFT on ALFWorld transitions does not, and shows a small reversal, as `rq1_score` found earlier. This format has no few-shot examples, and these readings carry no pre-specified rule.

## 6. What is not yet known, and the fallback

**The Llama tree in the shared plain format.** This was the core question, base → SFT → DPO → RL and the sibling recipes. It is not answered, because the base cannot do the task with three examples. The fixed fallback rescores the six plain-format Llama-tree checkpoints with seven examples (sets 1 and 2), with their floor rescorings. At the smoke's measured rates, with the longer prompts, that is about 2.0 L40S-hours plus about 1.3 A40-hours for floors plus loading: **about 3.5 to 4 GPU-hours.** The pilot has used about 8.7 GPU-hours (section 7), so this would exceed the approved 7.5–9. The design says to stop and report here.

A cheaper staged route, if it is approved, keeps the frozen rule unchanged:

1. Score the base's seven-example arm first (about 0.4 GPU-hours).
2. If the base fails its gate again, the lineage is "not interpretable in this format" and nothing more is run.
3. If it passes, score the other five checkpoints and the floors.

**Provenance comparison with lexsem.** It is fixed in design section 17 (every shared cell is expected to match bit for bit), and it waits for lexsem's scores (queued). It does not gate anything here.

## 7. Compute

All jobs ran in the serve2 stack.

| Part | GPU-hours |
|---|---|
| Smoke (10 jobs) | about 0.35 |
| Main and bridge scoring, L40S (9 jobs, 7349816–7349824) | 4.32 |
| Floors, A40 (8 jobs, 7349825–7349832) | 2.28 |
| D-PT2 rescoring of the five reuse arms, L40S (7358803, 7358804, 7358806, 7358808) | 1.77 |
| **Total** | **about 8.7** |

D-PT2 cost more than its 1.1–1.3 estimate because the Qwen job scored both of its arms in full.

## 8. Limits

- One lineage with released intermediate stages, and that lineage is not yet readable in the primary format. The Qwen pair has only its two ends.
- Every stage changes data and objective together. "Base" means the released base checkpoint.
- Plain text is off-distribution for post-trained models. The template arms show this matters: the Tülu checkpoints only barely pass competence in plain text.
- The readouts describe how predictions respond to sentences. They are not a claim about internal computation, and rejecting a pure account identifies neither a mechanism nor a mixture.
- Comparisons among sentences are relative to the goal effect, and the normative value of a log-odds shift is not zero for a simulator with residual doubt.
