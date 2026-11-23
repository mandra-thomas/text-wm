# Every model scored so far: a cross-model summary (2026-11-22)

This note gathers, in one place, the results for every model the project has scored. The individual notes report each study in full. Most summaries so far led with the two reference models, Qwen3-8B and Llama-3.1-8B, because the preregistered test (RQ2) and the two replications (TextCraft, τ-bench) were designed around them. In ALFWorld, however, eleven models have been scored on the same items, and the decision simulation scored five. The results below are read across all of them. No verdict changes. Every number is quoted from the metrics files named in each section, and nothing here is new computation.

## 1. ALFWorld: the goal effect in eleven models

**What is measured.** The items are the 4,666 failing actions whose failure the history already settles (RQ2's primary items). The goal effect is the score under the aligned task minus the score under the swapped task, which differs only in naming another object. Two formats are used: two-candidate log-odds (the success reply against "Nothing happens.") and the yes/no question ("will this action succeed, or will the environment reply 'Nothing happens.'?").

**Five families.** These come from `families_results.md` and `families_metrics.json`. The competence rule asks four things of the neutral-task yes/no answers: an AUC of at least 0.70, Yes on at most 95% of failures, Yes on at least 5% of successes, and a median answer mass of at least 0.5.

| Model | Developer, size | Passes the competence rule (neutral yes/no AUC) | Share of failures pushed toward Yes | Yes/no effect, nats | Two-candidate effect, nats | Failures flipped from No to Yes |
|---|---|---|---|---|---|---|
| Qwen3-8B | Alibaba, 8B | yes (0.99) | 0.887 | 2.80 | 6.12 | 0.2% |
| Llama-3.1-8B-Instruct | Meta, 8B | yes (0.97) | 0.896 | 2.21 | 1.64 | 16.5% |
| Mistral-7B-Instruct-v0.3 | Mistral, 7B | yes (0.94) | 0.879 | 2.67 | 1.06 | 37.5% |
| Llama-3.2-3B-Instruct | Meta, 3B | yes (0.88) | 0.840 | 0.70 | 0.98 | 24.2% |
| Gemma-2-9B-it | Google, 9B | yes (0.99) | 0.748 | 1.23 | 0.80 | 3.0% |
| Llama-3.2-1B-Instruct | Meta, 1B | no (0.68) | 0.598 | 0.09 | 0.31 | 3.2% |

**Six Qwen3 sizes.** These come from `exploratory_results.md` section 6 and `size_v2_metrics.json`. "Sign changes" is the share of items whose first-token log-odds changes sign between the two tasks, in either direction.

| Qwen3 size | Can do the task | Yes/no effect, nats | Two-candidate effect, nats | First-token sign changes |
|---|---|---|---|---|
| 0.6B | no: answers Yes to almost every question | 0.12 | 0.89 | 0.6% |
| 1.7B | no: answers Yes to almost every question | 2.08 | 2.49 | 24.8% |
| 4B | yes | 2.27 | 7.49 | 19.2% |
| 8B | yes | 2.80 | 6.12 | 9.7% |
| 14B | yes | 3.18 | 4.33 | 21.3% |
| 32B | yes | 1.55 | 2.35 | 17.8% |

**Reading.**

- **The effect is present in every model that can do the task.** Eight models qualify: Qwen3 4B, 8B, 14B and 32B, Llama-3.1-8B, Mistral-7B, Llama-3.2-3B and Gemma-2-9B. They come from four developers and span 3B to 32B parameters. Every one shows a positive effect in both formats. The three models that cannot do the task lean the same way, but their results are not interpreted.
- **Two-candidate nats do not rank the models.** Qwen3-8B's two-candidate effect (6.12 nats) is the largest in nats, but that does not make its effect larger than the other models'. Qwen spreads its scores over a much wider range, and 59% of its 6.12 nats is the success sentence's tail; scaled by each model's own spread of item-level shifts, Qwen is close to the other competent models (section 5, from `qwen_effect_audit_forensic.md`). *(Updated 2026-11-23 from the exploratory forensic audit; the table's values are unchanged.)* About half of the 6.12 nats also depends on Qwen3's thinking-off prompt format: without the empty `<think></think>` block the template places before the answer, the effect is 3.31 nats, while the standardized effect and the share of items moved do not change (section 5, "How to compare"). *(Added 2026-11-23, exploratory follow-up.)*
- **Practical impact depends on calibration.** Qwen3-8B changes its answer on only 0.2% of failures, because it is very confident that they fail. Less confident models change many answers: Mistral 37.5%, Llama-3.2-3B 24.2%, Llama-3.1-8B 16.5%. A shift in log-odds is converted into changed decisions by the model's calibration, so a ranking by nats alone misstates where the effect matters most (`families_results.md`, "Practical size depends on calibration").
- **Scale does not remove it.** Among the competent Qwen3 sizes, the two-candidate effect in nats falls with size (7.5, 6.1, 4.3, 2.4), but the share of sign changes does not (19%, 10%, 21%, 18%), and the standardized effect is flat (0.90, 1.02, 0.96, 0.86; section 5). The fall in nats reflects how widely each size spreads its scores, not a weaker effect. *(Updated 2026-11-23, exploratory follow-up: these scores carry Qwen3's empty thinking-off block, which inflates each size's scale by a different amount. In plain text, on 300 items, the standardized series stays flat for 4B and 8B (1.05, 1.15) but not for 14B (0.72); 32B was not rescored. See section 5.)* Under the aligned task, Qwen3-32B answers Yes to 11.6% of failing actions, against 3.8% under the swapped task.
- **Limits.**
  - One scoring run per model.
  - Comparisons across families change the developer, data, size, tokenizer, template and post-training together, so they are descriptive.
  - All of these models were released in 2024–2025 and none is larger than 32B. Newer models are covered by the wave B design (section 4).

## 2. Decision simulation, Stage 1a: five verifier models

This comes from `decisionsim_results.md`. The two primary models' numbers are from `decisionsim_metrics.json`; the three secondary models' numbers are from `decisionsim_dx_metrics.json` (`frozen_functions_secondary_models`, the frozen functions applied post hoc), because `decisionsim_metrics.json` lists those models as still waiting. *(Citation corrected 2026-11-23; it named only `decisionsim_metrics.json`.)* A verifier asks the world model whether the agent's proposed action will succeed. If the answer is No, the agent takes the best-scored alternative from a fixed candidate menu. The comparison is the failure rate of the chosen action when the verifier sees the real task, against a verifier with the task sentence deleted ("task-blind").

| Verifier model | Extra failures, real task − task-blind | Gate category |
|---|---|---|
| Qwen3-8B (decides the gate) | +19.5 points [17.2, 22.3] | worth doing |
| Llama-3.2-3B (secondary) | +15.4 points | worth doing; its task-blind verifier is itself worse than no verifier |
| Gemma-2-9B (secondary) | +8.9 points | worth doing; the same two-part mechanism as Qwen3-8B |
| Mistral-7B (secondary) | +3.3 points | worth doing; 21 of its 24 net harmed states at step 0 |
| Llama-3.1-8B (primary) | +1.1 points [−0.1, 2.6] | no reliable change |

*(Corrected 2026-11-23: the Mistral row read "all at step 0"; by `decisionsim_dx_metrics.json`, 21 of its 24 harmed states are at step 0 and 3 at later steps, with no helped state.)*

Four of the five models can be pushed into choosing failing actions on these menus. With a fallback that contains no failing goal-object actions, all three secondary models drop to "no reliable change" or "too rare". In every model the size of the harm depends on the menu and on the decision rule (`decisionsim_dx.md`). *(Updated 2026-11-23, exploratory follow-up: for Qwen3-8B the step-0 part of the harm, about two thirds of its net, also depends on the prompt format. At the 134 step-0 states it is 71.6 points with Qwen3's empty thinking-off block before the answer and 38.8 points [30.6, 47.0] in plain text; the block roughly doubles it but does not create it (`decisionsim_results.md`, "Sensitivity of the step-0 harm to the prompt format").)*

## 3. Mechanism battery: three models

This comes from `mechanism2_results.md` and `mechanism2_metrics.json` (merged into the main line; design `mechanism2_design.md`, frozen at tag `mechanism2-design-frozen-2026-11-22`, commit `6eab8af`). Qwen3-8B and Llama-3.1-8B are the primary models; Mistral-7B-Instruct-v0.3 is scored as a secondary, descriptive model, as fixed in the design (section 11.9). The frozen rules give the same verdict for all three: no single account. Each candidate account is contradicted as the sole explanation, so the effect is a mixture or an account not on the list. Results are in [mechanism2_results.md](mechanism2_results.md), and the robustness deep-dive is in [mechanism2_dx.md](mechanism2_dx.md). The models differ in what moves them. For Qwen3-8B, adding "Ignore the {obj}." to the swapped task reproduces the whole goal effect (r 1.03 two-candidate). For Llama-3.1-8B a bare mention gives about a third of it. Describing the object instead of naming it keeps most or all of the effect in both models. All three models read an explicit claim about the object's location strongly in the stated direction. The user's hope barely matters for Qwen or Llama, while Mistral moves in the wishful direction. Qwen's yes/no answers stay "No" under every line, so its yes/no effects are shifts of confidence. Llama's Yes rate on these failures goes from 6.1% (swapped) to 22.4% (aligned). The key contrasts keep their sign across games, task types and formats.

## 4. What still rests on two models, and what is being added

The TextCraft and τ-bench results, and the reading of the effect as task relevance that they support, have so far been scored only on Qwen3-8B and Llama-3.1-8B. The following runs add models. *(Added 2026-11-23.)* The τ-bench authorization experiment (exploratory, frozen at tag `tau-auth-design-frozen-2026-11-22`; [tau_auth_results.md](tau_auth_results.md), metrics `tau_auth_metrics.json`) tested that reading directly on the same two models, adds none, and changes no verdict. For Qwen3-8B, which passed both gates, a backend description that states and demonstrates that the task never changes the output removes none of the request component (−0.09 nats [−0.42, +0.26] of +2.30) or of the naming component (−0.08 of +1.87): both are "relevance-like (reading (a))". For Llama-3.1-8B both categories are "inconclusive (gate)": it failed the task-blind comprehension check (question K2, 0 of 24) and its positive control was +0.30 nats [+0.19, +0.40], below the 1-nat threshold. *(Updated 2026-11-23: TextCraft round 2 is now scored for all five models; its row gives the result, and `textcraft_r2_results.md` reports every contrast for every model.)* *(Rows updated 2026-11-23 to match the README's status table; designs fixed at a tag are called "frozen".)* *(Updated 2026-11-23: the τ-bench families extension is scored. Llama-3.2-3B, the one interpretable new model, shows the τ-bench goal effect in the ALFWorld direction and, like both reference models, a shift shared by working and failing calls; Mistral-7B fails the competence rule on τ-bench, so every interpretable τ-bench model is still from Qwen or Meta. In nats Llama-3.2-3B's effect is a quarter of Qwen3-8B's, but the two move about the same share of items (0.82 and 0.78), which is the scale point of section 5. [tau_families_results.md](tau_families_results.md).)*

| Study | Models added | State |
|---|---|---|
| TextCraft round 2 | Gemma-2-9B, Mistral-7B, Llama-3.2-3B (Qwen3-8B and Llama-3.1-8B rescored under the clean prompt) | Done (frozen at tag `textcraft-r2-design-frozen-2026-11-23`, commit `4ccf84b`). Within-item penalty confirmed below zero in all three models that discriminate on goal crafting (Qwen3-8B −1.02, Llama-3.1-8B −1.68, Gemma-2-9B −1.27 nats): "penalty general across the competent models". Lift: Qwen3-8B +0.67 confirmed above zero, Llama-3.1-8B equivalent to zero (does not discriminate on ingredients), Gemma-2-9B −0.30 confirmed below zero. Mistral-7B and Llama-3.2-3B discriminate for no contrast. Results in `textcraft_r2_results.md` |
| τ-bench retail, model families | Mistral-7B, Llama-3.2-3B. Gemma-2-9B is not run: every prompt exceeds its 8,192-token context | Done (frozen at tag `tau-families-design-frozen-2026-11-23`, commit `a078a8d`). Llama-3.2-3B "replicates (ALFWorld direction)", +0.99 nats [0.90, 1.08], "a shift shared by working and failing calls (working calls move more)"; item share 0.82 against Qwen3-8B's 0.78 ("no difference detected") although Qwen's effect is four times larger in nats. Mistral-7B "not interpretable (fails the families competence rule on the first-token readout)" (AUC 0.58). Results in `tau_families_results.md` |
| ALFWorld, wave B | Qwen3.5-9B, Ministral-3-8B, Gemma 4 E4B and 31B, Qwen3.8-27B, plus a base-against-instruct comparison | Design frozen at tag `families-waveb-design-frozen-2026-11-22` (commit `f6bbffb`) after one review and its fixes; environment check running; no wave B score is read unless it passes |
| TextCraft round 3 | Qwen3.5-9B, Ministral-3-8B, Gemma-4-12B, plus a larger-model extension and an exploratory Qwen3.5 thinking arm | Proposal and compatibility checks done; built after round 2 is frozen |
| Decision simulation, Stage 1b | Llama-3.1-8B and Mistral-7B (priority), Qwen3-8B (contrast) | Done (frozen at tag `decisionsim-stage1b-frozen-2026-11-22`, commit `2a05707`). "No reliable difference" for Llama-3.1-8B and Mistral-7B, overall and on encounter states; Qwen3-8B (contrast) +2.29 points [+1.36, +3.33], mostly from rejecting working alternatives under any task. Results in `decisionsim_agent_results.md` |

## 5. Comparing effect sizes across models

This comes from the exploratory forensic audit of Qwen3-8B's effect (`qwen_effect_audit_forensic.md`, metrics `qwen_effect_audit_forensic_metrics.json`), which re-read the stored scores of every model above; its analysis list and interpretation rule were committed before any number was computed. It changes no verdict.

**Two-candidate nats are not one scale.** Models spread their scores over very different ranges: across items, Qwen3-8B's two-candidate score under the swapped task has an SD of 7.8 nats, Llama-3.1-8B's 3.1. A shift of one nat therefore means different things in different models, and the same holds across Qwen3 sizes. Two-candidate nats should not be compared across models or across Qwen sizes as if they were one scale.

**The standardized effect.** The audit divides each model's mean shift by the SD of its item-level shifts,

$$
d_z = \frac{\overline{\Delta L}}{\operatorname{SD}(\Delta L)},
$$

where $\Delta L$ is an item's aligned-minus-swapped difference. In the table, "first token" is the two-candidate score's opening word alone ("You" against "Nothing").

| Model | Two-candidate | First token | Yes/no |
|---|---|---|---|
| Qwen3-8B | 1.02 [0.91, 1.14] | 0.75 [0.68, 0.82] | 0.88 [0.81, 0.94] |
| Llama-3.1-8B | 0.85 [0.78, 0.92] | 0.68 [0.61, 0.74] | 1.12 [1.03, 1.21] |
| Mistral-7B | 0.52 [0.44, 0.61] | 0.74 [0.64, 0.84] | 0.96 [0.88, 1.05] |
| Llama-3.2-3B | 0.75 [0.64, 0.86] | 0.81 [0.71, 0.92] | 1.00 [0.88, 1.12] |
| Gemma-2-9B | 0.31 [0.18, 0.44] | 0.31 [0.18, 0.44] | 0.49 [0.39, 0.59] |
| Qwen3-4B | 0.90 | 0.82 | 0.27 |
| Qwen3-14B | 0.96 | 1.09 | 0.46 |
| Qwen3-32B | 0.86 | 0.60 | 0.46 |

**Qwen is close to the other models on that scale.** Qwen3-8B's two-candidate effect is 1.2 times Llama-3.1-8B's (1.02 against 0.85), not 3.7 times as in nats. On the first token it is inside the other competent models' range (0.75, against 0.31 to 0.81), and on yes/no it is below Llama-3.1-8B, Mistral and Llama-3.2-3B (0.88 against 0.96 to 1.12).

**The Qwen3 size series is flat when standardized.** In nats the competent sizes fall from 7.49 (4B) to 2.35 (32B). Standardized, the two-candidate effect is 0.90, 1.02, 0.96 and 0.86 for 4B, 8B, 14B and 32B. What changes with size is how far each model spreads its scores (the SD of the swapped-task score falls from 10.9 to 4.0 nats), not how strongly the task moves them relative to that spread. *(Updated 2026-11-23, exploratory follow-up: this holds in the study's thinking-off format, where Qwen3's template puts an empty `<think></think>` block before the answer. On a seeded 300-item subsample the standardized effect is 0.97, 1.02 and 1.00 for 4B, 8B and 14B with the block, and 1.05, 1.15 and 0.72 in plain text (no chat template), so the series stays flat for 4B and 8B but not for 14B, whose plain-text interval (0.57 to 0.89) does not overlap 8B's (0.96 to 1.38). The block inflates each size's nats by a different amount (plain text keeps 31%, 53% and 23% of the with-block effect), so the fall in nats is mostly the block's scale inflation, which is not monotone in size. Qwen3-32B was not rescored; its weights were no longer available. `qwen_effect_audit_pipeline.md` §7.2 B.)*

**59% of Qwen's 6.12 nats is the success tail.** 3.60 nats are the goal raising the probability of the rest of the success sentence after "You"; the first token carries 2.52. Llama-3.1-8B has the same 59% share, but the share is 45% for Gemma-2-9B, 29% for Llama-3.2-3B and negative for Mistral (−12%), and it varies across Qwen3 sizes (17%, 59%, 10% and 49% for 4B to 32B), so the tail inflates some models' nats and not others'.

**Why Qwen rarely flips.** A changed answer needs an item near the decision boundary. In the yes/no format Qwen3-8B's median failure sits at −26.9 nats under the swapped task, and only 0.1% of items lie within 5 nats of the boundary, so a 2.8-nat average shift almost never crosses it (0.2% of answers change). Llama-3.1-8B's median failure sits at −4.4 nats, with 59% of items within 5 nats, and it changes 16.5% of answers; Mistral's sits at −0.25 and it changes 37.5%. The flip rate is set by where each model's scores start, not by how far the task moves them.

**How to compare.** Cross-model comparisons lead with a scale-free readout (the standardized effect, the share of items moved, the pooled AUC) and the yes/no format, and give nats per model. For Qwen3-8B that means leading with a standardized effect of 1.02 and 88% of items moved toward success, and quoting the 6.12 nats together with its format sensitivity, described next. *(Rewritten 2026-11-23 after the exploratory follow-up; it replaces this paragraph's earlier 200-item summary.)*

The second, independent audit (`notes/qwen_effect_audit_pipeline.md`, 2026-11-23) re-implemented the Qwen scoring on GPU: a scorer written from scratch reproduces the stored scores exactly on the same GPU type, and vLLM and float32 weights agree, so no bug produced the number. About half of the effect's size in nats depends on one prompt-format detail: with thinking off, Qwen3's chat template inserts an empty `<think></think>` block before the answer. Its follow-up (section 7 of that note; plan committed before scoring at `8810f24`) rescored all 4,666 items without the block on the same GPU types. The effect is 3.31 nats (2.95 to 3.64) against 6.12, a ratio of 0.54, but the standardized effect (1.02 with the block, 1.05 without) and the share of items moved (87.9% and 88.7%) are unchanged. The block stretches the scale on which Qwen expresses its preferences; it does not make the model more goal-sensitive, so the standardized comparison in the table above is unaffected. One correction to the audit's first write-up: the render without the block is Qwen's template with thinking on, where the two answers get under 0.1% of the next-token probability, so it is not an ordinary alternative format. The comparison to rely on is plain text (no chat template; the answers keep at least 90% of the probability), which gives the same halving for Qwen3-8B (3.26 against 3.28 nats on 300 items). In every format tried the effect stays positive and 2 to 5 times Llama-3.1-8B's in nats. The step-0 concentration of Qwen's effect exists only with the block (10.99 nats at step 0 with it, 2.73 without, against 3.3 to 3.6 at later steps).

From this note on, results summaries report every scored model. The reference models are singled out only where a design makes them primary.
