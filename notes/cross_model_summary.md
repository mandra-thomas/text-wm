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
- **Effect size and practical impact rank the models differently.** Qwen3-8B has the largest log-odds effect but changes its answer on only 0.2% of failures, because it is very confident that they fail. Less confident models change many answers: Mistral 37.5%, Llama-3.2-3B 24.2%, Llama-3.1-8B 16.5%. A shift in log-odds is converted into changed decisions by the model's calibration, so a ranking by nats alone misstates where the effect matters most (`families_results.md`, "Practical size depends on calibration").
- **Scale does not remove it.** Among the competent Qwen3 sizes, the two-candidate effect in nats falls with size (7.5, 6.1, 4.3, 2.4), but the share of sign changes does not (19%, 10%, 21%, 18%). Under the aligned task, Qwen3-32B answers Yes to 11.6% of failing actions, against 3.8% under the swapped task.
- **Limits.**
  - One scoring run per model.
  - Comparisons across families change the developer, data, size, tokenizer, template and post-training together, so they are descriptive.
  - All of these models were released in 2024–2025 and none is larger than 32B. Newer models are covered by the wave B design (section 4).

## 2. Decision simulation, Stage 1a: five verifier models

This comes from `decisionsim_results.md` and `decisionsim_metrics.json`. A verifier asks the world model whether the agent's proposed action will succeed. If the answer is No, the agent takes the best-scored alternative from a fixed candidate menu. The comparison is the failure rate of the chosen action when the verifier sees the real task, against a verifier with the task sentence deleted ("task-blind").

| Verifier model | Extra failures, real task − task-blind | Gate category |
|---|---|---|
| Qwen3-8B (decides the gate) | +19.5 points [17.2, 22.3] | worth doing |
| Llama-3.2-3B (secondary) | +15.4 points | worth doing; its task-blind verifier is itself worse than no verifier |
| Gemma-2-9B (secondary) | +8.9 points | worth doing; the same two-part mechanism as Qwen3-8B |
| Mistral-7B (secondary) | +3.3 points | worth doing; all at step 0 |
| Llama-3.1-8B (primary) | +1.1 points [−0.1, 2.6] | no reliable change |

Four of the five models can be pushed into choosing failing actions on these menus. With a fallback that contains no failing goal-object actions, all three secondary models drop to "no reliable change" or "too rare". In every model the size of the harm depends on the menu and on the decision rule (`decisionsim_dx.md`).

## 3. Mechanism battery: three models

This comes from branch `mechanism2`, with the design frozen at tag `mechanism2-design-frozen-2026-11-22`. Qwen3-8B and Llama-3.1-8B are the primary models; Mistral-7B-Instruct-v0.3 is scored as a secondary, descriptive model, as fixed in the design (section 11.9). The frozen rules give the same verdict for all three: no single account. Each candidate account is contradicted as the sole explanation, so the effect is a mixture or an account not on the list. Results are in [mechanism2_results.md](mechanism2_results.md), and the robustness deep-dive is in [mechanism2_dx.md](mechanism2_dx.md). The models differ in what moves them. For Qwen3-8B, adding "Ignore the {obj}." to the swapped task reproduces the whole goal effect (r 1.03 two-candidate). For Llama-3.1-8B a bare mention gives about a third of it. Describing the object instead of naming it keeps most or all of the effect in both models. All three models read an explicit claim about the object's location strongly in the stated direction. The user's hope barely matters for Qwen or Llama, while Mistral moves in the wishful direction. Qwen's yes/no answers stay "No" under every line, so its yes/no effects are shifts of confidence. Llama's Yes rate on these failures goes from 6.1% (swapped) to 22.4% (aligned). The key contrasts keep their sign across games, task types and formats.

## 4. What still rests on two models, and what is being added

The TextCraft and τ-bench results, and the reading of the effect as task relevance that they support, have so far been scored only on Qwen3-8B and Llama-3.1-8B. The following runs add models:

| Study | Models added | State |
|---|---|---|
| TextCraft round 2 | Gemma-2-9B, Mistral-7B, Llama-3.2-3B (Qwen3-8B and Llama-3.1-8B rescored under the clean prompt) | Design complete; frozen and scored after an independent pre-freeze audit |
| τ-bench retail, model families | Mistral-7B, Llama-3.2-3B. Gemma-2-9B is not run: every prompt exceeds its 8,192-token context | Built; revising after its first review |
| ALFWorld, wave B | Qwen3.5-9B, Ministral-3-8B, Gemma 4 (12B or E4B; 31B), Qwen3.8-27B, plus a base-against-instruct comparison | Built; revising after its first review |
| TextCraft round 3 | Qwen3.5-9B, Ministral-3-8B, Gemma-4-12B, plus a larger-model extension and an exploratory Qwen3.5 thinking arm | Proposal and compatibility checks done; built after round 2 is frozen |
| Decision simulation, Stage 1b | Llama-3.1-8B and Mistral-7B (priority), Qwen3-8B (contrast) | Design frozen at tag `decisionsim-stage1b-frozen-2026-11-22`; verifier scoring running |

From this note on, results summaries report every scored model. The reference models are singled out only where a design makes them primary.
