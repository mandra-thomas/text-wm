# Model families, wave A: results (2026-11-20)

Exploratory. The design, predictions and analysis rules are in [families_design.md](families_design.md); they were fixed, including three amendments, before any wave-A score was read. Analysis run `families_analysis` at ca76bb9 (ICE job 7201550); every number below is from [families_metrics.json](families_metrics.json). Scoring ran at 65eb591 on the 4,666 primary RQ2 items (determined failures) and 292 determined successes, three task lines each (aligned, swapped, neutral), two-candidate and yes/no formats.

**Primary measure** (yes/no format): the share of failure items whose log-odds of "Yes" is higher under the aligned task than the swapped one (ties count as not moved); 0.5 means no effect. A model is interpretable only if, under the neutral task, its yes/no AUC is at least 0.70, it answers Yes to at most 95% of failures and at least 5% of successes, and the median P(Yes)+P(No) is at least 0.5. Intervals are game-clustered bootstraps (95%).

| Model | Interpretable | Share pushed toward Yes | ΔL yes/no (nats) | ΔL two-candidate (nats) | Answers Yes on failures, swapped → aligned | Flipped No → Yes |
|---|---|---|---|---|---|---|
| Qwen3-8B (reference) | yes (AUC 0.99) | 0.887 [0.859, 0.911] | 2.80 [2.59, 3.01] | 6.12 [5.50, 6.74] | 0.0% → 0.2% | 0.2% |
| Llama-3.1-8B (reference) | yes (AUC 0.97) | 0.896 [0.872, 0.918] | 2.21 [2.05, 2.39] | 1.64 [1.50, 1.79] | 5.6% → 21.0% | 16.5% |
| Gemma-2-9B-it | yes (AUC 0.99) | 0.748 [0.695, 0.801] | 1.23 [1.01, 1.44] | 0.80 [0.47, 1.12] | 1.1% → 3.5% | 3.0% |
| Mistral-7B-Instruct-v0.3 | yes (AUC 0.94) | 0.879 [0.846, 0.907] | 2.67 [2.41, 2.90] | 1.06 [0.90, 1.23] | 44.9% → 80.3% | 37.5% |
| Llama-3.2-3B-Instruct | yes (AUC 0.88) | 0.840 [0.794, 0.880] | 0.70 [0.60, 0.80] | 0.98 [0.81, 1.13] | 14.1% → 36.9% | 24.2% |
| Llama-3.2-1B-Instruct | **no** (AUC 0.68) | 0.598 [0.557, 0.636] | 0.09 [0.07, 0.10] | 0.31 [0.22, 0.41] | 4.0% → 6.6% | 3.2% |

All six models show a positive effect in both formats (the logged prediction holds for all four wave-A models). In the primary family (18 tests, Holm-corrected), the share is supported for Gemma 2, Mistral and Llama 3.2 3B; Llama 3.2 1B is not interpretable (its neutral-task AUC, 0.68, is below 0.70), so its share is not interpreted.

**Planned comparisons of the share** (paired by item, game-clustered; "differs" from the 95% interval of the paired difference; equivalence would need the 90% interval inside ±0.05):

| Comparison | Difference in share | Reading |
|---|---|---|
| gemma2-9b-it − llama3.1-8b | -0.147 [-0.198, -0.100] | differs |
| gemma2-9b-it − mistral-7b-v0.3 | -0.131 [-0.178, -0.085] | differs |
| gemma2-9b-it − qwen3-8b | -0.139 [-0.193, -0.090] | differs |
| llama3.2-1b − llama3.1-8b | -0.297 [-0.338, -0.256] | not interpreted (Llama 3.2 1B fails the competence rule) |
| llama3.2-1b − qwen3-8b | -0.289 [-0.333, -0.242] | not interpreted; outside the planned comparisons |
| llama3.2-3b − llama3.1-8b | -0.055 [-0.088, -0.026] | differs |
| llama3.2-3b − qwen3-8b | -0.047 [-0.083, -0.012] | differs (descriptive; outside the planned comparisons) |
| mistral-7b-v0.3 − llama3.1-8b | -0.017 [-0.048, +0.013] | no difference detected |
| mistral-7b-v0.3 − qwen3-8b | -0.009 [-0.046, +0.030] | no difference detected |

No comparison met the ±0.05 equivalence margin after Holm correction. Mistral is within about 2 points of both reference models but equivalence is not shown.

**Practical size depends on calibration.** Models that are very confident that failures fail (Qwen3-8B, Gemma 2) rarely change their answer under the aligned task, although the task still moves their scores. Less confident models change many answers: Mistral already answers Yes to 45% of failures under the swapped task and 80% under the aligned task. The goal-to-evidence ratio R (secondary) is in the metrics file; across families it mostly reflects how far each model pushes its log-odds (design note, third amendment), so it is not used to rank models.

**Limits.** One scoring run per model; cross-family comparisons change developer, data, size, tokenizer, template and post-training at once, so they are descriptive. All four wave-A models are from 2024. Mistral's chat template places the system text after the few-shot examples; a placement check (300 items, system text in the first turn) is queued and not yet included. The two-candidate format's validity is unchecked for the non-Qwen models, so the yes/no format is the primary cross-model scale.
