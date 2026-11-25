# Model families, wave A: results (2026-11-20)

Exploratory. The design, predictions and analysis rules are in [families_design.md](families_design.md); they were fixed, including three amendments, before any wave-A score was read. Analysis run `families_analysis` at ca76bb9 (ICE job 7201550); every number below is from [families_metrics.json](families_metrics.json). *(Corrected 2026-11-23: the metrics file on `main` is from a rerun of `families_analysis` at the same commit, ICE job 7204329, made once the Mistral placement scores were in (commit d0b30c1). Against the job 7201550 version first committed (a32eaeb), only its `placement`, `waiting` and `provenance` blocks differ; every other block is identical, so every number in the first table and the comparisons is the same in both.)* Scoring ran at 65eb591 on the 4,666 primary RQ2 items (determined failures) and 292 determined successes, three task lines each (aligned, swapped, neutral), two-candidate and yes/no formats.

**Primary measure** (yes/no format): the share of failure items whose log-odds of "Yes" is higher under the aligned task than the swapped one (ties count as not moved); 0.5 means no effect. A model is interpretable only if, under the neutral task, its yes/no AUC is at least 0.70, it answers Yes to at most 95% of failures and at least 5% of successes, and the median P(Yes)+P(No) is at least 0.5. Intervals are game-clustered bootstraps (95%). *(revised 2026-11-25)* Option C's step-0 check ([optionc_results.md](optionc_results.md), section 3) found Qwen3-8B partly reading this anchored yes/no question as asking about task success ("question-specific (supports the task-success reading)"), so the share may include a task-progress judgement; the two-candidate effect, which does not use the question, is positive for every model in the table.

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

*(Clarified 2026-11-23.)* Two rules give a reading here, and they are not the same rule. The **primary** reading of each of the seven planned comparisons comes from the primary family of 18 tests (`primary_family.comparison_readings` in the metrics file): "differs" needs the Holm-adjusted game sign-flip permutation p of the paired difference (`p_differs`) below 0.05, and "matches" needs the Holm-adjusted TOST p for a difference within ±0.05 (`p_equivalent`) below 0.05; differs without matching reads "differs", neither reads "inconclusive (no difference detected)". The **descriptive** reading in the `comparisons` block, which the table's column follows, says "differs" when the 95% bootstrap interval of the difference excludes zero and "matches" when its 90% interval lies inside ±0.05, with no correction. For the seven planned comparisons the two agree: the four Gemma and Llama-3.2-3B-against-Llama-3.1-8B comparisons differ (Holm p 0.018 each), and both Mistral comparisons are inconclusive (Holm p of the difference 1.0; of equivalence 0.156 and 0.126). The two rows outside the planned comparisons (Llama-3.2-3B and Llama-3.2-1B against Qwen3-8B) have only the descriptive reading, and the Llama-3.2-1B comparisons are not interpreted.


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

**One key name, two quantities.** *(Added 2026-11-23.)* The code key `yes_share_failures` means different things in two modules, and both appear in `families_metrics.json`. In `wf/sizes.py` (and so under `models.<model>.yes_no.yes_share_failures` in `families_metrics.json`, and in `size_metrics.json` and `size_v2_metrics.json`) it is the mean of σ(L) over failing actions, a mean probability of "Yes". In `wf/families.py` (under `models.<model>.argmax_answers.yes_share_failures`) it is the share of failing actions whose yes/no log-odds is above zero, that is, answered Yes. This note's "Answers Yes on failures" column is the second (for Mistral under the aligned task, 0.803 against a mean σ of 0.757). The keys are not renamed.

**Practical size depends on calibration.** Models that are very confident that failures fail (Qwen3-8B, Gemma 2) rarely change their answer under the aligned task, although the task still moves their scores. Less confident models change many answers: Mistral already answers Yes to 45% of failures under the swapped task and 80% under the aligned task. The goal-to-evidence ratio R (secondary) is in the metrics file; across families it mostly reflects how far each model pushes its log-odds (design note, third amendment), so it is not used to rank models.

**Limits.** One scoring run per model; cross-family comparisons change developer, data, size, tokenizer, template and post-training at once, so they are descriptive. All four wave-A models are from 2024. Mistral's chat template places the system text after the few-shot examples; the placement check below tests whether that matters. The two-candidate format's validity is unchecked for the non-Qwen models, so the yes/no format is the primary cross-model scale.

**Placement check: Mistral's system text** (logged before data in [families_design.md](families_design.md), "Placement check"; run `family_placement_mistral-7b-v0.3`, ICE job 7199733, read by `families_analysis` at ca76bb9, ICE job 7204329). Mistral 7B v0.3 was rescored on 300 primary items (102 games) with the system text folded into the first user turn, as for Gemma, instead of the native last-turn position. Goal effect (aligned minus swapped, nats, game-clustered 95% CI) on the same 300 items:

| Format | Native placement | First-turn placement | First turn minus native |
|---|---|---|---|
| Yes/no | 2.46 [2.06, 2.88] | 2.76 [2.27, 3.27] | +0.30 [+0.13, +0.48] |
| Two-candidate | 0.85 [0.58, 1.13] | 0.87 [0.61, 1.15] | +0.02 [−0.15, +0.20] |

By the logged rule (the difference's CI excludes zero *and* the difference exceeds a fifth of the native effect), placement does not move the effect on this sample: in the two-candidate format the difference is near zero; in the yes/no format it is reliably positive but 12% of the native effect, below the one-fifth threshold (0.49 nats). Moving the system text earlier makes Mistral's yes/no effect slightly larger, not smaller, so the native placement does not inflate Mistral's position in the comparisons. The tokenization gate passed on all 300 items. This is one model, one alternative placement and 300 items; it does not make the other families' formats comparable. Adding the run changed nothing else in `families_metrics.json`.
