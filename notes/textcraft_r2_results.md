# TextCraft round 2: results (2026-11-23)

Confirmatory for the 50-test family that was frozen before any score, at the annotated tag `textcraft-r2-design-frozen-2026-11-23` (commit `4ccf84b`, branch `textcraft-r2`). The design is [textcraft_r2_design.md](textcraft_r2_design.md), including its sign-off corrections (section 15.F) and the independent pre-freeze audit ([textcraft_r2_audit.md](textcraft_r2_audit.md)). The scoring runs `textcraft_r2_score_<model>` (four shards per model, ICE jobs 7247899–7247918) and the analysis run `textcraft_r2_analysis` (ICE job 7247919) all ran at that commit. Every result below is copied from [textcraft_r2_metrics.json](textcraft_r2_metrics.json), which is the analysis run's `metrics.json` byte for byte. The provenance facts in section 10 were read, without modification, from the run folders on the cluster. No verdict here was recomputed or re-derived, and nothing in this note is a new analysis. Section 10 gives the full provenance, and section 11 reproduces the analysis run's STATUS text.

The analysis code (`wf/textcraft/r2_analysis.py` and the rest of `wf/textcraft/r2*.py`) lives at the tag and on the `textcraft-r2` branch. It is not on `main`.

## Contents

1. The question, in plain words
2. What was done: records, goal lines, the five contrasts and the rules
3. Discrimination (competence) for each model
4. Confirmatory results, contrast by contrast
5. Readings and rejection rules that follow from the verdicts
6. Descriptive diagnostics: R3, R4 and the central-question mapping
7. Other pre-registered descriptive readouts
8. The pre-registered predictions against the results
9. Interpretation: what round 2 says about Llama's negative TextCraft effect
10. Provenance
11. The analysis run's STATUS text

---

## 1. The question, in plain words

The project asks whether a language model used as a *world model* (a model that predicts how an environment will respond to an action) lets the agent's stated task leak into its predictions. In TextCraft, a text version of Minecraft crafting, the environment's replies are fully determined by the recipe list, the history and the action. Whether a craft succeeds never depends on the goal printed at the top of the screen. A model that reads the state correctly should therefore give the same prediction whatever the goal line says. Any shift caused by changing only the goal line is a leak of the task into the prediction.

The basic measurement is $L$, the *two-candidate log-odds*. The model reads the prompt, and we compute the log-probability it assigns to the environment's success reply ("Crafted ...") and to its failure reply ("Could not find ..."), each forced word by word after the prompt. $L$ is the success log-probability minus the failure log-probability, in *nats* (natural-log units). $L$ above zero means the model favours the success reply. All round-2 items in the confirmatory contrasts are actions that fail, so a correct world model should give them low $L$, and the goal line should not move it.

**What round 1 found.** The confirmatory TextCraft test ([textcraft_results.md](textcraft_results.md)) compared the *aligned* goal line (the goal whose recipe tree contains the action's item) with the *swapped* line (the other goal of the item's pair). Under the round-1 prompt the aligned line raised Qwen3-8B's $L$ by +0.94 nats on average. For Llama-3.1-8B it *lowered* $L$ slightly, by −0.19 nats. The exploratory follow-ups ([textcraft_llama_dx.md](textcraft_llama_dx.md)) suggested why. Actions that craft an *ingredient* of the goal seemed to get a lift from the aligned line in Qwen (about +0.9) and none in Llama (about 0). Actions that craft the *goal item itself* seemed to score about 0.8 nats lower than matched ingredient actions in both models. That "lift plus terminal penalty" reading was post hoc and confounded with item structure. Goal crafts and ingredient crafts are different actions with different inputs.

**The account round 2 tests.** For an action that crafts item $X$ under goal line $g$, the account writes the shift of $L$, relative to a line that makes $X$ irrelevant, as

$$L(a \mid g) - L(a \mid g_{\text{irrelevant}}) = \lambda_m \,[X \in \text{tree}(g)] + \nu_m \,[g \text{ names } X] + \kappa_m(h)\,[\text{crafting } X \text{ completes } g] + \varepsilon .$$

Here $m$ is the model and $h$ the history. $\lambda_m$ is an *intermediate lift* for any item in the goal's recipe tree. $\nu_m$ is a *naming term*, applied when the goal line names the item. $\kappa_m(h)$ is a *completion term*, applied when crafting the item would finish the task, and it may depend on the history. In ordinary TextCraft the naming and completion terms always act together, and their sum is round 1's "terminal penalty". The account claims three things. The penalty is negative and of similar size across models; round 2 tests only its sign, not the similarity of sizes. The lift differs across models, and that difference is why Qwen's total was positive and Llama's negative. The penalty is largest early in a game and shrinks as the history fills.

**The central question**, as frozen (design section 1 and 14.U5): *does Llama's negative TextCraft effect arise from a goal-completion or naming mechanism, from poor reasoning about the environment's state, or from a combination?* The design answers it only descriptively, with neutral labels (section 6 below). Those labels describe how the scores co-vary with simple text heuristics. They do not identify an internal mechanism.

## 2. What was done: records, goal lines, the five contrasts and the rules

**Models and prompt.** Five instruction-tuned models were each scored on the same 25,743 (record, goal-line) combinations: Qwen3-8B (thinking off), Llama-3.1-8B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3 and Llama-3.2-3B-Instruct. That is 128,715 scores in all. Every prompt used the *clean* world-model system text, from which every sentence addressed to an agent had been removed. The yes/no question was *anchored* to the environment's replies: "will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that it failed ("Could not find ...")?". Under round 1's question, "Will this action succeed?", a model could read "succeed" as task success. Round-2 yes/no figures are therefore a different quantity from round 1's. Qwen and Llama-3.1 are rescored under the clean prompt, so their numbers here are new estimates and not re-estimates of the round-1 confirmatory numbers. The round-1 verdicts stand as they were.

**Records** (design section 4). The 1,500 frozen main failures and 600 frozen successes from round 1 are reused. They come as *same-history couples*: pairs of items that share a goal pair, a history, a failure class and a role, with one item in each goal's tree. *Goal role* means the action crafts the aligned goal itself. *Ingredient role* means it crafts a direct ingredient of the goal. There are 882 new *parent* records. Each is a goal-role craft of an item G that is also a direct ingredient of some other item Z (the "parent"), shown on a recipe list extended with Z's recipe lines. The 920 main failures with 1 to 8 history steps also get *padded* versions, with 1 or 8 no-op inventory look-ups ("inventory" / "Inventory: You are not carrying anything.") placed before the first step.

**Goal lines.** On main records:

- the *aligned* line, "Goal: craft G.";
- the *swapped* line, which names the other goal of the pair;
- the *neutral* line, "Goal: explore the crafting commands.";
- the *two-item* line, "Goal: craft A and B.", naming both goals of the pair.

On parent records, on the extended list:

- the aligned line, "craft G";
- the *parent* line, "craft Z", under which G is an ingredient of the goal;
- the swapped line.

**The five confirmatory contrasts** are computed on two-candidate $L$, on failures only. Each is clustered by *goal pair*: resampling and sign-flipping always move whole goal pairs, because items that share a pair are not independent.

The **intermediate lift** $\Lambda$ asks whether a model raises its predicted success for an action that serves the named goal without being the goal. It is the mean, over the 990 ingredient-role main failures, of aligned minus swapped:

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}} .$$

Many ingredient names share words with their goal's name (683 of 990), so $\Lambda$ includes any effect of partial naming.

The **within-item penalty** $P$ asks whether naming G as the goal, so that crafting it completes the task, lowers predicted success compared with a goal for which G is only an ingredient. The action, history and displayed list are identical, and only the goal line changes. It is the mean over the 675 parent failures:

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})} .$$

G is relevant under both lines, so the lift cancels. $P$ estimates the naming and completion terms together on one fixed craft, free of round 1's item-structure confound.

The **naming / partial-completion** contrast $D_n$ asks what happens when the goal line names the crafted item but crafting it does not finish the task. The two-item line names G, and crafting G completes only one of two named items. $D_n$ compares two-item minus swapped on goal-role failures with the same difference on ingredient failures, where the two-item line only makes the item relevant:

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}} .$$

The **completion with naming held** contrast $D_c$ asks what completing the whole task adds, given that the item is named. Both lines name G, and only the one-item line makes the craft finish the task:

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}} .$$

Both use the same 996 rows: 500 goal-role failures that have a two-item line, and 496 *class-matched* ingredient failures (only the "too few" and "never obtained" failure classes, as for the goal role). On those rows $D_n + D_c$ equals the *role gap*, the goal-role aligned-minus-swapped effect minus the ingredient one, exactly. The split is algebraic. Reading the parts as causal "naming" and "completion" effects needs four untested assumptions (design 14.U4): additivity, equal dilution by the second named item for both roles, no interaction between recipe structure and the goal line, and no effect of which item is named first.

The **timing** contrast $T$ asks whether the penalty shrinks when the history gets longer without gaining content. Writing $\Delta L^{(k)}$ for aligned minus swapped with $k$ look-ups prepended,

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}} .$$

$T$ is read only where there is a penalty to shrink. If the goal-minus-ingredient effect at one look-up does not have a 95% interval entirely below zero, $T$'s verdict is "not applicable" whatever its test says. That is a frozen gate, not a test.

**Statistics.** For $\Lambda$ and $P$, which are means of per-item differences, the "differs" p-value is a two-sided *sign-flip permutation* test: whole goal pairs have the signs of their differences flipped at random, 20,000 times. For $D_n$, $D_c$ and $T$, which compare two groups of items, it is a two-sided pair-bootstrap p-value. Every estimate has a 95% percentile *pair bootstrap* interval: goal pairs are resampled with replacement and the estimate recomputed, 20,000 times for the family tests. Each contrast also has an equivalence test, a *TOST* ("two one-sided tests"). It asks whether the effect is reliably inside a margin of plus or minus $\delta$. Its p is the larger of the two one-sided bootstrap shares beyond $\pm\delta$. The margin is fixed per model from the item spread of the main-failure aligned-minus-swapped effect:

$$\delta_m = \min\left(0.3 \times \mathrm{SD}_m,\ 0.4\ \text{nats}\right).$$

All 50 tests (five models, five contrasts, a differs test and an equivalence test each) were *Holm-adjusted* together at a familywise level of 0.05. Holm's procedure sorts the p-values and multiplies the smallest by 50, the next by 49, and so on, keeping the result monotone. It controls the chance of any false claim across the whole family. With 20,000 resamples the smallest possible raw p is $1/20001 \approx 0.00005$, so the smallest possible Holm-adjusted p is about 0.0025. Several tests sit at that floor.

**Verdicts** (design section 8), applied in this order:

- "confirmed above zero" or "confirmed below zero": the Holm-adjusted differs p is below 0.05, and the estimate and its whole 95% interval are on that side of zero;
- "equivalent to zero": otherwise, if the Holm-adjusted TOST p is below 0.05. This means "no effect larger than the margin", never "no effect";
- "inconclusive": otherwise. The interpretation adds "(equivalence underpowered)" when the equivalence test's power at a true zero, at Holm's first step, is below 0.8;
- "not applicable": for the timing contrast only, when the applicability gate above fails.

**Discrimination ("competence") changes the wording, never the verdict.** Each contrast has its own gate (design 14.U1, worded at sign-off 15.F4). Two conditions must hold:

- the *AUC* of the yes/no answer under the neutral line is at least 0.70 in every population the contrast uses. The AUC is the probability that a randomly chosen success gets a higher score than a randomly chosen failure: 0.5 is chance and 1.0 is perfect separation;
- the other three conditions of the overall gate hold on the full population: Yes on at most 95% of failures, Yes on at least 5% of successes, and a median yes-plus-no probability of at least 0.5.

Passing means the model's yes/no answers *discriminate* successes from failures in those action categories. It does not show that the model applies the inventory rule or understands the environment. A confirmed verdict in a model that passes is read as "distortion in a model that discriminates successes from failures in these categories". In a model that fails, it is read as a "statistically measurable goal effect; the model does not discriminate successes from failures in this contrast's populations (its competence gate fails), so it is not read as a distortion of a discriminating prediction". An equivalence verdict reads "no effect larger than the margin, in a model that discriminates successes from failures in these categories" or "no effect larger than the margin; limited interpretation (does not discriminate successes from failures in these populations)". These four strings are copied from `metrics.json`.

**What is confirmatory, what is a decision rule and what is descriptive.** The analysis output has an `inference_map` that sets out three tiers:

- *Confirmatory:* only the 50 family tests.
- *Decision rules applied to the verdicts, without adding a test:* the per-model readings and the rejection rules R1, R2 and R5.
- *Descriptive, with intervals and no p-values:* everything else. That includes R3, R4, the central-question mapping, the role gap and its matched version, the parent subsets, the competence AUCs and every other readout.

This note keeps the same three tiers. It contains no exploratory analysis. Every number is a pre-registered output of the frozen analysis.

## 3. Discrimination (competence) for each model

This section comes first because it governs how every verdict below is worded. The first table gives the overall gate, wave A's four-part rule, on the 1,500 main failures and 600 successes under the neutral line with the yes/no question. The overall gate no longer decides how any contrast is read. It is reported for every model.

| Model | Yes/no AUC | Yes on failures | Yes on successes | Median yes-plus-no mass | Overall gate |
|---|---|---|---|---|---|
| Qwen3-8B | 0.860 | 7.8% | 54.2% | 1.000 | passes |
| Llama-3.1-8B | 0.564 | 53.7% | 60.8% | 1.000 | fails (AUC) |
| Gemma-2-9B-it | 0.810 | 21.6% | 71.2% | 0.998 | passes |
| Mistral-7B-v0.3 | 0.566 | 99.9% | 99.5% | 0.998 | fails (AUC; Yes on failures) |
| Llama-3.2-3B | 0.720 | 96.9% | 99.0% | 0.993 | fails (Yes on failures) |

The contrast-specific gates are what the verdicts use. The second table gives, for each contrast, the yes/no neutral-line AUC in each of its populations and whether the model passes ("discriminates") or fails ("does not discriminate"). The populations are as follows:

- the lift uses all ingredient failures against ingredient successes;
- the penalty uses goal-role main failures against goal-role successes, *and* parent failures against parent successes;
- the naming and completion contrasts use goal-role failures against goal-role successes, *and* class-matched ingredient failures against ingredient successes;
- timing uses the padded goal-role and padded class-matched ingredient failures against the successes at 1 to 8 steps.

| Model | Lift | Within-item penalty | Naming and completion | Timing |
|---|---|---|---|---|
| Qwen3-8B | 0.869: discriminates | goal 0.864, parent 0.917: discriminates | goal 0.864, ingredient 0.862: discriminates | goal 0.885, ingredient 0.893: discriminates |
| Llama-3.1-8B | 0.556: does not discriminate | goal 0.716, parent 0.824: discriminates | goal 0.716, ingredient 0.571: does not discriminate | goal 0.693, ingredient 0.587: does not discriminate |
| Gemma-2-9B-it | 0.771: discriminates | goal 0.865, parent 0.939: discriminates | goal 0.865, ingredient 0.715: discriminates | goal 0.857, ingredient 0.713: discriminates |
| Mistral-7B-v0.3 | 0.596: does not discriminate | goal 0.501, parent 0.520: does not discriminate | goal 0.501, ingredient 0.468: does not discriminate | goal 0.499, ingredient 0.521: does not discriminate |
| Llama-3.2-3B | 0.660: does not discriminate | goal 0.769, parent 0.725: does not discriminate (AUCs pass; Yes on 96.9% of failures fails the full-population condition) | goal 0.769, ingredient 0.632: does not discriminate | goal 0.778, ingredient 0.665: does not discriminate |

**What the gates show.** Qwen3-8B and Gemma-2-9B discriminate in every population. Each contrast is therefore read for them as a possible distortion of a discriminating prediction.

Llama-3.1-8B has the profile round 1 led us to expect. It discriminates on goal crafting: 0.716 on the main goal-role items and 0.824 on the parent records, so it passes the penalty's gate. It does not discriminate on ingredients (0.556 overall, 0.571 class-matched), which is near chance. It misses the timing gate narrowly, at 0.693 on the padded goal-role items. Its overall AUC of 0.564 fails because ingredients make up most of the population.

Mistral-7B answers Yes to almost everything: 99.9% of failures and 99.5% of successes. Its AUCs are at chance, so it discriminates nowhere.

Llama-3.2-3B ranks goal-crafting items better than chance (AUC 0.769 and 0.725, both above 0.70). It still fails every contrast's gate, because it answers Yes to 96.9% of failures and the full-population condition caps that at 95%. Its "does not discriminate" for the penalty therefore comes from near-universal Yes answers, not from the AUC. The frozen wording does not distinguish the two routes to failing a gate. Section 9 returns to this.

## 4. Confirmatory results, contrast by contrast

Each table gives, for every model:

- the estimate and its 95% pair-bootstrap interval, in nats;
- the Holm-adjusted p of the differs test;
- the Holm-adjusted p of the equivalence test, with that test's power at a true zero at Holm's first step in parentheses;
- the verdict string from `metrics.json`;
- the model's discrimination status for this contrast.

The equivalence margin is 0.40 nats for every model except Llama-3.1-8B, whose smaller spread gives 0.38. The item SD of the main-failure aligned-minus-swapped effect, three tenths of that SD, and the margin are as follows:

| Model | Item SD of the main-failure effect (nats) | Three tenths of the SD | Margin used (nats) |
|---|---|---|---|
| Qwen3-8B | 4.52 | 1.35 | 0.40 |
| Llama-3.1-8B | 1.27 | 0.38 | 0.38 |
| Gemma-2-9B-it | 1.55 | 0.47 | 0.40 |
| Mistral-7B-v0.3 | 1.63 | 0.49 | 0.40 |
| Llama-3.2-3B | 1.56 | 0.47 | 0.40 |

Qwen spreads its scores about three times as widely as the others. As the design anticipated (section 10), equivalence within 0.4 nats is out of reach for Qwen on every contrast: the power at Holm's first step is 0.00 throughout. For Qwen, any contrast that is not confirmed can only be "inconclusive", never "equivalent".

### 4.1 Intermediate lift: does the aligned line raise predicted success for ingredient crafts?

990 ingredient-role main failures in 162 goal pairs.

| Model | Estimate [95% CI] | Differs, Holm p | Equivalent, Holm p (power) | Verdict | Discrimination |
|---|---|---|---|---|---|
| Qwen3-8B | +0.67 [+0.32, +1.02] | 0.0144 | 1.000 (0.00) | "confirmed above zero" | discriminates |
| Llama-3.1-8B | +0.095 [−0.004, +0.193] | 1.000 | 0.0025 (1.00) | "equivalent to zero" | does not discriminate |
| Gemma-2-9B-it | −0.30 [−0.42, −0.19] | 0.0025 | 1.000 (1.00) | "confirmed below zero" | discriminates |
| Mistral-7B-v0.3 | +0.27 [+0.19, +0.35] | 0.0025 | 0.0261 (1.00) | "confirmed above zero" | does not discriminate |
| Llama-3.2-3B | −0.04 [−0.16, +0.08] | 1.000 | 0.0025 (1.00) | "equivalent to zero" | does not discriminate |

Qwen3-8B's lift is confirmed above zero, at about two thirds of a nat, in a model that discriminates on ingredients. Its interpretation is "distortion in a model that discriminates successes from failures in these categories". The aligned line makes Qwen more optimistic about failing ingredient crafts.

Llama-3.1-8B's lift is "equivalent to zero": the interval lies within its 0.38-nat margin. Llama does not discriminate on ingredients, so the interpretation is "no effect larger than the margin; limited interpretation (does not discriminate successes from failures in these populations)". The absence of a lift is measured in a population where Llama's yes/no answers are near chance. "No lift" there does not mean that a working state judgement was left untouched.

Gemma-2-9B has a *negative* lift, confirmed below zero and read as a distortion in a model that discriminates. The aligned line makes Gemma more pessimistic about failing ingredient crafts. That is the correct direction for failures, but still a shift that the goal line causes.

Mistral-7B's lift is confirmed above zero, as a measurable effect only. Its equivalence test also passes (Holm p 0.026). The lift is reliably different from zero and reliably smaller than 0.4 nats. Under the frozen rule the differs verdict takes precedence.

Llama-3.2-3B is "equivalent to zero", with limited interpretation.

### 4.2 Within-item penalty: naming and completing against "ingredient of the goal", on one fixed craft

675 parent failures in 111 goal pairs.

| Model | Estimate [95% CI] | Differs, Holm p | Equivalent, Holm p (power) | Verdict | Discrimination |
|---|---|---|---|---|---|
| Qwen3-8B | −1.02 [−1.62, −0.45] | 0.0232 | 1.000 (0.00) | "confirmed below zero" | discriminates |
| Llama-3.1-8B | −1.68 [−1.94, −1.42] | 0.0025 | 1.000 (0.00) | "confirmed below zero" | discriminates |
| Gemma-2-9B-it | −1.27 [−1.48, −1.06] | 0.0025 | 1.000 (0.49) | "confirmed below zero" | discriminates |
| Mistral-7B-v0.3 | +0.27 [+0.13, +0.40] | 0.0087 | 0.7050 (0.99) | "confirmed above zero" | does not discriminate |
| Llama-3.2-3B | −0.50 [−0.68, −0.33] | 0.0025 | 1.000 (0.84) | "confirmed below zero" | does not discriminate |

This is the contrast round 2 was built around. It holds the action, the history and the displayed list fixed, and changes only whether the goal line names G (so that crafting G completes the task) or names a parent Z (so that G is merely an ingredient of the goal). Three models discriminate on goal crafting: Qwen3-8B, Llama-3.1-8B and Gemma-2-9B. All three show a penalty confirmed below zero, between about −1.0 and −1.7 nats. Each is read as a "distortion in a model that discriminates successes from failures in these categories". The goal-crafting penalty from round 1 therefore survives when item structure is removed. It is not an artefact of comparing different crafts.

Llama-3.2-3B also shows a confirmed penalty (−0.50). It is a measurable effect only, because the model answers Yes to nearly every failure. Mistral-7B goes the other way: naming and completing *raises* its predicted success, confirmed above zero, as a measurable effect only.

### 4.3 Naming / partial completion: the item is named but its craft does not finish the task

500 goal-role and 496 class-matched ingredient failures in 208 goal pairs.

| Model | Estimate [95% CI] | Differs, Holm p | Equivalent, Holm p (power) | Verdict | Discrimination |
|---|---|---|---|---|---|
| Qwen3-8B | +0.10 [−0.67, +0.86] | 1.000 | 1.000 (0.00) | "inconclusive" (equivalence underpowered) | discriminates |
| Llama-3.1-8B | +0.78 [+0.56, +0.99] | 0.0025 | 1.000 (0.29) | "confirmed above zero" | does not discriminate |
| Gemma-2-9B-it | −0.57 [−0.74, −0.40] | 0.0025 | 1.000 (0.87) | "confirmed below zero" | discriminates |
| Mistral-7B-v0.3 | +0.94 [+0.69, +1.20] | 0.0025 | 1.000 (0.00) | "confirmed above zero" | does not discriminate |
| Llama-3.2-3B | +0.15 [−0.07, +0.38] | 1.000 | 0.4441 (0.30) | "inconclusive" (equivalence underpowered) | does not discriminate |

Gemma-2-9B is the only model with a negative naming / partial-completion contrast. It is confirmed below zero, in a model that discriminates in both populations. Naming the crafted item, without the craft finishing the task, lowers Gemma's predicted success relative to the ingredient control.

Llama-3.1-8B and Mistral-7B go the other way: naming the item without completing the task *raises* their predicted success. For both this is a measurable effect only. Llama's goal-role population passes (0.716), but its class-matched ingredient population (0.571) does not, and the gate takes the smaller AUC.

Qwen3-8B is inconclusive. Its interval runs from about −0.7 to +0.9, and equivalence cannot be reached at its spread. Llama-3.2-3B is inconclusive as well.

### 4.4 Completion with naming held: does finishing the whole task add to naming?

The same 996 rows.

| Model | Estimate [95% CI] | Differs, Holm p | Equivalent, Holm p (power) | Verdict | Discrimination |
|---|---|---|---|---|---|
| Qwen3-8B | +0.65 [−0.003, +1.31] | 1.000 | 1.000 (0.00) | "inconclusive" (equivalence underpowered) | discriminates |
| Llama-3.1-8B | −1.26 [−1.45, −1.08] | 0.0025 | 1.000 (0.62) | "confirmed below zero" | does not discriminate |
| Gemma-2-9B-it | −0.05 [−0.24, +0.16] | 1.000 | 0.0087 (0.62) | "equivalent to zero" | discriminates |
| Mistral-7B-v0.3 | +0.05 [−0.07, +0.17] | 1.000 | 0.0025 (1.00) | "equivalent to zero" | does not discriminate |
| Llama-3.2-3B | −0.12 [−0.27, +0.03] | 1.000 | 0.0099 (0.96) | "equivalent to zero" | does not discriminate |

Llama-3.1-8B's completion contrast is large and negative, confirmed below zero. When the item is already named, making its craft finish the task lowers Llama's predicted success by about 1.3 nats relative to the ingredient control. It is a measurable effect only, for the same gate reason as for naming.

Gemma-2-9B's completion contrast is "equivalent to zero", in a model that discriminates. Once the item is named, finishing the task adds no effect larger than 0.4 nats. Mistral and Llama-3.2-3B are also equivalent to zero, with limited interpretation.

Qwen3-8B is inconclusive. Its estimate is positive (+0.65) and its interval just touches zero (raw p 0.050), but it is far from significant after Holm adjustment.

### 4.5 Timing: does the penalty shrink when seven empty look-ups are added?

336 padded goal-role and 332 padded class-matched ingredient failures in 185 goal pairs. Beside $T$, the table gives the frozen applicability gate: the goal-minus-ingredient effect at one look-up, which must have a 95% interval below zero for $T$ to be read.

| Model | Estimate [95% CI] | Differs, Holm p | Equivalent, Holm p (power) | Role gap at one look-up [95% CI] | Verdict | Discrimination |
|---|---|---|---|---|---|---|
| Qwen3-8B | +0.80 [+0.30, +1.33] | 0.0490 | 1.000 (0.00) | +0.71 [−0.05, +1.47] | "not applicable" | discriminates |
| Llama-3.1-8B | +0.03 [−0.12, +0.18] | 1.000 | 0.0025 (0.95) | −0.84 [−1.06, −0.61] | "equivalent to zero" | does not discriminate |
| Gemma-2-9B-it | −0.13 [−0.30, +0.04] | 1.000 | 0.0255 (0.88) | −0.91 [−1.24, −0.56] | "equivalent to zero" | discriminates |
| Mistral-7B-v0.3 | −0.40 [−0.55, −0.25] | 0.0025 | 1.000 (0.97) | +1.65 [+1.28, +2.03] | "not applicable" | does not discriminate |
| Llama-3.2-3B | −0.29 [−0.40, −0.18] | 0.0025 | 0.6786 (1.00) | −0.13 [−0.38, +0.12] | "not applicable" | does not discriminate |

Two models have a penalty to shrink at one look-up, Llama-3.1-8B and Gemma-2-9B, and for both $T$ is "equivalent to zero". Adding seven empty look-ups, about 100 tokens that carry no new information, does not change the goal-role penalty by more than the margin. For Gemma, which discriminates in both padded populations, the interpretation is "no effect larger than the margin, in a model that discriminates successes from failures in these categories". For Llama it is "no effect larger than the margin; limited interpretation (does not discriminate successes from failures in these populations)".

Three models get "not applicable". For Qwen3-8B, Mistral-7B and Llama-3.2-3B, the goal-role effect at one look-up is not reliably below the ingredient effect, so under the frozen rule there is no penalty whose shrinkage $T$ could measure. In all three the differs test alone would have passed Holm's correction with an interval excluding zero. Qwen's +0.80 has Holm p 0.049, right at the threshold, and Mistral's −0.40 and Llama-3.2-3B's −0.29 are both at the 0.0025 floor. The applicability gate overrides the test by design, so none of the three is read.

### 4.6 All 25 verdicts at a glance

The verdict strings are those in `metrics.json`. Discrimination status follows each verdict.

| Model | Lift | Within-item penalty | Naming / partial completion | Completion with naming held | Timing |
|---|---|---|---|---|---|
| Qwen3-8B | confirmed above zero (discriminates) | confirmed below zero (discriminates) | inconclusive (discriminates) | inconclusive (discriminates) | not applicable (discriminates) |
| Llama-3.1-8B | equivalent to zero (does not discriminate) | confirmed below zero (discriminates) | confirmed above zero (does not discriminate) | confirmed below zero (does not discriminate) | equivalent to zero (does not discriminate) |
| Gemma-2-9B-it | confirmed below zero (discriminates) | confirmed below zero (discriminates) | confirmed below zero (discriminates) | equivalent to zero (discriminates) | equivalent to zero (discriminates) |
| Mistral-7B-v0.3 | confirmed above zero (does not discriminate) | confirmed above zero (does not discriminate) | confirmed above zero (does not discriminate) | equivalent to zero (does not discriminate) | not applicable (does not discriminate) |
| Llama-3.2-3B | equivalent to zero (does not discriminate) | confirmed below zero (does not discriminate) | inconclusive (does not discriminate) | equivalent to zero (does not discriminate) | not applicable (does not discriminate) |

Of the 25 verdicts, 12 are confirmed (7 below zero and 5 above), 7 are equivalent to zero, 3 are inconclusive and 3 are not applicable. Six of the 12 confirmed verdicts are in a model that discriminates for that contrast, and so are read as distortions: Qwen's lift and penalty, Llama-3.1's penalty, and Gemma's lift, penalty and naming contrast. The other six are measurable effects only.

## 5. Readings and rejection rules that follow from the verdicts

These rules add no test. They combine the confirmed verdicts with the competence flags, as the design fixed in advance (section 8, 15.R). The reading strings are copied from `metrics.json`.

**Per-model readings of the penalty.** These combine the within-item penalty with the naming and completion contrasts, using the design's table:

- *Qwen3-8B:* "penalty present, mechanism unresolved". The penalty is confirmed, but both the naming and the completion contrast are inconclusive.
- *Llama-3.1-8B:* "completion penalty, naming constrained; measurable effect only; does not discriminate successes from failures for naming, completion". The penalty is confirmed below zero, completion is confirmed below zero, and naming is confirmed *above* zero. That matches the design's completion-only row. The suffix records that the two mechanism contrasts are measured where Llama does not discriminate.
- *Gemma-2-9B-it:* "naming / partial-completion penalty (naming the crafted item, or completing one item of a named two-item goal, lowers predicted success; a completion part cannot be separated from the dilution of naming)".
- *Mistral-7B-v0.3:* "completing a named goal raises predicted success (no penalty); measurable effect only; does not discriminate successes from failures for parent_penalty, naming, completion".
- *Llama-3.2-3B:* "penalty present, mechanism unresolved; measurable effect only; does not discriminate successes from failures for parent_penalty, naming, completion".

**Per-model readings of the lift.** Qwen3-8B, "lift present". Llama-3.1-8B, "no lift larger than the margin; measurable effect only; does not discriminate successes from failures for lift". Gemma-2-9B-it, "negative lift". Mistral-7B-v0.3, "lift present; measurable effect only; does not discriminate successes from failures for lift". Llama-3.2-3B, "no lift larger than the margin; measurable effect only; does not discriminate successes from failures for lift".

**Per-model readings of timing.** Llama-3.1-8B, "no sensitivity to added empty-inventory steps larger than the margin; measurable effect only; does not discriminate successes from failures for timing". Gemma-2-9B-it, "no sensitivity to added empty-inventory steps larger than the margin". Qwen3-8B, "not applicable". Mistral-7B-v0.3 and Llama-3.2-3B, "not applicable; measurable effect only; does not discriminate successes from failures for timing".

**Rejection rules.**

- *R1, no penalty.* For any model that discriminates for the penalty, a penalty equivalent to zero or confirmed above zero rejects the penalty component of the decomposition for that model. R1 was testable for Qwen3-8B, Llama-3.1-8B and Gemma-2-9B, and it fires for none of them. It is not testable for Mistral-7B or Llama-3.2-3B, because neither discriminates for the penalty. Mistral's positive penalty therefore does not trigger it. The round-1 falsifier F1 (no penalty in Llama-3.1) is recorded as false.
- *R2, the penalty is not shared.* Among the models that discriminate for the penalty, R2 would fire if some model other than Llama-3.1 had a penalty equivalent to zero or above zero and none had one confirmed below zero. Qwen3-8B and Gemma-2-9B are both confirmed below zero, so R2 does not fire. F3 is recorded as false.
- *R5, added steps deepen the penalty.* For a model that discriminates for timing, a timing contrast confirmed below zero would reject the predicted attenuation of the penalty with added steps. R5 was testable for Qwen3-8B and Gemma-2-9B and fires for neither. Qwen's timing is not applicable, and Gemma's is equivalent to zero.

**Across models.** The analysis's pattern is "penalty general across the competent models". The models competent for the penalty are Qwen3-8B, Llama-3.1-8B and Gemma-2-9B, and all three are confirmed below zero. In the design's words, the penalty is general across these families under the clean prompt. The design's further across-model row, "the same reading of the decomposition in every model where it is read", does not hold. Among the models that discriminate for all three penalty contrasts, Qwen's reading is "mechanism unresolved" and Gemma's is "naming / partial-completion penalty". Llama-3.1's "completion penalty, naming constrained" rests on contrasts where it does not discriminate.

## 6. Descriptive diagnostics: R3, R4 and the central-question mapping

These sit outside the Holm family and carry no familywise error control.

### 6.1 R3: do the within-item penalty and the between-item role gap agree?

R3 compares two measures of the same penalty. The first is the within-item penalty $P$. The second is the round-2 *role gap*: the aligned-minus-swapped effect on the 500 goal-role failures minus the same effect on the 496 class-matched ingredient failures. On these rows the role gap equals $D_n + D_c$. The design also defines a *matched* role gap. It averages the role gap within cells that share history-length bin, kind of missing inputs (all craftable, all raw or mixed) and number of missing inputs, and so removes the obvious item-structure differences between the roles. R3 fires when three things hold together:

- the model discriminates for the penalty, naming and completion;
- the penalty is confirmed in one direction while the unmatched role gap's 95% interval lies on the other side of zero;
- the matched role gap's interval lies on that same other side.

| Model | Within-item penalty verdict | Role gap [95% CI] | Matched role gap [95% CI] (cells, items) | R3 testable | R3 fires |
|---|---|---|---|---|---|
| Qwen3-8B | confirmed below zero | +0.74 [+0.02, +1.48] | +0.77 [+0.01, +1.54] (13, 981) | yes | **yes** |
| Llama-3.1-8B | confirmed below zero | −0.49 [−0.68, −0.30] | −0.44 [−0.68, −0.20] (13, 981) | no | no |
| Gemma-2-9B-it | confirmed below zero | −0.62 [−0.87, −0.35] | −0.37 [−0.67, +0.004] (13, 981) | yes | no |
| Mistral-7B-v0.3 | confirmed above zero | +0.98 [+0.72, +1.26] | +1.49 [+1.13, +1.82] (13, 981) | no | no |
| Llama-3.2-3B | confirmed below zero | +0.03 [−0.19, +0.26] | +0.32 [+0.07, +0.54] (13, 981) | no | no |

(The intervals are from the 20,000-resample computation used by R3. The descriptive copy of the role gap in `descriptive.role_gap_main` uses 2,000 resamples, and for Qwen it gives [+0.04, +1.49].)

**R3 fires for Qwen3-8B.** The analysis's note reads "P and the role gap disagree, and the matched role gap confirms the role gap". On one fixed craft, naming and completing the goal lowers Qwen's $L$ by about a nat compared with making the craft an ingredient of the goal. Yet across items, Qwen's goal-role crafts get a *larger* aligned-minus-swapped effect than its ingredient crafts, by about +0.75 nats, and matching on item structure does not remove that. By the frozen wording this "counts against the decomposition for this model (descriptive diagnostic; no familywise error control)". The two-component account adds a role-independent lift to a negative goal-role penalty, so it predicts a role gap at or below zero whenever the penalty is negative. Qwen's between-item measure contradicts that prediction.

Two cautions apply. Both intervals only just exclude zero, with lower ends at +0.018 and +0.014. And R3 has no multiplicity control. *(revised 2026-11-25)* The reading is also numerically fragile: when the same records are rescored in the newer software stack (`serve2`) on the same GPU type, the unmatched role gap's lower end moves from about +0.01/+0.02 to −0.019, so R3 would not fire there (`numerical_sensitivity.md` §5.1). The matched role gap was not recomputed. "R3 fires for Qwen3-8B" is therefore not established. For Gemma-2-9B the penalty and role gap agree in sign, so R3 does not fire. For Llama-3.1-8B they also agree, but R3 is not testable because Llama does not discriminate for naming and completion.

### 6.2 R4: does the lift carry the Qwen–Llama difference?

R4 was withdrawn as a rejection rule at sign-off and is now purely descriptive. It reports the difference in lift between Llama-3.1-8B and Qwen3-8B on the same 990 ingredient failures, with a paired goal-pair bootstrap interval of 2,000 resamples:

$$\Lambda_{\text{Llama-3.1}} - \Lambda_{\text{Qwen3}} = -0.58 \quad [-0.90,\ -0.23].$$

Qwen's lift is reliably larger than Llama's on the same items. That is consistent with the lift carrying part of the Qwen–Llama difference in the total effect, which is the account's second claim. Llama's lift is measured where Llama does not discriminate, and no test of the difference is in the family, so this is a description only.

The design's across-model table (section 8) also lists a descriptive row: whether the ordering of the lift matches the ordering of the overall goal effect. The analysis does not code this row. It can be read off the reported estimates, and it holds. Among the three models competent for the penalty, the lift orders Qwen (+0.67) above Llama-3.1 (+0.09) above Gemma (−0.30), and the overall main-failure effect (section 7.1) orders them the same way: +1.28, −0.02 and −0.41. Across all five models the two orderings also coincide (Qwen, Mistral, Llama-3.1, Llama-3.2-3B, Gemma). With five models this is a description, not a test.

### 6.3 The central-question mapping

For each model the analysis gives one of four labels:

- "penalty with inventory-consistent tracking";
- "penalty with shortcut-consistent tracking";
- "no confirmed penalty; shortcut-consistent or non-discriminating";
- "undetermined".

The label depends on three inputs: the penalty's verdict, the model's discrimination for the penalty, and how well two text heuristics *track* the model's yes/no scores in the goal-role population compared with the true inventory rule. The heuristics are H1 (every input's name is mentioned somewhere in the history) and H2∧H9 (the counts gathered reach each input's need, and the output is not a top item of the list). The true inventory rule is H6. Tracking is measured as the AUC of the model's score between the items a heuristic calls a success and those it calls a failure. H2∧H9 calls every goal-role craft a failure, so its tracking AUC there is undefined (NaN by construction), and the comparison rests on H1 alone.

| Model | Penalty verdict | Discriminates for the penalty | H1 minus H6 tracking AUC [95% CI] | Answer |
|---|---|---|---|---|
| Qwen3-8B | confirmed below zero | yes | +0.001 [−0.034, +0.040] | "undetermined" |
| Llama-3.1-8B | confirmed below zero | yes | −0.022 [−0.059, +0.014] | "penalty with inventory-consistent tracking" |
| Gemma-2-9B-it | confirmed below zero | yes | +0.023 [−0.010, +0.056] | "undetermined" |
| Mistral-7B-v0.3 | confirmed above zero | no | −0.004 [−0.045, +0.036] | "no confirmed penalty; shortcut-consistent or non-discriminating" |
| Llama-3.2-3B | confirmed below zero | no | −0.050 [−0.088, −0.015] | "undetermined" |

For Llama-3.1-8B, the model the question is about, the answer is "penalty with inventory-consistent tracking". Its penalty is confirmed, it discriminates for the penalty, and the name-mention shortcut H1 tracks its goal-role scores no better than the true rule. The point estimate is slightly lower, and the interval spans zero.

Qwen3-8B and Gemma-2-9B are "undetermined", because H1's tracking is fractionally *above* H6's by point estimate (+0.001 and +0.023) without its interval excluding zero. Qwen's case turns on a difference of 0.001 in AUC. Llama-3.2-3B is "undetermined", because its penalty is confirmed but it does not discriminate for the penalty, a combination no label covers. Mistral-7B gets "no confirmed penalty; shortcut-consistent or non-discriminating".

These labels describe how the scores co-vary with heuristics. "Inventory-consistent" means the true rule tracks Llama's goal-role scores at least as well as the name-mention shortcut. It does not mean Llama computes the inventory.

## 7. Other pre-registered descriptive readouts

These are pre-registered in the design (section 9) and reported with 95% pair-bootstrap intervals (2,000 resamples) and no p-values. They change no verdict. Only the readouts that bear on the interpretation are shown. The rest are in `per_model.<model>.descriptive` in the metrics file.

### 7.1 The overall goal effect under the clean prompt

The overall goal effect is aligned minus swapped over the 1,500 main failures, the round-1 estimand, here under the clean prompt.

| Model | All main failures | Goal role (510) | Ingredient role (990) | Successes (600) |
|---|---|---|---|---|
| Qwen3-8B | +1.28 [+1.01, +1.57] | +2.47 [+1.95, +2.97] | +0.67 [+0.32, +1.01] | +3.16 [+2.36, +3.89] |
| Llama-3.1-8B | −0.02 [−0.11, +0.08] | −0.23 [−0.37, −0.09] | +0.095 [−0.002, +0.191] | +0.37 [+0.22, +0.52] |
| Gemma-2-9B-it | −0.41 [−0.52, −0.30] | −0.63 [−0.86, −0.39] | −0.30 [−0.42, −0.19] | +1.00 [+0.82, +1.18] |
| Mistral-7B-v0.3 | +0.62 [+0.50, +0.73] | +1.29 [+1.05, +1.55] | +0.27 [+0.19, +0.35] | +1.24 [+1.03, +1.46] |
| Llama-3.2-3B | −0.06 [−0.17, +0.05] | −0.11 [−0.29, +0.09] | −0.04 [−0.17, +0.08] | +0.36 [+0.19, +0.53] |

Under the clean prompt, Qwen's effect stays positive and Llama-3.1's is near zero. Both match the design's descriptive expectations ("Qwen above zero; Llama near zero or slightly negative"). Gemma-2-9B has a negative overall effect on failures, larger than Llama's, and it is negative in both roles. On successes, every model's aligned line raises predicted success.

### 7.2 The penalty on parent records, by subset and by item kind

| Model | Strict parents (no shared word; 176 items, 29 pairs) | Strict minus overlapping parents | Penalty on parent *successes* (207) | Never obtained (522) | Too few (150) |
|---|---|---|---|---|---|
| Qwen3-8B | −0.89 [−2.19, +0.31] | +0.17 [−1.36, +1.56] | −0.67 [−1.67, +0.31] | −1.40 [−2.10, −0.79] | +0.33 [−1.03, +1.67] |
| Llama-3.1-8B | −2.03 [−2.72, −1.39] | −0.47 [−1.21, +0.24] | −1.43 [−1.84, −1.04] | −1.71 [−2.02, −1.43] | −1.53 [−1.93, −1.11] |
| Gemma-2-9B-it | −1.70 [−2.30, −1.12] | −0.58 [−1.24, +0.07] | +0.05 [−0.18, +0.28] | −1.50 [−1.73, −1.26] | −0.52 [−0.84, −0.16] |
| Mistral-7B-v0.3 | +0.23 [−0.06, +0.53] | −0.05 [−0.36, +0.30] | +0.61 [+0.44, +0.79] | +0.16 [+0.01, +0.30] | +0.64 [+0.39, +0.90] |
| Llama-3.2-3B | −0.56 [−0.97, −0.18] | −0.08 [−0.54, +0.36] | +0.28 [+0.07, +0.48] | −0.70 [−0.89, −0.53] | +0.22 [−0.03, +0.45] |

**Partial naming.** Under the parent line, G is often partly named, because the parent's name shares a word with G's in 499 of the 675 failures. For Llama-3.1 and Gemma the penalty is clearly present on the strict parents, which share no word with G. In those two models it does not depend on partial naming of G in the parent line. For no model does the strict-minus-overlapping comparison exclude zero. By the design's rule, that does not show that partial naming plays no part. Qwen's strict subset (29 pairs) has an interval across zero. That is not evidence that partial naming drives Qwen's pooled penalty.

**Failure-specific or general.** The penalty on parent *successes* separates two patterns. For Llama-3.1-8B the naming-and-completing line lowers predicted success on crafts that work (−1.43) about as much as on crafts that fail (−1.68). The shift is general and does not depend on whether the craft can succeed. For Gemma-2-9B the penalty is confined to failures (+0.05 on successes). Its goal line moves failing crafts toward the correct answer and leaves working crafts alone. Qwen's estimate on successes is negative, with an interval across zero.

**Qwen's pre-registered class expectation.** Qwen's penalty is more negative on never-obtained records (−1.40) than on too-few records (+0.33). This matches the design's expectation for Qwen in direction. The design pre-registered no interval on that difference, and none is given here.

### 7.3 The within-item split on the extended list, and the matched sensitivity analyses

The parent records also carry the two-item line on the extended list. That splits $P$ exactly into a "completion" part (aligned minus two-item) and a "naming" part (two-item minus parent), on one fixed craft. The design reports this split descriptively. The between-item naming and completion contrasts were also recomputed within matched cells (design 14.V2) as a sensitivity analysis.

| Model | Within-item: aligned minus two-item | Within-item: two-item minus parent | Matched naming / partial completion | Matched completion with naming held |
|---|---|---|---|---|
| Qwen3-8B | −0.46 [−0.96, +0.09] | −0.55 [−1.04, −0.09] | −0.22 [−0.98, +0.55] | +0.99 [+0.27, +1.68] |
| Llama-3.1-8B | −2.00 [−2.19, −1.81] | +0.32 [+0.07, +0.55] | +0.78 [+0.54, +1.03] | −1.22 [−1.45, −0.99] |
| Gemma-2-9B-it | −0.82 [−1.00, −0.63] | −0.45 [−0.65, −0.26] | −0.57 [−0.77, −0.36] | +0.20 [−0.04, +0.50] |
| Mistral-7B-v0.3 | −0.22 [−0.31, −0.12] | +0.49 [+0.35, +0.63] | +1.41 [+1.06, +1.72] | +0.09 [−0.06, +0.22] |
| Llama-3.2-3B | −0.49 [−0.63, −0.35] | −0.01 [−0.18, +0.17] | +0.48 [+0.22, +0.77] | −0.16 [−0.37, +0.01] |

For Llama-3.1-8B every measure tells the same story. Within one craft, completion carries the whole penalty (−2.00) and naming alone slightly raises $L$ (+0.32). Between items, matched or not, naming is positive and completion is strongly negative.

For Gemma-2-9B the within-item split puts part of the penalty on each side, and more on completion (−0.82) than on naming (−0.45). The between-item contrasts put it on naming (−0.57) and none on completion. The two measures therefore disagree about how Gemma's penalty divides, though not about its sign.

For Qwen3-8B the matched completion contrast is positive (+0.99) where the within-item completion part is near zero or negative. This is the same between-item against within-item tension that R3 flags.

### 7.4 Other expectations from the design

The design stated three further descriptive expectations.

The first was that the lift would transfer: on parent records, parent line minus swapped line should be close to the model's lift. The transfer is close for Mistral (+0.34 against a lift of +0.27) and Qwen (+0.87 against +0.67). It is less close for the others: Llama-3.1 +0.29 against +0.09, Gemma +0.02 against −0.30, and Llama-3.2-3B −0.24 against −0.04.

The second was that adding the parent's lines under the aligned goal would change $L$ little. That did not hold for three models. The extended list raised $L$ under the aligned line by +0.77 for Qwen, +0.82 for Llama-3.1 and +1.36 for Gemma, against +0.05 for Mistral and −0.11 for Llama-3.2-3B. $P$ is computed with both lines on the same extended list, so this does not enter $P$. It does make the penalty measured on each line's own display larger (Qwen −1.78, Llama-3.1 −2.49, Gemma −2.64).

The third was that one look-up against none would move the goal-role effect up slightly. It did not for Llama-3.1 (−0.10 [−0.19, −0.01]) and moved it down for Gemma and Llama-3.2-3B. Only Mistral moved up (+0.24).

On the real history-length bins (not padding), Llama-3.1's penalty shrinks with history length: −2.05 at 1–3 steps, −1.66 at 4–8 and −1.28 at 9–12. Gemma's barely changes (−1.34, −1.30, −1.16).

### 7.5 Yes/no versions

The design pre-registered yes/no versions of the lift, the naming and the completion contrasts as descriptive readouts:

| Model | Lift, yes/no | Naming / partial completion, yes/no | Completion with naming held, yes/no | Overall main-failure effect, yes/no |
|---|---|---|---|---|
| Qwen3-8B | +0.30 [−0.03, +0.65] | +0.94 [+0.28, +1.61] | +0.03 [−0.55, +0.61] | +0.93 [+0.64, +1.24] |
| Llama-3.1-8B | +0.14 [+0.09, +0.20] | +0.24 [+0.10, +0.39] | +0.09 [−0.03, +0.20] | +0.24 [+0.18, +0.30] |
| Gemma-2-9B-it | −0.13 [−0.29, +0.01] | −0.26 [−0.54, +0.01] | +0.45 [+0.21, +0.72] | +0.05 [−0.07, +0.19] |
| Mistral-7B-v0.3 | +0.84 [+0.75, +0.93] | +2.03 [+1.70, +2.38] | +0.19 [+0.04, +0.34] | +1.66 [+1.49, +1.84] |
| Llama-3.2-3B | −0.02 [−0.03, +0.00] | +0.08 [+0.05, +0.11] | −0.06 [−0.09, −0.03] | −0.02 [−0.04, +0.00] |

The anchored yes/no scale does not reproduce the two-candidate pattern closely. Llama-3.1-8B's strong two-candidate completion penalty (−1.26) is +0.09 on yes/no. Gemma's completion contrast is equivalent to zero on two-candidate but +0.45 on yes/no. The confirmatory scale is the two-candidate one, and the yes/no versions are descriptive.

## 8. The pre-registered predictions against the results

The design's predictions table (section 7, revised by 14.R4 and 14.V3) recorded expectations so that they could be checked afterwards. No confirmatory rule depends on them.

| Model and contrast | What was predicted or expected | Result |
|---|---|---|
| Qwen3-8B, lift | above zero | confirmed above zero (+0.67) |
| Qwen3-8B, penalty | no overall direction; more negative on never-obtained than too-few records | confirmed below zero (−1.02); never obtained −1.40 against too few +0.33 (descriptive) |
| Llama-3.1-8B, lift | equivalent to zero | equivalent to zero (+0.095), limited interpretation |
| Llama-3.1-8B, penalty | below zero, about −0.5 to −0.9 | confirmed below zero, at −1.68, larger than expected |
| Llama-3.1-8B, completion with naming held | motivated expectation below zero, not a firm prediction | confirmed below zero (−1.26), measurable effect only |
| Gemma-2-9B, Mistral-7B, Llama-3.2-3B, penalty | below zero (the account's generality claim only) | Gemma confirmed below zero; Llama-3.2-3B confirmed below zero (measurable only); Mistral confirmed above zero (measurable only) |
| Every other cell | no directional prediction | see section 4 |
| The account: lift | differs across models and carries the between-model difference | lifts range from −0.30 to +0.67; Llama minus Qwen −0.58 [−0.90, −0.23]; the lift ordering matches the overall effect ordering (descriptive) |
| The account: penalty | below zero | below zero in all three models that discriminate for it |
| The account: timing | sensitivity reading above zero; no-sensitivity reading equivalent to zero | equivalent to zero where readable (Gemma; Llama-3.1 with limited interpretation); not applicable elsewhere |

## 9. Interpretation: what round 2 says about Llama's negative TextCraft effect

**The penalty survives holding the craft fixed, and it is not specific to Llama.** The question round 2 was built to answer first is whether round 1's goal-crafting penalty survives when the craft is held fixed. It does. In all three models whose yes/no answers discriminate on goal crafting (Qwen3-8B, Llama-3.1-8B and Gemma-2-9B), naming the crafted item as the goal, so that crafting it completes the task, lowers predicted success by about 1 to 1.7 nats. The comparison is with making the same item an ingredient of the goal, on the identical action, history and recipe list. Each is confirmed after Holm correction over 50 tests. The rejection rules R1 and R2 do not fire, and the analysis's across-model reading is "penalty general across the competent models".

This shows that the goal line's naming-and-completing status changes $L$ for a fixed craft. The penalty therefore does not need differences in recipe structure to appear: under the clean prompt it appears on one and the same craft. It does not show *why* the models do this. In particular it does not show that they "think" the task is premature.

**For Llama-3.1-8B, the answer to the central question is a penalty on goal crafting, alongside no measurable lift on ingredients.** By the coded mapping the answer is "penalty with inventory-consistent tracking": the penalty is confirmed in a population where Llama discriminates, and the name-mention shortcut does not track Llama's goal-role scores better than the true inventory rule. Llama's lift on ingredients is equivalent to zero (+0.095, within its ±0.38 margin). With a penalty on goal crafts and no lift on ingredients, the account predicts a negative effect on goal-role items, a null one on ingredient items and a negative role gap. The clean-prompt scores show exactly that: the goal-role effect is −0.23, the ingredient effect +0.095 and the role gap −0.49. The descriptive R4 difference points the same way, with Llama's lift 0.58 nats below Qwen's on the same items.

Under the clean prompt, Llama's overall effect on all main failures is near zero (−0.02 [−0.11, +0.08]), not negative. Two thirds of those failures are ingredient crafts, where Llama shows no effect. Round 1's −0.19 was measured under the verified round-1 prompt, which still contained sentences addressed to the agent, so it is a different quantity (design section 2).

Two qualifications keep this short of the account's full story for Llama. First, the no-lift result sits in the population where Llama does not discriminate (ingredient AUC 0.556). The verdict there is "limited interpretation". We cannot tell a model that sees the ingredient's state and ignores the goal from one that cannot read the ingredient's state and has nothing for the goal to distort. In that sense the original design's three options are not cleanly separated. Llama looks like the "combination" pattern that design 14.U5 first sketched: a penalty where it discriminates, a null lift where it does not. The coded label that superseded that sketch is "penalty with inventory-consistent tracking", and that coded label is the one that applies. Second, the descriptive readouts show that Llama's penalty is not failure-specific. On parent successes the naming-and-completing line lowers $L$ by −1.43, about as much as on failures. That fits a general downward shift on goal crafts more than a sharpened state judgement. It is a descriptive observation, not a test.

**Naming against completion.** For Llama-3.1-8B every measure points the same way. Completing the named task lowers predicted success: −1.26 between items, −2.00 within one craft, and −1.22 matched. Naming the item without completing the task raises it slightly. The reading is "completion penalty, naming constrained". The between-item naming and completion contrasts are measured where Llama does not discriminate on the ingredient control, so this decomposition is a "measurable effect only". It is not read as a distortion of a discriminating prediction.

Gemma-2-9B discriminates everywhere, and it gives the other reading, "naming / partial-completion penalty". Its naming contrast is confirmed below zero, and its completion contrast is equivalent to zero. Gemma's within-item split, however, puts more of the penalty on completion than on naming. The two measures agree on the existence of the penalty but not on how it divides.

For Qwen3-8B the mechanism is unresolved: both contrasts are inconclusive, and equivalence is unreachable at its spread. So round 2 does not find one decomposition reading shared across families.

**Qwen3-8B does not fit the additive account.** R3 fires for Qwen. Within one craft Qwen penalizes the named, completing goal by about a nat. Across items, its goal-role crafts get a *larger* aligned-minus-swapped boost than its ingredient crafts (+0.74, and +0.77 after matching on item structure). An account that adds a role-independent lift to a negative goal-role penalty cannot produce both. By the frozen wording this "counts against the decomposition for this model". It is a descriptive diagnostic without familywise control, and both intervals clear zero by less than 0.02 nats, so it is a weak signal. *(revised 2026-11-25)* A change of software stack alone undoes it (`numerical_sensitivity.md` §5.1), so it is not established. If it holds, Qwen's goal effect needs something the two-component account lacks, for example a relevance boost that is larger for the goal item itself than for its ingredients, with the completion penalty acting within it.

**Gemma-2-9B is a second model with a negative TextCraft effect, and for a partly different reason.** Gemma's overall clean-prompt effect is −0.41, more negative than Llama's. Unlike Llama, Gemma discriminates in every population. Its lift is *negative* (confirmed below zero, −0.30): the aligned line makes Gemma more pessimistic about failing ingredient crafts. Its penalty (−1.27) is confined to failures, with +0.05 on parent successes. On failures, then, Gemma's goal line pushes predictions toward the correct answer in both roles. On successes its aligned line raises predicted success (+1.00), which is also toward the correct answer. That is still a goal leak, because the true reply does not depend on the goal, but on these items it is a leak toward accuracy. The account did not anticipate a negative lift. It made no prediction for Gemma.

**"Largest early" is not about elapsed steps, where it can be read.** In the two models with a penalty at one look-up, adding seven empty look-ups changed the penalty by less than the 0.4-nat margin. Gemma discriminates, and Llama-3.1 is read with limited interpretation. This is consistent with the content reading, in which the penalty depends on what the history has gathered and not on how long it is. It is not proof of it, and R5 does not fire. Llama's penalty does shrink along the real history-length bins (from −2.05 to −1.28), and padding reproduces none of that. For the other three models the frozen gate makes timing "not applicable", although their timing tests alone would have passed Holm. The design reads timing only as the change of an existing penalty, and those three models had no reliable penalty at one look-up to change.

**Mistral-7B and Llama-3.2-3B.** Neither discriminates for any contrast, so every effect they show is a "statistically measurable goal effect" only. Mistral answers Yes to 99.9% of failures, and its effects run in the ALFWorld-like direction: positive lift, positive penalty, positive naming. Llama-3.2-3B shows a confirmed negative penalty (−0.50) that resembles the competent models'. It fails the penalty's gate only through its 96.9% Yes rate on failures, since its goal-crafting AUCs (0.769 and 0.725) pass 0.70. The frozen wording "does not discriminate" covers both routes to failing the gate. For Llama-3.2-3B, "ranks goal crafts above chance but answers Yes to nearly all of them" would be more exact. This note keeps the frozen wording.

**What would change these conclusions.**

- The penalty conclusion rests on three discriminating models and one prompt. A model that discriminates on goal crafting and shows a penalty equivalent to zero or above zero would fire R1. If it is not Llama-3.1, it would also count toward R2. Round 3's contemporary models (`textcraft-r3`) are the planned test.
- Llama's "no lift" would mean more if Llama discriminated on ingredients. A version of the ingredient population on which Llama passes the gate, for example with the inventory shown, would separate "ignores the goal" from "cannot read the state".
- Qwen's R3 result is marginal and uncorrected. A replication with more goal-role items, or on round 3's Qwen3.5, would show whether the between-item and within-item measures really disagree for Qwen.
- The naming and completion split rests on the constructed two-item line and the four assumptions of 14.U4. Gemma's within-item and between-item splits disagree. That is a reason to treat any single model's decomposition reading as provisional.
- Everything here is one scoring run per model, on scripted goal-blind walks, under one prompt and on one GPU type, so these are not agent histories.

**Implications.** The goal-crafting penalty is not specific to Llama. It belongs alongside the positive goal effect as a second, opposite way the task leaks into TextCraft predictions. Which of the two dominates a model's total depends on its ingredient-level response. The intermediate lift is positive for Qwen and Mistral, about zero for both Llamas and negative for Gemma, and its ordering matches the ordering of the total effects. For reporting, the TextCraft effect should not be summarized by the sign of one aligned-minus-swapped number. The within-item penalty and the lift answer different questions. In Qwen, the descriptive R3 diagnostic suggests that they do not combine additively.

## 10. Provenance

**Freeze.** Tag `textcraft-r2-design-frozen-2026-11-23`, commit `4ccf84be1a104d222ed6af1a1d49eb62f787b89b` (`4ccf84b`). The checkout on ICE was `text-wm-fix42`, clean (`git_dirty: false` in every run's `inputs.json`), at that commit.

**Inputs** (sha256, identical in all 20 score shards and the analysis):

- records `wf/textcraft/data/r2_items.jsonl`: `e6da991f9eb6c927213d31a6c22814c8080db1945543060e67779eb7e47c22f0`, matching design 14.C;
- manifest `wf/textcraft/data/r2_manifest.json`: `ef0b3af92aab98029b3dda52cd6fdc7321a75717449d6bf1d6c07cef1e771bda`;
- in-context examples `wf/textcraft/data/fewshot.json`: `e6ddab93966981f9a6bfa70cd26ce0a1552c733a1ac46142f306ea41adfb1253`.

**Scoring** (`textcraft_r2_score_<model>`, four shards each, one ICE job per shard). Every shard's `git_sha` is `4ccf84b`, and every shard ran on an NVIDIA L40S in bfloat16.

| Model | ICE jobs (shards 0–3) | Pinned revision (every score) | Attention | Scores |
|---|---|---|---|---|
| Qwen3-8B | 7247899, 7247900, 7247901, 7247902 | b968826d9c46 | sdpa | 25,743 |
| Llama-3.1-8B | 7247903, 7247904, 7247905, 7247906 | 0e9e39f249a1 | sdpa | 25,743 |
| Gemma-2-9B-it | 7247907, 7247908, 7247909, 7247910 | 11c9b309abf7 | eager | 25,743 |
| Mistral-7B-v0.3 | 7247911, 7247912, 7247913, 7247914 | c170c708c41d | sdpa | 25,743 |
| Llama-3.2-3B | 7247915, 7247916, 7247917, 7247918 | 0cb88a4f764b | sdpa | 25,743 |

The shards hold 829, 624, 702 and 827 records, which is 2,982 records and 25,743 scores per model and 128,715 in all. The following were checked read-only on the cluster when this note was written:

- no (record, goal line) pair is duplicated, and none is missing (`missing_scores` is 0 for every model in the metrics);
- every score's two-candidate log-odds is finite;
- every score carries its model's pinned revision;
- each of the 2,982 records carries one 16-hex-digit item hash, identical across all five models (sha256 of the sorted `transition_id:item_hash` list: `4d1f10dab78b5d5b2033520fa624389d1b551645739d91caf25527455a497122`);
- the tokenization gate passed in every shard, with zero problems, zero golden-rendering problems and no input over 4,096 tokens;
- the longest inputs (1,819, 1,837, 1,706, 2,054 and 1,837 tokens) equal the CPU check's in design 14.C.

**Smoke runs** (`textcraft_r2_smoke_<model>`, 20 records and 188 scores each, jobs 7246002–7246006, L40S, at `4ccf84b`) were read for format and timing only, as the design required.

**Analysis.** `textcraft_r2_analysis`, ICE job 7247919, a CPU job (`gpu: null`) on host atl1-1-01-005-1-2, at `4ccf84b`, run on 2026-11-23. Status PASS. The family has 50 tests, and no test entered padded (no missing or NaN p). The analysis wrote `metrics.json` (sha256 `85c76f591d5673c4a88a711f3438222c76e9d9d7ebc27fb23c6d56bf89158e4f`, copied unchanged to `notes/textcraft_r2_metrics.json`) and `STATUS.md`. It does not write a separate `REPORT.md`. The STATUS text is in section 11.

**Not-a-number values in the metrics.** There are 1,530, all in heuristic-tracking readouts, and all undefined by construction. In those cells one side of the comparison is empty: a heuristic calls every item in the population one class (H2∧H9 in the goal-role population, or the within-successes and within-failures splits), or the two-candidate neutral arm does not exist (parent records). None is in a contrast, a family test or a verdict.

## 11. The analysis run's STATUS text

Copied verbatim from `runs/textcraft_r2_analysis/STATUS.md` (sha256 `0aaa0b14…5deae4b`). In it, "naming" is the naming / partial-completion contrast, "completion" is completion with naming held, "parent_penalty" is the within-item penalty, and "timing" is the padding contrast. The parenthesis after each verdict is the contrast-specific discrimination status.

```
# STATUS — textcraft_r2_analysis — PASS

qwen3-8b: overall gate pass; lift confirmed above zero (discriminates); parent_penalty confirmed below zero (discriminates); naming inconclusive (discriminates); completion inconclusive (discriminates); timing not applicable (discriminates)
llama3.1-8b: overall gate fail; lift equivalent to zero (does not discriminate); parent_penalty confirmed below zero (discriminates); naming confirmed above zero (does not discriminate); completion confirmed below zero (does not discriminate); timing equivalent to zero (does not discriminate)
gemma2-9b-it: overall gate pass; lift confirmed below zero (discriminates); parent_penalty confirmed below zero (discriminates); naming confirmed below zero (discriminates); completion equivalent to zero (discriminates); timing equivalent to zero (discriminates)
mistral-7b-v0.3: overall gate fail; lift confirmed above zero (does not discriminate); parent_penalty confirmed above zero (does not discriminate); naming confirmed above zero (does not discriminate); completion equivalent to zero (does not discriminate); timing not applicable (does not discriminate)
llama3.2-3b: overall gate fail; lift equivalent to zero (does not discriminate); parent_penalty confirmed below zero (does not discriminate); naming inconclusive (does not discriminate); completion equivalent to zero (does not discriminate); timing not applicable (does not discriminate)
across: penalty general across the competent models
```
