# Numerical sensitivity of small effects and equivalence tests, measured from the archived paired rescorings (2026-11-25)

*Exploratory and post hoc. No model was scored for this note and no Slurm job was submitted. Everything below comes from score files the project already had on ICE, copied read-only. Code is in `scripts/numsens/` and its outputs are in `scripts/numsens/out/*.json`. Section 9 says how to rerun it.*

## 1. The question and the short answer

The question was whether the project's small effects and equivalence tests can be trusted given how the scores are computed. Every score is a log-probability difference read from a model running in bfloat16 (bf16) on a GPU, and bf16 arithmetic is not exactly reproducible across hardware and software. Until now the project has mostly reasoned about this as if it were independent rounding noise that averages out. The request was to measure it instead. That meant finding every pair of runs that scored the same prompts under different numerics, measuring how scores and the analysed contrasts move between them, and checking whether any reported conclusion could change.

The short answer has five parts.

1. **Within one environment, scores are deterministic.** "Environment" here means the same software stack, the same GPU type, the same code path and the same batch shape. Every same-environment rerun found reproduces every score bit for bit. That covers about 36,000 scores in seven pairs of runs, four models, both benchmarks and three GPU types, on different nodes and days. A same-environment rerun therefore checks provenance, but it tells you nothing about numerical sensitivity.
2. **Changing any one ingredient adds noise of a predictable size.** The ingredients are the GPU type, the software stack, the numerical precision, the inference engine, the attention implementation or the batch shape. The size depends mainly on how large the model's logits are, which in turn sets the spacing of the bf16 grid.
   - For Qwen3-8B, whose yes/no logits sit between 32 and 64, an individual score moves by about 0.25 to 0.39 nats on average. An item's analysed contrast moves with a standard deviation of about 0.45 to 0.63 nats.
   - For Llama-3.1-8B, Mistral-7B and Gemma-2-9B, the per-score figure is 0.07 to 0.12 nats and the contrast figure 0.14 to 0.26.
3. **It is not independent rounding noise, in three ways.**
   - The two arms of a contrast move almost independently. Their differences correlate only about 0 to 0.3, so pairing cancels little of the noise.
   - A change of environment moves conditions by different average amounts, so it shifts the mean of a contrast systematically, not just randomly. The largest case is Qwen3-8B's step-0 yes/no aware-minus-neutral contrast, which moves by −0.117 nats between an A100 and an L40S, 7 standard errors from zero.
   - In the TextCraft yes/no readout the differences have heavy tails: excess kurtosis 6.6 for Qwen and 15.5 for Gemma, counting exact repeats.
4. **For means over hundreds to thousands of items, the resulting "numerical floor" is small.** It is about 0.04 to 0.08 nats (one standard deviation) for Qwen-sized logits and about 0.01 to 0.03 nats for the others. That is far below the project's large effects and below most of its margins.
5. **One reported conclusion flips under an environment change alone.**
   - TextCraft round 2 reports that rejection criterion R3 "fires" for Qwen3-8B. When the same records are rescored in the newer stack on the same GPU type, R3 no longer fires: the role gap's interval moves from just above zero to just below it.
   - Five other readings sit close enough to their boundaries to be fragile. Each is listed in section 6, together with the environment change that would put it at risk.
   - Every confirmatory verdict that could be recomputed under two environments survives (TextCraft round 2, three models). One confirmatory equivalence that could not be recomputed sits within one typical environment shift of its margin: TextCraft round 1, Llama-3.1-8B, two-candidate.
   - One design rule is calibrated below the numerical floor and currently blocks a planned comparison: the eager-against-SDPA attention check that stopped Gemma 4 in wave B and round 3.

## 2. Where numerics enter a score, and the terms used below

A score is computed from one forward pass of the model per prompt, with one unpadded sequence and batch size 1. The pass runs in bf16, a 16-bit floating-point format with 8 significant bits. The model's final logits are bf16 numbers. They are cast to float32 before the log-softmax, which is exact enough. The logits themselves, however, sit on a grid whose spacing grows with their size: 1/16 nat for logits between 8 and 16, 1/8 between 16 and 32, and 1/4 between 32 and 64.

The yes/no score is the log-odds between two answer tokens, and it equals the difference of their two logits exactly:

$$
L_{\text{yn}} = \log p(\text{Yes}) - \log p(\text{No}) = z_{\text{Yes}} - z_{\text{No}} .
$$

The normalising constant of the softmax cancels. So any change to $L_{\text{yn}}$ is a whole number of grid steps of the two logits. The two-candidate score $L$ is the difference between the summed log-probabilities of a success reply and a failure reply, each forced token by token. It is a sum over many positions, so its changes are not confined to a grid.

The order of floating-point additions inside the GPU's matrix and attention kernels depends on the GPU type, the library versions, the attention implementation, the shape of the input and the batch composition. Different orders round differently. That is the only source of the differences studied here. No prompt, model weight or code logic differs within any pair compared below; section 3 says how that was checked.

**Terms.**
- An **environment** is a full specification of the numerics: software stack (the old stack is torch 2.6.0 with transformers 4.51.3; "serve2" is torch 2.14.1 with transformers 5.23.0), GPU type, dtype, attention implementation, code path and batch shape.
- A **per-score difference** is the same prompt's score in environment B minus environment A.
- A **contrast** is what the analyses use: condition A minus condition B for the same item, for example aligned minus swapped.
- A **per-item contrast difference** is that contrast in environment B minus the same contrast in environment A.
- The **shift of an aggregate** is the change in a mean, share or test result when every score comes from environment B instead of A.
- **Game-clustered** means that resampling moves whole games (or goal pairs in TextCraft), because items from one game are not independent.
- The **design effect** is the squared ratio of the clustered standard error to the naive one.

## 3. The paired rescorings that exist

I searched the notes and every run folder on ICE for runs that scored the same prompts more than once. The GPU of each job was read from its job record (`jobs.jsonl`) or, for older runs, from the `nvidia-smi` banner in its Slurm log, as the Qwen forensic audit did. For multi-shard runs, each record was assigned its own shard's GPU. Duplicate submissions that ran for under 15 seconds were excluded. Prompts are identical within each pair, and the evidence for that is strong:
- in every pair, some stratum where nothing else changed reproduces bit for bit;
- the Option C follow-up had already checked its prompt hashes;
- the serve2 rescorings passed token-identity gates.

| What differs | Run A against run B | Model(s) | Scores compared | Environments |
|---|---|---|---|---|
| GPU type, or nothing | `rq2_score` against `size_score_qwen3-8b` | Qwen3-8B | 9,332 two-candidate | A40 or L40S against L40S, old stack |
| GPU type, or nothing | `rq1_score` against `size_score_qwen3-8b` | Qwen3-8B | 13,998 two-candidate | L40S, A40, A100-40GB or A100-80GB against L40S |
| GPU type, or code path | `rq1v_score` against `size_score_qwen3-8b` | Qwen3-8B | 14,582 yes/no and 9,332 two-candidate | A40 or L40S against L40S |
| GPU type, or nothing | `rq2_score` and `rq1_score` against `rq1v_score` | Llama-3.1-8B | 9,332 and 9,332 two-candidate | mixed against mixed |
| GPU type | Stage 1a (`decisionsim_score`) against Option C step 0 | Qwen3-8B, Llama-3.1-8B | 3,867 yes/no each | A100-40GB (Qwen) or A40 (Llama) against L40S |
| Nothing | Option C follow-up rescoring against Stage 1a | both | 3,867 yes/no each | same GPU type, one node for Qwen |
| Software stack | wave A/RQ scores against `family_rescore_<model>` (serve2) | Qwen3-8B, Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Qwen3-32B | 29,164 per model | same GPU type in each pair (L40S, L40S, A40, L40S, A100-80GB) |
| Nothing | `family_envcheck_<model>` against `family_rescore_<model>` | Qwen3-8B, Mistral-7B | 1,800 each | serve2, same GPU type, different job and day |
| Software stack | `textcraft_r2_score` against `textcraft_r3_rescore` | Qwen3-8B, Mistral-7B, Gemma-2-9B | 25,743 per model | L40S both |
| Nothing | `textcraft_r3_envcheck` against `textcraft_r3_rescore` | the same three | 2,602 each | serve2, L40S, different jobs |
| Precision and engine | Qwen pipeline audit: bf16 against float32 weights with eager attention, and against vLLM | Qwen3-8B, Llama-3.1-8B | 200 items, aligned and swapped | A40 and L40S |
| Attention implementation | `attention_agreement` blocks of Gemma 4 jobs | Gemma 4 E4B, 12B, 31B | 5 items each | serve2, L40S or A100 |
| Batch shape and padding | lexsem batched smoke; interpretability batch-row check; TextCraft thinking-arm padding diagnostic | Qwen3-8B, Llama | small; only maxima or means archived | L40S |

The attention-implementation evidence is thin: five items per job, aligned arm only, so no contrast can be computed. The batch-shape evidence survives only as summary numbers. Everything else has full per-score records.

## 4. What the comparisons show

### 4.1 Same environment: bit-identical, with one instructive exception

Every pair in which nothing changed agrees exactly on every score. That includes:
- `rq2_score` shard 3 against the size series (2,120 scores, L40S);
- `rq1_score` against the size series on L40S (6,900);
- the Llama runs on A40 against A40 (1,090) and on L40S against L40S (6,720);
- Option C's follow-up rescoring against Stage 1a on an A100-40GB and on an A40 (3,867 each);
- the serve2 environment checks against the serve2 rescorings, for Qwen3-8B on L40S and Mistral-7B on A40 (1,800 each);
- the three TextCraft serve2 environment checks against the TextCraft serve2 rescorings (2,602 each).

These ran in different jobs, on different nodes and on different days. So the GPU type, not the individual card, is what matters. This is consistent with the Option C follow-up's finding.

The exception is useful. `rq1v_score` and `size_score_qwen3-8b` ran on the same GPU type in the same stack. Their two-candidate scores agree exactly (8,242 of 8,242), but 0.8% of their yes/no scores differ (105 of 12,869), by 1 to 4 grid steps, up to 1.0 nat. The cause is a code-path change between their commits:
- `rq1v_score` (commit `2155b8d`) computed yes/no with `score_pair`, which runs two forward passes, prompt plus " Yes" and prompt plus " No", one token longer than the prompt;
- the size series (commit `8bfd84c`) used the new `score_one_token_pair`, one pass over the prompt alone.

A sequence one token longer changes the shape of the matrix products, and occasionally that changes a logit's last bit. All 105 differing scores have $|L_{\text{yn}}|$ between 22 and 31, where both logits are large. So "same environment" has to include the code path and the input shape, not just the stack and the GPU.

### 4.2 How large the differences are

The table gives per-score and per-item contrast noise for each kind of change, on the largest set available for each. The "contrast" is aligned minus swapped. "Mean absolute" is per score. The contrast column is the standard deviation of the per-item contrast difference.

| Change | Model | Readout | Scores identical | Mean absolute per-score difference (nats) | Largest (nats) | Contrast SD (nats) | Items whose contrast changes sign |
|---|---|---|---|---|---|---|---|
| GPU type (A40 against L40S) | Qwen3-8B | two-candidate | 0.1% | 0.35 | 2.5 | 0.58 | 2.3% |
| GPU type (A100 against L40S) | Qwen3-8B | two-candidate | 0.1% | 0.35–0.36 | 2.8 | 0.57–0.59 | 1.6–2.5% |
| GPU type (A40 against L40S) | Qwen3-8B | yes/no | 29% | 0.27 | 2.25 | 0.45 | 1.8% |
| GPU type (A100 against L40S), step 0 | Qwen3-8B | yes/no | 27% | 0.29 | 2.0 | 0.57 | 2.5% |
| GPU type (five pairings) | Llama-3.1-8B | two-candidate | 0% | 0.12–0.13 | 0.9 | 0.19–0.21 | 2.2–3.0% |
| GPU type (A40 against L40S), step 0 | Llama-3.1-8B | yes/no | 38% | 0.10 | 1.1 | 0.19 | 4.7% |
| Stack, L40S | Qwen3-8B | two-candidate | 0.7% | 0.33 | 3.4 | 0.56 | 2.0% |
| Stack, L40S | Qwen3-8B | yes/no | 30% | 0.26 | 3.75 | 0.45 | 1.3% |
| Stack, L40S | Llama-3.1-8B | two-candidate / yes/no | 0% / 44% | 0.10 / 0.08 | 0.6 / 0.6 | 0.18 / 0.16 | 2.6% / 1.2% |
| Stack, A40 | Mistral-7B | two-candidate / yes/no | 0% / 50% | 0.09 / 0.07 | 0.6 / 0.5 | 0.16 / 0.14 | 2.7% / 1.4% |
| Stack, L40S | Gemma-2-9B | two-candidate / yes/no | 0% / 37% | 0.10 / 0.07 | 1.1 / 1.5 | 0.18 / 0.15 | 2.3% / 2.5% |
| Stack, A100-80GB | Qwen3-32B | two-candidate / yes/no | 0% / 35% | 0.19 / 0.15 | 1.4 / 1.0 | 0.32 / 0.29 | 2.5% / 2.9% |
| Stack, L40S, TextCraft | Qwen3-8B | two-candidate / yes/no | 4.5% / 25% | 0.36 / 0.39 | 5.4 / 4.75 | 0.58 / 0.63 | 4.4% / 5.5% |
| Stack, L40S, TextCraft | Mistral-7B | two-candidate / yes/no | 0% / 27% | 0.09 / 0.09 | 0.8 / 0.6 | 0.16 / 0.16 | 5.8% / 3.3% |
| Stack, L40S, TextCraft | Gemma-2-9B | two-candidate / yes/no | 0% / 30% | 0.10 / 0.12 | 0.8 / 2.75 | 0.18 / 0.26 | 5.2% / 4.9% |
| bf16 against float32 weights and eager attention | Qwen3-8B | two-candidate | — | 0.33 | 1.5 | 0.46 | — |
| bf16 against vLLM 0.8.5 | Qwen3-8B / Llama-3.1-8B | two-candidate | — | 0.37 / 0.12 | 1.7 / 0.6 | 0.57 / 0.20 | — |

Three things stand out.

**A change of stack is about as large as a change of GPU type.** For Qwen3-8B's yes/no scores it is 0.26 against 0.27 to 0.29. The wave B diagnostic's original "seven times" comparison had used a reference that was mostly same-GPU reruns; this has since been corrected in `deviations.md`, and the full-data numbers here confirm the correction.

**The bf16 error itself is the same size as these differences.** On 200 Qwen items, the study's bf16 scores differ from a float32 run by 0.33 nats per score, about as much as two bf16 runs on different GPUs differ from each other. So neither environment is "right", and the spread between environments is a fair estimate of how far bf16 is from exact arithmetic.

**Sign flips are a few percent of items in every case.** Flips concentrate where the item contrast is near zero.

### 4.3 Do the two arms' differences cancel?

If an environment change moved both arms of an item's contrast by the same amount, the contrast would not move at all. If it moved them independently, the contrast's variance would be the sum of the two arms' variances. The ratio below distinguishes the two:

$$
R = \frac{\operatorname{Var}(d_A - d_B)}{\operatorname{Var}(d_A) + \operatorname{Var}(d_B)} ,
$$

where $d_A$ and $d_B$ are the per-score differences of the two arms of the same item. $R = 1$ means independent, $R < 1$ means they partly cancel, and $R > 1$ means they add.

Across the 59 cross-environment contrast comparisons (ALFWorld, Option C step 0 and TextCraft), $R$ lies between 0.74 and 1.0, and the correlation between the arms' differences between about 0 and +0.25. Most values are near 0.9 and 0.1. The precision and engine comparisons of section 4.7 give 0.69 to 1.0. So the arms of a contrast move almost independently, with a small positive correlation that cancels about a tenth of the variance. The project's earlier statement that "the rounding is the same in every condition and averages out" ("Scorer precision", 2026-11-18) holds for the mean over many items. It does not hold item by item: an item's contrast carries roughly $\sqrt{2}$ times the per-score noise.

### 4.4 Dependence on logit magnitude, score magnitude and prompt length

**Logit magnitude is the main driver.** The yes/no differences show it directly, because they are whole grid steps:
- For Qwen3-8B, 100% of nonzero yes/no differences are multiples of 1/8 nat and 92 to 99% are multiples of 1/4. So its Yes and No logits sit between 32 and 64, where one step is 1/4. This matches the median readout logit of 52 measured in the wave B float64 diagnostic.
- For Qwen3-32B about half are multiples of 1/4 (logits straddling 32).
- For Llama-3.1-8B and Mistral-7B, 10 to 23% are multiples of 1/4 and nearly all are multiples of 1/8. That means logits between 16 and 32, with mostly single-step changes.
- For Gemma-2-9B only 35 to 50% are even multiples of 1/8, so its logits are mostly below 16. Gemma 2 soft-caps its final logits at 30.

Per-score noise follows this ordering exactly: Qwen3-8B about 0.26, Qwen3-32B 0.15, Llama 0.08, Mistral and Gemma 0.07. The same ordering holds for the two-candidate readout. That is why the noise is model-specific, and why a margin that is comfortable for Llama can be tight for Qwen.

**Within a model, score magnitude matters for yes/no but hardly for two-candidate.**
- For yes/no, noise is largest where the answer is uncertain. For Qwen3-8B under the stack change, the mean absolute difference is about 0.5 to 0.66 nats for scores with absolute value below 8, against about 0.25 for saturated scores above 16. Option C's step-0 yes/no shows the same pattern: 0.66 nats at absolute scores of 4 to 8, against 0.25 at 24 to 32.
- This matters for decision rules that threshold near zero, because the scores that cross zero are the ones that move most.
- For two-candidate scores, noise is flat across magnitude except for the most extreme bin (above 32 nats), where it rises by about a third.

**Prompt length does not matter.** I used three proxies: characters of history and state text, history steps for ALFWorld, and exact prompt tokens (500 to 650) for Option C step 0. In every pair the rank correlation between absolute difference and length is between −0.04 and +0.11, and binned means are flat.

### 4.5 Dependence on condition: systematic shifts, not just noise

An environment change moves every score of a condition a little on average, by a different amount for each condition. When two conditions of a contrast shift by different averages, the contrast's mean shifts, and averaging over items does not remove it. Some examples, all with game-clustered 95% intervals:
- **Option C step 0, Qwen3-8B, A100 against L40S (yes/no).** The neutral prompt's scores move +0.125 [+0.105, +0.146], while the aware prompt's move +0.008 and the swapped prompt's +0.018. The aware-minus-neutral contrast therefore shifts by −0.117 [−0.149, −0.086], on 1,322 candidates where item-level noise alone would allow about ±0.03.
- **The same comparison for Llama-3.1-8B, A40 against L40S.** Aware-minus-swapped shifts by −0.031 [−0.042, −0.019].
- **ALFWorld two-candidate, Qwen3-8B, A100-40GB against L40S.** Aligned scores move +0.12 and swapped +0.06, and the contrast shifts by +0.064 [+0.012, +0.116].
- **Stack change for Qwen3-32B.** Aligned-minus-absent shifts by +0.030 [+0.019, +0.041] in two-candidate and −0.020 [−0.030, −0.010] in yes/no.

To see how general this is, I pooled every cross-environment comparison of a mean contrast (59 of them) and compared each shift $s_k$ with the standard error $\mathrm{se}_k$ that item-level noise alone would give (game-clustered bootstrap). If the noise were independent and unbiased, $s_k/\mathrm{se}_k$ would be roughly standard normal and about 5% would exceed 2 in absolute value. Instead 24% (non-Qwen models) to 31% (Qwen models) do. A method-of-moments estimate of the systematic spread is

$$
\hat\tau^2 = \frac{1}{K}\sum_{k=1}^{K}\left(s_k^2 - \mathrm{se}_k^2\right) ,
$$

and it comes out as follows.

| Group of comparisons | Number | Root-mean-square shift (nats) | Largest shift (nats) | Systematic part (nats) |
|---|---|---|---|---|
| Qwen3-8B, change of GPU type | 12 | 0.048 | 0.117 | 0.042 |
| Qwen3-8B, change of stack (ALFWorld and TextCraft) | 10 | 0.040 | 0.108 | 0.024 |
| Qwen3-32B, change of stack | 4 | 0.020 | 0.030 | 0.019 |
| Llama-3.1-8B, change of GPU type | 7 | 0.015 | 0.031 | 0.013 |
| Llama, Mistral, Gemma, change of stack | 26 | 0.007 | 0.016 | about 0 |

So a change of environment shifts a Qwen-family mean contrast by a few hundredths of a nat on average, and by about a tenth at worst, beyond what item-level noise predicts. For the smaller-logit models the shift is about a hundredth, and a stack change on the same GPU type is consistent with pure noise. Two cautions: the four Qwen3-32B comparisons share one pair of runs, and the systematic part is estimated from few comparisons, so these numbers carry a factor-of-two uncertainty.

One possible source of condition-correlated bias does not apply here. Condition-correlated batching or padding, which the request raised, would bias contrasts directly. The project's scorers never batch or pad: every score is one unpadded sequence. The shifts above arise anyway, because the conditions' texts differ and the kernels respond to the text.

### 4.6 Tails

On ALFWorld the per-score differences that are not exact repeats are close to Gaussian. Their excess kurtosis is −0.5 to 2.2, and the 99th percentile is about 2.7 standard deviations, against 2.6 for a normal distribution.

TextCraft is heavier-tailed. The excess kurtosis of nonzero differences is 3.8 for Qwen two-candidate, 4.3 for Qwen yes/no and 10.0 for Gemma yes/no (6.6 and 15.5 for the two yes/no readouts when exact repeats are counted). Single yes/no scores move by up to 4.75 nats (Qwen) and 2.75 (Gemma). Item-level contrast differences have excess kurtosis of about 1 to 2 (Option C step 0) and reach 3.2 nats for Qwen on ALFWorld.

For means over hundreds of items the tails make no practical difference. For statistics that depend on single items, or on small subsets of a few dozen items, they do. Examples are a maximum, a per-item threshold, or the 50-item float64 diagnostic, whose −0.11 goal-effect shift is +0.016 on the full 4,666 items.

### 4.7 Precision, engine, attention implementation and batch shape

**Precision and engine.** On the Qwen audit's 200 items:
- bf16 against float32 weights with eager attention moves the aligned-minus-swapped mean by −0.054 (standard error 0.032);
- bf16 against vLLM moves it by −0.013 (0.040) for Qwen and +0.012 (0.014) for Llama.

These are inside the range of section 4.5.

**Attention implementation.** The `attention_agreement` checks of Gemma 4 jobs compare eager and SDPA attention on five items, in the same stack and on the same GPU. The mean absolute yes/no difference was:
- 0.050 in wave B's placement job and 0.275 in wave B's scoring job, both Gemma 4 E4B;
- 0.125 (E4B), 0.216 (12B) and 0.083 (31B) in round 3's smokes.

Every individual yes/no difference is a whole number of 1/8 steps (0, 1, 2 or 5 steps). In float32 the same comparison agrees to 0.00001 nats (round 3's deviation D-G1). So eager against SDPA in bf16 is another environment change of the size measured above.

**Batch shape and padding.** The scorers avoid both, but three archived checks show what they would cost:
- The lexsem smoke batched 8 to 32 sequences. Readouts changed by up to 0.375 nats in plain text, and by up to 1.25 nats for Qwen with the empty think block.
- The interpretability pilot measured 0.132 nats between batch rows.
- TextCraft's thinking arm measured 0.084 nats median between identical copies in a batch and 0.068 to 0.088 under left padding, against bit-identical results when a sequence ran alone.

So batch shape is an environment ingredient like the others, of roughly the same size.

### 4.8 The numerical floor of aggregate effects and decision statistics

Combining the item-level noise with the systematic part gives a simple model. For a mean contrast over $n$ items with item-level contrast noise $\sigma_c$ and design effect $D$, the standard deviation of the estimate across environments is about

$$
\mathrm{SD}_{\text{num}}(n) \approx \sqrt{\tau^2 + \frac{D\,\sigma_c^2}{n}} .
$$

With the measured values, the floor works out as follows. For Qwen3-8B: $\tau \approx 0.04$, $\sigma_c \approx 0.58$, $D \approx 1.3$. For Llama-3.1-8B: $\tau \approx 0.013$, $\sigma_c \approx 0.20$. For Mistral and Gemma under a stack change: $\tau \approx 0$, $\sigma_c \approx 0.15$. Design effects observed range from 0.5 to 2.2.

| Items in the mean | Qwen3-8B, one SD (nats) | Llama-3.1-8B, one SD (nats) | Mistral or Gemma, stack change, one SD (nats) |
|---|---|---|---|
| 100 | 0.077 | 0.026 | 0.017 |
| 300 | 0.055 | 0.018 | 0.010 |
| 1,000 | 0.045 | 0.015 | 0.005 |
| 5,000 | 0.041 | 0.013 | 0.002 |

For Qwen-sized logits the floor stops falling at about 0.04 nats, because the systematic part does not average out. Twice the floor, about 0.1 nats, is the smallest Qwen effect or margin distance that is safe against a change of environment. For Llama-sized logits the corresponding figure is about 0.03 to 0.05 nats.

The same comparisons give the floor of the project's decision statistics directly.
- **Item shares** (the share of items whose contrast is positive) moved by at most 0.008 in every comparison. About 2 to 5% of items flip sign, but flips in both directions largely offset each other.
- **Standardized effects** (mean contrast divided by the item SD) moved by at most 0.007 on ALFWorld and by at most 0.034 for the TextCraft round-2 contrasts.
- **Acceptance of a yes/no verifier** (rounded score above zero) flipped for 0.16% of Qwen's step-0 scores and 0.9% of Llama's. Llama's scores sit closer to zero.
- **Bootstrap and permutation p-values** of the TextCraft round-2 contrasts changed by up to a factor of about 2 when only the scores changed, with resamples held fixed. Qwen's timing Holm p went from 0.052 to 0.028.

## 5. The small effects and equivalence tests, one by one

Each entry says what is tested, how close the result sits to its decision boundary, and whether the measured numerical variation could change it. Two kinds of variation are distinguished. A rerun in the same environment changes nothing. A different environment means another GPU type, stack, precision, engine, attention implementation or batch shape.

### 5.1 TextCraft round 2's confirmatory family, re-run on both stacks

This was the one place where a full test family could be recomputed under two environments. The three round-2 models that round 3 rescored (Qwen3-8B, Mistral-7B, Gemma-2-9B) have every round-2 record scored in the old stack and in serve2, both on L40S. I reimplemented the five confirmatory contrasts exactly as `wf/textcraft/r2_analysis.py` defines them at the frozen tag:
- the populations come from the tag's `r2_items.jsonl`;
- the margin is the smaller of 0.3 item SDs and 0.4 nats;
- there are 20,000 goal-pair resamples, a sign-flip or bootstrap p for "differs", and TOST for "equivalent";
- Holm runs over all 50 tests, with the two models that were not rescored taken from the frozen metrics.

My point estimates on the old scores equal the frozen ones to every printed digit. The same resamples were used for both stacks, so any difference between the two columns comes from the scores alone.

| Model and contrast | Frozen verdict | Estimate, old stack, then serve2 (nats) | Numerical shift and its noise SE (nats) | Verdict in serve2 |
|---|---|---|---|---|
| Qwen3-8B lift | confirmed above zero | 0.670, 0.663 | −0.008 (0.019) | same |
| Qwen3-8B parent penalty | confirmed below zero | −1.019, −0.994 | +0.025 (0.022) | same |
| Qwen3-8B naming | inconclusive | 0.097, 0.108 | +0.011 (0.040) | same |
| Qwen3-8B completion | inconclusive | 0.645, 0.606 | −0.039 (0.046) | same |
| Qwen3-8B timing | not applicable (gate) | 0.802, 0.910 | +0.108 (0.061) | same; the differs test alone would cross Holm 0.05 (0.052 to 0.028) |
| Mistral-7B, all five | lift, penalty and naming above zero; completion equivalent; timing not applicable | shifts at most 0.011 | noise SE 0.005 to 0.017 | same |
| Gemma-2-9B, all five | lift, penalty and naming below zero; completion and timing equivalent | shifts at most 0.016 | noise SE 0.005 to 0.023 | same; timing's equivalence Holm p 0.027 to 0.032 |

The equivalence margins are unchanged (0.40 nats for all three; the SDs move by less than 0.3%). Timing's applicability gate also holds in both stacks:
- Qwen fails it either way, because its role gap at one look-up is +0.71 and +0.64, not below zero;
- Mistral fails it either way;
- Gemma passes it either way (−0.91 and −0.90).

**So no confirmatory round-2 verdict depends on the stack for these three models.** The equivalence verdicts that matter most here are Gemma's completion and timing and Mistral's completion. They sit 0.12 nats or more inside their 0.40-nat margins, about 5 to 25 times the numerical SD of their estimates. The two Llama models were not rescored, so their 10 verdicts could not be recomputed. Their equivalence verdicts (Llama-3.1-8B lift +0.095, Llama-3.2-3B completion −0.12, Llama-3.1-8B timing +0.03) have 95% intervals ending at least 0.13 nats inside the margins, and Llama-sized logits have a floor of about 0.02, so these are not at risk. Their other verdicts were not checked here.

**The descriptive R3 reading does depend on the stack.** R3 fires for Qwen3-8B because, among other conditions, the unmatched role gap's 95% interval lies above zero while the parent penalty is confirmed below zero.
- The frozen interval is [+0.018, +1.48]; with my resamples on the old scores it is [+0.010, +1.48].
- On the serve2 scores, with the same resamples, it is [−0.019, +1.45].
- The role gap moves by −0.028 nats, well within its noise SE of 0.038.

The interval's lower end crosses zero, so in serve2 R3 would not fire. The results note already calls Qwen's R3 "marginal and uncorrected"; this shows that a change of stack alone, with nothing else changed, is enough to undo it. The interpretation built on it ("Qwen3-8B does not fit the additive account") should be read as unresolved. The matched role gap, whose lower end was +0.014, was not recomputed here because its cell matching was not reimplemented. Its lower end is as close to zero, so it is probably just as fragile.

### 5.2 TextCraft round 3: the B3 equivalence test and the A3 margins

B3 compares an older model with its newer counterpart on standardized contrasts: z is a mean divided by an item SD, and the test asks whether Δz, the difference in z, is within a fixed margin of 0.20 standardized units, by TOST. The environment check failed, so the designed fallback applies, and B3 reads the older models' serve2 rescorings, not their round-2 scores. All scoring involved ran in serve2 on L40S, at batch size 1.

Three numbers matter here.
- How far an environment change alone moves an older model's z: at most 0.034 (Qwen timing), and at most 0.011 for every other contrast of the three models. Against a 0.20 margin, numerics can use up at most about a sixth of it.
- A B3 comparison involves two models, so numerical shifts of both enter Δz. They are independent across models, so the combined spread is still at most about 0.05.
- Round 3's design already finds that B3 equivalence has almost no power for Dn, Dc and T, so the realistic outcomes are "differs" (by 0.3 SD or more) or "not distinguishable". A numerical shift of a few hundredths cannot turn either into the other except for an estimate within about 0.05 of a boundary.

**B3 is not at numerical risk as designed.** It would have been at moderate risk had it read round-2 scores, because Qwen3-8B's timing z moves 0.034 between stacks. The fallback rule was the right protection.

Round 3's A3 and A3x equivalence tests use round 2's margin, the smaller of 0.3 item SDs and 0.4 nats, in nats and within one model's run. Within one run there is no numerical variation. Under an environment change, the floors of section 4.8 apply: about 0.05 to 0.1 nats for Qwen-like logits and 0.02 to 0.04 for the others. They would matter only for a 90% interval ending within about 0.1 nats of the margin.

The Gemma 4 slot in B3 is "not scored" because both Gemma 4 sizes failed the eager-against-SDPA attention check, at 0.216 and 0.125 nats against a limit of 0.10. Section 5.6 discusses that rule.

### 5.3 Families wave B: the environment check, the rescoring rule and the primary tests

**The environment check was settled by numerical chance.** Its per-score limit for yes/no was 0.07 nats. For Qwen3-8B, a change of GPU type moves yes/no scores by 0.27 to 0.29 and a change of stack by 0.25 to 0.26. A same-environment rerun moves them by exactly zero. So for Qwen the limit could only be met by bit identity, and a stack change could not pass it whatever its effect on contrasts. Mistral passed at 0.0688 against 0.07. Its full-data stack change on A40 is 0.070 per score, so whether it passed or failed was a coin toss.

The check's contrast criteria were sensible but measured on too few items to be informative. The goal effect had to stay within 0.10 nats and the share within 0.02, on 300 items, or on 50 in the float64 diagnostic. On all 4,666 items the stack change moves the yes/no goal effect by +0.016 for Qwen3-8B, the two-candidate by −0.007, and the share by −0.002. The 50-item float64 diagnostic's −0.11 was a small-sample fluctuation of a mean whose item-level noise SD is about 0.45.

**The adopted rule is the right one.** Every older reference is rescored in serve2, and every comparison reads the rescorings. Its "environment-dependent reference" flag, which fires when the share moves by more than 0.02 or the yes/no goal effect by more than 0.10 nats, can now be evaluated on the full rescorings.

| Reference model | Share of items with aligned above swapped (yes/no), original then serve2 | Change in yes/no goal effect (nats) |
|---|---|---|
| Qwen3-8B | 0.887 then 0.885 | +0.016 |
| Llama-3.1-8B (L40S part of the original) | 0.899 then 0.901 | +0.001 |
| Mistral-7B | 0.879 then 0.880 | 0.000 |
| Gemma-2-9B | 0.748 then 0.744 | −0.003 |
| Qwen3-32B | 0.715 then 0.707 | −0.009 |

None comes near the flag, so for the five references the original-environment and serve2 readings will agree.

The wave B primary tests compare item shares with a TOST at ±0.05. Measured shares move by at most 0.008, so these tests are not at numerical risk even though wave B models may run on L40S, A40 or A100. The 0.3-nat "noise floor" for differences in yes/no goal effect between models is conservative by a factor of three or more for means over the full item set. It would be about right for subsets of 50 to 100 items of a Qwen-sized model.

### 5.4 The τ-bench authorization experiment

Each removal is a difference between arms scored in the same job on the same GPU, so a rerun would reproduce it. The question is how far another environment would move it.

For Qwen3-8B the margin is 0.72 nats and the removals are −0.09 [−0.42, +0.26] and −0.08 [−0.33, +0.18]. Their intervals end at least 0.46 nats inside the margin. Qwen's numerical floor for a mean over 192 items is about 0.06 to 0.1, so the "relevance-like" categories are robust to any environment change.

For Llama-3.1-8B the margin is 0.17 nats. Llama fails both gates, so its categories are "inconclusive (gate)" and nothing rests on the ungated labels. Two ungated readings are nonetheless fragile.
- The request component's removed part, −0.20, is only 0.03 beyond the −0.17 needed for "grows". Llama's largest measured shift of a mean contrast across GPU types is 0.031.
- The naming component's arm (i) interval reaches −0.21, 0.04 beyond the margin, which is what keeps it from "absent".

Both would be at risk under a change of GPU type, and not under a same-environment rerun. τ-auth's shards ran on A40 (shards 0 to 2) and L40S (shard 3), so the estimates are already averages over two environments. That does not bias anything, because each contrast stays within one job.

### 5.5 The lexsem diagnostic

All of lexsem's decisions compare arms within one run: L40S, serve2, pinned revisions, bf16, batch size 1. Its gate refuses mixed environments. A rerun would reproduce every number exactly.

Its provisional "numerical floor" of 0.13 nats came from the interpretability pilot's batch-row noise. The design asks that it be replaced by a measured paired-effect sensitivity before freezing, and this note supplies one.
- **Qwen3-8B with the empty think block, yes/no, at step 0.** This is the setting of lexsem's step-0 bridge, and it is also where the largest systematic shift was measured: −0.117 nats for aware-minus-neutral between two GPU types.
- **Qwen means over a few hundred items in general.** The one-SD floor is about 0.05 to 0.08.
- **Llama.** About 0.02 to 0.03.

**So 0.13 is about right as a floor for Qwen with the block.** It roughly equals the largest observed shift and is about twice the typical one. It is conservative for Qwen in plain text, which is less extreme than the block case in all evidence available, and for Llama a floor of about 0.05 would be justified.

The zones are read only when a quarter of the reference effect exceeds the floor, which requires a reference above 0.52 nats. The motivating reference effects are 2.2 to 2.3 nats on later steps and a plain-text G of about 1.3. These clear the requirement by a wide margin, so the zone readings are not at numerical risk. The step-0 reference values of about 0.23 to 0.37, which the code does not use, would not clear it.

The one lexsem comparison that crosses environments is the descriptive "new minus archived" stage-0 comparison, from the old stack with the block to the serve2 rerun. Environment alone can move it by up to about ±0.12 nats in the mean and by about 0.5 nats per item. It should be read with that allowance, never as evidence of a change in the model's behaviour.

### 5.6 Rules that compare one environment with another inside a job

The wave B and round 3 **attention-agreement rule** requires the eager and SDPA implementations to agree within 0.10 nats on average on five yes/no scores. One bf16 step at logits between 16 and 32 is 0.125 nats. So the rule passes only when at most three of the five scores move by a single step, and it fails on the kind of difference measured in section 4.7.
- The float32 comparison shows that the two implementations agree to 0.00001 nats for Gemma 4 E4B. The failures are bf16 numerics, not a wrong implementation.
- The same model passed in wave B's placement job (0.050) and failed in its scoring job (0.275), on different items.

This rule currently stops wave B's Gemma 4 E4B scoring job ("stopped before scoring", status FAIL), and with it the planned Gemma 4 against Gemma 2 comparison, one of wave B's sixteen primary tests, which cannot be tested without those scores. It also excluded Gemma 4 from round 3's B3. Scoring uses one implementation consistently within a model, so the difference between implementations never enters a contrast. The check can therefore only catch a broken implementation, and for that a float32 comparison, or a bf16 limit set well above the bf16 floor, would serve.

The **thinking-arm start check** (padded against alone) had the same problem and was already moved to float32. The interpretability pilot's 0.15-nat **reproduction gate** passed easily (2.782 against 2.783); means over thousands of items move by a few hundredths.

### 5.7 Other reported effects or margins under about 0.3 nats

| Reading | Distance to the boundary that would change it | Environment that could change it | At risk? |
|---|---|---|---|
| rq1x: Llama-3.1-8B goal effect on 292 determined successes, 0.16 [0.01, 0.31], p 0.04 | lower end 0.01 above zero | a different GPU type or stack (Llama mean shifts up to 0.03; noise SD for 292 items about 0.01) | yes; the "significant" reading is fragile, the size (about 0.1 to 0.2) is not |
| τ families: Llama-3.2-3B failing minus witness −0.08 [−0.12, −0.03], which gives the "(working calls move more)" suffix | upper end 0.03 below zero | a different GPU type or stack | marginal: within the largest Llama-family shifts, beyond typical ones |
| TextCraft round 1: Llama-3.1-8B standardized equivalence, −0.14 against ±0.14 ("not inside") | 90% interval [−0.207, −0.080]: the lower end is 0.067 beyond the margin | none plausible (standardized shifts measured up to 0.034) | no |
| TextCraft round 1: Llama-3.1-8B two-candidate and yes/no equivalence (confirmatory; inside ±0.3 and ±0.37) | 90% intervals end 0.016 and 0.17 inside | the two-candidate one under a typical Llama environment shift (0.013 systematic, up to 0.031) | two-candidate yes, marginal (the interval ends at −0.284 against −0.3); yes/no no |
| Families wave A: Llama-3.1-8B against Qwen3-8B share, 0.008 with 90% interval [−0.013, 0.030], "matches" (p 0.0015) | 0.02 inside ±0.05 | none (shares move by at most 0.008) | no |
| Families wave A: Mistral against references, "inconclusive" (equivalence Holm p 0.126 and 0.156) | far from 0.05 | none | no |
| τ families: Mistral +0.23 (not interpretable); placement difference +0.30 against a 0.046 rule | far | none | no |
| TextCraft round 2: Mistral lift +0.27 (also inside the margin, TOST p 0.026); Llama-3.2-3B naming +0.15 inconclusive | far from changing verdicts | none | no |
| Interpretability reproduction gate (0.15 nats on means) | 0.001 measured | none | no |
| Decision-simulation acceptance of near-zero scores (Llama-3.2-3B "scores within ±0.6 nats of zero, flips are tiny threshold crossings") | per-score | a different GPU type (0.9% of Llama step-0 acceptances flip) | individual decisions yes; rates no (both arms in one run) |

## 6. Conclusions at risk

Listed in order of how much rests on them. "Cross-environment" means the reading would probably or possibly change if the same items were rescored with a different GPU type, stack, precision, engine, attention implementation or batch shape. "Same environment" means a rerun with everything identical; none of these readings would change under that.

1. **TextCraft round 2, R3 fires for Qwen3-8B** (descriptive, uncorrected). It does not fire when the same records are rescored in serve2 on the same GPU type: the role gap's lower end moves from +0.010/+0.018 to −0.019. The reading that Qwen3-8B "does not fit the additive account" rests on it. **At risk under a change of stack, and by extension under a change of GPU type.** This is the one measured flip.
2. **rq1x, Llama-3.1-8B's positive goal effect on the 292 determined successes** (0.16 [0.01, 0.31], p 0.04). The lower end sits within one typical Llama-sized environment shift of zero. At risk cross-environment. The effect's size and the main text's point (much smaller than on failures) are not.
3. **τ families, Llama-3.2-3B's "(working calls move more)" suffix**, which rests on failing minus witness −0.08 with an upper end of −0.03. Marginal, cross-environment.
4. **TextCraft round 1, Llama-3.1-8B's two-candidate equivalence** (confirmatory C4: the 90% interval [−0.284, −0.109] lies inside ±0.3). Its lower end is 0.016 nats inside the margin, about one typical Llama-sized environment shift. A rescore on another GPU type or stack could plausibly push it outside, changing that row from "equivalent" to "not shown". Cross-environment only. The yes/no equivalence (0.17 inside) and the negative-effect reading are not at risk.
5. **τ-auth, Llama-3.1-8B's ungated "grows" label and the near-"absent" naming component.** Each is 0.03 to 0.04 nats from its boundary. Cross-environment; these are reported "for completeness" and decide nothing.
6. **The Gemma 4 exclusions** from round 3's B3 and from wave B's scoring. These are not findings but design outcomes, and they rest on a bf16 check whose limit is below one grid step. Wave B's Gemma 4 against Gemma 2 comparison is currently blocked by it.
7. **Families wave B's environment-check verdicts** (Qwen fail, Mistral pass at 0.0688 against 0.07). Already superseded by rescoring every reference, which was the right response. Recorded here only because a check of this form cannot distinguish a harmful change from bit-level noise.

Not at risk:
- every confirmatory verdict of TextCraft round 2 that could be recomputed (15 of 25 model-by-contrast verdicts; the 10 for the Llama models could not be recomputed, and their equivalence verdicts end at least 0.13 nats inside their margins);
- every share-based comparison in families waves A and B;
- τ-auth's Qwen categories;
- round 3's B3 as now configured;
- every large effect in the project, which are 1 to 7 nats against a floor of at most about 0.1.

## 7. Recommended rules

**Rule 1. Define the environment completely and keep every paired comparison inside one.**
- The environment is the stack version, GPU type (not card), dtype, attention implementation, code path, and batch shape and padding. Section 4.1 showed that a one-token change in input shape is enough to move 0.8% of yes/no scores.
- Both arms of any contrast, and every arm of a test family, must come from one environment.
- A resumed shard must be pinned to the original GPU type. The 2026-11-20 deviation notes that a resume could cross GPU types, and Option C's yes/no shards 0 and 2 did resume on A40 after starting on L40S.
- Comparisons of the same model across environments (archived against new, round 2 against round 3, original against rescored) are descriptive only, read with the allowance of Rule 3.
- Comparisons between different models may cross environments when they are made on shares or standardized scales. Shares move by at most 0.008, and standardized effects by at most 0.034. Comparisons in nats between models should share a stack, as wave B and round 3 now do.

**Rule 2. Do not use bit-level agreement criteria in bf16.**
- Any check that compares two environments in bf16 should either run in float32, which separates a wrong implementation from rounding (as the thinking-arm check now does), or use limits above the measured floor of that model family.
- For the attention-agreement rule specifically, a float32 check with a tolerance of about 0.01 nats separates a wrong implementation from rounding, as round 3's D-G1 diagnostic effectively did. Wave B and round 3 keep their frozen rules; applying this to them would be a deviation from a frozen design and is not made here.

**Rule 3. Set margins and minimum effects relative to the measured floor.**
- A confirmatory effect, equivalence margin or decision boundary should be at least five times the numerical SD of its own statistic, so that numerics use at most a fifth of it. The SD follows the formula in section 4.8, with the model family's values: for Qwen-sized logits (yes/no logits above 32) about 0.05 to 0.08 nats for means of 100 to 1,000 items; for Llama, Mistral and Gemma about 0.01 to 0.03.
- In practice:
  - nats-scale margins should be at least about 0.3 to 0.4 for Qwen-family models, which round 2 and round 3 satisfy, and at least about 0.1 for the others;
  - standardized margins should be at least about 0.15, which B3's 0.20 satisfies;
  - share margins of 0.05 are ample.
- Separately, any reading whose interval ends within about twice the numerical SD of its boundary should be labelled "numerically fragile" in the results note, whatever its p-value. That is about 0.1 nats for Qwen means and 0.03 to 0.05 for the others. This applies to the five readings in items 1 to 5 of section 6.

**Rule 4. For small effects and equivalence tests, include a cross-environment rescore of a subset, not a same-environment rerun.**
- A same-environment rerun reproduces every bit, so it checks provenance and determinism, which is worth a few minutes as a smoke test, but it says nothing about sensitivity.
- What measures sensitivity is rescoring a subset under a different GPU type in the same stack: at least 300 items, every arm, the readouts the tests use. Then report the shift of each decision statistic, recomputed on that subset with the same resamples, beside the main result. At the project's speeds this costs a fraction of a GPU-hour per model.
- A reading whose decision changes on that subset, or whose estimate moves by more than a fifth of its margin, should be reported as environment-dependent.

**Rule 5. Prefer readouts with smaller logits for near-zero decisions, and threshold with care.**
- Yes/no scores are noisiest exactly where they are uncertain, at absolute values below about 8, and Qwen's large logits put them on a coarse grid.
- Rules that threshold individual scores near zero (verifier acceptance, sign flips) should be reported as rates within one environment, never as counts of individual changed decisions across runs.

**Rule 6. Use the full item set, not small subsets, for environment diagnostics.** On 50 items the mean shift of a Qwen contrast has a noise SD of about 0.07, so a −0.11 shift there is not evidence of bias. On thousands of items the same comparison resolves shifts of 0.01.

## 8. What this does and does not show

**What it shows.** These are measurements on the project's own prompts, models and readouts. They cover four kinds of environment change across five models, at the scale of the actual analyses (tens of thousands of scores per comparison). Same-environment determinism was shown on about 30,000 scores.

**What it does not show.**
- It does not measure H100 or H200 GPUs. Wave B allows none of them for the old references, but some designs allow H100.
- It does not measure Llama-3.2-3B, Llama-3.2-1B, Ministral, OLMo or the Qwen3.5/3.8 models. Their logit sizes are not known here, and since noise scales with logit size, each new model family needs its own floor (Rule 4).
- The attention-implementation and batch-shape evidence is small, and the batch-shape evidence survives only as summary numbers.
- The systematic part is estimated from 4 to 26 comparisons per group and could be off by a factor of two.
- The R3 recomputation covers the unmatched role gap only.
- The τ-bench data were not re-analysed. The τ readings in section 5 are judged by the distance from their reported intervals to their boundaries, using the ALFWorld/TextCraft floors for the same model families.

**What would change these conclusions.** A cross-GPU rescore of the TextCraft records would show whether R3 also flips under a change of GPU type. I expect it would sometimes, given a noise SE of 0.038 against a lower end of 0.01 to 0.02. A Llama rq1x rescore on another GPU type would settle item 2 of section 6. If a new model family turned out to have logits above 64 (grid 0.5), its floor would be about twice Qwen3-8B's, and the margins in Rule 3 would need doubling.

## 9. Files and how to rerun

Code is in `scripts/numsens/`:
- `fetch.sh` copies the inputs read-only from ICE and records the GPU of every job from its log.
- `alfworld_pairs.py` handles the ALFWorld pairs: GPU type, stack and reruns.
- `optionc_step0.py` handles Stage 1a against Option C step 0, and the follow-up rescoring.
- `textcraft.py` handles the two TextCraft stacks and re-runs the round-2 family on both with common resamples.
- `precision_attention.py` handles bf16 against float32, vLLM, the attention checks and the batched smoke.
- `floor.py` pools the shifts and estimates the systematic part.
- `summarize.py` prints the tables.

Outputs are in `scripts/numsens/out/` (`alfworld_pairs.json`, `optionc_step0.json`, `textcraft.json`, `precision_attention.json`, `floor.json`).

To rerun, set `NUMSENS_DATA` to an empty folder, run `bash fetch.sh`, then run the five Python scripts in the order above with the project's `.venv`. Each takes under two minutes on a laptop CPU. The TextCraft script needs the tag `textcraft-r2-design-frozen-2026-11-23` for the round-2 records, and reads `notes/textcraft_r2_metrics.json` for the two models that were not rescored.
