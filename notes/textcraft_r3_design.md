# TextCraft round 3: design (written 2026-11-23, before any round-3 code and before any round-3 score)

This note fixes, before any round-3 code exists and before any model scores a round-3 record, which models round 3 scores, which records and prompts, which tests are confirmatory, how every result will be read, and what counts as replication, failure to replicate and inconclusive evidence. It is derived from the proposal ([textcraft_r3_proposal.md](textcraft_r3_proposal.md)) and its "Decisions (2026-11-22)" section; where the two differ, this note governs.

**Status and provenance.**

- Round 2 is frozen at tag `textcraft-r2-design-frozen-2026-11-23` (4ccf84b). Model families wave B, whose serve2 scorer fixes round 3 needs, is frozen at `families-waveb-design-frozen-2026-11-22` (f6bbffb). Both tags were merged into branch `textcraft-r3` before this note, with both studies' frozen code paths unchanged (section 12).
- **Written without round-2 results.** No round-2 score, analysis output or result note was read while writing this note or building round 3. The round-3 hypotheses are carried forward from the round-2 *design*. Round 3 uses round 2's outcomes in exactly two places, both fixed here: as the target of the replication rule (section 7.3), and as the older half of the generation pairs (section 7.4). Any hypothesis added later because of what round 2 finds must be labelled "motivated by round-2 outcomes", reported as such, and never presented as independently prespecified.
- **Amended after review 1:** section 14 states the rules as amended and governs where an earlier section differs.
- The design is reviewed independently and then frozen at a git tag (`textcraft-r3-design-frozen-<date>`) before any round-3 GPU job, the smoke runs included. Any change after the tag is a reported deviation.
- Round 3 changes no round-2 or wave-B file's behaviour. It has its own run names (`textcraft_r3_*`), configs, results folders, golden file and code (`wf/textcraft/r3*.py`), and it imports round-2 and wave-B code without editing it.

## 1. Question

Round 2's five models were released between May 2024 and April 2025. Round 3 asks whether round 2's measured quantities behave the same way in recent open-weight instruction models when everything else is held fixed: the same 2,982 records, the same goal interventions (arms), the same clean world-model prompt and anchored yes/no question, the same scoring and the same estimands.

Three analyses answer three different questions and are never pooled:

- **(a) Direct replication.** Each new model goes through round 2's five confirmatory contrasts exactly as round 2's models do. The verdicts are read against the account's own prediction from the round-2 design (H3-P) and against the pattern round 2 actually finds (the replication rule).
- **(b) Older against newer releases.** Three pairs of a round-2 model and a contemporary model of the same developer line, compared contrast by contrast on the same items, on a scale-free primary measure.
- **(c) Exploratory.** Size, recency, thinking mode, format, capability and architecture. Estimates with intervals; nothing confirmed or rejected.

**What round 3 cannot show.** The models are a small purposive sample, one per developer line. No result licenses a claim about "model generations", "2026 models" or any model not scored. A difference between two releases changes data, size, tokenizer, template, post-training, architecture and software at once; it is a difference between two released artifacts.

## 2. Models

All are scored in bf16 with log-softmax in float32, in serve2 (transformers 5.23.0), from node-local weights checked against a committed hash manifest (route A, section 9). Revisions are the full commits in `wf.constants.REVISIONS`.

| Tag | Hugging Face id | Revision | Role | GPU type (pinned) |
|---|---|---|---|---|
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | core; pairs with Qwen3-8B | NVIDIA L40S |
| `ministral3-8b-it` | mistralai/Ministral-3-8B-Instruct-2512-BF16 | f6fae9795746f63c9be8344932f01275f3c63734 | core; pairs with Mistral-7B-v0.3 | NVIDIA L40S |
| `gemma4-12b-it` | google/gemma-4-12B-it | 43c9d014b231474fa7212dbb48a762566c2b255d | core; pairs with Gemma-2-9B-it | NVIDIA L40S |
| `gemma4-e4b-it` | google/gemma-4-E4B-it | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | declared fallback for `gemma4-12b-it` only | NVIDIA L40S |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 | extension | A100 80 GB |
| `gemma4-31b-it` | google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | extension | A100 80 GB |
| `olmo3-7b-it` | allenai/Olmo-3-7B-Instruct | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | extension | NVIDIA L40S |
| `qwen3.5-9b-think` | Qwen/Qwen3.5-9B (as `qwen3.5-9b`) | as `qwen3.5-9b` | exploratory thinking arm | NVIDIA L40S |

The handling of each template is wave B's, unchanged, for the five models wave B pinned (`wf.families.WAVE_B`): thinking off (`enable_thinking=False`) with each model's declared thinking-off ending checked on every prompt; Ministral 3's tokenizer loaded with `mistral_format=False`, its prompts checked against mistral-common, and its default system prompt forbidden; Gemma 4 31B's empty thought channel declared as a generation suffix; Qwen3.8 as shipped (empty think blocks in the example replies); Gemma 4's eager-against-SDPA agreement check at job start. The two new specs:

- **`gemma4-12b-it`.** Its chat template is byte-for-byte Gemma 4 31B's, so it takes the 31B handling: the generation suffix `<|channel>thought\n<channel|>`, the thinking-off ending `<|turn>model\n<|channel>thought\n<channel|>`, the marker `<|think|>`, and the attention-agreement check. Its class is `Gemma4UnifiedForConditionalGeneration`, loaded through `AutoModelForImageTextToText` with the strict loading check and **no** unexpected checkpoint tensor allowed: every language-model tensor matches the class (proposal, section 2.4), and the ten vision-embedder tensors stored under older names must be renamed by transformers on load, or the check fails on the missing parameters. GPU memory needed: 24 GB or more (23.9 GB of weights).
- **`olmo3-7b-it`.** ChatML with a native system role (its default function-calling system prompt appears only when no system message is given, and is forbidden in every prompt); no thinking mode; plain causal-LM class (`Olmo3ForCausalLM`), strict loading, no unexpected tensor allowed (the checkpoint's 355 tensors match the class exactly). Its thinking-off ending is the bare generation prompt `<|im_start|>assistant\n`.

**The Gemma fallback rule (mechanical).** `gemma4-12b-it` is replaced by `gemma4-e4b-it` in every core role (family A3, pair B3, replication) if, at its smoke run and before any round-3 contrast is computed, any of these holds: the strict loading check fails; the tokenization gate fails; the eager-against-SDPA agreement check fails; or the median yes/no mass P(Yes) + P(No) over the smoke run's yes/no prompts is below 0.5. The replacement is recorded as a deviation, and the size caveat (E4B has about 4 B effective parameters against Gemma 2's 9 B) is printed beside every Gemma reading. No other model has a fallback: a core or extension model that fails its gate is not scored, and its tests enter their family with p = 1.

**Not used** (decisions D3 and D4): Ministral 3 Reasoning, OLMo 3 Think, Granite 4.2, Ministral 3 14B, Gemma 3 12B, gpt-oss-20b.

## 3. Records, prompts and arms

Exactly round 2's: `wf/textcraft/data/r2_items.jsonl` (sha256 `e6da991f…7e47c22f0`, verified by `r2_manifest.json`), 2,982 records and 25,743 (record, arm) combinations per model; the clean system text (`prompts.CLEAN_SYSTEM`), the three verified examples (`fewshot.json`), the anchored yes/no question (`r2.YN_QUESTION`); prompts built by `r2.messages`. Every round-3 model takes the system text natively, so no prompt is folded (except in the Gemma placement check, section 8.3). Round 2's item hash (`r2.item_hash`) is used unchanged, so every round-3 score pairs item by item with every round-2 score of the same record. Yes/no arms are scored in one forward pass (`score_one_token_pair`), two-candidate arms with `score_pair`, as in round 2.

## 4. Gates before any score

On every record and arm of the shard, before the first score and again on every resume, through the scorer's own encoding (`LogoddsScorer._encode`, which unwraps transformers 5's `BatchEncoding` with `template_input_ids`):

1. round 2's tokenization gate: each candidate tokenized alone equals the template's reply tokens after that arm's prompt (with the model's declared generation suffix stripped); "Yes" and "No" are single tokens equal to the template's reply after every yes/no prompt; the two candidates' first tokens differ;
2. wave B's prompt checks: every prompt ends with the declared thinking-off ending, the thinking switch is read where the template has one, no forbidden text appears, and for Ministral 3 every prompt's ids equal mistral-common's encoding;
3. arm isolation: for every pair of arms of a record that differ only in the goal line, the token ids differ only inside that line;
4. golden renderings: five fixed record-arm cases per model (round 2's golden cases), rendered text, id count and id hash, compared with `wf/textcraft/data/r3_chat_golden.json`.

The same gate runs on CPU on ICE for every model, every record and arm, before the freeze (section 13). At job start also: the environment guard (transformers 5 or later, no `kernels` package), the resource check (GPU memory and, for the 27B and 31B models, job memory), the weights check against the manifest, the GPU-type check (section 10) and, for Gemma 4, the attention-agreement check (wave B's rule: five records, aligned two-candidate and yes/no arms; mean absolute difference at most 0.10 nats yes/no and 0.50 two-candidate, no single score more than 1.0 nats apart). Any failure stops the job before it writes a score.

## 5. Smoke runs

20 seeded records (`random.Random(f"{RNG_SEED}|textcraft|r3|smoke")`), every arm, one shard, the same gates. A smoke run is read for four things only: the loading record, the attention-agreement record, timing, and the median yes/no mass (for the Gemma fallback rule). No contrast, AUC, competence value or effect is computed on it, and the analysis never reads it.

## 6. The estimands (unchanged from round 2)

All contrasts are on the two-candidate log-odds L (the log-probability of the success reply minus that of the failure reply, each teacher-forced after the prompt), on failures, with goal pairs as clusters, computed by round 2's frozen code (`r2_analysis.contrast_inputs` and `contrasts`). With ΔL an item's aligned-minus-swapped difference:

The intermediate lift, over the 990 ingredient main failures,

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}}.$$

The within-item penalty, over the 675 parent failures,

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})}.$$

Naming / partial completion and completion with naming held, over the 500 goal-role and 496 class-matched ingredient failures with a two-item line,

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}},$$

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

The sensitivity to added empty-inventory steps, over the padded failures,

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}}.$$

Statistics, verdicts, readings, interpretations, the rejection rules R1, R3 (a descriptive diagnostic) and R5, the central-question mapping, the matched role gap, the heuristic tracking and every descriptive readout are round 2's frozen functions, called on round-3 scores: estimate, 95% percentile goal-pair bootstrap (20,000 resamples for family tests, 2,000 descriptive), two-sided sign-flip p for Λ and P and bootstrap p for Dn, Dc and T, and two one-sided equivalence tests against the margin. Verdicts: confirmed above or below zero (Holm-adjusted differs-p below 0.05 and the 95% interval on that side), equivalent to zero (Holm-adjusted equivalence p below 0.05), inconclusive (flagged "equivalence underpowered" when power at a true zero at Holm's first step is below 0.8), not applicable (T only, when the one-look-up role gap has no interval below zero).

## 7. Analyses and hypotheses

### 7.1 Competence: assessment of transfer and the rule used

Round 2's rule is used unchanged: the overall gate (yes/no AUC under the neutral line at least 0.70 between the main failures and the 600 successes; Yes on at most 95% of failures and at least 5% of successes; median P(Yes) + P(No) at least 0.5) is reported, and each contrast is read against its own rule (the yes/no neutral-line AUC at least 0.70 in each of its populations, with the other three criteria on the full population). Competence never withholds a verdict; it decides whether a confirmed effect is read as a distortion in a model that discriminates successes from failures, or only as a measurable goal effect.

Assessment, fixed before any round-3 effect exists:

- The AUC is a rank statistic: it does not depend on a model's logit scale or calibration, so it transfers across models unchanged. Newer models may sit near its ceiling, which makes the heuristic tracking (H1, H2, H6, H2 and H9) more important for saying how a model discriminates; a high AUC can come from a shortcut that coincides with the truth.
- The two Yes shares are read at L = 0 and guard against a model that answers one word to everything; that concern is model-independent, so they transfer.
- The mass depends on the reply format. The CPU check confirmed that every model's template replies to "Yes" and "No" with the single bare token the scorer reads; whether the model puts its probability there is seen only with the weights (smoke run, and the gate's mass criterion).

Adaptations: none to the thresholds, so "competent" means the same in rounds 2 and 3. Added, descriptive only: each contrast's competence at a stricter floor of 0.80 (`r3_analysis.STRICT_AUC`), and the thinking arm's competence after reasoning, which is a different quantity and gates nothing.

### 7.2 The equivalence margin: assessment of transfer and the rule used

Round 2's margin is used unchanged for family A3 and A3x: for model m, on the two-candidate scale and for all five contrasts,

$$\delta_m = \min\left(0.3 \times \mathrm{SD}_m,\ 0.4\ \text{nats}\right),$$

where the SD is the item standard deviation of ΔL over the 1,500 main failures, from the point estimates, before any equivalence test is read.

Assessment: the cap of 0.4 nats (an odds ratio of about 1.5) is an absolute size on a scale whose meaning differs between models. Newer, more confident models are likely to have larger spreads, so the cap will usually bind and, as for Qwen3-8B in round 2's power table, equivalence will often be unreachable for P, Dn, Dc and T. The consequence is stated now: in such a model a contrast can fail to replicate only by being confirmed in the opposite direction, and a "general null" from round 2 can rarely be replicated. The margin is still kept exactly, because an equivalence verdict must mean "no effect larger than δ nats" in both rounds and a margin chosen with round 3's spreads in view could be tuned to the data. Every test reports its bootstrap SE, its equivalence power at a true zero (at Holm's first step and at 0.05) and the scale-free readout of section 7.6.

### 7.3 Analysis (a): direct replication

**Confirmatory hypothesis H3-P** (carried forward from the round-2 design, section 7, the account's prediction for any model): for each core model, the within-item penalty P is below zero, read as "confirmed below zero" in family A3. No directional prediction is made for Λ, Dn, Dc or T, because the round-2 design made none for models other than its two reference models; they are tested and read through round 2's tables.

**Rejection criteria, per core model,** as frozen in round 2: R1 (a model competent for P with P equivalent to zero or confirmed above zero: the penalty component is rejected for that model); R5 (a model competent for T with T confirmed below zero); R3 as round 2's descriptive diagnostic. **R2 restated for round 3** (`r3_analysis.r2_contemporary`): among the core models competent for P, if any has P confirmed below zero, R2 does not fire; otherwise, if any has P equivalent to zero or confirmed above zero, R2 fires (the penalty is not shared among these contemporary models); otherwise it is undecided. Round 2's R4 compares Llama-3.1 with Qwen3 and has no round-3 counterpart.

**H3-P, read per model and overall** (`r3_analysis.h3p_reading`). Per core model: *supported* (competent for P and P confirmed below zero), *rejected* (R1 fires), *measurable only* (P confirmed below zero in a model not competent for P), otherwise *inconclusive*. Overall:

- **supported**: at least two core models are competent for P, every one of them is supported, and R2 does not fire;
- **rejected**: R2 fires, or R1 fires in at least one competent core model and none is supported;
- **partial**: at least one competent core model supported and at least one rejected or inconclusive; listed per model, labelled "partial", and never counted as support;
- **inconclusive**: anything else, with the reason (fewer than two competent core models; every competent model inconclusive).

**Round 2's general finding, per contrast** (`r3_analysis.round2_finding`), computed mechanically from round 2's filed analysis (its verdicts and per-contrast competence, `runs/textcraft_r2_analysis/metrics.json`), once that exists:

- *general, sign s*: at least two round-2 models are competent for the contrast, and it is confirmed in direction s in every one of them;
- *general null*: at least two competent, and equivalent to zero in every one;
- *model-specific*: among the competent models, confirmed in some and equivalent or opposite in others;
- *not established*: anything else (fewer than two competent; inconclusive or not applicable among them).

**The replication rule** (`r3_analysis.replication`), among the core models competent for the contrast (models not competent are listed beside the reading and do not count), needing at least two competent:

| Round 2's finding | Replicated | Not replicated | Partial (reported per model, never counted as replication) | Inconclusive |
|---|---|---|---|---|
| general, sign s | confirmed in direction s in every competent core model | none confirmed in direction s, and at least one equivalent to zero or confirmed opposite | at least one confirmed in direction s, and at least one other inconclusive, equivalent or opposite | fewer than two competent core models; or none confirmed in direction s and none equivalent or opposite |
| general null | equivalent to zero in every competent core model | at least one confirmed (either direction) | at least one equivalent and the others inconclusive, none confirmed | fewer than two competent; or every competent model inconclusive |
| model-specific or not established | no replication target: the per-model verdicts are new evidence on the account (H3-P and the rejection criteria) | | | |

The extension models get the same readings computed separately (family A3x; `replication` over the extension tags), never merged with the core.

### 7.4 Analysis (b): older against newer releases

**Pairs (fixed):** Qwen3-8B → Qwen3.5-9B; Mistral-7B-Instruct-v0.3 → Ministral-3-8B-Instruct; Gemma-2-9B-it → Gemma-4-12B-it (or its fallback). The older model's scores are round 2's (`textcraft_r2_score_<tag>`), read only after round 2 is scored and filed, under round 2's provenance check.

**Why the primary scale is standardized.** Two-candidate nats are not comparable across models: the forensic audit on main (`notes/qwen_effect_audit_forensic.md`) found Qwen3-8B's 6.12-nat ALFWorld effect to be about 1.0 standardized against Llama-3.1's 0.85, and the Qwen3 size series flat once standardized. The models' logit scales differ, and they split the replies into different numbers of tokens (round 3's CPU check: 9.2 to 11.8 tokens for the success reply). So every cross-model or cross-generation size comparison in round 3 uses a standardized effect as its primary scale; raw nats are descriptive only.

**The standardized contrast** of model m (`r3_analysis.standardized`): the contrast divided by the standard deviation of the item-level values that enter it. For Λ and P, which are means of per-item differences d, it is the mean of d over the SD of d:

$$z_m = \frac{\bar d}{\mathrm{SD}(d)}.$$

For Dn, Dc and T, which are a goal-role mean minus an ingredient mean of per-item values, it is that difference over the pooled within-role SD of the values:

$$z_m = \frac{\bar x_{\text{goal}} - \bar x_{\text{ingredient}}}{\sqrt{\dfrac{(n_g - 1)s_g^2 + (n_i - 1)s_i^2}{n_g + n_i - 2}}}.$$

**The estimand of a pair** is the difference of the standardized contrasts on the same records,

$$\Delta z = z_{\text{new}} - z_{\text{old}},$$

computed only on records both models scored. **Uncertainty:** one resample of goal pairs serves both models; in each of 20,000 resamples (seed `RNG_SEED`) both models' standardized contrasts are recomputed on the resampled pairs and differenced. The 95% percentile interval of those differences is the interval; the two-sided bootstrap p uses round 2's formula; the equivalence test uses round 2's TOST against a fixed margin of 0.20 standardized units (`r3_analysis.B3_MARGIN`; a fifth of a standard deviation, Cohen's "small"). No reading compares two point estimates or two separate intervals.

**Family B3** (section 7.7) holds the differs and equivalence tests of the 3 pairs × 5 contrasts. **Readings** (`r3_analysis.pair_reading`): *differs, newer higher* or *differs, newer lower* (Holm-adjusted differs-p below 0.05 and the 95% interval excluding zero); *similar within the margin* (Holm-adjusted equivalence p below 0.05); *not distinguishable* otherwise, never read as "the same". If the environment check (section 8.1) has not passed, every B3 reading is "not interpretable (environment check)" unless the declared fallback (full rescoring of the older models in serve2) has been run, in which case the pairs use those rescores.

**Descriptive beside each pair, never verdicts:**

- the raw difference in nats, with its paired interval;
- for Λ and P, each model's item share (the share of items whose per-item difference is above zero) and the paired difference of shares;
- for each model and contrast, the first-token / tail split: the contrast computed on the first-token log-odds (the difference between the two replies' first-token log-probabilities) and on the tail (L minus that), each with its interval, so a reader can see how much of a difference is carried by the reply's later tokens;
- each model's competence for the contrast;
- what changes within the pair: developer data and post-training, parameters, tokenizer and segmentation (Ministral 3 splits "Crafted" into "C" + "rafted", as Gemma 2 did), template (system text first in Ministral 3, in the last user turn in Mistral 7B; a system turn and an empty thought channel in Gemma 4 against the folded first turn of Gemma 2), attention (Qwen3.5's linear layers; Gemma 4's 1,024-token windows, exceeded by 99% of inputs, against Gemma 2's 4,096), and the software environment.

With three purposive pairs, the directions of the differences are described, never summarized as a trend.

### 7.5 Analysis (c): exploratory

Everything here is estimation with 95% intervals (2,000 goal-pair resamples) and no test enters any family.

- **Size and recency within a line:** Qwen3.8-27B against Qwen3.5-9B, Gemma 4 31B against Gemma 4 12B, each as a paired standardized difference with raw nats beside it.
- **Format checks** (section 8.3), read by wave A's one-fifth rule.
- **Thinking arm** (section 8.2): Qwen3.5-9B thinking on against the same weights thinking off.
- **Capability:** for every instruction model of rounds 2 and 3, each contrast's standardized value against the model's competence AUC; a table, no test, no trend claim.
- **Architecture:** named beside each model (full attention; sliding-window hybrids; linear-attention hybrids), never credited with an effect.

### 7.6 Scale-free readout in analyses (a) and (c)

Analysis (a) compares only verdicts across models, never sizes. Wherever a size is placed beside another model's in a table (the across-model table, analysis c), the standardized contrast is the primary column and nats are descriptive. Every round-3 model block also carries each contrast's standardized value and its first-token / tail split.

### 7.7 Multiplicity

Each family is Holm-adjusted at a familywise 0.05, with its size fixed now. A test whose model was not scored, or whose p is missing or not a number, enters with p = 1, so no family shrinks.

| Family | Tests | Size |
|---|---|---|
| A3 (analysis a, core) | 3 core models × 5 contrasts × (differs, equivalent), raw nats as round 2 | 30 |
| A3x (analysis a, extension) | `qwen3.8-27b`, `gemma4-31b-it`, `olmo3-7b-it` × 5 × 2 | 30 |
| B3 (analysis b) | 3 pairs × 5 contrasts × (differs, equivalent), on the standardized difference | 30 |
| Analysis (c), the thinking arm, the format checks, every descriptive readout | none | 0 |

Round 2's family of 50 is untouched and never combined with these. A3, A3x and B3 answer different questions and are not corrected across each other.

### 7.8 Success, failure and inconclusive evidence

- **Analysis (a), per contrast:** the replication table of 7.3. **For the account:** H3-P overall as defined in 7.3 (supported, rejected, partial, inconclusive).
- **Analysis (b):** no success or failure. Each pair and contrast reads differs (newer higher or lower), similar within the margin, or not distinguishable.
- **Analysis (c):** nothing is confirmed or rejected.

## 8. The environment check, the thinking arm and the format checks

### 8.1 Environment check (wave B's pattern, on TextCraft records)

Round 2 scored in the study's older environment (transformers 4.51.3, torch 2.6.0); round 3 scores in serve2 (transformers 5.23.0, torch 2.14.1). Before any B3 reading:

- **Part 1, tokens (CPU, before the freeze):** the round-3 gate, run in serve2 with Qwen3-8B's and Mistral-7B-v0.3's pinned tokenizers on every record and arm, must pass, and their golden renderings (round 2's five cases) must have the same id hashes as round 2's stored goldens (`r2_chat_golden.json`, made in 4.51.3).
- **Part 2, scores (GPU):** `textcraft_r3_envcheck_<tag>` for `qwen3-8b` and `mistral-7b-v0.3` rescores 300 seeded records (`random.Random(f"{RNG_SEED}|textcraft|r3|envcheck")` over all 2,982), every arm, in serve2, with wave B's serve2 loading of the earlier models (`families.envcheck_spec`), scores saved under the tag `<tag>-serve2`. It runs on the GPU type that the model's round-2 shards recorded; the analysis compares only records whose round-2 shard ran on exactly that type and needs at least 200 such records.
- **Pass rule** (wave B's thresholds): yes/no mean absolute difference per score at most 0.07 nats; two-candidate at most 0.35 nats; on the compared main failures, the aligned-minus-swapped effect differs by at most 0.10 nats on each scale, and the yes/no share of items with aligned above swapped by at most 0.02; every expected score present; the GPU type as required.
- **If it fails:** no B3 reading is made from round-2 scores. The three older models are rescored on all records in serve2 (`textcraft_r3_rescore_<tag>`, about 7 GPU-hours) and B3 is computed serve2 against serve2. Analyses (a) and (c) do not depend on it.

### 8.2 The thinking arm (exploratory; outside every family; never gated or pooled with the instruction runs)

Qwen3.5-9B at the same revision, its template with `enable_thinking=True` (the generation prompt ends `<think>\n`), generated with HF `generate` in serve2 (`wf/textcraft/r3_think.py`, run `textcraft_r3_think_qwen3.5-9b`):

- **Records and arms (4,530 generations):** the 675 parent failures under `ext_aligned` and `ext_parent` (P); the 990 ingredient main failures under `aligned` and `swapped` (Λ); and, for competence, `yn_neutral` on the 600 determined successes and a seeded 600 of the 1,500 main failures (`random.Random(f"{RNG_SEED}|textcraft|r3|think|failures")`).
- **Generation:** Qwen's recommended thinking sampling (temperature 0.6, top-p 0.95, top-k 20, as the project's thinking probe), at most 2,048 new tokens, stopping at the end of the reasoning block. Prompts go in fixed batches of 8 in record-and-arm order; each batch's sampling seed is fixed from `RNG_SEED` and the batch's first record and arm. Every batch is saved as it finishes and a resumed job skips saved generations.
- **Budget forcing:** a generation that has not closed its reasoning within 2,048 tokens is closed with `\n</think>\n\n` and flagged `forced`; every readout is reported on all generations and on unforced ones only.
- **Where the readout is taken:** after the rendered prompt, the reasoning, the closing `</think>` and a blank line. The two candidates (two-candidate arms) or "Yes" and "No" (the yes/no arm) are teacher-forced there with the scorer's own forward pass, and L after reasoning is their log-odds. The round-3 CPU check found no candidate and neither answer word merging across `</think>\n\n`; the run checks the boundary again for every scored item and records it.
- **Readouts:** P and Λ after reasoning, each with its 95% interval and standardized value; their paired differences from the same records scored with thinking off (`textcraft_r3_score_qwen3.5-9b`), on the standardized scale with nats beside it; the yes/no AUC after reasoning; the forced share, the median reasoning length, and the share of traces that mention "goal" or "task" or the crafted item. Read as: what Qwen3.5-9B's own reasoning does to the goal-line effects on these records, mixed with sampling noise.

### 8.3 Format checks (exploratory, one-fifth rule)

On 300 seeded main failures (`random.Random(f"{RNG_SEED}|textcraft|r3|format")`), arms aligned, swapped, yn_aligned and yn_swapped:

- `textcraft_r3_placement_gemma4-12b-it`: Gemma 4 12B with the system text folded into the first user turn (as Gemma 2 had it), tag `gemma4-12b-it-firstturn`;
- `textcraft_r3_nopreserve_qwen3.8-27b`: Qwen3.8 with `preserve_thinking=False` (no empty think blocks in the example replies), tag `qwen3.8-27b-nopreserve`.

Each is compared with its model's native scores on the same records: the paired difference of the goal effect (aligned minus swapped) per format. Wave A's rule: if the difference's 95% interval excludes zero and its size exceeds a fifth of the native effect, the model's readings are reported as format-dependent, with both numbers.

## 9. Storage and weights

Route A (decision D5): each GPU job downloads the pinned snapshot into the node's local disk (`--cache <node-local> --node-local`, never under /storage), with the hf-xet chunk cache node-local too; the free space is checked against the manifest first; after the download every file is hashed against the committed manifest (`wf/config/weights_manifests/<org>__<name>.json`, built from Hub metadata only); any missing, extra or different file stops the job before loading; the verification record is copied into the shard. The two new manifests (Gemma 4 12B, OLMo 3 7B Instruct) are built before the freeze.

## 10. GPU types

Each model runs every shard on one pinned GPU type (section 2): NVIDIA L40S for the 7 to 12 B models and the thinking arm; an A100 with 80 GB for the 27B and 31B models, with two L40S on one node as the declared alternative, chosen at submission and identical for every shard of that model. The job checks the type before loading; the analysis checks that every shard of a model recorded the same GPU list and that it is the pinned one (or the declared alternative), and refuses otherwise. The format checks run on their model's type; the environment check on the older model's round-2 type (8.1). Across pairs the two models necessarily ran on whatever types their rounds pinned; that is part of the environment difference the check bounds.

## 11. Run order and cost

Order (`r3.RUN_ORDER`): the environment check (both models); the smoke runs of the three core models (the Gemma fallback rule is applied here); the three core models; then, as GPUs allow, `olmo3-7b-it`, `qwen3.8-27b` with its format check, `gemma4-31b-it`, the Gemma 4 12B placement check, and the thinking arm (smoke first). Whether the extension and the thinking arm run depends on GPU availability only, recorded before any core result is read; their families stay fixed in size. The analysis runs once each stage's scores are filed.

| Stage | Runs | GPU-hours (estimates) |
|---|---|---|
| Minimum | environment check (0.6), core smoke runs (0.3), `qwen3.5-9b` (2–4), `ministral3-8b-it` (1.7–2.5), `gemma4-12b-it` (2.5–3.5), downloads (1–2.5) | about 9–13; budget 14 |
| Extension | `olmo3-7b-it` (1.5–2), `qwen3.8-27b` (6–9) and `gemma4-31b-it` (6–9) on 80 GB cards, two format checks (0.5), smoke runs (0.3), downloads (1–2.5) | about 15–23, 12–18 of them on 80 GB cards |
| Thinking arm | smoke (0.2) and 4,530 generations with scoring | about 3–6 |
| All | | about 27–42; budget 45 |
| If the environment check fails | full serve2 rescoring of the three older models | about 7 more |

## 12. Code

Built after this note is committed; nothing round-2 or wave-B specific changes behaviour.

- Merge (before this note; merge commits ed533a9 and 7dbbccf): round 2's tag, then wave B's tag. The full local suite on the merge: 1,154 passed, 25 skipped. Every file under `wf/textcraft/`, `tests/test_textcraft_r2.py` and round 2's data equals round 2's tag; `wf/families.py`, `wf/scoring/`, `wf/constants.py`, `wf/tools/` and the wave B tests equal wave B's tag. Three files conflicted only because both studies appended to the same lists (`wf/runs.py` runners, `scripts/mutation_check.py` mutants and runner, `notes/testing.md`); both sides were kept, the mutation runner takes wave B's strict named-kill rule, and round 2's screening switch (`MUTATION_FAST_ONLY`) is kept in it.
- `wf/constants.py`: the two new pinned revisions (additions only).
- `wf/textcraft/r3.py`: the model specs (wave B's specs reused for its five models; new specs for `gemma4-12b-it`, `olmo3-7b-it` and the thinking arm), tags, GPU pins, families, pairs, run order, seeds.
- `wf/textcraft/r3_tokcheck.py`: the gate of section 4 and the golden renderings, CPU, every model.
- `wf/textcraft/r3_prefetch.py`: route A for the round-3 tags, from wave B's prefetch and manifest functions (wave B's tool knows only its own tags, and its files are frozen).
- `wf/textcraft/r3_score.py`: `textcraft_r3_score_<tag>`, `textcraft_r3_smoke_<tag>`, `textcraft_r3_envcheck_<tag>`, `textcraft_r3_rescore_<tag>`, `textcraft_r3_placement_gemma4-12b-it`, `textcraft_r3_nopreserve_qwen3.8-27b`.
- `wf/textcraft/r3_think.py`: `textcraft_r3_think_qwen3.5-9b` and its smoke run.
- `wf/textcraft/r3_analysis.py`: `textcraft_r3_analysis` (families A3, A3x, B3; H3-P; the replication rule; R2 for round 3; standardized and first-token / tail readouts; the environment check; the thinking arm; the format checks; provenance and GPU checks).
- Configs `wf/config/textcraft_r3_*.yaml`; jobs `pace/textcraft_r3_score.sbatch` (GPU, serve2, node-local weights) and `pace/textcraft_r3_tokcheck.sbatch` (CPU); tests `tests/test_textcraft_r3.py` and the serve2-only `tests/test_textcraft_r3_serve2.py`; round-3 mutants in `scripts/mutation_check.py` under the strict named-kill rule.

## 13. Checks before the freeze (2026-11-22 and 23; no model loaded with real weights, no round-3 score)

**Merge.** Round 2's tag, then wave B's (merge commits ed533a9 and 7dbbccf). Every round-2 file and round-2 test equals round 2's tag, and every wave-B file equals wave B's tag; three files that both studies append to were merged by keeping both sides (section 12). Full local suite on the merge: 1,154 passed, 25 skipped.

**The round-3 gate on every record and arm, all twelve tags** (ICE, fresh checkout `text-wm-fix43` from a git bundle at 799f8c3, CPU job 7246224, serve2: transformers 5.23.0, tokenizers 0.23.3, offline, tokenizer and config files only, in a cache inside the checkout; `wf/textcraft/data/r3_tokcheck_ice.json`). For each of the seven models, the two format variants and the three earlier models as serve2 loads them: 32,942 candidate checks, 9,272 yes/no prompts, 17,057 arm pairs isolated, **0 problems**. Every round-3 model's 25,743 prompts end with its declared thinking-off ending; Ministral 3's 25,743 prompts equal mistral-common's encoding.

| Tag | Yes / No ids | First tokens (success vs failure) | Longest input | Minutes |
|---|---|---|---|---|
| `qwen3.5-9b` | 9175 / 2665 | "Craft" vs "Could" | 1,832 | 11.6 |
| `ministral3-8b-it` | 16860 / 4753 | "C" vs "Could" | 1,848 | 17.1 |
| `gemma4-12b-it` | 10784 / 3771 | "Craft" vs "Could" | 1,717 | 11.8 |
| `gemma4-e4b-it` | 10784 / 3771 | "Craft" vs "Could" | 1,713 | 11.7 |
| `qwen3.8-27b` | 9175 / 2665 | "Craft" vs "Could" | 1,844 | 11.7 |
| `gemma4-31b-it` | 10784 / 3771 | "Craft" vs "Could" | 1,717 | 11.8 |
| `olmo3-7b-it` | 9642 / 2822 | "Craft" vs "Could" | 1,815 | 11.6 |
| `gemma4-12b-it-firstturn` | 10784 / 3771 | "Craft" vs "Could" | 1,713 | 12.4 |
| `qwen3.8-27b-nopreserve` | 9175 / 2665 | "Craft" vs "Could" | 1,832 | 11.8 |
| `qwen3-8b-serve2` | 9454 / 2753 | "Craft" vs "Could" | 1,819 | 11.0 |
| `mistral-7b-v0.3-serve2` | 6360 / 2538 | "Craft" vs "Could" | 2,054 | 9.1 |
| `gemma2-9b-it-serve2` | 3553 / 1294 | "C" vs "Could" | 1,706 | 7.8 |

**Golden renderings** (`wf/textcraft/data/r3_chat_golden.json`, from the same job): round 2's five cases for all twelve tags. **Environment check, part 1:** the serve2 renderings of Qwen3-8B, Mistral-7B-v0.3 and Gemma-2-9B-it equal round 2's stored renderings (made with transformers 4.51.3) in text, id count and id hash, in all five cases (a test pins this).

**Weights manifests** (`wf/config/weights_manifests/`, Hub metadata only, no token; neither repository is gated): Gemma 4 12B-it, 7 files, 23.95 GB; OLMo 3 7B Instruct, 12 files, 14.61 GB; `snapshot_download(dry_run=True)` plans exactly the same files and sizes. The other five models use wave B's committed manifests.

**serve2 forward tests** (`tests/test_textcraft_r3_serve2.py`, ICE CPU job 7246366: 4 passed). Tiny random-weight models built from the pinned configs of `Gemma4UnifiedForConditionalGeneration` and `Olmo3ForCausalLM` pass the round-3 gate on a round-2 record with the real pinned tokenizers and score it end to end (two-candidate and yes/no; the one-pass and two-pass yes/no scores agree). The Gemma 4 12B tiny model, saved with its vision-embedder tensors under the names the real checkpoint stores, loads back under the strict loading check with no missing, mismatched or unexpected parameter and with identical tensors: transformers renames the stored names on load. The eager-against-SDPA agreement check passes on the real class and switches back to SDPA. Wave B's own serve2 forward tests pass on the merged code in the same environment (job 7246496: 14 passed, 1 skipped, the skip being the Qwen3.5 base model, whose tokenizer the checkout's cache does not hold).

**Mocked dry run, end to end** (`scripts/textcraft_r3_dryrun.py`, every scorer the mock scorer, round 2's full records): in a fresh runs folder, round 2's five models (128,715 scores, round 2's own run) and round 2's analysis, then every round-3 run in its config's shards: the six core and extension models (25,743 scores each, none missing), both environment-check rescorings (2,602 scores each: every arm of the 300 seeded records), both format checks and the thinking arm (4,530 generations and scores, a quarter of them budget-forced by the mock). The round-3 analysis exits 0 and writes every block: families A3, A3x and B3 of 30 tests each, H3-P, the replication readings against round 2's filed (mock) verdicts, the environment check (passed on the mock scores, with part 1 read from the golden files), all three pairs, the size comparisons, both format checks, the thinking arm and the capability table; 22 minutes on a laptop. The mock scores carry no effect, so no reading means anything.

**Tests and mutants.** `tests/test_textcraft_r3.py`: 52 tests (every decision rule of section 7 and 8, the gate, the runs, the thinking arm, two end-to-end mock runs, the golden file). Mutation testing under wave B's strict named-kill rule (`notes/textcraft_r3_mutation_log.txt`, `notes/testing.md`): 67 round-3 mutants, all killed, by 38 different tests; a screen run first found nine survivors, and tests for each were added before the official run.

**Full suite, once at the end** (local, at bbfe6fa): 1,208 passed, 29 skipped (the skips need transformers 5, torch or cached tokenizers; the serve2-only tests ran on ICE).

**What has not been run.** No GPU job, no smoke run, no weights downloaded or loaded, no round-3 score, and no round-2 score or result read. Not checkable before the weights load: the real checkpoints' strict loading, the real attention agreement, speed, and each model's yes/no mass; each is checked at the start of each GPU job (the smoke runs first), before its first score.

## 14. Amendments after review 1 (2026-11-23, before the freeze and before any round-3 score)

Review 1 ([textcraft_r3_review1.md](textcraft_r3_review1.md), commit 2a3a768, merged into this branch) found four must-fix and thirteen should-fix items. The coordinator decided the should-fix items; this section states the rules as amended. **Where an earlier section of this note differs, this section governs.** No round-2 score or result was read, and no model has scored a round-3 record. Code: `wf/textcraft/r3_gates.py` (new), and changes to `r3.py`, `r3_score.py`, `r3_think.py` and `r3_analysis.py`.

### 14.M1 GPUs are checked per job

`wf.provenance.run_record` keeps a shard's GPUs as a set of names, so a job on two L40S would read as one L40S and fail the 27B and 31B models' declared alternative. The analysis now reads each job's own GPU list from its shard's `jobs.jsonl` (`r3_gates.job_gpus`). Every job of every shard of a model must have recorded the identical list, and that list must be the model's pinned type or its declared alternative. A problem refuses only the family it belongs to (A3, A3x, or B3 through the pair's models or older scores); the other families are read.

### 14.M2 A model that fails its gate is "not scored"

A model is **not scored** when any filed gate record of its smoke or score run has `ok: false`: the strict loading check (now caught and filed by the scoring run as `loading_failed`), the tokenization gate, or the attention agreement (`r3_gates.model_state`). Other failures (a download, the weights hash check, a crash, an out-of-memory error) are not filed as gate failures; the job is rerun. A model not scored gets verdict "not scored" and competence false for every contrast, its tests enter their family at p = 1, and H3-P, R2 and the replication readings proceed with the other models. A B3 pair whose newer model is not scored reads "not scored" (its tests at p = 1). "Waiting" is kept only for a model whose run is not filed yet.

### 14.M3 and S9 The Gemma fallback is code

`r3_gates.gemma_slot` applies the rule to Gemma 4 12B's filed smoke record (`runs/textcraft_r3_smoke_gemma4-12b-it/shard_00_of_01/`). It returns `gemma4-e4b-it` when any of these holds: the loading check failed; the tokenization gate failed; the attention agreement failed; or a **format mismatch** (S9): the median yes/no mass P(Yes) + P(No) over every yes/no score of the smoke run is below one half **and** on more than half of those prompts the model's most probable next token is a structural or template marker (a special token, whitespace, markdown, or a bracketed marker) rather than an answer word. Every smoke run records that token (`top_token`, `top_token_special`) for every yes/no prompt, so the diagnostic is reported for every core model. A low mass without that evidence does not fire the fallback; it goes to the competence gate, and the model is reported "not competent", not substituted. Until 12B's smoke record is filed the slot is "waiting".

The analysis takes the slot from this function and refuses a configured `gemma_core` that disagrees. The `gemma4-e4b-it` smoke and score runs refuse to start unless the function returns E4B. **If E4B then fails its own gate, the Gemma slot is "not scored"** (14.M2): Gemma's core tests enter at p = 1, and there is no further substitution.

### 14.M4 The environment check mirrors round 2's shards

*Why.* Round 2 scores each model in four shards, and its job script allows L40S, A40 or A100. With mixed types, at least 200 comparable records out of 300 would often be unreachable for a reason unrelated to software. (The coordinator reports that round 2's scoring now runs with every shard pinned to L40S, so the single-type shortcut below should apply; the general rule is implemented regardless.)

*The rule* (`r3_gates.envcheck_plan`, `r3_score.run_textcraft_r3_envcheck`, `r3_analysis.envcheck_block`):

1. **Plan from metadata only**, before any environment-check job: from the GPU names in each round-2 shard's `jobs.jsonl` for Qwen3-8B, Mistral-7B-v0.3 and Gemma-2-9B-it (S8). A shard is *typed* when every job recorded the same single-GPU list; otherwise its records are not comparable. The plan (shard, GPU name, seeded records) is written by `python -m wf.textcraft.r3_gates plan --r2-runs <round-2 runs>` to `wf/config/textcraft_r3_envcheck_plan.json`. **It is committed before the freeze tag** (review 2, R2-M1): the freeze waits for it, so every round-3 job runs from the tag's commit with the plan in it (no exemption from 14.S12). With the per-score criterion resolved to wave B's original one (item 5), the plan is the only item the freeze waits for. **Outcome (2026-11-23, committed before the freeze):** round 2 ran all four shards of each of the three models from `text-wm-fix42` at 4ccf84b, and every job recorded exactly `NVIDIA L40S`, so every shard is typed and the plan for each model is a **single L40S job covering round-2 shards 0 to 3, 300 of 300 seeded records comparable** (75, 55, 68 and 102 per shard), runnable. The planner read only each job's recorded GPU names (copied from `jobs.jsonl`, nothing else) and round 2's committed record file (`wf/textcraft/data/r2_items.jsonl`, sha256 identical to the one in `text-wm-fix42`) for the shard assignment; no round-2 score, metrics, report or status file was read.
2. **Mirror round 2's shards.** One job per typed round-2 shard (`WF_ENVCHECK_JOB`): the seeded records that lie in that shard by round 2's own partition (`r2_score.shard_records(all records, i, 4)`), every arm, pinned by `--constraint` and by a job-start check that refuses any GPU list other than exactly that shard's name.
3. **Shortcut.** When every typed shard shares one type, one job on that type scores all of them.
4. **Comparison, record by record.** A record is comparable when its round-2 shard and the environment-check shard that rescored it are each typed and their GPU lists are identical.
5. **Pass rule:** on the union of comparable records, with at least 200 comparable, **wave B's original criteria, unamended**, written into round 3's own constant (`r3.ENVCHECK_RULE`) and pinned with its sha256 in `wf/config/textcraft_r3_envcheck_rule.json` (the analysis refuses if the constant and the pin differ): the mean absolute difference per score at most 0.07 nats on the yes/no question and 0.35 nats on the two-candidate score; on the comparable main failures, the aligned-minus-swapped effect within 0.10 nats on each scale and the yes/no share of items with aligned above swapped within 0.02. Round 3 does not read wave B's rule from `wf/families.py` and calls no wave B function, so a later change in wave B cannot change round 3. *(Resolved after review 2: wave B's rounding hypothesis was refuted; its diagnostic found that, before any bf16 rounding, Qwen3-8B's yes/no L moves about 0.22 nats in float64 between the two environments, a final hidden state perturbed by about one bf16 epsilon and amplified by Qwen's large readout logits (median 52 to 62). Wave B adopts no amendment, and neither does round 3.)*

   **What to expect.** On wave B's evidence with the same two models and software, Qwen3-8B will very likely fail the yes/no per-score criterion (wave B measured 0.248 nats against 0.07), and Mistral-7B-v0.3 sits at the limit (wave B: 0.0688). The declared consequence then applies: the environment check reads "fail", and B3 uses the full serve2 rescoring of the three older models, each job on exactly one L40S (about 7 GPU-hours, inside the budget of 14.C). A failure here is not evidence about round 3's models; it only decides that B3 compares serve2 with serve2.
6. **Outcomes.** "pass" (every model passes and its golden tokens agree with round 2's); "fail" (any model fails, or its golden tokens differ); **"not run (GPU types)"** (fewer than 200 comparable records because the plan could not reach 200, or a planned GPU type was not obtainable within 7 days of the first submission, filed as `gpu_unobtainable.json` in that model's run folder); "waiting" otherwise. "Not run" is not "failed": no software difference was detected; the check could not be made.
7. **B3's older scores follow from the outcome (S5).** "pass": round 2's scores, after round 2's provenance check (S6). "fail" or "not run (GPU types)": the declared full serve2 rescoring of the three older models (`textcraft_r3_rescore_<tag>`), each job on exactly one L40S. A configured `b3_old_scores` that disagrees is refused.

S8: Gemma-2-9B-it is now one of the environment check's models (`textcraft_r3_envcheck_gemma2-9b-it`, about 0.3 GPU-hours), under the same rule; its serve2 renderings already equal round 2's.

### 14.S1 One competent model is too few for H3-P

With fewer than two core models competent for P, H3-P overall is "inconclusive (fewer than two competent)", also when the one competent model is supported. "Rejected" keeps its precedence, inherited from round 2's R2: one competent model with P equivalent to zero or confirmed above zero, and none confirmed below, rejects H3-P (note N1: the rule is asymmetric, and is stated beside the reading).

Note N2: an inconclusive round-2 model that is competent makes a contrast "not established", so round 3 reads "no replication target" for it. That means round 2 did not establish a general finding, not that round 2 found nothing; for P the headline may then rest on H3-P alone.

### 14.S2 Equivalence power at Holm's first step of 30

Round 2's power fields use 0.05/50. Round 3 recomputes each A3 and A3x contrast's equivalence power at a true zero at 0.05/30 (one-sided z = 2.935) and sets the "equivalence underpowered" flag from it (`r3_analysis.holm30_power`); round 2's values are kept beside it.

### 14.S3 B3 equivalence power, the flag, and a fixed-denominator column

Each B3 test reports its equivalence power at a true zero at Holm's first step (z = 2.935) and at 0.05, and "not distinguishable" reads "not distinguishable (equivalence underpowered)" when the first is below 0.8. The margin stays 0.20 standardized units. **Equivalence is practically unreachable in B3 for Dn, Dc and T at these sample sizes:** on the real goal-pair structure the reviewer's simulation gave equivalence power at Holm's first step of 0.00 for Dn, Dc and T unless the two models' item values correlate at about 0.8 (T: 0.41 there), against 0.39–0.92 for Λ and 0.02–0.74 for P (correlation 0 to 0.5). For those contrasts the realistic readings are "differs" (for differences of roughly 0.3 to 0.5 SD or more) or "not distinguishable". A descriptive column divides each model's contrast by a fixed denominator, its SD of ΔL over the 1,500 main failures (the SD that sets its margin), so a reader can see whether a B3 difference comes from the mean or from the spread.

### 14.S4 Plain-words directions

"Differs, newer higher" and "newer lower" carry the size in words: for two negative values, "smaller in size" or "larger in size", and for P "a smaller penalty" or "a larger penalty"; "opposite signs" when the signs differ.

### 14.S6, S7 Round 2's provenance and the resample count

Before B3 reads round 2's scores, round 2's own provenance check runs (`r2_analysis.provenance` on the three older models); any problem refuses B3. Round 3's metrics record the sha256 of round 2's `metrics.json` and its status line. B3's tests use 20,000 resamples; a different `b_draws` is refused outside a mock run.

### 14.S10 The thinking arm

- The forced flag is split: `forced_reason` is "cap" (the reasoning reached 2,048 tokens without closing) or "ended unclosed" (the turn ended without closing it); the unforced readouts exclude both.
- At job start, before generating: every thinking prompt is checked (it ends with the thinking-on generation prompt, holds the clean system text and no forbidden text), and the next-token log-probabilities of the shortest prompt of the first batch, alone and left-padded inside the batch, must agree over its 20 most probable tokens within the attention-agreement tolerances (mean 0.10 nats, any 1.0). Either failure stops the job (`start_checks.json`).
- Reproducibility depends on the shard count and the batch size (they decide which prompts share a batch, its padding and its random stream), and GPU sampling is not bitwise deterministic. Every generation is saved with its batch key and seed, and the readout is scored from the saved text.
- **Cost, re-estimated from the smoke plan.** 4,530 generations make 567 batches of 8; a batch lasts as long as its longest trace, up to 2,048 tokens. At the 15–25 tokens per second per sequence expected for a 9 B model with PyTorch linear attention on an L40S, a batch takes about 60–140 seconds, so generation takes about 9–22 hours, and scoring after the traces about 0.5. The thinking smoke run (two batches) measures seconds per batch, and the full run is planned as 567 times that. The estimate below uses 9–22 hours.

### 14.S11 No p-values outside the families

The exploratory blocks (the size comparisons and the thinking arm's paired differences) carry estimates and intervals only.

### 14.S12 Every job runs from the freeze tag

Every round-3 GPU job runs from a checkout at the freeze tag's commit (detached, never pulled), and checks at job start that its HEAD is that commit or differs from it only outside `wf/`, `scripts/` and `pace/` (`r3_gates.require_freeze_commit`). The analysis checks every job of every shard of every round-3 run (scores, smoke runs, environment check, rescoring, format checks and the thinking arm) against the same rule.

### 14.S13 and the reviewer's mutants

Tests now pin the environment check's main-failure restriction, its yes/no share criterion and its per-record GPU match, the thinking arm's unforced subset, per-batch seeds and record-and-arm order, the revision check on a mixed set, and B3's resample count. The reviewer's eleven mutants, adapted to the amended code, are in `scripts/mutation_check.py` with the round-3 set.

### 14.N Other corrections

- Qwen3.5-9B has 9.41 B parameters in its architecture (8.95 B in the language model; the checkpoint stores 9.65 B tensors with its multi-token-prediction heads); `PAIR_CHANGES` said 9.7 B.

### 14.C Cost, amended

| Stage | Runs | GPU-hours (estimates) |
|---|---|---|
| Minimum | environment check, three models, mirroring round 2's shards (about 0.9 in all, a few short jobs per model when the shards are mixed), core smoke runs (0.3), `qwen3.5-9b` (2–4), `ministral3-8b-it` (1.7–2.5), `gemma4-12b-it` (2.5–3.5), downloads (1–2.5) | about 9.5–13.5; budget 15 |
| Extension | as section 11 | about 15–23, 12–18 on 80 GB cards |
| Thinking arm | smoke (two batches, 0.1) and 4,530 generations with scoring | about 9–22 |
| All | | about 34–59; budget 60 |
| If the environment check does not pass | full serve2 rescoring of the three older models on L40S | about 7 more |

### 14.T Checks after the review-1 fixes (2026-11-23; nothing scored, no round-2 result read)

- **Frozen code.** Round 2's files and wave B's files are byte-identical to their tags; `wf/constants.py` only adds the two round-3 pins.
- **The round-3 gate on every record and arm, all twelve tags,** on ICE in a fresh checkout (`text-wm-fix46` at d29731a, from a git bundle; CPU job 7248427, serve2, offline): every tag 32,942 candidate checks, 9,272 yes/no prompts and 17,057 arm pairs isolated, **0 problems**; longest inputs as in section 13. The golden renderings remade from this job are byte-identical to the committed `r3_chat_golden.json`.
- **Tests.** `tests/test_textcraft_r3.py`: 78 tests. Full local suite at d29731a, once at the end: **1,234 passed, 29 skipped**.
- **Mutants.** Strict named-kill rule, official run at d29731a (`notes/textcraft_r3_mutation_log.txt`): **109 of 109 killed, by 57 different tests**: the 67 earlier round-3 mutants (9 rewritten for the amended code), the reviewer's 11 (adapted; 8 had survived the whole suite), and 31 for the fixes. The screen run's single survivor got its test before the official run.
- **Not run:** no GPU job, no smoke run, no environment-check job, no plan file (it is written from round 2's recorded GPU names once round 2's scoring has filed them, then committed before any environment-check job).

### 14.R2 Amendments after review 2 (2026-11-23, before the freeze and before any round-3 score)

Review 2 ([textcraft_r3_review2.md](textcraft_r3_review2.md), commit 3393d0b, merged) found every review-1 item fixed, one must-fix condition and seven should-fix items. The coordinator decided them; this section governs where earlier text differs.

- **R2-M1 (decided, then simplified).** The freeze waits for the environment-check plan file, committed before the freeze tag, with no exemption from 14.S12 (14.M4, item 1). Wave B adopts no amendment to its per-score criterion, so round 3 waits for nothing from wave B: its rule is its own constant (item 5).
- **S1.** A provenance problem of a pair's newer model (the Gemma slot's model included) refuses B3 as well as its own family (`r3_analysis.newer_model_refusals`).
- **S2.** A job refuses to start from a checkout with uncommitted changes to tracked files (`r3_gates.require_freeze_commit`, through `wf.util.git_dirty`), and the analysis refuses any shard whose `inputs.json` records `git_dirty: true`.
- **S3.** A model not scored reads "not scored" in H3-P's per-model column.
- **S4.** `gpu_unobtainable.json` must name the first submission's Slurm job id (`first_submission_job_id`) and date (`first_submission_date`), and the analysis checks that at least 7 days have passed since that date (`r3_gates.unobtainable_check`); a declaration that fails this is a provenance problem and does not make the check "not run".
- **S5.** A filed gate failure is final: a run into a shard folder that holds a gate record with `ok: false` refuses to start (`r3_gates.refuse_after_gate_failure`), so the record is never overwritten.
- **S6.** An extension model may be declared not run for GPU availability by a dated file, `runs/textcraft_r3_score_<tag>/not_run.json` (`date`, ISO 8601, and `reason`). It counts as not scored (its tests at p = 1, A3x proceeds) only if it is an extension model and is dated before the analysis first read any core result. That moment is recorded once, by the analysis, in `runs/textcraft_r3_first_core_reading.json`, and is never overwritten. A late or core-model declaration is a provenance problem and refuses A3x (`r3_gates.not_run_declaration`).
- **S7.** A test pins `pace/` (and `wf/`, `scripts/`) among the protected code paths of 14.S12.
- **The rule and its pin.** Round 3's rule is the constant `r3.ENVCHECK_RULE` (wave B's original criteria, unamended); `python -m wf.textcraft.r3_gates pin-rule` writes it with its sha256 to `wf/config/textcraft_r3_envcheck_rule.json`, and the analysis refuses to run if the constant and the pin differ (`r3_gates.rule_problems`). The live reading of `families.ENVCHECK_RULE` and the wave B hook interface of the first draft of this item are withdrawn (coordinator's decision after wave B's diagnostic).

### 14.C2 Cost, amended after review 2

The full serve2 rescoring of the three older models (about 7 GPU-hours on L40S) is now expected, not contingent (14.M4, item 5), and is counted in the total.

| Stage | GPU-hours (estimates) |
|---|---|
| Minimum (14.C) | about 9.5–13.5 |
| Serve2 rescoring of Qwen3-8B, Mistral-7B-v0.3 and Gemma-2-9B-it (expected) | about 7 |
| Extension | about 15–23, 12–18 on 80 GB cards |
| Thinking arm | about 9–22 |
| All | **about 41–66** |

The previous budget of 60 GPU-hours covers the lower and middle of this range; at the upper end the study needs up to **6 GPU-hours more** (66 against 60). The budget is raised to 66. If the thinking arm runs near its upper estimate, it is the stage to stop at its next saved batch (it resumes per batch), so that the confirmatory stages are never short of time.

### 14.T2 Checks after the review-2 fixes (2026-11-23; nothing scored, no round-2 result read)

- **Tests.** `tests/test_textcraft_r3.py`: 87 tests. Full local suite at 64c1e8a, once at the end: **1,243 passed, 29 skipped**.
- **Mutants.** Strict named-kill rule, official run at 64c1e8a (`notes/textcraft_r3_mutation_log.txt`): **135 of 135 killed, by 65 different tests**: the 109 of 14.T, review 2's twelve (rv2-4, which survived the reviewer's run, now killed by `test_freeze_commit_and_job_start_check`) and 14 for the review-2 fixes (two written for the withdrawn wave B hook were replaced by "rule read from wave B" and "rule constant loosened").
- **Not run:** no GPU job, no environment-check job, no plan file, no tag.

### 14.P The environment-check plan (2026-11-23, before the freeze, no score read)

`wf/config/textcraft_r3_envcheck_plan.json`, written by `python -m wf.textcraft.r3_gates plan` (14.M4, item 1): for Qwen3-8B, Mistral-7B-v0.3 and Gemma-2-9B-it, one job on `NVIDIA L40S` covering round-2 shards 0 to 3, 300 of 300 seeded records comparable. Nothing under `wf/` changed except this file, so the ICE CPU gate of 14.T was not rerun. Full local suite with the plan file in place: **1,243 passed, 29 skipped**.


## 15. Deviations after the freeze

Each deviation has its own dated note. Where they differ from sections 1 to 14, the notes govern. Every hypothesis,
family, test, threshold, Holm size, margin and reading rule is unchanged.

- **D-T1** (2026-11-23), [textcraft_r3_deviation_thinking_padding.md](textcraft_r3_deviation_thinking_padding.md):
  the thinking arm's padded-against-alone start check runs in float32.
- **D-G1** (2026-11-23), [textcraft_r3_deviation_gemma_e4b.md](textcraft_r3_deviation_gemma_e4b.md): an exploratory
  Gemma 4 E4B run, outside every family.
- **D-T2** (2026-11-25), [textcraft_r3_deviation_analysis_config.md](textcraft_r3_deviation_analysis_config.md): the
  analysis config takes the Gemma slot and B3's older scores from the gates.
- **D-G2** (2026-11-25), [textcraft_r3_deviation_attention_float32.md](textcraft_r3_deviation_attention_float32.md):
  the attention-agreement criterion of section 4 (and 14.M3) is evaluated in float32, by
  `textcraft_r3_attnf32_<tag>`, with the same function, items, arms and limits, for every Gemma 4 model. Gemma 4 12B's
  smoke is rerun (`textcraft_r3_smoke2_gemma4-12b-it`) so that the fallback rule's format diagnostic has rows. The
  Gemma slot then follows the frozen rule: 12B unless a trigger fires, else E4B. The analysis that is read is
  `textcraft_r3_analysis_dg2`, at D-G2's tag. D-G2 was decided before the round-3 analysis ran and before any round-3
  score was read.
