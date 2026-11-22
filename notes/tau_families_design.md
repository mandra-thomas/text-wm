# τ-bench retail, wave-A model families: design (written 2026-11-22, before any score of these models)

This note fixes the design of an extension of the τ-bench retail replication (`notes/tau_design.md`, results in
`notes/tau_results.md`) to the instruct models of the model-families study's wave A (`notes/families_design.md`,
`notes/families_results.md` on `main`). It is committed before any code of the extension is written and before any
of these models is scored on τ-bench items. Nothing is scored until an independent review (engineering and
scientific validity, validation standard steps 9 and 9b) passes and the design is frozen at an annotated git tag.
Anything changed after this commit goes in a dated "Changes" section at the end, with its reason.

What existed before this note, and was looked at: the τ-bench item build and its pins, the τ-bench design and
results for Qwen3-8B and Llama-3.1-8B, the ALFWorld wave-A results, the three models' configs and chat templates at
their pinned revisions, and one tokenizer-only measurement made for this note (section 2.2): prompt lengths and the
tokens each template writes at the start of a reply, on the 856 primary items. No model of this extension has been
run on a τ-bench item, and no forward pass of any kind has been made.

Every rule of `notes/tau_design.md` (as amended through review 4) holds unless this note changes it. The items, the
prompts, the few-shot examples and their counterbalanced order, the cues and the per-model decision rule are the
same.

## 1. Question

**The τ-bench finding.** In the τ-bench retail backend, a test item is a write call on a target order that fails a
precondition, and the deciding fact (the order's status, a variant's availability, the original payment method, a
gift card's balance) is printed in an earlier tool output in the history. The task line names the target order and
product (aligned), another order of the same user and its product (swapped), or nothing specific (neutral). The
primary readout is the first token of the predicted tool output: $L_1 = \log P(\texttt{\{"}) - \log P(\texttt{Error})$,
since every success reply begins with `{"` and every failure reply with `Error`. The goal effect is the mean over
the 856 status-matched primary items of $L_1$ under the aligned task minus $L_1$ under the swapped task.

**What it showed.** Qwen3-8B: +3.98 nats [3.61, 4.35], "replicates", read as a general task-consistency bias (the
working witness call on the same order moves as much). Llama-3.1-8B: +0.47 [0.41, 0.54], "small effect, positive",
specific to failing calls (fail minus witness above 0). The joint verdict was "ALFWorld direction in both models, size
met by one; not a replication".

**What this extension asks.** Does the τ-bench goal effect appear in instruct models of other families and of
Llama's next release, and do they look like Qwen3-8B (naming the order raises the success belief of any call on it)
or like Llama-3.1-8B (the shift is specific to calls the record shows will fail)? Two readings of the two reference
models are open, and the extension can speak to them: the split could follow the developer (Qwen against Meta), or
it could be a property of individual models with no family pattern. One model of another developer (Mistral) and one
of Meta's next release (Llama 3.2) can describe this; they cannot separate family, size, tokenizer, chat format and
post-training, so every cross-model statement is descriptive (section 7.5).

## 2. Models and template handling

### 2.1 The three wave-A models considered

| Tag | Hugging Face id | Pinned revision | Trained context | Template handling (as wave A) | Attention | In this extension |
|---|---|---|---|---|---|---|
| `mistral-7b-v0.3` | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41d | 32,768 | native: the template writes the system text into the last user turn | SDPA (default) | scored |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b | 131,072 (RoPE scaled from 8,192) | native system header; `date_string` fixed at "26 Jul 2024" | SDPA (default) | scored |
| `gemma2-9b-it` | google/gemma-2-9b-it | 11c9b309abf7 | 8,192 (no RoPE scaling) | system text folded into the first user turn | eager (soft-capping) | **excluded** (2.3) |

Llama-3.2-1B-Instruct is not considered: it fails the families competence rule in ALFWorld (neutral yes/no AUC
0.68). Revisions are the full hashes in `wf.constants.REVISIONS`; the weights of all three are in the ICE cache
(`/storage/ice1/2/1/gmat3/hfcache`, the `models--<org>--<name>` folders that `TRANSFORMERS_CACHE` points at) at exactly
these snapshots, checked on 2026-11-22. No download is needed.

**Mistral 7B v0.3, native placement.** As in wave A, the template's own placement is kept: the system text (the
instruction, the backend's rules, the retail policy and the tool schemas, about 5,750 Mistral tokens) is written into
the last user turn, `[INST] <system text>\n\n<query turn>[/INST]`, so here it sits after the four few-shot examples
and immediately before the query's task line, where every other model has it at the start. The template writes a
reply after a space and trims it, so the scored tokens are the ones it writes: `▁{"` and `▁Error` for the first-token
readout and `▁Yes` and `▁No` for the questions (2.2). In ALFWorld, folding Mistral's system text into the first turn
raised its yes/no effect by 0.30 nats (12% of the effect, below the one-fifth rule). Here the system text is about
half of the prompt and lands between the examples and the task line, so placement could matter more; a small
placement check is part of the protocol (section 7.6).

**Llama 3.2 3B.** The Llama 3 header format, with the date fixed so the prompt cannot change with the day a shard
runs; its prompts are byte-identical to Llama-3.1-8B's apart from the model.

### 2.2 Tokenizer-only measurement made for this note

On ICE (CPU job, transformers 4.51.3, each pinned tokenizer offline, the items rebuilt at the base commit 5741e81
with the pinned hashes reproduced and `tau_verify` passing), every primary item's prompt was rendered through the
scorer's own encoding (`LogoddsScorer._encode`, with each model's template settings and Gemma's fold), under the
aligned and swapped task lines, for the failing call's output cue, yes/no question and manipulation question and the
witness's output cue (1,712 prompts per variant and model):

| Model | Prompt tokens, median | 90th percentile | Maximum (any variant) | Share above 8,192 |
|---|---|---|---|---|
| Mistral 7B v0.3 | 11,870 | 14,032 | 14,642 | 100% |
| Llama 3.2 3B | 9,767 | 11,414 | 11,907 | 100% |
| Gemma 2 9B | 11,348 | 13,456 | 14,051 | 100% |

For comparison, the reference runs had Qwen3 median 10,696 (maximum 15,250) and Llama-3.1 median 9,675 (maximum
13,284). Under Gemma's tokenizer the part of the prompt that is the same for every item, the system message and the
four few-shot examples with the template's markup, is 9,575 tokens (the system message alone 5,531, or 4,051 without
the agent policy; the examples about 4,000). With the policy removed (the τ design's rules-only system message) that
part is still about 8,100 tokens, Gemma's prompts are 9,868 tokens at the median, and every one is above 8,192. The
same fixed part is 10,007 tokens for Mistral and 8,411 for Llama 3.2.

Reply starts, read on every tenth primary item with the template's own rendering of the reply after the prompt, and
the scorer's tokenization of each scored word alone:

- **Llama 3.2 3B and Gemma 2:** `{"`, `Error`, `Yes` and `No` are single tokens, and the template's replies begin with
  exactly them (success twin and witness output with `{"`, failure reply and witness failure with `Error`).
- **Mistral:** the words tokenize alone as `▁{"`, `▁Error`, `▁Yes` and `▁No`, single tokens, and the template's
  replies (written after `[/INST]` and a space) begin with exactly these.

So the τ readouts, `{"` against `Error` and `Yes` against `No`, carry over unchanged in text; what is checked per
model is that the token the scorer reads is the token the template writes. This replaces, for these models, the τ
design's token check (A10: "single tokens without a leading space", "generation prompt ends in a newline"), which
was written for Qwen3 and Llama 3.1; Mistral's template has no generation prompt and writes a space before a reply,
by design. The full check on every scored prompt is a gate (section 6).

### 2.3 Gemma 2 9B is excluded (decided now, before any score)

**Decision.** Gemma 2 9B is not scored in this extension. It is reported as "not run: the τ-bench prompt does not fit
its 8,192-token context", never as a null result or as evidence about goal sensitivity.

**Why it does not fit.** The fixed part of every prompt, the system message and the four few-shot examples, is 9,575
Gemma tokens before any item's history, so no item fits; the shortest primary prompt is longer than the window under
every variant measured, and without the agent policy the fixed part alone would leave about 100 tokens for an item. Gemma 2 was trained with 8,192 positions, rotary base 10,000 and no position scaling.
Its local layers attend over a 4,096-token sliding window, but its global layers would have to relate the task line
and the deciding record across relative distances it never saw in training. A result would mix goal sensitivity with
out-of-range position behaviour, and the sliding window does not remove that: the global layers are the ones that
carry information across the prompt.

**Why not a variant that fits.** Three variants were considered and rejected.

- **Truncating the history.** Not admissible: the history holds the deciding record, which is what makes each failure
  determined, and the fixed part alone already overflows.
- **A compact prompt** (no agent policy, no tool schemas, fewer or no examples). It would fit, but it is a different
  prompt and arguably a different task. The τ result depends on what the prompt says about the backend (the reference
  models' effect was larger without the statement that the backend does not see the task), so a Gemma result on a
  compact prompt could not be compared with the four models' results on the main prompt, and reading it under the
  same verdict rule would need every other model scored on the compact prompt too, an arm as large as this extension.
- **Ignoring the limit.** Running Gemma at 11–14k tokens and reporting the numbers with a caveat would put a number
  that is not interpretable next to numbers that are.

**Cost, for the record.** Had it fit, Gemma with eager attention (needed for its attention-logit soft-capping) would
take about 3.4 s a pass at its prompt lengths, about 11 GPU-hours for the protocol below, twice the budget.

**Consequence.** The extension has one model from a developer other than Qwen's and Meta's (Mistral). The planned
later wave of the families study (Gemma 4, with a longer context) can add Gemma under its own amendment.

## 3. Items, prompts and population: unchanged

- **Items.** The τ-bench build at its pins: items `2c8ffeac…`, natural items `9bd6c3f7…`, few-shot examples `231607f7…`
  (`wf/config/tau_score.yaml`). A rebuild on ICE at this branch's base commit reproduced all three hashes and
  `tau_verify` passed on 3,391 items (2026-11-22). Scoring refuses files that are not verified or not pinned (τ design
  §8.8, A11, B10).
- **Population.** The 856 primary items (status-matched main population, 175 users): `status` 409, `unavailable` 405,
  `wrong_return_pm` 36, `gc_balance` 6. Nothing outside the primary population is scored (section 4).
- **Prompts.** The main arm only: the system message whose rendered sha256 is pinned (`7cc04248…`), the four few-shot
  examples in the design order or reversed by item, the output cue, the reply-anchored yes/no question (C1) and the
  manipulation question, all exactly as for the reference models. The only differences between models are the ones
  their chat templates impose (2.1).

## 4. The cost-reduced protocol

### 4.1 What is scored, per primary item and model

| Call | Readout | Task lines | Passes | What it is for |
|---|---|---|---|---|
| failing call | first token | aligned, swapped, neutral | 3 | the confirmatory primary contrast; aligned and swapped each against neutral |
| failing call | reply-anchored yes/no | aligned, swapped, neutral | 3 | secondary readout; the families study's format |
| failing call | first token | mention-aligned, mention-swapped | 2 | the mention pair: write minus mention in the reading table |
| failing call | manipulation question | aligned, swapped | 2 | the manipulation check that gates a null |
| witness (the working call that achieves the aligned task) | first token | aligned, swapped, neutral | 3 | discrimination (gates a null and a reversal), the witness's own effect, fail minus witness |
| witness | reply-anchored yes/no | neutral | 1 | the families competence rule on the yes/no format |

That is 14 forward passes per item, **11,984 passes per model**. Every one is a single forward pass that reads one
next-token distribution (`score_one_token_pair`), with the same scorer, precision and log-softmax as the reference
runs.

### 4.2 What is dropped, and why each cut is safe

- **Full-reply readout** (12 passes per item for the failing call and the witness, each about twice the time of a
  single-token pass; about 60% of the reference runs' GPU time per item). It was secondary, never a verdict input,
  and its interpreted form (per-token) is dominated by long JSON twins. Its by-product, the check that the first-token
  pass agrees with the full pass's first token, was a tokenization-boundary check; here every scored token is checked
  against the template's own reply on every item without a GPU (section 6).
- **The rules-only and ALFWorld-matched arms** (500 items each, 6 passes, plus 3 for the matched arm's witness: 7,500
  passes per model). Both were exploratory, with no confirmatory role. They asked how the prompt's statements about
  the backend move the effect, which the reference models answered (the effect remained under both prompts and was
  larger without the statement). Repeating them for new models answers a different question from the one this
  extension asks.
- **The main pool beyond the primary population** (529 items) **and the natural set** (28 items). The full pool was
  secondary and its extra stratum (swapped-task achievability) is confounded with status matching and never read
  separately (A3); the natural set was too small for inference even for the reference models.
- **The mention pair on the yes/no question** (2 passes). The reading table reads write minus mention on the first
  token only (`checks_of`); the yes/no mention contrasts were descriptive.
- **The witness's yes/no under the aligned and swapped task lines** (2 passes). Descriptive in the reference design;
  only the neutral one is needed, for the competence rule.
- **The contamination check.** Exploratory. Both models' training data predate τ-bench's release (June 2024): Mistral
  7B v0.3 was released in May 2024, and Llama 3.2's stated knowledge cutoff is December 2023. tau2's database extends
  τ-bench's, so this is approximate.

Every pre-registered input of the per-model verdict and of the reading table (B3, C2) is kept: the primary contrast,
the manipulation check, discrimination, the witness's aligned minus swapped, fail minus witness and write minus
mention, all on the first token. So the per-model verdict and its reading are computed exactly as for the reference
models.

### 4.3 The Mistral placement check (exploratory, 600 passes)

300 primary items, drawn with a fixed seed (`random.Random("20260924|tau_family_placement").sample` of the primary
items in id order), the failing call's first token under the aligned and swapped task lines, with Mistral's system
text folded into the first user turn (Gemma's handling in wave A) instead of the template's last-turn placement.
Scores are saved under the tag `mistral-7b-v0.3-firstturn`, so they cannot mix with the native run. Reading rule in
section 7.6.

### 4.4 GPU cost

Seconds per pass are estimated from the reference smoke run (Qwen3-8B 1.50 s at a median of 10,696 tokens, Llama-3.1-8B
1.20 s at 9,675; τ design, smoke record) by compute per token and prompt length:

- **Mistral 7B v0.3.** Per token it does nearly the work of Llama-3.1-8B (32 layers, width 4,096, MLP 14,336; a smaller
  output layer), and its prompts are 23% longer: about 1.5 s a pass.
- **Llama 3.2 3B.** About 0.43 of Llama-3.1-8B's compute per token at the same length; allowing for small models'
  lower efficiency, about 0.65 s a pass.

| Run | Mistral 7B v0.3 | Llama 3.2 3B |
|---|---|---|
| protocol (11,984 passes) | 5.0 h | 2.2 h |
| smoke (21 items, 294 passes) | 0.1 h | 0.05 h |
| placement check (600 passes) | 0.25 h | none |
| **total per model** | **about 5.4 GPU-hours** | **about 2.2 GPU-hours** |

About 7.6 GPU-hours for the extension, against about 38 for the reference models' full run; the protocol itself is
about a quarter of what the reference protocol would cost for the same model. These are estimates; the smoke run
measures the seconds per pass by readout and the projection decides (section 6).

**Jobs and GPUs.** One model per job, 4 shards per model by user (users split as in `tau_score`), a 3-hour limit per
job, any L40S, A100 or H100 (A40s are left out: about half the speed in the reference sizing, which would put a
Mistral shard near the limit). Every score is saved as it finishes; a shard cut off resumes, on the same GPU type
(`--constraint`), as in wave A.

**Memory.** Weights in bf16: Mistral about 14.5 GB, Llama 3.2 3B about 6.4 GB. The scorer computes logits at every
position of the prompt: at the longest prompts about 1 GB for Mistral (32,768-entry vocabulary, 14.6k positions) and
3 GB for Llama 3.2 (128,256 entries, 11.9k positions), plus the returned key-value cache (about 2 GB and 1.3 GB).
SDPA attention does not materialise the attention matrix. Both fit comfortably on any of the allowed cards (40 to 80
GB), and on an A40 or L40S (48 GB). Gemma, had it been run, would have needed eager attention's full attention
matrices (about 12.5 GB transient per layer at 14k tokens in float32).

## 5. Readout definitions

For each scored job the record holds $\log P(t_+)$ and $\log P(t_-)$ at the first answer position, with
$L = \log P(t_+) - \log P(t_-)$, where for the first-token readout $t_+$ and $t_-$ are the scorer's tokens for `{"` and
`Error` (as the template writes a reply: section 2.2) and for the two questions those for `Yes` and `No`. Each record
also stores the prompt's length in tokens.

## 6. Gates before and during scoring

1. **Items verified and pinned** (unchanged τ gate).
2. **Per-model token check on every scored prompt** (run `tau_family_token_check`, CPU, offline tokenizers at the
   pinned revisions, through the scorer's own encoding and each model's template settings). For each scored model it
   must pass all of:
   - `{"`, `Error`, `Yes` and `No`, each tokenized alone as the scorer reads it, is one token, and the two tokens of
     each readout differ;
   - on every primary item, the template's own rendering of the success twin and the witness output as the reply
     (after the item's output-cue prompt for that call) begins with the `{"` token, and of the failure reply and the
     witness failure begins with the `Error` token; each whole reply tokenized alone equals the template's tokens for
     it (no merge at the prompt/answer boundary);
   - after every yes/no and manipulation prompt that will be scored (6 per item), `Yes` and `No` as the template writes
     them as the reply equal the scored single tokens;
   - for every call and cue, the rendered prompts under the different task lines differ only in the task line;
   - every scored prompt is shorter than the model's trained context (Mistral 32,768, Llama 3.2 131,072).

   The check also covers the placement check's 600 prompts for Mistral, and records Gemma's prompt lengths for the
   exclusion (length only). Scoring refuses a model whose check has not passed at this commit (or at a commit the
   scoring config names) on exactly the pinned item files.
3. **In the scoring job.** Before the first score of a model, the same readout check on the shard's first item with
   the loaded tokenizer; and for every job, the prompt's length in tokens is computed and the job refuses a prompt at
   or above the model's context. Gemma is refused by name.
4. **Smoke run** (`tau_family_score_smoke`, both models, 20 hash-chosen primary items plus the longest primary item,
   every job type; then `tau_family_smoke_analysis`). Read for format, timing and precision only, never for effects:
   the median two-way masses of each readout (below 0.5 is a red flag), the token reports, mean seconds per job by
   readout, the projected hours of the largest shard of each model against the job limit and the projected total
   against the 6-hour budget, and the standard deviation of the per-item aligned minus swapped differences with the
   projected 95% half-widths (no mean, no estimate, no verdict). The projection is copied into this note and
   committed before full scoring.

## 7. Analysis and pre-registered rules

### 7.1 Per-model verdict: the τ design's rule, unchanged

For each scored model, on the primary population, $\overline{\Delta L_1}$ is the mean over items of $L_1$ under the
aligned task minus $L_1$ under the swapped task, with a 95% user-clustered percentile bootstrap (2,000 resamples) and
a two-sided user-level sign-flip permutation p (1,000 permutations), seed 20260924. With $p_{\text{Holm}}$ the
Holm-adjusted p of section 7.2, the verdict is "replicates (ALFWorld direction)" when

$$\overline{\Delta L_1} \ge 0.5 \text{ nats}, \quad \text{CI excludes } 0, \quad p_{\text{Holm}} < 0.01,$$

"reversed" at −0.5 or below under the same conditions, "small effect" (with its direction) when only the last two
hold, "no effect larger than 0.5 nats" when the CI lies inside (−0.5, 0.5), and "inconclusive" otherwise. The τ gates
apply as before: a null needs the manipulation check (task line read: aligned minus swapped on the manipulation
question at least 1 nat with the CI above 0) and discrimination (on the items whose witness the text-only checker
determines, the witness's neutral $L_1$ minus the failing call's, at least 1 nat with the CI above 0); a reversal
needs discrimination. Every verdict is read through the outcome table of B3 with C2 (witness effect, fail minus
witness, write minus mention), by the same code (`wf.tau.pipeline.verdicts` and `outcome_reading`). The
sensitivity analysis without the few-shot tool classes (B6) and the product-clustered analysis are computed as before.

**Third gate: competence (section 7.3).** A model that fails the families competence rule on the first-token readout
gets the verdict "not interpretable (fails the families competence rule)", whatever its estimate, as in the families
study; its estimate and interval are still reported.

**No joint verdict.** Each model's verdict stands alone. The reference design's joint rule asked whether two
co-primary models agreed; here the question is descriptive (section 7.5), and no statement of the form "replicates
across models" is made from this extension.

### 7.2 Multiplicity

The confirmatory family is the per-model primary test of each scored model: two tests (Mistral 7B v0.3, Llama 3.2 3B),
Holm-adjusted together at a fixed size of two. A model whose score is missing enters with p = 1, so the other is
never corrected for a smaller family. Gemma, excluded before data, is not in the family. The reference models'
verdicts were decided in their own family (Holm over two) and are not re-tested. Everything else is descriptive,
with an interval and, where stated, a p that belongs to no family: the yes/no readout, aligned and swapped against
neutral, the per-class results (`status` and `unavailable`), the product-clustered and few-shot-tool sensitivity
analyses, the item shares, the competence numbers, every cross-model comparison and the placement check.

### 7.3 Competence (the families rule, on τ-bench's discrimination population)

The discrimination population $D$ is the primary items whose witness the text-only checker determines (PASS): 447
items (`unavailable` 405, `wrong_return_pm` 36, `gc_balance` 6; the `status` witnesses need a variant's availability
that their histories do not show). On $D$, under the neutral task, the witnesses are the determined successes and
the failing calls the determined failures. For a readout with log-odds $L$, write $s_i$ for item $i$'s witness score
and $f_i$ for its failing call's. The rule is the families study's four conditions:

$$\mathrm{AUC} = \frac{1}{|D|^2} \sum_{i \in D} \sum_{j \in D} \left( \mathbf{1}[s_i > f_j] + \tfrac{1}{2}\,\mathbf{1}[s_i = f_j] \right) \ge 0.70,$$

the share of failing calls with $f_j > 0$ (the model's argmax is the success reply, or Yes) at most 0.95, the share of
witnesses with $s_i > 0$ at least 0.05, and the median two-way mass $P(t_+) + P(t_-)$ over the $2|D|$ neutral records
at least 0.5.

It is computed for both readouts:

- **First token** (`{"` as the success answer). This is the gate of section 7.1: it is the readout the verdict is
  about, and a model that cannot tell working from failing calls on it, or answers one way to everything, says
  nothing about goal sensitivity.
- **Yes/no** (the families study's format, exactly). Reported; it gates only the yes/no readings (a model that fails it
  has its yes/no results labelled "not interpretable on the yes/no readout").

The τ discrimination check of section 7.1 stays as it was; the two answer different questions (a mean separation of
at least 1 nat, against a rank-based separation and two coverage bounds). Both rules are also reported for the two
reference models, descriptively: their published verdicts stand, and a failure would be disclosed beside them, not
turned into a new verdict (as was done for Llama-3.1-8B in TextCraft).

### 7.4 Secondary estimates (descriptive)

Per model, on the primary population, each with a user-clustered 95% interval and sign-flip p (no family):

- the yes/no $\Delta L$, and aligned minus neutral and swapped minus neutral for both readouts;
- the practical size: the mean over items of $\sigma(L_1)$ under aligned minus under swapped, with
  $\sigma(L_1) = P(\texttt{\{"}) / (P(\texttt{\{"}) + P(\texttt{Error}))$;
- the flip shares, toward success, toward failure and either;
- the **item share** of each readout, the families study's scale-free measure: with $F$ the primary items,

$$\text{share} = \frac{1}{|F|} \sum_{i \in F} \mathbf{1}\left[L(i, \text{aligned}) > L(i, \text{swapped})\right],$$

  null one half, ties counted as not moved;
- per class (`status`, `unavailable`), and the descriptive classes with estimate and n only;
- the controls: witness effects, discrimination (first token and yes/no, and per class), the mention pair, the
  manipulation check, fail minus witness and write minus mention;
- format: median two-way masses and seconds per job by readout; prompt lengths in tokens.

### 7.5 Comparisons across models (descriptive, rules fixed now)

The reference blocks (Qwen3-8B, Llama-3.1-8B) are recomputed by the same code from the reference scoring run
(`tau_score`, scored at `c737cd6`, linked into this checkout's `runs/` before the analysis), restricted to exactly the
jobs of section 4.1. The analysis checks that the recomputed primary estimates and intervals equal the published ones
in `notes/tau_metrics.json` (a results check, validation step 12) and that every reference record's item hash equals
the current item's.

Comparisons are made for each scored model against each reference model, and between the two scored models, on the
items both have, with a paired user-clustered bootstrap (users resampled once, both models read on the same
resample). For each comparison of models A and B:

- **Item share on the first token.** With $m_A(i) = 1$ when A's $L_1$ is higher under aligned than under swapped and 0
  otherwise, the difference is the mean of $m_A(i) - m_B(i)$. It "differs" when its 95% paired interval excludes 0;
  it "matches" when its 90% paired interval lies inside ±0.05 (the families margin); both: "differs, within the
  margin"; neither: "no difference detected", never "the same".
- **$\Delta L_1$ difference** in nats, with its 95% paired interval: "differs" when the interval excludes 0 and the
  difference is at least 0.3 nats in size (the families noise floor; a smaller difference is "no difference
  detected"). Log-odds scales differ across models (Qwen3-8B's τ effect is 8 times Llama-3.1-8B's in nats), so this is
  read beside the item share, never alone.
- **The same two numbers on the yes/no readout**, for models interpretable on it.
- **The reading row** of each model (section 7.1), set side by side.

A model that is not interpretable (fails the competence rule) or whose verdict is gated to "inconclusive" is left
out of every cross-model reading, with the reason.

How the patterns will be read (all descriptive):

| Pattern | Reading |
|---|---|
| Both scored models' first-token effect has a CI above 0 | The direction of the τ goal effect is not specific to Qwen3-8B and Llama-3.1-8B: it appears in every interpretable model tested on τ-bench |
| A scored model, interpretable, with a CI inside ±0.5 and its gates passed | No leak of more than 0.5 nats in that model on τ-bench (with the C2 witness qualification) |
| Both scored models have the task-consistency reading (fail minus witness not above 0), as Qwen3-8B | The task-consistency pattern is not specific to Qwen3; the failure-specific pattern is the exception among the models tested |
| Both scored models have the failure-specific reading, as Llama-3.1-8B | The task-consistency pattern is specific to Qwen3-8B among the models tested |
| Mistral and Llama 3.2 3B differ in reading row | The pattern varies by model; with one model per developer outside Meta, no family or developer explanation is drawn |
| Llama 3.2 3B has Llama-3.1-8B's reading row and their item shares "match" | Descriptive similarity within the Llama line; size, distillation and post-training all differ, so no cause is named |
| A comparison "no difference detected" or a $\Delta L_1$ difference under 0.3 nats | No difference detected, never "the same" |
| Mistral's placement check moves its effect (7.6) | Mistral's position in every comparison is reported as format-dependent, with both numbers |

### 7.6 The Mistral placement check: reading rule

On the 300 items: the goal effect (aligned minus swapped, first token) under the native and the first-turn
placement, each with a user-clustered 95% interval, and the per-item difference (first turn minus native) with its
own interval. If that interval excludes 0 and the difference exceeds one fifth of the native effect on the same items
in absolute size, Mistral's τ result is reported as depending on prompt format, with both numbers; otherwise placement
is reported as not shown to move the effect on this sample. It changes no verdict.

### 7.7 What each outcome would mean, per model (written before data)

| Outcome for a scored model | Meaning |
|---|---|
| fails the competence rule on the first token | Not interpretable on τ-bench: it does not tell working calls from failing ones, or answers one way to everything, on these prompts; no statement about goal sensitivity |
| "replicates" or "small effect, positive", gates passed, fail minus witness above 0 | The task overrides a visible fact about the named object, more for failing calls than working ones (Llama-3.1-8B's pattern) |
| the same, fail minus witness about 0 or negative | Goal leakage as a general task-consistency bias or an authorization belief (Qwen3-8B's pattern) |
| a positive effect with discrimination failing | The task shifts predictions, but the model is not shown to use the record |
| "reversed", discrimination holding | Naming the order sharpens failure prediction in that model; reported as a counter-example to the direction found in every model so far |
| null with both gates | No leak of more than 0.5 nats (or, if the witness moves, the C2 reading) |
| null with a gate failing, or "inconclusive" | Inconclusive for that model; not a null |

The extension can show that the direction generalizes (or does not) across two more models, and describe whether
the reference models' two patterns recur. It cannot show why a model has one pattern or the other.

## 8. Predictions (logged before any score of these models)

The predictions use the ALFWorld wave-A first-token effects (the same readout family: the first token of the
two-candidate score) and the two reference models' ratio of τ-bench to ALFWorld first-token effect. In ALFWorld the
first-token $\Delta L$ was: Qwen3-8B 2.51 [2.27, 2.77], Llama-3.1-8B 0.68 [0.62, 0.76], Mistral 7B v0.3 1.25 [1.08, 1.41],
Llama 3.2 3B 0.69 [0.59, 0.77]. On τ-bench the reference models came out at 1.59 times (Qwen3-8B) and 0.69 times
(Llama-3.1-8B) their ALFWorld first-token effects, so the ratio itself varies by more than a factor of two between
models.

- **Mistral 7B v0.3.** Scaled by the two reference ratios, 0.86 to 2.0 nats. Predicted: a positive first-token effect
  with the CI above 0, and the verdict "replicates" (at least 0.5 nats). The reading row is not predicted: there is no
  reference model of its developer. Competence is the main risk: in ALFWorld Mistral answered Yes to 56% of failures
  under the neutral task (yes/no), far more than any other interpretable model, while its AUC was 0.94.
- **Llama 3.2 3B.** Scaled by Llama-3.1-8B's ratio (same developer, same tokenizer and prompt bytes), about 0.48 nats,
  at the threshold. Predicted: a positive first-token effect with the CI above 0 and an estimate below 0.5 nats
  ("small effect, positive"); the failure-specific reading of Llama-3.1-8B, weakly. Competence and the manipulation
  check are a real risk for a 3B model reading 10k-token JSON prompts.
- **Gemma 2 9B.** Excluded (2.3); no prediction.

The same, as logged forecasts (subjective probabilities, used by no rule):

| Model | CI above 0 | Verdict "replicates" | Failure-specific reading, given a positive effect | Passes competence (first token) and both τ gates |
|---|---|---|---|---|
| Mistral 7B v0.3 | 0.85 | 0.6 | 0.5 | 0.7 |
| Llama 3.2 3B | 0.7 | 0.35 | 0.55 | 0.6 |

Not predicted: the yes/no results (the reference models split there, Qwen3-8B +7.68 and Llama-3.1-8B −0.02), the
ordering of the models, and the placement check's direction.

## 9. Risks and caveats stated in advance

- **One model outside Qwen and Meta.** With Gemma excluded, any statement about families rests on Mistral alone, and
  Llama 3.2 is Meta's. Cross-model readings are descriptive.
- **Everything changes at once across models:** developer, data, size, tokenizer, chat format and post-training.
  Mistral's chat format also moves the whole system text (half the prompt) next to the task line; the placement check
  measures this on 300 items only.
- **Long prompts.** Mistral reads up to 14.6k tokens; long-context reading may differ between models, and a model that
  fails the manipulation check or discrimination is gated, not read.
- **Reduced protocol.** No full-reply readout and no exploratory arms for these models, so the reference models'
  matched-arm finding (the effect is larger without the statement that the backend does not see the task) is not
  tested for them. The natural (agent-made) failures are not scored.
- **One scoring run per model**; GPU numerics move scores slightly across GPU types (about 0.02 nats in $\Delta L$ on
  the archived ALFWorld runs); the reference models were scored on a mix of GPU types.
- **The `status` class has no discrimination evidence** (its witness is not determined by the history), as for the
  reference models.
- Every caveat of the τ design (A16 and its additions) applies.

## 10. Runs, code and order of work

| Run | Device | What it does |
|---|---|---|
| `tau_build`, `tau_verify` | CPU | unchanged; must reproduce the pins at the scoring commit |
| `tau_family_token_check` | CPU | the per-model token check of section 6, every scored prompt; Gemma's lengths |
| `tau_family_score_smoke` | GPU | 21 items, both models, every job type; format and timing |
| `tau_family_smoke_analysis` | CPU | format, timing, projection and precision; no estimate |
| `tau_family_score_mistral-7b-v0.3`, `tau_family_score_llama3.2-3b` | GPU | the protocol of 4.1, 4 shards by user each |
| `tau_family_placement_mistral-7b-v0.3` | GPU | the placement check of 4.3 |
| `tau_family_analysis` | CPU | verdicts, competence, readings, comparisons with the reference run |

Code: `wf/tau/families.py` (prompts with each model's template handling, jobs, gates, scoring, analysis), reusing
`wf/families.py` (model specs, the fold, `template_reply`) and `wf/tau/pipeline.py` (item gate, effects, verdict rule,
outcome table); configs `wf/config/tau_family_*.yaml`; job scripts `pace/tau_family_score.sbatch` and
`pace/tau_family_items.sbatch`; tests `tests/test_tau_families.py`; mutants in `scripts/mutation_check.py`.

Order: this note; code, tests, mutants and a dry run without models; the token check on ICE (results added below);
independent review (engineering, and scientific validity) with fix rounds until a round finds no must-fix; an
annotated tag freezing the design; the smoke run, recorded here; full scoring; the analysis; a results check.
