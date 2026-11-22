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

**What it showed** (corrected after review 1, R1-M1, from the exploratory diagnostics `notes/tau_dx.md` on `main`).
Qwen3-8B: +3.98 nats [3.61, 4.35], "replicates", with the pre-registered reading "a general task-consistency bias".
The working witness call on the same order does not move "as much": it moves +9.45 [8.94, 9.94], more than twice as
much, so failing minus witness is −5.47 [−5.87, −5.05]. Llama-3.1-8B: +0.47 [0.41, 0.54], "small effect, positive",
pre-registered as "specific to failing calls". That row rests on failing minus witness +0.09 [0.04, 0.14], about a
fifth of the effect: the witness moves +0.38, about 80% as much as the failing call; failing minus witness is positive
in 49% of items, +0.68 in one status cell and −0.27 in the unavailable class; and the pooled +0.47 averages +1.23 in the
status class and −0.33 in the unavailable class. So on closer reading both reference models shift working and failing
calls together, and Llama-3.1-8B's pooled effect depends on the class mix. The joint verdict was "ALFWorld direction in
both models, size met by one; not a replication".

**What this extension asks** (amended after review 1, R1-M1). Does the τ-bench goal effect appear in instruct models
of other families and of Llama's next release? Is the shift shared by working and failing calls, as in both reference
models on closer reading, or specific to failing calls by the rule fixed in section 7.5? And does the pooled effect
hold across the two main failure classes, or depend on their mix, as Llama-3.1-8B's did? One model of another developer (Mistral) and one
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
| witness | reply-anchored yes/no | aligned, swapped, neutral | 3 | the families competence rule on the yes/no format (neutral); the task judgement on working calls and yes/no fail minus witness (aligned, swapped; restored after review 1, R1-S4) |

That is 16 forward passes per item, **13,696 passes per model** (14 and 11,984 before review 1). Every one is a single forward pass that reads one
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
- *(Restored after review 1, R1-S4; the cut is withdrawn.)* ~~The witness's yes/no under the aligned and swapped task
  lines (2 passes).~~ They measure whether the yes/no answer on working calls carries a task judgement (Qwen3-8B
  answered No for 91% of working calls it confidently predicted to succeed under the swapped line), which is one of the
  cross-cutting findings, and they make yes/no fail minus witness identifiable.
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
| protocol (13,696 passes; R1-S4) | 5.7 h | 2.5 h |
| smoke (21 items, 336 passes) | 0.14 h | 0.06 h |
| placement check (600 passes) | 0.25 h | none |
| **total per model** | **about 6.1 GPU-hours** | **about 2.5 GPU-hours** |

(Before review 1: 5.4 and 2.2.) About 8.6 GPU-hours for the extension, against about 38 for the reference models' full run; the protocol itself is
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
   - after every yes/no and manipulation prompt that will be scored (8 per item since R1-S4), `Yes` and `No` as the template writes
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
   against the budget (6 hours as first written; flagged above 6.5 since R1-S4, informational), and the standard deviation of the per-item aligned minus swapped differences with the
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

Of the 447 items, 405 are `unavailable`, so the gate is in practice a rule on that class; a model whose effect lives in
the `status` class (as Llama-3.1-8B's does) is gated on the other class (review 1, note).

### 7.4 Secondary estimates (descriptive)

Per model, on the primary population, each with a user-clustered 95% interval and sign-flip p (no family):

- the yes/no $\Delta L$, and aligned minus neutral and swapped minus neutral for both readouts;
- the practical size: the mean over items of $\sigma(L_1)$ under aligned minus under swapped, with
  $\sigma(L_1) = P(\texttt{\{"}) / (P(\texttt{\{"}) + P(\texttt{Error}))$;
- the flip shares, toward success, toward failure and either;
- the **item share** of each readout, the families study's scale-free measure: with $F$ the primary items,

$$\text{share} = \frac{1}{|F|} \sum_{i \in F} \mathbf{1}\left[L(i, \text{aligned}) > L(i, \text{swapped})\right],$$

  null one half, ties counted as not moved, with the share of exact ties and the median item effect reported beside
  it (R1-S1);
- per class (`status`, `unavailable`), and the descriptive classes with estimate and n only;
- the controls: witness effects, discrimination (first token and yes/no, and per class), the mention pair, the
  manipulation check, fail minus witness and write minus mention;
- format: median two-way masses and seconds per job by readout; prompt lengths in tokens;
- (R1-S4) fail minus witness anchored on the neutral line, on both readouts: the failing call's aligned minus neutral
  minus the witness's, and the same for swapped minus neutral (an authorization belief lowers swapped below neutral for
  both calls); yes/no fail minus witness;
- (R1-S4) the yes/no task judgement, per call (failing and witness) and task line: among calls whose first token
  confidently predicts JSON ($L_1 \ge 3$, the threshold of `notes/tau_dx.md`), the share answered No on yes/no; and
  yes/no regressed on $L_1$ with one slope and one intercept per line, with user-clustered intervals for the intercept
  differences;
- (R1-M1) per class (`status`, `unavailable`): the failing call's effect, the witness's and fail minus witness.

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
- **The label** of each model by the rule below, set side by side, with its qualifiers (R1-M1).

A model that is not interpretable (fails the competence rule) or whose verdict is gated to "inconclusive" is left
out of every cross-model reading, with the reason; in the metrics file every comparison entry of such a model, and
every yes/no comparison of a model that fails the yes/no competence rule, has its readings replaced by "not read:
<why>" (the computed reading kept as `reading_unread`), so no unmarked "differs" or "matches" remains (R1-S6).

**"Matches" is expected to be unreachable (R1-S3, stated before data).** "Matches" needs the 90% paired interval of
the item-share difference inside ±0.05. With 856 items in 175 users the 90% half-width is about 0.04 without
clustering (two models with shares near 0.6 whose moves are uncorrelated: per-item variance about 0.48, standard error
about 0.024) and about 0.06 to 0.10 with the within-user correlation seen in ALFWorld wave A. So a small difference
will almost always read "no difference detected", and the Llama-line row below will very probably not be met whatever
the models do. The bar is not lowered.

**The cross-model label (R1-M1, fixed before data).** The τ outcome table and its reading stay exactly as they are
(they are the τ rule), but the table's row is not used as the cross-model label. Instead, for each model with a positive
verdict, discrimination holding and the competence rule passed:

- the row is named by its mechanical criterion, "fail minus witness above 0" or "fail minus witness not above 0";
- always reported beside it: the witness's effect as a share of the failing call's effect, and per pre-specified class
  (`status`, `unavailable`) the failing call's effect and fail minus witness, each with its interval;
- the label "failure-specific" may appear in cross-model text only if all three hold: fail minus witness has its 95%
  interval above 0; the witness's effect is less than half the failing call's; and fail minus witness is not below 0
  (its interval wholly below 0) in either class. Otherwise the model is described as "a shift shared by working and
  failing calls", with the numbers (and "working calls move more" when fail minus witness has its interval below 0).
  By this rule neither reference model is failure-specific (Llama-3.1-8B: witness at 80%, unavailable class below 0).
- **class mix:** when the two classes' failing-call effects have intervals on opposite sides of 0, "the pooled effect
  depends on the class mix; reported per class";
- **minority (R1-S1):** when the verdict is positive but the item share's 95% interval includes or lies below one half,
  "the mean is carried by a minority of items; reported as such, not as a general shift". The mean governs the verdict;
  the share qualifies the reading and governs statements about breadth, $\Delta L_1$ about size on each model's scale.

How the patterns will be read (all descriptive):

| Pattern | Reading |
|---|---|
| Both scored models' first-token effect has a CI above 0 | The direction of the τ goal effect is not specific to Qwen3-8B and Llama-3.1-8B: it appears in every interpretable model tested on τ-bench; qualified by name for a model carried by a minority of items or not shown to use the record (R1-S1) |
| A scored model, interpretable, with a CI inside ±0.5 and its gates passed | No leak of more than 0.5 nats in that model on τ-bench (with the C2 witness qualification) |
| Both scored models labelled "a shift shared by working and failing calls" (R1-M1) | In both scored models the task shifts working and failing calls alike (or working calls more), as in both reference models; no model tested shows a shift specific to failing calls by the pre-stated rule |
| Both scored models labelled "failure-specific" (R1-M1) | Both scored models show a shift specific to failing calls by the pre-stated rule, which neither reference model meets |
| Mistral and Llama 3.2 3B differ in label | The pattern varies by model; with one model per developer outside Meta, no family or developer explanation is drawn |
| A scored model's class effects have intervals on opposite sides of 0 (R1-M1) | The pooled effect depends on the class mix; reported per class |
| Llama 3.2 3B has Llama-3.1-8B's label by the rule and their item shares "match" | Descriptive similarity within the Llama line; size, distillation and post-training all differ, so no cause is named (very probably unreachable, R1-S3) |
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
| "replicates" or "small effect, positive", gates passed, labelled "failure-specific" by the rule of 7.5 | The task overrides a visible fact about the named object, more for failing calls than working ones; neither reference model meets this rule (R1-M1) |
| the same, labelled "a shift shared by working and failing calls" (including fail minus witness above 0 that does not meet the rule) | Goal leakage shared by working and failing calls: a task-consistency bias or an authorization belief, the pattern of both reference models on closer reading (`notes/tau_dx.md`); the τ table's own reading is printed beside it unchanged |
| class effects with intervals on opposite sides of 0 (R1-M1) | The pooled effect depends on the class mix; reported per class |
| a positive mean with an item share whose interval includes or lies below one half (R1-S1) | The mean is carried by a minority of items; reported as such, not as a general shift; the median item effect is given |
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
- **Llama 3.2 3B** (restated after review 1, R1-M1). Scaled by Llama-3.1-8B's ratio (same developer, same tokenizer
  and prompt bytes), about 0.48 nats, at the threshold. Predicted: a positive first-token effect with the CI above 0
  and an estimate below 0.5 nats ("small effect, positive"); labelled "a shift shared by working and failing calls"
  (the witness moving more than half as much, as Llama-3.1-8B's did), and plausibly the class-mix reading (status
  positive, unavailable at or below 0). Competence and the manipulation
  check are a real risk for a 3B model reading 10k-token JSON prompts.
- **Gemma 2 9B.** Excluded (2.3); no prediction.

The same, as logged forecasts (subjective probabilities, used by no rule):

| Model | CI above 0 | Verdict "replicates" | Labelled "failure-specific" by the rule of 7.5, given a positive effect (R1-M1) | Class-mix reading | Passes competence (first token) and both τ gates |
|---|---|---|---|---|---|
| Mistral 7B v0.3 | 0.85 | 0.6 | 0.2 | 0.3 | 0.7 |
| Llama 3.2 3B | 0.7 | 0.35 | 0.15 | 0.5 | 0.6 |

(Before review 1 the third column was "the failure-specific reading", by the τ table's row: 0.5 and 0.55. The new
column uses the stricter rule, which neither reference model meets, so its forecasts are lower.)

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
- **The `status` class has no determined witness** (reworded after review 1, R1-S6): its witness's success needs a
  variant's availability that the history does not show, so the discrimination check reads the other classes. That is
  not a lack of evidence that the models read the status record: both reference models score a call whose tool matches
  the order's status +4.9 (Qwen3-8B) and +3.0 (Llama-3.1-8B) nats higher (`notes/tau_dx.md`).
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

## Implementation record (2026-11-22, after the design commit 1b3a756, before any score)

No design rule changed. What was built and checked:

- **Code** (commit 80a59fa): `wf/tau/families.py`, the seven runs and their configs, `pace/tau_family_score.sbatch`
  (3-hour limit, L40S, A100 or H100) and `pace/tau_family_items.sbatch` (CPU). `wf.tau.pipeline.verdicts` and
  `sensitivity_verdicts` now take the model family as an argument; the default is the two reference models, so the
  reference design's behaviour is unchanged (the τ pipeline and families tests pass: 185 passed, 4 skipped). The extension calls them with its own
  family, so the verdict rule, the gates and the reading table are the reference design's own code.
- **Token check on ICE** (`tau_family_token_check`, job 7240557, at 80a59fa, after a rebuild and verification at that
  commit that reproduced all three item pins; 9 minutes on 4 CPUs): **PASS** for both scored models.
  - For each of Mistral 7B v0.3 and Llama 3.2 3B: all 11,984 scored prompts rendered and measured; all 3,424 replies
    (twin, failure reply, witness output and witness failure of each of the 856 items) begin with the scored token
    and tokenize alone exactly as the template writes them; `Yes` and `No` are the template's tokens after all 5,136
    yes/no and manipulation prompts; the prompts of every call and cue differ only in the task line; no prompt
    reaches the context. Problems of every kind: 0.
  - Scored tokens: Mistral `▁{"`, `▁Error`, `▁Yes`, `▁No`; Llama 3.2 `{"`, `Error`, `Yes`, `No`; each a single token.
  - Prompt lengths in tokens (median and maximum over the 856 items, failing call, output cue): Mistral 11,868 and
    14,607 (longest prompt of any variant 14,642, the yes/no question on tau-g-77cc59152d73); Llama 3.2 9,763 and
    11,874 (longest 11,907).
  - The Mistral placement check's 600 prompts (system text in the first turn): no problem, longest 14,516 tokens.
  - Gemma 2 9B (lengths only): all 11,984 prompts are at or above its 8,192-token context (median 11,339, longest
    14,051), confirming the exclusion of section 2.3.
- **Golden prompts** (`tests/golden/tau_families_rendered.json`): one real primary item (tau-g-f73576420d8e,
  `unavailable`, few-shot order reversed), the sha256 of the messages of every scored job under each model's handling
  (checked by the tests everywhere), and the sha256 and token count of each rendered prompt with each pinned tokenizer,
  written and re-checked identical on ICE (job 7240556; `scripts/tau_families_golden.py`).
- **Dry run without models** (`scripts/tau_families_dry_run.py`, mock scoring on the full item build in a temporary
  runs folder, real configs and shard counts): every stage ran. 856 primary items from 175 users; 11,984 passes per
  model, in shards of 3,080, 2,940, 3,038 and 2,926 (the largest about 1.3 h for Mistral at 1.5 s a pass); a resume of
  a finished shard adds nothing; 600 placement passes; the smoke set is 21 items (294 passes per model); the smoke
  analysis wrote a projection and no estimate; the families analysis found no missing or stale score, read all 14
  jobs of both reference models from a mock reference run, and its reference results check matched that run's τ
  analysis exactly. (A first attempt stopped at its token check because the code was committed while it was
  verifying the items: verification is tied to the commit, so every later stage refused. Rerun at a fixed commit.)
- **Mutation testing** (`scripts/mutation_check.py`, `MUTANTS_MATCHING="tau-fam"`, 4 shards): 58 mutants of the
  extension's code (the job list, cues and readout words, template handling, the placement spec, every check of the
  token check and its gate, the in-job length guard and readout report, resume and sharding, the competence rule and
  its population, task and thresholds, the competence gate, the Holm family, the item share, the comparison rules and
  their pairing, the placement rule, the reference commit, hash and results checks, the projection and budget, the
  smoke analysis's estimate-free output, the reading rows and patterns, and the pipeline's verdict family). 58 of 58
  killed, each by a named test in `tests/test_tau_families.py` (the runner now records the first failing test of
  every kill). One mutant (the comparison's bootstrap resampling the two models separately) first survived the
  targeted tests, because the test's synthetic users all had the same number of moved items, so even unpaired draws
  gave a mean difference of exactly 0; the test now has users that differ in that number and the mutant is killed.
  Six older mutants in the same file (targeting rewritten code in `wf/runs.py`) were already inapplicable at the
  base commit; they are untouched and outside this extension.
- **Tests.** `tests/test_tau_families.py`: 60 tests (57 run everywhere; the 3 rendered-prompt tests need the pinned
  tokenizers and skip elsewhere, and their check passed on ICE through the golden script). Full suite at 137dd01:
  957 passed, 10 skipped.
- **Status.** Steps 1 (design), 7 (prompt integrity and tokenization per model) and 8 (mutation testing) are done for
  the extension; steps 2 to 6 are inherited unchanged from the τ-bench build (the items are the same files). Not yet
  done: the independent engineering and scientific-validity reviews (steps 9 and 9b), the freeze at an annotated tag,
  the smoke run, scoring and the results check. Nothing has been scored.

## Amendments after review 1, before any score (2026-11-22)

Review 1 (`notes/tau_families_review1.md`, engineering and scientific validity, steps 9 and 9b) read the design and code
at 7c4a571 and found the engineering sound but the extension not ready to freeze: one must-fix and six should-fix.
`notes/tau_dx.md` (on `main`, 4df8ed9, with the results-note qualification eac03f7) had landed 28 seconds before this
note's design commit and was not taken into account; it overturns both reference descriptions. Every amendment below
was decided before any score of these models exists. The confirmatory rule (the per-model verdict, its gates, the Holm
family of the two scored models and the τ outcome table's reading) is unchanged; the amendments concern the
descriptive layer, one restored readout and the tests. Sections edited in place are marked "R1-".

- **R1-M1, the reference patterns and the cross-model label.** Sections 1, 7.5, 7.7 and 8 now describe the reference
  models as `notes/tau_dx.md` found them (Qwen3-8B's witness moves +9.45 against +3.98; Llama-3.1-8B's "failure-specific"
  row rests on +0.09 of +0.47 with its witness at about 80%, and its pooled effect averages +1.23 and −0.33 by class).
  `wf.tau.pipeline.verdicts` and `outcome_reading` are unchanged and still printed. The cross-model label is now the
  extension's own (`describe` in `wf/tau/families.py`): the row by its mechanical criterion, the witness share and the
  per-class failing effect and fail minus witness always beside it, "failure-specific" only under the three conditions
  of section 7.5 (fail minus witness interval above 0; witness less than half the failing effect; no class with fail
  minus witness below 0), otherwise "a shift shared by working and failing calls"; a class-mix row for class effects of
  opposite sign. The Llama 3.2 3B prediction is restated in these terms. Tests: the reviewer's scenario end to end
  (+0.40-like, witness about 80%, fail minus witness above 0, classes of opposite sign: not failure-specific, class mix
  reported, no pattern sentence calls it failure-specific), the Llama-3.1-8B and Qwen3-8B profiles, and each of the
  three conditions at its boundary.
- **R1-S1, minority of items.** A positive mean with an item share whose interval includes or lies below one half is
  reported as carried by a minority of items (a qualifier beside the verdict, and named in the direction sentence); the
  median item effect and the share of exact ties are reported.
- **R1-S2, the Holm family on the run path.** `run_tau_family_analysis` now passes both scored models to
  `family_verdicts` and `sensitivity_verdicts` whatever the config lists, so a model left out enters with p = 1. Test:
  the analysis with only Mistral in its config gives Mistral's Holm p as twice its raw p.
- **R1-S3, "matches".** Stated in section 7.5 before data: very probably unreachable at this sample size; the bar is not
  lowered.
- **R1-S4, the witness's yes/no restored** under the aligned and swapped lines (2 passes per item; 16 per item, 13,696
  per model). New descriptive analyses: fail minus witness anchored on neutral (both readouts), yes/no fail minus
  witness, per-class witness and fail minus witness, and the yes/no task judgement (share answered No among calls whose
  first token confidently predicts JSON, per call and line; line intercepts of yes/no on the first token). Cost: about
  6.1 GPU-hours for Mistral and 2.5 for Llama 3.2 3B; the largest shard is 3,520 passes, about 1.5 h for Mistral at
  1.5 s a pass, well inside the 3-hour limit. The budget flag in the smoke projection is raised from 6 to 6.5 hours
  (informational; the shard limit decides). The token check and the golden renders are rerun on ICE (record below).
- **R1-S5, tests for the reviewer's surviving mutants:** the placement threshold at 0.42 and 0.38 against a native
  effect of 2.0; no cross-model label without discrimination; the competence bounds at exactly AUC 0.70 and a Yes share
  of exactly 0.95 (both pass); the two-way mass as a median (a distribution whose mean and median lie on opposite sides
  of 0.5); the paired comparison bootstrap's interval width against the cluster bootstrap; the item share's interval
  clustered by user; the Llama-line row refusing "differs, within the margin"; the in-job readout report's witness
  check. The reviewer's mutants and mutants of the new logic are in `scripts/mutation_check.py` and recorded in
  `notes/testing.md`.
- **R1-S6, reporting.** Comparison entries of gated models are marked "not read: <why>"; section 9's status-class
  sentence is reworded.

What remains as the reviewer noted, unchanged: `wf.analysis.stats.holm` drops a NaN p from its count, which could only
arise with an empty population, impossible on the real items.

**Checks rerun after these amendments** (at 68ba60e, which holds the amendments and their code):

- **Token check on ICE** (`tau_family_token_check`, job 7244974, 17 minutes, after a rebuild and verification at that
  commit that reproduced all three item pins; report committed as `notes/tau_families_token_check.json`): **PASS** for
  both scored models. Each: all 13,696 scored prompts; all 3,424 reply checks; `Yes` and `No` the template's tokens
  after all 6,848 yes/no and manipulation prompts; prompts differ only in the task line; none reaches the context;
  0 problems. Longest prompts unchanged (Mistral 14,642, Llama 3.2 11,907). The 600 placement prompts pass. Gemma 2:
  all 13,696 prompts at or above 8,192.
- **Golden prompts** (job 7244973): the two restored jobs added per model; every earlier hash unchanged; re-checked
  identical.
- **Dry run without models** (at 68ba60e): every stage ran; 13,696 passes per model in shards of 3,520, 3,360, 3,472
  and 3,344 (the largest about 1.5 h for Mistral at 1.5 s a pass); 600 placement passes; 21 smoke items (336 passes per
  model); the analysis found no missing or stale score and its reference results check matched exactly.
