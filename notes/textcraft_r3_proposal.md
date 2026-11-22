# TextCraft round 3: a contemporary-model replication of round 2 (proposal, 2026-11-22)

**Status.** A proposal for the owner's review. It is not a frozen design and nothing in it has been scored. It was written on branch `textcraft-r3`, cut from `origin/textcraft-r2` at f610413, while round 2's last corrections were still in progress and before round 2's freeze tag existed.

**Written without round-2 results.** No round-2 score existed when this note was written, and none was read: no model has scored a round-2 record. The round-3 hypotheses below are carried forward from the round-2 *design* (`notes/textcraft_r2_design.md` at f610413), not from its outcomes. Any hypothesis added later because of what round 2 finds must be labelled "motivated by round-2 outcomes" and must not be presented as independently prespecified. Section 7.3 states how round 3 uses round 2's outcomes once they exist: as the targets of a replication rule fixed here, never as a source of new directional hypotheses.

**What may be built, and when.** The only round-3 code is the compatibility check of section 3 (`scripts/textcraft_r3_compat.py`, `pace/textcraft_r3_compat.sbatch`), which loads tokenizers and templates and scores nothing. The real round-3 code (section 6) is built only after (1) the owner has reviewed this proposal, and (2) round 2's freeze tag (`textcraft-r2-design-frozen-<date>`) exists. Round 3 is then frozen at its own tag (`textcraft-r3-design-frozen-<date>`) after an independent review, before any round-3 score, and any change after that tag is a reported deviation.

**Round 2 is not touched.** Round 3 modifies no round-2 file. It has its own configs, run names, results directory and code version, and it reuses validated round-2 code by import (section 6). Round 3 never delays round 2: it scores nothing before round 2 is frozen, and its cross-generation analysis (section 7.4) reads round-2 scores only after round 2 has been scored and its results filed.

## Decisions (2026-11-22)

The owner and the coordinating reviewer decided the open questions of section 8 on 2026-11-22, before round 2's freeze and before any round-2 or round-3 score. Where an earlier or later section of this proposal differs, this section governs.

- **D1. Gemma generation.** Gemma 4 12B-it (`google/gemma-4-12B-it` at 43c9d014b231…) is the core Gemma model, with Gemma 4 E4B-it (4ecc2ba63bca…) as the declared fallback under the rule of section 2.4. Gemma 3 12B is not used.
- **D2. Scope and staging.** One freeze tag covers the minimum and the expanded design. The minimum (the three core models, the TextCraft environment check, the smoke runs) runs first. The extension models `qwen3.8-27b`, `gemma4-31b-it` and `olmo3-7b-it`, the two format checks and the reasoning arm follow as GPUs allow. Whether each runs depends on GPU availability, not on core results; the families stay fixed in size (a test whose model was not scored enters with p = 1).
- **D3. Reasoning arm.** Only Qwen3.5-9B with thinking on (`qwen3.5-9b-think`), compared with the same weights with thinking off. It is exploratory (analysis c), reported separately, never pooled with the instruction runs, never entered in a test family and never gated with them, and generated with HF `generate` in serve2 (no vLLM environment). Ministral 3 Reasoning and OLMo 3 Think are dropped, so the question about Ministral's reasoning system prompt (section 8, item 8) no longer arises. OLMo 3 7B Instruct stays in the extension as a fourth developer, without its reasoning twin.
- **D4. No extra developers.** Granite 4.2 8B and Ministral 3 14B are not used. Family A3x therefore holds the three extension models (30 tests).
- **D5. Storage.** Route A, as in wave B: weights downloaded per job to node-local disk, every file checked against a per-model hash manifest committed before the freeze.
- **D6. Replication rule.** The minimum of two competent models in both rounds is accepted. A partial replication (a contrast confirmed in the round-2 direction in some competent round-3 models, with the others inconclusive, equivalent or opposite) is reported per model with its contrasts, labelled "partial", and never counted as a replication. In the table of section 7.3, the "partly replicated" and "replicated in some models, not in general" sub-cases are both reported under this label.
- **D7. When code is built.** Round-3 code is built only after both of these hold: wave B's scorer fixes have passed their re-review and are merged, and round 2's freeze tag exists. Nothing beyond the compatibility check is built before then.

**Cost for the chosen scope** (estimates, not timings; section 4.4 gives the rates behind them):

| Stage | Runs | GPU-hours |
|---|---|---|
| Minimum | `qwen3.5-9b` (2–4), `ministral3-8b-it` (1.7–2.5), `gemma4-12b-it` (2.5–3.5); environment check on Qwen3-8B and Mistral-7B-v0.3 (0.6); smoke runs (0.3); node-local downloads (about 1–2.5) | about 9–13; budget 14; 48 GB cards suffice |
| Extension | `qwen3.8-27b` (6–9) and `gemma4-31b-it` (6–9) on A100-80GB or two L40S; `olmo3-7b-it` (1.5–2); format checks (0.5); smoke runs (0.3); downloads (about 1–2.5) | about 15–23, of which 12–18 on 80 GB cards |
| Reasoning arm | `qwen3.5-9b-think` on about 4,500 record-arm generations, then scoring after each trace | about 3–6 |
| All | | about 27–42; budget 45 |

If the environment check fails, rescoring the three older models fully in serve2 adds about 7 GPU-hours (section 6.3).

## 1. The question

**Background in one paragraph.** The project tests whether LLM world models violate goal-independence: in a deterministic environment an action's outcome depends only on the state and the action, yet changing the task line changes the predicted success. TextCraft round 1 confirmed a goal effect for Qwen3-8B and not for Llama-3.1-8B, and its exploratory follow-ups suggested a two-part account: an *intermediate lift* (crafts that serve the named goal become more likely to be predicted to succeed) and a *penalty* on crafting the named goal item itself. Round 2 tests that account prospectively in five models with five contrasts (section 7.1), under a clean world-model prompt and an anchored yes/no question.

**Why round 3.** Round 2's five models were released between May 2024 and April 2025 (Mistral-7B-Instruct-v0.3, May 2024; Gemma-2-9B-it, June 2024; Llama-3.1-8B-Instruct, July 2024; Llama-3.2-3B-Instruct, September 2024; Qwen3-8B, April 2025). Whatever round 2 finds, it says little about the open-weight models in use at the end of 2026, which have had heavier agentic and reasoning post-training, newer tokenizers and templates, and in two cases new attention architectures. Round 3 asks whether round 2's measured quantities behave the same way in recent open-weight models when everything else is held fixed: the same records, the same goal interventions, the same prompt, the same scoring, and the same estimands.

**Three analyses, kept apart.** Round 3 has three analyses that answer different questions and are never pooled:

- **(a) Direct replication.** Each new model is run through round 2's confirmatory contrasts exactly as round 2 runs its own models, and the verdicts are read against (i) the account's own predictions as stated in the round-2 design and (ii) the pattern round 2 actually finds, through a replication rule fixed here (section 7.3).
- **(b) Older against newer releases.** For each developer line that has both a round-2 model and a contemporary model, the difference in each contrast between the two, estimated on the same items with a paired cluster bootstrap of the difference (section 7.4).
- **(c) Exploratory.** Size, reasoning mode, capability and architecture (section 7.5). Estimates with intervals, no confirmatory claims.

**What round 3 cannot show.** The models are a small, purposive sample: one model per developer line, chosen for recency, comparability and feasibility. No result in round 3 licenses a claim about "model generations" in general, about "2026 models", or about any model not scored. A difference between an older and a newer release changes developer data, size, tokenizer, template, post-training and sometimes architecture at once, so it is a difference between two released artifacts, not a "generation effect".

## 2. Model shortlist (verified)

### 2.1 How each entry was verified

Every identifier, revision, date, license, parameter count and configuration below was read on 2026-11-22 from public Hugging Face metadata, without authentication:

- the Hub API (`/api/models/<id>?blobs=true`): the current commit (`sha`), the repository's creation date, the license in the model card's metadata, gating, the safetensors parameter count by dtype, and every file's size;
- each repository's `config.json`, tokenizer files and chat template, downloaded at the pinned commit (config and tokenizer files only; no weights);
- for the two architectures new to the project (Gemma 4 12B and OLMo 3), the safetensors *headers* (tensor names, shapes and dtypes, read with HTTP range requests, no weights), compared with the architecture instantiated on PyTorch's meta device from the pinned config in serve2 (transformers 5.23.0);
- release dates from the developers' release pages or announcements where checked (Gemma 4 12B: Google's Gemma release page lists 3 June 2026; Qwen3.8-27B: the QwenLM repository lists 14 August 2026; Granite 4.2: announced 25 August 2026), otherwise the month recorded in wave B's design note; the Hub repository's creation date is given beside it, because a repository is often created before the public release.

### 2.2 The shortlist

The *core* models form the minimum viable replication; the *extension* models are the larger external-validity test; the *reasoning* models are a separate exploratory arm with a different estimand (section 7.5) and are never pooled with the instruction models.

| Tag | Hugging Face id | Pinned revision | Release (Hub created) | License | Parameters (safetensors) | bf16 weights | Context | Attention | Role |
|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5-9b` | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | Feb 2026 (2026-02-27) | Apache-2.0 | 9.65 B | 19.3 GB | 262,144 | hybrid: 24 linear-attention (gated DeltaNet) and 8 full-attention layers | core; newer half of the Qwen pair (Qwen3-8B) |
| `ministral3-8b-it` | mistralai/Ministral-3-8B-Instruct-2512-BF16 | f6fae9795746f63c9be8344932f01275f3c63734 | Dec 2025 (2025-10-31) | Apache-2.0 | 8.92 B | 17.8 GB | 262,144 (YaRN from 16,384) | full | core; newer half of the Mistral pair (Mistral-7B-v0.3) |
| `gemma4-12b-it` | google/gemma-4-12B-it | 43c9d014b231474fa7212dbb48a762566c2b255d | 3 Jun 2026 (2026-05-23) | Apache-2.0 | 11.96 B | 23.9 GB | 262,144 | 40 sliding-window (1,024) and 8 global layers | core; newer half of the Gemma pair (Gemma-2-9B-it) |
| `gemma4-e4b-it` | google/gemma-4-E4B-it | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | Mar 2026 (2026-03-02) | Apache-2.0 | 8.00 B (about 4 B effective) | 16.0 GB | 131,072 | 35 sliding-window (512) and 7 global layers | pre-declared fallback for `gemma4-12b-it` only |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 | 14 Aug 2026 (2026-08-05) | Apache-2.0 | 27.78 B | 55.6 GB | 262,144 | hybrid: 48 linear and 16 full layers | extension; the newest open dense model under 30 B from a round-2 developer |
| `gemma4-31b-it` | google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | Mar 2026 (2026-03-11) | Apache-2.0 | 31.27 B | 62.5 GB | 262,144 | 50 sliding-window (1,024) and 10 global layers | extension; a large model from a second developer, same template as `gemma4-12b-it` |
| `olmo3-7b-it` | allenai/Olmo-3-7B-Instruct | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | Nov 2025 (2025-11-19) | Apache-2.0 | 7.30 B | 14.6 GB | 65,536 (YaRN from 8,192) | 24 sliding-window (4,096) and 8 full layers | extension; a fourth developer with fully open training data, and the instruction twin of `olmo3-7b-think` |
| `qwen3.5-9b-think` | Qwen/Qwen3.5-9B (same weights as `qwen3.5-9b`) | as `qwen3.5-9b` | as above | Apache-2.0 | 9.65 B | 19.3 GB | 262,144 | as above | reasoning arm: the same weights with thinking on |
| `ministral3-8b-reasoning` | mistralai/Ministral-3-8B-Reasoning-2512 | 81eaece1948f3875421d9a45bc55487d10e2d894 | Dec 2025 (2025-10-31) | Apache-2.0 | 8.92 B | 17.8 GB | 262,144 | full | reasoning arm: Ministral 3's reasoning release, same size and tokenizer as the instruct model |
| `olmo3-7b-think` | allenai/Olmo-3-7B-Think | d97e442d7cc678210054dbcc9b440894d62c89a4 | Nov 2025 (2025-11-18) | Apache-2.0 | 7.30 B | 14.6 GB | 65,536 | as `olmo3-7b-it` | reasoning arm: OLMo 3's reasoning model, from the same base as `olmo3-7b-it` |

The revisions of `qwen3.5-9b`, `ministral3-8b-it`, `gemma4-e4b-it`, `qwen3.8-27b` and `gemma4-31b-it` are exactly the ones wave B pinned (`notes/families_waveb_design.md`, section 3), so the two studies score the same artifacts. None of the ten repositories is gated. The parameter counts are the Hub's safetensors totals; for the vision-language checkpoints they include the unused image (and for Gemma, audio) parts (wave B's meta-device counts: Qwen3.5-9B 9.41 B in all, 8.95 B in the language model; Ministral 3 8.49 B in the language model; Gemma 4 12B about 11.91 B outside its small vision and audio projections, counted here). Qwen3.8-27B's repository was modified on 2026-11-07 and Qwen3.6-27B's on 2026-11-20, which is why every job loads the pinned commit, never the branch head.

**The round-2 developer lines and their newer halves.** Qwen3-8B pairs with Qwen3.5-9B, Mistral-7B-v0.3 with Ministral-3-8B, and Gemma-2-9B-it with Gemma-4-12B-it. Meta has published no open Llama model of comparable size after Llama 3.2 (the Hub lists no meta-llama text model created after June 2025; Llama 4 is a mixture-of-experts model of at least 109 B parameters), so Llama-3.1-8B and Llama-3.2-3B have no newer half, and the Llama line is absent from analysis (b).

### 2.3 Why these three core models

- **Qwen3.5-9B.** The user's first candidate, verified as named. It is the direct successor of the study's main model at nearly the same size, and its thinking-off format is the one Qwen3-8B used in round 2 (an empty think block before the reply), so the Qwen pair differs in weights and post-training far more than in format. It adds an architecture the study has not scored: three of every four layers use linear attention.
- **Ministral 3 8B.** The user's third candidate, verified; the correct identifier is `Ministral-3-8B-Instruct-2512`. The default repository ships the language model in FP8 (7.42 B of its parameters are `F8_E4M3`), so round 3 uses Mistral's bf16 release of the same instruct model, `Ministral-3-8B-Instruct-2512-BF16`, as wave B does; every model in the study is scored in bf16, and an FP8 model would add a quantization difference to the Mistral pair.
- **Gemma 4 12B.** See 2.4.

### 2.4 Which Gemma generation: Gemma 4 12B, not Gemma 3 12B

The user named "Gemma 3 12B" as the newer Google model, while wave B uses Gemma 4. The recommendation is **Gemma 4 12B-it**, with Gemma 4 E4B-it as the pre-declared fallback, for five reasons.

1. **Recency.** Gemma 3 12B (`google/gemma-3-12b-it`, revision 96b6f1eccf38, March 2025) is older than Qwen3-8B, the newest round-2 model. It would not add a contemporary model. Gemma 4 is Google's current open family (March to June 2026).
2. **Size match.** Gemma 4 12B (11.96 B) is the size-matched successor of Gemma 2 9B (9.24 B). Wave B's review 1 asked for exactly this swap for its own Gemma pair, conditional on the model existing at a pinnable revision, sharing the template family, and fitting the same GPU tier. All three hold: it is pinned above; its chat template is byte-for-byte identical to Gemma 4 31B-it's (checked), so wave B's handling of the 31B template (the empty thought channel as a declared generation suffix) carries over unchanged; and its 23.9 GB of bf16 weights fit a 40 or 48 GB card.
3. **Access and license.** Gemma 3 is gated under the Gemma Terms of Use: its metadata is public, but every file, including the tokenizer, returned HTTP 401 without an accepted license and a token. As instructed, no token was read, so Gemma 3's tokenizer check could not be run. Gemma 4 is Apache-2.0 and ungated.
4. **Infrastructure.** Wave B has already pinned, templated and gated Gemma 4 E4B and 31B in serve2. Gemma 3 would need its own handling.
5. **What Gemma 3 would offer.** Its template has no system role and folds the system text into the first user turn, as Gemma 2's does, so a Gemma 2 to Gemma 3 pair would differ less in format than the Gemma 2 to Gemma 4 pair. That is a real advantage for a clean comparison, but not for a contemporary replication. If the owner wants it, Gemma 3 12B can be an exploratory "bridge" model in analysis (c), once the owner has accepted Google's license on the Hub and arranged access in the job environment; round 3 would not handle the token itself.

**What is new about Gemma 4 12B, and what must still be checked with the weights.** Its class is `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`, an encoder-free multimodal model), not the `Gemma4ForConditionalGeneration` of E4B and 31B. transformers 5.23.0 in serve2 has the class and maps it from both `AutoModelForCausalLM` and `AutoModelForImageTextToText`. Against the meta-device architecture, every language-model tensor of the checkpoint (666 tensors) matches in name and shape. The ten vision-embedder tensors are stored as `model.vision_embedder.*` and `model.embed_vision.embedding_projection`, where the class expects `model.embed_vision.*`, a renaming transformers is expected to apply on load; the output layer is tied to the embeddings. So, as in wave B, the job must load with `output_loading_info=True` and stop unless every parameter is loaded (none missing, none mismatched in shape, no unexpected tensor outside a pattern pre-registered for this model). That can only be checked with the weights, at the start of the first job.

**Fallback rule.** If Gemma 4 12B fails the loading check, the tokenization gate, the eager-against-SDPA agreement check (section 6.3) or the smoke run's format checks, it is replaced by Gemma 4 E4B-it before any round-3 contrast is computed, and the replacement is recorded as a deviation. The fallback's own size confound (about 4 B effective parameters against Gemma 2's 9 B) is then stated beside every Gemma reading.

### 2.5 Other 2025–2026 open-weight models considered

| Model | Verified | Decision and reason |
|---|---|---|
| Qwen3.6-27B (Apr 2026, 1cda14dd71a3, Apache-2.0) | yes; passes the compatibility check (section 3) | Not included: same architecture and size as Qwen3.8-27B, which is newer and already pinned by wave B. |
| Ministral-3-14B-Instruct-2512-BF16 (Dec 2025, 3cea74c1ebaf, Apache-2.0, 27.9 GB) | yes; passes | Optional size step within the Mistral line (fits a 40 or 48 GB card). Not in the recommended designs; the owner may add it to analysis (c). |
| IBM Granite 4.2 8B (Aug 2026, a7042144125c, Apache-2.0, 17.6 GB, dense, 131,072 context) | yes; passes with thinking off | Optional fifth developer. Recent and easy to run, but it has no round-2 predecessor and adds developer breadth rather than replication; its thinking mode is on by default. Not recommended for the minimum; a reasonable addition to the extension if the owner wants another developer. |
| OLMo 3 7B Instruct and Think (Nov 2025, Apache-2.0) | yes; both pass their checks | In the extension and the reasoning arm: a same-base instruction and reasoning pair, plain causal-LM class (no vision tower), and fully open pre- and post-training data, which in principle allows a contamination search for TextCraft text. OLMo 3.1 7B returned 401 (not public). |
| gpt-oss-20b (Aug 2025, 6cee5e81ee83, Apache-2.0) | yes; template renders | Not recommended. Reasoning-only (no thinking-off mode), mixture-of-experts weights stored in MXFP4 (19.1 B of its parameters as packed 4-bit), so it cannot be scored in bf16 like every other model; the harmony format moves the system text into a developer message; its sliding window is 128 tokens. |
| Gemma 3 12B (Mar 2025) | metadata only (gated) | Not recommended for the core (2.4). |
| Mixture-of-experts and frontier releases (Qwen3.8-Flash-Next, Qwen3.8-2.4T-A95B, DeepSeek-V4, Mistral Large 4, Mistral Small 4 119B, Nemotron 3.5) | Hub listings only | Out of reach on the cluster's GPUs in bf16, and not comparable in kind to round 2's dense 3–9 B models. |

### 2.6 Instruction models and reasoning models are kept apart

Round 2 scores a reply teacher-forced directly after the generation prompt, with thinking off. A model whose template always opens a reasoning block (Ministral 3 Reasoning ends its prompt with `[/INST]` and then writes `[THINK]`; OLMo 3 Think's generation prompt ends with `<think>`; gpt-oss writes an analysis channel) cannot be scored that way without putting it off its own distribution. The compatibility check confirms that none of the three has a thinking-off switch (rendering with `enable_thinking=False` gives the same prompt). Qwen3.5 has a switch: with it off, the model is an instruction model scored like Qwen3-8B; with it on, it is a reasoning model.

So round 3 has two kinds of run, which are never pooled, never entered in the same test family and never described as interchangeable:

- *instruction runs* (thinking off, or no thinking mode): round 2's estimand exactly, in analyses (a), (b) and (c);
- *reasoning runs* (the model writes its own reasoning first): a different estimand, the log-odds of the two replies after the model's own reasoning (section 7.5), in analysis (c) only.

## 3. Compatibility checks (tokenizers and templates only; no weights, no score)

### 3.1 What was run

`scripts/textcraft_r3_compat.py` imports round 2's code and records, changes nothing in them, and runs every candidate's pinned tokenizer and chat template over round 2's frozen records (`wf/textcraft/data/r2_items.jsonl`, sha256 `e6da991f…7e47c22f0`, checked by the script), round 2's prompt (`r2.messages`: the clean system text, the three verified examples, the anchored yes/no question) and round 2's arms. Every record and arm was checked for every instruction model: 25,743 prompts per model, of which 16,471 two-candidate prompts (32,942 candidate checks) and 9,272 yes/no prompts. It ran on ICE as one CPU job (`pace/textcraft_r3_compat.sbatch`, job 7241623, partition coc-cpu), in a fresh checkout of this branch (`text-wm-r3check`, deployed from a git bundle), with serve2's Python only read (transformers 5.23.0, tokenizers 0.23.3), offline, and tokenizers loaded from folders downloaded at the pinned commits. The recipe database was not needed for rendering; a copy was placed in the checkout (sha256 `5216d3a5…d26d65f`, as pinned) and is not committed. The per-model reports are in `notes/textcraft_r3_compat/<tag>.json`.

The checks, for every instruction model, on every record and arm:

1. **Round 2's tokenization gate.** Each candidate, tokenized alone as the scorer tokenizes it, equals the template's tokens for that text as the assistant's reply after that arm's prompt (with a declared generation suffix stripped first, as wave B does for Gemma 4 31B); "Yes" and "No" are single tokens equal to the template's reply after every yes/no prompt; the two candidates' first tokens differ.
2. **Wave B's prompt checks.** Every prompt ends with the model's declared thinking-off ending; the thinking switch is read (thinking on renders differently, ends as declared and carries the declared marker, and thinking off carries no marker); no forbidden default text; and for Ministral 3, every prompt's ids and every distinct candidate's ids equal mistral-common's encoding.
3. **Arm isolation in tokens (new).** For each pair of arms of a record that differ only in the goal line (aligned against swapped, neutral and two-item; the same for the yes/no, padded and extended-list arms: 11 kinds of pair), the token ids differ only inside the goal line, so no merge across the line's boundary changes a neighbouring token.
4. **Fit.** The longest input (prompt plus the longer candidate) against the context window and against the sliding-attention window.
5. **Segmentation.** Token counts of the success and failure replies, and the first-token pairs.

For the reasoning models only the template was checked (on round 2's five golden cases): the generation prompt's ending, whether a thinking-off render differs, whether the clean system text appears verbatim, and whether every distinct candidate (830 strings) and "Yes" and "No" tokenize without merging across the end of a closed reasoning block, which a score after a generated trace needs.

### 3.2 Results

The job ran at commit 55f2fa1 and took 17 minutes of wall time (10.5 to 16 minutes per model). **Every instruction model passed every check with no problem:** for each one, all 32,942 candidate checks, all 9,272 yes/no prompts, all 25,743 thinking-off endings and all 17,057 arm pairs.

| Tag | Gate | Problems | Yes / No ids | First tokens (success vs failure) | Reply tokens, success / failure (mean) | Longest input | Inputs over the sliding window | Arm pairs not isolated |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5-9b` | pass | 0 | 9175 / 2665 | "Craft" vs "Could" | 9.2 / 12.2 | 1,832 | none (no sliding window) | 0 of 17,057 |
| `ministral3-8b-it` | pass | 0 | 16860 / 4753 | "C" vs "Could" | 11.8 / 13.8 | 1,848 | none | 0 |
| `gemma4-12b-it` | pass | 0 | 10784 / 3771 | "Craft" vs "Could" | 9.7 / 12.6 | 1,717 | 41,755 of 42,214 over 1,024 | 0 |
| `gemma4-e4b-it` | pass | 0 | 10784 / 3771 | "Craft" vs "Could" | 9.7 / 12.6 | 1,713 | all 42,214 over 512 | 0 |
| `qwen3.8-27b` | pass | 0 | 9175 / 2665 | "Craft" vs "Could" | 9.2 / 12.2 | 1,844 | none | 0 |
| `gemma4-31b-it` | pass | 0 | 10784 / 3771 | "Craft" vs "Could" | 9.7 / 12.6 | 1,717 | 41,755 of 42,214 over 1,024 | 0 |
| `olmo3-7b-it` | pass | 0 | 9642 / 2822 | "Craft" vs "Could" | 9.2 / 12.2 | 1,815 | none over 4,096 | 0 |
| `qwen3.6-27b` (considered) | pass | 0 | 9175 / 2665 | "Craft" vs "Could" | 9.2 / 12.2 | 1,832 | none | 0 |
| `ministral3-14b-it` (optional) | pass | 0 | 16860 / 4753 | "C" vs "Could" | 11.8 / 13.8 | 1,848 | none | 0 |
| `granite4.2-8b` (optional) | pass | 0 | 9642 / 2822 | "Craft" vs "Could" | 10.2 / 13.2 | 1,866 | none | 0 |

"Longest input" is the prompt plus the longer reply, in tokens (round 2's models: 1,706 to 2,054). "Inputs over the sliding window" counts each scored sequence (32,942 candidate sequences and 9,272 yes/no prompts). For reference, round 2's models split the replies the same way except Gemma 2, which, like Ministral 3, splits "Crafted" into "C" + "rafted".

Also, for every instruction model:

- the template's ids equal the plain tokenization of its rendered text, and the rendered prompt and a reply joined as plain text tokenize as the prompt followed by the reply (0 differences of 25,743 and 32,942), so no template inserts a space before the reply (Mistral 7B v0.3's did);
- `apply_chat_template(tokenize=True)` returned a `BatchEncoding`, not a tensor;
- for Ministral 3 8B and 14B, all 25,743 prompts and all 830 distinct replies tokenize exactly as mistral-common encodes them, although transformers warns that the tokenizer's regex pattern is "incorrect" (6.3);
- the thinking switch was read where the model has one (Qwen, Gemma 4, Granite: thinking on renders differently, ends as declared and carries its marker, and thinking off carries none), and Ministral 3's and OLMo 3's renders are the same with or without it;
- no Ministral 3 prompt contains its default system prompt ("Le Chat", "The current date"), and no OLMo 3 Instruct prompt contains its default function-calling system prompt;
- Qwen3.8-27B writes an empty think block into the example replies of all 25,743 prompts; Qwen3.5-9B and Qwen3.6-27B write none.

Reasoning models (template checks on round 2's five golden cases; 832 strings: the 830 distinct replies plus "Yes" and "No"):

| Tag | Generation prompt ends with | Thinking-off render differs | Clean system text verbatim | Replies merging across the closing marker | "Yes" / "No" single tokens |
|---|---|---|---|---|---|
| `qwen3.5-9b-think` | `<|im_start|>assistant\n<think>\n` | yes (a switch exists) | yes | 0 of 832 after `</think>\n\n` | yes |
| `ministral3-8b-reasoning` | `[/INST]` (the model writes `[THINK]` itself) | no | yes (replaces the shipped reasoning prompt) | 0 of 832 after `[/THINK]` | yes |
| `olmo3-7b-think` | `<|im_start|>assistant\n<think>` | no | yes, followed by an appended sentence | 423 of 832 after a bare `</think>`; 0 after `</think>\n\n` | yes |
| `gpt-oss-20b` | `<|end|><|start|>assistant` | no | yes | 0 of 832 after the final-channel header | yes |

Gemma 3 12B could not be checked: its files are gated (HTTP 401 without an accepted license).

### 3.3 What the checks show, and what they do not

**What they show.** Round 2's records, prompts and arms can be scored by every core and extension model without any change to the records or the prompt text: the replies the scorer reads are exactly the replies each template writes, "Yes" and "No" are the single tokens the scorer reads, the two replies always start with different tokens, every prompt has its model's thinking-off ending, and every goal intervention changes only the tokens of the goal line itself. That last point matters for the cross-model comparisons: no model's tokenizer turns the goal-line change into a change in neighbouring tokens, so the contrasts measure the same intervention in every model.

**Systematic differences in what is measured, and how round 3 handles each.**

- *Reply segmentation.* The two-candidate L sums log-probabilities over each reply's tokens, and the replies are split differently by different tokenizers: about 9.2 and 12.2 tokens for Qwen and OLMo, 9.7 and 12.6 for Gemma 4, 11.8 and 13.8 for Ministral 3. Ministral 3 also splits "Crafted" into "C" + "rafted", as Gemma 2 did, so its first-token terms compare "C" with "Could" while the others compare "Craft" with "Could". Within a model every contrast compares the same replies under different goal lines, so segmentation cancels out of each contrast; across models it affects the scale of L, which is why analysis (b) reports a scale-free difference beside the raw one (7.4) and why first-token readouts are not compared across models.
- *Templates.* Qwen3.5 thinking off writes the same empty think block before the reply as Qwen3-8B did in round 2. Gemma 4 12B and 31B put an empty thought channel before the reply; Gemma 4 E4B does not. Ministral 3 puts the system text first, where Mistral 7B v0.3 put it in the last user turn, and writes no space before the reply, where Mistral 7B wrote one. Gemma 4 has a system turn, where Gemma 2 had the system text folded into the first user turn. Qwen3.8 writes empty think blocks into the three example replies of every prompt. Each of these is a format difference inside a generation pair or within a line; they are listed beside every pair reading (7.4), and the two that can be isolated cheaply (Gemma's system placement and Qwen3.8's example think blocks) have their own exploratory checks in the expanded design (6.3).
- *Attention windows.* 99% of Gemma 4 12B's and 31B's scored sequences, and all of Gemma 4 E4B's, are longer than their sliding windows; no other model's are (4.3).
- *Reasoning modes.* None of the three reasoning-only models can be switched off, so they cannot be scored with round 2's protocol; their compatible protocol (score after the model's own reasoning, with `</think>` followed by a blank line for OLMo 3) is in 7.5.
- *Software.* transformers 5.23 returns a `BatchEncoding` from `apply_chat_template(tokenize=True)` for every candidate, which round 2's scorer cannot take (6.3).

**What they do not show.** Nothing here involves the weights. The loading of each checkpoint (in particular Gemma 4 12B's renamed vision-embedder tensors), the agreement between eager and SDPA attention for Gemma 4, the speed of each model, and whether each model puts its yes/no probability mass on "Yes" and "No" are checked only at the start of each GPU job, before its first score. The checks also say nothing about competence or about any effect.

## 4. Compute, memory, quantization and context

### 4.1 Load per model

Each instruction model scores round 2's full set: 25,743 (record, arm) combinations, which is 16,471 two-candidate arms at two forward passes each and 9,272 yes/no arms at one pass each, about 42,200 forward passes per model (round 2, section 11). Inputs are at most 1,866 tokens for the new models (section 3.2; round 2's models reach 2,054).

### 4.2 Memory and quantization

All models are scored in bf16 with log-softmax in float32, as in rounds 1 and 2 and wave B. Memory is dominated by the weights; activations for one sequence of under 1,900 tokens are small.

| Tag | bf16 weights | Fits | Proposed GPU |
|---|---|---|---|
| `olmo3-7b-it` | 14.6 GB | any card of 24 GB or more | L40S, A40 or A100 |
| `ministral3-8b-it` | 17.8 GB | 24 GB or more | L40S, A40 or A100 |
| `qwen3.5-9b` (and `-think`) | 19.3 GB | 24 GB or more | L40S, A40 or A100 |
| `gemma4-12b-it` | 23.9 GB | 40 GB or more | L40S, A40 or A100 |
| `qwen3.8-27b` | 55.6 GB | 80 GB, or two 48 GB cards | A100-80GB; two L40S as fallback |
| `gemma4-31b-it` | 62.5 GB | 80 GB, or two 48 GB cards | A100-80GB; two L40S as fallback |

**No quantization in any confirmatory or extension run.** Quantizing weights changes every log-probability, so a quantized model's contrasts would not be the same quantity as a bf16 model's, and the paired comparisons of analysis (b) would mix model differences with quantization error. Three consequences:

- Ministral 3 is scored from the bf16 release (2.3), not the FP8 default repository.
- The two large models run unquantized on an 80 GB card or split over two 48 GB cards of one node (`device_map="auto"`; the loading check refuses any CPU or disk placement). The cluster's coc-gpu partition has A100 nodes (two cards each), L40S nodes (eight cards each) and A40 nodes; wave B found two of the A100 nodes have 80 GB cards.
- If neither is available, the large models are dropped from the run rather than quantized. A quantized run (for example Google's QAT checkpoints, which are different released artifacts, or 8-bit weights, which need bitsandbytes, absent from serve2) could only be an exploratory addition, labelled as quantized and accompanied by a bf16-against-quantized agreement check on a seeded subset. It is not proposed.

### 4.3 Context fit

Every candidate's context window (65,536 tokens for OLMo 3, 131,072 for Gemma 4 E4B, 262,144 for the others) is more than 30 times the longest TextCraft input (1,866 tokens), so no prompt is truncated and positional extrapolation (YaRN in Ministral 3 and OLMo 3) is not exercised. The windows that matter are the *sliding* attention windows: every scored sequence exceeds Gemma 4 E4B's 512 tokens, and 41,755 of 42,214 exceed Gemma 4 12B's and 31B's 1,024 tokens (section 3.2), so on those models' local layers the recipe list and the first examples are out of view and reach the reply only through the global layers. That is how the deployed models work, and it is recorded rather than avoided, but it is a structural difference from round 2's models (Gemma 2's window of 4,096 tokens was never exceeded) and is named beside every Gemma reading. No input reaches OLMo 3's window of 4,096.

### 4.4 GPU-hours (estimates, not timings)

Round 2 assumed about 25,600 forward passes per GPU-hour for an 8 B model on these prompts (round 1's measured rate), and wave B measured Mistral 7B at 43,746 ALFWorld passes in 1 h 43 min on an A40 with shorter prompts. Qwen3.5's and Qwen3.8's linear-attention layers run in PyTorch in serve2 (no `fla`, `causal-conv1d` or `kernels` package is installed, and wave B forbids the last), so their rate is the least certain. Gemma 4 runs with SDPA attention after the agreement check of section 6.3.

| Run | Forward passes | Estimate |
|---|---|---|
| `ministral3-8b-it` | about 42,200 | 1.7–2.5 h |
| `qwen3.5-9b` | about 42,200 | 2–4 h |
| `gemma4-12b-it` | about 42,200 | 2.5–3.5 h |
| `olmo3-7b-it` | about 42,200 | 1.5–2 h |
| `qwen3.8-27b` | about 42,200 | 6–9 h (A100-80GB) |
| `gemma4-31b-it` | about 42,200 | 6–9 h (A100-80GB) |
| TextCraft environment check: Qwen3-8B and Mistral-7B-v0.3 rescored in serve2 on 300 seeded records, all arms | about 4,300 each | 0.3 h each |
| Smoke run, 20 seeded records per model, all arms | about 330 each | 0.1 h each, plus loading |
| Reasoning arm, `qwen3.5-9b-think` only (Decisions, D3; section 7.5): about 4,500 generations of up to 2,048 tokens with HF `generate` in serve2, then the two replies scored after each trace | about 4,500 generations and 9,000 passes | 3–6 h |
| Format checks (Gemma 4 12B system placement; Qwen3.8 `preserve_thinking=False`), 300 seeded records each | about 1,800 each | 0.5 h together |
| Node-local weight download (wave B's route A), per job | — | 0.2–0.4 h |

## 5. Minimum viable replication against an expanded study

**Minimum viable replication (recommended to run first).** The three core instruction models (`qwen3.5-9b`, `ministral3-8b-it`, `gemma4-12b-it`) on round 2's full records and arms; the TextCraft environment check on two round-2 models; smoke runs. It answers analysis (a) for three contemporary models from three of round 2's four developers and gives analysis (b) three generation pairs. Estimated 9–13 GPU-hours with downloads; budget 14. Everything fits 48 GB cards.

**Expanded study.** The minimum, plus:

- the extension models `qwen3.8-27b` and `gemma4-31b-it` (the larger external-validity test; 12–18 h on 80 GB cards) and `olmo3-7b-it` (a fourth developer; 1.5–2 h);
- the reasoning arm: `qwen3.5-9b-think` only, beside the same weights with thinking off (3–6 h; Decisions, D3);
- two format checks: Gemma 4 12B with the system text folded into the first user turn, as Gemma 2 had it, and Qwen3.8 with `preserve_thinking=False` (about 0.5 h together; section 6.3).

Estimated 27–42 GPU-hours in all, of which 12–18 on 80 GB cards; budget 45 (cost table in the Decisions section).

**Recommendation.** Freeze the whole design (minimum, extension and exploratory arms) at one tag, so that nothing about the extension is chosen after seeing core results, and run it in stages: the minimum first; then the extension and the reasoning arm as GPU time allows. The core verdicts do not depend on whether the extension runs (separate test families, section 7.6). Whether the extension runs is decided by budget, recorded before the core results are read, and not by them. If the owner wants to spend less, the minimum alone is a complete study.

## 6. Pipeline changes (built only after this proposal is reviewed and round 2 is frozen)

### 6.1 Reused unchanged, by import

- **Records and interventions.** `wf/textcraft/data/r2_items.jsonl` (2,982 records; 25,743 record-arm combinations), `r2_manifest.json` (must be verified and hash the records exactly), the few-shot examples (`fewshot.json`), `r2.messages`, `r2.candidates`, `r2.parse_arm`, `r2.jobs`, `r2.YN_QUESTION`, `r2.SYSTEM` (`prompts.CLEAN_SYSTEM`). Every round-3 model sees exactly the prompts every round-2 model sees, up to each model's own template.
- **Item hash.** `r2.item_hash`, unchanged, so a round-3 score and a round-2 score of the same record carry the same hash and can be paired item by item; scores carry the model tag and revision.
- **Analysis.** The pure functions of `r2_analysis`: populations, the five contrasts, the overall and contrast-specific competence, the margin, the family, the verdicts, readings, interpretations, the rejection rules, the central-question mapping, the matched role gap, the heuristic tracking and the descriptive block. Round 3 calls them on its own scores; it does not copy or edit them.
- **Gate logic.** `r2_tokcheck.gate`'s checks and wave B's additions.

### 6.2 New round-3 files

- `wf/textcraft/r3.py`: the round-3 model specs (built on wave B's `ModelSpec` fields: model class, tokenizer arguments, generation suffix, thinking-off ending, forbidden text, mistral-common check, unexpected-tensor pattern, minimum GPU memory), the tags, the test families of section 7.6, and the replication rule.
- `wf/textcraft/r3_score.py`: the run `textcraft_r3_score_<tag>`, sharded and resumable like `r2_score` (which cannot be reused directly because it looks models up in `r2.MODELS`), with the gate on every resume and the golden renderings of `wf/textcraft/data/r3_chat_golden.json`.
- `wf/textcraft/r3_reasoning.py`: the reasoning arm (generation with a budget and forced close, then scoring after the trace), modelled on `wf/thinking.py`.
- `wf/textcraft/r3_envcheck.py`: the TextCraft environment check (6.3).
- `wf/textcraft/r3_analysis.py`: the run `textcraft_r3_analysis`, which reads round-3 scores and, for analysis (b) only, the frozen round-2 scores of the older models.
- Configs `wf/config/textcraft_r3_*.yaml`; jobs `pace/textcraft_r3_score.sbatch` (serve2, node-local weights) and `pace/textcraft_r3_tokcheck.sbatch`; results under `runs/textcraft_r3_*`; tests `tests/test_textcraft_r3.py`; round-3 mutants in the mutation check.

### 6.3 What round 3 needs from wave B, and the checks it adds

Round 3 must run in serve2 (transformers 5.23), because none of the new models loads in the study's older environment. The scorer changes it needs exist only on branch `families-waveb`, where a fix round after review 1 is in progress. Round 3 reuses them after they are reviewed and merged; it does not re-implement them. In particular:

- **The BatchEncoding unwrap** (wave B review 1, M1). The compatibility check confirms that in serve2 `apply_chat_template(tokenize=True)` returns a `BatchEncoding` for every candidate's tokenizer. Round 2's scorer (`LogoddsScorer._encode`) passes the result straight to `torch.cat` and to the model, so it would crash on every chat score in serve2. Round 3 cannot start until the fixed scorer is merged.
- **Loading.** The vision-language model classes, `mistral_format=False` for Ministral tokenizers, `dtype`, the strict loading check with per-model unexpected-tensor patterns (a new pattern for Gemma 4 12B's vision-embedder renaming, fixed from its loading record at the smoke run, before any contrast), the refusal of CPU or disk placement, and the environment guard (transformers 5 or later; no `kernels` package).
- **Templates.** The generation suffix for Gemma 4 12B and 31B, the thinking-off endings, the forbidden default text for Ministral 3, and mistral-common agreement.
- **Storage route A** with the per-model file-hash manifest (review 1, M3): weights downloaded per job to node-local disk, every file hashed against a manifest committed before the freeze. The scratch quota had 53 GB free on 2026-11-22, less than the two large models need.
- **Gemma 4 attention** (review 1, S9): at the start of each Gemma 4 job, about five records scored under both eager and SDPA attention, agreeing within a stated tolerance, or the job stops.

Round 3 adds:

- **A TextCraft environment check.** Round 2 scores its models in the study's older environment (transformers 4.51.3), and round 3 scores in serve2, so analysis (b) compares scores made with different software. Wave B's environment check covers ALFWorld prompts only. Before any round-3 score is read, Qwen3-8B and Mistral-7B-v0.3 rescore 300 seeded round-2 records under all their arms in serve2, on the GPU type of their round-2 runs, and the scores are compared with their round-2 scores. Pass rule, fixed now from wave B's: mean absolute difference per score at most 0.35 nats (two-candidate) and 0.07 nats (yes/no); the sample's aligned-minus-swapped effect differs by at most 0.10 nats on each scale; every score present in both. This comparison of raw scores across environments is made only after round 2 is scored and frozen. If it fails, analysis (b) rescores the three older models fully in serve2 (about 7 GPU-hours) and compares serve2 with serve2.
- **The gate, extended,** with the arm-isolation check of section 3 run on every resume.
- **Mistral tokenizer regex.** transformers 5.23 warns that the Ministral 3 tokenizers carry "an incorrect regex pattern" and suggests `fix_mistral_regex=True`. The compatibility check compared the fast tokenizer as loaded (without that flag, as wave B loads it) with mistral-common on all 25,743 TextCraft prompts and all 830 distinct replies, for Ministral 3 8B and 14B, and found them identical, so the warning does not touch these strings. The tokenizer is therefore loaded exactly as wave B loads it, and the round-3 gate repeats the comparison on every resume, so a difference would stop the job.
- **Format checks (expanded design only, exploratory).** Gemma 4 12B with the system text folded into the first user turn, on 300 seeded records, aligned and swapped arms in both formats, read by wave A's one-fifth rule (the Gemma pair is "format-dependent" if the change's interval excludes zero and it exceeds a fifth of the native effect). Qwen3.8 with `preserve_thinking=False` on the same records and rule (its template writes empty think blocks into every example reply; section 3.2).

### 6.4 Reproducibility

- One code commit for every shard of every round-3 model, refused otherwise; every score carries its model's pinned revision (round 2's provenance rule).
- Pinned revisions in `wf.constants.REVISIONS`, a committed hash manifest per model, and the environment versions recorded per shard.
- Golden renderings per model, compared before scoring; the tokenization gate on every record and arm before any score and on every resume.
- Mock dry run of the full pipeline, tests, and mutants with named killing tests, as for round 2, before the freeze.
- The reasoning arm's generations are saved with their seeds and token counts, and scored from the saved text, so the scoring can be rerun without regenerating.

## 7. Hypotheses, rules and analysis plan

### 7.1 The estimands, unchanged

All contrasts are on the two-candidate log-odds L (the log-probability of the success reply minus that of the failure reply, both teacher-forced after the prompt), on failures, with goal pairs as clusters, exactly as in round 2 (`textcraft_r2_design.md`, section 6). With ΔL the aligned-minus-swapped difference of an item, the five are:

The intermediate lift, the mean goal effect on the 990 ingredient main failures,

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}}.$$

The within-item penalty on the 675 parent failures, where action, history and displayed list are identical and only the goal line changes,

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})}.$$

The naming / partial-completion contrast and the completion contrast with naming held, on the 500 goal-role and 496 class-matched ingredient failures with a two-item line,

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}},$$

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

The sensitivity to added empty-inventory steps, on the padded failures, where the superscript counts the look-ups put before the history,

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}}.$$

Round 3 keeps round 2's statistics for each: an estimate, a 95% percentile pair bootstrap, a two-sided p (sign-flip permutation of whole pairs for Λ and P, the bootstrap p for Dn, Dc and T), and a two one-sided equivalence test against a margin δ, with 20,000 resamples and flips for family tests and 2,000 for descriptive intervals. It keeps round 2's verdict table (confirmed above or below zero; equivalent to zero; inconclusive, with "equivalence underpowered" where flagged; not applicable for T), its reading tables, its descriptive readouts and its heuristic tracking. Where round 2's final frozen text differs from what is quoted here (its last corrections were in progress when this was written: the matched role gap of R3 and the Λ comparison of R4 become descriptive, R5 is reworded, the competence wording separates discrimination from understanding of the inventory rule, and the central-question labels are renamed), the frozen round-2 text governs and round 3 inherits it.

### 7.2 Competence: does round 2's rule transfer?

**Round 2's rule.** The overall gate (yes/no AUC at least 0.70 between successes and failures under the neutral line; Yes on at most 95% of failures and at least 5% of successes; median P(Yes) + P(No) at least 0.5) is reported, and each contrast is read against its own competence rule: the yes/no neutral-line AUC at least 0.70 in each population the contrast uses, with the other three criteria on the full population. Competence never withholds a verdict; it decides whether a confirmed effect is read as a distortion in a competent predictor or only as a measurable goal effect.

**Assessment, criterion by criterion.**

- *The AUC.* It is a rank statistic: it does not depend on a model's logit scale or calibration, only on whether successes score above failures. It transfers across models without change. Newer models may sit near its ceiling; that does not harm the reading, but it means the 0.70 floor will rarely bind, and the heuristic tracking becomes more important for saying *how* a model discriminates (14.V4: a high AUC can come from a shortcut that coincides with the truth).
- *The two Yes shares.* They are read at L = 0, so they depend on calibration and on a model's prior towards one answer. They are protections against a degenerate model that answers one word to everything, which is a model-independent concern. They transfer.
- *The mass.* The median P(Yes) + P(No) depends on the template's reply format: a model whose reply after thinking-off text or an empty thought channel starts with something else (a space, markdown, a quotation mark) would put little mass on the bare words. The compatibility check confirms that, for every core and extension model, the template's own reply to "Yes" and "No" is the single bare token the scorer reads. Whether the models put their mass there can only be seen with the weights.
- *The question itself.* Round 2's anchored question is new to every model; nothing suggests it behaves differently for newer models, and the competence it measures is the same quantity for all of them.

**Adaptations, fixed before any round-3 effect exists.**

1. The thresholds are kept exactly, so that "competent" means the same thing in rounds 2 and 3.
2. The smoke run (20 seeded records, all arms) may be read for three things only: the loading record, timing, and the median yes/no mass. If a core model's median mass on the smoke run is below 0.5, the owner decides, before any full run, whether to replace it (Gemma 4 12B by its fallback; the others have none) or to run it and let the gate decide. No contrast, AUC or effect is computed on the smoke run.
3. Added descriptive readouts: each contrast's competence verdict at a stricter AUC floor of 0.80 (so that a reader can see where a reading depends on a model sitting between 0.70 and 0.80), and the two-candidate AUC beside the yes/no AUC in every population. Neither changes a verdict.
4. Reasoning runs are not gated with the instruction runs. Their competence (the yes/no AUC after reasoning, on the subset they score) is a different quantity and is reported only.

### 7.3 Analysis (a): direct replication, with its hypotheses

**Confirmatory hypothesis, carried forward from the round-2 design.** Round 2's design states the account's predictions for *any* model (its section 7, last row of the predictions table): the within-item penalty is below zero. It made that prediction for each of its three non-reference models as "the account's generality claim". Round 3 carries it forward unchanged, for each core model:

- **H3-P.** P is below zero (read as "confirmed below zero" in family A3).

No directional prediction is made for Λ, Dn, Dc or T, because the round-2 design made none for models other than Qwen3-8B and Llama-3.1-8B, and none for either of those on Dn or T. Those four contrasts are tested (differs and equivalent) and read through round 2's tables. No expectation in this section rests on round-2 outcomes.

**The rejection list, applied to the round-3 models.** Round 2's coded criteria are computed for each core model with the competence each needs, as frozen in round 2: R1 (for a model competent for P, P equivalent to zero or confirmed above zero rejects the penalty component for that model); R2, restated for round 3 (among the core models competent for P, at least one has P equivalent to zero or confirmed above zero and none is confirmed below zero: the penalty is not shared among these contemporary models; if any is confirmed below zero it does not fire; otherwise undecided); R5 (for a model competent for T, T confirmed below zero). R3 and R4 are applied in whatever form round 2 freezes them (both are becoming descriptive); R4 compares Llama-3.1 with Qwen3 and has no round-3 counterpart.

**The replication rule (fixed now, applied once round 2's verdicts exist).** For each contrast c, round 2's frozen analysis gives a verdict per round-2 model and a competence flag. From those alone, mechanically, round 3 derives round 2's *general finding* for c:

- *general, sign s*: at least two round-2 models are competent for c, and c is confirmed in direction s in every one of them;
- *general null*: at least two round-2 models are competent for c, and c is equivalent to zero in every one of them;
- *model-specific*: among the competent round-2 models, c is confirmed in some and equivalent to zero or confirmed in the other direction in others;
- *not established*: anything else (fewer than two competent models, or inconclusive verdicts among them).

Round 3 then reads c among the core models competent for c (and, separately, among the extension models):

| Round 2's general finding | Replicated | Not replicated | Inconclusive |
|---|---|---|---|
| general, sign s | c confirmed in direction s in every competent core model, with at least two competent | c equivalent to zero or confirmed opposite to s in at least one competent core model, and confirmed in direction s in none | anything else: fewer than two competent core models; or some confirmed in direction s with the rest inconclusive ("partly replicated"); or some confirmed in direction s and some equivalent or opposite ("replicated in some models, not in general"), each sub-case named |
| general null | c equivalent to zero in every competent core model, with at least two competent | c confirmed (either direction) in at least one competent core model | anything else; for a model whose equivalence test is underpowered, "inconclusive (equivalence underpowered)" |
| model-specific, or not established | no replication target: round 3's per-model verdicts are reported as new evidence on the account (H3-P and the rejection list), not as a replication | — | — |

Verdicts in models not competent for c are reported beside the reading but do not count towards it. The headline of analysis (a) is the P row, because the penalty is the account's general claim; the other four contrasts are reported in the same form.

**The headline for P, in plain words.** If round 2 finds the penalty in every competent model, round 3 replicates it when every competent contemporary model also shows a confirmed negative P; it fails to replicate when a competent contemporary model shows no penalty larger than δ or a confirmed positive one and none shows a confirmed negative one. If round 2's penalty is model-specific or not established, round 3 has no round-2 finding to replicate and tests H3-P afresh in three new models.

### 7.4 Analysis (b): older against newer releases, paired on items

**Pairs (fixed now).** Qwen3-8B against Qwen3.5-9B; Mistral-7B-Instruct-v0.3 against Ministral-3-8B-Instruct; Gemma-2-9B-it against Gemma-4-12B-it (or its fallback, Gemma 4 E4B, with the size confound stated).

**The estimand.** For pair (old, new) and contrast c, the difference of the contrast between the two models on the same items,

$$\Delta_c = c_{\text{new}} - c_{\text{old}}.$$

For Λ and P this is the mean of item-level differences: with d the per-item contrast (for example the aligned-minus-swapped difference of one ingredient failure), each item contributes the newer model's d minus the older model's d on the same record. For Dn, Dc and T it is the difference of the two models' difference-in-differences on the same rows.

**The uncertainty.** Both models scored the same records, so one bootstrap resample of goal pairs serves both: in each of 20,000 resamples, c is computed for both models on the resampled pairs, and the difference is taken. The 95% percentile interval of those differences is the interval for the difference; the two-sided bootstrap p uses round 2's formula; and a two one-sided equivalence test uses the margin of the pair, the smaller of the two models' margins. No reading in analysis (b) rests on comparing two point estimates or on whether two separate intervals overlap.

**Logit scale.** The two-candidate L sums log-probabilities over each reply's tokens, and the models differ both in how confident they are (round 1's item spread of ΔL was about 1.4 nats for Llama-3.1 and about 4.8 for Qwen3) and in how they split the replies into tokens (section 3.2). A raw difference in nats can therefore reflect scale as much as goal sensitivity. So beside the raw difference, the analysis reports a scale-free version: each model's contrast divided by its own item standard deviation of ΔL over the 1,500 main failures (the SD that sets its margin), bootstrapped jointly with the same resamples. A raw "differs" verdict is called *robust to scale* only when the standardized difference's 95% interval excludes zero in the same direction; otherwise it is reported as "differs in nats; not distinguishable once scale is removed".

**Readings.** "Differs" (Holm-adjusted p below 0.05 and the interval excluding zero), "similar within the margin" (equivalence test passes after Holm), or "not distinguishable" (neither; never read as "the same"). Each reading names the two released models and lists what changes between them: developer data and post-training, parameter count, tokenizer and segmentation, template (Qwen: empty think block in both; Mistral: system text in the last user turn and a space before the reply in Mistral 7B, system text first and no space in Ministral 3; Gemma: system text folded into the first user turn in Gemma 2, a system turn and an empty thought channel in Gemma 4), attention (Qwen3.5's linear layers; Gemma 4's 1,024-token windows against Gemma 2's 4,096), and the software environment. With three purposive pairs, the direction of the three differences is described, never summarized as a trend.

### 7.5 Analysis (c): exploratory

Everything here is estimation with 95% intervals and no p-values in any family.

- **Size and recency within a line** (expanded design): Qwen3.8-27B against Qwen3.5-9B (size and six months of development), Gemma 4 31B against Gemma 4 12B (size and a different model class, same template and release family), each as a paired difference by the method of 7.4. The extension models also get the replication reading of 7.3, as a separate line, never merged with the core.
- **Reasoning mode** (expanded design). A different estimand: the model writes its own reasoning, and the two replies are then scored after that reasoning (the project's thinking-mode method, `wf/thinking.py`). Adaptations, fixed now:
  - *Records and arms:* the 675 parent failures under the extended-list aligned and parent lines (P), the 990 ingredient failures under the aligned and swapped lines (Λ), and, for competence, the yes/no question under the neutral line on the 600 successes and a seeded 600 of the main failures; about 4,500 generations per model.
  - *Generation:* each model's recommended sampling settings, a fixed seed per record and arm, at most 2,048 reasoning tokens; reasoning that reaches the cap is closed with the model's own closing marker and the reply is then scored (budget forcing), and such cases are flagged and every result is reported with and without them.
  - *Closing markers:* `</think>` followed by a blank line (Qwen3.5 and OLMo 3; the compatibility check found that OLMo 3's bare `</think>` merges with half of the replies' first tokens, while with the blank line no reply merges), `[/THINK]` (Ministral 3 Reasoning).
  - *System text:* Ministral 3 Reasoning's shipped system prompt is what tells it to reason in `[THINK]` blocks, and round 2's clean system text would replace it; so for that model the shipped reasoning prompt (from `SYSTEM_PROMPT.txt` at the pinned revision) is put before the clean text, identically in every arm, and the smoke run checks that traces appear. OLMo 3 Think's template appends a "no functions available" sentence to the system text, identically in every arm; it is recorded.
  - *Pairs read:* each reasoning run against its instruction twin on the same items (Qwen3.5-9B thinking on against off, same weights; Ministral 3 Reasoning against Instruct; OLMo 3 Think against Instruct). A difference mixes the reasoning itself with the reasoning model's different post-training (except for Qwen3.5) and with sampling noise; it is described as such.
  - *Compute:* serve2 has no vLLM, so generation uses HF `generate` (3–6 GPU-hours per model). A vLLM build would be faster but is a third software environment with its own equivalence check; not proposed.
- **Capability.** Across every instruction model scored in rounds 2 and 3, each contrast against the model's competence AUC and its two-candidate AUC, as a table and a plot. With at most nine models chosen on purpose, no correlation is tested and no trend is claimed.
- **Architecture.** Full attention (Ministral 3, the round-2 models), sliding-window hybrids (Gemma 4, OLMo 3) and linear-attention hybrids (Qwen3.5, Qwen3.8) are named beside each model's results. Architecture is confounded with everything else, so nothing is attributed to it.

### 7.6 Multiplicity

Each confirmatory question has its own family, Holm-adjusted at a familywise 0.05, with its size fixed at the freeze. A test whose model was not scored, or whose p is missing, enters with p = 1, so no family shrinks.

| Family | Tests | Size |
|---|---|---|
| A3 (analysis a, core) | 3 core models × 5 contrasts × 2 tests (differs, equivalent) | 30 |
| A3x (analysis a, extension; expanded design only) | 3 extension models (`qwen3.8-27b`, `gemma4-31b-it`, `olmo3-7b-it`) × 5 × 2 | 30 |
| B3 (analysis b) | 3 pairs × 5 contrasts × 2 tests (differs, equivalent) | 30 |
| Analysis (c) | none: estimates and intervals only | 0 |

Round 2's own family of 50 tests is untouched and is not combined with any round-3 family. Families A3 and B3 answer different questions and are not corrected across each other; a reader who wants to combine claims across families should note that. The scale-free versions of analysis (b) and every descriptive readout are outside every family.

### 7.7 The equivalence margin: does round 2's transfer?

**Round 2's margin.** For model m, the margin on the two-candidate scale, the same for all five contrasts, is

$$\delta_m = \min\left(0.3 \times \mathrm{SD}_m,\ 0.4\ \text{nats}\right),$$

where the SD is the item standard deviation of ΔL over the 1,500 main failures under the clean prompt, computed once from the point estimates before any equivalence test is read.

**Assessment.** The margin mixes a relative part (0.3 SD) and an absolute cap (0.4 nats, an odds ratio of about 1.5). In round 1 the cap bound for both reference models, and round 2's power table shows that for a model with Qwen-sized spreads equivalence is unreachable for P, Dn, Dc and T at these sample sizes. Newer models are likely to be more confident than round 2's, with larger spreads, so for them the cap will probably bind and equivalence will often be unreachable. The consequence for round 3 is specific: in such a model, a contrast can "not replicate" only by being confirmed in the opposite direction, never by equivalence, and a "general null" from round 2 can rarely be replicated.

**Adaptation.** The margin is kept exactly, because an equivalence verdict must mean the same thing ("no effect larger than δ nats") in rounds 2 and 3, and a margin chosen with round 3's spreads in view could be tuned to the data. Two additions, fixed now:

1. For every model and contrast, the realized bootstrap SE, the power of the equivalence test at a true zero at Holm's first step and at 0.05, and the largest estimate that could still be declared equivalent, as in round 2.
2. A scale-free descriptive readout: each contrast and its interval divided by the model's SD. It never changes a verdict, but it lets a reader see whether a "not distinguishable" result in a high-spread model is a tight null in standard-deviation units or a wide interval.

For analysis (b) the margin of a pair is the smaller of its two models' margins.

### 7.8 What would count as success, failure and inconclusive evidence

For analysis (a), contrast by contrast, the replication table of 7.3 is the criterion, and the headline is P. In addition, for the account itself:

- **Supported in contemporary models:** H3-P holds (P confirmed below zero) in every core model competent for P, with at least two competent, and no rejection criterion fires for any core model.
- **Rejected in contemporary models:** R1 fires for any core model competent for P, or R2 (restated) fires among the core models.
- **Inconclusive:** anything else, with the reason named (fewer than two core models competent for P; P inconclusive in a competent model; R2 undecided).

For analysis (b), there is no success or failure: each pair and contrast reads "differs" (with or without scale robustness), "similar within the margin" or "not distinguishable", and the reading is a statement about two released models.

For analysis (c), nothing is confirmed or rejected.

## 8. Open decisions for the owner

*All decided on 2026-11-22; see the Decisions section near the top, which governs. The list is kept as first written.*

1. **Gemma generation.** Gemma 4 12B-it (recommended), Gemma 4 E4B-it (wave B's choice; weaker size match), or Gemma 3 12B (older than round 2's newest model; gated, needs the owner's license acceptance and access arrangement; possible only as an exploratory bridge).
2. **Minimum or expanded.** The minimum (about 14 GPU-hours, 48 GB cards) or the expanded study (about 55 GPU-hours, 12–18 of them on 80 GB cards), and whether to freeze both at one tag and stage the runs (recommended).
3. **The reasoning arm.** Whether to include it, which of the three twins, and whether HF `generate` in serve2 is acceptable or a vLLM environment should be built (a third environment with its own equivalence check).
4. **Extra developers.** Whether to add Granite 4.2 8B or Ministral 3 14B to analysis (c) or to the extension family (each adds 10 tests to A3x if added there).
5. **Storage.** Route A (node-local weights with the hash manifest) as wave B decides, or any other route wave B adopts.
6. **The replication rule's thresholds.** Whether "at least two competent models" is the right minimum for a general finding in both rounds, and whether partly-replicated outcomes should be reported under "inconclusive" (as proposed) or as a separate category.
7. **Dependency on wave B.** Round-3 code needs wave B's reviewed scorer changes (above all the BatchEncoding fix) merged first; the owner decides whether round 3 waits for wave B's freeze or takes the reviewed fixes as soon as they are merged.
8. **Ministral reasoning system text.** Whether prepending Mistral's shipped reasoning prompt to the clean text (7.5) is acceptable, or the reasoning model should be run without it and the smoke run used to see whether it reasons at all.

## 9. Sequencing

1. The owner reviews this proposal (now). Round 2 continues independently.
2. Round 2 is frozen at its tag and scored. Nothing in round 3 runs before the tag exists.
3. Wave B's scorer fixes are reviewed and merged.
4. Round-3 code is built (section 6) at a commit after round 2's tag; tests, mutants, the full tokenization gate in serve2, golden renderings, the hash manifests and a mock dry run are completed; an independent review follows; round 3 is frozen at `textcraft-r3-design-frozen-<date>`.
5. The TextCraft environment check runs and its verdict is recorded before any round-3 score is read.
6. Smoke runs (format, timing, loading and mass only), then the core models, then (expanded design) the extension models, the format checks and the reasoning arm.
7. `textcraft_r3_analysis` runs once all scores of a stage are filed; analysis (b) reads round-2 scores only from round 2's filed, frozen runs.
