# TextCraft round 3: independent review 1 (2026-11-22)

Reviewer's read of branch `textcraft-r3` at 451ab3b (checked out as `textcraft-r3-review1`). Nothing was run on a GPU, nothing on ICE was touched, and no round-2 score, analysis output or result note was read. The only data read are round 2's records (`wf/textcraft/data/r2_items.jsonl`) and the committed design notes, code, tests and golden files. Every number below comes either from the code and records or from synthetic values; none comes from a model.

**Verdict: not yet ready to freeze.** The design is careful, the frozen code paths are intact, and the statistics of analyses (a) and (b) are correct as written. Four items must be fixed before the tag, each a small code or rule change, because each leads to a wrong or blocked confirmatory reading on an outcome that is plausible:

1. a 27B or 31B model run on the declared alternative of two L40S cards blocks the whole analysis;
2. a core model that fails its gate leaves H3-P and every replication reading unread for ever;
3. the "mechanical" Gemma fallback rule exists only as a hand-set config value;
4. with round 2's four shards and its three allowed GPU types, the environment check will probably fail for lack of comparable records, for a reason unrelated to software.

The should-fix items are mostly labelling and disclosure. Section 8 gives the recommended rule for the environment-check GPU.

How findings are classed: **must-fix** means the code or rule produces a wrong, blocked or discretionary confirmatory result on a plausible outcome; **should-fix** means a reading could be misread, a disclosure is missing, or a rule is not stated as precisely as the design claims; **note** means something to keep in mind, with no change required.

## 1. Provenance and frozen code

**What was checked.** Each frozen study's files were compared with its own tag by `git diff <tag> HEAD -- <paths>`.

- Against `textcraft-r2-design-frozen-2026-11-23`, everything under `wf/textcraft/`, `tests/test_textcraft_r2.py`, `tests/test_textcraft_dx.py`, `pace/textcraft_r2_score.sbatch`, the round-2 configs and `notes/textcraft_r2_design.md` is unchanged. The only differences under `wf/textcraft/` are the eight new round-3 files (`r3*.py`, `r3_chat_golden.json`, `r3_tokcheck_ice.json`).
- Against `families-waveb-design-frozen-2026-11-22`, `wf/families.py`, `wf/scoring/`, `wf/tools/`, the three wave B test files, `tests/conftest.py`, `pace/family_score_serve2.sbatch` and the `family_*` configs are unchanged (empty diff).

**Shared files: additions only.**

- `wf/constants.py` adds two `REVISIONS` keys (Gemma 4 12B, OLMo 3 7B Instruct) and changes no earlier pin.
- `wf/runs.py` adds `textcraft_r3_*` runner keys. No key collides with an earlier runner, so no frozen run name is redirected.
- `scripts/mutation_check.py` adds the round-3 mutants and keeps round 2's and wave B's mutant lists unchanged. The runner is wave B's strict named-kill version, plus round 2's `MUTATION_FAST_ONLY` screen switch, which an official run never sets. The runner change affects how a mutant is counted, never what a study computes.

Neither frozen study's behaviour can change. The design note's claim in section 12 holds.

## 2. Replication logic (analysis a)

### 2.1 H3-P

The per-model reading (`h3p_reading`, `r3_analysis.py:254-283`) matches design 7.3: *supported*, *rejected* (R1), *measurable only*, otherwise *inconclusive*. The overall "rejected" branch (R2 fires, or R1 fires with none supported) is consistent with R2 as restated: R2 fires exactly when some competent model has P equivalent to zero or above and none is confirmed below.

**Should-fix S1: one competent core model, supported, is read "partial", but the design says "inconclusive".** Take the case where only one core model is competent for P and it is supported, and the other two are not competent (each "measurable only" or inconclusive). The code returns `overall = "partial"` with no reason (line 274), and a test (`test_h3p_reading`, `tests/test_textcraft_r3.py:626-627`) pins that behaviour. Design 7.3, however, defines "partial" as "at least one competent core model supported and at least one rejected or inconclusive", and lists "fewer than two competent core models" as a reason for "inconclusive". Here no *competent* model is rejected or inconclusive, so the design's answer is "inconclusive (fewer than two competent core models)". A newer model failing P's competence gate is plausible (the parent-record yes/no AUC is new to every model), so this case may well occur. "Partial" is never counted as support, so the harm is a mislabelled headline rather than a false confirmation. Either make the code require `len(comp) >= MIN_COMPETENT` before "partial", or amend the design's wording. Whichever is chosen must be fixed before the tag.

**Note N1: the rule is asymmetric.** A single competent core model with P equivalent to zero rejects H3-P overall (through R2), while support needs two. This is inherited from round 2's R2 and is consistent with it. Because equivalence for P is reachable only in low-spread models (design 7.2), the asymmetry will bind rarely. It should still be stated beside the H3-P reading.

### 2.2 The replication rule and round 2's general finding

`round2_finding` (`r3_analysis.py:286-300`) and `replication` (`303-338`) implement the design table exactly, including "partial" for a general finding and for a general null, never counted as replicated. `MIN_COMPETENT = 2` is used in both rounds. Verdicts of models that are not competent are listed and not counted.

**Is the mapping mechanical and fixed before round 2's results?** Yes. It reads only round 2's filed verdict strings and per-contrast competence flags (`read_round2`), and it was committed (799f8c3) before any round-2 result existed. The mapping also agrees with round 2's own across-model reading (`r2_analysis.across`): "general" requires every competent model to be confirmed in one direction, and "model-specific" requires a confirmed verdict in some models and an equivalent or opposite one in others.

**Note N2: the most likely misreading.** One competent round-2 model that is merely inconclusive turns a contrast into "not established". Round 2's own power table makes this likely for any model with Qwen-sized spreads, where P's minimum detectable effect is about 1 nat and equivalence cannot be reached. Round 3 then reads "no replication target" for that contrast. Two things follow. First, the rule is conservative in the right direction. Second, for P the headline of analysis (a) may then rest on H3-P alone. The design states the corresponding consequence for margins (7.2) but not this one; one sentence in 7.3 would prevent a reader from taking "no replication target" as "round 2 found nothing".

### 2.3 Families A3 and A3x

Both call round 2's frozen `family()` over the three tags. A model that is missing or has a NaN p enters with p = 1, and the size is checked to be 30 (`family_a`). That is correct.

**Must-fix M2: when a core model is never scored, H3-P and replication are never computed** (`r3_analysis.py:609`). They are computed only when all three tags have complete scores, and otherwise the output says `waiting [...]`. The test `test_family_a_has_thirty_tests_and_refuses_another_size` (line 717-718) pins that behaviour. Design section 2 says a core model that fails its gate "is not scored, and its tests enter their family with p = 1", so the analysis is meant to proceed without it.

*Failure scenario:* Ministral 3 fails the strict loading check at its first job, so it is never scored. Its 10 tests correctly enter A3 at p = 1. But H3-P overall, R2 and all five replication readings stay "waiting" indefinitely. Making them appear would require a code change after the freeze, decided with the other two models' verdicts in view.

*Fix:* add a mechanical "not scored" state, derived from the filed gate record. A model counts as not scored when its smoke or score run's `tokenization_checks.json` has `ok: false`, or when the loading check raised. Such a model gets verdict "not scored" and competence false for every contrast, and the readings proceed. Do the same for A3x and for B3 pairs, which otherwise read "waiting" for ever. Keep "waiting" only for a model whose run simply has not been filed yet.

**Should-fix S2: the "equivalence underpowered" flag in A3 is computed at the wrong Holm step.** Round 2's `_power_block` uses `Z_HOLM_FIRST = 3.09`, the one-sided quantile at 0.05/50 (`r2_analysis.py:45`). Round 3's families have 30 tests, so the first step is 0.05/30 (z = 2.94). The flag that appends "(equivalence underpowered)" to an inconclusive verdict is therefore slightly too eager. It changes no verdict. Since `r2_analysis.py` is frozen, recompute the power at z = 2.935 in `r3_extras` and use that in round 3's text.

## 3. Older-versus-newer pairs (analysis b)

### 3.1 The standardization

**What the code computes.** For Λ and P, each model's value is the mean of the per-item differences divided by their SD (the paired "dz"). For Dn, Dc and T, it is the goal-minus-ingredient difference divided by the pooled within-role SD. Both are computed from per-goal-pair sufficient statistics (`_cluster_sums`, `_stats`). The pooled SD uses the usual n − 1 weights; a mutant that drops them is killed.

**Well defined?** Yes. A zero SD yields NaN; such a resample is dropped, and a NaN p enters B3 at 1.

**Comparable across models?** Dividing by each model's own item SD removes any multiplicative logit scale. A test confirms that a model whose scores are three times another's gives a standardized difference of exactly zero. This is the same standardizer the forensic audit used ("mean shift over the SD of the item-level shifts"). The audit found that no other standardizer reordered the models.

**Paired bootstrap.** In every resample one draw of goal pairs (`ks`) serves both models, and both models' SDs are recomputed from the resampled pairs (`paired_difference`, `r3_analysis.py:154-180`). The records are the intersection of what both models scored. This is right, and two tests plus a mutant confirm it.

**Is the denominator estimated in a way that biases the difference?** The small-sample bias of mean/SD is of order 1/n. Even counting goal-pair clustering, that is under 1% here, so negligible.

**Note N3: what z measures.** The SD that divides is the spread of the *same* contrast across items. With deterministic scoring there is no measurement noise, so this spread is real item heterogeneity plus bf16 rounding. A pair can therefore read "differs" because the newer model's per-item effects are more or less *heterogeneous*, with the same mean shift relative to its logit scale. The SD is also sensitive to a few extreme items, and newer, more confident models may have heavier tails. Neither point invalidates the choice. The proposal (7.4) had planned a fixed denominator, the model's SD of ΔL over the 1,500 main failures (the SD that sets its margin). Reporting that as one more descriptive column beside the primary z would let a reader see whether a B3 difference comes from the mean or from the spread. This is optional.

### 3.2 The 0.20 standardized-unit margin, and power

**Is the margin defensible?** As a choice, yes. It is fixed before data, scale-free, and of the size the audit treated as "within the range" (Qwen3-8B at about 1.0 against Llama-3.1 at 0.85 differs by 0.17). "Cohen's small" is a loose anchor, because 0.2 in paired dz units is not the same size as 0.2 in between-group d units. The margin is, however, only ever applied to two models' values of the same contrast, so it is internally consistent.

**Is it reachable?** Only for Λ and P, and only if the two models' item values are correlated. The figures below were computed on the real records and goal-pair structure, with synthetic item values. Each model's value was set to a pair effect plus item noise, with intraclass correlation 0.10 (round 1 measured 0.06 to 0.15). The two models were correlated by ρ at both levels, the true difference was zero, and the bootstrap SE was read from `paired_difference` itself (2,000 resamples, 3 replicates; `scripts/review_textcraft_r3_b3_power.py`). Holm's first step in a family of 30 is one-sided z = 2.94 for equivalence and two-sided z = 3.14 for "differs".

The table gives, for each contrast and cross-model correlation, the bootstrap SE of the standardized difference, the power of the equivalence test at a true zero (at Holm's first step and at 0.05), and the smallest difference "differs" detects with 80% power at Holm's first step.

| Contrast (items, pairs) | Correlation between the models | SE | Equivalence power, first step | Equivalence power, 0.05 | Detectable difference, first step |
|---|---|---|---|---|---|
| Λ (990, 162) | 0 | 0.058 | 0.39 | 0.93 | 0.23 |
| Λ | 0.5 | 0.043 | 0.92 | 1.00 | 0.17 |
| P (675, 111) | 0 | 0.068 | 0.02 | 0.81 | 0.27 |
| P | 0.5 | 0.049 | 0.74 | 0.98 | 0.20 |
| Dn or Dc (996, 208) | 0 | 0.098 | 0.00 | 0.31 | 0.39 |
| Dn or Dc | 0.5 | 0.071 | 0.00 | 0.77 | 0.28 |
| T (668, 185) | 0 | 0.125 | 0.00 | 0.00 | 0.50 |
| T | 0.5 | 0.091 | 0.00 | 0.43 | 0.36 |
| T | 0.8 | 0.058 | 0.41 | 0.93 | 0.23 |

How to read this. For Dn, Dc and T, "similar within the margin" is essentially unreachable unless the two models' per-item values are highly correlated, or unless many other B3 tests reject first and Holm's threshold relaxes. So for those contrasts the realistic readings are "differs" (only for differences of about 0.3 to 0.5 SD or more) or "not distinguishable". For Λ, and for P at moderate correlation, both directions are within reach.

**Should-fix S3: disclose this, and carry the flag into the readings.** Each B3 test should report its equivalence power at a true zero (at 2.935 and at 1.645). "Not distinguishable" should become "not distinguishable (equivalence underpowered)" when the power at the first step is below 0.8, as A3's verdicts do. The design should also state, as 7.2 does for A3, that B3 equivalence is unreachable in practice for Dn, Dc and T. The margin itself need not change.

### 3.3 Wording of the pair readings

**Should-fix S4: "newer higher" and "newer lower" are signed, and P is expected to be negative.** "Differs, newer higher" on P means the newer model's standardized P lies *above* the older's: less negative, or positive. In words, it has a *smaller* penalty, or none. A reader skimming the status line (`B3: ... parent_penalty differs, newer higher`) will likely read "higher" as "larger effect". Add the direction in plain words, for example "newer above older (smaller in size)" when both are negative, or add a magnitude column. The same applies to any contrast whose two values have opposite signs.

**Generation claims.** None is made. The readings are keyed by the two tags. The caveat (`PAIR_CAVEAT`) and the list of what changes within each pair (`PAIR_CHANGES`) travel with the output, and the design forbids trend summaries. One small inconsistency (note N4): `PAIR_CHANGES` gives Qwen3.5-9B 9.7 B parameters, while its spec says 9.41 B.

### 3.4 Old scores and provenance

**Should-fix S5: `b3_old_scores` is a free config value** (`r3_analysis.py:638, 660, 764`). The design says the round-2 scores are used if the environment check passes, and the serve2 rescores only if it fails. The code uses whichever the config names, even "rescore" after a passing check. Derive the value from the environment check instead: round 2's scores if it passed; the rescores, which must exist and be complete, if it failed or was not run; and refuse a config that disagrees.

**Should-fix S6: round 2's scores are read for B3 without round 2's provenance check** (`r3_analysis.py:768`). The design says they are read "under round 2's provenance check". Call `r2_analysis.provenance([...old tags...], r2_root)` and refuse on any problem. Also record in round 3's metrics the sha256 of round 2's `metrics.json` and its status line, so it is clear which round-2 analysis the replication rule read.

**Should-fix S7: `b_draws` can override B3's 20,000 resamples** (`r3_analysis.py:647`). The knob is a test convenience. A real run should refuse any value other than `B_DRAWS` unless `WF_MOCK_SCORE=1`.

**Should-fix S8: the environment check does not cover the Gemma pair's older model.** The check rescores Qwen3-8B and Mistral-7B-v0.3 only, as wave B did. Gemma-2-9B-it differs from both in exactly the ways a software change could break: eager attention with attention-logit soft-capping, final-logit capping, and the folded system text. Its serve2 renderings already match round 2's (section 13), and an envcheck config for it is one line, with its rescoring runner already present. Either add `textcraft_r3_envcheck_gemma2-9b-it` (about 0.3 GPU-h) and require it for the Gemma pair only, or print beside the Gemma pair that the environment check did not cover Gemma 2. This is not an expansion of the study; it is about whether the third B3 pair is measured validly.

## 4. Competence and margins

The transfer argument in design 7.1 and 7.2 is sound. The AUC is rank-based, so it is scale-free. The two Yes-share floors guard against degenerate one-word models, which is a model-independent concern. The mass is a format question, checked first on the CPU and again by the weights. Keeping round 2's thresholds exactly makes "competent" mean the same thing in both rounds, and the stricter 0.80 floor shows which readings sit between 0.70 and 0.80. The margin is kept exactly, and its consequence (equivalence unreachable when the cap binds) is stated. No change is needed.

Two notes. N5: the competence AUC is a point estimate with no interval, so a model at 0.70 to 0.72 can flip between runs; the 0.80 floor column already covers this. N6: Gemma 4's 1,024-token windows are exceeded by 99% of inputs; that is part of the model and is named beside it, not a reason to change the gate.

## 5. Gemma fallback

**Must-fix M3: the rule is not implemented as code.** Design section 2 calls it mechanical. In the code, the Gemma slot comes from `cfg["gemma_core"]` (`r3_analysis.py:624, 716`), set by hand. Nothing reads the smoke run's filed record, nothing computes "the median yes/no mass over the smoke run's yes/no prompts", and nothing refuses a full run of `gemma4-e4b-it` when the rule did not fire. The runner and config exist, and `RUN_ORDER` does not include E4B.

*Failure scenario:* both Gemma 4 12B and E4B get scored (E4B "as a backup", since GPUs were free). The analyst sets `gemma_core` to whichever gives the cleaner A3 and B3 readings. The code accepts it silently; only a deviation entry, if anyone writes one, would show it.

*Fix:* write a function that reads `runs/textcraft_r3_smoke_gemma4-12b-it/shard_00_of_01/`. It should take the loading record, the gate's `ok`, the attention-agreement `ok`, and the median over every yes/no score in `scoring.jsonl` of `exp(logp_plus) + exp(logp_minus)` (say "every yes/no arm", since round 2's mass is on `yn_neutral` only). From these it returns the slot. A smoke run that crashed before writing its gate record counts as a loading failure. The analysis then refuses a `gemma_core` that disagrees with the function, and the E4B score run refuses to start unless the function returns E4B. Also state what happens if E4B fails its own smoke gate: the Gemma slot is then "not scored" (M2).

**Should-fix S9: the mass criterion overlaps round 2's competence gate.** Median P(Yes) + P(No) ≥ 0.5 is criterion 4 of round 2's overall gate, and part of every contrast's competence. A Gemma 4 12B that puts its probability elsewhere would, under round 2's rule, be a *non-competent model*. That is a finding. The fallback instead replaces it with a model that may be competent, and does so for the Gemma slot only: Ministral 3 or Qwen3.5 with low mass are scored and found not competent. The loading, gate and attention triggers are engineering failures and are sound. The mass trigger is sound only if a low mass means a *format* mismatch, where the model wants to emit something other than a bare answer after the empty thought channel.

To keep it mechanical and avoid selecting on competence, either drop the mass trigger and let the gate decide (the proposal's own alternative, 7.2 item 2), or make it diagnostic. In the diagnostic version, fire only if the mass is below 0.5 *and* on most smoke yes/no prompts the model's most probable next token is neither "Yes" nor "No" but a structural token (a channel or turn marker, whitespace, markdown). Record the same diagnostic descriptively for every core model.

## 6. Thinking arm

**The cap and how it is closed.** `max_new_tokens = 2,048`, with stopping at `</think>`. A trace that never closes gets `\n</think>`, then the readout's blank line, and is flagged `forced`. This matches design 8.2. Text after the closing marker is dropped.

**Should-fix S10, four small items.**

- (a) `forced` also marks a trace that ended its turn (`<|im_end|>`) without closing its reasoning (`r3_think.py:69-72`). That is not budget forcing. Record the two cases separately, for example `forced: "cap"` and `"ended unclosed"`.
- (b) Generation uses left-padded batches of 8 through Qwen3.5's linear-attention layers, in the PyTorch fallback (no `kernels`). Whether padding is masked correctly in those layers has not been checked. A failure would not show in any gate, because the readout is scored unbatched. Add to the thinking smoke run one check: the next-token logits after one prompt, alone and left-padded inside a batch, agree within the attention-agreement tolerance.
- (c) The arm has no prompt check at job start. `THINK_SPEC.thinking_on_ending` (`<|im_start|>assistant\n<think>\n`) is declared but never checked. The CPU compat check covered it once, but nothing does so in the job. Check every prompt's ending and the forbidden texts before generating.
- (d) Reproducibility depends on the shard count and the batch size (which prompts share a batch, and so the padding and the random stream), and GPU sampling is not bitwise deterministic. State both in 8.2.

**The readout position** is right: prompt, then reasoning up to and including `</think>`, then `\n\n`, then the candidate. It re-tokenizes the text rather than reusing the generated ids, which is acceptable, and the boundary is recorded per item.

**Exclusion from the families** holds: `think_block` is called only inside `exploratory`, and no family function sees the arm.

**Should-fix S11: exploratory blocks carry p-values.** `think_block` (line 534) and the size comparisons (lines 671-673) call `paired_difference`, which returns `p_differs` and `p_equivalent` (the TOST against 0.20). Round 2's audit held to the standard that no p-value appears outside the family. Strip both from exploratory outputs, or rename them as non-inferential.

**Note N7: cost.** 4,530 generations of up to 2,048 tokens each, with linear attention in PyTorch, may exceed the 3 to 6 GPU-hour estimate. The run resumes per batch, so this costs time but not validity.

## 7. Engineering

**Must-fix M1: the declared two-L40S alternative fails the analysis's provenance check.** `provenance.run_record` reports a shard's GPUs as a *set* of names (`wf/provenance.py:53`), so a job on two L40S is recorded as `["NVIDIA L40S"]`. `provenance_problems` (`r3_analysis.py:568`) passes that through `gpu_class`, gets "NVIDIA L40S", and finds it is not in `GPU_PIN["qwen3.8-27b"] = ("A100 80GB", "2x NVIDIA L40S")`. `run_textcraft_r3_analysis` then returns 1 *before writing any metrics*, so A3, A3x and B3 are all blocked. The job-start check (`gpu_check`) uses the raw `nvidia-smi` list, so it passes. The failure shows up only at analysis time, after the GPU hours are spent.

Reproduced by `scripts/review_textcraft_r3_checks.py`: a shard whose `jobs.jsonl` records `["NVIDIA L40S", "NVIDIA L40S"]` gives the problem "qwen3.8-27b: GPU ['NVIDIA L40S'] is not the pinned ('A100 80GB', '2x NVIDIA L40S')".

*Fix:* in `provenance_problems`, read each job's own GPU list from `jobs.jsonl`. Require every job of every shard of a model to have the identical list, and require `gpu_type_ok` on it. Add a test with a two-L40S shard. Separately, the commit check should not stop A3 because an extension shard has a problem: report per family, and refuse only the affected family.

**Should-fix S12: the one-commit rule will block on ordinary note commits.** All shards of every round-3 model and format check must share one commit (`r3_analysis.py:561`). The core runs come first and the extension "as GPUs allow", possibly days later. Any commit pulled into the ICE checkout in between (a README line, a design-note result paragraph) gives the extension shards a different commit, and the whole analysis refuses. A post-freeze deviation fix would also require rescoring the core. Fix it by rule and by a check. All round-3 GPU jobs run from a checkout at the freeze tag's commit (detached, never pulled). The analysis checks `commit == <tag commit>` for every shard (environment check and thinking arm included, which the current check skips), or, if a deviation is ever needed, it accepts commits whose diff from the tag touches no file under `wf/`, `scripts/`, `pace/` or `wf/config/`.

**Checked and found sound:**

- **Tokenization gate:** on every record and arm before the first score and on every resume, through the scorer's own `_encode`.
- **Arm isolation:** the common prefix and suffix are stripped, and what differs must decode to text inside each arm's goal line. That is weak in principle but adequate, and its mutants are killed.
- **Golden renderings:** compared by text, id count and id hash for all twelve tags. Environment check part 1 is pinned by a test.
- **Weights manifests and prefetch route A:** `r3_prefetch.py` uses wave B's `prefetch` with the committed manifest, refuses `/storage`, checks free space, and hashes every file. A failure exits non-zero before Python loads anything, and the job re-checks the record in `run_gate`.
- **`gpu_check`:** refuses an unpinned type at job start. Earlier models are recorded only.
- **`attention_agreement`:** five main failures, both formats, each switch confirmed on the config, the cache cleared.
- **Resume:** refuses a different commit, config or input, and refuses saved scores whose record hash or model differs. The gate is re-run before more scores.
- **Sharding:** by goal pair, as in round 2.

**Tests.** `tests/test_textcraft_r3.py` was run locally (Python 3.14, the project's `.venv`): **52 passed** in 52 s, none skipped.

**Mutants.** See section 9.

## 8. The environment-check GPU: recommended rule

**What the current rule needs.** It needs at least 200 of the 300 seeded records to come from round-2 shards that ran on the same GPU type as the serve2 rescoring. Round 2 scores each model in four shards (`n_shards: 4`, sharded by goal pair), and its job script allows L40S, A40 or A100. The 300 seeded environment-check records fall in round 2's shards as follows (computed from the records by `scripts/review_textcraft_r3_envcheck_shards.py`):

| Round-2 shard | Seeded records | Of which main failures |
|---|---|---|
| 0 | 75 | 39 |
| 1 | 55 | 25 |
| 2 | 68 | 47 |
| 3 | 102 | 37 |

**Why this is a problem.** Unless all four shards ran on one type, reaching 200 needs three shards on one type, and not shards 0, 1 and 2 (together 198). Any two shards give at most 177. A mixed placement is quite possible under round 2's constraint, and it would then "fail" the environment check for a reason unrelated to software. That forces the 7-GPU-hour fallback and changes B3's older scores. The analysis code also assumes one type for the whole rescoring (`envcheck_block`, line 426: a single `want`), so no multi-type remedy can be added after the freeze without a deviation. This is **must-fix M4**: fix the rule and the code now.

**Recommended rule (fixed now, applied mechanically once round 2's scoring is filed):**

1. **Plan from metadata only.** Before any environment-check job, read the GPU names recorded in each round-2 shard's `jobs.jsonl` for Qwen3-8B and Mistral-7B-v0.3. These are names, not scores. A shard is *typed* when every one of its jobs recorded the same single-GPU list. Otherwise (a resume on another type, or no record) it is *untyped*, and its records are not comparable. Commit the plan (shard, GPU name, number of seeded records) before submission.
2. **Mirror round 2's shards.** Run the environment check as one job per typed round-2 shard. Shard i scores the seeded records that lie in round 2's shard i, using round 2's own partition `r2_score.shard_records(all_records, i, 4)`, not a re-partition of the 300. Each job is pinned to that shard's exact GPU name, by `--constraint` and by a job-start check that refuses any other name. Each record is compared with the rescoring from its own shard's job. The cost is four jobs of a few minutes each instead of one.
3. **Shortcut.** If all typed shards share one type, a single job on that type is equivalent (the current code path).
4. **Pass rule.** Wave B's thresholds on the union of comparable records, with at least 200 comparable. With every shard typed, all 300 are comparable.
5. **Mechanical fallback.** If fewer than 200 records are comparable (untyped shards, or a needed GPU type not obtainable within 7 days of the first submission), the environment check is recorded as **"not run (GPU types)"**, not "failed", and B3 uses the declared fallback (full serve2 rescoring of the three older models, on L40S like the round-3 models). The distinction matters in the report: a software difference was not detected; the check simply could not be run.

**Code needed:** `envcheck_block` must match the GPU per record (the old shard's list against the new shard that scored that record), instead of one `want`. The envcheck run must accept a shard index that selects round 2's shard, and must refuse a GPU name other than the planned one.

(If any round-2 shards of these two models have not been submitted yet, submitting them with `--constraint=gpu-l40s` would make all of this moot. That is an operational choice for round 2's owner, not something round 3 can require.)

## 9. Mutants of my own (strict rule)

**How they were run.** Eleven mutants, chosen to probe code the round-3 mutant list does not target. They were applied with the project's own runner (`scripts/mutation_check.py` loaded by `scripts/review_textcraft_r3_mutants.py`) under the strict rule: a mutant is killed only when pytest exits 1 and names a failing test. A screen run used the covering test files only (`MUTATION_FAST_ONLY=1`). Every survivor of the screen was then rerun officially, which means the covering files first and then the whole suite.

| Mutant | What it breaks | Result |
|---|---|---|
| rev1 | the standardized SD divides by n, not n − 1 | killed (`test_stats_match_the_definitions`) |
| rev2 | the environment check's goal effect is taken over every chosen record, not only main failures | **survived** the whole suite |
| rev3 | the environment check's yes/no share criterion is switched off | **survived** the whole suite |
| rev4 | the thinking arm's "unforced" subset keeps forced traces | **survived** the whole suite |
| rev5 | every thinking batch gets the same sampling seed | **survived** the whole suite |
| rev6 | provenance accepts a model whose scores carry mixed revisions, if the pinned one is among them | **survived** the whole suite |
| rev7 | B3's confirmatory tests use 2,000 resamples instead of 20,000 | **survived** the whole suite |
| rev8 | the attention-agreement check uses one item instead of five | killed (`test_attention_agreement_rule`) |
| rev9 | the environment check accepts rescoring shards that ran on different GPUs | **survived** the whole suite |
| rev10 | the thinking trace keeps the text after `</think>` | killed (`test_budget_forcing_and_the_readout_point`) |
| rev11 | the thinking jobs are not put in record-and-arm order (batch composition) | **survived** the whole suite |

**Totals.** 3 of 11 were killed by named tests. The other 8 survived the whole suite, with no timeout and no unrunnable suite. Logs (the screen first, then the four official shards): `notes/textcraft_r3_review1_mutation_log.txt`.

**Should-fix S13: add tests for these gaps.** Each survivor is a rule written in the design that no test pins:

- the environment check's main-failure restriction, its yes/no share criterion and its single-GPU requirement (rev2, rev3, rev9);
- the unforced subset of the thinking arm (rev4);
- the per-batch seeds and the job order, which carry the arm's reproducibility (rev5, rev11);
- the revision check catching a *mixed* set (rev6);
- B3's resample count (rev7; the fix is S7).

None of these mutants changes a result that exists today. They show where a later edit could silently weaken a rule.

## 10. Summary of items

| Id | Class | Where | Item |
|---|---|---|---|
| M1 | must-fix | `r3_analysis.py:568`, `provenance.py:53` | Two-L40S alternative fails provenance; the whole analysis exits before writing metrics |
| M2 | must-fix | `r3_analysis.py:609` | A core model that fails its gate leaves H3-P, R2 and replication "waiting" for ever |
| M3 | must-fix | `r3_analysis.py:624, 716` | The Gemma fallback rule is a hand-set config value, not code |
| M4 | must-fix | `r3_analysis.py:421-471`; design 8.1 | Environment check: with round 2's four shards, at least 200 comparable records is unlikely unless all shards share a type; fix the rule and the code (section 8) |
| S1 | should-fix | `r3_analysis.py:274` | One competent model supported reads "partial"; the design says "inconclusive" |
| S2 | should-fix | `r2_analysis.py:45` via `model_block` | A3's underpowered flag uses 0.05/50, not 0.05/30 |
| S3 | should-fix | `pair_reading`; design 7.4 | B3 equivalence power not reported or flagged; unreachable for Dn, Dc, T |
| S4 | should-fix | `r3_analysis.py:391-393` | "Newer higher" on a negative P means a smaller penalty; say so |
| S5 | should-fix | `r3_analysis.py:638, 660, 764` | `b3_old_scores` is free; derive it from the environment check |
| S6 | should-fix | `r3_analysis.py:768` | Round-2 scores read for B3 without round 2's provenance check |
| S7 | should-fix | `r3_analysis.py:647` | `b_draws` can lower B3's resamples in a real run |
| S8 | should-fix | design 8.1 | Gemma 2 not covered by the environment check |
| S9 | should-fix | design section 2 | The mass trigger of the Gemma fallback overlaps round 2's competence gate |
| S10 | should-fix | `r3_think.py` | Forced flag mixes two cases; padded batching unchecked; no prompt check; reproducibility conditions |
| S11 | should-fix | `r3_analysis.py:534, 671-673` | p-values in exploratory outputs |
| S12 | should-fix | `r3_analysis.py:561` | One-commit rule blocks on any commit between stages; pin jobs to the tag commit; include the envcheck and thinking runs |
| S13 | should-fix | `tests/test_textcraft_r3.py` | Tests for the reviewer's surviving mutants (section 9) |
| N1-N7 | note | | Asymmetric H3-P; "not established" likely; what z measures; 9.7 B vs 9.41 B; AUC point estimate; Gemma windows; thinking cost |
