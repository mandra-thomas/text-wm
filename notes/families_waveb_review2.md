# Model families, wave B: review 2, the section 8 amendment (2026-11-23, before any wave B score)

This is an independent review of the amendment that follows the failed environment check. It covers branch `families-waveb-serve2-rescore` at 901bbd4, against the freeze tag `families-waveb-design-frozen-2026-11-22` (f6bbffb). The diff reviewed is `git diff families-waveb-design-frozen-2026-11-22...901bbd4`: the amendment in section 8 of [families_waveb_design.md](families_waveb_design.md), `wf/families.py`, the five `wf/config/family_rescore_*.yaml`, `families_waveb_analysis.yaml`, `wf/runs.py`, the two job scripts, `wf/tools/envcheck_tokens.py --rescore`, `wf/tools/prefetch_models.py`, the Qwen3-32B manifest, the tests and the mutant list. The reviewer worked read-only on ICE: no job was submitted and no file was changed there. No GPU was used.

## Verdict

**Not yet ready to re-freeze, but close.** The design is sound: rescoring every older comparison model in serve2 does put every comparison in one environment. The rescoring list is complete. The gate has no path that lets a partial or failed rescoring through, and no path that falls back to the original scores. The frozen family, thresholds and predictions are unchanged in effect.

Three things block the freeze:

- The description of the diagnostic contains claims the data do not support. One of them contradicts the frozen design's own cross-GPU numbers.
- Four sections of the design note still state the old rule. Once frozen, the document would contradict itself.
- 13 of my 25 mutants survived the wave B test files. Three of them change what a real rescoring scores or loads, and no test notices.

All of these can be fixed in an hour or two, with no GPU time.

## Must-fix

**M1. The diagnostic is described more favourably than its data allow** (section 8, "The cause, found with a precision diagnostic", and `RESCORE_AMENDMENT` in `wf/families.py`; the same sentence is in the `deviations.md` entry on `main`). I checked every number against `text-wm-fix45/runs/families_envcheck_precision/report.json`, its `old.jsonl` and `new.jsonl` (job 7248092), and `text-wm-fix39/runs/families_envcheck/metrics.json` (job 7247973). The numbers themselves are right: 300 of 300 scores reproduced in each environment, 450 of 450 prompts with identical ids, 205 of 207 changed logits equal to their rounded float64 values, 100 of 101 changed scores predicted, 0.217 / 0.16 / 0.96, a median relative logit change of 0.44%, and the per-condition means and standard errors. Four statements are not right.

1. *"perturbs the final hidden state by about one bf16 epsilon, as a change of GPU type does."* The second half is contradicted by the frozen design's own basis for the thresholds. In the archived Qwen3-8B runs, scoring the same prompts on different GPU types moved the yes/no score by 0.035 nats per score. The stack change moved it by 0.248, about seven times as much. I also recomputed the cross-GPU comparison in the old environment: `rq1v_score` against `size_score_qwen3-8b`, with the same prompts, 14,582 yes/no scores. 91% of the scores were identical to the last bit (mean absolute difference 0.035). Across the stack change only 30% were identical. Whatever a GPU change does to the hidden state, it moves Qwen's yes/no scores far less than the stack change does.
   - Failure scenario: a reader of the frozen note concludes that the environment difference is the same tolerated kind as the GPU differences already present in every earlier comparison. That is the conclusion the check was built to test, and it failed.
   - Fix: delete "as a change of GPU type does". State the factor plainly: per score, about seven times the cross-GPU difference in yes/no and about twice in two-candidate (0.341 against 0.18).

2. *"about one bf16 epsilon (0.39%)".* The machine epsilon of bf16 is 2⁻⁷ = 0.78%. The value 0.39% is 2⁻⁸, the unit roundoff, which is half the spacing. Fix: write "about half a bf16 ulp (the unit roundoff, 2⁻⁸ = 0.39%)", or "about 0.4% of the logit". Also say that what was measured is the change in the float64 logits. The hidden-state change is inferred from it: the same weights in float64 give different logits, so the hidden states differ. It was not measured directly.

3. *"Mistral 7B's logits are smaller, so the same kind of perturbation moved its scores less."* This was not measured. The Mistral mode of the diagnostic (commit 445b179) was never run: `families_envcheck_precision` holds Qwen3-8B only, and I found no other run on ICE. It also understates how close Mistral came: its yes/no per-score difference was 0.0688 against a limit of 0.07. Fix: mark the sentence as an untested explanation, and report Mistral's margin.

4. *"The perturbation is not tied to one condition."* The per-condition float64 means are quoted (−0.059 aligned, +0.054 swapped), but their consequence is not. On the diagnostic's 50 items, the yes/no goal effect (aligned minus swapped) moves by −0.11 nats in float64 (standard error 0.053) and by −0.125 in bf16 (0.064). That is larger than the check's 0.10 limit, at about two standard errors. On the full 300 items it moved +0.026. The 300-item figure is the better estimate, so the subset's shift is probably noise, but the note should say so rather than leave it for the reader to compute. Fix: add one sentence giving the 50-item shift with its standard error beside the 300-item value.

None of this changes the decision, because rescoring every older model makes the cause irrelevant to the comparisons. It matters because the text will be frozen.

**M2. Sections of the frozen note still state the rule the amendment replaces.** Section 14 has a dated note. These do not:

- Section 3, "Reference models (already scored with exactly the same prompts; read, not rescored)". They are now rescored.
- Section 4, "a wave A or reference model is scored there only by the environment check (section 8)". The rescoring scores them there too.
- Section 11, the outcome-table row "The environment check fails | No wave B result is read until the cause is fixed and documented … The analysis writes no wave B number". After the amendment the check has failed, and the analysis does write wave B numbers once the rescorings pass.
- Section 18's S5 description ("no wave B score is read unless the environment check passed").

Failure scenario: someone reading section 11 of the frozen design sees that a failed environment check means no wave B number, and then finds a report full of them. Fix: add a dated pointer to the section 8 amendment in each place, and rewrite the section 11 row along the lines of "fails → every older comparison model rescored in serve2; wave B read only when every rescoring passes `rescore_check` (amendment 2026-11-23)". Also say, in section 8 or 9, that the label `not interpretable (environment check)` now means "the serve2 rescorings have not passed". The label is kept so that the frozen outputs do not change, but its meaning has.

**M3. Nothing in the suite pins what a rescoring actually scores and loads** (details under point 7). Three of my mutants change exactly the like-for-like property the amendment rests on, and survive:

- *"Rescoring prompts not folded (Gemma 2)"*: in `run_family_rescore`, `family_messages(spec, t, arm)` becomes `sizes.arm_messages(t, arm)`.
- *"Rescoring loads with the original spec's settings"*: `open_scorer(spec)` becomes `open_scorer(spec_for(tag), spec.tag)`. That drops `strict_load` (the loading check) and, for Mistral, `mistral_format=False`.
- *"Token dump without the fold"*: in `envcheck_tokens.dump`, `family_messages(spec, t, arm)` becomes `sizes.arm_messages(t, arm)`.

`test_rescore_spec_keeps_the_earlier_prompts_and_loads_as_wave_b` checks `rescore_spec`, but nothing checks that `run_family_rescore` and the token dump use it. The mock scorer ignores the tokenizer settings and the loading check, and the tests never compare the prompts.

Failure scenarios:

- The first and third mutants would fail loudly on ICE today, because Gemma 2's template raises on a system turn.
- The second would fail loudly for Mistral, because `MistralCommonBackend` refuses the gate's reply render.
- But for Qwen3-8B, Qwen3-32B, Llama and Gemma 2, the second mutant only drops the loading check. That is silent. If transformers 5's renaming of an older architecture's checkpoint tensors ever left a layer at its random initialisation, the reference scores would be garbage. Every test would pass, and so would the gate, because the gate never looks at `strict_load`.

Fix: add a mock-mode test that runs `run_family_rescore` for every rescoring model, through a spy on `sizes.score_shard` (or `LogoddsScorer.__init__`), and asserts:

- the scored messages equal `family_messages(spec_for(tag), t, arm)`;
- the scorer gets the `rescore_spec` settings (`tokenizer_kwargs`, `attn_implementation`, `strict_load=True`, `template_kwargs`, `model_tag`).

Assert the same messages for `envcheck_tokens.dump` on a stub tokenizer. Also record `strict_load` and `load_info["ok"]` in the shard's gate record, and require them in `rescore_check`.

## Should-fix

S1. **Llama-3.1-8B's original-environment column is not a pure environment comparison.** Prompts, few-shot examples, template, `enable_thinking=False`, revision and item population are the same. Three things are not:

- **The yes/no scorer path.** `rq1v_score` scored "Yes" against "No" with `score_pair`: two forward passes over the prompt plus the candidate. The rescoring uses `score_one_token_pair`: one pass over the prompt. These are equal in exact arithmetic, but the kernels see different sequence lengths.
- **The GPU types.** The rq1 runs ran on mixed GPU types, and none was recorded (`jobs.jsonl` gpu is empty).
- **The revision.** The rq1 runs loaded an unpinned snapshot. That snapshot is the pinned one only because the cache held a single snapshot: I checked on ICE that `hfcache/models--meta-llama--Llama-3.1-8B-Instruct` holds exactly `0e9e39f…`. The same holds for `size_score_qwen3-8b` and `size_score_qwen3-32b`.

In the old environment, the Qwen3-8B analogue of the first two together (`rq1v_score` against `size_score_qwen3-8b`, yes/no) gives a mean absolute difference of 0.035 with 91% identical. That is small, but it is not zero. Fix: say in the amendment that Llama's environment-sensitivity numbers also contain the scorer path and mixed GPU types. Do the same for Qwen3-8B's original column in two-candidate format if its shards' GPU types are not all L40S. Then nobody reads Llama's column as the size of the stack effect.

S2. **No pre-registered reading for an environment-sensitive reference.** The gate checks tokens, completeness, GPU type, the gate pass, revision and hashes. It never looks at whether the serve2 scores are sane relative to the original ones. Suppose transformers 5 silently mishandled one older architecture, for example Gemma 2's attention-logit soft-capping in the eager path. Gemma 4 is a different class, so only Gemma 2 would be hit. The gate would pass, and the Gemma generation pair would be read against a broken Gemma 2. The only trace would be a large number in the descriptive `environment_sensitivity` column, with no rule that acts on it.

Fix: fix a rule now, before any data, in the style of the one-fifth rule. For example: if a reference's item share moves by more than 0.02, or its yes/no ΔL by more than 0.10 (the check's effect-level limits), the planned pair involving it is reported as "environment-dependent", with both readings. The reading stays the serve2 one. Also add the competence verdict, the item share's CI and the AUC to `environment_sensitivity`. A reference's interpretability, and with it a planned pair's, now comes from serve2. Wave A's margins are wide (AUC 0.94 to 0.99), but the column should show it.

S3. **The analysis does not tie `reference` to `rescore` in code.** `rescore_gate` checks `cfg["rescore"][tag]["run"]`, while the scores come from `cfg["reference"][tag]`, a separate entry. Only the configuration test ties them together. Failure scenario: the analysis is run with an edited or overridden configuration in which `reference.qwen3-8b` reads `size_score_qwen3-8b`. The gate still passes on the rescoring, and every Qwen comparison silently reads original-environment scores. My mutant of this kind was killed only through the configuration file. Fix: in `run_families_analysis` for wave B, require `cfg["reference"][t] == {"model": t + RESCORE_SUFFIX, "runs": {cfg["rescore"][t]["run"]: list(ENVCHECK_ARMS)}}` for every `t` in `RESCORE_MODELS`, and block otherwise. Add a test.

S4. **The token comparison carries no provenance, and the gate trusts any file with `ok: true`.** `compare_<tag>.json` records no commit, and `rescore_check` does not check that its left side is the study stack (transformers 4.51.3, tokenizers 0.21.4) and its right side transformers 5.x. If the comparison were run at a commit whose prompt code differs from the rescoring's, it would not cover the prompts the rescoring scored. If both dumps came from one stack, it would be trivially identical. Fix: have `dump` write `git_sha`. In `rescore_check`, require `left.transformers == "4.51.3"`, a right-hand major version of at least 5, both revisions equal to the pin, and the comparison's commit equal to the rescoring's commit (or to the re-freeze tag).

S5. **No coverage or item-identity check on the rescorings.** `complete_shards` checks status and commit, not that every (item, condition) of `size_jobs` is present. `compare_models` silently pairs on the intersection. Runs are copied between checkouts (`text-wm-fix23`, `fix39`, `fix45`), so an item-set drift between the rescoring's checkout and the analysis's would shrink the comparison without a word. Fix: in `rescore_check`, require the record count to equal `len(size_jobs(...))` on the analysis's items, and every record's `item_hash` to equal `_item_hash` of the analysis's item. Wave B's own runs have the same gap from before the amendment; the same check would close it for them too.

S6. The configuration comment in `families_waveb_analysis.yaml` says "pace/families_envcheck_tokens.sbatch with ENVTOK_TAGS and ENVTOK_OUT=rescore_tokens". The script reads `ENVTOK_RESCORE=1`. Correct the comment.

S7. Two smaller wording points:

- "the same load-time settings… TF32 off". The report shows matmul TF32 off but cuDNN `allow_tf32` true in both environments. Write "TF32 matmuls off in both".
- `PAIR_CAVEAT` and `QWEN3_SIZE_SERIES` quote Qwen3 shares (0.49 / 0.89 / 0.70 / 0.71) from the original environment. They are still the right basis for B1 and the caveat, but say "original environment", since Qwen3-8B's and 32B's serve2 shares will be printed beside them.

S8. **The per-shard aggregation in `rescore_check` is untested.** Five of my mutants survive: `any` in place of `all` for the GPU type, the gate and the weight hashes; a shard that ran on two GPU types accepted; and a second spec or revision tolerated (`<=` in place of `==`). Every test builds one-shard rescorings, so `all` and `any` coincide. Qwen3-32B really is rescored in two shards, and it is the one model whose weights are hashed. Failure scenario: shard 1 is resumed by hand with a different `WF_WEIGHTS`, or on another node. The job-start checks stop that today, but under these mutants the analysis would not. Fix: build a two-shard rescoring in the test, make one shard fail each check in turn, and assert that the check fails.

S9. **The descriptive original-environment outputs are only checked for their keys.** Two mutants survive: `environment_sensitivity` comparing the original with itself, and `comparisons_original_environment` reading the serve2 scores. In mock mode the original and serve2 scores are identical (the mock is deterministic), so no test can tell them apart. Fix: perturb the mock original-environment records (for example, add 0.5 to every L of `size_score_qwen3-8b`), then assert `serve2_minus_original` and a changed `comparisons_original_environment` entry.

S10. Three minor survivors: a token comparison with no `ok` key accepted (`is not False`); a gate record without `environment.checked` accepted; and a mock gate record accepted without a mock environment. None is reachable by the current code paths: `compare` always writes `ok`, `require_environment` always sets `checked`, and mock runs write to `runs_mock`. A one-line test each would close them.

S11. The token CPU job's estimate ("one to three hours") is high. The two-model run took 5 minutes 32 seconds (job 7240543), so five models should take about 15 to 20 minutes, plus building the old stack and downloading Qwen3-32B's tokenizer files. The 3-hour limit is fine. Lower the estimate so that a slow job is noticed.

## Answers to the eight questions

**1. Is the rescoring list complete? Yes.** I traced every read in `run_families_analysis` for wave B.

- **Per-model blocks and share tests.** These read `family_score_<tag>` for wave B tags only.
- **References.** `cfg["reference"]` holds exactly the five rescoring runs, with model `<tag>-serve2` and the six arms.
- **Comparisons.** `pairs` is built from `scores` (wave B) and `ref_scores`, the serve2 rescorings. It covers each wave B model against Qwen3-8B and Llama-3.1-8B, the five planned pairs, and the three descriptive pairs, which are all wave B against wave B.
- **The primary family.** `primary_family` reads `report["models"]` and `report["reference"]` (serve2) together with `report["comparisons"]`.
- **The checks and predictions.** The placement and template checks read wave B runs only. `b3_fallback` reads wave B scores only. `waveb_predictions` reads the family.
- **Original-environment scores.** These enter only `environment_sensitivity` (the reference models' own blocks) and `comparisons_original_environment`, which are descriptive and withheld with the rest.

I found no other path that reads an original-environment score. The planned pairs' older members (Gemma 2, Mistral 7B, Qwen3-8B, Qwen3-32B) and the two descriptive references (Qwen3-8B, Llama-3.1-8B) are exactly `RESCORE_MODELS`. Each rescoring covers every arm of `size_jobs` on all 4,666 primary items and 292 successes (43,746 forward passes), which is what each comparison reads. The only gap: the list is enforced against the configuration by a test, not in code (S3).

**2. Are the prompts like for like? Yes for the prompts; not fully for Llama's original column (S1).**

- `rescore_spec` keeps each model's `hf_id`, pinned revision, `system` placement (Gemma 2 folded), attention (`eager` for Gemma 2; Gemma 2's soft-capping is in its config at the pinned revision), and `template_kwargs`, so the prompts come from `family_messages(spec)`.
- For Llama-3.1-8B, I checked the original runs' code at their commits (`rq1_score` de40795, `rq1x_score` 39722d6, `rq1v_score` 2155b8d) against the current code. `rq1.arm_messages` and `rq1v.arm_messages` are what `sizes.arm_messages` calls. The prompt-building code changed only in `_replace_task_line`, which now raises where it used to prepend. Any prompt that scores at all is therefore byte-identical. The same few-shot set (set 1) and the same `enable_thinking=False` are used.
- The size runs (8bfd84c) and wave A's Gemma 2 run (65eb591) build prompts with the same functions as now.
- The scorer path differs for Llama's yes/no (S1), and Mistral's tokenizer is read through a different backend. Its ids were identical in part 1, and the new token job re-checks all five models.
- The like-for-like settings themselves are not pinned by a test (M3).

**3. Is wave B gated correctly? Yes; no path found.** `rescore_gate` requires all five tags in the configuration and every `rescore_check` to pass. `blocked` depends only on that, and blocking runs `apply_environment_gate` and `withhold_waveb_readings` as before. Specifically:

- An incomplete shard fails `complete` and is listed as waiting.
- A missing reference run leaves its comparisons out, so the tests show "waiting" at p = 1.
- Nothing falls back to original scores. I mutated in such a fallback myself; it was killed.
- A mock gate record is accepted only with a mock environment, and mock runs write to `runs_mock`, which a real analysis never reads.
- The GPU check is per shard against `[want]`. The job-start check uses the unsorted GPU list, so it also refuses a two-GPU job, which the analysis-side check (a set of names) could not see.
- The weights check is per shard against the pin.

Hardening is suggested in S3 to S5.

**4. Did anything else change? No, in effect.**

- `WAVE_B_SHARE_TESTS`, `WAVE_B_PLANNED`, `COMPETENCE`, `SHARE_MARGIN`, `NOISE_FLOOR_NATS`, `B3_FALLBACK`, the one-fifth rule and `PREDICTION_B` are untouched.
- The original `reference` block was moved byte for byte to `reference_original` (I diffed it).
- `descriptive_comparisons`, `placement` and `template_checks` are unchanged.
- The wave B runs and their order among themselves are unchanged; `RUN_ORDER_B` only inserts the rescorings.
- Wave A's path is unaffected, since every new branch is under `wave != "A"`.
- The diff also carries the diagnostic branch's script and job (`scripts/families_envcheck_precision.py`, `pace/families_envcheck_precision.sbatch`). These are harmless, but should be named in the amendment's code list.

**5. Provenance and job-start checks: correct.** I checked the GPU pins against the records.

- Gemma 2's wave A shard: "NVIDIA L40S".
- Mistral's: "NVIDIA A40".
- Qwen3-32B's size shards: "NVIDIA A100 80GB PCIe", from the Slurm logs (jobs 7174610 and 7174613, node atl1-1-01-005-13-0, 2 h 0 min and 1 h 49 min). That is the exact string `gpu_names` records on that node type, and `A100-80GB` is a valid ICE feature.
- Qwen3-8B's earlier type (L40S) is from its logs, as review 1 found.

The job checks the GPU and node-local weights before any download or load, and refuses a non-rescoring tag in `run_family_rescore`. `require_environment` demands transformers 5 or later and no `kernels` package. `RUN_ORDER_B` puts the five rescorings after the check and before Ministral 3. The token job writes to its own `rescore_tokens/` folder. Qwen3-32B is not in any ICE cache (`hfcache` has no `models--Qwen--Qwen3-32B`), so node-local weights are required.

**6. Is the amendment text honest? Mostly; see M1 and M2.** It quotes the frozen rule verbatim. It records the failed verdict with every criterion and says plainly that it "stays FAILED". It reports `pass: false, gates: false` in the output, and nowhere implies the check passed. The exceptions are the four claims in M1 and the stale sections in M2. I would also add one sentence saying in so many words that this replaces the gate rather than satisfying it: under the frozen rule, wave B would stay unread until the check was rerun and passed.

**7. Tests and mutants.** Wave B tests: 89 passed, 7 skipped (`tests/test_families_waveb.py`, local `.venv`). Full suite: 948 passed, 77 skipped, matching the author's count.

*My own mutants.* I wrote 25 mutants on the new gating, the rescoring selection and the token tool (`R2 …`). I ran them under the strict rule, using `scripts/mutation_check.py` from `origin/main` for its kill rule. "Killed" means pytest exited 1 and named a failing test. Each mutant was run against the three files that cover the mutated modules (`tests/test_families_waveb.py`, `tests/test_families_waveb_review1.py`, `tests/test_families.py`).

12 were killed by named tests:

- partial rescoring counts as complete;
- weights never collected (vacuous pass);
- rescore gate reduced to completeness;
- missing rescoring falls back to original scores;
- rescoring environment not checked;
- Gemma 2 rescored with SDPA;
- the environment check said to gate;
- predictions read whatever the gate;
- a failed (complete) rescoring reported as waiting;
- token dump with the original tokenizer settings;
- job accepts a node with the pinned GPU among others;
- Gemma 2 pinned to an A40.

13 survived those files:

- like-for-like (M3): prompts not folded; scorer opened with the original spec; token dump without the fold;
- shard aggregation (S8): one shard on the pinned GPU enough; a shard on two GPU types accepted; one shard's gate enough; one shard's weight hashes enough; a second spec or revision tolerated;
- descriptive outputs (S9): environment sensitivity compares the original with itself; original-environment comparisons read serve2;
- minor (S10): a token comparison without a verdict accepted; an unchecked environment accepted; a mock gate without a mock environment.

The full-suite pass that the strict runner requires before a survivor counts was started for all 13 but not finished. A session restart and machine load (a full-suite run took over 45 minutes under load) stopped it. These 13 are therefore survivors of the covering test files, not yet confirmed survivors of the whole suite. No other test file imports `wf.families` or `wf.tools.envcheck_tokens`, so I expect them to survive there too, but that is not checked.

*The author's 33 rescoring mutants*, rerun under the strict rule: 29 of 33 were killed by named tests, matching the author's record. The other four were not completed:

- "analysis reads the original references" (it mutates the YAML configuration, so the runner falls back to the whole suite, and the baseline timed out under load);
- "job allows Qwen3-32B from the shared cache";
- "token tool always environment-check models";
- "prefetch refuses the reference models".

The author records all four as killed; I did not confirm them.

**8. Is the cost plausible? Yes.**

- **Wave A timings.** Mistral 1 h 43 min on an A40, Gemma 2 with eager attention 1 h 47 min on an L40S, Qwen3-8B about 1 h on L40S, and Qwen3-32B 3 h 49 min on one A100 80GB (two shards, from the logs).
- **serve2 speed.** It is not much slower: the environment check's 2,700 passes took 6.5 minutes for Qwen3-8B and 9 minutes for Mistral, loading and gating included. Scaled to 43,746 passes, that is about 1.5 to 2 hours each.
- **The total.** The table's ranges add to 11.1 to 19 GPU-hours, of which 4.6 to 9 are on 80 GB cards, as stated.
- **Qwen3-32B's memory.** The 65.54 GB of bf16 weights fit an A100 80GB, which reports about 85 GB, with about 19 GB to spare. Logits for a 1,200-token prompt are about 0.4 GB. `device_map="auto"` will not offload at that size, and the loading check would refuse CPU placement if it did. The 128 GB job is generous for a memory-mapped load. Two shards of about 2 hours each fit the 8-hour limit easily.
- **Downloads.** Two downloads of 65.5 GB to node-local NVMe, plus hashing, at 0.3 to 0.5 h each, are plausible on ICE's network. That is the least certain figure.
