# TextCraft round 3: focused re-review (review 2, 2026-11-23)

**What this is.** A check of `textcraft-r3` at f68949d against review 1 ([textcraft_r3_review1.md](textcraft_r3_review1.md)), and a search for new defects in the fixes. The work was done on branch `textcraft-r3-review2`, cut from f68949d.

**What was and was not used.** No GPU job was run. ICE was not touched. No round-2 score, analysis output or result was read. The only data read were round 2's records, plus synthetic job records that contain GPU names and nothing else. The scripts are `scripts/review2_textcraft_r3_checks.py`, `scripts/review2_textcraft_r3_plan.py` and `scripts/review2_textcraft_r3_mutants.py`. Review 1's scripts were rerun where they still apply.

## Verdict

**Ready to freeze once the two open items land, with one condition that must be written into the design first.** Both open items have to be *committed before the freeze tag*.

- **Why the order matters.** Open item 2 (the plan file) lives in `wf/config/`, and open item 1 (wave B's amended per-score rule) lives in `wf/families.py`. The new rule S12 lets a job run only at the tag's commit or at a commit that changes nothing under `wf/`, `scripts/` or `pace/`. Either file committed after the tag therefore makes every round-3 job refuse to start. For the plan file the environment check cannot run at all, because the job also needs the plan in its own checkout.
- **What the design says now.** Design 14.M4 only says the plan is committed "before any environment-check job". That wording allows the order that deadlocks. This is the one must-fix item, **R2-M1**.

All four of review 1's must-fix items and all thirteen should-fix items are fixed, and I checked each against its failure scenario. The fixes introduced no new defect that would give a wrong confirmatory reading. Seven should-fix items remain (section 3). Two are cheap enough to make before the tag: R2-S1, a provenance gap in B3, and R2-S2, uncommitted changes in the checkout.

## 1. Review 1's items against their failure scenarios

| Item | Failure scenario in review 1 | What I checked | Status |
|---|---|---|---|
| M1 | A shard on two L40S failed the analysis's provenance check and blocked every family | `review2_textcraft_r3_checks.py`: a shard whose job recorded `["NVIDIA L40S", "NVIDIA L40S"]` now gives no problem, because each job's own list is read from `jobs.jsonl` (`r3_gates.job_gpus`). A problem now refuses only its family (`test_a_family_with_a_provenance_problem_is_refused_alone`) | fixed (but see R2-S1 for B3) |
| M2 | A core model that fails its gate left H3-P and replication "waiting" for ever | A filed `loading_failed` record gives `model_state` "not scored". In `family_a` that model enters at p = 1 with competence false, and H3-P, R2 and replication proceed (`test_a_core_model_not_scored_enters_at_p_one_and_the_readings_proceed`). The B3 pair of a model that is not scored reads "not scored". Loading failures are now caught and filed (`r3_score.py`, `loading_failed`) | fixed |
| M3 | The Gemma slot was a hand-set config value | `r3_gates.gemma_slot` reads the filed smoke record. The analysis refuses a `gemma_core` that disagrees. The E4B smoke and score runs refuse to start unless the rule fired. If E4B then fails its own gate, the slot is "not scored" | fixed |
| M4 | With round 2's four shards spread over mixed GPU types, 200 comparable records was unlikely | `review2_textcraft_r3_plan.py`, synthetic job records with GPU names only. All four shards on L40S: one job, 300 records comparable. A 2-2 split: four jobs on exact GPU names, 300 comparable. One shard untyped: 198 comparable, plan "not runnable", status "not run (GPU types)". The analysis now matches GPUs record by record | fixed |
| S1 | One competent model that is supported read "partial" | It now reads "inconclusive (fewer than two competent)" | fixed |
| S2 | A3's power was computed at the 0.05/50 Holm step | `holm30_power` recomputes it at z = 2.935, before the verdicts and interpretations are formed | fixed |
| S3 | B3 power and the underpowered flag were missing | Both are reported. The fixed-denominator column is added. The design states that equivalence for Dn, Dc and T is unreachable | fixed |
| S4 | "Newer higher" on a negative P was misleading | `direction_words` now adds "smaller in size (a smaller penalty)" or "opposite signs" | fixed |
| S5 | `b3_old_scores` was a free config value | It is derived by `b3_source`, which refuses a config value that disagrees | fixed |
| S6 | B3 read round 2's scores without round 2's provenance check | `r2_analysis.provenance` now runs on the three older models first, and the sha256 and status line of round 2's metrics are recorded | fixed |
| S7 | `b_draws` could be lowered in a real run | It is refused outside a mock run | fixed |
| S8 | Gemma 2 was not covered by the environment check | Gemma 2 is added to the check, its configuration and the run order | fixed |
| S9 | The mass trigger overlapped the competence gate | The fallback now fires only when the mass is low *and* the top next tokens are structural. A low mass with answer-like top tokens goes to the competence gate (shown on synthetic smoke rows) | fixed, sound (section 2.3) |
| S10 | Thinking arm: forced flag mixed two cases; padding unchecked; no prompt check | The forced flag is split into "cap" and "ended unclosed". A check of padded against unpadded logits and a prompt check run at job start | fixed |
| S11 | Exploratory blocks carried p-values | `descriptive_only` strips them | fixed |
| S12 | The one-commit rule broke on ordinary commits | Every job now runs from the freeze tag's commit, and the analysis checks every job of every run | fixed (see R2-M1, R2-S2) |
| S13 | Review 1's eight surviving mutants had no tests | Tests were added. A rerun of review 1's mutant script on the amended code is in section 4 | fixed |

`tests/test_textcraft_r3.py` ran locally: **78 passed**.

## 2. New code: what I checked

### 2.1 `r3_gates`

- **`job_gpus` and `shard_gpu`.** These read each job's own GPU list. A shard counts as typed only when every job recorded the same list. They are correct, and my mutants on them were killed.
- **`model_state`.** A model is "not scored" on any filed gate failure from its smoke run or any score shard. It is "scored" once a score-run gate record exists with no failure, and "waiting" otherwise. This is correct. Note N1: the attention agreement runs on each shard's own first five items. So Gemma 4 12B can pass at its smoke run (which fixes the slot as 12B) and then fail on a score shard. It is then "not scored" with no substitution, which is consistent with the rule that the fallback is decided at the smoke run only.
- **`gemma_slot`.** Correct. The slot reads "waiting" until 12B's smoke gate record exists, and the analysis then uses 12B, whose state is also "waiting".
- **`require_freeze_commit`.** It checks HEAD against the tag but not the working tree; see R2-S2. It raises an error if more than one tag matches `textcraft-r3-design-frozen-*`. That is deliberate, and it means a second freeze tag (for example after a deviation) needs a new prefix.
- **The environment-check plan.** It uses round 2's own partition and the exact GPU names, takes the single-type shortcut when it can, and treats untyped shards as not comparable. Both the plan run and the jobs refuse a plan that is not runnable.

### 2.2 How "not scored" propagates

- **A3 and A3x.** The model's tests enter at p = 1, its verdicts are "not scored", and its competence is false for every contrast. R2 and replication count only competent models, so they are unaffected. H3-P proceeds.
- **B3.** A pair whose newer model is not scored reads "not scored", and its 10 tests enter at p = 1.
- **One mislabel (R2-S3).** In H3-P's per-model column, a model that was not scored is shown as "inconclusive".

### 2.3 The S9 structural-token trigger

`structural()` counts a top token as structural when it is any of the following, unless it is "yes" or "no" in any case:

- a special token;
- empty after stripping (whitespace or a newline);
- a token starting with one of the characters `* # \` > - _ ~ | < [ {`.

On the checks script, `"\n"`, `" "`, `"**"`, `"<"`, `"["` and `"-"` count as structural, and `"Yes"`, `" no"`, `"The"`, `"Craft"` and `"C"` do not.

The fallback fires only when the median mass is below 0.5 *and* more than half of the smoke run's yes/no prompts have a structural top token. A model that wants to reply in environment words ("Could", "Crafted") therefore goes to the competence gate and is not replaced. That is the right separation between a format failure and a finding. For Gemma 4 the top token is read after the empty thought channel, because `_encode` includes the declared generation suffix, which is the right position.

### 2.4 The S12 loophole: "changes nothing under `wf/`, `scripts/` or `pace/`"

**What it allows.** The tracked top level of the repository is `wf/`, `scripts/`, `pace/`, `notes/`, `tests/`, `runs/` (`.gitkeep`, `FILED.txt`, `env.json`), `README.md`, `PILOT_PROTOCOL.md`, `pyproject.toml` and `.gitignore`. Configs, weight manifests, records, golden files and the plan file all sit under `wf/`. So the rule only allows commits that touch:

- notes and other documentation;
- tests;
- `runs/` bookkeeping, where nothing on the scoring path reads `env.json`, and `FILED.txt` is only checked for presence;
- `pyproject.toml`, which has no effect on an already-installed environment.

Checked with the checks script: `notes/x.md`, `tests/test_x.py`, `pyproject.toml` and `runs/FILED.txt` are accepted; `wf/config/textcraft_r3_analysis.yaml` and `wf/config/textcraft_r3_envcheck_plan.json` are refused. **The loophole is safe as to what code can run.**

**Two consequences.**

- The plan file is caught by the rule, which is R2-M1.
- HEAD is compared, but the working tree is not, which is R2-S2.

### 2.5 The builder's two discretionary choices

**All three models or rescore all.** If any of the three models fails, or cannot be run, all three older models are rescored. I agree with this choice.

- It keeps one source of older scores across the B3 family.
- A failure on one architecture is evidence that the software differs, and that difference could reach the others.
- The cost is about 7 GPU-hours, inside the budget.

A per-pair rule would be defensible too, but it would mix serve2-against-serve2 and environment-against-environment comparisons inside one Holm family.

**`gpu_unobtainable.json`, with "waiting" until it is filed.** Acceptable.

- Round 2's shards are all pinned to L40S, so the plan will be one L40S job, a type that is always obtainable. The file should never be needed.
- The operator's discretion is only whether and when to file it, and the only effect of filing is the declared rescoring.
- The code checks only that the file exists, not the 7-day rule. R2-S4 hardens this.

## 3. Findings

### Must-fix

**R2-M1. The plan file and wave B's amendment must be committed before the freeze tag; state this in 14.M4.**

- **Where:** `r3_gates.CODE_PATHS`, `code_identical`, `require_freeze_commit`; `r3_gates.PLAN_FILE` is `wf/config/textcraft_r3_envcheck_plan.json`; `plan_job` reads that file from the job's own checkout. The design text is 14.M4, item 1, and 14.T, "Not run".
- **Failure scenario:** the tag is cut now and round 2 finishes afterwards. The plan is then committed under `wf/config/`.
  - Every round-3 job from a checkout that contains the plan stops with "this checkout is at …, not the freeze tag's … (or a code-identical commit)".
  - An environment-check job from a checkout at the tag stops with "no committed environment-check plan".
  - The environment check therefore cannot run without a deviation.
  - The same applies to wave B's amended per-score rule, which lands in `wf/families.py`.
- **Fix:** write into 14.M4 that the freeze waits for both open items and that both are committed before the tag. If the freeze cannot wait, there is an alternative. Exempt exactly `PLAN_FILE` from `code_identical`. Then have the analysis re-derive the plan from round 2's `jobs.jsonl` and refuse a committed plan that differs. That keeps the plan mechanical even though it is committed late.

### Should-fix

**R2-S1. B3 is not refused when a pair's newer model has a provenance problem** (`r3_analysis.py:923-927, 956-973`).

- Design 14.M1 says a problem refuses its family, including "B3 through the pair's models". The code refuses A3 or A3x from a core or extension model's problems, but it sets `refused["B3"]` only from round 2's provenance or from the rescoring.
- **Scenario:** one job of a Ministral 3 score shard recorded no GPU, because the second `nvidia-smi` call in `job_record` timed out. A3 is refused, but B3 still reads the Mistral pair from those same scores. The trigger is rare, because the job-start checks guard the commit and the GPU type, but the output would contradict itself.
- **Fix:** refuse B3 when any pair's newer model (the Gemma slot's model included) has problems, and add a test.
- Also, the environment check's own commit problems (`envcheck:<tag>`) are recorded but do not affect its status. With the job-start check in place this is a note only.

**R2-S2. A checkout with uncommitted changes passes the freeze check.**

- `require_freeze_commit` compares HEAD only. A job at the tag's commit with uncommitted edits under `wf/` runs, and only `inputs.json`'s `git_dirty` records it.
- **Fix:**
  - refuse at job start when `wf.util.git_dirty()` is true (it already ignores untracked files and the calibrated configs);
  - have the analysis refuse any shard whose `inputs.json` has `git_dirty: true`.

**R2-S3. A model that was not scored shows as "inconclusive" in H3-P's per-model column; label it "not scored".**

**R2-S4. `gpu_unobtainable.json` is checked only for existence.** Require it to name the first submission's job id and date, and have the analysis check that at least 7 days have passed, so that "not run (GPU types)" stays mechanical.

**R2-S5. A filed gate failure can be erased by rerunning into the same folder.** `tokenization_checks.json` is overwritten on a rerun. The fix is either to refuse a rerun into a shard folder that already holds an `ok: false` record, or to keep the earlier records (for example `tokenization_checks.<n>.json`) and have `model_state` read all of them. The design says gate failures are final, and the code should enforce it.

**R2-S6. No extension model can be closed as "not run for GPU availability", so A3x reads "waiting" for ever.** If one extension model is never run, A3x's H3-P and replication are never computed. Design section 11 says the decision is "recorded before any core result is read".

- **Fix:** a filed declaration, for example `runs/textcraft_r3_score_<tag>/not_run.json` with its date. The analysis counts it like "not scored" (p = 1) and refuses it if dated after the first core analysis.

**R2-S7. No test pins `pace/` in the protected code paths.** Mutant rv2-4 removes `pace/` from `CODE_PATHS` and survives the whole suite. A commit that changed only a job script would then pass as "code-identical". Add a case to `test_freeze_commit_and_job_start_check` with a changed `pace/textcraft_r3_score.sbatch`.

### Notes

- **N1.** As in 2.1: a Gemma 4 12B that passes its smoke run but fails a score shard's attention agreement is not substituted.
- **N2.** The fixed-denominator column divides by a fixed SD that is not resampled. Its interval therefore understates the uncertainty a little. It is descriptive, so this is fine.

## 4. The two known open items

**1. The per-score criterion of the environment check (placeholder awaiting wave B's amendment).**

**Safe as it stands.** If nothing lands, `per_score_check` uses wave B's current 0.07 nats. Wave B's own check failed on Qwen3-8B, so this criterion would very likely fail here too. The status would be "fail" and B3 would use the declared full rescoring: about 7 more GPU-hours, with no false pass.

**Two things are needed when the amendment lands, beyond committing it before the tag (R2-M1).**

- **The interface.** The hook is called as `envcheck_per_score_check(fmt, diffs, old, new)`. It receives one format's pooled differences and L values, without the model's tag and without the condition labels. The amended rule needs:
  - a g fixed per model at its readout-logit magnitude, which needs the tag;
  - a per-condition bias bound, which needs the differences keyed by condition;
  - possibly the stored `logp_plus` and `logp_minus`.

  `envcheck_block` should pass the tag and the condition-keyed pairs. Otherwise the amended rule cannot be applied without editing `r3_analysis` after the freeze.
- **Pinning.** The analysis reads wave B's rule from `wf/families.py` "live" at analysis time, and the analysis run is not itself commit-checked. A later wave B edit would therefore silently change round 3's criterion.
  - Record the rule's values and the sha256 of the hook's source in round 3's metrics.
  - Refuse if they differ from the values committed at the freeze, for example a constant in `r3.py` that holds the hash.

**2. The plan file, written after round 2 is filed.**

**Handled safely, provided it is committed before the tag (R2-M1).** With every round-2 shard pinned to L40S, the plan is one job on L40S covering round 2's shards 0 to 3, with all 300 records comparable (checked on synthetic job records). A shard resumed on another GPU type would make that shard untyped. If it is shard 3 (102 records), only 198 remain, the plan is not runnable, the outcome is "not run (GPU types)", and the declared rescoring follows. That is the intended conservative behaviour.

## 5. Mutants (strict named-kill rule)

**Twelve new mutants on the code added after review 1** (`scripts/review2_textcraft_r3_mutants.py`). A screen on the covering tests came first, and the survivor was then rerun on the whole suite.

| Mutant | What it breaks | Result |
|---|---|---|
| rv2-1 | `structural` ignores the special-token flag | killed (`test_structural_tokens`) |
| rv2-2 | the format trigger fires on a low mass alone | killed (`test_gemma_slot_rule`) |
| rv2-3 | `model_state` ignores smoke-run gate failures | killed (`test_model_state_from_filed_gate_records`) |
| rv2-4 | the protected code paths omit `pace/` | **survived** the whole suite (should-fix R2-S7) |
| rv2-5 | a shard counts as typed despite mixed job GPUs | killed (`test_envcheck_plan_from_round_two_gpu_names`) |
| rv2-6 | a golden-token disagreement does not fail the environment check | killed (`test_environment_status_and_b3_source`) |
| rv2-7 | a failed environment check keeps round 2's scores | killed (`test_environment_status_and_b3_source`) |
| rv2-8 | environment-check records compared across GPU types | killed (`test_environment_check_matches_gpus_record_by_record`) |
| rv2-9 | a model that was not scored counts as competent | killed (`test_a_core_model_not_scored_enters_at_p_one_and_the_readings_proceed`) |
| rv2-10 | "cap" needs more than the 2,048-token budget | killed (`test_budget_forcing_and_the_readout_point`) |
| rv2-11 | the freeze commit is not enforced at job start | killed (`test_freeze_commit_and_job_start_check`) |
| rv2-12 | `b_draws` accepted outside a mock run | killed (`test_b3_uses_twenty_thousand_resamples_outside_mock`) |

**Review 1's eleven mutants, rerun unchanged on f68949d** (`scripts/review_textcraft_r3_mutants.py`, covering tests).

- **Applied and killed: 8 of 8.** This includes all five former survivors that still apply: rev2, rev4, rev5, rev6 and rev11, plus rev7. Each was killed by the tests added for S13.
- **Not applied: rev3, rev9 and rev10.** The code they targeted was rewritten by the fixes, so their strings no longer occur. The builder's adapted versions of these three are in `scripts/mutation_check.py`, and its official log at d29731a reports them killed. I did not rerun the builder's 109.

Logs: `notes/textcraft_r3_review2_mutation_log.txt`.
