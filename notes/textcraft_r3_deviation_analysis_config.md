# TextCraft round 3, deviation D-T2: the analysis config takes the Gemma slot and B3's older scores from the gates

Date: 2026-11-25. This comes after the freeze tag `textcraft-r3-design-frozen-2026-11-23` (0d6f460) and after D-T1/D-G1's
tag `textcraft-r3-deviation-t1-2026-11-23` (98dfbbf). The user approved it. It was written before any round-3
analysis ran on real scores, so no contrast, verdict or effect had been read.

## What does not change

Every frozen hypothesis, family, test, threshold, Holm size, gate and interpretation rule stays as frozen. So do the
analysis code and the gates code. `wf/textcraft/r3_analysis.py` and `wf/textcraft/r3_gates.py` are byte-identical to
D-T1's tag. The frozen config `wf/config/textcraft_r3_analysis.yaml` is kept unchanged as the original record.

## Why the frozen config cannot run

The frozen config sets `gemma_core: gemma4-12b-it` and `b3_old_scores: round2`. Both were meant to follow the gates:
the file's own comments say E4B "only if the fallback rule fired" and rescore "only if the environment check failed".
The analysis refuses any configured value that disagrees with the gate outcome (design 14.M3 and 14.S5), and the
filed gate outcomes disagree with both:

- **Gemma slot.** Gemma 4 12B's smoke record fails the attention-agreement check (yes/no 0.216 nats against 0.10), so
  `r3_gates.gemma_slot` gives `gemma4-e4b-it`. The analysis stops with "gemma_core is 'gemma4-12b-it', but the
  fallback rule gives 'gemma4-e4b-it'".
- **B3's older scores.** The environment check failed, so `r3_analysis.b3_source` gives `rescore`, and a configured
  `round2` raises.

The dispatcher (`python -m wf.runs <run id>`) reads `wf/config/<run id>.yaml` and accepts no override. As frozen, the
analysis therefore cannot run at the freeze commit or at D-T1's commit. Test
`test_dt2_the_corrected_analysis_matches_the_frozen_code_on_a_replay` shows both refusals, under the frozen code and
under the current code.

## The correction

The new config `wf/config/textcraft_r3_analysis_dt2.yaml` copies the frozen config key for key. It differs in only
three places:

- the run id is `textcraft_r3_analysis_dt2`;
- `gemma_core` is `null`;
- `b3_old_scores` is `null`.

The new run id is registered in `wf/runs.py` with the same function, `run_textcraft_r3_analysis`. The dispatcher, the
analysis code and the gates code are not changed.

**Why null is not a new rule.** For a null value, the frozen code already uses the gates' own result:

- The Gemma check refuses only a value that is neither null nor the slot `gemma_slot` returns.
- `b3_source` returns the environment check's outcome when the configured value is null.

So null means "as the gates decide". That is the frozen rule (14.M3: "the analysis takes the slot from this
function"; 14.S5).

**What the gates give on the filed records:**

- the Gemma slot is `gemma4-e4b-it`, which is itself "not scored": its smoke record also failed attention agreement
  (0.125 against 0.10), design 14.M2, unchanged by D-G1;
- B3 reads `rescore`, the serve2 rescores of jobs 7252995 to 7253000.

The metrics record both (`gemma_core`, `gemma_slot` with its triggers, `environment_check.status`,
`B3.old_scores`), so the reader checks the values on the output itself.

**The output folder.** The run writes `runs/textcraft_r3_analysis_dt2/`, which no frozen run id uses.

## Where the analysis runs (authorized here)

`textcraft_r3_analysis_dt2` runs from a clean checkout at the commit of the annotated tag
`textcraft-r3-deviation-t2-<date>`, which the coordinator creates after review. This applies to an interim analysis
and to the final one. This note authorizes it.

**Why this commit.** It is the only analysis code that has both of these:

- the new run id;
- D-T1/D-G1's reading of the thinking arm and of the exploratory E4B run. The frozen analysis never reads E4B, and it
  flags thinking shards made at D-T1's commit.

**How the commit is checked.** The analysis never checked its own commit; `require_freeze_commit` is called only at
the start of scoring and thinking jobs. So the reader checks the run folder before reading anything:

- `git_sha` must be the t2 tag's commit;
- `inputs.json` must show `git_dirty: false`.

**Tags the checkout must carry.** The bundle must contain the freeze tag, exactly one `textcraft-r3-deviation-t1-*`
tag and the t2 tag.

- The t2 prefix `textcraft-r3-deviation-t2-` is distinct from `r3_gates.DEVIATION_TAG_PREFIX`, so
  `deviation_commit()` still finds the single t1 tag.
- Without the t1 tag, `deviation_commit()` returns none. The thinking and E4B shards would then carry commit problems
  in `provenance_problems`. These refuse no family, but they would be wrong.

**Which commit each shard must come from.** The analysis at the t2 commit accepts:

- the thinking arm's and the E4B run's shards made at D-T1's commit (or the freeze commit);
- every other shard only from the freeze commit (or a code-identical one). This covers the core and extension scoring,
  the smoke runs, the environment check and the rescoring.

**No new job-start exemption.** D-T2 adds no exemption at job start. `require_freeze_commit` is unchanged. A thinking
or exploratory run cannot start at the t2 commit, because that commit changes `wf/runs.py` relative to D-T1's. Every
GPU job keeps its frozen checkout: the core and extension models at the freeze tag; the thinking arm and E4B at D-T1's
tag. Test `test_dt2_a_second_deviation_tag_leaves_d_t1_intact` shows this with real git, using three tagged commits.

## The first-core-reading marker

The user decided on 2026-11-25 to run all three extension models (OLMo 3 7B, Qwen3.8-27B and Gemma 4 31B). They are
submitted from a freeze-tag checkout, and no `not_run.json` declaration will be filed. The marker
(`runs/textcraft_r3_first_core_reading.json`) decides only whether such a declaration is on time (14.R2, S6), so with
no declaration it decides nothing.

- **If a single analysis runs after every arm** (the plan), the marker is moot. That run writes it once.
- **If an interim analysis runs anyway,** two steps are needed:
  - Its marker must be copied, unchanged and with its original time, into the `runs/` folder of the checkout that
    runs the final analysis, before that run. The interim run's own folder is moved aside as
    `runs/_interim_textcraft_r3_analysis_dt2_<date>/`.
  - Any `not_run.json` filed after the interim reading is late. It is then a provenance problem and refuses A3x, as
    the frozen rule says.

## Checking that the confirmatory verdicts are preserved

The keys below carry or decide a confirmatory verdict. They are compared as the sha256 of each value serialised as
JSON with sorted keys.

- the families `A3`, `A3x` and `B3`, including H3-P, R2, the replication readings, every test, its raw and
  Holm-adjusted p, and every verdict and pair reading;
- `per_model`, `model_states`, `gemma_core`, `gemma_slot`, `fallback_used`, `environment_check`, `environment_rule`,
  `declared_not_run`, `missing_scores`, `round2_analysis_read` and `freeze_commit`;
- `provenance_problems`, without the key `think` and the keys starting `r3x:`. Those name the exploratory runs, which
  no family reads.

The keys expected to differ are `exploratory` (the E4B block), `inference_map` (one added line) and `deviation_t1`
(absent in the frozen code).

1. **Final run against the frozen code.** The final analysis's metrics against the freeze commit's analysis code,
   run on the same filed runs with this config's values: every key above must be identical. Test
   `test_dt2_the_corrected_analysis_matches_the_frozen_code_on_a_replay` does this on a mock replay of the filed
   state, using byte copies of the freeze commit's `r3_analysis.py` and `r3_gates.py` in
   `tests/golden/textcraft_r3_frozen/`. The replay covers:
   - two core models scored and both Gemma 4 smoke gates failed;
   - the environment check failed and the rescores filed;
   - the thinking arm and the E4B run;
   - no extension model.

   Test `test_dt2_the_frozen_copies_are_the_freeze_commits_files` checks that the copies equal the commit's files.
2. **Interim against final** (only if an interim analysis ran). The extension models will be scored between the two
   runs, so some keys change legitimately: `A3x`, `declared_not_run`, and the extension models' entries in `per_model`,
   `model_states` and `missing_scores`. Everything else must be byte-identical:
   - `A3` and `B3`;
   - `gemma_core`, `gemma_slot`, `fallback_used`, `environment_check`, `environment_rule`, `round2_analysis_read`
     and `freeze_commit`;
   - the core models' entries in `per_model`, `model_states` and `missing_scores`;
   - the filtered `provenance_problems`, apart from the extension models' keys.

   A3 and B3 read only the core models, the older models and round 2.

## Tests

In `tests/test_textcraft_r3.py`:

- `test_dg1_the_exploratory_block_cannot_enter_any_family` now passes a real environment result. The environment check
  fails and B3 reads the older models' scores, so B3 holds the two scored pairs with 20 unpadded tests. It compares A3,
  A3x and B3 across three runs: the freeze commit's code, the current code without E4B, and the current code with E4B.
  Its always-true assertion (`or True`) is removed.
- `test_dt2_the_corrected_config_takes_both_values_from_the_gates` checks two things:
  - the frozen config is unchanged;
  - the new config equals it but for the run id and the two nulls, and a null gives the gates' value in both code
    versions.
- `test_dt2_the_corrected_analysis_matches_the_frozen_code_on_a_replay` is the replay described above.
- `test_dt2_a_second_deviation_tag_leaves_d_t1_intact` runs with real git:
  - `deviation_commit()` with a t2 tag present;
  - thinking and E4B shards accepted at D-T1's commit, and refused at the t2 commit;
  - rescoring and core shards refused at D-T1's commit;
  - a thinking run refused at job start at the t2 commit, and started at D-T1's.
- `test_dt2_the_frozen_copies_are_the_freeze_commits_files`.

Three hand-made mutants were each caught by these tests: B3's margin changed from 0.20 to 0.25; E4B dropped from the
D-T1 commit exemption; an explicit `gemma_core` in the new config. They are not in `scripts/mutation_check.py`.

## Cost

CPU only.
