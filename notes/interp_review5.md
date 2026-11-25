# Interpretability pilot: review 5 (confirming review of fix round 4)

Date 2026-11-22. Branch reviewed: `interp` at 8fadc39 (design amendment 4 in 48d62e8, code and tests in 8fadc39). Reviewer worked on a local branch `interp-review5` (not pushed). Nothing was run on a GPU or on the cluster; no interpretability job has run.

## What was done

1. Read the amendment-4 diff (`git log -p 48d62e8^..8fadc39`), sections C1 to C8, M and B1 to B4 of `notes/interp_design.md`, the round-4 entry of `notes/testing.md` and the review-4 block of `scripts/mutation_check.py`. The review-4 report itself is not on the branch, so its items were reconstructed from amendment 4 (D1 to D11), the deviations entry and the testing table.
2. Checked each D item in code (`wf/interp/analysis.py`, `wf/interp/pilot.py`) and checked that a test fails if the fix is reverted.
3. Looked for remaining validity problems in the rules the minimal pilot will report, especially the onset rule after the direct-path exclusion.
4. Built a git-ignored CPU environment in this worktree (`.venv`: Python 3.12, torch 2.2.2 CPU, transformers 4.51.3, numpy below 2, scikit-learn) and ran the interpretability tests, the full suite and 10 mutants of my own (a kill counts only when pytest exits 1 with a named failing test).

## Verdict

**One must-fix item remains (R5-MF1). Not ready to freeze until it is fixed.** Every review-4 item is fixed in code as the note says, and each has at least one test that fails on reversal. Three parts of the new code are not pinned by any test (T1 to T3 below: the D7 gate's effect on the reported value, its "both folds" condition, and the D10 ablation plan). The remaining validity problem is a mirror image of review 4's D1 problem that D1's own premise rules out by mistake.

## R5-MF1 (must-fix): excluding a failing layer can move the onset *earlier* and manufacture "at or before arrival"

Where: `notes/interp_design.md:530` (D1: "removing a layer can only move the onset later … Exclusions after the onset change nothing"), the rule in C2 (`notes/interp_design.md:478`), `wf/interp/analysis.py:718-733` (`direct_path_kept`) and `wf/interp/analysis.py:977-979` (`ablation_summary` reads the onset on the kept layers).

The onset is the earliest kept layer from which every *kept* deeper layer has a removal lower bound of at least 0.6. The direct-path rule removes a layer whatever its removal is. If the removed layer *fails* (lower bound below 0.6) and sits after a run of passing layers, removing it lets the onset jump back to that earlier run. So removing a layer can move the onset earlier, and D1's premise is false; D1's check guards only the other direction (onset pushed after the arrival).

Concrete case, run against the code at 8fadc39 (profile 4, …, 28; answer-position arrival 12; removal lower bounds 0.10, 0.65, 0.70, 0.45, 0.70, 0.75, 0.80 at layers 4, 8, …, 28):

| Direct-path exclusions | Kept layers | Onset | `onset_versus_arrival` |
|---|---|---|---|
| none | 4 to 28 | 20 | after arrival |
| layer 16 only (upper bound 0.33) | 4, 8, 12, 20, 24, 28 | 8 | at or before arrival |
| 16 to 28 (direct-path share rising with depth, the likely pattern) | 4, 8, 12 | 8 | at or before arrival |

Why it matters. The exclusion rule exists because a large direct-path share *inflates* a layer's removal, so a pass there may be an artifact. Inflation cannot produce a failure: a layer whose removal fails even with the direct path's help is real evidence that one direction does not carry 60% of the effect there. Dropping it turns a violated "stays at or above 0.6 from L on" condition into a pass. The flipped reading is the one B1 maps to H3/H4/H5/H6 in the full pilot ("at or before arrival"), so the rule can produce that reading by construction, exactly the failure review 4 found in the other direction. The pattern is not exotic: the layer where the direct-path share first crosses an upper bound of 0.3 is the transitional layer, where a removal around 0.4 to 0.6 (direct path plus a partial direction) is expected. The existing tests pass on this case because none puts a failing layer among the excluded ones.

Smallest fix (a rule change in the note plus one line of code): a profile layer is excluded only if its direct-path upper bound is at least 0.3 **and** its removal lower bound is at least 0.6; a failing layer stays in the onset's profile, since the direct path cannot explain a failure. With that, exclusion removes only passing layers, which can only move the onset later or make it undefined, D1's premise becomes true, and D1's "not assessable" check stays correct as written. `direct_path_kept` would need the removal profile (or `ablation_summary` would filter `kept` by it), and a test should cover the case above (excluding a failing layer 16 must not give onset 8). An equivalent alternative is to make the row "not assessable" whenever an excluded layer after the onset has a lower bound below 0.6, but the first form is simpler and keeps the reading. Correct D1's sentence either way. The current tests (`test_direct_path_share_tracks_the_removal_and_the_rule_excludes_it`, `test_an_excluded_layer_between_arrival_and_onset_makes_it_not_assessable`) exclude only passing layers and stay valid under the fix.

## Review-4 items, checked one by one

| Item | What the note says | Code | Test that fails on reversal | Status |
|---|---|---|---|---|
| D1 exclusion between arrival and onset | onset after arrival with an excluded layer in [arrival, onset) is "not assessable" | `onset_versus_arrival(..., all_layers=)` `analysis.py:674-676`; `table_rows` passes the whole profile (`pilot.py:883-885`) | M1, M2 and my V1 (arrival layer itself excluded) are killed by `test_an_excluded_layer_between_arrival_and_onset_makes_it_not_assessable` and `test_table_rows_read_interval_bounds_onset_and_h4` | Fixed for that direction; premise wrong in the other direction (R5-MF1) |
| D2 tests for the exclusion | non-deepest exclusion, review 4's four cases, arrival at deepest kept, upper bound exactly 0.3 | — | present in the two tests above and `test_onset_and_deepest_bound_come_from_the_kept_layers`; R4-N7, N8, N10, N11 killed | Fixed |
| D3 steering ratio gated | read only where `steer_task`'s lower bound is at least 0.3; median over those layers or "not assessable" | `gate_on_lower_bound` `analysis.py:1029-1035`, applied at `:1003-1005`; `h4_rows` medians over `steer_ok` (`pilot.py:806-807, 838-841`) | M4, M5, M6, and my V9 (gate on the wrong reference) killed | Fixed |
| D4 H4 layers before the earlier arrival | cut at min of the final group's 0.5 arrival and the answer-position arrival | `pilot.py:875-876` | M7 and R4-N31 killed (test checks both orders) | Fixed |
| D5 stage-0 quantities SE only | H4 cosine and direct-path share: "not applicable" | `pilot.py:930-937` | M8 killed | Fixed |
| D6 "can conclude" restated | calibration readings, no labels | note only; `table_rows` names no hypothesis (end-to-end test checks) | — | Fixed (text) |
| D7 second projection gated | read only where the answer direction's absolute cosine with the readout-free success direction is below 0.5 in both folds | `h4_cosine_report` `pilot.py:747-750`, `h4_rows` `pilot.py:809-813, 819-822` | M9 (the layer list) and M10 are killed. My V4 (the row's *value* taken over every H4 layer instead of the gated ones) **survives**: in the test the ungated layer does not move the median. My V2 (gate on the smaller of the two folds' values, so "both folds" becomes "either fold") **survives**: both folds carry the same value in the test. V3 (gate at 0.5 inclusive) survives; it only moves the boundary | Code fixed; test gaps T1, T2 |
| D8 direct path on the whole calibration fold, NaN bounds, reliabilities | every edited-fold failure, unweighted; NaN upper bound excluded; disattenuation needs 0.2 | `pilot.py:643-650`; `analysis.py:718-733`; `analysis.py:603-611` | R4-N2, N3, N4, M11, M12 killed; my V5 (other fold) and V6 (layer with no entry excluded) killed | Fixed. The direct-path population (all calibration-fold failures with aligned and swapped prompts) matches the ablation's eligible population (`ablation_selection` uses no effect filter), so the unweighted share and the weighted removal estimate the same population |
| D9 limits and linearity diagnostic | stated; ratio reported, not read | `analysis.py:1006-1010` | M16 is killed. My V10 (the ratio taken against another layer's removal) survives, because the test has only one layer | Fixed (a diagnostic that is reported, not read) |
| D10 sample sizes combine as the largest need, capped at 400 | per stratum, patching and ablation separately; half-widths when a cap binds | `plan_sizes`, `half_width_at` `analysis.py:736-753`; `pilot.py:938-947` | M13, M14 and M15 are killed (they hit the patching plan and `plan_sizes` directly). My V7 (the ablation plan leaves out the steering ratio) and V8 (the ablation cap set to 100) **survive**: no test checks the ablation plan | Code fixed; test gap T3; notes N2, N3 |
| D11 tests, mutants, sbatch comment | 22 survivors killed; 31 + 15 mutants added; 55 edits | sbatch comment matches (7 kinds × 7 layers + 4 random + caveat + last = 55) | see test results | Fixed |

## Notes (do not block the freeze)

- **N1. "Not assessable" for an empty H4 layer set is only partly written.** When no profile layer lies before the arrival, `h4_rows([])` returns "not assessable" for the steering ratio, the second projection and the perp-over-task ratio, but NaN for the main projected cosine and the steering shares (`pilot.py:816`, `842-844`). B2 says the H4 rows are then "not assessable". NaN cannot be misread as a value, so this is cosmetic; a string would be clearer.
- **N2. The object-mention share's sample size does not scale with its own items.** Its SE comes from the roughly 25 mention items, but `sample_sizes` scales it by the stratum's whole patching n (`pilot.py:919-921`), which assumes the mention items grow in proportion. `scaled_minimum` scales the mention minimum down for draws below 100 but never up, so a full pilot of 400 patching items would draw about 25 + 3% of the rest, not 100, mention items. Example: SE 0.09 at 25 items needs about 166 mention items; the plan reports a half-width at the cap of 0.097 for 400 stratum items, which those 400 items would not reach. Fix before the full pilot by computing this need in mention items (or scaling the mention quota up with n); it does not affect any minimal-pilot reading.
- **N3. The "all" plan has no stated use.** `plan` is computed for step 0, later and all. The draw is made per weighting stratum, so the pooled plan's role (the sum of the two strata, or a check on it) should be one sentence in D10.
- **N4. The second-projection gate reads one fold when the other is missing.** `ans_cos` takes the maximum over the folds present (`pilot.py:809-812`); with one fold missing the "both folds" condition becomes "one fold". Cannot happen with the planned sizes; a one-line guard would make it exact.
- **N5. Mutation-harness wording.** `testing.md` says a kill needs "a named test" to fail; the harness counts pytest exit code 1 with `--tb=no`, which is the same thing in practice (exit 1 only when some test failed) but prints no name.

## Test results (CPU, this worktree's `.venv`)

- `tests/test_interp.py` and `tests/test_interp_pilot.py`: 100 passed, none skipped (147 s). Without scikit-learn both files skip entirely; it must be in the environment for the interp tests to run at all.
- Full suite (`tests/`): 720 passed, 2 skipped (40 minutes on a machine shared with other test runs).
- Ten spot-check mutants of my own, V1 to V10 (definitions in the git-ignored `.venv/review5/my_mutants.py`). For a mutant that survives the interp test files, the repository harness reruns the whole 40-minute suite. On this busy machine that rerun hit the harness's 900 s timeout, which the harness reports as "inconclusive", never as a kill. So each mutant was run against `tests/test_interp.py` and `tests/test_interp_pilot.py`, the only test files that import `wf.interp`. A mutant counts as killed only when pytest exits with code 1 and names the failing test.

| Mutant | Result | Failing test |
|---|---|---|
| V1: the arrival layer itself is not counted as "between" | killed | `test_table_rows_read_interval_bounds_onset_and_h4` |
| V2: the second-projection gate reads the smaller of the two folds' absolute cosines | survived | none |
| V3: the second-projection gate becomes "at most 0.5" | survived (boundary only) | none |
| V4: the second-projection value is taken over every H4 layer | survived | none |
| V5: the direct path is computed on the unedited fold | killed | `test_minimal_pilot_end_to_end` |
| V6: a layer without a direct-path entry is excluded | killed | `test_direct_path_share_tracks_the_removal_and_the_rule_excludes_it` |
| V7: the ablation plan leaves out the steering ratio | survived | none |
| V8: the ablation plan is capped at 100 instead of 400 | survived | none |
| V9: the steering gate reads the readout-free steering instead of the task steering | killed | `test_steering_ratio_is_gated_on_the_task_steering_and_uses_the_readout_free_direction` |
| V10: the linearity ratio uses the deepest layer's removal | survived | none |

## Test gaps (close them in the same round as R5-MF1; the code they guard is correct today)

- **T1 (D7, the reported value).** In `_report` of `tests/test_interp_pilot.py`, give a gated-out layer a value that moves the median over all layers (for example, gate out two of the three layers with values far from the third). Then a median taken over every layer fails. This kills V4.
- **T2 (D7, "in both folds").** Give the two folds different answer cosines at one layer (0.2 in fold 0, 0.7 in fold 1) and assert that the layer is gated out. This kills V2. Pinning the boundary (V3) needs a value of exactly 0.5.
- **T3 (D10, the ablation plan).** Assert that `plan["ablation"]["n_wanted"]` is the largest of the removal, gap-ratio, steering-ratio and perp-over-task needs, and that its cap is 400. This kills V7 and V8.
- V10 guards a diagnostic that is reported, not read. A two-layer case in `test_steering_ratio_is_gated_on_the_task_steering_and_uses_the_readout_free_direction` would kill it.
