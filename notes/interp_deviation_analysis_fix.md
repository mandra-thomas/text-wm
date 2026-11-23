# Interpretability pilot: analysis fix after a crash (deviation, 2026-11-23)

Frozen design: tag `interp-design-frozen-2026-11-22` (commit 7ade3b9), `notes/interp_design.md` with amendments 1 to 5. This fix is on branch `interp-analysis-fix`, made from the tag. It was made before any analysis output was read: the crashed run wrote no metrics (its folder holds only `config.json`, `git_sha`, `inputs.json` and `jobs.jsonl`), and the diagnosis read only item metadata (item sets, folds, verbs, hidden-state labels, the stage-0 index) and counts of items, never an activation value or an estimate.

## The crash

The CPU analysis `interp_pilot_analysis` (job 7252128, checkout at 7ade3b9, scikit-learn supplied as below) stopped after about six minutes:

```
wf/interp/pilot.py:655 run_interp_pilot_analysis -> hidden_probe_report
wf/interp/pilot.py:1038 hidden_probe_report -> answer_projection
wf/interp/pilot.py:351 answer_projection -> success_direction
wf/interp/analysis.py:421 success_direction: ValueError("no verb has both a success and a failure")
```

The failing call is the hidden-state belief probe (amendment 1, A4) at the answer position, not the H4 cosines (`h4_cosine_report` had already returned). It fails at the first probe layer (layer 0) for the first held-out fold (fold 0, so the probe is trained on fold 1's games), and it would fail in the same way at every probe layer (0, 4, ..., 32, 35) and for both folds, because which rows are used does not depend on the layer.

## The cause: the wrong rows for the success direction (a code bug)

At the answer position the probe is trained and read after two directions are projected out: the readout direction and "the neutral success direction of A3" (amendment 1, A1). A3 defines that direction on the training fold's determined successes and determined failures under the neutral task, and amendment 2 (B2) makes it a weighted mean of within-verb differences, using only failures after step 0 (`analysis.success_direction`). `fold_directions` and `h4_cosine_report` select those rows.

The frozen `hidden_probe_report` instead took the successes and later failures from the probe's own training rows. Those rows have a hidden-state label and are balanced within true outcome (A4). In the real data every labelled success has the same label:

- 130 of the 292 determined successes have a label, and every one of them is a "take X from Y" whose object was at Y at reset (label true). A take that works needs its object to be there.
- The other 162 successes have no label. Either their action names no object and receptacle (examine 56, go to 24, close 16, use 6, inventory 4), or the history acts on the object, which may have moved since reset (move 33, heat 9, clean 6, cool 5, take 3).

Balancing makes the two label values equally common among successes. With 130 true and 0 false, it keeps no success at all. The success set was therefore empty in both folds, so no verb had both a success and a failure:

| Held-out fold | Labelled training items (successes) | Kept after balancing | Successes kept | Failures after step 0 kept |
|---|---|---|---|---|
| 0 (trained on fold 1) | 2,401 (64) | 710 | 0 | 609 |
| 1 (trained on fold 0) | 2,395 (66) | 732 | 0 | 628 |

The population the design names is not empty. The training fold's determined successes and its failures after step 0, all with neutral prompts in stage 0, are:

| Held-out fold | Successes | Failures after step 0 | Verbs in both |
|---|---|---|---|
| 0 | 146 | 2,069 | clean, heat, move, take |
| 1 | 146 | 2,083 | clean, cool, heat, move, take |

So this is a row-selection bug, not an empty stratum the design failed to foresee: the design asks for a direction that can be estimated, and the code asked a different, always-empty set of rows for it. The mock end-to-end test did not catch it because its balanced training set has fewer than 20 items, so that fold is skipped before the answer position is reached.

## The fix

`pilot.success_population(sets, fold, have, f)` returns fold `f`'s determined successes with a neutral prompt and its failures after step 0 with neutral, aligned and swapped prompts. This is the same selection as `fold_directions` and `h4_cosine_report`. `hidden_probe_report` uses the training fold's population (`1 - f` for held-out fold `f`) for the success direction it projects out at the answer position, and it also loads the neutral-prompt residuals of those items (162 of them are outside the labelled set). Nothing else changes. In particular, the probe's training set is still balanced within true outcome as A4 says. A side effect of that rule, not changed here: because balancing keeps no success, the probe in effect trains on failures only.

No "not estimable" fallback was added. With the design's rows, both folds have four or five shared verbs. If a future item set had none, `success_direction` would still raise, which is the same behaviour as `fold_directions` and `h4_cosine_report`.

## What the fix changes, and what it does not

- **Hidden probe at the action's receptacle token.** Unchanged. It never uses the projection. The extra rows are looked up by key, so no value read there moves.
- **Hidden probe at the answer position.** This is the only quantity affected. At the frozen commit it had no value, because the run stopped at its first cell. It is now estimated as the design describes.
- **Everything else.** No other quantity reads the hidden probe: `table_rows`, `sample_size`, the ablation and patching readouts, the H4 rows and the direct-path share do not. The hidden probe is descriptive in the minimal pilot (section M). The GPU stages (stage 0, patching, ablation) never call this code, so their outputs stand. The analysis computes the hidden probe after the direct-path, ablation and H4-cosine sections, so in the crashed run those sections had been computed but never written: the metrics file is written only at the end.

## Checks

- **A test that reproduces the crash.** `test_hidden_probe_success_direction_is_the_training_folds_not_the_balanced_rows` in `tests/test_interp_pilot.py` builds items shaped like the real data: every labelled success is a take with label true, the other successes are unlabelled examine and move actions, and there are step-0 failures. On the frozen `pilot.py` it fails with the crash's own error (`analysis.py:421: ValueError: no verb has both a success and a failure`). On the fix, the direction projected out at every probe layer and fold equals `success_direction` on the training fold's successes, labelled or not, and its failures after step 0. Both positions' AUCs are finite.
- **Interp tests.** 104 passed: the 103 at the tag plus this one. They ran with scikit-learn 1.9.1 in a local CPU venv (torch 2.2.2, transformers 4.51.3, numpy 1.26.4).
- **Mutants.** Four mutants, AF1 to AF4, are in `scripts/mutation_check.py`. AF1 puts the frozen selection back, AF2 takes the direction from the held-out fold, AF3 lets step-0 failures in, and AF4 does not load the extra rows. Under origin/main's strict runner (a kill needs pytest exit 1 and a named failing test), all four were killed by the test above. The 240 earlier interpretability mutants still match their target text exactly as at the tag (`notes/testing.md`).

## Operational deviation: scikit-learn supplied outside the study environment

The study environment (`.envs/serve`) has no scikit-learn, and an earlier analysis attempt failed on the import. Job 7252128 ran with `PYTHONPATH=/storage/ice1/2/1/gmat3/pylib/interp_sklearn_1.9.1`, a separate `pip install --target` folder holding scikit-learn 1.9.1 with joblib 1.6.0, threadpoolctl 3.8.0 and narwhals 2.28.0. The study environment itself was not changed.
