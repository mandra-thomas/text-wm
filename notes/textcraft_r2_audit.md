# TextCraft round 2: independent pre-freeze audit (2026-11-22)

This note is the independent auditor's check of branch `textcraft-r2` at fb7e86b, before the freeze. Nothing was scored, and no round-2 score was read. No GPU or ICE job was used. Every check either reads the code or runs it on synthetic data, or runs it on the real round-2 records with random scores.

**Audited.** The six sign-off corrections reported as applied in 170db45, 6734c35 and fb7e86b (design 15.F1 to 15.F5 and 15.A), and whether anything that defines the experiment changed after the review-2 head f610413.

**Tools.** Two scripts are kept on the audit branch:

- `scripts/audit_textcraft_r2_stats.py` re-derives the statistics independently;
- `scripts/audit_textcraft_r2_mutants.py` runs the auditor's own mutants on throw-away copies of the repo.

## Verdict

**Fix first. The fixes are small: text edits to the design note, plus three tests. No change to the analysis code is needed for correctness.**

The code does what the six corrections require. Each statistical property was re-derived independently and holds:

- Holm over a fixed family of 50, with missing or NaN entering as 1;
- the TOST;
- the margin and when it is computed;
- goal-pair clustering;
- the Dn + Dc identity;
- R2;
- R3;
- no p-value outside the family.

The design note still contains two rule statements that contradict the sign-off corrections. Since the note is the document being frozen, those two sentences should be fixed first (must-fix 1 and 2). Four of the auditor's 12 mutants survive. Three of them point at properties that 15.A or 15.F says are tested when the tests do not pin them down; the fourth is a tie case of a descriptive rule. The code is correct today, but the tests should be added before the freeze so that the 15.A table is true as written (must-fix 3).

## Must-fix before the freeze

**1. The R5 sentence in 15.R still states the rejection that 15.F3 withdrew** (`notes/textcraft_r2_design.md:710`). The sentence reads: "*Rejects* the claim that the penalty is largest early, in either reading." Only a parenthetical follows it: "Reworded at sign-off, 15.F: R5 rejects the predicted attenuation, not content-based explanations." The sentence itself was not reworded. "In either reading" includes the content reading, which is exactly what 15.F3 says R5 does *not* rule out.

The code is right. At `r2_analysis.py:721` the rule rejects "the predicted attenuation of the penalty with added steps … not content-based explanations".

Failure scenario: a model is competent for T and T is confirmed below zero. A reader of the frozen 15.R would then report that both the timing and the content explanations are rejected. The code output and 15.F3 say only the timing (attenuation) claim is.

Fix: reword the sentence to match 15.F3.

**2. The R3 text in 15.R still says "Rejects" and "does not reject"** (`notes/textcraft_r2_design.md:708`). The text reads: "… and does not reject. *Rejects* the decomposition for that model, since …". The parenthetical at line 703 says R3 is a descriptive diagnostic with no familywise error control, but the body still uses the language of a test decision.

15.F1 and the code (`r2_analysis.py:716-717`, "counts against the decomposition for this model (descriptive diagnostic; no familywise error control)") have the right wording.

Failure scenario: someone quotes 15.R as the pre-registered rule. They would then read R3 as a formal rejection, which is the error-control claim the owner asked to remove.

Fix: replace "does not reject" and "*Rejects*" with the "counts against …, descriptively" wording of 15.F1.

**3. Three properties that 15.A or 15.F reports as tested are not pinned by the tests.** The code is correct for all three; my scripts verify it. Each mutant below leaves all 141 tests passing.

- **X4, the role gap on another row set.** In `contrast_inputs` (`r2_analysis.py:553`), the role gap was changed to read goal-role and *all* ingredient failures that have a two-item line. The 15.A row "Dn + Dc equals the role gap exactly … on the 996 shared rows" cites `test_audit_naming_plus_completion_is_the_role_gap_on_the_shared_rows`. That test recomputes the gap from `two_item_rows` itself and never calls `contrast_inputs`.

  The other identity test, `test_role_gap_descriptive_equals_naming_plus_completion`, uses `_planted` scores without noise. In those scores every ingredient has the same ΔL, so a change in the ingredient row set cannot show.

  Failure scenario: a later edit points R3's role gap at a different row set. R3 would then compare P with a gap that is no longer Dn + Dc, and no test would fail.

  Fix: a test that calls `A.contrasts` (or `contrast_inputs`) with noisy scores and asserts that `role_gap.estimate == naming.estimate + completion.estimate` to 1e-12.
- **X1, R3 firing without a signed unmatched gap.** At `r2_analysis.py:709`, `bool(p_sign and rg_sign and p_sign != rg_sign)` was changed to `bool(p_sign and p_sign != rg_sign)`. Under the mutant, R3 fires when P is confirmed and *both* role-gap intervals cross zero. That violates 15.F1's condition that "the unmatched role gap's 95% interval lies entirely on the other side of zero".

  `test_rejection_list_per_model` has no case in which the unmatched interval crosses zero.

  Fix: add the case where `rg_ci=(-0.2, 0.3)` and `m_ci=(-0.2, 0.3)`, and assert that R3 does not fire.
- **X12, the R4 difference reversed at the call site.** At `r2_analysis.py:1001` the arguments were swapped, so the code computed Qwen minus Llama while the output still labels it `difference_llama_minus_qwen`. The builder's own mutant reverses the subtraction inside `lift_difference`, but no test checks the sign of the reported difference in `analyse()`; `test_audit_inference_map` only checks `n_items > 0`.

  Failure scenario: the one descriptive number that replaced R4 would be reported with its sign reversed.

  Fix: in an `analyse()` test, plant different lifts for Llama and Qwen and assert the sign and value of `difference_llama_minus_qwen`.

## Should-fix (wording; no rule changes)

- **The central question in section 1** (`notes/textcraft_r2_design.md:42`) still asks whether the effect arises "from a goal-completion or naming mechanism, from poor reasoning about the environment's state, or from a combination". It has no pointer to 15.C or 15.F5. 14.U5 (lines 573-582) carries a superseded marker, but section 1 does not.

  Suggest adding: "(answered descriptively by 15.C under the neutral labels of 15.F5; no answer identifies an internal mechanism)".
- **The across-models table in section 8** (`notes/textcraft_r2_design.md:362`) has the row "The same mechanism reading in every model where it is read → One mechanism across families." Suggest "The same decomposition reading in every model → the same reading across families".

  The "mechanism readings" heading (line 326) and "penalty present, mechanism unresolved" (code `r2_analysis.py:658`) refer to which term of the algebraic decomposition is negative, not to an internal mechanism. They can stay, but "decomposition reading" would be more consistent with 15.F5.
- **The R2 wording in 15.R** (line 698), "and none is confirmed below zero", and in 14.R6 (line 601) leaves it open whether "none" includes Llama-3.1. The code (`r2_analysis.py:769-775`) applies "none *other than Llama-3.1*". That is the only reading under which R2 means "the penalty is not shared beyond Llama", and it matches F3. Suggest writing "none of them".
- **The 15.S bullet** (line 750) quotes the old reading suffix "measurable effect only; not competent for …". The code now says "does not discriminate successes from failures for …".
- **The size of the Holm family is not enforced in code.** `family()` (`r2_analysis.py:582-594`) builds 10 tests per model in `expected`, and `expected = list(cfg.get("models", R.TAGS))` (line 1057). The frozen config lists the five models, so the family is 50.

  Failure scenario: a run with a config listing two models (for example an early look at Qwen and Llama) produces verdicts from a Holm family of 20, with no warning. My script confirms this: four models give a family of 40.

  Suggest `assert set(expected) == set(R.TAGS)` in `analyse`, or building the family always over `R.TAGS`. This is hardening only; nothing in the frozen pipeline triggers it.
- **The inference map does not name T's applicability gate.** The gate (`r2_analysis.py:570`, "the one-look-up role gap's 95% interval below zero") and the "comparable to the one-look-up change" check (line 575) are interval-based decisions outside Holm. They are pre-registered (section 6), and the gate can only remove a verdict, never create one, so they add no familywise risk. `INFERENCE_MAP` covers them only as "every other readout". Suggest listing them as pre-registered gates.
- **The central question's tie case is not tested.** Mutant X10 changes "H6 tracks at least as well" (`<= 0`, `r2_analysis.py:735`) to "strictly better" (`< 0`), and no test notices. A test with a tracking difference of exactly 0 would pin it down.
- **`contrasts()` computes ps for three auxiliary comparisons** (the role gap, the timing role gap at one look-up, and one look-up against none) and then drops them before output. This is harmless, and my output walk found no p outside the 50, but the computation is wasted.

## (a) The six corrections

| Item | Code | Note | Remaining wording |
|---|---|---|---|
| 1. R3 descriptive, outside the 50, exact specification | Holds. `kind = DIAGNOSTIC` ("outside the Holm family; no familywise error control"). Rows are `two_item_rows` (996). The matched gap is computed as 15.F1 states: length bin × craftable/raw/mixed × one/several; at least 5 items per role in a cell; each cell weighted by its smaller role count; cells recomputed in each resample; 20,000 pair resamples. On the records, 13 of 15 cells (981 items) are used. | 15.F1 is exact and matches the code | 15.R line 708, "Rejects" / "does not reject" (must-fix 2) |
| 2. R4 descriptive, with a paired interval | Holds. No `fires` key. `lift_difference` takes per-item differences on the 990 ingredient failures (162 pairs) and resamples goal pairs once for both models, with 2,000 draws | 15.F2 holds | 15.R line 709 keeps the old rule text, but under a clear "Withdrawn" marker: acceptable |
| 3. A negative T rejects only the attenuation | Holds (line 721) | 15.F3 holds | 15.R line 710, "in either reading" (must-fix 1) |
| 4. Discrimination, not competence | Holds. The interpretation, reading and STATUS strings never say "competent" or "understand", and the gates are unchanged | 15.F4 and the section 8 table hold. The note's "competent for X" is defined in 15.F4 | 15.S line 750 (stale suffix); deviations.md line 483 (a historical log entry, acceptable) |
| 5. Neutral central-question labels | Holds. The criteria are byte-identical to f610413 apart from the label strings | 15.C and 15.F5 hold | Section 1 line 42 and the section 8 table line 362 (should-fix) |
| 6. Statistical audit | Every property holds (section (b) below) | 15.A's table is accurate on the code, but two "how it was checked" entries overstate the tests (must-fix 3) | n/a |

No code string, note or test in round 2 claims familywise error control for R3 or R4. The phrase "competent predictor" survives only in historical text: the mutation log and testing.md line 142 (an old test list). It does not occur in code.

## (b) Statistics re-derived

These are the results of `scripts/audit_textcraft_r2_stats.py`. It uses synthetic data, and the real records (`r2_items.jsonl`, sha256 `e6da991f…`) with Gaussian random scores. The full log is reproduced at the end of this note.

**Holm over a fixed 50.** I compared `family()` with a textbook Holm step-down written separately, on 200 random families with unscored models, missing contrasts and NaN ps. It matched to 1e-15 every time, with 50 entries each time.

A single real p of 0.0005 among 49 padded tests is adjusted to 0.025, that is, multiplied by 50. The only caveat is the config dependence noted above.

**TOST and its smoothing.** On 500 random draw sets, `tost_p` equals the larger of the two smoothed one-sided shares:

$$p_{\text{TOST}} = \max\left(\frac{\#\{b \le -\delta\} + 1}{B + 1},\ \frac{\#\{b \ge \delta\} + 1}{B + 1}\right).$$

Here $b$ ranges over the bootstrap draws, $B$ is their number and $\delta$ is the margin. `bootstrap_p` equals section 6's two-sided formula exactly.

On 20,000 draws, TOST p below 0.05 agrees with the 90% percentile interval lying inside $(-\delta, \delta)$. The smallest attainable p is $1/(B+1)$, never 0. A margin that is NaN, zero or negative gives NaN, which enters the family as 1.

**The margin and its timing.** On the real records, the margin's SD is the $n - 1$ item SD of ΔL over exactly the 1,500 main failures, and

$$\delta = \min(0.3\,\mathrm{SD},\ 0.4).$$

When the SD is small, the 0.3 SD rule binds. I instrumented `model_block`: `margin_of` is called exactly once per model, before the first `tost_p` call, and all 8 TOST calls receive that value. Changing every score outside the main failures leaves δ unchanged.

**Goal-pair resampling everywhere.** In a full `analyse()` of all five models on the real records, I intercepted every `draws` and `sign_flip_p` call (1,431 calls) and every `random.Random.choices` call (18,470). Every cluster list and every resampled population was a set of goal-pair ids. None was item-level. The fast paths (mean and gap) give the same draws as the general path, and the sign flip flips whole pairs.

**The Dn + Dc identity on the real 996 rows.** `two_item_rows` holds 500 goal-role and 496 class-matched ingredient failures. All have a two-item line, none is used up, the ingredients are "too few" or "never obtained" only, and none has its partner goal already done.

In `contrast_inputs`, naming, completion and the role gap use the same 996 rows and clusters in the same order. Over 5 random score sets, Dn + Dc minus the role gap is at most $5.6 \times 10^{-17}$ in the estimate and $7.8 \times 10^{-16}$ in every bootstrap draw, because the same resamples are used. The weighting is item-weighted in both.

The matched gap equals my own computation: the sum over used cells of the smaller role count times the difference in role means, divided by the sum of the smaller role counts. It uses 13 cells and 981 items.

**R2.** I enumerated all $5^5$ verdict patterns (below, equivalent, above, inconclusive, not applicable) against all $3^5$ competence patterns (true, false, missing), 759,375 cases in all. Every case agreed with a spec written independently from 15.R / 14.R6, with Llama-3.1 excluded from "the others": it fires, does not fire, or is undecided as the spec says.

**R3.** I enumerated 384 cases: the P verdict, the side of the unmatched interval, the side of the matched interval (including NaN) and competence for P, Dn and Dc. All agreed with 15.F1, and every case carries `kind = DIAGNOSTIC`. An interval whose lower bound is exactly 0 does not count as "above zero".

**No p outside the family.** I walked the whole `analyse()` output except `family`. The only `p_*` keys are the 50 family tests (5 models × 5 contrasts × differs/equivalent): none extra, none missing. `R4_lift_comparison` has no `fires` and no p. R3 is labelled a diagnostic for every model. The R4 difference uses 990 items in 162 pairs.

## (c) Diff check, f610413 to fb7e86b

`git diff --name-only f610413 fb7e86b` lists only these files:

- `notes/deviations.md`
- `notes/testing.md`
- `notes/textcraft_r2_design.md`
- `notes/textcraft_r2_mutation_log.txt`
- `scripts/mutation_check.py`
- `tests/test_textcraft_r2.py`
- `wf/textcraft/r2_analysis.py`

The following show no diff at all:

- `wf/textcraft/prompts.py`, `r2.py`, `r2_items.py`, `r2_score.py`, `r2_heuristics.py`, `build.py`;
- `wf/config/` (the five-model list and the YN question), `wf/constants.py` (revisions), `wf/families.py`, `wf/runs.py`;
- `wf/textcraft/data/` (records, few-shot examples, manifest, tokenization check);
- `pace/` (sbatch scripts).

The `r2_analysis.py` diff touches only wording, R3's label, R4 (now descriptive, plus `lift_difference`) and `INFERENCE_MAP`. `populations`, `two_item_rows`, `contrast_inputs`, the competence gates, `margin_of`, `family`, `verdict` and the central-question criteria are unchanged.

The hashes all agree:

- **Records.** `r2_items.jsonl` has sha256 `e6da991f9eb6c927…`, which equals the manifest, the tokenization-check file and the f610413 manifest (the two manifests are byte-identical).
- **Few-shot examples.** `fewshot.json` has sha256 `e6ddab93…`, equal to the manifest.
- **YN question.** sha256 of `R.YN_QUESTION` is `5e31ac7a4a013636…`; its file `r2.py` is unchanged.
- **Item hashes.** The digest over all 2,982 records is `c0da9f1136632d6a…`; both the records and `item_hash`'s inputs are unchanged.

The population sizes are as designed:

- 1,500 main failures and 600 successes;
- 504 goal-role failures and 990 ingredient failures, of which 496 are class-matched;
- 336 + 332 padded failures;
- 675 parent failures and 207 parent successes.

**Result: nothing that defines the experiment changed.**

## (d) Tests and mutants

`tests/test_textcraft_r2.py`: **141 passed** (5 min 3 s, at fb7e86b). The frozen recipe database had to be copied in first, because it is not committed; it was copied from the `textcraft-r2` worktree and its hash was checked against `FROZEN_SHA256`.

I ran my own mutants of the changed logic under the strict rule: a mutant counts as killed only if pytest exits 1 and names a failing test. Each mutant ran on a throw-away copy of the repo.

| Mutant | Result | Killing test or outcome |
|---|---|---|
| X1 R3 fires without a signed unmatched gap | survived | exit 0; 141 passed in 224.25s (0:03:44) |
| X2 R3 matched gap only needs not to be opposite | killed | `test_rejection_list_per_model` |
| X3 R2 counts Llama-3.1 among the others | killed | `test_across_models_patterns_and_f3` |
| X4 role gap on a different row set from Dn and Dc | survived | exit 0; 141 passed in 224.54s (0:03:44) |
| X5 margin SD from the ingredient failures | killed | `test_margin_is_capped_at_four_tenths` |
| X6 TOST takes the smaller one-sided p | killed | `test_statistics_helpers` |
| X7 Λ and P sign flip by item | killed | `test_audit_every_resample_is_over_goal_pairs` |
| X8 equivalence verdict on the unadjusted p | killed | `test_verdicts_are_statistical_and_never_withheld` |
| X9 lift difference over class-matched ingredients only | killed | `test_lift_difference_is_paired` |
| X10 H6-best needs strictly better tracking | survived | exit 0; 141 passed in 290.98s (0:04:50) |
| X11 NaN ps dropped from the family instead of entering as 1 | killed | `test_family_is_fixed_at_fifty_and_pads_missing` |
| X12 R4 difference reported as Qwen minus Llama | survived | exit 0; 141 passed in 250.98s (0:04:10) |

8 of the 12 mutants were killed. The 4 survivors:

- X1, X4 and X12 are test gaps for properties that 15.A or 15.F claims (must-fix 3). In each case the current code is correct, as `scripts/audit_textcraft_r2_stats.py` checks directly; for X12 the call site passes Llama as model a, so the difference is Llama minus Qwen as labelled.
- X10 changes only the tie case of the central question's "H6 tracks at least as well" condition (a difference of exactly zero). Exact zero is unlikely with continuous scores, and the mapping is descriptive. This is a minor test gap.

The builder's 145 mutants were not rerun; their log is at `notes/textcraft_r2_mutation_log.txt`.

## Full log of the statistics script

```
PASS Holm equals textbook Holm over a fixed 50 (NaN/missing -> 1), 200 random families
PASS single real p of 0.0005 among 49 padded -> Holm 0.025
PASS family size depends on the expected list (4 models -> 40): no hard guard on 50 [finding]  [len=40]
PASS TOST p = max((#<=-δ + 1)/(B+1), (#>=δ + 1)/(B+1)) on 500 random draw sets
PASS two-sided bootstrap p = min(1, (2 min(#<=0, #>=0) + 1)/(B+1)) on 500 random draw sets
PASS TOST p < 0.05 agrees with the 90% percentile interval inside (-δ, δ), 51 true means from -0.5 to 0.5  [51 of 51 agree]   (rerun after fixing a degenerate first version of this check)
PASS TOST undefined margin -> NaN (enters the family as 1)
PASS TOST smoothed minimum is 1/(B+1), never 0
PASS sign-flip p smoothed minimum is 1/(n+1)
INFO item-hash digest over 2982 records: c0da9f1136632d6ae8285057b6b78a9c7ac0ebeabdd91a2a43f1a989072cc51b
INFO sha256(R.YN_QUESTION): 5e31ac7a4a0136367d51dfec938347caa71c0b2f9d740fac95d34b0034180f05
INFO population sizes: {"main_failures": 1500, "successes": 600, "goal": 504, "ingredient": 990, "ingredient_matched": 496, "padded": 920, "padded_goal": 336, "padded_ingredient_matched": 332, "parent_failures": 675, "parent_successes": 207, "goal_successes": 300, "ingredient_successes": 300, "goal_successes_padded_bins": 200, "ingredient_successes_padded_bins": 200}
PASS population sizes as designed (1500 main failures, 600 successes, 675 parent failures, 207 parent successes, 990 ingredient, 500+6 goal)  [goal=504]
PASS margin SD is the n-1 item SD of ΔL over the 1,500 main failures  [n=1500, sd=2.8816]
PASS δ = min(0.3 SD, 0.4)
PASS δ follows 0.3 SD when that is below 0.4  [δ=0.0864]
PASS margin computed once per model, before the first TOST, and every TOST uses it  [1 margin call(s), 8 TOST calls]
PASS δ depends only on the main failures' aligned and swapped scores
PASS two_item_rows: 996 rows, 500 goal-role and 496 class-matched ingredient  [{'goal': 500, 'ingredient': 496}]
PASS two-item rows: goal rows exclude used up; ingredient rows are too few / never obtained only
PASS two-item rows all have a two-item line; none has its partner goal already done
PASS Dn, Dc and the role gap use the same 996 rows and clusters (contrast_inputs)
PASS Dn + Dc = role gap (item-weighted), 5 random score sets  [max |err| 5.55e-17]
PASS Dn + Dc = role gap in every bootstrap draw (same resamples)  [max |err| 7.77e-16]
PASS role_gap and matched role gap read two_item_rows (same row set as Dn, Dc)
PASS matched role gap: 13 of 15 cells used, 981 items (as 15.F1 states)  [13 of 15 cells, 981 items]
PASS every two-item row has at least one missing input (profile well defined)
PASS matched role gap = Σ min(n_g, n_i)(mean_g - mean_i) / Σ min(n_g, n_i) over cells with ≥5 per role
PASS R2 three-way logic matches the spec on all verdict x competence combinations  [759375 cases, 0 mismatches]
PASS R3 fires iff competent for P, Dn, Dc; P confirmed; unmatched gap CI on the other side; matched CI on the same side as the unmatched  [384 cases, 0 mismatches]
PASS R3: an unmatched interval with lower bound exactly 0 does not count as above zero
PASS every draws()/sign_flip_p() call in a full analyse() gets goal-pair ids as clusters  [1431 calls, 0 bad]
PASS every rng.choices population (the resampled units) is a set of goal-pair ids  [18470 choices calls, 0 non-pair]
PASS clusters are coarser than items in every call (no item-level resampling)
PASS fast mean path gives the same draws as the general path
PASS fast gap path gives the same draws as the general path
PASS sign flip flips whole pairs (two clusters -> p can only take the 2-cluster values)  [p=1.000]
PASS the only p-values in the output (outside `family`) are the 50 family tests  [extra=[], missing=0]
PASS family has 50 entries and verdicts exist
PASS R4 output has no 'fires' and no p; it is labelled descriptive
PASS R3 labelled a diagnostic in every model's output
PASS R4 lift difference on the 990 ingredient failures, 162 pairs  [990 items, 162 pairs]

39 checks, 0 failed
```
