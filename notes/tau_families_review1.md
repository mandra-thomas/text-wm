# τ-bench wave-A families extension: independent review 1 (validation steps 9 and 9b)

Reviewed 2026-11-22 on branch `tau-families` at 7c4a571 (commits 1b3a756, 80a59fa, 47e2c60, 137dd01, 7c4a571), checked
out as `tau-families-review1`. Nothing was run on a GPU or on ICE. Read: `notes/tau_families_design.md`,
`wf/tau/families.py`, the diff of `wf/tau/pipeline.py` and `wf/runs.py`, the seven configs, both sbatch scripts,
`tests/test_tau_families.py`, the tau-fam mutants in `scripts/mutation_check.py`, and for context `notes/tau_design.md`,
`notes/tau_metrics.json`, `notes/families_results.md`, `wf/families.py`, `wf/analysis/stats.py`, and on `main`
`notes/tau_dx.md` (the exploratory τ diagnostics, 4df8ed9) and the results-note qualification eac03f7.

**Verdict: not yet ready to freeze.** The engineering is sound: the per-model verdict really is the τ pipeline's own
code, the change to `pipeline.py` cannot alter the published τ verdicts, the competence rule, the paired comparison
bootstrap, the reference results check and the placement rule compute what the note says, and the timing has a wide
margin. One scientific problem blocks the freeze. The note's description of the two reference patterns, and the
cross-model readings built on them, were written without the τ diagnostics that landed on `main` 28 seconds before the
design commit, and those diagnostics overturn both descriptions. Left as is, a likely outcome for Llama 3.2 3B (and
possibly Mistral) would be printed by the analysis code as a reading the project already knows to be misleading. The fix
is a design amendment and a few lines of reporting code; it needs no GPU and no change to the confirmatory rule.

Classification: **must-fix** blocks the freeze; **should-fix** should be done before the freeze but would not by
itself make a result wrong; **note** is for the record.

## 1. Must-fix

### M1. The two reference patterns are described as they were pre-registered, not as the diagnostics found them, and the cross-model readings would misread a likely outcome

**Where.** `notes/tau_families_design.md` lines 30–32 and 36–38 (what the reference models showed and the question
asked of the new models), lines 385–394 (the pattern table), lines 409–410 (the outcome rows "Llama-3.1-8B's pattern"
and "Qwen3-8B's pattern"), lines 432–434 (Llama 3.2 3B predicted to have "the failure-specific reading of
Llama-3.1-8B"); `wf/tau/families.py` lines 610–618 (`specificity`) and 684–709 (`pattern_readings`).

**What the note says.** Qwen3-8B is described as a general task-consistency bias in which "the working witness call on
the same order moves as much" (line 31, and "fail minus witness about 0 or negative" at line 410). Llama-3.1-8B is
described as "specific to failing calls (fail minus witness above 0)", i.e. "the task overrides a visible fact about
the named object, more for failing calls than working ones" (line 409). The new models are to be sorted into one or
the other by the binary rule of the τ outcome table: the row is "failure-specific" when the 95% interval of fail minus
witness lies above zero, and "task-consistency" otherwise. The pattern table then draws conclusions such as "both
scored models have the failure-specific reading, as Llama-3.1-8B: the task-consistency pattern is specific to Qwen3-8B
among the models tested", and "Llama 3.2 3B has Llama-3.1-8B's reading row".

**What the project has since found** (`notes/tau_dx.md` on `main`, exploratory, from the confirmatory scores):

- For Qwen3-8B the witness does not move "as much": it moves $+9.45$ nats against $+3.98$ for the failing call, so fail
  minus witness is $-5.47$ $[-5.87, -5.05]$. The task-consistency reading survives, in a stronger form.
- For Llama-3.1-8B the "failure-specific" row rests on a pooled fail minus witness of $+0.09$ $[0.04, 0.14]$ on an
  effect of $+0.47$. The witness moves $+0.38$, about 80% as much as the failing call; fail minus witness is positive in
  only 49% of items; it is $+0.68$ in one status cell and on the 36 returns, and $-0.27$ in the unavailable class. The
  pooled $+0.47$ is itself an average of $+1.23$ (status class) and $-0.33$ (unavailable class, a small reversal). The
  diagnostics conclude that the pre-registered reading "the ALFWorld effect in its stronger form" does not hold up
  item by item, by class or across prompts, and `main` (eac03f7) now qualifies it in the results note.

**Why it blocks the freeze.** The extension's main descriptive question is whether the new models "look like Qwen3-8B
or like Llama-3.1-8B" (line 36). The two templates it compares against are mislabelled, and the binary row that sorts
new models has no magnitude criterion: any positive effect whose fail-minus-witness interval clears zero is called
"failure-specific", however much of the effect the witness shares. With 856 items and an interval half-width of about
$0.05$ nats for a Llama-sized model, a fail-minus-witness of $+0.08$ will clear zero.

**Failure scenario (likely, not contrived).** Llama 3.2 3B, which shares Llama-3.1-8B's tokenizer and prompt bytes and
was predicted to resemble it, comes out at $+0.40$ $[0.33, 0.47]$ with the witness at $+0.32$ and fail minus witness at
$+0.08$ $[0.03, 0.13]$, positive in the status class and negative in the unavailable class. Mistral does the same at a
larger scale. Then `specificity` returns "failure-specific" for both, `outcome_reading` prints "The task overrides a
visible fact about the named object, more for failing calls than for working ones: the ALFWorld effect in its stronger
form", and `pattern_readings` prints "The task-consistency pattern is specific to Qwen3-8B among the models tested" and,
if the item shares happen to match, "Llama 3.2 3B has Llama-3.1-8B's reading row". Every one of these sentences would
be the reading the project has already retracted for Llama-3.1-8B.

**Fix (amendment before the freeze; no GPU; the confirmatory rule untouched).**

1. Correct the reference descriptions in sections 1, 7.5, 7.7 and 8, citing `notes/tau_dx.md`: Qwen's working calls move
   more than twice as much as its failing calls; Llama-3.1-8B's pre-registered "failure-specific" row comes from about
   a fifth of its effect, from one cell, with opposite-sign class effects.
2. Keep the per-model verdict and `outcome_reading` exactly as they are (they are the τ rule, and must stay identical
   to the reference analysis), but do not use the outcome table's row as the cross-model label. In the extension's own
   layer (`specificity`, `pattern_readings` and the pattern table), name the row by its mechanical criterion ("fail
   minus witness above 0" / "not above 0") and pre-state descriptive qualifiers that are always reported beside it: the
   witness effect as a share of the failing call's effect, and fail minus witness and the failing effect in each
   pre-specified class (`status`, `unavailable`), all of which the protocol already scores.
3. Pre-state when a cross-model sentence about "failure-specific" behaviour may be written. One workable rule: only
   when fail minus witness is above 0 with its interval, and the witness moves less than half as much as the failing
   call, and fail minus witness is not below 0 in either pre-specified class. Otherwise the model is described as
   "a shift shared by working and failing calls" with the numbers. The author can choose a different threshold, but it
   must be fixed now.
4. Add an outcome row and a pattern row for opposite-sign class effects (the two pre-specified classes with intervals
   on opposite sides of 0): "the pooled effect depends on the class mix; reported per class". This is the outcome the
   reference Llama actually had, and a Llama 3.2 model is the most likely to repeat it. The per-class numbers are
   already computed (`population_block` → `by_class`); only the reading is missing.
5. Restate the prediction for Llama 3.2 3B (line 434) in these terms, and drop "Llama-3.1-8B's pattern" wording from
   the reading table.

Add tests for the new rule (a constructed model with fail minus witness above 0 but a witness at 80% of the failing
effect must not be called failure-specific in the cross-model text; opposite-sign classes must produce the class-mix
reading).

## 2. Should-fix

### S1. The share-against-mean divergence (Wave B precedent) has no pre-stated reading

**Where.** Design §7.4 (line 347–351), §7.5 table, §7.7; `families.py` `item_share` (472–486) and `pattern_readings`.

The verdict is on the mean, which is right, but the note never says how a positive mean with an item share at or below
one half is read. Wave B's review found exactly this in a competent model (Qwen3-4B in ALFWorld, share 0.49, clear mean
effect from a minority of items), and the reference Llama-3.1-8B here is not far from it (59% of items positive, median
item effect $+0.25$ against a mean of $+0.47$). As written, such a model would get "replicates" or "small effect,
positive", the outcome-table reading "the task overrides a visible fact", and it would count toward "the direction ...
appears in every interpretable model", with nothing saying the effect is carried by a minority of items. The families
design already had the needed row (`notes/families_design.md` line 207: "the effect is carried by a minority of items
with large shifts; reported as such, not as a general shift"). Add it here, for the first-token readout: when the
verdict is positive but the item share's 95% interval includes or lies below 0.5, the result is reported as carried by
a minority of items, and the direction sentence of the pattern table is qualified for that model. Also state how the
two comparison numbers are read when they disagree (the share governs statements about breadth, $\Delta L_1$ about size
on each model's own scale), and report the median item effect beside the mean.

### S2. The fixed Holm family of two is not what the run path does

**Where.** `wf/tau/families.py` lines 849 and 853.

`family_verdicts` and `sensitivity_verdicts` are called with `tuple(m for m in SCORED if m in cfg["models"])`. The
analysis refuses to run while any listed model has a missing shard (`waiting`), so the design's case "a model whose
score is missing enters with p = 1" (line 305) can only be reached by removing that model from `models` in the
config, and that shrinks the family to one. Scenario: Llama 3.2 3B fails its smoke or scoring and is dropped from
`tau_family_analysis.yaml`; Mistral's permutation p of 0.006 is then compared with 0.01 unadjusted ("replicates")
instead of at 0.012 (not "replicates"). The unit test `test_holm_runs_over_the_two_scored_models_at_a_fixed_size` checks
`family_verdicts` with its default argument, so it does not see this. Pass `SCORED` unconditionally in both calls, and
add an end-to-end test with `models: [mistral-7b-v0.3]`.

A related edge, a note rather than a fix: `wf.analysis.stats.holm` drops a NaN p from the count, so a scored model with
a NaN permutation p would also shrink the family. `P.paired` filters NaN differences before the permutation, so this
needs an empty population; it cannot happen on the real items.

### S3. The "matches" rule is almost certainly unreachable, and the note does not say so

**Where.** Design §7.5 lines 369–372; `families.py` line 656.

"Matches" needs the 90% paired interval of the item-share difference inside $\pm 0.05$. In ALFWorld wave A, with 4,666
items, the share differences had 95% half-widths of about 0.03 to 0.05, and no comparison met the margin after Holm.
Here there are 856 items in 175 users. Scaling by the square root of the item ratio, a factor of about 2.3, gives 95%
half-widths of about 0.07 to 0.11 and 90% half-widths of about 0.06 to 0.10, wider than the margin itself. A direct
estimate gives the same picture: for two models with shares near 0.6 whose moves are uncorrelated, the per-item
difference has variance about 0.48, so the unclustered standard error is about $\sqrt{0.48/856} \approx 0.024$, the
90% half-width about 0.039, and with any within-user correlation more. "Matches" would then need an estimated
difference within about $\pm 0.01$, and usually could not be reached at all. Only strongly correlated models (Llama 3.2
3B against Llama-3.1-8B, perhaps) have a chance. The note should say this before data: the expected reading of a small
difference is "no difference detected", and the "Llama line" pattern row that needs "matches" will very probably not
be met whatever the models do. The smoke analysis could report a projected 90% half-width of the share difference for
the record. Do not lower the bar; state it.

### S4. The decomposition is complete on the first token but not on yes/no, and the analysis does not compute all of it

**Where.** Design §4.2 lines 183–185 and 190–193; `families.py` `controls_block` (539–565).

On the first token the protocol keeps the failing call and the witness under aligned, swapped and neutral, so goal
facilitation (aligned against neutral), wrong-goal suppression (swapped against neutral) and failure-specificity
(failing minus witness) are all identified. The analysis computes fail minus witness only for aligned minus swapped.
The neutral-anchored components of fail minus witness, i.e. (aligned minus neutral) and (swapped minus neutral) for the
failing call minus the same for the witness, are what the diagnostics used to separate the authorization reading
(swapped below neutral for both calls) from the others. Add these two contrasts with `contrast_of_effects` (no new
passes).

On yes/no the protocol keeps the failing call under all three lines but the witness only under neutral. So facilitation
and suppression on yes/no are identified for failing calls, but yes/no failure-specificity is not, and neither is the
working-call side of the yes/no task judgement. The diagnostics' evidence that Qwen's anchored yes/no carries a task
judgement has two halves: among calls whose first token confidently predicts JSON, Qwen answers No for 17%, 67% and 93%
of failing calls under aligned, neutral and swapped, and for 32%, 51% and 91% of working calls. The first half (failing
calls, and the per-line intercepts of yes/no regressed on the first token) is computable from this protocol and would
detect the finding in a new model, but the analysis does not compute it; pre-specify it as descriptive. The second half
needs the witness's yes/no under aligned and swapped, which were cut, and it is the τ analogue of the any-task
rejection of working moves that the decision simulation found to be half of the harm mechanism. Either restore those
two passes per item (1,712 passes per model, about 0.7 GPU-hours for Mistral and 0.3 for Llama 3.2 3B; Mistral's
total would then be about 6.1 hours, just over the 6-hour aim) or state plainly in §4.2 and §9 that the working-call
side of the yes/no task judgement and yes/no failure-specificity are not measured for these models. The statement that
the witness's aligned and swapped yes/no are "descriptive in the reference design; only the neutral one is needed" is
true of the confirmatory rules but undersells what is lost.

### S5. Test gaps: 10 of my 15 non-equivalent mutants survive

**Where.** `tests/test_tau_families.py`; the mutants are listed in section 4.

The code under each of these mutants is correct as written, so nothing computed today is wrong. But the author's 58 of
58 kill rate does not mean the analysis logic is pinned: my own mutants, chosen at the points this review cares about,
mostly survive. The ones that matter, with the test that would kill each:

- **Placement threshold** (R1, R2). One quarter and one sixth both survive, because the test's cases (a shift of 0.52
  and 0.32 against a native effect of 2.0, threshold 0.4) sit far from the threshold. Add 0.42 and 0.38, intervals off
  zero.
- **Reading row without discrimination** (R3). `specificity` can drop its discrimination check unnoticed; every
  pattern-reading test has `discrimination_holds: True`. Add a positive verdict with discrimination failing and
  assert the row is None.
- **Competence boundaries and mass** (R4, R5, R6). The median of the two-way mass can become a mean, and the 0.95 and
  0.70 bounds can become strict, without a failure. Add constructed populations at exactly AUC 0.70 and a failing Yes
  share of exactly 0.95 (both must pass), and a mass distribution whose mean and median fall on opposite sides of 0.5.
- **Comparison bootstrap size** (R7). Resampling half the users per draw survives; no test checks an interval's width.
  Add a case with a known clustered spread (or compare `_paired_draws` against `cluster_bootstrap` on one model's
  rows, which should agree in distribution).
- **Item-share clustering** (R11). The item share's interval can ignore users. Add users whose items all move or all
  stay, and assert the clustered interval is wider than the per-item one.
- **Llama-line pattern** (R12). "differs, within the margin" is accepted as "matches" unnoticed. Add the case.
- **In-job readout report** (R13). The witness's reply start can be dropped from the in-job check. The full token check
  covers every item before scoring, so this is defence in depth only; add a witness whose output starts with another
  token.

R8 (the token check's "all replies checked" count removed) is equivalent: a reply goes unchecked only when the template
raises, and that already records a problem. Mutants R9, R10, R14, R15 and R16 are killed.

### S6. Minor reporting gaps that a write-up could trip on

- The comparison entries carry a `reading` field ("differs", "matches") even for a model that is not interpretable or is
  gated to inconclusive, and the yes/no comparisons are computed for models that fail the yes/no competence rule. The
  design says these are left out of every cross-model reading, but the metrics file does not mark them. Add the two
  models' interpretability (first token and yes/no) to each comparison entry.
- Design §9 (lines 462–463) repeats "The `status` class has no discrimination evidence". The diagnostics showed both
  reference models read order status against the tool (status-rule term $+4.9$ and $+3.0$ nats); it lacks a
  determined witness, not evidence of record use. Reword.

## 3. Notes (no change needed)

**The per-model verdict is the τ pipeline's own code.** `family_verdicts` builds the same inputs the reference
analysis builds (`P.checks_of` on a controls block whose first-token entries come from the same `P.discrimination`,
`P.paired`, `P.contrast_of_effects` and `P.effects`) and calls `P.verdicts`, which calls `tau_verdict` and
`outcome_reading` unchanged. The competence gate is applied afterwards and keeps the pre-gate verdict. No joint verdict
is produced (`overall` is dropped).

**The change to `pipeline.py` cannot alter the published verdicts.** It adds a `models` argument with default
`CO_PRIMARY` to `verdicts` and `sensitivity_verdicts`; with the default, the dictionaries passed to `holm` and the loop
order are identical to before. `run_tau_analysis` does not pass the argument. The τ and families tests pass (below).
`wf/analysis/stats.py` and the rest of `wf/tau` are unchanged since c737cd6.

**The reference results check can and should reproduce `notes/tau_metrics.json` exactly.** The recomputation uses the
same `P.paired` with the same seed on the same 856 items in the same relative order (the build's order, filtered by
`is_primary`, in both analyses), so the bootstrap visits clusters in the same order and the estimate, interval and
$n$ must agree to floating-point identity; the 1e-9 tolerance is appropriate, and a mismatch sets the run to failed.
The dry run confirmed the mechanics on mock data; `notes/tau_dx.md` independently confirmed that the published numbers
follow from the raw records. Only the primary first-token contrast is compared; that is enough to detect a wrong
reference run or item set.

**Competence rule.** Population: primary items whose witness the text-only checker determines (`witness_checker ==
"PASS"`), 447 items, matching the τ discrimination population. Under neutral, witnesses are successes and failing calls
failures; AUC over all success–failure pairs with ties as one half; thresholds are `wf.families.COMPETENCE` (0.70,
0.95, 0.05, 0.5), the families rule exactly; the two-way mass median is over all $2|D|$ neutral records. Gate on the
first token, reported on yes/no. Correct. Note that 405 of the 447 items are `unavailable`, so the gate is in practice a
one-class rule; a model whose effect lives in the status class (as Llama-3.1-8B's does) is gated on the other class.
That is a property of the items, worth one sentence in §7.3.

**Paired comparison bootstrap.** Users are resampled once per draw and both models' rows are read on the same resample;
item-share and $\Delta L$ differences come from the same draws; 95% interval for "differs", 90% for "matches"; the noise
floor applies to the $\Delta L$ reading. Correct; the author's test now kills the unpaired mutant.

**Placement check.** 300 items by a seeded sample in id order, the same in the scoring, token check and analysis; the
first-turn scores live under `mistral-7b-v0.3-firstturn`; the rule (interval of first turn minus native excludes 0 and
its size exceeds one fifth of the native effect on the same items) is implemented as written. S5 covers its test.

**Sharding and the 3-hour limit.** Shards of 3,080, 2,940, 3,038 and 2,926 passes; at the estimated 1.5 s a pass the
largest Mistral shard needs about 1.3 h, at 0.65 s the largest Llama 3.2 shard about 0.55 h. The margin is 2.3 times for
Mistral, the estimate already includes the extra tokenization per pass (it is inside the timed span, so the smoke run
measures it), A40s are excluded, shards resume, and the smoke projection decides before full scoring. Adequate.

**Gemma 2 exclusion.** Justified and correctly reported. The fixed part of every prompt (9,575 Gemma tokens) exceeds
the 8,192-token trained context, the token check on ICE confirmed all 11,984 prompts at or above it, the alternatives
(truncation, a compact prompt, ignoring the limit) are rightly rejected, and the analysis carries the exclusion text
("excluded before data ... reported as not run, never as a null") into `metrics.json`. The results note should use the
short form "not run (context window)" in any model table.

**The other cuts are safe.** The full-reply readout (secondary, never a verdict input, and per the diagnostics mostly
copying of the order number and product name), the rules-only and ALFWorld-matched arms (exploratory; the loss of the
authorization test via the matched arm is stated in §9), the main pool beyond the primary items (its extra stratum was
never read separately), the natural set (28 items, too small), yes/no on the mention pair (descriptive), and the
contamination check (both models predate τ-bench's release) do not touch any confirmatory input. Every input of the
per-model verdict and reading table is kept on the first token. The one cut with a real cost is the witness's yes/no
under aligned and swapped (S4).

**Item share and ties.** First-token scores sit on a grid of about 0.125 nats (bf16 logits), so a few item differences
are exactly zero (0.4–0.5% for the reference models). Ties count as not moved, so the share's null is very slightly
below one half; negligible at these rates, but the tie share should be reported.

**Direction sentence and discrimination.** A positive verdict whose discrimination gate fails stays "replicates" or
"small effect" (with the "not shown to use the record" reading) and counts as interpretable for the direction sentence
"appears in every interpretable model". That follows the τ rule; the sentence should name such a model's caveat.

## 4. Tests and mutants

**Tests.** With the repository's virtual environment on macOS: `tests/test_tau_families.py` 57 passed, 3 skipped (the
rendered-prompt tests need the pinned tokenizers); all τ and families test files together
(`tests/test_tau*.py tests/test_families*.py`) 332 passed, 7 skipped. I did not rerun the full suite.

**The tests fail when the main logic breaks** in these places: the job list, the token-check conditions, the
competence population and the condition it reads, the Holm family in `family_verdicts`, the competence gate, the
pairing of the comparison bootstrap, the reference results check, and the direction of the placement rule. They do
not pin thresholds, interval widths or the discrimination condition of the reading row (S5).

**My own mutants** (strict rule: killed only if pytest exits 1 with a named failing test; each run on a clean copy of
the tree outside the worktree, `tests/test_tau_families.py` only):

| Mutant | Change | Result |
|---|---|---|
| R1 | placement rule at one quarter of the native effect | survived |
| R2 | placement rule at one sixth | survived |
| R3 | reading row assigned without discrimination | survived |
| R4 | competence two-way mass: mean instead of median | survived |
| R5 | Yes share on failures: strict bound at 0.95 | survived (boundary) |
| R6 | AUC: strict bound at 0.70 | survived (boundary) |
| R7 | paired comparison bootstrap draws half the users | survived |
| R8 | token check passes without all replies counted | survived, equivalent (a missed reply already records a problem) |
| R9 | sensitivity verdicts over the reference family | killed by `test_analysis_end_to_end` |
| R10 | comparison counts ties as moved | killed by `test_compare_is_a_minus_b_on_shared_items_with_paired_draws` |
| R11 | item-share interval not clustered by user | survived |
| R12 | Llama-line reading accepts "differs, within the margin" | survived |
| R13 | in-job readout report ignores the witness's reply start | survived |
| R14 | a missing model enters Holm with p = 0 | killed by `test_holm_runs_over_the_two_scored_models_at_a_fixed_size` |
| R15 | competence reads the swapped witness | killed by `test_competence_reads_the_neutral_witness_against_the_neutral_failing_call_on_pass_items` |
| R16 | reference check tolerance 1e-2 | killed by `test_analysis_flags_a_reference_result_that_differs_from_the_published_one` |

Five killed, one equivalent, ten survived. S5 lists what each survivor needs.

## 5. What must happen before the freeze

1. M1: amend the design (dated Changes section) and the extension's reading layer; tests for the new rules.
2. S1–S6 as the author decides, each either done or answered in the Changes section.
3. A second review round on the amendment only (the engineering otherwise passes).
