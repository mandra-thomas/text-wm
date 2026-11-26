# Controlled Bayesian environment: independent review of design and code (review 1)

Scope: `notes/bayesenv_design.md`, `wf/bayesenv/` (world, render, design, score, analysis, simulate, pipeline), `tests/test_bayesenv.py`, `wf/config/bayesenv_*.yaml`, `pace/bayesenv.sbatch` and the bayesenv entries in `wf/runs.py`, at commit 1d79feb, before any freeze tag and before any model score. Context read: `notes/numerical_sensitivity.md` sections 4.8 and 7, `scripts/qwen_audit/followup_score.py` (`plain_text`), `wf/scoring/logodds.py`, `wf/runs.py` (`_init_run`, `_check_resume`), `wf/util.py`, `pace/_common.sh`. The 42 tests pass (207 s). Checks written for this review were run against the code with planted models; their commands and outputs are quoted where they support a finding.

## Verdict

The world model, the generators and the exact posterior are correct for the story the code encodes: the enumeration agrees with a forward simulation, every residual in the confirmatory family has an exact Bayes value of 0, matched prompt types differ only in their defining line, the sequence counts and cost figures in the note are reproduced exactly, and the wording random-effect variance and its Satterthwaite degrees of freedom are derived correctly and keep nominal coverage in simulation. The design is not ready to freeze. One readout is invalid as rendered (half of the state-inference narrator prompts never mention a key, so the question-free state readout does not ask about the object whose posterior is computed), the toy prompt leaves open what happens to an object that is not hidden, and several decision rules are implemented differently from the note or not at all: the specificity rule only recognises excess in the positive direction (a world-model-only goal *penalty* is labelled "no excess shown"), the sizing band's "no negative conclusion" is never passed to the analysis (equivalence can be declared with an MDE above 1.5 times the target), the nats and standardized equivalence conditions are never combined, sign filters not in the note drop negative residuals from the pattern and constant rules, a single two-prompt residual (history-only provenance) counts as a "pattern", the reply-form gate passes when the sampled continuations are missing or empty and never checks the second candidate, the description check decides nothing, and the cross-GPU subset is accepted with a different design and analysed with silently dropped cells. Each is cheap to fix before the freeze. None of the findings is about the numerical environment rules, which the scorer and pipeline implement as written.

## MUST-FIX

These would produce a wrong or misleading number or decision, or leave a stated requirement unmet.

### M1. The narrator readout asks about no particular object in half of its prompts

**What is wrong.** The state-inference narrator prompt ends "Complete the narrator's sentence about box 3." followed by "Narrator: Right now, box 3", read as " holds" against " does". The exact posterior is about the key (in box 3 and box 3 unlocked). In the state-inference framing nothing else names the key unless the cue, note or history does: the generator sentence names no object by design, the take rule says "<object>", and there is no action line. In every explore, coin, other-tag and described-tag prompt under the histories H0, H2, H3, H6 and H7, the word "key" never occurs, so " holds" measures whether box 3 holds anything (the coin is hidden under the same rule), not the queried outcome.

**Evidence.** A scan of the rendered state-inference narrator prompts (`render(t, 'A', h, 0, 'si', 'narr')`, all text and the prefix, word "key" counted) finds no occurrence in 20 of the 40 (type, history) pairs checked: explore, coin, other tag and described tag, each under H0, H2, H3, H6 and H7. Example (explore, H2): the prompt ends "Your task is to: explore the house. House tag: none." then four "It looks empty." lines, then "Complete the narrator's sentence about box 3." / "Narrator: Right now, box 3".

**Consequences.** (a) The narrator readout's evidence slope, competence gate and the slope that normalizes it in the specificity rule are computed from explore prompts (`analysis.py:368-371`), the very prompts that do not mention the key in five of seven histories. (b) Every narrator contrast against explore, other tag, coin or described tag compares a prompt that names the key with one that does not, so it changes what " holds" refers to as well as the cue: zero-reliability note minus explore, tag naming (key tag minus other tag), goal minus tag in its four-prompt form, name minus description. (c) Arm (ii) of the specificity rule (section 10) uses this readout. The world-model framing is not affected because the action line "take the key from box 3" names the key in every prompt, and the state-inference yes/no question names the key.

**Fix.** Make every state-inference prompt name the queried object in a part that is identical across prompt types, without putting the object's word in the generator sentence. Two options: (1) an instruction and prefix that name the key and keep the two continuations distinct and question-free, for example the instruction "Complete the narrator's sentence about the key and box 3." with the prefix "Narrator: Right now, taking the key from box 3 would", read as " succeed" against " fail" (both single tokens to be confirmed for each tokenizer); (2) a fixed rules sentence listing the house's possible objects (see M2), which names the key in every prompt. Re-run the reply-form gate's classifier design for whichever form is chosen. Add a test that every rendered prompt of every type, framing and readout contains the queried object's word in a line shared by all prompt types.

### M2. The toy prompt does not say that an object that is not hidden is not in the house

**What is wrong.** The exact posterior sets $P(\ell = \text{none}) = 1 - \pi$, i.e. an object that is not hidden is absent from the house. The toy's generator sentence says only "every object that the task or the house tag refers to was hidden with probability 90%, and every other object with probability 50%. A hidden object was put in one of the six boxes" (`render.py:119-124`). It does not say what happens to an object that is not hidden (absent, or present but not in a box), nor which objects the house can contain. The note's text ("There is a key in this house."), the game master ("there is a key somewhere in this house") and the state questions all depend on that reading. The ALFWorld-surface generator does state it ("otherwise the room has no keychain", `render.py:345-350`); the toy does not.

**Fix.** Add one sentence to the rules, identical in every toy prompt, e.g. "An object that is not hidden is not in the house at all." If it is acceptable for the key to be named in a sentence shared by every prompt, a sentence listing the possible objects ("The house may contain a key and a coin.") also resolves M1; the generator sentence itself can stay object-free, which is what requirement 4 asks.

### M3. The specificity rule is sign-dependent and mislabels a missing arm

**What is wrong.** Section 10 counts a failure as specific to predicting an action's outcome only if the 95% interval of

$$\Delta_{\text{spec}} = \frac{\theta^{\text{WM}}}{a^{\text{WM}}} - \frac{\theta^{\text{SI}}}{a^{\text{SI}}}$$

lies above 0, and `analysis.py:600-602` implements exactly that. If the failure is negative (the task line lowers the predicted success, as goal effects in this project's natural environments have often done), a larger failure in the world-model framing gives $\Delta_{\text{spec}} < 0$ and the rule reports "no excess shown". The fallback (`analysis.py:607-611`, "reply readout's standardized effect at least as large as the yes/no one") has the same sign dependence. Separately, when a yes/no readout fails its gate but both question-free readouts pass, `yn` is `None`, `yn_pos` is `False`, and the result is labelled "no excess shown" (`analysis.py:602-603`) instead of "not assessable".

**Evidence.** Planted model, world-model-only goal role, `Design(n=20)`, seed 3 (a planted-model script written for this review): with $+0.8$ the rule returns "specific by the rule" (yes/no $\Delta_{\text{spec}} = 1.17$, CI $[0.99, 1.35]$); with $-0.8$ it returns "no excess shown" (yes/no $\Delta_{\text{spec}} = -1.02$, CI $[-1.19, -0.84]$), although the planted failure is equally world-model-specific.

**Fix.** Define the excess on the magnitude in the direction of the world-model effect: with $s = \operatorname{sign}(\hat\theta^{\text{WM}}/\hat a^{\text{WM}})$ on the primary readout, require the 95% interval of $s\,\Delta_{\text{spec}}$ to lie above 0 on both arms, and require the two framings' normalized effects to have the same sign or report a sign reversal separately. Apply the same sign alignment in the fallback. Return "not assessable" whenever either arm cannot be computed. Add the negative case to `test_specificity_rule_separates_world_model_only_from_general`.

### M4. The sizing band's "no negative conclusion" never reaches the analysis

**What is wrong.** Section 9 says that if the base design's predicted MDE of goal minus tag is above $1.5\,T$ (or above $T$ with no affordable redesign), goal minus tag is not used for a negative conclusion, and a pattern contrast whose predicted MDE stays above 0.5 nats at 90 items is not used for one either. The smoke readout records the band (`analysis.py:911-924`) and an `underpowered` list (`analysis.py:948-950`), but `analyse`/`readings` (`analysis.py:515-549`, `723-762`) never read them and the analysis config (`wf/config/bayesenv_analysis.yaml`) has no field for them. The equivalence test in `readings` can succeed whenever the realised SE is a little smaller than predicted or the estimate is near 0.

**Evidence.** With $T = 0.35$, an estimate of 0.01 and an SE that gives $\mathrm{MDE} = 1.56\,T$ (df 200), `readings(..., low_floor_family=True, target=0.35)` returns `equivalent_within_margin: True` and `equivalent_standardized: True`. With 90% half-width $1.65\,\mathrm{SE}$ and $\mathrm{SE} = \mathrm{MDE}/2.80$, the interval fits inside $\pm T$ for any MDE up to about $1.7\,T$ when the estimate is 0.

**Fix.** Write the band and the underpowered list into a committed file (or into the score config) that the analysis must read; have `readings` refuse equivalence for goal minus tag in the two "not used for negative conclusions" bands and for any listed pattern contrast, and emit "not assessable at this noise" for goal minus tag in band 3 unless detected. Add a test.

### M5. The equivalence decision is never formed as the note defines it

**What is wrong.** Section 8: equivalence for Llama-3.1-8B requires the 90% interval inside $\pm M$ **and** the standardized 90% interval inside $\pm 0.20$. `readings` returns two independent booleans (`analysis.py:534-537`) and no combined decision; `no_pattern_shown` (`analysis.py:547-548`) and the cross-GPU comparison (`analysis.py:805-806`) use the nats condition only; the pattern contrasts have no standardized condition at all. A reader of `metrics.json` can take either flag as "the" decision.

**Evidence.** The case in M4 at $\mathrm{MDE} = 1.71\,T$ gives `equivalent_within_margin: False`, `equivalent_standardized: True`.

**Fix.** Add one field per estimand, `equivalent = nats_condition and standardized_condition`, use it everywhere a negative decision is read (readings table, `no_pattern_shown`, cross-GPU), and state in the note whether the standardized condition also applies to the redundancy pattern and the announced slope.

### M6. Sign filters in the pattern and constant rules are not in the note

**What is wrong.** `readings` counts a pattern contrast as detected only if Holm rejects it **and** its estimate is positive (except history-only provenance), and a content-blind constant only if it is rejected and positive (`analysis.py:521-522`). The note (sections 7 and 11) states no direction for either rule; its family is two-sided. A Holm-significant negative redundancy pattern, announced slope, zero-reliability note, tag-naming effect or announced effect is silently dropped from the readings.

**Evidence.** With the zero-reliability note set to $-0.6$ (SE 0.05) and the redundancy pattern to $-0.9$ (SE 0.1), Holm rejects both, `content_blind_constants_detected` is `[]`, `pattern_contrasts_detected` is `[]`, `residual_pattern_detected` is `False`.

**Fix.** Either pre-specify the directions in the note with a reason (a failed redundancy screen predicts a positive redundancy pattern; an ignored announcement predicts an announced slope with the sign of the observational goal effect, which need not be positive) and report rejected contrasts of the other sign as "departure in the unpredicted direction", or drop the filters. The announced slope's predicted sign should follow the observed goal slope, not be fixed at positive.

### M7. History-only provenance is a single residual counted as a "pattern"

**What is wrong.** Requirement: a single non-zero residual must not count as a pattern, because a mention constant produces one. The note (section 11) and the code (`PATTERN_CONTRASTS`, `analysis.py:488`; `readings`, line 521) count "helper minus tester" as a pattern contrast and as sufficient on its own for "a structured departure ... that a content-blind constant cannot produce". It is one contrast between two prompts that differ in a whole provenance sentence (the tester sentence against a helper sentence that mentions a helper that "tries to take ... the object its task names"). A constant attached to that sentence, independent of any content, produces it; so does any lexical effect of the words "helper" or "tries to take".

**Fix.** Replace it in the pattern list with a contrast whose Bayes value is 0 and which a sentence constant cannot produce, e.g. the provenance slope

$$\big[\overline{(\text{helper} - \text{tester})}\big]_{\text{generator C}} - \big[\overline{(\text{helper} - \text{tester})}\big]_{\text{generator A}},$$

which an over-reading of the helper (treating it like the seeing guide) makes non-zero and a constant does not. Keep "helper minus tester" in the family as a residual, and reword section 11.

### M8. The reply-form gate passes on no evidence and ignores the second candidate and the "other" rate

**What is wrong.** (a) `pipeline.py:70-71` and `111-112` load `checks.json` only if it exists and otherwise pass `None`; `gates` then sets the reply-form share to `None` and passes the gate (`analysis.py:640-645`). (b) If no sampled continuation begins with the first candidate, the share is `None` and the gate passes; there is no minimum count. (c) Only continuations beginning with the first candidate are checked. For the narrator readout the second candidate " does" can begin a success ("does hold the key and is not locked." is classed `other` and never counted), so a failure-form check on the second candidate is needed as much as the success-form check on the first. (d) Requirement 6 asks for validation of the "other" rate (continuations that begin with neither candidate); it is computed in `checks.json` but used by no gate and reported by no analysis output. (e) The checks run itself is not validated (no status, commit, dirty-tree or GPU check, `pipeline.py:70-71`, `111-112`).

**Evidence.** End-to-end `analyse` on planted records with `checks=None` (an end-to-end planted-model script written for this review): every gate shows `"success_form_share": null, "passed": true`. Of the 120 sampled prompts, 63 are world-model reply prompts, 49 narrator prompts and 8 ALFWorld-surface reply prompts (seed 20261125); at the exact posterior the expected number of continuations beginning with the first candidate is about 105, 103 and 14 respectively, and a model that rarely predicts success produces far fewer. `classify_reply` also classes " holds a key and is not locked." as `a_other` (only "holds the key" is accepted), which would fail the gate spuriously.

**Fix.** Require the checks run to exist and to pass the same status/commit/clean/GPU checks as scoring shards; fail the gate (report "not assessable") when fewer than a stated number of continuations (e.g. 30) begin with either candidate; add a failure-form share for continuations beginning with the second candidate with the same 80% threshold; report the "other" rate per readout and set a threshold for it; sample per readout (a fixed number of prompts for each question-free readout, stratified over cells so that high-posterior prompts are included) instead of 120 prompts shared at random; accept "holds a key" in the success form.

### M9. The description check decides nothing

**What is wrong.** Section 5: if the described tag is not recognised as a key, the name-against-description estimand is reported as not assessable. `analyse` never reads `checks["description"]` (`analysis.py:723-762`); name minus description is always reported and always stays in the Holm family. The note does not say whether a not-assessable member leaves the family (changing $m$ for Holm) or stays.

**Fix.** Read the description check in `analyse`; when it fails, mark name minus description "not assessable" and remove it from the family before Holm, and say so in the note. Also note that the check is one prompt and one forward pass; a handful of paraphrased checks (and both framings' system texts) would make "recognised" less fragile.

### M10. The cross-GPU subset is accepted with a different design, and missing cells are dropped silently

**What is wrong.** (a) `run_bayesenv_analysis` takes the design from the main run's `config.json` (`pipeline.py:47-49`, `113`) and never compares it with the cross-GPU run's design (the configs say only "must equal the full run's"). (b) `mean_contrast` drops any cell that has no items with all the needed prompt types and re-weights over the cells present (`analysis.py:172-181`), so a missing prompt type changes an estimand's definition without an error. (c) The analysis never checks that each record's `unit_hash` equals the hash of the prompts the current code renders; scoring and analysis run at different commits by design (the smoke readout commits in between), so a later change to `world.py` or `render.py` would attach scores to different exact values silently.

**Evidence.** Main run designed with three goal and four tag wordings ($W_G = 3$, $W_T = 4$), cross-GPU subset scored with the base design: `analyse` runs without error and reports for goal minus tag a cross-GPU shift of 0.355 nats, "decision_changed: True, environment_dependent: True"; the subset's goal minus tag is computed on the 8 panel cells only, because the 20 other cells lack `key3` (an end-to-end planted-model script written for this review).

**Fix.** Refuse the analysis when the two designs differ; make `mean_contrast` and `item_contrast` raise when a requested cell has no complete items (or when item sets differ between prompt types beyond a stated tolerance); recompute `D.unit_hash` for every record at analysis time and refuse mismatches.

### M11. The ALFWorld-surface implicit prompts do not say who chose the actions

**What is wrong.** Requirement 2: every prompt says who chose the earlier actions. The implicit arm (`alf_types`, `render.py:307-308`; `alf_render`, `render.py:342-351`) drops the whole set-up paragraph, including "Every action in this episode, the next one included, was picked at random by a tester". The rendered implicit prompt goes straight from the few-shot examples to "History:".

**Fix.** Either keep the provenance sentence in the implicit prompts (it carries no generator information) or state in sections 2 and 13 that the implicit arm is exempt and why.

## SHOULD-FIX

### S1. Pre-specified estimands and reports that the code does not produce

- The calibration ratios (goal and tag slope over the base-rate slope, note slope over the history slope; section 7) are not computed anywhere (`grep -n calibration wf/bayesenv/` finds nothing). The seeing-guide ratio is not compared with $a$.
- Section 8 says the announcement, provenance and existence statements' two wordings are reported separately; no estimate by `variant` exists.
- Section 7 asks for Brier and log-score regret under both conventions; only the log-score regret has an unnormalized version (`analysis.py:650-669`).
- Section 11's label for Qwen3-8B nulls ("not detected at an MDE of $X$ nats") is not produced; `readings` gives the MDE but no label.

Fix: implement them (ratios with the stratified bootstrap drawing numerator and denominator from the same resample) or remove them from the note.

### S2. Goal minus tag by generator omits the wording term the note gives it

Section 8 lists "goal minus tag ... by generator" among the estimands that add wording variance. `goal_minus_tag_gen_{g}` uses `mc` only (`analysis.py:461-465`), and the likelihood-ratio slope and Cochran's $Q$ across histories are built from these item-only SEs (`analysis.py:466-481`). Wording-by-cell variation then inflates $Q$ and the slope's apparent precision. Fix: apply `add_wording` with $C = 7$ per generator; for the slope and $Q$, either use the full variance or state that they are item-conditional.

### S3. Single-wording residuals are not declared conditional on their wording

Tag naming at zero information compares one key-tag wording with one other-tag wording, and the competing goal compares one coin task with one explore task line; neither carries a wording term, yet section 8 names only the note, announcement and provenance statements as conditional on wording. Fix: state that these two family members are conditional on their wordings, or average tag naming over the $W_T$ key-tag wordings with a matched other-object wording for each.

### S4. The numerical floor uses the wrong item count for pattern contrasts and the cross-GPU margins

`numerical_floor(cls, e.n_items)` uses the total items summed over every cell the estimand touches (`analysis.py:527`, `542`). For the redundancy pattern that is about $28n$, although its variance (and its numerical noise) is dominated by the four existence-settled cells. In the largest logit class ($\tau = 0.08$, $\sigma_c = 1.16$) the floor is

$$\mathrm{SD}_{\text{num}}(120) = \sqrt{0.08^2 + 1.3 \times 1.16^2 / 120} = 0.145, \qquad \mathrm{SD}_{\text{num}}(840) = 0.092,$$

so five times the floor is 0.72 nats on the relevant count but 0.46 on the count used, and the margin stays at 0.5. The cross-GPU rows use the subset's item count for every member except goal minus tag (`analysis.py:798-799`), so their margins differ from the main run's. Fix: use the effective item count of the contrast (the harmonic combination implied by its weights) and the main run's margins in the cross-GPU comparison.

### S5. Pattern targets are fixed in nats on the model's scale

The goal-role target is $T = 0.5\,a$ (half a nat of exact evidence on the model's scale), but the pattern target and margin are 0.5 nats of the model's own log-odds (section 8; `PATTERN_TARGET`). For a model with $a = 0.3$, a 0.5-nat pattern margin corresponds to about 1.7 nats of exact evidence, so "no structured residual at the target size" would cover residuals more than three times the size of the goal-role target. Fix: use $\max(0.5\,a, 5\,\mathrm{SD}_{\text{num}})$ for the pattern contrasts too, or give the reason for a fixed nat scale in the note.

### S6. The redesign band can choose a design the budget then stops

Band 2 picks the cheapest menu design with predicted MDE at most $T$ and GPU time at most twice the base design's; the budget check comes afterwards and stops the full run if exceeded (`analysis.py:914-958`). At the note's 0.08 s per sequence, the projected two-model totals (full, cross-GPU, smoke and 0.3 h start-up per model) are:

| Goal wordings, tag wordings, items per cell | Projected total GPU-hours | Within 6.3 |
|---|---|---|
| 2, 3, 30 (base) | 4.69 | yes |
| 3, 4, 30 | 4.97 | yes |
| 4, 6, 30 | 5.39 | yes |
| 6, 8, 30 | 5.94 | yes |
| 8, 8, 30 | 6.22 | yes |
| 2, 3, 45 | 6.06 | yes |
| 3, 4, 45 | 6.45 | no |
| any design with 60 items per cell | 7.4 to 10.1 | no |

Seven of the twelve designs that pass the "at most twice the base" filter exceed the budget, in which case the rule stops the study rather than falling back to the base design with "no negative conclusion". Fix: add "and whose projected total is within the budget" to the band-2 selection, and fall back to the base design otherwise. The pattern-item increase (existence-settled and announced cells up to 90 items) adds about 0.4 GPU-hours for two models and should be inside the same check.

### S7. The announced slope includes cells in which it has no signal

The announced arm runs on the six graded histories including H4, where existence is settled and the observational goal effect is exactly 0. Under the account the announced slope is meant to detect (the announcement is ignored and the task line is read as informative), the three H4 cells contribute expected 0 to the slope and only add variance. Fix: drop H4 from the announced cells (fifteen cells), or keep it and say that the slope is diluted by design.

### S8. Chat-mode tokenization is not checked as the note claims

Section 6: every candidate's joint tokenization with the prompt must extend the prompt by exactly that token. In chat mode `encode` checks only that the candidate is one token and, with an answer prefix, that prefix plus candidate tokenizes as prefix ids plus candidate id (`score.py:72-86`); the rendered template plus prefix plus candidate is never tokenized jointly, and no chat-mode path has a test. Fix: render the template as text (`tokenize=False`), append prefix and candidate, tokenize once, and require equality with the concatenated ids; add a test with a stub tokenizer for both modes, including a positive case for the plain mode that checks the candidate ids carry the leading space.

### S9. Wording ambiguities in the provenance statements

- Seeing guide, wording 1 (`render.py:89-92`): "When the object its task names is in a box that is not locked, the guide takes it from that box 90% of the time; otherwise it picks one of the twelve take actions ... at random." "Otherwise" can be read as covering only the case where the key is not in an unlocked box, leaving the remaining 10% unspecified; wording 2 ("in every other case") is unambiguous. Under the alternative reading (the 10% case is not a take action) the exact log-odds of the guide prompts shift by 0.02 to 0.19 nats (largest under generator C with H0: 1.35 against 1.53). The seeing-guide ratio is descriptive, so this is not decision-relevant, but the two wordings should state the same policy.
- History-only helper, wording 1: "picks, at random, a box it has not looked into", although the helper looked into nothing (the tester did). Wording 2 ("the boxes not yet looked into") is clear. The exact posterior is unaffected (either reading is independent of the hidden state), but the two wordings describe different policies.
- Both statements refer to "its task" / "the object its task names"; neither says that this is the task shown in the task line.

Fix: align wording 1 with wording 2 in each pair and say "the task shown in the task line".

### S10. "Nothing is read for that model" is not implemented

Section 5: if both world-model readouts fail their gates, nothing is read for that model. `analyse` still computes Holm readings for every state-inference readout that passes and runs the specificity rule (`analysis.py:734-753`); only the `primary_readout` field becomes `None`. Fix: when the primary is `None`, emit readouts descriptively without `readings`, and say in the note what the smoke readout's "stop" band (`analysis.py:858-860`) means for the full run.

### S11. Tests that could pass while the code is wrong

- No test ties the numbers written into the prompts (30%, 2%, 90% and 10% for the glance, 20% for locks, 90% and "twelve" for the guide, the generator percentages, the note's reliability) to the constants the posterior uses; the forward simulation uses the same `Evidence` constants as the enumeration, so a drift between text and code would pass every test. Add a test that parses each rendered rule sentence and compares it with `world`.
- No test checks that every prompt names the queried object (would have caught M1).
- The planted-model tests compare each estimand's shift against the Bayes-planted model with the same seed, so any bias common to all planted models cancels; the Bayes-planted residuals are never checked against 0, and `announced_goal_effect` and `redundancy_residual` are not in `EXPECT` although the note says lexical mention moves them.
- The planted mass is a constant 0.9 and the planted logit size gives the small floor class, so the mass gate, the floor classes, the primary-readout switch, the narrator fallback and the "not assessable" paths are never exercised. Planted deviations are identical across readouts unless `wm_only`, so a mix-up of readout keys inside the analysis would not be detected.
- `test_smoke_readout_bands_follow_wording_variance` line 276 is vacuous: `"goal_minus_tag" not in json.dumps(quiet["variance"])` is always true. A real leakage test would plant a large goal role and assert that no reported quantity changes except the variance components that legitimately include it.
- `test_posterior_matches_forward_simulation` seeds its generator with Python's `hash((hist, prov))`, which changes between processes unless `PYTHONHASHSEED` is fixed, so the Monte Carlo draws differ run to run.
- Nothing tests `cross_gpu`, `check_run` on the cross-GPU pairing, or the refusal of incomplete cells (M10).

### S12. Freeze bookkeeping

Section 16 exempts "section 17 and the two sizing configs" from the deviations log, but the procedure also edits `bayesenv_xgpu_<model>.yaml` (design must match) and `bayesenv_smoke_readout_llama3.1-8b.yaml` (`fixed_gpu_h`, set to what Qwen3-8B used). List every file that the procedure edits after the tag. The analysis is required to run "from a clean checkout at the tag", but the code checks only that the checkout is clean, not that the tag is an ancestor of `HEAD`.

## NOTES

- **N1. A zero-Bayes contrast that is pooled away.** Explore prompts under generators A, R3 and C state different named-object rates (50%, 75%, 90%) but have the same exact posterior. The evidence and history slopes treat them as replicates and the base-rate slope pools them into one comparison group whose bootstrap resamples across the three cells together (`analysis.py:376-387`). A model that applies the named rate to the key even when nothing names it would bias the base-rate slope (and the calibration ratios built on it). Reporting explore under C minus explore under A, Bayes value 0, would detect this directly.
- **N2.** H4 enters the base-rate slope with an exact difference of 0 (existence settled makes the explore posterior identical across generators: $-1.39$ in every generator), so it adds variance and dilutes the denominator average without adding signal.
- **N3.** The delta method in the specificity rule ignores the covariance between the two framings' slopes (estimated on the same items) and between each slope and its estimand; for same-sign effects this is conservative. The interval keeps the Satterthwaite degrees of freedom of the item and wording terms after adding the slope variance. The six specificity intervals (three estimands, two arms) are not multiplicity-controlled; the note should say so. `normalized_difference` pairs items by array length, not by item index (`analysis.py:573`); with complete data this is harmless.
- **N4.** The take rule says the take "succeeds only if the object is in that box and the box is not locked" (`render.py:74-76`); "if and only if" is what the posterior assumes, and the following clause ("the environment then replies ...") only implies it.
- **N5.** The low-floor decision (equivalence allowed) comes from the config flag `low_floor_family`, not from the measured floor class. Qwen3-8B in plain text may fall below the 32-logit threshold and be assigned the small class constants, which were measured on Llama-3.1-8B in its chat template. Report the measured class beside the flag and stop if Llama-3.1-8B's measured class is not "small".
- **N6.** The cost projection prices the cross-GPU subset at the L40S seconds per sequence; A40 throughput is lower, so the projection is optimistic by a fraction of the 0.45 GPU-hours. The checks job is not in the projection (covered in practice by the 0.3-hour start-up allowance).
- **N7.** The ALFWorld-surface implicit arm reads " a" against " nothing", which with no stated generator measures whether the drawer holds anything, not a keychain; its "swap" and "goal against neutral" effects should be described that way.
- **N8.** In the public note, section 3 cites agreement "with the planning code's posterior to $4\times10^{-16}$", and section 13 refers to "the original toy"; neither is defined or present in the repository, so a reader cannot check them. Section 7 writes the four-prompt form as plain text, "(goal − explore) − (key tag − other tag)", where a display equation fits better, and the table in section 4 carries LaTeX symbols ($\pi_{\text{o}}$, $\pi_{\text{n}}$), which some renderers do not display inside tables; plain words ("the generator's rate for unnamed objects") would be safer there. The note is otherwise impersonal and factual and contains no plans or venues.
- **N9.** `sample_check` contains a no-op (`score.py:278-279`, `text = text`). `combine` and `goal_minus_matched_note` call `np.nanmean` on two NaNs on the cross-GPU subset and emit runtime warnings.
- **N10.** Resuming a Qwen3-8B shard after the Llama-3.1-8B smoke readout has been committed is refused by `_check_resume` (commit changed); this is the intended guard but it means a Qwen shard that times out late must be rerun from scratch, or the order of commits planned around it.
- **N11.** What was checked and found correct: the enumeration against a fresh forward simulation, the prior under the announced replacement ($(\pi_{\text{n}} + 2\pi_{\text{o}})/3$), the equality of the history-only helper's and the tester's posteriors, the matched-information note ($\Lambda(0.8) = 9$ equals generator C's ratio exactly; $\Lambda(0.95) = 39$ against generator D's 39.01), the wording mean squares and their expectations, $\mathbb{E}[M_W] = \sigma_e^2 + n\sigma_\gamma^2 + nC_p\sigma_\alpha^2$ and $\mathbb{E}[M_{WC}] = \sigma_e^2 + n\sigma_\gamma^2$, the Satterthwaite formula, interval coverage with wording-by-cell effects only (0.95 over 40 planted replicates) and with both wording effects (0.975), the KL regret formula, Holm's step-down, the redundancy pattern's item-by-item combination, the sequence counts (9,300 smoke; 63,720 full; 19,116 cross-GPU over 324 items), the 1.4 and 0.45 GPU-hour figures, the environment record and resume guards in `run_bayesenv_score`, and the plain-text rendering (system text, blank line, user turn, answer prefix after one space).

## Response (before the freeze)

Every finding was addressed in the design and code before the freeze tag; nothing had been scored. Where a fix differs from the recommendation, the reason is given.

- **M1.** The narrator instruction now names the key: "Complete the narrator's sentence about the key and box 3." It is the same line in every state-inference narrator prompt of an item. The prefix and the candidates (" holds" against " does") are unchanged. A rephrasing about the take succeeding was not used, because it would turn the state question into a question about an action's outcome, which is what the specificity rule compares against. A test now checks that every prompt of every type, framing and readout names the key in a line shared by all prompt types of the item.
- **M2.** The generator sentence now ends "an object that was not hidden is not in the house at all", and the take rule says "if and only if" (N4). The generator still names no object.
- **M3.** The excess is now read in the direction of the world-model effect, with $s$ the sign of the world model's normalized estimate on each arm. Opposite world-model signs on the two arms, or a missing arm, give "not assessable". A test covers both signs and the missing arm.
- **M4.** The smoke readout writes a `sizing` outcome: whether goal minus tag may support a negative conclusion, and which pattern contrasts may not. It is committed into `wf/config/bayesenv_analysis.yaml`. The analysis refuses to run without it, and `readings` applies it.
- **M5.** Each equivalence decision is now one field, `equivalent`. It requires the low-floor family, a measured logit-size class of "small" (N5), the sizing permission, and both the nats and the standardized conditions. The same applies to the redundancy pattern and the announced slope. "No residual pattern" uses it.
- **M6.** The sign filters were removed. The pattern rule and the constant list are two-sided, and the sign is reported.
- **M7.** History-only provenance is now a single residual. The pattern contrast added in its place is the provenance slope (helper minus tester under generator C, minus the same under generator A).
- **M8.**
  - The checks run is now required. It must be a complete real run from a clean checkout on L40S.
  - Sampling is stratified per question-free readout: 40 prompts each, half with an exact success probability of at least 0.25, 10 continuations each.
  - The gate now requires at least 30 continuations on each candidate word, at least 80% success form after the first candidate, at least 80% failure form after the second, and at most 50% beginning with neither.
  - "holds a key" is accepted as a success form.
- **M9.** The description check is now three phrasings, and all must favour Yes. If any does not, name minus description leaves the Holm family before adjustment.
- **M10.**
  - The analysis refuses a cross-GPU run with a different design, or one that scores different items or prompts.
  - Every requested cell must have complete items, otherwise the analysis stops; nothing is reweighted.
  - Stored prompt hashes are recomputed from the current code, and mismatches stop the readout and the analysis.
- **M11.** The ALFWorld-surface implicit prompts keep the sentence that every action was picked at random by a tester.
- **S1.**
  - The calibration ratios are now computed by the delta method, treating the two slopes as independent; this is stated in the note.
  - The seeing-guide ratio is now divided by the evidence slope.
  - The two statement wordings are reported separately.
  - The unnormalized Brier regret is now reported.
  - The Qwen null label is now emitted.
- **S2.** Goal minus tag by generator and by history now carries its wording term.
- **S3.** The note states that tag naming and the competing-goal residual are conditional on their wordings.
- **S4.** The numerical floor now uses each estimand's effective item count, $1/\sum_c w_c^2/n_c$. Cross-GPU margins are the main run's.
- **S5.** The pattern target is now $0.5\,a/0.7$: 0.5 nats at the simulated evidence slope, scaled to the model's own.
- **S6.**
  - The redesign must fit the budget, or the rule falls back to the base design.
  - The extra pattern items yield to the budget step by step.
  - The cross-GPU subset is priced at 1.3 times the L40S time (N6).
- **S7.** H4 was dropped from the announced cells, leaving 15.
- **S8.** Chat mode now renders the template as text and requires its tokenization to equal the template's ids. Each candidate must extend the rendered text by exactly one token. A stub-tokenizer test covers both modes. A tokenizer-only check of every prompt form for both models (`scripts/bayesenv_tokcheck.py`) is run on the cluster before the freeze.
- **S9.** Both wordings of each provenance statement now describe the same policy and say "the task in the task line".
- **S10.** If no world-model readout passes, no readout gets readings and the specificity rule is not applied.
- **S11.** The tests now check that the numbers in the rules match the posterior constants, that every prompt names the key, and that the Bayes-planted residuals are zero within error. The expected effects of the lexical and announcement models now include the announced effect and the redundancy residual. A deviation planted on one readout stays on it. The leakage test is real, the forward-simulation seed is deterministic, and the cross-GPU refusal and incomplete cells are tested.
- **S12.** The note lists every file the procedure edits after the tag. The analysis checks that the tag is an ancestor of the checkout.
- **N1, N2.** A descriptive "named-rate leakage" estimand was added: explore under generator C minus explore under generator A. The base-rate slope excludes H4.
- **N3.** The note states that covariances are ignored in the delta method and that the six specificity intervals are not adjusted for multiplicity.
- **N7.** The note describes the implicit arm's reply readout as measuring whether the drawer holds anything.
- **N8.** The references to code outside the repository were removed. The table's symbols were replaced by words, and the four-prompt form is now a display equation.
- **N9.** The no-op was removed.
- **N10.** The note states that each model's scoring finishes before the next commit the procedure makes.
