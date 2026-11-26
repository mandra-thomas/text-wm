# Controlled Bayesian environment: results (confirmatory family, frozen analysis run once)

**Status.** These are the results of the study designed in `notes/bayesenv_design.md`. The design, code and analysis were frozen at tag `bayesenv-design-frozen-2026-11-25` (commit 02c407a) before any model score existed. One deviation, D-BE1, was logged and tagged before any score was read (`notes/deviations.md`, tag `bayesenv-deviation-be1-2026-11-26`). It fixed a provenance check that had stopped the first smoke readout. Each model's smoke readout was recorded in section 17 of the design note before that model's full run was submitted. The frozen analysis ran once, from a clean checkout containing the tag (CPU job 7354564, commit 0cb04d4). Every number below is taken from that run's output, which is committed as `notes/bayesenv_metrics.json`. The independent review of the design is in `notes/bayesenv_review1.md`.

## 1. What was done and what it asks

**The world.** The study builds a small world in which every prompt states exactly how the world was generated.

- A house has six boxes. The queried action is always "take the key from box 3".
- Each box is locked with probability 20%.
- A stated rule hides any object that the task line or a "house tag" refers to with one probability, and every other object with another.
- Four versions of the rule (the *generators*) give the cue a likelihood ratio on the key's existence of 1, 3, 9 and 39. The cue carries no information under the first, and the last also lowers the base rate.
- Seven *histories* range from nothing seen to failure settled: looks into other boxes that can miss, looks that cannot miss, existence of a key known for certain, a lock report that is wrong 2% of the time, box 3 seen empty, and a glance suggesting a key.
- Every prompt therefore has one exact probability that the take succeeds, the *exact posterior*.

**What the model is asked.** Two framings share the same story.

- *World model:* predict the environment's reply to the take.
- *State inference:* say whether box 3 now holds the key and is unlocked.

Each framing has two readouts:

- a question-free one: the first token of the reply, "You" against "Nothing"; or a narrator sentence, "holds" against "does";
- a yes/no question.

**The cues compared.** All non-goal cues sit on the neutral task "explore the house".

- *Goal line:* a task naming the key (2 wordings).
- *House tag:* a tag naming the key (3 wordings). It carries the same information as the goal line, by the same route, in the same position.
- A tag naming the coin.
- A tag describing the key without naming it.
- The coin task, alone and with the key tag.
- Caretaker's notes "There is a key in this house." of stated reliability 0%, 50%, 80% and 95%.
- An *announced* arm: the prompt says the task line was redrawn at random after the house was set up, so the task carries no information.
- Two *provenance* arms: the action was chosen by a helper who saw only the history, or by a guide who can see inside the boxes.
- The same structure written in ALFWorld's own text (six drawers, a keychain).

**The four questions** for Qwen3-8B (in plain text, without the chat template's empty think block) and Llama-3.1-8B (in its chat template):

1. Does the model use stated evidence?
2. Does a goal line move the prediction differently from a matched house tag ("goal minus tag")?
3. Is there a structured residual where the exact posterior does not move?
4. Is any failure larger when predicting the action's outcome than when inferring the state from the same story?

## 2. What ran

**Stack and hardware.** All main scores were made in one numerical environment per model: torch 2.14.1 with transformers 5.23.0 ("serve2"), NVIDIA L40S, bf16, one unpadded sequence at a time, pinned model revisions. All shards are complete, came from clean checkouts, and share one commit and one environment.

| Model | Smoke (L40S) | Checks | Full run, 4 shards (L40S) | Cross-GPU subset (A40) |
|---|---|---|---|---|
| Qwen3-8B | 7330239 (9 min) | 7330244 | 7347327–7347330 (13–17 min each) | 7347331 (34 min) |
| Llama-3.1-8B | 7347335 (9 min) | 7347336 | 7350438–7350441 (13–17 min each) | 7350442 (32 min) |

**Size and cost.** Per model, 63,360 sequences were scored in the full run and 19,008 in the cross-GPU subset. The study used 3.45 GPU-hours.

**Sizing outcome.** The smoke readout's frozen sizing rule placed goal minus tag in the band "above 1.5 times the target" for both models:

- Qwen3-8B: predicted minimum detectable effect (MDE) 0.60 nats against a target of 0.10.
- Llama-3.1-8B: predicted MDE 0.35 nats against a target of 0.086.

So both ran the base design (2 goal wordings, 3 tag wordings, 30 items per cell), and goal minus tag may not support a negative conclusion for either model.

## 3. Which readouts are valid

A readout decides anything only if it passes three gates:

- the evidence slope (the slope of the model's log-odds on the exact log-odds for the neutral prompt over the 24 graded cells) has a 95% lower bound above 0.1;
- the median probability on the two candidate words is at least 0.5;
- for the question-free readouts, sampled continuations confirm that each candidate word starts its intended reply.

| Readout | Qwen3-8B: slope (95% lower bound); mass | Gate | Llama-3.1-8B: slope (95% lower bound); mass | Gate |
|---|---|---|---|---|
| World model, reply first token | −0.16 (−0.19); 0.43 | fails: log-odds fall as the exact probability rises | 0.41 (0.39); 0.93 | fails: only 40% of sampled "You" replies are "You take the key from box 3" |
| World model, yes/no | 0.21 (0.19); 0.997 | **passes; primary** | 0.16 (0.16); 0.74 | **passes; primary** |
| State inference, narrator | 0.52 (0.51); 0.012 | fails: 98% of continuations start with neither word | 0.37 (0.34); 0.003 | fails: same |
| State inference, yes/no | 0.62 (0.61); 0.55 | passes (secondary) | 0.10 (0.085); 0.75 | fails: slope |
| ALFWorld surface, reply | −0.18 (−0.31); 0.997 | fails: slope | 0.45 (0.27); 0.80 | passes (secondary) |
| ALFWorld surface, yes/no | −0.38 (−0.45); 0.99 | fails: slope | 0.08 (−0.01); 0.89 | fails: slope |

**Consequences under the frozen rules.**

- **Primary readout.** For both models the world model's reply readout failed its gates, so the primary readout is the world model's yes/no question. The detection targets on it are $T = 0.5\,a$ and $T_p = 0.5\,a/0.7$, where $a$ is the evidence slope; $T$ is about 0.10 nats for Qwen3-8B and 0.08 for Llama-3.1-8B, and $T_p$ about 0.15 and 0.12.
- **Description check.** Llama-3.1-8B rejected one of the three phrasings of the description check (−0.125 nats), so name minus description left its family.
- **Specificity rule.** For Llama-3.1-8B neither arm can be computed. For Qwen3-8B the question-free arm cannot be computed (both its readouts fail) and the fallback needs the reply readout. So **the specificity rule is "not assessable" for both models**: the study cannot say whether any failure is specific to predicting an action's outcome.
- **Logit sizes.** Every readout of both models has median candidate logits below 32, so all are in the small numerical-floor class.

These failures are themselves findings about the readouts.

- **Qwen3-8B's reply readout runs backwards.** In plain text, its first-token probability of "You" over "Nothing" moves against the exact probability of success.
- **Llama-3.1-8B's reply readout is ambiguous.** It tracks the exact probability, but most of its "You" continuations are not the success reply.
- **The narrator prompt is not taken up.** Neither model's narrator readout is usable, because both models almost never continue it with either candidate word.

## 4. Evidence use (primary readout: world model, yes/no)

| Slope (exact-reasoner value: equal to the evidence slope, or 1 for a ratio) | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Evidence slope (neutral prompt, all graded cells) | 0.207 [0.192, 0.222] | 0.162 [0.155, 0.170] |
| History slope (generators with base rate 0.5) | 0.272 [0.256, 0.287] | 0.198 [0.189, 0.206] |
| Base-rate slope (base rate 0.2 against 0.5) | −0.237 [−0.298, −0.176] | −0.150 [−0.172, −0.128] |
| Goal slope (goal minus neutral on the exact goal effect) | 0.045 [0.006, 0.085] | −0.006 [−0.019, 0.006] |
| Tag slope (key tag minus coin tag on its exact effect) | −0.043 [−0.084, −0.002] | −0.018 [−0.029, −0.007] |
| Note slope (note minus neutral on its exact effect) | 0.102 [0.056, 0.148] | −0.241 [−0.258, −0.225] |
| Note ordering (95% note minus 50% note; exact value 0.60) | 0.066 [0.055, 0.077] | −0.054 [−0.062, −0.045] |
| Seeing-guide effect divided by its exact effect | 0.533 [0.506, 0.560] | −0.056 [−0.074, −0.037] |

**What the table shows.**

- **Histories are used, weakly.** Both models use the history's evidence in the right direction but weakly. The slopes of 0.2–0.3 mean their log-odds move about a quarter as much as the exact log-odds.
- **The stated base rate is used in the wrong direction.** When the rule hides unnamed objects less often, the predicted success of the take rises.
- **The task's and the tag's information are nearly unused.** The goal and tag slopes are near zero or slightly negative.
- **Notes split by model.**
  - Qwen3-8B uses the caretaker's reliability in the right direction but compressed.
  - Llama-3.1-8B moves against it: the more reliable the note, the lower its predicted success.
- **The seeing guide splits by model.**
  - Qwen3-8B takes the seeing guide's choice as strong evidence: about half the exact shift, more than twice its own evidence slope.
  - Llama-3.1-8B does not.

The calibration ratios (cue slope over base-rate or history slope) are reported in the metrics file. Because the base-rate slopes are negative, they cannot be read as calibration.

## 5. The confirmatory family (primary readout)

Every residual below has an exact value of 0, because its two prompts share the exact posterior. Each one is either a *pattern* contrast, which a constant attached to one sentence or one mention cannot produce, or a *single residual*, which such a constant produces on its own. The family is Holm-adjusted at two-sided 0.05 per model. Estimates are in nats of yes/no log-odds, with 95% intervals.

| Residual (kind) | Qwen3-8B | Holm | Llama-3.1-8B | Holm |
|---|---|---|---|---|
| Goal minus tag (goal role at matched information) | −0.39 [−0.81, +0.03] | not rejected | −0.01 [−0.26, +0.24] | not rejected |
| Tag naming at zero information: key tag minus coin tag under the uninformative generator (single) | +0.31 [+0.26, +0.36] | rejected | −0.13 [−0.14, −0.11] | rejected |
| Competing goal: coin task with key tag minus explore with key tag (single) | +0.05 [+0.03, +0.06] | rejected | +0.19 [+0.18, +0.20] | rejected |
| Name minus description (single) | +0.29 [−0.45, +1.03] | not rejected | not assessable (description check failed) | — |
| Zero-reliability note (single) | −0.25 [−0.27, −0.22] | rejected | +0.27 [+0.26, +0.28] | rejected |
| Redundancy residual: notes where existence is settled (single) | +0.71 [+0.62, +0.80] | rejected | +0.27 [+0.24, +0.30] | rejected |
| **Redundancy pattern**: redundancy residual minus zero-reliability note (pattern) | **+0.96 [+0.88, +1.03]** | **rejected** | +0.00 [−0.02, +0.03] | not rejected |
| Announced goal effect: key task minus coin task, task line said to be redrawn at random (single) | +1.41 [+1.37, +1.44] | rejected | +0.42 [+0.41, +0.44] | rejected |
| Announced slope: announced effect under informative generators minus uninformative (pattern) | +0.06 [−0.01, +0.13] | not rejected | +0.02 [−0.01, +0.04] | not rejected |
| History-only provenance: helper minus tester (single) | +1.32 [+1.24, +1.40] | rejected | −0.05 [−0.08, −0.03] | rejected |
| History-only provenance slope: generator C minus generator A (pattern) | +0.03 [−0.12, +0.18] | not rejected | −0.02 [−0.08, +0.03] | not rejected |

### Readings by the frozen rules

**Goal role.** Goal minus tag is not detected in either model.

- Qwen3-8B: the reading is "not detected at an MDE of 0.59 nats". Its point estimate is negative (−0.39) and its standardized 90% interval is [−3.29, −0.23].
- Llama-3.1-8B: −0.01, with an MDE of 0.34 nats. The sizing rule withholds a negative conclusion, so the reading is "not detected; not assessable for a negative conclusion at this noise".

The shared wording effects of the goal and tag lines are large relative to the target, which is what set these MDEs. So the study cannot say whether the goal line differs from a matched non-goal line at the target size of about 0.1 nats. Goal minus tag shows no dependence on the likelihood ratio in either model: the slope on $\log\Lambda$ is −0.02 for Qwen and 0.00 for Llama.

**Residual pattern, Qwen3-8B: detected.** Where existence is already settled, a caretaker's note about existence has no exact effect. Even so, Qwen3-8B's yes/no log-odds rise by 0.71 nats under the 50–95% notes. Its 0% note lowers them by 0.25. The difference, 0.96 nats, is a structured departure from the exact posterior: it is not the same at zero information as where the variable is settled, so a content-blind constant attached to the note cannot produce it. The cross-GPU rescore labels this reading environment-dependent (section 8), because its estimate on the subset moved by 0.033 nats, just over a fifth of its 0.148-nat margin. Its detection did not change, and its size is about thirty times that shift.

**Residual pattern, Llama-3.1-8B: not detected.**

- The redundancy pattern is equivalent to zero within its margin of 0.14 nats, and within 0.20 on the standardized scale.
- The announced slope is inside its nats margin but not inside the standardized one, so "no residual pattern" is not declared.
- The provenance slope's MDE is 0.08 nats.

Llama-3.1-8B's redundancy residual (+0.27) equals its zero-reliability intercept (+0.27). That is the signature of a constant attached to the note, not of failed screening.

**Content-blind constants, both models.** Several single residuals where the exact posterior is unmoved are detected:

- **Tag naming:** +0.31 for Qwen3-8B, −0.13 for Llama-3.1-8B.
- **Zero-reliability note:** −0.25 and +0.27.
- **Competing goal:** +0.05 and +0.19.
- **Announced effect:** a key task raises predicted success over a coin task even when the prompt says the task line was redrawn at random: +1.41 nats for Qwen3-8B and +0.42 for Llama-3.1-8B. The effect does not grow with the generator's informativeness: the announced slopes are 0.06 and 0.02, both not detected. So it reads as a constant, not as a failure to drop the task as evidence.
  - For comparison, outside the announced arm and under the uninformative generator, the key task lowers yes/no log-odds relative to the coin task by 0.20 nats in both models.
- **History-only helper:** saying the action was chosen by a helper who saw only the history raises Qwen3-8B's prediction by 1.32 nats. The exact effect is zero, and the effect does not vary with the generator (slope 0.03), so it too reads as a constant. Llama-3.1-8B: −0.05.

**Readout disagreement.** For Qwen3-8B, the zero-reliability note is detected with opposite signs on the two gate-passing readouts: −0.25 on the world model's yes/no and +0.27 on the state-inference yes/no. Per the design this is reported as a finding, not resolved.

## 6. Secondary readout: Qwen3-8B, state inference, yes/no

This readout passed its gates. Its evidence slope is 0.62; the goal slope is 0.25 and the note slope 0.34. Holm-adjusted within this readout:

- **Pattern:** the redundancy pattern is detected, +0.40 [+0.34, +0.46].
- **Constants:** redundancy residual +0.67, zero-reliability note +0.27, tag naming +0.18, competing goal +0.10.
- **Goal minus tag:** +0.17 [−0.09, +0.43], not detected (MDE 0.35).

The redundancy pattern therefore appears in both of Qwen3-8B's gate-passing readouts, at different sizes.

## 7. The specificity question and the ALFWorld surface

**Specificity: not assessable for either model** (section 3). For Qwen3-8B the yes/no arm alone was computed, from the world model's yes/no against the state-inference yes/no, each divided by its own evidence slope. In the direction of the world-model effect it shows a larger normalized failure in the world-model framing for all three estimands:

- goal minus tag: −2.16 [−4.19, −0.13];
- redundancy residual: +2.33 [+1.76, +2.90];
- zero-reliability note: −1.64 [−1.79, −1.48].

The rule needs the question-free arm too, so these are descriptive. Being on the yes/no readout only, they cannot exclude a difference in how the two questions are read.

**ALFWorld surface (secondary transfer check).** Only Llama-3.1-8B's reply readout passed its gates there. Its evidence slope is 0.45 [0.27, 0.63] and its goal slope 0.17. Under the uninformative generator:

- goal minus tag: −0.69 [−0.75, −0.62];
- tag naming: −0.07;
- goal against the neutral task: −0.19;
- keychain task against creditcard task: −0.56.

With no generator stated, the keychain task raises the probability that the drawer shows something by 1.18 nats over the neutral task. Its standardized goal-minus-tag effect (−1.55) has the same sign as the toy reply readout's (−0.77), but that toy readout failed its gate for Llama-3.1-8B. Qwen3-8B's ALFWorld readouts both failed their gates, and their values are reported in the metrics file only.

## 8. Numerical sensitivity (cross-GPU subset, rule 4)

The 324-item subset was rescored on A40 in the same stack. Shifts of the family estimates were 0.00 to 0.09 nats. A reading is labelled environment-dependent if its decision changes, or if its estimate moves by more than a fifth of its margin. Labelled environment-dependent:

- **Qwen3-8B, primary readout:**
  - competing goal (decision changed on the subset; estimate +0.040 against +0.053);
  - redundancy pattern (shift 0.033 against a fifth-margin of 0.030; decision unchanged);
  - zero-reliability note (shift −0.043);
  - history-only provenance (shift −0.039);
  - provenance slope (shift +0.088).
- **Llama-3.1-8B, primary readout:** history-only provenance (decision changed on the subset; estimate −0.056 against −0.063).

Every other family decision, goal minus tag's equivalence status and both pattern equivalences are unchanged.

## 9. What these results do and do not show

**What the results show.**

- **Weak use of stated evidence.** On their only valid world-model readout, both models use the history's evidence weakly, use the stated base rate in the wrong direction, and use the task's or tag's stated informativeness hardly at all.
- **Departures where the exact posterior is unmoved.** Both models move their prediction in many places where the exact posterior does not move. A cue that should be screened off, an announcement that the task is uninformative, and a provenance statement that carries no information all shift the prediction, mostly by amounts that do not depend on the evidence (constants).
- **One structured residual.** Qwen3-8B also shows a residual that such a constant cannot produce: notes still raise its prediction when existence is already settled.
- **Goal role at the target size: not assessable.** Whether the goal line acts differently from a matched non-goal line could not be assessed at the size of interest. Wording effects shared across cells make goal minus tag too imprecise in both models: MDEs of 0.59 and 0.34 nats against targets of about 0.1.

**What the results do not show.**

- **No mechanism or mixture.** Rejecting pure accounts (for example, "a constant attached to the note" for Qwen3-8B's redundancy pattern) does not identify a mechanism or a mixture; an account not considered may produce the same pattern. Any fitted components would only suffice for these prompts.
- **Nothing about world-model specificity.** The specificity rule could not be applied, because the question-free state readout, and for Qwen3-8B also the reply readout, failed their gates.
- **No statement about other formats.** The readouts disagree in places (section 5, and the descriptive values of the failed readouts in the metrics file), and each is reported separately. The yes/no readout may be read partly as a question about task progress.
- **Toy only.** The ALFWorld-surface check passes for one readout of one model only.
- **Qwen nulls are not zeros.** Qwen3-8B nulls are "not detected at the stated MDE", never zero. For Llama-3.1-8B, only the redundancy pattern was shown equivalent to zero.

## 10. Files

- Design: `notes/bayesenv_design.md`, including the smoke readouts in section 17.
- Review: `notes/bayesenv_review1.md`.
- Deviation D-BE1: `notes/deviations.md`.
- Analysis output: `notes/bayesenv_metrics.json`, a copy of `runs/bayesenv_analysis/metrics.json` from job 7354564. Its `provenance` block lists every input file's SHA-256.
- Code: `wf/bayesenv/`.
- Runs: `wf/config/bayesenv_*.yaml`.
- Jobs: `pace/bayesenv.sbatch` and `pace/bayesenv_cpu.sbatch`.
- Tests: `tests/test_bayesenv.py`.
- Score files are on ICE under `runs/bayesenv_*` in the study's checkout and are not in git.
- Rerun the analysis with `python -m wf.runs bayesenv_analysis`, from a clean checkout containing the tag, with the run folders present.
