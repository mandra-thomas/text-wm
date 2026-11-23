# Mechanism battery: results (2026-11-22)

Exploratory, not preregistered. Design, predictions and reading rules: [mechanism2_design.md](mechanism2_design.md), frozen before any score at tag `mechanism2-design-frozen-2026-11-22` (commit `6eab8af`, branch `mechanism2`); where its sections differ, the later section holds (11 to 15 amend 1 to 10). Scoring: `mech2_score`, 8 game shards on L40S GPUs (ICE jobs 7241509 to 7241516), all at `6eab8af`; the tokenization gate passed for every model in every shard. Analysis: `mech2_analysis` (ICE job 7244859, at `6eab8af`, clean checkout). Integrity check before analysis: Qwen3-8B and Llama-3.1-8B 41,440 scores each, Mistral-7B-Instruct-v0.3 23,756, each equal to the number expected; no score missing, duplicated, unexpected, stale (item hash) or undefined. Every number in this note is from [mechanism2_metrics.json](mechanism2_metrics.json), a copy of the run's `metrics.json`, unless it is marked as coming from the robustness deep-dive [mechanism2_dx.md](mechanism2_dx.md) (exploratory, post hoc). An independent recompute in that note reproduces every primary estimate to floating-point precision.

## 1. What was run, and what question it asks

**The effect being explained.** RQ2 (preregistered) found that on ALFWorld actions that the history already shows will fail ("determined failures"), a task line that names the action's object raises the model's log-odds of the success text against "Nothing happens.": the aligned-minus-swapped effect was 6.12 nats for Qwen3-8B and 1.65 for Llama-3.1-8B. Here "aligned" is the episode's real task (for example "put a clean bowl in cabinet.", with the action `take bowl 1 from cabinet 1`) and "swapped" is the same task with another object of the room (for example "put a clean pan in cabinet."). The battery asks why naming the object has this effect.

**Six candidate accounts** (design §1 and §11.1; they are not mutually exclusive):

- **H1, no state tracking.** The model does not recover the facts that decide the failure, and falls back on the task. The battery cannot test it through the arms; its readout is the knowledge split (§7).
- **H2, task as evidence.** The task line is read as information about hidden facts: a bowl task implies there is a bowl that can be got. The belief changes and the prediction follows it.
- **H3, referent salience.** Making the object salient in the task line, by name or by description, raises the success reading whatever the line says about it. It is content-blind by definition.
- **H4, wishful prediction.** The model predicts the outcome the user wants.
- **H5, policy blending.** The model blends "what would an agent with this task do" into "what would happen": an action that is a sensible step for the task reads as one that works.
- **H6, prompt artefacts,** split into **H6_lexical** (word overlap between the task line and the action string) and **H6_position** (where the task line sits in the prompt).

**Items, models, formats.** A stratified sample of 1,000 of RQ2's 4,666 primary items (124 games; strata are game by failure reason), plus all 514 items at step 0 (no action yet in the history), 1,403 items in all. Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct are the primary models, few-shot with example set 1 and history only, as in RQ2. Mistral-7B-Instruct-v0.3 is a secondary model, reported descriptively only, scored on the arms the primary contrasts need and on the sample only (plus the position arms at step 0). Every arm is scored in two formats:

- the **two-candidate** format, whose score L is the log-odds of the item's success text against "Nothing happens.";
- the **yes/no** format, where the prompt ends with the question "will this action succeed, or will the environment reply "Nothing happens."?" and L is the log-odds of "Yes" against "No".

**Arms.** Every arm changes only the task line (or, for the position arms, only where it sits). The ones the primary contrasts read, with the bowl example:

| Arm | Task line in the bowl example |
| --- | --- |
| aligned | Your task is to: put a clean bowl in cabinet. |
| swapped | Your task is to: put a clean pan in cabinet. |
| mention_ignore | Your task is to: put a clean pan in cabinet. Ignore the bowl. |
| desire_keep_sw | Your task is to: put a clean pan in cabinet. Make sure that the bowl is not taken; leave it where it is. |
| desire_check | Your task is to: find out whether a bowl can be taken from the cabinet 1; the user hopes it cannot. |
| desire_check_can | the same line ending "the user hopes it can." |
| described | Your task is to: put a clean deep round dish for soup or cereal in cabinet. |
| done | Your task was to: put a clean bowl in cabinet. It is already done; nothing is left to do. |
| cancelled | Your task was to: put a clean bowl in cabinet. The user has cancelled it; nothing is left to do. |
| negated (take actions only) | Your task is to: put a clean pan in cabinet. The user says there is no bowl in the cabinet 1. |
| affirmed (take actions only) | the same line with "there is a bowl in the cabinet 1." |
| aligned_start, swapped_start (step 0 only) | the aligned or swapped line moved from the last line before "Action:" to the top of the history |

**The eight primary contrasts, in plain words.** Each is a mean over items of a within-item difference of L, in nats. Its size ratio r divides it by the aligned-minus-swapped effect on the same items and in the same format:

$$r = \frac{\text{mean contrast}}{\text{mean of } L(\text{aligned}) - L(\text{swapped})}.$$

So r = 1 means "as large as the whole goal effect", and r = −1 "as large, in the other direction".

- **P1, keep against ignore** (desire_keep_sw − mention_ignore, 1,000 items). Both lines are the swapped task plus one sentence about the object; only what the sentence asks differs: "make sure it is not taken" against "ignore it". It asks whether the user wanting the action *not* to happen lowers the reading (wishful prediction says it should).
- **P2, hope polarity** (desire_check − desire_check_can, 1,000 items). The same sentence word for word except "the user hopes it cannot" against "it can". Only the user's hope differs, so only wishful prediction expects a difference (below zero).
- **P3, mention without need** (mention_ignore − swapped, 1,000 items). Adding "Ignore the bowl." to the swapped task names the object without making the action wanted or sensible. Salience and word overlap expect a positive value; wishful prediction and policy blending expect about zero or below.
- **P4, need without name** (described − swapped, 1,000 items). The real task with the object described instead of named. Every content account expects a positive value except H6_lexical, which needs the name and expects about zero.
- **P5, done against cancelled** (746 placement items whose action does not take from a target-type receptacle and whose history does not contradict "done"). Both lines quote the real task; "done" adds a claim about the world (the object now sits at the target), "cancelled" does not. Task as evidence expects done below cancelled.
- **P6, position at step 0** (514 step-0 items). The change in the goal effect when the task line moves from just before "Action:" to the top:

$$P6 = \big[L(\text{aligned\_start}) - L(\text{swapped\_start})\big] - \big[L(\text{aligned}) - L(\text{swapped})\big].$$

- **P7, negated existence** (negated − swapped, 503 take items whose history does not show the object in the receptacle). The sentence names the object and the receptacle but reports that the object is not there. Salience and word overlap expect a positive value; task as evidence expects below or about zero.
- **P8, negated against affirmed** (the same 503 items). The same sentence with "no" against "a": mention and word overlap cancel, only the claim's direction differs. Task as evidence expects below zero; the other accounts about zero.

**Fixed rules.** Each contrast's 95% interval is a game-clustered percentile bootstrap (2,000 resamples), its p a game-level sign-flip permutation (1,000), Holm-corrected over a fixed family of eight per model and format. r's interval is placed on four zones: N (below −0.25), Z (−0.25 to 0.25, "about zero"), S (above 0.25 and below 0.5, "small") and L (0.5 or above, "large"). The category names which zones the interval touches: NEG (N), NEG_OR_TRIVIAL (N and Z), NULL (Z with Holm p ≥ 0.05), TRIVIAL (Z with Holm p < 0.05), TRIVIAL_OR_SMALL (Z and S), POS_SMALL (S), POS (S and L), POS_LARGE (L). An account's prediction for a contrast is a set of zones (design §13.3). In one format the account is *consistent* on the contrast when every zone the interval touches is allowed, *contradicted* when none is, and *undecided* otherwise. Jointly, a contrast is decided only when the two formats agree; one consistent and one contradicted is *format-dependent*. An account contradicted on any contrast "cannot be the sole account"; it is not rejected, because the accounts are not exclusive. When every content account is contradicted, the rules print "no single account; mixture or an account not on the list", and the readout becomes the estimates of P5, P7 and P8 (design §12.5).

## 2. The frozen verdicts

The status line of `mech2_analysis`, verbatim except for line breaks:

> **qwen3-8b-fewshot:** P1 1.17 (TRIVIAL_OR_SMALL; yes/no POS_SMALL), P2 0.42 (TRIVIAL; yes/no TRIVIAL), P3 6.43 (POS_LARGE; yes/no POS_LARGE), P4 5.25 (POS_LARGE; yes/no POS_LARGE), P5 1.73 (TRIVIAL_OR_SMALL; yes/no POS_SMALL), P6 0.65 (NULL; yes/no NEG_OR_TRIVIAL), P7 0.26 (NULL; yes/no POS_LARGE), P8 -11.83 (NEG; yes/no NEG); both formats: no single account; mixture or an account not on the list (each content account is contradicted, so none can be the sole account); the readout is the estimates and intervals of P5, P7 and P8; consistent: none; untested: none; contradicted (cannot be the sole account): H2_evidence [P5], H3_salience [P8], H4_wishful [P1, P2, P3, P8], H5_policy_blend [P3, P8], H6_lexical [P4, P8]. Salience cannot be the sole account. A mention-associated residual of r = 1.18 [1.05, 1.34] two-candidate and 1.86 [1.67, 2.08] yes/no remains after averaging opposing claims; it is consistent with salience but also with asymmetric processing of affirmations and denials, so it does not show that salience contributes. Word overlap with the action would give the same.; position untested
>
> **llama3.1-8b-fewshot:** P1 0.28 (TRIVIAL_OR_SMALL; yes/no TRIVIAL_OR_SMALL), P2 0.09 (NULL; yes/no NULL), P3 0.49 (TRIVIAL_OR_SMALL; yes/no TRIVIAL_OR_SMALL), P4 1.80 (POS_LARGE; yes/no POS_LARGE), P5 0.54 (POS_SMALL; yes/no TRIVIAL_OR_SMALL), P6 0.51 (TRIVIAL_OR_SMALL; yes/no NEG_OR_TRIVIAL), P7 -1.70 (NEG; yes/no NEG_OR_TRIVIAL), P8 -4.98 (NEG; yes/no NEG); both formats: no single account; mixture or an account not on the list (each content account is contradicted, so none can be the sole account); the readout is the estimates and intervals of P5, P7 and P8; consistent: none; untested: none; contradicted (cannot be the sole account): H2_evidence [P5], H3_salience [P7, P8], H4_wishful [P1, P2, P8], H5_policy_blend [P8], H6_lexical [P4, P7, P8]; position untested
>
> **mistral-7b-v0.3** (secondary, descriptive): P1 -0.47 (NEG_OR_TRIVIAL; yes/no TRIVIAL_OR_SMALL), P2 -1.82 (NEG; yes/no NEG), P3 0.19 (TRIVIAL_OR_SMALL; yes/no NULL), P4 1.15 (POS_LARGE; yes/no POS_LARGE), P5 0.08 (NULL; yes/no TRIVIAL), P6 -1.07 (NEG; yes/no NEG_OR_TRIVIAL), P7 -4.88 (NEG; yes/no NEG), P8 -6.63 (NEG; yes/no NEG); both formats: no single account; mixture or an account not on the list (each content account is contradicted, so none can be the sole account); the readout is the estimates and intervals of P5, P7 and P8; consistent: none; untested: none; contradicted (cannot be the sole account): H2_evidence [P2, P5], H3_salience [P2, P7, P8], H4_wishful [P7, P8], H5_policy_blend [P2, P7, P8], H6_lexical [P2, P4, P7, P8]; position untested

**In words.** For each of the three models, every one of the five content accounts (task as evidence, salience, wishful prediction, policy blending, word overlap) is contradicted on at least one contrast in both formats, so none of them can be the sole explanation. The frozen reading is "no single account; mixture or an account not on the list", and by design the readout is then the estimates and intervals of P5, P7 and P8 (section 4). The position account is untested for every model: no model's P6 was decided in both formats.

**Which fixed sentences the rules gave, and why.**

- *The mention-residual sentence (design §14.1), Qwen only.* Its condition is that salience is contradicted only on contrasts that add an explicit claim (P5, P7, P8) and that M's r interval lies wholly above 0.25 in both formats. For Qwen salience is contradicted only on P8 and M is 1.18 [1.05, 1.34] and 1.86 [1.67, 2.08], so the sentence was given. M is, in the design's fixed wording, "a mention-associated residual after averaging opposing claims"; the sentence says it "is consistent with salience but also with asymmetric processing of affirmations and denials, so it does not show that salience contributes". For Llama salience is also contradicted only on claim contrasts (P7, P8), but its M interval touches the band in the two-candidate format (r 0.35 [0.25, 0.44], zones Z and S), so no M sentence applies.
- *The claims sentence (design §13.1 (b)) was not given for any model,* because it needs task as evidence to be called consistent, and task as evidence is contradicted on P5 for all three.
- *The negation-placebo sentence (design §14.2) was not given,* because the placebo did not lie below the band in both formats for either primary model (section 4.3; Mistral was not scored on the placebo).
- *The desire_check − desire_keep line (design §14.4) was not added to the status line,* because it is added only when wishful prediction and policy blending both stand (consistent or untested), and both are contradicted for every model. The readout is reported in section 5.2 anyway.

**Each format's own verdict,** reported next to the joint one as the design requires (§11.7). The lists name the contrasts that contradict each account in that format.

| Model, format | Task as evidence | Salience | Wishful prediction | Policy blending | Word overlap | Position |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen, two-candidate | P5 | P7, P8 | P1, P2, P3, P8 | P3, P8 | P4, P7, P8 | contradicted (P6) |
| Qwen, yes/no | P5, P7 | P5, P8 | P1, P2, P3, P5, P7, P8 | P1, P3, P5, P7, P8 | P4, P5, P8 | untested |
| Llama, two-candidate | P5 | P5, P7, P8 | P1, P2, P5, P7, P8 | P5, P7, P8 | P4, P5, P7, P8 | untested |
| Llama, yes/no | P5 | P7, P8 | P1, P2, P8 | P8 | P4, P7, P8 | untested |
| Mistral, two-candidate | P2, P5 | P2, P7, P8 | P7, P8 | P2, P7, P8 | P2, P4, P7, P8 | consistent (P6) |
| Mistral, yes/no | P2, P5 | P2, P3, P7, P8 | P1, P7, P8 | P2, P7, P8 | P2, P3, P4, P7, P8 | untested |

Every content account is contradicted in every format for every model, so the single-format readings agree with the joint one. The one format-dependent contrast is Qwen's P7: about zero in the two-candidate format (NULL) and large in the yes/no format (POS_LARGE), so every account's reading of it is format-dependent and it counts as undecided.

## 3. The eight contrasts, per model

Estimates are in nats with 95% intervals; r is the contrast over the aligned effect on the same items and format; "Holm p" is within model and format (0.008 is the smallest value 1,000 permutations allow after Holm's correction). The reference row gives the aligned effect that r divides by.

**Qwen3-8B**

| Contrast | Two-candidate: estimate | r | Holm p | Category | Yes/no: estimate | r | Holm p | Category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aligned effect on the sample (reference) | +6.24 [5.58, 6.91] | 1 | | | +2.80 [2.55, 3.04] | 1 | | |
| P1 keep against ignore | +1.17 [0.63, 1.72] | 0.19 [0.11, 0.26] | 0.008 | TRIVIAL_OR_SMALL | +1.20 [1.03, 1.38] | 0.43 [0.37, 0.50] | 0.008 | POS_SMALL |
| P2 hope polarity | +0.42 [0.20, 0.63] | 0.07 [0.03, 0.10] | 0.008 | TRIVIAL | +0.48 [0.32, 0.65] | 0.17 [0.11, 0.23] | 0.008 | TRIVIAL |
| P3 mention without need | +6.43 [5.81, 7.09] | 1.03 [0.92, 1.15] | 0.008 | POS_LARGE | +3.21 [2.95, 3.48] | 1.15 [1.05, 1.26] | 0.008 | POS_LARGE |
| P4 need without name | +5.25 [4.74, 5.74] | 0.84 [0.75, 0.95] | 0.008 | POS_LARGE | +2.86 [2.53, 3.21] | 1.02 [0.93, 1.12] | 0.008 | POS_LARGE |
| P5 done against cancelled | +1.73 [1.42, 2.03] | 0.26 [0.21, 0.31] | 0.008 | TRIVIAL_OR_SMALL | +0.92 [0.79, 1.06] | 0.31 [0.26, 0.36] | 0.008 | POS_SMALL |
| P6 position at step 0 | +0.65 [0.08, 1.22] | 0.06 [0.01, 0.11] | 0.072 | NULL | −1.77 [−2.10, −1.41] | −0.23 [−0.27, −0.19] | 0.008 | NEG_OR_TRIVIAL |
| P7 negated existence | +0.26 [−0.38, 0.84] | 0.05 [−0.08, 0.15] | 0.438 | NULL | +1.97 [1.70, 2.25] | 0.68 [0.59, 0.78] | 0.008 | POS_LARGE |
| P8 negated against affirmed | −11.83 [−12.86, −10.84] | −2.26 [−2.64, −1.98] | 0.008 | NEG | −6.90 [−7.89, −6.00] | −2.37 [−2.76, −2.02] | 0.008 | NEG |

The aligned effects behind r on the other populations are 6.76 and 2.97 (P5's 746 items), 11.04 and 7.71 (the 514 step-0 items) and 5.24 and 2.91 (P7's 503 items), two-candidate and yes/no.

**Llama-3.1-8B**

| Contrast | Two-candidate: estimate | r | Holm p | Category | Yes/no: estimate | r | Holm p | Category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aligned effect on the sample (reference) | +1.60 [1.45, 1.75] | 1 | | | +2.22 [2.06, 2.39] | 1 | | |
| P1 keep against ignore | +0.28 [0.14, 0.44] | 0.18 [0.08, 0.27] | 0.008 | TRIVIAL_OR_SMALL | +0.48 [0.33, 0.63] | 0.22 [0.15, 0.28] | 0.008 | TRIVIAL_OR_SMALL |
| P2 hope polarity | +0.09 [−0.03, 0.21] | 0.06 [−0.02, 0.14] | 0.152 | NULL | −0.02 [−0.12, 0.10] | −0.01 [−0.06, 0.04] | 0.771 | NULL |
| P3 mention without need | +0.49 [0.30, 0.67] | 0.31 [0.20, 0.41] | 0.008 | TRIVIAL_OR_SMALL | +0.65 [0.52, 0.78] | 0.29 [0.24, 0.35] | 0.008 | TRIVIAL_OR_SMALL |
| P4 need without name | +1.80 [1.65, 1.96] | 1.13 [1.03, 1.23] | 0.008 | POS_LARGE | +2.58 [2.40, 2.77] | 1.16 [1.07, 1.26] | 0.008 | POS_LARGE |
| P5 done against cancelled | +0.54 [0.42, 0.66] | 0.34 [0.26, 0.43] | 0.008 | POS_SMALL | +0.66 [0.53, 0.80] | 0.27 [0.22, 0.34] | 0.008 | TRIVIAL_OR_SMALL |
| P6 position at step 0 | +0.51 [0.34, 0.67] | 0.34 [0.22, 0.46] | 0.008 | TRIVIAL_OR_SMALL | −0.80 [−1.05, −0.56] | −0.24 [−0.30, −0.18] | 0.008 | NEG_OR_TRIVIAL |
| P7 negated existence | −1.70 [−2.10, −1.32] | −0.75 [−0.99, −0.54] | 0.008 | NEG | −0.55 [−0.81, −0.31] | −0.21 [−0.31, −0.12] | 0.008 | NEG_OR_TRIVIAL |
| P8 negated against affirmed | −4.98 [−5.42, −4.54] | −2.19 [−2.56, −1.87] | 0.008 | NEG | −5.23 [−5.60, −4.85] | −1.99 [−2.22, −1.79] | 0.008 | NEG |

Aligned effects on the other populations: 1.57 and 2.43 (P5), 1.51 and 3.34 (step 0), 2.28 and 2.62 (P7).

**Mistral-7B-Instruct-v0.3** (secondary, descriptive; no conclusion rests on it)

| Contrast | Two-candidate: estimate | r | Holm p | Category | Yes/no: estimate | r | Holm p | Category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aligned effect on the sample (reference) | +1.12 [0.94, 1.31] | 1 | | | +2.70 [2.43, 2.98] | 1 | | |
| P1 keep against ignore | −0.47 [−0.69, −0.24] | −0.42 [−0.63, −0.21] | 0.008 | NEG_OR_TRIVIAL | +0.78 [0.60, 0.96] | 0.29 [0.22, 0.36] | 0.008 | TRIVIAL_OR_SMALL |
| P2 hope polarity | −1.82 [−1.97, −1.67] | −1.63 [−1.97, −1.36] | 0.008 | NEG | −3.34 [−3.58, −3.12] | −1.24 [−1.38, −1.11] | 0.008 | NEG |
| P3 mention without need | +0.19 [−0.03, 0.41] | 0.17 [−0.03, 0.35] | 0.174 | TRIVIAL_OR_SMALL | −0.19 [−0.41, 0.04] | −0.07 [−0.16, 0.01] | 0.138 | NULL |
| P4 need without name | +1.15 [0.94, 1.36] | 1.02 [0.89, 1.16] | 0.008 | POS_LARGE | +2.87 [2.57, 3.17] | 1.06 [1.00, 1.13] | 0.008 | POS_LARGE |
| P5 done against cancelled | +0.08 [−0.07, 0.23] | 0.07 [−0.06, 0.19] | 0.296 | NULL | +0.21 [0.06, 0.36] | 0.07 [0.02, 0.13] | 0.012 | TRIVIAL |
| P6 position at step 0 | −1.07 [−1.34, −0.78] | −0.40 [−0.47, −0.32] | 0.008 | NEG | −1.70 [−2.11, −1.29] | −0.28 [−0.34, −0.23] | 0.008 | NEG_OR_TRIVIAL |
| P7 negated existence | −4.88 [−5.32, −4.48] | −5.01 [−7.11, −3.79] | 0.008 | NEG | −1.96 [−2.27, −1.64] | −0.71 [−0.91, −0.55] | 0.008 | NEG |
| P8 negated against affirmed | −6.63 [−7.14, −6.18] | −6.79 [−9.38, −5.35] | 0.008 | NEG | −5.69 [−6.06, −5.35] | −2.07 [−2.33, −1.84] | 0.008 | NEG |

Mistral's two-candidate aligned effect is small (1.12 on the sample, 0.98 on P7's items), so its two-candidate r values for P7 and P8 are large and wide; the yes/no ratios are the better guide to its sizes.

**Joint readings, account by contrast.** "cons" is consistent in both formats, "CONTRA" contradicted in both, "fmt" format-dependent, "und" undecided. Position (P6) was undecided for every model.

| Account | Qwen P1 to P8 (without P6) | Llama P1 to P8 | Mistral P1 to P8 |
| --- | --- | --- | --- |
| Task as evidence | cons, cons, cons, cons, CONTRA, fmt, cons | cons, cons, cons, cons, CONTRA, cons, cons | und, CONTRA, cons, cons, CONTRA, cons, cons |
| Salience | cons, cons, cons, cons, und, fmt, CONTRA | cons, cons, und, cons, und, CONTRA, CONTRA | und, CONTRA, und, cons, cons, CONTRA, CONTRA |
| Wishful prediction | CONTRA, CONTRA, CONTRA, cons, und, fmt, CONTRA | CONTRA, CONTRA, und, cons, und, und, CONTRA | und, cons, und, cons, cons, CONTRA, CONTRA |
| Policy blending | und, cons, CONTRA, cons, und, fmt, CONTRA | und, cons, und, cons, und, und, CONTRA | und, CONTRA, und, cons, cons, CONTRA, CONTRA |
| Word overlap (H6_lexical) | cons, cons, cons, CONTRA, und, fmt, CONTRA | cons, cons, und, CONTRA, und, CONTRA, CONTRA | und, CONTRA, und, CONTRA, cons, CONTRA, CONTRA |

The pattern that matters: for both primary models, task as evidence fails only on P5, and salience fails only on contrasts that add an explicit claim (P8 for Qwen; P7 and P8 for Llama). Wishful prediction and policy blending fail on contrasts without any claim as well (P1, P2 and P3 for Qwen's wishful prediction; P3 for Qwen's policy blending).

## 4. The mixture readout: P5, P7, P8, M, and the negation placebo

The design fixes this as the readout when no single account fits (§12.5), and it is reported for every model.

### 4.1 P5, P7 and P8

These are the estimates in the tables above. Read together:

- **P8 is large and negative for every model and format:** r between −1.99 and −2.37 for the primary models (−11.83 and −6.90 nats for Qwen, −4.98 and −5.23 for Llama). A report that the object is not in the receptacle, against the same report with "a", moves the reading toward failure by about twice the whole goal effect. The models read the claim and follow it.
- **P7 differs by model.** For Llama the denial sentence lowers the reading below the swapped task (r −0.75 two-candidate, −0.21 yes/no). For Qwen it does not: about zero in the two-candidate format (r 0.05 [−0.08, 0.15]) and positive in the yes/no format (r 0.68 [0.59, 0.78]). A sentence that names the bowl and denies it is in the cabinet leaves Qwen's reading at or above the swapped task's.
- **P5 is positive for both primary models, where task as evidence predicted it negative.** Qwen +1.73 (r 0.26) and +0.92 (r 0.31); Llama +0.54 (r 0.34) and +0.66 (r 0.27). The design fixed in advance how to read this: "A positive P5 (done above cancelled) is predicted by no account; it would be read as a reaction to the completion wording itself ("done"), not as support for any account." (§5; also §11.7). Mistral's P5 is about zero (NULL and TRIVIAL).

**P5 against cancelled's own lift** (§12.7). Task as evidence expects "done" to give back part of what the cancelled line gained over the swapped task, a fraction q between −1 and 0:

$$q = \frac{L(\text{done}) - L(\text{cancelled})}{L(\text{cancelled}) - L(\text{swapped})}.$$

On P5's items cancelled's lift is +4.11 nats (r 0.61) and +1.93 (r 0.65) for Qwen, so q is +0.42 [0.33, 0.53] two-candidate and +0.48 [0.41, 0.56] yes/no: "done" adds to the lift instead of taking it back. For Llama the cancelled lift is about zero in the two-candidate format (+0.07 [−0.14, 0.28]), so q is not estimable there (524 of 2,000 draws undefined); in the yes/no format the lift is +1.55 and q +0.43 [0.32, 0.55]. P5 is also positive on the placement items it leaves out: on the 94 items that take from a target-type receptacle (where task as evidence itself predicts done above cancelled) +1.24 (Qwen) and +0.91 (Llama); on the 62 items whose history contradicts "done" +2.36 and +0.53; on all 887 placement items +1.72 and +0.58 (two-candidate).

### 4.2 M and its two halves

M is defined on P7's 503 items as the mean of the negated and affirmed arms minus the swapped arm:

$$M = \tfrac{1}{2}\big[L(\text{negated}) + L(\text{affirmed})\big] - L(\text{swapped}) = \tfrac{1}{2}\Big(\big[L(\text{affirmed}) - L(\text{swapped})\big] - \big[L(\text{swapped}) - L(\text{negated})\big]\Big).$$

Averaging the two opposite reports cancels the claim only if the denial lowers L exactly as much as the affirmation raises it. The design's fixed reading (§14.1) is that M is "a mention-associated residual after averaging opposing claims": consistent with salience, but also with asymmetric processing of affirmations and denials, or with word overlap with the action, so it does not show that salience contributes. Its two halves are reported so that the asymmetry is visible; they describe the pattern and do not attribute it.

| Model | M, two-candidate | M, yes/no | Affirmed − swapped (two-cand.; yes/no) | Swapped − negated (two-cand.; yes/no) |
| --- | --- | --- | --- | --- |
| Qwen | +6.17 [5.46, 6.84], r 1.18 [1.05, 1.34] | +5.42 [4.96, 5.92], r 1.86 [1.67, 2.08] | +12.09 (r 2.31); +8.87 (r 3.05) | −0.26 (r −0.05); −1.97 (r −0.68) |
| Llama | +0.79 [0.53, 1.03], r 0.35 [0.25, 0.44] | +2.07 [1.89, 2.24], r 0.79 [0.71, 0.87] | +3.28 (r 1.44); +4.68 (r 1.78) | +1.70 (r 0.75); +0.55 (r 0.21) |
| Mistral | −1.57 [−1.87, −1.28], r −1.61 [−2.47, −1.10] | +0.88 [0.56, 1.21], r 0.32 [0.22, 0.41] | +1.74 (r 1.79); +3.73 (r 1.35) | +4.88 (r 5.01); +1.96 (r 0.71) |

For Qwen, the affirmation lifts the reading by about twice the goal effect, while the denial does not lower it below the swapped task at all (in the yes/no format the denial line sits above it). The halves are therefore very unequal, and M is about as large as the whole goal effect. For Llama both halves point the expected way (the affirmation up, the denial down), with the affirmation larger. For Mistral the denial weighs more in the two-candidate format and the affirmation in the yes/no format. In the design's words, a given pair of halves fits either salience plus a symmetric reading of the claims or task as evidence read asymmetrically.

### 4.3 The negation placebo and P8 net of it

The placebo pair repeats P8's construction for the swapped task's object ("The user says there is no pan in the cabinet 1." against "… a pan …"), which says nothing about the action's object. A model whose reading drops at the word "no" whatever it denies would show it below zero (design §13.5). P8 net of the placebo is

$$S_3 = \big[L(\text{negated}) - L(\text{affirmed})\big] - \big[L(\text{placebo\_negated}) - L(\text{placebo\_affirmed})\big].$$

| Model | Placebo, two-candidate | Placebo, yes/no | Placebo without the 87 items, two-cand.; yes/no | P8 net of the placebo, two-candidate | Yes/no |
| --- | --- | --- | --- | --- | --- |
| Qwen | +0.84 [0.50, 1.20], r 0.16 [0.09, 0.23], zone Z | −0.07 [−0.26, 0.11], r −0.02 [−0.09, 0.04], Z | +1.02 (r 0.18, zones Z and S); +0.02 (r 0.01, Z) | −12.67 [−13.78, −11.66], r −2.42 | −6.83 [−7.82, −5.91], r −2.35 |
| Llama | +0.52 [0.30, 0.74], r 0.23 [0.13, 0.34], Z and S | +0.36 [0.10, 0.61], r 0.14 [0.04, 0.23], Z | +0.57 (r 0.24, Z and S); +0.40 (r 0.14, Z) | −5.50 [−6.01, −4.98], r −2.42 | −5.59 [−6.06, −5.11], r −2.13 |

The placebo is not below the band in either format for either primary model; if anything "no pan" reads slightly more toward success than "a pan". The negation artefact the placebo tests is contradicted in both formats for both models, and P8 net of the placebo is slightly larger than P8 itself. So P8 is not a reaction to the word "no". As fixed in the design (§14.5), the placebo cannot detect a negation effect bound to the object (in the two-candidate format the candidate repeats the object's name); the yes/no format, whose answers do not repeat it, gives P8 the same size. The placebo is also reported without the 87 items whose swapped task targets the action's receptacle type, where its claim bears on the swapped task; the reading does not change (the deep-dive, section 7, reads the 87 alone).

### 4.4 The first-token term of P7 and P8

The first-token term (the part of the two-candidate L carried by "You" against "Nothing") is fixed as descriptive and flagged when its sign differs from the full L's (§12.9). It does not split for any model: P7's first-token term is +1.73 (Qwen), −0.55 (Llama), −1.73 (Mistral), the same sign as the full L; P8's is −5.19, −1.85 and −3.62.

## 5. Other secondary readouts

### 5.1 Every arm against the swapped task, and practical size

The table gives each arm's effect on the sample against the swapped task as r, two-candidate and yes/no, with the yes/no **item share toward Yes**: the share of items whose yes/no log-odds are higher under the arm than under the swapped task, the families' cross-model metric. Values above 0.5 mean most items move toward Yes. The negated, affirmed and placebo arms are read on the 522 take items.

| Arm | Qwen r (two-cand.; yes/no) | Qwen share toward Yes | Llama r | Llama share | Mistral r | Mistral share |
| --- | --- | --- | --- | --- | --- | --- |
| aligned | 1; 1 | 0.895 | 1; 1 | 0.900 | 1; 1 | 0.877 |
| mention_ignore | 1.03; 1.15 | 0.936 | 0.31; 0.29 | 0.713 | 0.17; −0.07 | 0.511 |
| mention_exists ("There is a bowl somewhere in the room.") | 0.98; 0.76 | 0.851 | 0.56; 0.88 | 0.899 | not scored | |
| described | 0.84; 1.02 | 0.782 | 1.13; 1.16 | 0.957 | 1.02; 1.06 | 0.888 |
| desire_keep_sw | 1.22; 1.58 | 0.961 | 0.48; 0.51 | 0.764 | −0.24; 0.22 | 0.622 |
| desire_keep (the keep line alone, no household task) | 1.43; 1.48 | 0.936 | 0.40; 0.35 | 0.709 | not scored | |
| desire_check | 0.16; 1.54 | 0.932 | −0.47; 0.26 | 0.657 | −4.19; −1.09 | 0.146 |
| desire_check_can | 0.10; 1.37 | 0.916 | −0.53; 0.27 | 0.653 | −2.56; 0.15 | 0.509 |
| done | 0.91; 0.99 | 0.862 | 0.52; 0.94 | 0.870 | −0.12; 0.36 | 0.683 |
| cancelled | 0.62; 0.65 | 0.781 | 0.14; 0.62 | 0.766 | −0.24; 0.29 | 0.648 |
| negated | 0.07; 0.69 | 0.824 | −0.74; −0.20 | 0.356 | −4.65; −0.69 | 0.213 |
| affirmed | 2.28; 3.00 | 0.977 | 1.39; 1.74 | 0.981 | 1.70; 1.34 | 0.931 |
| neutral ("explore the room") | −0.43; 0.10 | 0.516 | −0.75; −0.10 | 0.471 | not scored | |

The swapped task's own lift over the neutral line, re-measured here, is +2.68 nats for Qwen and +1.21 for Llama (two-candidate; the design quoted +2.56 and +1.20 from the earlier run).

**Practical size.** A large log-odds effect need not change what the model would answer. In the two-candidate format, the score s = σ(L) moves by +6.5 points between the swapped and aligned tasks for Qwen, +16.2 for Llama and +11.9 for Mistral; adding "Ignore the bowl." moves it by +4.4, +4.1 and +0.1 points. In the yes/no format the mean σ (the average probability of "Yes" against "No") moves by +0.01 points for Qwen, +16.38 for Llama and +30.69 for Mistral between swapped and aligned. Qwen's yes/no answers on these determined failures are essentially always "No" whatever the task line: the deep-dive (section 9) finds it answers "Yes" on 0.0% of sample items under both the aligned and the swapped task, and on 6.4% of P7's 503 take items under the affirmed line. So for Qwen the battery's yes/no effects are shifts of confidence within "No", not changes of answer. For Llama they are changes of answer: 22.4% Yes under the aligned task against 6.1% under the swapped one, 5.9% under mention_ignore, 28.5% under the described task, and 48.3% under the affirmed line against 2.4% under the swapped task on P7's 503 take items (deep-dive, exploratory).

### 5.2 Desire_check − desire_keep (S1), the second separator of wishful prediction and policy blending

Both lines are bare desire lines without the household task, and both want the action not to happen; wishful prediction expects about zero, policy blending positive (design §13.4). Qwen: −7.87 [−8.43, −7.30], r −1.26 (zone N) two-candidate, +0.17 [−0.11, 0.46], r 0.06 (Z) yes/no. Llama: −1.40 [−1.59, −1.21], r −0.88 (N) and −0.21 [−0.32, −0.08], r −0.09 (Z). Jointly, policy blending is contradicted on S1 for both models (neither format is positive), and wishful prediction is format-dependent. The design adds this line to the status only when both accounts stand; they do not, so it is reported here as a descriptive check that agrees with the primary verdicts against policy blending. Mistral was not scored on desire_keep.

### 5.3 The first P1 and P2, and the desire gap

The first versions of P1 and P2 (design §12.1), kept as secondary readouts: desire_keep − mention_ignore is +2.46 (r 0.39) and +0.93 (r 0.33) for Qwen, +0.15 (r 0.09) and +0.13 (r 0.06) for Llama; desire_check − swapped is +1.02 (r 0.16) and +4.31 (r 1.54) for Qwen, −0.76 (r −0.47) and +0.58 (r 0.26) for Llama. The gap the review worried about (the bare keep line against the same sentence after the swapped task) is +1.29 (r 0.21) and −0.27 (r −0.10) for Qwen and −0.13 and −0.35 for Llama: dropping the household task costs the keep line little or nothing here.

### 5.4 Mention_ignore against the existence sentence

"Ignore the bowl." against "There is a bowl somewhere in the room." (both after the swapped task): +0.30 [−0.33, 0.90] for Qwen and −0.41 [−0.58, −0.24] for Llama. For Qwen, asserting that the object exists adds nothing beyond mentioning it; for Llama the assertion adds a little.

## 6. Position and step 0

**P6 (position at step 0).** Moving the task line from just before "Action:" to the top of the history at step 0 lowers L for both lines by similar amounts, so the goal effect barely changes. For Qwen, moving the aligned line lowers L by 4.10 nats and moving the swapped line by 4.75, so P6 is +0.65 (r 0.06, NULL) two-candidate and −1.77 (r −0.23, NEG_OR_TRIVIAL) yes/no. For Llama the two moves lower L by 0.92 and 1.43 (P6 +0.51, r 0.34; yes/no −0.80, r −0.24). For Mistral both moves raise L (+1.08 and +2.15; P6 −1.07, r −0.40). No model's P6 is decided in both formats, so the position account is untested for all three. In the design's terms (§12.8), moving the line also moves it above the room description and matches the examples' layout, and every step-0 item is a "never moved" failure, so P6 speaks only to that failure type.

**The content contrasts at step 0 and later** (secondary, `primary_by_step`; step 0 is all 514 step-0 items, "later" the sample's 889 items with history; two-candidate, r over the aligned effect on the same items).

| Contrast | Qwen step 0 | Qwen later | Llama step 0 | Llama later |
| --- | --- | --- | --- | --- |
| P1 keep against ignore | −2.52 (r −0.23) | +1.62 (r 0.29) | −0.14 (r −0.09) | +0.33 (r 0.21) |
| P2 hope polarity | +1.08 (r 0.10) | +0.33 (r 0.06) | −0.79 (r −0.52) | +0.19 (r 0.12) |
| P3 mention without need | +10.49 (r 0.95) | +6.04 (r 1.06) | −0.64 (r −0.42) | +0.63 (r 0.40) |
| P4 need without name | +14.34 (r 1.30) | +4.17 (r 0.74) | +1.73 (r 1.14) | +1.81 (r 1.13) |
| P5 done against cancelled | −2.03 (r −0.17) | +2.33 (r 0.38) | −0.27 (r −0.16) | +0.64 (r 0.41) |
| P7 negated existence | −6.75 (r −0.67) | +1.11 (r 0.23) | −3.32 (r −2.27) | −1.50 (r −0.63) |
| P8 negated against affirmed | −22.49 (r −2.23) | −10.55 (r −2.22) | −4.80 (r −3.28) | −4.98 (r −2.11) |

Qwen's aligned effect is 11.04 nats at step 0 against 5.67 later. In proportion to it, Qwen's mention effect (P3) is about the same at step 0 and later. But at step 0, where the agent has not moved and the receptacle has never been seen, Qwen's done line falls below the cancelled line and the denial falls below the swapped task. Both are what task as evidence predicts, and both reverse after step 0. For Llama, mentioning the object without need lowers the reading at step 0 and raises it later. The deep-dive (section 4) shows that Qwen's format-dependent P7 comes from this split: negative at step 0 in the two-candidate format, positive later in both formats. *(Updated 2026-11-23, exploratory follow-up: the battery's Qwen scores use Qwen3's thinking-off format, which puts an empty think block before the answer. In the RQ2 items, Qwen's step-0 excess exists only with that block (10.99 nats with it, 2.73 without, against 3.3 to 3.6 later; `qwen_effect_audit_pipeline.md` §7.2 A). The battery was not rescored, so whether these step-0 reversals survive without the block is not known.)*

## 7. H1: the knowledge split, and the few-shot overlap check

**Knowledge split** (`mech_score`, analysis only; design §6). Each contrast on sample items whose state question and its twin the same model answers correctly under the neutral task ("known" items: 421 of 1,000 for Qwen, 283 for Llama), as a share of its effect on all sample items. H1 predicts shares below 0.75 for positive contrasts. Note that this readout reads each contrast on every sample item that carries its arms (P5 on all 1,000, P7 and P8 on all 522 take items), not on the primary contrasts' restricted populations.

| Contrast | Qwen known share | Llama known share |
| --- | --- | --- |
| Aligned effect | 1.10 [1.03, 1.18] | 0.94 [0.79, 1.10] |
| P3 mention without need | 1.02 [0.95, 1.09] | 0.66 [0.18, 0.99] |
| P4 need without name | 1.27 [1.18, 1.37] | 0.95 [0.82, 1.06] |
| P8 negated against affirmed | 0.99 [0.92, 1.07] | 1.11 [1.03, 1.19] |

The aligned effect is fully present on items whose deciding fact the model answers correctly, as in the earlier mechanism diagnostic (1.11 and 1.01), so the effect is not confined to items whose state the model fails to track. Llama's P3 share (0.66) lies below 0.75 but with a wide interval reaching 0.99. As the design states (§9), the state question is a weak measure of knowledge, so this readout bounds H1 only loosely; the battery cannot test H1 otherwise.

**Few-shot overlap** (`rq1_score`, analysis only; descriptive because each example set's aligned-overlap items come from two or three games, and the swapped-overlap items from 2, 2 and 13). H6 predicts a positive difference for items whose real task line equals an example's task line, and a negative one where the swapped line does. For the aligned overlap, Qwen's differences are −0.43 [−0.90, 0.25] (set 1, 49 items in 2 games), −2.29 [−3.00, −0.08] (set 2, 91 items in 3 games) and −0.93 [−1.20, 2.35] (set 3, 66 items in 2 games); Llama's are −0.89, +0.47 and −0.79. For the swapped overlap, the differences are negative for both models in every set (Qwen −6.83, −0.80, −2.54; Llama −0.27, −0.38, −0.67). The aligned-overlap prediction fails in five of six cells and the swapped-overlap prediction holds in six of six; with two or three games per set, neither is evidence either way.

## 8. H7_check (exploratory account, outside the verdicts)

H7_check says that the task directs which part of the state the model checks; where the line points the check at the deciding fact, the effect should shrink (design §13.7). Its readouts, against the predictions written before any score (§13.6), are mixed:

- *Desire_check where the history shows the receptacle without the object, minus where it does not show it* (H7_check predicts below zero). Qwen −3.00 (r −0.57) and −1.63 (r −0.56): zone N in both formats, so H7_check is consistent and every other account contradicted. Llama +0.52 (r 0.23) and +0.52 (r 0.20): H7_check contradicted. The split is confounded with the failure reason. Within the one reason with both groups populated (contents or open state, 162 shown against 57 not shown), Qwen's difference is −0.63 [−2.65, 1.36] and −0.80 [−1.74, 0.18], indistinguishable from zero, and Llama's is +2.93 and +1.19, the wrong sign for H7_check.
- *Desire_check on known minus unknown items* (two-candidate only; H7_check predicts below zero). Qwen −4.12 [−5.15, −3.14], r −0.66 (N): H7_check consistent, the others contradicted. Llama −0.04, r −0.03 [−0.29, 0.27]: undecided.
- *The aligned effect on actions that would complete the task, minus other actions* (152 completing items; H7_check predicts below zero). Qwen +1.08 (r 0.17, zones Z and S) and +0.09 (r 0.03, Z): H7_check contradicted jointly. Llama −1.27 (r −0.79, N) and −0.21 (r −0.10, Z): format-dependent. Mistral +0.46 and +1.35: contradicted.

H7_check is supported by two Qwen readouts, one of which loses its support within a single failure reason. It is contradicted by Qwen's completing-action readout and by Llama's receptacle split. It remains an exploratory idea, not a finding.

## 9. What the pattern implies (exploratory reading)

The verdicts above are the frozen output. This section interprets them and is not part of the confirmatory-style reading. The battery was built to contradict sole accounts, and it can do that; it cannot identify which mixture is at work (design §11.1, §12.10: a 50/50 mixture of task as evidence and salience was read as task as evidence alone in simulation).

**Qwen3-8B.** Three facts stand out, and the deep-dive finds each stable across games, task types and formats.

1. *Mentioning the object is enough.* Adding "Ignore the bowl." to the swapped task reproduces the whole goal effect: r 1.03 two-candidate and 1.15 yes/no. The line does not ask for the bowl, makes no step involving the bowl sensible, and tells the agent the bowl does not matter. A line that wants the action *not* to happen does the same or more ("make sure the bowl is not taken", r 1.22 after the swapped task and 1.43 alone). The user's hope barely matters (P2, r 0.07 and 0.17). This is what rules out wishful prediction and policy blending as sole accounts. On these contrasts Qwen behaves as salience predicts: what the line says about the object does not matter.
2. *The name is not needed.* The described task reproduces most or all of the effect (P4, r 0.84 and 1.02), so word overlap with the action's object name cannot be the whole story.
3. *Explicit claims about the object's whereabouts are read, and strongly.* The affirmed and negated reports differ by about twice the goal effect (P8, r −2.26 and −2.37), and this is not a reaction to the word "no" (the placebo is about zero). Content-blind salience cannot produce this. But a denial cancels only the lift of mentioning the object, not more: the negated line ends level with the swapped task (two-candidate) or above it (yes/no). "Done" does not undo the lift of mentioning the real task either; it adds to it (P5 positive, q about +0.4).

Taken together, the Qwen numbers fit a simple additive description. Any sentence that refers to the object raises the success reading by about the size of the goal effect (P3, M, the done and cancelled lines), and an explicit claim about the object's presence then moves it by roughly another goal effect in the claim's direction. This is what review 3's planted scenario "salience plus ordinary reading of claims" looks like (design §13.1). But the design's adversarial check (§14.8) showed that task as evidence read asymmetrically produces the same P7, P8 and M, so this description does not attribute the effect to salience. The contrast that does not involve claims, P3, is where the accounts differ, and there task as evidence also allows a positive value (the line presupposes the object). What the battery does show for Qwen: the goal effect does not need the object to be wanted, needed or named, and does not follow the user's wishes. Whatever drives it responds to the object being referred to at all, and separately to explicit claims about it.

**Llama-3.1-8B.** The pattern is closer to task as evidence. Mentioning the object without need gives about a third of the effect (P3, r 0.31 and 0.29), the described task all of it (P4, r 1.13 and 1.16), and a denial lowers the reading below the swapped task (P7, r −0.75 and −0.21). Task as evidence is contradicted only on P5, where "done" reads slightly above "cancelled" with no cancelled lift to give back in the two-candidate format. By the design's fixed reading, a positive P5 is a reaction to the completion wording, which no account predicts. Wishful prediction is contradicted on P1, P2 and P8, and policy blending on P8 (and on S1).

**Mistral-7B** (descriptive only). It is the one model whose reading follows the user's hope (P2, r −1.63 and −1.24). This is wishful-like, but it is also the answer-format and word artefact the design flagged for P2 ("cannot" a few lines above a yes/no question; §12.4). Mentioning the object without need does nothing (P3 about zero), and denials weigh heavily.

**Across models.** Two results hold for all three: the effect survives describing the object instead of naming it (P4 large everywhere), and explicit claims about the object's presence move predictions strongly in the stated direction (P8 about −2 goal effects everywhere). Neither wishful prediction nor policy blending is a sole account for either primary model. How much of the effect a bare mention carries is model-dependent: all of it for Qwen, a third for Llama, none for Mistral.

**What this changes in the project's story.** Earlier work read the effect as a task-relevance bias at answer time, not a failure to track state. The battery narrows that for Qwen. The task does not have to make the object relevant: telling the model to ignore the object reproduces the effect, so "relevance" overstates what is needed. It is not wishful thinking, and it is not borrowed from what an agent would sensibly do. The knowledge split again finds it fully present where the model knows the deciding fact. For Llama the effect looks more like the task being read as weak evidence about the object, plus a reaction to completion wording. The models differ in mechanism, not only in size. No confirmatory verdict anywhere in the project changes.

**Practical size.** For Qwen, the yes/no effects are shifts within "No": it answers "Yes" to almost none of these failing actions under any line. The two-candidate score moves by a few points (aligned +6.5, mention_ignore +4.4). For Llama, the mention effect changes answers rarely (5.9% Yes under mention_ignore against 6.1% under the swapped task), while the described task (28.5%) and the affirmed report (48.3% against 2.4%) change many.

## 10. Limits

From design §9 and the amendments, those that bear on this reading:

- **Mixtures.** The accounts are not exclusive, and a weighted mixture can produce any value between the accounts it mixes. The battery can contradict a sole account; it cannot show that one account alone is at work, nor identify a mixture (§11.1, §12.10).
- **Arms are not all minimal pairs.** After review 2, P1, P2 and P8 are minimal pairs; the other contrasts change several features at once (§4, §9).
- **Salience against word overlap.** The two make the same content predictions except on P4 (§9, §12.3). M contains word overlap too, since both claim lines name the object and the receptacle.
- **M does not identify salience.** An asymmetric reading of affirmations and denials moves M up or down with no salience at all (§14.1, §14.8).
- **Policy blending and its rational-agent reading** predict the same pattern (§9).
- **H1** is tested only through the knowledge split, whose state question is a weak measure of knowledge (§9).
- **Unusual lines.** The desire, done and cancelled lines are unlike any ALFWorld task, so part of any difference may be a reaction to an unusual line. "Done" states a falsehood on purpose; the history contradicts it on 62 sample items, which P5 leaves out (§9).
- **The two-candidate L is not a probability of success.** It compares one success wording with "Nothing happens.". The yes/no format is the check, and the item share is the cross-model metric (§9).
- **Copying from the success text** differs between arms and acts only in the two-candidate format: desire_check, negated and affirmed name the action's receptacle, which the take success text repeats. This is one reason the two-format rule exists (§9, §12.4). Qwen's format-dependent P7 is the case the design anticipated.
- **The placebo** cannot detect a negation effect bound to the object in the two-candidate format (§14.5).
- **P6** changes three things at once (distance to the action, position above the room description, match with the examples' layout) and covers only "never moved" failures (§12.8).
- **Inherited wording** ("put a alarmclock", "a eating utensil") is kept for comparability (§9).
- **Few-shot overlap** rests on two or three games per example set (§9).
- **H7_check's readouts** are confounded with the failure reason (§13.7).
- **Intervals** are unadjusted for the eight contrasts and six accounts; error control comes from the band and the two-format rule (§12.6). The game-bootstrap intervals were slightly narrow in simulation (93–95% coverage; §9).
- **Estimand.** Unweighted means over the 1,000-item sample, which stand for the strata it covers. 117 items in 72 small strata (all four items of one game) could not be drawn (§11.8).
- **Mistral** is descriptive, scored on fewer arms and on the sample only, and was not part of any decision (§11.9, §13.5).
- **One scoring run per model,** one prompt format (example set 1, history only) and one environment.
