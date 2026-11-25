# Lexical repetition, semantic identity, existence content and task relevance: a small diagnostic (design, revised 2026-11-25)

**Status: design, not frozen.** No effect has been scored. Two 5-item smoke runs measured timing and validity only (section 11). The rules (section 9, `wf/lexsem_rules.py`) and the analysis (`wf/lexsem_analysis.py`) are written and tested on planted data; the design is to be reviewed again before the freeze tag (section 13). Exploratory, frozen procedure: the same standing as the mechanism battery and τ-auth.

**Decisions by the user (2026-11-25).** (1) The full run uses the `serve2` software stack. (2) Every sequence is scored one at a time, unpadded, as every stored project score was made; no float32 output layer. (3) Included arms: the core; "Ignore the X" with its control; the three-position arm; E4's stated-reliability pair; the failure-text sequence on every core cell (the full two-candidate score); Qwen3-8B with the empty think block in yes/no, eight conditions; Llama-3.1-8B in yes/no, twelve conditions. Expected cost about one GPU-hour. On the same day the user added six scientific qualifications (identifiability, statistics, controls, environment, the reliability arm, readouts); sections 2, 7, 10, 5, 12 and 4 answer them.

**Revision history.** First version 064edb6; reviewed twice on 2026-11-25 (section 17 lists every point and what was done).

## 1. The question

When the task line names the object an action acts on (the *aligned* task) instead of another object in the room (the *swapped* task), language-model world models predict more success for ALFWorld actions whose failure the history already determines. Call the mean within-item difference in log-odds the **goal effect**.

The interpretability pilot's stage 0 (run `interp_pilot_stage0`, Qwen3-8B, yes/no format, with Qwen3's empty think block) also scored the swapped task followed by "There is a saltshaker somewhere in the room.", naming the action's object without making it the goal. Read from its stored index, that sentence reproduces 2.26 of the 2.29-nat goal effect on the 3,861 later-step failures whose history does not name the object, but only 2.21 of 7.70 nats on the 514 step-0 items. The think block inflates Qwen's effect about four-fold at step 0 and under two-fold later (`qwen_effect_audit_pipeline.md` §7.2), and the equality is one of means: item by item, the goal and mention effects correlate at only 0.45 on those later items.

That sentence, like every mention used so far, also asserts or presupposes that the object exists, so the result cannot tell apart the explanations of the goal effect that section 2 sets out. This diagnostic crosses the *form* of the reference (exact name, spaced spelling, alias, description, the same word in another sense, matched controls) with its *existence content* (asserted, denied, placed elsewhere, merely listed, to be ignored), scores Qwen3-8B without the think block (and, for the bridge, with it), adds Llama-3.1-8B, and reads step 0 and later steps separately.

## 2. Identifiability: four levels, and what the contrasts can and cannot separate

**The candidate explanations need not be exclusive.** Recognising a word, activating the entity it names, inferring something about that entity in this room, and letting that inference move a prediction may be successive stages of one computation. A model that reads "There is a saltshaker somewhere in the room." as evidence must first recognise the word and resolve the entity. So the explanations are restated at four levels:

- **(a) Identification.** How the model identifies the entity the sentence is about: by the token string (lexical), or by the entity whatever its wording (semantic).
- **(b) Inference.** What the model infers about the entity: nothing in particular (a content-blind lift), that it exists in this room (evidence), or that it matters to the task (relevance).
- **(c) Influence.** How that inference moves the transition prediction: through the action's own tokens (which can attend to earlier text) or later, at the answer; through the predicted observation or through a reading of the question.
- **(d) Appropriateness.** Whether the influence is warranted. Every item here is a failure the history already decides, so a correct simulator should show no influence at all; any lift is inappropriate, whatever level produces it.

The diagnostic's accounts are read at levels (a) and (b): **lexical repetition** (identification by string, lift without regard to content), **semantic activation** (identification by entity, lift without regard to content), **task as evidence** (lift follows the inferred existence of the object in this room), and **task relevance** (lift follows a goal or hint role). Level (c) is probed descriptively by the position arm and by comparing readouts; level (d) is fixed by the item pool (section 12).

A reading such as "only task as evidence fits" therefore says that the *lift* depends on what the sentence claims and not on how the object is named. It does not say that the model skips lexical recognition or entity activation; those stages may well occur, and are needed to read the claim, but on their own they do not move the prediction.

**What each contrast can separate.** "Pure" means a single proposed account; "mixture" means a particular combination of them; "equivalent" lists mechanisms that predict the same outcome on that contrast, so it cannot tell them apart.

| Contrast (in words) | Level | Separates pure accounts | Separates particular mixtures | Cannot separate (behaviourally equivalent) |
|---|---|---|---|---|
| Name assertion over the goal effect | (b) | relevance (sole) from the three content accounts | the size of a goal-role component (its shortfall from 1) | lexical, semantic and evidence (all predict a lift) |
| Name listed, no claim | (b) | evidence from lexical and semantic | the claim share in a content-blind plus evidence mixture | lexical from semantic; a content-blind lift from a listed word read as a hint |
| Name placed in the house next door | (b) | evidence about this room from lexical and semantic | as above | a content-blind lift from evidence about existence in general |
| Name denied | (b) | evidence from lexical and semantic | the evidence weight (a denial may lower) | lexical from semantic; a negation-blind salience mechanism from both |
| Description asserted | (a) | lexical from semantic and evidence | the name share in lexical plus semantic, or lexical plus evidence | semantic from evidence |
| Description listed, no claim | (a), (b) | semantic from lexical and evidence | the name-by-claim interaction (with the listed name) | lexical from evidence |
| Alias asserted | (a) | lexical from semantic and evidence (never alone: rarer words) | as the description | semantic from evidence; a frequency effect from lexical |
| Spaced spelling asserted | (a) | token-sequence from word-level repetition | none | semantic, evidence and word-level repetition |
| Homonym (same token, other sense) | (a) | lexical (lifts) from evidence (no lift), never alone | a lexical component's presence | lexical from semantic activation that crosses senses |
| "Ignore the X" | (b) | relevance read as a hint from the content accounts | none | lexical, semantic and evidence (all presuppose or name the object) |
| Position: task line, end of history, after the action | (c) | none of the content accounts | none | every content account can act at any position |
| Stated reliability of the assertion | (b), (d) | qualitative evidence-weighting from content-blind accounts | none | a reliability effect from a carrier-wording effect (the two carriers differ in wording) |
| Readouts compared | (c) | none | none | a channel effect (question reading, output copying) from a belief change |

Four kinds of equivalence remain whatever the outcome.

- A computation that *activates the entity and then reads the claim* behaves exactly like pure evidence on every contrast: identification is a prerequisite, not a separate lift.
- A mechanism not on the list that lifts every mention alike, including the homonym, is read as "semantic activation not contradicted": the homonym is the only contrast that separates them, and it is barred from deciding alone (section 5). The simulation (section 10) shows this.
- The same components with different weights in different models cannot be told from different mechanisms.
- A component smaller than about a quarter to a third of the mention lift is usually missed.

**Wording.** When every proposed account is contradicted, the reading is "no single proposed account fits; mixtures and unlisted mechanisms are not distinguished". It is never worded as "a mixture of the proposed accounts": a mechanism not on the list produces the same reading.

## 3. What each account predicts

Each mention condition's lift is read relative to the lift of asserting the name (section 7): 1 means "lifts as much as asserting the name", 0 "lifts nothing".

- **Lexical repetition**: every sentence containing the exact name lifts fully, whether it asserts, denies, places the object elsewhere or lists the word; the homonym lifts; description and alias lift nothing. A strict token-sequence version predicts that the spaced spelling lifts less than the closed name.
- **Semantic activation**: every reference to the object lifts fully whatever it claims; a merely listed word may activate the entity less; a homonym may lift partly if it also activates the original concept.
- **Task as evidence** (literal claim about this room): an assertion lifts fully whatever its form; a denial, the house next door, a listed word and the homonym lift nothing.
- **Task relevance**: no non-goal sentence lifts. In its broad form (a sentence on the task card read as a hint), "Ignore the X" lifts nothing.

## 4. Model, format and readouts

**Model and format.** Qwen3-8B in plain text: no chat template, the study's own messages rendered by the Qwen audit's `plain_text`, the reply after one space. This format matched the no-block render on the goal effect (3.26 against 3.28 nats; `qwen_effect_audit_pipeline.md` §4, §7.2 B); it has not been validated for the effects of individual sentences. The with-block arm scores the name conditions in Qwen's own thinking-off format in yes/no, so format dependence of the content pattern can be read directly (section 11). Conclusions are worded per model and format.

**Three readouts, three separate conclusions.** One sequence per item, condition and format; from them:

- **First-token observation**: log p(" You") minus log p(" Nothing") at the reply's first position. Neither token names the object, and it predicts the environment's reply rather than answering a question. Threat: it is a two-word log-odds, so a sentence can raise it by moving probability off " Nothing" onto other words; the unnormalized log p(" You") and the success text's log-probability are reported beside it, with a flag when they disagree, and a condition failing the mass gate is read on log p(" You").
- **Yes/no**: log p(" Yes") minus log p(" No") for `rq1v`'s anchored question. Threat: the question may be read as "does this action advance the task?" (TextCraft round 1, τ-bench, Option C's question-specific step-0 collapse).
- **Full two-candidate**: the success text's log-probability minus that of "Nothing happens." (a third, failure-text sequence on every core cell). Threat: its success text names the object, so the exact-name conditions may gain by output-side copying; the token split (verb, object tokens, rest) is reported beside it.

None of the three is the definitive measure of the model's belief about the transition. The rules (section 9) give each readout its own reading line; there is no pooled verdict and no "primary readout decides". Disagreements between readouts are reported in their own section of the results as findings, contrast by contrast. The note does not choose, after the fact, the readout that best supports an account.

## 5. Items, conditions and controls

**Population, strata and sample.** RQ2's 4,666 primary items (replay-verified determined failures). Step 0 (no action yet; 514 items) and later steps (history after the task line does not name the action's object; 3,861 items) are separate strata; the 291 later items whose history already names the object are left out. 300 later items (108 games) and 150 step-0 items (96 games), drawn by `wf.lexsem.sample` (seeded, at most three and two per game); sorted ids hash to `cc8da36181976edd`. The 56 sampled items whose object also appears in the few-shot examples (soapbar, soapbottle) are reported apart and every reading is repeated without them.

**Conditions.** A mention condition adds one sentence to the swapped task's prompt, by default on the task line where stage 0 put its sentence (`name_assert` reproduces stage 0's prompt byte for byte). Example: action `take saltshaker 1 from shelf 2`, swapped task "put some vase on cabinet", control object toiletpaper.

| Condition | Content | Added sentence | Items |
|---|---|---|---|
| aligned, swapped | anchors | (task lines) | all |
| name, assertion | in this room | There is a saltshaker somewhere in the room. | all |
| name, denial | not in this room | There is no saltshaker in this room. | all |
| name, elsewhere | exists, not here | There is a saltshaker somewhere in the house next door. | all |
| name, listed | no claim | The following word appears in this note: saltshaker. | all |
| description, assertion | in this room | There is a container with holes for sprinkling white seasoning somewhere in the room. | all |
| description, listed | no claim | The following phrase appears in this note: container with holes for sprinkling white seasoning. | all |
| spaced spelling | in this room | There is a salt shaker somewhere in the room. | 111 compound items |
| alias | in this room | There is a skillet somewhere in the room. (pan) | 122 alias items |
| homonym | another sense | Critics often pan sequels to popular films. (pan) | 160 homonym items |
| "Ignore the X" | presupposes, irrelevant | Ignore the saltshaker. | all |
| name assertion at the end of the history | in this room | (last line of the history, just before "Action:") | all |
| name assertion after the action | in this room | (a line just after the action line) | all |
| stated reliability | in this room, reliable or not | The room's inventory list says there is a saltshaker in the room. / A tester inserted this sentence at random: there is a saltshaker in the room. | all |
| controls | the same carriers with the control object's name or description, or the homonym sentence with a control word | There is a toiletpaper somewhere in the room. / Critics often praise sequels to popular films. | as their pairs |

**Tokens.** Every name carrier keeps the name's leading space and ends it at a word boundary, so the name has exactly the same tokens as in the action line (checked for all 55 names in all name carriers with both tokenizers; checked per item at run time).

**Homonyms are not a pure lexical isolate.** Nine object types have a natural sense with the identical token (mug, watch, pan, plate, book, pencil, cup, bowl, egg). An unrelated-sense homonym may still activate the original concept. Its reading is therefore "the lexical form, without the intended sense in context"; semantic activation is allowed a partial lift on it, and the homonym never decides an account's reading on its own. A full homonym lift is not taken as proof of lexical-only processing.

**Recognition checks.** Descriptions are put to rq1x's naming question and read by rq1x's reader plus a frozen lenient list (fixed after reading Qwen's rq1x answers; about 77–79% of sampled items expected to pass); description contrasts are read on items that pass. Aliases are checked by forced choice ("Can the word "skillet" refer to a pan?" must favour Yes; the same with a frozen foil must favour No). Each model's own checks are used for its own arm.

**Controls and their audit.** Each item has one control object: an ALFWorld object type of another coarse class, absent from the room at reset, not named in the history, the action or the few-shot examples, with the same number of tokens as the action object's name, and the closest description length. Its name controls the name carriers and its description the description carriers. The audit below compares each target with the controls it received, item-weighted: token count (by construction) and tokenization; word frequency (Zipf scale from the `wordfreq` package, Speer et al., version 3); familiarity (Zipf of the everyday spelling, for example "salt shaker", since ALFWorld's closed compounds are rare as written); semantic plausibility (every control is an ALFWorld household object, so plausible in the household context, and from another class by construction); and prior probability of existence (the share of the 134 `valid_unseen` rooms whose scene at reset contains the type, from the collected transitions' scene records).

| Target (items) | Target tokens | Controls same token count | Zipf, target / controls | Familiarity, target / controls | Scene prevalence, target / controls | Most used controls |
|---|---|---|---|---|---|---|
| soapbar (46) | _soap bar | yes | 0.00 / 1.75 | 4.11 / 3.07 | 0.14 / 0.21 | spatula, wateringcan, ladle |
| mug (43) | _mug | yes | 3.71 / 4.35 | 3.71 / 4.35 | 0.78 / 0.08 | basketball, pencil, pillow |
| tomato (26) | _tomato | yes | 3.85 / 4.09 | 3.85 / 4.09 | 0.57 / 0.19 | pillow, pencil, basketball |
| apple (26) | _apple | yes | 4.76 / 4.70 | 4.76 / 4.70 | 0.57 / 0.18 | book, cloth, cd |
| saltshaker (23) | _salt sh aker | yes | 1.20 / 0.84 | 3.21 / 3.76 | 0.57 / 0.05 | tennisracket, toiletpaper, teddybear |
| potato (23) | _potato | yes | 4.11 / 4.84 | 4.11 / 4.84 | 0.57 / 0.12 | box, pen |
| pan (23) | _pan | yes | 4.39 / 4.99 | 4.39 / 4.99 | 0.57 / 0.08 | watch, newspaper |
| bowl (23) | _bowl | yes | 4.63 / 4.48 | 4.63 / 4.48 | 0.78 / 0.30 | cd, cloth, apple |
| egg (21) | _egg | yes | 4.46 / 4.18 | 4.46 / 4.18 | 0.57 / 0.13 | pencil, basketball, pillow |
| cd (20) | _cd | yes | 4.36 / 4.19 | 4.36 / 4.19 | 0.19 / 0.38 | cloth, apple, lettuce |
| cloth (16) | _cloth | yes | 4.03 / 4.78 | 4.03 / 4.78 | 0.13 / 0.44 | book, bowl, cd |
| peppershaker (15) | _peppers h aker | yes | 0.00 / 0.94 | 3.18 / 3.93 | 0.57 / 0.09 | toiletpaper, tennisracket, teddybear |
| pencil (15) | _pencil | yes | 3.95 / 4.29 | 3.95 / 4.29 | 0.20 / 0.23 | basketball, spoon |
| vase (14) | _vase | yes | 3.54 / 3.06 | 3.54 / 4.24 | 0.66 / 0.57 | butterknife, knife, fork |
| lettuce (14) | _lettuce | yes | 3.56 / 4.40 | 3.56 / 4.40 | 0.57 / 0.16 | cloth, cd, book |
| bread (14) | _bread | yes | 4.50 / 4.85 | 4.50 / 4.85 | 0.57 / 0.12 | box, pen |
| knife (10) | _knife | yes | 4.41 / 4.97 | 4.41 / 4.97 | 0.57 / 0.08 | newspaper, watch |
| book (10) | _book | yes | 5.43 / 4.20 | 5.43 / 4.20 | 0.20 / 0.31 | cloth, apple, lettuce |
| soapbottle (10) | _soap b ottle | yes | 0.00 / 1.59 | 4.03 / 3.82 | 0.72 / 0.00 | teddybear |
| pillow (9) | _pillow | yes | 3.96 / 4.01 | 3.96 / 4.01 | 0.28 / 0.35 | pencil, mug, basketball |
| keychain (9) | _key chain | yes | 2.76 / 1.62 | 4.52 / 3.43 | 0.28 / 0.04 | plunger, alarmclock, baseballbat |
| watch (9) | _watch | yes | 5.34 / 3.34 | 5.34 / 4.25 | 0.08 / 0.57 | pan, knife, fork |
| plate (9) | _plate | yes | 4.55 / 4.27 | 4.55 / 4.27 | 0.57 / 0.15 | pencil, basketball, cloth |
| alarmclock (8) | _alarm clock | yes | 0.00 / 1.09 | 4.05 / 3.66 | 0.20 / 0.00 | baseballbat, plunger |
| cup (7) | _cup | yes | 5.11 / 4.92 | 5.11 / 4.92 | 0.57 / 0.08 | newspaper, watch |
| spatula (7) | _spat ula | yes | 2.86 / 0.00 | 2.86 / 3.57 | 0.57 / 0.00 | wateringcan |

Item-weighted over the sample, the targets' and controls' Zipf frequencies differ by −0.29 on average (SD 1.17) and their familiarity by −0.09 (SD 0.75): balanced overall, with large per-type gaps (watch 5.34 against 3.34). **Prior probability of existence is not balanced**: the targets occur in 48% of rooms on average and their controls in 19% (mean gap +0.30, SD 0.34), because a control must be absent from the item's room, which favours rarer types. A sentence asserting a rare object (a basketball in a kitchen) may itself read differently from one asserting a common one. This is a stated limitation, and a pre-specified covariate: the analysis reports, per stratum and readout, the item-level correlation and slope of the name assertion's net effect on the prevalence gap (descriptive; no rule reads it).

**Position arm: what each contrast separates.** The name assertion and its own control are placed at three positions: (1) the task line; (2) the last line of the history, just before "Action:"; (3) a line just after the action line. Each position's net effect is taken against the control at the same position, so the carrier's own effect at that position is absorbed.

- (2) against (1) changes **recency** (the sentence moves closer to the answer) while keeping **causal accessibility** (the action's tokens can attend to it in both). It also changes the **prompt region**: the sentence leaves the task line and joins the history, where it reads like an observation.
- (3) against (2) changes **causal accessibility** (after the action, the action's tokens can no longer attend to it) at roughly constant recency (the two positions are a few tokens apart). It also changes the region: the sentence now sits between the action and the reply label, where it may be read as part of the observation or the answer.

So the "recency" contrast is confounded with the move from task line to history, and the "visibility" contrast with the move from history to the reply region; neither region change is controlled, because the controls are matched within position, not across. The arm's readings are worded accordingly ("recency, with the region change", "visibility, with the region change"), and it does not enter the account readings.

## 6. Estimands

For item $i$ and condition $c$, let $y_i(c)$ be a readout. The goal effect in a stratum is

$$G = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(\text{aligned}) - y_i(\text{swapped})\big].$$

Each mention condition $c$ has a control $k(c)$; its **absolute net effect**, in nats, is

$$\Delta_c = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(c) - y_i(k(c))\big].$$

The **mention lift** is $A = \Delta_{\text{name, assert}} / G$, and every other mention condition's **relative lift** is

$$\rho_c = \frac{\Delta_c}{\Delta_{\text{name, assert}}},$$

on the same items (subsets for alias, spelling, homonym, and the recognised descriptions).

**No ratio is reported alone.** Each ratio comes with its numerator and its denominator as absolute effects in nats, each with its own interval, and with the correlation of the two across bootstrap draws.

Further estimands: the unnormalized versions (log p(" You"), log p(" Nothing"), the success text's log-probability); the stage-0 bridge (name assertion minus the plain swapped task, over $G$); the item-level correlation and slope of each mention effect on the goal effect; the item-level correlation of the name assertion's net effect with the scene-prevalence gap; and each control's carrier effect against the plain swapped task.

## 7. Statistics

**Joint uncertainty.** One game-clustered bootstrap per stratum (2,000 resamples of games with replacement, seed 20260924) is shared by every estimand of that stratum, so a ratio, its numerator, its denominator, differences between ratios and the mixture weights all come from the same draws. The step comparison uses one joint bootstrap over the games of both strata. Percentile 95% intervals; a reading that changes under basic intervals is flagged "interval-sensitive".

**A small or poorly estimated reference.** A ratio is read only if its reference (the denominator: $G$ for the mention lift, the name assertion's net effect on the same items for a relative lift) has an interval that lies above zero and whose lower end is at least half its estimate. Otherwise only the absolute effects are reported for that contrast, and it does not enter the account readings. The threshold, fixed before scoring, keeps the reference's coefficient of variation at or below about 0.26 (the interval's lower half-width at most 50% of the estimate, so the standard error at most about a quarter of it). At that precision the ratio's first-order bias, roughly the squared coefficient of variation, is under 7%, the reference almost never crosses zero in a bootstrap draw (so the ratio's interval is bounded), and the same rule already governs $G$ (gate 1). In addition, relative lifts are read only if $A$'s lower bound is at least 0.25, that is, only if there is a mention lift that is not about zero to decompose. (An earlier draft required 0.5; the simulation in section 10 showed that this made every decomposition unreadable whenever the true lift was half the goal effect or less, which is plausible at step 0, while the reference rule already guards the ratios' stability.)

**Equal-width bands do not give equal power.** The predicted zones are bounded and equally wide (section 9), but power still depends on the reference size, the noise of each condition and the correlation of contrasts. Section 10 validates the whole procedure by simulation across a grid of those.

**Components under about a third of the lift are usually missed.** This limitation stands; the wording "no other component larger than about a quarter of the mention lift was detected" is used only when every contrast is consistent with one account.

## 8. Gates (per stratum and readout)

1. **Goal effect**: $G$'s interval above zero, lower end at least half the estimate; otherwise nats only.
2. **Readout mass**: first-token mass at least 0.7 and within 0.1 of the swapped condition's (a failing condition is read on log p(" You") and flagged); yes/no mass at least 0.9 (otherwise invalid in yes/no).
3. **A lift to decompose**: relative lifts read only if $A$'s lower bound is at least 0.25.
4. **Reference**: the small-reference rule of section 7, per contrast.
5. **Precision**: a relative lift with half-width above 0.25 is "imprecise" and counts as undecided.
6. **Recognition**: description and alias contrasts on recognised items only; not assessable below 100 later or 50 step-0 items.
7. **Environment**: every score file must record the same GPU type (L40S), stack, model revision, dtype (bf16) and unbatched scoring (section 11), or the analysis refuses to run.

## 9. Interpretation rules (pre-committed; `wf/lexsem_rules.py`)

**Zones.** $A$ and every relative lift are placed on five zones: *below zero* (under −0.25), *about zero* (−0.25 to 0.25), *partial* (0.25 to 0.75), *full* (0.75 to 1.25), *above* (over 1.25). A contrast's category is the set of zones its 95% interval touches; an account is consistent with a contrast when every zone is allowed, contradicted when none is, otherwise undecided.

| Relative lift | Lexical repetition | Semantic activation | Task as evidence |
|---|---|---|---|
| Name listed, no claim | full | partial or full | about zero |
| Name in the house next door | full | full | about zero |
| Name denied | full | full | about zero or below zero |
| Description asserted | about zero | full | full |
| Description listed, no claim | about zero | partial or full | about zero |
| Alias asserted | about zero | full | full |
| Homonym | partial or full | about zero or partial | about zero |

An account is contradicted overall if any contrast contradicts it, except that the alias and the homonym never do so alone; consistent if every contrast is consistent; otherwise undecided.

**Readings, one line per readout and stratum:**

- one account consistent, the others contradicted: "only X fits: no other component larger than about a quarter of the mention lift was detected";
- one account undecided, the others contradicted: "X not contradicted, not shown consistent";
- all contradicted: "no single proposed account fits; mixtures and unlisted mechanisms are not distinguished";
- otherwise: "not distinguished between …", naming every account not contradicted, with no preference.

Evidence is read in two forms: literal claim about this room (as above), and including pragmatic cues, contradicted only if the denial contradicts it too.

**Other readings.** Mention sufficiency and task relevance from $A$ (as sole account: about zero or below; as a component: the shortfall of $A$ from 1). The homonym descriptively: "the lexical form lifts without the intended sense in context", or "does not lift". "Ignore the X" against relevance read as a hint. Mixture weights (least squares on the three pure patterns, with intervals) and shares, always descriptive. The step difference in $A$ ("step 0 lower", "no step difference beyond 0.25", "step 0 higher", inconclusive), and "stratum-specific" only when an account is contradicted in one stratum, consistent in the other, and the responsible contrast's step difference lies outside ±0.25. Position: "recency, with the region change" and "visibility to the action's tokens, with the region change", each "matters" if its interval lies wholly outside ±0.25 of the goal effect, "no effect beyond 0.25" if wholly within. Item-level agreement: "tracks the goal effect item by item" (correlation lower bound at least 0.7), "weak item tracking" (upper bound below 0.5), otherwise intermediate.

**Readout disagreements.** For each pair of readouts (first-token against yes/no, first-token against full two-candidate, yes/no against full two-candidate) and each contrast, a disagreement is recorded when the two categories share no zone. The normalization flag compares the first-token readout with log p(" You"). These lists are reported as findings; no readout overrides another.

**Llama.** Llama's arm is yes/no only; the comparison with Qwen uses Qwen's yes/no line, and states that formats also differ (Qwen in plain text, Llama in its chat template).

## 10. Precision and simulated error rates

`scripts/lexsem/simulate_rules.py procedure` generates whole strata of item-level scores and reads them with the frozen analysis itself (`wf.lexsem_analysis.analyse`: bootstrap, gates, reference rule, rules). The generator has the plain-text first-token goal effect (1.31 nats, item SD 1.10; Qwen audit sample); an item-level mention response correlated 0.45 with the goal effect (stage 0) with a game component, shared by every mention condition of the item; one control per carrier shared by the conditions that use it; unequal carrier noise (0.35 to 0.55 nats by carrier); and condition-specific item heterogeneity at two levels, an optimistic one (relative-lift half-widths about 0.06 to 0.10 on the later stratum) and the planning one (about 0.15 to 0.20 later, 0.20 to 0.25 at step 0). The grid crosses reference sizes ($A$ of 0.95, 0.5 and 0.3) with planted truths: each pure account; semantic with a homonym that activates the concept at 0.4; partial-strength evidence (claims at 0.7); 85/15 and 70/30 evidence plus semantic; even mixtures; and an unlisted mechanism that lifts every mention alike. 100 datasets per cell.

SIM_TABLE

## 11. Environment and cost

**Pinned for every comparison.** GPU type L40S; the `serve2` stack (torch 2.14.1+cu126, transformers 5.23.0, Python 3.11.18; decided by the user); Qwen3-8B at revision b968826d9c46dd6066d109eabc6255188de91218 and Llama-3.1-8B-Instruct at 0e9e39f249a16976918f6564b8830bc894c89659; bf16 weights with a float32 log-softmax of the bf16 logits; one unpadded sequence per forward pass through the full model. The scorer writes these to an environment record beside each score file, and the analysis refuses to run unless every file's record matches (gate 7).

**The bridge to stage 0 is rescored inside this run.** Stage 0's yes/no prompts for the aligned task, the swapped task and its mention sentence are, byte for byte, the with-block arm's aligned, swapped and name-assertion yes/no prompts. The bridge is therefore read within the run, on the same items, GPU type, stack and bootstrap draws: the plain-text bridge ratio against the with-block one, with the rule "plain text higher than the block format", "the same in both formats (within 0.25)" or "plain text lower" for their difference; at step 0, "plain text higher" is read as "stage 0's low step-0 share belongs to the block format". This costs nothing beyond the with-block arm. The archived stage-0 scores (original stack) are compared with their rescoring only descriptively, as differences in the *paired* effects (goal effect and mention effect, with intervals) and the mean per-score difference; the earlier 0.2-nat per-score stack difference is not assumed to carry over to paired effects.

SMOKE_RESULTS

## 12. The stated-reliability arm (E4) and screening off

**What it can test.** Two sentences assert the object's presence with different stated reliability: "The room's inventory list says there is a saltshaker in the room." and "A tester inserted this sentence at random: there is a saltshaker in the room." Evidence weighting predicts the reliable source lifts more; content-blind accounts predict no difference. The reading is three-way ("the reliable source lifts more", "no reliability effect beyond 0.25 of the goal effect", "the unreliable source lifts more"). The two carriers differ in wording and length, so a difference could also be a carrier effect; the arm has no control that separates the two.

**Screening off cannot be tested here.** The arm was meant to cross stated reliability with whether the history already settles the action's outcome. Every item in the pool is a determined failure: the history settles it. The collections hold 23 undetermined transitions that act on the goal object, in 10 games, 20 of them "examine" actions, which is not a usable comparison set. So the arm measures responsiveness to stated reliability only where it should not matter at all. A test of whether the shift shrinks when the history settles the outcome needs undetermined items (the direction-A review's E3); it is not attempted here.

**Qualitative, not Bayesian.** The arm can show at most qualitative responsiveness: whether the shift orders with stated reliability. Whether the update has the right size requires a known joint distribution of evidence and outcome and is reserved for a controlled Bayesian environment. The results will claim no more than the former.

## 13. Freeze plan

1. Review of this revision.
2. Analysis code: written (`wf/lexsem_analysis.py`), tested on planted scores (`tests/test_lexsem_analysis.py`).
3. Smoke records committed in section 11 before the full run (the second smoke, with every arm and the unbatched scorer, is recorded there).
4. Tag `lexsem-design-frozen-<date>` on the commit with this note, the builder, the rules, the scorer and the analysis.
5. Run the three arms from a fresh checkout at the tag, on L40S, in `serve2`; the analysis refuses scores from another commit or environment.
6. Deviations after the tag in a dated note committed before the affected scores are read.

## 14. Implementation

- `wf/lexsem.py`: frozen tables, carriers, conditions and controls, the control-object rule, scene prevalence, strata and sample, prompt builder.
- `wf/lexsem_rules.py`: zones, predictions, readings.
- `wf/lexsem_analysis.py`: estimands with absolute effects beside every ratio, the shared game bootstrap, gates (including the small-reference rule and the environment check), readings per readout, readout disagreements, normalization flags, item-level agreement, the prevalence covariate, the few-shot exclusion repeat, the step comparison, the in-run bridge and the archived-score comparison; optional arms read only when present.
- `scripts/lexsem/score.py`: loads each model with the study scorer's settings, records its environment, checks tokens and the sample digest, runs the recognition checks, checks token identity per item, scores one unpadded sequence at a time, appends one line per item.
- `scripts/lexsem/simulate_rules.py`: section 10.
- `pace/lexsem.sbatch` (three arms), `pace/lexsem_smoke.sbatch`.
- `tests/test_lexsem.py`, `tests/test_lexsem_analysis.py`.

## 15. Limits

- One environment; two models; Qwen in plain text with a yes/no bridge to its thinking-off format; Llama in yes/no only.
- The added sentences are artificial lines on a task card; controls absorb each carrier's own effect, not interactions between a carrier's oddity and the object.
- The listed word and the elsewhere sentence are free of literal claims about this room, not of pragmatic content.
- Controls are rarer in ALFWorld rooms than their targets (scene prevalence 0.19 against 0.48); reported as a covariate.
- Aliases exist for 8 of 27 types and homonyms for 9; neither decides alone. The homonym may activate the original concept.
- The position arm's contrasts are confounded with changes of prompt region.
- The reliability arm cannot test screening off (no unsettled items) and its two carriers differ in wording.
- Behavioural contrasts cannot separate staged computations from single mechanisms, different weights from different mechanisms, or an unlisted uniform mechanism from semantic activation.
- Components under about a quarter to a third of the lift are usually missed.
- No multiplicity correction; the 0.25 margins, the precision and reference gates and the simulation are the protection.

## 16. Earlier smoke run (batched scorer, superseded)

Job 7305613 (commit e6f6630, 5 items, L40S, `serve2`): all token checks passed; batching saved about 28% (0.088 s per sequence alone, 0.063 at batch 8) but changed readouts by up to 0.375 nats in plain text and 1.25 nats with the think block; first-token mass was lowest for the denial (0.754 against 0.885 for the swapped task, on 5 items); in plain text the tokens after " Nothing" carry −2.1 to −3.7 nats, so the full two-candidate score needs its own failure-text sequence. The user chose one-at-a-time scoring.

## 17. Review points and what was done

| Point | What was done |
|---|---|
| M1 quoted carrier changed the name's tokens | Carrier keeps the name's tokens; checked with both tokenizers; per-item run-time check; test refuses the old carrier |
| M2 mixtures read as single accounts | Level contrasts relative to the name assertion; simulated (section 10) |
| M3 unnormalized probability, full score | Unnormalized estimands, normalization flag, mass-gate fallback; full score from a failure-text sequence on every core cell |
| M4 recognition gate | Lenient reader; recognised items only; alias forced choice |
| M5 position | Three-position arm with controls (included by the user); confounds stated (section 5) |
| S1 precision, zone asymmetry | Bounded equal-width zones; precision and reference gates; procedure-level simulation |
| S2 bridge numerator | As stage 0 defined it; now rescored within the run |
| S3 controls not parallel | One absent object for both controls; audit table |
| S4 carriers not claim-free | Elsewhere sentence; broad evidence form needs the denial |
| S5 "Ignore the X" | Included |
| S6 stratum-specific rule | Needs contradiction, consistency and a step difference |
| S7 few-shot overlap | Reported apart; every reading repeated without them |
| S8 Llama | Included, yes/no; comparison on Qwen's yes/no line |
| S9 transfer from plain text | With-block arm included; conclusions per format |
| S10 batching | Measured; replaced by one-at-a-time scoring (user) |
| S11 minor | Done or stated |
| Interp M1 item-level agreement | Reported for every mention condition |
| Interp M2 homonym | Included, not a pure lexical isolate (section 5) |
| User qualification 1, identifiability | Section 2; rule wording changed |
| User qualification 2, statistics | Sections 6, 7, 10; absolute effects beside every ratio; small-reference rule; simulation grid |
| User qualification 3, controls | Section 5: audit, homonym wording, position confounds; prevalence covariate |
| User qualification 4, environment | Section 11: pinned and checked; bridge rescored in-run; archived comparison on paired effects |
| User qualification 5, reliability | Section 12: no unsettled items; qualitative only |
| User qualification 6, readouts | Section 4 and 9: separate reading lines, disagreement section, threats stated |
