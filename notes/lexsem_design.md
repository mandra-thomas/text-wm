# Lexical repetition, semantic identity, existence content and task relevance: a small diagnostic (design, revised 2026-11-25)

**Status: design, not frozen.** No effect has been scored. A 5-item smoke run measured timing and validity only (section 10). The interpretation rules in section 8 are to be reviewed again and committed at an annotated tag before any effect is scored (section 11). Exploratory, frozen procedure: the same standing as the mechanism battery and τ-auth.

**Revision.** The first version (064edb6) was reviewed twice on 2026-11-25. Section 14 lists every review point and what was done about it. In short: the quoted-word carrier no longer changes the name's tokens; the rules now read levels relative to the name assertion, so two-component mixtures are no longer read as single accounts (checked by simulation); the model's unnormalized probabilities are estimands, and the full two-candidate score is a priced addition (it is not free in plain text); the recognition checks were repaired; a homonym condition, an "elsewhere" existential, "Ignore the X" and a three-position arm were added (the last two as priced additions); one absent ALFWorld object now supplies both the name control and the description control; item-level agreement with the goal effect is reported; and the scorer batches an item's sequences (about 28% cheaper, measured).

## 1. The question

When the task line names the object an action acts on (the *aligned* task) instead of another object in the room (the *swapped* task), language-model world models predict more success for ALFWorld actions whose failure the history already determines. Call the mean within-item difference in log-odds the **goal effect**.

The interpretability pilot's stage 0 (run `interp_pilot_stage0`, Qwen3-8B, yes/no format, with Qwen3's empty think block) also scored the swapped task followed by "There is a saltshaker somewhere in the room.", naming the action's object without making it the goal. Read from its stored index, that sentence reproduces 2.26 of the 2.29-nat goal effect on the 3,861 later-step failures whose history does not name the object, but only 2.21 of 7.70 nats on the 514 step-0 items. Two caveats come with this. The empty think block inflates Qwen's effect about four-fold at step 0 and under two-fold later (`qwen_effect_audit_pipeline.md` §7.2). And the equality is one of means: item by item, the goal and mention effects correlate at only 0.45 on those later items.

That sentence, like every mention used so far ("Ignore the bowl.", the described goals of `rq1x`), also asserts or presupposes that the object exists. So the result cannot tell apart five explanations of the goal effect:

1. **Mention sufficiency** (descriptive): a non-goal sentence naming the object reproduces the effect.
2. **Literal lexical repetition**: the object's word (its token sequence) occurring earlier raises predicted success, whatever the sentence says and whatever sense the word has there.
3. **Semantic activation**: any reference to the object, by name or by meaning, raises it, whatever the sentence says.
4. **Task relevance**: the effect needs the object to matter for the task, as a goal or as a hint.
5. **Task as evidence**: the model reads what the line says about the object's existence in the room and predicts accordingly.

This diagnostic crosses the *form* of the reference (exact name, spaced spelling, alias, description, the same word in another sense, and matched controls) with its *existence content* (asserted, denied, placed elsewhere, merely listed), scores Qwen3-8B without the think block, and reads step 0 and later steps separately. It contains experiment E1 of the direction-A review (use-mention, described presence and surface-form lines without the think block) and its step-0 point; the review's E4 (stated reliability of the existence sentence) is a priced addition.

## 2. What each explanation predicts

Each mention condition's lift is read *relative to the lift of asserting the name* (section 6): 1 means "lifts as much as asserting the name", 0 "lifts nothing".

- **Lexical repetition** predicts that every sentence containing the exact name lifts fully, whether it asserts, denies, places the object elsewhere or merely lists the word; that the same token in another sense (the homonym) also lifts; and that sentences without the name (description, alias) lift nothing. A strict token-sequence version predicts that the spaced spelling ("salt shaker") lifts less than the closed name.
- **Semantic activation** predicts that every reference to the object lifts fully, whatever its form and whatever it claims; a merely listed word may activate the entity less. The homonym, which refers to something else, lifts nothing.
- **Task as evidence** predicts that what lifts is the claim that the object is in this room: an assertion lifts fully whatever its form (name, alias, description); a denial, a sentence placing it in the house next door, a listed word and the homonym lift nothing.
- **Task relevance** predicts that no non-goal sentence lifts. In a broader form (a sentence on the task card is read as a hint), "Ignore the X", which keeps the name and the existence presupposition but removes relevance, lifts nothing.

## 3. Model, format and readouts

**Model and format.** Qwen3-8B in plain text: no chat template, the study's own messages rendered by the Qwen audit's `plain_text` (`scripts/qwen_audit/followup_score.py`), the reply after one space. This is the format without the empty think block that the audit validated for the goal effect (3.26 against 3.28 nats in the no-block render; first-token mass 0.89; `qwen_effect_audit_pipeline.md` §4, §7.2 B). It has not been validated for the effects of individual sentences; every conclusion is therefore worded as one about Qwen3-8B in plain text, unless the with-block arm (section 10) is added.

**Readouts.** One sequence per item, condition and format.

- **Observation sequence**: the observation prompt followed by the item's success text. At the reply's first position it gives the log-probabilities of " You" and " Nothing", each over the full vocabulary. Along the success text it gives every token's log-probability.
- **Yes/no sequence**: `rq1v`'s anchored question; the log-probabilities of " Yes" and " No".

From these, per condition:

- **Primary: the first-token log-odds**, log p(" You") minus log p(" Nothing"). Neither token contains the object, so no output-side copying of the name can produce a difference; it predicts the environment's reply rather than answering a question, so it avoids the documented question-reading problem of yes/no formats (TextCraft round 1, τ-bench, Option C's "question-specific" step-0 collapse).
- **Unnormalized, beside every primary contrast**: the effect on log p(" You") alone, on log p(" Nothing") alone, and on the log-probability of the whole success text (the model's actual predictive probability of the success observation). A condition can raise the two-token log-odds by moving probability off " Nothing" onto other tokens without raising the probability of success; these show whether it does.
- **The full two-candidate score**: the success text's log-probability minus that of "Nothing happens.". In the battery's chat-format scores the tokens after " Nothing" carry 0.000 nats for Qwen, so it would have been free; the smoke run found that in plain text they carry −2.1 to −3.7 nats (section 10). The full score therefore needs one more sequence per condition and is a priced addition: on the anchors and the name assertion pair it fits the budget, on every cell it does not. Where scored, it is reported descriptively with the token split (verb, object tokens, rest), because its success text contains the object's name and so is the readout most exposed to output copying.
- **Secondary: the yes/no log-odds**, read with the same rules and reported beside the primary. It is the readout of the stage-0 result and of the battery's denial contrast.

## 4. Items and strata

**Population.** RQ2's 4,666 primary items (replay-verified determined failures; `rq2_results.md`), all with rq1x descriptions.

**Strata,** each with its own estimates and readings:

- **Step 0**: no action yet (514 items, 125 games).
- **Later**: at least one action, and the history after the task line does not name the action's object (3,861 items). The 291 later items whose history already names the object are left out, because the word is then already repeated.

**Sample.** 300 later and 150 step-0 items, drawn by `wf.lexsem.sample` (sorted ids shuffled with the seed `lexsem|2026-11-24|<stratum>`, at most three items per game later and two at step 0): 108 and 96 games. The sorted ids hash to `cc8da36181976edd`; the scorer refuses to run if the rebuilt sample differs.

**Few-shot overlap.** The prompt's few-shot examples name five ALFWorld objects (candle, handtowel, soapbar, soapbottle, spraybottle). In 56 sampled items (46 soapbar, 10 soapbottle) the action's object is one of them, so its exact name is already in context in every condition. Every reading is repeated without these 56 items as a sensitivity analysis, and they are reported apart.

**Matched working actions are not included.** Whether the naming effect is specific to failures is already answered (it is not; `exploratory_results.md` §1), and the 37 matched successes come from few games.

## 5. Conditions

A mention condition adds one sentence to the swapped task's prompt. By default the sentence follows the swapped task line on the same line, where stage 0 put its sentence; `name_assert` reproduces stage 0's prompt byte for byte. Nothing else differs between conditions (tested). Example: action `take saltshaker 1 from shelf 2`, swapped task "put some vase on cabinet", control object toiletpaper.

**Core conditions.**

| Condition | Reference and content | Added sentence | Items |
|---|---|---|---|
| aligned | anchor: the real task | (task line "put some saltshaker on cabinet.") | all |
| swapped | anchor: the swapped task | (none) | all |
| name, assertion | exact name; in this room | There is a saltshaker somewhere in the room. | all |
| name, denial | exact name; not in this room | There is no saltshaker in this room. | all |
| name, elsewhere | exact name; exists, but not in this room | There is a saltshaker somewhere in the house next door. | all |
| name, listed | exact name; no claim | The following word appears in this note: saltshaker. | all |
| description, assertion | rq1x's description; in this room | There is a container with holes for sprinkling white seasoning somewhere in the room. | all |
| description, listed | the description; no claim | The following phrase appears in this note: container with holes for sprinkling white seasoning. | all |
| spaced spelling, assertion | compound name as two words | There is a salt shaker somewhere in the room. | 111 compound items |
| alias, assertion | another word for the object | There is a skillet somewhere in the room. (for pan) | 122 alias items |
| homonym | the same token in an unrelated sense | Critics often pan sequels to popular films. (for pan) | 160 homonym items |
| controls | each name carrier with the control object's name; each description carrier with its description; the homonym sentence with a control word | There is a toiletpaper somewhere in the room. / … soft absorbent sheets on a spindle used in the bathroom … / Critics often praise sequels to popular films. | as their pairs |

**Priced additions** (section 10): "Ignore the saltshaker." with its control; the name assertion and its control at two more positions, as the last line of the history just before "Action:" and as a line just after the action line; and E4's reliability pair ("The room's inventory list says there is a saltshaker in the room." against "A tester inserted this sentence at random: there is a saltshaker in the room.").

**Tokens.** Every name carrier keeps the name's leading space and ends it at a word boundary, so the name has exactly the same tokens as in the action line. This was checked for all 55 object names in all five name carriers with both the Qwen3-8B and the Llama-3.1-8B tokenizers, and the scorer checks it for every item at run time: each name or homonym condition must add exactly one copy of the name's token sequence to the swapped prompt, and every other condition none, or the run stops (`item_jobs`; a test shows the first version's quoted carrier would be refused).

**Existence content.** The assertion says the object is in this room. The denial says it is not. The elsewhere sentence has the assertion's syntax and slot but places the object in another house, so it asserts existence without saying anything about this room. The listed word names the word without any claim. Neither of the last two is wholly free of pragmatic content (a word listed on a task card may be read as a hint), which is why the denial is required as well before evidence-reading is counted as contradicted in its broad form (section 8). The generic statement of the first version ("A saltshaker is a common household item.") was replaced by the elsewhere sentence, because a base-rate claim is evidence a Bayesian should use.

**Homonyms.** Nine of RQ2's 27 goal-object types have a natural sense with the identical space-prefixed token: mug (to rob), watch (to look at), pan (to criticise), plate (tectonic), book (to reserve), pencil (to schedule), cup (to cup one's hands), bowl (in cricket), egg (to egg on). Each has a hand-written sentence with no claim about any room and a control sentence in which one word of the same token count replaces the homonym (rob, play, praise, fault, find, fit, use, play, urge). Both tokenizers give the homonym the same token as in the action line, and the two sentences the same length. The homonym is the one condition whose lift is predicted only by lexical repetition.

**Controls.** Each item has one control object: an ALFWorld object type with an rq1x description, from a different coarse class than the action's object (food, kitchenware, cleaning, devices, personal, paper, decor, sports), not the goal or swapped object, absent from the room at reset, not named in the history, the action or the few-shot examples, with the same number of tokens as the action object's name, and among those the closest in description length (ties by item hash). Its name fills the name carriers' control and its description the description carriers' control, so name and description are controlled by the same absent household object (on the sample, 75% of description pairs have equal token counts, the largest gap is 3). The spelling and alias conditions use the name's control; their token counts differ from it by one or two tokens. Zipf frequency is no longer matched: it is undefined for the compound names.

**Position.** The name assertion is at the task line in the core. The position addition places it, with its own control at the same place, (a) as the last line of the history, more recent but still visible to the action's tokens, and (b) just after the action line, about as recent but no longer visible to them. Comparing (a) with the task line isolates recency with visibility kept; comparing (b) with (a) isolates visibility at roughly constant recency.

**Recognition checks.** Before scoring, every description used (control descriptions included) is put to rq1x's naming question with rq1x's decoding, and read by rq1x's reader plus a frozen lenient list ("bar of soap" for soapbar, "key ring" for keychain, "shaker" for the two shakers, "tumbler" for cup, "compact disc" for cd). The list was fixed after reading Qwen's rq1x answers; under it, Qwen's rq1x answers pass for 77% of later and 79% of step-0 sampled items (mug, knife, spatula, soapbottle, peppershaker and cloth fail). Each alias is checked by forced choice: "Can the word "skillet" refer to a pan? Answer Yes or No." must favour Yes, and the same question with a frozen foil object (pillow) must favour No. Four aliases occur inside their object's own description (disc, blade, loaf, cushion), so alias and description are not independent for those objects.

## 6. Estimands

For item $i$ and condition $c$, let $y_i(c)$ be a readout (the primary is the first-token log-odds). The goal effect in a stratum is

$$G = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(\text{aligned}) - y_i(\text{swapped})\big].$$

Each mention condition $c$ has a control $k(c)$. Its net effect is

$$\Delta_c = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(c) - y_i(k(c))\big].$$

The **mention lift** is the name assertion's net effect over the goal effect,

$$A = \frac{\Delta_{\text{name, assert}}}{G},$$

and every other mention condition is read by its **relative lift**, its net effect over the name assertion's on the same items,

$$\rho_c = \frac{\Delta_c}{\Delta_{\text{name, assert}}}.$$

The alias, spelling and homonym relative lifts use their subsets, with the name assertion on the same subset.

Further estimands, all from the same scores:

- **Unnormalized versions** of $A$ and every $\rho_c$, with $y$ replaced by log p(" You"), by log p(" Nothing") and by the success text's log-probability.
- **The bridge to stage 0**, as stage 0 defined it: the name assertion minus the plain swapped task, over $G$, in the yes/no readout; reported with the control-netted version beside it.
- **Item-level agreement** of each mention condition with the goal effect: the correlation across items between $y_i(c) - y_i(k(c))$ and $y_i(\text{aligned}) - y_i(\text{swapped})$, and the slope of the first on the second. Equal means with weak item tracking point to two routes that happen to move the mean by the same amount.
- **Carrier effects**: each control against the plain swapped task (descriptive).

**Intervals.** 95% percentile bootstrap over games (2,000 resamples, seed 20260924), recomputing every ratio from resampled sums. The basic bootstrap interval is computed as a sensitivity; a reading that changes under it is flagged "interval-sensitive".

## 7. Gates (per stratum and readout)

1. **Goal effect.** The interval of $G$ must lie above zero with its lower end at least half the estimate; otherwise nats only, no reading.
2. **Readout validity.** For each condition, the mean probability of " You" plus " Nothing" must be at least 0.7 and within 0.1 of the swapped condition's; for yes/no, " Yes" plus " No" at least 0.9. A condition that fails the first-token check is read on the unnormalized log p(" You") instead of the two-word log-odds and flagged "normalization-dependent" (amended after the smoke run, where the denial's mass was 0.13 below the swapped condition's on 5 items). A condition that fails the yes/no check is "readout invalid" in yes/no.
3. **A lift to decompose.** The relative lifts are read only if $A$'s lower bound is at least 0.5 (below that, ratios of ratios are unstable).
4. **Precision.** A relative lift whose interval half-width exceeds 0.25 is "imprecise" and counts as undecided.
5. **Recognition.** The description relative lifts are read on items whose own description passed its check (and, as a sensitivity, on all items); the alias relative lift on alias objects that passed the forced choice. If fewer than 100 later or 50 step-0 items remain, the contrast is "not assessable".

## 8. Interpretation rules (pre-committed; implemented in `wf/lexsem_rules.py`)

**Zones.** $A$ and every relative lift are placed on five zones: *below zero* (under −0.25), *about zero* (−0.25 to 0.25), *partial* (0.25 to 0.75), *full* (0.75 to 1.25) and *above* (over 1.25). The two zones the accounts predict, about zero and full, are bounded and equally wide, so no account's prediction is easier to confirm than another's (review S1). A contrast's **category** is the set of zones its 95% interval touches. An account is **consistent** with a contrast when every zone in the category is one it allows, **contradicted** when none is, and **undecided** otherwise.

**What each account allows, by relative lift.**

| Relative lift (in words) | Lexical repetition | Semantic activation | Task as evidence |
|---|---|---|---|
| Name listed, no claim | full | partial or full | about zero |
| Name in the house next door | full | full | about zero |
| Name denied | full | full | about zero or below zero |
| Description asserted | about zero | full | full |
| Description listed, no claim | about zero | partial or full | about zero |
| Alias asserted (alias items) | about zero | full | full |
| Homonym (homonym items) | partial or full | about zero | about zero |

An account is contradicted overall if any contrast contradicts it, except that the alias contrast never does so alone (aliases are rarer words than the names); it is consistent overall if every contrast is consistent; otherwise undecided.

**Single-account reading,** per stratum, from the primary readout:

- one account consistent and the others contradicted: "only X fits: no other component larger than about a quarter of the mention lift was detected";
- one account undecided and the others contradicted: "X not contradicted, not shown consistent";
- all contradicted: "no single account of the mention lift; read the mixture estimates";
- otherwise: "not distinguished between …", naming every account not contradicted, with no preference among them.

The evidence account is read in two forms. *Literal-claim evidence* is contradicted by any contrast as above. *Evidence including pragmatic cues* (a listed word read as a hint, the elsewhere sentence read loosely) is counted as contradicted only if the denial contradicts it as well. Both lines are reported.

**Mention sufficiency (explanation 1)** is read from $A$: "a non-goal assertion reproduces the goal effect" if its category lies within full and above; "reproduces part of it" if partial only; "does not reproduce it" if within about zero and below zero; otherwise inconclusive.

**Task relevance (explanation 4).** As a sole account, consistent if $A$ is about zero or below, contradicted if it touches neither. As a component: "the goal adds beyond a non-goal assertion" if $A$'s upper bound is below 0.75; "no goal-role component beyond a quarter of the goal effect" if its lower bound is at least 0.75. If "Ignore the X" is added, its relative lift is read against the broad, hint form of relevance, which allows only about zero or below zero; the three content accounts all allow full.

**Mixture estimates** (always reported, descriptive): least-squares weights of the three pure patterns in the table that best reproduce the observed relative lifts, with game-bootstrap intervals, plus the shares they imply (the name share, one minus the description's relative lift; the claim share, one minus the listed name's).

**Step 0 against later steps.** Every reading is made in each stratum. The step difference in $A$ (step 0 minus later, resampling games across both strata) is read "step 0 lower" if wholly below −0.25, "no step difference beyond 0.25" if wholly within ±0.25, "step 0 higher" if wholly above 0.25, otherwise inconclusive. An account is "stratum-specific" only if it is contradicted in one stratum and consistent in the other, and the step difference of a contrast that contradicts it lies outside ±0.25. Different readings that come only from different precision are not called stratum-specific.

**Position** (if added; descriptive, outside the account verdicts). The position-(a) net lift minus the task-line net lift, over $G$: "recency matters" if outside ±0.25, "no recency effect beyond 0.25" if within. The position-(b) net lift minus the position-(a) one: "visibility to the action's tokens matters" if wholly below −0.25, "no visibility effect beyond 0.25" if within ±0.25, otherwise inconclusive.

**Bridge to stage 0** (yes/no readout, the same 450 items). With the block, stage 0's ratio for this sentence on these items is 0.29 [0.23, 0.37] at step 0 and 0.92 [0.78, 1.05] later. The step-0 low share is read "specific to the block format" if the plain-text step-0 bridge ratio lies wholly above 0.5, "reproduced in plain text" if wholly below 0.5, otherwise inconclusive. Stage 0 ran in the study's original software stack; if the full run uses another stack (section 10), the bridge carries that difference too, and the with-block addition is the like-for-like comparison.

**Flags.** A contrast whose primary and yes/no categories share no zone is "readout-dependent". A contrast whose normalized and unnormalized (log p(" You")) categories share no zone is "normalization-dependent": the condition moved probability between " Nothing" and other tokens rather than toward " You". The primary readout decides every reading; flagged contrasts are listed with it.

**Item-level agreement** (descriptive): "tracks the goal effect item by item" if the correlation's lower bound is at least 0.7; "weak item tracking" if its upper bound is below 0.5; otherwise intermediate.

**Llama** (if added): the comparison between models uses the yes/no readout for both (Qwen's secondary, Llama's only), and states that formats also differ (Qwen in plain text, Llama in its chat template).

**What the readings do not cover.** They describe how predictions respond to sentences about the object; they do not identify an internal mechanism. "Only X fits" means that no other component larger than about a quarter of the mention lift was detected.

## 9. Precision and how the rules behave

**Expected precision.** For the goal effect in plain text, the audit's 300-item sample gives first-token effects of 1.31 nats later (item SD 1.10, standardized 1.19) and 1.65 at step 0 (38 items). The independent review measured, on these same 450 items in the battery's first-token scores with the block, 95% half-widths of 0.22 later and 0.10 at step 0 for differences involving four conditions, over $G$. Plain text should do better later (standardized 1.19 against 0.85) and worse at step 0 (the block inflates step 0 most). The planning values are half-widths of about 0.15 to 0.2 later and 0.1 to 0.15 at step 0 for the relative lifts; homonym, alias and spelling, on 111 to 160 items pooled over strata, about 0.2 to 0.25.

**Simulation of the rules.** `scripts/lexsem/simulate_rules.py` draws each relative lift around its planted value with a given half-width (independently; real errors are correlated, so this is approximate) and tallies the reading over 3,000 draws:

| Planted truth | Half-width 0 | 0.12 | 0.18 | 0.24 |
|---|---|---|---|---|
| Pure lexical, semantic or evidence | only that account fits (100%) | only that account 80%, not contradicted 20% | not contradicted about 97% | not contradicted about 95%, not assessable 5% |
| Half lexical, half evidence | no single account (100%) | no single account 100% | 99% | 86% |
| Half semantic, half evidence | no single account (100%) | 100% | 95% | 71% (semantic not contradicted 20%) |
| Half lexical, half semantic | no single account (100%) | 100% | 89% | 55% (one of the two not contradicted 40%) |
| 70% evidence, 30% semantic | no single account (100%) | evidence not contradicted 58% | 72% | 74% |
| 70% lexical, 30% semantic | no single account (100%) | lexical not contradicted 77% | 84% | 85% |
| 85% evidence, 15% semantic | only evidence fits | evidence not contradicted 98% | 99% | 95% |

Three things follow. A pure account is never contradicted at these precisions, and the three accounts behave alike (the first version's rules made "only evidence fits" much easier to reach than "only lexical fits"). An even mixture is read as "no single account" at the planned precision, the half-lexical, half-semantic one least reliably when intervals are wide. A 70/30 mixture is usually read as the majority account "not contradicted, not shown consistent": a residual component below about a third of the lift is often missed, which is why the wording says "no other component larger than about a quarter of the mention lift was detected" only when every contrast is consistent. At the planned precision "not distinguished" is rare (under 1% in every scenario), and "only X fits" needs half-widths near 0.12.

## 10. Cost and what fits in about 20 GPU-minutes

**Environment.** The study's original software stack (torch 2.6, transformers 4.51) lives inside the main checkout's folder on the cluster, which this work was asked not to touch. The smoke run used the newer `serve2` stack (torch 2.14, transformers 5.23), from a fresh checkout (`text-wm-lexsem`). Within one run every comparison is like-for-like whichever stack is used; only the bridge to the stored stage-0 scores crosses stacks (wave B measured about 0.2 nats per yes/no score between them). Which stack the full run uses is the user's decision.

**Smoke run** (job 7305613, commit e6f6630, one L40S, 5 items, 5 min 20 s of wall time for three model loads and three passes; timing and validity only, no effect computed). Outputs are in `runs/lexsem/smoke_*.json` of the `text-wm-lexsem` checkout.

- *Checks.* The frozen token counts matched both tokenizers, the readout tokens were single tokens in every mode, and the per-item token-identity check passed for every condition of every group (each name and homonym condition added exactly one copy of the name's tokens, every other condition none).
- *Throughput.* Qwen3-8B in plain text, 228 sequences averaging 605 tokens: 0.088 s per sequence one at a time, 0.063 with batches of 8, 0.066 with 16, 0.067 with 32. Batching saves about 28%, not several-fold: a single 600-token sequence already keeps the GPU busy. Qwen with the think block, yes/no only: 0.076 and 0.071 s (batch 1 and 16). Llama-3.1-8B, yes/no only: 0.071 and 0.066 s. Loading a model and starting the process costs about 1.3 minutes.
- *Batching changes readouts by bf16 rounding.* Against one-at-a-time scoring, the largest change in any first-token or yes/no log-odds was 0.375 nats in plain text (the same at batch 8, 16 and 32) and 1.25 nats in Qwen's think-block yes/no, whose logits are large. These are steps of bf16 logit rounding (0.125 to 0.25 nats at these magnitudes), the same kind of noise as the 0.13-nat batch noise the interpretability pilot measured. They add noise, not bias. Computing the output layer in float32 at the few positions read would remove the rounding at no measurable cost; that change is recommended before the freeze, with a repeat smoke.
- *Probability mass of the two readout words.* Yes/no: at least 0.998 in every condition (Qwen plain and with the block), 0.99 for Llama. First token, plain text: condition means from 0.75 to 0.94 (swapped 0.885), item minimum 0.52. The **denial** is the lowest (0.754, 0.13 below the swapped condition), so at scale it may fail gate 2; the elsewhere sentence (0.839) and the two moved positions (0.836 and 0.864) sit close to the 0.1 margin. This is the case M3 anticipated: a sentence that moves probability off " Nothing" onto other words. Gate 2 is therefore amended: a condition that fails it is read on the unnormalized log p(" You") instead of the two-word log-odds, flagged "normalization-dependent", rather than dropped.
- *"Nothing happens." after " Nothing".* In plain text the tokens after " Nothing" carry −2.1 to −3.7 nats (10 prompts), not about 0 as in the battery's chat format. So the full two-candidate score is not free in plain text: it needs a third sequence per condition (the failure text). It is a priced addition below; without it the full score is not reported, and the success text's own log-probability (free) remains the unnormalized readout.

**Cost, at the measured rates on one L40S.** The core is 450 items with 14 conditions on all items plus the spelling, alias and homonym conditions on their subsets: 6,853 item-condition cells, two sequences each, 13,706 sequences.

| Run | Sequences | Batch 8 (0.063 s) | One at a time (0.088 s) |
|---|---|---|---|
| Core, Qwen3-8B plain text, both readouts | 13,706 | about 14.4 min + 1.3 load + 0.5 checks ≈ **16 min** | about 20.1 min + 1.8 ≈ **22 min** |

| Addition | Sequences | Extra GPU-minutes (batch 8) |
|---|---|---|
| "Ignore the X" with its control (S5) | 1,800 | about 1.9 |
| E4, stated reliability of the assertion | 1,800 | about 1.9 |
| Failure-text pass for the full two-candidate score, on the anchors and the name assertion pair only | 1,800 | about 1.9 |
| The same on every core cell | 6,853 | about 7.2 |
| Position: name assertion and control at the history's end and after the action line (M5) | 3,600 | about 3.8 |
| With-block yes/no arm, eight conditions (anchors; name and control assertion, denial and listed word) (S9) | 3,600 | about 4.3 + 1.3 load ≈ 5.6 |
| With-block yes/no arm, twelve conditions (adds elsewhere and the description pair) | 5,400 | about 6.4 + 1.3 ≈ 7.7 |
| Llama-3.1-8B reduced arm, yes/no, twelve conditions (S8) | 5,400 | about 5.9 + 1.3 ≈ 7.2 |

**What fits in about 20 GPU-minutes, in priority order.**

1. The core with batches of 8: about 16 minutes. Homonyms (the interpretability review's unique lexical contrast) are in the core and cost about 0.7 minutes of it.
2. "Ignore the X" with its control: about 18 minutes in all.
3. The failure-text pass on the anchors and the name assertion pair: about 20 minutes in all. It gives the study's headline readout for the anchors and the stage-0 sentence.

**Beyond about 20 minutes, for the user to decide**, each priced on top of the 20:

- position, about +3.8 (the user asked for position to be distinguished; without it the design states position is out of scope);
- the with-block yes/no arm, about +5.6 (eight conditions) or +7.7 (twelve);
- the Llama reduced arm, about +7.2;
- E4, about +1.9;
- the full two-candidate score on every core cell, about +5.3 beyond item 3.

If scoring one sequence at a time is preferred over batches (no bf16 rounding differences between batched and unbatched scores), every figure grows by about 40% and the core alone is about 22 minutes. With the float32 output layer recommended above, batching's rounding differences should largely disappear.

## 11. Freeze plan

1. **Review** of this revision.
2. **Analysis code before the tag**: load the scores, compute section 6's estimands with the game bootstrap, and apply `wf/lexsem_rules.py`. Test it on simulated item-level scores with the planted scenarios of section 9 (pure accounts, the even mixtures, 70/30 mixtures), with noise at the planned precision, and report how often each is misread. Run the repository's mutation check on the rules and the analysis.
3. **Smoke figures** are committed in section 10 before the full run (done for timing and validity; repeated if the code changes).
4. **Tag** `lexsem-design-frozen-<date>` on the commit with this note, the builder, the rules, the scorer and the analysis; the tag message states the sample digest and the run's condition groups and stack.
5. **Run** from the tagged commit; the analysis refuses scores from any other commit.
6. **Deviations** after the tag go in a dated note committed before the affected scores are read.

## 12. Implementation

- `wf/lexsem.py`: frozen tables (spellings, aliases and foils, homonym sentences and control words, name token counts, object classes, few-shot objects, lenient reader), carrier sentences, conditions and controls, the control-object rule, strata and the seeded game-capped sample, and the prompt builder (task line; history end or after-action position; rq1v's yes/no question).
- `wf/lexsem_rules.py`: zones, account predictions, single-account and other readings, mixture weights, planted scenarios.
- `scripts/lexsem/score.py`: loads the model with the study scorer's settings, checks frozen token counts, readout tokens and the sample digest, runs the recognition checks, checks token identity per item, scores each item's sequences in right-padded batches (only the needed positions go through the output layer) and appends one line per item; `--smoke` writes timing and validity figures only.
- `scripts/lexsem/simulate_rules.py`: section 9's simulation.
- `pace/lexsem.sbatch`, `pace/lexsem_smoke.sbatch`: job scripts (L40S only).
- `tests/test_lexsem.py`: builder, token-identity check (including refusal of the old quoted carrier), control-object rules, frozen tables, lenient reader, sampling, and the rules on planted scenarios (including the two mixtures the first rules misread).

## 13. Limits

- One model in the core, one environment, one prompt format, and (unless the with-block addition is run) no direct check that sentence effects transfer from plain text to the study's thinking-off format.
- The added sentences are artificial lines on a task card; matched controls absorb each carrier's own effect, not an interaction between a carrier's oddity and the object.
- The listed word and the elsewhere sentence are free of literal claims about this room, not of pragmatic content; the denial carries the broad form of the evidence test.
- Aliases exist for 8 of 27 object types and homonyms for 9; both are read on subsets with wider intervals, and the alias never decides a reading alone.
- Behavioural contrasts cannot tell "different mechanisms" from "the same mechanisms with different weights"; the mixture weights describe, they do not identify. The rules miss residual components below about a quarter to a third of the lift.
- The simulation treats contrasts as independent; their errors are correlated through shared conditions.
- Intervals are not corrected for multiplicity; the 0.25 margins, the precision gate and the requirement to read patterns across contrasts are the protection, and the simulation reports how often a pure account would be wrongly contradicted (never, at the planned precision).

## 14. Review points and what was done

| Point | What was done |
|---|---|
| M1, quoted carrier changes the name's tokens | Carrier "The following word appears in this note: X." keeps the name's tokens; checked for all names with both tokenizers; per-item run-time check; test refuses the old carrier |
| M2, mixtures read as single accounts | Rules rewritten as levels relative to the name assertion (listed name, elsewhere, description, homonym levels); both failing mixtures now read "no single account"; simulation with noise in section 9; wording "no other component larger than about a quarter" |
| M3, unnormalized probability and full score | log p(" You"), log p(" Nothing") and the success text's log-probability are estimands; normalization-dependence flag and fallback (gate 2). The smoke run found the failure text's tail is not about 0 in plain text (−2.1 to −3.7 nats), so the full score needs a third sequence: priced, and inside the 20 minutes for the anchors and the name assertion pair |
| M4, recognition gate | Lenient reader frozen; descriptions read on passing items (77–79% expected); alias forced choice with a foil |
| M5, position | Three-position arm with controls defined and priced; inclusion is the user's decision (section 10) |
| S1, precision and zone asymmetry | Bounded equal-width predicted zones; precision gate; $A \geq 0.5$ gate; relative lifts; expected outcome rates simulated |
| S2, bridge numerator | Bridge defined as stage 0 did (minus the swapped task); control-netted version beside it |
| S3, controls not parallel | One absent ALFWorld object of another class supplies both the name and the description control |
| S4, carriers not claim-free | Generic statement replaced by the elsewhere existential; denial required for the broad evidence form |
| S5, "Ignore the X" | Defined with its control and read against the hint form of relevance; priced addition |
| S6, stratum-specific rule | Needs contradiction in one stratum, consistency in the other, and a step difference outside ±0.25 |
| S7, few-shot overlap | 56 items reported apart; every reading repeated without them |
| S8, Llama | Reduced arm uses yes/no for both models in the comparison; priced |
| S9, transfer from plain text | Conclusions worded for plain text; with-block yes/no addition priced |
| S10, batching | Implemented and measured: about 28% cheaper at batch 8, not several-fold; bf16 rounding differences up to 0.375 nats (plain) against one-at-a-time scoring; float32 output layer recommended |
| S11, minor | Alias and spelling control token gaps stated; Zipf matching dropped; no "an ukulele" (control pool replaced); multiplicity addressed by simulation |
| Interp M1, item-level agreement | Correlation and slope of each mention effect with the goal effect, with intervals |
| Interp M2, homonym | Nine object types, 160 sampled items, hand-written matched sentences, token identity checked |
| Interp, residual capture | Not added; it would have to be committed before the freeze |
