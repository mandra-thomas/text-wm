# Lexical repetition, semantic identity, existence content and task relevance: a small diagnostic (design, 2026-11-24)

**Status: design, not frozen.** Nothing in this note has been scored. The interpretation rules in section 8 are to be reviewed and then committed at an annotated tag before any score exists (section 11). Exploratory, frozen procedure: the same standing as the mechanism battery and τ-auth.

## 1. The question

When the task line names the object an action acts on (the *aligned* task) instead of another object in the room (the *swapped* task), language-model world models predict more success for ALFWorld actions whose failure the history already determines. Call the mean within-item difference in log-odds the **goal effect**.

The interpretability pilot's stage 0 (run `interp_pilot_stage0`, Qwen3-8B, yes/no format, with Qwen3's empty think block) also scored the swapped task followed by the sentence "There is a saltshaker somewhere in the room.", naming the action's object without making it the goal. Read from its stored index, that sentence reproduces 2.26 of the 2.29-nat goal effect on the 3,861 later-step failures whose history does not name the object (99%), but only 2.21 of 7.70 nats on the 514 step-0 items (29%). The step-0 figures are suspect, because the empty think block inflates Qwen's effect about four-fold at step 0 and under two-fold later (`qwen_effect_audit_pipeline.md` §7.2).

That sentence, like every mention used so far ("Ignore the bowl.", the described goals of `rq1x`), also asserts or presupposes that the object exists. So the result cannot tell apart five explanations of the goal effect:

1. **Mention sufficiency** (descriptive): a non-goal sentence naming the object reproduces the effect.
2. **Literal lexical repetition**: the object's word occurring earlier in context raises predicted success, whatever the sentence says.
3. **Semantic activation**: any reference to the object, by name or by meaning, raises it, whatever the sentence says.
4. **Task relevance**: the effect needs the object to matter for the task (a goal role).
5. **Task as evidence**: the model reads what the line says about the object's existence and predicts accordingly.

This diagnostic crosses the *form* of the reference (exact name, spaced spelling, alias, description, unrelated control) with its *existence content* (asserted, denied, a generic statement with no claim about the room, a quoted word with no claim), scores Qwen3-8B without the think block, and reads step 0 and later steps separately. It folds in experiment E1 of the direction-A review (use-mention, described presence and surface-form lines without the think block) and its step-0 point; the review's E4 (stated reliability of the existence sentence) is priced as an option (section 10).

## 2. What each explanation predicts

Explanation 1 is a statement about one contrast and is read directly from it. The other four predict patterns across conditions:

- **Lexical repetition** predicts that every sentence containing the exact name lifts about equally, whether it asserts, denies, quotes or generalizes, and that sentences without the name (description, alias) lift about nothing. A strict token-sequence version also predicts that the spaced spelling ("salt shaker") lifts less than the closed name ("saltshaker").
- **Semantic activation** predicts that every reference to the object lifts about equally, whatever its form (name, spelling, alias, description) and whatever it claims. A quoted word might activate the entity less than a used one, so the quoted conditions may be lower.
- **Task as evidence** predicts that what lifts is the existence claim: an assertion lifts whatever its form (name, alias, description); a denial does not lift (or lowers); a quoted word and a generic statement about such objects lift little.
- **Task relevance** predicts that no non-goal sentence lifts: only the aligned goal does.

## 3. Model, format and readouts

**Model and format.** Qwen3-8B in plain text: no chat template, the study's own messages rendered by the Qwen audit's `plain_text` (`scripts/qwen_audit/followup_score.py`), the reply after one space. This is the format without the empty think block that the audit validated (first-token mass 0.89 on its 300-item sample; same goal effect as the no-block render, 3.26 against 3.28 nats; `qwen_effect_audit_pipeline.md` §4, §7.2 B). The block is not used, because it inflates the scale by different factors at step 0 and later, which is exactly what the step comparison must not depend on.

**Readouts.** Two forward passes per item and condition.

- **Primary: the first token of the predicted observation**, " You" against " Nothing", read at the reply's first position. Neither token contains the object, so no output-side copying of the name can produce a difference. It predicts the environment's reply rather than answering a question, so it avoids the documented question-reading problem of yes/no formats (the yes/no question read partly as task success in TextCraft round 1 and τ-bench, and the step-0 rejection collapse that Option C found to be "question-specific"). Relevance-type effects that act through how a question is read would otherwise inflate the goal effect and its role component in yes/no but not in a predicted observation. On the audit's 300 plain-text items its goal effect is 1.31 nats later (standardized 1.19) and 1.65 at step 0 (38 items; standardized 1.50).
- **Secondary: the anchored yes/no question** of `rq1v` (" Yes" against " No"). It also contains no object, it is the readout of the stage-0 mention result and of the battery's denial contrast, and the same items were scored in it with the block at stage 0, so it bridges to the existing result. It is read with the same rules and reported beside the primary.
- **Descriptive: the success text's continuation.** The first-token pass is run on the prompt followed by the item's success text, so the same pass gives each token's log-probability. Splitting the change into the verb, the object's tokens and the rest (as `scripts/qwen_audit/token_split.py` does) shows whether naming the object raises its tokens in the predicted reply more than describing it does: a direct check of output copying. No verdict reads it.

**Not scored: the full two-candidate log-odds** (success text against "Nothing happens."). Its success text contains the object's name, so it is the readout most exposed to output copying in exactly the comparison this study makes (exact name against other forms), and 59% of Qwen's headline nats sit in its continuation. It would also need a third forward pass per cell for the failure text. It is priced as an option.

## 4. Items and strata

**Population.** RQ2's 4,666 primary items (replay-verified determined failures; `rq2_results.md`), all of which have rq1x descriptions for both objects.

**Strata,** each with its own estimates and readings:

- **Step 0**: the agent has not acted yet (514 items, 125 games).
- **Later**: at least one action, and the history after the task line does not name the action's object (3,861 items). The 291 later items whose history already names the object are left out, because the word is then already repeated in context; they would mix the repetition being tested with one already present.

**Sample.** 300 later items and 150 step-0 items, drawn by `wf.lexsem.sample`: within each stratum the items are sorted by id, shuffled with the seed `lexsem|2026-11-24|<stratum>`, and taken in order with at most three items per game (later) or two (step 0). The draw gives 300 items from 108 games and 150 from 96 games; its sorted ids hash to `cc8da36181976edd` (first 16 hex digits of SHA-256), and the scoring script refuses to run if the rebuilt sample differs. The per-game cap spreads the sample over games, which raises the effective sample size under game clustering.

Of the 450 items, 111 have a compound object name (and so a spaced-spelling condition) and 122 have an object with a frozen alias.

**Matched working actions are not included.** Whether the naming effect is specific to failures is already answered (it is not: `exploratory_results.md` §1), and the 37 matched successes come from few games, so they would add cost and wide intervals. They are priced as an option.

## 5. Conditions

Every condition except the two anchors is the swapped task line followed, on the same line, by one sentence. That is exactly where stage 0 put its mention sentence (`rq1.goal_line(t, "swapped_plus")`), and the `name_assert` condition reproduces stage 0's sentence byte for byte. Nothing else in the prompt differs between conditions (tested in `tests/test_lexsem.py`). With the stage-0 example (action `take saltshaker 1 from shelf 2`; swapped task "put some vase on cabinet"):

| Condition | What the added sentence does | Example (added sentence) | Items |
|---|---|---|---|
| aligned | anchor: the real task, no sentence | (task line "put some saltshaker on cabinet.") | all |
| swapped | anchor: the swapped task, no sentence | (none) | all |
| name, assertion | exact name; asserts existence in the room | There is a saltshaker somewhere in the room. | all |
| name, denial | exact name; denies existence in the room | There is no saltshaker in this room. | all |
| name, quoted word | exact name; no claim about the room | The word "saltshaker" appears in this note. | all |
| name, generic | exact name used as a word with meaning; no claim about this room | A saltshaker is a common household item. | all |
| control, assertion | matched unrelated word in the same sentence | There is a snowglobe somewhere in the room. | all |
| control, denial | same | There is no snowglobe in this room. | all |
| control, quoted word | same | The word "snowglobe" appears in this note. | all |
| control, generic | same | A snowglobe is a common household item. | all |
| description, assertion | rq1x's object-free description; asserts existence | There is a container with holes for sprinkling white seasoning somewhere in the room. | all |
| description, quoted | the description, quoted; no claim | The phrase "container with holes for sprinkling white seasoning" appears in this note. | all |
| control description, assertion | the description of another object type absent from the room | There is a long-handled deep scoop for serving soup somewhere in the room. | all |
| control description, quoted | same, quoted | The phrase "long-handled deep scoop for serving soup" appears in this note. | all |
| spaced spelling, assertion | the compound name written as two words | There is a salt shaker somewhere in the room. | 111 compound items |
| alias, assertion | a different word for the same object | There is a skillet somewhere in the room. (for pan) | 122 alias items |

**The existence factor.** The assertion is evidence that the object is in the room. The denial is evidence that it is not. The quoted word mentions the word without saying anything about the room (the use-mention distinction). The generic sentence uses the word with its meaning but makes no claim about this room; it carries at most a weak base-rate suggestion that such objects are common, which is why the quoted word is included as the cleaner zero-claim condition. "Ignore the X" is not in the core: it presupposes existence, so it predicts the same as the assertion under explanations 2, 3 and 5 and cannot separate them, and it was already measured with the block (battery P3). It is priced as an option, as the bridge to P3.

**Controls for the reference.** Each mention condition has a control with the same carrier sentence, so position, recency, syntax and carrier length are matched within each pair:

- **Control word** (for the name, spelling and alias conditions): an unrelated household-plausible noun from a frozen pool (tools, music, toys, outdoor gear, clothing; never an ALFWorld object or receptacle; nothing from kitchens, bathrooms, cleaning, writing or timekeeping, which relate to RQ2's objects; no word containing an object's name or contained in one). It has the same number of Qwen3 tokens as the object's name in a sentence, and it is drawn by a hash of the item id from the three pool words closest in Zipf frequency (wordfreq, English). Drawing among three spreads the control over several words. Token counts and frequencies are frozen in `wf/lexsem.py`; the scoring script checks every count against the loaded tokenizer and refuses to run on a mismatch. Llama-3.1-8B's tokenizer gives identical counts for every name and pool word.
- **Control description** (for the description conditions): the rq1x description of another object type that is not the goal or swapped object, not in the room at reset, and not named in the history or action, chosen as the closest in Qwen3 token count to the item's own description. On the sample, 98% of pairs have equal token counts and the largest gap is 2 tokens.

**Controls the factors themselves need.**

- *Tokenization.* Seven of the 27 object types have multi-token compound names. The spaced spelling changes the token sequence while keeping the word: under Qwen3's tokenizer "salt shaker" shares two of its three tokens with "saltshaker", "soap bar" one of two with "soapbar", and "toilet paper" none with "toiletpaper". The overlap is reported per type; the spelling contrast is graded by it. Results are also reported separately for single-token and multi-token names (descriptive).
- *Frequency.* Matched for the control word as above. Aliases are less frequent than names (for example "skillet" Zipf 3.08 against "pan" 4.39), so an alias drop could be a frequency effect; the alias contrast is therefore never the only basis for a reading (section 8).
- *Prompt length.* Matched within each mention–control pair (names and control words by token count; descriptions by closest token count). Lengths differ across carriers, which is why every contrast between carriers is taken net of each carrier's own control.
- *Recency and position.* Every sentence sits at the same place (directly after the swapped task line), so recency is matched across all mention conditions.
- *Syntactic role.* Each carrier fixes the reference's slot (object of "There is", complement of "The word", subject of the generic sentence); the controls fill the same slot.
- *Evidence content.* This is the existence factor itself.

**Position after the action line is not varied.** A sentence placed after "Action: …" changes two things at once: it is closer to the answer, and it can no longer be seen by the action's own tokens, because a causal model's earlier tokens cannot attend to later ones. The pilot's patching shows that the effect normally travels through the action's object word (`interp_results.md`), so moving the mention after the action blocks that route for every one of explanations 2, 3 and 5 alike, while an unrelated word at the same position would control recency but not route. The outcome would locate the computation, not separate the explanations; moving the task line further away also changed the goal effect little in the battery (P6). It is priced as an option, with its own control word at the same position.

**Manipulation check.** Each alias and description used (control descriptions included) is put to rq1x's naming question ("Which household object is described here: "…"? Answer with just the name of the object.") with rq1x's reader, in the model's own chat template, before scoring. About 90 short generations.

## 6. Estimands

For item $i$ and condition $c$, let $y_i(c)$ be the readout's log-odds (" You" against " Nothing" for the primary readout). The goal effect in a stratum is

$$G = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(\text{aligned}) - y_i(\text{swapped})\big].$$

Each mention condition $c$ has a control $k(c)$ (section 5). Its net effect is

$$\Delta_c = \frac{1}{n}\sum_{i=1}^{n}\big[y_i(c) - y_i(k(c))\big],$$

and its ratio to the goal effect is

$$r_c = \frac{\Delta_c}{G}.$$

A ratio of 1 means the sentence moves the readout as much as making the object the goal. Comparisons between two conditions $c$ and $c'$ on the same items are differences of ratios,

$$d_{c,c'} = r_c - r_{c'} = \frac{\Delta_c - \Delta_{c'}}{G}.$$

The spelling and alias contrasts are computed on their subsets, with the name assertion on the same subset. Each carrier's own effect against the plain swapped task, $\frac{1}{n}\sum_i [y_i(k) - y_i(\text{swapped})]$, is reported descriptively.

**Intervals.** 95% percentile bootstrap over games (2,000 resamples, seed 20260924), resampling whole games and recomputing every ratio from the resampled sums, as in the earlier studies. As a sensitivity, the basic bootstrap interval is also computed; a reading that changes under it is flagged "interval-sensitive" (the percentile reading is kept). No multiplicity correction is applied: as in the mechanism battery, error control comes from the bands below, which require an interval to clear a 0.25 margin, and from reading patterns across several contrasts rather than single tests.

## 7. Gates (per stratum and readout, before any contrast is read)

1. **Goal effect.** The interval of $G$ must lie above zero and its lower end must be at least half its estimate. Otherwise ratios are unstable; the stratum reports nats only and no reading.
2. **Readout validity.** For the first-token readout, the mean probability mass of " You" and " Nothing" must be at least 0.7 in every condition and within 0.1 of the swapped condition's. For yes/no, the mass of " Yes" and " No" must be at least 0.9. A condition that fails is "readout invalid" in that readout, and so is every contrast it enters.
3. **A lift to decompose.** The form and content contrasts (section 8, B to H) are read only if the name assertion's ratio has a lower bound of at least 0.25. Otherwise they are "not assessable: no mention lift to decompose", and only contrast A and the relevance reading apply.
4. **Recognition.** The description and alias contrasts are read on all their items and, as a sensitivity, on items whose form passed the naming check. If fewer than 70% of a contrast's items have a passing form, a "lower" result on it is flagged "recognition weak" and does not count against semantic activation or task as evidence (the drop could be a failure to identify the object).

## 8. Contrasts and the pre-committed interpretation rules

**Zones.** A ratio's interval is placed on four zones: *below zero* (under −0.25), *about zero* (−0.25 to 0.25), *partial* (0.25 to 0.75) and *full* (0.75 or more). A difference of ratios is placed on three: *lower* (under −0.25), *same* (−0.25 to 0.25) and *higher* (over 0.25). A contrast's **category** is the set of zones its 95% interval touches. An explanation is **consistent** with a contrast when every zone in the category is one it allows, **contradicted** when none is, and **undecided** otherwise. This rule is symmetric: a wide interval yields "undecided", never a substantive reading.

**The contrasts and what each explanation allows.** Contrast A is a ratio; B to H are differences or ratios as stated.

| Contrast (in words) | Lexical repetition allows | Semantic activation allows | Task as evidence allows |
|---|---|---|---|
| A. Name with an existence assertion, against its control, over the goal effect | partial or full | partial or full | partial or full |
| B. Generic statement against assertion, name held | same | same | lower |
| C. Quoted word against assertion, name held | same | same or lower | lower |
| D. Denial against assertion, name held | same | same | lower |
| E. Denial alone, over the goal effect | partial or full | partial or full | below zero or about zero |
| F. Description against name, assertion held | lower | same | same |
| G. Alias against name, assertion held (alias items) | lower | same | same |
| H. Quoted description against asserted description | same | same or lower | lower |
| I. Spaced spelling against name, assertion held (compound items) | lower or same | same | same |

**Single-account readings, per stratum, from the primary readout.** Each of the three accounts in the table is consistent, contradicted or undecided on each assessable contrast; it is *contradicted* overall if contradicted on any contrast, *consistent* overall if consistent on all, and *undecided* otherwise. Then:

- exactly one account consistent and the other two contradicted: "only [account] fits as a single account of the mention lift";
- all three contradicted: "no single account of the mention lift; read the mixture estimates";
- anything else: "not distinguished between [every account not contradicted]". No preference among them is stated, and the most plausible one is not singled out.

Contrast G (alias) never decides a reading alone: if it is the only contrast contradicting an account, that account is reported as "undecided (alias only)", because aliases are less frequent than names (section 5). Contrast I splits lexical repetition descriptively: "lower" reads as "token-sequence repetition", "same" as "word-level repetition"; it contradicts semantic activation and task as evidence only if "lower".

**Mention sufficiency (explanation 1)** is read from contrast A alone: "a non-goal assertion reproduces the goal effect" if A's category is "full"; "reproduces part of it" if "partial"; "does not reproduce it" if "about zero" or "below zero"; otherwise "inconclusive". The reading is about this sentence, not about mention in general.

**Task relevance (explanation 4).** As a single account it is consistent if contrast A is "about zero" or "below zero" and contradicted if A is "partial" or "full". As a component, the goal role beyond a non-goal assertion is the shortfall of contrast A from 1 (one minus A's ratio, with A's interval): "the goal adds beyond the assertion" if A's upper bound is below 0.75; "no goal-role component beyond 0.25" if A's lower bound is at least 0.75.

**Mixture estimates (always reported, read only under gate 3).** The *claim share* is the fraction of the assertion's lift that needs the claim, one minus the generic statement's ratio over the assertion's ratio (and the same with the quoted word). The *name share* is the fraction that needs the exact name, one minus the description's ratio over the name's. The *name-by-claim interaction* is the difference between the claim's effect with the name and with the description:

$$I = \big(r_{\text{name, assert}} - r_{\text{name, quoted}}\big) - \big(r_{\text{desc, assert}} - r_{\text{desc, quoted}}\big).$$

It is read "additive" if its interval lies within ±0.25, "the claim matters more with the name" if above, "less" if below, and "inconclusive" otherwise. A lexical-plus-evidence mixture would show a name share and a claim share both between 0.25 and 0.75 with an additive interaction.

**Step 0 against later steps.** Both strata get every reading above. In addition, the step difference in contrast A, the step-0 ratio minus the later ratio (one bootstrap resampling games across both strata), is read "step 0 lower" if its interval lies wholly below −0.25, "no step difference beyond 0.25" if wholly within ±0.25, "step 0 higher" if wholly above 0.25, and "inconclusive" otherwise. If the single-account readings differ between strata, the result is "stratum-specific"; it is not pooled.

**Bridge to the stage-0 result (secondary readout, same items).** With the block, stage 0's yes/no ratio for the same sentence on these items is 0.29 [0.23, 0.37] at step 0 and 0.92 [0.78, 1.05] later. The step-0 low share is read "specific to the block format" if the plain-text yes/no step-0 ratio for the name assertion has its interval wholly above 0.5, "reproduced without the block" if wholly below 0.5, and "inconclusive" otherwise. The same rule is applied to the primary readout and reported beside it.

**Readout dependence.** The primary readout decides every reading. The yes/no readout is read with the same rules and reported in a parallel column. A contrast whose two categories share no zone is flagged "readout-dependent"; the battery's denial contrast (ratio 0.97 in yes/no against 0.23 in two-candidate, Qwen later steps, with the block) is the known case.

**Expectations, not rules.** From the block-format data: later-step assertion ratio near 1 in yes/no; later-step denial near 1 in yes/no and near 0.2 in two-candidate; step 0 lower in both. A first-token readout without the block has no precedent for any mention sentence. No reading depends on these.

**What the readings do not cover.** They describe how the model's prediction responds to sentences about the object; they do not identify an internal mechanism, and "consistent" means "not contradicted by these contrasts", not "shown". A mixture can produce any pattern between its parts (the battery's simulation read a 50/50 evidence-plus-salience mixture as evidence alone), which is why the mixture estimates are reported in every case.

## 9. Sample size

The precision was set from the stored stage-0 yes/no scores (with the block) on exactly the sampled items: the assertion's ratio has an interval of [0.78, 1.05] later (half-width 0.13, 300 items in 108 games) and [0.23, 0.37] at step 0 (half-width 0.07, 150 items in 96 games). Across the full later stratum, subsampling gives ratio-interval widths of 0.43, 0.30, 0.25 and 0.22 for 100, 200, 300 and 400 items, so 300 is where the width falls to about half a zone (the partial zone is 0.5 wide) and further items buy little. For the plain-text first-token readout the standardized goal effect is at least as large (1.19 later and 1.50 at step 0 on the audit's sample, against 1.01 and 1.47 for the block yes/no), so its ratio intervals should be no wider. At step 0 the plain-text effect is smaller relative to its spread than with the block (standardized 0.89 against 1.45 in two-candidate), so its intervals may be about 1.5 times as wide as the block's, still about ±0.1 with 150 items. The spelling and alias contrasts use 111 and 122 items pooled over strata; as paired differences with the name assertion on the same items, their intervals are expected near ±0.2, enough to separate "lower" from "same" only for large differences, which is why neither decides a reading alone.

## 10. Cost, models and options

**Throughput, from recorded runs.** The `rq1v_score` shards of 2026-11-19 ran the study's unbatched transformers scorer, two forward passes per score, both 8B models in one job. On the seven L40S shards each forward pass took 0.078 to 0.082 seconds of wall time including model loading (7,048 to 10,562 scores in 19.0 to 28.1 minutes); the one A40 shard took 0.157 seconds. The Qwen audit's plain-text 8B run on an A40 took about 4 minutes for 1,200 forward passes including loading. The pilot's stage 0 scored 28,872 batched rows with activation capture in about 41 minutes on an L40S. Forward passes here are a single unbatched prompt of about 600 tokens, as in `rq1v_score`, so 0.08 seconds per pass on an L40S is the planning rate. The job script requests an L40S only; on an A40 the same run would take about twice as long.

**Qwen3-8B core.** 450 items times 14 conditions plus 111 spelling and 122 alias cells gives 6,533 item-condition cells, two forward passes each: 13,066 passes, about 17.4 minutes, plus about a minute to load the model and half a minute for the naming checks. **About 19 GPU-minutes on one L40S**, inside the approved 20.

**Llama-3.1-8B.** The reconciliation turns on Llama responding to what a sentence claims more than Qwen does, so a Llama arm would be informative. It cannot be added at small cost:

- *Full Llama arm* (same items, conditions and readouts, its own chat template, which has no think block): about 19 more GPU-minutes. **This doubles the budget.**
- *Reduced Llama arm*: yes/no only (Llama's yes/no effect is 2.21 nats, standardized 1.12, while its first-token effect is 0.68 nats with first-token mass 0.77 to 0.83), twelve conditions (the anchors, the four name conditions with their four controls, the description assertion and its control). 5,400 passes: about 8 more GPU-minutes, about 27 in all.

Both exceed the approved 20 minutes, so adding Llama is the user's decision. If one is added, the reduced arm is the better value: it carries the content contrasts (B to E) and the description contrast (F), which are where the models are expected to differ. The scoring script already supports it (`--mode chat --conditions …`).

**Other options, each priced on an L40S, none included:**

| Option | What it adds | Extra GPU-minutes |
|---|---|---|
| Stated reliability of the assertion (direction-A review, E4) | "The room's inventory list says there is an X in the room." against "A tester inserted this sentence at random: there is an X in the room.", both readouts; evidence weighting predicts the reliable one higher, lexical and semantic accounts predict no difference | about 2.5 |
| "Ignore the X" and its control | the bridge to the battery's P3, without the block | about 2.5 |
| Mention after the action line, with a control word at the same position | where the computation happens (section 5), not which explanation | about 2.5 |
| Failure-text pass for the full two-candidate log-odds | the headline readout, on all core cells (or on the anchors, the name assertion and its control only) | about 9 (or 2.5) |
| The 37 matched working actions, all core conditions | whether the mention lift is failure-specific, at low precision | about 1.5 |

The reliability option (E4) would fit if yes/no were dropped from the control conditions, but that would cost the secondary readout's contrasts; it is left to the user.

## 11. Freeze plan

1. **Review.** One independent review of this design, as for other exploratory studies with a frozen procedure.
2. **Analysis code before the tag.** Implement section 8 exactly (`wf/lexsem_analysis.py`), with tests that plant each scenario in simulated scores (pure lexical repetition, pure semantic activation, pure existence evidence, pure relevance, and a 50/50 lexical-plus-evidence mixture) and check that the rules return the intended readings, including "not distinguished" where they should. Run the repository's mutation check on it.
3. **Smoke run.** Five items on an L40S (under two minutes): the token-count and readout-token checks, the plain-text render, the first-token and yes/no masses per condition, and seconds per pass. Only those validity figures are committed to this note, with the smoke job's id, before the full run; no effect is computed from the smoke.
4. **Tag.** An annotated tag `lexsem-design-frozen-<date>` on the commit that holds this note (with the smoke figures), the builder, the scoring script and the analysis code, pushed to the remote. The tag message states the sample digest `cc8da36181976edd` and the commit.
5. **Run.** The scoring job is submitted from the tagged commit and records it (`runs/lexsem/<name>_commit.txt`); the analysis refuses scores from any other commit.
6. **Deviations.** Any change after the tag is written in a dated deviation note and committed before the scores it affects are read.

## 12. Implementation

- `wf/lexsem.py`: the frozen tables (spellings, aliases, object token counts and frequencies, the control pool), the carrier sentences, the condition definitions and controls, the control-word and control-description rules, the strata and the seeded, game-capped sample, and the prompt builder (the study's `build_messages` with the condition's task line; rq1v's yes/no question). It builds prompts only.
- `scripts/lexsem/score.py`: loads the model through the study's `LogoddsScorer` (pinned revision, bf16, float32 log-softmax), checks the frozen token counts and readout tokens against the loaded tokenizer, rebuilds the sample and checks its digest, runs the naming checks, then two forward passes per item and condition, appending one line per item so a rerun resumes. Plain mode reuses the audit's `plain_text`; chat mode reads as `score_one_token_pair` does.
- `pace/lexsem.sbatch`: the job script (L40S only).
- `tests/test_lexsem.py`: every mention condition is the swapped line plus one sentence; `name_assert` equals stage 0's sentence byte for byte; only name conditions contain the exact name; prompts differ only in the task line, in both formats and in plain text; control words match token counts, come from the nearest frequencies and never occur in the item; the frozen tables avoid ALFWorld words and the object's own parts; control descriptions are absent objects of the closest length; the strata, the seeded draw and the per-game cap; alias and spelling conditions only where they exist; the script refuses mismatched token counts.

## 13. Limits

- One model in the core, one environment, one prompt format. The plain-text format is off the model's chat path; the audit found it agrees with the no-block render on the goal effect, but not every sentence's effect need agree.
- The sentences are artificial lines appended to the task. A quoted-word note or a generic statement is unlike anything in ALFWorld, so part of any difference could be a reaction to an unusual line; the matched controls absorb the carrier's own effect but not an interaction between carrier oddity and the object.
- The generic statement carries a weak base-rate suggestion that such objects are common; the quoted word is the cleaner zero-claim condition, and both are reported.
- Aliases exist for 8 of the 27 object types and are less frequent than the names; descriptions are longer than names (controlled within pairs, not across forms).
- The spaced spelling shares some tokens with the closed name for most types, so it is a graded, not a clean, test of token-sequence repetition.
- Behavioural contrasts cannot tell "different mechanisms" from "the same mechanisms with different weights"; the mixture estimates describe, they do not identify.
- Gates and readings use 95% intervals without multiplicity correction; the 0.25 margins and the requirement to read patterns across contrasts are the protection.
