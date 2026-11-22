# Mechanism battery: why does naming the action's object push predictions toward success? (design, 2026-11-20)

Exploratory, not preregistered. This note fixes the design, the predictions and the readouts before any score of this run exists. **Amended after review 1 (section 11), after review 2 (section 12) and after review 3 (section 13), still before any score: where a later section differs from an earlier one, the later one holds.** Code: `wf/mech2.py`; runs `mech2_score` (8 game shards), `mech2_score_smoke` (one game, format and timing only) and `mech2_analysis`; checks on the real items: `scripts/check_mech2_prompts.py`; tests: `tests/test_mech2.py`.

## 1. Question and candidate accounts

On RQ2's determined failures, a task line that names the action's object raises the log-odds of the success text against "Nothing happens." (ΔL = 6.12 nats for Qwen3-8B, 1.65 for Llama-3.1-8B). Earlier exploratory runs constrain the explanation: most of the effect survives when the object is described instead of named (61% and 92%); for Qwen, a sentence stating that the object is somewhere in the room reproduces most of it; stating the deciding facts barely reduces it, and the effect is fully present on items whose deciding fact the model answers correctly (`exploratory_results.md`, sections 1, 3 and 5). This battery asks which of six accounts, not mutually exclusive, the effect follows.

- **H1, no state tracking.** The model does not recover the facts that decide the failure and falls back on the task as a guide to what happens. Its signature is where the effect sits: on items whose deciding fact the model does not know. The mechanism diagnostic already found the aligned effect fully present on known-state items (share 1.11 for Qwen, 1.01 for Llama), against H1 for the aligned task.
- **H2, task as evidence.** The task line is read as information about hidden facts: a mug task implies there is a mug in the room that can be got; a finished placement task implies the mug now sits at the target. The belief changes, and the prediction follows the belief.
- **H3, salience.** Any mention of the action's object in the task line raises the success reading at answer time, without a change in belief. What the line says about the object (wanted, unwanted, finished, to be ignored) does not matter, only that it is mentioned.
- **H4, wishful or helpful prediction.** The model predicts the outcome the user wants, a disposition plausibly learned in post-training. Wanting the action's outcome raises L; wanting it not to happen lowers it; indifference leaves it.
- **H5, policy blending.** The model blends "what would an agent with this task do" with "what would happen": an action that is a sensible step for the task reads as one that works. This includes a rational-agent reading (the agent tries it, so it presumably expects it to work); the battery does not separate that reading from blending.
- **H6, prompt artefacts.** The effect rides on surface features of the prompt: the task line's position (at step 0 it is the last line before "Action:"), overlap between the task line and a prompt example's task line, and word overlap. Lexical overlap with the object's name predicts the same content pattern as H3, so the battery separates H6 from H3 only through position and example overlap.

## 2. Items, prompt, models and formats

**Items.** RQ2's 4,666 primary items (`rq1.rq1_sets(...)["primary"]`).

- A **stratified sample of 1,000 items** (`SAMPLE_N`): strata are (game, failure reason), 495 strata; each stratum gets its proportional share of 1,000, rounded by largest remainder (ties broken by a hash of the stratum), and within a stratum the items with the smallest `sha256("20261120|mech2|<item id>")` are taken. ~~Inclusion probability is about 0.214 for every item, so sample means estimate the population's item-weighted means.~~ *Corrected after review 1 (section 11.8):* rounding small strata makes inclusion probabilities range from 0 to 0.33 (mean 0.214, SD 0.046), and 117 items in 72 strata of one or two items can never be drawn, among them all four items of one game (valid_unseen/73). Sample means are unweighted and estimate the mean over the strata the sample covers, not exactly the population's item-weighted mean. The sample holds 1,000 items in 124 games: 113 never-moved, 416 away from the receptacle, 197 not holding the object, 236 contents or open state, 38 hands full; by task type 132 pick, 153 pick-two, 251 clean, 168 heat, 183 cool, 113 examine; 111 are step-0 items.
- **Every step-0 item** (history length 0): 514 items in all 125 games, every one decided by "agent has not moved".
- Every arm is scored on the **union**, 1,403 items. Contrasts P1 to P5 read the sample; P6 reads the 514 step-0 items.

**Prompt.** RQ2's prompt: example set 1, history only, `build_messages(item, "aligned", task_line=<arm's line>)`. The arms reuse the earlier runs' prompts exactly where they overlap (aligned, swapped and neutral are `rq1_score`'s; the exists sentence is `rq1_score`'s `swapped_plus`; the described lines are `rq1x_score`'s; the yes/no question is `rq1v_score`'s), which `tests/test_mech2.py` checks message for message.

**Models.** Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, few-shot, loaded at the snapshots pinned in `wf/constants.py` (`REVISIONS`). The runner refuses a model whose loaded revision is not the pinned one, records the revisions in each shard's `config.json`, and each score carries `model_revision`.

**Formats.** Every arm in both formats:

- **two-candidate** (`cand`): L is the log-odds of the item's success text against "Nothing happens.", read with `score_pair`, giving L, the first-token term and the two-candidate score s;
- **yes/no** (`yn`, condition `yn_<arm>`): the final "Next observation:" is replaced by rq1v's question ("will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens."), and L is the log-odds of "Yes" against "No" from one forward pass (`score_one_token_pair`).

**Tokenization gate** (as on branch `families`, audit C9), run once a model is loaded and before any score, on a resume too: the gate renders prompts exactly as scoring does (checked against `LogoddsScorer._encode`); each item's two candidates, tokenized alone, equal the chat template's tokens for them as the reply after the item's aligned prompt, and their first tokens differ; "Yes" and "No" are single, distinct tokens and equal the template's reply after every yes/no prompt the shard scores. Results go to `tokenization_checks_<model>.json` in the shard folder; a failure stops the job before that model scores anything.

## 3. Arms and exact templates

Every arm differs from the aligned arm only in the task line, except the position arms, which differ from their plain arm only in where the task line sits. `{obj}` is the goal object's type (the action names an instance of it), `{a}` its article, `{pp}` the action verb's participle (taken, moved, heated, cooled, cleaned), `{prep}` the verb's preposition (from, to, with), `{rec}` the receptacle instance the action names, `{swapped}` RQ2's swapped task line, `{task}` the real task line without "Your task is to: ".

| Arm | Template | Purpose |
| --- | --- | --- |
| aligned | the real task line | reference |
| swapped | RQ2's swapped line (another object in the room) | reference for every contrast |
| neutral | Your task is to: explore the room. | reference |
| desire_keep | Your task is to: make sure that the {obj} is not {pp}; leave it where it is. | reversed desire, the object presupposed ("the") |
| desire_check | Your task is to: find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it cannot. | reversed desire, existence not presupposed ("a"); trying the action is the test asked for |
| mention_ignore | {swapped} Ignore the {obj}. | mention without need |
| mention_exists | {swapped} There is {a} {obj} somewhere in the room. | the earlier extra-mention sentence, rescored for pairing and the yes/no format |
| described | the real task with the object replaced by rq1x's description | need without name |
| swapped_described | the swapped task with its object described | rq1x's control |
| done | Your task was to: {task} It is already done; nothing is left to do. | goal already achieved |
| cancelled | Your task was to: {task} The user has cancelled it; nothing is left to do. | same mention, desire withdrawn, no claim about the world |
| aligned_start, swapped_start | aligned or swapped line, moved to the first line of the history | position |
| aligned_end, swapped_end | aligned or swapped line, moved to the last line of the history, just before "Action:" | position |

Examples, for an item with the real task "put a clean bowl in cabinet.", swapped object pan, action `take bowl 1 from cabinet 1`:

- desire_keep: "Your task is to: make sure that the bowl is not taken; leave it where it is."
- desire_check: "Your task is to: find out whether a bowl can be taken from the cabinet 1; the user hopes it cannot."
- mention_ignore: "Your task is to: put a clean pan in cabinet. Ignore the bowl."
- described: "Your task is to: put a clean deep round dish for soup or cereal in cabinet."
- done: "Your task was to: put a clean bowl in cabinet. It is already done; nothing is left to do."

For the other verbs the desire lines read, for instance, "make sure that the tomato is not heated; leave it where it is." and "find out whether a saltshaker can be moved to the cabinet 1; the user hopes it cannot."

**Templates by task type.** The desire arms are built from the action, so one template per verb covers every task type; `scripts/check_mech2_prompts.py` built them for all 22 task-type and verb combinations present (pick, pick-two, clean, heat, cool, examine × take, move, heat, cool, clean). The arms built on a task line keep its wording, so they fit every type: mention_ignore and mention_exists append a sentence to the swapped line; done and cancelled quote the real task after "Your task was to: ", which reads for every template ("…examine the pencil with the desklamp. It is already done…", "…put two pillow in sofa. It is already done…"); described replaces the one object word, as in rq1x.

**Position, per step.** In every item's history the opening observation is line 1 and the task line is line 2. At step 0 the history has no actions, so the task line is the last line before "Action:"; after step 0 it is the second line, followed by the actions and observations. The prompt examples put their task line first. So:

- at step 0, `_end` repeats the plain prompt exactly and is not scored (the analysis reads the plain score), and `_start` moves the task line from just before the action to the top, above the room description;
- after step 0, `_start` swaps lines 1 and 2 (a small move; it also matches the examples' format), and `_end` moves the task line from the top to just before "Action:".

**What each line claims.** Checked on every item of the union by `scripts/check_mech2_prompts.py`.

- desire_keep and mention_ignore presuppose that an {obj} exists in the room. That is true for every item: the goal object's type is in the scene on all 4,666 items. They state no location and no instance.
- desire_check names no instance but the receptacle the action names, which exists, and states nothing about where the object is.
- mention_exists asserts that an {obj} is in the room (true). The swapped task presupposes its object (in the scene on all 4,666 items).
- done asserts that the real task is finished. This is false by design: it is the arm's manipulation. For placement tasks it implies the object now sits at a target receptacle. On 62 of the 1,000 sample items the history contradicts it outright (every target-type receptacle was seen with its contents listed and none held the goal object's type, `done_contradicted`); the done contrast is also reported without them. Examine tasks are not placement tasks, so done implies no location there, and P5 leaves them out.
- cancelled stipulates the user's decision and makes no claim about the world.
- The desires ("make sure", "the user hopes") are stipulations, not facts the model could check.
- Inherited wording is kept unchanged for comparability: a mismatched article in RQ2's swapped line ("put a alarmclock", 8 items of the union, all in the sample; carried into mention_ignore and mention_exists) and in rq1x's described swapped line ("a eating utensil", 6 items). The battery's own text has correct articles.

## 4. Predictions

The table gives each arm's predicted value minus the swapped arm's, as a fraction r of the aligned effect on the same items. Write $D_X$ for the mean over items of $L(X) - L(\text{swapped})$; then

$$r_X = \frac{D_X}{D_{\text{aligned}}}.$$

"Large" means $r \geq 0.5$, "small" means $0 < r < 0.5$, "≈0" means $|r| \leq 0.25$, "<0" means below the swapped arm. The arm features behind the predictions are listed first.

| Arm | Names the action's object | Implies the object exists | Wants the action's outcome | Action is a sensible step for the line | Implies the object is elsewhere |
| --- | --- | --- | --- | --- | --- |
| aligned | yes | yes | yes | yes | no |
| desire_keep | yes | yes (presupposed) | no, wants it not to happen | no, counter to it | no |
| desire_check | yes | no | no, hopes it fails | yes, the test asked for | no |
| mention_ignore | yes | weakly (presupposed) | indifferent | no | no |
| mention_exists | yes | yes (asserted) | indifferent | no | no |
| described | no (described) | yes | yes | yes | no |
| done | yes | yes | no longer (achieved) | no | yes, for placement tasks |
| cancelled | yes | yes | no (withdrawn) | no | no |

**Prediction table, against the swapped task.**

| Arm (contrast) | H2 evidence | H3 salience | H4 wishful | H5 policy blending | H6 artefact |
| --- | --- | --- | --- | --- | --- |
| desire_keep (P1) | positive, large or small | positive, large or small | below 0 | ≈0 or below 0 | as H3 |
| desire_check (P2) | small or ≈0 | positive, large or small | below 0 | positive, large or small | as H3 |
| mention_ignore (P3) | small or ≈0 | positive, large or small | ≈0 | ≈0 or below 0 | as H3 |
| described (P4) | large | ≈0 or small | large | large | ≈0 (no name to overlap) |
| done minus cancelled (P5) | below 0 | ≈0 | ≈0 | ≈0 | ≈0 |
| task line moved to the top at step 0 (P6) | ≈0 | ≈0 or below 0 | ≈0 | ≈0 | not 0 (expected below 0) |
| mention_exists (secondary) | large | large | ≈0 | ≈0 | as H3 |
| cancelled (secondary) | positive | positive | ≈0 | ≈0 | as H3 |
| done (secondary) | at most cancelled's value | positive | ≈0 | ≈0 | as H3 |

H1 makes no prediction of its own about the arms' signs: under H1 the task guides the guess where the state is unknown, so the content pattern is whichever of H2, H4 or H5 describes that guide, but confined to items whose deciding fact the model does not know. Its readout is the knowledge split (section 6): the share of each positive contrast found on known-state items, which H1 predicts below 0.75.

**Against the neutral task.** Write $g$ for the swapped task's lift over the neutral one, $g = L(\text{swapped}) - L(\text{neutral})$, which was +2.56 nats for Qwen and +1.20 for Llama in `rq1x_score` and is re-measured here. Each arm's value against neutral is its value against swapped plus $g$:

$$L(X) - L(\text{neutral}) = D_X + g.$$

So an arm predicted "≈0" against swapped is predicted at about $g$ above neutral (it carries the generic lift of having a household task), "large" means well above $g$, and "below 0" means below $g$. H4 predicts the two desire arms below neutral as well (a user who wants the action to fail should push below a task that wants nothing); H5 predicts desire_keep at or below neutral (the action works against the task). The other accounts make no prediction against neutral beyond the shift by $g$.

**Patterns.** Over P1 to P6 the accounts predict:

| Contrast | H2 | H3 | H4 | H5 | H6 |
| --- | --- | --- | --- | --- | --- |
| P1 desire_keep | + | + | − | 0 or − | (no prediction) |
| P2 desire_check | small or 0 | + | − | + | (no prediction) |
| P3 mention_ignore | small or 0 | + | 0 | 0 or − | (no prediction) |
| P4 described | + large | 0 or small | + large | + large | (no prediction) |
| P5 done minus cancelled | − | 0 | 0 | 0 | (no prediction) |
| P6 position, step 0 | 0 | 0 or − | 0 | 0 | not 0 |

H2 and H3 differ on P4 and P5; H4 and H5 on P2; H2 and H3 against H4 and H5 on P1; H3 against H5 on P1 and P3; H6 against H2, H4 and H5 on P6. H6's content predictions coincide with H3's and are not used.

## 5. Primary contrasts, tests and mapping

Six named contrasts per model, each a within-item difference on the two-candidate format, with the aligned-minus-swapped effect on the same items as the denominator of its size ratio r:

1. **P1_desire_keep**: desire_keep − swapped, on the sample.
2. **P2_desire_check**: desire_check − swapped, on the sample.
3. **P3_mention_without_need**: mention_ignore − swapped, on the sample.
4. **P4_need_without_name**: described − swapped, on the sample.
5. **P5_done_vs_cancelled**: done − cancelled, on the sample's placement-task items (887 of 1,000; examine tasks left out).
6. **P6_position_step0**: the change in the goal effect when the task line moves to the top, on all 514 step-0 items:

$$P6 = \big[L(\text{aligned\_start}) - L(\text{swapped\_start})\big] - \big[L(\text{aligned}) - L(\text{swapped})\big].$$

**Tests.** Each contrast's mean has a game-clustered percentile bootstrap 95% interval (2,000 resamples) and a two-sided game-level sign-flip permutation p (1,000 permutations); r's interval comes from the same resamples. Holm's correction runs over the six p-values within each model. The yes/no format repeats the six contrasts as a second family with its own Holm correction (the replication); the two-candidate family is primary.

**Categories**, from the Holm p and r's interval:

- **NEG**: Holm p < 0.05 and the mean below 0;
- **POS_LARGE**: Holm p < 0.05, mean above 0, r's lower bound at least 0.5;
- **POS_SMALL**: Holm p < 0.05, mean above 0, r's upper bound at most 0.5;
- **POS**: Holm p < 0.05, mean above 0, r's interval straddling 0.5 (size unresolved);
- **NULL**: Holm p at least 0.05 and r's interval inside [−0.25, 0.25];
- **INCONCLUSIVE**: anything else.

**Mapping to accounts** (`PREDICTIONS` in `wf/mech2.py`, `readout`). For each account and contrast the observed category is *consistent* if the prediction allows it, *contradicted* if it does not, and *undecided* if the category is INCONCLUSIVE. POS with unresolved size counts as consistent with any positive prediction. Allowed categories:

- H2: P1 POS_LARGE or POS_SMALL; P2 POS_SMALL or NULL; P3 POS_SMALL or NULL; P4 POS_LARGE; P5 NEG; P6 NULL.
- H3: P1, P2 and P3 POS_LARGE or POS_SMALL; P4 NULL or POS_SMALL; P5 NULL; P6 NULL or NEG.
- H4: P1 NEG; P2 NEG; P3 NULL; P4 POS_LARGE; P5 NULL; P6 NULL.
- H5: P1 NULL or NEG; P2 POS_LARGE or POS_SMALL; P3 NULL or NEG; P4 POS_LARGE; P5 NULL; P6 NULL.
- H6: P6 NEG, POS_SMALL or POS_LARGE.

**Reading.** The report gives, per model, each contrast's estimate, interval, r, Holm p and category, and the account-by-contrast table of consistent, contradicted and undecided. An account with no contradicted contrast is called consistent with the battery; an account with one or more is called contradicted on those contrasts. The accounts are not exclusive, so a mixture produces intermediate values: H3 and H4 together could cancel on P1 and look like H5 there, but they would differ from H5 on P2 (H3 and H4 pull in opposite directions, so near 0, against H5's positive) and on P3 (positive under H3, 0 under H5). Where no single account is consistent, the report describes which contrasts each account fails rather than forcing a choice. A contrast whose category differs between the two-candidate and yes/no families is reported as format-dependent, and the reading names the disagreement. A positive P5 (done above cancelled) is predicted by no account; it would be read as a reaction to the completion wording itself ("done"), not as support for any account. No other conclusion is drawn from the primary family.

Prior results already bear on the table: P4's predecessor (described against described swapped, rq1x) kept 61% and 92%, and the knowledge split was 1.11 and 1.01. The battery remeasures P4 against the named swapped task, in the same job as the other arms and in the yes/no format.

## 6. Secondary readouts (descriptive, unadjusted)

Per model, from `mech2_analysis`:

- every content arm against swapped (L, first-token term, two-candidate score s in points, r), against neutral and against aligned, on the sample; in the yes/no format, ΔL, the shift in mean σ in points, and the **item share toward Yes** (the share of items whose yes/no log-odds are higher under the arm than under the swapped task, ties counted against; the families' cross-model metric), which is also tabulated across the two models;
- each primary content contrast at step 0 (all 514 step-0 items) and after (sample items with history), for take against other actions, and by failure reason;
- cancelled − swapped and done − swapped; mention_ignore − mention_exists (a presupposed against an asserted object); described − swapped_described (rq1x's contrast, replicated); desire_check − desire_keep; done − cancelled without the 62 contradicted items;
- position: P6 in the yes/no format; start against usual over the sample; end against usual after step 0; end against start at step 0 and after;
- **few-shot task overlap** (analysis-only, `rq1_score`, which scored every primary item with example sets 1, 2 and 3): for set k and item i, the item's aligned-minus-swapped effect under set k minus its mean under the other two sets, compared between items whose real task line equals a task line of set k's examples and all other items (the plain comparison, set k's effect on those items against the other items, is reported next to it). H6 predicts a positive difference; the same comparison for items whose swapped line equals an example's task line, where H6 predicts a negative one. Exact task-line matches among the primary items: 49 items in 2 games for set 1, 91 in 3 games for set 2, 66 in 2 games for set 3 (206 item-set pairs; 42, 120 and 48 for the swapped line). The brief that commissioned this battery cited 232 items; no exact-match count on the primary items gives 232, so the counts above are the ones used. With two or three games per set, this is descriptive only;
- **knowledge split for H1** (analysis-only, `mech_score`): each primary content contrast and the aligned effect on sample items whose state question and its twin the same model answers correctly under the neutral task, as a share of the effect on all sample items with a verdict.

## 7. Sample size and power

Spreads from the RQ2 items (`rq1x_v2_metrics.json`, `named_goal`): the per-item SD of ΔL is 6.0 nats for Qwen and 1.94 for Llama, and the within-game correlation is 0.26 and 0.12. The sample has about 8.1 items per game ($m$), so the design effect $1 + (m - 1)\rho$ is 2.84 for Qwen and 1.85 for Llama. The standard error of a contrast's mean over $n = 1000$ items is

$$\mathrm{SE} = \mathrm{SD}\sqrt{\frac{1 + (m-1)\rho}{n}},$$

0.32 nats for Qwen and 0.082 for Llama, if a contrast between two new arms spreads as much as ΔL does. At Holm's strictest step (two-sided α of 0.05/6) and 80% power the detectable contrast is $(2.64 + 0.84)\,\mathrm{SE}$: 1.11 nats for Qwen and 0.28 for Llama, about 18% and 17% of the aligned effect. At Holm's last step (α = 0.05) it is 0.90 and 0.23. If the new contrasts spread 1.5 times as much, these become 1.67 and 0.43 nats (27% and 26%).

The NULL category needs r's interval inside ±0.25. The interval's half-width is about $1.96\,\mathrm{SE}/D_{\text{aligned}}$, 0.10 for both models, so a true r of 0 lands in NULL with high probability (its estimate must fall within ±0.15 of 0, about 2.9 standard errors). Likewise a true r of 0.75 is called POS_LARGE unless its estimate falls below 0.6.

For P6 (514 items, 4.1 per game) the difference of two effects is taken to spread √2 times as much as ΔL (8.5 and 2.7 nats), which is conservative when the two effects are positively correlated. The design effect is 1.81 and 1.37, the SE 0.50 and 0.14 nats, and the detectable change at the strictest Holm step 1.75 nats for Qwen (16% of its step-0 effect of 11.0) and 0.48 for Llama (about a third of its step-0 effect of about 1.5). Llama's P6 may therefore end INCONCLUSIVE rather than NULL even with no position effect; that outcome is reported as such.

## 8. GPU estimate and sharding

Per model the run scores 20,017 two-candidate and 20,017 yes/no prompts (`scripts/check_mech2_prompts.py`; the step-0 `_end` aliases are not scored). Over both models that is 40,034 two-candidate scores (two forward passes each) and 40,034 yes/no scores (one pass each), 120,102 forward passes. `rq2_score` scored about 1,100 items per hour with both models, about 12.8 passes per item including its short generations, or about 0.26 s a pass; the families sizing used 0.4 s. At 0.26 to 0.4 s a pass the scoring takes 8.7 to 13.3 GPU-hours, plus about 0.8 for loading each model in each of 8 shards and a few minutes per model and shard for the tokenization gate (CPU): about 10 to 15 GPU-hours in all, 1.2 to 1.9 hours per shard. `pace/mech2_score.sbatch` asks for 3 hours per shard on any L40S, A40 or A100 with 32 GB of memory; scores are saved one by one and resume with `WF_RESUME=1`. The scores file is about 50 MB in all. Before the eight shards, `mech2_score_smoke` (one game) is run and read for format, gate and timing only.

## 9. Limits

- The six accounts are not exclusive and the arms are not minimal pairs of one another: each differs from the swapped arm in several features (section 4). The design rests on the predicted sign patterns being different, not on isolating one feature per arm; mixtures can mimic a single account on one contrast (section 5).
- H3 and the lexical part of H6 make the same content predictions; only position and example overlap separate H6.
- H5's blending and the rational-agent reading of it predict the same pattern.
- H1 is tested only through the knowledge split, whose state question is a weak measure of knowledge (both models lean toward No; `exploratory_results.md`, section 5).
- The desire, done and cancelled lines are unlike any ALFWorld task, so part of any difference may be a reaction to an unusual line; the neutral and swapped references are ordinary task lines. Done states a falsehood on purpose, and the history contradicts it on 62 sample items.
- The two-candidate L compares one success wording with "Nothing happens." and is not a probability of success; the yes/no format is the check on that, and the item share is the cross-model metric.
- Word overlap with the success text differs between arms. desire_check names the action's receptacle ("the cabinet 1"), which the take and move success texts repeat, so on the two-candidate format it may gain from copying alone; the yes/no format, whose answers are "Yes" and "No", does not share this, so P2's yes/no category is read alongside its two-candidate one. The desire lines' participles share a stem with the move, heat, cool and clean success texts, as the aligned task's verbs do.
- Inherited wording ("put a alarmclock", "a eating utensil") is kept for comparability with the earlier runs.
- Few-shot overlap rests on two or three games per example set.
- The game bootstrap intervals were slightly narrow in simulation (93–95% coverage; `testing.md`).

## 10. Stub for `notes/deviations.md` (exploratory entry, to be merged)

**Exploratory: mechanism battery, which account the goal effect follows (decided 2026-11-20, before any of these scores).** `wf/mech2.py`, runs `mech2_score` (8 shards), `mech2_score_smoke` and `mech2_analysis`; design, templates, prediction table and readouts in `notes/mechanism2_design.md`. Not preregistered. Few-shot Qwen3-8B and Llama-3.1-8B at their pinned revisions, example set 1, history only, two-candidate and yes/no formats. Items: a 1,000-item sample of RQ2's primary items stratified by game and failure reason (seed "20261120|mech2"), plus all 514 step-0 items (1,403 in all). Six accounts (no state tracking, task as evidence, salience, wishful prediction, policy blending, prompt artefacts) predict different signs for task lines that differ from the aligned and swapped tasks only in the task line: a reversed desire with the object presupposed ("make sure that the {obj} is not {pp}; leave it where it is.") and without it ("find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it cannot."), the swapped task plus "Ignore the {obj}.", the object-free description of the real task, the real task marked done or cancelled, and the aligned and swapped lines moved to the top or bottom of the history. Primary: six contrasts per model on the two-candidate format, Holm within model (desire_keep − swapped, desire_check − swapped, mention_ignore − swapped, described − swapped, done − cancelled on placement tasks, and the step-0 change in the goal effect when the task line moves to the top), each classified NEG, NULL, POS_SMALL, POS_LARGE, POS or INCONCLUSIVE by fixed rules and read against the prediction table; the yes/no format is the replication family. Analysis-only checks on earlier runs: few-shot task overlap (`rq1_score`, 206 aligned-overlap item-set pairs) and the knowledge split (`mech_score`). Estimated 10 to 15 GPU-hours. *Amended after review 1, still before any score (section 11):* H3 defined as referent salience and name salience read with word overlap (H6_lexical); a negated-existence arm on take items ("{swapped} The user says there is no {obj} in the {rec}."); seven primary contrasts per model and format (desire_keep − mention_ignore replaces desire_keep − swapped; negated − swapped added; P5 on 746 items), Holm within model and format; size zones with a TRIVIAL label (an r interval inside ±0.25 counts as ≈0 whatever its p) and a 2% dropped-draw rule for r; P6 counts only for the position account; a contrast is decided only when both formats agree; verdicts consistent, contradicted or untested; the sampling claim corrected; Mistral-7B-Instruct-v0.3 scored as a secondary, descriptive model. Estimated 15 to 22 GPU-hours. *Amended after review 2, still before any score (section 12):* P1 and P2 made minimal pairs (desire_keep_sw = "{swapped} Make sure that the {obj} is not {pp}; leave it where it is." against mention_ignore; desire_check against desire_check_can, "…; the user hopes it can."), the first P1 and P2 kept as secondary readouts; an affirmed arm ("{swapped} The user says there is {a} {obj} in the {rec}.") on the negated arm's items and P8 = negated − affirmed, a family of eight with Holm at a fixed size of eight; prediction cells amended (P4 positive for every content account but H6_lexical; H4 on P3 and P5 below zero or about zero; H2 on P1 about zero, small or large, on P7 below zero or about zero); H6_lexical defined as overlap with the action string; an account is consistent only where a rival fails, otherwise untested; a contradiction reads "cannot be the sole account"; under a mixture the estimates of P5, P7 and P8 are the readout; P5 also read against cancelled's own lift; P7 and P8 also on the 293 items whose receptacle was not seen, the first-token term fixed as descriptive; every score checked against the current items before analysis; Mistral scored on the 13 arms the primary contrasts read, position arms at step 0 only, swapped_described dropped. Estimated 13 to 19 GPU-hours. *Amended after review 3, still before any score (section 13):* the mention component M = (negated + affirmed)/2 − swapped on P7's items, in the mixture readout and the summary, with two fixed sentences when salience is contradicted only on contrasts that add an explicit claim (salience likely part of the effect when M lies above the band in both formats; claims read, evidence not shown, when task as evidence is consistent); an account is consistent only on a risky prediction (a discriminating contrast whose allowed zones exclude about zero), else untested, and each account is listed with the contrasts its verdict rests on; task as evidence allows P3 about zero; desire_check − desire_keep in both formats as a second wishful-against-blending readout; a negation placebo pair ("{swapped} The user says there is no {swapped_obj} in the {rec}." against the article) as a named secondary, paid for by scoring Mistral's content arms on the sample only; H7_check (the task directs which part of the state the model checks) as an exploratory account read on existing arms, outside the verdicts; an undefined score on a primary arm refuses the analysis. Estimated 13 to 19 GPU-hours. *Amended after review 4, still before any score (section 14):* M is described as a mention-associated residual after averaging opposing claims, and the summary no longer reads a positive M as salience being part of the effect (it says the residual is consistent with salience but also with asymmetric processing of affirmations and denials, so it does not show that salience contributes); an adversarial check planted pure task as evidence with 25 asymmetric claim-response patterns; M's two halves (affirmed − swapped, swapped − negated) are reported; a sentence when M lies inside the band in both formats; when the negation placebo lies below the band in both formats the summary names the verdicts that rest on P8 and gives P8 net of the placebo; the placebo also without the 87 items whose swapped task targets the action's receptacle type; the status line gives desire_check − desire_keep when wishful prediction and policy blending both stand. No prediction, contrast, category or verdict rule changed. Estimated 13 to 19 GPU-hours.

## 11. Amendments after review 1, before any score (2026-11-20)

A review of the design and code (decision rules simulated on the real items, 20 deliberate bugs, a sampling check) found that the decision rules could contradict accounts on effects too small to matter, that P6 counted against accounts that predict nothing about position, that some of H2's predictions did not follow from its own definition, and that the sampling note was wrong. Nothing had been scored. The amendments below were made, tested and committed before any GPU run of this battery; where they differ from sections 1 to 9, they hold. Code: `wf/mech2.py`; synthetic check: `scripts/mech2_simulate.py`.

### 11.1 Accounts, made precise

- **H1 (no state tracking) is not falsifiable by this battery.** Its prediction is about where the effect sits (items whose deciding fact the model does not know), not about the arms, so no pattern of the seven contrasts can contradict it. Its only readout is the knowledge split (section 6). It is left out of the readout.
- **H3 is referent salience**: making the action's object salient in the task line, by name or by description, raises the success reading at answer time, whatever the line says about it. A described object is still salient, so H3 now predicts P4 positive.
- **Name salience** (only the object's word matters) is not separable here from word overlap between the task line and the action, so it is read as part of H6: **H6_lexical** carries the content predictions of name salience and word overlap (P4 near zero, since the described line does not contain the name), and **H6_position** carries P6 alone. The few-shot overlap check (section 6) stays descriptive.
- **Mixtures can fit any pattern.** The accounts are not exclusive, and a weighted mixture of two accounts can produce any value between their predictions on every contrast. The battery can contradict a single account; it cannot show that one account alone is at work, and it cannot rule out a mixture.

### 11.2 New arm: negated existence

Template, for take actions only (the one verb whose success needs the object in the receptacle the action names):

> {swapped} The user says there is no {obj} in the {rec}.

Example: "Your task is to: put a clean pan in cabinet. The user says there is no bowl in the cabinet 1." It names the action's object (and receptacle) exactly as the other mention arms do, but denies that the object is where the action needs it. So it separates mention from evidence: under H3 the mention raises the success reading; under H2 the claim lowers it.

Why "the user says". On the 522 take items of the sample the claim is true of the scene at the start of the game on 356 and false on 166. The history shows the receptacle's contents without the object on 210 items (the claim is established), shows them with the object on 19 (the claim is contradicted), and does not show them on 293. A bare "There is no …" would put a false statement in the narrator's voice on a third of the items. Attributed to the user, the sentence is true as a report on every item, and it is still information about a hidden fact, which is what H2 says the task line is read as. The 19 items whose history contradicts the claim are left out of P7 (a rational reader would discount the claim there), leaving 503 items. "in" is used for every receptacle, as ALFWorld's own task lines write it ("put a cool apple in countertop.").

The arm adds 714 prompts per format and model (the take items of the union). `scripts/check_mech2_prompts.py` checks it on every real item: it is built exactly for take items of the goal object, names no instance but the action's receptacle, and differs from the aligned prompt only in the task line.

### 11.3 Primary family: seven contrasts per model and format

| Contrast | Arms | Items | What it separates |
| --- | --- | --- | --- |
| P1 keep vs ignore | desire_keep − mention_ignore | sample (1,000) | same presupposed object ("the {obj}") in both lines, only the desire differs: wishful prediction against salience and evidence |
| P2 desire check | desire_check − swapped | sample | wishful prediction against the rest |
| P3 mention without need | mention_ignore − swapped | sample | policy blending and wishful prediction against mention |
| P4 need without name | described − swapped | sample | name or word overlap against the rest |
| P5 done vs cancelled | done − cancelled | 746 placement items (11.6) | evidence against the rest |
| P6 position at step 0 | as in section 5 | 514 step-0 items | position artefact |
| P7 negated existence | negated − swapped | 503 take items | evidence against salience (and word overlap) |

desire_keep − swapped (the old P1) is now secondary: it equals P1 plus P3. Holm's correction runs over the seven p-values within each model and format, at a fixed family size of seven (a missing or undefined p enters as 1). Each contrast's r uses the aligned-minus-swapped effect on the same items and in the same format as its denominator.

### 11.4 Size zones, TRIVIAL, and the 2% rule

r is read on four zones that do not overlap:

- N: r below −0.25;
- Z (≈0): r between −0.25 and 0.25, both included;
- S (small): r above 0.25 and below 0.5;
- L (large): r at 0.5 or above.

A contrast's observed zones are the run of zones its 95% interval for r touches. Labels: NEG (N), NEG_OR_TRIVIAL (N and Z), NULL (Z, Holm p at least 0.05), **TRIVIAL (Z, Holm p below 0.05)**, TRIVIAL_OR_SMALL (Z and S), POS_SMALL (S), POS (S and L), POS_LARGE (L), WIDE (three or more zones). An interval inside ±0.25 counts as ≈0 whatever its p-value, so a small but significant effect never contradicts a ≈0 prediction. A contrast is INCONCLUSIVE (undecided for every account) when its estimate or interval is undefined, when its interval lies wholly outside the band (a claim of a non-trivial effect) but Holm's p is at least 0.05, or when r is not estimable: the reference effect is at or below zero, or more than 2% of the bootstrap draws leave r undefined (a resample whose reference mean is at or below zero), the rule the families analysis uses. Every r reports its number of dropped draws.

### 11.5 Prediction table (amended)

Each cell is the set of zones the account allows; "+" means S or L (positive beyond the band). An empty cell is no prediction.

| Contrast | H2 evidence | H3 referent salience | H4 wishful | H5 policy blending | H6_lexical | H6_position |
| --- | --- | --- | --- | --- | --- | --- |
| P1 keep vs ignore | Z or S | Z or S | N | N or Z | Z or S | |
| P2 desire check | + | + | N | + | + | |
| P3 mention without need | + | + | Z | N or Z | + | |
| P4 need without name | L | + | L | L | Z | |
| P5 done vs cancelled | N | Z | Z | Z | Z | |
| P6 position at step 0 | | | | | | N, S or L |
| P7 negated existence | N | + | Z | Z | + | |

Reasons for the cells that changed:

- **H2 on P1, P2, P3.** mention_ignore and desire_keep both presuppose the object ("the {obj}"), and the earlier sentence "There is a {obj} somewhere in the room." added to the swapped task reproduced 6.00 of Qwen's 6.12 nats and 0.99 of Llama's 1.65 (`exploratory_results.md`, section 1), so H2 predicts P3 positive. It does not insist on large: "Ignore the {obj}." presupposes the object instead of asserting it. P1 compares two lines with the same presupposition; "leave it where it is" may add that the object sits somewhere untaken, so H2 allows Z or S. desire_check names the receptacle and asks whether the object can be taken from it, an open question that raises the possibility that it is there, so H2 predicts P2 positive.
- **H3 on P4.** Referent salience predicts the described object raises the success reading too (positive). H6_lexical, which needs the name, predicts ≈0.
- **H4, H5 on P1.** Wishful prediction: the user wants the action not to happen in desire_keep and is indifferent in mention_ignore, so N. Policy blending: neither line makes the action a sensible step, and desire_keep makes it a counter-step, so N or Z.
- **P7.** H2: the user's report that the object is not in the receptacle lowers the success reading (N). H3 and H6_lexical: the line names the object (and the receptacle, which the take success text repeats), so positive. H4 and H5: the desire and the sensible steps are the swapped task's, unchanged (Z).
- **P5.** On the 746 items (11.6) every reading of H2 predicts done below cancelled (N); the other accounts predict Z.
- **P6 counts only for H6_position.** H2, H4 and H5 predict nothing about position, and H3 as defined does not either (salience by mention, not by distance). P6 no longer counts for or against them. Reading P6 as "the effect rides on the task line's position" needs r's interval to lie wholly beyond ±0.25 (|r| above 0.25 with the interval excluding the band, which also excludes 0) and Holm's p below 0.05. A P6 inside the band contradicts H6_position; one that straddles 0.25 leaves it undecided, and is reported as "position moves the effect, size not established" when its interval excludes 0.

Discriminating contrasts: H2 against H3 on P5 and P7 (and P4 only weakly: L against +); H2 against H6_lexical on P4, P5, P7; H3 against H6_lexical on P4; H4 against the rest on P1 and P2; H5 against H4 on P2, against H2, H3 and H6_lexical on P3, against H3 and H6_lexical on P7.

### 11.6 P5's items

Of the sample's 887 placement items, 94 take the object from a receptacle of the task's target type (for example "take cloth 1 from cabinet 1" for "clean some cloth and put it in cabinet."). There "done" says the object sits at a target-type receptacle, possibly this one, which makes the take more likely to work under H2, not less. 62 items have a history that contradicts "done" (section 3). 15 items are both. P5 reads the remaining 746. Secondary, descriptive: done − cancelled on the 94 take-from-target items (H2 predicts done above cancelled there), on the 62 contradicted items, and on all 887 placement items as first designed. Move, heat, cool and clean actions toward a target-type receptacle stay in P5: "done" says the object has been put down, so the agent no longer holds it, and these actions need it in hand.

### 11.7 Reading rules

Per format, each (account, contrast) the account predicts is **consistent** when every observed zone is allowed, **contradicted** when none is, and **undecided** otherwise (or when the contrast is INCONCLUSIVE).

**Formats: both must agree.** A contrast is decided for an account only when the two-candidate and yes/no readings agree (both consistent or both contradicted). One consistent and one contradicted is reported as **format-dependent** and counts as undecided; one decided and one undecided is undecided. Reasons for this rule over naming one deciding format per contrast: the two-candidate format can gain from copying words of the success text (desire_check and negated name the action's receptacle, which the take success text repeats; the desire lines' participles share a stem with the move, heat, cool and clean texts), while the yes/no format has its own answer biases (both models lean toward No on state questions), so neither format is clean for every contrast, and picking one per contrast would rest on guesses about which artefact is larger. Requiring agreement makes a contradiction harder to get and is the conservative choice when the question is which accounts survive. Each format's own readout is reported next to the joint one.

**Verdicts, per model:**

- **contradicted**: at least one contrast contradicted (in both formats); the report names them;
- **consistent**: no contrast contradicted and at least one contrast consistent (in both formats); the report gives the count, since one consistent contrast is weak support;
- **untested**: no contrast decided either way (only undecided or format-dependent contrasts). An untested account is not called consistent.

**When every content account is contradicted** (H2, H3, H4, H5 and H6_lexical), the report says "no single account; mixture or an account not on the list", lists the contrasts each account fails, and does not pick the least contradicted one. When the formats disagree on a contrast, the report names the contrast and the two readings. A positive P5 or a P7 above zero together with a P5 below zero are reported as such; no account predicts the first, and the second fits no single account. H1 is reported from the knowledge split alone, and the report says that this battery cannot test it.

### 11.8 Sampling claim, corrected

The note said inclusion probability was about 0.214 for every item. It is not: rounding the strata (game by failure reason, many of one to three items) makes inclusion probabilities range from 0 to 0.33 (mean 0.214, SD 0.046); 117 items in 72 strata of one or two items can never be drawn, among them all four items of one game (valid_unseen/73, whose two step-0 items still enter P6). The design is not reweighted. The battery's questions are about the sign and size of within-item contrasts, read against predictions that hold item by item, not about a population mean; inverse-probability weights would only reweight strata the sample already covers and cannot bring back the 117 items. The estimand is stated instead: unweighted means over the sample, which stand for the strata it covers, with each game represented about in proportion to its items (game sample share over population share between 0.52 and 1.33, apart from the one game left out).

### 11.9 Mistral-7B-Instruct-v0.3 as a secondary model

Mistral showed the largest practical effect in the families' wave A (item share toward Yes 0.879). It is scored on every prompt of the battery, both formats, at the families registry's pin (`mistralai/Mistral-7B-Instruct-v0.3`, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, `wf/constants.py`), with its chat-template handling from `wf/families.py`: the system text is passed as a system message and the template places it in the last user turn, as in its family run (`family_score_mistral-7b-v0.3`), no extra template settings, default attention. The tokenization gate runs for it as for the others. It is not part of any primary family; its contrasts, categories and readouts are reported descriptively, and no conclusion rests on them. The few-shot overlap and knowledge-split checks need earlier runs that did not score Mistral and are not computed for it.

### 11.10 Synthetic check of the rules

`scripts/mech2_simulate.py` plants patterns of r in synthetic scores for the battery's 1,403 real items (real games, populations and arms), both formats, with each model's spread and within-game correlation from the earlier runs (two-candidate: aligned effect 6.12 nats, item SD 6.0, correlation 0.26 for Qwen; 1.65, 1.94, 0.12 for Llama; yes/no: 2.80, 3.20, 0.10 and 2.21, 1.97, 0.19 from `rq1v_v2_metrics.json`), and reads them with the amended code: 40 replications per scenario and model, 500 bootstrap resamples and 500 permutations each (the real analysis uses 2,000 and 1,000). A false contradiction is an account contradicted on a contrast whose planted r lies in a zone the account allows.

Scenarios: no effect anywhere (noise); every primary contrast at r = +0.10 or −0.10 (inside the band; review 1's case of small but significant effects); and one planted fingerprint per account, each value well inside its predicted zones (for example H2: desire_keep, desire_check and mention_ignore 0.6, described 0.8, done 0 and cancelled 0.6, negated −0.5).

| Scenario | False contradictions, any account (Qwen / Llama), each format alone and both formats | Content accounts not contradicted under the two-format rule (Qwen; Llama) |
| --- | --- | --- |
| noise | 0 of 40 / 0 of 40 | none in 40; none in 40 |
| every contrast +0.10 | 0 / 0 | none in 35, H6_lexical in 4, H3 and H6_lexical in 1; none in 36, H6_lexical in 3, both in 1 |
| every contrast −0.10 | 0 / 0 | none in 40; none in 40 |
| H2 planted | 0 / 0 | H2 alone in 40; H2 alone in 40 |
| H3 planted | 0 / 0 | H3 alone in 40; 40 |
| H4 planted | 0 / 0 | H4 alone in 40; 40 |
| H5 planted | 0 / 0 | H5 alone in 40; 40 |
| H6_lexical planted | 0 / 0 | H6_lexical alone in 38, with H3 in 2; alone in 40 |
| H3 content with a position shift of −0.5 at step 0 | 0 / 0 | H3 alone in 40; 40 (P6 NEG in 37 of 40 for each model, NEG_OR_TRIVIAL in 3) |

No account was falsely contradicted in any of the 720 replications, in either format alone or under the two-format rule. In review 1's simulation of the first rules, small effects of ±0.10 were called POS_SMALL or NEG in up to 5 of 15 replications per contrast and counted against ≈0 predictions; with the band they are NULL, TRIVIAL, NEG_OR_TRIVIAL or TRIVIAL_OR_SMALL, which never contradict a ≈0 prediction. Each planted account was the only content account left uncontradicted in 398 of 400 planted-account replications; in the other 2, P4's interval straddled 0.25, leaving H3 undecided on it. Where nothing is planted and every contrast sits at +0.10, H6_lexical (and once H3) escaped contradiction in 4 or 5 of 40 replications because their positive predictions met intervals straddling 0.25, which leaves them undecided rather than contradicted. Limits of this check: the synthetic yes/no scores follow the same pattern as the two-candidate ones, so it does not test what happens when the formats truly disagree (that case is reported as format-dependent by rule); effects are additive and homogeneous across items; and mixtures were not planted, since by construction they can land anywhere between the accounts they mix.

### 11.11 Tests, deliberate bugs, GPU estimate

**Prompts.** `scripts/check_mech2_prompts.py` on every real item (2026-11-20): no problem; 20,731 two-candidate and 20,731 yes/no prompts per model (20,017 before, plus 714 negated prompts per format).

**Tokenization gate on the real tokenizers.** Run on ICE (login node, CPU, the job's environment) with the pinned Qwen3-8B, Llama-3.1-8B and Mistral-7B-Instruct-v0.3 tokenizers through `wf/mech2.py`'s own gate and each model's template handling: both candidates of all 1,403 items and all 20,731 yes/no prompts per model, no problem for any model. "Yes" and "No" are single tokens (Qwen 9454 and 2753, Llama 9642 and 2822, Mistral 6360 and 2538), and every item's first tokens are "You" against "Nothing". Longest prompt in a sample of 300: 1,138 tokens (Qwen), 1,010 (Llama), 958 (Mistral). The gate runs again in every shard before any score.

**Tests and deliberate bugs.** `tests/test_mech2.py` (57 tests) covers every amendment; the review's twenty mutants and fourteen new ones were added to `scripts/mutation_check.py`, and all 62 mech2 mutants are killed, each with its killing test logged in `notes/testing.md`.

**GPU estimate.** Per model 20,731 two-candidate scores (two forward passes each) and 20,731 yes/no scores (one pass); over three models 186,579 passes. At 0.26 to 0.4 s a pass that is 13.5 to 20.7 GPU-hours, plus about 1.2 for 24 model loads (three per shard) and a few minutes of CPU per model and shard for the gate (about 2.5 minutes per model for all prompts of the battery on the ICE login node): about 15 to 22 GPU-hours in all, 1.9 to 2.7 hours per shard. `pace/mech2_score.sbatch` now asks for 4 hours per shard; scores resume with `WF_RESUME=1`. Mistral's share is a third of the total (5 to 7 GPU-hours). `mech2_score_smoke` (one game, all three models) is still run first and read for format, gate and timing only.

## 12. Amendments after review 2, before any score (2026-11-20)

A second review read the amended design against the code, simulated the decision rules on the real items with scenarios the first synthetic check left out, and planted 22 more deliberate bugs. It passed the engineering with fixes and failed the science as designed: two of the primary contrasts were not the comparisons they were meant to be, so a true account could be contradicted by an artefact of the design. Nothing had been scored. Every change below was made, tested and committed before any GPU run of this battery. Where section 12 differs from sections 1 to 11, section 12 holds. Code: `wf/mech2.py`; synthetic check: `scripts/mech2_simulate.py`; checks on the real items: `scripts/check_mech2_prompts.py`.

### 12.1 P1 and P2 made minimal pairs

**The problem.** P1 compared desire_keep with mention_ignore, and P2 compared desire_check with the swapped task. The two desire lines replace the household task altogether ("Your task is to: make sure that the bowl is not taken; …"), while mention_ignore and the swapped arm keep the swapped household task ("Your task is to: put a clean pan in cabinet. …"). So each contrast changed two things at once: the desire, and whether any household task is present. The second change is not small. In `rq1x_score` the swapped task alone raised L above the neutral task by 2.56 nats for Qwen and 1.20 for Llama, which is 0.42 and 0.73 of the aligned effect on the same scale. If salience (H3) were the whole story, the desire lines would still lose that generic lift, so P1 and P2 would come out below zero by up to that amount, in zone N, and H3 would be contradicted although it is true. The review's simulation of pure H3 with that gap (0.5 of the aligned effect on the two desire lines) gave "no single account" in every replication.

**The fix.** Two arms were added so that each contrast changes one thing:

- **desire_keep_sw**: "{swapped} Make sure that the {obj} is not {pp}; leave it where it is." It is desire_keep's sentence placed after the swapped task line, exactly as mention_ignore places "Ignore the {obj}." after it. **P1 = desire_keep_sw − mention_ignore**: both lines are the swapped task plus one sentence about "the {obj}", and only what that sentence asks differs (keep the object untouched, or ignore it).
- **desire_check_can**: "Your task is to: find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it can." The existing desire_check ends "the user hopes it cannot." **P2 = desire_check − desire_check_can**: the two lines are word for word the same except "cannot." against "can.", so only the polarity of the user's hope differs. Both lines lack the household task equally, so the gap cancels.

Example, for "put a clean bowl in cabinet." with swapped object pan and action `take bowl 1 from cabinet 1`: desire_keep_sw reads "Your task is to: put a clean pan in cabinet. Make sure that the bowl is not taken; leave it where it is.", and desire_check_can reads "Your task is to: find out whether a bowl can be taken from the cabinet 1; the user hopes it can." The prompt check confirms on all 1,403 items of the union that desire_keep_sw is the swapped line followed by desire_keep's sentence and mention_ignore is the swapped line followed by "Ignore the {obj}.", and that desire_check and desire_check_can differ only in "cannot." against "can."; the tests check the same for every task type and verb.

The first P1 (desire_keep − mention_ignore) and the first P2 (desire_check − swapped) are kept as secondary readouts in both formats, with r, and so is desire_keep − desire_keep_sw, which measures the gap itself (what the bare desire line loses by dropping the household task).

### 12.2 New arm and contrast: affirmed, P8

P7 (negated − swapped) adds a whole sentence that names the object and the receptacle and denies that the object is there. A salience effect of the mention (up) and an evidence effect of the denial (down) add inside the same contrast, so P7 near zero could be both at once. The new arm keeps the mention and turns the claim around:

> affirmed: {swapped} The user says there is {a} {obj} in the {rec}.

It is built on exactly the negated arm's items (take actions of the goal object, 714 items of the union) and differs from the negated line only in "no" against the article ("a" or "an"), which the prompt check confirms on all 714. **P8 = negated − affirmed**, on P7's 503 sample items. Both lines mention the object and the receptacle equally, so salience and word overlap cancel, and only the claim's direction remains. Task as evidence (H2) predicts P8 below zero; the other accounts predict about zero.

The primary family is now eight contrasts per model and format. Holm's correction runs at a fixed family size of eight; a missing or undefined p enters as 1, so the family never shrinks.

### 12.3 Prediction table (amended after review 2)

Each cell is the set of size zones the account allows (section 11.4: N below the band, Z inside it, S small, L large); "+" means S or L; an empty cell is no prediction. This table is read cell by cell by `tests/test_mech2.py`, which asserts it equals `PREDICTIONS` in `wf/mech2.py` exactly.

| Contrast | H2 evidence | H3 referent salience | H4 wishful | H5 policy blending | H6_lexical | H6_position |
| --- | --- | --- | --- | --- | --- | --- |
| P1 keep vs ignore, after the swapped task | Z, S or L | Z or S | N | N or Z | Z or S | |
| P2 hope polarity, cannot vs can | Z | Z | N | Z | Z | |
| P3 mention without need | + | + | N or Z | N or Z | + | |
| P4 need without name | + | + | + | + | Z | |
| P5 done vs cancelled | N | Z | N or Z | Z | Z | |
| P6 position at step 0 | | | | | | N, S or L |
| P7 negated existence | N or Z | + | Z | Z | + | |
| P8 negated vs affirmed | N | Z | Z | Z | Z | |

Reasons for the cells that changed since section 11.5:

- **P1, task as evidence: Z, S or L.** "Make sure that the bowl is not taken; leave it where it is" says that the object sits somewhere it could be taken from, which is evidence that the take can work; "Ignore the bowl" only presupposes it. The earlier existence sentence alone reproduced nearly all of the effect for Qwen, so the difference can be large. Below zero is not allowed. The other accounts keep their section 11.5 cells: salience Z or S (both lines name the object; the keep sentence refers to it once more), wishful N (the user wants the action not to happen), policy blending N or Z (the keep sentence makes the action a counter-step), H6_lexical Z or S (the keep sentence's participle shares a stem with the action's verb).
- **P2, every account but wishful: Z.** The two lines differ only in what the user hopes, so only wishful prediction reads them differently (N). Policy blending used to predict a positive P2 because trying the action is the test the line asks for; that is true of both lines now, so it cancels. The first P2 (desire_check − swapped), kept as a secondary readout, still carries that reading.
- **P3, wishful: N or Z.** "Ignore the bowl" can be read as the user not wanting the bowl handled, which a wishful predictor would push slightly down.
- **P4, task as evidence, wishful and policy blending: "+" instead of L only.** A described object ("a deep round dish for soup or cereal") is less specific than its name, so the need it carries may be weaker without being absent; the earlier runs kept 61% and 92% of the effect, the first not far above the large threshold. P4 now separates only H6_lexical (about zero, since the line does not contain the name) from the other content accounts.
- **P5, wishful: N or Z.** Once the task is done, a user would not want the object moved again, so a wishful predictor may lower the success reading. Task as evidence keeps N, read against cancelled's own lift (12.7).
- **P7, task as evidence: N or Z.** The user's report may be discounted (it is false at the start of the game on about half of the items where the history does not show the receptacle), and the sentence still names the object and the receptacle. P8 now carries task as evidence's test of the claim without the mention.
- **P8**: task as evidence N; every other account Z.

**Which contrasts separate which accounts.** Two accounts are separated by a contrast when their allowed zones there do not overlap, so any decided reading contradicts at least one of them; partly separated when the zones differ but overlap. No two accounts have identical rows (checked by a test, `separations()` in `wf/mech2.py`, also written to the analysis report).

| Pair | Separated by | Partly separated by |
| --- | --- | --- |
| H2 evidence and H3 salience | P5, P7, P8 | P1 |
| H2 evidence and H4 wishful | P1, P2, P3, P8 | P5, P7 |
| H2 evidence and H5 policy blending | P3, P5, P8 | P1, P7 |
| H2 evidence and H6_lexical | P4, P5, P7, P8 | P1 |
| H3 salience and H4 wishful | P1, P2, P3, P7 | P5 |
| H3 salience and H5 policy blending | P3, P7 | P1 |
| H3 salience and H6_lexical | P4 only | none |
| H4 wishful and H5 policy blending | P2 only | P1, P5 |
| H4 wishful and H6_lexical | P1, P2, P3, P4, P7 | P5 |
| H5 policy blending and H6_lexical | P3, P4, P7 | P1 |

**Flagged: two pairs rest on a single contrast.** Referent salience and H6_lexical differ only on P4 (the described line), as they did before. Wishful prediction and policy blending are now fully separated only by P2, the hope's polarity; P1 and P5 separate them only partly (an N reading contradicts neither). If P2 ends undecided or format-dependent, the battery cannot tell these two pairs apart. H6_position predicts only P6, which no content account predicts; it answers a different question and is not exclusive of any content account.

### 12.4 H6_lexical redefined: overlap with the action string

H6_lexical is now **word overlap between the task line and the action string** (the object's type word, the action's receptacle instance, the verb's stem). The action string is in both formats' prompts, so this overlap is present in both formats, and H6_lexical's predictions apply to both. Copying from the **success text** is a different artefact: desire_check, desire_check_can, negated and affirmed name the action's receptacle ("the cabinet 1"), which the take success text repeats, and the desire lines' participles share a stem with the move, heat, cool and clean success texts. That copying can only act in the two-candidate format, whose candidate is the success text; the yes/no format's answers are "Yes" and "No". It is counted neither for nor against any account; it is one of the reasons the two-format rule exists (section 11.7).

P2 and P7 are the contrasts most likely to come out format-dependent. P7, because negated names the receptacle that the take success text repeats and swapped does not, so the two-candidate format can gain from copying alone. P2, because "cannot" and "can" sit a few lines above a yes/no question about whether the action will succeed, so in the yes/no format the word itself may tilt the answer toward "No" or "Yes", an answer-format artefact the two-candidate format does not share. In P1 and P8 both lines name the same words of the success text, so copying cancels.

### 12.5 Verdicts and wording

- **Consistent needs a rival that fails.** An account is called consistent only if it is consistent on at least one contrast where some rival account is contradicted. A reading that every account allows shows nothing about any of them: a large P4 fits four accounts at once, and only H6_lexical's contradiction there makes it informative. For P6, which only H6_position predicts, the rival is the plain reading "position does not matter" (zone Z). An account with no contradicted contrast and no such discriminating contrast is **untested**. The report lists, per account, the contrasts on which it was consistent while a rival failed.
- **Three lists.** The summary lists the content accounts consistent, untested and contradicted separately. The status line gives, per model, each primary contrast's two-candidate estimate and category with its yes/no category next to it, then the three lists and the position verdict.
- **A contradiction reads "cannot be the sole account".** The accounts are not exclusive, so a contradicted account is not rejected: the battery shows that it cannot explain the pattern alone, not that it plays no part.
- **Mixtures.** When every content account is contradicted, the report says "no single account; mixture or an account not on the list". It is fixed now that the readout is then continuous: the point estimates and 95% intervals of P5, P7 and P8, in nats and as r, in both formats (`mixture_readout`, reported for every model whatever the verdicts). These three are the contrasts that carry evidence against salience; their sizes say how much of each a mixture holds. A mixture does not always produce contradictions everywhere: the synthetic check (12.10) shows that a 50/50 mixture of task as evidence and salience is read as task as evidence alone. So "consistent" never excludes a mixture, and the mixture readout is reported in every case.

### 12.6 Where the error control comes from

Holm's correction rarely decides anything here. A category depends on where r's unadjusted 95% interval lies relative to the band of plus or minus 0.25 and the large threshold of 0.5; Holm's p enters only to tell NULL from TRIVIAL inside the band (both read as about zero) and to leave a contrast undecided when its interval lies wholly outside the band but Holm's p is at least 0.05. With the effects these models show, an interval wholly outside the band nearly always comes with a tiny p. So the protection against false contradictions comes mainly from two rules applied to unadjusted 95% intervals: the band (an interval must leave it entirely before a contrast counts as non-zero, and a contradiction of an about-zero prediction needs the whole interval outside it), and the two-format rule (a contrast is decided only when both formats agree). Neither the intervals nor the verdicts are adjusted for the eight contrasts or the six accounts. The synthetic false-contradiction rates (12.10) are the evidence for how well this works.

### 12.7 P5 read against cancelled's own lift

Task as evidence predicts done below cancelled because "It is already done" says the object has been put at the target, which undoes what the mention of the real task gave the cancelled line. Write $c$ for cancelled's lift over the swapped task on P5's items, $c = L(\text{cancelled}) - L(\text{swapped})$. The analysis now also reports P5 as a fraction of that lift, in both formats,

$$q = \frac{L(\text{done}) - L(\text{cancelled})}{L(\text{cancelled}) - L(\text{swapped})},$$

with its bootstrap interval, undefined when the lift is at or below zero, and the lift itself with r. Under task as evidence $q$ lies between −1 and 0 when done gives back part or all of the lift, and below −1 when done also says the object is out of reach. A P5 below zero with no lift to give back ($c$ about zero) is not the removal of evidence and is read as one of the other routes below.

Routes to a P5 below zero other than task as evidence, named in advance so that a negative P5 is not read as specific to it:

- the completion wording itself: "It is already done; nothing is left to do" may cue that the episode is over and nothing further happens, a generic lean toward "Nothing happens." (cancelled shares "nothing is left to do", so only "It is already done" differs);
- wishful prediction: a user whose goal is achieved would not want the object moved again (now allowed in H4's cell);
- a falsehood check: on the 62 items whose history contradicts "done", left out of P5, a model that checks the claim against the history may react to the contradiction; on the 746 P5 items the history does not contradict it, but a model may still treat the claim as implausible;
- the two lines quote the same real task, so word overlap and copying do not differ between them.

P8 is task as evidence's clean test; P5 is a second, less specific one.

### 12.8 P6, what moving the line also changes

At step 0, moving the task line to the top does three things at once: it moves the line away from the action (from the last line before "Action:" to the first), it puts it above the room description, and it makes the prompt match the prompt examples' layout, whose task line comes first. P6 cannot tell these apart; a P6 beyond the band is read as "the effect depends on where the task line sits", not as distance alone. Every step-0 item is a "never moved" failure (the agent has not moved, so the object is not at hand), so P6 speaks only to that failure type. The band in nats: Qwen's step-0 aligned effect is about 11.0 nats, so the band of plus or minus 0.25 is about plus or minus 2.7 nats, and a position effect of up to 2.7 nats counts as about zero; for Llama, whose step-0 effect is about 1.5 nats, the band is about plus or minus 0.4 nats.

### 12.9 P7: items whose receptacle was not seen, and the first-token term

P7 and P8 are also reported, in both formats, on the 293 take items whose history does not show the receptacle's contents (146 where the user's claim is true at the start of the game, 147 where it is false), next to the full 503. On these the user's report is the only information the prompt gives about the receptacle. On the 210 items whose history already shows the receptacle without the object, the claim adds nothing a reader of the history does not know, so task as evidence expects a smaller shift there.

The first-token term is fixed in advance as follows. In `rq1d_score` a denial ("There is no mug in this room.") lowered both models' full L, but on the outcome word it raised Qwen's (+0.71) and lowered Llama's (−0.38) (`exploratory_results.md`, section 3). So for P7 and P8 the first-token term (two-candidate) is reported, does not enter any category, and is flagged when its sign differs from the full L's (`first_token_P7_P8`, `sign_split_with_full_L`). With such a split, the two-candidate reading of P7 is described as resting on the continuation after the first token (the wording penalty of the success text), not on the first-token decision; the yes/no format, a single-token decision, is the check, and the two-format rule already requires it to agree before P7 decides anything.

### 12.10 Synthetic check, extended

`scripts/mech2_simulate.py` was extended and rerun with the amended code (2026-11-20): synthetic scores for the battery's 1,403 real items, both formats, each model's spread and within-game correlation as in section 11.10, 40 replications per scenario and model, 500 bootstrap resamples and 500 permutations each; 20 scenarios, 1,600 replications in all. Two changes in how the check counts:

- **A false contradiction is judged against the account's intended fingerprint**, not against the planted values. When a scenario adds an artefact (a gap in the desire lines' generic lift, word copying in one format, an answer bias in the other), an account contradicted on a contrast where its own pattern, without the artefact, lies in an allowed zone counts as falsely contradicted. Under the two-format rule a contrast counts as truly allowed when either format's true value is allowed.
- **The sole-survivor rate** is how often the planted account is the only content account not contradicted and is called consistent (for the position scenario, how often H6_position is called consistent).

Scenarios. Every primary contrast at one value of r: 0 (noise), +0.10 and −0.10 (inside the band), −0.20 (inside the band near its lower edge, true zone Z), +0.25 (the band's upper edge, true zone Z), +0.30 and +0.45 (small, S) and +0.50 (the large threshold, L). One fingerprint per account, each value well inside its predicted zones, and H3 with a position shift of −0.5 at step 0. Review 2's desire-arm gap on the new contrasts: pure salience and pure policy blending with the three bare desire lines (desire_keep, desire_check, desire_check_can) 0.5 of the aligned effect lower, since they lack the household task. Formats that disagree: salience in the two-candidate format and task as evidence in the yes/no format; task as evidence with word copying in the two-candidate format only (negated and affirmed, which name the receptacle, +0.6 each); salience with a yes/no answer bias toward "No" after "cannot" (desire_check −0.4 in the yes/no format only). And review 2's 50/50 mixture of the task-as-evidence and salience fingerprints.

**False contradictions.** Under the two-format rule, one in 1,600 replications (counting every account in each): at r = +0.25 exactly, the band's edge, Llama's P1 interval lay wholly in S in both formats once, contradicting policy blending's "N or Z". In each format alone, false contradictions appear only where they are expected: at the band's edge (+0.25: up to 9 of 40 replications for one account in one format for Qwen, 10 of 40 for Llama), near it (−0.20: up to 3 of 40; +0.30: up to 2 of 40, Llama), and in the yes/no format under the planted answer bias (P2 below zero in 34 of 40 replications for Qwen and 37 for Llama, which contradicts every account predicting about zero there). The two-format rule removed all of these but the one. Every other scenario had no false contradiction in either format.

**Sole survivors.** Each planted content account was the sole survivor in 40 of 40 replications for both models, except policy blending for Llama (39; once wishful prediction escaped contradiction as well). Review 2's desire-arm gap no longer produces a false "no single account": with the gap, salience was the sole survivor in 40 of 40 for both models and policy blending in 39 of 40 (Qwen) and 40 of 40 (Llama), where before the amendment the salience case gave "no single account" in every replication. With word copying in the two-candidate format, task as evidence was still the sole survivor in 40 of 40 for both models (P7 came out about zero in the two-candidate format and below zero in the yes/no format, both allowed by its "N or Z"). With the yes/no answer bias, salience was still the sole survivor in 40 of 40 (P2 was format-dependent in every replication). The position scenario was read as position-dependent (H6_position consistent) in 32 of 40 replications for Qwen and 38 of 40 for Llama; in the rest one format's P6 interval touched the band, leaving it undecided; salience stayed the sole content survivor in all 80.

**Boundaries and how often contrasts are decided.** The share of (account, contrast) readings decided under the two-format rule was 89 to 97% for the planted fingerprints, but 19% at r = +0.25 (276 and 277 of 1,440 for Qwen and Llama), 20 to 21% at +0.30, 39 to 42% at −0.20, 65 to 76% at ±0.10, 84 to 85% at +0.45 and 91 to 93% at +0.50. Near the band's edges the battery mostly says "undecided", as designed. The verdicts at the edge need care: with every contrast at +0.25, salience and H6_lexical were called consistent in 38 of 40 replications for each model, on P1 alone (P1 at +0.25 is allowed by their "Z or S", and wishful prediction's "N" is contradicted there), although their positive predictions for P3, P4 and P7 are wrong at +0.25 (undecided, not contradicted). So a "consistent" verdict that rests on one or two contrasts is weak, and the report gives the contrasts it rests on (`consistent_where_a_rival_fails`). At −0.20, −0.10, +0.45, +0.50 and with no effect, every content account was contradicted in every replication ("no single account"), which is correct, since no account predicts every contrast at one common value; at +0.10, policy blending or H6_lexical escaped contradiction in 10 to 11 of 40 replications, undecided where their positive predictions met intervals straddling 0.25.

**Formats that disagree.** With salience in one format and task as evidence in the other, P5, P7 and P8 were format-dependent in nearly every replication (460 and 463 format-dependent readings over 40 replications), and both accounts were called consistent in 40 of 40, each on the contrasts where they agree and a third account fails. Neither is contradicted, which is the conservative outcome the rule was chosen for; the report must then be read for its list of format-dependent contrasts, which names exactly the contrasts that separate the two.

**The 50/50 mixture of task as evidence and salience.** The mixture's true values are P5 −0.30, P7 +0.10 and P8 −0.60. P8 was NEG in 40 of 40 replications in both formats, P7 mostly NULL or TRIVIAL_OR_SMALL, and P5 NEG_OR_TRIVIAL in most and NEG in some. The verdict was "task as evidence consistent, salience contradicted" in 40 of 40 for both models: salience is contradicted by P8 and P7, while task as evidence allows P7 about zero and P1 anything but below zero. So the battery does not flag this mixture as a mixture. That is why the readout under a mixture is fixed as the estimates of P5, P7 and P8 (12.5) and they are reported in every case: here the mean r of P8 was −0.60, half of the pure task-as-evidence value (−1.2), and P7 was +0.10 against −0.5 under pure task as evidence; a reader comparing these sizes with the single-account fingerprints can see that task as evidence alone does not produce them. The verdict wording, "consistent", never claims that an account is the only one at work (11.1).

Limits of this check, as in section 11.10: effects are additive and the same on every item, the formats differ only where a scenario says so, and the planted fingerprints are the design's own idea of each account.

### 12.11 Analysis integrity

`mech2_analysis` now checks every score against the current items before it computes anything (`score_integrity`, written to `integrity.json` and the report). Per model it counts scores of items not in the battery, scores whose item hash differs from the current item's (`_item_hash`: action, history, task lines and candidates), duplicate (item, condition) pairs, scores missing and scores not expected, and per arm and format the scores expected (the model's jobs) against those present. A mismatch for any model, or a primary model with no score at all, stops the analysis with nothing analysed; a secondary model with no score at all is reported as not scored. The scoring run now records the item files' hashes (`transitions.jsonl` of each source) with its inputs, and a resume refuses to continue when they changed (`_init_run`); the analysis records the scoring files' hashes the same way.

### 12.12 GPU budget

- **Mistral on the arms the primary contrasts read.** Mistral-7B-Instruct-v0.3 is descriptive and enters no conclusion, so it is scored on 13 arms: aligned, swapped, desire_keep_sw, mention_ignore, desire_check, desire_check_can, described, done, cancelled, negated, affirmed and the two step-0 position arms. Its secondary readouts that need neutral, desire_keep or mention_exists are empty.
- **Position arms at step 0 only.** After step 0 no primary contrast read them; the `_end` arms are gone (at step 0 `_end` repeated the plain prompt). The secondary position readouts are now P6 in the yes/no format and its two parts at step 0 (aligned_start − aligned, swapped_start − swapped).
- **swapped_described dropped.** described − swapped_described was measured in `rq1x_score` on all primary items; the battery's P4 reads described against the named swapped task.

Per model, the primary models are scored on 16 arms: 19,292 two-candidate and 19,292 yes/no prompts (`scripts/check_mech2_prompts.py`, 2026-11-20; 20,731 before). Mistral: 15,083 of each. Over three models the two-candidate scores take two forward passes each and the yes/no scores one,

$$2 \times (2 \times 19{,}292 + 15{,}083) + (2 \times 19{,}292 + 15{,}083) = 107{,}334 + 53{,}667 = 161{,}001$$

forward passes in all (186,579 before). At 0.26 to 0.4 s a pass that is 11.6 to 17.9 GPU-hours, plus about 1.2 for 24 model loads and a few minutes of CPU per model and shard for the tokenization gate: about 13 to 19 GPU-hours, 1.6 to 2.4 hours per shard. `pace/mech2_score.sbatch` keeps 4 hours per shard. Mistral's share is about 28% (3.7 to 5.4 GPU-hours).

### 12.13 Prompts, tests, deliberate bugs and the gate

**Prompts.** `scripts/check_mech2_prompts.py` on every real item (2026-11-20): no problem. 19,292 two-candidate and 19,292 yes/no prompts per primary model on 16 arms, 15,083 of each for Mistral on 13 arms. The minimal pairs were checked on every item that carries them: P1's pair on all 1,403 items of the union, P2's on all 1,403, P8's on all 714 take items; the position arms exist on the 514 step-0 items only; negated and affirmed are built on exactly the same 714 items. The sample's P7 and P8 items are 503 (522 take items less 19 whose history shows the object in the receptacle), 293 of them with the receptacle's contents not shown.

**Tokenization gate on the real tokenizers, rerun.** The new arms add yes/no prompts the gate had not seen, so it was rerun on ICE (CPU job 7205732 on the `coc-cpu` partition, the job's environment, offline, the pinned Qwen3-8B, Llama-3.1-8B and Mistral-7B-Instruct-v0.3 tokenizers, `wf/mech2.py`'s own gate with each model's template handling): both candidates of all 1,403 items for each model and every yes/no prompt each model is scored on (19,292, 19,292 and 15,083, among them 3,520 per model from desire_keep_sw, desire_check_can and affirmed), no problem for any model. "Yes" and "No" are the same single tokens as before (Qwen 9454 and 2753, Llama 9642 and 2822, Mistral 6360 and 2538), and every item's first tokens are "You" against "Nothing". Longest prompt in a sample of 300: 961 tokens (Qwen), 1,013 (Llama), 1,066 (Mistral). The gate still runs in every shard before any score.

**Tests and deliberate bugs.** `tests/test_mech2.py` has 77 tests, all passing; they cover every amendment above. The review's 22 mutants (15 had survived the earlier tests) and 29 new ones were added to `scripts/mutation_check.py` (three mutants of the removed `_end` arms were replaced); all 110 mech2 mutants are killed, each with its killing test logged in `notes/testing.md`. The full test suite passes (773 passed, 7 skipped, 2026-11-20).

## 13. Amendments after review 3, before any score (2026-11-21)

A third review read the design after section 12 against the code, reran the decision rules on the real items with new scenarios, and planted 22 more deliberate bugs. Its verdicts: engineering passed with fixes (14 of its 22 mutants survived the tests, so the tests had gaps), science passed with fixes (one problem that must be fixed before any score, five that should be), and the GPU budget passed. Nothing had been scored. Every change below was made, tested and committed before any GPU run of this battery. Where section 13 differs from sections 1 to 12, section 13 holds. Code: `wf/mech2.py`; synthetic check: `scripts/mech2_simulate.py`; checks on the real items: `scripts/check_mech2_prompts.py`.

### 13.1 The verdict could not tell task as evidence from salience plus ordinary reading of claims (must-fix)

**The problem.** Three primary contrasts add an explicit claim to the task line: P5 ("It is already done"), P7 ("The user says there is no {obj} in the {rec}") and P8 (the same report denied against affirmed). Referent salience (H3), as defined in 11.1, is content-blind: what the line says about the object does not matter, only that the object is made salient. So H3 predicts that a denial and an affirmation move the success reading equally, and it is contradicted on P5, P7 and P8 by any model that reads claims the way a competent reader does, lowering its success reading after "there is no apple in the countertop 2" and raising it after "there is an apple". That reading of claims is not the account under test. The question the battery asks is why naming the action's object in an ordinary task line ("put a clean apple in fridge.") raises the success reading; a model can read explicit claims normally and still owe the goal effect to salience. Review 3 planted exactly that (the salience fingerprint, with the denial at 0.1 and the affirmation at 0.9 of the aligned effect, "done" at 0.2 and "cancelled" at 0.7) and the battery called task as evidence consistent and salience contradicted in every replication for both models. The verdict "H2 consistent, H3 contradicted" therefore cannot by itself mean that the goal effect comes from evidence.

**The fix: M, a mention-associated residual after averaging opposing claims** (called "the mention component" until review 4; see 14.1 and 14.8 for why the name and its reading changed). On P7's 503 items, both formats, with r over the aligned-minus-swapped effect on the same items:

$$M = \tfrac{1}{2}\big[L(\text{negated}) + L(\text{affirmed})\big] - L(\text{swapped}).$$

The negated and affirmed lines name the object and the receptacle in the same words and differ only in "no" against the article. Averaging them averages the two opposing claims: only if the denial lowers L by exactly as much as the affirmation raises it does the claim cancel, leaving the effect of adding a sentence that mentions the object and the receptacle. Under pure salience M is the mention's whole effect (0.7 of the aligned effect in the planted fingerprint); under task as evidence with a symmetric reading of the claims it is near zero (−0.5 and +0.7 in the fingerprint give +0.1), but with an asymmetric reading it can be anywhere (14.8). M is computed in `model_block` and reported in the mixture readout (`mixture_readout["M_mention_residual"]`, `M_mention_component` until review 4, with its two halves since review 4: estimate and interval in nats, r with its interval, the zones its r interval touches) in both formats, for every model, whatever the verdicts. It is not in the Holm family; its zones are read from r's unadjusted 95% interval.

**Two fixed sentences in the summary** (`joint_readout`, which now receives M). Both apply only when salience is contradicted (in both formats) on at least one contrast and on no contrast outside P5, P7 and P8 (`CLAIM_CONTRASTS`):

- (a) when, in addition, M's r interval lies wholly above 0.25 in both formats (and r is estimable in both), the summary added ~~"Salience cannot be the sole account, but a mention component remains with the claim balanced (r = M [CI] in each format): salience is likely part of the effect."~~ *Amended after review 4 and the adversarial check of M (sections 14.1 and 14.8), since a positive M does not show salience:* "Salience cannot be the sole account. A mention-associated residual of r = M [CI] remains after averaging opposing claims; it is consistent with salience but also with asymmetric processing of affirmations and denials, so it does not show that salience contributes. Word overlap with the action would give the same."
- (b) when, in addition, task as evidence is called consistent, the summary adds: "H3 is contradicted only on contrasts that add an explicit claim (done; the user's report). This shows the model reads such claims, not that the goal effect comes from evidence rather than salience; see M."

Both can appear together. The summary also carries the flags (`salience_contradicted_only_on_claims`, `mention_component_above_band_both_formats` (now `mention_residual_above_band_both_formats`), `salience_likely_part`, renamed `mention_residual_remains` after review 4, `claims_read_not_evidence_shown`) and M's values.

**Caveats, fixed now.** M assumes that the denial lowers L about as much as the affirmation raises it. If a model reacts more strongly to one direction, the claim does not cancel: task as evidence's own fingerprint gives M of about +0.1, and a model that is moved more by "there is an apple" than by "there is no apple" gives a positive M with no salience at all. *Corrected after the adversarial check of M (14.8; an earlier version of this paragraph read M as bounding the salience part, which was wrong):* M is neither an estimate nor a bound of the salience part. An asymmetric reading of the claims moves it up (affirmations weigh more) or down (denials weigh more) with no salience at all, and could equally offset a real salience effect; M is a mention-associated residual after averaging opposing claims, consistent with salience but never evidence for it on its own. M also contains word overlap (H6_lexical): both lines name the object and the action's receptacle, so a positive M is "salience or word overlap"; P4 is still the only contrast that separates those two. And the claim in both lines is attributed to the user, so a model that discounts the user's reports shrinks both halves of the claim but not the mention.

### 13.2 Consistent needs a risky prediction (should-fix)

Section 12.5 called an account consistent when it was consistent on at least one contrast where some rival was contradicted. Review 3 showed that this is too easy when the account's allowed zones include Z (about zero): such an account cannot be contradicted when nothing happens on that contrast, so being consistent there is not a test of it. In review 2's synthetic check, with every contrast at r = +0.25, salience and H6_lexical were called consistent on P1 alone, where both allow "Z or S" and wishful prediction's "N" fails.

**The rule now.** An account is **consistent** only if at least one of its discriminating contrasts (consistent there while a rival is contradicted on the same contrast) is a **risky prediction**: a cell whose allowed zones exclude Z. Otherwise it is **untested**, and the report gives the reason: "consistent only where it allows zero" when it had discriminating contrasts that were all non-risky, or "no contrast decided where a rival fails". The risky cells of the amended table (13.3) are:

- task as evidence: P4, P5 and P8;
- salience: P3, P4 and P7;
- wishful prediction: P1, P2 and P4;
- policy blending: P4 only;
- H6_lexical: P3 and P7;
- H6_position: P6.

Policy blending can now be called consistent only through P4, its one cell that excludes zero, and only when H6_lexical is contradicted there (P4 is the only contrast where any rival's cell is disjoint from its "+"); when P4 decides, it also makes task as evidence, salience and wishful prediction consistent on it. So a "consistent" verdict for policy blending is weak unless wishful prediction is contradicted on P2 at the same time, and the report's per-account contrast list shows which case holds.

**What the report prints.** Each account is listed with the contrasts its verdict rests on (`_label`): a consistent account with its risky discriminating contrasts and, separately, the discriminating contrasts where it allows zero, for example "consistent: H2_evidence [risky: P5, P8; allows zero: P1]"; an untested account with the contrasts where it was consistent while allowing zero, "H3_salience [consistent only where it allows zero: P1]"; a contradicted account with the contrasts that contradict it, "H3_salience [P5, P7, P8]". The same text is the status line's joint reading. The per-account rows add `risky_where_a_rival_fails` and `untested_reason`.

### 13.3 Prediction table (amended after review 3)

Each cell is the set of size zones the account allows (section 11.4: N below the band, Z inside it, S small, L large); "+" means S or L; an empty cell is no prediction. This table is read cell by cell by `tests/test_mech2.py`, which asserts it equals `PREDICTIONS` in `wf/mech2.py` exactly. The only change from 12.3 is task as evidence on P3.

| Contrast | H2 evidence | H3 referent salience | H4 wishful | H5 policy blending | H6_lexical | H6_position |
| --- | --- | --- | --- | --- | --- | --- |
| P1 keep vs ignore, after the swapped task | Z, S or L | Z or S | N | N or Z | Z or S | |
| P2 hope polarity, cannot vs can | Z | Z | N | Z | Z | |
| P3 mention without need | Z, S or L | + | N or Z | N or Z | + | |
| P4 need without name | + | + | + | + | Z | |
| P5 done vs cancelled | N | Z | N or Z | Z | Z | |
| P6 position at step 0 | | | | | | N, S or L |
| P7 negated existence | N or Z | + | Z | Z | + | |
| P8 negated vs affirmed | N | Z | Z | Z | Z | |

**Why task as evidence now allows P3 about zero.** P3 compares the swapped task plus "Ignore the {obj}." with the swapped task alone. Section 11.5 predicted P3 positive under task as evidence because the earlier sentence "There is a {obj} somewhere in the room." reproduced most of Qwen's effect. But "Ignore the bowl" does not assert anything; it only presupposes that a bowl exists, and it tells the agent that the bowl does not matter. Whether a model draws evidence from a presupposition inside an instruction to disregard the object is not fixed by the account's definition (11.1 and 1: the task line read as information about hidden facts); a reader who takes "ignore it" as "it is not relevant to what happens" may draw none. Predicting "+" made task as evidence fail on a reading that its own definition allows. Its cell is now "Z, S or L": anything but below zero, since nothing in the line suggests the object is absent.

**Which contrasts separate which accounts**, recomputed (`separations()`), with the same meaning as in 12.3:

| Pair | Separated by | Partly separated by |
| --- | --- | --- |
| H2 evidence and H3 salience | P5, P7, P8 | P1, P3 |
| H2 evidence and H4 wishful | P1, P2, P8 | P3, P5, P7 |
| H2 evidence and H5 policy blending | P5, P8 | P1, P3, P7 |
| H2 evidence and H6_lexical | P4, P5, P7, P8 | P1, P3 |
| H3 salience and H4 wishful | P1, P2, P3, P7 | P5 |
| H3 salience and H5 policy blending | P3, P7 | P1 |
| H3 salience and H6_lexical | P4 only | none |
| H4 wishful and H5 policy blending | P2 only | P1, P5 |
| H4 wishful and H6_lexical | P1, P2, P3, P4, P7 | P5 |
| H5 policy blending and H6_lexical | P3, P4, P7 | P1 |

P3 moved from "separated" to "partly separated" for task as evidence against wishful prediction and against policy blending; both pairs remain separated by other contrasts (P1, P2 and P8; P5 and P8). The two thin pairs flagged in 12.3 are unchanged: salience and H6_lexical differ only on P4, wishful prediction and policy blending only on P2 (13.4 adds a secondary readout for the latter).

### 13.4 A second wishful-against-blending readout: desire_check − desire_keep (should-fix)

P2 is the only primary contrast that fully separates wishful prediction from policy blending. The existing secondary desire_check − desire_keep is now reported in both formats, with r, as `S1_desire_check_vs_keep`, and named as a second separator of the two. Both lines are bare desire lines without the household task, so the generic lift of having a household task (review 2's gap) cancels between them. Both want the action not to happen ("make sure that the apple is not taken; leave it where it is" and "find out whether an apple can be taken from the countertop 2; the user hopes it cannot"), so wishful prediction expects about zero. For policy blending the two differ: trying the action is exactly the test the check line asks for, while the keep line makes the action a counter-step, so it expects the check line above the keep line. Its predictions for every account are in the table of 13.6. It is secondary: it does not enter the Holm family or the verdicts, and the two lines also differ in wording beyond the desire (the check line names the receptacle and does not presuppose the object), which is why it is not a primary contrast.

### 13.5 The negation placebo (should-fix, adopted)

P8 (negated − affirmed) carries task as evidence's cleanest test, but the two lines differ in the word "no". A model whose success reading drops after a negation word, whatever it denies, would show P8 below zero without reading the claim as evidence about the object. The placebo pair repeats P8's construction for an object the action does not concern, the swapped task's object:

> placebo_negated: {swapped} The user says there is no {swapped_obj} in the {rec}.
>
> placebo_affirmed: {swapped} The user says there is {a} {swapped_obj} in the {rec}.

For "put a clean bowl in cabinet.", swapped object pan, action `take bowl 1 from cabinet 1`: "Your task is to: put a clean pan in cabinet. The user says there is no pan in the cabinet 1." against "… there is a pan in the cabinet 1." The pair is built on exactly the negated arm's items (the 714 take items of the union), differs within itself only in "no" against the swapped object's article, and differs from negated and affirmed only in the object named; the prompt check confirms all three on all 714 items and that the placebo lines never name the action's object. The readout `S2_negation_placebo` is placebo_negated − placebo_affirmed on P7's 503 items, both formats, with r; `S3_P8_net_of_placebo` is P8 minus that difference, the claim about the action's object net of any reaction to "no" itself:

$$S_3 = \big[L(\text{negated}) - L(\text{affirmed})\big] - \big[L(\text{placebo\_negated}) - L(\text{placebo\_affirmed})\big].$$

Predictions, written now (table in 13.6): a claim about the swapped object says nothing about whether the action's object is in the receptacle, so task as evidence and every other content account predict the placebo about zero; the negation artefact predicts it below zero. On the net readout task as evidence predicts below zero and the artefact about zero. Both are named secondaries, outside the Holm family and the verdicts. If the placebo comes out below zero in both formats, a P8 below zero is read through `S3_P8_net_of_placebo` rather than at face value.

**How it is paid for.** Mistral-7B-Instruct-v0.3, the descriptive secondary model, is now scored on its 13 arms on the 1,000 sample items only, plus P6's arms (aligned, swapped, aligned_start, swapped_start) on all 514 step-0 items, so that its P6 still reads the same items as the primary models' (`model_jobs`). Its secondary readouts that read step-0 items outside the sample (the by-step splits) lose those items. The placebo pair is scored for the two primary models only. Numbers in 13.9.

### 13.6 Named secondary readouts and their predictions

These readouts are reported for every model (`secondary_named` in each model's block; the known-minus-unknown readout inside `knowledge_split`) and read descriptively: the zones that r's unadjusted 95% interval touches, against each account's allowed zones, in each format and jointly by the two-format rule (`read_secondary`). They enter neither the Holm family nor the verdicts of 12.5 and 13.2. Each cell is the set of zones the account allows, "+" is S or L, and an empty cell is no prediction. This table is read cell by cell by `tests/test_mech2.py`, which asserts it equals `SECONDARY_PREDICTIONS` in `wf/mech2.py`.

| Readout | H2 evidence | H3 referent salience | H4 wishful | H5 policy blending | H6_lexical | H7_check | negation artefact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S1_desire_check_vs_keep` (desire_check − desire_keep, sample) | N or Z | Z | Z | + | Z or S | N or Z | Z |
| `S2_negation_placebo` (placebo_negated − placebo_affirmed, P7 items) | Z | Z | Z | Z | Z | Z | N |
| `S3_P8_net_of_placebo` (P8 − the placebo difference, P7 items) | N | Z | Z | Z | Z | | Z |
| `H7a_desire_check_shown_vs_not` (desire_check − swapped, shown without the object minus not shown) | Z | Z | Z | Z | Z | N | |
| `H7a_aligned_shown_vs_not` (aligned − swapped, the same split) | N | Z | Z | N or Z | Z | N or Z | |
| `H7b_desire_check_known_vs_unknown` (desire_check − swapped, known state minus unknown) | Z | Z | Z | Z | Z | N | |
| `H7c_aligned_completing_vs_other` (aligned − swapped, action completes the task minus other actions) | Z | Z | Z or S | Z, S or L | Z, S or L | N | |

Reasons, row by row:

- **S1.** Task as evidence: the keep line presupposes the object and says it sits where it could be taken, while the check line only raises the question, so the check line carries no more evidence and possibly less (N or Z). Salience: both lines name the object once (Z). Wishful prediction: both want the action not to happen (Z). Policy blending: the action is the test the check line asks for and a counter-step for the keep line (+). H6_lexical: the check line also names the action's receptacle, which adds overlap with the action string (Z or S). H7_check: the check line directs the model to check the receptacle, and every item is a determined failure, so checking can only find reasons for failure (N or Z). Negation artefact: both lines contain a negation ("not", "cannot"), so it cancels (Z).
- **S2 and S3.** As in 13.5. H7_check makes no prediction on S3: a report about the receptacle may direct a check of it whichever way the report goes.
- **H7a, both rows.** On P7's items, the effect where the history shows the receptacle without the object (210 items) minus where the history does not show the receptacle (293 items), each r over the aligned effect on both groups together (`group_difference`). H7_check: desire_check tells the model to check whether the object can be taken from this receptacle; where the history already shows the receptacle without it, the check finds the failure, so desire_check's lift is smaller there (N). For the aligned task it predicts a smaller effect there too or none (N or Z), since the real task points at the object, not at this receptacle. Task as evidence: the aligned task implies the object can be got, which the history contradicts for this receptacle on the shown items, so the effect should be smaller there (N); desire_check carries little evidence either way (Z). Salience, wishful prediction and H6_lexical do not depend on what the history shows (Z). Policy blending: taking from a receptacle seen without the object is not a sensible step, so the aligned effect may be smaller (N or Z), while desire_check asks for the try in both groups (Z).
- **H7b.** desire_check − swapped on sample items whose deciding fact the model answers correctly in `mech_score` (both the state question and its twin) minus the other items with a verdict, two-candidate format only (`knowledge_split["desire_check_vs_swapped"]["known_minus_unknown"]`; every entry of the knowledge split now carries this difference). H7_check: where the model knows the deciding fact, a line that directs it to check finds the failure (N). No other account ties desire_check's lift to the model's knowledge (Z). H1 (no state tracking) is read, as before, from the aligned effect's known share.
- **H7c.** The aligned effect on sample items whose action would complete the task if it succeeded (`completes_task`: the goal object moved into a receptacle of a placement task's target type; 152 of the 1,000) minus the other items. H7_check: the task directs the check toward the completing step's preconditions (holding the object, the object already cleaned, heated or cooled), so the effect is smaller on completing actions (N). Wishful prediction: completion is what the user wants most directly (Z or S). Policy blending: the completing action is the most sensible step for the task (Z, S or L). H6_lexical: the completing action names a receptacle of the task's target type, more overlap with the task line (Z, S or L). Task as evidence and salience do not depend on the action's role (Z).

### 13.7 H7_check, an exploratory account (optional item, adopted)

**H7_check: the task directs which part of the state the model checks.** On this account the task line does not change the model's beliefs about hidden facts (task as evidence) or raise the object's salience at answer time; it changes which facts in the history the model consults before predicting. Where the consulted facts settle the outcome, the prediction follows them; where the task points elsewhere, the model fills in by default. The motivation is the TextCraft runs: crafting the goal item with an ingredient the agent never obtained gives a negative goal effect for both models, the opposite of ALFWorld's positive effect, which fits a model that, told the goal, checks the inventory for the goal's ingredients and finds one missing. In ALFWorld the facts that decide a failure are rarely the ones the task line points to (the task names the object and the target, while failures turn on where the agent is, what it holds and what a receptacle contains), so H7_check is compatible with a positive goal effect there; it predicts that the effect shrinks or turns negative when a line directs the check to the deciding fact.

Its readouts are the three in 13.6, on existing arms (no new prompts):

- (i) `H7a_*`: the split of P7's items by what the history shows of the receptacle. **Confound:** the split is tied to the failure reason. Of the 210 items whose history shows the receptacle without the object, 162 fail on contents or open state; of the 293 whose history does not show it, 179 fail on distance (the agent is away from the receptacle) and 51 more are step-0 items that never moved. The readout is therefore also reported within the one failure reason where both groups are well populated, contents or open state (162 shown, 57 not shown; `H7a_within_contents_or_open_state`), and the groups' reasons are tabulated (`H7a_groups_by_reason`).
- (ii) `H7b_desire_check_known_vs_unknown`: the knowledge split of desire_check's lift.
- (iii) `H7c_aligned_completing_vs_other`: the aligned effect by the action's role, the analogue of TextCraft's goal role.

**Decision: H7_check stays out of the primary verdicts.** It could be added as a seventh column of `PREDICTIONS` without changing the Holm family of eight, since Holm runs over contrasts, not accounts. It is kept out anyway, for three reasons. First, it makes no derived prediction on most of the eight primary contrasts (its claims are about which facts are consulted, and the primary contrasts vary what the line says, not where the deciding fact sits), so its row would be mostly empty and it could be called consistent or contradicted on one or two cells. Second, adding an account changes the other accounts' verdicts: under 12.5 and 13.2 an account is consistent only where a rival is contradicted, so a new rival can turn an untested account into a consistent one after the reviews have read the rules. Third, its own tests are confounded (failure reason, above) and rest on a split of the history, not on a manipulation of the prompt. It is reported as an exploratory account with predictions written before any score, read descriptively against the table of 13.6, and named in the report next to the verdicts, not inside them.

### 13.8 Undefined scores and the integrity check (should-fix)

`score_integrity` counted scores whose L is undefined (`n_undefined_L`, NaN or missing) but did not act on them; such scores were silently dropped by the analysis. Now a score with an undefined L on an arm the primary contrasts read (`PRIMARY_ARMS`) makes the model a mismatch, so the analysis refuses before computing anything (`n_undefined_L_primary_arms`), and undefined scores on other arms are counted and named in the status line ("undefined L on n scores of secondary arms"). Two smaller changes came with the review's mutants: `n_unexpected` now counts only scores of battery items under a condition the model is not scored on, so that scores of unknown items are counted once, under `n_unknown_items` (before, an unknown item was counted under both, and removing either check changed nothing); and the expected jobs come from `model_jobs`, which knows Mistral's restriction to the sample.

### 13.9 GPU budget

The primary models are scored on 18 arms (the placebo pair added): 20,720 two-candidate and 20,720 yes/no prompts each (19,292 before; `scripts/check_mech2_prompts.py`, 2026-11-21). Mistral is scored on its 13 arms on the sample and on P6's four arms at the 403 step-0 items outside the sample: 11,878 prompts of each format (15,083 before). The two-candidate scores take two forward passes and the yes/no scores one, so the run needs

$$2 \times (2 \times 20{,}720 + 11{,}878) + (2 \times 20{,}720 + 11{,}878) = 106{,}636 + 53{,}318 = 159{,}954$$

forward passes (161,001 before). The placebo pair adds 2 × 1,428 × 3 = 8,568 passes; restricting Mistral removes 3 × 3,205 = 9,615. At 0.26 to 0.4 s a pass the scoring takes 11.6 to 17.8 GPU-hours, plus about 1.2 for 24 model loads and a few minutes of CPU per model and shard for the gate: about 13 to 19 GPU-hours, 1.6 to 2.4 hours per shard. `pace/mech2_score.sbatch` keeps 4 hours per shard. Mistral's share is now 22% (2.6 to 4.0 GPU-hours).

**The cost of step-0 items outside the sample, named.** The union is the 1,000 sample items plus the 403 step-0 items the sample missed, and the primary models score every arm on all 1,403. On those 403 items only P6's four arms enter a primary contrast (1,612 prompts per format and model); the other 4,798 prompts per format and model serve only secondary readouts at step 0 (each arm against the swapped task at step 0, and the primary content contrasts on all 514 step-0 items). That is 2 × 4,798 × 3 = 28,788 forward passes, 2.1 to 3.2 GPU-hours, 18% of the run. They are kept, so that the step-0 secondary readouts cover the same 514 items P6 reads, and they are the first thing to cut if the hours run short.

### 13.10 Synthetic check, rerun with the amended rules

`scripts/mech2_simulate.py` was rerun with the amended code (2026-11-21) on the battery's real items, with the same settings as in 12.10: both formats, each model's spread and within-game correlation, 40 replications per scenario and model, 500 bootstrap resamples and 500 permutations each. It now also reports, for every scenario, M per format (mean r and the zones of its interval), the zones of the negation placebo, how often each of the two fixed summary sentences is given, how often an account is untested because it was consistent only where it allows zero, and the most common summary text. The synthetic placebo arms equal the swapped arm unless a scenario says otherwise. Review 3's two new scenarios were added: salience plus ordinary reading of claims (`H3_plus_comprehension`: the salience fingerprint with negated 0.1, affirmed 0.9, done 0.2 and cancelled 0.7 of the aligned effect) and salience with a lean toward failure after negation words in both formats (`H3_negation_priming`: desire_keep_sw, desire_check and negated 0.4 lower than under salience, and the placebo's "no" line 0.4 below its affirmed line). 22 scenarios, 1,760 replications in all.

**Planted accounts keep their verdicts under the risky rule.** Task as evidence, salience, wishful prediction and H6_lexical were each the sole survivor (the only content account not contradicted, and called consistent) in 40 of 40 replications for both models; policy blending in 39 of 40 for both (once wishful prediction was also called consistent). With review 2's desire-arm gap, salience and policy blending were the sole survivor in 40 of 40 for both models. With word copying in the two-candidate format, task as evidence was the sole survivor in 40 of 40; with the yes/no answer bias after "cannot", salience was the sole survivor in 40 of 40 (P2 format-dependent in 31 and 38 replications). The position scenario was read as position-dependent (H6_position consistent) in 31 of 40 replications for Qwen and 33 of 40 for Llama (32 and 38 in 12.10, with different synthetic draws now that the placebo arms are generated), and salience stayed the only content account called consistent in all 80. With the formats disagreeing (salience in one, task as evidence in the other), both accounts were called consistent in 40 of 40, as before, with 438 and 452 format-dependent readings naming the contrasts that separate them.

**No false contradiction under the two-format rule in the twenty scenarios of 12.10** (1,600 replications); round 2's one false contradiction, at r = +0.25 exactly, did not recur. In each format alone the false contradictions near the band's edge are as in 12.10 (at +0.25 up to 11 of 40 for one account in one format, at −0.20 up to 2 of 40) and the yes/no answer bias contradicts the about-zero predictions on P2 in that format (31 of 40 for Qwen, 38 for Llama); the two-format rule removes all of them.

**What the risky rule changed.** At r = +0.25, the band's edge, salience and H6_lexical were consistent on P1 alone in 38 of 40 replications in 12.10; now both are untested, "consistent only where it allows zero", in 40 of 40 for both models, and no account is called consistent. With every contrast at +0.10, the accounts that escaped contradiction are untested with that reason (policy blending in 9 of 40 for Qwen and 6 for Llama, H6_lexical in 7 and 4, salience in 4 and 0). At +0.30, where the true values lie in S and salience's "+" cells are right, salience was called consistent in 2 of 40 replications for Qwen and 5 for Llama and untested in the rest. With no effect anywhere, every content account was contradicted in 39 of 40 (Qwen; once policy blending was untested) and 40 of 40 (Llama).

**M and the two sentences, by scenario.** Mean r of M (two-candidate; yes/no), and how often each sentence was given (the mention sentence, (a), which read "salience is likely part of the effect" until review 4 withdrew that reading, 14.1; claims read, evidence not shown), Qwen and Llama:

| Scenario | M, Qwen | M, Llama | Mention sentence (a) | Claims sentence | Verdict (both models) |
| --- | --- | --- | --- | --- | --- |
| task as evidence | +0.10; +0.10 | +0.10; +0.09 | 0 and 0 of 40 | 40 and 40 | task as evidence consistent, alone, 40 of 40 |
| salience | +0.71; +0.71 | +0.71; +0.70 | 0 and 0 | 0 and 0 | salience consistent, alone, 40 of 40 |
| wishful prediction | −0.00; +0.02 | +0.01; +0.01 | 0 and 0 | 0 and 0 | wishful consistent, alone, 40 of 40 |
| policy blending | −0.01; −0.01 | −0.00; +0.00 | 0 and 0 | 0 and 0 | blending consistent, alone, 39 of 40 |
| H6_lexical | +0.70; +0.70 | +0.70; +0.71 | 0 and 0 | 0 and 0 | H6_lexical consistent, alone, 40 of 40 |
| salience plus reading claims (review 3) | +0.49; +0.50 | +0.50; +0.49 | 40 and 40 | 40 and 40 | task as evidence consistent, 40 of 40 |
| 50/50 mixture of task as evidence and salience | +0.39; +0.40 | +0.39; +0.39 | 27 and 31 | 40 and 40 | task as evidence consistent, 40 of 40 |
| salience with negation priming (review 3) | +0.51; +0.50 | +0.50; +0.50 | 3 and 1 | 0 and 0 | no single account, 35 and 38 of 40 |
| task as evidence with copying (two-candidate) | +0.71; +0.09 | +0.71; +0.10 | 0 and 0 | 40 and 40 | task as evidence consistent, alone, 40 of 40 |
| no effect | −0.01; +0.00 | −0.00; +0.01 | 0 and 0 | 0 and 0 | no single account |

Reading. In review 3's scenario, salience plus ordinary reading of claims, the verdict is still "task as evidence consistent, salience contradicted" in every replication, because salience as defined (content-blind) is contradicted by any reading of claims. But the summary now says so: M lies wholly above the band in both formats in 40 of 40 replications for each model, so both sentences are given, and the report stated that salience is likely part of the effect and that the win on P5, P7 and P8 shows claim reading, not evidence. *The first half is withdrawn after review 4 (14.1, 14.8):* task as evidence read asymmetrically gives the same M, so a positive M is a mention-associated residual after averaging opposing claims, not a sign that salience is part of the effect. Under pure task as evidence M is about +0.10, inside the band, and only the claims sentence is given, as it should be: the battery cannot show from P5, P7 and P8 alone that the goal effect comes from evidence. The 50/50 mixture lands between the two (M about +0.39, the mention sentence in 27 and 31 of 40). Word copying in the two-candidate format inflates M in that format (+0.71) because both claim lines name the receptacle that the take success text repeats; the yes/no format does not copy (+0.09), and requiring both formats keeps the mention sentence out. Under negation priming the negation placebo was below zero (zones N, or N and Z) in every replication in both formats for both models, against at most 7 of 40 replications touching N (always together with Z, never N alone) in one format in any scenario without the artefact; so the placebo flags the artefact that would otherwise make P8 look like evidence.

Limits are those of 11.10 and 12.10: additive, homogeneous effects; the formats differ only where a scenario says so; the fingerprints, the claim-reading pattern and the negation lean are the design's own versions of each account.

### 13.11 Prompts, the gate, tests and deliberate bugs

**Prompts.** `scripts/check_mech2_prompts.py` on every real item (2026-11-21): no problem. 20,720 two-candidate and 20,720 yes/no prompts per primary model on 18 arms; 11,878 of each for Mistral on 13 arms (1,612 of them P6's arms on the 403 step-0 items outside the sample). The placebo pair was checked on all 714 take items of the union: built on exactly the negated arm's items, differing within the pair only in "no" against the swapped object's article, differing from negated and affirmed only in the object named, and never naming the action's object. The H7_check groups on the real items: 210 P7 items whose history shows the receptacle without the object and 293 where it does not show it (162 and 57 of them failing on contents or open state); 152 of the 1,000 sample items have an action that would complete the task (pick 40, pick-two 23, clean 39, cool 27, heat 23).

**Tokenization gate on the real tokenizers, rerun.** The placebo pair adds yes/no prompts the gate had not seen, and Mistral's prompts changed, so the gate was rerun on ICE (CPU job 7219082 on the `coc-cpu` partition, the job's environment read-only, offline, the pinned Qwen3-8B, Llama-3.1-8B and Mistral-7B-Instruct-v0.3 tokenizers, `wf/mech2.py`'s own gate with each model's template handling and `model_jobs`; 2026-11-21): both candidates of all 1,403 items for each model and every yes/no prompt each model is scored on (20,720, 20,720 and 11,878, among them 1,428 placebo prompts per primary model), no problem for any model. "Yes" and "No" are the same single tokens as before (Qwen 9454 and 2753, Llama 9642 and 2822, Mistral 6360 and 2538), and every item's first tokens are "You" against "Nothing". Longest prompt in a sample of 300: 919 tokens (Qwen), 1,117 (Llama), 1,193 (Mistral). The gate still runs in every shard before any score.

**Tests and deliberate bugs.** `tests/test_mech2.py` has 104 tests (27 new), all passing; they cover every amendment above, including a test that reads the secondary prediction table of 13.6 cell by cell. Review 3's mutants were added to `scripts/mutation_check.py` (twenty of its 22, adapted where the code moved; A15 and A19 are equivalent and left out, with the reason given there), with 38 mutants of the amendment; all 168 mech2 mutants are killed, each with its killing test logged in `notes/testing.md`. Review 3's surviving mutants A1 to A9, A11, A18 and A21 each have a test written to kill them. The full test suite passes (800 passed, 7 skipped, 2026-11-21).

## 14. Amendments after review 4, before any score (2026-11-21)

A fourth review read the design and code after section 13, ran new scenarios and 20 deliberate bugs, and passed engineering, science and readiness with fixes. After it, an adversarial check of M was asked for (14.8). Nothing had been scored, and no real model score was looked at. No prediction, contrast, category or verdict rule changed in this round; what changed is how the summary reads M and the placebo, what is reported next to them, and the tests. Where section 14 differs from sections 1 to 13, section 14 holds.

### 14.1 M does not show salience (must-fix, then the adversarial check)

Review 4 found that sentence (a) of 13.1 read M wrongly. Under task as evidence with an asymmetric response to the claims (the affirmation moving the reading more than the denial, which `rq1d_score`'s split between the full L and the first token makes plausible), M exceeds the band with no salience at all, and the summary then said that salience was likely part of the effect. The review proposed calling M an upper bound on salience; the adversarial check (14.8) showed that this is not right either, since an asymmetric reading can push M below zero, or offset a real salience effect, as easily as raise it. So:

- M is described everywhere (code, flags, this note, the deviations entry) as **a mention-associated residual after averaging opposing claims**: consistent with salience, but also explained by asymmetric processing of affirmations and denials, or by word overlap with the action.
- Sentence (a) now reads: "Salience cannot be the sole account. A mention-associated residual of r = M [CI] remains after averaging opposing claims; it is consistent with salience but also with asymmetric processing of affirmations and denials, so it does not show that salience contributes. Word overlap with the action would give the same." Its conditions are unchanged (salience contradicted in both formats, only on P5, P7 or P8, and M's r interval wholly above 0.25 in both formats). The flag is `mention_residual_remains` (it was `salience_likely_part`); the mixture readout's key is `M_mention_residual`.
- M's two halves are reported in the mixture readout, each as r in both formats on P7's items, so that a reader can see the asymmetry: the affirmation's lift, affirmed − swapped (`M_half_affirmed_minus_swapped`), and the denial's drop, swapped − negated (`M_half_swapped_minus_negated`). M is half their difference:

$$M = \tfrac{1}{2}\Big(\big[L(\text{affirmed}) - L(\text{swapped})\big] - \big[L(\text{swapped}) - L(\text{negated})\big]\Big).$$

With a symmetric reading of the claims, M equals the mention's effect and the halves differ by 2M; with no mention effect, unequal halves mean an asymmetric reading. A given pair of halves fits either explanation, so the halves describe the pattern and do not attribute it.

### 14.2 When the negation placebo lies below the band (should-fix 3)

`joint_readout` now receives the placebo (S2) and P8 net of it (S3), as it receives M. When the placebo's r interval lies wholly below −0.25 in both formats (r estimable in both), the summary adds: "The negation placebo lies below the band in both formats (r = …): the word "no" lowers the success reading whatever it denies, so P8 is read net of it. Verdicts that rest on P8: …; P8 net of the placebo: r = …." The verdicts that rest on P8 are the content accounts contradicted on P8, and those consistent with P8 among the contrasts where a rival fails. The verdicts themselves do not change (a test checks every account's verdict and per-contrast readings with and without the placebo). Flags: `negation_placebo_below_band_both_formats`, `verdicts_resting_on_P8`. The reviewer's scenario, policy blending with a lean toward failure at the word "no" (`H5_negation_art`), was added to the simulation (14.8).

### 14.3 When M lies inside the band (should-fix 4)

The symmetric sentence: under the same condition as (a) (salience contradicted only on claim contrasts), when M's r interval lies inside ±0.25 in both formats the summary adds: "No mention-associated residual remains after averaging opposing claims (r = …, inside the band in both formats); task as evidence with a symmetric reading of the claims predicts this, but salience offset by an asymmetric reading would give the same, so it does not show that salience is absent." Flag `mention_residual_absent`. Like (a), it reports the residual and does not turn it into a verdict.

### 14.4 S1 in the status line (should-fix 5)

Wishful prediction and policy blending are fully separated by P2 alone (13.3). When neither is contradicted (both consistent or untested), the status line adds S1's joint reading, desire_check − desire_keep: its zones in each format and the two accounts' joint readings against S1's predictions (`s1_line`). For example: "H4 and H5 not told apart by the verdicts; S1 (desire_check - desire_keep): zones two-candidate S/L, yes/no S; joint: H4_wishful contradicted, H5_policy_blend consistent".

### 14.5 Caveats of the placebo (should-fix 6)

- On 87 of P7's 503 items the swapped task's target receptacle is of the type of the receptacle the action names (for example the swapped task "put a clean pan in cabinet." and the action `take bowl 1 from cabinet 1`). There the placebo's claim, "there is no pan in the cabinet 1", bears on the swapped task itself, so it is not a pure placebo. The placebo is also reported without these items (`S2_negation_placebo_without_swapped_target_at_rec`, descriptive, with the number left out).
- The placebo detects a negation effect that does not depend on what is denied. It cannot detect one bound to the object: in the two-candidate format the candidate repeats the object's name ("You pick up the bowl 1 …"), so "no bowl" may lower that particular candidate through the shared word, which a denial of the pan would not do. In that format such an effect is indistinguishable from evidence. The guard is the yes/no format, whose answers do not repeat the object, together with the two-format rule, which decides P8 only when both formats agree.

### 14.6 Review 4's surviving mutants and new tests (should-fix 2)

The review's 20 mutants (R1 to R20) were added to `scripts/mutation_check.py`, adapted where the code moved, except R10 and R19, which are equivalent (R10 makes the risky rule read `PREDICTIONS` only, but `_risky` is only called for accounts, whose cells are `PREDICTIONS`' own; R19 only adds a comment). Tests were written for each survivor: salience's contradictions read from the joint rows, not one format (R1); the claims sentence needs task as evidence consistent jointly (R2); no claim or M sentence when salience is not contradicted (R3); an interval starting exactly at 0.25 is not above the band (R6); the placebo reads P7's items (R7); M's zones per format (R9); the placebo is built only on takes of the goal object (R13); P8 net of the placebo per format (R15); the within-reason H7 readout keeps its sign (R16); a stale item hash of the secondary model refuses the analysis (R18).

### 14.7 The job script (should-fix 7)

`pace/mech2_score.sbatch`'s per-shard counts were those of review 2. They now read: about 2,590 two-candidate and 2,590 yes/no prompts per primary model and shard (20,720 of each over the run, 18 arms), about 1,485 of each for Mistral (11,878), about 20,000 forward passes per shard (159,954 over the run), 1.4 to 2.2 hours at 0.26 to 0.4 s a pass. The 4-hour limit is unchanged.

### 14.8 Adversarial check of M: pure task as evidence with asymmetric claims (2026-11-21)

**The concern.** M averages the two opposing reports and subtracts the swapped task. If the model reacts to "there is an apple in the countertop 2" and "there is no apple in the countertop 2" by different amounts, the claims do not cancel. Then, under pure task as evidence with no salience at all, M can be large and positive (affirmations weigh more) or negative (denials weigh more), and it can equally hide a real salience effect. The check asks whether the battery, with its real analysis and verdict code, misreads such a pattern as salience.

**What was run.** `scripts/mech2_simulate.py`, the same procedure as every other scenario (13.10): synthetic scores on the battery's 1,403 real items, both formats, each model's spread and within-game correlation, 40 replications per scenario and model, 500 bootstrap resamples and 500 permutations, read with `primary_family` and `joint_readout` exactly as `mech2_analysis` reads them (M, the placebo and P8 net of it passed to the summary). Every scenario plants pure task as evidence: the H2 fingerprint (desire_keep 0.6, desire_keep_sw 0.7, desire_check and desire_check_can 0.5, mention_ignore 0.6, described 0.8, done 0, cancelled 0.6, each as a fraction of the aligned-minus-swapped effect, which is 1), with no salience, no word overlap or copying, no wishful or policy-blending component and no generic negation priming, the same pattern in both formats. Only the affirmed and negated arms change, set by an asymmetric response to the claim:

- (i) a grid: affirmed − swapped in {+0.6, +1.0, +1.4, +1.8} and negated − swapped in {−1.0, −0.6, −0.2, 0} (16 scenarios);
- (ii) a denial-to-affirmation ratio family: the affirmation lifts the reading by 0.8 and the denial lowers it by 0.8 times g, g in {0.25, 0.5, 1, 2} (4 scenarios);
- (iii) a logistic saturation family: the model's belief that the object is in the receptacle sits at log-odds x0 under the swapped task, each claim moves it by 2 in its direction, and L follows the probability, so the affirmation's response is 2[σ(x0 + 2) − σ(x0)] and the denial's 2[σ(x0 − 2) − σ(x0)], for x0 in {−2, −1, 0, 1, 2} (5 scenarios). At low x0 affirmations move the reading more than denials; at high x0 denials do (the reading saturates near success).

Each of the 25 was run twice: with the placebo at zero, and with a placebo effect (`_pl`: the placebo's "no" line 0.4 below its affirmed line, the planted negated arm unchanged), 50 scenarios and 4,000 replications in all. Together they put the true M between −0.40 and +0.90, with twelve above +0.25 and three at or below −0.24. Outputs: the session scratchpad, folder `r4m` (`sim/`, one file per scenario, `aggregate.txt` and `table_asym.md`, generated by `tables.py`, which also checks that the table below is the one in this note); the scratchpad is temporary, and the simulation is seeded per model, scenario and replication, so `scripts/mech2_simulate.py 40 500 <scenario>` reproduces every number (a full rerun on 2026-11-22 reproduced them exactly).

**Results.** In every one of the 4,000 replications, for both models, with and without the placebo effect, task as evidence was called consistent and was the only content account not contradicted; salience was contradicted (on P5, P7 or P8 only), never called consistent next to task as evidence, and the summary never contained a phrase attributing any of the effect to salience. Mean M recovered the true value within 0.04 in each format, and the two halves recovered the planted responses (for example +1.81 and +0.01 at affirmed +1.8, negated 0).

| Scenario | Affirmed − swapped | Swapped − negated | True M | Mean M, Qwen (two-candidate; yes/no) | Mean M, Llama | Task as evidence consistent and sole survivor | Salience attributed | Residual sentence (Qwen; Llama) | No-residual sentence | Placebo variant: placebo sentence (Qwen; Llama), verdicts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `asym_grid_a+0.6_n-1.0` | +0.60 | +1.00 | -0.20 | -0.24; -0.20 | -0.22; -0.20 | 40/40, 40/40 | 0 | 0; 0 | 0; 1 | 14; 13, unchanged |
| `asym_grid_a+0.6_n-0.6` | +0.60 | +0.60 | +0.00 | -0.01; -0.01 | -0.00; +0.00 | 40/40, 40/40 | 0 | 0; 0 | 37; 37 | 10; 16, unchanged |
| `asym_grid_a+0.6_n-0.2` | +0.60 | +0.20 | +0.20 | +0.20; +0.19 | +0.20; +0.18 | 40/40, 40/40 | 0 | 0; 0 | 3; 3 | 7; 20, unchanged |
| `asym_grid_a+0.6_n+0.0` | +0.60 | -0.00 | +0.30 | +0.29; +0.30 | +0.30; +0.30 | 40/40, 40/40 | 0 | 0; 2 | 0; 0 | 13; 19, unchanged |
| `asym_grid_a+1.0_n-1.0` | +1.00 | +1.00 | +0.00 | -0.01; -0.00 | +0.01; +0.01 | 40/40, 40/40 | 0 | 0; 0 | 34; 38 | 17; 19, unchanged |
| `asym_grid_a+1.0_n-0.6` | +1.00 | +0.60 | +0.20 | +0.20; +0.21 | +0.20; +0.19 | 40/40, 40/40 | 0 | 0; 0 | 1; 3 | 17; 14, unchanged |
| `asym_grid_a+1.0_n-0.2` | +1.00 | +0.20 | +0.40 | +0.41; +0.40 | +0.39; +0.40 | 40/40, 40/40 | 0 | 31; 31 | 0; 0 | 11; 19, unchanged |
| `asym_grid_a+1.0_n+0.0` | +1.00 | -0.00 | +0.50 | +0.50; +0.50 | +0.50; +0.50 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 12; 22, unchanged |
| `asym_grid_a+1.4_n-1.0` | +1.40 | +1.00 | +0.20 | +0.20; +0.22 | +0.20; +0.20 | 40/40, 40/40 | 0 | 0; 0 | 1; 1 | 12; 22, unchanged |
| `asym_grid_a+1.4_n-0.6` | +1.40 | +0.60 | +0.40 | +0.40; +0.38 | +0.40; +0.40 | 40/40, 40/40 | 0 | 25; 32 | 0; 0 | 16; 19, unchanged |
| `asym_grid_a+1.4_n-0.2` | +1.40 | +0.20 | +0.60 | +0.61; +0.61 | +0.59; +0.60 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 16; 21, unchanged |
| `asym_grid_a+1.4_n+0.0` | +1.40 | -0.00 | +0.70 | +0.71; +0.69 | +0.71; +0.69 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 17; 18, unchanged |
| `asym_grid_a+1.8_n-1.0` | +1.80 | +1.00 | +0.40 | +0.40; +0.41 | +0.40; +0.39 | 40/40, 40/40 | 0 | 31; 32 | 0; 0 | 22; 25, unchanged |
| `asym_grid_a+1.8_n-0.6` | +1.80 | +0.60 | +0.60 | +0.61; +0.60 | +0.61; +0.59 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 16; 20, unchanged |
| `asym_grid_a+1.8_n-0.2` | +1.80 | +0.20 | +0.80 | +0.79; +0.82 | +0.81; +0.79 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 17; 12, unchanged |
| `asym_grid_a+1.8_n+0.0` | +1.80 | -0.00 | +0.90 | +0.90; +0.92 | +0.90; +0.88 | 40/40, 40/40 | 0 | 40; 40 | 0; 0 | 12; 20, unchanged |
| `asym_ratio_g0.25` | +0.80 | +0.20 | +0.30 | +0.29; +0.29 | +0.29; +0.30 | 40/40, 40/40 | 0 | 2; 1 | 0; 0 | 17; 16, unchanged |
| `asym_ratio_g0.50` | +0.80 | +0.40 | +0.20 | +0.20; +0.20 | +0.20; +0.21 | 40/40, 40/40 | 0 | 0; 0 | 5; 3 | 13; 17, unchanged |
| `asym_ratio_g1.00` | +0.80 | +0.80 | +0.00 | -0.01; -0.00 | -0.01; +0.00 | 40/40, 40/40 | 0 | 0; 0 | 37; 38 | 14; 15, unchanged |
| `asym_ratio_g2.00` | +0.80 | +1.60 | -0.40 | -0.41; -0.40 | -0.41; -0.39 | 40/40, 40/40 | 0 | 0; 0 | 0; 0 | 12; 21, unchanged |
| `asym_logit_x-2` | +0.76 | +0.20 | +0.28 | +0.29; +0.29 | +0.29; +0.28 | 40/40, 40/40 | 0 | 0; 0 | 0; 0 | 19; 21, unchanged |
| `asym_logit_x-1` | +0.92 | +0.44 | +0.24 | +0.24; +0.23 | +0.25; +0.24 | 40/40, 40/40 | 0 | 0; 0 | 0; 0 | 15; 20, unchanged |
| `asym_logit_x+0` | +0.76 | +0.76 | +0.00 | -0.02; -0.01 | -0.01; +0.01 | 40/40, 40/40 | 0 | 0; 0 | 34; 38 | 17; 18, unchanged |
| `asym_logit_x+1` | +0.44 | +0.92 | -0.24 | -0.23; -0.24 | -0.26; -0.24 | 40/40, 40/40 | 0 | 0; 0 | 0; 0 | 13; 17, unchanged |
| `asym_logit_x+2` | +0.20 | +0.76 | -0.28 | -0.27; -0.28 | -0.28; -0.30 | 40/40, 40/40 | 0 | 0; 0 | 0; 0 | 15; 14, unchanged |

The columns: the planted responses, the true M, mean M over replications, the joint verdict (task as evidence consistent and sole survivor, 40 of 40 for each model), how often the summary attributed the effect to salience, how often the residual sentence (M's interval wholly above the band in both formats) and the no-residual sentence (inside the band in both formats) were given, and, for the placebo variant, how often the placebo sentence was given and whether the verdicts changed.

**Reading.** Under the wording of 13.1 before review 4, the residual sentence would have told a reader that "salience is likely part of the effect" in 25 to 40 of 40 replications of each of the nine pure-task-as-evidence scenarios whose M is +0.40 or more (40 of 40 from +0.50 up), and in 1 or 2 of 40 of some scenarios with M near +0.30: exactly the misreading the check was asked to find. With the amended wording the same replications say that a mention-associated residual remains, that it is consistent with salience but also with asymmetric processing of affirmations and denials, and that it does not show that salience contributes. Where M is about zero (a symmetric response, for example affirmed +1.0 and negated −1.0) the no-residual sentence appears, and it says that this does not show salience is absent. Where M is negative the battery says nothing about M beyond reporting it and its halves. So the diagnostic did not misidentify planted pure task as evidence in its verdicts; it would have misread it in its prose, and the prose is what changed. No experimental arm, primary contrast, prediction or verdict rule changed: the check found no validity problem in them.

**The placebo.** With a placebo effect of −0.4 (0.15 beyond the band's edge, so its interval clears the band in only part of the replications), the placebo sentence was given in 7 to 25 of 40 replications per scenario and model, and the verdicts were the same as without it in every scenario. The sentence names the verdicts that rest on P8 and gives P8 net of the placebo; under pure task as evidence P8 net of the placebo stays below zero, so it does not change what a reader should conclude.

**Other scenarios, rerun with the amended summary** (40 replications, both models; summary texts in `aggregate.txt`):

| Scenario | Verdict | Mean M (Qwen; Llama, two-candidate) | Residual sentence | No-residual sentence | Claims sentence | Placebo sentence |
| --- | --- | --- | --- | --- | --- | --- |
| task as evidence, symmetric fingerprint | task as evidence alone, 40 and 40 | +0.10; +0.10 | 0; 0 | 30; 35 | 40; 40 | 0; 0 |
| salience | salience alone, 40 and 40 | +0.71; +0.71 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| salience plus reading claims (review 3) | task as evidence consistent, 40 and 40 | +0.49; +0.50 | 40; 40 | 0; 0 | 40; 40 | 0; 0 |
| 50/50 mixture of task as evidence and salience | task as evidence consistent, 40 and 40 | +0.39; +0.39 | 27; 31 | 0; 0 | 40; 40 | 0; 0 |
| salience with negation priming (review 3) | mostly no single account | +0.51; +0.50 | 3; 1 | 0; 0 | 0; 0 | 12; 18 |
| policy blending plus a negation lean (review 4) | policy blending alone in 18 and 19 of 40, otherwise no single account | −0.21; −0.19 | 0; 0 | 0; 0 | 0; 0 | 14; 19 |
| wishful prediction | wishful alone, 40 and 40 | −0.00; +0.01 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| policy blending | policy blending alone, 39 and 39 | −0.01; −0.00 | 0; 0 | 0; 0 | 0; 0 | 0; 0 |

The salience-plus-claims scenario and the asymmetric task-as-evidence scenarios now get the same sentences, which is the point: the battery cannot tell them apart from P5, P7, P8 and M, and the summary no longer pretends to. Within P5, P7, P8 and M (halves included) nothing separates salience from an asymmetric reading of the claims: a content-blind salience effect leaves the halves' sum (affirmed − negated, that is −P8) unchanged and makes the halves unequal by 2M, and an asymmetric claim reading also makes them unequal. The contrasts without a claim, P3 against its predictions and P4 for word overlap, are where the accounts differ. With a negation lean, policy blending is contradicted in about half of the replications (on P7 or P8, which the planted lean pushes below zero); the placebo sentence appears in about a third to a half, naming the verdicts that rest on P8.

**Regression tests.** `test_pure_task_as_evidence_with_asymmetric_claims_is_never_read_as_salience` (four planted pure-task-as-evidence patterns, affirmed/negated +1.8/0, +1.4/−0.2, +0.6/−1.0 and +0.2/−1.0, through `model_block` and `joint_readout`: task as evidence consistent and the only consistent account, no salience-attribution phrase, M and its halves recovered, the residual sentence given with its new wording where M exceeds the band, no M sentence where M is below it) and `test_the_m_sentences_never_attribute_the_effect_to_salience` (both M sentences, the exact wording pinned in `test_summary_sentences_on_salience_contradicted_only_by_claims` and `test_mention_band_edges`).

### 14.9 Tests, deliberate bugs, prompts and the gate

**Tests and deliberate bugs.** `tests/test_mech2.py` has 122 tests collected (18 new in this round), all passing. Review 4's mutants were added to `scripts/mutation_check.py` (18 of its 20; R10 and R19 are equivalent and left out, with the reason given there), with 12 mutants of the review-4 amendment and 2 of the adversarial check, and 8 round-3 mutants adapted where the code moved; all 200 mech2 mutants are killed, each by a named failing test logged in `notes/testing.md`. The full test suite passes (818 passed, 7 skipped, 2026-11-22).

**Prompts and the gate.** No arm, template, item or job changed in this round, so the prompt check (13.11) and the tokenization gate on ICE (job 7219082, 13.11) still describe exactly what the run scores; neither was rerun.

## 15. Amendments after review 5, before any score (2026-11-22)

A fifth review passed the code, arms, contrasts, predictions and verdict rules as ready to freeze, with fixes to the note and the tests. Nothing had been scored. Two passages of section 14 were corrected because they said the halves of M could separate salience from an asymmetric reading of the claims. They cannot: a content-blind salience effect leaves the halves' sum (affirmed − negated, that is −P8) unchanged and makes them unequal by 2M, and an asymmetric reading also makes them unequal (14.1, 14.8; P7 is no longer listed as a separating contrast, since it carries a claim). The optional sentence for M below the band was adopted: under the same condition as the other M sentences (salience contradicted only on claim contrasts), when M's r interval lies wholly below −0.25 in both formats the summary adds "After averaging opposing claims the mention-associated residual lies below the band in both formats (r = …): the denial moves the reading more than the affirmation; this does not attribute anything to salience." (flag `mention_residual_below_band`). Tests now pin the whole summary text for M above, inside and below the band (so that no sentence can be appended unnoticed), the S1 status line when one of wishful prediction and policy blending is consistent and the other untested, and P8 net of the placebo being quoted only when it has an interval in both formats. `tests/test_mech2.py` has 126 tests (4 new), all passing. The review's surviving mutants X1, X2, X8 and X9 and two mutants of the new sentence (Y1, Y2) were added to `scripts/mutation_check.py` (206 mech2 mutants), and U5 was anchored to the placebo's test now that M's below-band test uses the same comparison; these seven were run and each is killed by a named test (`notes/testing.md`). Only `wf/mech2.py`, its tests, the mutation list and these notes changed, so the full suite was not rerun (last run: 818 passed, 7 skipped, 14.9). No arm, template, item, contrast, prediction or verdict rule changed; the prompt check and the gate (13.11) still apply.
