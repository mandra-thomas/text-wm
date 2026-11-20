# Mechanism battery: why does naming the action's object push predictions toward success? (design, 2026-11-20)

Exploratory, not preregistered. This note fixes the design, the predictions and the readouts before any score of this run exists. Code: `wf/mech2.py`; runs `mech2_score` (8 game shards), `mech2_score_smoke` (one game, format and timing only) and `mech2_analysis`; checks on the real items: `scripts/check_mech2_prompts.py`; tests: `tests/test_mech2.py`.

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

- A **stratified sample of 1,000 items** (`SAMPLE_N`): strata are (game, failure reason), 495 strata; each stratum gets its proportional share of 1,000, rounded by largest remainder (ties broken by a hash of the stratum), and within a stratum the items with the smallest `sha256("20261120|mech2|<item id>")` are taken. Inclusion probability is about 0.214 for every item, so sample means estimate the population's item-weighted means. The sample holds 1,000 items in 124 games: 113 never-moved, 416 away from the receptacle, 197 not holding the object, 236 contents or open state, 38 hands full; by task type 132 pick, 153 pick-two, 251 clean, 168 heat, 183 cool, 113 examine; 111 are step-0 items.
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

**Exploratory: mechanism battery, which account the goal effect follows (decided 2026-11-20, before any of these scores).** `wf/mech2.py`, runs `mech2_score` (8 shards), `mech2_score_smoke` and `mech2_analysis`; design, templates, prediction table and readouts in `notes/mechanism2_design.md`. Not preregistered. Few-shot Qwen3-8B and Llama-3.1-8B at their pinned revisions, example set 1, history only, two-candidate and yes/no formats. Items: a 1,000-item sample of RQ2's primary items stratified by game and failure reason (seed "20261120|mech2"), plus all 514 step-0 items (1,403 in all). Six accounts (no state tracking, task as evidence, salience, wishful prediction, policy blending, prompt artefacts) predict different signs for task lines that differ from the aligned and swapped tasks only in the task line: a reversed desire with the object presupposed ("make sure that the {obj} is not {pp}; leave it where it is.") and without it ("find out whether {a} {obj} can be {pp} {prep} the {rec}; the user hopes it cannot."), the swapped task plus "Ignore the {obj}.", the object-free description of the real task, the real task marked done or cancelled, and the aligned and swapped lines moved to the top or bottom of the history. Primary: six contrasts per model on the two-candidate format, Holm within model (desire_keep − swapped, desire_check − swapped, mention_ignore − swapped, described − swapped, done − cancelled on placement tasks, and the step-0 change in the goal effect when the task line moves to the top), each classified NEG, NULL, POS_SMALL, POS_LARGE, POS or INCONCLUSIVE by fixed rules and read against the prediction table; the yes/no format is the replication family. Analysis-only checks on earlier runs: few-shot task overlap (`rq1_score`, 206 aligned-overlap item-set pairs) and the knowledge split (`mech_score`). Estimated 10 to 15 GPU-hours.
