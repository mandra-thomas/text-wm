# Natural ALFWorld bridge: who chose the action, and existence settled without naming the object (design, written 2026-11-25 before any score)

**Status.** Exploratory study with a frozen procedure. This note, the code (`wf/bridge.py`, `wf/bridge_analysis.py`, `wf/scoring/prefix.py`), the configurations (`wf/config/bridge_*.yaml`), the job script (`pace/bridge.sbatch`) and the tests (`tests/test_bridge.py`) are frozen together at an annotated tag before any score. The smoke readout is added to section 13 before the full run; the analysis runs once, from a clean checkout at the tag. Nothing here changes a confirmatory result. A companion CPU check of an assumption the existence arm relies on is in [a1prime_check.md](a1prime_check.md).

## 0. What was known when this was written

RQ2 scored 4,666 ALFWorld actions whose failure the history already settles (for example, taking an object from a receptacle the agent is not at). Each item was scored under the **aligned** task line (the episode's real task, which names the action's object) and the **swapped** line (the same template naming another object in the room). The **goal effect** is the mean, over items, of the model's log-odds of success under the aligned line minus that under the swapped line. It was 6.12 nats for Qwen3-8B and 1.65 for Llama-3.1-8B (`rq2_results.md`). In plain text, without the empty think block that Qwen3's template adds when thinking is off, Qwen's effect is about 3.3 nats, with the same standardized size (about 1.05) and share of items moved (about 89%) (`qwen_effect_audit_pipeline.md` §7).

Two objections stop that effect from being read as an error of the model, and this study tests one natural-data version of each.

**Who chose the action.** Every RQ2 action is a probe written by the item collector, which saw the whole room and chose failing actions on the task's object. The prompt shows only "Action: …". A model that reads an unexplained action as chosen by an expert who could see the room, as ALFRED's demonstrations were, would legitimately take an action that fits the aligned task as evidence that it works. Under that reading the task line is informative even about a failure the history settles, because the model may not be certain of what the history shows. The project's question is what the environment would reply if the action were taken regardless of who chose it, so this reading is a candidate explanation, not a licence; but it has never been tested by telling the model who chose the action.

**What the task says about existence.** A task naming an object implies that the object exists. In an earlier exploratory test (`exploratory_results.md` §3), a sentence stating the object's existence placed in both prompts removed the goal effect on the first answer word (Qwen −0.16 nats and Llama 0.07, against 2.42 and 0.66 without it, on the 4,382 items with one known location). Every sentence used then named the object, so the collapse cannot be told apart from the extra mention saturating whatever the name does. A sentence that settles existence without naming or describing the object separates the two.

## 1. Questions

1. **Provenance.** When the prompt says the action was chosen at random by a tester, or by an agent that has seen only the history shown, does the goal effect shrink relative to a prompt that carries a content-free sentence in the same place? If it does, the effect behaves like inference from the action's provenance. If it does not, while the model shows that it reads such statements (a positive control) and answers a question about them correctly (a comprehension gate), provenance does not account for the effect.
2. **Existence without reference.** When both prompts state "Every object any task in this house could need is present in the room", does the goal effect shrink on the items where the model, asked directly, says the object is present? If it does, the existence information the task line carries can account for the effect. If it persists, the effect needs the object's reference (entity or role), and the earlier collapse needed the name.

## 2. Assumption A1′, checked on CPU first

Reading an existence collapse as legitimate screening needs assumption A1′: once the object's existence is known, where it is (which receptacle, whether inside a closed one, how many instances) does not depend on which task was drawn. If A1′ fails, a correct model keeps a legitimate residual after existence is settled, equal to the task's information about location. [a1prime_check.md](a1prime_check.md) measures this on ALFWorld's generated initial states; its result enters the reading of the existence arm in section 10.

## 3. Items

**Sample.** 1,500 of the 4,666 RQ2 primary items, stratified by history-length bin (0, 1–3, 4–7, 8 or more actions) and the fact that decides the failure (`determination_reason`: away from the receptacle, never moved, contents or open state, hands full, not holding the object). Each stratum gets its proportional share rounded down; the remaining places go to the largest remainders; within a stratum, items are taken in the order of the hash of `20260924|bridge-sample|<item id>`. The 14 strata hold between 13 and 781 items in the population, so the sample holds between 4 and 251. Deterministic; tested (`tests/test_bridge.py`).

**Subset.** The first 300 sample items in the order of the hash of `20260924|bridge-subset|<item id>`. It carries the comprehension questions, Qwen's chat-format link (section 4) and the cross-GPU rescore (section 11).

**Smoke items.** The first 20 subset items, scored in a separate run that the analysis never reads.

The items, histories, candidates and swapped lines are RQ2's, read from the verified collection (`rq12_transitions_s0`–`s2`, which the run refuses unless their verification passed); nothing is rebuilt.

## 4. Models, formats and environment

**Qwen3-8B in plain text** for every item: no chat template; the system text, each example as its user text followed by one space and its answer, and the query, separated by blank lines, with the answer after one space. This is the pipeline audit's plain-text format, where the two answers' first tokens hold about 90% of the next-token probability, and the format the project treats as Qwen3's reference without the empty think block. The render is tested to equal the audit's.

**Is RQ2's thinking-off chat format (with the empty think block) also needed?** Not for the questions asked. Both questions are about how much of the effect a statement removes, measured as a share of the same model's effect in the same format, and as a standardized difference. The audit showed that the block stretches the scale of Qwen's scores without changing its standardized effect (1.02 with the block, 1.05 without, on all 4,666 items) or the share of items moved (88% and 89%). A share removed is scale-free, so it should not depend on the block if the block only stretches the scale. That is an assumption, and it fails at step 0, where the block changes more than the scale. So the block format is scored on the 300-item subset, every arm and readout, as a **descriptive link** to the RQ2 headline: the same estimands on the same items in both formats, with no decision taken from it. Its cost is about a fifth of one model's main run.

**Llama-3.1-8B** in its chat template, exactly as RQ2 rendered it (`LogoddsScorer`'s template settings).

**Environment.** One environment for every comparison (`numerical_sensitivity.md` Rule 1): the serve2 stack (torch 2.14.1, transformers 5.23.0; the job refuses any other), bf16 weights at the pinned revisions of `wf.constants.REVISIONS`, the default attention implementation, one sequence per forward pass with no padding, every main-run shard on an L40S (the run refuses any other GPU type, including on resume). The cross-GPU rescore runs on an A100 in the same stack.

**Scoring routine.** For each item and task line, the prompts of the provenance arms share everything up to the action line. `wf/scoring/prefix.py` runs that shared prefix once and scores every statement, readout and question from its key-value cache, cut back to the same point before each sequence. Every score of the study comes from this routine, so every arm is computed the same way. In float32 it equals a plain forward pass to about 0.00001 nats (tested on small random Llama and Qwen3 models). In bf16 the split changes the order of additions, as any change of input shape does, so its scores differ from a plain forward pass by rounding noise; the smoke run reports the size of that difference on one item per format.

## 5. Arms and exact texts

Every arm is scored under both the aligned and the swapped task line. The prompt is RQ2's (example set 1, history only).

**Provenance statements** sit on their own line between the action and the readout ("Action: …", a blank line, the statement, a blank line, then "Next observation:" or the question):

| Arm | Text |
|---|---|
| no statement | (nothing) |
| random tester, wording A | A tester picked this action at random from the possible actions. |
| random tester, wording B | A tester chose this action by drawing it at random from a list of commands, without looking at the room. |
| history-only agent, wording A | An agent who has seen only what is shown above chose this action. |
| history-only agent, wording B | This action was chosen by an agent whose only information about the room is the history shown above. |
| positive control | This action was chosen by an agent who only proposes actions that will work. |
| content-free sentence (placebo) | This command is written in lower-case letters, like every command in this game. |

Wording A of each statement is the text fixed when the study was proposed. Wording B is written to remove one ambiguity: "the possible actions" could be read as ALFWorld's list of currently admissible actions, which would imply the action works; the comprehension gate (section 8) catches that reading for wording A. The content-free sentence is added because an extra sentence near the action can dilute or otherwise change a task effect regardless of its content; the provenance contrasts are measured against it, so that a removal is attributed to what the statement says rather than to its presence.

**Existence statements** are appended to the task line in both prompts, as in the earlier dissociation test:

- **existence without reference:** "Every object any task in this house could need is present in the room." It neither names nor describes the object. It is true of every item (the action's object and the swapped object both exist in the room), but it is a universal claim the model may not apply to the specific object, which the existence check (section 6) measures.
- **existence with the name (reference arm):** "There is a {object} somewhere in the room.", the earlier test's sentence. It is the comparison that shows whether the collapse appears in this format and stack at all.

## 6. Readouts

Every arm is read three ways, and each gets its own conclusions:

- **Two-candidate:** the teacher-forced log-probability of the environment's real success text minus that of "Nothing happens.".
- **First token:** the log-probability of the success text's first token minus that of "Nothing", read from the same position. It carries no wording penalty and asks no question.
- **Yes/no:** RQ2's question ("Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens.\nAnswer:") in place of "Next observation:", read as log P(Yes) − log P(No). It is known to carry part of a task-progress judgement (`cross_model_summary.md`), so a removal seen only here is reported as such.

Three further yes/no prompts:

- **Comprehension questions** (subset, aligned task line, after each provenance statement and after no statement), in place of "Next observation:":
  - **C1:** "Question: did the choice of this action depend on anything about the room that is not shown above? Answer Yes or No." The stated answer is No for both tester and both agent wordings, Yes for the positive control.
  - **C2:** "Question: could this action have been chosen even if it will not work? Answer Yes or No." The stated answer is Yes for the tester and agent wordings, No for the positive control.

  C1 is the specified question about whether the chooser used knowledge of the room's contents. C2 has the opposite stated answer, so that a model leaning toward one answer cannot pass both (the pattern of τ-auth's K1–K4).
- **Existence check** (every item, both task lines, under no statement and under existence without reference): "Question: is there a {object} in the room? Answer Yes or No." It names the object, so it is a check only, never the estimand.

## 7. Estimands

Write $L_i(g, s)$ for one readout's log-odds of success on item $i$ under task line $g$ (aligned or swapped) and arm $s$. The goal effect under arm $s$ is

$$G_i(s) = L_i(\text{aligned}, s) - L_i(\text{swapped}, s).$$

For a statement type $k$ (random tester or history-only agent) with the set $W_k$ of its wordings that pass the comprehension gate, the per-item effect under the statement averages over those wordings:

$$\bar G_i(k) = \frac{1}{|W_k|} \sum_{w \in W_k} G_i(w).$$

The estimands, each by model and readout, with plain names:

1. **No-statement effect**, the mean of $G_i(\text{none})$: RQ2's effect in this format on the sample.
2. **Removed by the statement, beyond the content-free sentence** (primary for each statement type):

$$R(k) = \frac{1}{N}\sum_{i=1}^{N} \big[G_i(\text{placebo}) - \bar G_i(k)\big],$$

and its **share of the no-statement effect**,

$$\rho(k) = \frac{\sum_i \big[G_i(\text{placebo}) - \bar G_i(k)\big]}{\sum_i G_i(\text{none})}.$$

3. **Removed by the content-free sentence**, the mean of $G_i(\text{none}) - G_i(\text{placebo})$, and **removed by the statement against no statement**, the mean of $G_i(\text{none}) - \bar G_i(k)$ (both reported).
4. **Positive-control shift**: how much the positive control raises predicted success over the content-free sentence, averaged over both task lines,

$$PC = \frac{1}{N}\sum_i \frac{1}{2}\sum_{g} \big[L_i(g, \text{positive}) - L_i(g, \text{placebo})\big].$$

5. **Removed by existence settled without reference**, on the set $F$ of items whose existence check under that statement favours "present" (log-odds above zero) under both task lines:

$$R_{\exists} = \frac{1}{|F|}\sum_{i \in F} \big[G_i(\text{none}) - G_i(\text{exist})\big],$$

with its share of the no-statement effect on the same items.

6. **Removed by existence stated with the name** (reference), the same quantity for the named sentence, on $F$ and on all items.
7. **Wording spread**: the mean difference between the two wordings' effects, per statement type.
8. **Standardized removal**: a removal divided by the item-level standard deviation of the reference effect ($G_i(\text{placebo})$ for provenance, $G_i(\text{none})$ for existence). It is the scale on which Qwen's negative decisions are read (section 9).

Reported with each: the standardized effect and share of items moved toward success per arm; for the first-token and yes/no readouts the probability-scale goal effect (the mean of the difference of the two sigmoids); each task line's level shift against no statement; the median two-way probability mass per arm; and every estimand by history-length bin. The probability scale matters for the following reason. Suppose a model holds the deciding fact (say, that the agent is not at the receptacle) with a small residual doubt $\epsilon$, and the task acts only through what would happen if that fact were false. The probability of success is then about $\epsilon$ times the success probability in that world, so the task's effect on the probability vanishes as $\epsilon$ does, while its effect on the log-odds is the same for every small $\epsilon$. A removal seen in log-odds is therefore informative, but a persisting log-odds effect is not by itself excess.

**Intervals and tests.** 95% percentile intervals from a bootstrap over games (2,000 resamples, seed 20260924; one fixed set of resamples per item list, shared by every statistic on that list). The primary tests are two-sided sign-flip permutation tests of $R(\text{tester})$, $R(\text{agent})$ and $R_{\exists}$ in which all of a game's items flip together (10,000 permutations), for each model and readout: 18 tests, Holm-adjusted at a family-wise 0.05. The family always has 18 members, whether or not a gate passes.

## 8. Gates

1. **Comprehension (per model and wording).** On the 300 subset items, the answer to C1 and the answer to C2 each favour the stated answer on at least 80% of items. A statement type is read only through its passing wordings; with none passing, its categories are "not readable: comprehension gate failed". The positive control's answers and the no-statement answers are reported, not gated.
2. **Positive control (per model and readout).** $PC$ at least 1 nat with its 95% interval above 0 (τ-auth's threshold). If it fails on a readout, that readout's provenance categories are "not readable: positive control failed", because a statement that removes nothing could then not be told from a statement the model does not use.
3. **Existence check (per model).** At least 300 items in $F$; otherwise the existence categories are "not assessable".
4. **Format (descriptive red flag).** An arm whose median first-token mass (the probability of the two first tokens together) is below 0.5 is reported, and its first-token category is marked as resting on a small share of the model's probability.

## 9. Decision rules

**Margin.** For each model and readout,

$$\delta = \max\big(0.25\,\lvert \text{no-statement effect} \rvert,\ \delta_{\min}\big),$$

with $\delta_{\min}$ 0.4 nats for Qwen and 0.1 for Llama: at least five times the numerical standard deviation of a mean over hundreds of items in each family (`numerical_sensitivity.md` Rule 3, about 0.05–0.08 for Qwen-sized logits and 0.01–0.03 for Llama), and a quarter of the effect, below which a removal is not taken to change what carries the effect.

**Categories** for each primary removal $R$ (positive means the statement removes part of the effect):

- **removes most:** $R$'s 95% interval above 0, Holm-adjusted p below 0.05, and the share removed with its 95% interval at or above 0.5;
- **removes a part:** $R$'s interval above 0 and Holm p below 0.05, but not "most";
- **increases the effect:** $R$'s interval below 0 and Holm p below 0.05;
- **removes nothing (equivalent to zero within the margin):** $R$'s 90% interval inside $(-\delta, \delta)$, and, for Qwen only, the standardized removal's 90% interval inside $(-0.15, 0.15)$ (negative decisions are read on the low-floor family or the standardized scale, `numerical_sensitivity.md` Rule 3);
- **undetermined:** none of these.

A failed gate replaces the category as in section 8. A category whose deciding interval ends within twice the family's numerical standard deviation of its boundary (0.1 nats for Qwen, 0.03 for Llama) is labelled **numerically fragile**.

**Readouts are not combined.** Each readout gets its own categories. Where they disagree, the disagreement is reported as a finding, with the readouts' known properties beside it (the yes/no readout carries a task judgement; the two-candidate score carries the success text's wording, and for Llama covers about a fifth of the probability of a full reply).

## 10. What each outcome means

The readings, with the gates that make a null readable:

| Pattern (per model and readout) | Reading |
|---|---|
| Tester and agent statements each remove most of the effect beyond the content-free sentence; gates pass | The effect behaves like inference from the action's provenance: told the action carries no information about the room, the model stops letting the task line move its prediction. That is a finding in itself. The natural effect is then not shown to violate goal invariance under the interventional reading, because the expert-provenance reading carries most of it. |
| Removes a part | Part of the effect behaves like provenance inference; the rest persists when the action is stated to be uninformative. |
| Removes nothing, with the positive control and the comprehension gate passing | Provenance does not license the effect: the model reads provenance statements (positive control), states the right answer about the chooser, and the task line still moves its prediction as much. Under the interventional reading, what remains needs another explanation. It is not thereby shown to be an error on the log-odds scale, because a model with residual doubt need not shrink a log-odds effect (section 7); the probability-scale effect is reported for that reason. |
| A gate fails | Nothing is concluded about that statement or readout. |
| The content-free sentence removes as much as the statements | The removal belongs to an added sentence near the action, not to provenance. |
| Tester and agent statements disagree | Reported; the tester statement also makes the action random with respect to the task, the agent statement only with respect to the hidden room, so they are not interchangeable. |
| Existence without reference removes most of the effect on the items where the model says the object is present | The existence information the task line carries can account for the effect, and the earlier collapse did not need the object's name. |
| It removes nothing (or a part) while the named sentence removes most on the same items | The collapse needs the object's reference: an entity or role effect, or repetition of the name, which this contrast does not separate. A residual up to the size of the task's legitimate information about location given existence (A1′ index of [a1prime_check.md](a1prime_check.md)) is not excess. |
| Neither sentence removes it | The collapse seen earlier does not reappear in this format and stack; the existence reading is not supported here. |

**The interpretation rule.** Rejecting pure accounts does not identify a mechanism or a mixture. A removal shows what a statement changes, not why. In particular, "provenance carries it" means the effect responds to a provenance statement as provenance inference would, not that the model represents a chooser; and "persists" means the stated information does not reach the prediction, not which mechanism produces it.

## 11. Cross-GPU subset rescore

Rule 4 of `numerical_sensitivity.md`: the 300-item subset, every arm and readout in the primary format, rescored on an A100 in the same stack. For every decision statistic ($R(\text{tester})$, $R(\text{agent})$, $R_{\exists}$, $PC$ and the no-statement effect), per model and readout, the analysis recomputes the estimate and the category on the subset from the L40S scores and from the A100 scores with the same resamples, and reports the shift. A reading whose category changes on the subset, or whose estimate moves by more than a fifth of its margin, is reported as **environment-dependent**.

## 12. Safeguards

- **Checks at the start of every scoring run, which refuse to score on any failure:** every prompt renders; every candidate pair is non-empty with different first tokens; Yes and No are single, different tokens in the format; aligned and swapped prompts differ only in the task line; in plain text the number of continuations taken from the joint tokenization is recorded. The run also refuses a dirty checkout, a GPU of another type than its configuration names, mock records and unverified items.
- **Per-item saving and resume:** an item's records are written together, so a stopped shard resumes at the next item; resumed records must match the item hash; a resume refuses a different commit or configuration.
- **Analysis refusals:** a missing or incomplete shard; shards at different commits; scoring or analysis at a commit other than the freeze tag's; a main-run shard on a GPU other than L40S or a rescore shard on one other than A100; a duplicate or missing score; a primary family of other than 18 tests; a dirty checkout.
- **Tests** (`tests/test_bridge.py`): the sample's size, proportions, determinism and nesting; every arm's prompt differs from its pair only in the task line, with the statement between the action and the readout; the plain-text render equals the audit's; the job list per item; the stated answers; the cached routine against a plain forward pass in float32 (two small random models, any order, a prompt equal to the shared prefix, a one-sequence group); the analysis on planted data for each reading (provenance carries it; removes nothing; comprehension failed; positive control failed; the content-free sentence removes as much; existence removes most; existence check failed); the bootstrap, permutation test and Holm on known data; the cross-GPU comparison with and without a planted shift; duplicate records refused; the configurations' GPU types; a mock end-to-end run through the analysis.

## 13. Smoke run, cost and stop rule

**Smoke run.** Both models, the 20 smoke items, every arm, readout and question, on an L40S at the freeze tag. It reports only format and cost quantities, never an effect or a gate: seconds per item and the projected cost; the median first-token and yes/no masses per arm; the token checks; and the difference between the cached routine and a plain forward pass on one item per format.

**Cost.** Per item and model, the main run scores 14 provenance prompts and 4 existence prompts with two candidates each, 18 yes/no prompts and 4 existence checks, plus 12 comprehension prompts on the subset. With the cached routine the shared prefix is computed once per item and task line. The study's compute allowance is about 2.5 GPU-hours in all. The projection from the smoke run covers both main runs, Qwen's 300-item chat-format link, both A100 rescores (at the L40S speed, which is conservative) and model loading.

**Stop rule.** If the smoke projection exceeds 2.5 GPU-hours, nothing more is submitted; the projection and the reductions that would bring it within budget are reported for a decision.

*(Smoke readout: added here before the full run.)*

## 14. Limitations stated in advance

- The statements are stipulations about a hypothetical chooser. The real items were chosen by a collector that saw the room, so a model that believed the statements would be wrong about the real provenance; the study measures the model's response to stated provenance, which is what the objection concerns.
- The provenance arms leave the existence channel open: under any stated provenance, the aligned task still implies the object exists. A persisting effect under the provenance statements therefore can still include legitimate existence information held with residual doubt.
- The existence statement is a universal claim; the existence check names the object and may itself remind the model of it. The check selects the items the existence estimand is read on, so that estimand describes the items where the model applies the claim, not all items.
- A content-free sentence in the task line is not included; the named sentence is the existence arm's comparison. Dilution by an added sentence would shrink both, so "removes nothing" is robust to it and "removes most" is checked against the named sentence and the provenance arms' content-free sentence.
- The comprehension questions are behavioural: answering them correctly does not show that the same information is used when predicting.
- Two models, one environment, one item population; RQ2's determined failures only.

## 15. Files and run order

Code `wf/bridge.py` (items, prompts, scoring run), `wf/bridge_analysis.py` (analysis), `wf/scoring/prefix.py` (cached routine); configurations `wf/config/bridge_smoke_<model>.yaml`, `bridge_score_<model>.yaml` (two shards each), `bridge_rescore_<model>.yaml`, `bridge_analysis.yaml`; job scripts `pace/bridge.sbatch` and `pace/bridge_analysis.sbatch`; tests `tests/test_bridge.py`.

Order: freeze tag; smoke runs (both models); smoke readout committed here; main runs (two shards per model, L40S) and rescores (A100); the analysis once, on CPU, from a clean checkout at the tag; results in `notes/bridge_results.md`.
