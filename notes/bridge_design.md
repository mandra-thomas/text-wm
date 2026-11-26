# Natural ALFWorld bridge: who chose the action, and existence settled without naming the object (design, written 2026-11-25 before any score)

**Status.** Exploratory study with a frozen procedure. This note, the code (`wf/bridge.py`, `wf/bridge_analysis.py`, `wf/scoring/prefix.py`), the configurations (`wf/config/bridge_*.yaml`), the job script (`pace/bridge.sbatch`) and the tests (`tests/test_bridge.py`) are frozen together at an annotated tag before any score. The smoke readout is added to section 13 before the full run; the analysis runs once, from a clean checkout at the tag. Nothing here changes a confirmatory result. A companion CPU check of an assumption the existence arm relies on is in [a1prime_check.md](a1prime_check.md).

## 0. What was known when this was written

RQ2 scored 4,666 ALFWorld actions whose failure the history already settles (for example, taking an object from a receptacle the agent is not at). Each item was scored under the **aligned** task line (the episode's real task, which names the action's object) and the **swapped** line (the same template naming another object in the room). The **goal effect** is the mean, over items, of the model's log-odds of success under the aligned line minus that under the swapped line. It was 6.12 nats for Qwen3-8B and 1.65 for Llama-3.1-8B (`rq2_results.md`). In plain text, without the empty think block that Qwen3's template adds when thinking is off, Qwen's effect is about 3.3 nats, with the same standardized size (about 1.05) and share of items moved (about 89%) (`qwen_effect_audit_pipeline.md` §7).

Two objections stop that effect from being read as an error of the model, and this study tests one natural-data version of each.

**Who chose the action.** Every RQ2 action is a probe written by the item collector, which saw the whole room and chose failing actions on the task's object. The prompt shows only "Action: …". A model that reads an unexplained action as chosen by an expert who could see the room, as ALFRED's demonstrations were, would legitimately take an action that fits the aligned task as evidence that it works. Under that reading the task line is informative even about a failure the history settles, because the model may not be certain of what the history shows. The project's question is what the environment would reply if the action were taken regardless of who chose it, so this reading is a candidate explanation, not a licence; but it has never been tested by telling the model who chose the action.

**What the task says about existence.** A task naming an object implies that the object exists. In an earlier exploratory test (`exploratory_results.md` §3), a sentence stating the object's existence placed in both prompts removed the goal effect on the first answer word (Qwen −0.16 nats and Llama 0.07, against 2.42 and 0.66 without it, on the 4,382 items with one known location). Every sentence used then named the object, so the collapse cannot be told apart from the extra mention saturating whatever the name does. A sentence that settles existence without naming or describing the object separates the two.

## 1. Questions

1. **Provenance.** When the prompt says the action was chosen at random by a tester, or by an agent that has seen only the history shown, does the goal effect shrink relative to a prompt that carries a content-free sentence in the same place? If it does, the effect behaves as if the model inferred something from who chose the action. If it does not, while the model raises its prediction when told the chooser only proposes working actions (a positive control) and answers questions about the chooser as stated (a comprehension gate), the effect does not behave like provenance inference.
2. **Existence without reference.** When both prompts state "Every object any task in this house could need is present in the room", does the goal effect shrink, beyond what a content-free sentence in the same place does, on the items where the model, asked directly, says the object is present? If it does, the existence information the task line carries is enough to account for the effect in this format. If it persists while the named sentence removes it, the effect needs a reference to the object (entity, role or the name itself).

## 2. Assumption A1′, checked on CPU first

Reading an existence collapse as legitimate screening needs assumption A1′: once the object's existence is known, where it is (which receptacle, whether inside a closed one, how many instances) does not depend on which task was drawn. If A1′ fails, a correct model keeps a legitimate residual after existence is settled, equal to the task's information about location. [a1prime_check.md](a1prime_check.md) measures this on ALFWorld's generated initial states. A1′ fails as an exact statement (task objects are placed differently from other instances of their type, with existence fixed), but the legitimate change it implies in the log-probability that the object is at the action's receptacle is small on typical take items (median 0.06 nats, mean −0.29 because of a negative tail), so a near-zero residual after existence is settled remains consistent with legitimate screening, while a mean residual much above half a nat is not licensed by location information. The per-item index is reported beside the existence removal (section 7).

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

**Statements about who chose the action** sit on their own line between the action and the readout ("Action: …", a blank line, the statement, a blank line, then "Next observation:" or the question):

| Arm | Text |
|---|---|
| no statement | (nothing) |
| random tester, wording A | A tester picked this action at random from the possible actions. |
| random tester, wording B | A tester chose this action by drawing it at random from a list of commands, without looking at the room. |
| history-only agent, wording A | An agent who has seen only what is shown above chose this action. |
| history-only agent, wording B | This action was chosen by an agent whose only information about the room is the history shown above. |
| positive control | This action was chosen by an agent who only proposes actions that will work. |
| content-free sentence (placebo) | This command is written in lower-case letters, like every command in this game. |
| seeing agent pursuing the task (reported, not gated) | This action was chosen by an agent who could see inside every receptacle in the room and picks actions that advance its task. |

Wording A of the tester and agent statements is the text fixed when the study was proposed. Wording B removes one ambiguity: "the possible actions" could be read as ALFWorld's list of currently admissible actions, which would imply the action works; the comprehension gate (section 8) catches that reading for wording A.

The content-free sentence is there because an extra sentence near the action can dilute or otherwise change a task effect whatever it says. The tester and agent statements are measured against it, so that a removal is attributed to what the statement says rather than to its presence. It is not free of every possible reading: "like every command in this game" could be read as saying the command is well formed, which is why the positive control is also reported against no statement.

The positive control asserts that the chooser only proposes working actions, so it mainly tests whether the model raises predicted success when told that; it is the gate that makes a null readable on each readout, not evidence that the model reasons about choosers. The seeing-agent statement (added after review 1) tests the other direction: if the model reads provenance, an agent that sees the room and pursues the task line's task makes the action evidence of success under the aligned line and not under the swapped line, so the goal effect should grow. It is reported beside the removals and gates nothing.

**Existence statements** are appended to the task line in both prompts, as in the earlier dissociation test:

- **existence without reference:** "Every object any task in this house could need is present in the room." It neither names nor describes the object. It is true of every item (the action's object and the swapped object both exist in the room). It is a universal claim the model may not apply to the specific object, and it is not symmetric across task lines: under the aligned line the object any task needs includes the action's object directly; under the swapped line the action's object is covered only if the model takes "any task in this house" to include tasks other than the one stated. The existence check (section 6) measures, per item and task line, whether the model applies it.
- **existence with the name (reference arm):** "There is a {object} somewhere in the room.", the earlier test's sentence. It shows whether a name-based collapse appears in this format and stack at all.
- **content-free sentence in the task line (reference for both, added after review 1):** "This sentence and the task before it are written in plain English, as in every game." Both existence removals are measured against it, so that a removal is not the effect of appending any sentence to the task line.

## 6. Readouts

Every arm is read three ways, and each gets its own conclusions:

- **Two-candidate:** the teacher-forced log-probability of the environment's real success text minus that of "Nothing happens.".
- **First token:** the log-probability of the success text's first token minus that of "Nothing", read from the same position. It carries no wording penalty and asks no question.
- **Yes/no:** RQ2's question ("Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens.\nAnswer:") in place of "Next observation:", read as log P(Yes) − log P(No). It is known to carry part of a task-progress judgement (`cross_model_summary.md`), so a removal seen only here is reported as such.

Three further yes/no prompts:

- **Comprehension questions** (subset, aligned task line, after each statement except the content-free and seeing-agent ones, and after no statement), in place of "Next observation:":
  - **C1:** "Question: did the choice of this action depend on anything about the room that is not shown above? Answer Yes or No." The stated answer is No for both tester and both agent wordings, Yes for the positive control.
  - **C2:** "Question: could this action have been chosen even if it will not work? Answer Yes or No." The stated answer is Yes for the tester and agent wordings, No for the positive control.

  C1 is the specified question about whether the chooser used knowledge of the room's contents. C2 has the opposite stated answer, so that a model leaning toward one answer word cannot pass both. Because the tester and agent answers (No, Yes) may also be what a model says with no statement at all, the gate (section 8) also requires each item's answer to differ from that item's answer after the positive control, whose stated answers are the opposite; the answers with no statement are reported.
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

with its **share of the effect under the content-free sentence** (the reference the removal is measured from),

$$\rho(k) = \frac{\sum_i \big[G_i(\text{placebo}) - \bar G_i(k)\big]}{\sum_i G_i(\text{placebo})},$$

and its **share of the no-statement effect**, the same with $G_i(\text{none})$ in place of $G_i(\text{placebo})$ in both places.

3. **Removed by the content-free sentence**, the mean of $G_i(\text{none}) - G_i(\text{placebo})$, and **removed by the statement against no statement**, the mean of $G_i(\text{none}) - \bar G_i(k)$ (both reported).
4. **Positive-control shift**: how much the positive control raises predicted success over the content-free sentence, averaged over both task lines,

$$PC = \frac{1}{N}\sum_i \frac{1}{2}\sum_{g} \big[L_i(g, \text{positive}) - L_i(g, \text{placebo})\big],$$

also reported against no statement.

5. **Increase under the seeing agent** (reported): the mean of $G_i(\text{expert}) - G_i(\text{placebo})$.
6. **Removed by existence settled without reference**, against the content-free sentence in the task line, on the set $F$ of items whose existence check under the unnamed sentence favours "present" (log-odds above zero) under both task lines:

$$R_{\exists} = \frac{1}{|F|}\sum_{i \in F} \big[G_i(\text{task-line placebo}) - G_i(\text{exist})\big],$$

with its share of the effect under the task-line placebo and of the no-statement effect on the same items. Reported beside it: the same removal on the items where the statement changed the swapped line's check from "absent" (under no statement) to "present"; on the items in $F$ whose history does and does not name the object; by history-length bin; and, for the take items in $F$, the mean A1′ index of [a1prime_check.md](a1prime_check.md) (location plus instance number given existence, per item, from `notes/a1prime_items.jsonl`) beside the effect that remains under the unnamed sentence.

7. **Removed by existence stated with the name** (reference), the same quantity for the named sentence, on $F$ and on all items.
8. **Wording spread**: the mean difference between the two wordings' effects, per statement type.
9. **Standardized removal**: a removal divided by the item-level standard deviation of its reference effect ($G_i(\text{placebo})$ for provenance, $G_i(\text{task-line placebo})$ for existence). It is the scale on which Qwen's negative decisions are read (section 9).

Reported with each: the standardized effect and share of items moved toward success per arm; for the first-token and yes/no readouts the probability-scale goal effect (the mean of the difference of the two sigmoids); each task line's level shift against no statement; the median two-way probability mass per arm; and every estimand by history-length bin. The probability scale matters for the following reason. Suppose a model holds the deciding fact (say, that the agent is not at the receptacle) with a small residual doubt $\epsilon$, and the task acts only through what would happen if that fact were false. The probability of success is then about $\epsilon$ times the success probability in that world, so the task's effect on the probability vanishes as $\epsilon$ does, while its effect on the log-odds is the same for every small $\epsilon$. A removal seen in log-odds is therefore informative, but a persisting log-odds effect is not by itself excess.

**Intervals and tests.** 95% percentile intervals from a bootstrap over games (2,000 resamples, seed 20260924; one fixed set of resamples per item list, shared by every statistic on that list). The primary tests are two-sided sign-flip permutation tests of $R(\text{tester})$, $R(\text{agent})$ and $R_{\exists}$ in which all of a game's items flip together (10,000 permutations), for each model and readout: 18 tests, Holm-adjusted at a family-wise 0.05. The family always has 18 members, whether or not a gate passes.

## 8. Gates

1. **Comprehension (per model and wording).** On the 300 subset items, for C1 and for C2 separately: at least 80% of items favour the stated answer, and on at least 80% of items the answer lies on the stated side of the same item's answer after the positive control. A statement type is read only through its passing wordings; with none passing, its categories are "not readable: comprehension gate failed". The positive control's answers and the no-statement answers are reported, not gated.
2. **Positive control (per model and readout).** $PC$ at least 1 nat with its 95% interval above 0 (τ-auth's threshold). If it fails on a readout, that readout's provenance categories are "not readable: positive control failed", because a statement that removes nothing could then not be told from a statement the model does not take in.
3. **Existence check (per model).** At least a fifth of the items in $F$ (300 of the 1,500; 60 of the 300 subset items in the cross-GPU comparison); otherwise the existence categories are "not assessable".
4. **Format flag.** Every arm and task line whose median first-token mass (the probability of the two first tokens together) is below 0.5 is listed, and the first-token categories carry that list.

## 9. Decision rules

**Margin.** For each model and readout,

$$\delta = \max\big(0.25\,\lvert \text{reference effect} \rvert,\ \delta_{\min}\big),$$

where the reference effect is the effect under the content-free sentence in the statement's position, and $\delta_{\min}$ is 0.4 nats for Qwen and 0.1 for Llama: Rule 3's practical floors (`numerical_sensitivity.md`; for Qwen at least five times the numerical standard deviation of a mean over hundreds of items, about 0.05–0.08; for Llama the rule's 0.1, about five times its typical 0.015–0.02). A quarter of the effect is the size below which a removal is not taken to change what carries the effect.

**Categories** for each primary removal $R$ (positive means the statement removes part of the effect):

- **removes most:** $R$'s 95% interval above 0, Holm-adjusted p below 0.05, $R$ at least $\delta$, the share of the reference effect with its 95% interval at or above 0.5, and the share of the no-statement effect at least 0.5;
- **removes a part:** $R$'s interval above 0, Holm p below 0.05 and $R$ at least $\delta$, but not "most";
- **increases the effect:** $R$'s interval below 0, Holm p below 0.05 and $R$ at most $-\delta$;
- **removes nothing beyond the margin (equivalent to zero within it):** $R$'s 90% interval inside $(-\delta, \delta)$, and, for Qwen only, the standardized removal's 90% interval inside $(-0.15, 0.15)$ (negative decisions are read on the low-floor family or the standardized scale, `numerical_sensitivity.md` Rule 3). A removal can be detectably above zero and still in this category; that is reported, and read as a removal too small to change what carries the effect;
- **removes (or increases by) a small amount:** detectably non-zero by the first two conditions, smaller than $\delta$, and not shown equivalent;
- **undetermined:** none of these.

A failed gate replaces the category as in section 8. A category whose deciding interval ends within twice the family's numerical standard deviation of its boundary (0.1 nats for Qwen, 0.03 for Llama) is labelled **numerically fragile**.

**Readouts are not combined.** Each readout gets its own categories. Where they disagree, the disagreement is reported as a finding, with the readouts' known properties beside it (the yes/no readout carries a task judgement; the two-candidate score carries the success text's wording, and for Llama covers about a fifth of the probability of a full reply; a statement line before "Next observation:" can change which continuation the model expects, which the first-token mass shows).

## 10. What each outcome means

Every reading below says how the goal effect behaves when a statement is added. None identifies a mechanism.

| Pattern (per model and readout) | Reading |
|---|---|
| Tester and agent statements each remove most of the effect beyond the content-free sentence; gates pass | The effect behaves as if the model read the unexplained action as chosen by someone who knew the room: told the chooser had no such knowledge, the task line stops moving its prediction. The natural effect is then not shown to violate goal invariance under the interventional reading. Other routes are not excluded: the agent wordings also point at "the history shown above", which could shift weight to the history without any inference about the chooser. |
| Removes a part | Part of the effect behaves as above; the rest persists when the action is stated to be uninformative. |
| Removes nothing beyond the margin, with the positive control and the comprehension gate passing | The effect does not behave like inference from provenance: the model answers the questions about the chooser as stated, raises predicted success when told the chooser only proposes working actions, and the task line still moves its prediction as much. Whether the model uses those statements when predicting is not shown by the gate, which is behavioural. Nor is the remaining effect thereby an error on the log-odds scale, because a model with residual doubt need not shrink a log-odds effect (section 7); the probability-scale effect is reported for that reason. The seeing-agent statement is read beside it: if that does not increase the effect either, the goal effect is insensitive to stated provenance in both directions. |
| Removes or increases a small amount | Reported with its size; too small to change what carries the effect. |
| Increases the effect | Reported; the stated provenance makes the task line matter more, which no account under test predicts for these statements. |
| Undetermined | Neither a removal of the margin's size nor its absence is shown. |
| A gate fails | Nothing is concluded about that statement or readout. |
| The content-free sentence removes as much as the statements | The change belongs to an added sentence near the action, not to what it says about the chooser. |
| Tester and agent statements disagree | Reported; the tester statement makes the action random with respect to the task as well as the room, the agent statement only with respect to the hidden room, so they are not interchangeable. |
| Existence without reference removes most of the effect beyond the task-line placebo, on the items where the model says the object is present | In this format, settling existence without naming the object accounts for most of the effect on those items: the existence information the task line carries is enough. Whether the earlier collapse (measured in RQ2's chat format) needed the name is answered for that format only by the descriptive chat-format link. |
| It removes nothing (or a part) while the named sentence removes most on the same items | The collapse needs a reference to the object: an entity or role effect, or repetition of the name, which this contrast does not separate. A remaining effect up to the size of the task's legitimate information about location and instance number given existence (the A1′ index reported beside it) is not excess. |
| Neither sentence removes it | A name-based collapse does not appear in this format and stack; the existence reading is not supported here. |

**The interpretation rule.** Rejecting pure accounts does not identify a mechanism or a mixture. A removal shows what a statement changes, not why. "Behaves as if it reads provenance" means the effect responds to a provenance statement as provenance inference would, not that the model represents a chooser; and "persists" means the stated information does not change the prediction, not which mechanism produces the effect.

## 11. Cross-GPU subset rescore

Rule 4 of `numerical_sensitivity.md`: the 300-item subset, every arm and readout in the primary format, rescored on an A100 in the same stack. On the subset, the analysis recomputes every decision statistic from the L40S scores and from the A100 scores with the same resamples: each run's own comprehension gate, the existence-check count against its gate, $R(\text{tester})$, $R(\text{agent})$, $R_{\exists}$ with their categories, $PC$ with its gate, and the no-statement effect. A reading whose category or gate changes on the subset, or whose estimate moves by more than a fifth of its margin, is reported as **environment-dependent**.

## 12. Safeguards

- **Checks at the start of every scoring run, which refuse to score on any failure:** the checkout is at the freeze tag's commit and clean; the stack is serve2 (transformers 5.23.0, torch 2.14.1); the GPU is of the configured type (also on resume); the collection is finished and verified; every prompt renders; every candidate pair is non-empty with different first tokens; Yes and No are single, different tokens in the format; aligned and swapped prompts differ in exactly one line, the task line; in plain text the number of continuations not taken from the joint tokenization is recorded. Mock records are refused by the reader.
- **Smoke stop rule for the cached routine:** on one item per format, the cached scores are compared with plain forward passes; the smoke run fails if the mean absolute difference over the log-probabilities exceeds 0.5 nats or more than 5% of them differ by more than 1 nat. bf16 rounding moves log-probabilities by tenths of a nat; an offset or cache error moves most of them by nats.
- **Per-item saving and resume:** an item's records are written together; on resume, an item with fewer than all of its records is removed and scored again; resumed records must match the item hash; a resume refuses a different commit or configuration. Each shard records its software stack.
- **Analysis refusals:** a missing or incomplete shard; shards at different commits; scoring or analysis at a commit other than the freeze tag's; a main-run shard on a GPU other than L40S or a rescore shard on one other than A100; a shard whose recorded stack is not serve2; a record of another model, of an item outside the sample, or with a different item hash; a duplicate or missing score; a primary family of other than 18 tests; a dirty checkout.
- **Tests** (`tests/test_bridge.py`; run in serve2 with torch before the freeze, all passing): the sample's size, proportions, determinism and nesting; every arm's prompt differs from its pair only in the task line, with the statement between the action and the readout; the plain-text render equals the audit's; the job list per item; the stated answers; the cached routine against a plain forward pass in float32 (small random Llama and Qwen3 models, any order, a prompt equal to the shared prefix, a one-sequence group); the mapping from every scored sequence to its record through the real renderer (a word-level tokenizer and a tagging engine); the shared prefix ending at the action line; the analysis on planted data for each reading (removes most; removes nothing; comprehension failed; a model that answers the same whatever the statement fails the gate; positive control failed; the content-free sentence removes as much; a detectable removal below the margin is not "a part"; shares against the right reference; the first-token mass flag; existence removes most; existence removes nothing beyond the task-line placebo; existence check failed); the bootstrap, permutation test and Holm on known data; the cross-GPU comparison with and without planted shifts, with the existence category comparable on the subset; resume dropping partly saved items; records of another model or item refused; duplicate records refused; the configurations' GPU types; a mock end-to-end run through the analysis.

## 13. Smoke run, cost and stop rule

**Smoke run.** Both models, the 20 smoke items, every arm, readout and question, on an L40S at the freeze tag. It reports only format and cost quantities, never an effect or a gate: seconds per item and the projected cost; the median first-token and yes/no masses per arm; the token checks; and the cached-against-full-pass comparison with its stop rule.

**Cost.** Per item and model, the main run scores 9 statement arms and 3 task-line arms under two task lines, each with the two-candidate pair and the yes/no question, plus 4 existence checks and, on the subset, 12 comprehension questions: about 72 scored sequences and 8 shared prefixes of about 650–750 tokens per item. With the cached routine each shared prefix is computed once. The study's compute allowance is about 2.5 GPU-hours in all. The projection from the smoke run covers both main runs, Qwen's 300-item chat-format link, both A100 rescores (at the L40S speed, which is conservative) and model loading.

**Stop rule.** If the smoke projection exceeds 2.5 GPU-hours, nothing more is submitted; the projection and the reductions that would bring it within budget are reported for a decision.

*(Smoke readout: added here before the full run.)*

## 14. Limitations stated in advance

- The statements are stipulations about a hypothetical chooser. The real items were chosen by a collector that saw the room, so a model that believed the statements would be wrong about the real provenance; the study measures the model's response to stated provenance, which is what the objection concerns.
- The provenance arms leave the existence channel open: under any stated provenance, the aligned task still implies the object exists. A persisting effect under the provenance statements therefore can still include legitimate existence information held with residual doubt.
- The existence statement is a universal claim, not symmetric across task lines (section 5); the existence check names the object and may itself remind the model of it. The check selects the items the existence estimand is read on, so that estimand describes the items where the model applies the claim under both task lines, not all items; the subset where the statement changed the check, and the split by whether the history names the object, are reported for that reason.
- The comprehension questions are behavioural: answering them correctly does not show that the same information is used when predicting.
- The chat-format link for Qwen is descriptive and on 300 items; conclusions about RQ2's own format rest on it only descriptively.
- Two models, one environment, one item population; RQ2's determined failures only.

## 15. Files and run order

Code `wf/bridge.py` (items, prompts, scoring run), `wf/bridge_analysis.py` (analysis), `wf/scoring/prefix.py` (cached routine); configurations `wf/config/bridge_smoke_<model>.yaml`, `bridge_score_<model>.yaml` (two shards each), `bridge_rescore_<model>.yaml`, `bridge_analysis.yaml`; job scripts `pace/bridge.sbatch` and `pace/bridge_analysis.sbatch`; tests `tests/test_bridge.py`; the A1′ per-item index `notes/a1prime_items.jsonl` (from `scripts/a1prime_check.py`). The independent review and the changes it led to are in [bridge_review1.md](bridge_review1.md).

Order: freeze tag; smoke runs (both models); smoke readout committed here; main runs (two shards per model, L40S) and rescores (A100); the analysis once, on CPU, from a clean checkout at the tag; results in `notes/bridge_results.md`.
