# How does post-training change whether a language model keeps an agent's task out of its predicted environment dynamics? A staged-checkpoint pilot (design, 2026-11-26)

**Status: frozen design (tag `posttraining-design-frozen-2026-11-26`).** No effect has been scored. The access and tokenizer check (section 2) and a CPU dry run of the scorer and analysis with the real tokenizers and a tiny random model ran on CPU. Exploratory study with a frozen procedure: the design, the scorer and the analysis are fixed at an annotated tag after one independent review (`notes/posttraining_review1.md`, section 15), before any score. Approved by the user on 2026-11-26 with the scope in section 1.

## 1. The question and the scope

When a language model acts as a world model for an agent, the earlier history of a fully observed environment can already settle whether an action works. The agent's task should then not change the predicted outcome. Every model the project has scored lets it change the prediction anyway (the *goal effect*, `notes/rq2_results.md`). The question here is what post-training does to that dependence. It may introduce it, amplify it, suppress it, change its character (the same size made of different parts), or move different parts of it in opposite directions; one stage may add what a later stage removes. The design assumes none of these: every comparison is two-sided and every reading has a "the same" and an "inconclusive" outcome as well as "higher" and "lower".

Answering it needs checkpoints that differ only in training stage. The core lineage is the Llama-3.1-8B tree:

- the pretrained base, `meta-llama/Llama-3.1-8B`;
- AI2's Tülu 3 after supervised fine-tuning (SFT), after direct preference optimization (DPO), and after reinforcement learning with verifiable rewards using PPO (the released Tülu 3 8B);
- Tülu 3.1, a second RL run (GRPO) from the same DPO checkpoint;
- Meta's own Llama-3.1-8B-Instruct, a sibling post-training of the same base.

A second pair is Qwen3-8B-Base and Qwen3-8B (only the two ends are released). A *bridge* arm places the Word2World ALFWorld model, a third post-training of the same Llama base (domain SFT on ALFWorld transitions), in the same comparison.

**What this cannot identify** (stated before any score). Within a lineage each stage changes the training data and the objective together, so a stage difference cannot be attributed to either. One lineage is a case study, not a law about post-training. The readouts describe how predictions respond to the prompt; they say nothing about internal computation. Rejecting a pure account of what changed identifies neither a mechanism nor a mixture.

## 2. Checkpoints, access and tokenizers

| Tag | Checkpoint | Pinned revision | Role |
|---|---|---|---|
| llama-base | meta-llama/Llama-3.1-8B (gated) | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b | base |
| tulu-sft | allenai/Llama-3.1-Tulu-3-8B-SFT | f2a0b46b0cfda21003c6141b1ff837b7e165524d | SFT from the base |
| tulu-dpo | allenai/Llama-3.1-Tulu-3-8B-DPO | a7beb67e33ffd01cc87ac3b46cadc1000985b8db | DPO from SFT |
| tulu-rl | allenai/Llama-3.1-Tulu-3-8B | 666943798adbde0b1aff34626007e26986a3c107 | RLVR (PPO) from DPO |
| tulu31-grpo | allenai/Llama-3.1-Tulu-3.1-8B | 46239c2d07db76b412e1f1b0b4542f65b81fe01f | RLVR (GRPO) from the same DPO |
| llama-instruct | meta-llama/Llama-3.1-8B-Instruct (gated) | 0e9e39f249a16976918f6564b8830bc894c89659 | Meta's post-training of the base |
| w2w-llama | X1AOX1A/WorldModel-Alfworld-Llama3.1-8B | 8f089d2126f0406a296747041ebf33b902837cd3 | domain SFT of the base (bridge only) |
| qwen-base | Qwen/Qwen3-8B-Base | 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 | base |
| qwen | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | post-trained (distilled) |

The revisions are the Hub's `main` commit read from the Hugging Face API on 2026-11-26 before any download (the last three were already pinned in `wf/constants.py`).

**Access check** (CPU job 7347490, `scripts/posttrain/access_check.py`, commit 3786df4). With the cluster's token, every checkpoint's model information at the pinned revision returned the pinned commit, and a request for every weight file succeeded (both gated Meta repositories included; 16.06 GB each for the Llama tree, 16.38 GB for Qwen). The official tokenizer and config files were read at the pinned revisions:

- All seven Llama-tree tokenizers add `<|begin_of_text|>` (id 128000) by default. The four Tülu checkpoints add one `<pad>` token (id 128256) and have 128,264 embedding rows; the others have 128,256. Every other id is shared.
- The token ids of 71 fixed texts (the golden item's prompt in plain text under the aligned and the swapped task, the readout words with and without a leading space, a success and the failure text, all 55 ALFWorld object names with a leading space, and the four carrier sentences), each with and without the tokenizer's special tokens, are identical across all seven Llama-tree tokenizers, and across the two Qwen tokenizers.
- Chat templates: the base has none; the four Tülu checkpoints share one template (plain role tags `<|user|>`, `<|assistant|>`, which does not render a BOS token); Meta's Instruct template renders its BOS and header tokens; Word2World's renders `<|begin_of_text|>` and Llama-3 headers; the Qwen base ships a template that is never used for it.

The run-time gate (section 5) repeats the identity check on every prompt actually scored.

## 3. Prompt formats

**Primary: one plain-text few-shot format for every checkpoint.** The study's world-model prompt (system text, three few-shot examples, the history, the action, "Next observation:") is rendered with no chat template: the system text, each example as its user text, one space and its reply, then the query's user text, separated by blank lines (the Qwen audit's `plain_text`, which wave B's plain format and the lexical-versus-semantic diagnostic, "lexsem", also use). The prompt is tokenized with the tokenizer's own special tokens: the Llama tree's BOS token exactly once at the start, nothing for Qwen. A candidate reply is scored after one space. Because the examples show the environment's replies, predicting the next observation is ordinary next-token prediction, which a base model can do without instruction following.

**Secondary: each post-trained checkpoint's own chat template, as shipped at its pinned revision** (`apply_chat_template` with the generation prompt): Tülu's role-tag template for the four Tülu checkpoints (no BOS, as the template renders it; AI2's Tülu 3 8B SFT script, `scripts/train/tulu3/finetune_8b.sh` in open-instruct, does not pass `--add_bos`, the option with which open-instruct prepends BOS to the template, so training sequences most likely carried none either; this was not verified on the tokenized training data), Meta's template for Llama-3.1-8B-Instruct, and Qwen3's thinking-off template, which ends in the empty think block, for Qwen3-8B. A base model is never scored in a chat template. The difference between the two formats within one checkpoint is a *format effect*, and its change between stages is reported as a stage-by-format interaction.

**Bridge: the Word2World message format.** The system message carries the room's full state and the task line; earlier actions are user turns and observations assistant turns; the queried action is the last user turn (`rq1.arm_messages(..., ft=True)`, as `rq1_score` scored the Word2World models). It is rendered as plain text for the base, the four Tülu checkpoints and Meta Instruct, and in its own chat template for the Word2World Llama. The full-state contexts come from the `X1AOX1A/LLMasWorldModels` dataset snapshot ff6ae2b924d1 in the shared cache. Rendered as plain text this format has no few-shot examples and no "Next observation:" cue, and at step 0 nothing in it shows the reply format; the bridge is therefore descriptive only, with its own discrimination (successes against failures under the swapped task) reported beside every value.

**Seven-example fallback** (section 5): the same plain format with the three examples of set 1 followed by the four of set 2 (`notes/fewshot_sets_frozen.json`).

## 4. Items, conditions and the recognition check

**Items.** Lexsem's 450 determined failures (`wf.lexsem.sample`, sorted ids hash `cc8da36181976edd`): 300 at later steps in 108 games, whose history does not name the action's object, and 150 at step 0 (no action yet) in 96 games, drawn from RQ2's 4,666 replay-verified primary items. Every one is an action whose failure the history already settles. For the competence gate and a descriptive reading on working actions, all 292 of RQ2's determined successes that have a swapped task (111 games). The 56 failures whose object also appears in the few-shot examples are reported apart, as in lexsem.

**Conditions on the failures (12).** Each changes only the task line. The sentences and control objects are lexsem's (`wf/lexsem.py`), imported, so the prompts are byte-identical to lexsem's for the same item, condition and format. In the example the action is `take saltshaker 1 from shelf 2`, the swapped task names another object of the room, and the control object (toiletpaper) is an ALFWorld object of another class, absent from the room, with the same number of tokens.

| Condition | What it is, in words | Task line (example) |
|---|---|---|
| real task (aligned) | the episode's own task, which names the action's object | Your task is to: put some saltshaker on cabinet. |
| swapped task | the same task template with another object of the room | Your task is to: put some vase on cabinet. |
| neutral task | no object at all | Your task is to: explore the room. |
| assertion; control | the swapped task plus a claim that the object is in the room; the control names the absent control object | … There is a saltshaker somewhere in the room. / … There is a toiletpaper somewhere in the room. |
| denial; control | the swapped task plus a claim that the object is not in the room | … There is no saltshaker in this room. |
| listed word; control | the swapped task plus the name, with no claim | … The following word appears in this note: saltshaker. |
| "Ignore the X"; control | the swapped task plus an instruction to disregard the object | … Ignore the saltshaker. |
| described goal | the real task with the object described instead of named (rq1x's descriptions) | Your task is to: put some container with holes for sprinkling white seasoning on cabinet. |

**Conditions on the successes (3):** real, swapped and neutral tasks.

**Bridge conditions:** the real and the swapped task, on the failures and the successes.

**Recognition check for the described goal.** Rq1x's and lexsem's recognition checks ask the model to name the described object, which a base model cannot be expected to answer. Here each plain-format checkpoint scores, teacher-forced, every one of the 55 ALFWorld names that have a description after `The phrase "<description>" refers to the household object called`, and again after the same sentence with the neutral phrase "something". A name's association with the description is the difference of the two log-probabilities (so a name that is merely frequent gains nothing). An object is recognised if its own name ranks among the top three of the 55. The described-goal contrast is read on the items whose object every plain-format checkpoint of the lineage recognises, so its item set is the same at every stage.

## 5. Readouts and gates

**Readouts.** Each item, condition and format is scored as separate unpadded sequences: the prompt followed by the item's success observation (for example "You pick up the saltshaker 1 from the shelf 2."), the prompt followed by "Nothing happens.", and, for the chat-template arms only, the yes/no prompt (rq1v's anchored question). From them, with $\log p$ the model's log-probability:

The **first-token decision** is

$$L^{\text{first}} = \log p(\text{first token of the success text}) - \log p(\text{" Nothing"}),$$

both at the reply's first position of the success sequence. For every failure item the first token is " You"; 56 of the 292 success observations begin with another word ("There's", "The", "This"), and for them the decision is read from their own first token. The **full two-candidate score** is

$$L = \log p(\text{success text}) - \log p(\text{"Nothing happens."}),$$

each from its own sequence. The **continuation** is the remainder, $L^{\text{cont}} = L - L^{\text{first}}$: mostly the verb and the object's name once "You" is written, minus the tail of the failure text. The **yes/no** score is $\log p(\text{"Yes"}) - \log p(\text{"No"})$; it is recorded only for the post-trained checkpoints in their chat templates and is a secondary reading, never part of a comparison with a base model.

**Gates**, computed per arm (one checkpoint in one format) under the neutral task only, so no gate reads the goal contrast. RQ2's determined successes all have a history, so they are compared with the 300 later-step failures, the stratum of the primary readings; the step-0 discrimination is reported descriptively.

1. *Discrimination:* the area under the ROC curve separating successes from failures on the full two-candidate score is at least 0.70.
2. *Not degenerate:* the full score is above zero on at most 95% of failures and on at least 5% of successes.
3. *Mass:* the median probability of the two first tokens (the success text's first token plus " Nothing") is at least 0.7.
4. *Tokenization:* in plain text, the tokenizer's BOS exactly once where it adds one and never otherwise; on every failure, a name condition adds exactly one copy of the name's tokens to the swapped prompt and no other condition (other mentions, the described goal, the neutral task) adds one; every scored sequence's token ids (stored as a hash) are identical across the plain-format arms of a lineage with the same examples and in their floor rescorings, checked by the analysis; the success text of every failure starts with the readout token; the failure text starts with " Nothing" (chat: "Nothing"); lexsem's frozen name-token counts match; the rebuilt sample's hash matches. Any failure stops the job.
5. *Loading:* the loaded model's parameter names equal the checkpoint's safetensors index (allowing a tied output layer), at the pinned revision. The job also refuses to start on the wrong GPU type (an L40S for main and bridge arms, an A40 for floor arms).

An arm passing 1–3 is *interpretable*: its full score and continuation are read. Its first-token decision is read as a decision readout if, in addition, the first-token score alone passes 1 and 2. Yes/no is gated on its own (1 and 2 on the yes/no score, and a median "Yes" plus "No" mass of at least 0.9). An arm failing a gate is "not interpretable" on that readout, which is not a null result.

**Fixed fallback for a failing base.** If a lineage's base fails gates 1–3 in the three-example format, every plain-format checkpoint of that lineage is rescored with seven examples and only those arms are read for that lineage, so that a base rescued by extra examples is never compared with post-trained models that did not get them. Their floor rescorings are part of the fallback. It costs about 2.6 GPU-hours for both lineages; if it is needed, the study stops and reports before running it (section 10).

## 6. Estimands

For arm $k$, readout $r$ and the $n$ failure items of a stratum, write $L^r_{k,i}(c)$ for item $i$'s score under condition $c$. The goal effect is

$$G^r_k = \frac{1}{n}\sum_{i=1}^{n}\big[L^r_{k,i}(\text{aligned}) - L^r_{k,i}(\text{swapped})\big].$$

Because post-training can change how sharp a model's probabilities are, a shift of one nat does not mean the same thing at every stage. Every goal effect is therefore also reported on three scale-free measures: the **standardized effect** $Z^r_k$, the mean item effect over its item standard deviation (each arm's item standard deviation and the ratio between two arms are reported beside it, since a stage can change the spread as well as the mean); the **net share of items moved**, the share of items whose effect is positive minus the share negative; and the **rank gain**, each item's real-task rank minus its swapped-task rank among its three task lines (real, swapped, neutral), divided by two, which no monotone rescaling of an item's scores can change and which the mention sentences cannot move. The rank among all twelve conditions is reported descriptively. Beside these, the goal effect relative to the arm's own discrimination is

$$R^r_k = \frac{G^r_k}{E^r_k}, \qquad E^r_k = \bar{L}^r_{k,\text{successes}}(\text{neutral}) - \bar{L}^r_{k,\text{failures}}(\text{neutral}),$$

reported descriptively, as are the goal effect and shares without the 56 items whose object appears in the few-shot examples.

Each mention sentence $c$ with control $\kappa(c)$ has a net effect and a share of the goal effect on the same items:

$$\Delta^r_k(c) = \frac{1}{n}\sum_{i=1}^{n}\big[L^r_{k,i}(c) - L^r_{k,i}(\kappa(c))\big], \qquad s^r_k(c) = \frac{\Delta^r_k(c)}{G^r_k}.$$

The described goal's share is the same ratio with the described real task in place of $c$ and the swapped task in place of $\kappa(c)$, on recognised items. Every share is reported with its numerator and denominator in nats, the standardized net effect, and its rank-based counterpart (the sentence's rank gain over its control's, divided by the real task's rank gain).

The decision token's share of the goal effect is

$$\pi^{\vphantom{r}}_k = \frac{G^{\text{first}}_k}{G^{\text{full}}_k} = \frac{G^{\text{first}}_k}{G^{\text{first}}_k + G^{\text{cont}}_k}.$$

Two indices summarize how the decision token handles content. Content sensitivity is the gap between asserting and denying the object, and the content-blind lift is the share carried by a mention that asserts nothing:

$$C_k = s^{\text{first}}_k(\text{assertion}) - s^{\text{first}}_k(\text{denial}), \qquad B_k = s^{\text{first}}_k(\text{listed word}).$$

The same indices are computed on the continuation and the full score; "Ignore the X" is reported beside $B_k$. A change between checkpoints $k$ and $k'$ is a paired difference on the same items and the same bootstrap draws. The stage-by-meaning interaction for a sentence meaning $m$ asks whether a stage changes a sentence's share differently from the plain assertion's:

$$I_{k \to k'}(m) = \big[s_{k'}(m) - s_k(m)\big] - \big[s_{k'}(\text{assertion}) - s_k(\text{assertion})\big].$$

The stage-by-format interaction is the change between two post-trained stages of the format effect (the chat-template value minus the plain-text value of the same checkpoint). A **certainty-matched** reading reports every share by quartile of the neutral task's score, a prompt that enters neither the contrast nor the goal effect, with each quartile's range of neutral scores, so that the results note can state where two stages' ranges do not overlap; it is descriptive. If a lineage's base fails its gate (section 5), every comparison, secondary comparison and format effect that would use that lineage's three-example plain arms uses the seven-example arms instead or is marked "not read".

**Uncertainty.** One game-clustered bootstrap per stratum (10,000 resamples of games, seed 20261126, enough to resolve the smallest Holm threshold of 0.05/35) is shared by every arm, since every arm scores the same items. Intervals are 95% percentile intervals; two-one-sided equivalence readings use the 90% interval; p-values are two-sided bootstrap p-values for a difference of zero.

## 7. Reusing lexsem's scores

Lexsem scores some of the same cells in what is designed to be the identical environment (the serve2 stack on an L40S, bf16, one unpadded sequence per forward pass, the same scorer functions, the same pinned revision): Llama-3.1-8B base, Qwen3-8B-Base and Qwen3-8B in plain text, Qwen3-8B with the think block, and Llama-3.1-8B-Instruct's yes/no in its chat template. Lexsem reads the Llama base from the shared cache, where this pilot placed it at the pinned revision with every file's hash checked against the Hub (CPU job 7348413, `scripts/posttrain/fetch_shared.py`); this pilot's own Llama-base job reads the same copy. Within one environment the project's scores are deterministic (`notes/numerical_sensitivity.md` §4.1), so those failure cells are taken from lexsem's run files rather than rescored. This pilot scores only the failure cells lexsem lacks (for example the failure-text sequence of "Ignore the X", the described goal, and every failure-text sequence of the think-block arm) and every success cell itself (successes are never reused). The reuse is allowed only if all of these hold, checked by the analysis:

- lexsem's environment record for that arm equals this arm's (GPU type, torch, transformers, Python, revision, dtype, unbatched scoring, prompt mode);
- lexsem's run commit is recorded and every file that decides a score (the scorer, the item and prompt builders, the plain-text renderer, the pin table) is identical between that commit and the analysis commit;
- on the first five failures of each stratum this pilot also scores every reused cell, and every one of those values is bit-identical to lexsem's;
- every reused cell exists for all 450 failures.

Otherwise the arm is rescored with `--no-reuse` (a deviation entry first) and nothing from it is read until then. Reuse never crosses an environment.

## 8. Reading rules (fixed before any score)

**Strata.** The pre-specified readings use the 300 later items. Step 0 (150 items) is reported in the same form but carries no Holm-corrected reading, because lexsem's simulation shows it is much weaker.

**Pre-specified stage comparisons (primary, plain format).** Base against Tülu SFT; Tülu SFT against DPO; DPO against RL (PPO); base against Tülu's final model; base against Meta Instruct; Tülu's final model against Meta Instruct (sibling recipes); Qwen3-8B-Base against Qwen3-8B. For each, five primary quantities on the later stratum: the standardized goal effect on the first token and on the full score, the decision-token share $\pi$, content sensitivity $C$ and the content-blind lift $B$ (both on the first token). That is a fixed family of 35 readings, Holm-corrected together; a reading that is missing or fails a gate enters with p = 1, so the family never shrinks. A "higher" or "lower" reading in this family needs the Holm-adjusted p below 0.05 in addition to its own rule. Every other reading (the continuation, step 0, the other shares, the secondary comparisons) is computed with the same rules and labelled "descriptive". Secondary comparisons, without Holm: DPO against GRPO, PPO against GRPO, the same stage comparisons in the chat templates, the format effects and the stage-by-format interactions, and the bridge.

**A stage difference in the standardized goal effect** is read as:

- *higher* or *lower* if the paired 95% interval excludes zero, the estimate exceeds 0.15 in standardized units and five times the pair's numerical floor (the square root of the sum of the two arms' squared standardized floors), the Holm-adjusted p is below 0.05, **and** the difference in nats, in standardized units, in the net share moved and in the rank gain all have the same sign, with the rank-gain difference's interval excluding zero and its estimate at least 0.05 (about half a rank position);
- *scale-dependent* if the standardized criteria hold but the four measures disagree in sign or the rank gain does not clear its margin; this is reported as a finding about calibration, not as a null;
- *the same* if the 90% interval of the standardized difference lies within ±0.15 and 0.15 is at least five floors;
- *inconclusive* otherwise;
- *not readable* if either arm's floor was not measured.

**A checkpoint's own goal effect** is *suppressed to about zero* if the 90% interval of its standardized goal effect lies within ±0.10 and 0.10 is at least five floors; otherwise it is reported as above zero, below zero, or not shown to be about zero.

**Shares and their differences** ($\pi$, $C$, $B$, each sentence's share, and the stage-by-meaning interactions) use lexsem's three-way rule with its 0.25 margin: *lower* if the 95% interval lies wholly below −0.25, *the same* if it lies within ±0.25 (and 0.25 is at least five floors in share units, the pair's floor in nats over the smaller goal effect), *higher* if it lies wholly above 0.25, *inconclusive* otherwise. They are read only when both checkpoints' goal effects pass lexsem's reference rule made two-sided, so that a reversed effect is not excluded: the interval excludes zero and its end nearest zero is at least half the estimate in absolute value. Otherwise only the net effects in nats and standardized units are reported. The decision-token share $\pi$ also needs both arms' first-token and full readouts to pass their gates.

**Readouts are read separately.** The first token, the continuation and the full score each get their own reading line; disagreements between them are reported as findings, and none overrides another.

**Numerical floors** (`notes/numerical_sensitivity.md`, rules 1–6). Every main arm is rescored on an A40 (the same stack, a different GPU type) on all 300 later items (rule 4: at least 300 items, every arm): under the real and swapped tasks and the assertion pair for the chat-template arms, and additionally the denial and listed-word pairs, which $C$ and $B$ use, for the plain-format arms; success and failure sequences. The floor of a readout is the square root of the squared mean shift plus its bootstrap variance, for the goal effect and for each rescored mention pair's net effect, the largest, in nats and in standardized units. Margins must be at least five floors (rule 3); a reading whose interval ends within two floors of a boundary is labelled *numerically fragile*; a reading for which an arm's standardized goal effect moves on the A40 by more than a fifth of the margin is labelled *environment-dependent* (rule 4). Floors are measured on the later items and applied to step 0, which is stated with every step-0 reading. Every comparison is within one environment (rule 1): the analysis refuses scores from more than one stack, a main or bridge arm not scored on an L40S, a floor arm not on an A40, a revision other than the pin, scores from a commit whose scorer files differ from the analysis commit's, and any token-id mismatch.

## 9. What each outcome pattern would mean

Each row describes a pattern on the primary quantities that passes the rules of section 8. "The stage" means the first checkpoint at which the change appears.

| Pattern | Reading | What it does not mean |
|---|---|---|
| Base goal effect about zero; a later stage clearly positive | post-training introduces the dependence in this lineage, at that stage | not that the stage's objective rather than its data causes it; not that every recipe does |
| Base clearly positive; a later stage higher on every scale | present before post-training, amplified at that stage | not that pretraining "causes" it |
| Base clearly positive; a later stage lower or about zero | suppressed at that stage | not that the stage teaches environment dynamics; it may only reduce sensitivity to the task line in this format |
| Goal effect the same, but $C$, $B$ or $\pi$ changes | post-training changes the character, not the size | not a mechanism; the same components with different weights cannot be told from new components |
| Decision token and continuation move in opposite directions | the two readout components respond to post-training independently; the overall effect can hide the change | not that they are computed separately inside the model |
| A stage raises something and a later stage removes it | non-monotone | the final model says little about the process |
| Tülu's final model and Meta Instruct differ beyond the margin | the change belongs to a recipe, not to post-training of this base in general | not which part of the recipe |
| Formats agree early and diverge later | the dependence becomes format-bound with post-training | — |
| Measures disagree across scales | scale-dependent: the stage changes calibration, not a scale-free goal dependence | not a null |
| A base fails competence even with seven examples | not interpretable for that lineage in this format | not evidence that the base lacks the dependence |
| Qwen3-8B-Base shows a content-blind decision readout in plain text | the pattern predates Qwen's post-training, in this format | not Qwen-general; not the think block's role, which lexsem tests |

## 10. Environment, cost and stopping

**Environment.** The serve2 stack (torch 2.14.1+cu126, transformers 5.23.0, Python 3.11.18), named in the job scripts and checked at job start; L40S GPUs for every main and bridge arm, restricted to the `coc-gpu` and `ice-gpu` partitions; A40s for the floors; bf16 weights with a float32 log-softmax of the bf16 logits; one unpadded sequence per forward pass, through lexsem's scorer functions. Weights are downloaded at the pinned revision to node-local disk inside each job. Each arm's run folder (`runs/posttrain/<arm>/`) records the commit, configuration, input hashes, environment and the jobs that wrote it; a resumed job must match all of them, and the scorer refuses mock mode and a working tree with uncommitted changes.

**Cost**, from lexsem's second smoke rates (0.085 s per success sequence, 0.064 per failure-text sequence, about 0.07 per yes/no sequence on an L40S; A40 taken as twice as slow, a conservative guess), counting only the cells this pilot scores:

| Part | Arms | Sequences | GPU-hours |
|---|---|---|---|
| Plain format, Llama tree (lexsem's base failure cells reused) | 6 | 66,312 | 1.38 |
| Plain format, Qwen pair (lexsem failure cells reused) | 2 | 8,004 | 0.16 |
| Chat templates | 6 | 99,468 | 2.01 |
| Word2World bridge | 7 | 20,776 | 0.43 |
| Floors (A40), 300 items | 14 | 52,800 | 2.19 |
| Recognition check, 8 plain arms | 8 | about 12,500 short sequences | about 0.05 |
| Model downloads and loading, 17 jobs | | | about 1.4 |
| Smoke, all arms of all 9 checkpoints, 5 items | | | about 0.6 |
| **Total** | | | **about 8.2** |

The approved budget is about 7.5–9 GPU-hours. The smoke measures the real rates. **If the smoke projects more than 9 GPU-hours, or the seven-example fallback is needed, the study stops and reports before going on.**

## 11. Procedure

1. One independent adversarial review of this note and the code, recorded in `notes/posttraining_review1.md`, with the fixes applied.
2. An annotated tag `posttraining-design-frozen-2026-11-26` on the commit with this note, `wf/posttrain.py`, `wf/posttrain_score.py`, `wf/posttrain_analysis.py`, the job scripts and the tests, pushed before any score.
3. The smoke (`PT_SMOKE=5`, every arm of every checkpoint, timing and validity only: seconds per sequence, readout masses per condition, the token checks, no difference between conditions) from a fresh checkout at the tag. **Its readout is committed to section 12 of this note before the full run is submitted.**
4. The full run from the tag: one L40S job per checkpoint, one A40 floor job per checkpoint.
5. Any deviation in a dated entry here and in `notes/deviations.md`, committed before the affected scores are read.
6. The analysis once, from a clean checkout at the tag (`python -m wf.posttrain_analysis --runs runs/posttrain --lexsem <lexsem runs> --out notes/posttraining_metrics.json`), and the results in `notes/posttraining_results.md`.

## 12. Smoke record

**Smoke run, 2026-11-26** (recorded before the full run is submitted). Ten jobs, 7348558–7348567, ran from a fresh checkout at the freeze tag (commit e0e9c04) in the serve2 stack (torch 2.14.1+cu126, transformers 5.23.0): nine on L40S GPUs, one per checkpoint with all of its main and bridge arms, and one on an A40 with Tülu SFT's two floor arms. Each arm scored 5 failures and 5 successes in every cell, including the cells reused from lexsem. Only timing and validity were measured; no difference between conditions was computed.

**Checks.** No job stopped. The following all passed in every job:

- the strict loading check (parameter names equal the safetensors index);
- the GPU-type check;
- the stack assertion;
- the pinned revision;
- lexsem's sample hash and name-token counts;
- the BOS rule in plain text;
- the per-item name-copy check, including the described goal and the neutral task;
- the check that every failure's observation starts with the readout token.

The four Tülu checkpoints' 8 untrained embedding rows (ids 128,256 to 128,263) held at most $7 \times 10^{-6}$ of the probability at the reply's first position. The recognition check takes about 4.5 s per two objects, so about a minute per plain arm. The first five successes by id all begin with "You", so the smoke did not exercise a success whose first token is another word. Their first-token decision is read from the stored log-probability of their own first token, which needs no further code path.

**Rates**, seconds per unpadded sequence. The first arm of each job carries one-off warm-up and reads slightly higher.

| Arm kind | Success sequence | Failure-text sequence | Yes/no |
|---|---|---|---|
| Plain format, L40S, 8 checkpoints | 0.063–0.12 (0.063–0.075 after warm-up) | 0.059–0.063 | — |
| Chat template, L40S, 6 checkpoints | 0.066–0.071 | 0.064–0.068 | 0.068–0.074 |
| Word2World bridge, L40S, 7 checkpoints | 0.095–0.12 | 0.094–0.098 | — |
| Floors, A40 (Tülu SFT, plain and chat) | 0.125–0.130 | 0.115–0.122 | — |

**Readout masses under the neutral task** (mean over the 5 items; the gate uses the median over all items):

- *First-token mass on the failures:* the probability of " You" plus " Nothing" at the reply's first position. In plain text it was 0.86–0.94 for every checkpoint; the Llama base was at 0.91 and the Qwen base at 0.92. In the chat templates it was 0.78 for Meta Instruct, 0.91–0.95 for the Tülu checkpoints, and 1.00 for Qwen3-8B with the block.
- *First-token mass on the successes:* 0.95–1.00 everywhere.
- *Yes/no mass in the chat templates:* below 0.9 on some items for Tülu DPO, RL and GRPO (10–15 condition cells with a minimum under 0.9). Their yes/no readings may therefore fail the yes/no gate. Yes/no is secondary and is not part of any stage comparison.
- *Single items:* the plain-text Qwen3-8B, Tülu RL and the bridge arms each had a few condition cells in which one item fell below 0.7. No gate is read here.

**Tail of the failure text after " Nothing"** (log-probability of " happens."):

- −0.7 to −5.8 nats in plain text;
- −0.003 to −0.18 in the chat templates;
- exactly 0 for Qwen3-8B with the block and for the Word2World model in its own template.

So in plain text the full two-candidate score differs substantially from the first-token decision, which is why the failure-text sequence is scored on every cell.

**Cost.** The measured rates were applied to the cells of the full run. Projected scoring time:

- L40S: 3.8 GPU-hours for the main and bridge arms;
- A40: about 1.8 GPU-hours for the 14 floor arms, from Tülu SFT's measured A40 rates (0.12–0.13 s);
- download and loading: about 2.5 minutes per job, observed in the smoke (downloads to node-local disk were fast), about 0.7 GPU-hours over 17 jobs;
- the smoke itself: about 0.4 GPU-hours.

The total is about 6.7 GPU-hours. That is under the approved 7.5–9 and under the 9-hour stopping threshold, so the full run proceeds.

**CPU check of the success path (2026-11-26, after the full run was submitted, before any of its jobs started; CPU job 7349845, at commit e0e9c04).** The smoke never reached a success whose observation does not start with "You". So the frozen scorer's own job builder and readout bookkeeping (`build_jobs`, `readouts`), together with the analysis's `readout`, were run on CPU on the first success of each start word: t_000040 ("You"), t_000657 ("There's"), t_001375 ("The") and t_1000078 ("This"). The tokenizers were the real ones, from the access check's cache, and a tiny random model stood in for the weights. Seven arms were covered, spanning all prompt modes: Tülu SFT in plain text, in its template and in the bridge; the Llama base in plain text; Qwen3-8B in plain text and with the block; and the Word2World model in its template. Under the real, swapped and neutral tasks, every case passed four checks:

- the success sequence's continuation decodes to the item's own observation;
- the flag for "starts with the readout token" is false exactly for the three non-"You" words;
- the failure-text sequence starts with "Nothing";
- the analysis's first-token decision equals the stored log-probability of the item's own first token minus that of "Nothing", with the full and continuation scores finite.

No code was changed. The check script is outside the repository; it uses only the frozen functions.

## 13. Limits

- One lineage with intermediate stages, plus a base–final pair; a case study.
- Each stage changes data and objective together; Tülu's RL stage is narrow (math and instruction-constraint verifiers) and KL-regularized.
- "Base" means the released base checkpoint, whose pretraining data are closed and may contain instruction-like text.
- The plain format is off the usual input distribution for post-trained checkpoints; the chat-template arms measure how much that matters but do not remove it. The Tülu chat arm follows the template as shipped, which renders no BOS token.
- The sentences are artificial lines on a task card; controls absorb each carrier's own effect, not interactions between the carrier and the object.
- Behaviour, not mechanism: the first-token and continuation split is a split of the readout, not of the computation.
- The normative target is not zero on the log-odds scale (a simulator with residual doubt may move); sentences are compared with each other and checkpoints with each other.
- Step-0 readings are weak; shares need a goal effect that passes the reference rule.
- No multiplicity correction outside the 35 primary readings.

## 14. Files

- `wf/posttrain.py`: checkpoints, lineages, arms, conditions, cells, lexsem reuse bookkeeping, item sets, the bridge and seven-example prompts, the recognition pairs.
- `wf/posttrain_score.py`: the scorer (lexsem's functions; run folders, resume, refusals, smoke).
- `wf/posttrain_analysis.py`: gates, estimands, floors, reading rules, Holm, environment checks.
- `scripts/posttrain/access_check.py`: the CPU access and tokenizer check (section 2). `scripts/posttrain/cpu_dryrun.py`: a CPU dry run with real tokenizers and a tiny random model.
- `pace/posttrain_score.sbatch`, `pace/posttrain_cpu.sbatch`, `pace/posttrain_access.sbatch`.
- `tests/test_posttrain.py`: cells, reuse, readout definitions and the reading rules on planted scores.

## 15. Review

One independent adversarial review (2026-11-26) of the draft note and code found 7 must-fix, 12 should-fix and 6 minor points; every one was addressed before the freeze. The points and the changes are listed in `notes/posttraining_review1.md`. The main changes: the recognition check became a ranking among all 55 names by association; the Holm family is fixed at 35 readings, with every other reading labelled descriptive; a missing numerical floor makes a reading unreadable, and equivalence readings need their margin to be at least five floors; success cells are never reused from lexsem, and reuse needs recorded provenance, code identity, a bit-identity check in both strata and complete cells; the reference rule is two-sided; the competence gate compares later-step failures with the successes; the robustness rule's rank gain uses only the three task lines; token-id identity across a lineage is checked on every scored sequence.

## 16. Deviation D-PT1: lexsem's one-line smoke fix, and the provenance check's file list (2026-11-26)

Recorded after the freeze and before any pilot score was read. The scoring jobs had run, but only their job-log headers had been opened.

**What changed outside this pilot.** Lexsem commit 5aabb99 edited one line of `scripts/lexsem/score.py`, inside lexsem's `smoke()`. The success probe now runs only in arms that score the first-token format, because it had crashed lexsem's yes/no-only Llama chat arms. The diff:

```
-    for t in success_probe(list(successes)):
+    for t in success_probe(list(successes)) if "obs" in fmts else ():  # yes/no-only arms have no first token
```

The file's git blob went from fd6dabab20a831fe846146384b461ed0db7645a1 (lexsem 503ba47, also this pilot's freeze tag e0e9c04) to fcbf7c58c2233ad8102e475f060be51f495bf536. No other file that decides a score changed between 503ba47 and 5aabb99.

**Why it cannot affect this pilot's scores.**

- Every scoring and floor job (7349816–7349832) and the smoke (7348558–7348567) ran in `/storage/ice1/2/1/gmat3/text-wm-posttrain-run`, a clone made from this branch's own bundle at the tag. 5aabb99 is not in that clone's history. Its HEAD is e0e9c0481e3f027a8b24f215b61f5437eed2efac, with no modified tracked files. Its `scripts/lexsem/score.py` hashes to blob fd6dabab… (sha256 874cf8820cc044ee1d6eee459fac536718c7b45ae465ddf49a86aff3d34fac60), and the file was last written when the clone was made (08:25, before the first job). Every job log records commit e0e9c04.
- The pilot never calls lexsem's `smoke()`; its own smoke is `wf/posttrain_score.py`'s `smoke()`. From `scripts/lexsem/score.py` it uses only `load`, `prompt_ids`, `sequence`, `score_one`, `run_item`, `readout_ids`, `token_checks`, `sample_digest` and `environment`, none of which 5aabb99 touched.

**Effect on lexsem reuse.**

Lexsem's full-run scores used for reuse will come from a commit that contains 5aabb99. Their scoring path is unchanged too, since `readouts`, `score_successes`, `item_jobs`, `sequence` and `score_one` are identical. The analysis's provenance check, which requires identical files, is changed in two ways (`wf/posttrain_analysis.py`):

- *An exact exception for this one file version.* `ACCEPTED_BLOB_PAIRS` holds exactly (`scripts/lexsem/score.py`, fd6dabab…, fcbf7c58…), and a difference consisting only of these two blobs of that file is treated as identical. Any other difference in any file still fails the check, so the check is not loosened beyond this diff. A test confirms that 503ba47 and 5aabb99 pass against the tag while da2e312 still fails.
- *A latent bug, fixed.* For lexsem's runs, the check compared every pilot scorer file, including `wf/posttrain*.py` and the pin table, which a lexsem commit does not contain. So every reuse would have been refused. It now compares only the files that decide a lexsem score: `wf/lexsem.py`, `scripts/lexsem/score.py`, `scripts/qwen_audit/followup_score.py`, `wf/scoring`, `wf/rq1.py`, `wf/rq1x.py`, `wf/rq1v.py`, `wf/rq2.py` and `wf/data`. The reused checkpoint's pinned revision is compared through the environment records, as before. The pilot's own arms are still compared on the full scorer file list.

Nothing was rescored. Every other rule is unchanged.

## 17. Deviation D-PT2: the five reuse arms are scored by the pilot itself (2026-11-26)

Recorded and tagged (`posttraining-deviation-dpt2-2026-11-26`) before these jobs were submitted and before any pilot score was read.

**What changes.** The pilot scores every cell of five arms itself, with `--no-reuse`, through the same scoring path and on L40S GPUs. Under section 7, those cells would have come from lexsem's run files. The five arms are:

- Llama-3.1-8B base, plain text;
- Qwen3-8B-Base, plain text;
- Qwen3-8B, plain text;
- Qwen3-8B with the think block;
- Llama-3.1-8B-Instruct in its chat template.

Jobs are submitted from a clean checkout at this tag. The earlier run folders of these arms, scored at e0e9c04 with only the cells lexsem lacked, are moved unread to `runs/posttrain_superseded_reuse/` and are not used.

**Why.** When the pilot's full run finished, lexsem had not frozen and none of its full-run scores existed. The user chose to decouple the pilot from lexsem's schedule rather than wait.

**Cost.** About 1.1 to 1.3 L40S GPU-hours (four jobs, 75,312 sequences at the smoke's rates, plus loading). That brings the pilot to about 8.1 to 8.3 GPU-hours, within the approved 7.5 to 9.

**What does not change.** No decision rule, gate, margin, floor, Holm family or estimand changes. The floor arms already scored every cell themselves and are unchanged. The analysis code is unchanged: an arm scored with `reuse: false` is read from its own cells (the frozen `merge_lexsem` skips it).

**Provenance comparison with lexsem (fixed now, before either side's scores are read; it does not gate the analysis).** Lexsem froze at `lexsem-design-frozen-2026-11-26` (46e255b, scoring code identical to 5aabb99). When its scores exist, every cell scored by both studies will be compared, matched by item, condition and format. Lexsem's arms and the pilot's arms are paired as follows:

- `llama8b_base_plain` with the Llama base in plain text;
- `qwen8b_base_plain` with the Qwen base in plain text;
- `qwen_plain` with Qwen3-8B in plain text;
- `qwen_block` with Qwen3-8B with the think block;
- `llama_chat` with Llama-3.1-8B-Instruct in its chat template.

Both studies use the same GPU type, stack, revision, unbatched path, prompt builder and scorer functions, so the expectation is as follows.

- *Expected bit-identical* (the stored log-probabilities, rounded to five decimals as both studies write them; `notes/numerical_sensitivity.md` §4.1: same environment, same bits):
  - on the failures, the success-text sequence: log p(" You") and log p(" Nothing") at the first position, and every success-text token's log-probability;
  - on the failures, the failure-text sequence: every token's log-probability;
  - on the failures, the yes/no sequence: log p("Yes") and log p("No");
  - on the successes, under the neutral task (lexsem's base arms only): the failure-text and yes/no sequences, and the success-text token log-probabilities and log p(" Nothing"). Lexsem's own-first-token field (`lp_a` for those successes) is the same number as the pilot's first stored token log-probability.
- *Expected only within the floor:* none. Any difference is reported as a provenance finding. A cell differing by less than the arm's measured floor would point to an unrecorded environment difference (for example a different node's software or driver), and a larger one to a different prompt or code path. Neither changes this pilot's readings, which use only its own scores.

The comparison is reported in the results note, with counts of identical, within-floor and larger differences per arm and readout.

## 18. Deviation D-PT3: the seven-example fallback for the Llama tree, staged (2026-11-27)

Recorded and tagged (`posttraining-deviation-dpt3-2026-11-27`) before any seven-example arm was scored.

**Trigger.** In the frozen analysis (job 7364918, at 26d5ac1), the Llama-3.1-8B base failed the competence gate in the three-example plain format. The gate requires that, under the neutral task, the model separate the determined successes from the later-step failures. The base's area under the ROC curve was 0.54 on the full two-candidate score, against the required 0.70, and 0.50 on the first token. Lexsem independently found 0.50 for the same base.

**Frozen fallback** (section 5). Every plain-format checkpoint of the lineage is rescored with seven few-shot examples (sets 1 and 2), and only those arms are read for the lineage's plain-format comparisons.

**Staging, chosen by the user.** The fallback is run in two steps; the rule is unchanged.

1. The base's seven-example arm (`llama-base_plain7`) is scored first: about 0.4 GPU-hours on an L40S, from a clean checkout at this tag. Only its competence gate, its AUC values and its format checks (mass, degeneracy, the token and loading checks) are read.
2. If the base fails the gate again, the Llama lineage is recorded as "not interpretable in this format" and nothing more is scored.
3. If it passes, the other five checkpoints' seven-example arms and all six seven-example floor arms are scored, about 3.1 to 3.6 GPU-hours. The analysis then runs once more, with the seven-example arms in place, under the same frozen code and rules. The pilot total would then be about 12 GPU-hours, beyond the 7.5 to 9 first approved; the user approved this staging.

**Unchanged.** The gate and its 0.70 threshold, every decision rule, margin, floor rule, the Holm family of 35, and the analysis code. The seven-example arms are analysed exactly as the frozen code already provides (`SEVEN`, `resolve`); the Qwen readings are unaffected, since the Qwen base passed its gate.
