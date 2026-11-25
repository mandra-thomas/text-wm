# Option C: a closed-loop agent with a world-model verifier: results (2026-11-23; updated 2026-11-24)

**Status.** Exploratory study, frozen in procedure. The design and analysis were frozen before anything ran at the annotated tag `optionc-design-frozen-2026-11-22` (commit `54cb71e`, branch `optionc`; [optionc_design.md](optionc_design.md), review in [optionc_review1.md](optionc_review1.md)). The smoke run, the step-0 check and its analysis ran at that commit. After the step-0 check, and before any census episode existed, the pre-declared reallocation was exercised (commit `2a86683`, configuration and the test that pins it only; design note, final section). Both census runs and the analysis ran at `2a86683` in the clean checkout `text-wm-fix40` on ICE. Every number below is quoted from two files copied unchanged from the cluster: [optionc_step0_metrics.json](optionc_step0_metrics.json) (the step-0 analysis, job 7246046; SHA-256 `6e4512c7…`) and [optionc_metrics.json](optionc_metrics.json) (the main analysis, job 7252705; SHA-256 `1721ee72…`). Nothing here re-verdicts a frozen output or adds a test; where this note does arithmetic on the reported numbers (for example a rejection share from the counts of decision reasons), it says so, and anything not fixed by the design is labelled descriptive or exploratory.

**Update (2026-11-24).** Both analyses were rerun at `01cad81` in `text-wm-fix40` (step-0 analysis job 7295369, main analysis job 7295370), after the argmax-rule run had finished and Stage 1a's scoring runs had been linked into the checkout. The two metrics files named above are now those reruns' outputs, copied unchanged (step-0 SHA-256 `f34686ad…`, main `eb01f354…`). Compared key by key with the earlier files, the reruns add exactly two things and change nothing else: each model's step-0 `stage1a_agreement` entry (section 3.1) and the argmax run (section 6.3). Every other number in this note is therefore unchanged. The update also adds why the step-0 scores differ from Stage 1a's (section 3.1, with a diagnostic), the step-0 classification computed in plain text (section 10.3), and the assessment of the Llama loop with the user's decision not to run it (section 13). The two follow-ups were specified, committed and pushed before any of their scores existed, in [optionc_followups_2026-11-24.md](optionc_followups_2026-11-24.md) (branch `optionc`, commit `0b7518e`; their code is in `scripts/optionc_followup/` there).

## 1. The question

The project's main finding (RQ2, replicated in a yes/no format) is that when a language model is prompted to act as a world model for an ALFWorld agent, its predicted success for an action whose failure the history already settles goes up when the agent's stated task names the action's object. A "world model" here is simply the language model asked to predict what the environment will reply to an action. The finding is about predictions. Three later stages asked whether it changes decisions:

- **Stage 1a** (fixed candidate menus; [decisionsim_results.md](decisionsim_results.md)) found that a Qwen3-8B verifier shown the real task chose failing actions at 23.9% of states against 4.4% task-blind, mostly at the first step of an episode, through two ingredients acting together: any concrete task makes Qwen's yes/no verifier reject the agent's working first moves (the "step-0 collapse"), and naming the object ranks failing goal-object actions first in a fallback menu rich in them. Its pre-registered gate judged a closed-loop study "worth doing".
- **Stage 1b** (the agent's own sampled candidates, one decision at a time; [decisionsim_agent_results.md](decisionsim_agent_results.md)) found "no reliable difference" for the two susceptible priority verifiers. Its Qwen3-8B contrast did make chosen actions fail more often (+2.29 points per decision), but mainly because any concrete task made Qwen reject working alternatives and abstain, not through the object-naming bias.

Option C is the closed-loop study the gate licensed. The agent plays whole episodes, and a verifier screens every action before it is executed, so a wrong decision has its real cost (a wasted step, a loop, a lost game) and the agent's recovery is counted too. The question is:

**Does a world-model verifier that is shown the agent's task make the agent succeed less often than the same verifier without the task?**

Write the success of the episode of game $g$ under verifier arm $a$ and sampling seed $s$ as a 0 or 1, and its mean over seeds within the game as $\bar{y}_{g,a}$. The primary estimand is the game-paired difference in success between the task-aware and the task-blind verifier over the 134 games,

$$
\Delta_{\text{success}} = \frac{1}{G} \sum_{g=1}^{G} \left( \bar{y}_{g,\text{aware}} - \bar{y}_{g,\text{blind}} \right), \qquad G = 134 .
$$

A negative value means that showing the task costs success. The secondary questions are whether the task makes the agent execute more failing actions, whether the verifier's own interventions do more harm than good, and whether any difference comes from naming the task's object.

## 2. The loop, the rule and the arms

**Agent and environment.** The agent is the RQ3 Qwen3-8B ReAct agent with thinking off. "ReAct" means the agent writes a short thought and then an action line at each step; its prompt holds the current observation, its own history of observations and executed actions, and the list of admissible actions. Its proposal is its greedy reply (temperature 0). It plays the 134 solvable `valid_unseen` ALFWorld games, with at most 30 steps per episode. The verifier is invisible to the agent: no prediction, verdict or rejected proposal is ever written into the agent's prompt.

**The verifier and its two readouts.** The verifier is a second copy of Qwen3-8B, prompted as a world model with the history of the episode so far, the action, and the task line of its arm. It is read in one of two ways, and each run uses one readout throughout. The yes/no readout asks whether the action will succeed or the environment will reply "Nothing happens.", and scores the log-odds of the two answer tokens,

$$
L_{\text{yn}} = \log P(\text{Yes}) - \log P(\text{No}).
$$

The observation readout asks the model for the next observation and reads only the first token of its reply. Every ALFWorld failure reply is "Nothing happens.", so the probability that the reply does not begin with "Nothing" is the model's own predicted probability of success,

$$
L_{\text{obs}} = \log\bigl(1 - P(\text{Nothing})\bigr) - \log P(\text{Nothing}).
$$

Both scores are in nats (natural-log units), are rounded to 1/64 nat, and the verifier **accepts** an action when the rounded score is above zero. Both readouts go through the same scorer with Qwen3's thinking-off chat template, which inserts an empty `<think></think>` block before the answer (section 10.3).

**The decision rule ("accepted-only").** The verifier scores the agent's proposal. If it accepts, the proposal is executed. If it rejects, 20 alternatives are sampled from the agent's own unmodified prompt (Qwen's recommended non-thinking settings), the distinct actions are ranked by how often they were sampled (at most 8 candidates, the proposal first), and the first alternative the verifier accepts is executed. If none is accepted, or the samples contain no other action, the proposal is executed anyway; this is an **abstention**, and it hands the decision back to the agent. An **override** is a step where the executed action is not the proposal. At every override the proposal was also executed once on a private replay of the state, so each override can be classified by the true outcomes of both actions: **override harm** means the proposal would have worked and the chosen action failed, **override help** the reverse.

**The arms.** All arms share the verifier model, readout and rule; only the task line in the verifier's prompt differs.

- **No verifier:** the agent's proposal is always executed (this is the RQ3 no-foresight agent). It is greedy, so it has one episode per game.
- **Task-blind:** the task line is deleted from the verifier's prompt (the few-shot examples keep their own task lines, as in every earlier stage).
- **Task-aware:** the real task line.
- **Swapped:** the real task line with its object replaced by another object present in the room, one per game, fixed for the whole episode. 125 of the 134 games have such an object. In the loop this is a verifier told a wrong task, not a neutral "any concrete task" control: it can reject the real task's own progress steps (section 6.2).

**Matched-state scoring.** At every step of every task-aware episode, the proposal and every alternative the aware verifier scored were also scored under the blind and swapped prompts and never acted on. This gives the three prompts' judgements of the identical decision, free of the fact that different arms lead to different states.

**Common random numbers.** Within a game, every agent reply, sample, verifier score and replay outcome is cached under the exact prompt that produced it and reused by every later episode of that game in the same run. Two arms therefore play the identical episode up to the first step where their verifiers choose differently, and differ afterwards only because of that choice.

**Outcomes and inference.** Per episode: success; invalid actions executed (real reply "Nothing happens."); invalid actions among the first 10 steps; override harm minus help (the verifier's own net contribution to failures); and descriptive counts (abstentions, overrides, repeated action-and-reply pairs, steps, the RQ3 score $S$, which is minus the number of steps, or −31 for a lost episode). Every comparison is paired by game, as each game's difference of seed means. Intervals are 95% percentile bootstraps over games (2,000 resamples) and p-values two-sided whole-game sign-flip permutations (1,000), the tests used throughout the project. A verdict of **harm** needs the interval of the aware-minus-blind difference entirely on the worse side of zero and p below 0.05; **help** the reverse; otherwise **no reliable difference**. Success is the single primary test. Override harm minus help and invalid actions in the first 10 steps are the two key secondaries, and invalid actions per episode a secondary; they get the same categories with the sign reversed (more is worse), uncorrected for multiplicity. A harm verdict carries an attribution label, which is computed only when the obstruction check of section 6.2 is clear.

## 3. The step-0 check

**What it asked.** Before any episode, the design asked whether Qwen's step-0 collapse is specific to the yes/no question (Qwen reading it as "will this action accomplish the task?"), or whether a stated task also changes what Qwen predicts the next observation to be. It scored Stage 1a's 134 step-0 states (one per game, with Stage 1a's full candidate sets: the agent's step-0 proposal, which always works, random admissible actions, and failing goal-object actions) under the aware, blind, swapped and neutral prompts and both readouts, for Qwen3-8B and Llama-3.1-8B (jobs 7245678 and 7245679).

**The classification rule** (fixed at the freeze). For each readout, the gap is the aware-minus-blind difference in the share of the 134 working step-0 proposals accepted, paired by state, with its game-bootstrap interval. A readout *collapses* when the interval's upper end is below −30 points and shows *no gap* when the whole interval lies within ±10 points; otherwise it is *unclear*. A blind arm is *competent* when it passes the wave A competence rule on all step-0 candidates (AUC at least 0.70, accepting at most 95% of failures and at least 5% of successes; AUC is the probability that a random working candidate scores above a random failing one). "Question-specific" requires a competent blind yes/no arm, a yes/no collapse, and an observation readout that is competent and shows no gap.

| Step-0 check, 134 states | Qwen3-8B yes/no | Qwen3-8B observation | Llama-3.1-8B yes/no | Llama-3.1-8B observation |
|---|---|---|---|---|
| Working proposals accepted, task-aware | 7.5% [3.7, 11.9] | 100% | 89.6% [84.3, 94.8] | 100% |
| Working proposals accepted, task-blind | 89.6% [84.3, 94.8] | 100% | 100% | 100% |
| Paired gap, aware − blind (points) | −82.1 [−88.1, −74.6]; 110 games non-zero | 0.0 [0.0, 0.0]; no game non-zero | −10.4 [−15.7, −5.2]; 14 games non-zero | 0.0 [0.0, 0.0] |
| Gap state | collapse | no gap | unclear | no gap |
| Blind arm: AUC; failures accepted; successes accepted | 1.00; 0.0%; 97.2% | 1.00; 18.8%; 100% | 0.98; 5.0%; 97.8% | 1.00; 100%; 100% |
| Blind arm competent | yes | yes | yes | no (accepts every failing candidate) |
| Classification | question-specific (supports the task-success reading) | | mixed (the yes/no interval does not settle it) | |

For Qwen3-8B the yes/no readout reproduces the collapse: shown the real task, it accepts 7.5% of the agent's working first moves against 89.6% task-blind, and the swapped prompt behaves like the aware one (5.6%). The same states read through the observation readout show no gap at all: every working proposal is accepted under every prompt, in every game. The classification is therefore "question-specific (supports the task-success reading)": the collapse is a property of how the yes/no question is answered under a task, not of what the model predicts the environment will say. Per the design's rules, any yes/no result at step 0 is described below as a consequence of how the question is read.

"No gap" concerns only the acceptance of working proposals. The stated task does move the observation readout's scores, in both directions the earlier studies describe. The any-task drop (the mean aware-minus-blind score on working `go to` candidates) is −8.49 nats [−8.92, −8.09] under the observation readout, against −39.1 [−39.8, −38.4] under yes/no; under the observation readout the working `go to` scores stay far above zero (median 26.25 nats aware, 34.4 blind), so none is rejected, whereas under yes/no they are pushed below zero (median −22.5 aware, +19.25 blind). The naming lift (the mean aware-minus-swapped score on failing goal-object candidates) is +7.04 nats [6.45, 7.62] under the observation readout and +7.60 [7.10, 8.09] under yes/no. In acceptance terms the observation readout's aware arm accepts 77.2% of the failing step-0 candidates, against 18.8% task-blind, 28.8% swapped and 8.1% neutral; its AUC stays at 0.99 because the working candidates score even higher. So the observation readout removes the collapse but not the RQ2 bias: at step 0 it accepts most failing goal-object actions when the task names their object. This matters for reading the loop (section 5.3).

For Llama-3.1-8B the yes/no gap is −10.4 points: not a collapse, but its interval reaches beyond −10, so the classification is "mixed (the yes/no interval does not settle it)". Its observation readout is not competent (its blind arm accepts every failing step-0 candidate), so Llama would have had no observation-readout run. Llama's loop runs (an optional Tier 3) were not submitted. *(2026-11-24: not run, by the user's decision; section 13.)* Its yes/no readout, the one a Llama loop would use, is competent: its blind arm has AUC 0.98 and accepts 5.0% of failing and 97.8% of working candidates.

**One check promised by the design is missing from the metrics.** The design said the step-0 yes/no scores under the aware, swapped and neutral prompts, which are Stage 1a's prompts, would be compared with Stage 1a's saved scores and reported as the largest difference after rounding. The analysis computes this only when it finds Stage 1a's scoring runs, and those runs are not under the `text-wm-fix40` checkout's `runs/` folder, so the comparison was skipped without an error and `optionc_step0_metrics.json` has no `stage1a_agreement` entry. The rates themselves agree with the with-block rescoring of the same states reported in the Qwen pipeline audit (aware 7.5%, swapped 5.6%, blind 89.6%; [decisionsim_results.md](decisionsim_results.md), last section), but the score-by-score comparison was not made. *(2026-11-24: now made; section 3.1.)*

### 3.1 Agreement with Stage 1a's scores, and why they differ (added 2026-11-24)

**What the rerun found.** With Stage 1a's scoring runs linked into the checkout, the step-0 analysis rerun (job 7295369) computed `stage1a_agreement`. It compares each step-0 yes/no score under the aware, swapped and neutral prompts with Stage 1a's saved score for the same prompt: 3,867 pairs per model, both rounded to 1/64 nat. Most pairs differ.

| | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Pairs compared | 3,867 | 3,867 |
| Pairs that differ after rounding | 2,807 (72.6%) | 2,414 (62.4%) |
| Largest difference after rounding (nats) | 2.0 | 1.125 |
| Mean absolute difference before rounding (nats; computed here) | 0.289 | 0.101 |
| Pairs on opposite sides of zero, so accepted by one and rejected by the other (computed here) | 6 | 36 |
| GPU, Stage 1a / Option C (job records) | A100-PCIE-40GB / L40S | A40 / L40S |

The differences are small next to the scores: the two sets correlate at r = 0.9997 for Qwen and 0.9992 for Llama (computed here). The mean signed differences are small next to the spread of the differences. For Qwen they are +0.008 nats under the aware prompt, +0.018 under the swapped one and +0.125 under the neutral one, against a standard deviation of 0.40. For Llama they are −0.06 to −0.03, against 0.13.

Every difference is, to within 10⁻⁴, a whole multiple of 1/8 nat. That is what rounding the two logits to bf16 produces: bf16 numbers between 16 and 32 are spaced 1/8 apart, and between 32 and 64 they are spaced 1/4 apart. The working guess was the GPU: the scores are computed in bf16, and different GPUs can round the same computation differently.

**What was checked, and what each check rules out.** Everything below was specified before the diagnostic ran ([optionc_followups_2026-11-24.md](optionc_followups_2026-11-24.md), section 1).

- **Prompt text, item hashes, chat template, empty think block, model revision: ruled out.** Each run's own code was used to rebuild its prompts on CPU: the Option C checkout for Option C, and a copy of `wf/` at Stage 1a's scoring commit `eeeb7dd` for Stage 1a. Every rebuilt prompt matches the hash its run stored with its score (3,867 of 3,867 for each run and model). Across the two runs, the messages are identical for all 3,867 pairs, and so are the token ids the scorer reads, rendered by each code's own tokenizer at its pinned revision. The Qwen revision is `b968826d…` and the Llama revision `0e9e39f2…` in both. Each code's own template settings (thinking off) are the same, and every Qwen prompt ends with the empty think block in both runs.
- **Software stack: ruled out.** Both jobs ran the same Python environment, `.envs/serve`, with torch 2.6.0 and transformers 4.51.3. Nothing in its `site-packages` is newer than 2026-10-28, earlier than both jobs. The newer `serve2` stack (transformers 5.x) was not involved.
- **Batch composition, padding, dtype and code path: ruled out by reading the code.** Both runs make one unpadded forward pass of one prompt, with bf16 weights loaded by the same `open_scorer`. Both take the last position's logits from the full logits tensor and cast them to float32 before the log-softmax. Stage 1a used `score_one_token_pair`; Option C used `next_token_logprobs`. The step-0 run never turned on the last-position-only option, so these are the same computation. The only scorer change between the two commits adds that unused option.
- **GPU type: established for Qwen3-8B by the diagnostic.** Option C's own step-0 scoring path was run on the Qwen prompts on an A100-PCIE-40GB, Stage 1a's GPU type (job 7300308, 8.6 minutes; the node, `atl1-1-03-007-35-0`, is also the one Stage 1a's job used). It reproduced Stage 1a's 3,867 scores exactly: all 3,867 identical, mean absolute difference 0.0 before rounding. It differed from Option C's L40S scores exactly as Stage 1a does (1,060 identical, mean absolute difference 0.289, largest 2.0, 6 acceptance flips). The rescoring ran on the very node Stage 1a used, so on its own it cannot separate the GPU type from the individual card. But reruns on different L40S nodes reproduce each other at 99.2% (the reference below), so it is the hardware type that matters.
- **GPU type: established for Llama-3.1-8B by the diagnostic.** The same rescoring of the Llama prompts on an A40, Stage 1a's GPU type for Llama (job 7300309, 11.5 minutes), reproduced Stage 1a's 3,867 Llama scores exactly: all identical, mean absolute difference 0.0. It differed from Option C's L40S scores exactly as Stage 1a does: 1,453 identical, mean absolute difference 0.101, largest 1.125, 36 acceptance flips. So for both models the whole disagreement is the GPU type.

**How this compares with the project's reference.** The wave B diagnostic quotes, as the size of a change of GPU type in the old stack, Qwen yes/no scores differing by 0.035 nats per score with 91% identical, against 0.248 nats and 30% identical for the stack change. Recomputed here, that reference compares `rq1v_score` with `size_score_qwen3-8b` (14,582 scores) and gives the same 0.035 and 90.9%. But seven of `rq1v_score`'s eight shards and both `size_score` shards ran on L40S GPUs. Split by GPU, the 12,869 scores where both runs used L40S GPUs agree 99.2% (0.003 nats per score). The 1,713 scores from the one shard that ran on an A40 agree only 28.6% (0.273 nats per score, largest 2.25).

So the 0.035 figure mostly measures reruns on the *same* GPU type. A true change of GPU type moves Qwen's yes/no scores by about as much as the stack change did, and as much as is seen here (27.4% identical, 0.289 nats). This qualifies the wave B diagnostic's "about seven times the change measured across GPU types": against an A40-versus-L40S change in the same stack (0.273 nats), the stack change's 0.248 nats is about the same size per score. *(This is a reading of the archived records, not a re-verdict of the wave B check.)*

**What it means for Option C.** Nothing in the step-0 check changes. The classification is computed within one run on one GPU type, and the acceptance rates agree with Stage 1a's (aware 7.5%, swapped 5.6%, blind 89.6%). For Qwen, 6 of 3,867 scores (0.16%) change sides of zero between the GPU types. For Llama, 36 do (0.9%; its scores sit closer to zero). Comparisons *across* runs made on different GPU types carry per-score noise of a few tenths of a nat. That is negligible next to the multi-nat task effects, but it is enough to move individual near-zero accept/reject calls.

## 4. The reallocation, and why

The frozen plan (the user's decision of 2026-11-22, about 70 to 80 GPU-hours) made the yes/no readout primary, for continuity with the gate, and pre-declared one reallocation: if the step-0 check found Qwen3-8B "question-specific" with a competent observation readout, the three seeds of the aware and blind arms could move to the observation readout, which would then become the primary run, and yes/no would run at one seed as a secondary. The reasoning, fixed before any data, was that under "question-specific" a yes/no result says something about how a yes/no question is read under a task, while the observation readout is the direct test of what a world model predicts.

The condition held (section 3), and the reallocation was exercised at commit `2a86683`, after the step-0 check and before any census episode; the smoke run's two training games had been read for the engineering gate only. The primary run became `optionc_qwen3-8b_obs`: no verifier (seed-free), task-blind and task-aware at three seeds, swapped at one seed, with matched-state scoring. The secondary run `optionc_qwen3-8b_yn` has task-blind, task-aware and swapped at one seed each. The no-verifier episodes were moved to the primary run, because the analysis takes its reference episodes (and the drift check's) from there; the yes/no run's comparisons with no verifier therefore use those episodes and are paired by game but not by common random numbers. No code, threshold, rule or readout definition changed. The analysis checks the reallocation condition itself and refuses an observation-readout primary run without it.

## 5. The primary result: observation readout, accepted-only rule

### 5.1 The verdict

| Observation readout (arms at 3 seeds except none and swapped) | No verifier | Task-blind | Task-aware | Swapped (125 games) |
|---|---|---|---|---|
| Success | 45.5% [36.6, 53.7] | 47.3% [38.8, 55.2] | 46.0% [37.3, 54.2] | 44.0% [35.2, 52.8] |
| Invalid actions per episode | 4.80 [3.87, 5.79] | 3.72 [3.01, 4.56] | 3.80 [3.10, 4.63] | 3.86 [3.10, 4.70] |
| Invalid actions in the first 10 steps | 0.45 [0.30, 0.61] | 0.41 [0.28, 0.55] | 0.41 [0.28, 0.55] | 0.42 [0.30, 0.57] |
| Override harm minus help per episode | 0 | −0.53 [−0.76, −0.33] | −0.47 [−0.70, −0.29] | −0.55 [−0.86, −0.28] |
| Overrides; abstentions per episode | 0; 0 | 0.76; 0.68 | 0.61; 0.50 | 0.82; 0.54 |
| Proposals rejected (share of verified steps, from the reason counts) | | 6.5% | 5.0% | 6.0% |
| First proposal rejected | | 0% | 0% | 0% |
| Episodes; steps | 134; 2,987 | 402; 8,920 | 402; 8,937 | 125; 2,813 |

**Primary verdict: no reliable difference.** The task-aware verifier's success minus the task-blind verifier's is −1.2 points [−3.0, 0.0] (p 0.23, 134 games). In words: with the observation readout, the agent won 46.0% of its episodes when the verifier saw the task and 47.3% when it did not, and the difference is not distinguishable from zero.

The interval needs a careful reading. Only 3 of the 134 games have any success difference between the two arms, and in all three the aware arm did worse; that is why the upper end is exactly zero. With three games carrying the whole difference, the smallest p-value a sign-flip test can give is 0.25, so the test could not have reached significance in either direction, and the bootstrap interval describes those three games rather than giving a precise bound. What can be said firmly is that the two verifiers led to a different outcome in 3 games out of 134.

The reason is that the two verifiers almost never disagree. Of the 402 (game, seed) pairs of aware and blind episodes, 337 (83.8%) are identical action for action from start to finish; the other 65 diverge at a median step of 23, late in a 30-step episode. The aware verifier rejected 5.0% of proposals and the blind one 6.5% (shares computed here from the counts of decision reasons), and neither rejected a single first move.

**Key secondaries: no reliable difference for both.** Override harm minus help, aware minus blind, is +0.06 per episode [−0.03, +0.15] (p 0.24, 21 games non-zero). Invalid actions in the first 10 steps, aware minus blind, is +0.002 per episode [0.000, +0.007] (p 1.0); only one game differs, so this test could not have been significant (smallest attainable p 1.0). The secondary outcome invalid actions per episode is also "no reliable difference": +0.08 [−0.07, +0.24] (p 0.29).

**Obstruction check: clear** (section 6.2 explains it). The swapped verifier accepted 89.5% of the agent's working proposals on the real goal object against 96.8% for the aware one, a gap of 7.3 points, inside the 10-point margin. Because the primary verdict is not harm, no attribution label is computed.

### 5.2 Verifier value (descriptive)

Against no verifier, both verifiers remove about one invalid action per episode: task-blind minus none −1.08 [−1.53, −0.68] (p 0.001), task-aware minus none −1.00 [−1.48, −0.58] (p 0.001). Their own interventions are net helpful (override harm minus help −0.53 and −0.47 per episode), almost entirely through help: the blind verifier's overrides rescued a failing proposal 0.56 times per episode and harmed a working one 0.02 times. That does not reach success: blind minus none +1.7 points [−1.0, +4.7] (p 0.31, 6 games differ), aware minus none +0.5 [−1.7, +3.0]. The design predicted a blind-verifier gain of 0 to 8 points (central +3) and about one fewer invalid action; the point estimates fall in that range. The fewer failures came with fewer repeated action-and-reply pairs (blind minus none −0.48 per episode [−0.87, −0.07]), which fits the design's guess that the verifier helps mainly by breaking the agent's loops, though the metrics do not classify which steps were rescued.

### 5.3 What the observation verifier does with the task (descriptive)

The per-step acceptance rates show both sides of the stated task, on the agent's own proposals during the loop:

| Observation readout, per step, by the proposal's true outcome | Task-blind | Task-aware | Swapped |
|---|---|---|---|
| Working proposals accepted | 98.4% | 99.3% | 98.5% |
| Failing proposals accepted | 72.9% | 76.8% | 75.3% |
| Failing proposals on the real goal object accepted | 43.9% (264 steps, 23 games) | 67.1% (298 steps, 22 games) | 46.3% (82 steps, 16 games) |
| Working proposals on the real goal object accepted | 90.0% (852 steps, 86 games) | 96.8% (841 steps, 86 games) | 89.5% (266 steps, 77 games) |

Two things stand out. First, the observation verifier is a weak screen in the loop: it accepts about three quarters of the agent's failing proposals under every prompt, although at Stage 1a's step-0 candidates its blind arm accepted only 18.8% of failures. The agent's own failing proposals in the loop are evidently harder to call from the first token than the generated goal-object failures of the step-0 menus; the metrics do not say which kinds of failure they are. Second, the task works on goal-object actions in both directions: the aware verifier accepts failing goal-object proposals more often (the RQ2 bias) and also accepts working goal-object proposals more often (it rejects fewer of the agent's real progress steps). The matched-state analysis in section 7 confirms both on identical states. These two effects push outcomes in opposite directions, and failing goal-object proposals are rare (298 of 8,937 aware steps, in 22 games), so they leave almost no trace in the outcomes. The aware verifier did rescue failing proposals a little less often than the blind one (override help −0.08 per episode [−0.15, −0.005], p 0.050), descriptive and uncorrected.

## 6. The secondary result: yes/no readout, accepted-only rule, one seed

### 6.1 The verdicts

| Yes/no readout (one seed per verifier arm) | No verifier (from the primary run) | Task-blind | Task-aware | Swapped (125 games) |
|---|---|---|---|---|
| Success | 45.5% [36.6, 53.7] | 52.2% [44.0, 61.2] | 47.8% [38.8, 56.0] | 54.4% [45.6, 63.2] |
| Invalid actions per episode | 4.80 [3.87, 5.79] | 3.07 [2.40, 3.78] | 4.19 [3.25, 5.22] | 3.46 [2.62, 4.38] |
| Invalid actions in the first 10 steps | 0.45 [0.30, 0.61] | 0.31 [0.22, 0.41] | 0.49 [0.36, 0.64] | 0.42 [0.29, 0.57] |
| Override harm; help; harm minus help per episode | 0; 0; 0 | 0.15; 0.97; −0.82 | 0.22; 0.72; −0.50 | 0.19; 0.84; −0.65 |
| Abstentions per episode | 0 | 4.99 | 9.12 | 9.08 |
| Proposals rejected (share of verified steps, from the reason counts) | | 36.9% | 52.6% | 56.0% |
| Working proposals accepted, per step | | 72.7% | 55.9% | 51.7% |
| First proposal rejected; first proposal overridden | | 10.4%; 6.7% | 92.5%; 0% | 93.6%; 0% |
| Episodes; steps | 134; 2,987 | 134; 2,848 | 134; 2,962 | 125; 2,650 |

**Success: no reliable difference.** Aware minus blind is −4.5 points [−11.2, +2.2] (p 0.29, 22 games differ). At one seed the interval is wide, as the design's precision table expected (a half-width of 5.2 to 7.5 points with one seed); a success cost of up to about 11 points is not excluded.

**Invalid actions per episode: harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task).** Aware minus blind is +1.12 invalid actions per episode [+0.33, +1.96] (p 0.005, 73 games differ). The two key secondaries carry the same label: invalid actions in the first 10 steps +0.18 [+0.06, +0.31] (p 0.007), and override harm minus help +0.32 per episode [+0.08, +0.57] (p 0.010). These are secondary tests of a secondary run, uncorrected.

"Harm" here means worse than the task-blind verifier, not worse than having no verifier. Against no verifier, the aware yes/no verifier still removes 0.61 invalid actions per episode (interval −1.59 to +0.39, p 0.22) and its interventions are still net helpful (harm minus help −0.50); the blind verifier removes 1.73 [0.93, 2.57] and is more helpful (−0.82). The stated task takes away part of the verifier's benefit.

**How the harm arises (descriptive).** Under the yes/no readout the aware verifier rejects about half of all proposals (52.6%, against 36.9% task-blind), including 44% of the agent's working proposals (55.9% accepted). Under the accepted-only rule a rejection usually ends in an abstention (9.1 per episode aware, 5.0 blind; +4.13 [+3.27, +5.08]), which returns the agent's own proposal. So the aware verifier overrides no more often than the blind one (2.50 against 2.85 per episode), but its overrides are less useful: it rescues failing proposals less often (help −0.25 per episode [−0.47, −0.03], p 0.03) and does slightly more harm (+0.07 [−0.04, +0.18], p 0.28). The agent then repeats itself more (repeated action-and-reply pairs +1.27 per episode [+0.44, +2.26]). This is the route Stage 1b found for its Qwen contrast: any concrete task lowers Qwen's acceptance of working actions, so it gives up more often and loses rescues of the agent's failing repeats. In the loop it compounds to about one more invalid action per episode.

At step 0 the collapse appears in full (the aware verifier rejects 92.5% of first proposals, the swapped one 93.6%, the blind one 10.4%), but under this rule it costs nothing directly: the aware verifier never overrode a first move, because at step 0 it accepted no alternative either and abstained. The blind verifier overrode 6.7% of first moves.

### 6.2 The obstruction check, and what "attribution not identifiable" means

The design's attribution rule asks how much of an aware-versus-blind harm is recovered by telling the verifier a task about another object, on the 125 games that have one. With the cost $c$ and the recovery $r$ defined there,

$$
c = D_{\text{blind,aware}}, \qquad r = D_{\text{swapped,aware}}, \qquad \text{attributable to the real task when } r \ge \tfrac{1}{2} c ,
$$

where $D_{xy}$ is the mean over games of the per-game difference between arms $x$ and $y$. The logic is that a swapped task keeps "a concrete task is present" and removes only "the task names this object", so if the swapped verifier behaves like the blind one, naming the object is what does the harm. Review 1 pointed out that in a live loop this logic breaks: a verifier told to put away a different object rejects the real task's own progress steps (taking, cleaning or placing the real goal object), which the aware verifier accepts. The swapped arm then differs from the aware arm in two ways at once, naming and obstruction of the real task, and comparing their success cannot separate them.

The pre-registered **obstruction check** detects this from fields every step logs. For each arm it takes the agent's *working* proposals that act on the real goal object and computes the share the verifier accepted. If the swapped verifier accepts them more than 10 points less often than the aware verifier, the check fires, and any harm verdict is labelled "attribution not identifiable in the loop (the swapped verifier obstructs the real task)"; no "any concrete task" reading is allowed.

Under yes/no it fired. The aware verifier accepted 60.5% [51.0, 70.2] of working real-goal proposals (258 steps, 93 games), the swapped verifier 41.5% [35.1, 48.7] (229 steps, 86 games), a gap of 19.0 points against the 10-point margin. The blind verifier accepted 56.9% [52.0, 61.6]. The matched-state comparison on identical states agrees: the swapped prompt rejects 62.7% of the working real-goal proposals that the aware prompt rejects 40.3% of (difference −22.5 points [−29.3, −16.3]). Under the observation readout the gap was 7.3 points and the check was clear (section 5.1).

So the label means this: the aware yes/no verifier costs the agent about one invalid action per episode relative to a task-blind one, but the loop cannot say at the level of outcomes whether naming the real object contributes, because the only comparison arm that keeps a concrete task also blocks the real task's progress. Decision-level attribution, which does not have this problem, is in section 7.

For completeness, the exact decomposition on the 125 swap games (descriptive; the identity holds for every outcome) splits the aware-minus-blind invalid-action difference of +1.01 [+0.23, +1.84] into aware minus swapped +0.70 [+0.07, +1.46] and swapped minus blind +0.30 [−0.31, +0.99]. The success decomposition is aware minus swapped −6.4 points [−12.0, −1.6] (p 0.042) and swapped minus blind +2.4 [−4.0, +8.8]. Read naively, the swapped verifier did better than the aware one despite obstructing (54.4% success against 47.8%), and swapped minus none is +10.4 points [+3.2, +18.4] (p 0.014). These are uncorrected contrasts at one seed among many reported, and they are exactly the comparisons the obstruction check says not to interpret as attribution. One mechanical point explains why obstruction need not cost success under this rule: a rejected working proposal usually ends in an abstention, which executes the proposal anyway, so the swapped verifier's obstruction shows up mostly as abstentions (9.08 per episode, like the aware verifier's 9.12) rather than as lost progress.

### 6.3 The argmax rule (Tier 2, yes/no readout, one seed; added 2026-11-24)

**What ran.** `optionc_qwen3-8b_yn_argmax` is the pre-registered secondary rule, Stage 1a's rule 2, kept as the continuity check with the gate. It differs from the accepted-only rule in one place. When the verifier rejects the proposal, every sampled alternative is scored and the one with the highest rounded score is executed *whatever its sign*. So the verifier can swap in an action it itself predicts to fail. Only when the samples contain no other action is the proposal executed. Readout (yes/no), verifier, agent, games and arms (task-blind, task-aware and swapped at one seed, matched-state scoring on the aware steps) are those of the yes/no secondary. The no-verifier reference episodes come from the primary run, so comparisons with no verifier are paired by game but not by common random numbers. It ran at `2a86683` (jobs 7253142–7253145, four shards on L40S GPUs, every shard "shard complete", 393 episodes), and its analysis is the rerun job 7295370.

| Yes/no readout, argmax rule (one seed per verifier arm) | No verifier (from the primary run) | Task-blind | Task-aware | Swapped (125 games) |
|---|---|---|---|---|
| Success | 45.5% [36.6, 53.7] | 53.7% [45.5, 62.7] | 53.0% [44.0, 61.9] | 54.4% [45.6, 63.2] |
| Invalid actions per episode | 4.80 [3.87, 5.79] | 3.12 [2.40, 3.93] | 3.04 [2.36, 3.82] | 2.90 [2.22, 3.66] |
| Invalid actions in the first 10 steps | 0.45 [0.30, 0.61] | 0.28 [0.19, 0.38] | 0.51 [0.38, 0.65] | 0.42 [0.29, 0.55] |
| Overrides per episode; harm; help; harm minus help | 0 | 4.51; 0.46; 1.46; −1.01 | 6.84; 0.72; 1.53; −0.81 | 6.98; 0.69; 1.48; −0.79 |
| Proposal executed after a rejection with no other action (per episode) | 0 | 3.60 | 4.59 | 5.51 |
| Proposals rejected (share of verified steps, from the reason counts) | | 38.1% | 53.1% | 58.3% |
| Working proposals accepted, per step | | 72.4% | 53.1% | 47.0% |
| First proposal rejected; first proposal overridden | | 10.4%; 6.7% | 92.5%; 47.8% | 93.6%; 44.8% |
| Episodes; steps | 134; 2,987 | 134; 2,854 | 134; 2,885 | 125; 2,678 |

The rejection shares are computed here from each arm's decision-reason counts. For the aware arm these are 1,354 proposals accepted, 916 replaced by the highest-scored alternative and 615 rejected with no other action, out of 2,885 steps.

**Verdicts.**

- **Success: no reliable difference.** Aware minus blind is −0.7 points [−9.0, +7.5] (p 1.0, 29 games differ).
- **Invalid actions per episode: no reliable difference**, −0.07 [−0.85, +0.71] (p 0.88).
- **Override harm minus help (key secondary): no reliable difference**, +0.19 per episode [−0.20, +0.60] (p 0.37).
- **Invalid actions in the first 10 steps (key secondary): harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)**, +0.23 per episode [+0.10, +0.37] (p 0.002, 43 games differ).

The obstruction check fired. The aware verifier accepted 63.5% [56.0, 70.9] of the agent's working real-goal proposals (271 steps, 95 games), the swapped verifier 38.9% [32.4, 45.7] (247 steps, 83 games), a gap of 24.6 points against the 10-point margin. The blind verifier accepted 54.0% [49.0, 59.4] (265 steps, 100 games). These are secondary tests of a one-seed secondary run, uncorrected.

**How the early harm arises (descriptive).** Under this rule the step-0 collapse has a direct cost. The aware verifier rejected 92.5% of first proposals, which always work, and replaced 47.8% of them with the highest-scored alternative, against 6.7% for the blind verifier. In the accepted-only run the same rejections ended in abstentions and no first move was overridden. Over the whole episode the aware verifier overrides 2.32 more times per episode than the blind one [+1.49, +3.13]. Its overrides harm a working proposal 0.26 times more often [+0.08, +0.45] (p 0.007), while rescuing a failing one about as often (+0.07 [−0.32, +0.47]). That shows up as more failing actions early in the episode (+0.23 in the first 10 steps) but not over the whole episode or in success.

**Against no verifier (descriptive).** Under this rule every verifier helps:

- the blind verifier raises success by 8.2 points [+0.7, +15.7] (p 0.037, 27 games differ) and removes 1.68 invalid actions per episode [−2.53, −0.84];
- the aware verifier raises success by 7.5 points [−1.5, +16.4] (p 0.13);
- the swapped verifier raises success by 10.4 points [+1.6, +19.2] (p 0.042).

These are uncorrected contrasts at one seed among many reported.

**Decomposition on the 125 swap games (descriptive).** The first-10-step difference of +0.24 [+0.10, +0.40] splits into aware minus swapped +0.12 [+0.03, +0.22] (p 0.025) and swapped minus blind +0.12 [−0.03, +0.27]. These are the comparisons the obstruction check says not to read as attribution.

**Matched states (descriptive).** On identical states of the aware episodes, the aware prompt accepted failing goal-object proposals more often than the blind prompt (15.0% against 12.0%, +3.0 points [+0.9, +6.0], p 0.064, 167 steps in 31 games) and than the swapped prompt (19.1% against 13.0%, +6.1 [+2.6, +11.5], p 0.007, 131 steps in 28 games). It rejected 17.0 points more working proposals than the blind prompt ([+14.8, +19.2], 2,368 steps, 134 games). It rejected fewer working goal-object proposals than either the blind prompt (−10.7 [−15.8, −6.0]) or the swapped prompt (−24.8 [−31.3, −18.5]). These are the same three patterns as in the accepted-only yes/no run (section 7). Of the 134 pairs of aware and blind episodes, 17 (12.7%) never diverge. The others first execute different actions at a median step index of 1, counting from 0, so typically at the first or second action. In the primary run, by contrast, the 65 diverging pairs first diverged at a median step of 23.

**Against the prediction.** The design predicted (prediction 6) that under this rule the aware-minus-blind success difference would be more negative than under the accepted-only rule, between −10 and −2 points (central −5). It also predicted that at step 0 the aware verifier would override at least 80% of proposals, mostly to another working `go to`, so that step-0 harm would be small and the cost would come later. Neither held:

- success differs by −0.7 points, less negative than the accepted-only yes/no run's −4.5;
- the aware verifier overrode 47.8% of first moves, not 80% or more;
- the measurable cost is early (the first 10 steps), not later.

Stage 1a's large harm under this rule came from its candidate menus, which were rich in failing goal-object actions. The agent's own samples contain few such actions. Of the aware verifier's 6.84 overrides per episode, 0.72 replaced a working proposal with a failing action, so most of its replacements, including those of wrongly rejected first moves, did not cost a failed step.

## 7. Matched-state analysis (descriptive)

At every step of the aware episodes, the same proposal was scored under the aware, blind and swapped prompts. The table gives, for each kind of step, how often each prompt would make the stated call, and the paired difference with its game interval. "Accept failing goal" is the RQ2 bias seen in the loop; "reject working" is the any-task rejection.

| Matched states, aware episodes | Observation: aware | Observation: other | Observation: difference (points) | Yes/no: aware | Yes/no: other | Yes/no: difference (points) |
|---|---|---|---|---|---|---|
| Accept failing goal-object proposal, against blind | 67.1% | 39.6% | +27.5 [+11.5, +43.8]; 298 steps, 22 games, p 0.011 | 14.2% | 10.9% | +3.3 [+0.9, +6.5]; 211 steps, 30 games, p 0.039 |
| Accept failing goal-object proposal, against swapped | 72.1% | 41.9% | +30.2 [+12.3, +48.3]; 265 steps, 20 games, p 0.012 | 16.6% | 9.4% | +7.2 [+4.0, +12.5]; 181 steps, 28 games, p 0.001 |
| Reject working proposal, against blind | 0.66% | 1.43% | −0.76 [−1.18, −0.37]; 7,219 steps, 134 games | 44.1% | 27.6% | +16.5 [+14.4, +18.5]; 2,334 steps, 134 games |
| Reject working proposal, against swapped | 0.67% | 1.16% | −0.49 [−0.86, −0.09]; 6,733 steps, 125 games | 44.9% | 49.8% | −4.9 [−6.6, −3.3]; 2,166 steps, 125 games |
| Reject working goal-object proposal, against blind | 3.2% | 8.3% | −5.1 [−8.5, −2.5]; 841 steps, 86 games | 39.5% | 49.6% | −10.1 [−16.1, −4.2]; 258 steps, 93 games |
| Reject working goal-object proposal, against swapped | 3.6% | 8.3% | −4.8 [−8.6, −1.9]; 756 steps, 78 games | 40.3% | 62.7% | −22.5 [−29.3, −16.3]; 236 steps, 86 games |

Steps are counted across seeds (the observation run has three aware seeds), so the same state can appear more than once; the intervals resample games. Goal-object alternatives (whose outcomes are unknown, since only the chosen action is executed) were accepted at nearly the same rates under each prompt; the metrics give the paired differences only (observation: +3.2 points against blind, 31 alternatives; yes/no: −2.8 against blind and +5.3 against swapped).

What this shows, at the level of single decisions and with the loop's confounds removed:

- **The naming bias reaches the verifier's decisions under both readouts.** On identical states the aware prompt accepts the agent's failing goal-object proposals more often than either the blind or the swapped prompt. Under the observation readout the shift is large (about 28 to 30 points), under yes/no small (3 to 7 points), because yes/no rejects most such proposals under every prompt. These steps are rare: they occur in 20 to 30 of 134 games.
- **The any-task rejection is a yes/no phenomenon.** Shown the real task, the yes/no prompt rejects 16.5 points more of the agent's working proposals than the blind prompt, at every step of the episode, not just at step 0; the swapped prompt rejects slightly more still. Under the observation readout the task does the opposite, rejecting slightly fewer working proposals.
- **Naming the real object protects the real task's progress steps.** Both readouts reject working goal-object proposals less often under the real task than task-blind or swapped. A stated task makes goal-object actions look more likely to succeed whether they work or fail.

The design's prediction 5 expected the aware prompt to accept failing goal-object proposals more often than the swapped one (yes, under both readouts) and both concrete-task prompts to reject working proposals more often than the blind one (yes under yes/no; under the observation readout the aware prompt rejects fewer, against the prediction, though at rates near 1%).

## 8. Drift check against the archived RQ3 episodes

The no-verifier arm is the RQ3 no-foresight agent, replayed with a different GPU, vLLM build and checkout. Compared with the archived RQ3 episodes (reported, not gated): the first action agrees in 98.5% of the 134 games; 59.7% of games are reproduced action for action to the end (a reproduction rate, not a test, since one flipped near-tie token changes the rest of an episode); the 54 games that depart do so at quartile steps 4, 11 and 18. Success is 45.5% now against 47.0% archived, a paired difference of −1.5 points [−6.0, +2.2] (p 0.75, 8 games differ). The agent behaves as it did in RQ3. Comparisons between Option C's arms are unaffected by drift in any case, since every arm is played by the same served agent.

## 9. Predictions against outcomes

The design logged eight predictions before any data. They were written when yes/no was the primary readout, so each row says which run it is checked against.

| Prediction (logged before data) | Outcome | Held? |
|---|---|---|
| 1. Step-0 check: yes/no collapse (aware under 20%, blind at least 80% of working proposals accepted); observation readout no collapse (aware at least 80%), so "question-specific" (about 60% confidence); naming lift above 2 nats under both readouts; Llama no collapse under either | Qwen yes/no 7.5% against 89.6%; observation 100% aware; "question-specific"; naming lift 7.60 (yes/no) and 7.04 (observation) nats; Llama yes/no gap −10.4 points, "mixed", observation readout not competent | Yes for Qwen; partly for Llama (no collapse, but the yes/no interval does not show "no gap") |
| 2. Step 0 in the loop (yes/no): aware rejects at least 80% of first proposals but overrides at most 15%; blind overrides at most 12% | Aware rejects 92.5%, overrides 0%; blind overrides 6.7% | Yes |
| 3. Verifier value: blind raises success over none by 0 to 8 points (central +3) and lowers invalid actions (central −1) | Observation: +1.7 points, −1.08 invalid; yes/no: +6.7 points, −1.73 invalid; neither success gain reliable | Yes on point estimates (success gains not distinguishable from zero) |
| 4. Primary: aware minus blind between −8 and +2 points (central −2), most likely "no reliable difference"; invalid actions between 0 and +1 (central +0.3) | Observation (primary after reallocation): −1.2 points, "no reliable difference", invalid +0.08. Yes/no: −4.5 points, "no reliable difference", invalid +1.12 ("harm") | Yes for the primary run; for yes/no, success yes, invalid actions slightly above the predicted range |
| 5. Attribution: the obstruction check fires, so success-level attribution "not identifiable"; at matched states aware accepts more failing goal-object proposals than swapped, and both concrete-task prompts reject more working proposals than blind | Fires under yes/no (19.0-point gap), clear under observation (7.3); matched states as predicted under yes/no; under observation the naming shift holds but the aware prompt rejects fewer working proposals | Yes for yes/no; partly for observation |
| 6. Argmax rule more negative (−10 to −2 points, central −5); at step 0 the aware verifier overrides at least 80% of proposals, with small step-0 harm and the cost later | −0.7 points [−9.0, +7.5], "no reliable difference"; first proposals overridden 47.8%; the measurable cost is early (invalid actions in the first 10 steps +0.23, "harm, attribution not identifiable") (section 6.3) | No |
| 7. Llama-3.1-8B in the loop: within 3 points, "no reliable difference" | Not run (Tier 3; the user's decision of 2026-11-24, section 13) | Not tested |
| 8. Observation readout: smaller aware-minus-blind difference than yes/no in success and invalid actions | −1.2 against −4.5 points; +0.08 against +1.12 invalid actions (separate runs and seed counts, so a description, not a test) | Yes, descriptively |

## 10. Interpretation

### 10.1 What the null in the loop shows, and what it does not

The primary question has a clear answer for this configuration: with the observation readout and the accepted-only rule, showing Qwen3-8B the agent's task did not change this agent's success. The reason is as informative as the verdict. Under the observation readout the task does not trigger the step-0 collapse, and at later steps the aware and blind verifiers reject almost the same few proposals, so in 84% of seed pairs the agent plays the identical episode. Where the task does change a decision, it changes it in both directions: it makes the verifier accept more of the agent's failing goal-object proposals (the RQ2 bias, clearly present at the decision level, +27.5 points on identical states) and reject fewer of its working goal-object proposals. Failing goal-object proposals are rare in this agent's play (in about 20 of 134 games), so the bias has little to act on.

The design's precision section matters for how much the null excludes. It planned for a 95% half-width of about 3.0 to 4.4 points with three seeds and power of about 0.8 only for losses of 6 to 8 points, and it warned that its power model treats seeds as independent, which overstates their value when the verifier rarely rejects. That is what happened: the realised interval, −3.0 to 0.0 points, is narrower than planned, but because the arms differ in only 3 games, not because the experiment measured a large number of independent differences, and with three non-zero games the test could not have been significant. The fair summary is that this verifier changed the outcome of 3 of 134 games, all in the aware arm's disfavour; a cost of a few points is not excluded by any formal test, but there is no sign of a large one.

The null does not show that the RQ2 bias is harmless. It is visible in the loop's own decisions under both readouts. It shows that for a greedy agent that rarely proposes the failures RQ2 measures, screened one action at a time by a rule that never swaps in an action the verifier rejects, the bias has almost no decisions to change. Planners that rank many candidates by predicted success, that roll out imagined trajectories, or that fall back to the best-scored action whatever its score would give it more to act on, as Stage 1a's menus did.

### 10.2 How this relates to Stages 1a and 1b

The three stages now agree. Stage 1a's large harm (+19.5 points per state) needed the step-0 collapse together with a fallback that switches to the highest-scored action from a menu rich in failing goal-object actions. Stage 1b removed the fallback route by using the agent's own candidates and the accepted-only rule, and found no reliable harm for the susceptible verifiers; its Qwen contrast showed a smaller harm through abstention instead. Option C removes the same route in a live loop, and:

- under the observation readout, which also removes the step-0 collapse, nothing reaches success or invalid actions;
- under yes/no, the collapse is present (92.5% of first moves rejected) but costs nothing at step 0, because the rule abstains, while the any-task rejection at later steps (16.5 points more working proposals rejected) reproduces Stage 1b's abstention route and compounds to about one more invalid action per episode against a task-blind verifier, without a reliable success difference at one seed.

The practical reading is the one Stage 1b reached for Qwen: showing a yes/no verifier the task has a real cost, because the task makes the verifier say No to good actions and give up rescuing bad ones. That cost comes from the yes/no question's task reading, which the step-0 check classifies as question-specific, and the loop could not attribute it to naming the object. It is therefore evidence about how a yes/no verifier should be prompted, not evidence that the object-naming bias of RQ2 reaches task success.

### 10.3 The format caveat

Both readouts were scored through Qwen3's thinking-off template, which places an empty `<think></think>` block before the verifier's answer, as Stage 1a did. The Qwen pipeline audit ([qwen_effect_audit_pipeline.md](qwen_effect_audit_pipeline.md), section 7) found that this block about doubles Qwen's effects. At Stage 1a's step-0 states the real-task yes/no verifier accepts 7.5% of first moves with the block and 26.1% in plain text (97.8% task-blind), and its step-0 harm under the argmax fallback is 71.6 points with the block against 38.8 in plain text. On the RQ2 items the first-token effect, the quantity closest to the observation readout, is 2.52 nats with the block and 1.07 without.

For Option C this means three things. ~~The step-0 classification would very probably be unchanged in plain text: a 26.1% against 97.8% acceptance gap is far below the −30-point collapse threshold, although the step-0 check's interval rule was not computed in that format.~~ *(Replaced 2026-11-24 by the computed result below: the classification is unchanged in plain text.)* The yes/no step-0 collapse costs nothing under the accepted-only rule in either format, because the aware verifier abstains at step 0 rather than overriding. *(Under the argmax rule it does cost something; section 6.3.)* But the later-step any-task rejection that drives the yes/no secondary's invalid-action harm was never rescored without the block, so its size in plain text is unknown; and the observation readout's naming shift on failing goal-object proposals is probably smaller in plain text. Neither caveat changes a verdict; both limit how far the sizes carry beyond this prompt format.

**The step-0 classification in plain text (computed 2026-11-24; exploratory).**

*What was done.* The specification was committed and pushed before the score, in [optionc_followups_2026-11-24.md](optionc_followups_2026-11-24.md), section 2. Every one of the step-0 check's 10,378 Qwen3-8B scores was recomputed without the chat template, and so without the empty think block. That covers the 134 step-0 states, every candidate, the aware, blind, swapped and neutral prompts, and both readouts. The job was 7300307, in the fresh checkout `text-wm-fix50` at `0b7518e`, on an L40S, the GPU type of the logged with-block scores; it took about 20 minutes.

The prompts are Option C's own, and their hashes equal the logged run's on all 10,378. They are rendered in the Qwen pipeline audit's plain-text format: the system text, then each example with its answer after one space, then the query. The yes/no readout is read exactly as the audit read it, comparing " Yes" with " No" after one space. The observation readout is Option C's, read at the token where the reply " Nothing happens." would begin, which is " Nothing" in every prompt.

The frozen analysis code (`wf.optionc.step0_report`) was then applied unchanged. Two checks hold:

- run on the logged with-block scores, it reproduces the frozen classification, gaps and competence results exactly;
- the job's with-block rescoring of the first 3 states' 192 jobs equals the logged scores exactly (largest raw difference 0.0).

In plain text the two answers stay on the model's usual path. " Yes" and " No" hold 99.9% of the next-token probability (median; minimum 99.6%). " Nothing" and " You" hold a median of 96.7% for the observation readout (below 50% in 0.6% of prompts).

*Result: the classification is unchanged, "question-specific (supports the task-success reading)".*

| Step 0, Qwen3-8B, 134 states | Yes/no, with the block | Yes/no, plain text | Observation, with the block | Observation, plain text |
|---|---|---|---|---|
| Working first moves accepted: aware; blind | 7.5%; 89.6% | 27.6% [20.1, 35.8]; 99.3% [97.8, 100] | 100%; 100% | 100%; 100% |
| Working first moves accepted: swapped; neutral | 5.6%; 89.6% | 24.8%; 100% | 100%; 100% | 100%; 100% |
| Paired gap, aware − blind (points) | −82.1 [−88.1, −74.6]; 110 games non-zero | −71.6 [−79.1, −64.2]; 96 games non-zero | 0.0 [0.0, 0.0]; none non-zero | 0.0 [0.0, 0.0]; none non-zero |
| Gap state | collapse | collapse | no gap | no gap |
| Blind arm: AUC; failures accepted; successes accepted | 1.00; 0.0%; 97.2% | 1.00; 1.5%; 99.8% | 1.00; 18.8%; 100% | 1.00; 62.7%; 100% |
| Blind arm competent | yes | yes | yes | yes |
| Failing candidates accepted: aware; blind; swapped; neutral | 0.8%; 0.0%; 0.0%; 0.0% | 2.0%; 1.5%; 0.3%; 1.1% | 77.2%; 18.8%; 28.8%; 8.1% | 86.6%; 62.7%; 49.2%; 30.5% |
| Naming lift, aware − swapped on failing goal-object candidates (nats) | +7.60 [7.10, 8.09] | +3.43 [3.20, 3.65] | +7.04 [6.45, 7.62] | +1.67 [1.52, 1.82] |
| Any-task drop, aware − blind on working `go to` candidates (nats) | −39.1 [−39.8, −38.4] | −6.93 [−7.12, −6.75] | −8.49 [−8.92, −8.09] | −1.47 [−1.57, −1.36] |

*What it shows.*

- **The yes/no collapse survives in plain text.** It is smaller: the real-task verifier accepts 27.6% of the agent's working first moves against 7.5% with the block. The gap's interval still lies entirely below the −30-point threshold. The 27.6% agrees with the audit's 26.1% for Stage 1a's identical real-task prompt, measured there on an A40.
- **The observation readout still shows no gap in any game, and its blind arm is still competent.** So the step-0 check would have classified Qwen3-8B the same way in plain text, and the pre-declared reallocation would have been licensed in either format. The expectation stated before the score (about 70% confidence) held.
- **Plain text shrinks every score by the factors the audit reported.** The naming lift falls to 45% of its with-block size on yes/no (3.43 against 7.60 nats) and to 24% on the observation readout (1.67 against 7.04). The any-task drop falls to 18% and 17%. In acceptance terms the observation readout becomes a weaker screen in plain text: task-blind, it accepts 62.7% of the failing step-0 candidates instead of 18.8%, though its AUC stays 1.00. The real task still raises that acceptance, to 86.6% against 49.2% under the swapped task.

*What this does not cover.* The loop's later steps were not rescored, so whether the yes/no secondary's later-step any-task rejection (and with it its invalid-action harm) survives in plain text remains unknown, as stated above.

### 10.4 What would change the conclusion

A different agent would: one whose candidate lists put determined goal-object failures above its working actions more often, as Stage 1a's menus did. So might a verifier that ranks rather than screens. The argmax rule, which can execute an action the verifier rejects, was listed here as a candidate. It has now run with the yes/no readout (section 6.3), and it did not change success either: with this agent's own samples it adds an early cost of about a quarter of an invalid action per episode. On the yes/no side, an anchored or plain-text format that weakens the any-task rejection might remove the invalid-action harm. More games would tighten the success intervals (the 140 solvable `valid_seen` games would roughly halve the variance, per the design), but more seeds would add little under the observation readout, since most seed pairs never diverge.

## 11. What was not run, and what the metrics do not carry

- **The argmax rule (Tier 2, `optionc_qwen3-8b_yn_argmax`)** was part of the committed plan at one seed, as the continuity check with Stage 1a's rule 2. The plan runs it after Tier 1, so it had not been submitted when this note was first written. *(2026-11-24: it has now run and been analysed; section 6.3.)*
- **Llama-3.1-8B in the loop (Tier 3, optional): not run. Decided 2026-11-24 by the user**, because the step-0 check's competence result made its inferential value doubtful and the remaining allocation (about 58 GPU-hours of the committed budget) was too large without a convincing justification. Section 13 gives the assessment behind the decision. It agrees with the outcome and corrects two facts behind the reasons: the readout that failed competence is the observation readout, which a Llama loop would not use, and the loop itself would cost about 19 GPU-hours by the design.
- **Stage 1a score agreement at step 0** was promised by the design but skipped by the analysis because Stage 1a's scoring runs are not in this checkout (section 3). *(2026-11-24: computed in the rerun, and the disagreement explained; section 3.1.)*
- **How often the swap object is later named by an executed action** was to be reported according to the design; it is not in the metrics.

## 12. Provenance and cost

All jobs ran on ICE (`ssh ice`) in the checkout `/storage/ice1/2/1/gmat3/text-wm-fix40`, every shard reporting "shard complete"; both analyses record a clean checkout (`git_dirty: false`).

| Run | Commit | ICE jobs | GPU | GPU-hours (from accounting) |
|---|---|---|---|---|
| `optionc_smoke` (two training games, every arm, both rules) | `54cb71e` | 7245677 | L40S | 0.27 |
| `optionc_step0_qwen3-8b`, `optionc_step0_llama3.1-8b` (10,378 scores each) | `54cb71e` | 7245678, 7245679 | L40S | 0.46 |
| `optionc_smoke_analysis`, `optionc_step0_analysis` | `54cb71e` | 7246046 | CPU | |
| `optionc_qwen3-8b_obs` (primary; 4 shards, 1,063 episodes, 23,657 steps) | `2a86683` | 7247952–7247955 | L40S | 4.42 |
| `optionc_qwen3-8b_yn` (secondary; 4 shards, 393 episodes, 8,460 steps) | `2a86683` | 7247956–7247959, resumed 7251605 and 7251606 | L40S; the two resumed shards on A40 | 7.18 |
| `optionc_analysis` | `2a86683` | 7252705 | CPU | |
| `optionc_qwen3-8b_yn_argmax` (Tier 2; 4 shards, 393 episodes, 8,417 steps) | `2a86683` | 7253142–7253145 | L40S | 7.89 |
| `optionc_step0_analysis`, `optionc_analysis` (reruns, 2026-11-24: Stage 1a's runs linked, argmax run added) | `01cad81` | 7295369, 7295370 | CPU | |
| Follow-ups of 2026-11-24 (section 3.1 diagnostic; section 10.3 plain text), checkout `text-wm-fix50` | `0b7518e` | 7300307 (plain text), 7300308 (Qwen rescoring), 7300309 (Llama rescoring) | L40S; A100-PCIE-40GB; A40 | 0.34; 0.14; 0.19 |

**The two resumed shards.** Yes/no shards 0 and 2 (jobs 7247956 and 7247958, both on node `atl1-1-03-004-21-0`) stopped after 52 and 20 minutes with a `ConnectionError` while sending an action to the job's local ALFWorld game server ("Connection reset by peer" and "Remote end closed connection without response"). They were resumed from their saved episodes and cache as jobs 7251605 (shard 0) and 7251606 (shard 2), at the same commit, and completed. Resumption skips saved episodes and reuses every cached agent reply and verifier score, so the common-random-numbers pairing within a game is kept; scores computed after the resume came from an A40 instead of an L40S, which can move a near-zero score across a 1/64-nat rounding step. The episode records do not mark which GPU played them. The resumption is already recorded in [deviations.md](deviations.md) (2026-11-23).

**Gates.** The smoke gate passed (no format errors, the cache check replayed the first game's 7 episodes identically with every model call disabled, 1.26 seconds per step). Every census shard passed its run-order gate with no override (smoke passed, step-0 analysis reported, cost rule held), its tokenization check (Yes 9454 and No 2753; Nothing 23780 and You 2610 single tokens), and the last-position-logits check (8 prompts, largest raw difference 0.0, so the faster path was adopted); every observation shard passed the observation-readout gate. Format errors were 0% of proposals in every shard.

**Cost.** The committed budget was about 70 to 80 GPU-hours for Tiers 1 and 2; after the reallocation the configs carried upper bounds of 58 GPU-hours (observation) and 28 (yes/no), and the smoke-based cost rule projected 10.3 and 3.4. The study used 12.3 GPU-hours in all (0.73 before the census, 4.42 for the primary run, 7.18 for the secondary, including the two attempts that stopped). Per step, the observation run took 0.32 to 0.50 seconds and the yes/no run 2.2 to 2.7 seconds, against the design's assumption of 4.8 to 5.3; the yes/no run cost more than its projection because its verifier rejects about half of all proposals and each rejection triggers a 20-sample request, and more than the three-seed observation run because there later seeds reused almost every cached step.

**The argmax run (added 2026-11-24).** Every shard passed its run-order gate with no override: the cost rule projected 3.42 GPU-hours against the configuration's upper bound of 30. Every shard also passed its tokenization check, adopted the last-position logits (8 prompts, largest raw difference 0.0) and had 0% format errors. It took 2.5 to 3.2 seconds per step and 7.89 GPU-hours. With it, Option C has used about 20.2 GPU-hours of the committed 70 to 80, before the follow-ups of sections 3.1 and 10.3 (0.67 GPU-hours in all).

## 13. Tier 3, Llama-3.1-8B as the verifier in the loop: assessment and decision (2026-11-24)

**Decision (the user, 2026-11-24): not run.** The user's stated reasons were that the step-0 check's failed competence result makes its inferential value doubtful, and that the allocation is too large without a convincing justification. This assessment was written to inform that decision and agrees with the outcome. It corrects two factual points behind the reasons, so the record is accurate; neither point changes the recommendation.

### What Tier 3 is

Tier 3 is the run `optionc_llama3.1-8b_yn` in the frozen design. The agent stays the same RQ3 Qwen3-8B ReAct agent. Only the verifier changes: Llama-3.1-8B-Instruct, read through the **yes/no** readout, under the accepted-only rule. It has the task-blind, task-aware and swapped arms at one seed each, matched-state scoring on the aware steps, and reuses the primary run's no-verifier episodes. The primary contrast would be the same as in the main study: success of the task-aware verifier minus the task-blind one, paired by game over the 134 games.

### Two factual corrections to the stated reasons

**The failed competence check concerns a readout Tier 3 would not use.** At the step-0 check, Llama's *observation* readout failed the competence rule: its task-blind arm accepted every failing step-0 candidate (100%), although its AUC was 0.995. Its *yes/no* readout, the one Tier 3 uses, passed: under the task-blind prompt it had AUC 0.98, accepted 5.0% of failing candidates and 97.8% of working ones. Llama's classification is "mixed (the yes/no interval does not settle it)". That label is not a competence failure. It means the yes/no gap between the aware and blind arms (−10.4 points [−15.7, −5.2]) is neither a collapse (an interval entirely below −30) nor clearly no gap (an interval entirely within ±10). So a Llama loop would be interpretable as a yes/no-verifier experiment. The non-competent observation readout only rules out a Llama observation-readout run, which was never planned, and it means Llama gets no "question-specific" reading of its yes/no result.

**The cost is about 19 GPU-hours by the design, not 58.** The 58 GPU-hours figure is the budget left in the committed 70 to 80 GPU-hours. Option C has used about 20.2 GPU-hours: 12.3 for the smoke, the step-0 check and Tier 1, and 7.9 for the argmax run, from accounting. The design's estimate for Tier 3 is 19 GPU-hours (range 13 to 28), and the run's configuration carries an upper bound of 28. The closest measured analogues are much cheaper. The Qwen yes/no secondary, with the same arms, seed count and matched-state scoring, took 7.2 GPU-hours. The argmax run took 7.9. Llama's yes/no verifier rejects fewer proposals than Qwen's, at step 0 accepting 89.6% of working first moves under the task against Qwen's 7.5%, so it would trigger fewer 20-sample requests. My estimate is therefore about 5 to 8 GPU-hours. That is an extrapolation from the Qwen runs, not a measurement.

### What a Llama loop would test that the Qwen runs did not

The central question is whether task-induced distortion of a world model's predictions changes what an agent achieves. The Qwen runs answered it in two configurations. Each has a limit as a test of the naming bias:

- **Qwen's yes/no verifier.** Its task effect is dominated by the any-task rejection: told any concrete task, it says No to good actions. The loop could not attribute its invalid-action harm to naming the object, because the obstruction check fired.
- **Qwen's observation readout, the primary run.** It carries a large naming bias with no collapse. On identical states it accepted 27.5 points more of the agent's failing goal-object proposals when shown the real task than when task-blind. Yet aware and blind episodes differed in success in only 3 of 134 games.

Llama's yes/no verifier is a second verifier of the observation readout's kind: a strong naming bias with little any-task rejection. At step 0 the task-aware Llama accepted 52.4% of the failing step-0 candidates against 5.0% task-blind. Its naming lift, the mean aware-minus-swapped score on failing goal-object candidates, was +3.34 nats [3.00, 3.68]. It still accepted 89.6% of working first moves. So Tier 3 would be a cross-model replication of the primary run's situation, with a different verifier family and a different readout. It would show whether "a naming-biased verifier without the collapse leaves success unchanged" holds beyond Qwen.

### Why its inferential value is low anyway

The reason the primary run found nothing is on the agent's side, not the verifier's. This greedy agent rarely proposes the failures that RQ2 measures. Failing goal-object proposals appeared in about 20 to 30 of the 134 games: 298 of 8,937 task-aware steps, in 22 games, in the primary run. A verifier can change outcomes only through those steps, and through rejections of working steps, which Llama rarely makes. That constraint does not depend on which model verifies. Stage 1b already measured Llama-3.1-8B as a verifier on this agent's own candidates and found "no reliable difference" (+0.17 points [−0.92, +1.27] per decision), with harm on encounter states concentrated in one looping game.

At one seed, the design's precision table gives a 95% half-width of about 5.2 to 7.5 points of success and power of about 0.5 only for a 10-point loss. The likely result is "no reliable difference" with an interval of roughly ±6 points. The design's own prediction 7 was "within 3 points, no reliable difference". That would add a second, wider null beside a primary null that is already tighter (−1.2 points [−3.0, 0.0]).

### What each outcome would imply

- **No reliable difference** (most likely; I would put it at 75 to 85%): consistent with the Qwen observation-readout result and with Stage 1b. It would add little, because the interval would be about twice as wide as the primary run's and the explanation (few opportunities) is already known.
- **Harm, attributable or unresolved:** the naming bias reaching success through a second verifier family. This would be the informative outcome, but at one seed only a large loss would register, which neither Stage 1b nor the primary run makes plausible.
- **Harm, attribution not identifiable:** likely if any harm appears, since the swapped yes/no prompt obstructed the real task's progress steps for Qwen, and Llama's swapped arm also rejects working moves (it accepted 84.8% of working first moves). It would repeat the Qwen yes/no secondary's ambiguity.
- **Help:** possible if a task-aware Llama rescues more of the agent's failing repeats. It would be a fact about verifier value, not about the bias.

### Recommendation

**Do not run Tier 3**, in agreement with the user's decision. The reason is not readout competence, since the yes/no readout Tier 3 uses passed. The reasons are that the expected result is a one-seed null that cannot exclude effects of the size anyone predicts, and that the limiting factor is the agent, which a different verifier does not change.

If a cross-model check is wanted later, a cheaper and more informative option is **decision-level only**. Score the logged aware-episode states of the primary run (the agent's proposals and the alternatives the verifier scored) with Llama's aware, blind and swapped yes/no prompts, never acting on them. This repeats the matched-state analysis for a second verifier. By the step-0 check's throughput (about 10,000 scores in 14 minutes on an L40S), and allowing for the longer histories of later steps, it would take about 1 to 3 GPU-hours. It answers whether Llama's verdicts on this agent's real decisions move with the task, which is the step before outcomes. The budget that would change the success-level answer is on the agent's side instead: more games (the 140 solvable `valid_seen` games would roughly halve the variance), or a planner that ranks many candidates or falls back to the best-scored action, which is where Stage 1a's harm came from.
