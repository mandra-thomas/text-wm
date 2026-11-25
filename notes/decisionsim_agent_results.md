# Decision simulation, Stage 1b (the agent's own candidates): results (2026-11-22)

Exploratory stage. Design frozen before any census label was read at tag `decisionsim-stage1b-frozen-2026-11-22` (commit `2a05707`, branch `decisionsim-agent`; [decisionsim_agent_design.md](decisionsim_agent_design.md)). Sampling (ICE jobs 7240587–7240590, four shards) and labelling (jobs 7240591–7240598, eight shards) ran at `20cc63d`, whose sampling and labelling code the frozen commit leaves unchanged. The model-free analysis `decisionsim_agent_encounter` (job 7244860), the scoring runs (Llama-3.1-8B jobs 7244869–7244870, Mistral-7B-v0.3 jobs 7244871–7244872, Qwen3-8B jobs 7244873 and 7244875; two shards each, L40S) and the analysis `decisionsim_agent_analysis` (job 7245683) ran at `2a05707` in the checkout `text-wm-fix37`. Every scoring shard passed the run-order check against the model-free analysis with no override, and both Mistral shards passed the Yes/No tokenization gate. Numbers are from [decisionsim_agent_metrics.json](decisionsim_agent_metrics.json), a byte-identical copy of the analysis output (SHA-256 `df9c6c33…`). The exploratory follow-up is [decisionsim_agent_dx.md](decisionsim_agent_dx.md).

## Question

The main finding of the project (RQ2, replicated in the yes/no format) is that a prompted world model's predicted success for an action whose failure the history already settles rises when the agent's task names the action's object. Stage 1a showed that this can change decisions when a verifier screens fixed candidate menus written by our generators. Stage 1b asks the question that practical-impact claims rest on: when a real agent proposes its own candidates, does telling the verifier the task make the agent act on a failing action more often than not telling it?

At each of the 2,973 decision points of the 134 RQ3 episodes in which the Qwen3-8B ReAct agent saw no predictions, the agent's own ranked list was built: rank 1 is the action the agent actually took (its greedy choice), and the remaining ranks are the distinct actions among 20 sampled replies at Qwen's recommended non-thinking settings, ordered by how often they were sampled (at most eight candidates). Every candidate was executed once on a fresh replay of its state; it fails when the environment replies "Nothing happens.". A verifier, one of three language models asked the yes/no feasibility question, scores every candidate. Its score is the log-odds of the answer tokens,

$$
L = \log P(\text{Yes}) - \log P(\text{No}),
$$

rounded to 1/64 nat, and it accepts a candidate when $L > 0$. The selection rule is: act on the highest-ranked candidate the verifier accepts; if it accepts none, act on rank 1 (the verifier "abstains" and the agent behaves as if there were no verifier). The verifier sees the agent's real task line (task-aware), the same prompt with that line deleted (task-blind), the task with its object replaced by another object in the room (swapped), or "explore the room" (neutral; priority models only).

The primary estimand is the difference in the share of decisions whose chosen action fails,

$$
\Delta_{\text{invalid}} = \mathbb{E}_s\left[\mathbf{1}\{a^{\text{aware}}_s \text{ fails}\} - \mathbf{1}\{a^{\text{blind}}_s \text{ fails}\}\right],
$$

over all decision points $s$. The key secondary estimand $\Delta_{\text{invalid}}^{E}$ is the same difference restricted to the encounter states $E$: states whose list holds at least one regime candidate, meaning a failing action on the task's object whose failure the history determines, not at a shared spot (the kind of action RQ2 scores). The encounter label is model-free and was fixed before any score existed. Llama-3.1-8B and Mistral-7B-v0.3 are the priority verifiers, chosen because they are susceptible to the bias (under the aligned task they turn 16.5% and 37.5% of failing items from No to Yes, against 0.2% for Qwen3-8B); each estimand has its own Holm correction over these two. Qwen3-8B is a contrast and is in no Holm family. Every interval is a 95% percentile bootstrap over games (2,000 resamples) and every p a whole-game sign-flip permutation (1,000 flips).

## The frozen verdicts

These are the analysis code's own outputs, unchanged. Both priority models passed the competence rule (task-blind AUC at least 0.70, accepting at most 95% of failures and at least 5% of successes), so both are interpreted.

| Estimand | Llama-3.1-8B | Mistral-7B-v0.3 |
|---|---|---|
| Overall (2,973 states, 134 games): aware − blind | +0.17 points [−0.92, +1.27] | +0.17 points [−0.09, +0.47] |
| raw p, Holm p | 0.83, 0.83 | 0.39, 0.78 |
| games with a non-zero net difference | 41 | 8 |
| harm states / help states | 48 / 43 | 8 / 3 |
| share of harm from regime candidates | 15% (7 of 48) | 12% (1 of 8) |
| **Verdict** | **no reliable difference** | **no reliable difference** |
| Conditional on encounter (129 states, 22 games): aware − blind | +4.65 points [−1.16, +12.50] | 0.00 points [−2.46, +2.05] |
| raw p, Holm p | 0.48, 0.96 | 1.00, 1.00 |
| games with a non-zero net difference (smallest p a sign flip could give) | 4 (0.125) | 2 (0.5) |
| harm states / help states | 7 / 1 | 1 / 1 |
| share of harm from regime candidates | 100% | 100% |
| **Verdict** | **no reliable difference** | **no reliable difference** |

"Harm" is a state where the aware verifier's choice fails and the blind verifier's works; "help" is the reverse; harm minus help, divided by the number of states, is the difference exactly (the code checks this identity in every contrast).

**Decomposition (states with a swap object).** The design splits the aware-minus-blind difference into what naming the real task's object adds over a task of the same form about another object, and what showing any task of that form does,

$$
F_{\text{aware}} - F_{\text{blind}} = (F_{\text{aware}} - F_{\text{swapped}}) + (F_{\text{swapped}} - F_{\text{blind}}),
$$

where $F_v$ is the share of chosen actions that fail under verifier $v$. It is computed on the 2,616 states (125 games) that have a swap object, and on the 91 encounter states (18 games) that have one.

| Part | Llama, all swap states | Mistral, all swap states | Llama, encounter swap states | Mistral, encounter swap states |
|---|---|---|---|---|
| aware − blind | +0.31 [−0.92, +1.52], p 0.72 | +0.19 [−0.08, +0.55], p 0.40 | +5.5 [−3.0, +16.3], p 0.77 | 0.0 [−3.6, +2.7], p 1.0 |
| aware − swapped | +0.27 [−0.34, +0.97], p 0.51 | −0.11 [−0.32, +0.07], p 0.49 | +9.9 [0.0, +20.7], p 0.51 | 0.0 [−3.6, +2.7], p 1.0 |
| swapped − blind | +0.04 [−1.07, +1.17], p 1.0 | +0.31 [+0.04, +0.63], p 0.06 | −4.4 [−11.3, 0.0], p 0.51 | 0.0 |

Points; the identity holds in every column. No verdict is "toward failure", so attribution decides nothing here. The Llama encounter-state parts rest on 2 to 3 games with any net change.

## The pre-declared readings

**"Inert under the task" or "the bias rarely reaches decisions".** The design (amendment after review 3) fixed that a null from a model whose aware arm abstains at most multi-candidate states, or rejects most working rank-1 actions, is reported as "inert under the task", because such a verifier hands the decision back to the agent and cannot show the bias whatever its scores. Neither priority model meets that description:

| At the 1,684 states with more than one candidate | Llama aware | Llama blind | Mistral aware | Mistral blind | Qwen aware | Qwen blind |
|---|---|---|---|---|---|---|
| Abstains (accepts nothing), all steps | 2.6% | 1.6% | 0.2% | 0.0% | 29.4% | 14.5% |
| Abstains at step 0 (65 states) | 1.5% | 0.0% | 6.2% | 0.0% | 98.5% | 0.0% |
| Abstains at later steps (1,619 states) | 2.6% | 1.7% | 0.0% | 0.0% | 26.6% | 15.1% |
| Chooses rank 1 | 82.4% | 84.5% | 97.6% | 98.2% | 71.9% | 69.6% |
| Accepts the agent's working rank-1 action | 87–89% | 90–100% | 80–100% | 100% | 4–58% | 73–90% |

The last row gives the range over step 0 and later steps (the step-specific values are in the acceptance section below). Llama's and Mistral's aware arms abstain at under 3% of states and accept most working rank-1 actions, so by the pre-declared rule their nulls read as **the bias rarely reaches this agent's decisions**, not as "inert under the task".

One qualification follows from the same table, and it is descriptive rather than a reading the design fixed. Mistral's verifier is close to inert in the opposite direction: it accepts 92% of failing candidates even task-blind, so whatever the prompt it almost always accepts rank 1 (98%), and its aware and blind choices differ at only 0.9% of states. Its null therefore says little about the bias; it says that a verifier this permissive has almost no decision for the bias to act on.

**"Not evidence for the main finding".** This suffix is appended only to a "toward failure" verdict in which fewer than half of the harm states come from regime candidates. No verdict is "toward failure", so no suffix appears. For the record, the regime share of harm is 15% (Llama) and 12% (Mistral) overall, and 100% for both on the encounter states, where harm is 7 and 1 states.

**Regime harm.** The states where the aware verifier's choice is a regime candidate and the blind verifier's choice works number 7 for Llama (in 3 games; 5 of the 7 are in one game, `valid_unseen/150`, where the agent repeats `move apple 2 to garbagecan 1`), 1 for Mistral and 15 for Qwen (in 10 games). The futility rule did not apply: the encounter states that also list a working candidate fall in 19 games, above the 7 needed for a sign-flip p below 0.025. Once the scores were in, though, the encounter states with any net aware-minus-blind change fell in only 4 games for Llama and 2 for Mistral, so neither conditional test could have reached significance (smallest attainable p 0.125 and 0.5).

## What ran, and what the agent proposes (model-free)

**Bookkeeping.** All 2,973 states were usable: no replay mismatch, no history mismatch and no server error on any of the 10,193 distinct candidates. No sampled reply lacked an action line in either list; one primary reply and two temperature-1 replies hit the 256-token cap. A fresh greedy completion agreed with the archived action at 94.3% of states, above the 90% drift threshold, so the lists are not described as from a drifted sampler. The archived greedy action was never sampled at 9 states (primary) and 44 (temperature 1); 2 and 28 lists were cut at eight candidates. 357 states (9 games) have no swap object. vLLM 0.8.5 served the agent.

**The lists are short.** The primary list holds 2.17 candidates on average, and 1,289 of the 2,973 states (43%) have a single candidate, because at those states all 20 samples repeated the greedy action or one of them. At a single-candidate state every verifier must choose that candidate, so no verifier can change anything there. The temperature-1 list holds 3.28 on average, with 659 single-candidate states. Every failing candidate in either list is outside the environment's admissible-action list (free-form): the admissible list holds only working actions here.

**Encounter.** A regime candidate appears in the primary list at 4.3% of states [2.2, 6.8], 129 states in 22 of 134 games, at a median step of 22. It never appears at step 0 (0 of 134), at 0.45% of states at steps 1–11 [0.08, 0.89] and at 8.2% at steps 12–29 [4.2, 12.6]. The strict, RQ2-like encounter (the history never acted on the goal object) is 0.24% [0.07, 0.45], 7 states. The 159 regime candidates are all free-form; 61 of them are the agent's own rank-1 action (the dry run, which could not apply the shared-spot exclusion, gave 61 as an upper bound), and the share of candidates that are regime candidates is between 2% and 7% at each of ranks 1–6 (none at ranks 7 and 8, which hold 29 candidates). A regime candidate ranked above the first working candidate ("reachable") occurs at 2.5% of states [1.2, 4.1], in 17 games; that is 74 of the 129 encounter states. In the temperature-1 list the encounter is 5.8% [3.3, 8.6] in 31 games, strict 0.6%.

**Chosen-failure rates by configuration.** Percent of the 2,973 states whose chosen action fails.

| Configuration | Llama-3.1-8B | Mistral-7B-v0.3 | Qwen3-8B |
|---|---|---|---|
| None (the agent's own action) | 22.6 [19.0, 26.2] | 22.6 | 22.6 |
| Task-blind verifier | 18.0 [15.1, 20.8] | 21.8 [18.3, 25.3] | 13.7 [11.1, 16.4] |
| Neutral task ("explore the room") | 17.5 [14.7, 20.2] | 21.7 [18.2, 25.2] | not scored |
| Swapped task (2,616 swap states only) | 17.5 [14.4, 21.0] | 21.5 [17.7, 25.6] | 15.1 [12.0, 18.6] |
| Task-aware verifier | 18.2 [15.2, 21.2] | 22.0 [18.5, 25.5] | 16.0 [13.0, 19.0] |
| Oracle (knows every outcome) | 7.4 [5.3, 9.6] | 7.4 | 7.4 |

Every verifier improves on the bare agent. The improvement is about 4.4 to 4.6 points for Llama (none minus aware +4.4 [2.7, 6.2]; none minus blind +4.6 [3.0, 6.4]), 0.6 to 0.8 for Mistral (+0.6 [0.0, 1.5]; +0.8 [0.1, 1.7]) and 6.6 to 8.9 for Qwen (+6.6 [4.6, 8.7]; +8.9 [6.9, 10.8]). The gap to the oracle stays large: 10.6–10.8 points for Llama, 14.4–14.6 for Mistral, 6.3–8.6 for Qwen. The oracle's 7.4% is the share of states where no listed candidate works, mostly single-candidate states whose one candidate fails.

**Neutral arm (priority models; never decisive).** Aware minus neutral is +0.71 points [−0.27, +1.82] for Llama and +0.27 [−0.07, +0.66] for Mistral; neutral minus blind is −0.54 [−1.25, +0.10] and −0.10 [−0.35, +0.07].

## Pre-registered secondaries

### How decisions change (change paths)

Every state where the aware and blind verifiers choose different actions is assigned a path: (i) the aware verifier abstained, so the agent took rank 1; (iv) the blind verifier abstained; (ii) the aware verifier accepted an action ranked above the blind verifier's choice, which the blind verifier rejected; (iii) the aware verifier rejected the action the blind verifier chose, which lies above its own choice. The design notes that under this rule a blanket rejection cannot reach a fallback menu as in Stage 1a: it can only abstain.

| Path, aware against blind (harm / help / both fail / both work) | Llama | Mistral | Qwen |
|---|---|---|---|
| (i) aware abstained | 16 / 1 / 0 / 1 | 0 / 0 / 0 / 0 | 69 / 5 / 2 / 51 |
| (iv) blind abstained | 3 / 0 / 0 / 2 | 0 / 0 / 0 / 0 | 3 / 5 / 1 / 8 |
| (ii) aware accepted what blind rejected, ranked above | 21 / 7 / 5 / 23 | 6 / 1 / 1 / 1 | 4 / 0 / 0 / 19 |
| (iii) aware rejected what blind chose | 8 / 35 / 0 / 62 | 2 / 2 / 0 / 15 | 12 / 10 / 0 / 101 |
| All changed states | 184 (6.2% of states) | 28 (0.9%) | 290 (9.8%) |

**By step band.** No model has any harm or help at step 0. The only step-0 changes are both-work swaps between two working first moves: 7 for Llama and 13 for Mistral through path (iii), 7 for Qwen through path (i). Every harm and help state is at step 1 or later. The same holds in each part of the decomposition (aware against swapped, swapped against blind), overall and on the encounter states (all cells in the metrics file). For Llama the two directions nearly cancel: the task leads it to accept failing actions ranked above the blind choice (path ii, 21 harm against 7 help) and to reject failing actions the blind verifier chose (path iii, 35 help against 8 harm). Qwen's harm is almost all path (i), abstention.

### Hybrid verifiers

The design recombines the arms: the aware arm judges the agent's rank-1 action while the blind arm judges the alternatives, and the reverse. Each is compared with the blind verifier (points, with intervals and descriptive p-values).

| Hybrid minus blind | Llama | Mistral | Qwen |
|---|---|---|---|
| Aware judges rank 1, blind judges alternatives | −0.47 [−1.46, +0.44], p 0.39 | +0.17 [−0.09, +0.47], p 0.39 | +0.14 [−0.16, +0.43], p 0.51 |
| Blind judges rank 1, aware judges alternatives | +0.71 [+0.18, +1.34], p 0.02 | 0.00 | +2.15 [+1.25, +3.23], p 0.001 |

The task does two things to Llama's verifier that pull in opposite directions: on the agent's own action it slightly lowers failures, and on the alternatives it raises them; together they give the null +0.17. Mistral's small difference comes entirely from its judgement of rank 1. Qwen's harm comes entirely from its judgement of the alternatives.

### Acceptance by position, outcome and step

Yes rates in percent, over the candidates each arm scored (the aware, swapped and neutral arms are not scored at single-candidate states, so their rates cover fewer candidates than the blind arm's).

| Candidates | Llama aware | Llama blind | Mistral aware | Mistral blind | Qwen aware | Qwen blind |
|---|---|---|---|---|---|---|
| Working rank 1, step 0 | 88.6 | 100.0 | 80.0 | 100.0 | 3.8 | 90.3 |
| Working rank 1, later | 87.5 | 89.5 | 99.6 | 99.7 | 57.8 | 72.8 |
| Failing rank 1, later | 58.1 | 58.1 | 96.3 | 95.1 | 13.4 | 14.8 |
| Working alternatives, step 0 | 77.9 | 91.6 | 86.3 | 100.0 | 0.0 | 61.1 |
| Working alternatives, later | 89.7 | 91.4 | 99.6 | 99.5 | 52.6 | 67.5 |
| Failing alternatives, later | 71.3 | 67.3 | 92.2 | 88.4 | 19.9 | 22.3 |
| Regime candidates | 79.7 | 57.2 | 91.5 | 88.7 | 24.8 | 20.8 |

No failing candidate occurs at step 0. Paired on the same candidates, the aware-minus-blind shift in Yes on regime candidates is +21.6 points for Llama (153 candidates; mean $L$ +1.25 nats), +2.6 for Mistral (+0.50 nats) and +3.3 for Qwen (+1.90 nats). Split by the decomposition, Llama's regime shift is aware minus swapped +16.5 points and swapped minus blind +1.0 (103 candidates): the object-naming signature of RQ2, at the level of scores. Qwen's is aware minus swapped +8.7 and swapped minus blind −6.8. On working rank-1 actions at step 0 the swapped-minus-blind shift is −20.9 points (Llama), −29.7 (Mistral) and −86.8 (Qwen), and aware minus swapped is +11.0, +8.8 and +2.2: for Qwen any concrete task does almost all of it. The AUCs (accepting successes over failures, by $L$) are 0.73 aware against 0.75 blind (Llama), 0.76 against 0.82 (Mistral) and 0.77 against 0.85 (Qwen).

**Headline: Yes on working rank-1 actions at step 0.**

| Arm | Llama | Mistral | Qwen |
|---|---|---|---|
| Task-aware | 88.6% | 80.0% | 3.8% |
| Swapped | 79.1% | 70.3% | 2.2% |
| Task-blind | 100.0% | 100.0% | 90.3% |
| Neutral | 100.0% | 100.0% | not scored |

Qwen's Stage 1a pattern (a blanket No under any concrete task at step 0: 7% aware, 6% swapped, 90% blind) reproduces on the agent's own first moves.

### Fallback and abstention

The abstention table above is the pre-registered inert readout. Qwen's aware and swapped verifiers abstain at 98.5% and 100% of step-0 multi-candidate states; because the rule then returns the agent's own first move, which always works at step 0, this costs nothing there. After step 0 Qwen's aware verifier abstains at 26.6% of states against 15.1% for the blind verifier, and that is where its harm arises (69 of its 88 harm states are path (i)).

### Sensitivities (pre-registered)

Aware minus blind in points, with interval and descriptive p.

| Sensitivity | Llama | Mistral | Qwen |
|---|---|---|---|
| Primary, as frozen | +0.17 [−0.92, +1.27] | +0.17 [−0.09, +0.47] | +2.29 [+1.36, +3.33] |
| Each game weighted equally | +0.18 [−0.62, +1.00] | +0.15 [−0.05, +0.40] | +1.79 [+1.08, +2.64] |
| Steps 0–11 only (1,478 states) | +0.07 [−0.48, +0.68] | +0.34 [0.00, +0.77] | +1.15 [+0.40, +2.08] |
| List cut to 2 candidates | +0.27 [−0.86, +1.42] | +0.17 [−0.09, +0.47] | +1.78 [+1.03, +2.65] |
| List cut to 4 candidates | +0.17 [−0.92, +1.27] | +0.17 [−0.09, +0.47] | +2.15 [+1.23, +3.17] |
| Fallback to the highest-scored candidate instead of rank 1 | −0.30 [−1.31, +0.69] | +0.17 [−0.09, +0.47] | +0.94 [+0.30, +1.62] |
| Temperature-1 list | +0.13 [−1.14, +1.45] | +0.24 [−0.07, +0.60] | +3.26 [+2.18, +4.51] |
| Temperature-1 list, its encounter states (171 states, 31 games) | +4.09 [−2.21, +11.00], 7 games non-zero | 0.00 [−1.85, +1.59] | +21.05 [+13.60, +29.11] |
| Rank 1 and candidates sampled at least twice | +0.34 [−0.61, +1.33] | +0.17 [−0.04, +0.41] | +1.88 [+1.11, +2.82] |
| A failing action counts only at its first listing in a game | −0.32 [−0.62, 0.00], p 0.08 | +0.04 [0.00, +0.11] | +0.50 [+0.15, +0.89] |

For the priority models every sensitivity agrees with the primary: no interval excludes zero on the harmful side, and the largest point estimate is +0.34 points. The temperature-1 list's encounter states give Llama +4.1 points with 7 games changing, the same picture as the primary conditional estimand.

**The highest-$L$ fallback.** The design pre-declared that if this sensitivity diverges from the primary, the divergence is read as Stage 1a's fallback-ranking mechanism and is not evidence for the main finding. For Llama and Mistral it does not diverge (−0.30 against +0.17; identical for Mistral, which never abstains). For Qwen it diverges, in the direction opposite to Stage 1a's: replacing abstention with the highest-scored candidate shrinks Qwen's difference from +2.29 to +0.94 points, because when Qwen's aware verifier rejects everything, its highest-scored candidate is often the working alternative the blind verifier would have chosen. Under the pre-declared reading this divergence is not evidence for the main finding; what it shows is that most of Qwen's primary difference is produced by the abstention rule returning a failing rank-1 action.

### Progress (descriptive)

The share of chosen actions that are progress events (picking up, transforming, placing or revealing the goal object, or winning) is 10.0% for the bare agent and 10.8% for the oracle. Aware minus blind is −0.13 points [−0.40, +0.11] for Llama, −0.10 [−0.21, 0.00] for Mistral and −0.10 [−0.60, +0.37] for Qwen.

### Stage 2 feed

The aware and blind verifiers disagree at 184 states for Llama (48 harm in 26 games, 43 help, 5 both fail, 88 both work), 28 for Mistral (8 harm in 5 games) and 290 for Qwen (88 harm in 40 games, 20 help). The design's suggested gate for Stage 2 (a priority model "toward failure, attributable to the real task", harm mostly from regime candidates) is not met by either priority model.

## Qwen3-8B as the contrast

Qwen was scored as a contrast because it almost never turns a determined goal-object failure into Yes. It is in no Holm family, and the frozen code gives it no verdict; the numbers below are descriptive.

Qwen's aware verifier makes the chosen action fail more often than its blind verifier: +2.29 points [+1.36, +3.33] overall (p 0.001, 43 games non-zero; dropping any one game leaves +2.11 to +2.41) and +15.5 points [+8.8, +23.3] on the encounter states (p 0.002, 10 games non-zero). But the overall difference does not come from the mechanism RQ2 measures. Only 15 of its 88 harm states choose a regime candidate; 50 choose a failing `go to`, 9 another goal-object failure and 14 something else. On the swap states the difference is mostly what any task does (swapped minus blind +1.53 [+0.85, +2.35], p 0.002) rather than what naming the object adds (aware minus swapped +0.38 [−0.26, +1.15], p 0.39). The route is abstention: under any concrete task Qwen rejects working actions much more often (Yes on working alternatives after step 0 falls from 63–78% task-blind to 52–54%), so it accepts nothing at more states, and the rule returns the agent's rank-1 action. When that action is one of the agent's failing repeats and the blind verifier would have accepted a working alternative, the task has cost the agent a rescue. That is 69 of the 88 harm states. Seen against no verifier at all, Qwen's aware verifier still helps (+6.6 points), only less than its blind verifier (+8.9).

On the encounter states the picture is closer to RQ2's: 15 of 22 harm states choose a regime candidate, and the encounter-state decomposition puts +7.7 of the +12.1 points in aware minus swapped (p 0.12, 4 games non-zero). Of those 15, 9 are states where Qwen's aware verifier accepted the regime candidate (said Yes to a failure the history determines) in 5 games, 4 of them in one game; the other 6 are abstentions that returned a regime candidate at rank 1, which Qwen had rejected. The deep-dive follows these states one by one.

## Predictions, checked

The predictions were logged before any data of this stage. They were written for the temperature-1 lists, which became the sensitivity list; the design said prediction 1 applies to the tempered primary list as an upper-leaning guess.

| Prediction | Outcome | Held? |
|---|---|---|
| 1. A regime candidate at 8–30% of states (central 15%); most regime candidates free-form; reachable at under half of encounter states; strict encounter 0.5–5% | 4.3% primary, 5.8% temperature 1; all 159 free-form; reachable at 74 of 129 encounter states (57%); strict 0.24% primary, 0.6% temperature 1 | No on the rate (below the range in both lists) and on reachability; yes on free-form; strict holds only for the temperature-1 list |
| 2. Qwen: absolute difference below 0.5 points, interval including zero | +2.29 [+1.36, +3.33] | No |
| 3. Llama: positive, 0.2–1.5 points, concentrated in regime candidates the aware verifier accepts; may not be significant | +0.17, not significant; 15% of harm from regime candidates | No (estimate just below the range, harm not concentrated in regime candidates) |
| 4. Mistral: positive, 0.5–3 points; the largest decision-change rate of the three; the smallest verifier value, possibly negative | +0.17; change rate 0.9% (smallest of the three); verifier value +0.6 to +0.8 points (smallest, positive) | Partly (smallest verifier value held; the size and change rate did not) |
| 5. Llama and Mistral: aware minus swapped positive and most of aware minus blind; swapped minus blind either sign | Llama +0.27 of +0.31 (both near zero); Mistral −0.11 of +0.19 | Llama nominally, inside noise; Mistral no |
| 6. Qwen and Llama: every verifier below the bare agent's 22.6%; oracle below 3% | Qwen 13.7–16.0%, Llama 17.5–18.2%; oracle 7.4% (primary), 1.5% (temperature 1) | Yes on verifiers; the oracle bound holds only for the temperature-1 list |
| 7. Qwen (and possibly Mistral): high aware and swapped abstention at step 0 with near-zero harm there | Qwen 98.5% and 100%, zero step-0 harm; Mistral 6.2% and 7.5% | Yes for Qwen; not for Mistral |

## What the result means

**What was asked, and what came back.** Stage 1b asked whether, at the decision points of a real agent with the candidates that agent really proposes, a verifier that sees the task leads the agent to a failing action more often than one that does not. For the two verifiers chosen because they are susceptible, the frozen answer is "no reliable difference" on both estimands: +0.17 points per decision for each, with intervals reaching +1.27 (Llama) and +0.47 (Mistral). By the pre-declared rule this is read as "the bias rarely reaches this agent's decisions", because neither verifier hands its decisions back to the agent by abstaining.

**Why it rarely reaches decisions.** Three facts about the agent and the rule, all visible in the data, explain the null.

First, the agent rarely puts the vulnerable action in front of the verifier. A determined goal-object failure appears in the agent's list at 4.3% of states, in 22 games, almost all after step 12, and never at step 0. 43% of states have a single candidate, where no verifier can change anything.

Second, the selection rule protects the agent's own choice. The verifier acts on the highest-ranked candidate it accepts, and both priority verifiers accept the agent's working rank-1 action almost always (Llama 87–89%, Mistral 80–100%). Llama's aware verifier does say Yes to far more regime candidates than its blind verifier (80% against 57%, a paired shift of +22 points: the RQ2 bias is present in its scores). But of the 73 regime alternatives it accepts, 68 sit below a candidate it accepted first, so only 5 are ever chosen. The bias can only act when the regime candidate is the first one accepted.

Third, when the regime candidate is the agent's own rank-1 action (61 states), both verifiers usually let it through, so the aware verifier's extra Yes changes nothing: the agent would have taken that failing action anyway. Llama's blind verifier rejects only 10 of the 42 such rank-1 candidates at multi-candidate states, and Mistral's blind verifier 3 (its aware verifier rejects the same 3).

**The conditional estimand as the bound on harm where the vulnerability can act.** The design warned that an overall bound of about one point per decision corresponds to roughly $1/e$ times that among encounter states, where $e$ is the encounter rate,

$$
\text{bound per encounter state} \approx \frac{\text{bound per decision}}{e},
$$

so with $e = 0.043$ Llama's overall upper bound of 1.27 points would allow up to about 29 points of harm per encounter state if all of it sat there. The conditional estimand measures that directly. For Mistral it is tight: 0.0 points [−2.5, +2.1], so harm above about 2 points per encounter state is excluded. For Llama it is not: +4.7 points [−1.2, +12.5], with 7 harm states in 3 games, 5 of them in one game. That interval still contains the "modest" per-encounter effect the precision table assumed (net 8 points), and it cannot be read as showing the bias harmless where it applies; it excludes only effects above about 12.5 points per encounter state. Llama's conditional test also could not have reached significance with the changes it saw (4 games, smallest attainable p 0.125). The deep-dive states what each interval excludes in more detail.

**How this squares with Stage 1a's large effect.** Stage 1a found a task-aware Qwen verifier choosing failing actions at 23.9% of states against 4.4% task-blind. Its deep-dive showed that this needed two ingredients together: any concrete task makes Qwen's verifier say No to working first moves at step 0, and naming the object ranks failing goal-object actions first in a fallback menu that held many of them. Stage 1b reproduces the first ingredient exactly (Qwen accepts 3.8% of the agent's working first moves under the real task, 2.2% under the swapped task, 90% task-blind) and removes the second twice over. The agent's step-0 lists contain no failing candidate at all (185 `go to`, 43 `look` and 1 `inventory` across 134 states), so there is nothing to fall back to; and under this rule a verifier that accepts nothing abstains, returning the agent's own working first move. Step-0 harm is therefore zero for every model. After step 0, Stage 1a's Qwen difference was +7.6 points on even steps 2–10 with a fallback menu; Stage 1b gives +1.3 points on the same steps (deep-dive). The priority models' Stage 1a effects were also mostly at step 0 (Mistral +15.7 points at step 0, +0.5 later; Llama +3.0 and +0.7) and do not reappear. The two stages therefore agree: the vulnerability is real in scores, and it changes decisions when the candidates put the dangerous actions where the verifier will pick them, which this agent's own lists rarely do.

**What the Qwen contrast adds.** A verifier can still hurt this agent, and the one that does is the model chosen as insensitive to the bias. Its harm comes from any concrete task lowering its acceptance of working actions, which under this rule makes it give up more often and lose rescues of the agent's failing repeats. That is a real practical cost of showing a task to a yes/no verifier, but it is not the object-naming bias of RQ2: the swapped task does most of it, and most harm states involve no goal-object action. Read with the pre-declared readings, it would not count as evidence for the main finding.

**What would change the conclusion.** An agent whose lists put goal-object failures above its working actions more often (a broader or more exploratory sampler, a planner that ranks by predicted success, or a rule that falls back to the highest-scored candidate) would give the bias more decisions to act on, as Stage 1a's menus did. A tighter conditional interval for Llama would need encounter states spread over more games: here they sit in 22 games, 81 of the 129 in the six games with 13–14 encounter states each.

## Reading and limits

- **What this licenses:** for this agent (the RQ3 Qwen3-8B ReAct agent, thinking off), its own candidate lists, the yes/no prompt and the "highest-ranked accepted, else rank 1" rule, showing two susceptible verifiers the task does not reliably change how often the chosen action fails; the overall effect is below about 1.3 points per decision for Llama and 0.5 for Mistral, and below about 2 points per encounter state for Mistral.
- **What it does not license:** that the bias is harmless where it can act for Llama (its conditional interval reaches 12.5 points); that it is harmless for other agents, rules or planners; anything about end-to-end task success, since nothing is executed after the choice.
- **The agent saw the admissible actions** *(revised 2026-11-25)*. Its prompt included ALFWorld's admissible-action list, and every failing candidate in either list lay outside it (above, "What ran"). The nulls therefore mean "not detected in this setup"; an agent without the list may propose such failures more often.
- **Scope:** susceptible verifiers only, one agent, one environment, one prompt format, one rule. Stage 1b's priority pair differs from Stage 1a's co-primary pair, so only Llama is primary in both.
- **The task-blind arm** keeps the few-shot examples' own task lines, as in every arm.
- **Everything here is exploratory**: the design note labels the whole stage exploratory, and the frozen verdicts are exploratory verdicts reached by rules fixed before the data.
