# Walkthroughs, part 0: overview of the question, the studies and the shared measurement

*Written 2026-11-26. Descriptive; it adds no result. Every number is quoted from a committed results note or metrics file in `notes/`, named next to it, or, for the worked example of section 1, recomputed from the stored score files and checked against them by `scripts/walkthroughs/overview_worked_example.py`. Where this document and a results note disagree, the results note is authoritative.*

This is the first of four walkthrough documents. It sets out the question, works through one real decision from start to finish, maps every study the project has run, explains once the measurement machinery that all studies share, and draws the cross-benchmark summary figures. The three companion documents walk through many more sample decisions, benchmark by benchmark:

- [01_alfworld.md](01_alfworld.md): ALFWorld, the text household environment of the preregistered test and most follow-ups;
- [02_textcraft.md](02_textcraft.md): TextCraft, a Minecraft-style crafting game;
- [03_taubench.md](03_taubench.md): τ-bench retail, a customer-service backend driven by JSON tool calls.

Section 5 explains how to read them and ends with a glossary of every term used in the four documents.

## Contents

1. The question in plain words, and one decision worked end to end
2. A map of every study
3. The measurement machinery, explained once
4. Cross-benchmark summary figures
5. How to read the per-benchmark documents, and a glossary

---

## 1. The question in plain words, and one decision worked end to end

### 1.1 The question

A language-model agent acts in an environment by issuing text commands and reading the environment's text replies. Agents increasingly consult a **world model** before they act: a model asked to predict what the environment will reply to a proposed action. In this project the world model is an ordinary instruction-tuned language model prompted to play that role. Its prompt shows the episode so far (the **history**), the proposed **action**, and asks for the next observation. The history contains the agent's **task**, a one-line goal such as "put two soapbar in garbagecan".

All three environments studied here are **goal-independent**: what the environment replies depends on the state of the world and the action, not on what the agent is trying to achieve. Write $s_t$ for the state at step $t$, $a_t$ for the action and $g$ for the goal. Goal independence says

$$
P(s_{t+1} \mid s_t, a_t, g) = P(s_{t+1} \mid s_t, a_t).
$$

Once the state and action are known, the goal adds no information about the outcome. A world model that has learned such an environment should therefore give the same prediction whatever the task line says. The project tests whether it does on actions whose outcome is not in doubt: **determined failures**, actions that fail and whose failure the history already proves. The finding that runs through every study is that it does not. When the task names the object the action acts on, the world model's predicted success rises. The project calls this the **goal effect**. Its title, "The Task Is Not the World", is the correct reading the models fail to make.

Why it matters: an agent that checks its proposals with such a world model is most optimistic exactly about the actions that would advance its goal, which are the actions it is most likely to take. An error toward false success that is correlated with what the agent wants is not a random error. Whether it actually changes decisions is the subject of the decision studies (section 4.5).

### 1.2 One decision, from the stored data

The example is ALFWorld item `t_002295` from game `valid_unseen/191`. It is one of the 4,666 items of the preregistered test (RQ2), and it is also the repository's golden test fixture (`tests/golden/default_prompt.json`), so its rendering is checked by a test on every change to the code. The walkthrough for ALFWorld ([01_alfworld.md](01_alfworld.md)) goes through many more items of different kinds; this one was chosen because every model in the families study scored it and because it shows, on a single item, most of the things the later sections measure in aggregate.

**The history.** The agent has taken one action. This is the history exactly as the real ALFWorld server printed it:

```
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put two soapbar in garbagecan.
> go to cabinet 1
You arrive at cabinet 1. The cabinet 1 is closed.
```

**The action and the true reply.** The action to be predicted is `take soapbar 1 from cabinet 1`. In ALFWorld nothing can be taken from a closed receptacle, and the history says "The cabinet 1 is closed." So the action must fail, and the server's reply is `Nothing happens.` (ALFWorld gives that same reply to every inadmissible command). The item collector recorded the reason for the failure as `contents_or_open_state`, and the item was replayed from reset on a CPU node to confirm both the history and the reply.

**The two task lines.** The item is scored twice. Under the **aligned** task line the history keeps its real task, `Your task is to: put two soapbar in garbagecan.`, which names the action's object. Under the **swapped** task line only the object name changes: `Your task is to: put two spraybottle in garbagecan.` The swap object is drawn from object types in the same room that are not the goal type, are not named in the action, are not acted on in the history, have no instance at the action's receptacle at the start, and could do the same task by the game's own rules (`wf/data/goals.py`, `pick_swap`). Every other byte of the prompt is identical; the code refuses to replace anything but exactly one task line (`wf/scoring/prompts.py`, `_replace_task_line`).

**The full prompt.** `wf/scoring/logodds.py`, `build_messages`, turns the item into a chat with eight messages: a system message, three worked examples (each a user turn and the true reply as the assistant turn) and the query. The system message reads:

```
You are a text world model for household embodied tasks. Given the interaction history and the next action, predict the environment's next observation exactly in ALFWorld style. Output only the observation text.
```

The three examples are real server text from training games: one success (`take spraybottle 1 from cabinet 1` after opening the cabinet, reply "You pick up the spraybottle 1 from the cabinet 1.") and two failures (reply "Nothing happens."). The query turn is:

```
History:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, [...] and a towelholder 1.
Your task is to: put two soapbar in garbagecan.
> go to cabinet 1
You arrive at cabinet 1. The cabinet 1 is closed.

Action: take soapbar 1 from cabinet 1

Next observation:
```

Each model then renders the chat with its own chat template. For Qwen3-8B with thinking off, the template ends the prompt with `<|im_start|>assistant\n` followed by an empty reasoning block, `<think>\n\n</think>\n\n`, after which the answer is scored (`notes/qwen_effect_audit_pipeline.md`, "Chat template and the thinking-off block"). That empty block matters for the size of Qwen's numbers and comes up again in section 3.

The script `scripts/walkthroughs/overview_worked_example.py` re-renders the aligned and swapped messages with `build_messages` and checks them byte for byte against the golden fixture; it also recomputes the item's hash (`wf/runs.py`, `_item_hash`, a SHA-256 over the action, history, both task lines and both candidate replies), `337347b2bb51ddbb`, and requires every stored score line it reads to carry the same hash.

**The two candidate replies.** Rather than letting the model write a reply, the main readout feeds it two fixed replies and asks how probable each is. The failure candidate, written $o^{-}$, is the true reply `Nothing happens.`. The success candidate, written $o^{+}$, is `You pick up the soapbar 1 from the cabinet 1.`: what the real server prints when this exact action works. It was obtained by replaying a short plan from reset (go to the cabinet, open it, then take the soap bar) and recording the server's reply, never written by hand (`wf/data/candidates.py`); it is called the **success twin**.

**Teacher forcing and the log-odds.** Each candidate is fed to the model token by token after the prompt, and the log-probabilities the model assigns to its tokens are summed. If a candidate has tokens $o_1, \dots, o_n$ and the prompt is $x$,

$$
\log P(o \mid x) = \sum_{k=1}^{n} \log P(o_k \mid x, o_1, \dots, o_{k-1}).
$$

The **two-candidate log-odds** is the difference of the two sums,

$$
L = \log P(o^{+} \mid x) - \log P(o^{-} \mid x),
$$

measured in nats (natural-log units). A positive $L$ means the model finds the success sentence more probable than the failure sentence; $L = 0$ means it is indifferent between them.

For Qwen3-8B under the aligned task the stored log-probabilities are −13.565 for the success twin and −0.0019 for "Nothing happens.", so $L = -13.563$; under the swapped task they are −17.617 and −0.0001, so $L = -17.616$ (run `rq2_score`, commit `ddeabc2`). The script recomputes each $L$ as the difference of its two stored log-probabilities and finds the stored $L$ to within $10^{-6}$; the yes/no run `rq1v_score`, which repeated these two scores in a separate job, reproduced them to every printed digit.

**The goal effect on this item.** The item's goal effect is its aligned score minus its swapped score:

$$
\Delta L_i = L_i(\text{aligned}) - L_i(\text{swapped}) = -13.563 - (-17.616) = +4.053 \text{ nats}.
$$

Naming the soap bar in the task made the success sentence about $e^{4.05} \approx 57$ times more probable relative to "Nothing happens.", although the cabinet is closed under both task lines.

**The same shift in probability.** The logistic of $L$,

$$
s = \sigma(L) = \frac{1}{1 + e^{-L}},
$$

puts the score on a 0-to-1 scale: the probability of the success sentence relative to the two-sentence set. It is not the model's probability that the action succeeds (section 3.2 explains why). For Qwen3-8B, $s$ moves from $2.2 \times 10^{-8}$ under the swapped task to $1.3 \times 10^{-6}$ under the aligned task. A shift of 4 nats that changes nothing anyone would notice: Qwen remains certain the action fails.

**Every model on the same item.** The families study scored this item with four more models under the same prompts (run `family_score_<model>`, commit `65eb591`), and the yes/no question of section 3.2 was scored for all of them. The table lists the stored values; "neutral" is the task line `Your task is to: explore the room.`, which names no object. Columns give the two-candidate log-odds under each task line, the item's goal effect, the score $s$ under each line, and the yes/no log-odds (Yes against No) under each of the three lines.

| Model | L, swapped | L, aligned | Goal effect, nats | s, swapped | s, aligned | Yes/no, swapped | Yes/no, aligned | Yes/no, neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −17.62 | −13.56 | +4.05 | 0.00000002 | 0.0000013 | −26.25 | −23.75 | −25.50 |
| Llama-3.1-8B-Instruct | −1.42 | +4.63 | +6.04 | 0.195 | 0.990 | −5.00 | +2.25 | −4.75 |
| Mistral-7B-Instruct-v0.3 | +2.68 | +6.67 | +4.00 | 0.936 | 0.999 | −2.25 | +7.63 | +4.25 |
| Gemma-2-9B-it | −2.32 | +3.99 | +6.31 | 0.090 | 0.982 | −8.13 | −2.75 | −7.19 |
| Llama-3.2-3B-Instruct | +2.55 | +3.22 | +0.67 | 0.928 | 0.962 | −0.13 | 0.00 | −0.25 |
| Llama-3.2-1B-Instruct (not competent) | +0.23 | −0.14 | −0.37 | 0.557 | 0.465 | −0.75 | −0.88 | −0.50 |

![Dumbbell chart, one row per model: each row joins the item's score under the swapped task (hollow circle) to its score under the aligned task (filled circle), for the two-candidate readout (left) and the yes/no readout (right). Every competent model's filled circle lies to the right of its hollow one; Llama-3.1-8B, Gemma-2-9B and Mistral-7B cross zero on one or both readouts, while Qwen3-8B moves right but stays far left of zero.](figures/overview_worked_example.png)

*Figure 0. One failing action, scored by six models under the real and the swapped task. Look at the direction of each segment (left to right means the real task raised predicted success) and at whether it crosses the vertical line at zero, where the model's preference switches from failure to success. Data: the stored score lines of runs `rq2_score`, `rq1v_score` and `family_score_<model>`; script `scripts/walkthroughs/overview_worked_example.py`.*

Four things in the table recur throughout the project.

*Direction.* Every model that passes the competence rule (section 3.6) moves toward success under the real task, on both readouts. Llama-3.2-1B, which fails that rule, moves the other way by a little; its results are never interpreted.

*Size in nats is not comparable across models.* Qwen3-8B's shift of 4 nats is large in nats and invisible in probability, because Qwen's scores sit 14 to 26 nats below zero. Llama-3.1-8B's shift of 6 nats carries the item across zero: the success sentence goes from about one in five to 99% of the pair, and its yes/no answer changes from a firm No ($P(\text{Yes}) = 0.007$) to Yes ($P(\text{Yes}) = 0.90$). Whether a shift changes an answer depends on where the model's scores start, which is why section 3.4 introduces scale-free measures.

*Wrong answers that are not caused by the goal.* Mistral-7B already prefers the success sentence under the swapped task (s = 0.94) and answers Yes under the neutral task (+4.25): it gets this item wrong even without the goal. The goal effect is a shift; a model can be wrong for other reasons too, which is why each study checks competence.

*Ties and the 1/64 grid.* Llama-3.2-3B's aligned yes/no score is exactly 0.00, a tie between Yes and No. Scores are computed from logits held in bfloat16, so yes/no scores come in steps of 1/64 nat or coarser and exact ties are common; by the families rule a tie counts as not answering Yes (section 3.3).

**From one item to the result.** The preregistered test averages $\Delta L_i$ over all 4,666 items. For Qwen3-8B the mean is 6.12 nats [5.48, 6.75], for Llama-3.1-8B 1.65 [1.51, 1.78], both "supported (strong)" with Holm-adjusted p 0.002 (`notes/rq2_confirmatory_metrics.json`, `notes/rq2_results.md`). This item, at 4.05 nats for Qwen, is below Qwen's mean; at 6.04 for Llama it is far above Llama's. Section 3 explains each piece of that sentence: the mean, the interval, the p-value and the verdict.

---

## 2. A map of every study

The project has run, as of 2026-11-26, about thirty studies on three benchmarks. The tables below list each, grouped by what it asks. They are built from the README's status table and `notes/GUIDE.md`, section 8, and each headline was checked against the results note named in the last column. Each row says, in plain words, which models were scored, what the paired prompts manipulated, which readouts were used (section 3.2 defines them), the study's standing and freeze tag, its status, and its headline result.

Three words recur in the "standing" column. **Preregistered** means the test was filed in the preregistration of 2026-11-05 (not in this repository). **Frozen** means the design and analysis code were fixed at an annotated git tag before any score was read; such studies may be confirmatory (their verdicts license claims at a stated error rate) or exploratory with a frozen procedure. **Exploratory** means descriptive and hypothesis-generating; some exploratory numbers are marked **post hoc**, meaning they were computed after the results were in. Status is "done" when the analysis has run on every planned model, "interim" when the frozen analysis was run at the user's request with some models still waiting (their tests enter the multiple-comparison correction as p = 1), and "running" when no result exists yet. Studies marked running are listed without results.

### 2.1 ALFWorld: the core tests

These studies use ALFWorld's determined failures, collected by a Qwen3-8B agent from the 134 solvable unseen games and replay-verified. "Qwen3-8B" is always scored in its thinking-off chat format unless a row says otherwise.

| Study | Models | What the paired prompts change | Readouts | Standing, tag | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| Pilots A and B | Qwen3-8B, Llama-3.1-8B, a fine-tuned Word2World model; ReAct agent | Built and scored items on training games; checked that the agent uses predictions | All | Pilot, no research question | Done | Sized the study; no verdict | `PILOT_PROTOCOL.md`, `notes/deviations.md` (2026-11-17) |
| RQ2: the goal effect | Qwen3-8B, Llama-3.1-8B-Instruct | Real task against the same task naming another object in the room | Two-candidate (primary); first token, s, written predictions (secondary) | Preregistered; analysis reproducible at release tag `repro-rq2-2026-11-25` | Done | Supported (strong) for both: 6.12 nats [5.48, 6.75] and 1.65 [1.51, 1.78] | `notes/rq2_results.md` |
| RQ1 and H1: direction of written errors, plus secondary RQ2 conditions | Qwen3-8B, Llama-3.1-8B, two Word2World fine-tuned models | Three few-shot example sets; full hidden state shown; an instruction to ignore the goal; extra-mention sentences | Written predictions, parsed; two-candidate | RQ1 measure and H1 preregistered; rest decided after RQ2 | Done (parser never passed its audit rule) | Both lean toward predicting success, Llama far more (A = 0.19 against 0.77); the goal effect holds with every example set; the ignore instruction removes 9% of Qwen's effect, none of Llama's | `notes/rq1_results.md` |
| RQ3: cost of injected false successes to an agent | Qwen3-8B ReAct agent with an oracle that injects errors | Kept against dropped predictions; false successes against false failures | Episode score | Preregistered | Done | H3 not supported: interaction +0.94 steps [−0.20, 2.14]; the setup could not test the question well | `notes/rq3_results.md` |

### 2.2 ALFWorld: exploratory follow-ups to RQ2

Each of these was logged in `notes/deviations.md` before its data and is reported with uncorrected statistics.

| Study | Models | What changes | Readouts | Standing | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| Object-free goals and working actions | Qwen3-8B, Llama-3.1-8B | Every object name in the task replaced by a description | Two-candidate, AUC | Exploratory | Done | 61% of Qwen's effect remains [54%, 68%], 92% of Llama's [83%, 103%] | `notes/exploratory_results.md` §1 |
| Thinking mode | Qwen3-8B | Model reasons before answering | Two-candidate after sampled reasoning | Exploratory; reading revised 2026-11-25 | Done | Not determined: no reduction in nats (4.01 against 3.33 without the empty think block), but the standardized effect falls (0.25 against 1.10), partly through sampling noise | `notes/exploratory_results.md` §2 |
| Dissociation | Qwen3-8B, Llama-3.1-8B | Sentences that the object exists, is at a place, or is absent, in both prompts | First token, two-candidate | Exploratory | Done | Mixed readouts; led to the mechanism battery | `notes/exploratory_results.md` §3 |
| Counterfactual task and yes/no question | Qwen3-8B, Llama-3.1-8B | Task destination replaced; anchored yes/no question | Two-candidate, yes/no | Exploratory | Done | Effect unchanged with a non-real aligned task; replicates on yes/no (2.80 and 2.21 nats) | `notes/exploratory_results.md` §4 |
| Mechanism diagnostic | Qwen3-8B, Llama-3.1-8B | Deciding facts stated; environment rules stated; state questions | Two-candidate, yes/no | Exploratory | Done | Stating the facts removes 25% (Qwen) and 2% (Llama); stating the rules increases the effect; not a failure to track state | `notes/exploratory_results.md` §5 |
| Qwen3 size series | Qwen3 0.6B, 1.7B, 4B, 8B, 14B, 32B | Same prompts across sizes | Two-candidate, first token, yes/no | Exploratory | Done | Present at every size; the two smallest cannot do the task; standardized effect flat among competent sizes (0.90, 1.02, 0.96, 0.86) | `notes/exploratory_results.md` §6 |
| Qwen3-8B effect-size audits (forensic and pipeline) | Qwen3-8B, compared with every scored model | Stored scores re-read; scoring re-implemented; empty think block removed | All | Exploratory; plans committed before computing | Done | No pipeline fault; about half of Qwen's 6.12 nats is the empty think block's scale stretch (3.31 without it), standardized effect unchanged (1.02 against 1.05) | `notes/qwen_effect_audit_forensic.md`, `notes/qwen_effect_audit_pipeline.md` |
| Goal effect by history length | Five family models | None (stored scores by number of actions in the history) | Two-candidate, yes/no | Post hoc | Done | A step-0 concentration on yes/no also in Gemma-2-9B, Mistral-7B and Llama-3.1-8B, which have no think block | `notes/step0_by_history_metrics.json`, `notes/cross_model_summary.md` §5 |
| Mechanism battery | Qwen3-8B, Llama-3.1-8B; Mistral-7B secondary | Task lines built to separate accounts: mention with "ignore", desire, done or cancelled tasks, claims about the object | Two-candidate, yes/no | Exploratory, frozen at `mechanism2-design-frozen-2026-11-22` | Done | No single account for any model; for Qwen "Ignore the {object}." added to the swapped task reproduces the whole effect | `notes/mechanism2_results.md` |
| Interpretability pilot | Qwen3-8B | Activation patching and direction removal between aligned and swapped prompts | Yes/no | Exploratory calibration, frozen at `interp-design-frozen-2026-11-22` (analysis fix tags 2026-11-23, -24) | Done | Behaviour reproduced (2.782 nats); localization partial; mechanism and meaning not achieved | `notes/interp_results.md` §13 |

### 2.3 Model families

| Study | Models | What changes | Readouts | Standing, tag | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| Families, wave A | Gemma-2-9B-it, Mistral-7B-Instruct-v0.3, Llama-3.2-1B and 3B-Instruct, beside the two references | RQ2's aligned, swapped and neutral lines | Yes/no item share (primary), two-candidate | Exploratory, design fixed before data | Done | Effect in every competent model; shares toward Yes 0.75 to 0.90; Llama-3.2-1B not competent | `notes/families_results.md` |
| Families, wave B | Ministral-3-8B, Qwen3.5-9B (chat and plain) and its base model, Qwen3.8-27B; Gemma-4-E4B and 31B | As wave A, in a newer software stack; base against instruct | Yes/no item share | Exploratory, re-frozen at `families-waveb-design-refrozen-2026-11-25`; deviation tag `families-waveb-deviation-attn-float32-2026-11-25` | Interim (Gemma 4 waiting); final analysis running | Item share supported for the four scored models; Qwen3.5 instruct matches base | `notes/families_waveb_results_interim.md` |

### 2.4 Decisions: does the effect change what an agent does?

| Study | Models | What changes | Readouts | Standing, tag | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| Decision simulation, Stage 1a (fixed menus) | Qwen3-8B, Llama-3.1-8B; Gemma-2-9B, Mistral-7B, Llama-3.2-3B secondary | Verifier shown the real task, the swapped task, or no task | Yes/no as a verifier; chosen action's outcome | Frozen gate at `decisionsim-stage1a-frozen-2026-11-21` | Done | Gate "worth doing", decided by Qwen: +19.5 points [17.2, 22.3] more failing choices than a task-blind verifier; Llama no reliable change | `notes/decisionsim_results.md` |
| Decision simulation, Stage 1b (the agent's own candidates) | Llama-3.1-8B, Mistral-7B; Qwen3-8B as contrast | As Stage 1a, on candidates the agent proposed itself | Yes/no verifier; rate of acting on a failing action | Exploratory, frozen at `decisionsim-stage1b-frozen-2026-11-22` | Done | No reliable difference for Llama or Mistral; such candidates appear at 4.3% of states, never at step 0 | `notes/decisionsim_agent_results.md` |
| Option C (closed loop) | Qwen3-8B agent and verifier | Verifier shown the real task, the swapped task, no task, or no verifier | Observation readout (primary), yes/no | Exploratory, frozen at `optionc-design-frozen-2026-11-22` | Done | No reliable difference in success: −1.2 points [−3.0, 0.0]; the design could detect only losses of about 6 to 11 points | `notes/optionc_results.md` |

### 2.5 TextCraft

| Study | Models | What changes | Readouts | Standing, tag | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| TextCraft round 1 | Qwen3-8B, Llama-3.1-8B | Goal line names one of two paired goals | Two-candidate (primary), first token, yes/no (unanchored) | Confirmatory for the frozen test, `textcraft-design-frozen-2026-11-20` | Done | Confirmed for Qwen (+0.94 nats [0.63, 1.24]); small negative for Llama (−0.19 [−0.30, −0.09]); no replication across both | `notes/textcraft_results.md` |
| TextCraft follow-ups | Qwen3-8B, Llama-3.1-8B | Diagnostics, competence, comparison with ALFWorld | All | Exploratory, post hoc | Done | Qwen's unanchored yes/no answers partly judge the task; about 59% of ALFWorld's two-candidate effect is the success sentence's tail | `notes/textcraft_vs_alfworld.md` and the notes it links |
| TextCraft round 2 | Qwen3-8B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, Llama-3.2-3B | Goal names the crafted item, names it beside another goal, or makes it an ingredient of a parent goal | Two-candidate | Confirmatory for a 50-test family, `textcraft-r2-design-frozen-2026-11-23` | Done | Penalty for naming and completing the goal confirmed below zero in the three models that discriminate on goal crafting; lift on ingredients only for Qwen | `notes/textcraft_r2_results.md` |
| TextCraft round 3 | Qwen3.5-9B, Ministral-3-8B, OLMo-3-7B scored; Gemma slot, Qwen3.8-27B, Gemma-4-31B waiting | As round 2, newer models | Two-candidate | Frozen at `textcraft-r3-design-frozen-2026-11-23`; deviation tags through `textcraft-r3-deviation-g2-2026-11-25` | Interim; final analysis running | Penalty confirmed for Ministral (−1.23 nats), equivalent to zero for Qwen3.5 (numerically fragile); OLMo not competent | `notes/textcraft_r3_interim_results.md` |

### 2.6 τ-bench retail

| Study | Models | What changes | Readouts | Standing, tag | Status | Headline | Read |
|---|---|---|---|---|---|---|---|
| τ-bench retail | Qwen3-8B, Llama-3.1-8B | Request names the order and item the failing call acts on, another order, nothing, or only mentions it | First token (primary), yes/no, full reply | Confirmatory for the frozen test, `taubench-design-frozen-2026-11-21` | Done | Qwen +3.98 nats [3.61, 4.35] but a working call on the same order moves more (−5.47), so a general task-consistency bias; Llama +0.47, small | `notes/tau_results.md`, `notes/tau_dx.md` |
| τ-bench model families | Mistral-7B, Llama-3.2-3B | As τ-bench | First token | Confirmatory per model, `tau-families-design-frozen-2026-11-23` | Done | Llama-3.2-3B replicates (+0.99 [0.90, 1.08]); Mistral not interpretable | `notes/tau_families_results.md` |
| τ-bench authorization | Qwen3-8B, Llama-3.1-8B; Llama-3.2-3B extension | Backend described as ignoring the task, or as enforcing a request rule | First token | Exploratory, frozen at `tau-auth-design-frozen-2026-11-22`; extension `tau-auth-ext-llama32-2026-11-25` | Done | Qwen: stating that the backend ignores the task removes none of the effect; the Llama models fail the gates | `notes/tau_auth_results.md` |

### 2.7 Cross-cutting and paused

| Study | What it does | Status | Headline | Read |
|---|---|---|---|---|
| Audit reruns (2026-11-20) | Seven CPU analysis reruns after a documentation and rigour audit | Done | No verdict changed | `notes/deviations.md`, "Changes from the 2026-11-20 audit" |
| Numerical sensitivity | Compares every archived pair of runs that scored the same prompts under different numerics | Done | Same environment is bit-identical; changes add logit-scaled noise; four readings flagged numerically fragile | `notes/numerical_sensitivity.md` |
| ScienceWorld | Feasibility probe for a fourth environment | Paused | Almost no determined failures on the task's own object | README, status table |

### 2.8 Studies still running (no results)

These four studies have designs in the public repository; none has a result, and nothing about their outcome is reported here.

| Study | Benchmark | Models | What the paired prompts change | Readouts | Standing, tag | Status |
|---|---|---|---|---|---|---|
| Controlled Bayesian environment | A small synthetic world (six boxes, one key) whose prompt states exactly how it was generated, so every prompt has an exact posterior probability of success | Qwen3-8B, Llama-3.1-8B | Evidence strength in the history; a task line against a non-goal "house tag" carrying the same information | Reply readout (first token), yes/no, a narrator readout | Confirmatory, `bayesenv-design-frozen-2026-11-25`; deviation `bayesenv-deviation-be1-2026-11-26` | Running |
| Natural ALFWorld bridge | ALFWorld, 1,500 RQ2 items | Qwen3-8B (plain text), Llama-3.1-8B | Statements about who chose the action; a statement that every needed object is present | Two-candidate, first token, yes/no | Exploratory with a frozen procedure, `bridge-design-frozen-2026-11-25` | Running |
| Lexical-versus-semantic diagnostic | ALFWorld | Qwen3-8B (plain and with the think block), Llama-3.1-8B, and family arms | The form of a reference to the object (name, spelling, alias, description, homonym) crossed with what it says about existence | Two-candidate, first token, yes/no | Exploratory with a frozen procedure by design; design on branch `lexsem-diagnostic`, no freeze tag in the public repository as of 2026-11-26 | Running |
| Post-training pilot | ALFWorld | Qwen3-8B-Base in plain text, beside the instruct model | As the diagnostic's plain-text core | Two-candidate, first token | An arm of the lexical-versus-semantic diagnostic (same branch) | Running |

The final analyses of families wave B and TextCraft round 3 (rows above) are also running; their interim readings will be replaced.

---

## 3. The measurement machinery, explained once

Every study in section 2 turns pairs of prompts into numbers in the same few ways. This section explains each piece once, from the score of one prompt up to a verdict. The per-benchmark documents refer back here.

### 3.1 Teacher-forced scoring

All scores come from one forward pass of the model per prompt, with no text generated (`wf/scoring/logodds.py`, `LogoddsScorer`). The model is loaded at a pinned revision, the prompt is rendered with the model's own chat template, and a candidate reply is appended. Feeding a fixed continuation and reading the probabilities the model assigns to it, instead of sampling, is called **teacher forcing**. It gives exact, deterministic numbers and lets the same two candidates be compared across task lines. Log-probabilities are computed in float32 from the model's bfloat16 logits.

### 3.2 The readouts

A **readout** is the rule that turns the model's response to one prompt into one number. Four are used.

**Two-candidate log-odds.** Defined in section 1.2: the summed log-probability of the success twin minus that of the failure reply,

$$
L = \log P(o^{+} \mid x) - \log P(o^{-} \mid x).
$$

It uses the environment's real replies and is the preregistered readout of RQ2 and the primary readout of TextCraft. Its weakness is that it compares two particular sentences. If $q$ is the model's probability that the action succeeds in any wording, and $c \ge 0$ is the cost of guessing the twin's exact wording, then approximately

$$
L \approx \operatorname{logit}(q) - c,
$$

so a task that makes the words "the soapbar 1" more probable after "You pick up" moves $L$ without changing the model's belief that the action fails (`notes/deviations.md`, "Decision 5 in detail"). In ALFWorld 58 to 59% of the two-candidate effect sits in the tail of the success sentence after its first token, for both reference models (`notes/textcraft_vs_alfworld.md` §2). The score $s = \sigma(L)$ is a bounded version of $L$, useful for practical size but not a probability of success.

**First token.** Only each candidate's first token counts: "You" against "Nothing" in ALFWorld, "Craft" against "Could" in TextCraft, `{"` against `Error` in τ-bench.

$$
L_{\text{first}} = \log P(t^{+}_1 \mid x) - \log P(t^{-}_1 \mid x).
$$

This is where the two replies diverge, before any object name, so it carries no wording penalty and is the readout used to compare environments. Its risk is coverage: the two tokens may not hold most of the next-token probability. The scorer stores their total as the **first-token mass**; in the worked example it is above 0.9999 for Qwen and 0.72 for Llama under the swapped task.

**Continuation (full reply).** In τ-bench, whose success replies are long JSON objects, the log-probability of the whole reply is also reported, sometimes normalized by its length. It is a secondary readout there.

**Yes/no.** The prompt's last words, "Next observation:", are replaced by a question, and the single tokens "Yes" and "No" are compared at the next position:

$$
L_{\text{yn}} = \log P(\text{Yes} \mid x) - \log P(\text{No} \mid x).
$$

The two candidates are the same for every item, model and environment, so there is no wording penalty. ALFWorld's question is **anchored**: it names the environment's failure reply ("Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens.", `wf/rq1v.py`). TextCraft round 1's question was not anchored, and Qwen3-8B partly answered whether the action would succeed at the task rather than in the environment (`notes/textcraft_qwen_robustness.md` §9); later questions in TextCraft and τ-bench are anchored. Even anchored, a yes/no effect can include a judgement about the task: at step 0 Qwen's yes/no verifier rejects working first moves under any concrete task (`notes/optionc_results.md`). The notes' rule is that a yes/no effect is a goal effect on the model's answer, while a first-token effect is closer to a prediction of the reply.

### 3.3 Log-odds, nats, and the 1/64 grid

A log-odds is the natural logarithm of a ratio of two probabilities. It is measured in **nats**: one nat is a factor of $e \approx 2.72$ in that ratio, and a shift of $\Delta$ nats multiplies the ratio by $e^{\Delta}$. Zero means the two candidates are equally probable. Because the logits come from bfloat16 arithmetic, yes/no and first-token scores lie on a grid of 1/64 nat or coarser, and the grid widens as logits grow (1/16 nat for logits between 8 and 16, 1/4 between 32 and 64; `notes/numerical_sensitivity.md` §2). Exact ties are therefore common, which matters for any rule that picks the highest-scored action; the decision studies round every score to 1/64 before comparing (`wf/decisionsim.py`, `quantize`).

### 3.4 The goal effect and its scale-free versions

**The goal effect.** For a readout and a set of $N$ items, the goal effect is the mean of the item differences:

$$
\Delta L = \frac{1}{N}\sum_{i=1}^{N}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].
$$

A correct world model gives $\Delta L = 0$. Each item counts equally. The same formula is applied to every readout, and in the decision studies to rates (in percentage points) instead of log-odds.

**Why the swapped task, and what the neutral task adds.** The aligned and swapped lines are both concrete tasks of the same template, so anything a concrete task does to every action cancels in their difference, leaving the effect of which object is named. A neutral line ("explore the room") names no object; any concrete task lowers predicted success on actions unrelated to it relative to the neutral line (the **relevance penalty**, `notes/exploratory_results.md` §7). With all three lines scored on the same items, the difference from neutral splits exactly:

$$
L(\text{aligned}) - L(\text{neutral}) = \left[L(\text{aligned}) - L(\text{swapped})\right] + \left[L(\text{swapped}) - L(\text{neutral})\right].
$$

The first bracket is the goal effect; the second says whether the swapped task pushes predictions down. In ALFWorld the effect is mostly the real task lifting predicted success; in TextCraft and τ-bench, for Qwen on the first token, it is mostly the wrong task suppressing it (`notes/tau_dx.md` §8).

**Nats are not one scale across models.** Models spread their scores over very different ranges: across items, the swapped-task score has a standard deviation of 7.8 nats for Qwen3-8B and 3.1 for Llama-3.1-8B (`notes/cross_model_summary.md` §5). Two scale-free measures are therefore used to compare models.

The **standardized effect** divides the mean item difference by the standard deviation of the item differences:

$$
d = \frac{\overline{\Delta L_i}}{\operatorname{SD}(\Delta L_i)}.
$$

It says how consistently items move, relative to how much they vary. Qwen3-8B's two-candidate effect is 6.12 nats and $d = 1.02$; Llama-3.1-8B's is 1.65 nats and $d = 0.85$ (`notes/qwen_effect_audit_forensic_metrics.json`). Wave B's notes use a different standardization (the effect over the standard deviation of the swapped-task scores), which is not the same number; this overview uses only the item-difference version.

The **share of items moved** counts the items whose score is strictly higher under the real task than under the swapped task:

$$
\text{share} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\left\{L_i(\text{aligned}) > L_i(\text{swapped})\right\}.
$$

With no effect it is 0.5; ties count as not moved. It ignores size altogether, so it is unaffected by how extreme a model's scores are, which is why the families studies made it their primary measure on the yes/no readout (`wf/families.py`, `_share_toward_yes`).

**Practical size.** Neither measure says how many answers change. Several counts do: the shift in $s$ in percentage points; the share of items whose two-candidate score crosses zero from failure to success (a **flip**); and the share of failures whose yes/no answer goes from No under the swapped task to Yes under the real task. The same direction of effect changes 0.2% of Qwen3-8B's yes/no answers and 37.5% of Mistral-7B's, because Qwen's failures sit far below the decision boundary (`notes/families_results.md`, "Practical size depends on calibration").

**Ratios.** Some studies report a manipulation as the share of the goal effect it keeps (or removes): the effect under the changed prompt divided by the plain effect on the same items and readout. Their intervals bootstrap the ratio itself (`notes/audit_metrics.json`, `ratios`). The mechanism battery's ratio $r$ divides each contrast by the aligned-minus-swapped effect on the same items (`notes/mechanism2_results.md`).

### 3.5 Intervals, tests and verdicts

**Clustering.** Items are not independent: the 4,666 RQ2 items come from 125 games, and items of one game share a room, a task and a trajectory. Each study names a **cluster**, the unit treated as independent: games in ALFWorld and the decision studies, goal pairs in TextCraft, users in τ-bench.

**Game-clustered bootstrap intervals.** A **bootstrap** estimates uncertainty by recomputing a statistic on many resampled datasets. A **cluster bootstrap** resamples whole clusters with replacement: draw 125 games from the 125 with repeats, take all their items, recompute the mean. Repeating this 2,000 times with a fixed seed and taking the 2.5th and 97.5th percentiles of the resampled values gives the 95% **percentile interval** (`wf/analysis/stats.py`, `cluster_bootstrap`). In simulation these intervals covered the true value 93 to 95% of the time, so they are slightly narrow (`notes/testing.md`).

**Sign-flip permutation p-values.** With no effect, each cluster's summed difference $S_g$ is equally likely to be positive or negative. The test flips the sign of every cluster at random, recomputes $|\sum_g \epsilon_g S_g| / N$, and counts how often the result reaches the observed value. With $c$ such permutations out of $n = 1000$,

$$
p = \frac{c + 1}{n + 1},
$$

so the smallest possible value is $1/1001 \approx 0.001$. Many reported p-values sit exactly at that floor.

**Holm correction.** When several tests are read together, the chance of at least one false positive grows. The **Holm** procedure controls it: with $m$ p-values sorted from smallest, $p_{(1)} \le \dots \le p_{(m)}$, the adjusted value of the $k$-th is

$$
\tilde p_{(k)} = \max_{j \le k} \min\left\{1,\ (m - j + 1)\, p_{(j)}\right\}.
$$

Each study fixes its **family** (the set of tests corrected together) in advance: two for RQ2, 18 for families wave A, 50 for TextCraft round 2. A test whose model has not been scored enters the family with p = 1, so the family never shrinks. With two tests at the floor, the smaller becomes $2/1001 = 0.002$: RQ2's "Holm p 0.002".

**Decision rules.** A confirmatory test turns an estimate, an interval and a p-value into a verdict by a rule fixed before data. For RQ2 (`wf/constants.py`), a model's effect is "supported" when $\Delta L$ is at least 0.5 nats, the interval excludes zero and the Holm p is below 0.01, and "supported (strong)" at 1.0 nats or more.

**Equivalence (TOST) margins.** A non-significant difference does not show the absence of an effect. To claim an effect is negligible a study fixes a **margin** $\delta$ in advance and runs two one-sided tests (**TOST**): that the effect is above $-\delta$, and that it is below $+\delta$. Rejecting both at 5% is the same as the 90% interval lying inside the margin:

$$
\left[\hat\theta_{5\%},\ \hat\theta_{95\%}\right] \subset (-\delta,\ \delta).
$$

Margins differ by study: 5 points on the item share for the families; in TextCraft round 2,

$$
\delta_m = \min\left(0.3\,\operatorname{SD}_m,\ 0.4 \text{ nats}\right),
$$

where $\operatorname{SD}_m$ is the standard deviation of model $m$'s item differences. A result that is neither different from zero nor equivalent within the margin is **inconclusive**, which the notes keep distinct from "no effect".

### 3.6 Competence gates

A goal effect in a model that cannot do the task says little. Each study therefore checks **competence**: whether the model tells working actions from failing ones when no goal is in play, usually on scores under the neutral task. The basic measure is the **AUC**, the probability that a randomly chosen working action scores higher than a randomly chosen failing one (ties count half):

$$
\text{AUC} = \Pr(L_{\text{success}} > L_{\text{failure}}) + \tfrac{1}{2}\Pr(L_{\text{success}} = L_{\text{failure}}).
$$

The families rule, also used by later studies, requires a neutral yes/no AUC of at least 0.70, Yes on at most 95% of failures and at least 5% of successes, and a median Yes-plus-No probability of at least 0.5 (`wf/families.py`, `COMPETENCE`). A model that fails is "not interpretable", never counted as showing no effect. Some studies add a **gate** of their own, such as τ-bench authorization's comprehension questions and positive control; when a gate fails, the reading is "inconclusive (gate)".

### 3.7 The empty think block

Qwen3 models have a thinking mode. With thinking off, Qwen3's chat template still writes an empty reasoning block, `<think>\n\n</think>\n\n`, immediately before the scored answer. This block is Qwen's documented thinking-off format and is identical in the aligned and swapped prompts, so it cannot create the effect; but it stretches the scale on which Qwen expresses preferences. Rescored without it on all 4,666 RQ2 items, Qwen3-8B's effect is 3.31 nats [2.95, 3.64] instead of 6.12, while the standardized effect (1.05 against 1.02) and the share of items moved toward success (88.7% against 87.9%) are unchanged (`notes/qwen_effect_audit_followup_metrics.json`, `A_qwen3_8b_all_items`). Plain text without any chat template gives the same halving on 300 items. Two cautions from the notes: the render without the block is Qwen's thinking-on template, which is off the model's usual answer path; and whether Qwen's step-0 concentration disappears in plain text is checked on 22 items only (`notes/qwen_effect_audit_pipeline.md` §7). Quote Qwen's 6.12 nats with this format sensitivity, and compare models on the scale-free measures.

### 3.8 Numerical floors and "numerically fragile"

Scores are computed in bfloat16 on GPUs, and bfloat16 arithmetic is not exactly reproducible across hardware and software. `notes/numerical_sensitivity.md` compared every archived pair of runs that scored the same prompts under different numerics. Within one **environment** (same software stack, GPU type, code path and batch shape) every score repeats bit for bit; the worked example's two runs agree to every digit. Changing any one ingredient adds noise whose size grows with the model's logits: an item's contrast moves with a standard deviation of about 0.45 to 0.63 nats for Qwen3-8B and 0.14 to 0.26 for models with smaller logits, and conditions shift systematically, not only at random. For means over hundreds to thousands of items the resulting **numerical floor**, the standard deviation of an aggregate effect under an environment change, is about 0.04 to 0.08 nats for Qwen-sized logits and 0.01 to 0.03 for the others. A reading is **numerically fragile** when it sits close enough to a decision boundary that an environment change alone could flip it. One reading does flip under an environment change alone (TextCraft round 2's rejection rule R3 for Qwen3-8B). Four readings are annotated as numerically fragile in their notes: that R3 reading, the object-free-goal run's Llama successes, the τ-bench families' Llama-3.2-3B suffix result, and TextCraft round 1's Llama two-candidate equivalence (README, numerical-sensitivity row); TextCraft round 3's interim note flags its Qwen3.5-9B penalty and lift the same way. Every confirmatory verdict that could be recomputed survives.

---

## 4. Cross-benchmark summary figures

The figures below are drawn by `scripts/walkthroughs/overview_figures.py` from committed metrics files only; the script prints every plotted value with its source, and each caption names the files. Colours follow the model in every figure of the four walkthroughs (the mapping is in `scripts/walkthroughs/overview_palette.py`): Qwen3-8B blue, Llama-3.1-8B orange, Mistral-7B aqua, Gemma-2-9B yellow, Llama-3.2-3B magenta, Ministral-3-8B green, Qwen3.5-9B violet, Qwen3.8-27B red, the other Qwen3 sizes lighter and darker blues, and models that fail a competence rule grey. Every row also names its model, so colour is never the only way to tell models apart. Intervals are 95% cluster-bootstrap intervals throughout.

### 4.1 How many failing items move toward success

This is the most direct cross-model comparison: of the determined failures each model scored, what fraction scores higher under the real task than under the swapped task? With no effect the fraction is 0.5. For ALFWorld the readout is the anchored yes/no question, the primary metric of both families waves; for τ-bench it is the first token of the reply, the τ-bench families extension's item share. TextCraft does not appear because no committed metrics file holds an item share on its anchored readouts (its round-1 yes/no question was unanchored, and rounds 2 and 3 decompose the effect by the item's role rather than reporting one share).

![Dot-and-interval chart of the share of determined failures whose score is higher under the real task than under the swapped task. ALFWorld rows (eleven model and format variants) range from 0.65 to 0.91; tau-bench rows are 0.78 for Qwen3-8B, 0.59 for Llama-3.1-8B and 0.82 for Llama-3.2-3B. Every interval lies to the right of the 0.5 no-effect line. Hollow markers are interim results.](figures/overview_share_moved.png)

*Figure 1. Share of failing items moved toward success by the real task. Look at the distance of each dot from the 0.5 line, and at the hollow markers, which come from interim analyses whose final versions are still running. Sources: `notes/families_metrics.json` (wave A and the two reference models, key `primary_metrics/item_share_toward_yes`), `notes/families_waveb_interim_metrics_2026-11-25.json` (wave B interim, including Qwen3-32B rescored in the newer software stack), `notes/tau_families_metrics.json` (`primary/item_share/first`).*

Every competent model scored on ALFWorld moves most failing actions toward Yes: from 0.65 for Qwen3.8-27B to 0.91 for Qwen3.5-9B in plain text, with every interval well clear of 0.5. The four current-generation models of wave B (interim) behave like the older ones, and the Qwen3.5-9B base model moves as many items as its instruct version (0.900 against 0.906 in plain text), so the wave B note finds the instruct model matching its base model and its prediction B3, which expected them to differ, not holding (`notes/families_waveb_results_interim.md`). On τ-bench the share is also above 0.5 for all three interpretable models, but much less so for Llama-3.1-8B (0.59). Two models are absent because they fail competence rules: Llama-3.2-1B in ALFWorld and Mistral-7B on τ-bench. One caution applies to the ALFWorld shares: Option C's step-0 check found Qwen3-8B partly reading the anchored question as asking about task success, so the share may include a task-progress judgement; the two-candidate effect, which asks no question, is positive for every family model (`notes/families_results.md`).

### 4.2 How large the effect is relative to its own spread

The standardized effect $d$ of section 3.4 has committed intervals for every competent ALFWorld model on all three readouts (`notes/qwen_effect_audit_forensic_metrics.json`), and for TextCraft round 1 on the two-candidate and first-token readouts (`notes/textcraft_vs_alfworld_metrics.json`). τ-bench is not drawn: its metrics store Qwen3-8B's first-token item standard deviation (4.96 nats, `notes/tau_dx_metrics.json`) beside the mean of 3.98 nats, which gives $d \approx 0.80$, but no interval for $d$ is committed.

![Three panels of dot-and-interval rows, one per readout: two-candidate, first token, yes/no. ALFWorld rows for eight competent models lie between 0.27 and 1.12; TextCraft round 1 rows are 0.20 and 0.21 for Qwen3-8B and about -0.14 for Llama-3.1-8B. A hollow Qwen3-8B row without the empty think block sits at 1.05 next to 1.02 with it.](figures/overview_standardized.png)

*Figure 2. Standardized goal effect by readout. Look at how the ALFWorld rows cluster (mostly between 0.5 and 1.1) compared with the TextCraft rows (near zero), and at the hollow Qwen3-8B row, which shows that removing the empty think block halves the effect in nats but leaves $d$ where it was. Sources: `notes/qwen_effect_audit_forensic_metrics.json` (`models/<model>/scale/L|F/dz_ci`, `yes_no/scale/dz_ci`), `notes/qwen_effect_audit_followup_metrics.json` (`A_qwen3_8b_all_items/all`), `notes/textcraft_vs_alfworld_metrics.json` (`scales/textcraft_main`). TextCraft's yes/no row is left out because its question was not anchored to the reply.*

On the scale-free measure the models are close to each other. Qwen3-8B, whose 6.12 nats is 3.7 times Llama-3.1-8B's 1.65, is only 1.2 times Llama's on $d$ (1.02 against 0.85), and on yes/no it is below Llama, Mistral and Llama-3.2-3B. Among the five family models Gemma-2-9B has the smallest standardized effect (0.31 to 0.49); the Qwen3 sizes other than 8B are lower still on yes/no (0.27 to 0.46). The competent Qwen3 sizes are flat on the two-candidate readout (0.90, 1.02, 0.96, 0.86 for 4B to 32B) although their effects in nats fall from 7.5 to 2.4. TextCraft round 1 shows the contrast between environments: Qwen's effect is a fifth of its ALFWorld size on $d$ (0.20 against 1.02), and Llama's is reversed (−0.14). The TextCraft walkthrough ([02_textcraft.md](02_textcraft.md)) explains why: crafting the goal item itself is penalized while crafting its ingredients is lifted, and the two roles cancel differently in different models.

### 4.3 What changes the effect, and what does not

Several studies added or changed one element of the prompt and measured how much of the goal effect remained, on the same items and readout. The figure puts them on one scale: 1 means the effect is unchanged, 0 means it is gone, and values above 1 mean the change enlarged it. The lower panel shows a different kind of manipulation: a sentence added to the swapped task only, naming the action's object without making it the goal, measured as the share of the goal effect it reproduces.

![Two panels of dot-and-interval rows, three colours for Qwen3-8B, Llama-3.1-8B and Mistral-7B. Top panel (share remaining): object only described 0.61, 0.92, 1.02; ignore-the-goal note 0.91 and 1.03; deciding facts stated 0.75 and 0.98; rules stated 1.64 and 1.28; hidden layout shown 0.34 and 0.29; tau-bench backend said to ignore the task 1.04 and 1.04 (Qwen). Bottom panel (share reproduced): "Ignore the object." 1.03, 0.31, 0.17; "There is a object somewhere in the room." 0.98 and 0.56.](figures/overview_what_changes.png)

*Figure 3. Share of the goal effect kept (top) or reproduced (bottom) by changes to the prompt, two-candidate readout unless a row says otherwise. Look first at which rows sit near 1 (no help) and which near 0, then at where the three models disagree. Sources: `notes/audit_metrics.json` (`ratios`: object-free goal kept; ignore note and hidden layout as 1 minus the committed share removed), `notes/mech_metrics.json` (`interventions/facts|rules/share_of_effect_removed`, plotted as 1 minus the share removed), `notes/tau_auth_metrics.json` (`decision/per_model/qwen3-8b-fewshot/removed_shares`, first token, plotted the same way), `notes/mechanism2_metrics.json` (`arms_on_sample/<arm>/vs_swapped_L/ratio`; Mistral's "described" row is the battery's described-goal arm on its 1,000-item sample, not the object-free run).*

The pattern, read row by row:

- Telling the model to ignore the goal barely helps: Qwen keeps 91% of its effect, Llama all of it.
- Stating the facts that decide the failure (where the agent is, what it holds, what is open) removes a quarter of Qwen's effect and none of Llama's, so the effect is not mainly a failure to track the state.
- Stating the environment's rules makes the effect larger in both models (164% and 128%).
- Showing the game's full hidden starting layout cuts the effect by about two thirds in both, the largest reduction of any change, without removing it.
- Replacing the object's name with a description keeps most of the effect (61% for Qwen, 92% for Llama, about all of it for Mistral), so the task acts through the object's meaning, not only its name.
- In τ-bench, describing a backend that ignores the task removes none of Qwen's effect, although Qwen answers the comprehension questions about that backend correctly (`notes/tau_auth_results.md`).
- A bare mention of the object reproduces the whole effect for Qwen even when the sentence says to ignore it, but only a third for Llama and a sixth for Mistral; a sentence asserting that the object is in the room reproduces Qwen's whole effect and about half of Llama's.

The mechanism battery read these contrasts together and found no single account for any model (`notes/mechanism2_results.md`); the running studies of section 2.8 are designed to separate the remaining accounts.

Two further changes have committed numbers but cannot be drawn on this scale with intervals. **Thinking mode** (Qwen3-8B reasoning before it answers): after reasoning the effect is 4.01 nats on 1,000 items, against 3.33 without the empty think block on the same items, so no reduction in nats; but the standardized effect falls from 1.10 [0.98, 1.23] to 0.25 [0.18, 0.32] and the share of items moved from 88% to 61%, partly because each task line samples its own reasoning trace, which adds spread. The notes read the question as not determined (`notes/exploratory_results.md` §2; `notes/qwen_effect_audit_followup_metrics.json`, `thinking_probe_items`). **The empty think block** itself: section 3.7.

### 4.4 Step 0 against later steps

Many items are scored before the agent has acted at all (step 0, 514 of the 4,666 RQ2 items), where the history holds only the room description and the task. The goal effect is often much larger there.

![Five small line charts, one per model, of the goal effect in nats against the number of actions in the history (0, 1-3, 4-6, 7 or more), for the two-candidate readout (solid) and the yes/no readout (dashed), with shaded intervals. Qwen3-8B, Mistral-7B and Gemma-2-9B drop sharply after step 0 on both readouts; Llama-3.1-8B drops on yes/no but is flat on two-candidate; Llama-3.2-3B is nearly flat.](figures/overview_step0.png)

*Figure 4. Goal effect by history length, per model. Each panel has its own vertical scale, because nats are not comparable across models; compare the shape of each curve, not its height. Source: `notes/step0_by_history_metrics.json` (post hoc, from stored scores; game-clustered intervals, 1,000 resamples).*

In Qwen3-8B's thinking-off format the two-candidate effect is 11.0 nats at step 0 against 5.2 to 5.8 later. How much of that concentration comes from the empty think block is not settled (section 3.7): without the block, in the thinking-on render, step 0 gives 2.73 nats against 3.3 to 3.6 later (`notes/rq2_results.md`). The concentration is not specific to Qwen, however: on the yes/no readout Gemma-2-9B (4.65 nats at step 0 against 0.56 at seven or more actions), Mistral-7B (5.94 against 1.81) and Llama-3.1-8B (3.34 against 1.79) show it too, though they have no think block; Llama-3.2-3B is flat (0.80 against 0.71). Step 0 matters for decisions because it is where the decision simulation's harm concentrates (section 4.5) and where an agent's first move is checked.

### 4.5 Consequences for planning

Three studies asked whether the goal effect changes what an agent does when a world model is used as a **verifier**: a model asked whether the agent's proposed action will succeed, whose No sends the agent to another candidate. Each compares a verifier shown the agent's real task with a **task-blind** verifier (the same prompt with the task sentence deleted), on the same states.

![Three stacked panels. Stage 1a: extra states where the chosen action fails, real-task minus task-blind verifier: Qwen3-8B +19.5, Llama-3.2-3B +15.4, Gemma-2-9B +8.9, Mistral-7B +3.3, Llama-3.1-8B +1.1 points. Stage 1b: change in share of decisions acting on a failing action: Llama-3.1-8B +0.17, Mistral-7B +0.17, Qwen3-8B +2.29 points. Option C: change in episode success: observation readout -1.2, yes/no accepted-only -4.5, yes/no argmax -0.7 points, all intervals including zero or touching it.](figures/overview_decisions.png)

*Figure 5. Decision-level effects of showing the verifier the agent's task, in percentage points. The three panels measure different things and have their own axes; within each, look at whether the interval excludes zero. Sources: `notes/decisionsim_metrics.json` (`option_c/per_model`, Qwen3-8B and Llama-3.1-8B), `notes/decisionsim_dx_metrics.json` (`frozen_functions_secondary_models`, comparison with the task-blind verifier, the frozen functions applied post hoc to the secondary models), `notes/decisionsim_agent_metrics.json` (`primary/overall` for Llama and Mistral; `models/qwen3-8b/primary/difference` for the Qwen contrast), `notes/optionc_metrics.json` (`runs/<run>/rules/<rule>/contrasts/aware_minus_blind/success`).*

**Stage 1a** gave the verifier fixed candidate menus written by the project's generators, which include failing actions on the goal's object. There the effect clearly changes decisions: Qwen3-8B's real-task verifier makes the chosen action fail at 19.5 points more states than its task-blind verifier, most of it at step 0, and three of the four other models also reach the frozen gate's "worth doing" category; only Llama-3.1-8B shows no reliable change (`notes/decisionsim_results.md`). **Stage 1b** let the agent propose its own candidates. Failing goal-object candidates turned out to be rare (4.3% of states [2.2, 6.8], never at step 0), and the two priority verifiers show no reliable difference; Qwen's contrast of +2.29 points comes mostly from rejecting working alternatives under any task, not from naming the object (`notes/decisionsim_agent_results.md`). **Option C** closed the loop: a Qwen3-8B agent played whole episodes with a Qwen3-8B verifier. Success shows no reliable difference under any of the three variants (−1.2 points [−3.0, 0.0] for the primary observation readout), but the design could detect only losses of about 6 to 11 points and the primary rests on 3 games whose outcome differed (`notes/optionc_results.md`). In all agent studies the agent's prompt included ALFWorld's list of admissible actions, which already filters out most failing actions, so the nulls mean "not detected in this setup".

---

## 5. How to read the per-benchmark documents, and a glossary

### 5.1 How the walkthroughs are organized

Each per-benchmark document follows the same order: what the environment is and why its replies do not depend on the task; how items are built and verified; a set of real sample decisions, each with its rendered prompt (from the repository's own prompt builders), its candidate replies, its stored scores under each task line and what it shows; then the studies run on that benchmark, their designs and results, with figures. Read this overview's section 3 first: the per-benchmark documents use its terms without redefining them, and its section 1.2 is the template every sample decision follows (history, action, task lines, candidates, scores, effect in nats and in probability).

Sample decisions are illustrations, not evidence. A single item can sit far from the mean in either direction (the worked example is above Llama's mean and below Qwen's), and items were chosen to show the range of cases, not at random unless the document says so. The evidence is the aggregate result, quoted with its interval and the note it comes from.

When the documents quote a number, the file named beside it is the authority. Metrics files in `notes/` are byte-identical copies of the analysis runs' `metrics.json`; the run data themselves are kept off GitHub, apart from the public RQ2 release (README, "Reproduce the headline result"). Figures in the walkthroughs share the colour mapping of `scripts/walkthroughs/overview_palette.py`; figure files carry the document's prefix (`overview_`, and the other documents' own prefixes) in `notes/walkthroughs/figures/`.

### 5.2 Glossary

- **Action**: the command whose reply the world model predicts; in ALFWorld a text command, in TextCraft a get or craft command, in τ-bench a JSON tool call.
- **Aligned task line**: the episode's real task, which names the object (or order) the action acts on.
- **Anchored question**: a yes/no question that names the environment's actual failure reply, so that No means "the environment will reply with failure" and not "this does not help the task".
- **AUC**: area under the ROC curve; the probability that a random working action scores above a random failing one, ties counting half. 0.5 is chance, 1.0 perfect.
- **Arm, condition**: one version of a prompt in a paired design, such as aligned, swapped, neutral or task-blind.
- **Base model**: a model before instruction tuning (post-training), here Qwen3.5-9B-Base and, in a running study, Qwen3-8B-Base.
- **Bootstrap, cluster bootstrap**: estimating uncertainty by recomputing a statistic on resampled data; the cluster version resamples whole games, goal pairs or users.
- **Candidate reply**: a fixed reply whose probability is read by teacher forcing; the success twin and the failure reply.
- **Cluster**: the unit treated as independent in intervals and tests: games (ALFWorld), goal pairs (TextCraft), users (τ-bench).
- **Competence, competence gate**: whether a model tells working from failing actions without a goal in play; a model failing the study's rule is "not interpretable".
- **Confirmatory**: a test fixed before its data (question, items, readout, statistic, rule, family) whose verdict licenses a claim.
- **Couple (TextCraft)**: two items with the same history, failure class and role, one acting in each goal's recipe tree, so the history favours neither goal.
- **Decision simulation**: offline studies in which a world model acts as a verifier over candidate actions (Stages 1a and 1b).
- **Determined failure**: an action that fails and whose failure the history alone proves.
- **Determined success**: a working action whose reply reveals nothing new.
- **Empty think block**: the `<think>\n\n</think>\n\n` that Qwen3's thinking-off template places before the answer; it stretches Qwen's scale without changing its standardized effect.
- **Environment (numerical)**: the software stack, GPU type, code path and batch shape a score was computed in; within one environment scores repeat bit for bit.
- **Equivalence, TOST**: showing an effect lies within a prespecified margin by two one-sided tests.
- **Exploratory**: descriptive and hypothesis-generating; p-values unadjusted.
- **Family (Holm)**: the fixed set of tests corrected together.
- **First-token mass**: the total next-token probability of the two candidates' first tokens; low values mean the first-token readout covers little of what the model would say.
- **First-token readout**: the log-odds of the two candidates' first tokens only.
- **Flip**: an item whose two-candidate score crosses zero from failure (swapped) to success (aligned); a No-to-Yes flip is the same for the yes/no answer.
- **Freeze tag**: an annotated git tag at the commit where a design and its analysis code were fixed before scoring.
- **Gate**: a condition that must hold before a result is read (competence, comprehension, positive control, environment check).
- **Goal effect**: the mean aligned-minus-swapped difference in a readout over items; zero for a correct world model.
- **Goal-independent environment**: one whose reply depends only on state and action, not on the agent's goal.
- **History**: the episode so far as shown in the prompt: the first observation (with the task line) and each earlier action with its reply.
- **Holm correction**: a stepwise multiplication of sorted p-values that controls the chance of any false positive in a family.
- **Interim**: a frozen analysis run before every planned model was scored; missing models enter as p = 1, and the final analysis replaces it.
- **Item**: one history plus one next action, with the environment's true reply and the candidates.
- **Item hash**: a 16-character SHA-256 over what a score depends on; a stored score is refused if its hash does not match the item.
- **Item share, share of items moved**: the share of items whose score is strictly higher under the aligned task than under the swapped task; 0.5 means no effect.
- **Lift and suppression**: whether the goal effect comes from the real task raising predicted success above the neutral line (lift) or the wrong task lowering it below (suppression).
- **Log-odds**: the natural log of a ratio of two probabilities, in nats.
- **Margin**: the prespecified bound inside which an effect counts as negligible.
- **Mention sentence**: a sentence naming an object without making it the goal, such as "There is a soapbar somewhere in the room."
- **Nat**: the unit of a natural-log quantity; a one-nat shift multiplies a probability ratio by about 2.72.
- **Neutral task line**: a concrete task that names no object ("explore the room"; in TextCraft "explore the crafting commands"; in τ-bench "The user wants help with their account.").
- **Numerical floor**: the standard deviation of an aggregate effect under a change of numerical environment.
- **Numerically fragile**: a reading close enough to its decision boundary that a change of numerical environment could flip it.
- **Option C**: the closed-loop study in which an agent plays whole episodes with a world-model verifier.
- **Penalty (TextCraft)**: the lower predicted success of crafting the goal item itself when the task names and completes it, against making the same item an ingredient of another goal.
- **Percentile interval**: the 2.5th to 97.5th percentiles of the bootstrap values.
- **Permutation test, sign-flip**: a test that flips the sign of each cluster's differences at random to see how often an effect as large arises by chance.
- **Plain text**: a prompt rendered without any chat template; for Qwen3 the comparison format without the empty think block.
- **Post hoc**: computed after the results were in, often by a one-off script.
- **Preregistered**: filed in the preregistration of 2026-11-05 (RQ1, RQ2, RQ3).
- **Readout**: the rule that turns a model's response to one prompt into one number.
- **Relevance penalty**: the drop in predicted success that any concrete task causes on actions unrelated to it, relative to the neutral line.
- **s (two-candidate score)**: the logistic of the two-candidate log-odds; the success sentence's probability relative to the pair, not the probability of success.
- **Smoke run**: a small real run read only for format, timing and gates, never for effects.
- **Standardized effect, d**: the mean item difference divided by the standard deviation of the item differences.
- **Step 0**: an item scored before the agent has taken any action.
- **Success twin**: the real environment's reply when the same action works, obtained by replaying a plan that makes it work.
- **Swapped task line**: the real task with its object replaced by another object present in the room (in TextCraft the other goal of the pair; in τ-bench another order).
- **Tail (success tail)**: the part of the success sentence after its first token; in ALFWorld it carries 59% of the two-candidate effect.
- **Task-blind**: the aligned prompt with the task sentence deleted; used for verifiers in the decision studies.
- **Teacher forcing**: feeding a fixed reply to the model and reading the probabilities it assigns, without generating.
- **Thinking mode**: Qwen3's mode in which the model writes reasoning before answering.
- **Two-candidate log-odds, L**: the log-probability of the success twin minus that of the failure reply.
- **Verifier**: a world model asked whether a proposed action will succeed, whose answer decides which action the agent takes.
- **Witness (τ-bench)**: a working call that would achieve the aligned task from the item's state, scored as a control.
- **World model**: a model that predicts the environment's reply to an action.
- **Yes/no readout**: the log-odds of "Yes" against "No" after a question about whether the action succeeds.
