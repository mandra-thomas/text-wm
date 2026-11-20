# Interpretability pilot: where does the task enter the prediction? (design, 2026-11-20)

Status: design logged before any interpretability run. No activation, patching, probe or lens result exists at the time of writing. Exploratory, not preregistered with the filed study; once the commit that adds this file is on `interp`, every threshold and prediction below is fixed, and any later change is logged in `deviations.md` with its date and reason. Tooling is in `wf/interp/` (stages in section 8). Models: Qwen3-8B first (thinking off), then Llama-3.1-8B-Instruct, both at the pinned snapshots in `wf.constants.REVISIONS`, bfloat16, HF transformers 4.51.3, as in every scoring run.

## 1. The finding to explain and the competing accounts

On RQ2's 4,666 failing actions whose failure the history already decides, naming the action's object in the task line (the *aligned* task) instead of another object in the room (the *swapped* task) raises the model's log-odds that the action succeeds. In the two-candidate format the log-odds L compares the action's success text with "Nothing happens."; the confirmatory effect is 6.12 nats for Qwen3-8B and 1.65 for Llama. In the yes/no format, where the prompt ends with a fixed question and "Yes" is scored against "No", the effect is 2.80 and 2.21 nats (`rq1v`). The behavioural follow-ups (`exploratory_results.md`) narrowed the explanations but cannot separate the ones that differ only in where the task's influence enters the computation:

- **H1, no state tracking.** The model does not represent the facts that decide the failure (where the agent is, what it holds, what the receptacle contained), so it falls back on the task.
- **H2, belief change.** The model represents those facts, but the task changes them: with the aligned task it comes to believe, for example, that the object is at the receptacle or that the agent is there.
- **H3, relevance at answer time.** The state representation is unchanged; a salience or relevance signal ("this action is about the goal's object") is added to the answer late, near the answer position.
- **H4, wishful or helpful prediction from post-training.** Also late and also leaving the state alone, but the added signal is a generic lean toward success or toward the helpful outcome, not tied to the specific object match.
- **H5, policy and outcome blending.** The prediction reuses the agent's "should I take this action" representation, so whatever makes the action attractive under the task leaks into "will it work".
- **H6, prompt artefacts.** The effect is carried by surface machinery (lexical repetition of the object's name between task and action, or the few-shot format), not by anything about goals or states.

The mechanism diagnostic (`exploratory_results.md`, section 5) already argues against H1 at the level of behaviour: the effect is fully present on items whose state question the model answers correctly, and stating the deciding facts removes little of it. It also showed, for Qwen, a general shift toward "Yes" rather than a specific change in the stated belief. Those are verbal answers; this pilot asks the same questions of the internal computation. The central prediction it tests is:

- under H3 and H4 the goal's influence enters late, at or near the answer position, and leaves the internal representation of the state unchanged;
- under H1 and H2 it changes the representation of the state in the middle layers, at the history tokens that carry that state.

## 2. Readouts

Interpretability needs a readout that a single forward pass produces at one position. Two are used, both at the *answer position*, the last token of the chat-templated prompt (for Qwen3 with thinking off, the final token of the empty `<think></think>` block the template adds; for Llama, the token after the assistant header).

**Yes/no logit difference (primary).** The prompt is the `rq1v` yes/no prompt, byte for byte (`wf.rq1v.arm_messages`, arms `yn_aligned`, `yn_swapped`, `yn_absent`). The readout is the logit of "Yes" minus the logit of "No" at the answer position,

$$
D_{\text{yn}} = z_{\text{Yes}} - z_{\text{No}},
$$

which equals the yes/no log-odds L of `rq1v` exactly, because the softmax normaliser cancels. It is linear in the final normalised residual: if $h_L$ is the residual stream after the last layer, $\gamma$ the final RMSNorm, and $W_U$ the unembedding, then

$$
D_{\text{yn}} = (W_U[\text{Yes}] - W_U[\text{No}])^\top \, \gamma(h_L).
$$

**First-token two-candidate difference (secondary).** The prompt is the observation prompt of the confirmatory run (`wf.rq1.arm_messages`, arms `aligned`, `swapped`, `absent`), and the readout is the logit of "You" minus the logit of "Nothing" at the answer position. On all 4,666 failures the success text starts with "You" and the failure with "Nothing" (`audit_metrics.json`, `first_token`), so this is the first-token term ℓ_first, whose goal effect is 2.52 nats for Qwen and 0.68 for Llama. It carries about 40% of Qwen's full ΔL and 40% of Llama's; the rest lies in the later tokens of the success text and is not studied here (open question 1).

The goal effect on an item is the difference of a readout between its aligned and swapped prompts, written $\Delta D = D(\text{aligned}) - D(\text{swapped})$.

**Reproduction gate.** Stage 0 (section 8) recomputes both readouts on every item. Its mean ΔD must match the logged effects (2.80 yes/no and 2.52 first-token for Qwen; 2.21 and 0.68 for Llama) within 0.15 nats. Across GPU types a single item's effect varies with a standard deviation of 0.55 nats for Qwen (`deviations.md`, "Scores across GPU types"), which averages to under 0.01 nats over 4,666 items, so a larger gap cannot come from the GPU and means the prompt or tokenisation differs from the scored one, and nothing else runs until it is explained.

## 3. Items

**Population.** RQ2's 4,666 determined failures and the 292 determined successes of `rq1x` (`wf.rq1x.rq1x_sets`, the same sets the behavioural runs used), from the replay-verified collections `rq12_transitions_s0..s2`.

**Game split.** The 125 games are split 50/50 into a *train* half and a *test* half by a seeded shuffle (`RNG_SEED`, `interp|split|<seed>`), for three seeds (0, 1, 2). Everything that is estimated (probe weights, the task direction, the layer chosen for ablation, the readout lens) is fit on train games only, and every reported effect of an intervention is measured on test games only. Seed 0 is primary; seeds 1 and 2 are robustness checks.

**Patching set.** Patching is costly, so it runs on 200 items drawn from the test games of seed 0 among those with a clear effect: $\Delta D_{\text{yn}} \ge 1$ nat in stage 0 (for Llama, whose effects are smaller, the same rule). Selection is stratified so that the sample's mix matches the clear-effect items' mix over two factors: the reason for the failure (`determination_reason`) and whether the agent has acted yet (history length 0 or more). Qwen's effect is 11 nats at step 0 and about 5 afterwards, so an unstratified sample would be dominated by step 0. Within strata, items are drawn by a seeded shuffle (`interp|patch|<model>`). Selecting on a large effect makes fractions stable but describes only clear-effect items; a second sample of 100 test items drawn without the effect filter gets the residual-stream map only, to show whether the picture differs.

**Placebo pairs.** For the placebo control, each patching item also gets a second non-goal object: one of the swapped objects that other items of the same game received (so it passed the same feasibility checks for the same task, `pick_swap`), not the item's own swap, not named in its action, not acted on in its history, and not at the action's receptacle at reset. Items without such an object get no placebo; their count is reported.

## 4. Logit lens and readout lens: where do aligned and swapped diverge?

For each layer $\ell$, the residual stream at the answer position, $h_\ell$, is read out as if it were the last layer (the *logit lens*):

$$
D_\ell^{\text{lens}} = (W_U[\text{Yes}] - W_U[\text{No}])^\top \, \gamma(h_\ell).
$$

At the last layer this equals the model's own readout (a unit test checks the identity). The logit lens is known to be unreliable in early and middle layers, where the residual stream is not yet in the unembedding's basis. A full *tuned lens* (one learned affine map per layer, trained to match the final distribution) needs about 17 million parameters per layer for these models and its own training run, so the pilot uses a cheaper version restricted to the readout, a *readout lens*: for each layer, a ridge regression from $h_\ell$ to the final $D_{\text{yn}}$, fit on the neutral-task prompts of train games (penalty chosen by 5-fold cross-validation over games) and applied unchanged to the aligned and swapped prompts of test games. Because it is fit only on neutral prompts, it can show the goal effect only if the goal moves $h_\ell$ along a direction that already predicts the answer on neutral prompts.

**Summary measures.** For each lens, the per-layer goal effect $\Delta D_\ell$ (aligned minus swapped, mean over test items, with game-clustered bootstrap intervals), and the *divergence layer*, the first layer at which the lens effect reaches half the final effect and stays above it.

## 5. Activation patching: which positions and layers carry the effect?

**Pairing positions.** The aligned and swapped prompts are identical except for the object's name in the task line. After tokenisation they share a common prefix of tokens and a common suffix of tokens; between them is the *span*, the tokens of the object's name (and any neighbour whose tokenisation changed). The model is causal, so every activation at a prefix position is identical in the two runs, and patching there does nothing. All patching therefore happens at suffix positions, matched from the end of the sequence, and, when the two spans have the same number of tokens, at the span itself position by position. When the names have different token counts, suffix positions sit at absolute positions that differ by that count, and rotary position encoding sees the shift; results are therefore reported for all items and for the equal-length subset separately.

**Token groups** (assigned from character offsets in the chat-templated text; `wf.interp.prompts.token_groups`):

- *task rest*: the task line after the object's name;
- *history, action's object*: history tokens naming the action's object type (and its instance number);
- *history, swap object*: history tokens naming the swapped object type;
- *history, receptacle*: history tokens naming the action's receptacle instance;
- *history, other*: the remaining history tokens;
- *history, all*: the union of the four history groups;
- *action*: the "Action: …" line;
- *query*: "Next observation:" or the yes/no question;
- *final*: the template tokens after the user turn, including the answer position;
- *last*: the answer position alone;
- *span*: the object's name in the task line (equal-length items only).

**Sites.** The residual stream after each decoder layer (*resid*), the attention block's output (*attn*), and the MLP block's output (*mlp*), at every layer.

**Directions.** *Noising* runs the aligned prompt and overwrites the chosen activations with those of the swapped run; it asks whether those activations are necessary for the effect. *Denoising* runs the swapped prompt and writes in the aligned activations; it asks whether they are sufficient. The *fraction restored* by noising a cell (site, layer, group) is

$$
R = \frac{D(\text{aligned}) - D(\text{aligned with the cell patched from swapped})}{D(\text{aligned}) - D(\text{swapped})},
$$

so 1 means the cell carries the whole effect and 0 that it carries none; the denoising fraction is defined symmetrically. Fractions are aggregated across items as a ratio of sums (numerators summed over the denominators summed), which is stable when some items' effects are small, with game-clustered bootstrap intervals. Every batch of patched runs includes an unpatched row, and fractions use that row's readout, so batch composition cannot change the baseline.

**Grid.** Residual stream: every layer by every group, noising and denoising on the yes/no readout, and noising on the first-token readout. Attention and MLP outputs: every layer by the coarse groups (*task rest*, *history, all*, *action*, *query*, *final*, *last*, *span*), noising, yes/no readout only. Because residual patching of a suffix group at layer $\ell$ replaces the group's state at that depth while later layers can still read the unpatched span, the per-layer residual map shows where the information *resides* at each depth (as in causal tracing), and the attention and MLP maps show which blocks *write* it.

**Heads.** For the six layers with the largest attention-output fraction at the *last* or *final* group, each query head's contribution is patched separately (the head's slice of the input to the attention output projection) at the *final* group, noising, yes/no. For the top heads this gives a total effect; their direct effect on the readout is read by projecting the head's output at the answer position through the final norm (scale frozen at the clean run) onto the readout direction. Attention weights of the top heads from the answer position and the action tokens to the task-line span are recorded with eager attention on 50 items. Full path patching (isolating a head's effect through one downstream path) is not in the pilot; it is added only if the head-level total and direct effects disagree in a way that matters for the hypotheses.

**Summary measures fixed before data.** These are the quantities the verdicts in section 9 use; the full maps are descriptive.

- *History share*: the largest noising fraction of the *history, all* group over layers, on the yes/no readout.
- *Arrival layer*: the first layer at which noising the *final* group restores at least half of the effect, as a share of the number of layers (0 = first, 1 = last).
- *Action share*: the largest noising fraction of the *action* group over layers.
- *Early-span share*: for equal-length items, the fraction restored by noising the *span* at the end of the first third of layers, minus the fraction restored by noising the *span* at the end of the second third. A large drop means the information leaves the object's name in the first third of the network.

## 6. Linear probes: does the task change the represented state?

**Facts probed.** Each is computed from the history text alone with `wf.data.transitions.history_facts`, the same parser that decides which failures are determined. A fact is left unlabelled where the history does not establish it.

- *At receptacle*: the agent is at the action's receptacle.
- *Object listed*: the last observation of the action's receptacle listed the action's object (labelled only when that receptacle's contents were seen).
- *Holding anything*: the agent holds an object.
- *Holding the object*: the agent holds the action's object.
- *Receptacle closed*: the action's receptacle was last seen closed (labelled only when its state was seen).

**Training data.** Neutral-task prompts ("Your task is to: explore the room.") in the yes/no format, built for every collected item of the train games (all 12,987 collected items are eligible, including actions that succeed and items outside RQ2's population, so that every fact has both values often enough). The true outcome correlates with the deciding fact, and a probe at the answer position could learn the answer instead of the fact. The training set is therefore balanced by subsampling so that each fact's two values occur equally often among actions that succeed and among actions that fail; the probe's cosine with the readout direction is reported.

**Positions and layers.** Three positions: the answer position; the last token of the action line; and the action's receptacle token in the action line. Ten layers: every fourth layer and the last (for Qwen3-8B 0, 4, …, 32, 35; for Llama 0, 4, …, 28, 31). The probe is a logistic regression on the standardised residual (sklearn, L2 penalty chosen by 5-fold cross-validation over train games). A probe is *usable* if its held-out AUC on neutral prompts of test games is at least 0.75.

**Belief shift.** For each usable probe, on the test items of RQ2's population, the probe's logit is read under the aligned and the swapped prompt. Its mean difference is put in units of the probe's own separation on neutral test prompts (the mean logit of the positive class minus that of the negative class), and signed so that positive is the value that would make the action succeed (at the receptacle, object listed, not holding anything for a take, holding the object for a move, receptacle open). The standardised shift is

$$
\delta_f = \frac{\overline{\text{logit}_f(\text{aligned}) - \text{logit}_f(\text{swapped})}}{\overline{\text{logit}_f \mid y=1} - \overline{\text{logit}_f \mid y=0}} .
$$

The fact used for each item is its deciding fact (from `determination_reason`); the other facts serve as controls.

**Fixed thresholds.** *No belief change*: the 95% interval of $\delta_f$ lies within ±0.15 at every usable layer and position, for the deciding facts. *Belief change*: $\delta_f \ge 0.25$ with an interval excluding zero at some middle-third layer at the action or receptacle position. Shifts in between are reported as inconclusive. A shift seen only at the answer position, with none at the action and receptacle positions, is read as the answer leaking into the probe, not as a belief change.

**Probe failure as evidence for H1.** If no probe for a deciding fact reaches an AUC of 0.75 at any layer and position, the state is not linearly represented where looked, which is what H1 predicts. The per-item probe confidence on neutral prompts is also correlated (Spearman) with the item's goal effect; H1 predicts a negative correlation (a larger effect where the state is represented less clearly), of at most −0.2.

## 7. A task direction at the answer position

**Estimate.** At each layer, the mean over train items of the residual at the answer position under the aligned prompt minus that under the swapped prompt, normalised to unit length:

$$
u_\ell = \frac{\overline{h_\ell(\text{aligned}) - h_\ell(\text{swapped})}}{\big\lVert \overline{h_\ell(\text{aligned}) - h_\ell(\text{swapped})} \big\rVert}.
$$

**Intervention.** At the chosen layer, at the *final* group's positions, the component along $u_\ell$ is set to the midpoint $c_\ell$ of the train items' mean projections under the two prompts (a mean ablation along one direction, which stays in distribution better than setting it to zero):

$$
h \leftarrow h - \left(u_\ell^\top h - c_\ell\right) u_\ell .
$$

The layer is chosen on train items as the one whose ablation removes the most of the effect; a second variant ablates at that layer and every later one. Both are applied to the aligned, swapped and neutral prompts of the test items and of the test-game determined successes.

**Readouts.** The share of the goal effect removed on test items; discrimination, the AUC of $D_{\text{yn}}$ for determined successes against determined failures under the neutral task and under the aligned task, with and without the ablation; and the mean shift of $D_{\text{yn}}$ on neutral prompts (collateral damage).

**Controls and transfers.**

- *Random directions*: 20 random unit directions at the same layer and positions, with the same mean-ablation rule, on 300 test items; the removed share must exceed the 95th percentile of the random ones.
- *Placebo direction*: the same estimate from placebo pairs (two non-goal objects).
- *Described goals*: the direction estimated on named-object prompts, applied to the `rq1x` described-goal prompts ("cool some thick drinking vessel with a handle for hot drinks …"), where 61% (Qwen) and 92% (Llama) of the effect survives.
- *Across formats*: the yes/no direction applied to the first-token readout's prompts, and the reverse.
- *Policy direction*: the same estimate on the agent's prompt (`wf.mech.policy_messages`, aligned against swapped, at its answer position), applied to the world-model prompt.
- *Probe directions*: the cosine of $u_\ell$ with each usable probe's weight vector at the same layer and position.
- *Later, other environments*: the ALFWorld direction applied to TextCraft and τ-bench items (branches `textcraft`, `taubench`) once those are scored and show a goal effect. This is planned, not part of this pilot.

**Fixed thresholds.** *A single late direction carries the effect*: the ablation removes at least 60% of the effect on test items, the random directions' 95th percentile is below 10%, and the neutral-task AUC falls by less than 0.02. *It does not*: less than 30% removed. *Transfer*: an applied direction removes at least 50% of the effect where it is applied; *no transfer*: less than 20%.

## 8. Controls common to all methods, and stages

**Controls.**

- *Placebo patching*: the same grids between two non-goal objects (swapped against placebo) on 100 of the patching items. The goal effect there should be near zero; the placebo's per-cell absolute change in the readout gives the noise floor, and a cell counts as carrying the effect only if its restored effect exceeds the placebo's 95th percentile for that cell.
- *Position-matched control*: for the *action* and each history group, the same number of tokens drawn at random from the other suffix groups (seeded), at nine layers; a group matters only if it beats its matched random set.
- *Identical-prefix check*: patching any prefix position must change nothing; every job checks this on its first item and stops if not.
- *Seeds*: three game splits for everything estimated; numbers are reported for seed 0 and as a range over seeds.
- *Step-0 and later items separately*, for every summary measure, because Qwen's effect is twice as large at step 0.

**Stages** (`wf/interp/run.py`; each saves one record per item as it goes and resumes after a timeout):

0. *Baseline*: both readouts for every item under the aligned, swapped and neutral prompts, with the answer-position residual at every layer and the probe positions at the probe layers saved (fp16). Gives the reproduction gate, the item selection, both lenses and the direction estimates.
1. *Probe data*: neutral yes/no prompts for the remaining collected items of train and test games, at the probe positions and layers.
2. *Patching*: the grids of section 5 on the 200 selected items, the 100 unselected items, the placebo pairs and the position-matched controls.
3. *Heads*: section 5's head patching and the attention weights.
4. *Ablation*: section 7 on the test items.

Probes, lenses, direction estimates and every statistic are CPU analyses of the saved records.

## 9. Predictions and outcome table

The table says, for each hypothesis, what each outcome would be if that hypothesis were the main mechanism. "Late" and "middle" refer to thirds of the network's depth. The thresholds are those of sections 5–7.

| Outcome | H1 no state tracking | H2 belief change | H3 relevance at answer time | H4 wishful or helpful | H5 policy blending | H6 prompt artefact |
|---|---|---|---|---|---|---|
| Probes for deciding facts usable (held-out AUC at least 0.75) | No | Yes | Yes | Yes | Yes | Yes |
| Standardised belief shift at action or receptacle positions | None to define (probes fail) | At least 0.25 in the success direction, middle layers | Within ±0.15 | Within ±0.15 | Within ±0.15 | Within ±0.15 |
| History share of the effect (noising) | At least 0.3 | At least 0.3, rising in middle layers | Below 0.15 | Below 0.15 | Below 0.15 | Below 0.15 |
| Arrival layer of the effect at the final positions | Middle | Middle or late, after the history | Late (at least 0.5 of depth) | Late | Late | Early or middle |
| Action share | Any | Any | High (the match is made at the action) | Low to moderate | High | High, and early |
| Early-span share (information leaves the name in the first third) | Any | No | No | No | No | Yes |
| One answer-position direction removes the effect | Under 30% | Under 30%, or removes it while lowering the neutral AUC | At least 60%, neutral AUC kept | At least 60%, neutral AUC kept | At least 60% | At least 60% |
| Direction cosine with deciding-fact probes | Not defined | High (at least 0.3) | Low (below 0.1) | Low | Low | Low |
| Direction moves determined successes not on the goal object | Not applicable | Little | Little | As much as failures | Little | Little |
| Direction transfers to described-goal prompts | Not applicable | Yes | Yes | Yes | Yes | No |
| Policy-prompt direction removes the effect | No | No | No | No | Yes (at least 50%) | No |
| Correlation of probe confidence with the item's effect | At most −0.2 | Near zero | Near zero | Near zero | Near zero | Near zero |

**How a verdict is read.** H3 and H4 against H1 and H2 is the central contrast, and it rests on three rows: the belief shift, the history share and the direction's effect on the neutral AUC. "Late entry, state unchanged" (H3 or H4) needs all three of: no belief change, a history share below 0.15, and a single direction removing at least 60% with the neutral AUC kept. "State changed in the middle layers" (H1 or H2) needs a history share of at least 0.3 together with either a belief change (H2) or unusable probes (H1). Any other combination is reported as mixed, with the rows that disagree. H3 and H4 are separated by whether the direction moves working actions that do not involve the goal's object (H4 predicts it does, as much as failures); H5 by the policy-direction transfer; H6 by the early-span share together with no transfer to described goals. More than one hypothesis can hold in part, and the table is read for the main mechanism, not an exclusive one.

**What the behavioural results already lead one to expect**, stated so that a confirmation is not over-read: the mechanism diagnostic, the object-free goals and the 2×2 point toward H3 for Qwen and a mix with a goal-role part for Llama. An outcome that only repeats this is weak support, because the same experimenter chose the items and the methods; the interesting outcomes are the ones that could contradict it (a belief shift at the history tokens, or a direction that does not transfer to described goals).

## 10. Compute and storage

**Assumptions.** Prompts of about 700 tokens on average (system and three examples about 330, the query about 150 to 550; stage 0 records the real lengths and the estimate is redone before stage 2). A forward pass of an 8B model costs about twice its parameter count in floating-point operations per token, about 11.5 TFLOP per prompt; at 40% of an A100's bfloat16 peak that is about 0.1 s per prompt, and about 0.15 s on an L40S. Patched runs of one item are batched (same prompt, one patch per row, an unpatched row in each batch). Logits are computed for the last position only.

| Stage | Forward passes (Qwen3-8B) | A100 hours | L40S hours |
|---|---|---|---|
| 0 Baseline (4,958 items, 3 tasks, 2 formats, plus placebo prompts) | 34,400 | 1.0 | 1.4 |
| 1 Probe data (about 8,000 more items, neutral, 2 formats) | 16,000 | 0.4 | 0.7 |
| 2 Patching, 200 selected items (1,692 cells each) | 338,000 | 9.4 | 14.1 |
| 2 Patching, 100 unselected items, 100 placebo pairs, position-matched controls | 85,000 | 2.4 | 3.5 |
| 3 Heads (100 items) and attention weights (50 items) | 20,000 | 0.6 | 0.8 |
| 4 Ablation (test items, random directions, placebo, transfers, three seeds) | 97,000 | 2.7 | 4.0 |
| Total | about 590,000 | about 16 | about 25 |

Llama-3.1-8B has 32 layers instead of 36, so its patching costs about 11% less: about 15 A100 hours. With 30% for overheads and reruns, both models need about 40 A100 hours (about 60 L40S hours). Before stage 2 runs on all 200 items, it runs on 10 and the table is redone from the measured speed. Stage 2 dominates; if the measured speed is much slower than assumed, the first cut is the attention and MLP grids at the *history, other* and *task rest* groups. A shared-prefix key-value cache (the system turn and examples are the same in every prompt) would roughly halve every stage, and is an optimisation to add only if needed.

**Storage** (fp16, only the positions and layers needed):

- answer-position residual, every layer, per prompt: 36 × 4,096 × 2 bytes, about 0.3 MB; for stage 0's 34,400 prompts, 10.1 GB;
- probe positions (two besides the answer position), ten layers, per prompt: 0.16 MB; for stage 0's prompts 5.5 GB, and for stage 1's (all three positions) 3.9 GB;
- patching, head and ablation results are per-item numbers (one value per cell), under 100 MB in all; source activations for patching live in GPU memory for one item at a time and are never written.

That is about 20 GB for Qwen and 18 GB for Llama. Both fit in ICE scratch's roughly 50 GB free only with little margin, so the models run one after the other. After each model's CPU analyses have written their per-item summaries (lens values, probe logits, projections on every direction), its raw activation files are deleted, with their SHA-256 recorded, to keep at least 25 GB free; they can be regenerated deterministically on the same GPU type. Every job checks free space before writing and stops below 5 GB.

## 11. Limits known in advance

- The first-token readout carries about 40% of the two-candidate effect; the rest, in the later tokens of the success text, is outside this pilot.
- Patching shows which activations are necessary or sufficient for the readout difference, not what they mean. Probes show what is linearly decodable, not what the model uses. Redundant or self-repairing circuits can make a patch look small when the cell matters.
- Suffix positions of prompts whose object names differ in token count sit at shifted absolute positions; the equal-length subset is the check.
- The patching sample is selected on a clear effect, so its maps describe those items; the unselected sample is the check.
- The many cells of the patching maps are descriptive. Only the summary measures and thresholds above lead to verdicts, and no correction for multiple comparisons is made across them, as in the other exploratory analyses.
- All numbers will come from one GPU type per model and job; bfloat16 forward passes differ across GPU types (`deviations.md`, "Scores across GPU types"), and the aligned and swapped runs of one item always run in the same job.

## 12. Open questions

1. Should the full two-candidate L (teacher-forcing the success text) get a patching map on a small subset, to check that the first-token picture holds for the remaining 60% of the effect?
2. H4 would be tested more directly by a base-versus-instruct comparison (Qwen3-8B-Base, Llama-3.1-8B), which needs two new pinned snapshots and about 1 GPU hour each for stage 0 and the lens. Worth adding?
3. Attribution patching (a gradient approximation of every cell from one forward and backward pass) could replace most of stage 2's exhaustive grid. The exact grid is kept for the pilot because its numbers are exact; should it be screened first instead, if speed turns out to be the constraint?
4. Whether nnsight is used at all: the tooling uses plain PyTorch forward hooks, which need no extra package on ICE.
