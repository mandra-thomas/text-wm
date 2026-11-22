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

## Amendments after review 1, before any run (2026-11-20)

Status: an outside review of the design and the tooling (its scripts ran the tooling against the real Qwen3-8B and Llama-3.1 tokenizers and put 20 bugs back into the code one at a time) found problems in both. Everything below was decided after that review and before any activation, patching, probe or lens result existed; no interpretability job has run. Where an amendment contradicts sections 1 to 12, the amendment holds. `deviations.md` records the change ("Interpretability pilot: amendments after review 1").

### A1. A late belief change can hide at the answer position

A probe read at the answer position sees the answer itself: if the aligned task moves the readout, any probe whose weights overlap the readout direction moves with it, and a genuine belief change made late, at the answer position, would look the same as the answer leaking into the probe. Section 6's rule ("a shift seen only at the answer position is the answer leaking") therefore throws away exactly the late belief change that H2 allows. Two corrections are made, and both are reported:

- *Projected probes.* At the answer position, probes are trained and read after removing two directions from the residual: the readout direction $\gamma \odot (W_U[\text{Yes}] - W_U[\text{No}])$ (the direction along which the logit lens moves the readout) and the neutral success direction of A3. A probe that still separates the fact after this projection is not reading the answer.
- *Residual shift.* For each item the part of the probe's shift that the readout's own change predicts is subtracted. With $\beta_f$ the least-squares slope of the probe logit $z_f$ on the readout $D$ across neutral prompts of the held-out games, the residual shift is

$$
\delta_f^{\text{res}} = \frac{\overline{\left(z_f(\text{aligned}) - z_f(\text{swapped})\right) - \beta_f \left(D(\text{aligned}) - D(\text{swapped})\right)}}{\overline{z_f \mid y=1} - \overline{z_f \mid y=0}} .
$$

A belief change is read from the action and receptacle positions as before, and from the answer position only through the projected probe and the residual shift.

### A2. History share is not belief, and step 0 is its own stratum

A large history share says that the history tokens' activations differ between the aligned and the swapped run and that the difference matters; it does not say that what differs is a belief about the state. The history tokens that name the action's object (`hist_obj`) are the place where a goal could plausibly change the represented state of that object, so they are reported separately from the rest of the history, next to a position-matched control (A7).

At step 0 the agent has not acted, so no history token follows the task line: the history before the task line is a shared prefix, identical in both runs. The history share there is 0 by construction, not by measurement. Every verdict of section 9 is therefore applied per stratum, step 0 and later steps, and at step 0 the history-share row is undefined and the verdict rests on the other rows. Qwen's effect is about 11 nats at step 0 and about 5 later, so a pooled history share would mostly describe the step-0 items, where it cannot be anything but 0.

### A3. The direction test

The task direction $u_\ell$ of section 7 is the mean aligned-minus-swapped difference at the answer position. On its own it cannot separate "the goal adds a success signal" from "the goal adds something that happens to be read as success". Added:

- *Cosines.* The cosine of $u_\ell$ with the readout direction above, and with the neutral success direction $s_\ell$, the unit mean residual of determined successes minus that of determined failures under the neutral task (train fold), are reported at every ablation layer, together with the cosine between the two folds' estimates of $u_\ell$ (stability).
- *Ablation after projecting out success.* The task direction with its success component removed, $u_\ell^{\perp} = (u_\ell - (u_\ell^\top s_\ell) s_\ell) / \lVert \cdot \rVert$, is mean-ablated in the same way as $u_\ell$.
- *Mention-contrast direction.* The same estimate from the swapped prompt with a sentence naming the action's object added (rq1's `swapped_plus`) against the swapped prompt: the direction of merely naming the object, not as the goal. It is a control: if it removes as much as the task direction, the task direction is about the name, not the goal role.
- *Discrimination in nats.* The neutral-task AUC of determined successes against failures is 0.99 for Qwen, at its ceiling, so "the AUC falls by less than 0.02" cannot detect damage. Discrimination is instead the success-versus-failure gap of the readout under the neutral task,

$$
G = \overline{D \mid \text{success}} - \overline{D \mid \text{failure}},
$$

and it is *kept* when the ablated gap is at least 90% of the unablated one. This replaces the AUC criteria of sections 7 and 9; the AUC is still reported.

### A4. Probe failure is not evidence for H1; a hidden-state belief probe

A probe that does not reach an AUC of 0.75 shows that the fact is not linearly decodable at the positions and layers looked at. It could be represented nonlinearly, elsewhere, or only in combination; that is not evidence that the model does not track it. Section 6's paragraph "Probe failure as evidence for H1" and the corresponding cells of section 9 are withdrawn: unusable probes make the belief rows *inconclusive*, and H1 can be supported only by the behavioural evidence it already has against it, or by usable probes whose accuracy on the item's own deciding fact is near chance where the effect is large (the correlation row, kept).

Added: a *hidden-state belief probe* for "the action's object is at the action's receptacle", labelled from the scene at reset (`scene_locations`, `wf.interp.probes.hidden_labels`; unlabelled where the history acts on the object, since it may have moved). This is the belief H2 names most directly ("with the aligned task it comes to believe that the object is at the receptacle"), and its label does not come from the history parser, so it does not share that parser's blind spots. It is trained on neutral prompts (balanced within true outcome) and read with the belief shift and the residual shift of A1.

### A5. H4 restated and tested directly

H4 is restated as: *the object match triggers a generic success feature*, the same feature that separates working from failing actions under the neutral task. Its predictions, in place of section 9's H4 cells that only repeated H3's:

- the task direction has a cosine of at least 0.3 with the success direction (H3: below 0.1);
- ablating $u_\ell^{\perp}$ removes less than half of what ablating $u_\ell$ removes (H3: at least 80% of it);
- *steering*: adding to a swapped prompt, along $s_\ell$, the mean aligned-minus-swapped projection on $s_\ell$ raises its readout by at least half of the goal effect (H3: under 20%);
- the direction moves determined successes as much as failures (kept from section 9).

### A6. Thresholds

- *Belief-shift bound as a share of the effect.* A shift in the probe's own units says nothing about whether it matters for the answer. It is restated as the share of the readout effect the shift could carry. With $G_f$ the neutral-prompt readout gap between items whose deciding fact has its success value and items where it does not (train fold), a standardised shift $\delta_f$ could move the readout by about $\delta_f G_f$, so the share it could carry is

$$
\kappa_f = \frac{\delta_f \, G_f}{\Delta D} .
$$

*No belief change*: the upper bound of $\kappa_f$ is below 0.2 at every usable layer and position for the deciding facts. *Belief change*: the lower bound of $\kappa_f$ is at least 0.5 at some middle-third layer at the action or receptacle position (or at the answer position through A1's projected probe or residual shift). Between: inconclusive. This replaces the ±0.15 and 0.25 bounds of section 6.
- *Multiple testing.* Claims about single cells (a layer of a group, a layer and position of a probe) use intervals Bonferroni-corrected over the layers of that group (and over positions, for probes). Summary measures that are maxima over layers (the history share, the action share, the object-mention share) get a game-clustered bootstrap interval in which the maximising layer is re-chosen in every resample, so the interval includes the choice. The history share is *low* when that interval's upper bound is below 0.15 and *high* when its lower bound is at least 0.3. Section 11's "no correction across summary measures" stands for the summary measures themselves, which are few and fixed.
- *Variances before thresholds.* The bands above (0.15 and 0.3 for the history share, 0.6 and 0.3 for removal, 0.2 and 0.5 for $\kappa$, 90% for the gap) were set without any estimate of the item-to-item variance of these quantities. They are provisional. The minimal pilot (section M) runs its patching on the games of one fold only (the calibration fold); from its variances the bands are fixed, in a second dated amendment, before any patching on the other fold's games and before the full pilot. The minimal pilot's own readouts are calibration results, read against the provisional bands and not reported as verdicts.

### A7. Controls

- *Random directions* pass by construction: in a 4,096-dimensional residual a random unit direction has an expected squared cosine of 1/4,096 with the task direction, so its ablation removes almost nothing whatever the mechanism. They are kept as a sanity check of the editing code (two per layer at two layers) and removed from every pass criterion of section 7.
- *Placebo floor on the effect's scale.* The placebo pairs' per-cell change is in nats, the goal pairs' fractions are not, so the two were not comparable. Each cell's goal-pair removed effect in nats (clean minus patched, mean over items) is compared with the placebo pairs' mean absolute change in nats at the same cell; a cell carries the effect only if the lower bound of the former exceeds the upper bound of the latter (both Bonferroni over layers). Fractions are reported beside them.
- *Position-matched sets from the same region.* A matched control drawn from all other suffix groups mixes regions at different distances from the answer. Each matched set is drawn from the group's own region: for `hist_obj`, as many history tokens naming neither object (`hist_obj_matched`, seeded by item).
- *More pairs.* Besides aligned against swapped and the placebo pairs: aligned against neutral (the whole task line differs), the counterfactual-goal pairs of `rq1v` (the real task's destination replaced, so neither task in the pair is the episode's real one; `wf.interp.prompts` goals `cf_aligned`, `cf_swapped`) and the mention pairs (`swapped_plus` against `swapped`). In the full pilot each gets the residual map on 50 items; in the minimal pilot the mention pair gives the mention-contrast direction and stage 0 records the neutral prompts.

### A8. What a "mechanism" claim would need, and what the pilot aims at

A *localization* claim says where, in layers, positions and blocks, the goal's influence on the readout resides and is necessary or sufficient, and whether one direction at the answer position carries it. A *mechanism* claim would need, in addition: named components (heads, MLP blocks or features) shown necessary and sufficient through specific paths (path patching or an equivalent), not only in total; an account of what each computes, checked on inputs built to vary only that quantity; the same components doing the same job across models, prompt formats and described (object-free) goals; and at least one prediction derived from the account after the fact and confirmed on new items. The pilot aims at localization. Its write-up uses "mechanism" only for claims that meet the second standard, and none of the minimal pilot's outputs can.

### A9. Open questions of section 12, decided

1. Full two-candidate L: yes, a residual-stream map by the coarse groups on 50 items, teacher-forcing the success text (full pilot, not the minimal one).
2. Qwen3-8B-Base: yes, stage 0 only, as a behavioural comparison (readouts on the same prompts, no patching). Caveat: the prompts go through the instruct model's chat template, which the base model was not trained on, so a difference can come from the template as well as from post-training. The snapshot is pinned in `wf.constants.REVISIONS` (`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, the `main` revision reported by the Hugging Face API on 2026-11-20).
3. Attribution patching: only to screen heads; every head that is reported is confirmed by exact patching.
4. Splits: two-fold cross-fitting over games with one seed (seed 0): everything estimated on one fold is applied to the other, and both folds' items are reported. The three 50/50 seeds of section 3 are dropped.
5. nnsight: unchanged (plain hooks).

### A10. Tooling corrections (no design change intended)

- *Spans by character range.* The aligned/swapped token pairing matched token ids from both ends. Object names that share an ending (glassbottle and winebottle split as "glass"+"b"+"ottle" and "wine"+"b"+"ottle") had the shared sub-word tokens counted in the suffix and grouped with the rest of the task line; the review found 28 of 2,812 ALFWorld name pairs affected with the real tokenizers. The differing region is now found on characters and widened to whole words, and no token overlapping it is counted as shared (`prompts.char_diff`, `prompts.pair`); with the fix, 0 of 2,812.
- *One clean baseline per batch.* Every patching and editing batch now has exactly the batch size (the last one is padded with unpatched rows), the source run is a batch of the same size, and each cell's patched readout is paired with the unpatched row of its own batch (`cell_clean`, `edit_clean`). Before, fractions used the first batch's clean row for every cell, so on a GPU, where bfloat16 results can depend on the batch's shape, a cell's fraction could mix two batches.
- *Readouts in float32.* The readout is computed in float32 from the last layer's output through the final RMSNorm and the two unembedding rows (`Parts.readout`), not from the model's bfloat16 logits, whose spacing near a logit of 10 is about 0.06 nats. The logit lens uses the same computation.
- *Noise-based prefix tolerance.* The identical-prefix check had a fixed tolerance of 0.001 nats, which a GPU's bfloat16 noise could exceed. Each job now measures the batch noise (the spread of one prompt's readout across batch rows and sizes 1, 2 and 16; `run.batch_noise`) and uses the larger of 0.01 nats and five times that noise. Stage 0 records the noise.
- *Edits where the direction was estimated.* The task direction is estimated at the answer position, but the ablation was applied at every position of the template's tail. Edits now apply at the answer position only by default (`positions="last"`); the tail is allowed only for directions estimated per position.
- *Reproduction gate per item.* The gate compared only means. It now also requires a median absolute per-prompt difference from the logged rq1v score of at most 0.35 nats, correlations with the logged scores of at least 0.95 (prompts) and 0.90 (item effects), and at least 99% of the logged items reproduced, besides the mean effect on the same items within 0.15 nats (`analysis.reproduction_gate`). The per-item bounds allow for a GPU-type change, which moves one item's effect with an SD of 0.55 nats for Qwen.
- *Missing sign.* The belief shift treated a fact with no success value (holding something, for an action other than take) as "toward failure"; such items are now left out.
- *Streaming loads.* `ActStore.load` held every chunk in memory; it now fills a preallocated array one chunk at a time and can keep a selection (one layer, or the probe layers), so a 9 GB store is read in one pass.
- *Tests.* Tests now kill the four mutants that survived the review (probe positions shifted, edits at every position, the baseline from the wrong batch, the probe direction left in standardised units) and make the source-position mutant fail on values rather than on an index error. The review's 20 mutants and 12 mutants of the fixes are in `scripts/mutation_check.py`.

### M. The minimal pilot

The full pilot of sections 2 to 10 needs about 40 A100-hours. Before it, a minimal pilot on one model and one format calibrates the thresholds (A6), checks the tooling against the real model, and gives a first localization. It is wired as four runs (`wf/runs.py`, configs in `wf/config/interp_pilot_*.yaml`, `wf/interp/pilot.py`; job script `pace/interp_pilot.sbatch`, one A100 or L40S, one model).

| Run | What it does | Forward rows (Qwen3-8B) | A100 hours |
|---|---|---|---|
| `interp_pilot_stage0` | Yes/no readouts and the answer-position residual at every layer (plus the probe positions at the probe layers) for RQ2's 4,666 determined failures under the aligned, swapped, neutral and swapped-plus-mention tasks and the 292 determined successes under the first three; batch noise; the reproduction gate. Nothing else runs unless the gate passes. | about 19,500 | about 1 |
| `interp_pilot_patch` | Residual stream at every layer by the pilot's groups (span, task rest, object-mention history tokens and their matched control, whole history, action, query, final, last), noising and denoising, on 100 items with an effect of at least 1 nat drawn from the calibration fold's games (stratified by failure reason and step 0), and on 50 placebo pairs among them. | about 110,000 | about 3 |
| `interp_pilot_ablate` | Directions estimated at the answer position on one fold's games, applied at the answer position to the other fold's items (two-fold cross-fitting), at layers 4, 8, …, 32 and 35: the task direction, the task direction without its success component, the mention-contrast direction, the task direction from that layer on, steering along the success direction, and two random directions at two layers. 75 failures and 50 successes per fold, aligned, swapped and neutral prompts. | about 48,000 | about 1.5 |
| `interp_pilot_analysis` | CPU: the readouts below per stratum (step 0, later, all), and the hidden-state probe (descriptive). | none | none |

At the design's assumed 0.1 s per forward row on an A100 (about 0.15 s on an L40S), that is about 5.5 A100-hours, or about 8 L40S-hours; the job script asks for 8 hours and every run resumes per item. Stage 0 stores about 9 GB of float16 activations. A 10-item timing run of `interp_pilot_patch` comes first, and the estimate is redone from it.

**Readouts, per stratum.** From patching (noising and denoising): the history share with its bootstrap interval for the maximum over layers (undefined at step 0); the object-mention share against its matched control; the action share; the arrival layer of the final group; the early-span share on equal-length items; every group's per-layer map against the placebo floor in nats. From ablation, with the layer for each fold's items chosen on the other fold: the share of the goal effect removed by the task direction, by the task direction without its success component, by the mention direction, and by the task direction from the chosen layer on; the neutral success-versus-failure gap with and without the ablation; the collateral shift on neutral prompts; the share removed on determined successes; the steering gain along the success direction; the cosines of A3.

**What the minimal pilot can conclude.** Whether the tooling reproduces the scored effect item by item on the real model; how much of the yes/no effect the residual stream at the answer position, the final template tokens, the action line, the task line and (after step 0) the history carry at each depth, separately at step 0 and later; whether a single answer-position direction removes the effect without collapsing the neutral success-versus-failure gap; whether that direction is mostly the model's own success direction (the restated H4) or orthogonal to it (H3), and whether it is about the goal role or merely the name (mention control); and the variances needed to fix the thresholds of A6.

**What it cannot conclude.** Whether the represented state changes (H1, H2): the history-fact probes need the neutral prompts of all collected items (stage 1), which it does not run, so its hidden-state probe is descriptive only; anything about heads or which blocks write the effect (attention and MLP grids, stage 3); the policy-blending account (H5, no policy-prompt direction); transfer to described goals or across formats; the first-token or full-reply readouts; anything about Llama or about the base model; and any verdict at all, since its thresholds are provisional and its patching sample is the calibration fold's clear-effect items. It is one model, one prompt format, one GPU type and one split, and it localizes; it does not establish a mechanism (A8).

## Amendments after review 2, before any run (2026-11-20)

Status: a second outside review read the amended design, the minimal pilot's code and its tests, ran a simulation of the ablation, and put 35 bugs back into the code one at a time (21 survived the tests). Its verdicts were "pass with fixes" for the engineering and "fail as written" for the science, turning to "pass with fixes" after four must-fix items. Everything below was decided after that review and before any activation, patching, probe or lens result existed; no interpretability job has run. Where this amendment contradicts sections 1 to 12, amendment 1 or section M, this amendment holds. `deviations.md` records it ("Interpretability pilot: amendments after review 2").

### B1. The last layer made the ablation test tautological; it is replaced by a depth profile

At the last layer the readout is linear in the residual up to the final norm's per-prompt scale: with $h$ the residual after the last layer, $\gamma$ the final RMSNorm and $w = W_U[\text{Yes}] - W_U[\text{No}]$,

$$
D(h) = w^\top \gamma(h).
$$

The task direction $u$ at that layer is the unit mean of the items' aligned-minus-swapped differences, so the sum of those differences points exactly along $u$. A mean ablation sets the component along $u$ to the same value $c$ in both prompts of every item, which deletes every item's difference along $u$; what is left of the differences sums to zero, and so does its effect on a linear readout. The pooled removed share is therefore about 1 at the last layer whatever produced the effect, a single shared feature or thousands of item-specific changes. Cross-fitting does not rescue it, because the other fold's mean difference points the same way as soon as the effect has any shared part. Review 2 simulated this in 4,096 dimensions with mostly item-specific differences and a small shared part and found a removed share of about 0.99 for every size of the shared part. Section 7's rule ("the layer whose ablation removes the most") would then always have picked the last layer, and the row "one direction removes at least 60%" would always have said yes.

Changes:

- *Layers.* The ablation's *depth profile* is every fourth layer from 4 up to 0.8 of depth: 4, 8, …, 28 for Qwen3-8B (`analysis.profile_layers`; 4, …, 24 for Llama-3.1-8B). The last layer (35) is ablated only as a sanity edit, `task_last|35`, expected to remove about everything; it enters no readout. Layer 32 is decided as follows: only three decoder layers follow it, so the same argument holds approximately, and how nearly linear those layers are cannot be known before running. It is ablated with the task direction only (`task|32`) and reported with that caveat, but it is excluded from the summary, from the layer choice and from `task_from` (`analysis.caveat_layers`). The cost is that an effect carried by one direction only from layer 32 on would read as "no onset within the profile"; that is the conservative error.
- *Summary measure.* "The maximum over layers" is withdrawn for the ablation. The pre-registered summary is the *onset layer*: the earliest profile layer $L^*$ from which the cross-fitted removed share's 95% lower bound stays at or above 0.6 at every later profile layer (`analysis.onset_layer`). A claim about every layer from $L^*$ on is an intersection of per-layer tests, so the per-layer intervals need no correction. If the deepest profile layer's lower bound is below 0.6, the onset is undefined ("no onset within the profile"). The removed share at every profile layer is reported with its interval.
- *Outcome-table row, restated* (replaces section 9's row "One answer-position direction removes the effect"). The onset layer is compared with the patching arrival layer of section 5 (the first layer from which noising the final template tokens restores at least half of the effect), rounded up to the next layer of the ablation grid, since the onset can only be a grid layer (`analysis.onset_versus_arrival`). H1 and H2 predict that the onset comes after the arrival layer, or that there is none: the goal's influence reaches the final positions through a changed state, which the answer position integrates item by item before one direction can carry it. H3, H4 and H5 predict that it comes at or before the arrival layer: the signal is written at the answer position as one feature, so one direction carries it as soon as it arrives. H6 predicts the same as H3, early. The neutral-gap condition of A3 (the ablated gap at least 90% of the unablated one) is read at the onset layer.
- *Test.* `test_last_layer_ablation_is_tautological_and_the_depth_profile_is_not` builds review 2's case at small size (256 dimensions, 400 items per fold): middle-layer differences that are item-specific with mean zero, turned into a shared shift by a nonlinearity before a last layer read through an RMSNorm. The cross-fitted last-layer ablation removes about 0.95 of the effect, the middle-layer ablation about 0, the onset over the profile is undefined, and an onset that admitted the last layer would have reported it. With a shared middle-layer direction instead, the onset is found at the middle layer.

### B2. H4 against H3: the readout confound

Any direction that raises "Yes" at the answer position has a component along the readout direction $r = \gamma \odot w$ (A1). The task direction $u$ does, because the task raises "Yes"; the success direction $s$ does, because successes read "Yes". Their cosine can therefore be well above zero under H3, where the task's signal is not the model's success feature at all, and adding along $s$ raises the readout through $r$ whatever the mechanism. Section A5's first and third H4 predictions could not separate H4 from H3.

The H4 cosine is therefore read after the readout direction is projected out of both directions. With $\hat r = r / \lVert r \rVert$ and, for any direction $v$, $v_\perp = (v - (v^\top \hat r)\hat r) / \lVert v - (v^\top \hat r)\hat r \rVert$, it is

$$
\cos(u_\perp, s_\perp).
$$

Changes:

- *Success direction within verbs.* 106 of the 292 determined successes have verbs that no determined failure has (examine 56, go to 24, close 16, use 6, inventory 4), so the plain difference of means also encodes "which verb is this". $s$ is now the weighted mean over verbs present in both sets of the within-verb difference of mean neutral residuals (successes minus failures), each verb weighted by $n_s n_f / (n_s + n_f)$ (`analysis.success_direction`). The failures used are those after step 0, since no determined success is at step 0 (B4).
- *Cosines after projecting out the readout.* The projected cosine above is reported next to its null baselines: the same cosine for the mention-contrast direction (naming the action's object without the goal role), the cosine between the two folds' estimates of $u$ (how far the estimate agrees with itself; a cosine of $u$ with anything is read against it), the cross-fold cosine of one fold's $u_\perp$ with the other fold's $s_\perp$ (no shared estimation noise), and $1/\sqrt{4096} \approx 0.016$, the standard deviation of a random pair's cosine (`pilot.fold_cosines`). The raw cosine is still reported and not read.
- *Steering without the readout component.* A new edit, `steer_success_perp_readout|L`, adds along $s_\perp$ the mean aligned-minus-swapped projection on $s_\perp$; its gain on swapped prompts is reported as a share of the goal effect on the same items. Raw steering along $s$ is reported and not read.
- *Layers fixed now.* The H4 rows are read at the profile layers up to two thirds of depth (Qwen3-8B 4, …, 24; Llama 4, …, 20; `analysis.h4_layers`), and of those only at layers strictly before the patching arrival layer measured on the calibration fold. From the arrival layer on, the final positions already carry the effect and the answer position's residual is increasingly a function of the answer itself, so two directions tied to the answer align there whatever the mechanism. Two thirds is where the design's "late" third begins; the arrival cut can only remove layers, never add them. If no layer remains, the H4 rows are "not assessable".
- *H4 predictions, restated* (replace A5's first three; medians over the H4 layers, `pilot.h4_rows`). H4: the projected cosine is at least 0.3 and exceeds the mention direction's projected cosine; steering along $s_\perp$ raises swapped prompts by at least half of the goal effect; ablating the task direction without its success component removes less than half of what the task direction removes. H3: projected cosine below 0.1; steering along $s_\perp$ below 20% of the effect; the task direction without its success component removes at least 80% of what the task direction removes. "The direction moves determined successes as much as failures" (kept) is read after step 0 only (B4).

### B3. Calibration rule, and interventions on one fold only

**Calibration rule (dated 2026-11-20, after review 2, before any run).** The substantive bands stay as written: 0.15 and 0.3 for the history share, 0.6 and 0.3 for removal by one direction (now read through the onset layer, B1), 0.2 and 0.5 for κ, 90% for the neutral gap, and the H4 bands of B2. The minimal pilot does not move any of them, and A6's "the bands are fixed from the minimal pilot's variances" is withdrawn. The minimal pilot's per-stratum game-bootstrap standard errors are used for three things only, all for the full pilot: the number of items per stratum for each summary measure's 95% interval to reach a half-width of at most 0.075 (half of the narrowest gap between bands, 0.15 to 0.3), computed as the pilot's n times $(1.96 \, \mathrm{SE} / 0.075)^2$ (`analysis.n_for_half_width`, reported by the analysis as `sample_size`); the number of bootstrap resamples; and the form of the Bonferroni intervals (B5). These settings are written in a dated note before any full-pilot item is patched or edited.

The scaling assumes the SE falls as one over the square root of the number of items; with games as clusters it falls more slowly once items per game grow, so the computed n is a lower bound and the note adds a margin.

**Interventions on the calibration fold only.** The minimal pilot patches and edits only items of the calibration fold's games (fold 0 of split seed 0). Patching already did so (its items and placebo pairs come from the calibration fold); the ablation now does too (`edit_folds: [0]`): calibration-fold items are edited with directions estimated on the other fold's stage-0 residuals. The other fold's items are never patched or edited before the full pilot, so its intervention results cannot have shaped any choice made after the minimal pilot. Stage 0 still runs on both folds, because the reproduction gate needs every item and the cross-fitted directions need the other fold's residuals; those are observational readouts, already seen at the gate, not intervention results. With one fold edited, the secondary readouts at "the chosen layer" can only use a layer chosen on the same fold's items; they are flagged "in-sample" in the output, and the pre-registered summary (the onset layer) does not depend on a choice.

### B4. Step 0 and object mentions: quotas, weights, and what is undefined

At the minimal pilot's sizes, proportional sampling left too few step-0 items to say anything per stratum. In the calibration fold, 239 of about 1,840 failures with a clear logged effect are at step 0 (13%), and only 60 (3%) are later items whose history after the task line names the action's object (the items that have `hist_obj` tokens at all). Changes (`run.select_stratified`):

- *Patching*: 100 items, at least 30 at step 0 and at least 25 later items with an object mention; the rest (45) are later items without one, each stratum split over failure reasons in proportion.
- *Ablation*: 75 failures with at least 20 at step 0 (proportional would give about 8), and 50 determined successes, all on the calibration fold.
- *Weights.* Each item's weight is its stratum's share of the eligible items over its share of the sample (strata: step 0, later with a mention, later without), stored in the run's manifest. Every pooled ("all") estimate (ratio-of-sums fractions, removed shares, steering shares, gaps, shifts) uses the weights inside every bootstrap resample, so pooled numbers describe the population, not the oversampled mix. With the expected mix the weights are about 0.43, 0.13 and 1.86, and the pooled estimate's effective sample size is about 62 of 100. Within the step-0 stratum and within the object-mention share the weights are constant and change nothing.
- *No determined success at step 0.* All 292 determined successes have a history (`history_len` of 1 or more). The success direction is therefore estimated on later items (B2) and applied unchanged to step-0 prompts; the step-0 success-versus-failure gap, and the removal on determined successes at step 0, are undefined and are reported as such (`analysis.gap` returns `defined: false`), not as zero. At step 0 the discrimination row and the "moves successes" row are not filled.
- The object-mention share is computed on the oversampled mention items (about 25) against their position-matched control. Section M's object-mention readout is no longer uncalibrated, but with 25 items it is still a calibration number, not a verdict.

### B5. Bonferroni intervals

A percentile interval at $\alpha = 0.05/36$ reads the outer 0.07% of the bootstrap draws on each side, one or two of 2,000 draws, so its bounds are noise. Every Bonferroni-corrected interval (per-layer patching fractions, removed nats, the placebo floor) is now a normal interval from the bootstrap standard error: the point estimate plus or minus $z_{1-\alpha/2}$ times the standard deviation of the draws (`analysis._interval_normal`), which 2,000 draws estimate well. Pointwise and maximum-over-layers intervals stay percentile intervals.

### B6. First-token readout in stage 0

Stage 0 now also scores the first-token readout ("You" against "Nothing") on the 4,666 failures' aligned and swapped two-candidate prompts (9,332 prompts, readouts only, no residuals kept), and gates it against the logged first-token term `L_first` of `rq1v_score`'s plain arms, with the same five checks as the yes/no gate (mean effect 2.52 nats for Qwen, published). Stage 0 passes only if both gates pass. A patching grid in that format (noising only, about 25 items, paid for by dropping denoising from the calibration run) was considered and not adopted: 25 items calibrate nothing, and denoising is the sufficiency half of the history share, which the calibration needs. The first-token patching map stays in the full pilot.

### B7. Groups

The action line is split into its object (the action's object word and instance number, `action_obj`) and the rest (`action_rest`); "action" stays as their union. A large action share carried by the object's name points to lexical matching (H6); one carried by the verb and receptacle does not. `hist_swap` (history tokens naming the swapped object) is added to the minimal pilot's grid. The grid is now 12 groups by 36 layers.

### B8. GPU type and compute

The job script asks for 12 hours (was 8). Every GPU stage reads the GPUs recorded in stage 0's `jobs.jsonl` (and, when resuming, its own) and refuses to start, writing why, on any other GPU type, or when it cannot read its own GPU while one is on record (`pilot.check_gpu`), so a resubmission must use stage 0's GPU type. Recomputed forward rows, at the design's 0.1 s per row on an A100 and 0.15 s on an L40S:

| Run | Forward rows | A100 hours | L40S hours |
|---|---|---|---|
| `interp_pilot_stage0` (19,540 yes/no prompts with residuals, 9,332 first-token prompts without) | 28,900 | 0.8 | 1.2 |
| `interp_pilot_patch` (at most 432 cells per direction, batches of 15 cells and a clean row, a source run per direction; 100 items and 50 placebo pairs) | at most 144,000 | 4.0 | 6.0 |
| `interp_pilot_ablate` (calibration fold: 125 items, 3 prompts, 48 edits in 4 batches of 16) | 24,000 | 0.7 | 1.0 |
| Total | about 197,000 | about 5.5 | about 8.2 |

Patching is an upper bound: step-0 items have no history groups and items without an object mention no `hist_obj` cells. 12 hours leave about 45% over the L40S estimate.

### B9. Rounding of edits

An edit is computed in float32 and cast back to the model's bfloat16, which moves the edited component by up to about $2^{-8}$ of its size. On the first prompt each ablation job edits, the residual at the answer position is read after every edit at every edited layer, and the realized value is logged (`run.edit_items(checks=…)`, `edit_checks.json`): for a mean ablation the component along the direction minus its target, for steering the added component minus the amount.

### B10. Gate references

The logged `rq1v_score` rows carry no `model_revision` (null): the revision is the one pinned in `wf.constants` at the scoring run's commit and is not recorded per row; stage 0's manifest says so. Each logged row's `item_hash` (the scorer's hash of the action, history, task lines and candidates) is now checked against the current item's, and any mismatch fails the gate. On the archived scores, all 36,102 Qwen rows match.

### B11. Tests

Value checks on synthetic data with known answers now cover the 21 mutants that survived review 2 (the end-to-end test checked structure only), and every new piece of code has its own. The review's 35 mutants (seven adapted to changed lines) and 42 mutants of this round's code are in `scripts/mutation_check.py`; the killing tests are listed in `testing.md`. One existing test's tolerance was raised from $10^{-6}$ to $10^{-4}$ nats: multithreaded float32 matrix products on a CPU differ by a few $10^{-6}$ between batch shapes, and the identical-prefix check it guards fails on wrong positions by far more.

## Amendments after review 3, before any run (2026-11-21)

Status: a third outside review read amendment 2, its code and tests, simulated the ablation once more, checked the selection and hashes on the real items, and put 40 bugs into the code (19 survived the tests). Its verdicts: "pass with fixes" for the engineering; for the science, "fail as written" for the rows that carry hypothesis labels, and "pass with fixes" after four must-fix items. Everything below was decided after that review and before any activation, patching, probe or lens result existed; no interpretability job has run. Where this amendment contradicts anything earlier, it holds. `deviations.md` records it ("Interpretability pilot: amendments after review 3").

### C1. Onset against arrival: like with like, and no hypothesis labels

B1 compared the onset (a 95% lower bound staying at or above 0.6, on the ablation grid, at the answer position) with an arrival layer of a different kind (the first layer from which the *final* template tokens' point fraction stays at or above 0.5, at any layer). Changes:

- *Arrival on the same terms.* The arrival layer the onset is compared with is now read from the noising curve of the `last` group (the answer position, where the ablation acts) under the onset's own rule: the earliest layer of the grid 4, 8, …, 32 from which the pointwise 95% lower bound of the fraction restored stays at or above 0.6 (`analysis.patch_arrival_layer`, `pilot.arrival_on_answer_position`). The final group's 0.5 arrival is still reported, descriptively, with its index rounded to 6 decimals.
- *Rounding.* Any arrival index is rounded to 6 decimals before it is rounded up to the grid, so floating-point noise (32.000000001) cannot move it to the next grid layer.
- *Not assessable.* If there is no arrival on the grid, or the arrival lies after the deepest layer the onset is read on, the row is "not assessable". If the direct-path rule (C2) excludes every profile layer, it says so.
- *No onset.* "No single direction" only when the deepest layer's removal has an upper bound below 0.3; otherwise "inconclusive (no onset, interval too wide)". Before, a missing onset always read as evidence for H1 and H2, even when the interval was too wide to say.
- *No labels.* The minimal pilot's output (`table_rows`, formerly `verdict_rows`) names no hypothesis: "at or before arrival", "after arrival", "no single direction", "inconclusive", "not assessable". The mapping to hypotheses of B1 stays in this note and is applied only in the full pilot.

### C2. Direct-path share: a second tautology, and the rule for it

Review 3's simulation (`direct_path_sim.py`) gave the answer position at a middle layer an item-specific change $z_i$ with no shared direction, plus a part $a_i r$ already in readout form. A cross-fitted mean ablation along the mean difference then removes about the readout-form share of the effect by construction: as that share rose from 0 to 0.65, the removed share rose from 0 to 0.84, and the onset rule (lower bound of at least 0.6) passed once the share was about 0.55, although no single direction carries the item-specific change. Near the end of the profile (layer 28) a large readout-form share is likely under any account.

The pilot now reports, at every profile layer $\ell$ and per stratum, the *direct-path share*: the part of the goal effect already in readout form at that layer, read at the final norm's scale. With $r = \gamma \odot (W_U[\text{Yes}] - W_U[\text{No}])$ the readout direction, $h^A_{\ell,i}$ and $h^S_{\ell,i}$ the aligned and swapped answer-position residuals of item $i$ at layer $\ell$, $\rho^A_i$ and $\rho^S_i$ the root mean square of each prompt's last-layer residual (the final norm's scale), and $D^A_i$, $D^S_i$ the final yes/no readouts,

$$
\lambda_\ell = \frac{\sum_i \left( r^\top h^A_{\ell,i} / \rho^A_i - r^\top h^S_{\ell,i} / \rho^S_i \right)}{\sum_i \left( D^A_i - D^S_i \right)} .
$$

It is computed on CPU from stage 0's residuals on the ablation's failure items, weighted, with a game-bootstrap interval (`analysis.direct_path_share`, `pilot.direct_path_report`). At the last layer it equals 1.

**Pre-registered rule (one, chosen now): a profile layer whose direct-path share has a 95% upper bound at or above 0.3 is treated as a caveat layer: its removal is reported, but it is excluded from the onset, which is read on the remaining profile layers, per stratum (`analysis.direct_path_kept`, `DIRECT_PATH_MAX`).**

Why this rule rather than reading the onset on removal minus $\lambda$ with a paired bootstrap: the subtraction would make the 0.6 band unreachable wherever $\lambda$ exceeds 0.4, and it would also subtract the part H3 itself predicts. Under H3 the goal's signal is written in readout form at the answer position, so a large $\lambda$ is part of the hypothesis, not a nuisance to subtract. The question the onset can answer cleanly is whether one direction carries the effect *before* the effect is in readout form, and that is only answerable at layers where the direct path is small. The threshold 0.3 is the band below which a removal of at least 0.6 cannot be explained by the direct path alone. The cost: if every layer from the true onset on has a large direct-path share, the row is "not assessable". (Review 4, D1:) because an excluded layer can push the onset later, an onset after the arrival is read as "after arrival" only if no profile layer between the arrival and the onset was excluded; otherwise the row is "not assessable". `test_direct_path_share_tracks_the_removal_and_the_rule_excludes_it` reproduces the simulation's pattern (with readout-form share 0, 0.3 and about 0.6, the removal tracks it, and the rule removes the layer where the naive onset would have been found).

### C3. The H4 rows, restated again

- *Steering against a reference.* A new edit, `steer_task|L`, adds along the task direction its own mean aligned-minus-swapped projection. The H4 steering row is now the gain along $s_\perp$ (the success direction without its readout component) over the gain along $u$ at the same layer, summed over the same swapped prompts (`analysis.steer_relative`). Steering along $u$ is what the task itself does, so the ratio says how much of that a generic success feature reproduces, free of differences in scale between layers.
- *The perp-over-task removal ratio only where it means something.* The ratio of what the task direction without its success component removes to what the task direction removes is read only at layers where the task direction's removal has a lower bound of at least 0.3 (`analysis.perp_over_task`); elsewhere it is "not assessable", and the H4 row is the median over the assessable H4 layers, or "not assessable" if none.
- *Mention is not a null for H4.* A mention of the object is itself an object match, so "the projected cosine exceeds the mention direction's" is dropped from the H4 criterion. The mention direction's projected cosine with the success direction, and the task-minus-mention difference, are reported apart, as a goal-role row (`goal_role`).
- *Reliability and intervals.* Between-fold cosines are now reported for the projected task, success and mention directions. Each projected cosine is also reported disattenuated: divided by the square root of the two directions' between-fold cosines, what it would be between noise-free directions, to first order (`analysis.disattenuated`; undefined if a reliability is not positive). Every projected task-success cosine gets a game-bootstrap interval in which both directions are re-estimated in every resample (`analysis.projected_cosine_boot`, `pilot.h4_cosine_report`).
- *A layer-specific answer direction.* The final readout direction is only the last layer's notion of "the answer"; at a middle layer the answer may be carried along another direction. At every H4 layer a ridge regression of the final readout on that layer's neutral-prompt residual, fit after centring both within each (outcome, verb) cell so that it does not learn the success or verb contrast, gives a layer-specific answer direction (`analysis.answer_direction`). The H4 cosine is reported with the readout direction projected out, and with both the readout and the answer direction projected out.

(Review 4, D3, D4:) the steering ratio is read only at layers where the task steering's share of the goal effect has a 95% lower bound of at least 0.3, and the H4 row is the median over those layers or "not assessable"; the H4 layers are cut at the earlier of the two defined arrival layers (the final group's at 0.5 and the answer position's under the onset rule).

H4 predictions, restated (medians over the H4 layers of B2). H4: the projected cosine is at least 0.3 under both projections (its bootstrap lower bound above 0); steering along $s_\perp$ gains at least half of what steering along $u$ gains; where assessable, ablating the task direction without its success component removes less than half of what the task direction removes. H3: projected cosine below 0.1 under both projections; steering along $s_\perp$ below 20% of the gain along $u$; where assessable, the success-free direction removes at least 80% of what the task direction removes. The disattenuated cosine is reported and read only together with the raw one: a disattenuated value cannot rescue a raw cosine whose interval includes 0.

### C4. Steering amounts per stratum

Qwen's goal effect is about twice as large at step 0 as later, so one pooled steering amount over- or under-steers each stratum. Every steering amount (along the success direction, along it without its readout component, along the task direction) is now estimated on the other fold separately for step 0 and for later items (`estimate_directions(strata=…)`, `steer_by_stratum`), and each item is steered by its own stratum's amount: the ablation edits each stratum's items in a separate call (`run_interp_pilot_ablate`). Mean-ablation midpoints stay pooled. If the other fold had no step-0 items, the pooled amount would be used; it has 268.

### C5. Tests and mutants

Value tests on weighted bootstrap draws (the weights must enter every draw, not only the point estimate), the verb weighting, the mention rows, the first-token gate's arms, the selection when a minimum exceeds its stratum, the GPU check, the caveat layers and the defaults kill the 19 surviving mutants of review 3 (one of them, K36, by checking that every ablation item has a weight; K19 is equivalent and left out: with more eligible items than n the selection always takes exactly n). Review 3's 40 mutants (K4, K5, K26 and K32 adapted to changed lines, K19 left out) and 28 mutants of this round's code are in `scripts/mutation_check.py`; killing tests are in `testing.md`.

### C6. Calibration rule, completed; minimums for small draws; folds in the full pilot

The calibration rule of B3 is completed as follows (dated 2026-11-21, before any run):

- *Margin fixed.* The full pilot's n per stratum and measure is the pilot's n times $(1.96\,\mathrm{SE}/0.075)^2$, times 1.2 (`SAMPLE_SIZE_MARGIN`), the allowance for games as clusters.
- *B fixed.* Every bootstrap uses 2,000 game resamples, for percentile intervals and for the SEs behind normal intervals; the pilot does not set it.
- *Bonferroni form struck.* It was already fixed by B5 (normal intervals from the bootstrap SE); the pilot does not set it either. What the pilot's SEs set is the full pilot's sample sizes, nothing else.
- *More measures.* The sample sizes are now computed also for the object-mention share, the neutral-gap ratio (with a paired bootstrap SE), the answer-position arrival (the SE of the `last` curve at the arrival layer), the steering ratio and the perp-over-task ratio (`pilot.sample_sizes`). (Review 4, D5:) the H4 cosine and the direct-path share are stage-0 quantities, computed on every item of the fold, so for them only the SE is reported; a sample size does not apply.
- *Folds in the full pilot.* Its verdicts use fold 1's items only (the fold no intervention has touched). Fold 0's 100 patched items and 125 edited items are reported separately and never pooled into a verdict.

*Minimums for small draws.* The stratum minimums are set for 100 patching items and 75 ablation failures; when fewer are drawn they are scaled by n over that reference and rounded half up (`pilot.scaled_minimum`). The 10-item timing run of patching therefore draws at least 3 step-0 items and 3 with an object mention, instead of minimums larger than the run.

### C7. GPU names and compute

`check_gpu` compares GPU names exactly. `--constraint=gpu-a100` can land on another A100 model (40 GB against 80 GB, PCIe against SXM) whose name differs, and the check will then refuse it. That is kept as the conservative choice, since bfloat16 kernels are not guaranteed to match across models of a family; the job script says so, and a refused job is resubmitted to a node with stage 0's exact GPU (the refusal message names it). The ablation now has 55 edits per prompt (7 kinds at 7 profile layers, 4 random, the caveat and the last layer), which still fit in 4 batches of 15 edits and a clean row, so the forward rows are unchanged: about 197,000 in all, 5.5 A100-hours (about 5.6 with model loads and the per-stratum calls) or 8.2 L40S-hours, within the job's 12 hours.

### C8. What the yes/no format can and cannot show

- *An answer the model never gives.* On RQ2's failures, Qwen3-8B's mean probability of "Yes" is about 0.2% even under the aligned task (`exploratory_results.md`, section 4): its answer is at the floor in both conditions. The yes/no pilot therefore localizes a shift of 2.8 nats in log-odds inside an answer that stays "No", not a change of answer. That is a legitimate object for localization, but its findings should not be described as the mechanism by which the model "predicts success".
- *Not the headline effect.* The confirmatory effect is 6.12 nats in the two-candidate format, where the success text is often preferred. The yes/no effect is a different readout of the same manipulation. The minimal pilot does not explain the headline; at most its localization carries over. The first-token readout is now gated in stage 0 (B6), and a residual map of the full two-candidate reply (teacher-forcing the success text, by the coarse groups, 50 items; A9) is planned for the full pilot.
- *Transfer.* The first transfer target for any direction or localization found here is TextCraft's yes/no format (branch `textcraft`, once scored and showing a goal effect), before τ-bench. TextCraft's current yes/no question ("Will this action succeed? Answer Yes or No.") is not anchored to the environment's reply and is partly read as task relevance (about two-thirds of Qwen's TextCraft yes/no effect); transfer there requires a reply-anchored TextCraft question, validated before any direction is applied.

## Amendments after review 4, before any run (2026-11-21)

Status: a fourth outside review read amendment 3, its code and tests, simulated the direct-path exclusion, the steering ratio and the answer direction, and put 32 bugs into the code (22 survived the tests). Its verdicts: "pass with fixes" for both the engineering and the science, not yet ready to freeze. Everything below was decided after that review and before any activation, patching, probe or lens result existed; no interpretability job has run. Where this amendment contradicts anything earlier, it holds. `deviations.md` records it ("Interpretability pilot: amendments after review 4").

### D1. An exclusion between arrival and onset

The direct-path rule (C2) removes layers from the onset's profile, and removing a layer can only move the onset later. Review 4's simulation: with the removal's lower bound at or above 0.6 from layer 12 on and the arrival at 12, excluding only layer 12 (a direct-path upper bound of 0.32) moved the onset to 16 and turned "at or before arrival" into "after arrival" by construction. `onset_versus_arrival` now receives the whole profile; when the onset lies after the arrival and some profile layer from the arrival up to (not including) the onset was excluded, the row is "not assessable (a layer between arrival and onset excluded by the direct-path rule)". Exclusions after the onset change nothing.

### D2. Tests for the exclusion

Tests now cover an excluded layer that is not the deepest (layer 8 passes, layer 4 fails, 8 is excluded: no onset, and the deepest upper bound comes from layer 4), review 4's four exclusion cases, an arrival exactly at the deepest kept layer, and an upper bound of exactly 0.3.

### D3. The steering ratio is gated

Review 4's simulation: when steering along the task direction barely moves the readout (a gain of 0.02 to 0.15 nats), the ratio of the gain along the readout-free success direction to it ran from −2 to 54 with intervals spanning ±80. The ratio (`steer_perp_over_task`) is now read only at layers where the task steering's share of the goal effect has a 95% lower bound of at least 0.3 (`STEER_TASK_MIN_GAIN`, `analysis.gate_on_lower_bound`); the H4 row is the median over those layers, which are listed, or "not assessable".

### D4. H4 layers before either arrival

The H4 rows are read at layers strictly before the earlier of the two defined arrival layers: the final group's (point fraction 0.5) and the answer position's (C1). Either arrival marks where the effect already sits at the answer, and the H4 cosine must be read before both.

### D5. Sample sizes of stage-0 quantities

The H4 cosine and the direct-path share are computed on every item of the fold from stage 0, so only their SE is reported; their sample size is "not applicable (stage 0 uses every item of the fold)".

### D6. What section M says the minimal pilot "can conclude"

Section M lists what the minimal pilot "can conclude". Read with amendments 2 to 4, those are calibration readings, not conclusions. The minimal pilot reports the localization quantities (shares, arrival, onset, direction readouts) against provisional bands, without hypothesis labels. The mention comparison is a goal-role row, not a test of H4. The minimal pilot's variances set only the full pilot's sample sizes (B3, C6, D10). Section M's "cannot conclude" list stands.

### D7. When the second projection can be read

Review 4's simulation: when the success feature varies within cells, the layer-specific answer direction picks it up (cosine with the true success direction up to 0.96), and projecting the answer direction out removes the very feature H4 is about (the H4 cosine fell from 0.82 to 0.35). The cosine of the answer direction with the readout-free success direction is now reported per layer and fold (`answer_success_perp_readout_cos`). The "under both projections" condition of C3 is read only at layers where its absolute value is below 0.5 in both folds (`ANSWER_SUCCESS_MAX_COS`); elsewhere it is "not assessable", and the layers used are listed.

### D8. Direct-path share on the whole fold; NaN bounds; low reliabilities

- The direct-path share is now computed per stratum on every failure of the edited (calibration) fold, from stage 0 on CPU, rather than on the 75 sampled ablation failures. It is then the population's share and needs no weights. Its SE is reported with the sample sizes (D5).
- A profile layer whose direct-path share has no finite upper bound is excluded from the onset (before, a NaN bound kept the layer). A layer with no estimate at all is kept.
- A disattenuated cosine is undefined when either between-fold reliability is below 0.2 (`MIN_RELIABILITY`): dividing by a small, noisy reliability inflates it arbitrarily.

### D9. Limits that remain, and a free linearity check

- *Identity path only.* The direct-path share counts what is already along the readout direction at the final norm's scale, the residual stream's identity path to the readout. Readout-form information that later layers would amplify or rotate is not counted.
- *Different item sets.* The answer-position arrival is measured on patching items with an effect of at least 1 nat; the onset is measured on ablation items drawn without that filter. The comparison assumes the localization does not depend on the size of the effect.
- *"At or before arrival" is not H3 on its own.* Under H2 a changed state in the action or final tokens can also reach the answer position as one direction by the arrival layer. The row is not to be read for H3 without the probe and history rows of the full pilot.
- *Linearity diagnostic.* At each profile layer the task steering's share of the goal effect is reported next to the task direction's removed share (`linearity_steer_task_vs_removal`). If the rest of the network were close to linear along the task direction, adding its mean projection and ablating to the midpoint would move the readout by about the same amount; their ratio is reported, not read.

### D10. How the per-measure sample sizes combine

Pre-registered now. Per stratum, the full pilot's patching items are the largest number any patching measure needs (history share, action share, object-mention share, answer-position arrival). Its ablation failures are the largest number any ablation measure needs (removal at each profile layer, gap ratio, steering ratio, perp-over-task ratio). Each is capped at 400 per stratum (`PATCH_CAP_PER_STRATUM`, `ABLATE_CAP_PER_STRATUM`); 400 patching items cost about 11 A100-hours and 400 ablation failures about 2. If a cap binds, the full pilot runs at the cap, the analysis reports the half-width each measure reaches there, and the bands stay as they are (`analysis.plan_sizes`, reported as `sample_size.<stratum>.plan`). The stratum's availability also bounds it: fold 1 has 268 step-0 failures.

### D11. Tests, mutants and a comment

Tests now kill review 4's 22 surviving mutants. N2 is killed by giving the end-to-end model a non-unit final-norm weight, so that the readout direction differs from the plain unembedding difference, and the direct-path test's vacuous last-layer check is replaced by an exact one (the share is 1 at the last layer). Review 4's 32 mutants (N2, N3, N4 and N31 adapted to changed lines; N18 left out as equivalent) and 15 mutants of this round's code (one more was equivalent) are in `scripts/mutation_check.py`. A mutant counts as killed only when a named test fails (pytest exit code 1). The job script's comment now says 55 edits per prompt.
