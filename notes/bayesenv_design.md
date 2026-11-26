# Controlled Bayesian environment: design (confirmatory; frozen before any model score)

**Status.** Design note for a confirmatory study. Everything below is fixed at the annotated tag named in section 16 before any model is scored. The smoke run's readout (section 17) is appended and committed before the full run is submitted; it applies the sizing rule of section 9 and changes nothing else. Code: `wf/bayesenv/` (world and exact posterior, prompts, design, scorer, analysis, planted models), configs `wf/config/bayesenv_*.yaml`, job `pace/bayesenv.sbatch`, tests `tests/test_bayesenv.py`. The independent review of this design and its code is in `notes/bayesenv_review1.md`.

## 1. The question

In the natural environments (ALFWorld, TextCraft, τ-bench), replacing the object named in the agent's one-line task changes a language model's predicted success of an action whose outcome the history already settles. Natural data cannot say whether such a change is a miscalibrated use of legitimate evidence or an influence of the task that has nothing to do with evidence, because the natural environments' priors are unknown and the task can legitimately carry information about where objects are. This study builds a small world in which the prompt states exactly how the world was generated, so that every prompt has one exact posterior probability that the queried action succeeds. It then asks, for Qwen3-8B and Llama-3.1-8B:

1. whether the model uses stated evidence at all, and in proportion to its strength;
2. whether a task line that names the action's object moves the prediction more than a non-goal line that carries exactly the same information, by the same route, in the same position, naming the same object (the "house tag");
3. whether there is a structured residual where the exact posterior does not move;
4. whether any such failure is larger when the model predicts the environment's reply to an action than when it infers the current state from the same story;
5. whether the same signatures appear when the world is written in ALFWorld's own text.

The study can show that a model's predictions depart from the exact posterior in specified ways, and that particular simple accounts do not fit. It cannot identify a mechanism or a mixture of mechanisms: rejecting pure accounts leaves open accounts not considered, and components fitted to these prompts are reported only as sufficient for these prompts.

## 2. The world

A house has six boxes, numbered 1 to 6. The queried action is always "take the key from box 3". Each box is locked with probability 0.2, independently. A **generator** hides every object that the task line or the house tag refers to with probability $\pi_{\text{n}}$, and every other object with probability $\pi_{\text{o}}$; a hidden object goes into one of the six boxes, chosen uniformly. An object that is not hidden is not in the house. The take succeeds if and only if the key is in box 3 and box 3 is not locked; the environment then replies "You take the key from box 3."; otherwise it replies "Nothing happens.".

Write the task line's (or tag's) likelihood ratio on the key's existence as

$$\Lambda_g = \frac{\pi_{\text{n}}/(1-\pi_{\text{n}})}{\pi_{\text{o}}/(1-\pi_{\text{o}})}.$$

Four generators vary this ratio, and one also varies the base rate:

| Generator | Hiding probability of an object the task or tag does not refer to | Of an object it refers to | Likelihood ratio of the cue on the key's existence |
|---|---|---|---|
| A | 50% | 50% | 1 (the cue carries no information) |
| R3 | 50% | 75% | 3 |
| C | 50% | 90% | 9 |
| D | 20% | 90.7% | 39.0 |

The generator sentence in the prompt never names an object: "How the house was set up: first the player's task and the house tag (which may be none) were drawn. Then every object that the task or the house tag refers to was hidden with probability 90%, and every other object with probability 50%. A hidden object was put in one of the six boxes, chosen at random; an object that was not hidden is not in the house at all." The take rule says the take succeeds "if and only if the object is in that box and the box is not locked". The key is therefore named only in the action, in histories and notes that name it, and in cue lines that refer to it. The coin's hiding rate follows the same stated rule.

**Histories.** Seven histories, each shown as the agent's earlier actions and the environment's replies, with a rule sentence stating each instrument's accuracy.

| Code | What the history shows | Kind |
|---|---|---|
| H0 | nothing done yet | graded |
| H2 | four other boxes looked into and reported empty; a look misses an object that is there 30% of the time | graded |
| H3 | four other boxes looked into and reported empty; looks never miss | graded |
| H4 | two other boxes looked into (never miss), and a source that is always right says a key is somewhere in the house (two wordings: a house list, or a game master; alternated by item) | existence settled, outcome open |
| H6 | two other boxes looked into (30% miss), and a lock check, wrong 2% of the time, says box 3 is locked | graded, failure nearly settled |
| H7 | two other boxes looked into (30% miss), and a close look into box 3, which never misses, finds it empty | failure settled |
| H9 | a glance at box 3 suggested a key (a glance suggests a key 90% of the time when one is there, and 10% of the time when none is) | graded, high success |

The graded histories spread the exact log-odds of success under the neutral task from −5.9 to −0.6 nats.

**Who chose the actions.** Every prompt says who chose every action. In the main prompts: "Every action in this game, the next one included, was picked at random by a tester who cannot see inside the boxes." (state inference: "Every action in this game was picked at random …"). Two provenance arms replace the sentence for the queried action only, always saying that the earlier actions were picked at random by such a tester:

- **History-only helper:** the next action was chosen by a helper who knows only what is shown and picks, at random, one of the boxes not yet looked into, then tries to take from it the object named by the task in the task line. Its choice does not depend on the hidden state given the history, so the exact posterior equals the tester's. It is used only in histories where box 3 has not been looked into (H0, H2, H6), because in H7 this helper could not choose box 3 and the prompt would have no reference.
- **Seeing guide:** a guide who sees inside every box; when the object named by the task in the task line is in a box that is not locked, it takes it from there 90% of the time; in every other case it picks one of the twelve take actions (the key or the coin, from any box) at random. It is scored only under the key task, where its choice is informative about the key and box 3's lock and the coin plays no part.

Each provenance statement has two wordings, alternated by item.

## 3. The exact posterior

For a prompt whose generator, cue and history imply a prior $\pi$ on the key's existence, the success probability is computed by enumerating the key's location $\ell \in \{\text{none}, 1, \dots, 6\}$ and all $2^6$ lock patterns $\lambda$:

$$P(\text{success} \mid e) = \frac{\sum_{\ell,\lambda} P(\ell)\,P(\lambda)\,P(e \mid \ell,\lambda)\,\mathbf{1}[\ell = 3,\ \lambda_3 = 0]}{\sum_{\ell,\lambda} P(\ell)\,P(\lambda)\,P(e \mid \ell,\lambda)},$$

with $P(\ell = \text{none}) = 1-\pi$, $P(\ell = b) = \pi/6$, and $P(e \mid \ell, \lambda)$ the product of the stated instruments' likelihoods. A caretaker's note "There is a key in this house.", written by a caretaker who looks with probability $\rho$ and otherwise writes either sentence by a fair coin, multiplies the odds of existence by

$$\Lambda(\rho) = \frac{1+\rho}{1-\rho},$$

which is 1, 3, 9 and 39 at $\rho = 0, 0.5, 0.8, 0.95$. The seeing guide's choice of "take the key from box 3" has likelihood

$$\kappa\,\mathbf{1}[\ell = 3,\ \lambda_3 = 0] + \frac{1 - \kappa\,\mathbf{1}[\text{the key is in an unlocked box}]}{12}, \qquad \kappa = 0.9.$$

Under the announced replacement of the task line (section 4), the shown task carries no information and the prior is the generator's marginal over the original task, $(\pi_{\text{n}} + 2\pi_{\text{o}})/3$.

**Checks** (`tests/test_bayesenv.py`). The enumeration agrees with the closed form of a note's log-odds shift to $10^{-12}$, and with a forward simulation of the story as worded in the prompt (draw the house, the looks with their misses, the report errors, the note's coin, the guide's choice, and keep the draws that match the prompt) on every history, with and without the seeing guide, within Monte Carlo error. Every number written into the rules (lock rate, miss rate, report error, glance accuracies, the guide's 90% and twelve actions, the generators' rates, the note's reliability) is checked against the constant the posterior uses. The history-only helper's posterior equals the tester's exactly. Matched cues (goal and tag; tag and description; key tag with and without the coin task) share the exact posterior in every cell.

## 4. Prompts, conditions and items

Every prompt is a system text and one user turn. The user turn has the rules (rooms of the boxes, locks, the generator, the history's instruments, the take rule, who chose the actions, and, in some prompt types, an announcement or a note), then "History:", the task line with the tag line after it, the history's lines, and the framing's ending.

**Prompt types.** Each item is scored under every prompt type of its cell. Only the defining line differs between two prompt types of an item (a test checks this):

| Name | Task line, tag line, extra paragraph | Exact prior probability that the key is hidden |
|---|---|---|
| explore | "Your task is to: explore the house. House tag: none." | the generator's rate for objects not referred to |
| coin | "find the coin" instead of exploring; tag none | the rate for objects not referred to |
| goal (one per wording) | a task naming the key, e.g. "find the key"; tag none | the rate for objects referred to |
| note (one per reliability) | explore, tag none, plus the caretaker's note with stated reliability 0%, 50%, 80% or 95% | the rate for objects not referred to, then the note |
| key tag (one per wording) | explore, with a tag line naming the key, e.g. "House tag: KEY." | the rate for objects referred to |
| other tag | explore, "House tag: COIN." | the rate for objects not referred to |
| described tag | explore, "House tag: the small metal object used to open the front door." | the rate for objects referred to |
| coin task with key tag | "find the coin", "House tag: KEY." | the rate for objects referred to |
| announced goal, announced coin | goal wording 1 or the coin task, plus a statement that the task line is not the task the house was set up for and was redrawn at random from find the key, find the coin and explore the house, independently of the house (two wordings, alternated by item) | marginal |
| helper, guide | goal wording 1, with the history-only helper or the seeing guide choosing the action | as section 2 |

Every non-goal cue sits on the neutral task, so no competing goal is present when a tag or a note is compared with a goal. The coin task appears only in the competing-task checks and the ALFWorld-like swap.

**Wording pools.** Eight goal wordings ("find the key", "get the key", "pick up the key", "look for the key and take it", "fetch the key", "bring back the key", "track down the key", "collect the key") and eight tag wordings ("House tag: KEY.", "House tag: key.", "The tag on this house reads: key.", "This house carries the tag KEY.", "Tag attached to this house: the key.", "This house is tagged with the word key.", "House tag (set before the game): KEY.", "The house's tag names the key."). Every cell uses the first $W_G$ goal and the first $W_T$ tag wordings (base design: 2 and 3). Eight **panel cells** (A with H0 and H6, R3 with H2 and H9, C with H3 and H6, D with H0 and H9) carry all eight of each; they estimate the wording variance (section 8).

**Items.** An item is one draw of the surface in a cell (generator × history): which room each box stands in (six of twelve rooms) and which other boxes were looked into, in which order. Surfaces are deterministic functions of the cell and item index. All prompt types of an item share its surface, so every contrast is within item. Items of different cells are independent. Smoke items (index 1000 and above) are disjoint from the full run's.

**Cells and arms.** The core design crosses the four generators with the seven histories (28 cells), $n$ items each. The announced arm runs on generators A, C and D with the graded histories other than H4, where existence is settled and the observational goal effect is exactly 0 (15 cells); the provenance arms on generators A and C with H0, H2 and H6 (6 cells), on the core items.

## 5. Framings, readouts and their validity gates

**World-model framing.** System: "You are a text world model for a small house game. Given the rules, what has happened so far and the next action, predict the environment's next reply exactly. Output only the reply." The user turn ends with "Action: take the key from box 3" and then either

- "Next observation:", read at the reply's first token, "You" against "Nothing" (the **reply readout**, which asks no question); or
- the anchored question used throughout the project, "Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens. Answer:", read as "Yes" against "No" (the **yes/no readout**).

**State-inference framing.** The same rules, cue and history, with no action. System: "You are reasoning about a small house game. Given the rules and what has happened so far, describe the current state of the house exactly as asked." It ends with either

- "Complete the narrator's sentence about the key and box 3." and the answer prefix "Narrator: Right now, box 3", read as "holds" against "does" (the **narrator readout**, which asks no question; the intended sentences are "… holds the key and is not locked." and "… does not hold a key that can be taken."); or
- "Question: right now, does box 3 hold the key, and is box 3 not locked? Answer Yes or No. Answer:" (yes/no).

Because success is a deterministic function of the state, both framings have the same exact posterior on every prompt. Every prompt names the key in a line that is the same for every prompt type of an item: the action line in the world model, the narrator instruction or the question in state inference (a test checks this).

For each readout the model's score is the log-odds normalized over the two candidate tokens,

$$L = \log p(\text{first candidate}) - \log p(\text{second candidate}),$$

read from a float32 log-softmax of the model's bf16 logits at the position after the prompt. The candidate mass $p(\text{first}) + p(\text{second})$ and both raw logits are recorded for every prompt.

**Readout gates.** A readout is used for decisions only if all three hold, each measured on the run's own prompts:

1. **Competence:** the evidence slope (section 7) has a 95% lower bound above 0.1.
2. **Candidate mass:** the median mass on the two candidates is at least 0.5.
3. **Reply form and "other" rate** (the three question-free readouts: reply, narrator, and the ALFWorld-surface reply). The smoke's checks job samples, for each such readout, 40 prompts (20 whose exact success probability is at least 0.25 and 20 others) with 10 continuations each at temperature 1, up to 16 tokens. At least 30 continuations must begin with each candidate word; at least 80% of those beginning with the first candidate word must be the success form ("You take the key from box 3"; "holds the key" or "holds a key" followed by "not locked" or "unlocked"; "a keychain"), and at least 80% of those beginning with the second must be the failure form ("Nothing happens"; "does not hold"; "nothing"); and at most half of all continuations may begin with neither word. This checks that each candidate's first token stands for its reply (not, for example, "You cannot …" or "does hold …"). The checks run must exist and be a complete run from a clean checkout on the pinned GPU type; without it no readout passes.

A readout that fails is reported descriptively and decides nothing. The primary readout is the world model's reply readout; if it fails its gates, the world model's yes/no readout is primary; if both fail, nothing is read for that model (no readout gets decisions, and no full run is submitted for it if this is found in the smoke). Regrets (section 7) are reported under two conventions: normalized over the two candidates, and unnormalized (all other probability counted as failure).

**Description check.** The described tag must be recognized: three yes/no questions (whether the description describes a key; whether it is a way of referring to a key; whether a house tag reading the description refers to a key) must all favour Yes. If any does not, name minus description is reported as not assessable and leaves the Holm family before adjustment.

## 6. Models, rendering and numerical environment

- **Qwen3-8B** (pinned revision in `wf/constants.py`) in **plain text**: no chat template, so no empty think block. The messages are rendered by the Qwen audit's `plain_text` (`scripts/qwen_audit/followup_score.py`): the system text, then the user turn, separated by a blank line; an answer prefix follows after one space; candidates are read with a leading space. The scorer refuses a plain-text prompt containing chat-template markup or special tokens.
- **Llama-3.1-8B-Instruct** (pinned) in its own chat template with the generation prompt; candidates are the reply's first token, or follow an answer prefix after one space.

Every candidate must be a single token, and its joint tokenization with the prompt must extend the prompt by exactly that token; otherwise the run stops. A tokenizer-only check of every prompt form of the smoke design and every seventh unit of the full design with all wordings (`scripts/bayesenv_tokcheck.py`, before the freeze) found no failure for either model; prompts are about 300 to 530 tokens in the toy and 500 to 880 on the ALFWorld surface (jobs 7329780 and 7330116).

The numerical rules of `notes/numerical_sensitivity.md` (section 7) bind the study:

- **One environment per comparison.** For each model every main score comes from one stack (the project's newer stack, "serve2": torch 2.14.1 with transformers 5.23.0; the run refuses another version), one GPU type (L40S), bf16 weights, the model's default attention implementation, one unpadded sequence at a time. Shards record their environment (GPU name, torch and transformers versions, revision, dtype, attention implementation, batch size, rendering mode); a resumed shard refuses a different GPU type or environment; the analysis refuses shards that differ in any of these, that are incomplete, that came from a mock run or a checkout with uncommitted changes, or that ran at different commits.
- **No bit-level criteria in bf16.** None is used.
- **Margins at least five times the numerical floor** (section 8).
- **A cross-GPU rescore of a subset** (section 14), not a same-environment rerun.
- Near-zero thresholds are reported as rates within one environment; environment diagnostics use the full subset.

## 7. Estimands

Notation: $L_t$ is the model's score for prompt type $t$ on an item, $B_t$ the exact log-odds. "Bayes value" is the value an exact reasoner gives the estimand; every residual below has Bayes value exactly 0, because its two prompts share the exact posterior. Cells are equally weighted; within a cell, items are averaged. Each estimand is computed for every framing and readout.

**Evidence use (slopes).**

- *Evidence slope* $a$: the slope of $L_{\text{explore}}$ on $B_{\text{explore}}$ over items of the 24 graded cells (least squares, items pooled). An exact reasoner up to a temperature has $a$ equal to its inverse temperature. It is the scale every other quantity is compared with.
- *History slope*: the same on the three generators with base rate 0.5 (only the history moves the exact log-odds). *Base-rate slope*: the difference between generator D and the three base-rate-0.5 generators in mean $L_{\text{explore}}$, divided by the same difference in $B_{\text{explore}}$, averaged over the graded histories other than H4 (where existence is settled and the base rate has no effect).
- *Goal slope*, *tag slope*, *note slope*: the slope of the cue's effect (goal against explore; key tag against other tag; note against explore, reliabilities 50–95%) on its exact effect, over the graded histories in which a cue can matter (H0, H2, H3, H6, H9) and every generator. The cue's **calibration ratio** is its slope divided by the base-rate slope (goal, tag) or the history slope (note); an exact reasoner up to a temperature has ratio 1.
- *Note ordering*: the 95% note's effect minus the 50% note's, with its Bayes value.
- *Seeing-guide ratio*: the mean effect of the seeing guide (guide minus goal wording 1 under the tester) divided by its exact effect; compared with $a$.

**Residuals where the exact posterior does not move** (the confirmatory family; Holm-adjusted, section 8). Each is either a **pattern contrast**, which a constant attached to one sentence or one mention cannot produce, or a **single residual**, which such a constant produces on its own:

| Plain name | Contrast (per item) | Cells | Kind |
|---|---|---|---|
| Goal minus tag ("goal role at matched information") | mean over goal wordings minus mean over tag wordings | all 28 | the goal-role contrast |
| Tag naming at zero information | key tag minus other tag, same tag wording | generator A | single |
| Competing goal | coin task with key tag minus explore with key tag | all 28 | single (task lines) |
| Name minus description | mean over key-tag wordings minus the described tag | all 28 | single (identification) |
| Zero-reliability note | note at 0% minus explore | all 28 | single |
| Redundancy residual | mean of the 50%, 80% and 95% notes minus explore, where existence is settled | H4 (4 cells) | single |
| Redundancy pattern | redundancy residual minus the zero-reliability note | as above | pattern |
| Announced goal effect | announced goal minus announced coin | 15 announced cells | single |
| Announced slope | announced goal effect under generators C and D minus under generator A | announced cells | pattern |
| History-only provenance | helper minus tester (goal wording 1) | 6 provenance cells | single |
| History-only provenance slope | helper minus tester under generator C minus under generator A (a helper read as informative, like the seeing guide, moves it; a sentence constant does not) | provenance cells | pattern |

The announced and provenance arms exist only in the world-model framing, so the state-inference family has eight members. The tag-naming and competing-goal residuals compare single wordings and are conditional on them, as are the note, announcement, provenance and existence statements.

**Descriptive estimands** (reported with intervals; not in the family): the calibration ratios (by the delta method, the two slopes treated as independent); the seeing-guide ratio divided by the evidence slope; *named-rate leakage*, explore under generator C minus explore under generator A (the same exact posterior; non-zero if the model applies the named rate to an object nothing names); the announced goal effect, history-only provenance and redundancy residual separately for each of their two statement wordings; goal minus tag in the four-prompt form,

$$\big(L_{\text{goal}} - L_{\text{explore}}\big) - \big(L_{\text{key tag}} - L_{\text{other tag}}\big);$$

 goal minus tag under generator A alone; goal minus tag by generator and its least-squares slope on $\log\Lambda_g$ (a constant goal-role term predicts no slope; a different weighting of the two routes predicts one); its heterogeneity across histories (Cochran's $Q$); the competing task without a tag (coin minus explore); goal at zero information (goal minus explore, generator A); the ALFWorld-like swap at zero information (goal minus coin, generator A); goal minus the note of matched likelihood ratio (generator C against the 80% note, D against the 95% note), with the small exact difference subtracted; in the settled history H7, the probability-scale shift of each cue; expected Brier regret $(p_M - p_B)^2$ (under both readout conventions) and the expected log-score regret

$$\mathrm{KL}(p_B \,\|\, p_M) = p_B\,\operatorname{softplus}(-L) + (1-p_B)\,\operatorname{softplus}(L) - H(p_B),$$

with $H(0) = H(1) = 0$, by prompt family, under both readout conventions. No "share of error attributable to a residual" is computed.

## 8. Inference

**Linear estimands.** Almost every estimand has the form $\hat\theta = \sum_c w_c\,\bar y_c$, a weighted sum of cell means of per-item contrasts $y_{ci}$. Its variance is

$$\widehat{\operatorname{Var}}(\hat\theta) = \sum_c w_c^2\,\frac{s_c^2}{n_c},$$

with $s_c^2$ the within-cell variance across items. Terms that share items in a cell (the redundancy pattern) are combined item by item, so their covariance is kept.

**Wording as a random effect.** Estimands that average over goal or tag wordings (goal minus tag in both forms, by generator and by history, name minus description, goal at zero information, goal minus matched note) add the sampling variance of the wordings used. On the panel cells, a cue family's scores follow

$$L_{wci} = \mu_c + u_{ci} + \alpha_w + \gamma_{wc} + e_{wci},$$

with wording effects $\alpha_w$ (variance $\sigma^2_\alpha$), wording-by-cell effects $\gamma_{wc}$ ($\sigma^2_\gamma$) and residuals $e_{wci}$. From the panel's mean squares for wordings, wording × cell and residual ($M_W$, $M_{WC}$, $M_E$; $n$ items per cell, $C_p = 8$ cells), the variance contributed by averaging $W$ wordings over $C$ cells,

$$\frac{1}{W}\Big(\sigma^2_\alpha + \frac{\sigma^2_\gamma}{C}\Big) \quad\text{is estimated by}\quad \frac{1}{W}\Big(\frac{M_W - M_{WC}}{n C_p} + \frac{M_{WC} - M_E}{n C}\Big),$$

clipped at 0 as a whole, with Satterthwaite degrees of freedom. The estimand's total variance is the item term plus one such term per side (the described tag counts as one more draw of a tag wording). Intervals use Student's $t$ with Satterthwaite degrees of freedom for the total. A planted-model test checks the coverage of these intervals when wording effects are present, and that intervals from items alone are anti-conservative there. Conclusions about the note, the announcement and the provenance statements are conditional on their wordings; the announcement, provenance and existence statements have two wordings each, alternated by item, whose separate estimates are reported, and their main effects cancel in the pattern contrasts.

**Slopes and ratios** use a stratified bootstrap over items within cells (2,000 resamples, seed 20261125).

**Multiplicity.** The residuals of section 7 form one family per model and readout (eleven in the world-model framing, eight in state inference), Holm-adjusted at two-sided 0.05. Decisions are taken on the primary readout; the same family on every other readout is reported, Holm-adjusted within that readout, as secondary.

**Pattern rule.** A *residual pattern* is detected if at least one pattern contrast (redundancy pattern, announced slope, history-only provenance slope) is rejected, in either direction; its sign is reported. A rejected single residual is reported as a *content-blind constant* (with its sign), never as a pattern.

**Minimum detectable effect.** For an estimate with standard error $\mathrm{SE}$ and $\nu$ degrees of freedom,

$$\mathrm{MDE} = \big(t_{\nu,\,0.975} + t_{\nu,\,0.80}\big)\,\mathrm{SE},$$

80% power at two-sided 5%.

**Detection targets.** A goal role worth detecting equals half a nat of exact-posterior evidence; on the model's own scale that is

$$T = 0.5\,a,$$

with $a$ the evidence slope of the same framing and readout. For pattern contrasts the target is 0.5 nats at an evidence slope of 0.7 (the size of the planted failures they are built to detect), scaled to the model's own slope,

$$T_p = 0.5\,\frac{a}{0.7}.$$

**Numerical floor and margins.** The numerical standard deviation of a mean over $N$ items is taken from `notes/numerical_sensitivity.md` (section 4.8),

$$\mathrm{SD}_{\text{num}}(N) = \sqrt{\tau^2 + \frac{D\,\sigma_c^2}{N}}, \qquad D = 1.3,$$

where $N$ is the estimand's effective item count, $N = 1/\sum_c w_c^2/n_c$ for cell weights $w_c$ and $n_c$ items, with the constants of the readout's logit-size class, judged by the median of the larger candidate logit on that readout's prompts: below 32 ("small", $\tau = 0.013$, $\sigma_c = 0.20$, as Llama-3.1-8B), 32 to 64 ("large", $\tau = 0.04$, $\sigma_c = 0.58$, as Qwen3-8B), 64 or more (twice the large class). Equivalence margins are

$$M = \max\big(T,\ 5\,\mathrm{SD}_{\text{num}}(N)\big)$$

for goal minus tag, and $\max(T_p,\ 5\,\mathrm{SD}_{\text{num}}(N))$ for the redundancy pattern and the announced slope.

**Negative decisions.** Equivalence ("smaller than the target") is declared only if all of these hold: the model is Llama-3.1-8B (the family with the small numerical floor) and the readout's measured logit-size class is "small"; the sizing rule (section 9) did not withhold negative conclusions for that estimand; the 90% interval lies inside $\pm M$; and, on the standardized scale (estimate divided by the pooled within-cell item SD of the contrast), the 90% interval lies inside $\pm 0.20$. "No residual pattern" is declared only if no pattern contrast is detected and the redundancy pattern and the announced slope are both equivalent to zero in this sense (the provenance slope's MDE is reported beside it). For Qwen3-8B a null is reported only as "not detected at an MDE of $X$ nats"; its standardized interval is reported without an equivalence label.

## 9. Smoke run and the sizing rule

**Smoke run** (per model, before its full run; items disjoint from the full run): the explore prompt in both framings on all 28 cells (10 items each); the panel cells with all eight goal and eight tag wordings, explore, coin, other tag, described tag, coin task with key tag, and the 0% and 95% notes (10 items each); the announced and provenance arms with goal wording 1 on the six provenance cells (10 items each); and the ALFWorld-surface prompt types on its cells (5 items each). In all, 9,300 sequences. A separate short job samples continuations for the reply-form gate and runs the description check.

**What the smoke readout reports** (`python -m wf.runs bayesenv_smoke_readout_<model>`): the evidence slope of every readout; each readout's gates; the median larger candidate logit and its floor class; seconds per sequence; prompt-difference noise (the item SD of goal minus explore, key tag minus other tag, and 95% note minus explore, within cell); the item SD of single prompt types (arm-specific variance); and the wording-variance components of goal and tag wordings. It reports no difference between conditions.

**Sizing rule.** With $T = 0.5\,a$ on the primary readout, the predicted MDE of goal minus tag for a design with $W_G$ goal wordings, $W_T$ tag wordings and $n$ items per cell uses the smoke's components: item term $(s^2_u + M_{E,G}/W_G + M_{E,T}/W_T)/(28\,n)$, where $s^2_u$ is the item variance of the all-wording contrast less its residual part, plus the wording terms of section 8.

1. If the base design (2 goal wordings, 3 tag wordings, 30 items per cell) reaches an MDE at or below $T$: run the base design.
2. If its MDE is above $T$ but at most $1.5\,T$: one redesign. From the menu $(W_G, W_T) \in \{(2,3), (3,4), (4,6), (6,8), (8,8)\}$ and $n \in \{30, 45, 60\}$, take the cheapest design whose predicted MDE is at most $T$, whose GPU time is at most twice the base design's, and whose projected total stays within the budget. If none qualifies, run the base design and do not use goal minus tag for a negative conclusion.
3. If its MDE is above $1.5\,T$: run the base design and do not use goal minus tag for a negative conclusion; report goal role as "not assessable at this noise" unless it is detected.

The existence-settled cells' items for the explore and note prompts, and the announced cells' items, are set to the smallest of 30, 45, 60 and 90 (and at least $n$) at which the predicted MDE of the redundancy pattern and of the announced slope is at most $T_p$, reduced step by step if the projected total would exceed the budget. A pattern contrast whose predicted MDE stays above $T_p$ is not used for a negative conclusion.

The outcome (whether goal minus tag may support a negative conclusion, and which pattern contrasts may not) is written into `wf/config/bayesenv_analysis.yaml` for that model; the analysis refuses to run without it and applies it.

**Budget.** The projected GPU time of both models' smoke, checks, full and cross-GPU runs (the cross-GPU subset priced at 1.3 times the L40S time per sequence, and 0.3 GPU-hours per model of job start-up) must not exceed 6.3 GPU-hours; Qwen3-8B is sized first assuming Llama-3.1-8B costs the same, and Llama-3.1-8B's readout counts what Qwen3-8B actually used. If the projection exceeds the budget, the full run is not submitted and the sizing is reported.

The full run's design is written into `wf/config/bayesenv_score_<model>.yaml` and `bayesenv_xgpu_<model>.yaml`, the sizing outcome into `bayesenv_analysis.yaml`, and the smoke readout into section 17 of this note, in one commit before the full run is submitted.

## 10. The specificity rule

A failure counts as **specific to predicting an action's outcome** (as against inferring the state from the same story) only if, for the estimand $\theta$ divided by the evidence slope of the same framing and readout,

$$\Delta_{\text{spec}} = \frac{\theta^{\text{WM}}}{a^{\text{WM}}} - \frac{\theta^{\text{SI}}}{a^{\text{SI}}},$$

the 95% interval of $s\,\Delta_{\text{spec}}$ lies above 0 both (i) on the yes/no readout both framings share and (ii) between each framing's question-free readout (the reply readout against the narrator readout), where $s$ is the sign of the world model's normalized estimate on that arm, so that a larger failure in either direction counts. If the world-model estimate has opposite signs on the two arms, or either arm cannot be computed (a readout failed its gate, or a slope is not positive), the result is "not assessable". Normalizing by each framing's own evidence slope keeps a framing in which the model uses evidence more weakly from appearing to fail more. The interval uses the same items in both framings (per-item differences), the wording random effect computed on the transformed values, and the slopes' uncertainty by the delta method, ignoring the covariances between the slopes and the estimands. The six specificity intervals (three estimands, two arms) are not adjusted for multiplicity. The rule is applied to goal minus tag, the zero-reliability note and the redundancy residual.

If the narrator readout fails its gates, the rule falls back to (i) plus the requirement that the reply readout's standardized effect is at least as large, in the direction of the world-model effect, as the world model's yes/no standardized effect, and the reading is labelled "question reading not excluded". "Specific" here means only "triggered by asking about an action's outcome in this story"; it does not locate anything in a model component.

## 11. Readings

Each reading is per model and readout; disagreements between readouts are listed and treated as findings, not resolved by choosing one.

| Outcome | Reading | What it does not show |
|---|---|---|
| Evidence slope's lower bound at or below 0.1 | the model does not use stated evidence on this readout; nothing else is read on it | that it cannot use evidence in other formats |
| Goal minus tag detected | the task line moves the prediction differently from a non-goal line carrying the same information, by the same route, in the same position, naming the same object | that this difference is a "goal representation"; the verb and second person of the task line are part of the contrast, bounded only by the wording spread |
| Goal minus tag detected, flat across likelihood ratios, present under generator A | a constant difference between the goal and tag lines | which component produces it |
| Goal minus tag changes with the likelihood ratio | the goal and tag routes are weighted differently | which route is miscalibrated, without the slopes |
| Goal minus tag equivalent to zero (Llama only) | no goal-role difference as large as the target | absence of smaller effects |
| A residual pattern: redundancy pattern, announced slope or history-only provenance slope detected (either sign) | a structured departure from the exact posterior that a content-blind constant cannot produce | a mechanism, or that the components fitted suffice beyond these prompts |
| Only single residuals detected (tag naming, zero-reliability note, redundancy residual, announced effect or history-only provenance; no pattern) | a content-blind shift where the posterior is unmoved, of the kind a mention or a sentence constant would produce | that it is lexical rather than semantic |
| No pattern, both pattern contrasts equivalent to zero (Llama only) | no structured residual at the target size | absence of smaller residuals |
| Seeing-guide ratio near 0 | provenance stated in the prompt is not used | — |
| Specific by the rule of section 10 | the failure is larger when predicting an action's outcome than when inferring the state | that it is located in a world-model component |
| No excess by the rule | the failure is shared by state inference in the same story | that it is general to all probabilistic reasoning |

**Interpretation limits.** Rejecting the pure accounts does not identify a mechanism or a mixture, because an account not considered may produce the same pattern. Fitted components are described as "sufficient for these prompts", nothing more. A null for Qwen3-8B is "not detected at an MDE of $X$ nats", never zero.

## 12. Planted-model checks of the analysis

`wf/bayesenv/simulate.py` generates score files with known effects (a tempered exact reasoner, scale 0.7, with item effects, noise and optional wording effects), and `tests/test_bayesenv.py` checks that the analysis recovers each planted effect on its own estimand and on no other: a goal-role term moves only goal minus tag; a lexical mention (the word "key" in a cue line) moves tag naming at zero information, name minus description, the zero-reliability note and the announced effect but not goal minus tag; a competing goal moves only the competing-goal contrast; failed redundancy screening moves the redundancy pattern; an ignored announcement moves the announced slope; an over-read history-only helper moves the provenance residual and its slope. Under the tempered exact reasoner every family residual is zero within error, the slopes recover the planted scale and the calibration ratios are 1; a deviation planted on one readout appears only on that readout; the wording random effect keeps nominal coverage when wording effects are present while item-only intervals do not; the specificity rule separates a world-model-only goal role (of either sign) from one present in both framings, and returns "not assessable" when an arm is missing; the equivalence decision needs both scales, the low-floor family and the sizing permission; the pattern rule is two-sided and never counts a single residual; the gates fail without enough sampled continuations; a planted goal role changes nothing the smoke readout reports; the sizing rule picks the base design when wording variance is small and declares goal minus tag unusable for negative conclusions when it is large. Incomplete cells, an incomplete wording panel, a cross-GPU subset that scores different prompts, and stored prompt hashes that the current code does not reproduce all stop the analysis.

## 13. The toy on ALFWorld's surface

The same generator rendered in ALFWorld's text, with the study's world-model system text and its first frozen example set: six drawers (`drawer 1` to `drawer 6`), the object "keychain", the other task's object "creditcard", ALFWorld's own action syntax and observation strings. The history lists the room ("You are in the middle of a room. Looking quickly around you, you see …", receptacles drawn per item), then either nothing (AH0) or four other drawers gone to and opened, each "In it, you see nothing." (AH3; opening never misses), then "> go to drawer 3 / You arrive at drawer 3. The drawer 3 is closed." The action is "open drawer 3". Drawers have no locks, so the outcome is whether the keychain is in drawer 3. The reply readout continues the fixed reply prefix "You open the drawer 3. The drawer 3 is open. In it, you see" and reads " a" against " nothing"; the yes/no readout asks whether the agent will see a keychain in drawer 3. The stated generator names the keychain, because ALFWorld's replies list objects by name: "a keychain was put in one of the six drawers … with probability 90% if the task line or the room tag names the keychain, and with probability 50% otherwise … Nothing else is ever put in a drawer." Every prompt on this surface therefore names the keychain in the generator, and its goal-minus-tag contrast is read as a transfer check, not as an isolation of mention.

Prompt types: the neutral task "explore the room", the creditcard task, two keychain tasks ("put a keychain in safe", "put some keychain in safe"), three keychain room tags, and a creditcard tag; generators A, C and D; histories AH0 and AH3; plus an **implicit** arm with no generator stated and no tag (neutral, creditcard and keychain tasks), which has no exact reference; it keeps the sentence saying that every action was picked at random by a tester. Without a stated generator its reply readout (" a" against " nothing") measures whether the drawer holds anything, and its effects are described that way. Reported descriptively: the evidence slope, goal minus tag (wordings treated as fixed), tag naming and goal at zero information, the goal slope, the implicit arm's swap and goal-against-neutral effects, and, side by side with the toy, the standardized values of the shared estimands.

## 14. Cross-GPU subset rescore

After each model's full run, items with index below 9 of every core cell and every ALFWorld-surface cell (324 items), with every prompt type and arm scored on them, are rescored in the same stack on A40 GPUs. The subset must be scored with the main run's design; the analysis refuses it otherwise. For every decision statistic, the estimate on this subset is computed from both environments' scores. A reading is labelled **environment-dependent** if its decision changes between the two, or if its estimate moves by more than a fifth of its main-run margin (goal minus tag: $M$; pattern contrasts: $\max(T_p, 5\,\mathrm{SD}_{\text{num}})$; other residuals: $\max(T, 5\,\mathrm{SD}_{\text{num}})$). These comparisons are descriptive; the main analysis uses the L40S scores only.

## 15. Runs, budget and order

| Run (`python -m wf.runs …`) | What | GPU | Expected cost |
|---|---|---|---|
| `bayesenv_smoke_<model>` | smoke scores, one shard | L40S | 9,300 sequences, about 0.2 GPU-h |
| `bayesenv_checks_<model>` | sampled continuations, description check | L40S | a few minutes |
| `bayesenv_smoke_readout_<model>` | section 9 readout and sizing (CPU) | none | — |
| `bayesenv_score_<model>` | full run, four shards | L40S | base design 63,360 sequences, about 1.4 GPU-h at 0.08 s each |
| `bayesenv_xgpu_<model>` | cross-GPU subset | A40 | 19,008 sequences, about 0.55 GPU-h |
| `bayesenv_analysis` | sections 7–14, both models, once, from a clean checkout at the tag (CPU) | none | — |

Order: Qwen3-8B smoke and checks, its readout, the readout committed here, its full and cross-GPU runs; then the same for Llama-3.1-8B; then the analysis. A shard can be resumed only at the commit it started from, so each model's scoring finishes before the next commit that the procedure makes. Projected total at the base design: about 4.5 to 5 GPU-hours for both models, including start-up; the budget is 6.3.

## 16. Freeze

Tag: `bayesenv-design-frozen-2026-11-25` (annotated, pushed before any model score exists). After the tag the procedure edits only: section 17 of this note; `wf/config/bayesenv_score_<model>.yaml` and `bayesenv_xgpu_<model>.yaml` (the design set by the sizing rule); `wf/config/bayesenv_analysis.yaml` (each model's sizing outcome); and `wf/config/bayesenv_smoke_readout_llama3.1-8b.yaml` (the GPU-hours Qwen3-8B used). Any other change is logged in `notes/deviations.md` with its date before the affected data are read. The analysis refuses to run from a checkout that does not contain the tag.

## 17. Smoke readout

### 17.1 Qwen3-8B (recorded 2026-11-26, before its full run was submitted)

**What ran.** The smoke scores (job 7330239, 9,300 sequences, L40S, serve2 stack, plain text, at the freeze tag) took 492 s, 0.053 s per sequence. The checks job (7330244) sampled continuations and ran the description check. The readout (job 7347289, at the commit of deviation D-BE1; an earlier attempt, 7347247, stopped at the faulty hash check before computing anything) reports only slopes, gates, variance components and logit sizes. No difference between conditions was computed.

**Readout gates.** The evidence slope is the slope of the model's log-odds on the exact log-odds for the neutral prompt over the 24 graded cells; the mass is the median probability on the two candidate words; the form shares come from the sampled continuations.

| Readout | Evidence slope (SE) | Median candidate mass | Continuations starting with the first / second word / neither | Success form after the first word; failure form after the second | Gate |
|---|---|---|---|---|---|
| World model, reply first token | −0.17 (0.03) | 0.52 | 179 / 35 / 186 of 400 | 0.94; 1.00 | fails: negative slope |
| World model, yes/no | 0.20 (0.01) | 0.996 | (not sampled) | — | passes |
| State inference, narrator | 0.53 (0.01) | 0.013 | 7 / 0 / 393 of 400 | 0.14; none | fails: mass, form and "other" rate |
| State inference, yes/no | 0.63 (0.01) | 0.56 | (not sampled) | — | passes |
| ALFWorld surface, reply | −0.11 (0.12) | 0.997 | 368 / 31 / 1 of 400 | 0.86; 1.00 | fails: slope |
| ALFWorld surface, yes/no | −0.34 (0.09) | 0.99 | (not sampled) | — | fails: slope |

Every readout's median larger candidate logit is below 32 (21.5 to 30.8), so every readout is in the "small" floor class. The description check passed on all three phrasings (Yes minus No: 2.1, 1.2 and 1.0 nats).

**Consequences under the frozen rules.**

- The reply readout fails competence: its log-odds fall as the exact success probability rises. The world model's yes/no readout therefore becomes the primary readout. The reply, narrator and ALFWorld-surface readouts are reported descriptively and decide nothing.
- With the narrator and reply readouts both failing, neither the question-free arm of the specificity rule nor its fallback (which needs the reply readout) can be applied. The specificity rule will read "not assessable" for this model unless the full run's gates differ.
- The detection target on the primary readout is $T = 0.5 \times 0.200 = 0.100$ nats. The pattern target is $T_p = 0.5 \times 0.200 / 0.7 = 0.143$ nats.

**Noise and wording variance on the primary readout (world model, yes/no).**

- Item SD of within-item contrasts: 0.28 for goal against explore, 0.27 for key tag against other tag, and 0.21 for the 95% note against explore.
- Wording components: goal wordings $\sigma^2_\alpha = 0.014$ and $\sigma^2_\gamma = 0.064$; tag wordings $\sigma^2_\alpha = 0.077$ and $\sigma^2_\gamma = 0.113$; residual 0.018 and 0.020.
- The shared wording effects therefore dominate the item noise.

The other readouts' components are in the readout's `metrics.json`.

**Sizing rule.**

- The base design's predicted MDE for goal minus tag is 0.60 nats, which is 6.0 times the target. That is band 3: "above 1.5 times the target". The base design runs (2 goal wordings, 3 tag wordings, 30 items per cell), and goal minus tag is not used for a negative conclusion.
- No menu design would reach the target: 8 goal and 8 tag wordings still give 0.36.
- The predicted MDEs of the redundancy pattern (0.06) and the announced slope (0.11) are below $T_p$ at 30 items, so their cells keep 30 items and neither is withheld from negative conclusions. Equivalence is allowed only for Llama-3.1-8B in any case.
- Projected GPU time: 1.30 GPU-hours for Qwen3-8B's full and cross-GPU runs. The projected total for both models is 3.46 GPU-hours, within the 6.3 budget.

**Configurations written with this readout.** `wf/config/bayesenv_score_qwen3-8b.yaml` and `bayesenv_xgpu_qwen3-8b.yaml` keep the base design. `bayesenv_analysis.yaml` records Qwen3-8B's sizing outcome: goal minus tag not allowed for a negative conclusion; no pattern contrast withheld.

### 17.2 Llama-3.1-8B (recorded 2026-11-26, before its full run was submitted)

**What ran.**

- Smoke scores: job 7347335, 9,300 sequences, L40S, serve2 stack, chat template, at commit 9c71355, whose code is identical to the code used for Qwen3-8B. It ran at 0.052 s per sequence.
- Checks: job 7347336.
- Readout: job 7350407. It counts the 1.75 GPU-hours that Qwen3-8B's runs used. No difference between conditions was computed.

**Readout gates.**

| Readout | Evidence slope (SE) | Median candidate mass | Continuations starting with the first / second word / neither | Success form after the first word; failure form after the second | Gate |
|---|---|---|---|---|---|
| World model, reply first token | 0.41 (0.01) | 0.94 | 90 / 266 / 44 of 400 | 0.40; 1.00 | fails: only 40% of "You" continuations are the success reply |
| World model, yes/no | 0.17 (0.005) | 0.73 | (not sampled) | — | passes |
| State inference, narrator | 0.32 (0.02) | 0.002 | 2 / 1 / 397 of 400 | — | fails: mass, form and "other" rate |
| State inference, yes/no | 0.10 (0.009) | 0.78 | (not sampled) | — | fails: slope's lower bound 0.078 |
| ALFWorld surface, reply | 0.74 (0.30) | 0.80 | 200 / 91 / 109 of 400 | 0.91; 0.95 | passes |
| ALFWorld surface, yes/no | 0.05 (0.11) | 0.89 | (not sampled) | — | fails: slope |

Every readout is in the "small" floor class: median larger candidate logit 16.8 to 25.1.

The description check failed. Two phrasings favoured Yes (5.9 and 3.9 nats) but the third, "a house tag reads … Does the tag refer to a key?", favoured No by 0.125 nats. Name minus description is therefore not assessable for this model and leaves the Holm family before adjustment.

**Consequences under the frozen rules.**

- The world model's yes/no readout is primary.
- The detection target is $T = 0.5 \times 0.173 = 0.086$ nats, and the pattern target is $T_p = 0.123$ nats.
- The state-inference yes/no and narrator readouts both fail. Neither arm of the specificity rule can therefore be computed, and it will read "not assessable" for this model unless the full run's gates differ.
- The reply and narrator readouts decide nothing. The ALFWorld-surface reply readout passes and is reported, as the design states, as a secondary transfer check.

**Noise and wording variance on the primary readout.**

- Item SD of within-item contrasts: 0.13 for goal against explore, 0.10 for key tag against other tag, and 0.12 for the 95% note against explore.
- Goal wordings: $\sigma^2_\alpha = 0.021$ and $\sigma^2_\gamma = 0.006$. Tag wordings: $\sigma^2_\alpha = 0.004$ and $\sigma^2_\gamma = 0.011$.

**Sizing rule.**

- The base design's predicted MDE for goal minus tag is 0.35 nats, 4.1 times the target. That is band 3: the base design runs, and goal minus tag is not used for a negative conclusion. The menu's best design, 8 goal and 8 tag wordings, would give 0.18.
- The pattern contrasts' predicted MDEs are 0.03 for the redundancy pattern and 0.04 for the announced slope, both below $T_p$ at 30 items, so neither is withheld from a negative conclusion.
- Projected GPU time: 1.28 GPU-hours for Llama-3.1-8B's full and cross-GPU runs. The total for the study is 3.46 GPU-hours, within the 6.3 budget.

**Configurations written with this readout.**

- `wf/config/bayesenv_score_llama3.1-8b.yaml` and `bayesenv_xgpu_llama3.1-8b.yaml` keep the base design.
- `bayesenv_analysis.yaml` records Llama-3.1-8B's sizing outcome: goal minus tag not allowed for a negative conclusion; no pattern contrast withheld.

