# Lexical repetition, semantic identity, existence content and task relevance: results (2026-11-26)

Exploratory study with a frozen procedure. The design, its rules and its analysis were frozen at tag `lexsem-design-frozen-2026-11-26` (commit 46e255b) before any effect was scored. Every reading below is quoted as the frozen rules (`wf/lexsem_rules.py`, `wf/lexsem_analysis.py`) produced it. Any wording that goes beyond a rule's output is marked as description or interpretation. Design: `notes/lexsem_design.md`. Deviations: `notes/lexsem_deviations.md` (D1).

## 1. What was done and what it asks

Earlier work found that a sentence naming the action's object can reproduce much of the goal effect. The sentence appended to the swapped task was "There is a saltshaker somewhere in the room.", and it named the object without making it the goal. The *goal effect* is how much more a model expects an action on the goal object to succeed when the task is about that object (the aligned task) than when the task is swapped for another object. On Qwen3-8B with the empty think block, that sentence reproduced nearly all of the goal effect after the first step, but only about 29% of it at step 0. This study asks why a mere mention works. It compares four explanations:

1. Lexical repetition: the object's name string appearing again in the prompt.
2. Semantic identity: the object's concept being activated, whatever words are used.
3. Existence content: what the sentence asserts about the object, read as evidence that it is present.
4. Task relevance: the object being made to matter to the task.

Each *condition* adds one sentence to the swapped task's prompt. Each condition has a *control* that adds the same sentence about a different object (the control object, matched for token count). The effect of a condition is always measured net of its control. The sentences are:

- the plain assertion that the object is in the room (the reference sentence);
- the object's name inside a sentence that only lists it as a word ("The following word appears in this note: …"; the *listed word*);
- "Ignore the X.";
- the name with an assertion that the object is somewhere else;
- a denial that the object is in the room;
- a description of the object instead of its name, asserted or listed;
- a variant name or an alias;
- a homonym used in another sense;
- the assertion placed before the history or after the action instead of on the task line;
- the assertion attributed to a reliable source and to an unreliable one ("a tester who never looked in this room … flipping a coin");
- a *competition line*: the real task plus an assertion about the control object.

The items were 450 failed actions from RQ2's determined failures, 300 at later steps and 150 at step 0. Steps are counted from the start of the episode, and the two groups are analysed separately as *strata*. Ten *arms* were scored, each an unpadded, one-sequence-at-a-time scoring run of one model in one prompt format:

- Qwen3-8B, Qwen3-4B and Qwen3-14B, each in plain text (no chat template) and in Qwen3's chat template with thinking off, which ends in an empty think block ("block");
- Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct in their chat templates (yes/no readout only);
- Qwen3-8B-Base and Llama-3.1-8B base in plain text.

Each arm was also rescored on a second GPU type (an A40) on its first 100 items, to measure the numerical floor. Three readouts were used:

- the *first-token* readout: log p of the first token of the success text (" You") minus log p(" Nothing") at the reply's first position;
- the *yes/no* readout: log p("Yes") minus log p("No") after rq1v's anchored question;
- in plain text, the *two-candidate* readout: the log-probability of the whole success text minus that of "Nothing happens.".

The quantities, for item $i$ with readout value $y_i$, are the goal effect

$$G = \frac{1}{n}\sum_i \big(y_i(\text{aligned}) - y_i(\text{swapped})\big),$$

the net effect of condition $c$ against its control $k(c)$,

$$\Delta_c = \frac{1}{n}\sum_i \big(y_i(c) - y_i(k(c))\big),$$

the *mention lift* (the reference sentence's share of the goal effect)

$$A = \frac{\Delta_{\text{name assertion}}}{G},$$

and each sentence's *relative lift* against the reference sentence

$$\rho_c = \frac{\Delta_c}{\Delta_{\text{name assertion}}}.$$

All intervals are 95% game-clustered bootstrap intervals (2,000 draws). A relative lift is placed in zones: below zero (under −0.25), about zero, partial (0.25 to 0.75), full (0.75 to 1.25) and above (over 1.25). Its category is every zone its interval touches. Each account predicts a zone for each contrast, and the rules mark the account consistent, contradicted or undecided for that contrast. Before anything is read, gates apply:

- Gate 1 (goal effect): $G$'s interval must lie above zero with its lower end at least half the estimate, and a quarter of $G$ must be at least five times the measured numerical floor. Otherwise the cell is reported in nats only.
- Gate 2 (readout mass): the first-token candidates must hold at least 0.7 of the probability, and lie within 0.1 of the swapped task's mass. A failing cell is read on the unnormalized log p(" You") instead and flagged.
- Gate 3: the mention lift must be at least 0.25 for anything to be decomposed.
- Gate 6 (recognition): description and alias contrasts are read only on items whose description or alias the model recognised.
- Base models also pass a competence gate under a neutral task before anything of theirs is read.

## 2. Provenance

All ten arms and ten floor rescorings ran at the tag from a clean checkout on ICE (L40S for the arms, A40 for the floor rescorings), in the `serve2` stack (torch 2.14.1+cu126, transformers 5.23.0, Python 3.11.18), bf16, each model at its pinned revision. Jobs 7353885 to 7353904 and, for the Llama base arm after D1, 7360053. Each arm has 450 items, each floor rescoring 100, and each base arm 292 competence successes, all at the tag commit. The analysis ran once from the same clean checkout (CPU job 7365054), checking the commit and environment of every score file. Its output, `runs/lexsem/analysis.json` on ICE, has sha256 7cb64af6f0992fe2048365b21d63967fefe0766dbdbac88533b19fdffe7b5205. A gzip copy is committed as `notes/lexsem_metrics.json.gz` (sha256 605831cb…). GPU use was about 265 L40S-minutes and 34 A40-minutes, about 5.0 GPU-hours. The smoke projected about 4.6; Qwen3-14B plain took 64 minutes against 57 projected.

The measured numerical floors (A40 against L40S, same prompts) were 0.015 to 0.074 nats per arm, except Qwen3-4B with the block at 0.22 nats. The archived stage-0 scores (original stack) against their rescoring here differ by +0.01 nats in the goal effect at both strata (intervals −0.04 to +0.06 later, −0.05 to +0.09 at step 0). The mention effect differs by +0.03 and +0.00. The frozen reading at both strata is "within the numerical allowance of a change of environment" (±0.12 nats). Per score, the mean absolute difference is 0.24 nats, but it cancels in the paired effects.

## 3. The frozen readings, arm by arm

The table gives the goal effect in nats and standardized (the mean paired difference divided by its item-level standard deviation), the mention lift $A$, and the single-account reading for every arm, readout and stratum that was scored. "No single account fits" is short for the rule's full wording: "no single proposed account fits; mixtures and unlisted mechanisms are not distinguished". "Not distinguished between lexical and semantic" is short for "not distinguished between lexical and semantic; a mechanism not on the list with the same predictions is not excluded".

| Arm | Readout | Stratum | Goal effect, nats | Standardized | Mention lift (share) | Reading |
|---|---|---|---|---|---|---|
| Qwen3-8B plain | first token | later | 1.40 (1.24 to 1.56) | 1.29 (1.11 to 1.52) | 1.29 (1.19 to 1.40) | no single account fits |
| Qwen3-8B plain | first token | step 0 | 1.78 (1.59 to 1.97) | 1.49 (1.30 to 1.72) | 0.79 (0.69 to 0.89) | no single account fits |
| Qwen3-8B plain | yes/no | later | 1.24 (1.08 to 1.41) | 1.10 (0.94 to 1.29) | 1.21 (1.07 to 1.37) | no single account fits |
| Qwen3-8B plain | yes/no | step 0 | 3.39 (3.03 to 3.73) | 1.70 (1.50 to 1.94) | 0.63 (0.56 to 0.72) | no single account fits |
| Qwen3-8B plain | two-candidate | later | 3.59 (3.17 to 4.02) | 1.32 (1.12 to 1.58) | 0.98 (0.91 to 1.06) | no single account fits |
| Qwen3-8B plain | two-candidate | step 0 | 2.89 (2.53 to 3.26) | 1.49 (1.31 to 1.76) | 0.73 (0.64 to 0.82) | no single account fits |
| Qwen3-8B block | first token | later | 2.30 (1.93 to 2.68) | 0.85 (0.74 to 0.99) | 1.46 (1.27 to 1.69) | no single account fits |
| Qwen3-8B block | first token | step 0 | 7.87 (6.99 to 8.68) | 1.55 (1.34 to 1.79) | 0.63 (0.55 to 0.73) | no single account fits |
| Qwen3-8B block | yes/no | later | 2.40 (2.14 to 2.65) | 1.24 (1.06 to 1.48) | 1.02 (0.92 to 1.14) | no single account fits |
| Qwen3-8B block | yes/no | step 0 | 7.29 (6.48 to 8.12) | 1.46 (1.32 to 1.65) | 0.25 (0.19 to 0.32) | not assessable (mention lift below gate 3) |
| Llama-3.1-8B chat | yes/no | later | 2.19 (1.93 to 2.45) | 1.10 (0.98 to 1.23) | 0.93 (0.84 to 1.03) | no single account fits |
| Llama-3.1-8B chat | yes/no | step 0 | 3.36 (2.87 to 3.81) | 1.37 (1.17 to 1.62) | 0.90 (0.81 to 0.99) | no single account fits |
| Qwen3-4B plain | first token | later | 1.39 (1.20 to 1.57) | 1.07 (0.91 to 1.25) | 1.30 (1.15 to 1.48) | no single account fits |
| Qwen3-4B plain | first token | step 0 | 2.46 (2.02 to 2.91) | 0.97 (0.81 to 1.17) | 1.33 (1.17 to 1.54) | no single account fits |
| Qwen3-4B plain | yes/no | later | 1.12 (0.89 to 1.37) | 0.66 (0.53 to 0.80) | 1.70 (1.41 to 2.09) | no single account fits |
| Qwen3-4B plain | yes/no | step 0 | 3.93 (3.35 to 4.54) | 1.22 (1.05 to 1.43) | 0.87 (0.78 to 0.97) | no single account fits |
| Qwen3-4B plain | two-candidate | later | 2.47 (2.10 to 2.84) | 1.03 (0.87 to 1.22) | 1.23 (1.12 to 1.38) | no single account fits |
| Qwen3-4B plain | two-candidate | step 0 | 2.60 (2.16 to 3.02) | 1.02 (0.85 to 1.23) | 1.29 (1.12 to 1.49) | no single account fits |
| Qwen3-4B block | first token | later | 5.32 (4.30 to 6.28) | 0.80 (0.64 to 0.98) | 1.52 (1.31 to 1.80) | no single account fits |
| Qwen3-4B block | first token | step 0 | 10.91 (9.35 to 12.56) | 1.06 (0.92 to 1.25) | 0.95 (0.82 to 1.11) | no single account fits |
| Qwen3-4B block | yes/no | later | 0.72 (0.06 to 1.44) | 0.12 (0.01 to 0.21) | 2.46 (1.07 to 12.51) | not read (gate 1) |
| Qwen3-4B block | yes/no | step 0 | 12.17 (10.04 to 14.32) | 0.87 (0.76 to 0.99) | 0.67 (0.52 to 0.83) | not distinguished between lexical and semantic |
| Qwen3-14B plain | first token | later | 0.64 (0.56 to 0.72) | 1.09 (0.96 to 1.25) | 0.82 (0.74 to 0.92) | not read (gate 1) |
| Qwen3-14B plain | first token | step 0 | 0.89 (0.80 to 0.99) | 1.62 (1.41 to 1.91) | 0.93 (0.81 to 1.06) | not read (gate 1) |
| Qwen3-14B plain | yes/no | later | 1.47 (1.26 to 1.69) | 0.95 (0.78 to 1.22) | 1.33 (1.18 to 1.50) | no single account fits |
| Qwen3-14B plain | yes/no | step 0 | 5.18 (4.72 to 5.61) | 1.82 (1.64 to 2.07) | 0.72 (0.65 to 0.80) | no single account fits |
| Qwen3-14B plain | two-candidate | later | 0.92 (0.71 to 1.15) | 0.63 (0.50 to 0.77) | 0.96 (0.82 to 1.15) | not read (gate 1) |
| Qwen3-14B plain | two-candidate | step 0 | 1.47 (1.09 to 1.83) | 0.76 (0.59 to 0.95) | 1.52 (1.27 to 1.93) | no single account fits |
| Qwen3-14B block | first token | later | 3.71 (3.26 to 4.17) | 1.09 (0.93 to 1.27) | 0.61 (0.52 to 0.71) | not distinguished between lexical and semantic |
| Qwen3-14B block | first token | step 0 | 5.93 (5.30 to 6.56) | 1.47 (1.33 to 1.67) | 0.54 (0.44 to 0.65) | not distinguished between lexical and semantic |
| Qwen3-14B block | yes/no | later | 1.62 (1.11 to 2.19) | 0.42 (0.32 to 0.53) | 1.66 (1.26 to 2.30) | no single account fits |
| Qwen3-14B block | yes/no | step 0 | 14.60 (12.78 to 16.33) | 1.31 (1.17 to 1.49) | 0.61 (0.53 to 0.70) | no single account fits |
| Llama-3.2-3B chat | yes/no | later | 0.68 (0.58 to 0.78) | 1.04 (0.92 to 1.18) | 1.05 (0.98 to 1.13) | no single account fits |
| Llama-3.2-3B chat | yes/no | step 0 | 0.76 (0.67 to 0.85) | 1.37 (1.23 to 1.57) | 1.30 (1.21 to 1.41) | no single account fits |
| Qwen3-8B-Base plain | first token | later | 0.20 (0.14 to 0.27) | 0.47 (0.30 to 0.65) | 0.96 (0.71 to 1.31) | not read (gate 1) |
| Qwen3-8B-Base plain | first token | step 0 | 0.23 (0.15 to 0.32) | 0.53 (0.37 to 0.71) | −0.11 (−0.54 to 0.15) | not read (gate 1) |
| Qwen3-8B-Base plain | two-candidate | later | 0.44 (0.35 to 0.53) | 0.67 (0.52 to 0.85) | 0.92 (0.74 to 1.12) | not read (gate 1) |
| Qwen3-8B-Base plain | two-candidate | step 0 | 0.33 (0.24 to 0.44) | 0.61 (0.45 to 0.77) | −0.21 (−0.59 to 0.04) | not read (gate 1) |

**What the table shows.** Thirty cells pass gate 1, outside the base arms. The rules never pick out one account in any of them. "No single account fits" is the reading in 26 of the 30. In three more, the rules cannot tell lexical from semantic: Qwen3-14B with the block (first token, both strata) and Qwen3-4B with the block (yes/no, step 0). In the last, Qwen3-8B with the block (yes/no, step 0), the mention lift is below gate 3, so nothing is decomposed. The block arms carry only the name sentences (assertion, listed word, "Ignore the X", denial), so fewer contrasts are available to separate the accounts there. The task-relevance account, as the sole explanation, is "contradicted" in every readable cell except Qwen3-8B with the block at step 0 in yes/no ("undecided"). As the rules require, these rejections of the pure accounts identify neither a mechanism nor a mixture. A mechanism not on the list could produce every pattern below.

Three cells fail gate 1 for different reasons. For Qwen3-14B in plain text, the first-token and later-step two-candidate goal effects are small in nats (0.64 to 0.92), so a quarter of the effect is under five floors, even though the standardized effects are ordinary. For Qwen3-4B with the block, the later yes/no goal effect is too imprecise (interval 0.06 to 1.44 nats). The base models' cases are in section 6.

## 4. The contrasts behind the readings (Qwen3-8B and Llama-3.1-8B)

The two tables give each sentence's relative lift against the plain assertion, with its zones. "Fallback" marks a cell read on log p(" You") under gate 2 (section 7). "Fragile" marks a reading whose interval end lies within two numerical floors of a zone boundary. All of these are net of each sentence's control object.

Qwen3-8B plain text, first-token and yes/no readouts:

| Sentence | First token, later | First token, step 0 | Yes/no, later | Yes/no, step 0 |
|---|---|---|---|---|
| listed word | 0.74 (0.69 to 0.78) full/partial, fragile | 0.87 (0.79 to 0.97) full, fragile | 0.82 (0.75 to 0.92) full/partial, fragile | 1.23 (1.14 to 1.34) above/full |
| "Ignore the X" | 0.67 (0.63 to 0.73) partial, fragile | 0.76 (0.66 to 0.86) full/partial | 0.63 (0.55 to 0.72) partial, fragile | 0.81 (0.72 to 0.90) full/partial, fragile |
| elsewhere | 0.71 (0.67 to 0.74) partial, fallback, fragile | 0.33 (0.22 to 0.42) about zero/partial, fallback, fragile | 0.51 (0.44 to 0.56) partial | 0.41 (0.33 to 0.47) partial |
| denial | −0.25 (−0.38 to −0.13) about zero/below zero, fallback | −3.65 (−4.48 to −3.04) below zero, fallback | −0.35 (−0.48 to −0.23) about zero/below zero, fragile | −0.86 (−1.10 to −0.67) below zero |
| description assertion | 0.31 (0.26 to 0.36) partial, fragile | 1.12 (0.95 to 1.33) above/full | 0.57 (0.49 to 0.66) partial | 1.28 (1.13 to 1.45) above/full |
| description listed | 0.21 (0.17 to 0.25) about zero/partial, fragile | 1.23 (1.05 to 1.47) above/full | 0.62 (0.51 to 0.75) partial, fragile | 1.60 (1.39 to 1.83) above |
| variant name | 0.85 (0.77 to 0.90) full, fragile | 1.03 (0.84 to 1.25) full, fragile | 1.00 (0.89 to 1.11) full | 1.32 (1.15 to 1.70) above/full, fragile |
| homonym | 0.12 (0.05 to 0.18) about zero | 0.31 (0.17 to 0.41) about zero/partial | 0.27 (0.17 to 0.38) about zero/partial | 0.37 (0.28 to 0.47) partial, fragile |
| before the history | 1.30 (1.23 to 1.38) above/full, fragile | 1.12 (1.07 to 1.18) full | 1.91 (1.75 to 2.10) above | 0.97 (0.94 to 1.01) full |
| after the action | 0.20 (0.15 to 0.24) about zero, fragile | −0.10 (−0.21 to −0.01) about zero, fragile | 1.04 (0.93 to 1.16) full | 0.52 (0.46 to 0.60) partial |
| reliable source | 1.40 (1.33 to 1.49) above | 1.62 (1.51 to 1.76) above | 2.10 (1.93 to 2.33) above | 1.60 (1.49 to 1.73) above |
| unreliable source | 0.73 (0.69 to 0.77) full/partial, fallback, fragile | 0.98 (0.90 to 1.08) full, fallback | 1.02 (0.94 to 1.11) full | 1.39 (1.30 to 1.49) above |

The alias contrast is "not assessable" in every Qwen3-8B cell, because only 82 later and 40 step-0 items passed the alias recognition check, below the thresholds of 100 and 50.

Llama-3.1-8B-Instruct, yes/no:

| Sentence | Later | Step 0 |
|---|---|---|
| listed word | 0.47 (0.42 to 0.53) partial | 0.33 (0.25 to 0.41) partial, fragile |
| "Ignore the X" | 0.30 (0.23 to 0.36) about zero/partial | −0.11 (−0.22 to −0.02) about zero |
| elsewhere | 0.14 (0.06 to 0.22) about zero | 0.02 (−0.09 to 0.10) about zero |
| denial | −0.73 (−0.90 to −0.60) below zero | −1.02 (−1.25 to −0.84) below zero |
| description assertion | 0.93 (0.82 to 1.05) full | 0.90 (0.82 to 0.99) full |

**Why the pure accounts are rejected.**

- *Lexical and semantic accounts.* Both predict that the denial lifts as much as the assertion, since it repeats the name and evokes the object. Instead the denial is below zero or about zero in every Qwen3-8B and Llama cell, and far below zero at step 0. The "elsewhere" sentence also contradicts them, because it lifts only partly.
- *Evidence (existence) account.* It predicts that the listed word, which asserts nothing, does not lift. Instead the listed word lifts most of the way in Qwen3-8B (0.74 to 1.23 across readouts and strata) and partly in Llama (0.47 and 0.33).
- *Relevance as the sole account.* It predicts that a non-goal sentence reproduces nothing. Instead the plain assertion reproduces the whole goal effect at later steps in Qwen3-8B plain text. The reading is "a non-goal assertion reproduces the goal effect", with $A$ at 1.29 on the first token, 1.21 on yes/no and 0.98 on the two-candidate readout.

**Content readings.**

- *Homonym.* At later steps in Qwen3-8B plain text on the first token, the rule reads "the lexical form without the intended sense does not lift" (0.12). In the other Qwen3-8B cells the homonym reading is "undecided", except at step 0 on yes/no, where it reads "the lexical form lifts without the intended sense in context". In Qwen3-4B plain text it reads "the lexical form lifts without the intended sense in context" on the first-token and two-candidate readouts at both strata, and "undecided" on yes/no.
- *Stated reliability* (yes/no, net of each sentence's control object; scope: an ordering by stated reliability at most). At later steps the reading is "the reliable source lifts more" for Qwen3-8B, Qwen3-4B and Qwen3-14B. At step 0 it is "no reliability effect beyond 0.25" for all three. On the first token, the frozen analysis reports this contrast as "not scored" in the Qwen3-8B and Qwen3-4B plain arms, because one of its sentences fell back under gate 2 (section 7). It is read on log p(" You") in the unnormalized readout: "the reliable source lifts more" at both strata.
- *Competition line.* In every readable Qwen3-8B cell the reading is "no competition beyond 0.25 of the goal effect", with two exceptions: yes/no at step 0 in plain text is "inconclusive", and yes/no at step 0 with the block reads "competition: naming another object after the real task lowers its reading".

**Cross-arm readings.**

- *"Ignore the X".* Within the Qwen3-4B and Qwen3-8B arms the reading is "contradicts relevance as a hint" in every readable cell, except the two block-format yes/no cells at step 0, which read "imprecise". Qwen3-14B differs. Plain text on yes/no reads "consistent with relevance as a hint" at later steps and "undecided" at step 0. With the block, the first token reads "consistent with relevance as a hint" at both strata, while yes/no reads "contradicts relevance as a hint". Within Llama-3.1-8B it is "undecided" at later steps and "consistent with relevance as a hint" at step 0. Within Llama-3.2-3B it is "undecided" and "contradicts relevance as a hint". Comparing the three core arms on yes/no, the cross-arm reading is "inconclusive; neither outcome identifies a mechanism" at later steps. At step 0 it is "partly block-dependent: it falls without the block but stays above Llama's; neither outcome identifies a mechanism". The relative lifts are 0.81 in plain text, 1.55 with the block and −0.11 in Llama.
- *Bridge to stage 0.* The *bridge* is the name assertion minus the plain swapped task, as a share of the goal effect, which is how stage 0 measured it. It was rescored inside this run on the same items. In yes/no at step 0 it is 0.71 (0.63 to 0.80) in plain text and 0.29 (0.23 to 0.36) with the block, a difference of 0.42 (0.38 to 0.46). The frozen reading is "plain text higher than the block format", which the design reads as "stage 0's low step-0 share belongs to the block format". At later steps the difference is 0.21 (0.11 to 0.32) and the reading is "inconclusive". So stage 0's 29% at step 0 is reproduced here with the block, and in plain text the same sentence reproduces about 71%.
- *Qwen3-8B against Llama-3.1-8B* (yes/no, same items). The contrasts whose zones do not overlap are the "elsewhere" sentence and the description assertion at later steps. At step 0 they are $A$, the listed word, "elsewhere" and "Ignore the X". "Elsewhere", the listed word and "Ignore the X" are larger in Qwen. The description assertion (0.57 against 0.93) and the step-0 mention lift (0.63 against 0.90) are larger in Llama.

The frozen file also reports the pre-specified format × meaning, size × format × meaning and model × meaning interactions (yes/no) and the action-type breakdown. In Qwen3-8B, the block-minus-plain difference relative to the assertion reads "higher" at later steps for the reference without assertion, the denial and the listed word. At step 0 it reads "inconclusive" for the reference without assertion, "higher" for the denial and "the same within 0.25" for the listed word. Against Llama, Qwen3-8B with the block reads "higher" for all three sentences at both strata. Qwen3-8B plain reads "inconclusive" for all three at later steps and "higher" for all three at step 0.

Entries that involve a cell that failed its own gate 1 are not read here, although the cross-arm code reports them. Every Qwen3-4B "block minus plain" entry at later steps in yes/no rests on the Qwen3-4B block yes/no later cell (goal effect interval 0.06 to 1.44 nats) and is not read. Its intervals run to 30 or more. The between-model and interaction entries that use the Qwen3-14B plain first-token cells, or either base arm, are not read either.

## 5. Other readouts and disagreements

Readout disagreements are contrasts whose zones in two readouts share nothing. In Qwen3-8B plain text there are few:

- the assertion after the action (first token against yes/no, both strata);
- the unreliable source at step 0 (first token against yes/no);
- "elsewhere" and the listed description at step 0 (two-candidate against yes/no).

The placement after the action is the clearest. On the first token it lifts little (0.20 and −0.10). On yes/no it lifts fully at later steps (1.04) and partly at step 0 (0.52). The design names the confound: moving the sentence after the action also places it next to the reply label. The "normalization flags" are contrasts that read differently on the normalized first token and on log p(" You"). In Qwen3-8B plain text they are the listed word and the assertion before the history at step 0.

## 6. Base models

**Qwen3-8B-Base passes the competence gate.** This gate is applied under the neutral task ("Your task is to: explore the room."). On the first token, the model separates RQ2's 292 determined successes from the 450 sampled failures with an AUC of 0.85; 30% of competence items are read as success; the mean first-token mass is 0.88. The full two-candidate readout gives an AUC of 0.84. Its goal effect is small, however: 0.20 nats on the first token at later steps and 0.23 at step 0, against a measured floor of 0.033. A quarter of the effect (0.05) is under five floors (0.16), so every base-model cell is "not read: goal-effect gate failed" and is reported in nats only. Described in nats on the same items, post-training raises the first-token goal effect from 0.20 to 1.40 nats at later steps, and from 0.23 to 1.78 at step 0. Standardized, the rise is from 0.47 to 1.29 and from 0.53 to 1.49.

**Llama-3.1-8B base fails the competence gate.** On the first token its AUC is 0.50, and 81% of items are read as success. On the two-candidate readout the AUC is 0.52 and 79% are read as success. So none of its readouts is read, and its paired comparison with Qwen3-8B-Base is not read either. The post-training pilot found the same failure in its own plain format (AUC 0.54).

**Description and alias contrasts for the Llama base arm (deviation D1).** These are missing because the frozen recognition check runs through the tokenizer's chat template, and the Llama-3.1-8B base tokenizer has none. An empty check file was placed instead, and the frozen analysis reports those contrasts as "not assessable: 0 items passed the recognition check". The arm's other contrasts were unaffected by D1. Here they are not read anyway, because the arm fails its competence gate.

The base arms' yes/no answers were recorded and never read, as pre-specified.

## 7. The gate-2 fallback cells, with and without the fallback

The all-arms smoke (5 items per arm) had flagged seven first-token cells for gate 2's fallback:

- the name denial in Qwen3-8B plain and in Qwen3-4B plain;
- the assertion with stated reliability and with stated unreliability in Qwen3-4B plain;
- the assertion after the action, in both base arms.

The flag is "normalization-dependent": the cell's first-token contrasts are read on the unnormalized log p(" You"). In the full run, gate 2 is applied per stratum on all items, and it sends the following cells to the fallback:

- Qwen3-8B plain: the denial, "elsewhere" and the unreliable source, at both strata;
- Qwen3-8B with the block: the denial at later steps;
- Qwen3-4B plain: at later steps the denial and both reliability sentences; at step 0 the denial, "elsewhere", the homonym, the unreliable source and the unreliable control sentence;
- Qwen3-14B plain: the denial at both strata;
- Qwen3-8B-Base: the assertion after the action and its control, at both strata.

The Qwen3-14B plain and base cells are not read anyway (gate 1). The two lists differ because the smoke saw five items.

To show whether any conclusion depends on the fallback, the frozen analysis was rerun once with gate 2's first-token criterion switched off, so that every cell is read on the normalized first token. Nothing else was changed: same data, bootstrap size and seed (`scripts/lexsem/sensitivity_gate2.py`, CPU job 7365123). The output is `notes/lexsem_metrics_gate2_off.json.gz` (sha256 of the uncompressed file a5f695d9…). This rerun was requested after the freeze. It is a sensitivity check, not part of the frozen readings.

| Arm, stratum | Sentence | Relative lift on log p(" You") (frozen, fallback) | Relative lift on the normalized first token (gate 2 off) |
|---|---|---|---|
| Qwen3-8B plain, later | denial | −0.25 (−0.38 to −0.13), about zero/below zero | −0.10 (−0.21 to −0.00), about zero |
| Qwen3-8B plain, later | elsewhere | 0.71 (0.67 to 0.74), partial | 0.71 (0.67 to 0.74), partial |
| Qwen3-8B plain, later | unreliable source | 0.73 (0.69 to 0.77), full/partial | 0.76 (0.73 to 0.80), full/partial |
| Qwen3-8B plain, step 0 | denial | −3.65 (−4.48 to −3.04), below zero | −2.55 (−3.15 to −2.12), below zero |
| Qwen3-8B plain, step 0 | elsewhere | 0.33 (0.22 to 0.42), about zero/partial | 0.35 (0.24 to 0.43), about zero/partial |
| Qwen3-8B plain, step 0 | unreliable source | 0.98 (0.90 to 1.08), full | 1.04 (0.95 to 1.13), full |
| Qwen3-8B block, later | denial | 0.46 (0.38 to 0.54), partial | 0.42 (0.33 to 0.51), partial |
| Qwen3-4B plain, later | denial | −0.32 (−0.49 to −0.19), about zero/below zero | −0.24 (−0.41 to −0.12), about zero/below zero |
| Qwen3-4B plain, later | reliable source | 1.73 (1.62 to 1.88), above | 2.03 (1.87 to 2.23), above |
| Qwen3-4B plain, later | unreliable source | 1.04 (0.97 to 1.12), full | 1.13 (1.05 to 1.21), full |
| Qwen3-4B plain, step 0 | denial | −1.51 (−1.91 to −1.20), below zero | −1.13 (−1.42 to −0.91), below zero |
| Qwen3-4B plain, step 0 | elsewhere | 0.45 (0.38 to 0.51), partial | 0.32 (0.24 to 0.38), about zero/partial |
| Qwen3-4B plain, step 0 | homonym | 0.38 (0.28 to 0.48), partial | 0.39 (0.28 to 0.50), partial |
| Qwen3-4B plain, step 0 | unreliable source | 0.74 (0.67 to 0.83), full/partial | 0.60 (0.54 to 0.67), partial |

**No conclusion depends on the fallback.** With gate 2 off, every single-account reading, every account's overall status, every mention reading, relevance reading, homonym reading and broad-evidence reading is unchanged, in every arm and stratum. All cross-arm readings are unchanged too. The archived stage-0 comparison uses yes/no only and was not rerun.

What does change, all below the level of the conclusions:

- *One per-contrast status.* In Qwen3-4B plain text at step 0, "elsewhere" moves from "contradicted" to "undecided" for the evidence account. That account stays contradicted by the listed word, the listed description and the homonym.
- *Some zone memberships*, as in the table: the Qwen3-8B later denial loses "below zero", and the Qwen3-4B step-0 unreliable source loses "full".
- *Two item-agreement labels*, which describe how closely a sentence's effect tracks the goal effect item by item.
- *Two readout-disagreement lists.* In Qwen3-8B plain at later steps, the denial now disagrees between the first token and the two-candidate readout. In Qwen3-4B plain at step 0, the unreliable source now disagrees between the first token and yes/no.
- *The pre-specified sensitivity without the few-shot objects.* In Qwen3-4B plain at step 0, three per-contrast statuses tighten: the denial goes from "imprecise" to a definite status for all three accounts, and the description assertion from "undecided" to "contradicted" for the evidence and semantic accounts. No overall status changes.
- *Two contrasts that the fallback had left unscored on the first token.* These become computable when every sentence is on the same readout:
  - The reliability ordering, which reads "the reliable source lifts more" in Qwen3-8B and Qwen3-4B plain text at both strata. The frozen first-token analysis reports it as "not scored"; the frozen log p(" You") and yes/no readouts give the same ordering at later steps.
  - The position contrasts in Qwen3-8B-Base, which are not read anyway (gate 1).

## 8. What this shows and what it does not

**Where the plain text and the block disagree.** The clearest new fact is about format: the step-0 weakness of the mention belongs to Qwen3's empty think block. With the block, the plain assertion reproduces 29% of the yes/no goal effect at step 0, as stage 0 found. Without the block, it reproduces 71% on the same items, and 63% of the plain-text yes/no effect by the in-run reference measure ($A$). After step 0 the assertion reproduces the whole effect in both formats. Within the run, the archived stage-0 scores and their rescoring agree to within the numerical allowance, so this difference is not an environment change.

**What the pure accounts get wrong.** The pattern of sentences is not what any single proposed account predicts:

- A sentence that merely lists the name as a word lifts most of the way in Qwen and partly in Llama. This is against the pure existence-evidence account.
- A sentence that denies the object is present pushes the prediction the other way, strongly at step 0. This is against the pure lexical and semantic accounts.
- A sentence asserting the object without making it the goal reproduces the goal effect after step 0. This is against relevance as the sole account.

The rules therefore say no single account fits, and that mixtures and unlisted mechanisms are not distinguished. In particular, the result does not show that the effect is "half lexical, half evidence". The descriptive mixture weights in the metrics file are fits of the listed accounts only, and they carry no reading.

**What differs across models and sizes.** Description, not reading:

- Qwen's listed word, "elsewhere" and "Ignore the X" all lift more than Llama's.
- Llama's description assertion lifts fully at both strata, more than Qwen's at later steps.
- Llama's "Ignore the X" at step 0 behaves as relevance-as-a-hint predicts.
- Qwen3-14B with the block falls back to "not distinguished between lexical and semantic" on the first token. That reading comes with a mention lift of only about 0.6 and a goal-role component ("the goal adds beyond a non-goal assertion").

Three sizes of one family do not establish a scaling relationship. None of this describes internal computation: these are behavioural contrasts on predictions.

**What would change the conclusion.** A single account would be read only if:

- the denial behaved like the assertion (for the lexical and semantic accounts); or
- the listed word did not lift (for the evidence account); or
- a non-goal assertion reproduced nothing (for relevance).

None of these is close in the cells that are read. The step-0 format result would be overturned by an environment difference larger than the allowance. The archived-versus-rescored comparison rules that out within this run.

**Next steps this suggests.** If the question is which mixture or which unlisted mechanism, behavioural sentence contrasts of this kind have reached their limit: they reject pure accounts but cannot identify a mixture. A mechanistic readout, for example patching the mention sentence's representation into the swapped prompt, would be the informative next measurement. For the post-training question, Qwen3-8B-Base's goal effect is real but small (0.20 nats on the first token), below the numerical rule. Reading its decomposition would need a readout with a larger effect or more precision, not more sentences.

## 9. Limits

- **Success probe, low first-token mass for "There's".** The competence set's successes are read on their own first token against " Nothing". The smoke's probe showed that for success texts starting with "There's", the own first token (" There") holds little probability: 0.03 to 0.21 in plain text and below 0.001 with the block, because the model gives that mass to " You". Those 27 successes (of 292) are therefore read on a token the model rarely predicts there. The competence gate uses the mean mass over all 742 competence items (0.88 for Qwen3-8B-Base), so this does not change a gate decision, but the probe's figures for "There's" carry little information.
- **Fallback cells.** Section 7 lists them; no reading depends on them.
- **D1.** The Llama base arm's description and alias contrasts are not assessable, because the frozen recognition check needs a chat template and the base tokenizer has none. The arm's other contrasts are unaffected by D1, but are not read because the arm fails its competence gate.
- **Cross-arm entries and gate 1.** The cross-arm code does not apply each arm's own gate 1. Entries resting on a gated-out cell are reported in the metrics file and not read here (section 4).
- **Aliases.** They are not assessable in Qwen3-8B, because too few aliases were recognised.
- **Control objects.** They are matched on token count and familiarity but not on scene prevalence: the target objects occur in 48% of scenes against 19% for the controls. The analysis reports a prevalence covariate. In Qwen3-8B plain text at later steps, the item-level correlation of the name assertion's net effect with the prevalence gap is 0.15, with an interval from −0.00 to 0.30.
- **Position contrasts.** These carry the design's named confounds: the region of the prompt changes, and the sentence after the action sits next to the reply label.
