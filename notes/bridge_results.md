# Natural ALFWorld bridge: results (2026-11-26)

Exploratory study with a frozen procedure. The design ([bridge_design.md](bridge_design.md)), the code and the analysis were frozen before any score at the annotated tag `bridge-design-frozen-2026-11-25` (commit `298f282`, branch `natural-bridge`), after one independent review ([bridge_review1.md](bridge_review1.md)). The smoke runs, the four scoring shards, the two cross-GPU rescores and the analysis all ran at that commit, in a clean checkout, in the serve2 stack (torch 2.14.1, transformers 5.23.0). Every number below is quoted from [bridge_metrics.json](bridge_metrics.json), a byte-identical copy of the analysis run's `metrics.json` (sha256 `c77268949bdac0fb…`). No category is re-derived and no test is added. Paragraphs marked "exploratory reading" interpret the frozen numbers and decide nothing.

The companion CPU check of assumption A1′ is in [a1prime_check.md](a1prime_check.md).

## Summary

The study asks two questions about RQ2's goal effect, the rise in a model's predicted success of an action whose failure the history already settles when the task line names the action's object (aligned) rather than another object in the room (swapped). It was run on a stratified sample of 1,500 of RQ2's 4,666 items, for Qwen3-8B in plain text and Llama-3.1-8B in its chat template, each read three ways: the two-candidate score, the first answer word, and RQ2's yes/no question.

1. **Who chose the action.** Does the effect shrink when the prompt says the action was picked at random by a tester, or by an agent that saw only the history, compared with a content-free sentence in the same place?
2. **Existence without naming the object.** Does it shrink when both prompts say "Every object any task in this house could need is present in the room", on the items where the model then says the object is present?

**Frozen outcome.**

- **Who chose the action: not readable in either model, on any readout.** The comprehension gate failed for all four wordings in both models. The positive control also failed on every readout for Llama-3.1-8B. It passed for Qwen3-8B.
  - The gate failed mainly because of the first question. That question asks whether the choice depended on anything about the room not shown; both models answer "No" after the positive control as well, so the tester and agent answers could not be told apart from it.
  - Qwen3-8B gave the stated answer to both questions on at least 86% of items for three of the four wordings.
  - Llama-3.1-8B answered "No" to the second question under almost every statement.
- **Existence settled without naming the object:**
  - **Qwen3-8B: not assessable.** Too few items (222 of 1,500, against a minimum of 300) had the existence check favour "present" under both task lines.
  - **Llama-3.1-8B** (406 items pass): on the two-candidate readout, "removes a part", 0.32 nats [0.25, 0.40], a third of the effect under a content-free sentence in the same place; on the first token and yes/no, "removes a small amount (below the margin)", 0.18 [0.13, 0.24] and 0.39 [0.28, 0.52], each about a quarter.
  - On the same items, the sentence that names the object ("There is a {object} somewhere in the room.") removes all of the effect in both models and on every readout.
- **Numerical robustness:** none of the decision statistics moved enough to change its category, gate or reading when the 300-item subset was rescored on an A100.

**Exploratory reading, in brief.** None of these numbers decides a category; they are what the gated quantities measured.

- **Provenance.** Beyond a content-free sentence, the tester and agent statements remove at most about a quarter of the effect in any model and readout, mostly much less, and nothing at all from Qwen's yes/no effect.
- **Existence.** The effect's disappearance when existence is stated needs the object to be named. A universal existence claim that does not name the object removes a third or less of it in Llama, and Qwen does not accept that claim for most items.
- **A1′.** Placement varies with the task (see the A1′ section below), but by little: the task's legitimate information about where the object is, once its existence is known, is about zero on the items in question. The effect that remains is therefore far larger than location information licenses.

## 1. What was run

**Items.** 1,500 of RQ2's 4,666 primary items, stratified by history-length bin and the fact that settles the failure, from 123 of the 125 games. A 300-item subset carries the comprehension questions, the Qwen chat-format link and the cross-GPU rescore.

**Arms**, each under both task lines.

Statements on their own line after the action:
- none;
- a random tester, two wordings;
- an agent that saw only the history, two wordings;
- the positive control, an agent "who only proposes actions that will work";
- a content-free sentence;
- an agent that sees inside every receptacle and pursues its task (reported only).

Sentences appended to the task line:
- existence without naming the object;
- the named existence sentence;
- a content-free sentence.

Exact texts are in the design note, §5.

**Readouts.** For item $i$, task line $g$ and arm $s$, write $L_i(g, s)$ for the model's log-odds of success on one readout. The goal effect under arm $s$ is

$$G_i(s) = L_i(\text{aligned}, s) - L_i(\text{swapped}, s).$$

A statement's **removal** is the mean of $G_i(\text{reference}) - G_i(s)$. The reference is the content-free sentence in the statement's position:
- after the action, for the statements about who chose it;
- in the task line, for the existence sentences.

The removal's **share** divides it by the mean effect under that reference.

**Compute.** The smoke runs were jobs 7329828 and 7329829. The main runs were jobs 7347349–7347350 (Qwen3-8B) and 7347352–7347353 (Llama-3.1-8B), all on L40S. The rescores were 7347351 and 7347354, on A100s. The analysis was job 7353808, on CPU.

Every scoring shard ran:
- at commit `298f282`, on its pinned GPU type, in serve2;
- with complete record counts, and no resumed shard.

About 3.7 GPU-hours in all, against the 2.5 planned. The smoke run projected about 3.5; the stop rule halted the study until the user approved running the frozen design unchanged.

## 2. The no-statement effect (RQ2's effect in this format, on the sample)

| Model, readout | Goal effect, nats [95% CI] | Standardized effect | Share of items moved toward success |
|---|---|---|---|
| Qwen3-8B, two-candidate | 3.26 [2.86, 3.66] | 1.15 | 0.90 |
| Qwen3-8B, first token | 1.33 [1.18, 1.48] | 1.14 | 0.88 |
| Qwen3-8B, yes/no | 1.36 [1.19, 1.52] | 0.86 | 0.84 |
| Llama-3.1-8B, two-candidate | 1.60 [1.45, 1.75] | 0.83 | 0.81 |
| Llama-3.1-8B, first token | 0.65 [0.57, 0.72] | 0.65 | 0.73 |
| Llama-3.1-8B, yes/no | 2.20 [2.02, 2.39] | 1.08 | 0.90 |

The standardized effect is the mean effect divided by the item-level standard deviation of the effects. These values match the earlier measurements:
- RQ2's Llama-3.1-8B effect was 1.65 nats.
- Qwen3-8B's plain-text audit gave 3.26 nats and a standardized effect of 1.15 on 300 items.

So the sample and format reproduce the effect the bridge is meant to explain.

The first-token mass (the probability on the two first words together) has medians, across arms and task lines:
- 0.79–0.96 for Qwen in plain text;
- 0.73–0.86 for Llama.

No arm falls below the 0.5 flag. The yes/no mass is 0.99–1.00 everywhere.

## 3. Who chose the action

### 3.1 Gates

**Comprehension.** On the 300-item subset, after each statement, both models answered two questions. A wording passes if, on each question:
- at least 80% of items favour the stated answer;
- at least 80% of items answer on the stated side of the same item's answer after the positive control, whose stated answers are the opposite.

The first question (C1) asks whether the choice depended on anything about the room not shown; the stated answer is No. The second (C2) asks whether the action could have been chosen even if it fails; the stated answer is Yes.

| Wording | Qwen3-8B: C1 stated / beyond control | Qwen3-8B: C2 stated / beyond control | Llama-3.1-8B: C1 stated / beyond control | Llama-3.1-8B: C2 stated / beyond control |
|---|---|---|---|---|
| tester, wording A | 0.99 / 0.06 | 1.00 / 1.00 | 1.00 / 0.51 | 0.14 / 0.98 |
| tester, wording B | 0.86 / 0.00 | 1.00 / 1.00 | 1.00 / 0.07 | 0.04 / 0.99 |
| agent, wording A | 0.91 / 0.00 | 0.90 / 1.00 | 1.00 / 0.55 | 0.12 / 0.95 |
| agent, wording B | 0.41 / 0.00 | 0.96 / 1.00 | 0.92 / 0.00 | 0.39 / 1.00 |

Every wording fails in both models. With no statement:
- Qwen answers Yes to C1 on 6% of items and to C2 on 43%;
- Llama answers Yes to C1 on 1% and to C2 on 43%.

After the positive control, both models answer No to C1 on every item (its stated answer was Yes). On C2 (stated answer No) they answer No on every item (Llama) and on all items but one (Qwen).

**Positive control** (the rise in predicted success over the content-free sentence, averaged over both task lines; it passes at 1 nat or more with its interval above zero):

| Model | Two-candidate | First token | Yes/no |
|---|---|---|---|
| Qwen3-8B | 4.04 [3.86, 4.21], passes | 1.87 [1.80, 1.94], passes | 2.61 [2.53, 2.70], passes |
| Llama-3.1-8B | 0.13 [0.05, 0.21], fails | 0.21 [0.17, 0.25], fails | 0.13 [0.09, 0.16], fails |

**Frozen categories.** All twelve provenance categories (two models, three readouts, two statement types) are "not readable: comprehension gate failed for both wordings". For Llama the positive-control gate would also have failed.

### 3.2 What was measured (descriptive; the gates make none of it readable)

The removal beyond the content-free sentence, averaged over both wordings, in nats [95% CI]. In brackets after each is its share of the effect under the content-free sentence. The margin is the size below which a removal is not taken to change what carries the effect.

| Model, readout | Random tester | History-only agent | Margin | Content-free sentence's own removal against no statement |
|---|---|---|---|---|
| Qwen3-8B, two-candidate | 0.35 [0.27, 0.43] (0.13) | 0.18 [0.09, 0.27] (0.07) | 0.67 | 0.60 [0.48, 0.73] |
| Qwen3-8B, first token | 0.16 [0.12, 0.20] (0.16) | 0.09 [0.06, 0.12] (0.10) | 0.40 | 0.35 [0.30, 0.41] |
| Qwen3-8B, yes/no | −0.00 [−0.04, 0.03] (0.00) | 0.04 [0.00, 0.08] (0.04) | 0.40 | 0.29 [0.25, 0.34] |
| Llama-3.1-8B, two-candidate | 0.46 [0.39, 0.54] (0.24) | 0.10 [0.04, 0.16] (0.05) | 0.49 | −0.34 [−0.41, −0.28] |
| Llama-3.1-8B, first token | 0.13 [0.09, 0.16] (0.18) | −0.06 [−0.09, −0.03] (−0.09) | 0.18 | −0.08 [−0.11, −0.04] |
| Llama-3.1-8B, yes/no | 0.49 [0.44, 0.53] (0.23) | 0.30 [0.25, 0.34] (0.14) | 0.52 | 0.11 [0.06, 0.15] |

The seeing-agent statement should increase the effect if provenance is read. Its change against the content-free sentence:

| Model | Two-candidate | First token | Yes/no |
|---|---|---|---|
| Qwen3-8B | −0.36 [−0.46, −0.27] | +0.17 [0.14, 0.20] | +0.60 [0.54, 0.66] |
| Llama-3.1-8B | −0.15 [−0.20, −0.09] | +0.11 [0.07, 0.14] | −0.01 [−0.05, 0.03] |

**Exploratory reading.**

**What the statements remove.**
- Measured against a content-free sentence in the same place, saying who chose the action removes at most about a quarter of the effect on any readout, mostly much less, and nothing at all from Qwen's yes/no effect.
- Against no statement, the removals are larger for Qwen: 0.95 nats of the 3.26 on the two-candidate readout, 0.51 of 1.33 on the first token. But most of that is what any sentence placed after the action does: the content-free sentence alone removes 0.60 and 0.35.
- Every one of these removals is below its margin.
- The seeing-agent statement does not consistently increase the effect.

The pattern is not what an account in which the model reads an unexplained action as chosen by an expert would predict. But the gates exist precisely because a null is uninterpretable without them, so this is not a finding that provenance does not carry the effect.

**Why the gate failed.**
- The second question separated the statements cleanly for Qwen: Yes after the tester and agent statements on 90–100% of items, No after the positive control on all items but one.
- The first question did not, because neither model reads "an agent who only proposes actions that will work" as depending on hidden room contents.
- That is a weakness of the gate's design, not evidence about the effect. The gate was frozen; it is reported as it fell.

**Llama.** Its comprehension and positive-control failures repeat its τ-auth pattern: it answers No to C2 under almost every statement, and statements about the chooser barely move its predictions.

## 4. Existence settled without naming the object

**The existence check.** The share of items where the model says the object is present:

| Model | Aligned task, no statement | Aligned task, unnamed sentence | Swapped task, no statement | Swapped task, unnamed sentence | Items favouring "present" under both task lines (gate: 300) |
|---|---|---|---|---|---|
| Qwen3-8B | 0.11 | 0.34 | 0.08 | 0.16 | 222, not assessable |
| Llama-3.1-8B | 0.42 | 0.49 | 0.18 | 0.29 | 406, passes |

**Removals** on those items, against the content-free sentence in the task line, in nats [95% CI]. The share is in brackets.

| Model, readout | Unnamed sentence | Category | Named sentence, same items | Margin |
|---|---|---|---|---|
| Llama-3.1-8B, two-candidate | 0.32 [0.25, 0.40] (0.34) | removes a part | 1.02 [0.86, 1.18] (1.06) | 0.24 |
| Llama-3.1-8B, first token | 0.18 [0.13, 0.24] (0.25) | removes a small amount (below the margin, not shown equivalent) | 0.82 [0.70, 0.95] (1.12) | 0.18 |
| Llama-3.1-8B, yes/no | 0.39 [0.28, 0.52] (0.25) | removes a small amount (below the margin, not shown equivalent) | 1.54 [1.34, 1.76] (0.96) | 0.40 |
| Qwen3-8B, two-candidate | −0.15 [−0.24, −0.05] | not assessable | 0.84 [0.63, 1.04] (0.89) | 0.40 |
| Qwen3-8B, first token | −0.17 [−0.22, −0.12] | not assessable | 0.66 [0.50, 0.84] (0.99) | 0.40 |
| Qwen3-8B, yes/no | 0.06 [−0.09, 0.21] | not assessable | 1.25 [1.00, 1.52] (0.73) | 0.42 |

On all 1,500 items, under the named sentence the first-token goal effect falls to 0.18 nats for Qwen (from 1.33 with no statement) and 0.04 for Llama (from 0.65). This reproduces, in this format and stack, the earlier collapse when existence is stated with the object's name.

Reported beside Llama's removal on the two-candidate readout:
- 0.62 [0.48, 0.78] on the 138 items where the sentence turned the swapped prompt's existence check from "absent" to "present";
- 0.39 [0.30, 0.49] where the history does not name the object;
- 0.00 [−0.11, 0.10] where it does.

**Against A1′.** On the take items that pass the existence check (177 for Llama, 78 for Qwen), the mean legitimate index is:
- 0.01 nats for Llama and −0.13 for Qwen.

It is the change in the log-probability that the object is at the action's receptacle, plus whether the named instance exists, given existence ([a1prime_check.md](a1prime_check.md)). The goal effect that remains under the unnamed sentence on those items is:
- for Llama, 1.05 [0.79, 1.30] on two-candidate, 0.75 on first token, 2.03 on yes/no;
- for Qwen, 1.49, 1.12 and 1.99.

**Readouts.** For Llama the categories differ: a part on the two-candidate score, a small amount on the other two. The three estimates are close in share (a quarter to a third), and two of them sit within a hundredth of a nat of their margins. So the disagreement is about where a one-quarter removal falls against a one-quarter margin, not about direction.

**Exploratory reading.**

**Llama.** Stating, without naming the object, that everything any task could need is present removes a quarter to a third of the effect, concentrated where the statement changed what Llama believes about existence and where the history did not already name the object. Naming the object removes all of it on the same items. By the design's reading table:
- the existence information the task line carries accounts for some of the effect;
- the collapse seen when existence is stated needs a reference to the object, which this contrast cannot attribute to the entity, the role or the name itself.

Under ALFWorld's generator, the task carries almost no legitimate information about where the object is once its existence is known (index about zero on these items). So the effect that remains is not licensed by location information; it is many times larger.

**Qwen.** It mostly does not accept the universal claim. Even under it, it says the object is absent on two thirds of aligned prompts. On the items where it does accept it, the claim removes nothing.

These readings describe how the effect responds to the sentences. They do not identify the mechanism, and rejecting a pure existence account does not identify a mixture.

## 5. Cross-GPU rescore (Rule 4)

The 300-item subset was rescored in the primary format on an A100, in the same stack. Every decision statistic was recomputed with the same resamples:
- the comprehension gates;
- the existence-check counts;
- the three removals with their categories;
- the positive control;
- the no-statement effect.

**No statistic is environment-dependent in either model.** Removals shifted by at most 0.055 nats; no-statement effects by at most 0.033. Every category and gate is the same on both GPU types.

## 6. Qwen3-8B in RQ2's chat format (descriptive link, 300 items)

On the subset, the goal effect with the empty think block is:
- 5.95 nats on the two-candidate readout (plain text 3.22), standardized effect 1.09 in both formats;
- 2.33 on the first token (plain 1.26);
- 2.56 on yes/no (plain 1.23).

Every gate falls the same way in both formats:
- the comprehension gate fails for all four wordings;
- the positive control passes on every readout;
- the existence check is not assessable.

The provenance removals in that format are also small: the tester statement's share is 0.08 on two-candidate, 0.04 on the first token and 0.03 on yes/no. So the format that produced the RQ2 headline does not change the picture.

## 7. What this does not show

- **Provenance.** Because the gates failed, nothing here shows whether provenance carries the effect or does not. The descriptive removals are small, but a removal is readable only for a model shown to take in the statement.
- **Existence (Qwen).** The question was not assessable for Qwen3-8B.
- **Existence (Llama).** Its result is a partial removal by a universal claim that the model accepts on a quarter of items under both task lines; it describes those items.
- **Statements are stipulations.** They are about a hypothetical chooser. The real items were written by a collector that saw the room.
- **Scope.** Two models, one environment, RQ2's determined failures only.
- **Code path.** All scores come from one cached shared-prefix scoring routine. It equals a full forward pass to rounding, but is a different code path from other studies, so comparisons with other studies in nats are descriptive.

## 8. Files

- [bridge_design.md](bridge_design.md): design and smoke readout.
- [bridge_review1.md](bridge_review1.md): the review.
- [bridge_metrics.json](bridge_metrics.json): metrics.
- [a1prime_check.md](a1prime_check.md), [a1prime_metrics.json](a1prime_metrics.json) and `a1prime_items.jsonl`: the A1′ check.
- Code: `wf/bridge.py`, `wf/bridge_analysis.py`, `wf/scoring/prefix.py`, `scripts/a1prime_check.py`, `scripts/bridge_token_check.py`.
- Tests: `tests/test_bridge.py`.
- Run folders: `bridge_smoke_*`, `bridge_score_*`, `bridge_rescore_*` and `bridge_analysis`, in the ICE checkout `text-wm-bridge`.
