# TextCraft round 2: design (written 2026-11-21, before any round-2 score)

This note fixes, before any model scores a round-2 record, what round 2 asks, which records and prompts it scores, which tests are confirmatory, how each result will be read, and what would count against the account it tests. Nothing in round 2 has been scored. What had been seen when this was written: the confirmatory TextCraft results ([textcraft_results.md](textcraft_results.md), frozen design [textcraft_design.md](textcraft_design.md)), the exploratory Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md), branch `textcraft-llama-dx`), the cross-environment comparison (`notes/textcraft_vs_alfworld.md`, branch `xenv-compare`), the model-families wave A results in ALFWorld ([families_results.md](families_results.md)), and, for the planning figures and the competence disclosure in section 7, the confirmatory TextCraft scoring records (read again on 2026-11-21 for item spreads and the families competence rule; no new contrast was computed on them). The round-2 records were built and checked before this note was committed (section 4 gives their counts); no model has read them.

The design must be independently reviewed and then frozen at a git tag (`textcraft-r2-design-frozen-<date>`) before any scoring. Any change after the tag is a reported deviation.

## 1. Background and the question

**Terms.** Throughout, an *item* is a TextCraft state (a displayed recipe list, a history of actions and replies) plus a test action. *L* is the log-odds of the success reply against the failure reply, both teacher-forced after the prompt (the two-candidate score), or of "Yes" against "No" after the question "Will this action succeed?" (the yes/no score). A *goal line* is the line "Goal: craft X." that the environment prints under the recipe list; the *aligned* line names the goal whose recipe tree contains the action's object, the *swapped* line names the other goal of the item's pair (whose tree is also on screen), and the *neutral* line is "Goal: explore the crafting commands.". The *goal effect* of an item is

$$\Delta L_i = L_i(\text{aligned}) - L_i(\text{swapped}).$$

A *goal-role* action crafts the aligned goal itself; an *ingredient* action crafts a direct ingredient of it. *Same-history couples* are the confirmatory design's pairs of items sharing a goal pair, a history, a failure class and a role, one acting in each goal's tree; averaging over a couple removes any effect of how well the history fits one goal or the other (design section 12, B1).

**What round 1 found.** The confirmatory TextCraft test confirmed a positive goal effect for Qwen3-8B (+0.94 nats) and found a small negative one for Llama-3.1-8B (−0.19). The exploratory diagnostics then found that Llama's negative effect sits entirely in the goal role (−0.63) and that its ingredient role is null (+0.03), while Qwen is positive on both roles under the verified prompt. Matched on failure class, length, input structure and number of short inputs, the goal role's ΔL minus the ingredient role's was about −0.8 nats in both models. Making an ingredient's own object the goal ("Goal: craft Y." for an action crafting Y) lowered Llama's L by −1.40 on failures and −1.74 on successes; for Qwen it lowered L sharply only where inputs were never obtained (−4.00) and raised it on successes (+0.80). The penalty on goal crafts was largest early, with few or none of the goal's inputs held. A world-model-only system text (the "clean" prompt, agent instructions removed) raised the goal-role effect by +0.39 (Llama) and +1.66 (Qwen) on a 500-item sample and left competence about unchanged; showing the inventory raised competence but not ΔL.

**The two-component account (stated for testing).** For an action that crafts item X under goal line g, write the shift of L against a goal line that makes X irrelevant as

$$L(a \mid g) - L(a \mid g_{\text{irrelevant}}) = \lambda_m \,[X \in \text{tree}(g)] + \nu_m \,[g \text{ names } X] + \kappa_m(h)\,[\text{crafting } X \text{ completes } g] + \varepsilon,$$

where m is the model and h the history. The three terms are:

- the *intermediate lift* λ, the shift for an action that serves the goal without naming or completing it (Qwen about +0.9, Llama about 0 in round 1);
- the *naming term* ν, the shift when the goal line names the action's object;
- the *completion term* κ, the shift when the action would finish the named task, allowed to depend on the history (on how far the task has progressed, or on how long the history is).

In TextCraft's one-item goal format, ν and κ always act together: the only actions whose object the goal names are the crafts that complete it. Round 1's "terminal-craft penalty" is their sum, and the account claims three things about it and about λ:

1. **The penalty.** The sum of the naming and completion terms is negative and of similar size across models (round 1: about −0.8 nats relative to ingredients, in both models, under the verified prompt).
2. **The lift.** The intermediate lift differs across models, and that difference, not the penalty, is why Qwen's total effect is positive and Llama's negative.
3. **Early largest.** The penalty is largest early in the game and shrinks as the history fills.

Round 2 asks four questions about the account:

- Does the penalty survive when the action, the history and the displayed list are held fixed and only the goal line changes from naming-and-completing to neither (the *parent-goal* arm)? This removes the item-structure confound of the round-1 goal-versus-ingredient comparison.
- Is the penalty a *naming* term or a *completion* term (the *two-item goal* arm)?
- Is "largest early" about *time* (how many steps the history has) or *content* (what the history has gathered) (the *padding* arm)?
- Does the picture hold in three more model families, under one clean world-model prompt?

**Confirmatory and descriptive.** The confirmatory part is five contrasts per model (section 6), each with a test of difference and a test of equivalence, in one Holm family of fixed size 50 (section 8). Everything else is descriptive: estimates with 95% intervals and no p-values, including the overall goal effect under the clean prompt.

## 2. The prompt: clean world-model text, first-class

**Decision (the owner's, 2026-11-21).** The clean world-model system text is the default prompt for every TextCraft run after the confirmatory one. The confirmatory result stays reported under the prompt it was frozen with (`prompts.SYSTEM`). A run under the clean prompt measures a related but different quantity: Qwen and Llama rescored under it are new estimates, not re-estimates of the confirmatory numbers.

**Text.** `prompts.CLEAN_SYSTEM` (moved from the diagnostics' `dx.CLEAN_SYSTEM`, byte for byte unchanged; a test checks this):

> You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next action, predict the environment's next observation exactly in TextCraft style. Output only the observation text.
>
> The environment's rules: crafting commands are of the format "craft [target object] using [input ingredients]". Each action is "get" with a quantity and an object, "inventory" to look up the inventory, or "craft" with a quantity and one of the crafting commands shown. If a crafting command uses a generic ingredient like "planks", a special type of the same ingredient, e.g. "dark oak planks", can be used instead.

It keeps the world-model framing and states only the environment's rules and the action format. Every sentence addressed to the agent is removed, and so is "from the inventory or the environment", which hinted at a hidden store. It does not say that the inventory starts empty.

**Everything else is the verified prompt.** The three verified in-context examples (`fewshot.json`, sha256 `e6ddab93…adfb1253`), the turn format ("History:", "Action:", "Next observation:"), and the yes/no question ("Question: Will this action succeed? Answer Yes or No.\nAnswer:") are all unchanged.

**Safeguards before any score** (validation standard, steps 7 and 9b; these were skipped by the diagnostic runs):

- *Golden rendered prompts per model* (`wf/textcraft/data/r2_chat_golden.json`). For five fixed record and arm combinations, the rendered text, the number of ids and the sha256 of the ids, through each model's own chat template at its pinned revision. A test checks the stored text without a tokenizer; the scoring run re-renders and compares before scoring.
- *Tokenization through each scorer's own encoding*, for every model, on every record and arm the model will score (section 3).
- *An item hash covering the prompt context.* `r2.item_hash` covers every field any arm of a record shows or scores, plus the system text, the question, the examples and the padding step. The scorer refuses saved scores whose hash differs.

## 3. Models, templates and gates

| Tag | Model | Pinned revision | Chat-template handling |
|---|---|---|---|
| `qwen3-8b` | Qwen/Qwen3-8B | b968826d9c46 | native system message; thinking off |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a1 | native system message; the template's fixed date |
| `gemma2-9b-it` | google/gemma-2-9b-it | 11c9b309abf7 | no system role: the system text opens the first user turn, then a blank line (`families.fold_system`); eager attention, for the soft-capping |
| `mistral-7b-v0.3` | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41d | native; the template places the system text in the last user turn, as shipped; replies follow "[/INST]" after a space |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4f764b | native; `date_string` fixed at "26 Jul 2024" |

These are exactly wave A's handling and pins (`wf/families.py`, `wf.constants.REVISIONS`). Llama-3.2-1B is not included: it failed wave A's competence gate in ALFWorld. All five load in the study's serve environment (transformers 4.51.3), and the weights and tokenizers are in the ICE cache at these snapshots.

**Mistral's placement.** Mistral's system text goes into the last user turn, after the examples. Wave A's placement check found that moving it to the first turn changed Mistral's ALFWorld two-candidate effect by +0.02 nats and its yes/no effect by +0.30 (12% of the effect). The native placement is kept, as in wave A.

**Tokenization gate (wave A's, audit C9), before any score and on every resume.** Through the scorer's own `_encode` and `_encode_candidate`, on every record and arm of the shard:

1. Each candidate, tokenized alone as the scorer tokenizes it, must equal the chat template's tokens for that text as the assistant's reply after that arm's prompt.
2. "Yes" and "No" must each be one token, equal to the template's rendering of that word as the reply after every yes/no prompt of the shard.
3. The two candidates' first tokens must differ.

Any failure stops the job before it writes a score. Also recorded, for information only: whether the plain tokenization of the rendered prompt equals the template's ids, and whether the prompt text and candidate joined as plain text tokenize as the prompt followed by the candidate. Mistral fails the second check by design, because the plain join has no space after "[/INST]".

**CPU check on ICE before freezing.** The same gate, run with the five pinned tokenizers on every record and arm (`wf/textcraft/r2_tokcheck.py`, job `pace/textcraft_r2_tokcheck.sbatch`, CPU partition), plus the golden renderings and the longest prompt in tokens for each model. The results are recorded in section 13 and in `wf/textcraft/data/r2_tokcheck_ice.json`. Gemma 2 attends locally within a 4,096-token sliding window on alternate layers; the check records whether any prompt exceeds it.

**Competence gate: one rule for every model, fixed now.** This is wave A's rule (`families.COMPETENCE`, `notes/families_metrics.json`, `rules.competence`), computed on the round-2 main records under the neutral line, with the yes/no question. A model is interpretable for round 2 only if all four conditions hold:

- the AUC of the yes/no L between the 600 determined successes and the 1,500 main failures is at least 0.70;
- it answers Yes (L above 0) to at most 95% of the failures;
- it answers Yes to at least 5% of the successes;
- the median of P(Yes) plus P(No) is at least 0.5.

A model that fails the gate is reported as *not interpretable* for every confirmatory contrast and every effect estimate in this note. Its estimates are shown but not read, and its tests stay in the Holm family with their p-values, so the family's size does not change.

**Role-level competence (descriptive, pre-registered).** Llama's round-1 competence differed sharply by role (two-candidate neutral AUC 0.84 on the goal role, 0.58 on ingredients). For every model, the neutral-line AUC is reported on the goal role (goal-role successes against goal-role failures) and on the ingredient role, on both scales, with pair-bootstrap intervals. It is also reported per failure class (each class against all successes) and on the parent records' successes against their failures. These readouts do not change the gate's verdict. The two-candidate neutral AUC, overall and per role, and the median first-token mass of the two-candidate score (the probability of the two candidates' first tokens) are reported too.

## 4. Records

All records are in `wf/textcraft/data/r2_items.jsonl`, built by `python -m wf.textcraft.r2_items build` from the frozen items (`items.jsonl`, sha256 `3b81ac3f…5e421a6`) and the frozen builder. There are 2,982 records: 2,100 main and 882 parent. Each model scores 24,008 (record, arm) combinations.

### 4.1 Main records and the two-item goal line

**Records.** The 1,500 frozen main failures (750 same-history couples, 229 goal pairs) and the 600 frozen determined successes (300 couples), copied field for field. A record differs from its frozen item only in three added fields: `kind`, `both_task_line` and `arms`.

**Arms.** Each of four goal lines, as the two-candidate score and as the yes/no question (eight arms):

| Line | Text | What it says about a goal-role action (crafting G) |
|---|---|---|
| aligned | "Goal: craft G." | names G; crafting G completes the task |
| swapped | "Goal: craft P." | does not name G; G is not in P's tree |
| neutral | "Goal: explore the crafting commands." | names no item |
| two-item | "Goal: craft A and B." | names G; crafting G does not complete the task, because the other item remains |

In the two-item line, A and B are the pair's two goals in sorted order. Both members of a couple therefore see the same line, and across couples the action's object is named first in 255 goal-role items and second in 255.

**Is a two-item goal faithful to the environment?** Checked against AgentGym's code at 3ef9235 (`environment.py`, `env_wrapper.py`, the client `agentenv/envs/textcraft.py`):

- *Display.* TextCraft supports one goal only. `reset(commands, goal)` takes a single item id and prints "Goal: craft {name}.". So the environment never prints a two-item line, and in that sense the line is off-template. The neutral line is off-template in the same way, and was accepted in round 1 for the same reason: it sits where the goal line sits and changes nothing else.
- *Recipe list.* What the environment would show for a two-goal task is both goals' recipe trees, which is exactly what every round-1 item already shows (the union of the pair's two trees).
- *Replies.* Replies never depend on the goal. A craft's reply is "Crafted K minecraft:Y_" whether or not Y is the goal; only the reward and the `done` flag change, and neither is part of the text the world model reads. The server wrapper returns `observation`, `reward` and `done` as separate fields, and the client passes the observation through unchanged. The verify step replays every record's history and action with the goal set to G, to P and to none, and gets identical replies. So the success and failure replies are exactly those of a two-goal game.

The two-item line is therefore faithful in list and replies, and off-template only in its wording, like the neutral line. It is kept.

**Padding arms (main failures at 1 to 8 steps: 920 records).** The history is shown with k look-ups of the inventory put before its first step, for k = 1 and k = 8. Each look-up is the action "inventory" with its reply "Inventory: You are not carrying anything.", under the aligned and the swapped lines (four arms).

*Why this no-op.* An `inventory` look-up changes nothing in the environment, its reply is the environment's own, and the round-1 histories already contain it (the walk looks at the inventory with probability 0.10 per step). Placed before the first get, its reply is always the empty-inventory text. So the eight-step and the one-step padding show the same information (that the inventory starts empty) and differ only in length: seven steps, about 100 tokens.

*Rejected no-ops.* Looking at the inventory later in the history would reveal the held state, which changes content (the diagnostics found that showing the inventory raises competence). A failed `get` of a craftable item, or a failed craft, changes nothing but adds failure text and item names, which is content. A `get` of zero items is not an action the walk ever takes.

*Checks.* Verify replays each padded history in the environment under the goal, the partner and no goal: every look-up must reply exactly the padding text, the stored history and the action's reply must follow, and the twin must reproduce. The text-only checker must account for every padded history and give the same verdict, class and short inputs.

Padding uses only main failures in the 1–3 and 4–8 step bins (168 + 170 goal-role and 170 + 412 ingredient records, 202 pairs), where the round-1 gradient lies. Eight steps move a 1–3-step history into the 9–12-step range.

### 4.2 Parent records

**Rule (fixed before the build).** For a pair goal G with partner P, a *parent* Z is an item of the frozen database such that:

1. some recipe line for Z lists G itself (not a generic group) as an input, so G is a direct ingredient of Z;
2. Z is neither G nor P, not one of the three goals the fixed instruction names, and not a goal of the three prompt-example pairs;
3. Z's recipe tree shares no craftable item with P's tree, so the added lines share no craftable with the partner's tree and the swapped line stays as clean as in round 1.

Among the qualifying Z, the parent is one that shares no word with G's name if any qualifies (rule "strict"), otherwise any qualifying Z (rule "word overlap"). Among those, the one adding the fewest lines is taken, and a draw seeded with `random.Random(f"{SEED}|textcraft|r2|parent|{pair_id}|{G}")` breaks ties.

**Extended list.** The lines of Z's tree (the environment's `traverse_recipe_tree(Z)`, without distractors) that the item's list does not already show are added. Each added line, in sorted order, is inserted at a position drawn by `random.Random(f"{SEED}|textcraft|r2|extlist|{pair_id}|{G}|{Z}")`; the list's own lines keep their order. Under the goal line "craft Z" this is what the environment would show for goal Z: Z's tree, which contains G's tree, plus P's tree as other lines. Under "craft G" or "craft P" it is a list whose extra lines look like the distractor recipes AgentGym's `reset` adds to every real game. So all three lines are faithful to some environment display.

**Items.** These are goal-role crafts of G (failures with a history of at least one step, and determined successes) from the frozen builder's pool: the same 24 walks, the same rules and exclusions (tag reading, plural reading, wrong recipes). Items already in `items.jsonl` are excluded. Per qualifying goal, up to 2 failures and 1 success are drawn in each length bin (1–3, 4–8, 9–12), each cell seeded with `random.Random(f"{SEED}|textcraft|r2|parent|select|{pair_id}|{G}|{population}|{bin}")`.

**Counts.** 118 goals in 111 pairs qualify: 31 by the strict rule, 87 by word overlap (for example "pink dye" with parent "pink terracotta"). Most pair goals have no qualifying parent, because no recipe consumes them (signs, tools, blocks): 401 of the 510 main goal-role items have such a goal. That is why the parent records are newly drawn rather than taken from the main sample. The records hold 675 failures (522 never obtained, 150 too few, 3 used up; 370 with none of the goal recipe's inputs held) and 207 successes, in 111 pairs. 250 records (176 failures, 29 pairs) have a strict parent. 72 goals add one line, 8 add two, 35 add three, one adds four and two add eleven.

**Arms (two-candidate only, four arms).**

| Arm | List | Goal line | G is |
|---|---|---|---|
| `aligned` | the item's own list | "Goal: craft G." | named, and its craft completes the task |
| `ext_aligned` | extended | "Goal: craft G." | named, completing |
| `ext_parent` | extended | "Goal: craft Z." | not named (in strict-rule records not even by a word); a direct ingredient of the goal; its craft does not complete the task |
| `ext_swapped` | extended | "Goal: craft P." | not named; outside the goal's tree |

**Checks before scoring.** Verify runs:

- the round-1 replay (history, reply and twin, under the goal and under the partner) on the item's own list;
- the same replay on the extended list under G, P, Z and no goal;
- the text-only checker on both lists, which must give the same verdict, plus text-only versions of the rules: Z has a displayed line listing G; Z's displayed tree shares no craftable with P's displayed tree; a strict parent shares no word with G;
- a prompt diff: the extended list keeps the list's lines in order and adds only the added lines, and each arm differs from the aligned prompt only in what its name says;
- the parent rule recomputed from the environment's trees.

Three fresh processes (hash seeds 0, 1 and 2) run every check, and two fresh builds (hash seeds 0 and 1, 8 and 3 worker processes) must write the records byte for byte. The results are in section 13 and `r2_manifest.json`, which the scoring run requires to be verified and to hash the records file exactly.

## 5. Sample sizes and seeds (summary)

| Population | Records | Pairs | Arms per record | Used for |
|---|---|---|---|---|
| Main failures (frozen) | 1,500 | 229 | 8, plus 4 padding arms for 920 of them | lift, naming, completion, timing, competence, the overall effect |
| Determined successes (frozen) | 600 | 167 | 8 | competence; the effect on actions that work (descriptive) |
| Parent failures (new) | 675 | 111 | 4 | the within-item penalty |
| Parent successes (new) | 207 | 73 | 4 | descriptive |

All seeds derive from `RNG_SEED` = 20260924, with the strings given in section 4. Bootstraps use 4,000 resamples of whole goal pairs, and sign-flip permutations 4,000 flips of whole pairs, both seeded with `RNG_SEED`.

## 6. The five confirmatory contrasts (per model)

All are computed on two-candidate L, on failures, with goal pairs as clusters. The goal-role main items exclude the six used-up goal-role items (design B2).

**1. Intermediate lift (Λ).** The mean ΔL over the 990 ingredient main failures (495 couples, 162 pairs):

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}}.$$

An ingredient serves the aligned goal without completing it. The aligned line never names the ingredient's own object, although in 635 of the 990 items the object shares a word with the goal's name. So Λ estimates λ, plus any partial naming through shared words, which is reported as a stratum.

**2. Within-item penalty (P), on the parent failures** (675 records, 111 pairs):

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})}.$$

The action, the history and the displayed list are identical. G is relevant under both lines, so the intermediate lift cancels. Only whether the goal line names G and makes its craft complete the task changes. P estimates the sum of the naming and completion terms on one item. *What it separates:* the penalty from item structure. Round 1's goal-versus-ingredient comparison compared different crafts (goal crafts more often lack craftable inputs and have more short inputs); P compares one craft with itself. *What it does not separate:* naming from completion, because "craft G" does both, like the round-1 object-goal run in the opposite direction.

**3. Naming without completion (Dn), on main failures.** The difference in differences

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}}.$$

For a goal-role action, the two-item line names G and makes it relevant without completing the task; the swapped line does neither. For an ingredient action, the two-item line only makes it relevant. Subtracting the ingredient contrast removes the relevance lift (and the effect of mentioning a second goal, insofar as it acts on both roles alike), leaving the effect of naming the action's object when its craft does not finish the task.

**4. Completion with naming held (Dc), on main failures.**

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

Both lines name G; only the one-item line makes its craft complete the task. The ingredient term removes the general effect of adding a second item to the goal line.

**Identity.** By construction,

$$D_n + D_c = \overline{\Delta L}_{\text{goal}} - \overline{\Delta L}_{\text{ingredient}}:$$

the two-item line splits the round-1 role gap (unmatched) exactly into a naming part and a completion part, as ALFWorld's 2×2 split its effect into mention and goal-directed parts.

*What the two-item contrasts separate.* Naming from completion, holding the action and the history fixed: Dn varies naming with completion held off, and Dc varies completion with naming held on. *What they do not separate, and how each caveat bears on the reading:*

- (a) In the two-item line G is one of two named items, so a naming effect may be diluted. Under a pure naming account with dilution, Dn is negative and Dc is also negative but smaller in size than the naming effect itself. So *Dn's sign is the deciding contrast*: a pure completion account predicts Dn not below zero, and any naming penalty makes Dn negative.
- (b) The difference in differences compares goal-role with ingredient items, which differ in structure. Structure's main effect cancels because each item is compared with itself; an interaction between structure and goal line does not cancel. P has no such caveat.
- (c) The two-item line is off-template in wording (section 4.1).
- (d) Position: the object is named first in half the goal-role items. Dn and Dc are reported by position as a description.

**5. Timing (T), on the padded main failures.**

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}},$$

where ΔL with superscript k is aligned minus swapped with k inventory look-ups prepended. Under the account, the goal-role ΔL is the lift plus the penalty and the ingredient ΔL is the lift, so T is the change in the penalty when the history is seven steps longer with exactly the same content. *Timing reading:* the penalty depends on how many steps have passed, so it shrinks with padding and T is above zero. *Content reading:* the penalty depends on what has been gathered, so T is zero. T is only read where there is a penalty to shrink: if the goal-minus-ingredient ΔL at one look-up does not have a 95% interval below zero, T is reported as "not applicable" for that model. Padding lengthens the transcript in steps and tokens together, so T cannot separate those two; and look-ups are not progress, so a timing effect here would be about elapsed turns, not effort.

**Statistics.** Λ and P are means of per-item differences: estimate, 95% percentile pair bootstrap, and a two-sided sign-flip permutation p with whole pairs flipped. Dn, Dc and T compare two groups of items: estimate, 95% percentile pair bootstrap (the goal and ingredient items of a resampled pair go together), and a two-sided bootstrap p, equal to twice the smaller share of resamples on either side of zero, computed as

$$p = \min\left(1,\ \frac{2\min(\#\{b \le 0\},\ \#\{b \ge 0\}) + 1}{B + 1}\right).$$

## 7. Pre-registered predictions

The predictions follow from round 1 and from ALFWorld, not from what would make the account look good. Where the evidence does not support a direction, none is made. The *account's* predictions (what it must get right) are separated from our *expectations* (what we think will happen, which may be against the account).

**Which models are expected to show an intermediate lift.** The suggestion was to expect a positive ingredient-role effect for models with an ALFWorld *mention* effect. We do not use that criterion. A TextCraft ingredient action is not named by the goal line (apart from shared words), so the ALFWorld quantity that corresponds to Λ is the *goal-directed* part of ALFWorld's 2×2 (the action serves the task without being its named object), not the mention part. That part was measured only for the two reference models: +0.72 [0.05, 1.38] for Qwen and −0.18 [−0.31, −0.05] for Llama-3.1 (`rq2_results.md`). Their mention parts were +6.93 and +1.04. Gemma 2, Mistral and Llama 3.2 3B have positive ALFWorld goal effects (`families_results.md`: two-candidate 0.80, 1.06 and 0.98 nats; yes/no 1.23, 2.67 and 0.70), but no 2×2 was run for them. In both reference models the ALFWorld effect was mostly mention, so a positive total says little about the goal-directed part. Only Qwen qualifies for a positive prediction on Λ, Llama-3.1 is predicted null, and the three new models get no directional prediction. The word-overlap ingredient stratum is where a mention-like effect could act, and it is reported descriptively.

**Which reading of the penalty we expect.** In ALFWorld's 2×2, naming the action's object in a goal the action does not advance *raised* L for both reference models (mention +6.93 Qwen, +1.04 Llama). A naming *penalty* in TextCraft would contradict that, unless the verbatim item name works differently from an ALFWorld object. So for Qwen and Llama-3.1 we expect the completion reading: Dn not below zero, Dc below zero. For the three new models there is no 2×2 and no expectation; both readings are stated.

**The penalty under the clean prompt.** For Llama-3.1 round 1 supports P below zero: the object-goal run, which tests the same contrast in the other direction, gave −1.40 on failures. The clean prompt shrank Llama's goal-role effect by about 40%, so we expect roughly −0.5 to −0.9. For Qwen the evidence conflicts. Under the verified prompt the object-goal run gave −1.23 on failures, almost all of it on never-obtained items (−4.00), and +0.80 on successes. Under the clean prompt Qwen's goal-role effect (+3.03) exceeded its ingredient effect (+0.77) on the diagnostic sample. That comparison is unmatched, but it points the other way. 77% of the parent failures are never obtained, the class where Qwen's object-goal drop was concentrated. We therefore make no directional prediction for Qwen overall, and expect P to be more negative on never-obtained than on too-few records (descriptive). For the new models the account predicts P below zero, because it calls the penalty "not specific to Llama in kind". That is the account's generality claim, not an expectation backed by data.

**Timing.** For Llama, round 1's gradient among goal-role failures holding some inputs was −1.17, −0.49 and +0.02 nats at 1–3, 4–8 and 9–12 steps; by share of inputs held it was −0.85, −0.75 and −0.27. Length and progress are confounded there. If the length gradient were all timing, eight look-ups would move the padded items by about +0.5 to +1.2 nats, so T would be about +0.8. If it were all content, T would be zero. We make no directional prediction for any model.

**Competence (disclosed now).** Under the gate of section 3, applied today to the confirmatory TextCraft scoring records (verified prompt; 1,500 main failures and 600 successes under the neutral line), the results are:

- *Llama-3.1-8B:* yes/no AUC 0.61 (goal role 0.85, ingredient 0.55); Yes to 9.9% of failures and 11.3% of successes; median P(Yes) plus P(No) 0.999. It *would have been classed as not competent*, on the AUC condition. Its two-candidate neutral AUC on the same items is 0.61 (goal role 0.84, ingredient 0.58). The confirmatory metrics' 0.68 is the same model's two-candidate AUC under the aligned line, over the main and usefulness failures.
- *Qwen3-8B:* yes/no AUC 0.85; Yes to 18.7% of failures and 72.5% of successes; median mass 1.00. It passes.

This is a disclosed observation. It does not change the confirmatory verdicts, which were read under the confirmatory design's own competence rule. It does mean that **Llama-3.1-8B is likely to fail the round-2 gate**: the clean prompt lowered its two-candidate AUC by a further 0.016 on the diagnostic sample. If it fails, every round-2 contrast for Llama-3.1, the model the account was built on, is reported as not interpretable, and the account can be tested only on the models that pass. The role-level readout will show whether Llama discriminates on the goal role, as in round 1, but it does not change the verdict.

**Predictions table** (two-candidate, failures; "none" means no directional prediction; the account's own predictions are in the last row):

| Model | Intermediate lift Λ | Within-item penalty P | Naming without completion Dn | Completion with naming held Dc | Timing T |
|---|---|---|---|---|---|
| Qwen3-8B | above zero (round 1 +0.88; clean sample +0.77; ALFWorld goal-directed +0.72) | none overall; more negative on never-obtained than too-few records | not below zero (expected above: ALFWorld mention +6.93) | below zero (expected, completion reading) | none |
| Llama-3.1-8B (likely to fail the gate) | equivalent to zero (round 1 +0.03; ALFWorld goal-directed −0.18) | below zero (object-goal run −1.40; expected about −0.5 to −0.9 under the clean prompt) | not below zero (ALFWorld mention +1.04) | below zero (expected, completion reading) | none |
| Gemma-2-9B-it | none (no ALFWorld 2×2) | below zero (the account's generality claim only) | none | none | none |
| Mistral-7B-v0.3 | none (no ALFWorld 2×2) | below zero (the account's generality claim only) | none | none | none |
| Llama-3.2-3B | none (no ALFWorld 2×2) | below zero (the account's generality claim only) | none | none | none |
| The account, any model | differs across models and carries the between-model difference in the total effect | below zero | completion reading: not below zero; naming reading: below zero | completion reading: below zero; naming reading: at or near zero, below zero only by dilution | timing reading: above zero; content reading: equivalent to zero |

**Descriptive expectations** (not tested):

- the overall goal effect under the clean prompt: Qwen above zero (diagnostic sample +1.52), Llama near zero or slightly negative (−0.17), none for the new models;
- the lift transfers: on parent failures, L under the parent line minus L under the swapped line (G made an intermediate of the goal) is close to the model's Λ;
- adding the parent's lines under the aligned line changes L little;
- one inventory look-up against none moves the goal-role effect up slightly (round 1's empty-start sentence: +0.14 for Llama).

**What would falsify the account** (each applies to models that pass the competence gate):

- **F1, no within-item penalty in its home model.** For Llama-3.1-8B, P is equivalent to zero or confirmed above zero. The round-1 penalty would then reflect item structure or the agent-instruction prompt, not the goal line. If Llama-3.1 fails the gate, F1 cannot be tested, and that is reported.
- **F2, the lift does not explain the model difference.** Llama-3.1's Λ is confirmed above zero and at least Qwen's estimate, or, among the models that pass, the ordering of Λ is unrelated to the ordering of the total goal effect. The second half is descriptive (five models).
- **F3, the penalty is not shared.** P is confirmed below zero for no model that passes the gate other than Llama-3.1. The account's generality claim then fails; its Llama part could stand.
- **F4, "largest early" is wrong in kind.** If T is confirmed below zero (padding *deepens* the penalty), neither the timing nor the content reading holds.

The mechanism contrasts (Dn, Dc, T) choose between readings of the account; they do not falsify it except as in F4.

## 8. Tests, decision rules and multiplicity

**The family.** For each of the five models and each of the five contrasts there are two tests:

- *differs:* the two-sided p of section 6;
- *equivalent:* a two one-sided test (TOST) against a margin of plus or minus δ. Its p is the larger of the two one-sided bootstrap ps: the share of resamples at or below −δ and the share at or above +δ, each smoothed as (count + 1)/(B + 1).

That makes 50 tests, Holm-adjusted together at a familywise 0.05 level. The family's size is fixed at 50. A test whose model has not been scored, or whose p is missing or not a number, enters with p = 1.

**Equivalence margin.** For model m,

$$\delta_m = 0.3 \times \mathrm{SD}_m,$$

where SD is the item standard deviation of ΔL (aligned minus swapped) over the 1,500 main failures under the clean prompt. It is computed once per model from the point estimate, before any equivalence test is read. A standardized margin is used because the models' log-odds scales differ several-fold (on round 1's verified prompt the SD was 1.38 for Llama and 4.74 for Qwen; δ would have been 0.41 and 1.42). For Llama 0.41 is about half the round-1 matched penalty. For Qwen 1.42 is larger than its round-1 matched penalty (0.82), so for Qwen an "equivalent" verdict cannot exclude an effect of the round-1 size; this limit is reported with any Qwen equivalence.

**Verdicts per contrast and model.**

| Verdict | Rule |
|---|---|
| confirmed above zero | estimate above zero, 95% interval above zero, Holm-adjusted differs-p below 0.05 |
| confirmed below zero | the same, below zero |
| equivalent to zero | Holm-adjusted TOST p below 0.05 |
| inconclusive | none of the above |
| not interpretable | the model fails the competence gate (whatever the numbers) |
| not applicable | T only: no role gap at one look-up (section 6) |

**Readings per model** (models that pass the gate):

| P | Dn | Dc | Reading |
|---|---|---|---|
| below zero | not below zero (above, equivalent or inconclusive) | below zero | **Completion penalty.** Crafting the item that finishes the named task lowers predicted success; naming the object alone does not. |
| below zero | below zero | any | **Naming penalty.** Naming the action's object lowers predicted success even when its craft does not finish the task. Whether a completion part adds to it cannot be separated from the dilution of naming in the two-item line. |
| below zero | inconclusive | inconclusive | **Penalty present, mechanism unresolved** at this power. |
| equivalent to zero | any | any | **No penalty** under the clean prompt for this model: the round-1 role gap does not reappear when the item is held fixed. |
| above zero | any | any | **Completing a named goal raises predicted success** for this model (the ALFWorld-like direction); no penalty. |
| inconclusive | any | any | Inconclusive on the penalty; Dn and Dc are reported but not read as a mechanism. |

The timing reading is read alongside: T confirmed above zero means **timing** (the penalty shrinks with elapsed steps at fixed content); T equivalent to zero means **content** (steps that add nothing leave the penalty unchanged, so "largest early" is about what has been gathered); otherwise inconclusive or not applicable.

The lift: Λ confirmed above zero means the model raises predicted success for intermediate crafts of the named goal; Λ equivalent to zero means it does not (Llama's round-1 pattern).

**Across models** (descriptive, read from the per-model verdicts):

| Pattern | Reading |
|---|---|
| P confirmed below zero in every model that passes the gate | The penalty is general across these families under the clean prompt. |
| P confirmed below zero in some models, equivalent to zero in others | The penalty is model-specific; the models are named. |
| P below zero for Llama-3.1 only | The penalty is a Llama-3.1 property (F3). |
| The same reading of the penalty (completion or naming) in every model where it is read | One mechanism across families. |
| Λ ordering matches the ordering of the overall goal effect | Consistent with the lift carrying the between-model difference; with five models this is a description, not a test. |
| Fewer than two models pass the gate | Round 2 cannot speak across families; per-model results are reported as they are. |

**What each pattern does not show.**

- A confirmed P shows that the goal line's naming-and-completing status changes L for a fixed craft. It does not show that the model "thinks" the task is premature.
- A completion reading rests on Dn's sign and on the two-item line, which is off-template in wording.
- An equivalence verdict for Qwen cannot exclude round-1-sized effects (δ above 0.82).
- Nothing here bears on the confirmatory TextCraft verdicts. The clean prompt measures a different quantity.
- All items are failures and successes from goal-blind scripted walks, not agent histories.

## 9. Descriptive readouts (estimates and 95% pair-bootstrap intervals, no p-values)

1. The overall goal effect ΔL on the 1,500 main failures, and by role, class and length bin, per model. For Qwen and Llama it is also compared with the confirmatory (verified-prompt) scores on the same items: the per-item difference, clean minus verified.
2. The aligned-minus-neutral and swapped-minus-neutral contrasts; the two-item line against neutral.
3. Every confirmatory contrast on the yes/no scale.
4. P by failure class, by length bin, by inputs held (none against some), by parent rule (strict against word overlap), on the parent successes, and on the strict subset alone.
5. The lift on parent records: L under the parent line minus L under the swapped line, against Λ.
6. The list control: L under the aligned line on the extended list minus on the item's own list.
7. Dn and Dc by the object's position in the two-item line, by class, by length bin, and on successes. Λ on ingredients that share a word with the goal and on those that share none.
8. T's two parts (goal-role and ingredient changes) separately, by length bin. The empty-start contrast: the effect with one look-up against none, per role.
9. Competence: the gate's four values, the role-level and class-level AUCs on both scales, the two-candidate first-token mass per arm, and the AUC cost of each goal line.
10. Across models: a table of Λ, P, Dn, Dc, T and the overall effect, per model, with the gate result.

## 10. Power (planning figures, not results)

Planning uses the confirmatory TextCraft item spreads of ΔL (verified prompt) and the pair intraclass correlations:

- Llama: SD 1.56 on the goal role and 1.22 on ingredients; ICC 0.15 and 0.06.
- Qwen: SD 5.09 and 4.56; ICC 0.11 and 0.07.

Each contrast's standard error comes from the design effect of its items per pair. The minimum detectable effect is for a two-sided test at Holm's first step (0.05/50) with 80% power. These assume the contrasts spread like ΔL. P, Dn and Dc compare two lines that share more of the prompt than aligned and swapped do, so their spread is probably smaller; T is a difference of differences, so its spread could be larger.

| Contrast | Items (pairs) | Llama MDE (nats) | Qwen MDE (nats) |
|---|---|---|---|
| Λ | 990 (162) | 0.18 | 0.69 |
| P | 675 (111) | 0.33 | 0.99 |
| Dn, Dc | 504 goal (159) and 990 ingredient (162) | 0.38 | 1.25 |
| T | 338 goal (130) and 582 ingredient (128) | 0.45 | 1.52 |

For Llama-sized spreads, the expected P (−0.5 to −0.9) and an F1 equivalence verdict (|estimate| below about 0.15 with δ near 0.41) are both reachable. For Qwen-sized spreads, only large effects are detectable, and equivalence is reachable only for estimates near zero. The new models' spreads are unknown; in ALFWorld their effects were on Llama's scale (0.8–1.1 nats two-candidate).

## 11. Cost

Each model scores 24,008 (record, arm) combinations:

- main: 2,100 × 8 = 16,800;
- padding: 920 × 4 = 3,680;
- parent: 882 × 4 = 3,528.

That is 120,040 scores over five models. The yes/no arms (8,400 per model) are read from one forward pass each (`score_one_token_pair`, equal to `score_pair` by construction for one-token candidates). Two-candidate arms take two passes.

Round 1's rate was 51,200 scores (both models, all arms through `score_pair`) in about 4 GPU-hours on four shards, about 12,800 scores per GPU-hour; the diagnostics ran about 15,000 scores in 98 GPU-minutes. Prompts are of similar length. Estimated GPU time:

- Qwen3-8B, Llama-3.1-8B and Mistral-7B: about 1.6 hours each;
- Llama-3.2-3B: about 0.9 hours;
- Gemma-2-9B with eager attention (slower; taken as twice the rate's time): about 3.2 hours.

That totals about **9 GPU-hours**, plus about 0.5 for a 20-record smoke run per model and model loading; budget **10–12 GPU-hours**. Each model runs as 4 shards by goal pair, one job per shard of under 1.5 hours (Gemma's under 2 hours), saving every score as it is made, with resume. The CPU tokenization check took the time recorded in section 13.

## 12. Code (built after this note; see section 13 for the checks' results)

- `wf/textcraft/prompts.py`: `CLEAN_SYSTEM`, the default from round 2 on.
- `wf/textcraft/r2.py`: arms, prompts, models, item hash.
- `wf/textcraft/r2_items.py`: build and verify, writing `r2_manifest.json`.
- `wf/textcraft/r2_tokcheck.py`: tokenization gate and golden renderings, on CPU.
- `wf/textcraft/r2_score.py`: the scoring run `textcraft_r2_score_<tag>`, sharded and resumable, with the gate before any score.
- `wf/textcraft/r2_analysis.py`: the run `textcraft_r2_analysis`, with the gate, the contrasts, the Holm family, the verdicts and the descriptive readouts.
- Configs `wf/config/textcraft_r2_*.yaml`; jobs `pace/textcraft_r2_tokcheck.sbatch` and `pace/textcraft_r2_score.sbatch`.
- Tests `tests/test_textcraft_r2.py`; mutants in `scripts/mutation_check.py`, with their killing tests listed in `notes/testing.md`.

## 13. Checks run before scoring

*(Filled in by the build commit.)*
