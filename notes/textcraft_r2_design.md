# TextCraft round 2: design (written 2026-11-21, before any round-2 score)

This note fixes, before any model scores a round-2 record, what round 2 asks, which records and prompts it scores, which tests are confirmatory, how each result will be read, and what would count against the account it tests. Nothing in round 2 has been scored. What had been seen when this was written: the confirmatory TextCraft results ([textcraft_results.md](textcraft_results.md), frozen design [textcraft_design.md](textcraft_design.md)), the exploratory Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md), branch `textcraft-llama-dx`), the cross-environment comparison (`notes/textcraft_vs_alfworld.md`, branch `xenv-compare`), the model-families wave A results in ALFWorld ([families_results.md](families_results.md)), and, for the planning figures and the competence disclosure in section 7, the confirmatory TextCraft scoring records (read again on 2026-11-21 for item spreads and the families competence rule; no new contrast was computed on them). The round-2 records were built and checked before this note was committed (section 4 gives their counts); no model has read them.

The design must be independently reviewed and then frozen at a git tag (`textcraft-r2-design-frozen-<date>`) before any scoring. Any change after the tag is a reported deviation.

**Amended before the freeze.** Section 14 records the amendments made on 2026-11-21, before any round-2 score, from the owner's decisions and from review 1. Where an earlier section disagrees with section 14, section 14 governs. Factual errors found by review 1 are corrected in place, each marked "(corrected, 14.Rn)".

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

**What round 2 is, and is not (14.U5).** The round-1 TextCraft replication across models failed under its preregistered rule (confirmed for Qwen3-8B, not for Llama-3.1-8B), and round 2 changes nothing about that result. The diagnostics that motivated round 2 were exploratory and post hoc. Round 2 tests the explanations they suggested, prospectively, on a new prompt and new records, with rules fixed before any round-2 score. Its central question: *does Llama's negative TextCraft effect arise from a goal-completion or naming mechanism, from poor reasoning about the environment's state, or from a combination?* Section 14 states which results would reject the two-component decomposition.

**Confirmatory and descriptive.** The confirmatory part is five contrasts per model (section 6), each with a test of difference and a test of equivalence, in one Holm family of fixed size 50 (section 8). Everything else is descriptive: estimates with 95% intervals and no p-values, including the overall goal effect under the clean prompt.

## 2. The prompt: clean world-model text, first-class

**Decision (the owner's, 2026-11-21).** The clean world-model system text is the default prompt for every TextCraft run after the confirmatory one. The confirmatory result stays reported under the prompt it was frozen with (`prompts.SYSTEM`). A run under the clean prompt measures a related but different quantity: Qwen and Llama rescored under it are new estimates, not re-estimates of the confirmatory numbers.

**Text.** `prompts.CLEAN_SYSTEM` (moved from the diagnostics' `dx.CLEAN_SYSTEM`, byte for byte unchanged; a test checks this):

> You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next action, predict the environment's next observation exactly in TextCraft style. Output only the observation text.
>
> The environment's rules: crafting commands are of the format "craft [target object] using [input ingredients]". Each action is "get" with a quantity and an object, "inventory" to look up the inventory, or "craft" with a quantity and one of the crafting commands shown. If a crafting command uses a generic ingredient like "planks", a special type of the same ingredient, e.g. "dark oak planks", can be used instead.

It keeps the world-model framing and states only the environment's rules and the action format. Every sentence addressed to the agent is removed, and so is "from the inventory or the environment", which hinted at a hidden store. It does not say that the inventory starts empty.

**Everything else is the verified prompt.** The three verified in-context examples (`fewshot.json`, sha256 `e6ddab93…adfb1253`), the turn format ("History:", "Action:", "Next observation:"), and, as first written, the yes/no question ("Question: Will this action succeed? Answer Yes or No.\nAnswer:") were unchanged. *(Amended, 14.Q: round 2 uses a yes/no question anchored to TextCraft's replies; round 1's question is not used.)*

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

**Competence (amended, 14.U1).** The overall gate below is kept and reported for every model, but it no longer decides how a contrast is read: each contrast is read against its own competence rule (14.U1), and no estimate is withheld. The overall gate is wave A's rule (`families.COMPETENCE`, `notes/families_metrics.json`, `rules.competence`), computed on the round-2 main records under the neutral line, with the yes/no question. A model is interpretable for round 2 only if all four conditions hold:

- the AUC of the yes/no L between the 600 determined successes and the 1,500 main failures is at least 0.70;
- it answers Yes (L above 0) to at most 95% of the failures;
- it answers Yes to at least 5% of the successes;
- the median of P(Yes) plus P(No) is at least 0.5.

*(Superseded by 14.U1.)* The first version reported a model that fails this gate as not interpretable for every contrast. Now every model's estimates and verdicts are reported, read through the contrast-specific competence rules, and all tests stay in the Holm family.

**Role-level competence (descriptive, pre-registered).** Llama's round-1 competence differed sharply by role (two-candidate neutral AUC 0.84 on the goal role, 0.58 on ingredients). For every model, the neutral-line AUC is reported on the goal role (goal-role successes against goal-role failures) and on the ingredient role, on both scales, with pair-bootstrap intervals. It is also reported per failure class (each class against all successes), per history-length bin (the bin's failures against the bin's successes), role-matched for the used-up class (used-up ingredient failures against the 300 ingredient-role successes; the confirmatory used-up AUC set mostly-ingredient failures against successes of both roles), and on the parent records' successes against their failures. These readouts do not change the gate's verdict. The two-candidate neutral AUC, overall and per role, and the median first-token mass of the two-candidate score (the probability of the two candidates' first tokens) are reported too.

*Added 2026-11-21, before any round-2 score, from the finished competence analysis (`notes/textcraft_llama_competence.md`, heuristics fixed at 9a83738 before any fit):* Llama-3.1-8B's neutral two-candidate AUC on the main population is 0.61 (the 0.68 of the confirmatory metrics included the usefulness failures). It is expected to fail this gate; it stays in the design, and the gate decides.

## 4. Records

All records are in `wf/textcraft/data/r2_items.jsonl`, built by `python -m wf.textcraft.r2_items build` from the frozen items (`items.jsonl`, sha256 `3b81ac3f…5e421a6`) and the frozen builder. There are 2,982 records: 2,100 main and 882 parent. Each model scores 25,743 (record, arm) combinations (23,988 before the parent arms of 14.R8 were added).

### 4.1 Main records and the two-item goal line

**Records.** The 1,500 frozen main failures (750 same-history couples, 229 goal pairs) and the 600 frozen determined successes (300 couples), copied field for field. A record differs from its frozen item only in three added fields: `kind`, `both_task_line` and `arms`.

**Arms.** Each of four goal lines, as the two-candidate score and as the yes/no question (eight arms):

| Line | Text | What it says about a goal-role action (crafting G) |
|---|---|---|
| aligned | "Goal: craft G." | names G; crafting G completes the task |
| swapped | "Goal: craft P." | does not name G; G is not in P's tree |
| neutral | "Goal: explore the crafting commands." | names no item |
| two-item | "Goal: craft A and B." | names G; crafting G does not complete the task, because the other item remains |

In the two-item line, A and B are the pair's two goals in sorted order. Both members of a couple therefore see the same line, and across couples the action's object is named first in half the goal-role items and second in the other half.

*Added while building, before any score (2026-11-21).* One goal pair has a goal whose name contains the word "and" (flint and steel, paired with pink dye), so its two-item line, "Goal: craft flint and steel and pink dye.", cannot be read as two items. Its 10 records (4 goal-role failures, 6 goal-role successes) get no two-item arms and leave the two-item contrasts; they keep every other arm. The rule (`r2.two_item_line`): no two-item line when either goal's name contains the word "and". The naming and completion contrasts therefore use 500 goal-role failures (158 pairs) and the 990 ingredient failures.

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

**Arms (four two-candidate arms; amended, 14.R8: plus `ext_both`, the two-item line on the extended list, on the 873 records whose pair has one, and `yn_neutral`, the yes/no question under the neutral line on the item's own list, for competence).**

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

All seeds derive from `RNG_SEED` = 20260924, with the strings given in section 4. Bootstraps use 4,000 resamples of whole goal pairs, and sign-flip permutations 4,000 flips of whole pairs, both seeded with `RNG_SEED`. *(Superseded, 14 and 15.S: the 50 family tests and the rejection criteria's role gap use 20,000 resamples and 20,000 flips; descriptive intervals use 2,000.)*

## 6. The five confirmatory contrasts (per model)

All are computed on two-candidate L, on failures, with goal pairs as clusters. The goal-role main items exclude the six used-up goal-role items (design B2).

**1. Intermediate lift (Λ).** The mean ΔL over the 990 ingredient main failures (495 couples, 162 pairs):

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}}.$$

An ingredient serves the aligned goal without completing it. *(Corrected, 14.R1.)* The aligned line can name the ingredient in part: 683 of the 990 ingredient objects share a word with the goal's name, and in 180 of them the ingredient's whole name lies inside the goal's name (for example a planks ingredient of a goal named after its wood). So Λ estimates λ plus whatever partial naming does; it is reported separately for whole name contained, word overlap only and no shared word (section 14).

**2. Within-item penalty (P), on the parent failures** (675 records, 111 pairs):

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})}.$$

The action, the history and the displayed list are identical. G is relevant under both lines, so the intermediate lift cancels. Only whether the goal line names G and makes its craft complete the task changes. P estimates the sum of the naming and completion terms on one item. *What it separates:* the penalty from item structure. Round 1's goal-versus-ingredient comparison compared different crafts (goal crafts more often lack craftable inputs and have more short inputs); P compares one craft with itself. *What it does not separate:* naming from completion, because "craft G" does both, like the round-1 object-goal run in the opposite direction.

**3. Naming / partial-completion (Dn), on main failures.** *(Renamed, 14.V1: crafting G under "craft A and B" names G and completes one item of the two-item goal, so Dn is not pure naming.)* *(Ingredient rows class-matched, 14.R2: only too-few and never-obtained ingredient failures, 496 items, enter Dn, Dc and T.)* The difference in differences

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}}.$$

For a goal-role action, the two-item line names G and makes it relevant without completing the task; the swapped line does neither. For an ingredient action, the two-item line only makes it relevant. Subtracting the ingredient contrast removes the relevance lift (and the effect of mentioning a second goal, insofar as it acts on both roles alike), leaving the effect of naming the action's object when its craft does not finish the task.

**4. Completion with naming held (Dc), on main failures.**

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}}.$$

Both lines name G; only the one-item line makes its craft complete the task. The ingredient term removes the general effect of adding a second item to the goal line.

**Identity.** By construction,

$$D_n + D_c = \overline{\Delta L}_{\text{goal}} - \overline{\Delta L}_{\text{ingredient}}:$$

*(Corrected, 14.R3.)* The identity holds exactly only on the items that enter both sides: the 500 goal-role failures that have a two-item line and, after 14.R2, the 496 class-matched ingredient failures. It splits the *round-2* role gap under the clean prompt, unmatched on item structure, into a naming part and a completion part. It does not split round 1's gap, which was measured under another prompt. The split is algebraic; reading its parts as causal effects needs the assumptions of 14.U4.

*What the two-item contrasts separate.* Naming from completion, holding the action and the history fixed: Dn varies naming with completion held off, and Dc varies completion with naming held on. *What they do not separate, and how each caveat bears on the reading:*

- (a) In the two-item line G is one of two named items, so a naming effect may be diluted. Under a pure naming account with dilution, Dn is negative and Dc is also negative but smaller in size than the naming effect itself. Crafting G under "craft A and B" also completes a named sub-goal. *(Qualified, 14.R5.)* So a negative Dn reads as a *naming / partial-completion* penalty (14.V1), not as naming alone. A pure completion account of the whole task predicts Dn not below zero; an inconclusive Dn is not evidence that naming has no effect (14.U4).
- (b) The difference in differences compares goal-role with ingredient items, which differ in structure. Structure's main effect cancels because each item is compared with itself; an interaction between structure and goal line does not cancel. P has no such caveat.
- (c) The two-item line is off-template in wording (section 4.1).
- (d) Position: the object is named first in half the goal-role items. Dn and Dc are reported by position as a description.

**5. Timing (T), on the padded main failures.**

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}},$$

where ΔL with superscript k is aligned minus swapped with k inventory look-ups prepended. *(Reading amended, 14.U4 and 14.R9: T above zero reads as sensitivity to added empty-inventory steps, not as timing; the goal-role and ingredient rows are class-matched, 14.R2.)* Under the account, the goal-role ΔL is the lift plus the penalty and the ingredient ΔL is the lift, so T is the change in the penalty when the history is seven steps longer with exactly the same content. *Timing reading:* the penalty depends on how many steps have passed, so it shrinks with padding and T is above zero. *Content reading:* the penalty depends on what has been gathered, so T is zero. T is only read where there is a penalty to shrink: if the goal-minus-ingredient ΔL at one look-up does not have a 95% interval below zero, T is reported as "not applicable" for that model. Padding lengthens the transcript in steps and tokens together, so T cannot separate those two; and look-ups are not progress, so a timing effect here would be about elapsed turns, not effort.

**Statistics.** Λ and P are means of per-item differences: estimate, 95% percentile pair bootstrap, and a two-sided sign-flip permutation p with whole pairs flipped. Dn, Dc and T compare two groups of items: estimate, 95% percentile pair bootstrap (the goal and ingredient items of a resampled pair go together), and a two-sided bootstrap p, equal to twice the smaller share of resamples on either side of zero, computed as

$$p = \min\left(1,\ \frac{2\min(\#\{b \le 0\},\ \#\{b \ge 0\}) + 1}{B + 1}\right).$$

## 7. Pre-registered predictions

The predictions follow from round 1 and from ALFWorld, not from what would make the account look good. Where the evidence does not support a direction, none is made. The *account's* predictions (what it must get right) are separated from our *expectations* (what we think will happen, which may be against the account).

**Which models are expected to show an intermediate lift.** *(Superseded twice: by 14.R4, and by 14.V3, which makes every ALFWorld-based entry exploratory motivation only. The paragraph is kept as first written; the predictions table below is the current one.)* *(Corrected, 14.R4: the paragraph below mapped ALFWorld's goal-directed part to Λ. That part holds naming fixed and varies whether the action advances the task, so it corresponds to Dc, not Λ. Λ predictions now rest on TextCraft round 1 only; see the revised table.)* The suggestion was to expect a positive ingredient-role effect for models with an ALFWorld *mention* effect. We do not use that criterion. A TextCraft ingredient action is not named by the goal line (apart from shared words), so the ALFWorld quantity that corresponds to Λ is the *goal-directed* part of ALFWorld's 2×2 (the action serves the task without being its named object), not the mention part. That part was measured only for the two reference models: +0.72 [0.05, 1.38] for Qwen and −0.18 [−0.31, −0.05] for Llama-3.1 (`rq2_results.md`). Their mention parts were +6.93 and +1.04. Gemma 2, Mistral and Llama 3.2 3B have positive ALFWorld goal effects (`families_results.md`: two-candidate 0.80, 1.06 and 0.98 nats; yes/no 1.23, 2.67 and 0.70), but no 2×2 was run for them. In both reference models the ALFWorld effect was mostly mention, so a positive total says little about the goal-directed part. Only Qwen qualifies for a positive prediction on Λ, Llama-3.1 is predicted null, and the three new models get no directional prediction. The word-overlap ingredient stratum is where a mention-like effect could act, and it is reported descriptively.

**Which reading of the penalty we expect.** *(Superseded by 14.V3 and 15.S: no expectation in this paragraph is used. Dn has no expectation for any model; Dc has none for Qwen and only a motivated expectation for Llama; ALFWorld's goal-directed contrast is not the same intervention as Dc. The paragraph is kept as first written.)* In ALFWorld's 2×2, naming the action's object in a goal the action does not advance *raised* L for both reference models (mention +6.93 Qwen, +1.04 Llama). A naming *penalty* in TextCraft would contradict that, unless the verbatim item name works differently from an ALFWorld object. So for Qwen and Llama-3.1 we expect the completion reading: Dn not below zero, Dc below zero. For the three new models there is no 2×2 and no expectation; both readings are stated.

**The penalty under the clean prompt.** For Llama-3.1 round 1 supports P below zero: the object-goal run, which tests the same contrast in the other direction, gave −1.40 on failures. The clean prompt shrank Llama's goal-role effect by about 40%, so we expect roughly −0.5 to −0.9. For Qwen the evidence conflicts. Under the verified prompt the object-goal run gave −1.23 on failures, almost all of it on never-obtained items (−4.00), and +0.80 on successes. Under the clean prompt Qwen's goal-role effect (+3.03) exceeded its ingredient effect (+0.77) on the diagnostic sample. That comparison is unmatched, but it points the other way. 77% of the parent failures are never obtained, the class where Qwen's object-goal drop was concentrated. We therefore make no directional prediction for Qwen overall, and expect P to be more negative on never-obtained than on too-few records (descriptive). For the new models the account predicts P below zero, because it calls the penalty "not specific to Llama in kind". That is the account's generality claim, not an expectation backed by data.

**Timing.** For Llama, round 1's gradient among goal-role failures holding some inputs was −1.17, −0.49 and +0.02 nats at 1–3, 4–8 and 9–12 steps; by share of inputs held it was −0.85, −0.75 and −0.27. Length and progress are confounded there. If the length gradient were all timing, eight look-ups would move the padded items by about +0.5 to +1.2 nats, so T would be about +0.8. If it were all content, T would be zero. We make no directional prediction for any model.

**Competence (disclosed now).** *(Every yes/no figure in this paragraph, and in 14.U1's disclosure table, comes from round 1's unanchored yes/no question; round 2's anchored question (14.Q) measures a different quantity, so these are a guide, not a forecast.)* Under the gate of section 3, applied today to the confirmatory TextCraft scoring records (verified prompt; 1,500 main failures and 600 successes under the neutral line), the results are:

- *Llama-3.1-8B:* yes/no AUC 0.61 (goal role 0.85, ingredient 0.55); Yes to 9.9% of failures and 11.3% of successes; median P(Yes) plus P(No) 0.999. It *would have been classed as not competent*, on the AUC condition. Its two-candidate neutral AUC on the same items is 0.61 (goal role 0.84, ingredient 0.58). The confirmatory metrics' 0.68 is the same model's two-candidate AUC under the aligned line, over the main and usefulness failures.
- *Qwen3-8B:* yes/no AUC 0.85; Yes to 18.7% of failures and 72.5% of successes; median mass 1.00. It passes.

This is a disclosed observation. It does not change the confirmatory verdicts, which were read under the confirmatory design's own competence rule. *(Superseded by 14.U1: no contrast is withheld for competence, and competence is read per contrast; the overall gate below is reported only.)* It does mean that **Llama-3.1-8B is likely to fail the round-2 gate**: the clean prompt lowered its two-candidate AUC by a further 0.016 on the diagnostic sample. If it fails, every round-2 contrast for Llama-3.1, the model the account was built on, is reported as not interpretable, and the account can be tested only on the models that pass. The role-level readout will show whether Llama discriminates on the goal role, as in round 1, but it does not change the verdict.

**Predictions table** (two-candidate, failures; "none" means no directional prediction; the account's own predictions are in the last row). *Revised before the freeze (14.R4).* Λ rests on TextCraft round 1 only. Dc rests on ALFWorld's goal-directed part on completing moves (`notes/textcraft_vs_alfworld.md`, branch `xenv-compare`, section 10: the 2×2 part that holds naming fixed, on the moves that would complete the task). Dn has no directional expectation: ALFWorld's mention part names the object in a goal the action does *not* advance, while in Dn crafting G completes a named sub-goal.

| Model | Intermediate lift Λ | Within-item penalty P | Naming / partial-completion Dn | Completion with naming held Dc | Timing T |
|---|---|---|---|---|---|
| Qwen3-8B | above zero (TextCraft round 1 +0.88; clean sample +0.77) | none overall; more negative on never-obtained than too-few records | none | none (14.V3) | none |
| Llama-3.1-8B (likely to fail competence) | equivalent to zero (TextCraft round 1 +0.03 [−0.07, +0.13]) | below zero (object-goal run −1.40; expected about −0.5 to −0.9 under the clean prompt) | none | motivated expectation below zero, not a firm prediction (14.V3) | none |
| Gemma-2-9B-it | none | below zero (the account's generality claim only) | none | none | none |
| Mistral-7B-v0.3 | none | below zero (the account's generality claim only) | none | none | none |
| Llama-3.2-3B | none | below zero (the account's generality claim only) | none | none | none |
| The account, any model | differs across models and carries the between-model difference in the total effect | below zero | completion reading: not below zero; naming / partial-completion reading: below zero | completion reading: below zero; naming reading: at or near zero, below zero only by dilution | sensitivity reading: above zero; no-sensitivity reading: equivalent to zero |

*No confirmatory rule depends on any entry of this table (14.V3).* The tests, verdicts, interpretations and readings of section 8 are the same whatever was predicted; the table records expectations so that they can be checked afterwards.

*Revised again before the freeze (14.V3).* Dc expectations are now exploratory motivation only: ALFWorld's goal-directed contrast is not the same intervention as TextCraft's completion contrast (a different mention manipulation, different verbs and replies), and Qwen's positive ALFWorld estimate is not evidence that a penalty is absent, given its uncertainty. The previous revision gave Qwen "not below zero" and Llama "below zero" for Dc.

*The first version of this table* predicted Λ for Qwen and Llama from ALFWorld's goal-directed part (+0.72, −0.18), Dn "not below zero" for both from ALFWorld's mention part, and Dc "below zero" for both. Those mappings were wrong (14.R4).

**Descriptive expectations** (not tested):

- the overall goal effect under the clean prompt: Qwen above zero (diagnostic sample +1.52), Llama near zero or slightly negative (−0.17), none for the new models;
- the lift transfers: on parent failures, L under the parent line minus L under the swapped line (G made an intermediate of the goal) is close to the model's Λ;
- adding the parent's lines under the aligned line changes L little;
- one inventory look-up against none moves the goal-role effect up slightly (round 1's empty-start sentence: +0.14 for Llama).

**What would falsify the account** *(superseded by the coded rejection list of 15.R; F1 to F4 below are kept as first written. F1 becomes R1 for every model competent for P, F3 becomes R2, F2 becomes R4 and F4 becomes R5)* (each applies to models that pass the competence gate):

- **F1, no within-item penalty in its home model.** For Llama-3.1-8B, P is equivalent to zero or confirmed above zero. The round-1 penalty would then reflect item structure or the agent-instruction prompt, not the goal line. If Llama-3.1 fails the gate, F1 cannot be tested, and that is reported.
- **F2, the lift does not explain the model difference.** Llama-3.1's Λ is confirmed above zero and at least Qwen's estimate, or, among the models that pass, the ordering of Λ is unrelated to the ordering of the total goal effect. The second half is descriptive (five models).
- **F3, the penalty is not shared.** *(Rule corrected, 14.R6: F3 holds only if at least one competent model other than Llama-3.1 has P equivalent to zero or confirmed above zero, and none is confirmed below zero; if one is confirmed below zero, F3 does not hold; otherwise F3 is undecided.)* P is confirmed below zero for no model that passes the gate other than Llama-3.1. The account's generality claim then fails; its Llama part could stand.
- **F4, "largest early" is wrong in kind.** If T is confirmed below zero (padding *deepens* the penalty), neither the timing nor the content reading holds.

The mechanism contrasts (Dn, Dc, T) choose between readings of the account; they do not falsify it except as in F4.

## 8. Tests, decision rules and multiplicity

**The family.** For each of the five models and each of the five contrasts there are two tests:

- *differs:* the two-sided p of section 6;
- *equivalent:* a two one-sided test (TOST) against a margin of plus or minus δ. Its p is the larger of the two one-sided bootstrap ps: the share of resamples at or below −δ and the share at or above +δ, each smoothed as (count + 1)/(B + 1).

That makes 50 tests, Holm-adjusted together at a familywise 0.05 level. The family's size is fixed at 50. A test whose model has not been scored, or whose p is missing or not a number, enters with p = 1.

**Equivalence margin (amended, 14.U3).** For model m, on the two-candidate scale, the same for all five contrasts,

$$\delta_m = \min\left(0.3 \times \mathrm{SD}_m,\ 0.4\ \text{nats}\right),$$

where SD is the item standard deviation of ΔL (aligned minus swapped) over the 1,500 main failures under the clean prompt, computed once per model from the point estimate before any equivalence test is read. On round 1's spreads, 0.3 × SD was 0.41 for Llama and 1.42 for Qwen, so the cap of 0.4 nats applies to both, and probably to every model. An equivalence verdict means "no effect larger than δ", never "no effect". Each test also reports its power at a true zero (14.U3, section 10). Where equivalence is underpowered and not reached, the verdict is "inconclusive", never "no effect".

**Statistical verdicts per contrast and model.**

| Verdict | Rule |
|---|---|
| confirmed above zero | estimate above zero, 95% interval above zero, Holm-adjusted differs-p below 0.05 |
| confirmed below zero | the same, below zero |
| equivalent to zero | Holm-adjusted TOST p below 0.05 |
| inconclusive | none of the above ("inconclusive (equivalence underpowered)" when the test's power at a true zero, at Holm's first step, is below 0.8) |
| not applicable | T only: no role gap at one look-up (section 6) |

**How a verdict is read: competence (amended, 14.U1).** Competence no longer withholds a verdict. Every estimate and test is reported for every model, and all 50 tests stay in the Holm family. Each contrast has its own competence rule (14.U1). The verdict is read as follows:

| Verdict | Competent for this contrast | Reading |
|---|---|---|
| confirmed (either sign) | yes | a **distortion in a model that discriminates successes from failures in these categories**: the goal line shifts the predictions of a model whose yes/no AUC passes the gate in these action categories (discrimination, not demonstrated use of the inventory rule; 15.F) |
| confirmed (either sign) | no | a **statistically measurable goal effect** only: the model does not discriminate successes from failures in these action categories, so the shift is not read as a distortion of a discriminating prediction |
| equivalent to zero | yes | no effect larger than δ, in a model that discriminates successes from failures in these categories |
| equivalent to zero | no | no effect larger than δ; limited interpretation |
| inconclusive | either | inconclusive; not evidence of no effect |

**Mechanism readings per model (amended, 14.U4 and 14.R5).**

| P | Dn | Dc | Reading |
|---|---|---|---|
| below zero | below zero | any | **Naming / partial-completion penalty.** Naming the crafted item, or completing a named sub-goal, lowers predicted success. Whether completing the whole task adds to it cannot be separated from the dilution of naming in the two-item line. |
| below zero | equivalent to zero or confirmed above zero | below zero | **Completion penalty, naming constrained.** Crafting the item that finishes the named task lowers predicted success, and naming the object alone does not lower it by more than δ. |
| below zero | inconclusive | below zero | **Completion contributes; naming not constrained.** No completion-only conclusion. |
| below zero | inconclusive or equivalent | inconclusive or equivalent | **Penalty present, mechanism unresolved.** |
| equivalent to zero | any | any | **No penalty larger than δ** under the clean prompt: the round-1 role gap does not reappear when the craft is held fixed. |
| above zero | any | any | **Completing a named goal raises predicted success** (the ALFWorld-like direction); no penalty. |
| inconclusive | any | any | Inconclusive on the penalty; Dn and Dc are reported but not read as a mechanism. |

*Completion-only rule.* A reading that the penalty is about completion and not naming needs the naming contribution to be constrained: Dn equivalent to zero (within δ) or confirmed above zero. An inconclusive Dn is not evidence that naming has no effect.

*Strict against overlapping parents (14.U2, revised by 14.V4).* Beside the pooled P the analysis reports the direct comparison: P on strict parents minus P on overlapping parents (word overlap only and whole name together), with its 95% interval. This comparison is descriptive and is not part of any reading rule. A difference whose interval excludes zero shows that the penalty differs with partial naming of G in the parent line. That is consistent with partial naming contributing, but the two groups also differ in which goals and parents they hold. An interval across zero does not show that partial naming plays no part, and an inconclusive strict-subset estimate next to a significant pooled one does not show that partial naming drives the pooled result.

*The parts are read causally only under assumptions (14.U4).* The two-item split is an algebraic decomposition. Reading Dn and Dc as the effects of naming and of completion needs four assumptions:

- the goal line's effects add up without interacting with the action's role, except through completion;
- adding a second item to the goal line (the conjunctive goal) dilutes the goal line equally for goal-role and ingredient actions;
- recipe structure does not interact with the goal line;
- the object's position in the two-item line does not matter.

The two-item goals are constructed counterfactual prompts, not native TextCraft tasks.

**Padding (amended, 14.U4 and 14.R9).** T confirmed above zero reads as **sensitivity to added empty-inventory steps** (elapsed turns, or the repeated emphasis of the look-ups). It does not by itself separate elapsed time from prompt length, repetition or attention effects. The change from no look-up to one (goal role minus class-matched ingredient) is reported beside T. A T whose 95% interval contains that change is "comparable to the one-look-up change" and is not read as timing. T equivalent to zero reads as **no sensitivity to added empty-inventory steps larger than δ**, consistent with the content reading but not proof of it. T confirmed below zero is F4.

**The lift.** Λ confirmed above zero: the model raises predicted success for intermediate crafts of the named goal (including any partial naming, 14.R1). Λ equivalent to zero: no lift larger than δ (Llama's round-1 pattern).

**Across models** (descriptive, among the models competent for the penalty contrast).

| Pattern | Reading |
|---|---|
| P confirmed below zero in every competent model | The penalty is general across these families under the clean prompt. |
| P confirmed below zero in some competent models, equivalent to zero or above in others | The penalty is model-specific; the models are named. |
| P below zero for Llama-3.1 only | The penalty is a Llama-3.1 property; F3 is read by its corrected rule (14.R6). |
| The same mechanism reading in every model where it is read | One mechanism across families. |
| Λ ordering matches the ordering of the overall goal effect | Consistent with the lift carrying the between-model difference; with five models this is a description, not a test. |
| Fewer than two models competent for the penalty | Round 2 cannot speak across families; per-model results are reported as they are. |

**What each pattern does not show.**

- A confirmed P shows that the goal line's naming-and-completing status changes L for a fixed craft. It does not show that the model "thinks" the task is premature. If P is not negative (F1), the top-item status of round 1's goal crafts, a property of recipe structure, is a candidate explanation of the round-1 role gap.
- The pooled P is not free of lexical contamination: 499 of the 675 parent failures have a parent sharing a word with G (201 with G's whole name inside Z's). The strict subset is reported beside it (14.U2).
- A completion reading rests on Dn and on the two-item line, a constructed prompt.
- An equivalence verdict means no effect larger than δ = 0.4 nats on the two-candidate scale. For Qwen it is out of reach for P, Dn, Dc and T at this sample size (section 10).
- Nothing here bears on the confirmatory TextCraft verdicts. The clean prompt measures a different quantity.
- All items are failures and successes from goal-blind scripted walks, not agent histories.

## 9. Descriptive readouts (estimates and 95% pair-bootstrap intervals, no p-values)

1. The overall goal effect ΔL on the 1,500 main failures, and by role, class and length bin, per model. For Qwen and Llama it is also compared with the confirmatory (verified-prompt) scores on the same items: the per-item difference, clean minus verified.
2. The aligned-minus-neutral and swapped-minus-neutral contrasts; the two-item line against neutral.
3. *(Corrected, 14.R7.)* The yes/no versions of the contrasts that have yes/no arms: Λ, Dn and Dc (on the main records). P and T are two-candidate only: the parent records have the yes/no question under the neutral line only, for competence, and the padded arms are two-candidate.
4. P by failure class, by length bin, by inputs held (none against some), by parent rule (strict against word overlap), on the parent successes, and on the strict subset alone.
5. The lift on parent records: L under the parent line minus L under the swapped line, against Λ.
6. The list control: L under the aligned line on the extended list minus on the item's own list.
7. Dn and Dc by the object's position in the two-item line, by class, by length bin, and on successes. Λ by failure class (used up, too few, never obtained) and by the ingredient's name against the goal's (whole name contained, word overlap only, no shared word; 14.R1), and the ingredient controls of Dn and Dc by the same three kinds.
8. T's two parts (goal-role and ingredient changes) separately, by length bin. The empty-start contrast: the effect with one look-up against none, per role.
9. Competence: the gate's four values; the role-level, class-level, length-bin and role-matched used-up AUCs on both scales (section 3); the two-candidate first-token mass per arm; and the AUC cost of each goal line.
10. Heuristic tracking (added 2026-11-21, before any round-2 score). For every model, on the 1,500 main failures and 600 successes under the neutral line, on both scales, the fits of four heuristics of the competence note, with their definitions exactly as committed at 9a83738:

    - H1, every input's name mentioned somewhere in the history;
    - H2, the units obtained over the history reach each input's count, consumption ignored;
    - H6, the true inventory rule;
    - H2 and H9, H2 and the action's output is not a top item of the displayed list (so a goal craft fails).

    For each: the share of items it calls a success; the tracking AUC (the model's L between the items the heuristic calls a success and those it calls a failure; for H6 this is the competence AUC); the tracking AUC within the true failures and within the true successes; and the agreement and Cohen's kappa between the model's decision (L above zero) and the heuristic. `wf/textcraft/r2_heuristics.py` ports the competence script's definitions, and a test checks that the two agree on all 2,100 items. In round 1 Llama-3.1 followed H1 most closely (tracking AUC 0.87 against 0.61 for the truth), and H2 and H9 best by kappa. The readout says whether each round-2 model reads the state or a shortcut, which bears on how its contrasts can be read; it changes no verdict.
11. Across models: a table of Λ, P, Dn, Dc, T and the overall effect, per model, with the gate result.

## 10. Power (planning figures, not results)

Planning uses the confirmatory TextCraft item spreads of ΔL (verified prompt) and the pair intraclass correlations:

- Llama: SD 1.56 on the goal role and 1.22 on ingredients; ICC 0.15 and 0.06.
- Qwen: SD 5.09 and 4.56; ICC 0.11 and 0.07.

Each contrast's standard error comes from the design effect of its items per pair. The minimum detectable effect is for a two-sided test at Holm's first step (0.05/50) with 80% power. These assume the contrasts spread like ΔL. P, Dn and Dc compare two lines that share more of the prompt than aligned and swapped do, so their spread is probably smaller; T is a difference of differences, so its spread could be larger.

| Contrast | Items (pairs) | Llama SE | Llama MDE (nats) | Qwen SE | Qwen MDE (nats) |
|---|---|---|---|---|---|
| Λ | 990 (162) | 0.044 | 0.18 | 0.168 | 0.69 |
| P | 675 (111) | 0.080 | 0.33 | 0.24 | 0.99 |
| Dn, Dc | 500 goal (158) and 496 class-matched ingredient (116) | 0.099 | 0.41 | 0.33 | 1.38 |
| T | 336 goal (129) and 332 class-matched ingredient (95) | 0.118 | 0.49 | 0.40 | 1.65 |

(Amended for the class-matched ingredient rows, 14.R2.)

**Equivalence power and reachable bounds (14.U3).** With δ = 0.4 nats, a TOST verdict needs the estimate's distance from zero plus z standard errors to be below δ. Here z = 3.09 at Holm's first step (0.05/50, the most stringent level a test can face) and 1.645 at the least stringent step. The table gives, for each planning SE, the largest |estimate| that can still be declared equivalent at each level, and the power of the equivalence test at a true zero.

| Model, contrast | Largest equivalent \|estimate\|, Holm first step | at 0.05 | Power at a true zero, first step | at 0.05 |
|---|---|---|---|---|
| Llama, Λ | 0.26 | 0.33 | 1.00 | 1.00 |
| Llama, P | 0.15 | 0.27 | 0.94 | 1.00 |
| Llama, Dn or Dc | 0.09 | 0.24 | 0.66 | 0.98 |
| Llama, T | 0.04 | 0.21 | 0.24 | 0.92 |
| Qwen, Λ | unreachable | 0.12 | 0.00 | 0.54 |
| Qwen, P | unreachable | 0.01 | 0.00 | 0.02 |
| Qwen, Dn or Dc | unreachable | unreachable | 0.00 | 0.00 |
| Qwen, T | unreachable | unreachable | 0.00 | 0.00 |

"Unreachable" means z × SE exceeds δ, so no estimate can be declared equivalent. For Qwen, P, Dn, Dc and T are therefore "inconclusive" unless confirmed, and never "no effect". The analysis reports, for every model and contrast, the realized bootstrap SE and these two powers, and flags an equivalence test as underpowered when its power at a true zero at the first Holm step is below 0.8. The new models' spreads are unknown; in ALFWorld their effects were on Llama's scale (0.8–1.1 nats two-candidate), so Llama's rows are the closer guide for them.

For Llama-sized spreads the expected P (−0.5 to −0.9) is detectable, and an F1 equivalence verdict is reachable for |P| below about 0.15. For Qwen-sized spreads only large effects are detectable.

## 11. Cost

*(Updated for the amended arms, 14.R8.)* Each model scores 25,743 (record, arm) combinations:

- main: 2,090 × 8 + 10 × 6 = 16,780;
- padding: 920 × 4 = 3,680;
- parent: 882 × 5 + 873 = 5,283 (aligned, the three extended-list lines and the yes/no neutral question on every record, and the extended-list two-item line on the 873 that have one).

That is 128,715 scores over five models. The yes/no arms (9,272 per model: 8,390 main and 882 parent) are read from one forward pass each (`score_one_token_pair`, equal to `score_pair` by construction for one-token candidates). The 16,471 two-candidate arms take two passes each, so each model needs about 42,200 forward passes.

Round 1's rate was 51,200 scores (both models, all arms through `score_pair`) in about 4 GPU-hours on four shards, about 12,800 scores per GPU-hour; the diagnostics ran about 15,000 scores in 98 GPU-minutes. Prompts are of similar length. Estimated GPU time:

- Qwen3-8B, Llama-3.1-8B and Mistral-7B: about 1.65 hours each (round 1 ran about 25,600 forward passes per GPU-hour);
- Llama-3.2-3B: about 0.9 hours;
- Gemma-2-9B with eager attention (slower; taken as twice the rate's time): about 3.3 hours.

That totals about **9.2 GPU-hours**, plus about 0.5 for a 20-record smoke run per model and model loading; budget **10–12 GPU-hours**. Each model runs as 4 shards by goal pair, one job per shard of under 1.5 hours (Gemma's under 2 hours), saving every score as it is made, with resume. The CPU tokenization check took 9 to 12 minutes per model (section 13).

## 12. Code (built after this note; see section 13 for the checks' results)

- `wf/textcraft/prompts.py`: `CLEAN_SYSTEM`, the default from round 2 on.
- `wf/textcraft/r2.py`: arms, prompts, models, item hash.
- `wf/textcraft/r2_items.py`: build and verify, writing `r2_manifest.json`.
- `wf/textcraft/r2_tokcheck.py`: tokenization gate and golden renderings, on CPU.
- `wf/textcraft/r2_score.py`: the scoring run `textcraft_r2_score_<tag>`, sharded and resumable, with the gate before any score.
- `wf/textcraft/r2_analysis.py`: the run `textcraft_r2_analysis`, with the gate, the contrasts, the Holm family, the verdicts and the descriptive readouts.
- `wf/textcraft/r2_heuristics.py`: the competence note's heuristics H1, H2, H6, H9 and H2 and H9, ported from `scripts/textcraft_llama_competence.py` (definitions fixed at 9a83738).
- Configs `wf/config/textcraft_r2_*.yaml`; jobs `pace/textcraft_r2_tokcheck.sbatch` and `pace/textcraft_r2_score.sbatch`.
- Tests `tests/test_textcraft_r2.py`; mutants in `scripts/mutation_check.py`, with their killing tests listed in `notes/testing.md`.

## 13. Checks run before scoring (2026-11-21; no model has scored a round-2 record)

*These are the checks on the records before the amendments of section 14 (23,988 scores per model, records sha256 `ac971f15…`). The rerun on the amended records is in 14.C.*

**Records.** `r2_items.jsonl` (sha256 `ac971f15…f32ea837`): 2,982 records, 23,988 (record, arm) combinations per model. `python -m wf.textcraft.r2_items verify` marked the manifest verified:

- *Per-record checks in three fresh processes* (hash seeds 0, 1 and 2), on all 2,982 records: no replay, checker, structure or prompt failure, no duplicate id, no prompt-example goal. The replays covered the parent records on their own and their extended lists under goal, partner, parent and no goal, and every padded history at both padding levels under goal, partner and no goal. Every look-up replied with the padding text, and every reply and twin reproduced.
- *Determinism.* Two fresh builds (hash seed 0 with 8 worker processes, hash seed 1 with 3) wrote the records byte for byte.
- *Base items.* The frozen manifest is verified and hashes `items.jsonl` and `fewshot.json`.
- The recipe database copied to ICE hashes to the pinned `5216d3a5…d26d65f`.

**Tokenization, on ICE CPU** (checkout `text-wm-fix28` at 26de317, partition coc-cpu, jobs 7219636–7219640, the serve environment's transformers 4.51.3, each pinned tokenizer, offline; `wf/textcraft/data/r2_tokcheck_ice.json`). The gate of section 3 ran on every record and arm for each model: 31,196 candidate checks (both candidates of 15,598 two-candidate prompts) and 8,390 yes/no prompts per model.

| Model | Gate | Problems | Yes / No tokens | First tokens (success vs failure) | Longest input (tokens) | Over 4,096 | Plain join differs | Minutes |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | pass | 0 | single, ids 9454 / 2753 | "Craft" vs "Could" | 1,790 | 0 | 0 | 11 |
| Llama-3.1-8B | pass | 0 | single, ids 9642 / 2822 | "Craft" vs "Could" | 1,808 | 0 | 0 | 10 |
| Gemma-2-9B-it | pass | 0 | single, ids 3553 / 1294 | "C" vs "Could" | 1,677 | 0 | 0 | 9 |
| Mistral-7B-v0.3 | pass | 0 | single ("▁Yes", "▁No"), ids 6360 / 2538 | "Craft" vs "Could" | 2,021 | 0 | 31,196 (by design) | 10 |
| Llama-3.2-3B | pass | 0 | single, ids 9642 / 2822 | "Craft" vs "Could" | 1,808 | 0 | 0 | 12 |

The clean prompt therefore passes the tokenization checks for all five models, the condition the owner set for making it the default for later TextCraft runs. For every model the template's ids equal the plain tokenization of its rendered text. Mistral's plain join differs for every candidate because its template puts a space before the reply, which the gate's template-reply check takes into account (wave A found the same). No input reaches Gemma 2's 4,096-token window. One difference to keep in mind when reading first-token numbers: Gemma 2 splits "Crafted" as "C" + "rafted", so its first-token term compares "C" with "Could".

**Golden renderings.** `r2_chat_golden.json` holds, for each model, five renderings made in the same jobs: the first padded main record under the aligned, two-item, yes/no two-item and eight-look-up aligned arms, and the first parent record under the parent line. A test checks, without a tokenizer, that each stored text holds every turn in order, ends in the model's generation prompt, and that Mistral's system text sits after the examples.

**Dry run without models.** Mock scoring of all five models (4 shards each) and the analysis on the real records: 119,940 scores with none missing, and the analysis wrote every block and a family of 50 tests. As expected, every mock model fails the competence gate and every contrast reads "not interpretable".

**Tests and mutants.** `tests/test_textcraft_r2.py`: 72 passed (no model, no real tokenizer; the golden test reads the stored renderings). Full suite (`tests/`): 934 passed, 7 skipped. `scripts/mutation_check.py` holds 54 round-2 mutants, covering the prompts and arms, the item hash, the record builder and its checks, the scoring run and its gate, the tokenization gate, the analysis's contrasts, family, margins, competence gate, verdicts and readings, and the heuristics. The final run against the final tests killed all 54. In the first run two survived and were killed by tests added for them, and one equivalent mutant was removed (`notes/testing.md` lists each mutant with the test that killed it). The scope of this claim is these 54 mutants against the test suite.

**What has not been run.** No model has scored any round-2 record: there has been no smoke run and no GPU job. The design is to be reviewed and frozen at a tag before the smoke runs.

## 14. Amendments before freeze (user decisions and review 1, 2026-11-21)

These changes were made on 2026-11-21, before any round-2 score. No model has scored a round-2 record, and no round-2 effect has been computed or looked at. Each item is tagged by its source: **U** for a decision of the owner, **R** for review 1 (engineering and science, "pass with fixes", at f8dd14e), **S** for review 1's should-fix list. Sections 1 to 11 were corrected in place where they were wrong, with a pointer here; this section states the final rules.

### 14.U1 Competence by contrast (user decision; populations and thresholds from review 1)

The first version read every contrast through one overall gate, and a model that failed it was "not interpretable" for everything. Round 1 showed that Llama-3.1's competence differs by action category, so one gate would hide where it is and is not competent. **These rules were chosen with knowledge of round-1 data.** That knowledge includes Llama's round-1 role AUCs (yes/no 0.85 on the goal role, 0.55 on ingredients, both from round 1's *unanchored* question, 14.Q), its class profile and the competence note's heuristic fits. The disclosure table at the end of this item gives the values the rules would have produced on the confirmatory scores.

**The final competence rules (verbatim).**

1. *Overall gate (wave A's rule, unchanged, reported for every model).* On the 1,500 main failures and the 600 determined successes, under the neutral line, on the yes/no question, all four must hold:
   - the AUC is at least 0.70;
   - Yes (L above 0) on at most 95% of the failures;
   - Yes on at least 5% of the successes;
   - the median of P(Yes) plus P(No) is at least 0.5.
2. *Contrast-specific competence.* A model is competent for a contrast when both of these hold:
   - the yes/no neutral-line AUC (point estimate; no interval; no fallback to the two-candidate scale) is at least 0.70 in every population listed for that contrast;
   - the other three criteria of rule 1 (the two Yes shares and the mass) hold on the full population.

   The populations (failures against successes, each from the round-2 records):
   - *Λ:* ingredient-role main failures (990) against ingredient-role successes (300).
   - *P:* both (a) parent failures (675) against parent successes (207), on the parent records' new yes/no neutral arm, and (b) goal-role main failures without used up (504) against goal-role successes (300).
   - *Dn and Dc:* goal-role main failures (504) against goal-role successes (300), and class-matched ingredient failures (too few and never obtained, 496) against ingredient successes (300). The smaller of the two AUCs is the one compared with 0.70.
   - *T:* padded goal-role failures (336) against goal-role successes at 1 to 8 steps (200), and padded class-matched ingredient failures (332) against ingredient successes at 1 to 8 steps (200). The smaller AUC is compared with 0.70.
3. *Use.* Competence never withholds an estimate, a test or a verdict, and all 50 tests stay in the fixed Holm family. It decides how a verdict is read (section 8):
   - a confirmed effect in a competent model is a **distortion in a model that discriminates successes from failures in these categories** (worded at sign-off, 15.F: "competent" in this note always means passing an AUC gate, which is predictive discrimination, not demonstrated inventory-rule competence or understanding of the environment);
   - in a model that is not competent for that contrast it is a **statistically measurable goal effect** only.

**Reported for every model and population:**
- the success classification rate (Yes, or L above 0, on successes) and the failure classification rate (No, or L below 0, on failures), on the yes/no scale and, for the main records, on the two-candidate scale;
- both AUCs with pair-bootstrap intervals;
- the heuristic tracking of H1, H2, H6 and H2∧H9, with the definitions as committed at 9a83738, within that population.

Review 1 notes that Llama's goal-role competence in round 1 was largely "input names seen" (H1) coinciding with the truth there. The tracking within each population shows whether that recurs.

**Disclosure: these rules on the confirmatory scores** (verified prompt, round 1; yes/no neutral-line AUC from round 1's *unanchored* question, 14.Q; the parent population did not exist then):

| Population | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| ingredient, all (Λ) | 0.55 | 0.85 |
| ingredient, class-matched (Dn, Dc) | 0.68 (never obtained 0.81, too few 0.55; used up 0.42) | 0.80 |
| goal role (P, Dn, Dc) | 0.85 | 0.87 |
| padded goal role (T) | 0.85 | 0.89 |
| padded class-matched ingredient (T) | 0.67 | 0.85 |

Under these rules Llama-3.1 would have been competent for the goal-role part of P (its parent part is unknown) and not competent for Λ, Dn, Dc or T. Qwen would have been competent for all. The class-matched control changes Llama's ingredient competence population: its AUC there is 0.68, against 0.55 over all ingredients (14.R2).

### 14.U2 The parent goal (user decision; subsets from review 1)

The pooled P over all 675 parent failures stays the primary contrast. It is **not free of lexical contamination**: 499 of the 675 failures have a parent sharing a word with G. Three subsets are pre-registered, each with its estimate and 95% pair-bootstrap interval, plus the three pairwise differences (descriptive, no p-values):

- *strict*: the parent shares no word with G (176 failures, 29 pairs);
- *word overlap only*: a shared word but not G's whole name (298 failures, 52 pairs);
- *whole name*: G's whole name, as a contiguous word sequence, inside Z's name (201 failures, 33 pairs).

*Rule (withdrawn by 14.V4).* The first version of this item read a pooled penalty with a non-negative strict-subset estimate as "penalty possibly carried by partial naming". That inference is not valid. It is replaced by the direct comparison of strict and overlapping parents, with its interval (section 8).

*Also reported (S):*
- P on each line's own faithful display: L under "craft G" on the item's own list minus L under "craft Z" on the extended list, which is P plus the list control.
- The within-item split from the new extended-list two-item arm, "Goal: craft A and B." on the extended list (873 parent records; the pair whose goal name contains "and" has none). L(ext_aligned) − L(ext_both) and L(ext_both) − L(ext_parent) add up to P exactly on those records. This split is descriptive.

### 14.U3 The equivalence margin (user decision; fixed by review 1)

The margin is δ = min(0.3 × SD, 0.4 nats). It is fixed now (not provisional), applies on the two-candidate scale, and is the same for all five contrasts (section 8). On round 1's spreads the cap binds for both reference models. Section 10 gives each contrast's power at a true zero and the largest estimate that can still be declared equivalent. Every test also reports its realized bootstrap SE and power, and is flagged underpowered when its power at Holm's first step is below 0.8. When equivalence is not reached, the verdict is "inconclusive" (with "equivalence underpowered" where flagged), never "no effect". For Qwen, equivalence on P, Dn, Dc and T is unreachable at this sample size.

### 14.U4 Mechanism claims, tightened (user decision)

1. An inconclusive Dn is not evidence that naming has no effect. A completion-only reading needs the naming contribution to be constrained: Dn equivalent to zero (within δ) or confirmed above zero (section 8, completion-only rule). Otherwise the reading is "completion contributes; naming not constrained".
2. The two-item split is an algebraic decomposition. Reading its parts causally needs four assumptions:
   - the goal line's effects add up without interacting with the action's role, except through completion;
   - the conjunctive goal (a second named item) dilutes the line equally for goal-role and ingredient actions;
   - recipe structure does not interact with the goal line;
   - the object's position in the two-item line does not matter.

   None of these is tested here. Position is reported as a description.
3. Padding shows sensitivity to added inventory-check steps. It does not by itself distinguish elapsed time from prompt length, repetition or attention effects (section 8, padding).
4. Two-item goals are constructed counterfactual prompts, not native TextCraft tasks (section 4.1).

### 14.U5 The round-1 result and the central question (user decision)

The round-1 replication across models failed under its preregistered rule, and that verdict stands. Round 2 tests, prospectively, explanations that round 1's exploratory analyses suggested. The central question is whether Llama's negative TextCraft effect arises from:

- a goal-completion or naming mechanism;
- poor reasoning about the environment's state;
- or a combination.

*(The mapping below is superseded by the coded, mutually exclusive rule of 15.C, whose labels were renamed at sign-off (15.F), and the rejection list by 15.R as corrected in 15.F.)* The results map to it as follows (for Llama-3.1-8B; the same reading applies to any model):

- *Mechanism:* Llama is competent for P (both goal-crafting populations), P is confirmed below zero, and the mechanism reading of section 8 applies.
- *Poor state reasoning:* Llama is not competent in the populations where its effect sits, and its scores there track H1 or H2∧H9 more closely than H6. Any confirmed contrast is then a measurable effect, not a distortion of a working state judgement.
- *Combination:* P is confirmed below zero where Llama is competent (goal crafting), and Λ is null where it is not (ingredients). This was round 1's descriptive pattern.

**The two-component decomposition is rejected:**

- for a model competent for P, if P is equivalent to zero or confirmed above zero (F1 for Llama-3.1);
- across families, if F3 holds by its corrected rule (14.R6);
- for a model competent for P, Dn and Dc, if P and the round-2 role gap (Dn plus Dc, reported as `role_gap_main`) are both clearly signed but opposite. That is, P confirmed in one direction while the role gap's 95% interval lies on the other side: the within-item and between-item measures of the same penalty then disagree.

How the penalty depends on partial naming is read from the strict-against-overlapping comparison (section 8), not from a rule.

### Review 1: must-fix items

- **14.R1 Ingredient overlap.** 683 of the 990 ingredient objects share a word with the goal's name (the first version said 635), and in 180 the ingredient's whole name lies inside the goal's. "Never names" was dropped (section 6). Λ and the ingredient controls of Dn and Dc are reported separately for whole name contained (180), word overlap only (503) and no shared word (307).

  Containment is word-wise: "stone stairs" is a character substring of "smooth sandstone stairs" but not a word sequence in it, so it counts as word overlap only.
- **14.R2 Class-matched controls.** The ingredient rows of Dn, Dc and T are restricted to too-few and never-obtained failures: 496 main and 332 padded. Without this, the difference in differences also compared failure classes, since goal-role failures are never used up. Λ keeps all 990 and is also reported by class. This changes Llama's ingredient competence population (14.U1).
- **14.R3 The identity's scope.** Dn + Dc equals the goal-minus-ingredient ΔL exactly only on the 500 goal-role and 496 class-matched ingredient failures with a two-item line. It splits the round-2 role gap under the clean prompt, unmatched, and not round 1's (section 6).
- **14.R4 The ALFWorld mapping.** ALFWorld's goal-directed part, (aligned − mention) − (swapped − fourth), holds naming fixed and varies whether the action advances the task, so it maps to Dc, not Λ. Λ predictions now rest on TextCraft round 1 only. The Dc predictions use the goal-directed part on completing moves (`notes/textcraft_vs_alfworld.md`, branch `xenv-compare`, section 10): Llama −0.81 [−1.02, −0.60] gives below zero; Qwen +0.56 [0.17, 0.95] gives not below zero. The Dn expectation was softened to none, because ALFWorld's mention part names the object in a goal the action does not advance, while in Dn crafting G completes a named sub-goal. The predictions table in section 7 is the revised one.
- **14.R5 Reading a negative Dn.** It reads "naming or sub-goal-completion penalty" (section 6, caveat a; section 8); relabelled "naming / partial-completion" by 14.V1.
- **14.R6 F3.** F3 holds only if at least one model competent for P, other than Llama-3.1, has P equivalent to zero or confirmed above zero, and none is confirmed below zero. If one is confirmed below zero, F3 does not hold. Otherwise F3 is "undecided".
- **14.R7 Section 9's yes/no claim.** Only Λ, Dn and Dc have yes/no versions. P and T are two-candidate only. The wording is corrected rather than adding yes/no arms.
- **14.R8 The parent's neutral arm.** Each parent record gets `yn_neutral` on its own list (882 single-pass scores per model), so P's competence can be read on the neutral line. With the extended-list two-item arm (S, 873 two-candidate scores per model) each model now scores 25,743 (record, arm) combinations, 128,715 in all, about 9.2 GPU-hours (section 11).
- **14.R9 Padding's reading.** T above zero reads as sensitivity to added empty-inventory steps (elapsed turns or repeated emphasis). The change from no look-up to one, goal minus class-matched ingredient, is reported beside T, and a T whose 95% interval contains it is "comparable to the one-look-up change" and is not read as timing.

### Review 1: should-fix items (adopted)

- **Bootstrap size.** The 50 family tests use 20,000 pair resamples and 20,000 sign flips, computed from per-pair sums. These are the same resamples as the item-level computation; a test checks that.
- **Provenance.** The analysis refuses scores unless every shard of every model was scored at one git commit and every score carries its model's pinned revision, as `families.py` does.
- **The parent display and F1.** The penalty on each line's own display and the extended-list two-item split are reported (14.U2). If F1 holds, the top-item status of round 1's goal crafts, a property of recipe structure, is named as a candidate explanation of the round-1 role gap.
- **Counts in the scoring job.** The sbatch comment now gives 25,743 scores per model.
- **Mutants.** Review 1's surviving mutants M1–M4, M6, M9–M11, M13–M16, M21, M35, M38, M40, M48 and M51 now have killing tests, and are in `scripts/mutation_check.py` adapted to the amended code. M9 (excluded goals allowed as parents) is not equivalent: a test calls the rule with an excluded goal that consumes a real item. M52 targeted a check that repeated the expected-query check; that check was removed rather than tested. Results are in 14.C.

### 14.Q The yes/no question is anchored to the environment's replies (found by the Qwen competence analysis)

Round 1 asked "Question: Will this action succeed? Answer Yes or No." That question is not tied to the environment's reply, unlike ALFWorld's, which asks whether the environment will reply "Nothing happens." (`wf/rq1v.py`). A model can read it as "will this action help the task succeed". The Qwen competence analysis found the pattern that reading predicts: under the swapped line, Qwen's Yes on *working* goal crafts falls from 73% to 14% (−18 nats), although the craft works whatever the goal. Round 2 uses the yes/no question for every competence gate and as a scale, so it now asks:

> Question: will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that it failed ("Could not find ...")? Answer Yes if it succeeds and No if it fails.
> Answer:

(`r2.YN_QUESTION`; its sha256, `5e31ac7a…34180f05`, is pinned in a test.) It names every reply form a TextCraft action can get:

- a craft's success, "Crafted K minecraft:…";
- a get's success, "Got N …";
- both failures, "Could not find enough items to craft …" and "Could not find X".

Every round-2 candidate starts with one of these, and a test checks this. Round 1's question stays in `prompts.YN_QUESTION` for round 1 only.

Consequences:

- Round-2 yes/no scores are a different quantity from round 1's.
- Every round-1 yes/no figure quoted in this note came from the unanchored question: Llama's yes/no AUC of 0.61 overall and 0.85 on the goal role, the yes shares, and 14.U1's disclosure table. These figures are a guide to how the gates behave, not a forecast of round 2's.
- The item hash covers the question, so scores made with another question can never attach to these records.
- The tokenization check and the golden renderings were rerun with the new question for all five models (14.C). "Yes" and "No" must still be single tokens equal to the template's reply after every yes/no prompt.

### 14.V The owner's final points (2026-11-21, before any round-2 score)

**14.V1 Two-item completion semantics.** Under "craft A and B", crafting one of the two items would complete the whole objective if the other were already made or held. Every record with a two-item line was checked, text only (`r2.two_item_partner_done`): either goal appears as the product of a history step, or is held after the history. There are **no** such records among the 2,963 records with a two-item line (2,090 main and 873 parent; by role and class: none in any cell). This holds by construction, since the walk never crafts a goal and goals cannot be fetched. Verify counts them on every run.

The rule, fixed now: any such record leaves the naming and completion contrasts and the role-gap readout. It would be reported separately, but none exist.

Because crafting G under the two-item line still completes one *item* of a named two-item goal, Dn is labelled **"naming / partial-completion"** everywhere, and its negative reading is "naming / partial-completion penalty".

**14.V2 Remaining role-comparison confounds: sensitivity analyses (not primary).** Goal-role and class-matched ingredient failures still differ in history length, in whether the missing inputs are craftable intermediates or raw materials, and in how many inputs are missing. For Dn, Dc, T and the role gap (aligned minus swapped, goal minus ingredient), the analysis reports a matched estimate:

- cells are defined by length bin, by the missing inputs (all craftable, all raw, or mixed; an input is craftable when a displayed line outputs it, read from the prompt), and by the number of missing inputs (one, or several);
- within each cell holding at least 5 items of each role, the goal-minus-ingredient mean is computed;
- cells are averaged with the smaller count as the weight, as in round 1's matched comparison;
- intervals come from a pair bootstrap.

On the records, 13 of 15 cells (981 items, on the 996 rows of the two-item contrasts; corrected in 15.S from 985) qualify for Dn, Dc and the role gap, and 9 of 10 (663 items) for T. These estimates are reported as sensitivity beside the primary estimates. They do not enter the family or the readings.

Dn and Dc do not identify psychological mechanisms on their own. They are differences in predicted log-odds between constructed prompts. Read as "naming" and "completion" effects, they rest on the assumptions stated in 14.U4: additivity, no interaction between the conjunctive goal and the action's role, equal dilution, no interaction between structure and the goal line, and no position effect.

**14.V3 The ALFWorld analogy is motivation, not prediction.** ALFWorld's goal-directed contrast changes another part of the task (the destination or the transformation) for a different kind of action. It is not the same intervention as TextCraft's one-item against two-item contrast. Qwen's positive ALFWorld estimate (+0.56 [0.17, 0.95]) is not evidence that a penalty is absent. The Dc column of the predictions table now reads "none" for Qwen, and "motivated expectation below zero, not a firm prediction" for Llama. No confirmatory rule depends on any prediction or expectation in this note.

**14.V4 Subsets and competence, read carefully.**

- *Parents.* The partial-naming reading rule is withdrawn (section 8). The direct comparison of strict against overlapping parents is reported with its interval, together with what it can and cannot show.
- *Competence.* A competence AUC measures discrimination between successes and failures. It is not proof that the model reasons about the inventory correctly: a model can discriminate through a shortcut that coincides with the truth in a population. That is why the heuristic tracking (H1, H2, H6, H2∧H9) is reported within each gated population. Task effects are reported for every model whether or not its gates pass, with the limited interpretation of section 8 when they fail.

### 14.C Checks rerun after the amendments

**Records.** `r2_items.jsonl` (sha256 `e6da991f…7e47c22f0`): 2,982 records, 25,743 (record, arm) combinations per model. `python -m wf.textcraft.r2_items verify` marked the manifest verified:

- *Per-record checks in three fresh processes* (hash seeds 0, 1 and 2), on all 2,982 records: no replay, checker, structure or prompt failure, no duplicate id, and no prompt-example goal. No record with a two-item line has a named goal already made or held (14.V1).
- *Determinism.* Two fresh builds (8 and 3 worker processes) wrote the records byte for byte.

**Tokenization with the anchored question, on ICE CPU** (checkout `text-wm-fix28` at e7ca0a2, jobs 7225166–7225170, the serve environment's transformers 4.51.3, each pinned tokenizer, offline; `wf/textcraft/data/r2_tokcheck_ice.json`, which supersedes the reports of jobs 7219636–7219640). Per model: 32,942 candidate checks (both candidates of 16,471 two-candidate prompts) and 9,272 yes/no prompts, every one with round 2's anchored question.

| Model | Gate | Problems | Yes / No token ids | First tokens (success vs failure) | Longest input (tokens) | Over 4,096 | Plain join differs | Minutes |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | pass | 0 | 9454 / 2753, single | "Craft" vs "Could" | 1,819 | 0 | 0 | 13 |
| Llama-3.1-8B | pass | 0 | 9642 / 2822, single | "Craft" vs "Could" | 1,837 | 0 | 0 | 11 |
| Gemma-2-9B-it | pass | 0 | 3553 / 1294, single | "C" vs "Could" | 1,706 | 0 | 0 | 8 |
| Mistral-7B-v0.3 | pass | 0 | 6360 / 2538, single ("▁Yes", "▁No") | "Craft" vs "Could" | 2,054 | 0 | 32,942 (by design) | 9 |
| Llama-3.2-3B | pass | 0 | 9642 / 2822, single | "Craft" vs "Could" | 1,837 | 0 | 0 | 11 |

The anchored question keeps "Yes" and "No" as single tokens, equal to each template's reply after all 9,272 yes/no prompts. The golden renderings (`r2_chat_golden.json`) were remade from the same jobs.

**Dry run without models.** Mock scoring of all five models (4 shards each) and the analysis on the amended records gave 128,715 scores with none missing. The analysis wrote every block: the overall and contrast-specific competence, the 50-test family, verdicts, interpretations, readings and the sensitivity analyses.

**Tests and mutants.** `tests/test_textcraft_r2.py`: 102 tests pass. The full suite (`tests/`), run once at the end: 964 passed, 7 skipped. The round-2 test file was rerun after three of its tests were strengthened. `scripts/mutation_check.py` holds 102 round-2 mutants, and all are killed; `notes/testing.md` lists each with the test that killed it.

**What has not been run.** No model has scored any round-2 record: there has been no smoke run and no GPU job. Next come the confirming review, the summary to the owner and the freeze tag; then smoke runs and scoring.


## 15. Amendments after review 2, before freeze (2026-11-22)

Review 2 (engineering and science) read the design at a63e956 and passed it with fixes. Nothing has been scored. No model has scored a round-2 record, and no round-2 effect has been computed. The records and prompts do not change in this section, so the record hash, the manifest and the ICE tokenization check of 14.C stand. Everything here is analysis code, tests and text.

### 15.R The rejection list (coded; replaces the falsifiers of section 7 and 14.U5's list)

Each criterion is computed by the analysis: `rejections_for_model` per model, `across` across models. Each states the competence it needs. A criterion whose competence fails is "not testable" and never fires. The role gap uses 20,000 pair resamples, like the family tests.

- **R1, no penalty.** For any model competent for P, P is equivalent to zero or confirmed above zero. *Rejects* the penalty component of the decomposition for that model. (F1, now for every model, not only Llama-3.1.)
- **R2, the penalty is not shared.** Among the models competent for P:
  - at least one model other than Llama-3.1 has P equivalent to zero or confirmed above zero, and none is confirmed below zero → R2 fires;
  - if any is confirmed below zero, R2 does not fire;
  - otherwise R2 is "undecided".

  *Rejects* the claim that the penalty is shared across families. (F3, 14.R6.)
- **R3, P and the role gap disagree.** *(At sign-off, 15.F: a descriptive consistency diagnostic outside the 50-test family, with no familywise error control; its exact specification is in 15.F.)* This needs the model to be competent for P, Dn and Dc. It fires when three things hold together:
  - P is confirmed in one direction;
  - the round-2 role gap (aligned minus swapped, goal-role minus class-matched ingredient failures, on the 996 rows of the two-item contrasts; equal to Dn + Dc) has a 95% interval on the other side of zero;
  - the matched role gap (matched on length bin, craftable or raw missing inputs, and number of missing inputs, as in 14.V2) agrees in sign with the unmatched role gap, and its own 95% interval also excludes zero.

  If P and the unmatched role gap disagree but the matched role gap does not confirm, the analysis reports "P and the unmatched role gap disagree; the matched role gap does not confirm (item-structure confound)" and does not reject. *Rejects* the decomposition for that model, since the within-item and between-item measures of the same penalty then disagree. *(Owner decision, adopted from review 2's Qwen objection; flagged for the owner before the freeze.)*
- **R4, the lift does not explain the model difference.** *(Withdrawn as a rejection at sign-off, 15.F: R4 is now a descriptive comparison with a paired interval for Λ(Llama-3.1) − Λ(Qwen3).)* This needs both Llama-3.1 and Qwen3 to be competent for Λ. It fires when Llama-3.1's Λ is confirmed above zero and its estimate is at least Qwen3's. *Rejects* the claim that the lift carries the Qwen–Llama difference. (F2. Its second half, the Λ ordering across five models, stays a description and never rejects.)
- **R5, added steps deepen the penalty.** For a model competent for T, T is confirmed below zero. *Rejects* the claim that the penalty is largest early, in either reading. (F4.) *(Reworded at sign-off, 15.F: R5 rejects the predicted attenuation, not content-based explanations.)*

**Not tested.** The account's first claim, that the penalty is "of similar size across models", has no test here. Only its sign is tested (R1, R2). Sizes are reported, but no comparison of sizes across models is planned.

**Qwen's routes to rejection, stated explicitly.**

- R1 can fire for Qwen only through "P confirmed above zero": equivalence is unreachable for Qwen's P at this sample size (section 10).
- R3 can fire for Qwen if its P and its role gap point in opposite directions and the matched role gap confirms the role gap. Review 2's concern was that Qwen's unmatched role gap could be positive for item-structure reasons; the matched requirement addresses that.
- R2 counts Qwen like any other model competent for P.
- R4 uses Qwen's Λ only as the comparison.
- R5 applies if Qwen is competent for T.

### 15.C The central question: a descriptive mapping (outside the Holm family)

For each model the analysis gives exactly one of four answers (`central_question`). It uses:

- P's verdict;
- competence for P;
- within the gated goal-role population (goal-role main failures without used up, against goal-role successes, on the yes/no question under the neutral line), the tracking AUCs of H1 and of H2∧H9 minus that of H6, the true inventory rule. Each difference has a paired pair-bootstrap 95% interval (2,000 resamples; the same resample for both AUCs).

The answers:

*(Labels renamed at sign-off, 15.F; the operational criteria are unchanged. The first labels were Mechanism, Combination, Poor state reasoning and Undecided. The labels describe how the scores co-vary with text heuristics; they do not identify an internal mechanism.)*

- **Penalty with inventory-consistent tracking:** P confirmed below zero, the model is competent for P, and H6's tracking AUC is at least H1's and at least H2∧H9's (each difference at or below zero).
- **Penalty with shortcut-consistent tracking:** P confirmed below zero, and H1 or H2∧H9 tracks better than H6, with the difference's 95% interval above zero.
- **No confirmed penalty; shortcut-consistent or non-discriminating:** P not confirmed below zero, and either the model is not competent for P, or H1 or H2∧H9 tracks better than H6 with the difference's 95% interval above zero.
- **Undetermined:** anything else.

The answers are mutually exclusive:

- The first two both need a confirmed P, and the third needs its absence.
- The first needs both differences at or below zero, while the second needs one of them above zero with its interval.

*A difference that is undefined counts for neither side.* H2∧H9 calls every goal-role craft a failure, since the goal is a top item, so its tracking AUC within the goal-role population is undefined. The comparison then rests on H1. Inside the goal-role population, H2∧H9's signature is a low success classification rate on goal-role successes, which is reported.

"Λ null" in any description means Λ equivalent to zero, not inconclusive. The mapping is reported per model with its inputs. It is not a test and changes no verdict.

### 15.S Other changes

- **Readings say when competence fails.** `reading()` now appends "measurable effect only; not competent for …" whenever a contrast the reading uses (P, Dn, Dc for the penalty reading; T; Λ) is not competent.
- **Fallback in the reading table.** Any combination of P confirmed below zero with Dn and Dc not listed in section 8 (for example Dn confirmed above zero with Dc inconclusive, or Dc confirmed above zero) reads "penalty present, mechanism unresolved".
- **The same rows for the role gap.** The two-item contrasts, the role gap, its matched version and the descriptive two-item readouts all use the same 996 rows: 500 goal-role and 496 class-matched ingredient failures that have a two-item line and no named goal already made. The matched role gap uses 981 of them, in 13 of 15 cells.
- **What "padded" means for T's competence.** T's populations use the unpadded yes/no question under the neutral line, on the padded subset of failures (the padded records are scored with padding only under the aligned and swapped lines). The successes are the unpadded successes in the 1–3 and 4–8 step bins.
- **Score checks.** A score whose condition is not one of its record's arms now stops the analysis.
- **Stale text fixed.** Section 5 (resamples), section 7 (the two ALFWorld paragraphs, the competence paragraph, the falsifiers), 14.U1 (unanchored-question marker), 14.V2 (981 items), the analysis docstring, and the deviations entry (the partial-naming rule was withdrawn).
- **Tests and mutants.** Tests kill review 2's verdict mutants N1 (a confirmed effect above zero needs the interval), N10 (the two-sided bootstrap p), N19 (difference checked before equivalence) and N24 (the sign flip uses the absolute value), and its survivors N5–N9, N12, N14–N17, N22, N23 and N25. Each new rule has its own test and mutant. Results are in 15.T.

### 15.T Checks after review 2

- **Records and prompts.** Unchanged (sha256 `e6da991f…`). The verification and the ICE tokenization check of 14.C stand.
- **Tests.** `tests/test_textcraft_r2.py` holds 132 tests. The full suite was run once at the end, at 53665c5: 994 passed, 7 skipped.
- **Mutants.** The official run at ef73d40 (four shards; log `notes/textcraft_r2_mutation_log.txt`) killed 136 of 137 round-2 mutants. The one survivor had become equivalent once the role-gap readout read the shared row set, and it was removed together with the redundant guard it targeted. All 136 remaining mutants are killed, each by a named test (`notes/testing.md`). A screen run first found two of review 2's mutants (N11, N13) unkilled; tests for both were added before the official run.
- **What has not been run.** No model has scored any round-2 record.


## 15.F Final corrections at sign-off (2026-11-22)

The owner signed off on condition of the corrections below. Nothing has been scored, and no round-2 effect exists or was computed. The prompts, populations, sample sizes, records and model set are unchanged. The owner approved the revised R3 matching logic and keeping H1 as the shortcut comparison. The H2∧H9 population is not redesigned.

**15.F1 R3 is a descriptive consistency diagnostic.** R3 sits outside the 50-test confirmatory family, and no familywise error control is claimed for it. Its specification:

- *Rows.* The 996 rows of `two_item_rows`: goal-role main failures without used up (500 with a two-item line) and class-matched ingredient failures (too few and never obtained; 496). Records whose other named goal is already made or held are excluded; there are none.
- *Unmatched role gap.* The item-weighted mean of L(aligned) − L(swapped) over the goal-role rows, minus the same mean over the ingredient rows, with a 95% percentile interval from 20,000 resamples of whole goal pairs.
- *Matched role gap.* Cells are defined by three things:
  - length bin (1–3, 4–8, 9–12);
  - kind of missing inputs: all craftable (each input is the output of a displayed line), all raw, or mixed;
  - number of missing inputs: one, or several.

  A cell is used when it holds at least 5 items of each role. Within a used cell the estimate is the goal-role mean minus the ingredient mean. The cells are averaged with each cell weighted by its smaller role count. The 95% interval comes from 20,000 resamples of whole goal pairs, with the cells and weights recomputed in every resample. On the records, 13 of 15 cells (981 items) are used.
- *Firing condition.*
  - The model is competent for P, Dn and Dc.
  - P is confirmed: Holm-adjusted p below 0.05, with its 95% interval on the estimate's side.
  - The unmatched role gap's 95% interval lies entirely on the other side of zero.
  - The matched role gap's 95% interval lies entirely on the same side as the unmatched one.
- *Reporting.* When R3 fires, the report says the within-item and between-item measures of the penalty disagree, which counts against the decomposition for that model, as a descriptive finding. When P and the unmatched gap disagree but the matched gap does not confirm, the report says "P and the unmatched role gap disagree; the matched role gap does not confirm (item-structure confound)".

**15.F2 R4 is descriptive.** The account is not rejected on the ordering of two point estimates. The analysis reports both models' Λ and verdicts, with their competence, and the difference Λ(Llama-3.1) − Λ(Qwen3) on the same ingredient failures. The difference gets a paired pair-bootstrap 95% interval: one resample of goal pairs serves both models (`lift_difference`, 2,000 resamples).

A confirmatory version would need a test of the difference added to the Holm family, which would change the family's size after the design was reviewed. It would also need a margin for "at least as large". Neither can be fixed cleanly now without reopening the family, so R4 is descriptive.

**15.F3 R5 rejects the predicted attenuation only.** For a model competent for T, T confirmed below zero contradicts the predicted attenuation of the penalty with added inventory-check steps, that is, the "largest early" claim read as timing. It does not rule out content-based explanations of round 1's gradient.

**15.F4 Competence is discrimination.** The gates are unchanged. Passing one means the model's yes/no scores discriminate successes from failures in those action categories (AUC at least 0.70, with the other three criteria). It does not demonstrate that the model uses the inventory rule or understands the environment, which is why the heuristic tracking is reported in each gated population. Wording changed accordingly:

- in the note, section 8's table and 14.U1;
- in `interpretation()`: "distortion in a model that discriminates successes from failures in these categories";
- in `reading()`: "measurable effect only; does not discriminate successes from failures for …";
- in the STATUS lines: "discriminates" or "does not discriminate".

Where this note says "competent for X", it means passing X's discrimination gate.

**15.F5 Neutral labels for the central question.** The four answers are now:

- "penalty with inventory-consistent tracking";
- "penalty with shortcut-consistent tracking";
- "no confirmed penalty; shortcut-consistent or non-discriminating";
- "undetermined".

The criteria are unchanged (15.C). Agreement with a heuristic's predictions is a correlation and is not presented as an identified internal mechanism.

## 15.A Statistical audit at sign-off (2026-11-22)

Each item was checked by reading the code and by a test that fails if the property breaks (`tests/test_textcraft_r2.py`, the tests named here). An independent auditor will verify this afterwards.

| Property | Where | How it was checked | Result |
|---|---|---|---|
| Holm over the fixed family of 50, missing or NaN entering as p = 1 | `family()` | `test_audit_holm_family_of_fifty_with_missing_as_one` scores one model with NaN equivalence ps: 45 tests enter at p = 1, the family stays at 50, and each confirmed p is multiplied by 50 at the first step. Also `test_family_is_fixed_at_fifty_and_pads_missing` and `test_family_is_holm_adjusted` | holds |
| TOST from one-sided bootstrap ps | `tost_p()` | `test_audit_tost_uses_one_sided_bootstrap_ps` (hand-counted draws give the larger of the two smoothed one-sided shares); `test_n25_tost_low_side_uses_the_margin`; `test_statistics_helpers` | holds |
| δ = min(0.3 × SD, 0.4), computed once per model, before any equivalence test, and used by every contrast | `margin_of()`, `model_block()` | `test_audit_margin_is_fixed_before_equivalence` counts one call per model and checks every contrast's margin; `test_margin_is_capped_at_four_tenths` | holds |
| Every interval and permutation resamples whole goal pairs | `draws`, `_fast_draws`, `sign_flip_p`, `auc_ci` and every caller | `test_audit_every_resample_is_over_goal_pairs` intercepts every bootstrap and sign flip in a model block and checks that the clusters are goal-pair ids; `test_fast_draws_equal_the_general_path`; `test_sign_flip_flips_whole_pairs`; `test_lift_difference_is_paired` | holds |
| Dn + Dc equals the role gap exactly, under item weighting, on the 996 shared rows | `two_item_rows`, `contrast_inputs` | `test_audit_naming_plus_completion_is_the_role_gap_on_the_shared_rows` uses random scores on the real rows (500 goal-role, 496 ingredient), with agreement to 1e-12; `test_role_gap_descriptive_equals_naming_plus_completion` | holds |
| R2's firing condition | `across()` | `test_across_models_patterns_and_f3`: fires, does not fire and undecided, and a non-competent model counting for neither side | holds as stated in 15.R |
| R3's firing condition | `rejections_for_model()` | `test_rejection_list_per_model`: fires only with the matched confirmation; gives the confound note otherwise; not testable without P, Dn and Dc competence; labelled a descriptive diagnostic | holds as stated in 15.F1 |
| Every inferential comparison outside the 50 is labelled descriptive | `INFERENCE_MAP`; outputs | `test_audit_inference_map` checks the map, and that no p-value appears anywhere in a model block except in the five family contrasts | holds |

**Gaps found and closed.** Three:

- R4 had applied a decision rule to two point estimates; it is now descriptive (15.F2).
- R3 was described as rejecting without saying that it sits outside the multiplicity correction; it is now labelled (15.F1).
- The analysis output did not say which comparisons were confirmatory; `inference_map` now does.

No error was found in the Holm, TOST, margin, clustering or identity code.

**Tests and mutants at sign-off.**

- `tests/test_textcraft_r2.py`: 141 tests pass.
- Mutants: 145 round-2 mutants. The 22 new or changed ones were run in official mode at 170db45, and all were killed by named tests (log appended to `notes/textcraft_r2_mutation_log.txt`). The other 123 were killed in the official run at ef73d40, and their code is unchanged.
- The full suite was run once at the end: FULL_SUITE_SIGNOFF.
