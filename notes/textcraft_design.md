# TextCraft replication: design (written 2026-11-20, before any item was generated)

This note fixes the design of the TextCraft replication of RQ2 before items exist. It is meant to be copied into the decision log. Every rule a checker needs is stated here in words, so that a second checker can be written from this note alone, without reading our code.

## 1. Question and prediction

**Question.** On ALFWorld, a prompted world model predicts success more often for an action whose failure the history already settles when the task line names the action's object (aligned) than when it names another object (swapped). Does the same goal effect appear in a different environment, TextCraft, where failures follow from inventory arithmetic rather than from where objects are?

**Score.** For each item, L is the log-probability of the success reply minus that of the failure reply, both teacher-forced after the prompt (`LogoddsScorer.score_pair`). The goal effect is

$$\Delta L = \frac{1}{n}\sum_{i=1}^{n}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].$$

A world model that reads the history correctly gives a ΔL of zero, because the history alone decides the reply.

**Prediction (logged now).** *(The main population is drawn in same-history couples from section 12, B1 on; section 12, C7 adds named readouts below this test.)* ΔL > 0 on the main population for both Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct. For each model the prediction counts as confirmed when ΔL > 0, the 95% bootstrap interval clustered by goal pair excludes zero, and the permutation p-value (whole goal pairs flipped), Holm-adjusted over the two models, is below 0.01. The replication holds if it is confirmed for both models. We also report the RQ2 magnitude rule's verdict (support needs ΔL of at least 0.5 nats, strong at 1.0), as a description, not as the test.

**What a null would mean.** *(The ±0.25-nat criterion is replaced by equivalence margins, 1.0 nats for Qwen and 0.3 for Llama, in section 12, C4.)* A ΔL near zero with a narrow interval (say, within ±0.25 nats) would mean the ALFWorld effect does not carry over to a crafting game whose failures are inventory arithmetic. That would point to something specific to ALFWorld, such as the task line acting as evidence that its object exists somewhere, or to the "Nothing happens." reply format, rather than to a general bias toward the task's object. Two readings would need to be told apart before drawing that conclusion: the models may fail at the accounting itself, so that L is high under every task line and there is no room for a shift (the neutral condition and the first-token readout show this); or the goal line may matter less here because both goals' recipes are on screen. A negative ΔL would mean naming the goal makes the model more careful about it. The ALFWorld mechanism diagnostic found a task-relevance bias at answer time; that account predicts a positive effect here too.

## 2. Environment

- **Code.** AgentGym's TextCraft (`agentenv-textcraft`) at commit `3ef9235d23e68e7c2920c5422ad957dc8ced5c6c`. Its three source files are vendored in `wf/textcraft/upstream/`. Two of them carry patches, each listed at the top of its file: unused imports removed; no gymnasium base class (its reset only seeds a generator the class never uses); and the recipe loader reads the frozen file instead of listing a folder. Upstream sha256: `environment.py` 541d2f23…071e570, `crafting_tree.py` 6d688cfc…ec36d60, `utils.py` ee8528d1…14d14e4 (full hashes are in the file headers). A one-off check runs every item through the unmodified upstream package as well (section 7).
- **Frozen recipe database** (`wf/textcraft/data/recipes_frozen.json`, built by `wf/textcraft/freeze.py`; no longer committed, see section 11, A6). Upstream reads its 860 recipe files in the order the filesystem lists them. Two things depend on that order:
  - *Three two-way loops.* The loader keeps whichever direction it reads first: iron ingot and iron nuggets, gold ingot and gold nuggets, honey block and honey bottles.
  - *The bed group mapping.* The coloured beds' group is "bed" or "dyed_bed" depending on order. No recipe takes a bed group as an input, so this has no effect on any reply.
- **Loop rule.** Keep the direction that AgentGym's own released TextCraft data uses (AgentTraj-L training trajectories and the evaluation set), and drop the other file:
  - *Iron:* "craft 9 iron nugget using 1 iron ingot" is kept (15 occurrences, reverse 0). `iron_ingot_from_nuggets.json` is dropped, so iron ingot can be fetched.
  - *Gold:* "craft 1 gold ingot using 9 gold nugget" is kept (72 occurrences, reverse 0). `gold_nugget.json` is dropped, so gold ingot is craftable and cannot be fetched.
  - *Honey:* "craft 4 honey bottle using 1 honey block, 4 glass bottle" is kept (4 occurrences, reverse 0). `honey_block.json` is dropped.

  The remaining 857 files are stored in sorted file-name order, each with its JSON key order unchanged, because key order fixes the order of a recipe line's inputs. With the losers gone, no loop remains. A test checks that building the tree in sorted and in reversed order gives the same database.
- **No reset() randomness.** The environment's own `reset(data_idx)` samples distractor recipes, and the result depends on Python's hash seed. We never call it. Every game is created with `reset(commands=<our list>, goal=<goal>)`, and the recipe list is stored with each item.
- **Licences.**
  - AgentGym is MIT (Copyright 2023 FudanNLP-Agent); its licence is kept as `wf/textcraft/upstream/LICENSE-AgentGym`.
  - TextCraft comes from ADaPT (Prasad et al., 2023), also MIT (Copyright 2023 Archiki Prasad).
  - The recipe JSON files are Minecraft game data (© Mojang/Microsoft). AgentGym and ADaPT redistribute them, but the MIT licence does not cover Mojang's rights.
  - Recipes are game facts, and the use is non-commercial research. Before this repository is made public with the frozen file in it, the owner should decide whether to keep the file or ship a script that rebuilds it from an AgentGym checkout. The script exists (`python -m wf.textcraft.freeze <AgentGym checkout>`).

### How the environment replies (the rules a checker needs)

The environment's only state is the inventory, a map from item to count. It starts empty, and only the agent's own actions change it. The actions are:

1. **`get N X`** (N digits, X a name with spaces).
   - If X is craftable anywhere in the database, a generic group ("tag"), or not a known item, the reply is `Could not find X` (X exactly as typed) and nothing changes.
   - Otherwise the reply is `Got N X` and N of X are added.
2. **`craft N Y using c1 n1, c2 n2, ...`.** The inputs are checked against the inventory first. If any listed input has fewer than its count, the reply is `Could not find enough items to craft minecraft:Y_` and nothing changes. Here `Y_` is Y with spaces replaced by underscores. This check comes before any check that the recipe exists, so the reply does not depend on whether the recipe is right.
   - If every input is held but the inputs don't match a database recipe for Y, the reply is `Could not find a valid recipe for ...`. This case is excluded from the items.
   - Otherwise the listed inputs are removed, the recipe's output count K of Y is added, and the reply is `Crafted K minecraft:Y_`.
3. **`inventory`.** The reply is `Inventory: You are not carrying anything.` when the inventory is empty. Otherwise it is `Inventory: ` followed by `[name] (count) ` for each held item, with a trailing space, in the order the items were first added; an item whose count reached zero is removed and goes to the end if added again. It changes nothing.

Names in replies that begin `minecraft:` use underscores, and the checker turns them back into spaces. A generic ingredient on a recipe line (for example "planks") must be replaced by a specific kind in the action (for example "oak planks"). Every action in our histories and items uses specific kinds only. *(Corrected in section 11, A1: only planks and wool are generic inputs; 14 tag-like names such as "logs" are plain items. A1 also fixes the exact reply texts and the order of a craft's inventory changes.)*

## 3. Items: the union design

### Goals and goal pairs

- **Goal pool.** TextCraft's own goal set: every item with a recipe whose minimum crafting depth is at least 1 (`item_recipes_min_depth(1)`, 544 items on the frozen database). Three goals are removed because the fixed prompt text names them (the instruction's examples): blue dye, golden carrot and dark oak planks.
- **A goal's recipe tree.** The set of recipe lines that `traverse_recipe_tree(goal)` returns: every recipe for the goal, then recursively every recipe for each input, through generic groups. These are the lines the environment's `create_recipe_set` would show, without its sampled distractors.
- **A goal's craftable items.** The outputs of the lines in its tree, the goal included.
- **Partner.** Two goals A and B can be partners when their craftable-item sets are disjoint and they have the same minimum crafting depth. Disjoint sets also mean neither goal is in the other's tree. Base (fetchable) items may be shared.
- **Pairing.** For each depth, ascending:
  1. Take that depth's goals, sorted by item id, and shuffle them with `random.Random(f"{SEED}|textcraft|pairs|{depth}")`, where SEED = 20260924 (`wf.constants.RNG_SEED`).
  2. Walk the shuffled list. Each goal not yet paired takes as partner the first later goal in the list that is not yet paired and is compatible.
  3. Goals with no compatible partner are dropped.

  Each goal belongs to at most one pair, so pairs are independent clusters.
- **Pair id.** `p<index>`, numbered in the order the pairs were formed. Within a pair, the two goals are named in sorted order.
- **Displayed list.** The union of the two trees, deduplicated and sorted, then shuffled with `random.Random(f"{SEED}|textcraft|list|{A}|{B}")`, where A and B are the two goals' names in sorted order. The list therefore depends only on the unordered pair, and it shows two complete, genuine TextCraft tasks with nothing marking which one is real.
- **Prompt-example pairs.** Three pairs are drawn with `random.Random(f"{SEED}|textcraft|fewshot").sample(...)` from the pairs whose list has at most the median number of lines. They are drawn by pair index, ascending, and the three chosen pairs are sorted. They supply the three in-context examples, and none of their goals appears in any test item.

### Histories: goal-blind scripted exploration

A history is a prefix of a seeded walk over the pair's list. Each pair has six walks (seeds 0 to 5) of 12 steps, and every prefix of every walk, from 0 to 12 steps, is a history state. The walk is driven by `random.Random(f"{SEED}|textcraft|walk|{pair_id}|{seed}")`. It sees the list, the inventory and the two goal names only as a set, the "protected" outputs that it never crafts. It never knows which goal is real. Real is only a label attached later, by which tree the test action's object belongs to. The same histories therefore serve both goals of the pair. At each step:

1. Draw u uniform on [0, 1).
2. If u < 0.10, the action is `inventory`.
3. Otherwise, build the list of available crafts. A craft is available when its line's output is not one of the two goals and every input is held in at least the line's count. A generic input is filled by the kind held in the largest number, ties broken alphabetically, and only kinds held in at least the count qualify. The available crafts are written as action strings, deduplicated and sorted.
4. If u < 0.55 and some craft is available, the action is one of them, chosen uniformly.
5. Otherwise, the action is `get k b`. Here b is chosen uniformly from the bases: the sorted, fetchable, specific inputs of the list's lines. k is chosen uniformly from {1, 2, 3, 4}.

Every history action therefore succeeds. The policy never fails an action, never crafts a goal and never names a goal. Each prefix is labelled with its length: 0, 1–3, 4–8 or 9–12 steps.

### Test actions, outcomes and twins

At each history state, for each goal G of the pair (the item's aligned goal), the **targets** are G itself and every craftable specific direct ingredient of G. A direct ingredient is an input of a displayed line whose output is G, with generic inputs filled by the rule below. For each target Y:

- **Craft candidates.** For each displayed line whose output is Y, the line is written as an action. Each generic input is filled by the kind with the largest current holding (more than 0). If none is held, the kind with the largest peak holding (more than 0) is used. Failing that, the alphabetically first kind that is the output of a displayed line, else the alphabetically first kind. Ties are always broken alphabetically.
- **Get candidate.** `get 1 Y`.

Each candidate runs on a copy of the game, and the reply sorts it:

- `Crafted …` or `Got …`: a success, not an item.
- `Could not find a valid recipe …`: a wrong recipe, excluded and counted.
- `Could not find enough items to craft minecraft:Y_`: a craft failure item.
- `Could not find Y` for a get: a fetch-a-craftable item.

**Failure classes** (craft failures; stored names and the meaning of `kind` and peak are fixed in section 11, A1). Let inv be the inventory before the action and peak[n] the largest count of n held at any point in the history, the start included. An input (c, n) is short when inv[n] < c. Each short input is classified as follows:

- **used up** when peak[n] ≥ c: the agent once held enough, and an earlier craft took it below;
- **too few** when 0 < peak[n] < c: obtained, but never as many as needed at once;
- **never obtained** when peak[n] = 0.

The item's class is used up if any short input is used up, else too few if any is too few, else never obtained. The kinds of all short inputs are stored too.

**Populations.**

- *Main:* craft failures with a history of at least 1 step, of any of the three classes.
- *Secondary (fetch a craftable):* get failures at any history length. The rule that craftable items cannot be fetched is never stated to the agent, so this stratum is not part of the main test.
- *Empty history:* craft failures at history length 0. They are all never obtained, and they are reported only as exploratory.

**Success twin (o+).**

- *Craft:* a copy of the game gets exactly the missing amount of each short input (c − inv[n]) and runs the same action. The twin is the environment's real reply, `Crafted K minecraft:Y_`, and the added amounts are stored. An item whose topped-up craft does not succeed is excluded as a wrong recipe.
- *Get:* the environment cannot fetch a craftable item, so the twin is the template `Got 1 Y`, the environment's own get-success format. It is stored with twin source "template".
- *Failure (o−):* the environment's real reply.

**Duplicates.** Two items with the same pair, the same visible history and the same action are duplicates, and only the first, in walk order (seed, then prefix length), is kept. The action's object lies in one tree only, so the aligned goal is fixed by the action.

**Exclusions** (counted; section 11, A1 adds the tag-reading exclusion). The exclusions are:

- wrong-recipe crafts;
- crafts that name a generic group literally, which our candidates never do, so the count should be zero;
- successes;
- duplicates.

## 4. Sample

Selection is by a fixed rule with seeded draws, applied after deduplication.

*(The main population's rule is replaced by section 11, A3: couples balanced on history fit within class, length bin and role; A3 is in turn replaced by section 12, B1: couples of items with the same history, one for each goal of the pair.)*

- **Main population: 1,500 items, 500 per failure class.** If a class has fewer than its quota, all of its items are taken, and the remainder is split equally among the classes that still have spare items, repeated until the quota is filled or nothing is left.
- **Within a class,** the quota is split equally over the length bins 1–3, 4–8 and 9–12, with the same redistribution rule.
- **Within a cell,** items are taken round-robin over goal pairs: the pairs are shuffled once per cell, and each round takes one item, drawn at random, from each pair that still has items. This spreads items over clusters.
- **Secondary stratum: 400 items,** 100 per length bin (0, 1–3, 4–8, 9–12), by the same rule.
- **Empty-history stratum: 100 items,** by the same rule.
- **Seeds.** Each cell uses `random.Random(f"{SEED}|textcraft|select|{population}|{class}|{bin}")`.

The report states the counts per class, length bin and role (action on the goal or on a direct ingredient), and the number of pairs and goals.

## 5. Prompt

The prompt mirrors the ALFWorld world-model format (`wf/scoring/prompts.py`).

**System message.** A world-model framing followed by TextCraft's own instruction text, the one AgentGym gives its agents. Only the sentence fixing the agent's output format ("Your output must strictly follow this format…") is left out, because the world model does not act:

> You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next action, predict the environment's next observation exactly in TextCraft style. Output only the observation text.
>
> The agent was given these instructions:
> (AgentGym's instruction and reminders, verbatim)

**Each turn.** A user message, followed by an assistant message holding the true observation:

```
History:
Crafting commands:
<the list, one line each>

Goal: craft <goal>.
> <action 1>
<reply 1>
...

Action: <action>

Next observation:
```

The history is the transcript as the environment shows it: its first observation (the list and the goal line), then each action and reply.

**Task lines** (the goal line is the only part that changes):

| Condition | Goal line |
|---|---|
| Aligned | the item's real goal, `Goal: craft X.` |
| Swapped | the partner, `Goal: craft P.` |
| Neutral | `Goal: explore the crafting commands.` |

The neutral line mirrors ALFWorld's neutral line ("Your task is to: explore the room.") and keeps the environment's template, with a goal line in the same place. It names no item, so it favours neither tree. Dropping the line instead would change the format, as well as removing the goal. Tests check that the three prompts differ only in that line.

**In-context examples.** Three examples, one from each prompt-example pair. Each is drawn from that pair's walk with seed 0:

1. a successful craft: the walk's first craft action at step 3 or later, with the history before it;
2. a craft failure: an item of the main population from the second pair, chosen at random among its items with 2 to 5 history steps;
3. a successful get: the walk's first get action at step 3 or later. *(Its goal line is set by the rule of section 12, C8, so that one example acts off its goal.)*

Each example's goal line is its own goal. That is the goal whose tree holds the crafted object for example 1, and the item's goal for example 2. For a get, a coin seeded with `random.Random(f"{SEED}|textcraft|fewshot|goal|{k}")` picks one of the pair's goals, where k is the example's position (1 to 3). The examples are stored in `wf/textcraft/data/fewshot.json`.

**Yes/no question.** As in the ALFWorld validation runs (rq1v), the final "Next observation:" is replaced by `Question: Will this action succeed? Answer Yes or No.\nAnswer:`, and the single tokens "Yes" and "No" are scored.

## 6. Readouts and analysis

**Scored conditions.** Per item and model:

- the two-candidate L under aligned, swapped and neutral;
- the first-token term, from the same scores;
- the yes/no L under aligned, swapped and neutral.

The models are Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, through `LogoddsScorer`. Runs `textcraft_score` (sharded by pair) and `textcraft_analysis`.

**Primary test.** ΔL (aligned − swapped) on the main population, per model, as in RQ2:

- the mean over items, with a 95% percentile bootstrap that resamples goal pairs (the cluster, in place of ALFWorld's game; 2,000 resamples);
- a two-sided sign-flip permutation p-value that flips whole goal pairs (1,000 permutations);
- Holm adjustment over the two models.

**Everything else is exploratory:**

- aligned − neutral and swapped − neutral;
- the first-token ΔL and Δp;
- the yes/no effects;
- strata by failure class, history-length bin and role (goal or direct ingredient);
- the secondary and empty-history strata.

## 7. Safeguards (mirroring notes/testing.md)

1. **Determinism.** The item builder runs twice in fresh processes, with `PYTHONHASHSEED` 0 and 1, and must write byte-identical files. The frozen database must build the same tree in sorted and reversed file order.
2. **Full replay of every item** in fresh processes, with hash seeds 0, 1 and 2. A fresh game is built from the item's stored list and its aligned goal. Then:
   - the history replays to exactly the stored replies;
   - the action gives exactly the stored failure reply;
   - topping up the stored amounts gives exactly the stored twin.

   The history must also replay identically with the partner as the game's goal, since the reply never depends on which goal is real.
3. **Text-only determination checker** (`wf/textcraft/checker.py`). It imports no environment code, and a test enforces that. It reads only the item's recipe lines, the history's (action, reply) pairs and the action, and applies the rules in section 2. For each item it must:
   - call the action a sure failure;
   - give the stored failure reply exactly;
   - give the stored class and short inputs;
   - predict the twin's text (`Crafted K minecraft:Y_`, with K the action's count, or `Got 1 Y`).

   It also checks every "Inventory:" reply in the history against its own accounting.
4. **Upstream equivalence (one-off).** Every item replays through the unmodified AgentGym package at 3ef9235 (its own `os.listdir` replaced by the frozen sorted listing). This must give the same replies, and the same database as the frozen file.
5. **Known-answer tests.** Hand-worked games, with replies and classes computed by hand.
6. **No duplicates.** No two items share a pair, a visible history and an action.
7. **Prompts.** The aligned, swapped and neutral prompts differ only in the goal line, and the yes/no prompts only in the final question.
8. **Mutation tests** (scope and results in section 11, A5). At least 15 deliberate bugs in the builder, the checker and the prompts, each of which the TextCraft tests must catch (`python -m wf.textcraft.mutation_check`).
9. **Manifest.** `wf/textcraft/data/manifest.json` records:
   - the sha256 of the items file, the few-shot file and the frozen database;
   - the upstream commit;
   - the counts per stratum;
   - every check's result.

   The scoring run refuses an items file whose hash is not in a manifest marked verified.
10. **Blind second checker.** A second checker will be written later, from this note only, by someone who has not seen our code. It must agree with the stored failure reply and class on every item.

## 8. What is stored per item (`wf/textcraft/data/items.jsonl`; fields added in section 11 are listed there)

| Field | What it holds |
|---|---|
| `id` | the item id |
| `pair_id` | the cluster, also copied to `game_id` |
| `goal` | the aligned goal |
| `partner` | the swapped goal |
| `depth` | the pair's crafting depth |
| `recipe_lines` | the displayed list, in display order |
| `env_goal` | the aligned goal's item id |
| `walk_seed` | which walk the history comes from |
| `history` | (action, reply) pairs |
| `history_len` | the number of history steps |
| `length_bin` | 0, 1–3, 4–8 or 9–12 |
| `action` | the test action |
| `action_kind` | craft or get |
| `object` | the target Y |
| `role` | goal or ingredient |
| `failure_class` | the item's class |
| `short_inputs` | each short input's name, need, held count, peak and kind |
| `population` | main, secondary or empty history |
| `reply` | the failure reply, o− |
| `twin` | the success twin, o+ |
| `twin_source` | env_topup or template |
| `topup` | the amounts added for the twin |
| `task_line` | the aligned goal line |
| `swapped_task_line` | the swapped goal line |
| `neutral_task_line` | the neutral goal line |
| `candidates` | `{o_plus, o_minus}` |

## 9. Departures from the feasibility probe

The design changes the feasibility probe (scratch `envprobe/textcraft/`) in the following ways:

- **Frozen database.** The probe read the folder live.
- **No distractor recipes,** so the list is exactly the two trees.
- **Depth-matched, disjoint, one-to-one pairs,** with histories shared by both goals of a pair.
- **The twin tops up only the missing amounts.** The probe added each input's full count.
- **Classes defined by peak holding.** The probe used "consumed at some time".
- **Never-obtained items need a non-empty history** to be in the main test.
- **Fixed quotas.**

## 10. Clarifications written while building (2026-11-20, before any scoring)

These clarifications were added after the design commit (76626d6; this note first cited it as 2b4b16a, its hash before the branch history was rewritten in section 11, A6; corrected in section 12, B9). None changes a rule that affected which items were built.

- **Order independence (section 2).** Reading the frozen files in reverse gives the same recipes, generic groups and items. Only the order of an item's alternative recipes follows the file order, and the frozen file fixes that order. The environment's craft step takes the first alternative that matches the inputs. That order could matter only if two alternatives took the same inputs with different output counts, and none do (sugar lists "craft 1 sugar using 1 sugar cane" twice, identically). A test checks both points.
- **Splitting a quota (section 4).** When the remainder of a quota is smaller than the number of open classes or bins, one extra item goes to each of the first ones, in the order used in section 4 (classes used up, too few, never obtained; bins in ascending order). This is why the main cells hold 167, 167 and 166 items.
- **Example walks (section 5).** If a prompt-example pair's walk with seed 0 had no craft (or get) at step 3 or later, the next seed would be used. It was not needed: examples 1 and 3 come from walk seed 0. Example 2 is an item from walk seed 3.
- **Counts of the build.** The counts that fixed the sample:
  - 269 depth-matched pairs (538 of the 541 eligible goals), of which three are prompt-example pairs: p049 (gold ingot, light gray dye), p119 (acacia sign, fire charge) and p140 (birch sign, granite). That leaves 266 test pairs.
  - Every quota was filled.
  - The eligible pool and the exclusions (22,297 successes, 14,809 duplicates, and no wrong-recipe or generic-name crafts) are in `wf/textcraft/data/build_stats.json`.

## 11. Amendments after review 1, before any score (2026-11-20)

An adversarial review of the built items (review 1) found the labels trustworthy but the selection confounded, and parts of sections 2 and 3 wrong or underspecified. Every change below was made before any model scored any TextCraft item; nothing has been scored. The items were rebuilt under the amended rules and every safeguard of section 7 was rerun (results at the end of this section). The primary test of section 1 is unchanged: ΔL (aligned minus swapped) on the main population, per model, with the same bootstrap, permutation and Holm rule.

### A1. Generic groups, tag-derived names, and the rules a checker needs (corrects sections 2 and 3)

**Correction.** Sections 2 and 3 implied that recipe lines use many generic groups. In fact only two groups ever appear as an input on a recipe line:

- *planks*, whose members are acacia, birch, crimson, dark oak, jungle, oak, spruce and warped planks;
- *wool*, whose members are the fifteen coloured wools (black, blue, brown, cyan, gray, green, light blue, light gray, lime, magenta, orange, pink, purple, red, yellow). White wool is **not** a member, because its recipe carries no "wool" group, so "white wool" never fills a "wool" input.

Upstream's `clean_up_recipes` turns every JSON tag none of whose members has a recipe into a plain item. Fourteen tags are converted this way: logs, coals, stone crafting materials, stone tool materials, wooden slabs, soul fire base blocks, and the eight per-wood names oak logs, spruce logs, birch logs, jungle logs, acacia logs, dark oak logs, crimson stems and warped stems. Each is an ordinary fetchable item ("get 3 logs" replies "Got 3 logs"), and a craft input "3 logs" is filled only by holding 3 of the item "logs" itself. The database also holds 21 other group names that come from recipes' output "group" field (bed, boat, sign, wooden slab, sticks and so on). A get of any of these 23 group names fails, but no recipe line takes one of them as an input. The "generic-name crafts" exclusion of section 3 therefore concerns planks and wool only, and it stays at zero.

**Tag-reading exclusion (new).** The fixed instruction tells the agent that a generic ingredient such as "planks" may be filled by a specific kind. A reader who knows Minecraft may then take a held "warped stems" to fill a "logs" input, although in TextCraft it does not. For such an item the failure is not determined under a reading the prompt itself invites. We **exclude** these items rather than flag them as a stratum, because the main test needs failures that are determined for every reasonable reader. The rule: a craft item is excluded when one of its short inputs is one of the 14 tag-derived names and the inventory just before the action holds a positive amount of an item of the database that belongs to the same Minecraft tag. The member lists (in `build.TAG_READING`) are:

- each per-wood name: that wood's log, wood, stripped log and stripped wood (for the stems: stem, hyphae, stripped stem, stripped hyphae);
- logs: the eight per-wood names and all their members;
- coals: coal;
- stone crafting materials and stone tool materials: cobblestone and blackstone;
- wooden slabs: the eight wooden slabs (oak, spruce, birch, jungle, acacia, dark oak, crimson, warped);
- soul fire base blocks: none (soul sand and soul soil are not in the database).

Charcoal and cobbled deepslate belong to these Minecraft tags but are not items of the database. The rule excluded 154 candidate items before deduplication, among them the review's three examples (tc-p168-4a03744afe67, tc-p168-614ce24f7fb8 and tc-p133-0af517e8c19e). A test rebuilds the review's first example and checks that it is excluded.

**Rules left open in sections 2 and 3, now fixed** (the blind checker's list of ambiguities):

1. *The count in a success reply.* "Crafted K minecraft:Y_" carries the matched recipe's output count K, whatever count the action states ("craft 3 stick using 2 oak planks" replies "Crafted 4 minecraft:stick"). Every craft in our histories and items copies its line's count, so the action's count and K are the same number, which is what section 7 meant by "the action's count".
2. *Stored class names.* `failure_class` is one of `used_up`, `too_few`, `never_obtained`; get failures are stored as `get_craftable`.
3. *`kind` in `short_inputs`.* It is that input's own class (`used_up`, `too_few` or `never_obtained`), never a member of a group.
4. *Peak.* For a name, exactly as written in the action, the peak is the largest count held after any history step, with 0 at the start. Only gets and craft outputs raise a count. The state just before the action is the state after the last history step, so it is included.
5. *Order of a craft's changes.* The inputs are removed first, and an entry whose count reaches 0 is deleted. Then the output is added. An output already held keeps its place in the inventory listing; a new or re-added item goes to the end.
6. *The wrong-recipe reply.* Its exact text is the Python form of the recipe's output, for example `Could not find a valid recipe for ItemTagWithCount(item_tag=ItemTag(tag=None, item_id='minecraft:stick'), count=4)`. It occurs in no item.
7. *A get's failure.* `Could not find X` is the reply when X is the output of any recipe in the database, one of the 23 group names, or not an item. Every test get names an output of a displayed line.
8. *Matching an action to a recipe.* The environment matches the action's inputs to a recipe's inputs by name, in any order, and each count must be equal; the first matching alternative wins (section 10).
9. *The twin's top-up.* The top-up adds the missing amounts to the inventory directly; it is not a history step and does not change peaks.

### A2. Failures that can be decided without counting

Review 1 found that 660 of the 1,500 main items could be decided from names alone: an input that never appears in the history cannot be held. The class precedence (used up before too few before never obtained) also hid 163 too-few items that had a never-obtained input. Two flags are now stored with every craft item (both are `null` for gets):

- `pure_class` is true when all short inputs have the same class;
- `needs_arithmetic` is true when every short input was obtained at some point in the history (peak above 0). For such an item the names alone do not decide the failure: the reader has to count what was obtained and what was used.

**Named secondary readout (logged now; its population is narrowed in section 12, B6).** ΔL on the main items with `needs_arithmetic` true, per model, with the same pair-clustered bootstrap and pair-level sign-flip permutation as the primary test. It is reported whatever the primary result, and it is not part of the Holm family of the primary test. The primary population still holds all three classes. A failure decided by a missing name is still a determined failure, which is what the primary test is about. The secondary readout asks whether the effect survives when the failure can only be seen by counting.

On the rebuilt main sample, 847 items need arithmetic (used up 488, too few 359) and 653 do not (never obtained 500, too few 141, used up 12). 1,343 items are pure; the 157 mixed items are too few (141) and used up (16).

### A3. The history favours one goal: balanced selection

The walks are goal-blind, but selection made the history track the real goal. Review 1 asked which goal's tree the history fetches or crafts more of, and that guess picked the real goal for 63% of main items. Used-up items were the most extreme (89–100%), because a used-up input was usually consumed by crafting something in the real goal's tree. Never-obtained items on the goal went the other way (33%). If the aligned goal line agrees with what the history visibly works toward, ΔL could reflect that agreement rather than the goal naming the action's object.

**History fit (definition).** For a goal G, its *tree names* are every name on the lines of G's recipe tree (section 3), outputs and inputs alike. A history step's *object* is the item it produced: X for a reply "Got N X", and Y for a reply "Crafted K minecraft:Y_". Inventory steps have no object. Let the aligned count be the number of history steps whose object is in the aligned goal's tree names and not in the partner's, and the swapped count the same with the roles exchanged. The item's `history_fit` is `aligned` when the aligned count is larger, `swapped` when the swapped count is larger, and `tie` otherwise; both counts are stored in `history_fit_counts`. A text-only version computes the trees from the displayed lines: start from G, take every displayed line whose output is the current name (or, for planks and wool, one of the members listed in A1), and continue from each of its inputs. The checker recomputes every item's fit this way, and verify compares it with the builder's, which uses the environment's own tree.

**Selection (replaces section 4's rule for the main population; itself replaced by section 12, B1).** The main population is drawn in couples: one item whose history favours the aligned goal and one whose history favours the partner.

1. 750 couples (1,500 items), split over the three classes, then over length bins within a class, then over the two roles within a class and bin, each split equal with the redistribution rule of section 4. A cell's capacity is the smaller of its numbers of aligned-favouring and swapped-favouring items.
2. Within a class, bin and role cell with k couples, k aligned-favouring and k swapped-favouring items are drawn, each side round-robin over goal pairs with `random.Random(f"{SEED}|textcraft|select|main|{class}|{bin}|{role}|{fit}")`.
3. Items whose history favours neither goal (ties) are not drawn for the main population.

We rebalanced the sample rather than pre-logging a balanced subset, because the eligible pool allowed full balance at the full 1,500 items. The only cost is in the used-up class. Its 1–3 step bin has just two swapped-favouring items, so used-up items now sit almost entirely at 4–12 steps. Used up on the goal has only 7 swapped-favouring items, all from one pair, so that cell holds 14 items from 7 pairs. (Pairs in the table are distinct goal pairs per row.) The secondary (get) stratum was not rebalanced (its fit is 147 aligned, 124 swapped, 129 tie), and the empty-history stratum is all ties by definition.

**Class and role are confounded.** Before this amendment, 492 of the 500 used-up items had an ingredient as the action's object, and 335 of the 500 too-few items had the goal. The equal role split now gives too few and never obtained 252 goal and 248 ingredient items each, but used up remains 486 ingredient against 14 goal. An input is used up because an earlier craft consumed it, and the walk never crafts a goal, so used-up failures on the goal itself are rare. A difference between classes is therefore partly a difference between roles. We report role × class, and class × fit, as strata.

**Rebuilt main sample.** Each cell shows aligned-favouring / swapped-favouring items.

| Class, role | 1–3 steps | 4–8 steps | 9–12 steps | Items | Pairs |
|---|---|---|---|---|---|
| used up, goal | 0 / 0 | 3 / 3 | 4 / 4 | 14 | 7 |
| used up, ingredient | 2 / 2 | 121 / 121 | 120 / 120 | 486 | 168 |
| too few, goal | 42 / 42 | 42 / 42 | 42 / 42 | 252 | 148 |
| too few, ingredient | 42 / 42 | 41 / 41 | 41 / 41 | 248 | 109 |
| never obtained, goal | 42 / 42 | 42 / 42 | 42 / 42 | 252 | 165 |
| never obtained, ingredient | 42 / 42 | 41 / 41 | 41 / 41 | 248 | 155 |

The main population spans 257 goal pairs (514 goals); before this amendment it spanned 266. Review 1's heuristic now picks the real goal for exactly 50.0% of items in every class × role cell (50.6% overall, counting the craft inputs as the review's population-level version does).

### A4. Word priming: a prefix-only control and overlap strata

The success and failure replies both end in the object's name, and the aligned goal line often shares words with it (for a goal-role item it is the same name). A goal line could raise the name's probability and so move L without any belief about success. Two pre-logged controls separate the two.

**Prefix-only arms.** Two more arms, `prefix_aligned` and `prefix_swapped`, use exactly the prompt of the aligned and swapped conditions. Each scores the two replies with the object name removed from both: "Crafted K" against "Could not find enough items to craft" for crafts, and "Got 1" against "Could not find" for gets. They are stored per item as `prefix_candidates`, which the item hash covers. Writing $L^{\text{pre}}$ for this arm's log-odds,

$$\Delta L_{\text{prefix}} = \frac{1}{n}\sum_{i=1}^{n}\left[L^{\text{pre}}_i(\text{aligned}) - L^{\text{pre}}_i(\text{swapped})\right].$$

Because each full reply is its prefix followed by the name, L splits into the prefix part and a remainder: the log-probability of the name after the success prefix minus that after the failure prefix. The name-token remainder of the goal effect is

$$\Delta L_{\text{name}} = \frac{1}{n}\sum_{i=1}^{n}\left(\left[L_i(\text{aligned}) - L^{\text{pre}}_i(\text{aligned})\right] - \left[L_i(\text{swapped}) - L^{\text{pre}}_i(\text{swapped})\right]\right) = \Delta L - \Delta L_{\text{prefix}}.$$

Both are reported per model on every population with the primary test's statistics, as named controls rather than tests. If the prefix part carries ΔL, the goal line moves the predicted outcome. If the remainder carries it, the effect is word priming of the name. *(The last sentence is wrong and is corrected in section 12, B3: priming of the name largely cancels in the remainder, which can itself carry outcome belief.)* The split is exact only when a prefix's tokens are a leading part of the full reply's tokens. The GPU smoke run checks this for each model's tokenizer (validation standard, step 7) before full scoring. Where it fails, the remainder will be reported as approximate.

**Overlap flags and strata.** `object_goal_overlap` is true when the object's name shares a word (split on spaces) with the aligned goal's name, and `object_partner_overlap` the same for the partner. On the rebuilt main sample, 635 of the 982 ingredient-role items share a word with their aligned goal and 18 with the partner; 46 main items in all share a word with the partner. Before this amendment the counts were 641 of 869, and 50. Strata: role × goal overlap, and partner overlap.

### A5. Tests, mutation testing, and the scope of the earlier claim

The commit that added the mutation check said "31 deliberate bugs … all caught". That covered only our own 31 mutants. Review 1's 18 mutants left 12 alive, so the claim overstated how strong the tests were. The mutants that survived were seed changes, the round-robin shuffle, verify's partner-goal and prompt checks, the few-shot hash gate, the resume hash gate, the item hash, the missing-score count, and three readout wirings. Tests added:

- negative tests for verify: broken histories, replies, twins and top-ups, an environment whose replies depend on the goal, prompts that differ beyond the goal line, and broken structure fields must all be reported;
- the analysis joins scores to items by id and item hash, and stops when a score's item has changed (a stale score), with a test;
- a test that missing primary scores are counted and fail the run;
- readout wiring tests with a different known value on every arm and field: the first-token readout reads `L_first`, the yes/no effect keeps its sign (aligned minus swapped), strata use aligned minus swapped (never neutral), and the prefix and remainder readouts read their own arms;
- golden tests for seeded draws: selection on a fixed fake pool, the smoke subset, and a rebuild of the prompt examples, so that a changed seed or a lost shuffle fails the test suite and not only verify's full rebuild;
- the few-shot hash gate at scoring, and the resume refusal of saved scores whose item hash changed;
- known-answer tests for history fit, the flags, the tag-reading exclusion, the 14 tag-derived names, the generic members, the reply count and the wrong-recipe text.

`python -m wf.textcraft.mutation_check` now runs 75 mutants: our 31, review 1's 18 (two rewritten for the amended code), 6 more seed changes, and 20 for the amendments. Results are below.

### A6. The recipe database is rebuilt, not committed

The recipe JSON is Minecraft game data, which AgentGym's MIT licence does not cover. On the owner's decision, `wf/textcraft/data/recipes_frozen.json` is not committed and is listed in `.gitignore`. The branch's unpushed commits were rewritten so that none contains it. `python -m wf.textcraft.freeze --fetch` (a shallow, sparse git fetch of AgentGym at 3ef9235) or `--from <checkout>` rebuilds it with the rule of section 2: the 860 files in Python's sorted file-name order, minus the three loop losers, each recipe's key order kept, written as indented JSON. The script refuses a checkout at another commit, and it writes nothing unless the result's sha256 is the pinned `5216d3a5…d26d65f`, which the manifest records too. Both routes were run and gave that hash. Code that needs the database stops with these instructions when the file is missing. The vendored upstream Python files stay, with their MIT notice.

### Results of the rerun safeguards (rebuilt items, 2026-11-20)

- **Counts.** 2,000 items: 1,500 main (500 per class), 400 secondary gets (100 per bin), 100 empty-history items. Build exclusions: 22,297 successes, 14,807 duplicates, 154 tag-reading items, no wrong-recipe or generic-name crafts. 640 item ids carry over from the first build, 140 of them main.
- **Replay and the text-only checker** in three fresh processes (hash seeds 0, 1 and 2): no replay, checker, structure or prompt failures, no duplicates, no prompt-example goal in any item, and no main cell unbalanced on fit.
- **Determinism.** Two fresh builds (hash seeds 0 and 1) wrote the items and prompt-example files byte for byte. The prompt-example file did not change from the first build.
- **Upstream equivalence.** The unmodified AgentGym package at 3ef9235, given the frozen listing, builds the same database and reproduces every item's history, reply and twin (0 mismatches of 2,000).
- **Prompt diff.** On every item, the aligned, swapped and neutral prompts differ only in the goal line, each query's goal line is the item's own field, the yes/no prompts differ only in the question, and the prefix arms share their condition's prompt.
- **Blind checker.** The blind checker written from the pre-amendment note, rerun unchanged on the rebuilt items, agrees on every reply, class, short input, top-up and twin (2,000 of 2,000). A blind checker written from this amended note, covering history fit and the flags, is left for review 2.
- **Manifest.** `wf/textcraft/data/manifest.json` is marked verified for items sha256 `8f7de347…50ee6bc` and prompt examples `29e9f14d…1b34804f`.
- **Mutation check.** The first run of the 75 mutants killed 73. Two verify-side mutants survived. One switched off the balance condition in verify's gate, which no test reached because the gate sat inside the full verification run. The other switched off the goal-line check, which the negative prompt test failed to isolate because other checks also caught its broken prompt. The gate was moved into a tested function (`verify.replay_passes`), the negative test was narrowed so that only the goal-line check can catch it, and verify was rerun (verified, same hashes). The second run killed all 75. The scope of this claim: these are the 75 listed mutants, against `tests/test_textcraft_*.py`. It is not a claim that every possible bug would be caught.

## 12. Amendments after review 2, before any score (2026-11-20)

Two reviews read the rebuilt items of section 11: an adversarial review of the labels and controls (review 2), and a scientific-validity review of what the design can show. This section answers both. Nothing has been scored: no model has scored any TextCraft item. The items were rebuilt under the rules below and every safeguard of section 7 was rerun (results at the end). Part B answers review 2's must-fix list; part C answers the validity review. The primary test of section 1 keeps its statistic, its interval, its permutation, its Holm rule and its confirmation rule; what changes is how its population is drawn (B1) and how a null is read (C4).

### B1. The history still pointed to the real goal: same-history couples

**What review 2 found.** Section 11, A3 balanced one measure of how far the history works toward each goal: the number of steps whose object lies only in that goal's tree, and only its sign. Other measures computed from the same text still picked the real goal more often than chance. The number of distinct names produced in each goal's tree picked it for 0.685 of the used-up ingredient items. How close each goal was to completion picked it for 0.70 of the used-up ingredient items, and went the other way in the too-few ingredient (0.28) and never-obtained goal (0.36) cells. The first step in either tree picked it for about 0.63 of the too-few items. Within the balanced measure the margins differed (5.1 steps on the aligned-favouring side against 2.4 on the other), and the share of ingredient items whose name shares a word with the goal differed between the two sides (0.44 against 0.81). The coupling is structural: a used-up or too-few failure needs the history to have fetched or crafted the action's input, and that input lies in the real goal's tree.

**The ten fit measures (definitions, all computed from the prompt text by `wf/textcraft/fit.py`, which imports no environment code).** For an item with aligned goal G and partner P, the exclusive names of G are the names on the lines of G's tree (section 11, A3, text-only version) that are not on the lines of P's tree, and the same for P. A step's object is the item it produced (X for "Got N X", Y for "Crafted K minecraft:Y_") and its count is N or K. Each measure gives two numbers, one for each goal, and its sign is +1 when G's number is larger, −1 when P's is larger and 0 otherwise. Exchanging G and P exchanges the two numbers, so every measure changes sign with the goals.

1. *steps*: the number of history steps whose object is an exclusive name of the goal (this is section 11's `history_fit`).
2. *quantity*: the sum of the counts of those steps.
3. *crafts*: the number of those steps that are crafts.
4. *distinct names*: the number of different exclusive names of the goal among the step objects.
5. *first step*: the first step whose object is an exclusive name of either goal counts 1 for that goal; 0 for both when there is none.
6. *last step*: the same for the last such step.
7. *held now*: the total count held after the history (the checker's accounting) of the goal's exclusive names.
8. *completion*: over the displayed lines whose output is the goal, the largest mean over the line's inputs of the smaller of the amount held and the amount needed, divided by the amount needed; a planks or wool input counts every member held.
9. *goal words*: the number of words in the history (every action and reply, square brackets removed, "minecraft:x_y" read as "x y", split on spaces) that belong to the goal's name and not to the other goal's name.
10. *last inventory*: in the last "Inventory:" reply that lists anything, the number of listed names that are exclusive names of the goal.

On the rebuilt items, measures 1 to 8 and 10 agree in sign with review 2's own implementations (heur.py, heur2.py) on all 3,200 items; measure 9 agrees on 3,117, the rest because the reviewer's split keeps "minecraft:oak" and "[oak" as single words.

**Selection (replaces section 11, A3): same-history couples.** The main population is drawn in couples of two items that come from the same goal pair, show the same history, and have the same failure class and role, one acting in each goal's tree. Within a pair, both goals are tested at every state of the same goal-blind walks (section 3), so such couples exist wherever both trees offer a failure of one class and role at one state. A couple slot is one history of a pair at which both goals have such an item; the 750 couples are split over classes, then length bins, then roles, as in section 4; within a class, bin and role cell the slots are drawn round-robin over goal pairs with `random.Random(f"{SEED}|textcraft|select|main|{class}|{bin}|{role}|couples")`, a random slot per pair and round, and a random item of each goal in the slot. Each slot is used once. Ties in history fit are now drawn too, because the couple balances them.

**Why this removes the leak, and the assumption it needs.** Write the score of an item as

$$L_i(g) = r_i(g) + c(h_i, g) + e_i,$$

where g is the goal line, $r_i(g)$ is how the goal line bears on the action itself (the effect the study is about), $c(h, g)$ is any effect of how well the goal line agrees with the history h, and $e_i$ is everything else. The goal effect of an item whose aligned goal is G is $L_i(G) - L_i(P)$. In a couple, item 1 has aligned goal G and item 2 has aligned goal P, at the same history h, so the agreement terms contribute $c(h, G) - c(h, P)$ and $c(h, P) - c(h, G)$, which sum to zero. The mean ΔL over couples therefore contains no history–goal agreement at all, whatever measure of agreement one has in mind, listed or not, sign or magnitude. Every function of the history and the two goals that changes sign with the goals averages to exactly zero over the main sample, in every class, bin and role cell: all ten measures above, review 2's heuristics including the ones that do not use the history (the larger tree, the alphabetically first goal, the shorter name, whose top line is listed first), and the margins. On the rebuilt items each of review 2's heuristics picks the real goal for exactly 0.500 of main items, in every class × role cell. The assumption is additivity: the history pulls the two actions of a couple by the same amount, that is, c does not depend on which action follows. If agreement mattered more for one couple member's action than for the other's, a remainder would be left; the agreement contrast below measures how large c is, which bounds that remainder's importance.

**Why 24 walks.** Same-history couples are rarer than items. With the 6 walks of section 3, the used-up class had 141 couple slots from about 50 pairs, short of its 250 couples. With 24 walks per pair (seeds 0 to 23; every rule of the walk unchanged) it has 623, and every class fills its quota. Couple slots per class (all bins and roles): never obtained 71,012, too few 2,505, used up 623 (used up on the goal: 3, all in one pair). The prompt examples are still drawn from walks 0 to 5 only, so adding walks did not change them (the example file changed only for C8).

**Alternatives considered.** Exact mirroring of the ten signs across items with different histories (any pair, same cell) gave more capacity with 6 walks (used up 230 couples against 144, too few 1,465 against 684) but balances only the listed signs, not their magnitudes or any unlisted measure; we use it only where same-history couples are too concentrated (C2). Keeping only histories that favour neither goal on any measure leaves items only at 1 to 3 steps and only for never obtained (36 such items in the rebuilt main sample). Same-history couples balance every measure, magnitudes included, at no cost in quotas once the walks are extended.

**The cost.** The main population now spans 229 goal pairs (257 before). Too few on an ingredient has fewer pairs per cell (13 at 1–3 steps), and used up on the goal has 3 couples from one pair (6 items), see B2.

**The history–goal agreement contrast (a named readout, C7; descriptive from section 13, D2, which also corrects the additivity sentence above).** Half the difference between ΔL on the main items whose history favours their aligned goal (`history_fit` aligned) and ΔL on those whose history favours the partner. Within a couple these are the two members, so under the model above the half-difference estimates c, the amount history–goal agreement adds to an item's ΔL. This links TextCraft to ALFWorld, where agent histories pursued the aligned task by construction: there the measured effect is about ΔL plus c, here ΔL. Interval by pair bootstrap; no p-value.

**Sensitivity analyses (kept, now descriptive).** Review 2 asked for two sensitivity analyses. With couples both are close to redundant, and the validity review asked that they be kept as secondary or dropped with a reason; we keep them, reported with intervals and without p-values:

- *Regression.* ΔL of each main item regressed on the signs of the ten measures (least squares, intercept and ten slopes; interval of the intercept by pair bootstrap). Because every sign averages to zero over couples, the intercept equals ΔL exactly; the slopes describe how ΔL varies with fit, with the caveat that fit side is correlated with word overlap (B1 above: 0.57 of aligned-favouring and 0.78 of swapped-favouring ingredient items share a word with their goal).
- *Items whose history speaks for the aligned goal on no measure* (every sign 0 or −1). This is a selected subset: it keeps one member of a couple whose signs disagree in one direction, and both members only when every sign is 0. Under additive agreement its ΔL is at or below the full ΔL. Sizes in the rebuilt main sample, class × role: never obtained × goal 103 items (84 pairs), never obtained × ingredient 101 (77), too few × goal 21 (20), too few × ingredient 29 (17), used up × ingredient 35 (23), used up × goal 0; 289 in all, from 154 pairs.
- Exploratory regressions of aligned − neutral and swapped − neutral on the same signs.

**Counterfactual-goal arm: not added.** In ALFWorld a counterfactual task line was used because agent histories agreed with the real task by construction, and the line broke that agreement while keeping the action's relation to the task. Couples break the agreement here without a new arm: across a couple the history agrees with the aligned line exactly as often, and as much, as with the swapped line. A TextCraft counterfactual goal would also be off-distribution in a way ALFWorld's was not. The environment always displays the goal's own recipe tree, so a goal outside both displayed trees (or a different top item whose tree contains the action's object) names something whose recipe is not on the screen, and showing its recipe would change the list as well as the goal line.

### B2. Used up on the goal, and class contrasts

Used up on the goal role holds 3 couples (6 items) from one pair (p125: chain and grindstone). An input is used up because an earlier craft consumed it, and the walk never crafts a goal, so used-up failures on the goal itself are rare in both trees at once. This cell is not interpreted on its own and is left out of every named readout and class contrast; it stays in the primary population (0.4% of it), and the primary is also reported without it. Class contrasts are computed within one role only: used up, too few and never obtained against each other within the ingredient role, and too few against never obtained within the goal role. Any class contrast pooled over roles is also a role contrast, because the classes hold different mixes of roles. All class contrasts are descriptive (C7).

### B3. The prefix control: what the remainder means (corrects section 11, A4)

Section 11, A4 read a remainder that carries ΔL as word priming. That is wrong. Both full replies end in the object's name, so priming that raises the name's probability equally after both prefixes cancels inside L, and it cancels again inside the remainder, which is the log-probability of the name after the success prefix minus that after the failure prefix. What the remainder can carry is outcome belief: after "Crafted 1" a model that expects success finds the name more probable than after "Could not find enough items to craft". So the prefix part says how much of ΔL is decided before the name, and the remainder is the rest of the same judgement, not a priming effect. The real lexical threat is on the prompt side: word overlap between the goal line and the action string, present in 77% of section 11's main items (every goal-role item, and most ingredient items). The prefix arms cannot detect it, because the action is in both arms' prompts; only the overlap split addresses it (the goal-role, overlap and usefulness-only readouts of C2). The prefix decomposition is now descriptive. ΔL, the prefix ΔL and the remainder are computed on one item set, the items with all four scores (aligned, swapped, prefix aligned, prefix swapped), so that ΔL equals the prefix part plus the remainder exactly; and the analysis counts missing scores on every arm of every item, and withholds the verdict while any is missing.

### B4. A per-item certificate (validation standard, step 4)

The text-only checker now emits, for each item, the observation that settles it, and the item stores it as `certificate`. For a craft failure, one entry per short input: its name; `held_step`, the history step (counted from 1) whose reply set the count held now, or "never" when no step changed it (then nothing is held); and `peak_step`, the first step at which its peak was reached, or "never" when it was never obtained. With the need, these give the rule: the action fails because the count set at `held_step` is below the need; the class is used up when the peak reached at `peak_step` met the need, too few when it did not, never obtained when there is no peak step. For a success (C3) the entries are the same for every input. For a get failure, the entry is the displayed line (counted from 1) whose output is the object, which makes it craftable and so not fetchable. The builder computes the same steps from the environment's inventory after each step, and verify compares the two on every item.

### B5. Singular and plural twins

The database holds 11 pairs of items whose names differ by a final "s": acacia log/logs, birch log/logs, brick/bricks, coal/coals, crimson stem/stems, dark oak log/logs, jungle log/logs, nether brick/bricks, oak log/logs, spruce log/logs, warped stem/stems (a test checks that these are all). A reader may take a held "nether bricks" for "nether brick". We exclude, as for tag reading (section 11, A1): a craft failure is excluded when one of its short inputs has its twin held just before the action. The rule removed 1,130 candidates before deduplication.

### B6. The inventory shown after its last change

*(Section 13, D3: the counting readout takes whole couples, 766 items, and its competence gate is computed on them.)*

When a history ends with an inventory look-up after its last get or craft, the prompt shows the inventory before the action verbatim, so the failure can be read off without counting. Every item now stores `inventory_shown_after_last_change` (true when an "Inventory:" reply comes after the last reply that got or crafted something). The counting readout of section 11, A2 is narrowed to the main items that need arithmetic and do not show the inventory after its last change (835 items; of the 886 main items that need arithmetic, 51 show the inventory after its last change; without used up on the goal, the named readout has 833); the readout as logged in A2 is kept as descriptive.

### B7. Tokenization and golden chat prompts (validation standard, step 7)

*Review 2's result, recorded.* For all 1,270 distinct (prefix, full reply) pairs in the items, the prefix's tokens are a leading part of the full reply's tokens in both Qwen3-8B and Llama-3.1-8B-Instruct, so the prefix and remainder split of A4 is exact token-wise.

*On the rebuilt items, every arm.* `wf/textcraft/tokcheck.py` checks, for each item and arm, with the model's own tokenizer: the chat template's ids are the plain tokenization of its rendered text; each candidate tokenizes the same after the prompt as alone (the scorer joins separately tokenized ids); each prefix is a token prefix of its full reply, alone and after the prompt; "Yes" and "No" are single tokens. Run on ICE (tokenizers at the pinned snapshots, no GPU) over all 3,200 items and all eight arms (`wf/textcraft/data/tokcheck_ice.json`): no problem on any item for either model (0 of 3,200 each). Joined equals alone on every candidate, every prefix is a token prefix of its full reply alone and after the prompt, and Yes and No are single tokens in both. The golden prompts below were made in the same run and re-checked against both tokenizers (no difference).

*Golden rendered prompts.* `wf/textcraft/data/chat_golden.json` holds, for the first main item, the aligned and yes/no prompts rendered by each model's real chat template (Qwen3-8B snapshot b968826d, thinking off; Llama-3.1-8B-Instruct snapshot 0e9e39f2), with the number and sha256 of the ids. A test checks without a tokenizer that the stored text holds every message of today's prompt in order and ends in the model's generation prompt. The smoke run (`textcraft_score_smoke`, `token_check: true`) runs the tokenization check on its items with the scorer's own tokenizer and encoder, and re-renders the golden item, before scoring each model; any failure stops the run before a score is written.

### B8. Tests and mutation testing

Review 2's mutation run left three of its 20 mutants alive: N3 (stone crafting materials lose blackstone), N4 (verify's tag-reading check switched off) and N10 (the stale-score check ignores unknown item ids). Tests now kill each: the member lists of the 14 tag-derived names are pinned; verify must report an item whose short input is declared to have a held member (and, for B5, a held twin); and a score whose item id is unknown is stale. All 20 of review 2's mutants (N7 rewritten for the couple draw) and mutants for every amendment of this section were added to `wf/textcraft/mutation_check.py`, which now runs workers in parallel. Results are at the end of this section.

### B9. A stale commit reference

Section 10 cited the design commit as 2b4b16a, its hash before the branch's history was rewritten (section 11, A6). The design commit is 76626d6; the reference is corrected in place. The old object is unreachable and goes at the repository's next garbage collection, which was not run here because the object store is shared with the main checkout.

### C1. Same-history couples are the primary population

The validity review endorsed the couples of B1 as the primary population. The primary test is ΔL on the 1,500 main items (750 couples), as before.

### C2. Role-specific readouts, and usefulness-only couples oversampled

*(Section 13, D1: the extra usefulness population confounds goal identity; the named readout uses the main couples only.)*

The goal role and the ingredient role test different things. A goal-role action crafts the goal itself, so the action names the goal verbatim and completes the task; in ALFWorld the matching actions were intermediate steps on the task's object. An ingredient-role action is instrumentally relevant, a subgoal; when its name shares a word with the goal, naming can still act. Three named readouts split this (C7), each on whole couples so that the couple's balance holds within the readout:

- *goal role*: the direct replication of the ALFWorld effect;
- *ingredient with word overlap*: couples whose two items are both ingredients sharing a word with their own aligned goal;
- *usefulness only*: couples whose two items are both ingredients sharing no word with either goal, so the goal line can matter only through what the object is for. ALFWorld's 2 × 2 analysis puts this part at 0.72 nats for Qwen and −0.18 for Llama, small next to the mention part.

In the main sample, 239 ingredient couples have overlap on both members (478 items), 51 are usefulness-only on both (102 items), and 205 are mixed; mixed couples are in neither readout and are reported descriptively. The main sample holds only 51 usefulness-only couples, too few for that readout to detect an effect of ALFWorld's size for Qwen. A new population, `usefulness`, adds 300 couples (600 items) of never-obtained ingredient items whose name shares no word with either goal, at histories that no main couple uses, 100 couples per length bin. It is not part of the primary population. Same-history couples of this kind exist in only 23 goal pairs (both goals of a pair must have such an ingredient), which would leave the readout with about 30 items per pair and a minimum detectable effect near 2.4 nats for Qwen. The extra couples are therefore *sign-mirrored*: two items in the same length bin, possibly from different pairs, whose ten fit-measure signs (B1) are exact negatives of each other (two items with all signs zero also form a couple). Per bin, `random.Random(f"{SEED}|textcraft|select|usefulness|{bin}|mirrored")` walks the goal pairs round-robin (shuffled once), takes a random unused item of the pair and matches it with a random unused item of the opposite profile, dropping the item when none is left. Every listed measure's sign then sums to zero within the extra population, but magnitudes and unlisted measures are balanced only in expectation, a weaker guarantee than same-history couples; the readout states it. With the validity review's variance figures (item SD of ΔL 5.98 for Qwen and 1.93 for Llama, intraclass correlation 0.26 and 0.12), the readout's 702 items in 137 pairs give a minimum detectable effect, at a two-sided 0.005 level and 80% power, of about 1.2 nats for Qwen and 0.32 for Llama. The validity review asked for about 1 nat; usefulness-only items exist in only 139 goal pairs, so more items would add to the pairs already used and lower the figure little (1,000 items give about 1.1). The same calculation for the other readouts (items and pairs; Qwen and Llama, in nats):

| Readout | Items | Pairs | Qwen | Llama |
|---|---|---|---|---|
| primary (main) | 1,500 | 229 | 0.88 | 0.23 |
| goal role | 504 | 159 | 1.22 | 0.35 |
| ingredient with overlap | 478 | 79 | 1.52 | 0.41 |
| usefulness only | 702 | 137 | 1.19 | 0.32 |
| counting | 833 | 167 | 1.08 | 0.30 |

These assume ALFWorld's item spread and clustering; they are planning figures, not results.

### C3. Matched determined successes

Without scored successes, a null or the counting readout cannot be read: a model that fails at the accounting has no room to shift. A new population, `success`, holds 300 couples (600 items) of determined successes, drawn by the couple rule at the same kind of states (same pair and history, one craft in each goal's tree, same role), split over the length bins. A craft is a determined success when the text shows every input held in at least the line's count and the action is a displayed line with its generic inputs filled by members; the checker decides this from the text alone (`checker.matches_line`), and the environment's reply is "Crafted K minecraft:Y_". Its failure twin comes from the environment too: the same action after each input's holding is taken out (stored as `removal`), which gives "Could not find enough items to craft minecraft:Y_". For these items o+ is the reply and o− the twin. They are scored under all arms. They give:

- a competence check per class: the AUC of L, under the neutral line (and under the aligned line), between the successes and the main and usefulness failures, overall and per failure class, with pair-bootstrap intervals. A model *discriminates* a class when its AUC is at least 0.6 and the interval's lower end is above 0.5. A null on the primary test, and the counting readout, are interpreted only for a model that discriminates the classes involved;
- the AUC cost of the goal line (aligned against neutral);
- a standardizer: ΔL divided by the mean L gap between successes and failures under the neutral line;
- the goal effect on actions that work (ΔL on successes).

All are descriptive except the competence rule, which gates interpretation.

### C4. Reading a null: equivalence margins

The ±0.25-nat criterion of section 1 is not feasible for Qwen, whose 95% interval half-width is expected near 0.5 nats. It is replaced by equivalence margins of about one sixth of the ALFWorld effect: 1.0 nats for Qwen and 0.3 for Llama. The effect is equivalent to zero when the 90% pair-bootstrap interval of ΔL lies inside ± the margin (two one-sided tests at the 5% level). A ΔL that is neither confirmed nor equivalent is inconclusive.

### C5. The hidden-state account: the inventory-shown contrast

*(Section 13, D4: stratified by class, role and bin, readable inventories dropped, descriptive.)*

TextCraft has no hidden state only if the reader accepts that the inventory starts empty, and the prompt never says so; AgentGym's instruction even says items come "from the inventory or the environment". The inventory is displayed somewhere in the history of about half of the main items. A named readout contrasts ΔL on items whose history shows an "Inventory:" reply with ΔL on items whose history shows none, within each length bin, averaged with the bins' item counts as weights. The walk looks at the inventory by a coin flip (probability 0.10 a step), independent of the goals and the state, so within a bin the two groups differ by chance only. If the goal line acts as evidence about hidden state (the ALFWorld "task as evidence" account), ΔL is smaller where the inventory is shown. We do not add a rules arm stating the empty start: it would add two arms to every item and change the prompt, while the contrast is close to a randomized comparison at no cost.

### C6. Word overlap is the lexical threat; yes/no is the cross-benchmark metric

As B3 says, priming of the name cancels inside L; the lexical threat is overlap between the goal line and the action, and only the overlap split (C2) addresses it. The two-candidate L depends on each benchmark's reply texts, so it is not comparable across benchmarks. The yes/no ΔL (aligned minus swapped on "Yes" against "No") uses the same question in every benchmark, and is named as the cross-benchmark metric (C7).

### C7. A hierarchy of readouts

*(Replaced by section 13, D6: five named readouts.)*

1. **Primary** (unchanged statistic and rule): ΔL on the main population, per model, Holm over the two models, confirmed when ΔL > 0, the 95% interval excludes zero and the Holm p is below 0.01; equivalence by C4.
2. **Seven named readouts**, per model, on the main population without used up on the goal, each with its 95% pair-bootstrap interval; those with a sign-flip permutation p are Holm-adjusted within the model. Predictions, written now:
   - *goal role*: positive, of the order of ALFWorld's effect;
   - *ingredient with overlap*: positive, smaller than the goal role;
   - *usefulness only* (main and usefulness couples): small; near 0.7 nats for Qwen and near zero for Llama, as in ALFWorld's 2 × 2;
   - *history–goal agreement* (B1): positive if agreement adds to the effect; we predict a small positive value;
   - *inventory shown* (C5): negative under the hidden-state account, near zero under an answer-time relevance bias; we predict near zero;
   - *yes/no*: positive, as in the ALFWorld validation runs;
   - *counting* (B6): positive and smaller than the primary, if the effect survives where only counting settles the failure; interpreted only for a model that discriminates used up and too few (C3).
3. **Descriptive** (estimates and intervals, no p-values): every stratum, the class contrasts within role (B2), the sensitivity analyses (B1), the prefix decomposition (B3), the first-token readout, the neutral contrasts, the successes (C3), the secondary get and empty-history strata, the primary without used up on the goal, and the A2 readout as logged.

### C8. Outcomes and caveats, written before any score

*(The table is replaced by section 13, D7; the caveats stand, with D2's correction of the first.)*

| Result | Reading |
|---|---|
| Primary confirmed for both models, goal role positive | The ALFWorld goal effect replicates in an environment whose failures are inventory arithmetic. |
| Primary confirmed, usefulness only clearly positive | The goal line raises predicted success for actions that serve it, not only for actions that name it. |
| Primary confirmed, usefulness only near zero | The effect rides on the goal's name in the action (goal role, overlap), as ALFWorld's 2 × 2 suggested. |
| Primary equivalent to zero (C4), competence check passed | The effect does not carry over; with no hidden state here, this points to the ALFWorld effect working through the task as evidence about hidden state. |
| Primary null, competence check failed | Uninterpretable: the model does not read the accounting, so there is nothing for the goal to shift. |
| Primary negative | Naming the goal makes the model more careful about actions toward it. |
| Agreement contrast large and positive | Much of an effect measured on agent histories (ALFWorld) can come from the history agreeing with the goal. |
| Inventory-shown contrast clearly negative | Supports the hidden-state account: the goal matters less when the state is shown. |

Caveats:

- The estimand is conservative relative to ALFWorld's: couples remove history–goal agreement, which ALFWorld's histories had by construction (the agreement contrast estimates the difference).
- The goal role in TextCraft is a task-completing action that names the goal verbatim, unlike ALFWorld's intermediate actions on the task's object.
- The ingredient role tests instrumental relevance, an extension of the construct, not a replication of it.
- The empty starting inventory is never stated (C5).
- Couples need additivity (B1).
- Too few on an ingredient comes from few pairs per cell, and used up from 95 or so pairs; class contrasts are descriptive for that reason too.
- Histories come from a goal-blind scripted walk, not from agents.
- The prompt examples: before this amendment all three acted inside their own goal's tree (two successes and an ingredient failure), so no off-goal action was ever shown, unlike ALFWorld's example set 1, which has a goal-irrelevant success. We changed the third example's goal line, by rule, to the pair's goal whose tree does not hold the fetched item: it now shows "get 2 jungle logs" succeed under "Goal: craft granite.". Goal relevance no longer predicts the outcome within the examples (one on-goal success, one on-goal failure, one off-goal success). The rule for the get example now reads: the goal is the pair's goal whose tree names do not include the fetched item when exactly one does not; otherwise the coin of section 5. This changes the prompt of every item, before any score.

### Results of the rerun safeguards (rebuilt items, 2026-11-20)

- **Counts.** 3,200 items: 1,500 main (750 same-history couples; 500 per class), 600 usefulness (300 sign-mirrored couples, 200 per bin, 137 pairs), 600 successes (300 same-history couples, 200 per bin, 300 goal and 300 ingredient, 167 pairs), 400 secondary gets, 100 empty-history items. Main cells, items (pairs): used up × ingredient 2 (1) / 248 (60) / 244 (87) at 1–3 / 4–8 / 9–12 steps; used up × goal 0 / 2 (1) / 4 (1); too few × goal 84 (42) in each bin; too few × ingredient 84 (13) / 82 (34) / 82 (40); never obtained × goal 84 (42) in each bin; never obtained × ingredient 84 (42) / 82 (41) / 82 (41). The main population spans 229 goal pairs. History fit (section 11's measure) is aligned / swapped / tie for 216 / 216 / 68 used-up, 195 / 195 / 110 too-few and 215 / 215 / 70 never-obtained items. Build exclusions: 84,312 duplicates, 577 tag-reading and 1,130 plural-reading candidates, no wrong-recipe or generic-name crafts; every craft success at a non-empty history is now an eligible success item. 254 item ids carry over from section 11's build, 52 of them main.
- **Leak check.** Review 2's heuristics (heur.py, heur2.py), rerun unchanged on the main items: every one picks the real goal for exactly 0.500 of main items, overall and in every class × role cell; none is flagged.
- **Replay and the text-only checker** in three fresh processes (hash seeds 0, 1 and 2): no replay, checker (including certificates and successes), structure or prompt failure, no duplicate, no prompt-example goal in any item, no unbalanced cell, and every main, usefulness and success couple well formed.
- **Determinism.** Two fresh builds, hash seed 0 with 8 worker processes and hash seed 1 with 3, wrote the items and prompt-example files byte for byte.
- **Upstream equivalence.** The unmodified AgentGym package at 3ef9235 builds the same database and reproduces every item's history, reply and twin, the successes' removal twins included (0 mismatches of 3,200).
- **Blind checker.** The blind checker written from the pre-amendment note, rerun unchanged on the new items stripped to id, recipe lines, history and action, agrees with every stored reply, class, short input, top-up and twin, and calls every success item a success with the stored reply (3,200 of 3,200).
- **Tokenization and golden prompts.** As B7: no problem on any of 3,200 items for either model; goldens made and re-checked.
- **Manifest.** `wf/textcraft/data/manifest.json` is marked verified for items sha256 `3b81ac3f…5e421a6` and prompt examples `e6ddab93…adfb1253`.
- **Tests.** `tests/test_textcraft_*.py`: 126 passed. Full suite (`tests/`, at the items commit): 687 passed, 1 skipped; the tests added since are TextCraft tests only.
- **Mutation check.** `python -m wf.textcraft.mutation_check 4` (four workers, each with its own copy) runs 146 mutants: 73 of section 11's 75 (its two mutants of the A3 tie and one-sided rules no longer apply; seven strings rewritten for the amended code), review 2's 20 (N7 rewritten for the couple draw) and 53 for this section (37 for part B, 16 for part C). The first run killed 133. The 13 survivors were: the empty-history population label; the uncoupled selection's seed and shuffle (R1, R2: the synthetic pool's quotas no longer bound); the A2 readout as logged; verify's two mirror checks (same-history and sign-mirrored couples); the regression interval's statistic; the plural exclusion inside the builder; the all-arm missing-score check; the alone prefix check in tokcheck; the usefulness extras in the named readout; the success label check; and the 90% equivalence interval. A test was added for each, and rerun alone all 13 were killed. The second full run, with those tests, killed all 146. The scope of this claim: the 146 listed mutants, against `tests/test_textcraft_*.py`.

## 13. Final amendments after review 3 and the scientific re-review, before any score (2026-11-20)

Two more reviews read section 12's items and analysis: an adversarial engineering review (review 3), which planted synthetic effects in fake scores and ran them through `wf/textcraft/analysis.py`, and a scientific re-review of what each readout can show. This section answers both and is the last design change before scoring. Nothing has been scored. The items and the prompt examples are unchanged (`items.jsonl` and `fewshot.json` are byte for byte those of section 12, so the manifest's verification stands); what changes is which readouts carry a test, how they are computed, how a result is read, what the score's hash covers, and two exploratory scoring runs. The rule followed throughout: take the simplest change that makes each claim true; where a secondary readout cannot be made valid cheaply, demote it to a description and state its limit. The primary test and the goal-role readout are unchanged.

### D1. Usefulness only: same-history couples only (review 3, item 1; re-review S3)

**What review 3 found.** Section 12, C2 added 300 *sign-mirrored* usefulness couples, two items from possibly different goal pairs whose ten fit-measure signs are negatives of each other. Usefulness-only ingredients (sharing no word with either goal) usually exist in one goal's tree only, so within a pair the aligned goal is nearly constant: 117 of the extra population's 137 pairs have items under one goal line only. A couple then does not exchange the two goal lines, and anything about the goal line itself, independent of the action, enters ΔL. History-free guesses pick the real goal far above chance on these items: the goal with the larger tree 0.65, the goal with the shorter name 0.76, the alphabetically first 0.55. In fake scores where L depended only on the goal line's name length (every true effect zero), the readout as logged gave +0.39 nats with a Holm p of 0.005, while the primary gave 0.00. Section 12's sentence that magnitudes and unlisted measures are "balanced only in expectation" was also wrong: they are not balanced even in expectation (aligned minus partner, in the extra population: quantity +0.92, held now +0.95, distinct names +0.16, last inventory +0.08), and goal identity is not balanced at all.

**Options.** (a) Keep only pairs where both goals have usefulness-only items and weight the two directions equally within each pair: 20 such pairs and 172 items in the extra population, a weighting that undoes the couple's sign balance, for little power. (b) Subtract a pair-level goal-direction term estimated from the main couples: in a main couple the two members' ΔL differ by the term and by the difference between the two actions' true effects, which cannot be separated, so the term is not identified. (c) Use only the usefulness-only couples inside the main population, which are same-history couples: both goal lines appear on both members, so any function of the goal line, the history or both cancels exactly, as for every other named readout.

**Decision: (c).** The named usefulness readout is ΔL on the main population's usefulness-only couples: 51 couples, 102 items, 18 pairs. It is valid but weak: with ALFWorld's spread and clustering its minimum detectable effect (two-sided 0.005, 80% power) is about 3.2 nats for Qwen and 0.9 for Llama, against ALFWorld's 0.72 and −0.18. It can detect only an effect several times ALFWorld's; a null on it says nothing. The 600 extra items are reported descriptively (`usefulness_extra_population_goal_identity_confounded`) with the confound stated; they are not interpreted.

### D2. The agreement contrast is descriptive; a coherence-robust estimate (review 3, item 2; re-review S1, S2)

**What was found.** The agreement contrast compares the two members of the 626 opposite-fit couples, and they differ in their actions, not only in which goal the history favours. Within these couples the history-favoured member holds more of its action's inputs in 0.83 of the couples where the two differ (160), names its inputs in the history more often (0.87 of 140), is short by a smaller amount (0.61 of 417), and has the shorter object name (0.63 of 589). In fake scores with no agreement term at all, where ΔL grew with the number of the action's inputs held, the contrast came out at +0.14. The re-review added a second problem. Section 12 wrote the score as

$$L_i(g) = r_i(g) + c(h_i, g) + e_i$$

and concluded that the couple mean "contains no history–goal agreement at all". That holds when agreement is additive and symmetric. If instead the goal line helps only when the history favours it and the action lies in its tree (one-sided coherence),

$$d_i(g) = \delta \cdot [h_i \text{ favours } g] \cdot [a_i \in \text{tree}(g)],$$

then the favoured member's ΔL carries δ, its partner's carries nothing, the couple mean carries δ/2, and the agreement contrast is also δ/2. The contrast therefore cannot tell symmetric agreement (bias zero) from one-sided coherence (bias equal to the contrast).

**Changes.** The sentence in section 12, B1 is replaced by: *the couple mean contains no agreement term when agreement is additive and acts on both members alike; under one-sided coherence it contains half of it.* The agreement contrast stays as a description, reported with the four imbalances above, and is not read as an estimate of c. The outcome-table row that attributed part of ALFWorld's effect to agreement, and the claim that the contrast "bounds" the remainder, are removed.

**The coherence-robust estimate (pre-logged).** ΔL on the main items whose history does not favour their aligned goal (`history_fit` swapped or tie): the 626 partner-favoured members of the opposite-fit couples and both members of the 124 tie couples, 874 items from 229 pairs (minimum detectable effect about 1.0 nats for Qwen and 0.28 for Llama). When agreement helps rather than hurts, it is a lower bound on the effect under each account considered here. Under one-sided coherence no item in it receives δ (a craft failure's object is a craftable of its own goal's tree, and partners share no craftables), so it estimates the effect without coherence. Under symmetric agreement its partner-favoured members carry −c, so it falls below the effect by about 0.72c (626 of its 874 items). The action-level imbalances above push it down too. Review 3's fake scores, rerun through the amended analysis, show how conservative it can be: with symmetric agreement only (every true effect zero) it gave −0.76 while the primary gave 0.00, and with an effect that grows with the inputs held (primary +0.50) it gave −0.08. It is reported next to the primary with its interval and no p-value. Reading rule: a confirmed primary whose coherence-robust 95% interval reaches zero is reported as "not robust to one-sided coherence", meaning coherence could carry it, not that it does.

### D3. The counting readout on whole couples, and its competence gate (review 3, items 3 and 5)

The counting readout (section 12, B6) took single items. 67 of its 833 items have a couple partner that does not need arithmetic, so the couple's balance did not hold inside it. It now takes whole couples whose two members both need arithmetic and do not show the inventory after its last change: 766 items, 383 couples, 149 pairs (used up × ingredient 448, too few × ingredient 168, too few × goal 150); minimum detectable effect about 1.14 nats for Qwen and 0.31 for Llama. The inventory flag never splits a couple, because both members share the history.

Section 12, C3 gated the counting readout on discriminating "used up and too few", computed on all failures of those classes. The gate is now computed on the counting items themselves: the AUC of L under the neutral line between all 600 determined successes and the 766 counting items, discriminating when the AUC is at least 0.6 and the lower end of its 95% pair-bootstrap interval is above 0.5. The analysis writes the counting readout's reading as "withheld" when the gate fails, and the primary reading carries its own gate (D7).

### D4. The inventory-shown contrast: stratified, readable inventories dropped, descriptive (review 3, item 4; re-review S4)

The two groups of section 12, C5 differed a lot in composition: the share of items showing an inventory runs from 5% to 72% across class × role × length-bin cells (5% to 23% within the 1–3-step bin alone), because the walk's coin flip has more chances in longer histories and the item selection depends on the state. The contrast is now computed within class × role × length-bin cells, weighted by each cell's item count, and leaves out the items that show the inventory after its last change (136 main items, 134 outside used up on the goal) (their state is readable directly, so they confound the comparison with readability). That leaves 584 items with an inventory shown and 776 without. One difference cannot be stratified away: an inventory look takes a step, so within a cell the shown items have on average 1.0 fewer productive steps. Because of that, and because the two groups are different couples from different pairs, the contrast is descriptive. Its written prediction (near zero) stands.

### D5. Equivalence on three scales (re-review S5)

Section 12, C4's margins (1.0 nats for Qwen, 0.3 for Llama) are one sixth of ALFWorld's two-candidate effect, whose size depends on ALFWorld's reply texts. Two scales are added, each with a margin of one sixth of ALFWorld's value and the same test (the 90% pair-bootstrap interval inside ± the margin):

- the yes/no ΔL, with margins of 0.47 nats for Qwen and 0.37 for Llama (ALFWorld 2.80 and 2.21);
- ΔL divided by the item standard deviation of ΔL, recomputed in each resample, with margins of 0.17 for Qwen and 0.14 for Llama (ALFWorld 6.12 over 5.98 and 1.65 over 1.93).

A null is read as equivalent to zero only when all three scales say so (D7).

### D6. The Holm family and the hierarchy (re-review S7)

1. **Primary**, unchanged: ΔL on the 1,500 main items, Holm over the two models, the confirmation rule of section 1; read by the outcome table of D7.
2. **Named readouts**: five, Holm-adjusted together within each model (`HOLM_FAMILY`): the goal role (504 items, 159 pairs), ingredients sharing a word with their goal (478 items, 79 pairs), usefulness only (D1: 102 items), the yes/no effect (1,494 items), and counting (D3: 766 items). Each is confirmed by the rule of section 1 applied to its Holm-adjusted p; counting is read only when its gate passes. Written predictions as in section 12, C7, except that usefulness only can now detect only a large effect (D1).
3. **Descriptive** (estimates and intervals, no p-values): the coherence-robust estimate (D2), the agreement contrast (D2), the inventory-shown contrast (D4), the usefulness extra population (D1), the equivalence tests' estimates, and everything section 12, C7 listed as descriptive.

### D7. Outcomes, replacing section 12, C8's table (re-review S6; review 3, item 7)

The analysis writes each model's row (`verdicts.<model>.reading`) by these rules; "competence" is the neutral-line AUC over all failures (section 12, C3).

| Result for a model | Reading |
|---|---|
| Confirmed, competence passes, not equivalent | The ALFWorld goal effect replicates for this model. |
| Confirmed and equivalent on the two-candidate scale | The effect is real but smaller than one sixth of ALFWorld's. |
| Confirmed while competence fails | The goal line moves the prediction of a model that does not tell successes from failures here: a bias, not a shift in a working state judgement. |
| Confirmed, coherence-robust interval reaches zero | Reported as not robust to one-sided coherence (D2). |
| Negative: ΔL < 0, 95% interval below zero, Holm p < 0.01 | Naming the goal makes the model more careful about actions toward it. |
| Equivalent on all three scales, competence passes | The effect does not carry over to TextCraft as built here. |
| Equivalent on some scales only | Inconclusive; the scales are named. |
| Competence fails, not confirmed | Uninterpretable. |
| Anything else | Inconclusive. |

Across models (`verdicts.confirmed_for`): the replication holds when confirmed for both; "confirmed for one model only" is reported as such, with the other model's row, and is not a replication.

Named readouts: goal role confirmed, with the primary, says the effect appears where the action names the goal; ingredient-with-overlap confirmed says it reaches subgoals that share the goal's words; usefulness only confirmed (possible only for a large effect) says it reaches actions that serve the goal without naming it. The "usefulness only near zero" row of section 12 is removed: that readout cannot show an effect near zero (D1). The "agreement contrast large" row is removed (D2), and the inventory-shown row is now a description (D4).

**What a null does and does not show.** Section 12's null row said a null "points to the ALFWorld effect working through the task as evidence about hidden state". That over-reaches. Besides hidden state, TextCraft differs from ALFWorld in three ways that a null cannot separate: couples remove history–goal agreement that ALFWorld's agent histories had; both goals' recipe trees are on screen, so the swapped goal is a displayed second task rather than another object in the room; and the reply formats differ ("Could not find …" against "Nothing happens."). A null is reported with all four.

### D8. The prompt context in the item hash; withheld verdicts (review 3, item 7)

**Hash.** `score.item_hash` covered the item's own prompt inputs and candidates but not the system text, the yes/no question or the prompt examples, so a change to any of those after scoring would have left old scores looking current. The hash now covers them (`item_hash(item, shots, system)`), the scorer hashes with the examples and system text it actually uses, and the analysis checks with the manifest's examples. This was done before any score, smoke run included.

**Withheld verdicts.** With any score missing, `metrics.json` used to hold the verdicts and every estimate, and STATUS listed ΔL and "confirmed". Now `metrics.json` says `"verdicts": "withheld: …"`, holds the estimates only under `incomplete_estimates_not_results`, and STATUS gives only the counts of missing scores.

### D9. The off-goal prompt example biases against the effect (review 3, item 7)

Section 12, C8 put the third example ("get 2 jungle logs" succeeds) under the goal whose tree does not hold jungle logs, so the examples show one off-goal action and it succeeds. If the model learns anything about goal relevance from three examples, it learns that an off-goal action can succeed, which raises L under the swapped line and lowers ΔL. The bias, if any, is toward a null, so a confirmed effect is conservative on this account. The off-goal action is a base get, which always succeeds, so the example says nothing about arithmetic failures. D10's first variant measures the example's influence.

### D10. Two exploratory prompt variants (re-review, should-fix)

Both are scored only under the aligned and swapped lines, on seeded samples of whole main couples (`score.VARIANTS`, `random.Random(f"{SEED}|textcraft|variant|{name}")`), and read only against the verified prompt's scores on the same items: ΔL under each prompt and the per-item difference, clustered by pair, with no p-value (`exploratory_variants` in the analysis). Each run checks its own prompts' tokenization before scoring (`token_check: true`; the golden rendering is still checked with the verified examples).

- *Examples on goal* (`textcraft_score_examples_on_goal`, 200 couples): the third example goes back under its pair's other goal, birch sign, whose tree holds jungle logs (the example set before section 12, C8: every example on its own goal). It measures how much the off-goal example moves ΔL (D9).
- *Empty start* (`textcraft_score_empty_start`, 300 couples): one sentence is added to the system text, "The inventory is empty when the game starts; it holds only what the history's get and craft steps put in it." It is the cheap test of the hidden-state reading that D4 can no longer give: if the goal line acts as evidence about an inventory the prompt never describes, stating the empty start should shrink ΔL.

Scoring them costs 4,000 scores, against 51,200 for the main run. The natural-history arm is not part of this round.

### D11. How the estimand differs from ALFWorld's (re-review, should-fix)

ALFWorld's primary ΔL was measured on agent histories that pursued the aligned task, with a swapped line naming another object in the room, and on items chosen so the history did not touch the swapped object. Its counterfactual-aligned arm changed that number little (6.52 against 6.33 for Qwen), so whether the aligned task was the one the agent really pursued mattered little there. TextCraft's primary ΔL is measured on goal-blind scripted histories in same-history couples, so history–goal agreement is balanced out (fully under symmetric additivity, half under one-sided coherence, D2); the swapped line names the other displayed task, whose whole recipe tree is on screen; and failures are inventory arithmetic rather than object location. The TextCraft estimand is therefore the goal effect with agreement removed and a salient alternative task on screen. If ALFWorld's effect were mostly agreement, TextCraft's would be smaller for that reason alone; the ALFWorld counterfactual arm argues against that, which is why a large difference would still be informative. The goal-role readout is the closest to ALFWorld's construct, with the difference that a goal-role action completes the task and names the goal verbatim (section 12, C8 caveats).

### D12. Tests, mutation testing and safeguards

Tests in `tests/test_textcraft_review3.py` kill review 3's six surviving mutants: the competence interval's lower end (R3-11: an AUC of 0.68 carried by one pair, which resamples without it bring down to 0.5), per-class competence sets (R3-12), Holm adjustment over the listed family (R3-14), the overlap readout on whole couples (R3-15), cell weights in the inventory contrast (R3-17), and the counting readout's couple pairing (R3-21, now a fix, so its mutant is the reverse). Further tests cover the counting gate on its own items, the reading rules row by row, the three equivalence scales, the prompt context in the hash, withheld verdicts, and the two variants (their prompts, samples, hashes, token checks and comparison). The mutation script now gives each worker a fixed share of the mutants (every n-th in list order) and logs, for each mutant, the first failing test that killed it. Results are below.
