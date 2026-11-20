# TextCraft replication: design (written 2026-11-20, before any item was generated)

This note fixes the design of the TextCraft replication of RQ2 before items exist. It is meant to be copied into the decision log. Every rule a checker needs is stated here in words, so that a second checker can be written from this note alone, without reading our code.

## 1. Question and prediction

**Question.** On ALFWorld, a prompted world model predicts success more often for an action whose failure the history already settles when the task line names the action's object (aligned) than when it names another object (swapped). Does the same goal effect appear in a different environment, TextCraft, where failures follow from inventory arithmetic rather than from where objects are?

**Score.** For each item, L is the log-probability of the success reply minus that of the failure reply, both teacher-forced after the prompt (`LogoddsScorer.score_pair`). The goal effect is

$$\Delta L = \frac{1}{n}\sum_{i=1}^{n}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].$$

A world model that reads the history correctly gives a ΔL of zero, because the history alone decides the reply.

**Prediction (logged now).** ΔL > 0 on the main population for both Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct. For each model the prediction counts as confirmed when ΔL > 0, the 95% bootstrap interval clustered by goal pair excludes zero, and the permutation p-value (whole goal pairs flipped), Holm-adjusted over the two models, is below 0.01. The replication holds if it is confirmed for both models. We also report the RQ2 magnitude rule's verdict (support needs ΔL of at least 0.5 nats, strong at 1.0), as a description, not as the test.

**What a null would mean.** A ΔL near zero with a narrow interval (say, within ±0.25 nats) would mean the ALFWorld effect does not carry over to a crafting game whose failures are inventory arithmetic. That would point to something specific to ALFWorld, such as the task line acting as evidence that its object exists somewhere, or to the "Nothing happens." reply format, rather than to a general bias toward the task's object. Two readings would need to be told apart before drawing that conclusion: the models may fail at the accounting itself, so that L is high under every task line and there is no room for a shift (the neutral condition and the first-token readout show this); or the goal line may matter less here because both goals' recipes are on screen. A negative ΔL would mean naming the goal makes the model more careful about it. The ALFWorld mechanism diagnostic found a task-relevance bias at answer time; that account predicts a positive effect here too.

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

*(The main population's rule is replaced by section 11, A3: couples balanced on history fit within class, length bin and role.)*

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
3. a successful get: the walk's first get action at step 3 or later.

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

These clarifications were added after the design commit (2b4b16a). None changes a rule that affected which items were built.

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

**Named secondary readout (logged now).** ΔL on the main items with `needs_arithmetic` true, per model, with the same pair-clustered bootstrap and pair-level sign-flip permutation as the primary test. It is reported whatever the primary result, and it is not part of the Holm family of the primary test. The primary population still holds all three classes. A failure decided by a missing name is still a determined failure, which is what the primary test is about. The secondary readout asks whether the effect survives when the failure can only be seen by counting.

On the rebuilt main sample, 847 items need arithmetic (used up 488, too few 359) and 653 do not (never obtained 500, too few 141, used up 12). 1,343 items are pure; the 157 mixed items are too few (141) and used up (16).

### A3. The history favours one goal: balanced selection

The walks are goal-blind, but selection made the history track the real goal. Review 1 asked which goal's tree the history fetches or crafts more of, and that guess picked the real goal for 63% of main items. Used-up items were the most extreme (89–100%), because a used-up input was usually consumed by crafting something in the real goal's tree. Never-obtained items on the goal went the other way (33%). If the aligned goal line agrees with what the history visibly works toward, ΔL could reflect that agreement rather than the goal naming the action's object.

**History fit (definition).** For a goal G, its *tree names* are every name on the lines of G's recipe tree (section 3), outputs and inputs alike. A history step's *object* is the item it produced: X for a reply "Got N X", and Y for a reply "Crafted K minecraft:Y_". Inventory steps have no object. Let the aligned count be the number of history steps whose object is in the aligned goal's tree names and not in the partner's, and the swapped count the same with the roles exchanged. The item's `history_fit` is `aligned` when the aligned count is larger, `swapped` when the swapped count is larger, and `tie` otherwise; both counts are stored in `history_fit_counts`. A text-only version computes the trees from the displayed lines: start from G, take every displayed line whose output is the current name (or, for planks and wool, one of the members listed in A1), and continue from each of its inputs. The checker recomputes every item's fit this way, and verify compares it with the builder's, which uses the environment's own tree.

**Selection (replaces section 4's rule for the main population).** The main population is drawn in couples: one item whose history favours the aligned goal and one whose history favours the partner.

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

Both are reported per model on every population with the primary test's statistics, as named controls rather than tests. If the prefix part carries ΔL, the goal line moves the predicted outcome. If the remainder carries it, the effect is word priming of the name. The split is exact only when a prefix's tokens are a leading part of the full reply's tokens. The GPU smoke run checks this for each model's tokenizer (validation standard, step 7) before full scoring. Where it fails, the remainder will be reported as approximate.

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
