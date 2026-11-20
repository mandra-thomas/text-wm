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
- **Frozen recipe database** (`wf/textcraft/data/recipes_frozen.json`, built by `wf/textcraft/freeze.py`). Upstream reads its 860 recipe files in the order the filesystem lists them. Two things depend on that order:
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

Names in replies that begin `minecraft:` use underscores, and the checker turns them back into spaces. A generic ingredient on a recipe line (for example "planks") must be replaced by a specific kind in the action (for example "oak planks"). Every action in our histories and items uses specific kinds only.

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

**Failure classes** (craft failures). Let inv be the inventory before the action and peak[n] the largest count of n held at any point in the history, the start included. An input (c, n) is short when inv[n] < c. Each short input is classified as follows:

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

**Exclusions** (counted). The exclusions are:

- wrong-recipe crafts;
- crafts that name a generic group literally, which our candidates never do, so the count should be zero;
- successes;
- duplicates.

## 4. Sample

Selection is by a fixed rule with seeded draws, applied after deduplication.

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
8. **Mutation tests.** At least 15 deliberate bugs in the builder, the checker and the prompts, each of which the TextCraft tests must catch (`python -m wf.textcraft.mutation_check`).
9. **Manifest.** `wf/textcraft/data/manifest.json` records:
   - the sha256 of the items file, the few-shot file and the frozen database;
   - the upstream commit;
   - the counts per stratum;
   - every check's result.

   The scoring run refuses an items file whose hash is not in a manifest marked verified.
10. **Blind second checker.** A second checker will be written later, from this note only, by someone who has not seen our code. It must agree with the stored failure reply and class on every item.

## 8. What is stored per item (`wf/textcraft/data/items.jsonl`)

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
