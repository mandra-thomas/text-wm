# TextCraft walkthrough: the benchmark, the test items, the scoring and real sample decisions

This walkthrough explains, from first principles, how the project uses TextCraft to ask whether a language model acting as a *world model* lets the agent's task leak into its predictions. It describes the game, shows how test items are built from it, works through exactly how one prediction is turned into a number, and then walks through real decisions that the models made, with the prompts they saw and the scores they gave. The last part takes each TextCraft study in turn (round 1 and its follow-ups, round 2, and the final reading of round 3) and shows its headline numbers next to one or two items scored under its own conditions.

Nothing here is a new analysis. Every prompt below was rendered with the project's own prompt builder at the commit the scores were produced at, and every score was read from the stored score files; the item hash that each scoring run stored with every score was recomputed from the rendered inputs and matched in every case (section 7 says how). Headline numbers are quoted from the committed notes and metrics files, which are cited each time. Round 3 is reported from its final analysis of 2026-11-27 ([textcraft_r3_results.md](../textcraft_r3_results.md)), which supersedes the interim reading of 2026-11-25. The item-level scores of the three round-3 models that only the final analysis added (Gemma 4 12B, Qwen3.8-27B and Gemma 4 31B) were read from the same score files that analysis read, and checked the same way.

## Contents

1. The benchmark itself
2. How the project turns TextCraft into test items
3. How a prediction is scored
4. Sample decisions from round 1
5. The studies, one by one
6. Glossary
7. Provenance, licences and how to regenerate this document

---

## 1. The benchmark itself

### 1.1 Origin

TextCraft is a text-only crafting game modelled on Minecraft. It was introduced with ADaPT (Prasad et al., 2023, "ADaPT: As-Needed Decomposition and Planning with Language Models") as a test of whether a language-model agent can plan a multi-step crafting task, and it was later packaged in AgentGym (`WooooDyy/AgentGym`, environment `agentenv-textcraft`), whose version this project uses, pinned at commit `3ef9235`. The project vendors AgentGym's three TextCraft source files in `wf/textcraft/upstream/` and builds every game from a frozen copy of the recipe database (design note [textcraft_design.md](../textcraft_design.md), section 2).

In the original benchmark an agent is given a goal item, for example "craft blue dye", and a list of crafting commands that includes the goal's recipe tree plus some distractor recipes, and it must reach the goal by fetching base materials and crafting intermediates. This project never runs an agent to solve the task. It uses the game only as a source of exactly determined replies, which a model is asked to predict.

### 1.2 What the game consists of

The game's state is a single inventory: a map from item name to count. It starts empty, and only the agent's own actions change it. The design note puts it plainly: "The environment's only state is the inventory, a map from item to count. It starts empty, and only the agent's own actions change it."

The agent sees a **recipe list** ("Crafting commands:"), one recipe per line, such as `craft 4 polished blackstone using 4 blackstone`. A recipe line names an output count and item, then its inputs with their counts. Items that are the output of some recipe are **craftable**; items that are not are **base items**, which can be fetched directly.

There are three kinds of action, and each has a fixed reply.

A **get** action, `get N X`, fetches N of a base item X. Its reply is `Got N X`, and N of X are added to the inventory. If X is craftable (or a generic group name, or not an item at all) the reply is `Could not find X` and nothing changes. The rule that craftable items cannot be fetched is never stated to the agent.

A **craft** action, `craft N Y using c1 n1, c2 n2, ...`, uses the inputs listed. The environment first checks the inventory: if any listed input is held in a smaller amount than the line asks for, the reply is `Could not find enough items to craft minecraft:Y_` (the item's name with spaces turned into underscores) and nothing changes. Otherwise the inputs are removed, the recipe's output count K of Y is added, and the reply is `Crafted K minecraft:Y_`.

An **inventory** action, `inventory`, changes nothing and replies `Inventory: ` followed by `[name] (count) ` for each held item, in the order the items were first added, or `Inventory: You are not carrying anything.` when the inventory is empty.

A transcript of the game is what the agent would see: the recipe list and the goal line ("Goal: craft X."), then each action on a line starting with `> ` followed by the environment's reply.

### 1.3 What makes a craft succeed or fail

A craft succeeds exactly when every input it names is currently held in at least the stated count. Nothing else matters: not the goal, not which recipe tree the item belongs to, not how many steps have passed. Because every get succeeds or fails by a fixed rule and every craft changes the inventory by a fixed amount, a reader can reconstruct the current inventory from the transcript by simple bookkeeping: add what each `Got` reply adds, subtract each craft's inputs and add its output.

The project classes a failing craft by the history of the input that is short (section 2.3): the agent never obtained it, it obtained some but never enough at once, or it once held enough and an earlier craft used it up. The last two kinds can only be detected by counting.

### 1.4 Nothing is hidden

Unlike ALFWorld, where the agent cannot see what is inside a closed cabinet, TextCraft has no hidden state. The full recipe list is on screen, the inventory is a deterministic function of the actions shown, and the goal line has no influence on any reply. The project checks this directly: every item's history is replayed under both goals of its pair and must give byte-identical replies (section 2.7). A model that reads the transcript correctly therefore has everything it needs to predict the reply, and its prediction should not change when only the goal line changes. Any such change is the **goal effect** this project measures.

---

## 2. How the project turns TextCraft into test items

The full rules are in [textcraft_design.md](../textcraft_design.md) (sections 3, 11, 12 and 13); [GUIDE.md](../GUIDE.md) section 2.2 gives a shorter account. This section explains the parts needed to read the samples.

### 2.1 Goal pairs and the displayed recipe list

Every test item belongs to a **goal pair**: two goal items of equal crafting depth whose recipe trees share no craftable item. A goal's **recipe tree** is every recipe line for the goal, and recursively for each of its inputs. The recipe list the model sees is the union of the two goals' trees, deduplicated and shuffled with a seed that depends only on the pair. So the screen shows two complete, genuine TextCraft tasks, with nothing marking which one is the real goal. There are 269 such pairs; three of them supply the prompt examples and are excluded from testing.

### 2.2 Histories from a goal-blind walk

The history before each test action comes from a scripted walk over the pair's recipe list that never knows which goal is real. It fetches base items, crafts things from the list, and occasionally looks at the inventory. The walk "never fails an action, never crafts a goal and never names a goal" ([GUIDE.md](../GUIDE.md) §2.2). Every prefix of a walk, from 0 to 12 steps, is a history state, so the same history serves both goals of the pair.

### 2.3 Test actions, failure classes and success twins

At each history state, and for each goal G of the pair, the builder considers crafting G itself (the **goal role**) and crafting each craftable direct ingredient of G (the **ingredient role**). It runs each candidate action on a copy of the game. An action whose reply is a failure becomes a test item; its reply is the **failure reply**. The builder then tops up exactly the missing amounts of the short inputs and runs the same action again: the environment's success reply is the item's **success twin**. So each failing item comes with two candidate replies, the real failure and the reply it would have given had the inputs been there.

A failing craft is put into one of three **failure classes** by how the short input was held during the history. Writing the peak of an input for the largest count held at any point:

- **never obtained**: the short input never appeared in the inventory;
- **too few**: it was obtained, but never as many as the recipe needs;
- **used up**: it was once held in full, and an earlier craft consumed it.

An item whose short inputs were all obtained at some point **needs arithmetic**: its failure can only be seen by counting.

A failing `get` of a craftable item ("get 1 iron nugget" replies "Could not find iron nugget") forms a separate stratum, because its rule is never stated to the agent.

### 2.4 The populations

The frozen item set ([GUIDE.md](../GUIDE.md) §2.2) has 3,200 items. In plain words:

| Population | Items | What it is |
|---|---|---|
| Main failures | 1,500 | Failing crafts with a history of 1 to 12 steps, 500 per failure class, drawn as 750 same-history couples from 229 goal pairs |
| Determined successes | 600 | Crafts that work, drawn the same way, used to measure whether a model tells success from failure |
| Usefulness items | 600 | Ingredient crafts whose name shares no word with either goal; later demoted to descriptive |
| Failed gets | 400 | Gets of a craftable item, at any history length |
| Empty history | 100 | Failing crafts at step 0, all "never obtained" |

The main failures carry the primary test of round 1, and they are reused in rounds 2 and 3.

### 2.5 Same-history couples

Main items are drawn in **same-history couples**: two items from the same goal pair, with the same history, the same failure class and the same role, one acting in each goal's tree. Sample 1 below is such a couple. The reason is a confound found in review: a history that happens to fetch more of one goal's ingredients makes that goal look like the "real" one, and that agreement between history and goal line, not the naming of the action's object, could drive the effect. Inside a couple the history is identical for both members, and the two goal lines swap roles between them, so anything that depends only on the history, only on the goal line, or on both together cancels in the couple's mean (design note section 12, B1).

### 2.6 The goal lines

Only one line of the prompt ever changes between conditions: the goal line. The three basic lines are:

- **aligned**: the item's own goal, the goal whose tree contains the action's object ("Goal: craft polished blackstone pressure plate.");
- **swapped**: the other goal of the pair ("Goal: craft stone brick wall.");
- **neutral**: "Goal: explore the crafting commands.", which names no item.

Round 2 added lines that separate *why* naming a goal matters:

- the **two-item** line names both goals of the pair, "Goal: craft melon and stripped birch wood.", so that a goal-role craft is named but completes only half the task;
- the **parent** line is used on **parent records**: a goal-role craft of an item G that is also an ingredient of some other item Z. The recipe list is extended with Z's lines, and the goal line names either G ("craft acacia planks", so the craft completes the task) or Z ("craft acacia trapdoor", so the same craft is an intermediate step);
- **padded** histories prepend 1 or 8 empty inventory look-ups ("> inventory" / "Inventory: You are not carrying anything.") without adding any content.

A round-1 follow-up also used an **object-goal** line that makes an ingredient's own name the goal (section 5.2).

### 2.7 Verification

Before any score, every item was replayed through the environment in fresh processes and must reproduce its stored replies, failure reply and twin; the history must replay identically with the partner as the game's goal; a text-only checker that imports no environment code must decide from the recipe lines and history alone that the action fails, with the stored class; and the aligned, swapped and neutral prompts must differ only in the goal line. A blind second checker written from the design note alone agreed on all 3,200 items ([GUIDE.md](../GUIDE.md) §2.2). Scoring refuses an item file whose hash is not recorded as verified.

### 2.8 The lift and penalty decomposition

Round 1's follow-ups suggested that the goal effect has two parts with opposite signs: a **lift**, where an aligned goal makes the model more optimistic about crafting an ingredient it needs, and a **penalty**, where crafting the goal item itself looks less likely to succeed. Round 2 wrote this as an account. For an action that crafts item $X$ under goal line $g$, relative to a line that makes $X$ irrelevant,

$$L(a \mid g) - L(a \mid g_{\text{irrelevant}}) = \lambda_m \,[X \in \text{tree}(g)] + \nu_m \,[g \text{ names } X] + \kappa_m(h)\,[\text{crafting } X \text{ completes } g] + \varepsilon .$$

Here $m$ is the model and $h$ the history; $\lambda_m$ is the **intermediate lift** for any item in the goal's tree, $\nu_m$ is a **naming** term when the goal line names the item, and $\kappa_m(h)$ is a **completion** term when crafting the item finishes the task, which may depend on the history. In ordinary TextCraft the naming and completion terms always act together, because the only item a one-item goal line names is the one whose craft completes it. The new goal lines of section 2.6 pull them apart. The five confirmatory contrasts of round 2 are defined in section 5.3.

---

## 3. How a prediction is scored

### 3.1 The prompt

The prompt is a chat. A system message frames the model as a world model; three worked examples follow as user and assistant turns; the last user turn holds the item. No text is generated: the two candidate replies are scored by teacher forcing (section 3.3).

**System message, round 1.** It is the world-model framing followed by AgentGym's own instruction to its agents, with only the sentence that fixes the agent's output format removed (`wf/textcraft/prompts.py` at `5a3702a`, `SYSTEM`):

```text
You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next action, predict the environment's next observation exactly in TextCraft style. Output only the observation text.

The agent was given these instructions:
You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.

Reminder: 
1. Always specify the quantity when using "get" and "craft" commands. - Example of get: get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot
2. When using "get" command, do not specify whether the item comes from the inventory or the environment.
3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.
```

**System message, rounds 2 and 3 ("clean").** A round-1 follow-up found that the agent-directed sentences ("you have to respond an action…") change the effect, so later rounds use a clean text that states only the environment's rules (`CLEAN_SYSTEM`, commit `4ccf84b`):

```text
You are a text world model for TextCraft, a Minecraft crafting game. Given the interaction history and the next action, predict the environment's next observation exactly in TextCraft style. Output only the observation text.

The environment's rules: crafting commands are of the format "craft [target object] using [input ingredients]". Each action is "get" with a quantity and an object, "inventory" to look up the inventory, or "craft" with a quantity and one of the crafting commands shown. If a crafting command uses a generic ingredient like "planks", a special type of the same ingredient, e.g. "dark oak planks", can be used instead.
```

**The three prompt examples** come from three goal pairs that are never tested (`wf/textcraft/data/fewshot.json`). Each is a full user turn with a 12- or 13-line recipe list and a short history, answered by the true observation. The first is a successful craft under goal "light gray dye" (`craft 1 white dye using 1 lily of the valley`, answer `Crafted 1 minecraft:white_dye`); the second a failing craft under goal "acacia sign" (`craft 4 stick using 2 acacia planks`, answer `Could not find enough items to craft minecraft:stick`); the third a successful get under goal "granite" (`get 2 jungle logs`, answer `Got 2 jungle logs`), which acts off its own goal so that the examples do not teach "actions on the goal succeed".

**The query turn** shows the recipe list, the goal line, the history, the action and the cue "Next observation:". Sample 1 in section 4 shows one in full.

**How the chat is rendered for each model.** Each model's own chat template turns the messages into one string, at a pinned model revision. For Qwen3-8B the template is ChatML, and with thinking switched off it writes an **empty think block** at the start of the assistant turn; the scored string ends

```text
Next observation:<|im_end|>
<|im_start|>assistant
<think>

</think>


```

and the candidate replies are scored right after that. For Llama-3.1-8B-Instruct the string opens with a system header carrying the template's fixed dates ("Cutting Knowledge Date: December 2023", "Today Date: 26 Jul 2024") and ends `Next observation:<|eot_id|><|start_header_id|>assistant<|end_header_id|>` followed by a blank line. Gemma 2 has no system role, so the system text opens the first user turn; Mistral's template, as shipped, places the system text in the last user turn. These rendered strings are pinned per model in `wf/textcraft/data/chat_golden.json` and `r2_chat_golden.json`, and the scoring runs check them before starting. (The project examined in ALFWorld how much of Qwen3-8B's effect in nats depends on this block; see [qwen_effect_audit_forensic.md](../qwen_effect_audit_forensic.md) and its later qualifications. The TextCraft prompts use the block exactly as the template writes it, which is how Qwen3-8B is normally run with thinking off.)

### 3.2 The readouts

Every readout compares two continuations of the same prompt and reports the difference of their log-probabilities, in **nats** (natural-log units). One nat multiplies the odds by about 2.7; ten nats is a factor of about 22,000.

**Two-candidate log-odds.** The primary readout. The success twin $o^{+}$ and the failure reply $o^{-}$ are each teacher-forced after the prompt $x$, and

$$L = \log P(o^{+} \mid x) - \log P(o^{-} \mid x).$$

A positive $L$ means the model finds the exact success reply more probable than the exact failure reply. For a failing item, a correct world model should give a low $L$, whatever the goal line says.

**First token.** Only the first token of each reply counts. In TextCraft that is "Craft" against "Could" for crafts and "Got" against "Could" for gets, so

$$L_{\text{first}} = \log P(t^{+}_1 \mid x) - \log P(t^{-}_1 \mid x).$$

It is the closest thing to a pure success-or-failure decision, made before the reply names any item.

**Prefix control (round 1).** Both replies end in the object's name, and an aligned goal line often shares words with that name, so a goal line could raise the name's probability without any belief about success. The prefix arm scores the replies with the name removed ("Crafted 1" against "Could not find enough items to craft"), so that the difference between the full and the prefix effect isolates the part carried by the name tokens.

**Yes/no.** The "Next observation:" cue is replaced by a question and the single tokens "Yes" and "No" are scored, giving $\log P(\text{Yes}) - \log P(\text{No})$. Round 1 asked "Question: Will this action succeed? Answer Yes or No." A follow-up found that Qwen partly reads "succeed" as "serve the task" (sample 8 shows why), so rounds 2 and 3 ask a question anchored to the environment's own replies: "Question: will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that it failed ("Could not find ...")? Answer Yes if it succeeds and No if it fails."

### 3.3 Teacher forcing

To teacher-force a reply is to feed its tokens to the model one at a time after the prompt and add up the log-probability the model gave each token, given the prompt and the reply's earlier tokens. If the reply has tokens $o_1, \dots, o_n$,

$$\log P(o \mid x) = \sum_{k=1}^{n} \log P(o_k \mid x, o_1, \dots, o_{k-1}).$$

Nothing is sampled, so the score is a deterministic function of the prompt and the model. The tokenization checks of each round confirm that the candidate tokens are the ones the template would produce at that position.

### 3.4 A worked example

Take sample 1a below: the agent holds 2 melon slices, the recipe needs 9, and the action is `craft 1 melon using 9 melon slice`. Qwen3-8B's stored scores under the aligned line ("Goal: craft melon.") are a log-probability of $-17.2521$ for the 7-token success reply `Crafted 1 minecraft:melon` and $-0.0002$ for the 10-token failure reply `Could not find enough items to craft minecraft:melon`. So

$$L_{\text{aligned}} = -17.2521 - (-0.0002) = -17.2518 .$$

Under the swapped line ("Goal: craft stripped birch wood.") the two log-probabilities are $-10.0001$ and $-0.0002$, so $L_{\text{swapped}} = -10.0000$. Nearly all of the difference sits in the first token: $\log P(\text{"Craft"})$ is $-17.25$ under the aligned line and $-10.00$ under the swapped line, while "Could" has probability within $10^{-4}$ of one in both. The two first tokens together hold all of the next-token probability (the stored first-token mass is 1.0000).

The item's goal effect is the aligned minus the swapped score:

$$\Delta L = L_{\text{aligned}} - L_{\text{swapped}} = -17.25 - (-10.00) = -7.25 .$$

Converting to a probability between the two replies with the logistic function $\sigma(L) = 1/(1+e^{-L})$ gives about $3 \times 10^{-8}$ under the aligned line and $4.5 \times 10^{-5}$ under the swapped line. Both are confident and correct predictions of failure; the goal line moved the log-odds by 7.25 nats, but neither condition comes close to predicting success. That distinction, between a shift in log-odds and a change in the predicted outcome, recurs throughout.

### 3.5 From item scores to an effect

The study-level goal effect is the mean over items,

$$\Delta L = \frac{1}{n}\sum_{i=1}^{n}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].$$

Items from one goal pair share a recipe list and walks, so they are not independent. Every interval therefore resamples whole goal pairs (a **pair-clustered bootstrap**), and every permutation test flips the sign of whole goal pairs at once. Round 1 adjusted its two primary p-values by Holm's method; rounds 2 and 3 adjust within fixed families of tests (section 5).

A goal effect is only interesting in a model that can tell success from failure in the first place. **Competence** is measured under the neutral line as an **AUC**: the probability that a randomly chosen real success scores above a randomly chosen real failure (0.5 is chance, 1 is perfect).

---

## 4. Sample decisions from round 1

The samples in this section are main-population items, plus working crafts and two items from the secondary strata, scored by Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct under round 1's prompt (commit `5a3702a`, run `textcraft_score`). They were chosen to span the two roles, the three failure classes, short and long histories, and the range of outcomes: a typical shift, an extreme one, items where the effect is absent or reversed, working crafts, and the secondary strata. Short recipe lists were preferred so that the prompts can be shown in full. Each prompt below is the query turn exactly as rendered; the system text and the three examples precede it as described in section 3.1.

Every table gives, for each model, the two-candidate log-odds under the three goal lines, the aligned minus swapped difference on the two-candidate and first-token readouts, and the yes/no log-odds under the three lines. All values are in nats.

![Two scatter plots, one per model, of each of the 1,500 main failure items' two-candidate log-odds under the swapped goal line (horizontal) against the aligned goal line (vertical). Qwen's points spread widely around the diagonal, from about minus 30 to plus 25 nats, with a visible excess above it; Llama's lie in a narrower band from about minus 11 to plus 8 nats, close to the diagonal. Seven sample items are marked and labelled.](figures/textcraft_r1_items.png)

*Figure 1. Every main failure item of round 1 under the aligned and swapped goal lines, with the samples of this section labelled. Points above the diagonal are items where the aligned line raised predicted success. Look at how far individual items sit from the diagonal compared with the mean effect (+0.94 nats for Qwen, −0.19 for Llama): the mean is a small shift of a very wide cloud, and samples 3 and 4 show how large single items can be in either direction. Produced by `scripts/walkthroughs/textcraft_figures.py`.*

### Sample 1: a same-history couple (goal role, too few)

Two items share this history: the agent fetched 2 stripped birch logs and 2 melon slices. Item 1a (`tc-p001-1e56f799a78e`) crafts the melon goal, which needs 9 melon slices; item 1b (`tc-p001-e0e4f5606dd4`) crafts the stripped-birch-wood goal, which needs 4 stripped birch logs. Both fail for the same reason, too few of a fetched input. The prompt of 1a under its aligned line:

```text
History:
Crafting commands:
craft 3 stripped birch wood using 4 stripped birch log
craft 1 melon using 9 melon slice

Goal: craft melon.
> get 2 stripped birch log
Got 2 stripped birch log
> get 2 melon slice
Got 2 melon slice

Action: craft 1 melon using 9 melon slice

Next observation:
```

Item 1b's prompt is the same, except that its action is `craft 3 stripped birch wood using 4 stripped birch log` and its aligned line is "Goal: craft stripped birch wood.". The aligned line of one member is the swapped line of the other.

| Item | Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|---|
| 1a (melon) | Qwen3-8B | −17.25 | −10.00 | −15.31 | −7.25 | −7.25 | −31.25 | −30.00 | −31.62 |
| 1a (melon) | Llama-3.1-8B | −9.14 | −7.05 | −7.05 | −2.09 | −1.00 | −10.38 | −10.00 | −7.50 |
| 1b (birch wood) | Qwen3-8B | −17.74 | −19.83 | −20.60 | +2.09 | +2.00 | −30.00 | −27.00 | −28.50 |
| 1b (birch wood) | Llama-3.1-8B | −8.74 | −9.39 | −8.68 | +0.65 | −0.25 | −8.37 | −10.13 | −8.38 |

Both models predict failure confidently for both items under every line, so their predicted outcomes are right throughout. The goal line still moves the log-odds, in opposite directions for the two members: naming melon as the goal makes Qwen more sure the melon craft fails (−7.25), while naming stripped birch wood makes it slightly less sure the birch craft fails (+2.09). The couple's mean, −2.58 for Qwen and −0.72 for Llama, is what enters the study mean. The couple design guarantees that whatever is special about the word "melon" or about this history appears with both signs and cancels; what remains is how the line's relation to the action matters.

### Sample 2: a used-up input, where the predicted outcome flips (ingredient role)

Item `tc-p071-7f77863f789c` needs counting. The agent fetched 3 nether brick, then 2 nether wart and 1 more nether brick, and crafted red nether bricks once, which consumed both nether warts. Repeating the same craft fails: the nether wart is used up.

```text
History:
Crafting commands:
craft 1 red nether bricks using 2 nether wart, 2 nether brick
craft 4 red nether brick stairs using 6 red nether bricks
craft 1 nether bricks using 4 nether brick
craft 6 nether brick fence using 4 nether bricks, 2 nether brick

Goal: craft red nether brick stairs.
> get 3 nether brick
Got 3 nether brick
> inventory
Inventory: [nether brick] (3) 
> get 2 nether wart
Got 2 nether wart
> get 1 nether brick
Got 1 nether brick
> craft 1 red nether bricks using 2 nether wart, 2 nether brick
Crafted 1 minecraft:red_nether_bricks

Action: craft 1 red nether bricks using 2 nether wart, 2 nether brick

Next observation:
```

The aligned goal is red nether brick stairs, whose recipe needs the red nether bricks being crafted; the swapped goal is the nether brick fence.

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +0.76 | −0.75 | −0.25 | +1.50 | +1.50 | −12.25 | −12.00 | −10.25 |
| Llama-3.1-8B | −0.16 | −1.39 | −0.76 | +1.23 | +1.25 | −2.00 | −3.38 | −0.50 |

This is the effect at its most consequential. Qwen's two-candidate score is close to zero, meaning it is unsure, and the goal line decides which side it lands on: under the aligned line it favours the success reply ($\sigma(0.76) \approx 0.68$), under the swapped line the failure reply ($\sigma(-0.75) \approx 0.32$). The predicted observation flips. The same model's yes/no answer is a confident and correct "No" under all three lines, so the two readouts of one model disagree on this item: asked directly, Qwen gets the count right; asked to continue the transcript, it hesitates and the goal line tips it.

### Sample 3: the most extreme item (ingredient role, too few)

Item `tc-p154-c2758e20ccc1` is off by one unit: polished blackstone needs 4 blackstone and the agent holds 3.

```text
History:
Crafting commands:
craft 6 stone brick wall using 6 stone bricks
craft 4 polished blackstone using 4 blackstone
craft 4 stone bricks using 4 stone
craft 1 polished blackstone pressure plate using 2 polished blackstone

Goal: craft polished blackstone pressure plate.
> get 3 blackstone
Got 3 blackstone
> get 2 stone
Got 2 stone

Action: craft 4 polished blackstone using 4 blackstone

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +15.50 | −17.50 | −12.50 | +33.00 | +33.00 | +12.00 | −24.25 | −17.25 |
| Llama-3.1-8B | +2.64 | −2.42 | −1.18 | +5.07 | +4.88 | −1.75 | −3.75 | −1.62 | 

This is the largest Qwen effect among the 1,500 main items. Under the neutral and swapped lines Qwen is confidently right that the craft fails. Under the aligned line, which names the pressure plate whose recipe needs polished blackstone and whose name shares the word "blackstone", it is just as confident that the craft succeeds, on both the two-candidate and the yes/no readouts. Llama shows the same pattern at a fifth of the size. The round-1 results note reports that Qwen's effect is heavy-tailed: the median item moves only +0.25 nats while the top 5% of items carry 80% of the total ([textcraft_results.md](../textcraft_results.md), follow-up analyses). This is one of those items.

### Sample 4: a reversed item where both models are wrong under every line (goal role, never obtained)

Item `tc-p012-7de7a9ee1543` crafts the goal, white dye, through the recipe that needs a lily of the valley, which the agent never fetched. The list also offers a second white-dye recipe from bone meal, of which the agent has made 9.

```text
History:
Crafting commands:
craft 1 turtle helmet using 5 scute
craft 1 white dye using 1 bone meal
craft 1 white dye using 1 lily of the valley
craft 3 bone meal using 1 bone

Goal: craft white dye.
> get 4 bone
Got 4 bone
> craft 3 bone meal using 1 bone
Crafted 3 minecraft:bone_meal
> craft 3 bone meal using 1 bone
Crafted 3 minecraft:bone_meal
> get 1 bone
Got 1 bone
> craft 3 bone meal using 1 bone
Crafted 3 minecraft:bone_meal

Action: craft 1 white dye using 1 lily of the valley

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +10.00 | +24.00 | +25.63 | −14.00 | −14.25 | +8.50 | +14.75 | +17.75 |
| Llama-3.1-8B | +2.81 | +4.44 | +3.31 | −1.63 | −1.63 | −1.50 | −1.63 | −1.50 |

Both models predict that this craft succeeds under every line, which is wrong: the action names the lily recipe, and the environment checks only the inputs the action lists. A plausible reading is that the models judge whether white dye *can* be made (it can, from bone meal) rather than whether *this* command works. The goal line moves Qwen by −14 nats, in the "more careful" direction: naming white dye as the goal makes it less sure of success, though still sure. This is the goal-role, never-obtained cell where round 1 found a negative effect in both models (figure 2 in section 5.1).

### Sample 5: Llama's goal-completion penalty in its plainest form (goal role, never obtained)

Item `tc-p009-90026de11831` has a one-step history: the agent fetched 3 wheat and now tries to craft jungle planks from jungle logs it never fetched.

```text
History:
Crafting commands:
craft 4 jungle planks using 1 jungle logs
craft 8 cookie using 2 wheat, 1 cocoa beans

Goal: craft jungle planks.
> get 3 wheat
Got 3 wheat

Action: craft 4 jungle planks using 1 jungle logs

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −29.58 | −26.60 | −34.13 | −2.98 | −1.38 | −33.12 | −33.12 | −32.50 |
| Llama-3.1-8B | −9.54 | −5.84 | −10.43 | −3.69 | −1.81 | −9.12 | −8.13 | −7.37 |

Both models get this right everywhere. What the goal line does is make a craft that would *finish the task* look even less likely than when the goal is the unrelated cookie. For Llama this goal-role penalty is the whole of its round-1 effect: −0.63 nats on the 510 goal-role items against +0.03 on the 990 ingredient items ([textcraft_results.md](../textcraft_results.md)). Notice that Llama's neutral score is lower than its swapped score: the swapped line (a different, displayed task) raises its score, so "more careful under the aligned line" is relative to another named task, not to having no task.

### Sample 6: the effect absent in Qwen, and Llama predicting success on a used-up input (ingredient role, used up)

Item `tc-p068-b4203105258d` is the longest history among the samples. The agent fetched 3 sugar cane, crafted 3 paper (using all the sugar cane), made a minecart, and fetched another iron ingot. The second inventory reply even shows that no sugar cane is left. The action tries to craft paper again.

```text
History:
Crafting commands:
craft 1 skull banner pattern using 1 paper, 1 wither skeleton skull
craft 3 paper using 3 sugar cane
craft 1 minecart using 5 iron ingot
craft 1 furnace minecart using 1 furnace, 1 minecart
craft 1 furnace using 8 stone crafting materials

Goal: craft skull banner pattern.
> get 3 sugar cane
Got 3 sugar cane
> get 2 iron ingot
Got 2 iron ingot
> inventory
Inventory: [sugar cane] (3) [iron ingot] (2) 
> craft 3 paper using 3 sugar cane
Crafted 3 minecraft:paper
> get 4 iron ingot
Got 4 iron ingot
> craft 1 minecart using 5 iron ingot
Crafted 1 minecraft:minecart
> inventory
Inventory: [iron ingot] (1) [paper] (3) [minecart] (1) 
> get 1 iron ingot
Got 1 iron ingot

Action: craft 3 paper using 3 sugar cane

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −7.25 | −7.00 | −8.75 | −0.25 | −0.25 | −22.75 | −22.50 | −17.75 |
| Llama-3.1-8B | +1.62 | +3.80 | +1.95 | −2.18 | −2.12 | −0.12 | +0.00 | +1.12 |

Qwen reads the state correctly and the goal line barely moves it, which is what a good world model should do. Llama favours the success reply under every line. This matches the competence analysis of the round-1 notes: Llama tends to judge a craft by whether every input's name appears somewhere in the history (here sugar cane does), not by tracking what is left, and it ranks used-up failures above real successes ([textcraft_llama_competence.md](../textcraft_llama_competence.md)). Its negative effect on this item comes from a model that is wrong under both lines, so it says little about a goal line distorting a working judgement.

### Sample 7: a working craft that the swapped line turns into a predicted failure

Item `tc-p006-e4c60e48426a` is a determined success: the agent holds 9 wheat and bread needs 3.

```text
History:
Crafting commands:
craft 1 hay block using 9 wheat
craft 1 bread using 3 wheat

Goal: craft bread.
> get 2 wheat
Got 2 wheat
> get 3 wheat
Got 3 wheat
> get 4 wheat
Got 4 wheat

Action: craft 1 bread using 3 wheat

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +19.52 | −5.12 | +20.59 | +24.64 | +24.75 | +18.25 | −21.50 | +19.00 |
| Llama-3.1-8B | +3.01 | +2.35 | +4.73 | +0.66 | +0.75 | +0.12 | −1.75 | +0.62 |

Under the neutral and aligned lines Qwen is confidently right. Under the swapped line ("Goal: craft hay block.", the other displayed task, which also uses wheat) it predicts the failure reply and answers "No". Its prefix-control scores, with the item name removed from both replies, are +19.75 aligned and −5.00 swapped, so the shift is in "Crafted 1" against "Could not find enough items to craft", not in the word "bread". The effect reaches working actions too: the round-1 notes report that on determined successes Qwen's effect is +2.61 nats [+1.82, +3.38], larger than on failures ([textcraft_results.md](../textcraft_results.md)). Against the neutral line the change here is a penalty under the wrong goal, not a lift under the right one.

### Sample 8: the yes/no question read as "does this help the task" (working craft)

Item `tc-p017-6287c5725565` is another determined success: 6 quartz are held and a quartz block needs 4.

```text
History:
Crafting commands:
craft 1 quartz block using 4 quartz
craft 4 spruce planks using 1 spruce logs

Goal: craft quartz block.
> get 2 quartz
Got 2 quartz
> get 4 quartz
Got 4 quartz
> get 2 spruce logs
Got 2 spruce logs

Action: craft 1 quartz block using 4 quartz

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | +16.00 | +9.00 | +13.25 | +7.00 | +7.00 | +17.00 | −30.00 | +14.75 |
| Llama-3.1-8B | −0.47 | −2.97 | +0.79 | +2.51 | +2.61 | −0.88 | −4.38 | −0.75 |

Here Qwen's predicted observation stays on the success side under every line (+9.00 even when swapped), yet when asked "Will this action succeed?" under the swapped line ("Goal: craft spruce planks.") it answers "No" with 30 nats of confidence. The two readouts of the same model diverge by 39 nats. The most natural reading is that the question's "succeed" is taken as "succeed at the task", and crafting a quartz block does not help to craft spruce planks. This item type is why the round-1 yes/no effect (+2.93 for Qwen) is not compared with ALFWorld's, and why later rounds anchor the question to the environment's replies ([textcraft_qwen_robustness.md](../textcraft_qwen_robustness.md) §9). The round-1 notes count it: on working goal crafts that Qwen's own predicted observation confidently calls successes, it answers "No" to 72% under the swapped line, against 13% aligned and 1% neutral.

### Sample 9: fetching a craftable item (failed-get stratum, step 0)

Item `tc-p000-e4d689132db0` is at the very start of a game. The action asks to fetch an iron nugget, which is craftable and therefore cannot be fetched; the true reply is `Could not find iron nugget`, and the success twin is the template `Got 1 iron nugget`.

```text
History:
Crafting commands:
craft 1 flower pot using 3 brick
craft 9 iron nugget using 1 iron ingot

Goal: craft iron nugget.

Action: get 1 iron nugget

Next observation:
```

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −4.49 | −13.25 | −10.50 | +8.76 | +10.00 | −22.75 | −30.75 | −22.00 |
| Llama-3.1-8B | +8.24 | +6.04 | +7.21 | +2.21 | +3.12 | −1.50 | −3.25 | −1.50 |

The failure here depends on a rule the agent is never told, which is why gets are kept out of the main test. Still, the pattern is the familiar one: naming the iron nugget as the goal makes fetching it look more plausible to Qwen (+8.76), and Llama expects the fetch to succeed under every line.

### Sample 10: an empty history (empty-history stratum)

Item `tc-p002-b035902c35d2` crafts the honeycomb-block goal before anything has been fetched. Its prompt is the recipe list (`craft 1 honeycomb block using 4 honeycomb`, `craft 1 dried kelp block using 9 dried kelp`), "Goal: craft honeycomb block.", no history steps, and the action `craft 1 honeycomb block using 4 honeycomb`.

| Model | aligned | swapped | neutral | aligned minus swapped | same, first token | yes/no aligned | yes/no swapped | yes/no neutral |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | −18.75 | −20.26 | −8.50 | +1.51 | +1.50 | −20.50 | −33.38 | −10.75 |
| Llama-3.1-8B | −6.36 | −6.99 | −7.99 | +0.63 | +0.06 | −7.63 | −10.25 | −7.50 |

Every score correctly predicts failure. The interesting comparison is with the neutral line: for Qwen, both named goals make failure *more* certain than the neutral line does, by about 10 nats. A task line is not a neutral piece of context even when it names the right item. Empty-history items were reported only as exploratory, because their failures are all of the "never obtained" kind and the design wanted failures settled by a history.

---

## 5. The studies, one by one

### 5.1 Round 1: the confirmatory replication (2026-11-20)

**What it asked.** On ALFWorld, naming an action's object in the task line raises a world model's predicted success for actions that the history shows will fail. Round 1 asked whether the same goal effect appears in TextCraft, where failures follow from inventory arithmetic rather than from where objects are. The design was frozen at tag `textcraft-design-frozen-2026-11-20` before any score ([textcraft_design.md](../textcraft_design.md)).

**The rule fixed in advance.** For each model the prediction counts as confirmed when the mean aligned minus swapped effect on the 1,500 main failures is above zero, its 95% pair-clustered interval excludes zero, and the goal-pair permutation p-value, Holm-adjusted over the two models, is below 0.01. The replication holds only if it is confirmed for both models.

**What it found.** From [textcraft_results.md](../textcraft_results.md) and [textcraft_metrics.json](../textcraft_metrics.json):

| Model | Goal effect (nats), 95% interval | Holm-adjusted p | Competence (neutral-line AUC) | Reading |
|---|---|---|---|---|
| Qwen3-8B (thinking off) | +0.94 [+0.63, +1.24] | 0.002 | 0.87 [0.85, 0.89] | confirmed: the goal effect replicates |
| Llama-3.1-8B-Instruct | −0.19 [−0.30, −0.09] | 0.002 | 0.68 [0.65, 0.71] | negative: naming the goal makes the model more careful |

The replication therefore does not hold across models: it is confirmed for Qwen3-8B only. Qwen's effect sits in its first-token decision (+0.97), with the name-dependent remainder near zero (−0.01), so the goal line moves the predicted outcome rather than the copying of a name. Llama's negative effect lies entirely in goal-completing crafts.

![A dot plot with five rows, one per combination of action role and failure class, and two markers per row: Qwen3-8B in blue circles and Llama-3.1-8B in orange squares, each with a horizontal 95% interval. Qwen is clearly positive on the goal too-few row and on the two ingredient rows for never obtained and too few, slightly negative on goal never obtained and near zero on ingredient used up. Llama is negative on both goal rows and near zero on all ingredient rows.](figures/textcraft_r1_role_class.png)

*Figure 2. Round 1's goal effect by role and failure class (from `descriptive.main.strata.role_x_class` in [textcraft_metrics.json](../textcraft_metrics.json); goal role with used-up inputs has only 6 items and is omitted). Look at the two goal rows: both models are negative where the goal's input was never obtained (samples 4 and 5), and Qwen's positive total comes from the "too few" cells and from ingredient crafts (samples 2 and 3). Neither model shows an effect on used-up ingredient items (sample 6).*

The samples of section 4 fit these cells: sample 3 sits in Qwen's large positive ingredient-too-few cell, samples 4 and 5 in the goal never-obtained cell where both models go negative, and sample 6 in the used-up ingredient cell where Qwen shows nothing.

### 5.2 Round 1's follow-ups and the Llama diagnostics (2026-11-20 to 2026-11-22, exploratory)

**What they asked.** After the confirmatory result was read, a set of exploratory analyses and small GPU runs asked why Llama's effect is negative and how robust Qwen's is ([textcraft_llama_dx.md](../textcraft_llama_dx.md), [textcraft_llama_competence.md](../textcraft_llama_competence.md), [textcraft_qwen_robustness.md](../textcraft_qwen_robustness.md), [textcraft_qwen_competence.md](../textcraft_qwen_competence.md), [textcraft_vs_alfworld.md](../textcraft_vs_alfworld.md), summarised in the follow-up section of [textcraft_results.md](../textcraft_results.md)). None of them changes the confirmatory result.

**The object-goal run.** To see whether the goal-role penalty follows whichever item is named as the goal, ingredient crafts were rescored under a line naming the crafted item itself. Item `tc-p206-b641eac5a2c5` (run `textcraft_dx_objectgoal`, commit `610b458`) is one of them: the agent has only looked at its empty inventory, and the action crafts warped planks from warped stems it never fetched. Its rendered prompt under the object-goal line:

```text
History:
Crafting commands:
craft 4 warped stairs using 6 warped planks
craft 4 warped planks using 1 warped stems
craft 1 birch boat using 5 birch planks
craft 4 birch planks using 1 birch logs

Goal: craft warped planks.
> inventory
Inventory: You are not carrying anything.

Action: craft 4 warped planks using 1 warped stems

Next observation:
```

| Model | aligned line, "craft warped stairs" | swapped line, "craft birch boat" | object goal, "craft warped planks" | yes/no under the object goal |
|---|---|---|---|---|
| Qwen3-8B | −18.00 | −20.85 | −21.13 | −26.75 |
| Llama-3.1-8B | −0.40 | −2.62 | −3.07 | −1.00 |

Making the ingredient itself the goal lowers both models' scores below the aligned line, by 3.1 nats for Qwen and 2.7 for Llama on this item. Across the run the follow-up notes report that scoring an ingredient craft of Y under "Goal: craft Y." lowers Llama's score by 1.40 nats on failures and 1.74 on successes: the penalty follows the item named as the goal, not the top item of a tree.

**Prompt variants.** A second run (`textcraft_dx_prompts`, commit `f74c262`) rescored 500 items under the clean system text of section 3.1, with the inventory shown as one extra history step, and with both. Item `tc-p065-0228d31b0208` (goal role, cobblestone never obtained) shows the inventory variant, whose last history step is the environment's own inventory reply:

```text
History:
Crafting commands:
craft 1 mossy cobblestone using 1 cobblestone, 1 vine
craft 1 leather using 4 rabbit hide

Goal: craft mossy cobblestone.
> get 1 vine
Got 1 vine
> get 2 vine
Got 2 vine
> inventory
Inventory: [vine] (3) 

Action: craft 1 mossy cobblestone using 1 cobblestone, 1 vine

Next observation:
```

The table gives this item's aligned minus swapped effect under each prompt.

| Model | round-1 prompt | clean system text | inventory shown | clean and inventory shown |
|---|---|---|---|---|
| Qwen3-8B | −2.22 | +1.24 | −1.90 | +1.23 |
| Llama-3.1-8B | −2.28 | −0.88 | −1.57 | −1.37 |

On this goal-role item the clean text moves both models up, Qwen across zero; showing the inventory changes little. That is the run's finding in miniature: removing the agent instructions raises the goal-role effect by +0.39 nats for Llama and +1.66 for Qwen while leaving competence about unchanged, and showing the inventory raises Qwen's competence (AUC 0.94) without changing either model's effect ([textcraft_results.md](../textcraft_results.md), "The prompt matters" and "Explanations not supported").

**Other findings, in brief.** The effect has two components with opposite signs: a lift of about +0.9 nats on intermediate crafts for Qwen (+0.03 for Llama) and a penalty on crafting the named goal, about −0.8 nats in both models when goal and ingredient items are matched. Llama's overall negative is the penalty without the lift; Qwen's positive total is the lift outweighing it. Moving the off-goal prompt example, stating that the inventory starts empty, or removing the examples did not remove Llama's reversal. Llama's competence in TextCraft is weak (AUC 0.61 on the full failure-success comparison), and its scores follow the cue "every input's name appears in the history" better than the true inventory rule. Qwen's effect is robust in direction but heavy-tailed, and best described as a task-relevance bias between the matching and mismatched task lines in a competent but overconfident predictor. These findings led to round 2.

### 5.3 Round 2: the confirmatory decomposition (2026-11-23)

**What it asked.** Round 2 tested the lift-and-penalty account of section 2.8 under the clean prompt, in five models, with a 50-test confirmatory family frozen at tag `textcraft-r2-design-frozen-2026-11-23` ([textcraft_r2_design.md](../textcraft_r2_design.md), results in [textcraft_r2_results.md](../textcraft_r2_results.md), numbers in [textcraft_r2_metrics.json](../textcraft_r2_metrics.json)). The models were Qwen3-8B (thinking off), Llama-3.1-8B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3 and Llama-3.2-3B-Instruct; each scored the same 25,743 record-and-line combinations. The central question, as frozen, was whether Llama's negative TextCraft effect comes from a goal-completion or naming mechanism, from poor reasoning about the state, or from both.

**The five contrasts.** All are computed on the two-candidate log-odds of failing actions, clustered by goal pair. The **intermediate lift** is the aligned minus swapped effect on the 990 ingredient-role failures,

$$\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}} .$$

The **within-item penalty** compares, on 675 parent failures, the same craft under a goal line that names the crafted item (so the craft completes the task) and under one that names a parent it is an ingredient of; the action, history and displayed list are identical,

$$P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})} .$$

The **naming** contrast uses the two-item line, which names the crafted item without the craft completing the task, and compares goal-role with ingredient-role failures,

$$D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}} ,$$

and the **completion** contrast asks what finishing the whole task adds once the item is named,

$$D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}} .$$

The **timing** contrast asks whether the goal-role penalty shrinks when seven empty look-ups are added,

$$T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}} ,$$

where the superscript counts the look-ups placed before the history; it is read only if the one-look-up role gap has an interval below zero. Each contrast has a "differs from zero" test and an equivalence test against a margin of the smaller of 0.3 item standard deviations and 0.4 nats, all 50 Holm-adjusted together. A verdict is "confirmed above" or "below zero", "equivalent to zero", "inconclusive", or, for timing only, "not applicable". A model that fails the contrast's competence gate (yes/no AUC of at least 0.70 in each population the contrast uses, plus coverage conditions) has its effect read as "measurable only".

**A parent record.** Item `tc-p010-38a14f0049db` crafts acacia planks from acacia logs that were never fetched. On the extended list the parent goal is the acacia trapdoor, which needs 6 acacia planks. The two prompts that make up the penalty differ only in the goal line:

```text
History:
Crafting commands:
craft 3 iron door using 6 iron ingot
craft 2 acacia trapdoor using 6 acacia planks
craft 4 acacia planks using 1 acacia logs

Goal: craft acacia planks.
> get 3 iron ingot
Got 3 iron ingot
> get 1 iron ingot
Got 1 iron ingot

Action: craft 4 acacia planks using 1 acacia logs

Next observation:
```

The parent arm replaces "Goal: craft acacia planks." with "Goal: craft acacia trapdoor.". The table gives the two scores and their difference, the item's penalty.

| Model | goal names the planks | goal names the trapdoor | penalty on this item |
|---|---|---|---|
| Qwen3-8B | −26.01 | −22.25 | −3.76 |
| Llama-3.1-8B | −9.95 | −2.30 | −7.65 |
| Gemma-2-9B-it | −3.89 | −2.18 | −1.71 |
| Mistral-7B-v0.3 | +12.63 | +12.88 | −0.25 |
| Llama-3.2-3B | +1.20 | +0.99 | +0.20 |

The three models that discriminate on goal crafting (Qwen, Llama-3.1 and Gemma) all predict failure under both lines and are more sure of failure when the planks are the goal. Llama-3.1 moves by more than 7 nats on this one craft, which is now held fixed. Mistral predicts success under both lines, as it does for nearly every action (it answers Yes to 99.9% of failures), so its small difference says nothing about a distorted judgement.

**A main record with the two-item and padded lines.** Sample 1a's melon craft also carries the round-2 arms. Its two-item prompt differs only in the goal line, "Goal: craft melon and stripped birch wood."; its padded prompts put one or eight copies of

```text
> inventory
Inventory: You are not carrying anything.
```

before "> get 2 stripped birch log". Under the clean prompt, its per-item pieces are these (round 2 scores, nats):

| Model | aligned minus swapped | two-item minus swapped (naming part) | aligned minus two-item (completion part) | aligned minus swapped, one look-up | aligned minus swapped, eight look-ups |
|---|---|---|---|---|---|
| Qwen3-8B | −6.50 | −5.09 | −1.42 | −3.75 | −1.25 |
| Llama-3.1-8B | −1.44 | +0.43 | −1.86 | −0.97 | −0.14 |
| Gemma-2-9B-it | +0.51 | +0.63 | −0.12 | −0.73 | −0.78 |
| Mistral-7B-v0.3 | +7.34 | +7.13 | +0.21 | +6.71 | +6.10 |
| Llama-3.2-3B | +1.05 | +1.04 | +0.00 | +0.74 | −0.13 |

On this single goal-role item Llama's negative effect is entirely in the completion part, the two-item line alone even raising its score, which is the pattern of its round-2 verdicts (completion confirmed below zero, naming confirmed above zero, both measurable only). Mistral's large positive effect on this item, with its predictions on the success side throughout, is a non-discriminating model reacting to the goal's name. A single item is noisy: the contrasts below are means over hundreds of items and compare goal with ingredient roles, which one item cannot show.

**What it found.** From [textcraft_r2_results.md](../textcraft_r2_results.md) §4:

| Model | Intermediate lift | Within-item penalty | Naming | Completion | Added empty steps |
|---|---|---|---|---|---|
| Qwen3-8B | +0.67, confirmed above zero | −1.02, confirmed below zero | +0.10, inconclusive | +0.65, inconclusive | +0.80, not applicable |
| Llama-3.1-8B | +0.10, equivalent to zero (measurable only) | −1.68, confirmed below zero | +0.78, confirmed above zero (measurable only) | −1.26, confirmed below zero (measurable only) | +0.03, equivalent to zero (measurable only) |
| Gemma-2-9B-it | −0.30, confirmed below zero | −1.27, confirmed below zero | −0.57, confirmed below zero | −0.05, equivalent to zero | −0.13, equivalent to zero |
| Mistral-7B-v0.3 | +0.27, confirmed above zero (measurable only) | +0.27, confirmed above zero (measurable only) | +0.94, confirmed above zero (measurable only) | +0.05, equivalent to zero (measurable only) | −0.40, not applicable (measurable only) |
| Llama-3.2-3B | −0.04, equivalent to zero (measurable only) | −0.50, confirmed below zero (measurable only) | +0.15, inconclusive (measurable only) | −0.12, equivalent to zero (measurable only) | −0.29, not applicable (measurable only) |

The penalty survives when item structure is removed: all three models that discriminate on goal crafting show a within-item penalty confirmed below zero, between about −1.0 and −1.7 nats, so the across-model reading is "penalty general across the competent models". The lift is not shared: confirmed above zero for Qwen, negative for Gemma, and equivalent to zero (where it does not discriminate) for Llama-3.1. Qwen's reading is "penalty present, mechanism unresolved"; Gemma's is a naming or partial-completion penalty. The descriptive diagnostic R3 fired for Qwen (its within-item penalty and its between-item role gap disagree in sign), but the results note marks that reading as numerically fragile and not established, because a rescoring in newer software moves the role gap's interval to include zero (§6.1, revised 2026-11-25).

![Five side-by-side panels, one per contrast, each with the five models as rows and a dot with a 95% interval. Filled blue dots mark models that discriminate in that contrast's populations, hollow grey dots those that do not. In the penalty panel the three filled dots, for Qwen, Llama-3.1 and Gemma, all lie between minus 1 and minus 1.7 nats.](figures/textcraft_r2_contrasts.png)

*Figure 3. Round 2's 25 confirmatory estimates with their 95% pair-bootstrap intervals, from [textcraft_r2_metrics.json](../textcraft_r2_metrics.json) (`per_model.<model>.contrasts` and `competence.local.per_contrast`). Look at the within-item penalty panel: every filled (discriminating) dot is well below zero, while the filled dots of the lift panel fall on both sides of zero.*

### 5.4 Round 3: the final reading (2026-11-27)

**What it asked.** Round 3 asks whether round 2's quantities behave the same way in recent open-weight instruction models, holding everything else fixed: the same 2,982 records, the same goal lines, the same clean prompt, the same scoring and estimands. Its frozen comparisons have three parts. The first is a direct replication of round 2's five contrasts in each new model, in two Holm families of 30 tests each (three models, times five contrasts, times a "differs from zero" and an "equivalent to zero" test): the **core family** holds Qwen3.5-9B, Ministral-3-8B and one Gemma model chosen by a frozen rule (the "Gemma slot"), and the **extension family** holds Qwen3.8-27B, Gemma 4 31B and OLMo 3 7B. The second compares three older-against-newer pairs from the same developer lines (Qwen3-8B against Qwen3.5-9B, Mistral-7B against Ministral-3-8B, Gemma-2-9B against the Gemma slot) on a standardized scale. The third is a set of exploratory readouts with no test. The design was frozen at tag `textcraft-r3-design-frozen-2026-11-23`, with dated deviations D-T1, D-G1, D-T2, D-G2 and D-G2a.

**What this reading is.** The final analysis `textcraft_r3_analysis_dg2` ran once, after every arm was filed, from a clean checkout at the final tag `textcraft-r3-deviation-g2a-2026-11-26` (commit `f0ac203`), as ICE CPU job 7374476 ([textcraft_r3_results.md](../textcraft_r3_results.md), numbers in [textcraft_r3_metrics.json](../textcraft_r3_metrics.json)). It replaces the INTERIM reading of 2026-11-25, which was made at the user's request with only Qwen3.5-9B, Ministral-3-8B and OLMo 3 7B filed; that note is kept, marked superseded ([textcraft_r3_interim_results.md](../textcraft_r3_interim_results.md)). No verdict reported in the interim reading changed: the final run adds the verdicts of Gemma 4 12B, Qwen3.8-27B and Gemma 4 31B, the Gemma pair of the older-against-newer comparison, the cross-model readings of both families, two format checks and two size comparisons (results note, section 8).

**How the Gemma slot was filled.** Every Gemma 4 job starts with an attention-agreement check: it scores five items with two attention implementations that should give the same answer (eager and SDPA) and stops if they disagree. The frozen design ran this check in bf16, where rounding alone made the implementations disagree beyond the limits for Gemma 4 12B, E4B and every scoring shard of 31B (up to 0.81 nats on average). Deviation D-G2 evaluates the same criterion in float32, where all three pass with differences of at most 0.0003 nats, and the frozen fallback rule then gave the Gemma slot to **Gemma 4 12B**. Gemma 4 E4B, scored earlier as an exploratory run (D-G1), stays outside every family.

**Older models.** The environment check, which rescored 300 round-2 records of each older model in round 3's newer software stack (called serve2), failed for all three older models, so the older-against-newer comparison uses each older model's rescoring of every record in serve2. Two exploratory arms, outside every test family, are also reported: Gemma 4 E4B, and Qwen3.5-9B with reasoning switched on (its reasoning is sampled and then the two replies are teacher-forced after it).

**Gates and competence.** Every model passed every scoring gate (loading, tokenization, golden renderings, weights manifest, pinned GPU type, freeze commit) and the analysis found no provenance problem. A model is competent if it separates successes from failures on the yes/no question (AUC at least 0.70), does not answer Yes to nearly everything, and puts at least half its probability on Yes and No; a model that fails this has its effects read as "measurable only". From the results note, section 2:

| Model | Family | Competent | Yes/no AUC | Yes to failures / to successes | Equivalence margin (nats) |
|---|---|---|---|---|---|
| Qwen3.5-9B | core | yes | 0.905 | 4.4% / 56.5% | 0.175 |
| Ministral-3-8B | core | yes | 0.864 | 10.2% / 65.7% | 0.40 |
| Gemma 4 12B | core (the Gemma slot) | yes | 0.943 | 5.4% / 83.5% | 0.40 |
| Qwen3.8-27B | extension | yes | 0.999 | 0.7% / 96.2% | 0.115 |
| Gemma 4 31B | extension | yes | 0.997 | 1.9% / 97.5% | 0.40 |
| OLMo 3 7B | extension | **no** (Yes to 99.8% of failures) | 0.560 | 99.8% / 100% | 0.40 |

The equivalence margin is the smaller of 0.3 item standard deviations and 0.4 nats; it falls below 0.4 nats only where 0.3 item standard deviations is the smaller of the two, as for the two Qwen models.

**A parent record, round 3.** Item `tc-p057-a1b1aa9a2a86` crafts prismarine bricks from prismarine shards that were never fetched; the agent has only fetched oak logs. The parent goal is the prismarine brick stairs.

```text
History:
Crafting commands:
craft 4 oak planks using 1 oak logs
craft 1 prismarine bricks using 9 prismarine shard
craft 4 prismarine brick stairs using 6 prismarine bricks

Goal: craft prismarine bricks.
> get 1 oak logs
Got 1 oak logs
> get 2 oak logs
Got 2 oak logs
> get 4 oak logs
Got 4 oak logs

Action: craft 1 prismarine bricks using 9 prismarine shard

Next observation:
```

The parent arm reads "Goal: craft prismarine brick stairs." instead. Scores in nats (prompt of round 3 commit `fae00b3`; the runs were scored at four round-3 commits whose prompt builder is identical, section 7). Both Gemma 4 models were scored under deviation D-G2 (D-G2a for 31B), which reads the start check comparing eager and SDPA attention in float32; the scores themselves are bf16 with SDPA, as for every other model. The rows of Qwen3.5-9B and Qwen3.8-27B carry the results note's label for their mean penalties (section 6 of that note):

| Model | goal names the bricks | goal names the stairs | penalty on this item |
|---|---|---|---|
| Qwen3.5-9B (mean penalty numerically fragile) | −2.26 | −2.34 | +0.09 |
| Ministral-3-8B | −6.02 | −4.51 | −1.51 |
| Gemma 4 12B, the Gemma slot (scored under D-G2) | −12.88 | −13.87 | +0.99 |
| Qwen3.8-27B (mean penalty numerically fragile) | −2.38 | −2.25 | −0.13 |
| Gemma 4 31B (scored under D-G2a) | −30.07 | −32.08 | +2.01 |
| OLMo 3 7B | −4.76 | −4.18 | −0.58 |
| Qwen3-8B, rescored in round 3's software | −25.85 | −23.85 | −2.00 |
| Gemma-2-9B, rescored in round 3's software | −7.94 | −5.43 | −2.51 |
| Gemma 4 E4B (exploratory) | −5.98 | −6.63 | +0.65 |
| Qwen3.5-9B with reasoning (exploratory) | −1.75 | −2.25 | +0.50 |

Qwen3.5-9B predicts failure under both lines and is not moved by which item is the goal, which is its final verdict in miniature (penalty equivalent to zero). Ministral-3-8B shows a penalty of about the size of its mean. The two older models, rescored, still show theirs. Of the three models the final analysis added, both Gemma 4 models are more optimistic when the bricks are the goal (12B +0.99, 31B +2.01), on the same side as their mean penalties, while Qwen3.8-27B barely moves (−0.13) against a mean of +0.11. On the same item as round 2's acacia example, single-item values vary a good deal (Ministral is +0.73 there, Qwen3.5-9B +0.06); the verdicts rest on means over 675 records.

With reasoning on, Qwen3.5-9B writes out its bookkeeping before the replies are scored. An excerpt of its trace under the "bricks" goal (348 new tokens, not cut off):

```text
Looking at the history:
- The goal is to craft prismarine bricks
- The user has been gathering oak logs (3 times, totaling 7 oak logs)
- The action is to craft 1 prismarine bricks using 9 prismarine shard

However, looking at the inventory situation:
- The user has 7 oak logs but no prismarine shard mentioned in inventory
- The crafting command requires 9 prismarine shard
- The user doesn't have enough prismarine shard to craft
[...]
So for this case, it should be:
"Could not find enough items to craft minecraft:prismarine_bricks"
```

Under the "stairs" goal its trace (386 tokens) reaches the same conclusion, although it miscounts the oak logs as 4 along the way. Both traces name the goal, as 67.5% of all traces do. After reasoning, the item's two-candidate scores are −1.75 and −2.25, both on the failure side.

**What it found: the core family.** Estimates are in nats with the verdict of the deciding Holm-adjusted test, from [textcraft_r3_results.md](../textcraft_r3_results.md) §3 (the 95% goal-pair bootstrap intervals are given there and, for lift and penalty, in Figure 4):

| Model | Intermediate lift | Within-item penalty | Naming | Completion | Added empty steps |
|---|---|---|---|---|---|
| Qwen3.5-9B | +0.074, confirmed above zero | −0.020, equivalent to zero | +0.053, inconclusive | +0.125, inconclusive | −0.086, not applicable |
| Ministral-3-8B | +0.294, confirmed above zero | −1.231, confirmed below zero | −0.307, inconclusive | −0.014, equivalent to zero | −0.075, equivalent to zero |
| Gemma 4 12B | +0.148, inconclusive | +0.162, inconclusive | −0.195, inconclusive | +0.598, confirmed above zero | −0.430, not applicable |

Read model by model. Qwen3.5-9B has a small lift and no penalty larger than its 0.175-nat margin (interval −0.085 to +0.050), so rejection rule R1, which fires for a competent model whose penalty is equivalent to zero or confirmed above zero, fires for it. Ministral-3-8B has a penalty as large as round 2's (−1.231, interval −1.441 to −1.027); the results note adds that this penalty sits mostly in the later tokens of the replies (first token −0.11, the rest −1.12 nats), and that Ministral puts only about half its probability on the replies' first tokens (median 0.49, against 0.97 for Mistral-7B). Gemma 4 12B is inconclusive on the penalty (interval −0.019 to +0.367); the one contrast it decides is completion, confirmed above zero, which means that completing a named goal raises its predicted success relative to only naming it.

The prediction carried from round 2, H3-P, is that every core model's within-item penalty is confirmed below zero. Its reading is **partial**: Ministral-3-8B supports it, Qwen3.5-9B rejects it through R1, and Gemma 4 12B is inconclusive. "Partial" is reported per model and never counted as support. Rule R2, restated for round 3, fires if no competent model has its penalty confirmed below zero while at least one has it equivalent to zero or above zero; it **does not fire** here, because Ministral's penalty is confirmed below zero. The replication rule compares each contrast with round 2's finding: replication of round 2's general penalty is **partial** (Ministral confirms it, Qwen3.5 is equivalent to zero, Gemma 4 12B is inconclusive). Round 2 established no general finding for the other four contrasts, so they have no replication target, and their verdicts above are new evidence.

**What it found: the extension family.** From the same section:

| Model | Intermediate lift | Within-item penalty | Naming | Completion | Added empty steps |
|---|---|---|---|---|---|
| Qwen3.8-27B | +0.032, equivalent to zero | +0.106, confirmed above zero | +0.033, inconclusive | +0.092, confirmed above zero | −0.024, not applicable |
| Gemma 4 31B | −0.433, confirmed below zero | +0.652, confirmed above zero | −0.358, inconclusive | +0.939, confirmed above zero | −0.073, not applicable |
| OLMo 3 7B (not competent) | +0.456, confirmed above zero (measurable only) | +0.003, equivalent to zero (measurable only) | −0.270, inconclusive | +0.381, inconclusive | +0.105, not applicable |

Both competent extension models read "completing a named goal raises predicted success (no penalty)": their penalties are confirmed **above** zero (Qwen3.8-27B interval +0.072 to +0.139; Gemma 4 31B +0.468 to +0.830), so R1 fires for both, and Gemma 4 31B's lift is negative. OLMo 3 7B answers Yes to almost every failure, so its effects are measurable only. For this family H3-P is **rejected**, R2 **fires** (the penalty is not shared among these models), and the penalty is **not replicated**: both competent models are confirmed on the opposite side of zero from round 2.

**Older against newer releases.** Each pair is compared on a standardized scale, the contrast's mean divided by the standard deviation of its item values,

$$d = \frac{\bar{x}}{s_{\text{item}}},$$

and the comparison is the newer model's $d$ minus the older model's, with an equivalence margin of 0.20 and Holm adjustment over this comparison's own family of 30 tests. Raw nats are descriptive only. From the results note, section 4 (95% intervals in brackets):

| Pair (older → newer) | Contrast | Standardized difference, newer minus older | Reading |
|---|---|---|---|
| Qwen3-8B → Qwen3.5-9B | Intermediate lift | −0.016 [−0.137, +0.119] | similar within the margin |
| | Within-item penalty | +0.172 [+0.014, +0.319] | not distinguishable (equivalence underpowered) |
| | Naming, completion | +0.075, +0.048 | not distinguishable |
| | Added empty steps | −0.487 [−0.727, −0.258] | differs, newer lower (opposite signs) |
| Mistral-7B → Ministral-3-8B | Intermediate lift | −0.017 [−0.120, +0.081] | similar within the margin |
| | Within-item penalty | −1.022 [−1.225, −0.834] | differs, newer lower (opposite signs) |
| | Naming | −0.792 [−1.021, −0.569] | differs, newer lower (opposite signs) |
| | Completion | −0.055 [−0.292, +0.171] | not distinguishable |
| | Added empty steps | +0.353 [+0.144, +0.568] | differs, newer higher (smaller in size) |
| Gemma-2-9B → Gemma 4 12B | Intermediate lift | +0.305 [+0.200, +0.413] | differs, newer higher (opposite signs) |
| | Within-item penalty | +0.930 [+0.755, +1.111] | differs, newer higher (opposite signs) |
| | Naming | +0.369 [+0.160, +0.579] | differs, newer higher (smaller in size) |
| | Completion | +0.350 [+0.172, +0.542] | differs, newer higher (opposite signs) |
| | Added empty steps | −0.076 [−0.279, +0.142] | not distinguishable |

In the Qwen pair the penalty shrinks in raw nats from −0.99 to −0.02, but on the standardized scale only from −0.21 to −0.04 standard deviations, because Qwen3-8B's item values are about eight times as spread; the two are not distinguishable. In the Mistral pair the older model, Mistral-7B, is not competent (as in round 2), so a non-discriminating model is compared with a discriminating one. In the Gemma pair both models are competent, and Gemma-2-9B's penalty of −0.84 standard deviations (−1.29 nats, in its rescoring) becomes +0.09 (+0.16 nats) in Gemma 4 12B, while the lift changes sign from negative to positive: within the Gemma line the penalty is gone, a reading outside the equivalence margin and not numerically fragile.

**Exploratory readouts (no test).** With reasoning switched on, Qwen3.5-9B wrote 4,530 traces (median 671 new tokens, 36.7% closed at the 2,048-token cap; 67.5% name the goal and 99.96% the crafted item). Its penalty after reasoning is −0.42 nats [−0.74, −0.11], or −0.10 standard deviations [−0.18, −0.03], against −0.02 nats without reasoning; the change with reasoning is −0.40 nats [−0.71, −0.10] in raw nats but −0.07 standard deviations [−0.20, +0.07] on the standardized scale, so the conclusion depends on the scale. On the unforced traces only (a selected subset) the penalty is −0.18 nats [−0.57, +0.22], and yes/no AUC after reasoning is 0.965. Gemma 4 E4B is competent (AUC 0.811) and shows a lift of +0.100 [+0.031, +0.164] and a penalty of −0.296 [−0.450, −0.149] nats; it is not used for the slot, which Gemma 4 12B holds. Two format checks, each on 300 seeded failures and read by the rule that a change matters if its interval excludes zero and it exceeds a fifth of the native effect: Qwen3.8-27B rendered with its template option `preserve_thinking=False` is format-dependent on the two-candidate scale (the effect moves by −0.073 nats [−0.121, −0.027] against a native +0.067) but not on yes/no; Gemma 4 12B with its system text placed in the first user turn (the first-turn placement check) is not format-dependent on either scale. Within both developer lines the larger model's penalty is more positive: Gemma 4 12B to 31B, lift −0.19 [−0.29, −0.09] and penalty +0.25 [+0.09, +0.43] standard deviations; Qwen3.5-9B to Qwen3.8-27B, penalty +0.37 [+0.22, +0.53].

**Numerically fragile readings.** Under the project's numerical-sensitivity rule ([overview](00_overview.md), section 3.8) a verdict is fragile when its deciding interval ends within the model's numerical allowance of its boundary (zero for "confirmed", the margin for "equivalent"). The allowance is about 0.1 nats for Qwen-family models, 0.05 for Ministral and OLMo, and, for the Gemma 4 models, values that D-G2a fixed before any 31B score existed (for 31B, 0.065 nats for the lift and 0.078 for the penalty; 12B about half of each). Five verdicts are fragile, all in Qwen models:

- Qwen3.5-9B's lift confirmed above zero (its interval ends 0.034 above zero);
- Qwen3.5-9B's penalty equivalent to zero (its 90% interval ends 0.100 inside the 0.175 margin), on which its R1 rests;
- Qwen3.8-27B's lift equivalent to zero (0.061 inside its 0.115 margin);
- Qwen3.8-27B's penalty confirmed above zero (0.072 above zero), which feeds R1, H3-P and R2 in the extension family;
- Qwen3.8-27B's completion confirmed above zero (0.045 above zero).

No Ministral, OLMo, Gemma 4 or older-against-newer verdict is fragile. In particular Gemma 4 31B's penalty above zero ends 0.47 above zero, so the extension family's rejection of H3-P does not rest on the fragile Qwen3.8-27B readings alone.

**What the results say, and what they do not.** Round 2's general finding, a within-item penalty in every competent model, does not hold across these contemporary releases: one core model (Ministral-3-8B) shows it at round 2's size, one (Qwen3.5-9B) shows none larger than its margin (a numerically fragile reading), one (Gemma 4 12B) is inconclusive, and both competent larger models (Qwen3.8-27B and Gemma 4 31B) show the opposite, with completing a named goal raising their predicted success. The older-against-newer comparison separates the release lines: the penalty disappears within the Gemma line, the drop in nats within the Qwen line is mostly a change of scale, and the Mistral line compares a non-discriminating older model with a discriminating newer one. As the design states, these are a small purposive sample, one model per developer line, and they license no claim about "model generations": two releases differ in data, size, tokenizer, template, post-training, architecture and software at once.

![Two side-by-side dot plots sharing one list of thirteen models: the five round-2 models in blue circles, the six round-3 models of the final analysis in orange squares (marked core or extension), and the two exploratory runs in aqua diamonds; hollow markers for models that do not discriminate. The left panel shows the intermediate lift, mostly small and positive, with Gemma-2-9B and Gemma 4 31B below zero. The right panel shows the within-item penalty: clearly negative for Qwen3-8B, Llama-3.1-8B, Gemma-2-9B and Ministral-3-8B, near zero for Qwen3.5-9B and OLMo 3, slightly positive for Gemma 4 12B and Qwen3.8-27B, and clearly positive for Gemma 4 31B.](figures/textcraft_r3_lift_penalty.png)

*Figure 4. Lift and penalty for every TextCraft model, from [textcraft_r2_metrics.json](../textcraft_r2_metrics.json) and round 3's final [textcraft_r3_metrics.json](../textcraft_r3_metrics.json) (`per_model.<model>.contrasts`, and `exploratory` for the two aqua rows). Look at the penalty panel: the penalty is large in Ministral-3-8B, the newer Mistral, has vanished in Qwen3.5-9B, the newer Qwen, is inconclusive in Gemma 4 12B, the newer Gemma, and lies above zero in the two larger models, Qwen3.8-27B and Gemma 4 31B. The lift is mixed in sign, largest for round 2's Qwen3-8B and most negative for Gemma 4 31B.*

---

## 6. Glossary

**Aligned, swapped, neutral.** The three basic goal lines: the item's own goal, the other goal of its pair, and "Goal: explore the crafting commands.".

**AUC.** Area under the ROC curve: the probability that a randomly chosen real success gets a higher score than a randomly chosen real failure. 0.5 is chance.

**Competence gate.** A condition, fixed before data, on how well a model tells successes from failures; a model that fails it has its effects read as "measurable only", not as a distortion of a working prediction.

**Couple (same-history couple).** Two main items from one goal pair with the same history, class and role, one acting in each goal's tree.

**Craftable item, base item.** An item that is the output of some recipe line, and one that is not (and can be fetched).

**Failure classes.** Never obtained, too few, used up: how the short input was held during the history.

**First-token readout.** The log-odds of the two replies' first tokens ("Craft" or "Got" against "Could").

**Goal effect.** The mean change in a model's score when only the goal line changes, usually aligned minus swapped.

**Goal pair.** Two goals of equal crafting depth with no craftable item in common, whose trees make up the displayed recipe list; the clustering unit for all intervals.

**Holm adjustment.** A step-down correction of a family of p-values that controls the chance of any false claim in the family.

**Lift, penalty, naming, completion, timing.** Round 2's five contrasts (section 5.3).

**Log-odds, nats.** The difference of two log-probabilities, in natural-log units; one nat multiplies the odds by about 2.7.

**Main population.** The 1,500 failing crafts of round 1's primary test.

**Needs arithmetic.** A failing craft whose short inputs were all obtained at some point, so that only counting shows the failure.

**Parent record.** A goal-role craft of an item that is also an ingredient of another item (its parent), shown on a list extended with the parent's recipe.

**Prefix control.** Round 1's arm that scores the replies with the object's name removed.

**Role (goal, ingredient).** Whether the action crafts the goal itself or a direct ingredient of it.

**Success twin.** The environment's real reply to the same action after only the missing amounts are topped up.

**Teacher forcing.** Scoring a fixed reply by feeding its tokens after the prompt and summing their log-probabilities.

**Two-candidate log-odds.** The log-probability of the success twin minus that of the failure reply.

**Yes/no readout.** The log-odds of "Yes" against "No" after a question about whether the action succeeds.

---

## 7. Provenance, licences and how to regenerate this document

**Scores.** Round 1: run `textcraft_score` (ICE jobs 7204417–7204420, commit `5a3702a`); the object-goal and prompt-variant runs `textcraft_dx_objectgoal` (commit `610b458`) and `textcraft_dx_prompts` (commit `f74c262`). Round 2: runs `textcraft_r2_score_<model>` (commit `4ccf84b`). Round 3: the item-level scores shown here come from `textcraft_r3_score_qwen3.5-9b`, `textcraft_r3_score_ministral3-8b-it`, `textcraft_r3_score_olmo3-7b-it`, `textcraft_r3_score_qwen3.8-27b` and the three `textcraft_r3_rescore_*` runs (scored at the freeze commit `0d6f460`), `textcraft_r3_think_qwen3.5-9b` and `textcraft_r3x_gemma4-e4b-it` (D-T1's commit `98dfbbf`), `textcraft_r3_score_gemma4-12b-it` (D-G2's commit `fae00b3`) and `textcraft_r3_score_gemma4-31b-it` (D-G2a's commit `f0ac203`); the prompt builder (`wf/textcraft/r2.py`, `prompts.py` and the item and example files) is byte-identical at these four commits, and the prompts are shown as rendered at `fae00b3`. Round 3's headline numbers come from the final analysis `textcraft_r3_analysis_dg2`, run at tag `textcraft-r3-deviation-g2a-2026-11-26` (commit `f0ac203`, ICE CPU job 7374476), which read the same score runs; the copies of the Gemma 4 12B, Qwen3.8-27B and Gemma 4 31B runs used here were taken from that analysis's checkout (`text-wm-fix56`), and their scoring files' SHA-256 values equal the cluster files'. The score folders are archived outside the repository.

**Verification.** `scripts/walkthroughs/textcraft_samples.py` renders every prompt shown here with the prompt module of the commit its scores came from, recomputes that commit's item hash (which covers the item's prompt inputs and candidates, and, for rounds 1 to 3, the system text, question and prompt examples), and stops unless the hash equals the `item_hash` stored on every score record of that item; in round 3 it recomputes the hash with the checkout of each run's own scoring commit and checks it against that run's records. It matched for every sample (16 records per round-1 item; 30 to 60 per round-2 record and 62 to 121 per round-3 record). The two diagnostic-run samples are rendered with `wf/textcraft/dx.py` at commit `f74c262` and are not hash-checked by the walkthrough script; their scores are matched by item id and condition. `scripts/walkthroughs/textcraft_figures.py` draws the figures. Both read a local copy of the run archive and never rescore anything:

```text
python scripts/walkthroughs/textcraft_samples.py --runs RUNS_ROOT --trees TREES_ROOT
python scripts/walkthroughs/textcraft_figures.py --runs RUNS_ROOT --out notes/walkthroughs/figures
python scripts/walkthroughs/textcraft_figures.py --only r2 r3   # Figures 3 and 4 only; reads committed metrics, no run archive
```

**Licences.** TextCraft comes from ADaPT (`archiki/ADaPT`), under the MIT licence, "Copyright (c) 2023 Archiki Prasad"; AgentGym (`WooooDyy/AgentGym`), whose TextCraft code and instruction text are used here, is under the MIT licence, "Copyright (c) 2023 FudanNLP-Agent". Both licence files were read on 2026-11-26. The recipe data are Minecraft game data (Mojang), which those licences do not cover; following the project's convention ([DATA_LICENSES.md](../../DATA_LICENSES.md)), the recipe database itself is not committed, and this walkthrough quotes only the few recipe lines each example displays, as they appear in the project's committed item files. The quoted reasoning excerpt is output of Qwen3.5-9B; scores are numbers computed by this project from the models' log-probabilities.
