# A reader's guide to how the experiments are set up and scored

This guide explains, from first principles, what this repository measures and how. It is written for a reader who knows machine learning and statistics but has not followed the work day to day. It defines each term the first time it appears, shows real prompt text taken from the code, names the file and function that implement each mechanism, and points to the note where each result is reported in full.

The guide adds no results of its own. Every number in it is quoted from a results note or metrics file on `main`, or, for studies that exist only on branches, from the branch's design note. Where two sources disagree, the guide says so and follows the newer one; `notes/deviations.md` is the dated decision log and is current where it differs from older text. The status of each study is as of 2026-11-22. When this guide and a results note disagree, the results note is authoritative.

## Contents

1. The question in one page
2. The environments
3. The manipulation: task lines
4. Readouts: how a prediction is turned into a number
5. The effect measures
6. Statistics, in plain words
7. How studies are run and kept honest
8. Each study in brief
9. Glossary
10. A suggested reading order

---

## 1. The question in one page

### What a "world model" is here

A language-model agent acts in an environment by issuing text commands ("take mug 1 from cabinet 3") and reading the environment's text replies ("You pick up the mug 1 from the cabinet 3." or "Nothing happens."). A **world model** is a model that predicts the environment's reply to an action before the action is taken. Agents use such predictions to look ahead: to check a proposed action before committing to it, to rank candidate actions, or to plan several steps ahead without touching the real environment.

In this project the world model is an ordinary instruction-tuned language model (for example Qwen3-8B or Llama-3.1-8B-Instruct) prompted to play that role. Its system message in ALFWorld reads, verbatim (`wf/scoring/prompts.py`, `WM_SYSTEM`):

> You are a text world model for household embodied tasks. Given the interaction history and the next action, predict the environment's next observation exactly in ALFWorld style. Output only the observation text.

The prompt then shows a few worked examples, the history of the episode so far and the next action, and asks for the "Next observation:". Section 2 shows a full prompt.

The history in these environments contains the agent's task. In ALFWorld it is a sentence in the first observation ("Your task is to: put two soapbar in garbagecan."). As the README puts it, a world model consulted by an agent "is often shown the agent's task".

### Why the outcomes do not depend on the task

The environments studied here are **goal-independent**: what happens when an action is executed depends on the current state of the world and the action, and not on what the agent is trying to achieve. A cabinet that is closed stays closed whether the agent wants a soap bar or a spray bottle. Write $s_t$ for the state at step $t$, $a_t$ for the action, $g$ for the goal (the task) and $s_{t+1}$ for the next state. Goal independence is the statement

$$
P(s_{t+1} \mid s_t, a_t, g) = P(s_{t+1} \mid s_t, a_t).
$$

In words: once the state and the action are known, also knowing the goal tells you nothing more about the outcome. In all three environments the reply is in fact a deterministic function of state and action (Section 2 explains why for each), so the conditional distribution on the right puts all its mass on a single reply.

A world model that has learned the environment correctly should therefore satisfy the same equation. Its prediction for a given history and action should not change when only the task sentence changes. This project tests whether it does, on actions where the answer is not in doubt: actions whose failure is already settled by what the history shows.

### What a violation looks like, and why it matters

A violation is a prediction that moves toward success when the task names the object the action acts on. The canonical example, from the README: the agent has not gone to cabinet 3, and the next action is `take mug 1 from cabinet 3`. The environment will reply "Nothing happens.", because in ALFWorld you can only take things from a receptacle you are standing at. A goal-independent world model predicts failure under any task. A world model whose prediction shifts toward "You pick up the mug 1 from the cabinet 3." when the task is "cool some mug and put it in cabinet" than when it is "cool some plate and put it in cabinet" is letting the goal leak into a prediction it should not affect. The project calls this the **goal effect**.

The concern for agents is a specific failure mode. A verifier or planner that consults such a world model will be most optimistic exactly about the actions that would advance its goal, because the goal makes them look more likely to work. Those are the actions an agent is most likely to choose. An error toward false success on goal-relevant actions is therefore not a random error; it is correlated with what the agent wants. The decision-simulation studies (Section 8) ask whether this changes which action a verifier picks.

The preregistered primary test (RQ2) found the effect in both reference models on ALFWorld. Later studies ask whether it holds in other models and environments, what drives it, and whether it changes decisions. The README's summary table is the quickest overview of the results.

---

## 2. The environments

Three environments are used. Each has a different kind of state, action and reply, but all three share the property that matters: the reply is a deterministic function of the state and the action. For each, the project builds **items**. An item is one history plus one next action, with the environment's true reply. The core population is made of **determined failures**: actions that fail, and whose failure the history alone already proves, so a model that reads the history correctly has everything it needs to predict failure.

### 2.1 ALFWorld

**What it is.** ALFWorld is a text version of household tasks. In this repository "ALFWorld" means, specifically, ALFWorld as served by the AgentGym server that ships with Word2World (checkout `e8dc240`), with Word2World's renamed actions (`put` is called `move`, and there is a `help` action) and alfworld 0.3.3. Only solvable games are used: 134 of the 195 in AgentGym's `valid_unseen` list; the rest crash on reset and `wf/env/server.py` filters them out. The environment ends every episode at 50 steps (`notes/rq2_results.md`, "Methods").

**State, action and reply.** The state is the room: which receptacles exist, what is in or on each, which are open, where the agent stands and what it holds. The agent sees only what observations have revealed. Here is a real item from the confirmatory collection, stored as the golden test fixture `tests/golden/default_prompt.json` (item `t_002295`, game `valid_unseen/191`, real server text). Its history is:

```
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put two soapbar in garbagecan.
> go to cabinet 1
You arrive at cabinet 1. The cabinet 1 is closed.
```

The next action is `take soapbar 1 from cabinet 1`. The environment's true reply is `Nothing happens.`, because cabinet 1 is closed. The success reply the model is compared against (the "success twin", explained below) is `You pick up the soapbar 1 from the cabinet 1.`

**Why replies are deterministic.** The server has no randomness after reset. To any command that is not admissible at that moment it replies "Nothing happens.", so every failing action in the ALFWorld studies has that one reply. Every item is replayed from reset by its saved action prefix on a separate environment id (`wf/env/replay.py`), which is why the histories reproduce exactly.

**How items are collected.** A Qwen3-8B ReAct agent (greedy, thinking off, shown the real task and the admissible actions, no predictions) played each game once from reset. The collector stopped each episode after 12 steps (`step_cap: 12` in `wf/config/rq12_transitions*.yaml`), so item histories hold at most 11 actions; this cap is a collection setting, not part of the preregistration (`notes/deviations.md`, 2026-11-20). At each state two generators propose alternative actions, and each is run on a fresh replay to get its true reply (`wf/data/transitions.py`): a broad generator (taking the goal object from various receptacles, moving it, heating, cooling and cleaning it, wrong-object versions of the agent's action, and three random admissible actions as likely successes) at every second step and at any step where the agent's own action failed, and an exhaustive generator covering every failure decided by contents, open state or holding at the agent's receptacle. The schedule matters for state-level analyses: odd-step states get the broad candidates only when the agent's own action failed, so state-level rates use even steps (`notes/exploratory_results.md`, section 7).

**When a failure counts as determined.** The function `_failure_determined` reads the history for the agent's location ("You arrive at …"), what it holds ("You pick up …"), which receptacles are open or closed, and each receptacle's last-seen contents. The rules, verbatim in substance from `notes/rq2_results.md`: `take X from R` fails if the agent is known to hold something, is known to be elsewhere, R is known to be closed, or R's contents were seen without X; `move`, `heat`, `cool` and `clean` fail if the agent is known not to hold X or to be away from R; `open R` fails if the agent is away or R is known to be open (and `close` symmetrically); `go to R` fails if the agent is already at R or R is not in the room. A **determined success** is a working action whose reply reveals nothing new.

**Success twins.** The two-candidate readout (Section 4) needs a success reply for a failing action. It is never hand-written. `wf/data/candidates.py` builds a short plan from reset (go to the object's receptacle, open it, take the object, carry it to the action's receptacle), replays it, and executes the action; the real server's reply is the **twin**. Twins of `open` and `go to`, whose replies list contents, are dropped once the agent has changed that receptacle (`twin_contradicts_state`).

**Verification.** Every collected item was replayed on CPU nodes (`wf/tools/verify_items.py`): history and true outcome must reproduce exactly, the twin must be what the server prints when the action works, contents-decided failures are checked against Word2World's hidden state (an independent source), and no item the rules call a sure failure may have succeeded. RQ2 scoring refuses a collection shard without a full pass on its exact file (`notes/testing.md`).

**Shared spots.** Some receptacles share a spot: from `cabinet 1` the agent can act on `cabinet 2`, so "the agent is elsewhere" proves nothing there. Such items are flagged `shared_location` and left out. On the confirmatory data no item was excluded by this rule alone (`notes/rq2_results.md`, "Shared-spot sensitivity").

**The RQ2 population.** Of 12,987 collected items, 4,666 from 1,033 states in 125 of the 134 games form the preregistered population: determined failures whose action names the goal object, whose history does not act on the goal or swap object (`history_goal_agnostic`), with a twin and a swapped goal, not at a shared spot. The exclusion flow, stage by stage, is the table in `notes/rq2_results.md`, "Exclusion flow". The 292 determined successes used for working actions are built differently (most do not act on the goal object).

### 2.2 TextCraft

**What it is.** TextCraft is a Minecraft-style crafting game, from AgentGym at commit `3ef9235…`, vendored in `wf/textcraft/upstream/` and run on a frozen recipe database whose hash is pinned (`wf/textcraft/freeze.py` rebuilds it; despite its name this file is not the design freeze). The wrapper is `wf/textcraft/world.py`.

**State, action and reply.** The design note states it plainly (`notes/textcraft_design.md` §2): "The environment's only state is the inventory, a map from item to count. It starts empty, and only the agent's own actions change it." Actions are `get N X`, `craft N Y using …` and `inventory`. Replies are `Got N X`, `Crafted K minecraft:Y_`, `Could not find enough items to craft minecraft:Y_` (any input short), `Could not find X` (a get of something craftable or unknown) and `Inventory: …`.

The golden TextCraft item (`tc-p001-1e56f799a78e`, stored in `wf/textcraft/data/chat_golden.json`) has this query turn, rendered by `wf/textcraft/prompts.py`, `messages`:

```
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

The agent holds 2 melon slices and needs 9, so the true reply is `Could not find enough items to craft minecraft:melon`; the success candidate is `Crafted 1 minecraft:melon`. Both candidates name the item, which matters for Section 4.

**Why replies are deterministic.** Replies depend only on the inventory and the recipe database. The project never calls upstream's `reset(data_idx)`, which samples distractor recipes with a hash-seeded random draw; every game is created with `reset(commands=<our list>, goal=<goal>)`. Verification replays every item under both goals of its pair to show the reply does not depend on which goal is set.

**How items are built.** Goals are paired (`wf/textcraft/build.py`, `all_pairs`, `compatible`) so the two goals have disjoint craftable sets and equal crafting depth, and the prompt shows the union of both recipe trees, shuffled. Histories come from a goal-blind scripted walk (`walk`) that "never fails an action, never crafts a goal and never names a goal"; every prefix is a history state. Test actions are the goal itself and its direct ingredients. Failures are classed by the short input's peak holding (`classify`): **used up** (once held in full, then consumed), **too few** (obtained, never enough) and **never obtained**. Main items come in **same-history couples**: two items with the same pair, history, class and role, one acting in each goal's tree, so on average the history favours neither goal.

**Determined failures and verification.** A text-only checker, `wf/textcraft/checker.py` (`decide`), imports no environment code (a test enforces this) and decides from the history and recipe lines alone whether the action must fail. `wf/textcraft/verify.py` (`replay_problems`) checks that each history replays, does not depend on which goal is real, that the reply and twin reproduce, and that aligned, swapped and neutral prompts differ only in the goal line. A blind checker written from the note alone agreed on 3,200 of 3,200 items; scoring refuses unverified items (`wf/textcraft/score.py`, `verified_files`). The frozen set has 3,200 items: 1,500 main (750 couples, 229 goal pairs, 500 per class), 600 "usefulness" items (later demoted to descriptive), 600 successes, 400 failed `get` items and 100 empty-history items (`notes/textcraft_design.md` §12).

### 2.3 τ-bench retail

**What it is.** τ-bench retail is a customer-service backend: a database of users, orders and products, and tools the agent calls (`get_order_details`, `exchange_delivered_order_items`, `modify_pending_order_items`, and so on). The project uses tau2-bench retail at a pinned commit, ported without dependencies to `wf/tau/retail.py`, which keeps upstream's method names, the order of precondition checks, the error strings and the JSON serialisation. `RetailEnv.call(name, arguments)` returns the tool's JSON output, or on an exception `f"Error: {e}"`, exactly as upstream does.

**State, action and reply.** The state is the database. An item's history contains only tool calls and their outputs: `get_user_details`, `get_order_details` for the target order and a comparison order, and for one class `get_product_details`. A history call is rendered as

```
Call: {"name": "get_order_details", "arguments": {"order_id": "#W4316152"}}
```

followed by `Output:` and the JSON. An example produced by the item builder (`wf/tau/items.py`, `build_generated`) from the pinned database asks to exchange a tea kettle for a variant the history shows as `"available": false`. The failing call is

```
{"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W4316152", "item_ids": ["7292993796"], "new_item_ids": ["7274158061"], "payment_method_id": "gift_card_7245904"}}
```

and its reply is `Error: New item 7274158061 not found or available`. The success twin is the same call with only that one check bypassed (`call_bypassed`); it returns the updated order JSON, beginning `{"order_id": "#W4316152", "user_id": "aarav_anderson_8794", ...`.

**Why replies are deterministic.** From `notes/tau_design.md` §2: "The tools are pure functions of the database and the arguments. A failed write raises before it mutates anything." The port's call takes only a tool name and arguments, no task. An upstream-equivalence script (`scripts/tau_upstream_equivalence.py`) found identical outputs and databases on 10,000 random calls and on every item.

**Item classes and determined failures.** Every item is a write on an order that fails exactly one precondition, and that precondition is visible verbatim in the history: the order's status breaks the tool's status rule (`status`), the requested variant shows `available: false` (`unavailable`), a return goes to a payment method other than the original one (`wrong_return_pm`), or a payment change uses a gift card with too low a balance (`gc_balance`). A text-only checker, `wf/tau/checker.py` (`check`, `error_class`), must return FAIL with the same class and identical error text on every item; a blind second checker agreed on 3,391 of 3,391.

**The witness.** τ-bench adds a control ALFWorld does not have. For each item, a **witness** is a single write call that would achieve the aligned task from the item's state, found by trying candidate calls in a fixed order on a copy of the state and keeping the first that succeeds (`wf/tau/items.py`, `witness_calls`, `achievable`). In the example the witness is the same exchange to an available variant. The main population is restricted to items whose aligned task is achievable, so "the agent chose a wrong tool or a wrong argument for a task that a right one would achieve" (`notes/tau_design.md` §3.6). The witness is scored like the failing call, which allows the failure-specificity contrast of Section 5.

**Verification and counts.** `wf/tau/verify.py` checks byte-for-byte regeneration of history, failure reply, twin and witness output, that prompt variants differ only in the task line, and a 100-item replay in a fresh process; `tau_verify` passed with 0 of 3,391 items showing a problem, and scoring refuses item files that are not verified or whose hash is not pinned. The **primary** population is 856 status-matched items from 175 users (`notes/tau_metrics.json`, `items`). Users are the clustering unit.

### 2.4 ScienceWorld: paused

ScienceWorld is not in this repository. The README records that a pre-specified feasibility probe found almost no determined failures on the task's own object, and that 99.6% of the few it found were one generic parser reply. A study built on such items would test whether the model predicts a parser error, not whether the task changes a predicted state change, so it cannot replicate the main test. A set of control items on unrelated objects exists outside the repository. The ScienceWorld fallback for RQ3 in the original plan was never triggered (`notes/deviations.md`, "Closed 2026-11-20").

### 2.5 The three environments side by side

The table summarizes what differs. The main consequence for reading results is in the last two rows: in ALFWorld the success and failure replies differ in length and content, which affects the two-candidate readout (Section 4.8), while in TextCraft both replies name the item and in τ-bench the replies begin with distinct single tokens.

| | ALFWorld | TextCraft | τ-bench retail |
|---|---|---|---|
| State | Room layout, agent position, held object, open/closed receptacles | Inventory (item counts); recipes are fixed | Retail database (users, orders, products) |
| Action | Text command (`take soapbar 1 from cabinet 1`) | `get`, `craft … using …`, `inventory` | A JSON tool call |
| Failure reply | Always `Nothing happens.` | `Could not find enough items to craft minecraft:<item>` | A one-line `Error: …` string specific to the broken rule |
| Success reply | Action-specific sentence from the real server (the twin) | `Crafted N minecraft:<item>` | The JSON output of the call with one check bypassed |
| Items in the primary test | 4,666 determined failures, 125 games | 1,500 main failures in 750 couples, 229 goal pairs | 856 status-matched failures, 175 users |
| Clustering unit | Game | Goal pair | User |
| First tokens of the two replies | "You" against "Nothing" | "Craft" against "Could" | `{"` against `Error` |

---

## 3. The manipulation: task lines

Every study changes one thing between paired prompts: the **task line**, the sentence that states the agent's goal. Everything else in the prompt is byte-identical; `wf/scoring/prompts.py`, `_replace_task_line`, raises an error unless exactly one task line is replaced. Each version of the prompt is called a **condition** or **arm** (the notes use both words; "arm" is more common in later studies).

### 3.1 The four basic conditions

Take the golden ALFWorld item above (action `take soapbar 1 from cabinet 1`, cabinet 1 closed). The conditions and their task lines are:

- **Aligned:** the episode's real task, which names the action's object. `Your task is to: put two soapbar in garbagecan.`
- **Swapped:** the same template with another object substituted for the goal object. `Your task is to: put two spraybottle in garbagecan.` The swap object is drawn per item (`wf/data/goals.py`, `pick_swap`) among object types in the room that are not the goal type, are not named in the action, are not acted on in the history, have no instance at the action's receptacle at reset, and could do the task by the game's own rules. This is the RQ2 contrast: only the object name differs.
- **Neutral:** a concrete task that names no object. `Your task is to: explore the room.` (`wf/rq1.py`, `ABSENT_LINE`; the arm is called `absent` or `yn_absent` in code.)
- **Task-blind:** no task at all. The aligned prompt with the task sentence, and the line break before it, deleted, every other byte kept (`wf/decisionsim.py`, `blind_history` and `blind_messages`). For the golden item the history would begin with the room description and go straight to `> go to cabinet 1`. The few-shot examples keep their own task lines; a variant that strips those too (`yn_blind_all`, `strip_example_tasks`) is scored as a prompt-format check.

Task-blind is used only in the decision-simulation studies (Stages 1a and 1b) and in Option C. TextCraft and τ-bench have no task-blind condition: TextCraft's design chose the neutral line because "dropping the line instead would change the format, as well as removing the goal" (`notes/textcraft_design.md` §5), and τ-bench has aligned, swapped, neutral and a mention pair. The τ-bench authorization experiment has an arm called "blind", but it is a different thing: a backend description stating that the task never changes the output, not a prompt without a task (Section 8).

### 3.2 Why swapped and task-blind are the clean comparisons: the relevance penalty

It might seem that the cleanest test would compare the real task with a neutral one. It is not, and the reason was found in the data. `notes/exploratory_results.md` (section 7) reports that any concrete task lowers predicted success on actions unrelated to it, relative to the neutral line. On determined successes that do not involve the task's object, Qwen3-8B's yes/no log-odds for taking another object fall by 13.30 nats under the aligned task relative to neutral, but by only 0.43 relative to the swapped task. The note concludes: "A neutral comparison therefore mixes two things: the object-naming effect that RQ2 isolates (aligned against swapped), and a penalty that any concrete task puts on task-irrelevant actions. Only the aligned-against-swapped contrast isolates the first." This penalty is what the project calls the **relevance penalty**.

The aligned and swapped lines are both concrete tasks with the same template, so the penalty for having a concrete task is present in both and cancels in their difference. What remains is the effect of which object the task names. The task-blind prompt is clean in a different sense: it is the realistic alternative a system designer faces (show the verifier the task, or do not), and it differs from the aligned prompt in nothing but the presence of the task sentence. The neutral line is neither: it is a concrete task, but about nothing, and models treat it differently from no task. In Stage 1a, for instance, Qwen3-8B's neutral and task-blind verifiers both accept about 90% of working first moves while its aligned verifier accepts 7% (`notes/decisionsim_dx.md` §3). Section 5 shows how the neutral line is still used, in a decomposition.

### 3.3 The object-free (described) goal and other ALFWorld variants

**Described goal** (`wf/rq1x.py`, `describe_line`, `DESCRIPTIONS`). Every object name in the task is replaced by a fixed description containing no ALFWorld object name; the action still names the object. For the golden item:

- aligned, described: `Your task is to: put two small solid block you lather when washing in garbagecan.`
- swapped, described: `Your task is to: put two trigger-operated mister of cleaning liquid in garbagecan.`

A naming check asks each model which household object a description describes (`CHECK_PROMPT`), and results are also reported on the items where both descriptions passed it. This variant asks whether the effect needs the object's name to be repeated, or only its meaning.

**Extra-mention lines** (`wf/rq1.py`, `goal_line`, `mention_sentence`). The swapped task plus a sentence naming the action's object, or the aligned task plus one naming the swap object:

- swapped plus: `Your task is to: put two spraybottle in garbagecan. There is a soapbar somewhere in the room.`
- aligned plus: `Your task is to: put two soapbar in garbagecan. There is a spraybottle somewhere in the room.`

These let the goal effect split exactly into a "mention" part and a "goal role" part (`notes/rq1_results.md`, "Extra-mention control").

Other ALFWorld variants appear in single studies and are described where they are used: an "ignore the goal" note appended to the prompt ("(The agent's goal does not change what happens. Predict the observation from the history and the action alone.)", `wf/rq1.py`, `IGNORE_NOTE`); a full-state prompt that prepends the game's hidden starting layout (`HIDDEN_INTRO` in `wf/scoring/prompts.py`); a counterfactual aligned task whose destination is replaced by another receptacle type (`wf/rq1v.py`, `cf_destination`); and the dissociation sentences "There is a mug somewhere in the room.", "The mug 1 is at cabinet 3." and "There is no mug in this room." (`wf/rq1d.py`).

### 3.4 Task lines in TextCraft and τ-bench

For the golden TextCraft item (goal melon, partner stripped birch wood): aligned `Goal: craft melon.`, swapped `Goal: craft stripped birch wood.`, neutral `Goal: explore the crafting commands.` In TextCraft the swapped goal is the other goal of the pair, whose recipe tree is also on screen.

For the τ-bench example item (`wf/tau/items.py`, `TEMPLATES`, `NEUTRAL`, `MENTION_PRODUCT`): aligned `The user wants to swap the Tea Kettle in order #W4316152 for a different option of the same product.`; swapped `The user wants to swap the Bookshelf in order #W3470184 for a different option of the same product.`; neutral `The user wants help with their account.`; and a **mention pair** that names an order without asking for a write, `The user has a question about the Tea Kettle in order #W4316152.` and its swapped counterpart. The mention pair separates the effect of naming an order from the effect of requesting a change to it (Section 5).

### 3.5 The mechanism battery's arms

The mechanism battery (built on branch `mechanism2`, now merged into `main`; design `notes/mechanism2_design.md`, code `wf/mech2.py`) adds task lines built to separate candidate explanations of the effect. The main arms are listed below (the full list is in `wf/mech2.py`); the examples are rendered from the frozen templates on a test fixture (`tests/test_rq1.py`, `_item`: action `take saltshaker 1 from cabinet 1`, real task "put some saltshaker on cabinet", swap object mug). Each is the full task line shown in place of the real one.

| Arm | Task line |
|---|---|
| aligned | Your task is to: put some saltshaker on cabinet. |
| swapped | Your task is to: put some mug on cabinet. |
| desire_keep_sw | Your task is to: put some mug on cabinet. Make sure that the saltshaker is not taken; leave it where it is. |
| desire_check | Your task is to: find out whether a saltshaker can be taken from the cabinet 1; the user hopes it cannot. |
| desire_check_can | the same, ending "the user hopes it can." |
| mention_ignore | Your task is to: put some mug on cabinet. Ignore the saltshaker. |
| described | Your task is to: put some container with holes for sprinkling white seasoning on cabinet. |
| done | Your task was to: put some saltshaker on cabinet. It is already done; nothing is left to do. |
| cancelled | Your task was to: put some saltshaker on cabinet. The user has cancelled it; nothing is left to do. |
| negated (take actions) | Your task is to: put some mug on cabinet. The user says there is no saltshaker in the cabinet 1. |
| affirmed (take actions) | Your task is to: put some mug on cabinet. The user says there is a saltshaker in the cabinet 1. |
| placebo_negated / placebo_affirmed | as negated and affirmed, but about the swap object (mug) |
| aligned_start, swapped_start (step 0) | the aligned and swapped lines moved to the top of the history |

Each arm targets a prediction of one account. For example, if the effect comes from the task being evidence that its object is present and the action will work, then "the user says there is no saltshaker in the cabinet 1" should push predictions toward failure; if it comes from mere salience of the named object, negation and affirmation should act alike. The contrasts P1 to P8 built from these arms are described in Sections 5 and 6.

### 3.6 TextCraft round 2: two-item and parent goals

TextCraft round 2 (branch `textcraft-r2`, design `notes/textcraft_r2_design.md`) adds goal lines that separate naming from completion:

- **Two-item goal:** both goals of the pair in one line, sorted. A real record with goal iron nugget and partner flower pot gets `Goal: craft flower pot and iron nugget.` The action's item is named, but crafting it no longer completes the task by itself.
- **Parent goal:** a non-goal item whose recipe uses the goal item directly. For the record `tc-p000-51fd5b09ab3c` (goal iron nugget), the parent is chain, the recipe list gains the line `craft 1 chain using 1 iron ingot, 2 iron nugget`, and the parent task line is `Goal: craft chain.` The action's item is then an ingredient of the task rather than the task itself.
- **Padding:** 1 or 8 copies of `inventory` / `Inventory: You are not carrying anything.` placed before the first step, to test whether the timing of the goal relative to the action matters.

Round 2 also uses a **clean prompt** that drops every sentence addressed to the agent (`CLEAN_SYSTEM`) and an anchored yes/no question (Section 4.4).

---

## 4. Readouts: how a prediction is turned into a number

A **readout** is the procedure that turns the model's response to one prompt into one number. The project uses several, because each has a known weakness and the weaknesses differ. This section describes each: the exact prompt ending and candidates, the formula, what it measures, what can contaminate it, and what the project has found about that contamination.

All teacher-forced readouts go through one class, `LogoddsScorer` in `wf/scoring/logodds.py`. It loads the model at a pinned Hugging Face revision (`wf/constants.py`, `REVISIONS`; an unpinned model is refused), renders the prompt with the model's own chat template with thinking off (`template_settings`, `_encode`), and computes log-probabilities in float32 from the model's bfloat16 logits. For Qwen3, thinking off means the template inserts an empty `<think></think>` block right before the scored answer; this is Qwen3's documented thinking-off format, and it roughly doubles Qwen3-8B's effects in nats without changing their standardized size (Section 4.8, "Two-candidate nats are not comparable across models", and `notes/qwen_effect_audit_pipeline.md` §7; added 2026-11-23).

### 4.1 The two-candidate log-odds L

**Prompt ending and candidates.** The prompt is the world-model prompt of Section 1, ending with the query turn

```
Action: take soapbar 1 from cabinet 1

Next observation:
```

followed by the chat template's opening of the assistant turn. Two candidate replies are scored: the success twin, written $o^{+}$ (`You pick up the soapbar 1 from the cabinet 1.`), and the failure reply, written $o^{-}$ (`Nothing happens.`).

**Teacher forcing.** To teacher-force a candidate is to feed it to the model token by token after the prompt and add up the log-probability the model assigns to each of its tokens, given the prompt and the candidate's earlier tokens. If a candidate has tokens $o_1, \dots, o_n$ and the prompt is $x$, then

$$
\log P(o \mid x) = \sum_{k=1}^{n} \log P(o_k \mid x, o_1, \dots, o_{k-1}).
$$

No text is generated. `LogoddsScorer._teacher_force_logprob` does this; each candidate is tokenized alone, without special tokens (`_encode_candidate`), and the tokenization gates of Section 7 check that this matches how the template would render it as a reply.

**The formula.** The two-candidate log-odds is the difference of the two totals (`LogoddsScorer.score_pair`):

$$
L = \log P(o^{+} \mid x) - \log P(o^{-} \mid x).
$$

It is measured in nats (natural-log units). A positive $L$ means the model finds the exact success sentence more probable than the exact failure sentence after this prompt; $L = 0$ means it finds them equally probable.

**What it measures, and its main contamination: the wording penalty.** $L$ compares two particular sentences, not "success" and "failure". The decision log works this out (`notes/deviations.md`, "Decision 5 in detail"). Write $q$ for the model's probability that the action succeeds, in any wording. A failure in ALFWorld always reads "Nothing happens.", so its probability is close to $1 - q$. A success can be worded in many ways, and the twin is only one of them; write $c = -\log P(o^{+} \mid \text{success}) \ge 0$ for the cost of guessing its exact wording. Then

$$
L \approx \operatorname{logit}(q) - c .
$$

The **wording penalty** $c$ grows with every detail of the success sentence the model cannot guess: which instance number, which receptacle, whether contents are listed. It means the sign of $L$ is not a clean "predicts success" reading. On Tier 0's 100 train items, every case where the sign of $L$ disagreed with the model's own written prediction was a `go to` success whose text listed unseen contents.

The penalty matters for the goal effect because a task can change $c$ as well as $q$. A task that names the soap bar can make the words "the soapbar 1" more probable after "You pick up", without changing the model's belief that the action fails. Section 4.8 reports how large this part is.

**A validity check that one model failed.** Pilot A checked whether the sign of $L$ agrees with what the model writes when asked to predict freely. On 1,138 determined failures Qwen3-8B agreed on 91%, and Llama-3.1-8B on 65% (741 of 1,138), below the 90% line set in advance; Llama stayed co-primary with the failure disclosed (`notes/deviations.md`, "Closed 2026-11-20"). This is one reason later cross-family comparisons do not use $L$ (Section 4.9).

**Where it is the primary readout.** The preregistered RQ2 test, TextCraft round 1's primary test, and TextCraft round 2's contrasts. The mechanism battery reads it as one of its two formats.

### 4.2 The two-candidate score s = σ(L), and why it is not a probability of success

The logistic of $L$,

$$
s = \sigma(L) = \frac{1}{1 + e^{-L}},
$$

is logged in code and in older notes as `p_success`, "P(success)" or, for its shift, "Δp". It is the probability of the success sentence relative to the two-sentence set $\{o^{+}, o^{-}\}$. By the decomposition above it is pulled down by the wording penalty, so it understates the model's belief in success by an unknown, item-dependent amount. The README and `notes/rq2_results.md` both warn that $s$ "is not the model's probability that the action succeeds".

It is still useful as a bounded scale for practical size. A shift of 6 nats is enormous at $L = 0$ and negligible at $L = -20$. Qwen3-8B's mean $s$ on the RQ2 items is 0.075 under the aligned task and 0.017 under the swapped task; Llama-3.1-8B's is 0.408 and 0.244 (`notes/rq2_results.md`). The shift is 5.9 points for Qwen and 16.4 for Llama, the reverse of the ranking in nats (6.12 and 1.65), because Qwen's scores sit far below zero where $\sigma$ is flat.

### 4.3 The first-token readout

**Prompt and candidates.** The same prompt and candidates as the two-candidate score, but only each candidate's first token counts. In ALFWorld that is "You" against "Nothing" on every one of the 4,666 failing items, for every model and prompt (`notes/rq2_results.md`, "The first-token term"). In TextCraft it is "Craft" against "Could"; in τ-bench it is `{"` against `Error`.

**Formula.** With $t^{+}_1$ and $t^{-}_1$ the first tokens,

$$
L_{\text{first}} = \log P(t^{+}_1 \mid x) - \log P(t^{-}_1 \mid x).
$$

It is computed by the same pass as $L$ and stored as `L_first` on `ScoreResult`. It equals the logit of $q_{\text{first}} = P(t^{+}_1)/(P(t^{+}_1) + P(t^{-}_1))$, the model's preference between the two opening words. The difference $L_{\text{first}} - L$ is stored as `c_hat`, an estimate of the wording penalty.

**What it measures.** The first token is where the two replies diverge, before any object name, so it is "the closest thing either format has to a pure success-or-failure decision" (`notes/textcraft_vs_alfworld.md` §1). It carries no wording penalty from the tail of the sentence.

**What can contaminate it.** Two things. The first is coverage: the two tokens may not hold most of the model's next-token probability. The scorer stores their total mass as `first_mass`. For Qwen3-8B the median coverage is above 0.9999; for Llama-3.1-8B it is 0.85 under the aligned task and 0.80 under the swapped task, and below one half on 0.8% and 7.2% of items, so "for a typical Llama item 15–20% of the first-token probability goes to other words" (`notes/rq2_results.md`). The second is that the first word is not always decisive: some of Llama's failure sentences also start with "You", and among the 292 ALFWorld determined successes, 56 twins begin with "The", "There" or "This".

**Where it is primary.** τ-bench retail, whose success replies are long JSON objects with a heavy wording penalty, uses the first token as its primary readout. Cross-environment comparisons also lean on it, because it is the one teacher-forced scale on which the three environments' replies are comparable.

**Grid resolution: scores come in steps of 1/64 nat.** Models are run in bfloat16, so the logits themselves are rounded. `notes/exploratory_results.md` (section 7) found that every logged yes/no $L$ of both reference models, 14,874 per model, lies within 0.001 of a multiple of 1/64, "the bfloat16 resolution of the logit difference"; under the aligned task Qwen's $L$ takes only 212 distinct values and Llama's 118. Exact ties are therefore common. Analyses that compare scores across candidates, such as picking the highest-scored action, round to 1/64 first so that float32 noise does not break ties differently in different conditions; the decision simulation does this in `wf/decisionsim.py`, `quantize` (`GRID = 64`). Means over thousands of items are unaffected.

### 4.4 The yes/no question

**Construction.** The prompt is the ordinary prompt with its last words, `Next observation:`, replaced by a question, and the readout compares the single tokens "Yes" and "No" at the next position (`LogoddsScorer.score_one_token_pair`, which refuses candidates that are not single tokens):

$$
L_{\text{yn}} = \log P(\text{Yes} \mid x) - \log P(\text{No} \mid x).
$$

Because both candidates are one token and identical across items, models and environments, this readout has no wording penalty and no tail, and it means the same thing for every model whose tokenizer has single "Yes" and "No" tokens (checked by the tokenization gates). Its risk is the opposite one: a question about "success" can be read as a question about the task, not the environment. The wording of the question is therefore the critical design choice, and it differs between studies.

**ALFWorld's anchored wording** (`wf/rq1v.py`, `YN_QUESTION`, used by every ALFWorld yes/no study, the decision simulation and the families):

```
Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens.
Answer:
```

The question is **anchored**: it names the environment's actual failure reply, so "No" means "the environment will print Nothing happens.", not "this does not help my task". The golden item's yes/no prompt thus ends `Action: take soapbar 1 from cabinet 1` followed by a blank line and this question.

**TextCraft round 1's unanchored wording** (`wf/textcraft/prompts.py`, `YN_QUESTION`):

```
Question: Will this action succeed? Answer Yes or No.
Answer:
```

The design note described it as "as in the ALFWorld validation runs", but the wording differs (`notes/deviations.md`, 2026-11-22: TextCraft's question "is not anchored to the environment's reply, unlike ALFWorld's"). The consequence was found after scoring (`notes/textcraft_qwen_robustness.md` §9): "The TextCraft wording can be read as 'will this action succeed at my task?'". Two thirds of Qwen3-8B's TextCraft yes/no effect on failures (+1.95 of +2.92 nats) lies beyond the shift in its own predicted observation. Most tellingly, "Under the swapped line Qwen predicts that crafting the other task's final item will produce 'Crafted 1 minecraft:…', yet answers 'No' to whether it will succeed in 72% of such working crafts. It does so in 1% under the neutral line." The model is answering whether the action succeeds at the stated task. Llama-3.1-8B shows no such task-success component. The comparison of TextCraft's yes/no effect with ALFWorld's (2.92 against 2.80) was withdrawn in `notes/textcraft_results.md` ("Corrected 2026-11-21"); the older `notes/textcraft_vs_alfworld.md` (2026-11-20) still reports it uncorrected.

**TextCraft round 2's anchored wording** (branch `textcraft-r2`, `r2.YN_QUESTION`):

```
Question: will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that it failed ("Could not find ...")? Answer Yes if it succeeds and No if it fails.
Answer:
```

**τ-bench's anchored wording** (`wf/tau/prompts.py`, `YN_QUESTION`; it replaced a "will this tool call succeed" wording in fix round 3, before scoring):

```
Question: will the backend execute this call without returning an error? Answer Yes if its output is JSON and No if its output is a line starting with "Error: ".
Answer:
```

**Anchoring reduces but does not remove the task judgement.** In τ-bench, about 3.9 of Qwen3-8B's 7.68-nat yes/no effect lies beyond its own predicted output, and among calls whose first token confidently predicts JSON, Qwen answers No 90–93% of the time under the swapped line, 51–67% under neutral and 17–32% under aligned (`notes/tau_dx.md`). In ALFWorld, `notes/decisionsim_dx.md` notes that the anchored question "is better anchored than TextCraft's, but at step 0 the anchor evidently does not hold", where Qwen rejects working first moves under any concrete task (labelled there as an interpretation). The practical rule in the notes: a yes/no effect is a goal effect on the model's answer, which may include a judgement about the task; a first-token effect is closer to a prediction of the reply.

**A different yes/no format.** The older mechanism diagnostic (`wf/mech.py`) asks factual questions ("Before this action, is the agent at the cabinet 1?") with the ending `Question: {q} Answer Yes or No.\nAnswer:`. Those readouts are not comparable with the success question above.

### 4.5 Written predictions (RQ1) and the parser

**What is generated.** For RQ1 the model writes its own prediction: greedy decoding, at most 64 new tokens, after the same world-model prompt (`LogoddsScorer.free_generate`, called from `wf/rq1.py`). No candidates are imposed.

**The parser.** `wf/scoring/parser.py`, `parse_prediction`, classes each written prediction into one of ten categories. The ones that matter: `noop` ("Nothing happens.") and `noop_paraphrase` (failure in other words, such as "You can't …", "There is no …", "is already clean") count as predicting failure; `state_change`, `reveal_*` and `success_paraphrase` count as predicting the requested success; `process_only` (an effect with no stated result) and `different_effect` (an effect of another action or on another object) are "partial effects", counted on the success side by the primary rule and reported both ways; `unparseable` commits to nothing. The primary definition is "failure if and only if no effect is predicted" (`NO_EFFECT_CLASSES`).

**The measure A.** RQ1's asymmetry is the false-success rate minus the false-failure rate:

$$
A = \Pr(\text{predicts an effect} \mid \text{action fails}) - \Pr(\text{predicts no effect} \mid \text{action works}),
$$

computed on determined failures and determined successes (`wf/analysis/stats.py`, `asymmetry_A`). A positive $A$ is a lean toward predicting success.

**Contamination and status.** A parser decides the result here, so it was audited against a blind rater in six rounds (`wf/tools/parser_audit.py`). The rule fixed before the audits required at least 95% agreement on no-effect against effect and no row read in the opposite direction. Rounds five and six passed the agreement bar (95.5% and 96.6%) but each had rows read in the opposite direction, and the audits ended on 2026-11-19 without a pass. The notes use the final parser version `3c25d40cc71529e2`, which reads 110 of the 68,340 predictions differently from the first analysis; no verdict changes (`notes/rq1_results.md`, "Parser status"). Written predictions also feed the "written success shift" secondary measure of RQ2.

### 4.6 Item-level measures: shares, flips and changed answers

Means in nats say how far scores move, not how many answers change. Several item-level counts express practical size. Their definitions differ in small ways that matter.

- **Share of failures pushed toward Yes** (families wave A and B, `wf/families.py`, `_share_toward_yes`): the share of determined failures whose yes/no log-odds is strictly higher under the aligned task than under the swapped task. Ties count as not moved; with no effect the share is 0.5. It counts direction only, so it is unaffected by how extreme a model's scores are, which is why it is the primary cross-family metric. Only failures scored under all three yes/no arms are included. *(revised 2026-11-25)* Option C's step-0 check found Qwen3-8B partly reading the anchored question as asking about task success ("question-specific (supports the task-success reading)"; Section 4.4), so the share may include a task-progress judgement; the two-candidate effect, which does not use the question, is positive for every family model.
- **One-direction flip** (`flips_to_success`): the full two-candidate score is below zero under the swapped task and above zero under the aligned task.
- **First-token sign change** (`decision_flip`, `wf/analysis/stats.py`, `decision_flip_share`): the first-token log-odds changes sign between the two tasks, in either direction. For the same items this is larger than the one-direction flip; Qwen3-8B has 9.7% sign changes against 5.9% one-direction flips (`notes/exploratory_results.md` §6).
- **No-to-Yes flip** (`wf/families.py`, `argmax_answers`, `flip_no_to_yes`): the yes/no answer is No (log-odds at or below zero) under the swapped task and Yes under the aligned task, divided by all failures, not only those answered No under swapped. This is what the notes mean by "answers changed" or "failures flipped from No to Yes".
- **Share answered Yes**: the share of items whose yes/no log-odds is above zero.
- **Yes-share** (mean σ): the mean, over items, of the logistic of the yes/no log-odds: a mean probability of Yes against No. The notes define it in `notes/exploratory_results.md` (opening "Measures") and warn that it is "not the share of items on which Yes scores above No". The two differ when items sit near 0 or 1. For Qwen3-32B on failures under the aligned task, the share answered Yes is 11.6% and the mean σ is 15.0%. One code key, `yes_share_failures`, holds the mean σ in `wf/sizes.py`, `model_block`, and the argmax share in `wf/families.py`, `argmax_answers`.

**Calibration decides how log-odds turn into changed answers.** The families results show the same direction of effect in every competent model, but very different numbers of changed answers. Qwen3-8B moves 88.7% of failures toward Yes yet changes its answer from No to Yes on only 0.17% (rounded to 0.2% in the notes), because it is so confident that failures fail: under the neutral task the gap between its mean yes/no score on successes and on failures is 29.6 nats. Mistral-7B-Instruct-v0.3 moves a similar share (87.9%) but changes 37.5% of answers; it answers Yes on 44.9% of failures under the swapped task and 80.3% under the aligned task, and its success-failure gap is 6.2 nats (`notes/families_metrics.json`; `notes/families_results.md`, "Practical size depends on calibration"). The cross-model summary puts the lesson this way: "A shift in log-odds is converted into changed decisions by the model's calibration, so a ranking by nats alone misstates where the effect matters most."

### 4.7 Competence

A goal effect in a model that cannot do the task at all says little. Each study therefore checks **competence**: whether the model tells working actions from failing ones when no goal is in play.

**Neutral AUC.** The basic measure is the area under the ROC curve (AUC) between determined successes and determined failures, computed on scores under the neutral task line. It is the probability that a randomly chosen working action scores higher than a randomly chosen failing one, with ties counted as half:

$$
\text{AUC} = \Pr(L_{\text{success}} > L_{\text{failure}}) + \tfrac{1}{2}\Pr(L_{\text{success}} = L_{\text{failure}}).
$$

An AUC of 0.5 is chance and 1.0 is perfect separation. The neutral line is used so that the goal effect does not enter the competence measure.

**The competence rules.** They are fixed per study, before data:

- **Families (wave A and B), and the gate reported in TextCraft round 2** (`wf/families.py`, `COMPETENCE`, `competence_gate`; `notes/families_design.md`, second amendment): on yes/no scores under the neutral task, AUC at least 0.70; Yes on at most 95% of failures; Yes on at least 5% of determined successes; median total probability of "Yes" plus "No" at least 0.5. A model that fails is "not interpretable", never counted as a null result. The first version required Yes on 5–95% of failures, which would have failed Qwen3-8B (Yes on 0.15% of failures) for being right; the amendment, made before any wave A score was read, moved the lower bound to successes.
- **TextCraft round 1** (`notes/textcraft_design.md` §12 C3 and §13 D7; `wf/textcraft/analysis.py`, `competence`): two-candidate AUC under the neutral line, between the 600 successes and 2,100 failures, at least 0.6 with the lower end of its interval above 0.5. Qwen3-8B's was 0.868 and Llama-3.1-8B's 0.681; Llama's falls to 0.305 on "used up" items, which it ranks backwards.
- **τ-bench** (`notes/tau_design.md`, A4–A6): a discrimination check (the witness's neutral first-token score minus the failing call's, on the 447 items whose witness the checker can see, at least 1 nat with interval above zero) and a manipulation check, which asks whether the model reads the task line at all: the failing call's prompt ends with "Question: is this call on the order the user asked about? Answer Yes if it is and No if it is not." (`TARGET_QUESTION`), and the aligned-minus-swapped effect on that answer must be at least 1 nat with its interval above zero. The τ-bench families extension adds the families rule on first-token scores.
- **ALFWorld RQ2** had no competence gate; the Pilot A validity check of Section 4.1 played that role. The Qwen3 size series predates the gate; its two smallest models are judged post hoc as not competent because they answer Yes to almost every question.

### 4.8 What is known about contamination, in one place

**The success tail carries about 59% of ALFWorld's two-candidate effect.** $L$ splits exactly into a first-token term and the rest of each reply (`notes/textcraft_vs_alfworld.md` §1):

$$
L = \underbrace{\left[\log P(t^{+}_1) - \log P(t^{-}_1)\right]}_{\text{first token}} + \underbrace{\log P(o^{+}_{2:} \mid t^{+}_1)}_{\text{success tail}} - \underbrace{\log P(o^{-}_{2:} \mid t^{-}_1)}_{\text{failure tail}} .
$$

For Qwen3-8B on ALFWorld, 3.60 of the 6.12 nats (59%) sit in the success tail, that is, in the goal making the words after "You" ("pick up the soapbar 1 from the cabinet 1.") more probable; for Llama it is 0.96 of 1.65 (58%) (`notes/textcraft_vs_alfworld.md` §2). The failure tail contributes nothing ("Nothing" is always followed by " happens."). TextCraft's replies both name the item, so the tail cancels there (−0.03 nats). The ALFWorld two-candidate effect is therefore partly a naming effect on the wording of the success sentence, and the first-token effect (2.52 nats for Qwen, 0.68 for Llama) is the better reading of a change in predicted outcome.

**Two-candidate nats are not comparable across models.** An exploratory forensic audit of the stored scores (`notes/qwen_effect_audit_forensic.md`, 2026-11-22) finds that models spread their scores over very different ranges (across items, the SD of the swapped-task score is 7.8 nats for Qwen3-8B and 3.1 for Llama-3.1-8B), so the same nat means different things in different models and across Qwen3 sizes. Divided by the SD of each model's item-level shifts, Qwen3-8B's effect is 1.02 against Llama-3.1-8B's 0.85 on the two-candidate score, 0.75 against 0.68 on the first token (inside the other competent models' range of 0.31 to 0.81) and 0.88 against 1.12 on yes/no, and the Qwen3 size series is flat (0.90, 1.02, 0.96 and 0.86 for 4B to 32B) in the study's thinking-off format (in plain text, flat for 4B and 8B but not 14B; see below). For Qwen3-8B, 59% of the summed shift comes from items whose $L$ is below −10 nats under both tasks, where neither prompt's score could support a success prediction. The audit found no pipeline fault and changes no verdict. The second, independent audit (`notes/qwen_effect_audit_pipeline.md`, 2026-11-23) re-implemented the Qwen scoring on GPU: a scorer written from scratch reproduces the stored scores exactly on the same GPU type, and vLLM and float32 weights agree, so no bug produced the number. About half of the effect's size in nats depends on one prompt-format detail: with thinking off, Qwen3's chat template inserts an empty `<think></think>` block before the answer. *(Updated 2026-11-23, exploratory follow-up, `notes/qwen_effect_audit_pipeline.md` section 7.)* Rescored without the block on all 4,666 items, the effect is 3.31 nats against 6.12 (ratio 0.54), while the standardized effect (1.02 against 1.05) and the share of items moved toward success (87.9% against 88.7%) are unchanged: the block stretches the scale of Qwen's scores, not its goal sensitivity. So cross-model comparisons should lead with the standardized effect and the share of items moved, and quote 6.12 with this format sensitivity. The render without the block is Qwen's template with thinking on, where the two answers get under 0.1% of the next-token probability, so it is not a natural alternative format; plain text (no chat template, the answers keep at least 90%) is the comparison to rely on, and it gives the same halving (3.26 against 3.28 nats on 300 items). The step-0 concentration exists only with the block (10.99 nats at step 0 with it, 2.73 without). In plain text the standardized size series stays flat for 4B and 8B (1.05, 1.15) but not for 14B (0.72); 32B was not rescored. In every format tried the effect stays positive and 2 to 5 times Llama-3.1-8B's in nats.

**Yes/no answers carry a task judgement.** Strongly with an unanchored question (TextCraft round 1), and measurably even with an anchored one (τ-bench, ALFWorld at step 0). See Section 4.4.

**Calibration decides practical size.** The same direction of effect produces 0.2% changed answers in Qwen3-8B and 37.5% in Mistral-7B. See Section 4.6.

**Scores lie on a 1/64-nat grid**, which creates ties that matter for any argmax. See Section 4.3.

Two smaller facts belong here too. Individual scores repeat exactly only on the same GPU type; across GPU types they differ by noise of the size measured in `notes/deviations.md` ("Scores across GPU types", 2026-11-18, which supersedes a 2026-11-17 claim that only rounding differed). Each item's paired prompts are therefore scored in the same job. And log-probabilities are computed in float32 since 2026-11-18, because "a bfloat16 log_softmax rounds each log-probability to about 3 significant digits" (`notes/deviations.md`, "Scorer precision").

### 4.9 The readouts compared, and which is used for what

The table summarizes Sections 4.1 to 4.6.

| Readout | What it compares | Good for | Weaknesses | Where it is primary |
|---|---|---|---|---|
| Two-candidate log-odds | Exact success sentence against exact failure sentence, all tokens | Uses the environment's real replies; most sensitive | Wording penalty; goal can change the tail (59% of ALFWorld's effect); validity check failed for Llama | RQ2 (preregistered); TextCraft rounds 1 and 2; one of two mechanism-battery formats |
| Score s, the logistic of L | The same, on a 0–1 scale | Practical size on a bounded scale | Not a probability of success | Secondary measure (Δp) only |
| First token | Opening word of each reply | Closest to an outcome decision; no tail; comparable across environments | Low coverage for Llama; opening word not always decisive | τ-bench retail; cross-environment comparisons |
| Yes/no question | Single tokens Yes and No after a question | Same candidates for every model; no wording penalty; works for any model | Carries a task judgement, strongly if unanchored | Families (cross-family); decision simulation; Option C; second mechanism-battery format; interpretability pilot |
| Written prediction | The model's own free text, parsed | Behavioural; no imposed candidates | Depends on a parser that never passed its audit rule | RQ1 |
| Share toward Yes | Direction of each item's yes/no shift | Insensitive to how extreme a model's scores are | Ignores size; ties | Families primary metric |
| Flips and changed answers | Items crossing zero | What changes for a decision | Depends on calibration and threshold | Practical-size reporting |

Across models, nats on any of these log-odds readouts are not on one scale, because models spread their scores differently. The standardized effect, the mean shift divided by the SD of the item-level shifts, is the scale-free comparison (Section 4.8; `notes/qwen_effect_audit_forensic.md` §6), next to the share of items moved and the pooled AUC.

In short: the two-candidate log-odds is the preregistered measure for ALFWorld and the main measure within a model, not a cross-model scale; the first token is the measure across environments; the yes/no item share is the measure across model families; the yes/no log-odds is what a verifier would use, so the decision studies use it; and flips and changed answers translate any of these into practical size.

---

## 5. The effect measures

### 5.1 The goal effect ΔL

The basic effect is the mean, over items, of the aligned-minus-swapped difference in a readout:

$$
\Delta L = \frac{1}{N}\sum_{i=1}^{N}\left[L_i(\text{aligned}) - L_i(\text{swapped})\right].
$$

$N$ is the number of items. Each item contributes equally, so games or users with more items count for more; `notes/rq2_results.md` states that this "defines the estimand as the item-weighted mean". A correct world model gives $\Delta L = 0$. The same formula is applied to the first-token and yes/no readouts.

### 5.2 Against the neutral line: lift by the right task, suppression by the wrong one

Because all three task lines are scored on the same items, the difference from the neutral line splits exactly:

$$
L(\text{aligned}) - L(\text{neutral}) = \left[L(\text{aligned}) - L(\text{swapped})\right] + \left[L(\text{swapped}) - L(\text{neutral})\right].
$$

The first bracket is the goal effect. The second is what a task about another object does relative to a task about nothing; it includes the relevance penalty. Reading the two parts side by side shows whether the goal effect comes from the right task raising predicted success (the aligned line above neutral) or from a wrong task lowering it (the swapped line below neutral). The environments differ here. In ALFWorld, Qwen3-8B's first-token score rises by 3.52 nats from neutral to aligned and by 1.00 from neutral to swapped; in TextCraft by +0.43 and −0.54 (`notes/textcraft_vs_alfworld.md`, "The direction relative to the neutral task differs"); in τ-bench by +1.42 and −2.56 (`notes/tau_results.md`). In ALFWorld the effect is mostly the right task lifting; in TextCraft and τ-bench for Qwen it is mostly the wrong task suppressing. The notes treat any mechanistic reading of this sign pattern ("lift against suppression") as "a hypothesis from the sign structure, not a tested account".

### 5.3 Failure-specificity: failing call minus witness (τ-bench)

If the task overrides a visible fact about the failing action specifically, the failing call should move more than a working call on the same order. If the task simply makes every call consistent with the request look better, the witness (Section 2.3) should move at least as much. Per item,

$$
\text{FW}_i = \Delta L_i^{\text{fail}} - \Delta L_i^{\text{witness}},
$$

and the reported measure is its mean. A positive value is failure-specific. For Qwen3-8B on τ-bench it is −5.47 nats [−5.88, −5.05]: the witness moves far more than the failing call, so the effect reads as a general task-consistency bias. For Llama-3.1-8B it is +0.09 [0.04, 0.14], positive but small, and the follow-up finds it confined to two cells (`notes/tau_results.md`, `notes/tau_dx.md`).

τ-bench also reports **write minus mention**: the effect of a task that requests a change to the order, minus the effect of a task that only asks about it (Qwen +2.17, Llama +0.50). This separates requesting from naming, and is the target of the authorization experiment.

### 5.4 Decompositions in ALFWorld

The extra-mention control splits $\Delta L$ exactly into three parts that sum to it: the effect of removing the other-object sentence from the aligned prompt, the goal role with both objects named, and the mention of the action's object (`notes/rq1_results.md`). The object-mention 2×2 on 1,076 move, heat, cool and clean items splits it into a goal-directed part and a mention part (`notes/rq2_results.md`). Ratios are reported as **share removed** (for example, an "ignore the goal" note removes 9% of Qwen's effect) or **share kept** (an object-free goal keeps 61% for Qwen and 92% for Llama); each ratio's interval is bootstrapped with the ratio recomputed in every resample (`notes/audit_metrics.json`, `ratios`).

### 5.5 Effect sizes relative to something

Three scale-free versions appear.

The **goal-to-evidence ratio** (families, secondary) divides the yes/no goal effect on failures by the neutral-line gap between successes and failures:

$$
R = \frac{\operatorname{mean}_{F}\left[L(\text{aligned}) - L(\text{swapped})\right]}{\operatorname{mean}_{S} L(\text{neutral}) - \operatorname{mean}_{F} L(\text{neutral})}.
$$

Here F is the set of failures and S the set of successes. It was demoted to secondary because, across families, it "mostly reflects how far each model pushes its log-odds" (`notes/families_design.md`, third amendment).

The **size ratio** of the mechanism battery divides each contrast by the aligned-minus-swapped effect on the same items and format; Section 6 explains its zones.

The **standardized effect** divides the mean item difference by its standard deviation, used in cross-environment comparisons and as one of TextCraft's equivalence scales:

$$
d = \frac{\overline{\Delta L_i}}{\operatorname{SD}(\Delta L_i)}.
$$

### 5.6 TextCraft round 2's decomposition: Λ, P, Dn, Dc and T

Round 2 (branch `textcraft-r2`, `notes/textcraft_r2_design.md` §6) tests an account in which the goal's effect on an action that crafts item X is a sum of a lift when X is in the goal's recipe tree, a naming term when the goal names X, and a completion term when crafting X completes the goal. Its five confirmatory contrasts, all on two-candidate $L$ for failures, are as follows. The **intermediate lift** $\Lambda$ is the mean aligned-minus-swapped difference over 990 ingredient failures (actions on an item in the goal's tree that is not the goal):

$$
\Lambda = \overline{L(\text{aligned}) - L(\text{swapped})}_{\text{ingredient}} .
$$

The **within-item penalty** $P$ is the mean, over 675 parent failures, of the aligned line minus the parent line on the extended recipe list:

$$
P = \overline{L(\text{ext\_aligned}) - L(\text{ext\_parent})} .
$$

The **naming term** $D_n$ and the **completion term** $D_c$ are differences in differences between goal-role and ingredient-role items, using the two-item line:

$$
D_n = \overline{L(\text{both}) - L(\text{swapped})}_{\text{goal}} - \overline{L(\text{both}) - L(\text{swapped})}_{\text{ingredient}}, \qquad
D_c = \overline{L(\text{aligned}) - L(\text{both})}_{\text{goal}} - \overline{L(\text{aligned}) - L(\text{both})}_{\text{ingredient}} .
$$

On the rows where both are defined they add up exactly to the gap between goal-role and ingredient-role effects. The **timing** contrast $T$ is the change in the goal effect from one padded look-up to eight, goal role minus ingredient role:

$$
T = \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{goal}} - \overline{\Delta L^{(8)} - \Delta L^{(1)}}_{\text{ingredient}} .
$$

The letter P here is unrelated to the mechanism battery's contrasts P1 to P8. The results for every contrast and model are in `notes/textcraft_r2_results.md` (Section 8.9 summarizes them).

### 5.7 Decision-level effects

The decision simulation measures effects on choices, not scores. For a verifier arm, F is the share of states at which the action finally chosen fails. The gate's effect is

$$
\Delta F = F(\text{aligned}) - F(\text{task-blind}),
$$

in percentage points, together with the **net** count of states moved to failure (states harmed minus states helped) and the share of the effect that runs through the fallback menu (`wf/decisionsim.py`, `option_c_model`). Stage 1b's primary estimand, Δ_invalid, is the same kind of difference in the rate at which the agent acts on a failing action, over all decision points of the agent's own episodes. Option C's is Δ_success, the difference in episode success rate between the task-aware and task-blind verifiers.

### 5.8 Practical size

The notes report practical size in several ways, each defined above: the shift in the two-candidate score $s$ in percentage points (Δp or Δs), flips, sign changes, No-to-Yes answers, the share answered Yes under each task, and decision rates. The same goal effect can be large on one and small on another. For example, Qwen3-8B's two-candidate score shifts by 42 points at step 0 and 1.4 points afterwards (`notes/rq2_results.md`), its yes/no answers change on 0.2% of failures, and as a Stage 1a verifier it makes the chosen action fail at 23.9% of states against 4.4% when task-blind. Which one matters depends on how a system uses the world model.

---

## 6. Statistics, in plain words

### 6.1 Why items are clustered

Items are not independent. In ALFWorld, the 4,666 RQ2 items come from 125 games, between 4 and 90 per game; items from one game share a room, a task, an agent trajectory and often the same failing action at different states (`notes/rq2_results.md`, "Items per game and near-duplicates"). If one game happens to produce large effects, all its items do. Treating items as independent would make intervals too narrow and p-values too small. Every study therefore names a **cluster**, the unit that is treated as independent: games in ALFWorld and the decision simulation, goal pairs in TextCraft (`pair_id`), and users in τ-bench (each user's orders and products are shared across that user's items). The τ-bench results are also reported clustered by product as a sensitivity check.

### 6.2 The cluster bootstrap

A **bootstrap** estimates the uncertainty of a statistic by recomputing it on many resampled datasets. A **cluster bootstrap** resamples whole clusters with replacement: draw 125 games at random from the 125, allowing repeats, take all the items of the drawn games, and recompute the mean effect. Repeating this gives a distribution of plausible values. The project uses 2,000 resamples (`wf/constants.py`, `BOOTSTRAP_RESAMPLES`) and the **percentile** interval: the 2.5th and 97.5th percentiles of the resampled values form the 95% interval (`wf/analysis/stats.py`, `cluster_bootstrap`). The seed is fixed (20260924 for nearly every analysis), so intervals reproduce exactly; it also means intervals from different analyses on the same items are not independent draws.

**The coverage caveat.** A 95% interval should contain the true value in 95% of repeated experiments. `scripts/calibration_check.py` simulated 500 datasets per scenario with 134 games, about 20 items per game, and the pilot's intra-game correlation and spread. The percentile game-bootstrap interval covered the true value 93.2%, 93.4% and 94.8% of the time in three scenarios: "slightly narrow: a known small-sample property. Every bootstrap interval in the notes should be read with this in mind" (`notes/testing.md`, "Calibration of the statistics"). The preregistered verdicts barely depend on this, because they also require the permutation test below, which was calibrated.

### 6.3 The sign-flip permutation test

A **permutation test** asks how often a result at least as large as the observed one would arise if there were no effect. For a paired design, "no effect" means each item's aligned-minus-swapped difference is equally likely to have either sign. A **sign-flip** test with clusters flips the sign of all differences in a cluster together, at random, and recomputes the statistic. In `wf/analysis/stats.py`, `pair_permutation`: let $S_g$ be the sum of the paired differences in cluster $g$ and $N$ the number of items. The observed statistic is

$$
T_{\text{obs}} = \frac{\left|\sum_g S_g\right|}{N}.
$$

On each of $n = 1000$ permutations (`PERMUTATIONS`), each cluster gets an independent random sign $\epsilon_g = \pm 1$ and the statistic is $\left|\sum_g \epsilon_g S_g\right| / N$. If $c$ of the permutations reach at least $T_{\text{obs}}$, the two-sided p-value is

$$
p = \frac{c + 1}{n + 1}.
$$

The smallest possible value is $1/1001 \approx 0.001$. That is why many p-values in the notes are exactly 0.000999 or, after a Holm doubling, 0.001998: they are at the floor, and a stronger effect could not produce a smaller number with 1,000 permutations. In the calibration simulation this test rejected at the nominal rate under no effect (1.0%, 0.8% and 0.6% at p < 0.01), and with a true effect of 0.5 nats and Llama-like noise it rejected 98% of datasets at p < 0.01 (`notes/testing.md`). With very few clusters the floor rises: the smallest achievable sign-flip p with $g$ clusters that differ from zero is $2^{1-g}$, which is why Stage 1b's design, noting that the smallest achievable p with $g$ such games is $2^{1-g}$, requires at least 7 games with a non-zero difference (`MIN_GAMES_FOR_SIGNIFICANCE = 7`).

### 6.4 Holm correction with a fixed family

When several tests are read together, the chance that at least one is significant by accident grows. The **Holm** procedure controls this familywise error. Sort the $m$ p-values from smallest to largest, $p_{(1)} \le \dots \le p_{(m)}$; the adjusted p-value of the $k$-th is

$$
\tilde p_{(k)} = \max_{j \le k} \min\left\{1,\ (m - j + 1)\, p_{(j)}\right\}.
$$

So the smallest p is multiplied by $m$, the next by $m - 1$, and so on, with adjusted values kept in increasing order (`wf/analysis/stats.py`, `holm`). With two tests at the floor of 1/1001, the smaller one becomes 2/1001 = 0.002; that is where RQ2's "Holm p 0.002" comes from.

The **Holm family** is the set of tests corrected together. Each study fixes it in advance: two tests for RQ2 (one per co-primary model); two for TextCraft round 1 and τ-bench (one per model); 18 for families wave A (four per-model share tests plus two tests for each of seven planned comparisons); 16 for wave B; eight per model and format for the mechanism battery; 50 for TextCraft round 2 (a "differs" test and an equivalence test per model and contrast); eight for the τ-bench authorization experiment. The exploratory follow-ups to RQ2 are explicitly not corrected: "Holm's correction applies only to the two co-primary tests of the preregistered RQ2 result. The p-values and intervals here are unadjusted and should be read as descriptive" (`notes/exploratory_results.md`).

**Why "missing enters as p = 1".** The generic `holm` drops missing (NaN) values and corrects for the tests present. That would let a family shrink: if one model's scores were not yet in, the other tests would be corrected for a smaller family and look more significant. Every study that reads a family therefore pads it: a test whose model is still waiting, or whose p is missing or undefined, enters with p = 1, so the others are always corrected for the full planned family. The wording is in `wf/analysis/stats.py`, `prereg_rq2_verdicts` ("a model that was not scored counts as p = 1, so it is never a free pass"), `wf/textcraft/analysis.py`, `wf/tau/pipeline.py`, `wf/families.py` ("Holm over the full planned family (a missing or NaN p enters as 1)") and the mechanism battery's `primary_family` ("a missing or undefined p enters as 1, so the family never shrinks"). Families wave A originally counted only the p-values present; this was fixed before any data (`notes/families_design.md`).

### 6.5 Decision rules: what "supported" and "confirmed" mean

Each confirmatory test turns an estimate, an interval and a p-value into a verdict by a rule fixed before data.

**RQ2** (`wf/constants.py`, `DELTA_L_SUPPORT_NATS = 0.5`, `DELTA_L_STRONG_NATS = 1.0`, `PERMUTATION_P_MAX = 0.01`; `wf/analysis/stats.py`, `prereg_rq2_verdict`; interpreted in `notes/deviations.md` before confirmatory data): per model, a $\Delta L$ at or below zero, or an interval including zero, is "not supported"; a positive $\Delta L$ below 0.5 nats with an interval excluding zero is "weak (not support)"; $\Delta L$ of at least 0.5 with the interval excluding zero but Holm p at least 0.01 is "not supported (fails significance threshold)"; otherwise "supported", and "supported (strong)" at $\Delta L$ of at least 1.0. H2 overall is supported if at least one co-primary model is.

**TextCraft round 1**: confirmed for a model when $\Delta L > 0$, the 95% goal-pair interval excludes zero, and the Holm-adjusted p is below 0.01; the replication holds only if confirmed for both models.

**τ-bench** (`wf/tau/pipeline.py`, `tau_verdict`): "replicates" when the estimate is at least 0.5 nats, the interval excludes zero and Holm p < 0.01; "small effect" when only the interval and Holm conditions hold; "reversed" symmetrically; "no effect larger than 0.5 nats" when the interval lies inside (−0.5, 0.5) and the manipulation check passes; otherwise "inconclusive".

### 6.6 Equivalence tests and why "inconclusive" is not "no effect"

A non-significant difference does not show there is no difference; it may only show that the study could not tell. To claim an effect is negligible, a study must say in advance what "negligible" means, a **margin** $\delta$, and show the effect lies within it. The standard method is **TOST**, two one-sided tests: test that the effect is above $-\delta$, and separately that it is below $+\delta$. If both reject at level $\alpha$, the effect is declared equivalent to zero within the margin. Rejecting both one-sided tests at 5% is the same as the 90% interval lying inside $(-\delta, \delta)$.

The margins differ by study, and are fixed before data:

- **TextCraft round 1** (`notes/textcraft_design.md` §12 C4 and §13 D5; `wf/textcraft/analysis.py`, `equivalence`): about a sixth of the ALFWorld effect, 1.0 nats for Qwen and 0.3 for Llama on the two-candidate scale, plus yes/no margins (0.47 and 0.37 nats) and standardized margins (0.17 and 0.14). A null is read as equivalent only when all three scales say so.
- **Families**: ±0.05 on the item share (5 points), described as "well above run-to-run noise" and "about an eighth" of the effect. The test's p is the larger of the two one-sided bootstrap p-values, each computed as the count plus one over the draws plus one.
- **TextCraft round 2**: a per-model margin of 0.3 item standard deviations, capped at 0.4 nats (written out below the list).
- **τ-bench**: no separate TOST; the "no effect larger than 0.5 nats" verdict requires the 95% interval inside (−0.5, 0.5).
- **τ-bench authorization experiment**: 0.72 nats for Qwen and 0.17 for Llama.

For TextCraft round 2 the margin of model $m$, with $\mathrm{SD}_m$ the standard deviation of its item differences, is

$$
\delta_m = \min\left(0.3\,\mathrm{SD}_m,\ 0.4\right).
$$

A result that is neither significantly different from zero nor equivalent within the margin is **inconclusive**. The notes keep this distinct from "no effect". For τ-bench, the smoke run's precision estimate showed that for Qwen "its projected half-width exceeds the 0.5-nat margin, so the probability of a null verdict at a true zero is 0.0. A Qwen result that is not positive will be reported as inconclusive… and not as evidence of no effect" (`notes/tau_design.md`). Similarly a model that fails a competence gate is "not interpretable, never … a null result" (`notes/families_design.md`).

### 6.7 Effect zones in the mechanism battery

The mechanism battery (`wf/mech2.py`, `zone`, `categorize`; design §11.4; merged into `main` from branch `mechanism2`) does not ask only whether a contrast differs from zero. It asks how large it is relative to the main effect, through the size ratio of each contrast to the aligned-minus-swapped effect on the same items:

$$
r = \frac{\text{mean contrast}}{\operatorname{mean}\left[L(\text{aligned}) - L(\text{swapped})\right]}.
$$

The ratio's range is cut into four **zones** (`TRIVIAL_BAND = 0.25`, `LARGE = 0.5`): N, a ratio below −0.25; Z, from −0.25 to 0.25 ("about zero"); S, above 0.25 and below 0.5 (small); and L, 0.5 or above (large). The **category** of a contrast names the run of zones that the 95% interval of $r$ touches: NEG (N only), NEG_OR_TRIVIAL (N and Z), NULL (Z only, Holm p at least 0.05), TRIVIAL (Z only, Holm p below 0.05, so different from zero but negligible), TRIVIAL_OR_SMALL (Z and S), POS_SMALL (S only), POS (S and L), POS_LARGE (L only), WIDE (three or more zones), and INCONCLUSIVE (the estimate or interval is undefined, the ratio is not estimable because the reference effect is at or below zero or more than 2% of draws are undefined, or the interval lies outside Z while Holm p is at least 0.05).

Each candidate account predicts a set of allowed zones for each contrast (`PREDICTIONS`). A contrast is consistent with an account when every zone its interval touches is allowed, contradicts it when none is, and is undecided otherwise. A contrast counts only when both formats (two-candidate and yes/no) agree. An account contradicted on any contrast "cannot be the sole account". The battery has no TOST margins; "about zero" is the Z band.

---

## 7. How studies are run and kept honest

### 7.1 Confirmatory and exploratory

**Confirmatory** tests are fixed before their data: the question, the items, the readout, the statistic, the decision rule and the multiple-comparison family. A confirmatory verdict licenses a claim at the stated error rate. The project's original confirmatory tests are RQ1, RQ2 and RQ3, filed in a preregistration on 2026-11-05 (`runs/FILED.txt`; the filed proposal and preregistration are not in this repository). The preregistered constants are in `wf/constants.py` ("After filing, only change via human edit + disclosure"). Later studies (TextCraft round 1, τ-bench, the decision simulation's Stage 1a gate) are confirmatory in the sense that their tests were frozen at an annotated git tag before scoring; the notes call these "pre-registered" or "confirmatory for the frozen test", but they are not part of the filed preregistration. This guide uses "preregistered" for the filed tests and "frozen" for tag-fixed tests.

**Exploratory** analyses are everything else. The label licenses description and hypothesis generation, not a confirmatory claim. Within exploratory work the notes distinguish two levels. Most exploratory studies have their design and readouts logged in `notes/deviations.md` or a design note before their data. Numbers marked **post hoc** were computed after the results were in, often by one-off scripts, and are flagged as such where they appear. Exploratory p-values are unadjusted and read as descriptive.

Some later studies are exploratory in label but frozen in procedure: the mechanism battery, Stage 1b, Option C and the τ-bench authorization experiment each have their design and analysis plan frozen at a tag before scoring, and their results are still reported as exploratory.

### 7.2 Design notes written before data

Each study has a design note in `notes/` (or on its branch) committed before any of its data: for example `notes/decisionsim_design.md` before any decision result, `notes/textcraft_design.md` "before any item was generated", and the Option C and τ-bench authorization notes before any code. Amendments made after a review but before scores are added as dated sections ("Amendments after review N, before any score"), and the later section holds where they differ. Changes after data are logged in `notes/deviations.md` as deviations, with what changed and why.

### 7.3 Annotated freeze tags

An annotated git tag records a commit, a date and a message. The project tags the commit at which a design and its analysis code are frozen; anything changed later is a logged deviation. The tags on the public repository are:

| Tag | Commit | What it freezes |
|---|---|---|
| `textcraft-design-frozen-2026-11-20` | `5a3702a` | TextCraft round 1: the primary test, decision rule, outcome table and item and example hashes. Seen before the tag: one format-only smoke run. |
| `taubench-design-frozen-2026-11-21` | `c737cd6` | τ-bench retail: design and reading rules. The commit was pushed before full scoring; the tag itself was created on 2026-11-22, after scoring and before analysis, and adds no rule. |
| `decisionsim-stage1a-frozen-2026-11-21` | `eeeb7dd` | Decision simulation Stage 1a: arms, the 728 even-step states, the 1/64 rounding, the gate and its six categories. |
| `decisionsim-stage1b-frozen-2026-11-22` | `2a05707` (branch `decisionsim-agent`, merged into `main`) | Stage 1b: design and analysis, before any census label, futility output or verifier score was read. |
| `mechanism2-design-frozen-2026-11-22` | `6eab8af` (branch `mechanism2`, merged into `main`) | Mechanism battery: arms, contrasts P1 to P8, prediction table, verdict rules, the residual M. |
| `optionc-design-frozen-2026-11-22` | `54cb71e` (branch `optionc`) | Option C closed-loop study (exploratory), before anything ran. |
| `tau-auth-design-frozen-2026-11-22` | `f70b285` (branch `tau-auth`) | τ-bench authorization experiment (exploratory); only tokenization checks were seen before the tag. |
| `families-waveb-design-frozen-2026-11-22` | `f6bbffb` (branch `families-waveb`) | Families wave B: design and analysis, before any wave B weight was downloaded or scored. |
| `interp-design-frozen-2026-11-22` | `7ade3b9` (branch `interp`) | Interpretability pilot (a calibration study), before anything ran on a GPU. |
| `interp-deviation-analysis-fix-2026-11-23`, `interp-deviation-analysis-fix-2026-11-24` | `06e3871`, `c73147e` | Interpretability pilot, analysis-only fix after the frozen analysis crashed, before any output was read (`notes/interp_deviation_analysis_fix.md`). |
| `textcraft-r2-design-frozen-2026-11-23` | `4ccf84b` (branch `textcraft-r2`) | TextCraft round 2: the five contrasts, the 50-test Holm family, the equivalence margin and the reporting rules, after the sign-off corrections and an independent pre-freeze audit; nothing scored. Seen before the tag: tokenization checks and golden renders. |
| `textcraft-r3-design-frozen-2026-11-23` | `0d6f460` (branch `textcraft-r3`) | TextCraft round 3: contemporary-model replication of round 2's contrasts, the older-against-newer comparisons, the Gemma fallback rule, the environment check and its plan; content complete before round 2's analysis was read. |
| `tau-families-design-frozen-2026-11-23` | `a078a8d` (branch `tau-families`) | τ-bench families extension: the per-model verdict (the τ-bench rule with the families competence rule as a third gate), the Holm family of the two scored models, the cross-model label rule and the Mistral placement check, after one review and its fixes; only the item rebuild, tokenization checks and golden renders were seen before the tag. |

The tag messages (`git tag -l -n20`) list what was seen before each freeze and what the smoke run may be read for. *(Checked 2026-11-23 against the public repository's tag list (`git ls-remote --tags origin`): the ten tags above are all of them; the TextCraft round 2 row was added.)* *(Updated 2026-11-23: the τ-bench families row was added with its results; the remote also holds `textcraft-r3-design-frozen-2026-11-23`, which this table does not list.)*

### 7.4 Smoke runs read for format only

A **smoke** run is a small real run (a few items or games) made before the full run to check that prompts render, scores are produced, timing and memory are acceptable and gates work. It is read for format only: no effect estimate is computed or displayed. The TextCraft tag records exactly what was read ("score counts per model and arm, missing or non-numeric scores, tokenizer checks, model revisions, and the median probability mass on the candidate first tokens … No aligned-versus-swapped difference, per-item score or per-condition summary was computed or displayed"). `notes/tau_design.md` says the τ-bench smoke run "is read for format, timing and precision only". Later tags use the same rule. A change made after reading a smoke run (for example the thinking-mode reasoning cap of 3,072 tokens, `notes/deviations.md`, 2026-11-19) is logged as such.

### 7.5 Independent design reviews

Before a freeze, each design is reviewed by fresh reviewers who get the code and the specification but not the author's reasoning (`notes/testing.md`, layer 8). Reviews list must-fix items (which block the freeze) and should-fix items; the author amends the design and records the amendments in the design note and in `notes/deviations.md`, all before any score. The number of rounds varies: three for the decision simulation, four for τ-bench, five for the mechanism battery and the interpretability pilot. Review files are kept on the study branches (for example `notes/optionc_review1.md`, `notes/tau_auth_review1.md`, `notes/families_waveb_review1.md`). Judgements that need a rater, such as the parser audits, use a blind rater given only the rows and the category definitions.

### 7.6 Mutation testing

A test suite is only as good as the bugs it would catch. **Mutation testing** checks this directly: put a known bug back into the code, one at a time, and confirm that some test fails. Each deliberately broken version is a **mutant**; a mutant is **killed** when the tests catch it and **survives** when they pass anyway. `scripts/mutation_check.py` holds a list of mutants, each a tuple of a description, a file, the correct text and the buggy text, applied by exact string replacement on a copy of the repository ("so the working tree is never touched"). It runs the test files that cover the mutated module, then the whole suite if the mutant survived; a timeout "is never counted as a kill". TextCraft and τ-bench have their own runners (`wf/textcraft/mutation_check.py`, `scripts/tau_mutation_check.py`) that check the suite passes first and record each mutant's killing test.

**The strict named-kill rule.** The runners on `main` count any non-zero pytest exit as a kill: `scripts/mutation_check.py` (which also runs the Stage 1b mutants of `scripts/mutants_decisionsim_agent.py`), `wf/textcraft/mutation_check.py` and `scripts/tau_mutation_check.py`. The last two first check that the unmutated tests pass and record the first failing test of every kill (the τ-bench runner labels a kill without one "no test id (collection or import error)"); the main runner prints the first failed test but does not check the unmutated suite first. Counting any non-zero exit is too generous: a mutant that makes the suite fail to import "says nothing about whether a test would catch the bug". The strict rule is in the code of three branch runners. Families wave B's `scripts/mutation_check.py` (2026-11-22, from its first review) counts a kill only "when pytest exits 1 (tests ran and failed) and names a failing test. Any other non-zero exit (collection error, usage error, interrupted run) is 'broken'"; TextCraft round 2's `scripts/audit_textcraft_r2_mutants.py` applies the same two conditions; and the interpretability pilot's runner (from 2026-11-21) counts only exit code 1 as a kill. The freeze tags of Stage 1b, Option C, the τ-bench authorization experiment, wave B, the interpretability pilot and TextCraft round 2 all report every mutant killed by a named test (for example 175 for Stage 1b, 87 for the authorization experiment, 240 for the interpretability pilot, 159 for TextCraft round 2). For the studies whose runner lacks the strict rule in code (Stage 1b and Option C on the main runner, the authorization experiment on the τ-bench runner), that claim rests on the killing test recorded for each mutant. TextCraft round 1 killed 193 of 193 mutants before scoring, each with a named test (`notes/textcraft_design.md` §13). Counts by study are in `notes/testing.md`. *(Corrected 2026-11-23: this paragraph said the strict rule applied from 2026-11-22 without saying which runners apply it.)* *(Updated 2026-11-23: `scripts/mutation_check.py` on `main` now applies the strict rule by default; a recheck of the frozen tags killed every mutant by a named failing test: Stage 1b 175 of 175, Option C 131 of 131, the authorization experiment 87 of 87; `notes/mutation_strict_recheck_2026-11-23.jsonl`.)* *(Updated 2026-11-24: `wf/textcraft/mutation_check.py` and `scripts/tau_mutation_check.py` now also apply the strict rule by default. Like the main runner since cc1db87, they run pytest without bytecode on a copy made without `__pycache__`. Before this, a file restored within the second its same-length mutant was compiled could keep running the mutant's bytecode, which can only over-credit kills. A rerun of every mutant with the fix found TextCraft 195 of 195 and the authorization experiment 87 of 87, the same killing test for every mutant as the 2026-11-23 recheck. τ-bench gave 153 of 155: two builder mutants were error-only (the item build failed in a fixture's setup) and are now killed by two new tests. `notes/testing.md`, 2026-11-24.)*

### 7.7 Golden prompt renders and tokenization gates

A **golden render** is a stored copy of exactly what a prompt should look like, checked by a test so that no code change can silently alter it.

- `tests/golden/default_prompt.json` holds one real ALFWorld item with its full aligned and swapped message lists. `tests/test_audit_fixes_2026_11_20.py`, `test_golden_default_prompt_message_list`, asserts that `build_messages` reproduces them exactly and that they differ only by the task line. Its rule: "Regenerate the golden file only on purpose, and record why in notes/deviations.md."
- `tests/golden/families_rendered.json` holds the same item as rendered by the chat templates of Gemma-2-9B, Mistral-7B-v0.3 and Llama-3.2-1B and 3B at their pinned revisions.
- `wf/textcraft/data/chat_golden.json` (written by `wf/textcraft/golden_chat.py`) holds the first TextCraft main item rendered by both models' templates, with token counts and the hash of the token ids.

A **tokenization gate** checks, before any score is written, that the scorer reads the tokens it means to read. The checks differ by study but share a core: each candidate tokenized alone must equal the template's rendering of it as the assistant's reply (otherwise teacher forcing scores a different token sequence than the model would see); "Yes" and "No" (or `{"` and `Error`) must each be one token; the two candidates' first tokens must differ; and the prompts of an item's conditions must differ only in the task line. The implementations are `wf/families.py`, `tokenization_checks` (through the scorer's own `_encode` and `_encode_candidate`); `wf/textcraft/tokcheck.py`, `item_problems`; and `wf/tau/token_check.py`, `check_model`. A failure stops the job before it scores anything.

### 7.8 Item hashes and the refusal of stale or mock scores

Every saved score carries an **item hash**: a SHA-256 (truncated to 16 hex characters) over what the score depends on, namely the action, the history, the aligned and swapped task lines and the candidates (`wf/runs.py`, `_item_hash`: "so it can never attach to a different item with the same id"). TextCraft's hash also covers the system text, the question and the examples. A resume or analysis that finds a saved score whose hash does not match the current item refuses to continue ("saved scores belong to different items"). The τ-bench analysis passes only with zero missing and zero stale scores.

Each run folder records the commit (`git_sha`), the settings (`config.json`) and the SHA-256 of every input file (`inputs.json`). A resume refuses uncommitted code, a different commit, changed settings or changed inputs (`wf/runs.py`, `_check_resume`), so one result never mixes versions. A run may read another run's output only if that run finished with PASS or REPORTED at the same commit with unchanged inputs, or at a commit explicitly listed under `accept_commits` in its configuration (`require_passed`). Every analysis's `metrics.json` carries a provenance block with the analysis commit, each shard's commit, GPU and job id, and input hashes (`wf/provenance.py`, `provenance`).

**Mock mode.** The test suite and local runs use a simulator of the ALFWorld server (`wf/env/sim.py`) and a mock scorer. Mock runs write only to `runs_mock/` or a temporary folder, their records are marked `"mock": True`, real runs refuse to read them ("a real run must not read mock data"), and cluster jobs refuse to start if any mock switch is set (`pace/_common.sh`). A passing mock run shows the logic holds together and says nothing about real models.

### 7.9 Runs, shards, STATUS.md and the ICE cluster

A **run** is one named unit of work, defined by a function in `wf/runs.py` (or a study module) and a configuration in `wf/config/<run>.yaml`, started with `python -m wf.runs <run_id>`. Each writes to `runs/<run_id>/`. Large scoring runs are split by game into **shards**, one job each, writing to `runs/<run_id>/shard_<i>_of_<n>/`. Work is saved one record at a time with an fsync, so partial results survive a crash, and a resumed job (`WF_RESUME=1`) continues from the last saved record.

Each folder's **STATUS.md** has a first line of the form `# STATUS — <run> — PASS|FAIL|REPORTED (…)`. PASS means a target was met; REPORTED means the run has no target or outputs a rule; `BELOW TARGET` or `STOP` is stated when it applies. A scoring shard writes `REPORTED (shard complete)`, and an analysis waits until every shard reports complete at the same commit (`wf/rq2.py`, `complete_shards`).

The GPU jobs ran on the **PACE ICE** cluster, submitted with the scripts in `pace/` (`pace/submit.sh` adds account, partition and the GPU request; jobs accept any of L40S, A40 or A100 and use `--no-requeue`). Each batch of runs was made in its own clean checkout on ICE, named `text-wm-fix<N>`, "so that a resume never mixed commits" (`notes/reproducing.md` §2). This is what a "fix36-style checkout" refers to: a fresh, numbered checkout at one commit, used for one batch of jobs and not updated while its shards run. The notes name checkouts from `text-wm-fix4` to at least `text-wm-fix39` (the list is in `notes/reproducing.md` §2; updated 2026-11-23 from "at least `text-wm-fix35`"); for example TextCraft round 1 was scored in `text-wm-fix24`. Diagnostic runs used new checkouts and only read the confirmatory ones.

**Where data live.** `runs/**` is not tracked by git, so a fresh clone has no run output. The data are on ICE storage, and a full copy with a SHA-256 manifest is kept outside the repository; it is not public (`notes/reproducing.md`). Every metrics file in `notes/` is a byte-identical copy of the `metrics.json` of the analysis run named in its note, so every reported number can be traced without the raw data.

---

## 8. Each study in brief

Each entry says what the study asks, its design in brief, its readout, its status on 2026-11-22, and where to read more. Studies that exist only on a branch are marked as such; their design notes are read with `git show origin/<branch>:notes/<file>`.

### 8.1 Tier 0 and the pilots

**Tier 0** checks the machinery on train games (the real server's behaviour, replay, the parser, scorer agreement) and answers no research question. **Pilot A** built items from 50 train games and scored them with the two prompted models and a fine-tuned Word2World model, to estimate effect sizes, apply the scoring validity rule of Section 4.1, and plan the item count. **Pilot B** checked whether the agent uses predictions at all (perfect predictions had to change its action on at least 10% of steps), set the error-injection rates for RQ3, and chose the retained-prediction format. Pilot games come from `train`, so none enter the confirmatory tests. Results are assembled in `runs/PILOT_REPORT.md` on the cluster; the design is `PILOT_PROTOCOL.md`, and the decisions that followed are in `notes/deviations.md`, 2026-11-17.

### 8.2 RQ2: the preregistered goal effect (confirmatory)

**Question.** On determined failures, does naming the action's object in the task raise the world model's two-candidate log-odds of success? **Design.** The 4,666 items of Section 2.1, scored by Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct with three few-shot examples (set 1), aligned against swapped. **Readout.** Two-candidate $\Delta L$, game-clustered bootstrap and sign-flip permutation, Holm over the two models, the RQ2 decision rule of Section 6.5. **Result.** Supported (strong) for both: Qwen3-8B 6.12 nats [5.48, 6.75], Llama-3.1-8B 1.65 [1.51, 1.78], Holm p 0.002 each. Secondary measures: first-token $\Delta L$ 2.52 and 0.68; shift in $s$ 5.9 and 16.4 points. In the study's thinking-off format, Qwen's effect is concentrated at step 0 (11.0 nats, 42 points of $s$, on 514 items), a concentration that comes from that format's empty think block (see the update that follows); Llama's is flat across history lengths. *(Updated 2026-11-23, exploratory follow-up: Qwen's step-0 concentration depends on the empty think block that Qwen3's thinking-off template puts before the answer. Without it, the effect is 3.31 nats overall and 2.73 at step 0, against 3.3 to 3.6 later, while the standardized effect is unchanged (1.02 against 1.05). The confirmatory numbers stand; `notes/qwen_effect_audit_pipeline.md` §7.)* **Read:** `notes/rq2_results.md`, metrics `notes/rq2_confirmatory_metrics.json`; code `wf/rq2.py`.

### 8.3 RQ1 and H1: which way do written predictions err? (measure and H1 filed; parser caveat)

**Question.** Do world models lean toward predicting success or failure, and does that lean depend on information (history only versus full state, H1a) and on the model (H1b)? **Design.** Written predictions on 6,527 determined failures and 307 determined successes from 132 games, few-shot models with three example sets and a full-state prompt, plus the Word2World fine-tuned models. **Readout.** The asymmetry $A$ of Section 4.5. **Result.** Both prompted models lean toward success, Llama far more: $A$ is 0.193 for Qwen3-8B and 0.767 for Llama-3.1-8B (history only). H1b holds (+0.574); H1a holds for Llama and is reversed for Qwen, whom the full state makes more optimistic. The RQ1 measure and H1's predictions were fixed before data; the rest of this run's design was decided after the RQ2 result and is not preregistered (`notes/deviations.md`, 2026-11-18). The same scoring run carries RQ2's secondary conditions: the goal effect holds with every example set; an instruction to ignore the goal removes 9% of Qwen's effect and none of Llama's; the full state cuts it by about 70% without removing it; the fine-tuned Word2World Qwen model shows none and the Llama model a small reversed effect. **Caveat.** The parser never passed its audit rule (Section 4.5). The note's original reading that the effect comes mainly from naming the object was superseded by the object-free goals (Section 8.5). **Read:** `notes/rq1_results.md`, metrics `notes/rq1_v2_metrics.json` (supersedes `rq1_metrics.json`); code `wf/rq1.py`.

### 8.4 RQ3: what do false successes cost an agent? (confirmatory)

**Question (H3).** Do wrong predictions do more harm when the agent keeps them in its history than when it drops them, and do false successes cost more than false failures when kept? **Design.** A Qwen3-8B ReAct agent played all 134 games, seeing a predicted outcome before each action, from an oracle with injected errors: false successes (a failing proposal gets the real success text) or yoked false failures. The score per episode is minus the step count, capped at 30, with −31 for failure. The statistic is the interaction

$$
I = \left[S_{FS} - S_{FF}\right]_{\text{kept}} - \left[S_{FS} - S_{FF}\right]_{\text{dropped}},
$$

predicted negative; $S_{FS}$ and $S_{FF}$ are mean scores under false successes and false failures. **Result.** Not supported: $I = +0.94$ steps [−0.20, 2.14] on the primary set of 160 units in 61 games. The note explains why the experiment could not test the question well: keeping even perfect predictions lowered success (32.8%) below having no foresight (47.0%); most episodes in the primary set failed; and the errors were injected, not the world models' own. *(revised 2026-11-25)* The agent's prompt also included ALFWorld's admissible-action list, and oracle foresight that only said whether an action was valid did as well as perfect foresight (60.4% against 58.2% success), so the null means "not detected in this setup". **Read:** `notes/rq3_results.md`, metrics `notes/rq3_metrics.json`; code `wf/rq3.py`, statistics `wf/analysis/rq3_stats.py`.

### 8.5 The exploratory follow-ups to RQ2

All of these are exploratory, with designs logged in `notes/deviations.md` before their data, reported in `notes/exploratory_results.md` with uncorrected statistics.

- **Object-free goals** (`wf/rq1x.py`). With the object only described, 61% of Qwen's effect [54%, 68%] and 92% of Llama's [83%, 103%] remains, so the task acts through the object's meaning, not only its name. The same run reports the goal effect on 292 determined successes and the drop in discrimination (Llama's AUC falls from 0.87 to 0.77 when the aligned task replaces the neutral one).
- **Thinking mode** (`wf/thinking.py`). *(revised 2026-11-25)* Whether Qwen3-8B reasoning before answering reduces the goal effect depends on the scale, and is not determined. In nats it does not: after reasoning, the aligned-versus-swapped effect is 4.01 nats on 1,000 items, and without the empty think block of Qwen3's thinking-off format the same items give 3.33 (plain text gives the same no-block level for Qwen3-8B, 3.26 against 3.28 on 300 items). On the standardized scale of Section 5.5 it falls: 0.25 [0.18, 0.32] after reasoning, against 1.02 with the empty block and 1.10 without it on the same items, and the share of items moved toward success falls from 88% to 61% (post hoc, recomputed from the archived `thinking_probe` run; the no-reasoning values are `notes/qwen_effect_audit_followup_metrics.json`, `thinking_probe_items`). Part of the fall is sampling noise: each task line samples its own reasoning trace, which widens the spread of item shifts and biases the standardized effect after reasoning down. The reading of 2026-11-23, "does not reduce", rested on the nats alone. *(Reading revised 2026-11-23, exploratory follow-up.)* An earlier reading said reasoning keeps 66% of the gap (4.01 against 6.06 with thinking off) while leaving the lift over the neutral task (9.0 against 8.6); the thinking-off comparison point carries the empty block, which about doubles Qwen's effect in nats (6.09 with it, 3.33 without on these items), so that reading compared a filled think block with an empty one. The neutral-task comparisons have no baseline without the block and are not read. The comparison also crosses scoring engines (vLLM after sampled reasoning against Hugging Face scoring), which the note flags. The probe's numbers stand; only their reading changes.
- **Dissociation** (`wf/rq1d.py`). Adding "There is a mug somewhere in the room.", "The mug 1 is at cabinet 3." or "There is no mug in this room." to both prompts tested whether the task acts as evidence that its object exists. The readouts were mixed and the note calls for a confirmatory test with a design that separates the accounts, which became the mechanism battery.
- **Counterfactual aligned task and the yes/no question** (`wf/rq1v.py`). Replacing the task's destination leaves the effect unchanged (so it does not need the exact real task), and the effect replicates on the anchored yes/no question (2.80 nats for Qwen, 2.21 for Llama).
- **Mechanism diagnostic** (`wf/mech.py`). Stating the facts that decide the failure removes 25% of Qwen's effect and 2% of Llama's; stating the environment's rules increases it; the effect is fully present on items where the model answers state questions correctly. The note's exploratory reading is "a task-relevance bias applied at answer time", not missing state information. The first half holds; the second was narrowed by the mechanism battery (Section 8.17), which found no single account, and for Qwen3-8B a bare mention of the object with an instruction to ignore it reproduces the whole effect, which a relevance account does not predict.
- **Qwen3 size series** (`wf/sizes.py`). Qwen3 0.6B to 32B, thinking off, same prompts. The effect is positive at every size. The prediction logged before scoring, that the effect would not grow with log parameters, failed over the full range (slope +0.51 nats per e-fold [0.40, 0.63]), but only because the two smallest models cannot do the task; among the competent sizes (4B to 32B) the two-candidate effect falls (7.5, 6.1, 4.3, 2.4 nats) while practical size does not. Qwen3-32B answers Yes on 11.6% of failures under the aligned task against 3.8% under the swapped task. *(Updated 2026-11-23, exploratory follow-up: the fall in nats is mostly the thinking-off block's scale inflation, which differs by size. On 300 items, plain text keeps 31%, 53% and 23% of the with-block effect for 4B, 8B and 14B, and the standardized effect is 1.05, 1.15 and 0.72 in plain text against 0.97, 1.02 and 1.00 with the block, so the flat standardized series holds in plain text for 4B and 8B but not 14B. Qwen3-32B was not rescored.)*
- **Facts found while designing later studies** (section 7 of the note, post hoc): how rarely the RQ3 agent proposes the RQ2 kind of failure (30 fresh goal-object failures in 2,973 proposals, never at step 0), the step-0 concentration of the yes/no effect (measured with Qwen3's empty think block; on RQ2's items it was not rescored without it), the relevance penalty, the 1/64-nat grid and the even-step sampling rule.

### 8.6 Model families, wave A (exploratory)

**Question.** The comparison of Qwen3-8B with Llama-3.1-8B cannot separate model family, generation and post-training. Does the effect appear in models from other developers and in smaller distilled models? **Design.** Gemma-2-9B-it, Mistral-7B-Instruct-v0.3 and Llama-3.2-1B and 3B-Instruct scored on the RQ2 items and the 292 successes under aligned, swapped and neutral lines in both formats, beside the two reference models; a tokenization gate before scoring; the competence gate of Section 4.7; one primary metric (the share of failures pushed toward Yes); seven planned paired comparisons with "differs" and equivalence-within-5-points tests; Holm over a fixed family of 18; and a Mistral placement run checking whether moving its system text changes the effect (it does not, by the rule fixed in advance). **Result.** All three interpretable new models show the effect in both formats; Llama-3.2-1B fails the competence gate. Shares toward Yes: Qwen3-8B 0.887, Llama-3.1-8B 0.896, Mistral 0.879, Llama-3.2-3B 0.840, Gemma 0.748. Gemma differs from the references; the Mistral comparisons are within about 2 points but equivalence is not shown after Holm. Practical size depends on calibration (Section 4.6). *(revised 2026-11-25)* Option C's step-0 check found Qwen3-8B partly reading the anchored yes/no question as asking about task success, so the primary share may include a task-progress judgement; the two-candidate effect, which does not use the question, is positive for every model. **Read:** `notes/families_design.md`, `notes/families_results.md`, `notes/families_metrics.json`; code `wf/families.py`.

### 8.7 Model families, wave B (exploratory; branch `families-waveb`)

**Question.** Does the effect persist in current-generation open models, and does post-training (base against instruct) change it? **Design.** Seven runs: Ministral-3-8B-Instruct, Qwen3.5-9B in chat format and in plain format, Qwen3.5-9B-Base in plain format, Gemma-4-E4B-it, Qwen3.8-27B and Gemma-4-31B-it, scored in a separate pinned software environment (`serve2`, transformers 5.23.0) with an environment-equivalence gate that rescores 300 items for two wave A models within fixed tolerances. Holm over a fixed family of 16 (six share tests and five planned comparisons). Review 1 fixed a scorer incompatibility with the newer library, added a hash check of downloaded weights against committed manifests, and added a reading for an effect carried by a minority of items. **Status (updated 2026-11-23).** Frozen at tag `families-waveb-design-frozen-2026-11-22`. The environment check did not pass: Qwen3-8B's yes/no scores move 0.248 nats per score in the new stack (limit 0.07) while its goal effect moves 0.026. A float64 diagnostic showed this is a real numerical change, not bf16 rounding: the float64 readout logits move by about the bf16 unit roundoff, which Qwen's large readout logits amplify, about seven times the change measured across GPU types in the old stack. Decided before any wave B score: every older model used in a comparison is rescored in the new stack. *(revised 2026-11-25)* The amendment is written and re-reviewed (review 2 and its fixes on branch `families-waveb-serve2-rescore`); a re-freeze is in progress. No wave B model scored. **Read:** `notes/families_waveb_design.md` and `notes/families_waveb_review1.md` on the branch.

### 8.8 TextCraft round 1 (confirmatory for the frozen test) and its follow-ups

**Question.** Does the ALFWorld effect replicate in a second environment whose items remove history-goal agreement (same-history couples) and the success-tail asymmetry (both replies name the item)? **Design.** The 1,500 main items of Section 2.2, Qwen3-8B and Llama-3.1-8B, aligned against swapped on two-candidate $L$, clustered by goal pair, Holm over the two models at 0.01; competence and three-scale equivalence rules fixed before scoring at tag `textcraft-design-frozen-2026-11-20`. **Result.** Confirmed for Qwen3-8B (+0.94 nats [0.63, 1.24]); a small negative effect for Llama-3.1-8B (−0.19 [−0.30, −0.09]), whose competence is weak; so it does not replicate across both models. **Follow-ups** (exploratory, post hoc): the comparison with ALFWorld on every shared scale (`notes/textcraft_vs_alfworld.md`, source of the 59% tail finding), Qwen robustness and competence (`notes/textcraft_qwen_robustness.md`, source of the unanchored yes/no finding; `notes/textcraft_qwen_competence.md`), and Llama diagnostics and competence (`notes/textcraft_llama_dx.md`, `notes/textcraft_llama_competence.md`). The Llama diagnostics suggest an exploratory "lift plus terminal penalty" reading: an action on an ingredient of the goal is lifted by the aligned task by about +0.9 nats for Qwen and about zero for Llama, while actions that craft the goal item itself score about 0.8 nats lower than matched ingredient actions in both models (for Qwen the interval includes zero). In the note's words, "With a positive ingredient effect near +1 and the same goal-role penalty, Qwen's total stays positive. With an ingredient effect near zero, Llama's goes negative." It labels the mechanism "not established". None changes the confirmatory verdict. **Read:** `notes/textcraft_design.md`, `notes/textcraft_results.md`, `notes/textcraft_metrics.json`; code `wf/textcraft/`.

### 8.9 TextCraft round 2 (confirmatory for the frozen 50-test family)

**Question.** "Does Llama's negative TextCraft effect arise from a goal-completion or naming mechanism, from poor reasoning about the environment's state, or from a combination?" **Design.** Five models (Qwen3-8B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, Llama-3.2-3B), the clean prompt, the anchored question, the two-item, parent and padding arms of Section 3.6, and the five pre-registered contrasts $\Lambda$, $P$, $D_n$, $D_c$ and $T$ of Section 5.6, each with a "differs" and an equivalence test, Holm over 50 tests. Competence never withholds a verdict; it sets the wording ("discriminates" or "does not discriminate"). Frozen at tag `textcraft-r2-design-frozen-2026-11-23` (commit `4ccf84b`) after two reviews, the sign-off corrections and an independent pre-freeze audit; scored and analysed at that commit (ICE jobs 7247899–7247918 and 7247919). **Result.** The within-item penalty $P$ is confirmed below zero in all three models that discriminate on goal crafting: Qwen3-8B −1.02, Llama-3.1-8B −1.68 and Gemma-2-9B −1.27 nats. The analysis reads this as "penalty general across the competent models", and no rejection rule fires. The lift $\Lambda$ is confirmed above zero for Qwen (+0.67), equivalent to zero for Llama-3.1 (+0.095, in a population where Llama does not discriminate) and confirmed below zero for Gemma (−0.30). Llama-3.1's decomposition reads "completion penalty, naming constrained" as a measurable effect only, Gemma's reads "naming / partial-completion penalty", and Qwen's is unresolved. Added empty look-ups do not change the penalty beyond the margin where $T$ can be read. Descriptively, R3 fires for Qwen3-8B (its within-item penalty and between-item role gap disagree), and Llama-3.1's central-question label is "penalty with inventory-consistent tracking". Mistral-7B and Llama-3.2-3B discriminate for no contrast. **Read:** `notes/textcraft_r2_results.md` (every contrast for every model), `notes/textcraft_r2_design.md`, `notes/textcraft_r2_metrics.json`; code at the tag (`wf/textcraft/r2*.py` on branch `textcraft-r2`). *(Updated 2026-11-23: status changed from "frozen; nothing scored" to results.)*

### 8.10 TextCraft round 3 (branch `textcraft-r3`)

A proposal to rerun round 2's design on contemporary models (Qwen3.5-9B, Ministral-3-8B, Gemma-4-12B with Gemma-4-E4B as fallback), a larger-model extension and an exploratory Qwen3.5 thinking arm, with older-against-newer comparisons paired on items. **Status (updated 2026-11-23).** Built, reviewed twice and frozen at tag `textcraft-r3-design-frozen-2026-11-23` (commit `0d6f460`); its content was complete before round 2's analysis was read, and its replication targets follow mechanically from round 2's filed verdicts. The environment check uses wave B's original per-score criteria, pinned as round 3's own constant; on wave B's evidence Qwen3-8B is expected to fail it, which triggers the declared rescoring of the older models in the new stack. Smoke runs: Qwen3.5-9B and Ministral-3-8B pass every gate; Gemma-4-12B fails attention agreement, so the mechanical fallback gives the Gemma slot to Gemma-4-E4B; the thinking arm's padded-against-alone start check failed narrowly (0.111 against 0.10) and is under diagnosis. Core scoring is running. **Read:** `notes/textcraft_r3_design.md` (section 14 governs) and the two reviews on the branch.

### 8.11 τ-bench retail (confirmatory for the frozen test) and its follow-up

**Question.** Does the effect appear in a tool-calling environment with long structured replies, and is it specific to failing calls? **Design.** The 856 primary items of Section 2.3, Qwen3-8B and Llama-3.1-8B, aligned against swapped on the first token, clustered by user, Holm over the two models, the verdict rule of Section 6.5, with the witness, mention pair, manipulation and discrimination checks and a pre-registered reading table. **Result.** Qwen3-8B +3.98 nats [3.61, 4.35], "replicates"; but the witness moves more than the failing call (failing minus witness −5.47 [−5.88, −5.05]), so it reads as a general task-consistency bias rather than an override of a visible failure. Llama-3.1-8B +0.47 [0.41, 0.54], "small effect", failure-specific by the table; the follow-up finds the failure-specific part small (+0.09) and confined to two cells. Joint verdict: not a replication across both models. The design was pushed before scoring; its tag was created after scoring and before analysis. **Follow-up** (`notes/tau_dx.md`, exploratory, post hoc, plan committed before any item-level score was read): an independent recompute, per-class structure, competence, the yes/no task-judgement finding of Section 4.4, practical size, and a three-way reading in which Qwen's effect is mostly relevance and about a nat looks like authorization. **Read:** `notes/tau_design.md`, `notes/tau_results.md`, `notes/tau_metrics.json`; code `wf/tau/`.

### 8.12 τ-bench families extension (confirmatory per-model family; results on `main`, code on branch `tau-families`)

**Question.** Does the τ-bench goal effect appear in instruct models of other families and of Llama's next release, and is it shared by working and failing calls or specific to failing calls? **Design.** Mistral-7B-Instruct-v0.3 and Llama-3.2-3B-Instruct on the 856 primary items, the τ-bench verdict rule unchanged (first token, aligned against swapped, clustered by user, Holm over the two scored models at 0.01) with the families competence rule on first-token scores as a third gate; a cost-reduced protocol of 16 passes per item; cross-model readings descriptive, with "failure-specific" allowed only under three pre-stated conditions; an exploratory placement check for Mistral's chat template. Gemma-2-9B is not run because the fixed part of every prompt is 9,575 Gemma tokens against an 8,192-token context. Frozen at tag `tau-families-design-frozen-2026-11-23` (commit `a078a8d`) after one review and its fixes; scored and analysed at that commit (ICE jobs 7247978–7247986, every shard on an L40S; analysis job 7251608, after a first run failed for a missing reference-run link). **Result.** Llama-3.2-3B: "replicates (ALFWorld direction)", +0.99 nats [0.90, 1.08], every gate passed; its witness moves +1.07, so it is "a shift shared by working and failing calls (working calls move more)", the same label as Qwen3-8B; 82% of items move toward success; both failure classes positive, so no class-mix reading. Mistral-7B: "not interpretable (fails the families competence rule on the first-token readout)", because its first token barely separates working from failing calls (AUC 0.58 against 0.70; discrimination 0.36 nats against 1); its estimate (+0.23 [0.18, 0.27]) is reported and not read, and the placement check reads its result as format-dependent (+0.53 with the system text in the first turn against +0.23 native). Neither model passes competence on yes/no (Mistral says Yes and Llama-3.2-3B No to nearly everything). Against the reference models, Llama-3.2-3B moves the same share of items as Qwen3-8B ("no difference detected", 0.82 against 0.78) although its effect is a quarter of Qwen's in nats, and more items than Llama-3.1-8B ("differs"). **Read:** `notes/tau_families_results.md`, `notes/tau_families_metrics.json`, `notes/tau_families_design.md`; code `wf/tau/families.py` at the tag. *(Updated 2026-11-23: status changed from "not frozen and not scored" to results.)*

### 8.13 τ-bench authorization experiment (exploratory; results on `main`, code on branch `tau-auth`)

**Question.** Does part of the τ-bench effect reflect a belief that the backend refuses writes the user did not request ("authorization"), rather than relevance? **Design.** Three backend descriptions in the system message: the confirmatory prompt unchanged; an "auth" description stating a request rule ("a write call is executed only if the user's task asks for that change to that order", with the error "The user did not request this change"); and a "blind" description stating that "The user's task never changes a call's output". The write-minus-mention component is split into the part the blind description removes and the part that persists. 192 items, Holm over 8 tests at 0.01, margins 0.72 nats (Qwen) and 0.17 (Llama), with comprehension and positive-control gates. **Result** (2026-11-23; frozen at tag `tau-auth-design-frozen-2026-11-22`, scored and analysed at that commit). Qwen3-8B passed both gates (task-blind comprehension 24 of 24 on every question; positive control +3.38 nats [+3.01, +3.77]). The blind description removed none of its request component (removed −0.09 [−0.42, +0.26], persisting +2.39 [+1.94, +2.87] of +2.30) or naming component (removed −0.08, persisting +1.95): both "relevance-like (reading (a))", reading "(a) throughout", even though its answer to the rule question K1 switched from wrong under the confirmatory prompt to right. Llama-3.1-8B failed both gates (K2 wrong on 24 of 24 in the blind arm only; positive control +0.30 [+0.19, +0.40], below 1 nat), so both its categories are "inconclusive (gate)"; descriptively its request component grew (+0.54 to +0.74). **Read:** `notes/tau_auth_results.md`, `notes/tau_auth_metrics.json`, `notes/tau_auth_design.md`; code `wf/tau/auth.py` on branch `tau-auth`.

### 8.14 Decision simulation, Stage 1a: fixed candidate menus

**Question.** Does the bias change which action a verifier picks? **Design.** At 728 even-step states of the collected ALFWorld episodes (Section 2.1), a verifier asks the anchored yes/no question about the agent's own proposal; if the answer is No (log-odds at or below zero after rounding to 1/64 nat), rule 2 takes the highest-scored alternative from a fixed **menu** of every action collected at that state (8 to 24 candidates, about 85% of the working ones `go to` actions, nearly all failing ones acting on the task's object). Verifier arms: aligned, swapped, neutral, task-blind, plus no verifier and an oracle. The gate for a closed-loop study (Option C) requires, for at least one primary model, aligned against task-blind: $\Delta F$ above zero with p < 0.05, net harm at least half the headroom (21 of 42 states), the fallback cell supplying at most half, and the effect attributable to the real task by comparison with swapped. Frozen at tag `decisionsim-stage1a-frozen-2026-11-21`. **Result.** "Worth doing", decided by Qwen3-8B: the chosen action fails at 23.9% of states with a task-aware verifier, 5.8% with none and 4.4% task-blind (+19.5 points [17.2, 22.3]), mostly at step 0. Llama-3.1-8B shows no reliable change (+1.1). The three secondary models, applied post hoc, also reach "worth doing" but never decide the gate. The exploratory follow-up finds a two-part mechanism: any concrete task makes Qwen reject working first moves, and naming the object ranks failing goal-object actions first in the fallback. Under natural changes of rule or menu the effect keeps its direction but shrinks to 3–6 points. *(Updated 2026-11-23, exploratory follow-up: the step-0 harm depends on Qwen3's thinking-off format, which puts an empty think block before " Yes"/" No". At the 134 step-0 states the real-task verifier accepts 7.5% of first moves with the block against 90% task-blind, a harm of 71.6 points; in plain text it accepts 26% against 98%, a harm of 38.8 points [30.6, 47.0]. The block roughly doubles the step-0 harm but does not create it, and the real-versus-swapped gap persists in plain text (39 against 4 points). The frozen verdict stands; see `notes/decisionsim_results.md`, "Sensitivity of the step-0 harm to the prompt format".)* **Read:** `notes/decisionsim_design.md`, `notes/decisionsim_results.md`, `notes/decisionsim_dx.md`; code `wf/decisionsim.py`.

### 8.15 Decision simulation, Stage 1b: the agent's own candidates

**Question.** How often does a real agent put the relevant actions in front of a verifier, and does showing the task make it act on failing actions more often? **Design.** At the 2,973 decision points of the 134 RQ3 no-foresight episodes, the candidate list is the agent's greedy action plus up to 7 more distinct actions from 20 samples, ranked by frequency. A **regime candidate** is a failure that acts on the task's object, is determined by the history and is not at a shared spot, "the kind of action RQ2 scores"; an **encounter state** has at least one regime candidate in its list. The verifier accepts the highest-ranked candidate it says Yes to, otherwise the agent acts on its first choice. The priority verifiers are Llama-3.1-8B and Mistral-7B (chosen because they change many answers), with Qwen3-8B as a contrast. A futility rule, read before any verifier is scored, stops the study if encounter states with a working alternative fall in fewer than 7 games. Frozen at tag `decisionsim-stage1b-frozen-2026-11-22`. **Result.** *(Updated 2026-11-23: this entry described the study as still scoring; it is done and merged into `main`.)* The agent's own lists contain a determined goal-object failure at 4.3% of states [2.2, 6.8], in 22 of 134 games (predicted 8–30%), never at step 0; the futility rule was not met (19 games, at least 7 required), so the verifiers were scored. The frozen verdicts are "no reliable difference" for both priority verifiers, overall (+0.17 points [−0.92, +1.27] for Llama-3.1-8B, +0.17 [−0.09, +0.47] for Mistral-7B) and on encounter states (+4.65 [−1.16, +12.50] and 0.00 [−2.46, +2.05]); neither verifier abstains enough to count as inert, so the pre-declared reading is "the bias rarely reaches this agent's decisions". Qwen3-8B, the contrast, gives +2.29 [+1.36, +3.33], mostly from rejecting working alternatives under any task rather than from naming the object. **Deep-dive** (exploratory, post hoc): the nulls reproduce exactly and are stable to leaving out any game; Llama's encounter-state estimate rests largely on one game with a repeated action; the bias is visible in Llama's scores (it accepts more regime candidates under the real task) but the selection rule pre-empts almost all of them, and Stage 1a's step-0 harm cannot recur because the agent's step-0 lists hold no failing candidate. *(revised 2026-11-25)* The agent's prompt included ALFWorld's admissible-action list, and every failing candidate it proposed lay outside that list, so these nulls mean "not detected in this setup", not that the bias is harmless for agents in general. **Read:** `notes/decisionsim_agent_design.md`, `notes/decisionsim_agent_results.md`, `notes/decisionsim_agent_dx.md`, metrics `notes/decisionsim_agent_metrics.json`; code `wf/decisionsim_agent.py`.

### 8.16 Option C: a closed-loop agent with a world-model verifier (exploratory; results on `main`, code on branch `optionc`)

<<<<<<< HEAD
**Question.** In a live loop, does a task-aware verifier lower the agent's success compared with a task-blind one? **Design.** The RQ3 Qwen3-8B ReAct agent plays the 134 games with a Qwen3-8B verifier it does not see (yes/no readout as designed; the anchored observation readout became primary after the step-0 check). Under the primary "accepted-only" rule a rejected proposal triggers sampled alternatives and the first accepted one is taken; otherwise the proposal stands. Arms: no verifier, task-blind and task-aware at three seeds, swapped at one seed (descriptive), with every task-aware step also scored under the other prompts without acting on them. The primary measure is the aware-minus-blind difference in success, paired by game. A step-0 check classifies whether a readout "collapses" before any tier runs. **Result (2026-11-23).** Frozen at tag `optionc-design-frozen-2026-11-22`. The step-0 check classified Qwen3-8B as "question-specific (supports the task-success reading)": on yes/no the task-aware verifier accepts 82 points fewer of the agent's working first moves than the blind one [−88, −75], while on the anchored observation readout the gap is zero in every game, and that readout is competent. Under the pre-declared reallocation (commit `2a86683`, before any census episode) the observation readout became the primary run and yes/no a one-seed secondary. Primary (observation readout): success "no reliable difference", aware minus blind −1.2 points [−3.0, 0.0] (p 0.23); only 3 of 134 games differ and 84% of seed pairs play identical episodes, so the test could not have been significant; both key secondaries (override harm minus help, invalid actions in the first 10 steps) "no reliable difference"; obstruction check clear. Secondary (yes/no): success "no reliable difference" (−4.5 points [−11.2, +2.2]); invalid actions per episode "harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)", +1.12 [+0.33, +1.96], with the same label for both key secondaries. The route is Stage 1b's: under any concrete task the yes/no verifier rejects more working proposals, abstains more and rescues fewer failing ones; it remains better than no verifier. The swapped verifier accepted working real-goal proposals 19 points less often than the aware one, so success-level attribution is not identifiable. On identical states the real task raises acceptance of failing goal-object proposals under both readouts (+27.5 points observation, +3.3 yes/no). *(Updated 2026-11-24.)* Argmax rule (yes/no readout, the highest-scored candidate is taken; jobs 7253142–7253145, submitted after the Tier 1 analysis; analysis job 7295370 at `01cad81`): success 'no reliable difference', −0.7 points [−9.0, +7.5], with 29 of 134 games differing; invalid actions per episode 'no reliable difference', −0.07 [−0.85, +0.71]; invalid actions in the first 10 steps 'harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)', +0.23 [+0.10, +0.37]. The design's prediction for this rule (a loss of 2 to 10 points) is not supported. The Llama loop was not run. *(revised 2026-11-25)* The agent's prompt included ALFWorld's admissible-action list, so these nulls mean "not detected in this setup". The primary rests on 3 discordant games, and the design could detect only success losses of roughly 6 to 11 points, so smaller costs are not excluded. *(The verifier uses Qwen3's thinking-off format with the empty think block, as Stage 1a did; the results note, section 10.3, says what that does and does not affect.)* **Read:** `notes/optionc_results.md` (metrics `notes/optionc_metrics.json`, `notes/optionc_step0_metrics.json`); design `notes/optionc_design.md` and review `notes/optionc_review1.md`.
=======
**Question.** In a live loop, does a task-aware verifier lower the agent's success compared with a task-blind one? **Design.** The RQ3 Qwen3-8B ReAct agent plays the 134 games with a Qwen3-8B verifier it does not see (yes/no readout as designed; the anchored observation readout became primary after the step-0 check). Under the primary "accepted-only" rule a rejected proposal triggers sampled alternatives and the first accepted one is taken; otherwise the proposal stands. Arms: no verifier, task-blind and task-aware at three seeds, swapped at one seed (descriptive), with every task-aware step also scored under the other prompts without acting on them. The primary measure is the aware-minus-blind difference in success, paired by game. A step-0 check classifies whether a readout "collapses" before any tier runs. **Result (2026-11-23).** Frozen at tag `optionc-design-frozen-2026-11-22`. The step-0 check classified Qwen3-8B as "question-specific (supports the task-success reading)": on yes/no the task-aware verifier accepts 82 points fewer of the agent's working first moves than the blind one [−88, −75], while on the anchored observation readout the gap is zero in every game, and that readout is competent. Under the pre-declared reallocation (commit `2a86683`, before any census episode) the observation readout became the primary run and yes/no a one-seed secondary. Primary (observation readout): success "no reliable difference", aware minus blind −1.2 points [−3.0, 0.0] (p 0.23); only 3 of 134 games differ and 84% of seed pairs play identical episodes, so the test could not have been significant; both key secondaries (override harm minus help, invalid actions in the first 10 steps) "no reliable difference"; obstruction check clear. Secondary (yes/no): success "no reliable difference" (−4.5 points [−11.2, +2.2]); invalid actions per episode "harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)", +1.12 [+0.33, +1.96], with the same label for both key secondaries. The route is Stage 1b's: under any concrete task the yes/no verifier rejects more working proposals, abstains more and rescues fewer failing ones; it remains better than no verifier. The swapped verifier accepted working real-goal proposals 19 points less often than the aware one, so success-level attribution is not identifiable. On identical states the real task raises acceptance of failing goal-object proposals under both readouts (+27.5 points observation, +3.3 yes/no). *(Updated 2026-11-24.)* Argmax rule (yes/no readout; when the verifier rejects the proposal, the highest-scored sampled alternative is executed whatever its sign; jobs 7253142–7253145, submitted after the Tier 1 analysis; analysis job 7295370 at `01cad81`): success 'no reliable difference', −0.7 points [−9.0, +7.5], with 29 of 134 games differing; invalid actions per episode 'no reliable difference', −0.07 [−0.85, +0.71]; invalid actions in the first 10 steps 'harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)', +0.23 [+0.10, +0.37]. The design's prediction for this rule (a loss of 2 to 10 points) is not supported. The Llama loop was not run. *(The verifier uses Qwen3's thinking-off format with the empty think block, as Stage 1a did; the results note, section 10.3, says what that does and does not affect.)* **Read:** `notes/optionc_results.md` (metrics `notes/optionc_metrics.json`, `notes/optionc_step0_metrics.json`); design `notes/optionc_design.md` and review `notes/optionc_review1.md`.
>>>>>>> origin/optionc-results-update

### 8.17 The mechanism battery (exploratory)

**Question.** "Why does naming the action's object push predictions toward success?" **Design.** The arms of Section 3.5 on a stratified sample of 1,000 RQ2 items plus all 514 step-0 items, Qwen3-8B and Llama-3.1-8B (primary) and Mistral-7B (descriptive), in both formats. Eight primary contrasts P1 to P8 (for example P4 is described minus swapped, "need without name"; P8 is negated minus affirmed), each read through the zones and categories of Section 6.7 against a prediction table for the candidate accounts: task as evidence, referent salience, wishful prediction, policy blending and lexical overlap, plus a position account and an exploratory "which part of the state is checked" account. A secondary quantity M, the mention-associated residual, averages the negated and affirmed lines and subtracts the swapped one. Frozen at tag `mechanism2-design-frozen-2026-11-22`. **Status.** Scored and analysed. By the frozen rules the verdict for all three models is "no single account": each candidate account is contradicted as the sole explanation, so the effect is a mixture or an account not on the list. *(Updated 2026-11-23: this entry said the results were still being written on a branch; the results and the deep-dive are merged into `main`.)* The models differ in what moves them. For Qwen3-8B, adding "Ignore the {obj}." to the swapped task reproduces the whole effect (r 1.03 two-candidate), describing the object keeps most of it (r 0.84), and explicit claims about the object's location move predictions by about twice the effect in the stated direction (r −2.26); the user's hope barely matters. Llama-3.1-8B is closer to task as evidence (a bare mention gives about a third of the effect), and Mistral-7B is the one model whose reading follows the user's hope. **Deep-dive** (exploratory, post hoc, plan committed before computing): an independent recompute reproduces every frozen estimate; for Qwen the key contrasts keep their sign in all six task types and in at least 96% of games, and no contrast is carried by a few games or items; Llama's bare-mention contrast is smaller and less uniform; the one fragile key number is Qwen's two-candidate P7, a near-zero mean made of opposite parts. **Read:** `notes/mechanism2_design.md`, `notes/mechanism2_results.md`, `notes/mechanism2_dx.md`, metrics `notes/mechanism2_metrics.json` and `notes/mechanism2_dx_metrics.json`; code `wf/mech2.py`.

### 8.18 The interpretability pilot (branch `interp`)

**Question.** "Where does the task enter the prediction?" Under a relevance-at-answer-time account the task should enter late, at the answer position, leaving the model's internal representation of the state unchanged; under a belief-change account it should change that representation in middle layers at the history tokens. **Design.** A minimal pilot on Qwen3-8B with the yes/no readout only (the logit difference of Yes and No at the answer position), with activation patching on 100 calibration items and direction ablations on 125. Nothing past the first stage runs unless a reproduction gate passes: the mean effect within 0.15 nats of the logged 2.80, at least 99% of items reproduced and matching item hashes. The pilot is a calibration study whose output names no hypothesis; its variances set the sample sizes of a full pilot. **Status (updated 2026-11-24).** Done, as a calibration run with no hypothesis verdicts. The analysis ran (job 7295359) after an analysis-only deviation made before any output was read: the answer-position probe's success direction comes from the training fold's determined successes and failures, and the analysis trusts the GPU stages' freeze commit. Reproduction gates pass. *(revised 2026-11-25, to match the independent audit in `notes/interp_results.md` section 13; no reported number changes.)* The audit classifies the pilot as: replication of the behavioural effect achieved; causal localization partial; computational mechanism and semantic interpretation not achieved. Patching finds the task's effect at the action line's object word by layers 11–18 and at the answer position from layers 19–22. That route is largely forced by prompt order, no edge of it was tested and the single-group shares are not additive, so it says where the difference is used, not that it flows along a path or what it means. The history's share is low over all items (0.031) mainly because most histories do not name the object; on later items whose history does (7.0% of later failures, 6.2% of all failures) it is 0.40 [0.23, 0.72]. Aligned-against-swapped patching cannot detect the history being used as evidence, so a low share does not mean the history is unused. Ablation of one direction at the answer position reads "after arrival" on later steps only with percentile intervals (layer 24's lower bound 0.566 against 0.6) and "inconclusive" with normal or basic intervals; it is inconclusive at step 0 and pooled. The 0.44 cosine between the task direction and the neutral success direction is confounded with object mention (88% of determined successes name the object in their history, against 7% of later failures) and was read against the wrong null; the steering and removal rows that would separate a generic success feature from a relevance signal are empty by construction. A stage-0 mention control the note first left out: after step 0, a sentence asserting that the object is in the room, added to the swapped task, raises the yes/no log-odds about as much as the goal (2.26 against 2.29 nats; 2.21 against 7.70 at step 0); because the sentence asserts existence, it cannot separate a lexical account from the task as evidence that the object exists. The full pilot would need about 400 ablation failures after step 0 and about three times the object-mention items the other fold holds. **Read:** `notes/interp_results.md`, `notes/interp_design.md`.

---

## 9. Glossary

Terms are in alphabetical order. Each entry gives a plain definition and where the term is defined precisely.

- **A (asymmetry).** False-success rate minus false-failure rate of written predictions; positive means a lean toward predicting success. RQ1's measure. `wf/analysis/stats.py`, `asymmetry_A`; `notes/rq1_results.md`.
- **Accept-only rule; argmax rule.** Option C's decision rules: take the first sampled alternative the verifier accepts, or take the highest-scored candidate whatever its sign (Stage 1a's rule 2). `notes/optionc_design.md`; results in `notes/optionc_results.md`.
- **Aligned, swapped, neutral, task-blind.** The four basic task-line conditions: the real task; the real task with another object; "explore the room" (or its equivalent); no task sentence. Section 3.1; `wf/scoring/prompts.py`, `condition_task_line`; `wf/decisionsim.py`, `blind_messages`.
- **Anchored question.** A yes/no question that names the environment's actual replies, so No means a specific reply rather than "not useful for the task". Section 4.4.
- **Arm.** One prompt condition in a study; used interchangeably with "condition".
- **AUC.** Probability that a random working action scores above a random failing one; 0.5 is chance. Section 4.7.
- **Cluster.** The unit treated as independent in the statistics: game, goal pair or user. Section 6.1.
- **Competence gate.** A rule, fixed before data, that a model must pass for its effect to be interpreted. Section 4.7; `wf/families.py`, `competence_gate`.
- **Confirmatory, exploratory, post hoc.** Fixed before data and decided by a rule; designed before data but descriptive; computed after the results were seen. Section 7.1.
- **Couple.** In TextCraft, two items with the same history, pair, class and role, one acting in each goal's tree. `notes/textcraft_design.md` §12.
- **Determined failure; determined success.** An action whose failure (or success) the history alone proves. ALFWorld: `_failure_determined`, `notes/rq2_results.md`; TextCraft: `wf/textcraft/checker.py`, `decide`; τ-bench: `wf/tau/checker.py`.
- **ΔF.** In the decision simulation, the failure rate of the chosen action with a task-aware verifier minus that with a task-blind verifier, in points. `wf/decisionsim.py`.
- **ΔL.** The mean aligned-minus-swapped difference in a readout, in nats. Section 5.1.
- **Dn, Dc, Λ, P, T.** TextCraft round 2's naming term, completion term, intermediate lift, within-item penalty and timing contrast. Section 5.6; `notes/textcraft_r2_design.md` §6 (branch `textcraft-r2`).
- **Encounter (state).** In Stage 1b, a decision point whose candidate list contains at least one regime candidate; "strict" if the history also does not act on the goal object. `notes/decisionsim_agent_design.md`.
- **Failing minus witness (failure-specificity).** The failing call's goal effect minus the witness's, per item; positive means specific to failing calls. Section 5.3.
- **First token; first_mass; c_hat.** The readout on the first token of each reply; the probability the two first tokens cover; the first-token log-odds minus the full log-odds, an estimate of the wording penalty. Section 4.3; `wf/scoring/logodds.py`, `ScoreResult`.
- **fix<N> checkout ("fix36-style").** A fresh, numbered clone on the cluster (`text-wm-fix<N>`) at one commit, used for one batch of jobs so that resumes never mix commits. Section 7.9; `notes/reproducing.md` §2.
- **Flip; sign change; No-to-Yes.** Items crossing zero between swapped and aligned: one direction on the full score, either direction on the first token, or the yes/no answer changing from No to Yes. Section 4.6.
- **Freeze tag.** An annotated git tag marking the commit at which a design and its analysis code were fixed before scoring. Section 7.3.
- **Gate (Option C gate).** Stage 1a's pre-registered rule deciding whether a closed-loop study is worth doing, with six ordered categories. `notes/decisionsim_design.md`, "Option C is worth doing if".
- **Golden render.** A stored exact copy of a rendered prompt that a test compares against. Section 7.7.
- **H1 to H7.** These labels mean different things in different studies. In the filed study, H1 (with parts H1a and H1b) is RQ1's hypothesis, H2 the RQ2 test and H3 the RQ3 interaction. In the mechanism battery, H1 to H7 are candidate accounts: no state tracking, task as evidence, referent salience, wishful prediction, policy blending, lexical overlap or position (H6_lexical, H6_position), and which part of the state is checked (H7_check, exploratory) (`notes/mechanism2_design.md` §1 and §11.1). The interpretability pilot's H1 to H6 parallel these but with H2 as belief change and H3 as relevance at answer time (`notes/interp_design.md`). TextCraft round 2 uses H-labels for descriptive heuristic readouts (§9 of its design). Wave B's predictions are B1 to B3, and round 3's hypothesis is H3-P.
- **Holm family.** The fixed set of tests corrected together by Holm's procedure; missing tests enter with p = 1. Section 6.4.
- **ICE.** The PACE ICE cluster on which GPU jobs ran. Section 7.9; `pace/SETUP_ICE.md`.
- **Inconclusive.** Neither significantly different from zero nor equivalent within the margin; not the same as "no effect". Section 6.6.
- **Item.** One history plus one next action, with its true reply and candidates. Section 2.
- **Item hash.** A hash over what a score depends on, stored with each score so it cannot attach to a different item. Section 7.8; `wf/runs.py`, `_item_hash`.
- **M.** The mechanism battery's mention-associated residual: the average of the negated and affirmed lines minus the swapped line. `wf/mech2.py`, `MENTION_TERMS`.
- **Matched arm; mention pair; write minus mention.** τ-bench terms: a system message without the sentence about what the backend sees; task lines that name an order without requesting a write; the request effect net of the naming effect. Sections 3.4 and 5.3; `notes/tau_design.md`.
- **Menu.** In Stage 1a, the fixed set of candidate actions at a state from which the fallback chooses. Section 8.14.
- **Mock.** Simulator and fake-scorer mode for tests; its output is marked and refused by real runs. Section 7.8.
- **Mutant; killed; strict named-kill rule.** A deliberately broken copy of the code; caught by the tests; caught only if pytest ran, failed and named a failing test. Section 7.6.
- **Neutral AUC.** Competence AUC computed on scores under the neutral task line. Section 4.7.
- **P1 to P8.** The mechanism battery's eight primary contrasts. Section 8.17; `wf/mech2.py`, `PRIMARY`.
- **Readout.** The procedure that turns a model's response to a prompt into one number. Section 4.
- **Regime candidate.** A candidate action that is a determined failure on the task's object, not at a shared spot: the kind RQ2 scores. Section 8.15.
- **Relevance penalty.** The drop in predicted success that any concrete task puts on actions unrelated to it, relative to a neutral line. Section 3.2.
- **s; Δp; Δs; p_success.** The logistic of the two-candidate log-odds and its shift in points; not a probability of success. Section 4.2.
- **Shard.** One part of a run, split by game and run as its own job. Section 7.9.
- **Shared spot.** ALFWorld receptacles reachable from one position; items there are excluded. Section 2.1.
- **Share toward Yes.** The share of failures whose yes/no score is higher under the aligned than under the swapped task. Section 4.6.
- **Smoke.** A small real run read only for format, timing and gates, never for effects. Section 7.4.
- **STATUS.md.** A run folder's status file; its first line says PASS, FAIL or REPORTED. Section 7.9.
- **Step 0.** The first state of an episode, before the agent has acted; 514 of the RQ2 items.
- **Success tail.** The part of the two-candidate score after the success reply's first token; it carries about 59% of ALFWorld's two-candidate effect. Section 4.8.
- **TOST.** Two one-sided tests for equivalence within a margin. Section 6.6.
- **Twin (success twin).** The real environment's success reply for a failing action, obtained by running the action in a state where it works. Section 2.1; `wf/data/candidates.py`.
- **Two-candidate log-odds L.** Teacher-forced log-probability of the success reply minus that of the failure reply. Section 4.1.
- **Witness.** In τ-bench, a single write call that achieves the aligned task from the item's state, scored like the failing call as a control. Section 2.3; `wf/tau/items.py`, `achievable`.
- **Wording penalty.** The log-probability cost of guessing the success reply's exact wording, which lowers the two-candidate score below the model's belief in success. Section 4.1.
- **Word2World.** A project whose fine-tuned ALFWorld world models are scored as a comparison, and whose server and data define this repository's ALFWorld. Section 2.1.
- **Yes-share (mean σ) against share answered Yes.** The mean probability of Yes against No, as opposed to the share of items with Yes above No. Section 4.6.
- **Zones (N, Z, S, L) and categories (TRIVIAL, POS_LARGE, and others).** The mechanism battery's size bands for the ratio of a contrast to the main effect, and the labels for the run of bands an interval touches. Section 6.7.

---

## 10. A suggested reading order

**First, the results and the decision log.**

1. `README.md`: the question, the headline results and the status table of every study.
2. This guide, for definitions.
3. `notes/rq2_results.md`: the preregistered result, and its "Methods" section, which explains ALFWorld, item collection, determination, twins and swaps in full.
4. `notes/cross_model_summary.md`: every model scored so far, read together.
5. `notes/exploratory_results.md`: the follow-ups, including section 7's facts about the relevance penalty and the 1/64 grid.
6. `notes/textcraft_vs_alfworld.md`, section 1 for the readouts and section 2 for the success-tail finding (noting that its yes/no comparison is superseded by `notes/textcraft_results.md`).
7. `notes/rq1_results.md` and `notes/rq3_results.md`.
8. Per study, the design note and then the results note: families, TextCraft, τ-bench, decision simulation (file names in Section 8).
9. `notes/deviations.md`: the dated log of every change since filing. Read the section "Read before analysing or writing up" before using any data.
10. `notes/testing.md` and `notes/reproducing.md`: how the code is tested, which run and commit produced each result, and how to rerun.

**Then the code that implements the core.**

1. `wf/scoring/prompts.py`: the world-model prompt, the few-shot sets and task-line replacement.
2. `wf/scoring/logodds.py`: `LogoddsScorer` (`score_pair`, `score_one_token_pair`, `free_generate`) and `build_messages`.
3. `wf/data/transitions.py`, `wf/data/goals.py`, `wf/data/candidates.py`: item collection, determination, swaps and twins.
4. `wf/rq2.py`, then `wf/rq1.py` and `wf/rq1v.py`: the confirmatory scoring and the yes/no question.
5. `wf/analysis/stats.py`: bootstrap, permutation, Holm and the RQ2 verdict.
6. `wf/runs.py` and `wf/util.py`: run folders, resume, gates, item hashes and mock guards.
7. For the other environments, `wf/textcraft/` (start with `prompts.py`, `checker.py`, `analysis.py`) and `wf/tau/` (start with `prompts.py`, `items.py`, `pipeline.py`); for the later ALFWorld studies, `wf/families.py`, `wf/sizes.py` and `wf/decisionsim.py`.
8. `tests/golden/` and the tests named in Section 7.7, to see what each prompt must look like.

**Where data and runs live.** Run output is written to `runs/<run_id>/` (one folder per shard for sharded runs) and is not tracked by git; the data are on the cluster and in an archive outside the repository (`notes/reproducing.md`, section 2). Each run's settings are in `wf/config/<run_id>.yaml`, and its job script in `pace/`. The metrics behind every reported number are copied into `notes/*_metrics.json`, with a provenance block naming the run, commit and GPU. Branch-only designs are read with `git show origin/<branch>:notes/<file>`.
