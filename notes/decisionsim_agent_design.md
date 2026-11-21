# Stage 1b: verification of the agent's own ranked candidates (design, 2026-11-20; amended 2026-11-21)

Exploratory. This note was written and committed before any data of this stage existed (commit 644a98d): no agent candidate list has been sampled, no candidate has been labelled by replay, and no verifier has scored any candidate, so no decision of this stage could have been computed. It was amended three times, each before any data of this stage: with facts from the model-free dry run (2026-11-20), after review 1 and after review 2 (both 2026-11-21). The text below is the amended design; every change is listed with its reason in "Amendments after review 1, before any data" and "Amendments after review 2, before any data" at the end, which quote the original rule wherever a rule changed. Every number quoted comes from archived runs finished before this design (the RQ3 episodes at `text-wm-fix11/runs/rq3`, commit ddeabc2, and the collection at `text-wm-fix17`), recomputed on CPU without any model. Code: `wf/decisionsim_agent.py`; runs `decisionsim_agent_sample`, `decisionsim_agent_label`, `decisionsim_agent_score_<model>`, `decisionsim_agent_analysis` and the model-free `decisionsim_agent_dryrun`; the precision simulation is `scripts/decisionsim_agent_precision.py`.

## Where this sits

The main finding (RQ2, replicated in the yes/no format) is that a prompted world model's predicted success for an action whose failure the history already settles rises when the agent's task names the action's object. Stage 1a (`decisionsim_design.md`, branch `decisionsim`) asks whether that bias changes decisions over fixed candidate menus built by our generators at 761 collected states. Those menus are mostly goal-object failures against a few random working actions, so Stage 1a measures **vulnerability**: how a verifier behaves when the dangerous actions are put in front of it.

Stage 1b asks the complementary question, which is the one practical-impact claims rest on: **how often does a real agent put such actions in front of the verifier, and when it does, does telling the verifier the task make the agent act on a failing action more often than not telling it?** The candidates here are the agent's own, sampled from its own policy at the states it actually visited. Stage 1b is therefore primary for any statement about practical impact, and Stage 1a is the mechanism-level complement.

**Scope of the claims.** The priority verifiers were chosen because they are susceptible: under the aligned task they turn many determined failures into Yes (below). Any result is therefore a statement about **susceptible verifiers**, not about world models in general, and a positive result for them says nothing about a verifier as confident as Qwen3-8B. The priority pair (Llama-3.1-8B, Mistral-7B) also differs from Stage 1a's co-primary pair (Qwen3-8B, Llama-3.1-8B): only Llama is primary in both stages, and Stage 1b's Qwen3-8B is a contrast, so the two stages' primary verdicts are not directly comparable for Qwen or Mistral.

## Question

At each decision point of a real ALFWorld agent, the agent proposes a ranked list of its own candidate actions and a world-model verifier screens them: the agent acts on its highest-ranked candidate that the verifier predicts will succeed. The primary question is whether the verifier that sees the agent's real task line (task-aware) leads the agent to a failing action more often than the same verifier with the task line removed (task-blind). The estimand is the difference in the share of decisions whose chosen action fails,

$$
\Delta_{\text{invalid}} = \mathbb{E}_s\left[\mathbf{1}\{a^{\text{aware}}_s \text{ fails}\} - \mathbf{1}\{a^{\text{blind}}_s \text{ fails}\}\right],
$$

where the expectation runs over the agent's decision points and $a^{v}_s$ is the action chosen at state $s$ with verifier $v$.

The **key secondary estimand** (amended) is the same difference restricted to the encounter states $E$, the states whose list contains at least one action of the kind RQ2 studies (a failure the history determines, acting on the task's object, defined below). The label is model-free and fixed before any verifier score:

$$
\Delta_{\text{invalid}}^{E} = \mathbb{E}_{s \in E}\left[\mathbf{1}\{a^{\text{aware}}_s \text{ fails}\} - \mathbf{1}\{a^{\text{blind}}_s \text{ fails}\}\right].
$$

It is the effect where the vulnerability can act at all, and it is the better-powered of the two when encounters are rare. The model-free **encounter rate** (the share of states in $E$) is reported alongside.

## What this is and what it is not

It is a one-step decision study on real agent states with the agent's real candidate distribution. It is not an end-to-end measure: nothing is executed after the choice, so a failed action's cost (one wasted step, "Nothing happens.") is not followed into the rest of the episode, and recovery is not modelled. That is Stage 2's job (branching at disagreements, out of scope here; see "How Stage 1b feeds Stage 2"). It is also one agent (the RQ3 Qwen3-8B ReAct agent, thinking off), one environment (ALFWorld through the AgentGym server), one verifier prompt format (the yes/no feasibility prompt), and one selection rule.

## States

**Source.** The 134 episodes of condition "none" in the confirmatory RQ3 run (`rq3`, four shards, commit ddeabc2; archived at `text-wm-runs-archive/text-wm-fix11/runs/rq3`). In those episodes the Qwen3-8B ReAct agent decoded greedily, saw its real task, its history and the admissible actions, and saw no predictions; the episode cap was 30 steps (`STEP_CAP`). 63 of the 134 episodes were won; 73 hit the cap. Each episode records, per step, the agent's action and the server's real reply, so every state is reconstructable by replaying the recorded actions from reset.

**Why these episodes and not the item collection.** The collection that Stage 1a uses stopped at 12 steps and ran its broad generator only at even steps or after a failing proposal, which selects odd-step states on the outcome of the agent's action. The RQ3 episodes have neither problem: they run to 30 steps (the agent's first-attempt goal-object failures occur at a median step of 20, `exploratory_results.md` section 7), and nothing in them was generated conditionally on an outcome.

**Sampling.** A census: every decision point of every "none" episode, that is, the state before each of the agent's 2,973 actions, in all 134 games. No state is selected on any outcome, and there is no step-parity schedule, so the odd/even artefact of the collection cannot arise. Games are dealt to shards by their position in the sorted list of games, so the assignment does not depend on any outcome either.

**Replay check.** Each game is replayed from reset on the real server along its recorded actions. After every step the server's reply must equal the episode's recorded reply. If it does not, the states after the mismatch are dropped and counted (`replay_mismatch`); the earlier states stay. The world model's history text of each state (the opening observation, then each action and reply) is saved at sampling time, and every later replay of the state (one per candidate, during labelling) must reproduce it exactly; a state where any replay differs is unusable and counted (amended). RQ2's full verification reproduced every one of 12,987 collected items, so the expected counts are zero.

**Weighting.** The primary estimand weighs every decision equally. Long failing episodes (the 73 that hit the cap) therefore carry more weight than short successful ones, which is what "the agent's decisions" means for this agent. Two pre-declared sensitivity readouts change the weighting: each game weighted equally (the mean of per-game rates), and only the first 12 steps, steps 0–11 (comparable with Stage 1a's states).

## Candidate generation and ranking

**Options considered.** Three ways to obtain the agent's ranked candidates were considered.

- Ask the agent for a ranked list. This changes the agent's prompt and therefore its behaviour, so the list would describe a different agent; models also tend to pad such lists with plausible but never-chosen actions, and the order is not calibrated to anything.
- Score a fixed set of actions (the admissible list plus free-form ones) by the agent's log-probability. ALFWorld's admissible-commands list holds almost only working actions, so the failing candidates would have to be written by us, which reintroduces Stage 1a's generator-chosen menu. There is also a technical problem: the agent writes a thought before its action, and the log-probability of an action given one particular thought is close to 0 or close to minus infinity depending on whether the thought already named it, so it does not measure the agent's preference. Marginalising over thoughts requires sampling anyway.
- Sample complete replies from the agent and rank the distinct actions by how often they are sampled. Every candidate is the agent's own, including its free-form actions outside the admissible list (which is where its failures come from: the greedy agent's 671 failing actions include 482 `go to` actions, mostly to where it already is, and 108 actions on the task's object). The frequency of an action among the samples is a Monte Carlo estimate of the agent's marginal probability of that action, integrated over its thoughts, which is exactly the "agent's own probability" a ranking should use.

**Choice: sampling, ranked by frequency, with the greedy action first; two lists per state.** The procedure at each state:

1. The prompt is the RQ3 agent's prompt for condition "none", built by the same function (`agent_react_messages`) from the replayed state: the current observation, the history of earlier observations and actions, and the admissible actions. The agent always sees its real task.
2. **Rank 1 is the greedy action**: the action the agent actually took at this state in the archived episode (in condition "none" this is its greedy proposal; no step of the 134 episodes had a format error). This makes the no-verifier configuration exactly the real agent, and keeps the state sequence on the agent's own path.
3. **Primary list (amended).** One request to vLLM with $N = 20$ completions at Qwen's recommended non-thinking settings: temperature 0.7, top-p 0.8, top-k 20, min-p 0; at most 256 new tokens, thinking off (as in RQ3). These are the settings a practitioner sampling this agent would use.
4. **Sensitivity list (pre-registered).** In the same job, a second request of 20 completions at temperature 1, top-p 1, no top-k limit, min-p 0: the model's own distribution, whose tail holds more free-form actions. Its candidates share the primary list's labels and verifier scores through a cache keyed by (state, normalised action), so only candidates new to this list cost replays and forward passes.
5. **Request settings.** Each request carries its own seed, the first 8 hex digits of `sha256("20260924|decisionsim_agent|<list>|<game>|<step>")` read as an integer, and sends every sampling parameter explicitly (also repetition penalty 1, presence and frequency penalties 0). vLLM is started with `--generation-config vllm`, so the model's `generation_config.json` cannot change any default, and each shard records the vLLM version (`/version`) and the exact request settings in its metrics. The sampled replies are saved; the saved lists are the data (vLLM sampling repeats only on the same hardware and version).
6. **Parsing.** Each reply is read with the RQ3 parser (`parse_react_action`). A reply without an "Action:" line is a format error: it is counted and contributes no candidate (it is not mapped to `look`, because the agent did not propose `look`). Actions are normalised by collapsing whitespace, lower-casing and dropping a trailing full stop; the normalised string is the candidate and the one executed (rank 1 keeps its recorded string). The number of sampled actions that normalisation changed is recorded per list (amended after review 2).
7. **Deduplication and ranking.** Distinct normalised actions other than the greedy action are ranked by their count among the list's 20 samples, most frequent first; equal counts are ordered by a seeded hash of the action, the smallest of `sha256("20260924|<game>|<step>|<action>")` (the tie rule of Stage 1a, blind to outcomes).
8. **Length.** Each list keeps rank 1 and at most 7 sampled actions ($k_{\max} = 8$). The number of distinct actions cut off is recorded per state; with 20 samples, a cut-off action was sampled at most twice.
9. **Drift diagnostic (amended).** One fresh greedy completion (temperature 0) per state, on the same prompt. Its agreement with the archived greedy action is reported, with the number of states where the archived greedy action was never sampled in each list. Fixed now: if agreement is below 90%, the lists are described as coming from a drifted sampler (the agent served now does not reproduce the archived agent), and the results are read with that caveat. Rank 1 stays the archived action either way.

## Outcome labelling

Each distinct candidate of a state (the union of its two lists, primary first) is executed once on a fresh replay of its state on the real server (reset, the recorded actions, then the candidate), as the item collection did (`replay`, `step_counterfactual`). A candidate **fails** when the reply is exactly "Nothing happens."; any other reply is a success. A candidate whose step raises a server error is dropped and counted. Rank 1's replayed reply must equal the episode's recorded reply, and every replay must reproduce the saved history text; otherwise the state is unusable and counted.

Every candidate is also labelled with:

- whether it acts on the task's object (`names_object`, whole word, with the goal object parsed from the task line);
- whether a failure is **determined by the history** (`context_determined_label`, the RQ2 rules, applied to the world model's history text of the state: the opening observation, then each action and reply), with the deciding fact (`determination_reason`);
- whether it is at a shared spot (`at_shared_spot`, from the scene scan, never shown to a model);
- whether it is in the state's admissible list (free-form or not), its verb, and its rank and sample count in each list;
- whether the history acts on the goal object (`interacted_objects`), so the strict, RQ2-like subset can be formed;
- a **progress event** flag (below).

A **regime candidate** is a failure that acts on the task's object, is determined by the history and is not at a shared spot: the kind of action RQ2 scores. A **strict regime candidate** additionally has a history that does not act on the goal object, as RQ2's population required. An **encounter state** has at least one regime candidate in its (primary) list.

**Progress (descriptive only; amended).** ALFWorld's expert is not reachable through the AgentGym server (`on_expert_path` is None in every RQ3 step), so agreement with an expert's next action cannot be computed, and the code base has no PDDL distance. A descriptive progress event is defined from the candidate's real reply instead: the candidate succeeds and either wins the game, picks up an instance of the goal object, moves an instance of the goal object to a receptacle of the task's target type, applies the task's own transformation (heat, cool or clean, whichever the task asks for) to an instance of the goal object, turns on the lamp while holding an instance of the goal object (look-at tasks), or reveals an instance of the goal object that the history has not mentioned before. An action that already worked earlier in the history is never a progress event (repeats are dropped). Progress readouts never enter a verdict.

**Swap object.** For attribution, each state gets one swap object, drawn with a per-state seed (`20260924|decisionsim_agent|<game>|<step>`) by the rules of the collection's swapped goals (`swapped_goal`, `swap_feasible`): an object type in the room, other than the goal, that could do the task by the game's own flags, is not named by any candidate at the state, is not acted on in the history, and has no instance at any candidate's receptacle at reset. States without one (expected mostly in the nine games where the collection found no swap object) have no swapped verifier.

## Verifiers

Every verifier is the yes/no feasibility prompt of the earlier runs (`rq1v`'s `yn_*` arms: the world-model system text, the three frozen few-shot examples of set 1, the history, the action and the fixed question "will this action succeed, or will the environment reply "Nothing happens."?"). The score is the log-odds of the single tokens "Yes" and "No",

$$
L = \log P(\text{Yes}) - \log P(\text{No}),
$$

rounded to 1/64 nat (the bf16 resolution) before any comparison; the verifier **accepts** a candidate when the rounded $L$ is above zero. Only the task line of the query's history differs between verifiers; the few-shot examples are identical in every arm, as in every earlier run.

- **Task-aware:** the history as it is, with the agent's real task line.
- **Task-blind:** the same prompt with the query's task line deleted (the line and the newline before it, so the history reads as the room description followed by the first action). Nothing else changes. The deletion carries Stage 1a's guards (amended): exactly one task line, equal to the real task line, inside the first observation before the first action, never the first line; otherwise the prompt is refused. A test checks that the blind prompt equals the aware prompt with exactly that line removed. **Caveat (as in Stage 1a):** the few-shot examples keep their own task lines, about other games, while the blind query has none; the blind arm is therefore "no task for this episode in a prompt whose examples show tasks", not a prompt format that never mentions tasks.
- **Swapped (attribution):** the real task line with the goal object replaced by the state's swap object (`make_swapped_task_line`).
- **Neutral (secondary, priority models only; amended):** the task line "Your task is to: explore the room.", as in Stage 1a and the earlier runs. It links the two stages and never decides anything; it is not scored for Qwen3-8B.

Two configurations need no model: **none** (the agent's rank-1 action, that is, the real greedy agent) and **simulator oracle** (the verifier knows every candidate's true outcome). The agent always knows its real task in every configuration; only the verifier's prompt changes.

**Models.** Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 are the priority verifiers, because under the aligned task they turn many determined failures to Yes (wave A: 16.5% and 37.5% of failing items flip from No to Yes between the swapped and aligned task, against 0.2% for Qwen3-8B; `families_results.md`). Qwen3-8B (thinking off) is the contrast. All load at the pinned revisions in `wf.constants.REVISIONS`. Prompts are built exactly as before: `sizes.arm_messages` for Qwen and Llama, `families.family_messages` for Mistral (its native template placement, as in wave A), and Mistral gets wave A's Yes/No tokenization gate before any score. **Gate rule (amended):** if Mistral fails its Yes/No gate, it is not scored, it enters both Holm families with p = 1 and is reported as not interpreted; Llama then needs a raw p below 0.025 in each family.

**What is scored.** Every distinct candidate of a usable state under each arm of its model (aware, blind and swapped for all three; neutral for the priority models; swapped only where a swap object exists), with one exception fixed now: at a state with a single distinct candidate, every verifier must choose it, so only the blind arm is scored there (it is needed for the competence rule).

**Competence rule.** A verifier's decision results are interpretable only if, on this stage's candidates under the task-blind prompt, its AUC (accepting successes over failures, by $L$) is at least 0.70, it accepts at most 95% of failures and at least 5% of successes, the rule of wave A. A model that fails is reported but not interpreted; it stays in its Holm family with its own p.

## Selection rule

For a verifier $v$, the chosen action $a^{v}_s$ is the highest-ranked candidate that $v$ accepts. If $v$ accepts none, the agent acts on its rank-1 action (the verifier abstains, and the agent behaves as without one). A list of one candidate is chosen whatever the scores. The none configuration always chooses rank 1. The oracle chooses the highest-ranked candidate that truly succeeds, and rank 1 if none does. Candidates the server could not execute are dropped first; the list cuts of the sensitivity readouts ($k = 2$, $k = 4$) count the remaining candidates (amended).

## Outcomes and statistics

**Inference.** Every interval is a 95% game-clustered percentile bootstrap (2,000 resamples of games, seed 20260924); every p is a two-sided whole-game sign-flip permutation (1,000 permutations, seed 20260924), the tests used throughout the study. A sign-flip p cannot be smaller than $2^{1-g}$, where $g$ is the number of games whose net difference is not zero; at least 7 such games are needed for Holm's first threshold, 0.025 ($2^{-6} \approx 0.016$, while $2^{-5} \approx 0.031$). The number of such games and this floor are reported with every test.

**Primary (per priority model).** $\Delta_{\text{invalid}}$ over all usable states of the primary list, Holm-corrected over the two priority models at 0.05. The verdict, for an interpretable model (amended to add attribution, Stage 1a's rule):

- **toward failure, attributable to the real task:** Holm-adjusted p below 0.05, the interval above zero, and on the states with a swap object the aware-minus-swapped difference is positive and at least half of a positive aware-minus-blind difference;
- **toward failure, not attributable to the real task:** the first two conditions hold, the third does not (showing a task does it, but a task about another object does it nearly as much);
- **toward success:** Holm-adjusted p below 0.05 and the interval below zero;
- **no reliable difference:** otherwise.

Pre-declared: if fewer than half of the harm states (aware fails, blind succeeds) come from regime candidates, for example if the harm is driven by the agent's `go to` loops, a "toward failure" verdict is reported as **not evidence for the main finding**, whatever its attribution. The analysis reports the regime share of harm with every verdict, and in that case appends "(not evidence for the main finding: harm mostly from non-regime actions)" to the verdict text itself (amended after review 2; a share of exactly one half is not "fewer than half").

**Key secondary (per priority model; amended).** $\Delta_{\text{invalid}}^{E}$ on the encounter states of the primary list, with its own Holm correction over the two priority models and the same verdict categories (attribution read from the decomposition on the encounter states with a swap object).

**Secondary and descriptive (per model unless marked).**

- The chosen-failure rate $F_v$ under none, task-aware, task-blind, swapped, neutral and oracle; the share of chosen failures that are determined and that are regime candidates.
- The share of states where the aware and blind choices differ, split into harm (aware fails, blind succeeds), help (aware succeeds, blind fails), both fail and both succeed. Harm minus help equals $\Delta_{\text{invalid}}$ exactly; the analysis checks this identity. Harm is split by the kind of action the aware verifier let through (regime candidate, other goal-object failure, `go to`, other).
- **Attribution.** On the states with a swap object, the decomposition

$$
F_{\text{aware}} - F_{\text{blind}} = (F_{\text{aware}} - F_{\text{swapped}}) + (F_{\text{swapped}} - F_{\text{blind}}),
$$

where the first part is what naming the real task's object adds over a task of the same form about another object, and the second is what showing any task of that form does. Both parts get intervals and permutation p-values; the identity is checked. The same decomposition is reported on the encounter states.
- **Neutral (priority models).** $F_{\text{neutral}}$, aware minus neutral and neutral minus blind; never decisive.
- **Encounter (model-free).** The share of states with at least one regime candidate in the list, and with at least one strict regime candidate, each with a game bootstrap interval and its number of games; by rank, by step band (0, 1–11, 12–29), by whether the candidate is free-form; and the share of states where a regime candidate is **reachable**, that is, ranked above the first candidate that truly succeeds (a verifier that accepts it would choose it before any working action). Reported for both lists.
- $\Delta_{\text{invalid}}$ on the non-encounter states (the encounter and non-encounter parts, weighted by their shares, add up to the whole; checked).
- **Inert verifiers (amended).** Per arm, at states with more than one candidate: the fallback rate (the verifier accepted nothing) and the share of decisions that are rank 1. A verifier that almost always chooses rank 1 cannot change decisions whatever its bias.
- Verifier value: $F_{\text{none}} - F_v$ for each verifier, and the gap to the oracle $F_v - F_{\text{oracle}}$.
- Per-candidate acceptance rates by outcome, goal relevance and determination, per arm, and the aware-minus-blind shift in acceptance and in mean raw $L$; the AUC per arm.
- Progress (descriptive): the share of chosen actions that are progress events, per configuration, and the aware-minus-blind difference with its interval.
- Robustness of the primary: the leave-one-game-out range of $\Delta_{\text{invalid}}$ and the game whose removal moves it most.
- Sensitivity, pre-declared: game-weighted $\Delta_{\text{invalid}}$; steps 0–11 only; lists cut to $k = 2$ and $k = 4$; the fallback replaced by "the highest-$L$ candidate"; the temperature-1 list (overall and on its encounter states); the primary list restricted to rank 1 and candidates sampled at least twice; and a loop sensitivity in which a failing action counts only at the first state of its game where it is listed (later listings are removed).
- Bookkeeping: format errors per list, distinct actions per state, lists cut by $k_{\max}$, replay and history mismatches, candidates dropped for server errors, states without a swap, fresh-greedy agreement.

## Order of runs and the futility rule (amended)

Sampling and labelling run first; they need no verifier. The model-free analysis (`decisionsim_agent_encounter`) then runs with no model scores and reports the encounter readouts. Every scoring run refuses to start until that analysis has finished on the same label run; an explicit override (`run_order_override`, with a reason) is allowed and is logged with the scores (amended after review 2).

**The futility rule (corrected after review 2).** A state can show an aware-minus-blind difference only if one choice works and the other fails, so an encounter state can contribute regime harm, or any difference to the conditional estimand, only if its primary list also holds at least one working candidate. Fixed now: **if the encounter states that also list a working candidate fall in fewer than 7 games, the conditional estimand cannot reach significance and no regime harm can be significant** (the sign-flip floor above), and this is stated in advance of any score. The rule is a necessary condition: it can show futility, never promise significance, and it makes no claim about the overall primary, whose non-encounter states can differ for other reasons (which would not be evidence for the main finding). When it holds, the encounter rates are the result of Stage 1b and whether to score the verifiers is decided then; otherwise the three verifiers are scored (priority models first) and the full analysis is run. The number of games with a reachable encounter (a regime candidate ranked above the first working candidate) is kept as a descriptive readout; it is not the rule, because a verifier that rejects a working rank-1 action can still choose a regime candidate ranked below it (review 2's counterexample: one such state in each of 8 games, none reachable, gives a difference of 1 per state, 8 non-zero games and p below 0.025).

## Predictions (logged before any data of this stage)

These are left unchanged after review 1. They were written for the temperature-1 lists, which are now the sensitivity list; the primary (tempered) lists should, if anything, hold fewer free-form actions, so prediction 1 applies to the primary list as an upper-leaning guess.

1. **Encounter.** At least one regime candidate appears in the list at between 8% and 30% of states (central guess 15%); most regime candidates are free-form (outside the admissible list); a regime candidate is reachable at under half of the encounter states. The strict, RQ2-like encounter rate is much lower: 0.5–5% (amended before any data of this stage; the first version said 3–15%, see "Facts added before any data of this stage").
2. **Qwen3-8B (contrast).** $|\Delta_{\text{invalid}}|$ below 0.5 points, interval including zero: it accepts almost no determined goal-object failure under the aligned task (0.2%).
3. **Llama-3.1-8B.** A positive point estimate between 0.2 and 1.5 points, concentrated in regime candidates the aware verifier accepts; it may not reach significance (see "Precision").
4. **Mistral-7B-v0.3.** A positive point estimate between 0.5 and 3 points, and the largest decision-change rate of the three; its task-blind verifier will already accept many failures, so its verifier value over none will be the smallest of the three and may be negative.
5. **Attribution.** For Llama and Mistral, the aware-minus-swapped part is positive and is most of the aware-minus-blind difference. The swapped-minus-blind part can have either sign: a stated task lowers acceptance of task-irrelevant actions (section 7 of `exploratory_results.md`), which includes the agent's failing `go to` loops (help) as well as its working `go to` moves (fewer accepted, more fallbacks down the list).
6. **Verifier value.** For Qwen and Llama, every verifier lowers the chosen-failure rate below the greedy agent's 22.6%; the oracle's rate is below 3% (a state where no listed candidate works is rare).

## What each result would and would not show

- **Toward failure, attributable,** for a priority model, with harm mostly from regime candidates: at real decision points of a real agent, telling a susceptible verify-then-act world model the task makes the agent act on failing actions more often, at the rate estimated, and naming the task's object is what does it. It would not show a loss of task success (one wasted step can be recovered), would not transfer automatically to agents that search more broadly or use the world model's imagined outcomes over several steps, and is tied to this prompt format and selection rule. It is what would justify Stage 2.
- **Toward failure, not attributable,** or driven by non-regime harm: showing a task moves this agent's decisions toward failure, but not through the mechanism RQ2 found; it is not evidence for the main finding.
- **No reliable difference with a narrow interval:** the bias that RQ2 and Stage 1a measure rarely reaches this agent's decisions under verification. The overall interval bounds harm per decision: a bound of about one point over all decisions corresponds to about $1/e$ points among encounter states, where $e$ is the encounter rate (for example about 7 points if 15% of states are encounters), which is far from a bound of zero where the vulnerability can act. The conditional estimand is the readout for that question, and its interval must be read before calling the bias harmless where it applies. Together with a high Stage 1a vulnerability, a null would say the risk is real in scores but small per decision in this deployment; it would not rule out harm for planners that rank by predicted success, generate synthetic experience, or roll out imagined trajectories.
- **Toward success:** the task line helps the verifier at real decisions, for example by rejecting the agent's own looping `go to` actions; the swapped decomposition says whether that comes from the real task or from any task.
- **Encounter rate alone (model-free):** how often a sampling agent proposes the vulnerable actions at all; this is a fact about the agent, independent of any verifier. Under the futility rule it may be the stage's only result.
- **A verifier that fails the competence rule or its gate:** its numbers describe that model only and support no claim.

## Precision (from archived data, no model)

**Expected encounter.** The greedy agent's own action is a regime candidate (a determined goal-object failure) at 61 of the 2,973 states, 2.1%, in 15 games (amended: the first version gave 3.6% here, which counted all 108 goal-object failures, determined or not). This is an upper bound for rank 1, since the shared-spot exclusion needs the scene scan and could not be applied to the archived episodes. For the sampled alternatives, the closest archived evidence is what the same agent chose instead when told that its first choice would fail, in the other RQ3 conditions (dropped predictions, the agent's re-picks, recomputed on CPU):

| Situation in RQ3 | Re-picks | Re-pick fails | Re-pick is a goal-object failure | Determined goal-object failure |
| --- | --- | --- | --- | --- |
| Failing proposal shown as failing, perfect foresight | 248 | 20% | 12.5% | 4.0% |
| Failing proposal shown as failing, corrupted condition's true steps | 1,268 | 18% | 10.6% | 3.8% |
| Working proposal shown as failing (false failure) | 86 | 28% | 15.1% | 7.0% |
| Failing proposal, "not valid" verdict only | 244 | 18% | 4.9% | 2.5% |

So the agent's second choice is a goal-object failure about 5–15% of the time, and a determined one about 2.5–7%. If $m$ distinct sampled alternatives each had that chance independently, the share of states with at least one determined goal-object failure would be about 7–15% for $m = 2$ and 12–27% for $m = 4$ (rank 1 included). The independence assumption overstates the count when alternatives repeat one idea, and the re-pick prompt (with "Expected outcome: Nothing happens.") is not the sampling prompt, so these are rough; they put the expected number of encounter states somewhere between about 200 and 800 of 2,973. Because the archived encounters sit in a few long failing games, the number of games they fall in matters as much as their number.

**Power (amended after reviews 1 and 2; `scripts/decisionsim_agent_precision.py`, 300 replications, run on the archived episodes and the dry-run counts). Every power quoted is an upper bound:** the simulation has no between-game variation in the per-encounter effect, no loops, and no dependence between a game's states beyond their shared game. The simulation keeps the real 134 games and their state counts. Encounter states are placed in a fixed number of games, drawn with weights favouring long failing games (state count times one plus failing greedy actions), as the archived encounters are; among encounter states a difference is harm with probability $h$ and help with probability $h'$; every other state changes outcome in either direction with probability 0.5% each (symmetric background). "Power" is the chance that the interval is above zero and the permutation p is below 0.025, Holm's first threshold (the bar for one priority model when the other is null). Overall is the primary estimand, conditional the key secondary.

| Encounter scenario | Harm, help among encounters | Encounter states | Overall: true size, half-width | Overall power | Conditional: true size, half-width | Conditional power |
| --- | --- | --- | --- | --- | --- | --- |
| Rank 1 only: the 61 archived states in 15 games | 10%, 2% | 61 | 0.16 points, ±0.40 | 0.04 | 7.8 points, ±7.9 | 0.06 |
| Rank 1 only: the 61 archived states in 15 games | 20%, 3% | 61 | 0.34 points, ±0.47 | 0.15 | 16.7 points, ±10.4 | 0.44 |
| 15 games, about 15 each | 10%, 2% | 223 | 0.57 points, ±0.54 | 0.33 | 7.7 points, ±3.9 | 0.82 |
| 15 games, about 15 each | 20%, 3% | 223 | 1.24 points, ±0.80 | 0.90 | 16.7 points, ±5.2 | 1.00 |
| 30 games, about 8 each | 10%, 2% | 240 | 0.64 points, ±0.51 | 0.53 | 8.0 points, ±4.1 | 0.91 |
| 30 games, about 20 each | 10%, 2% | 590 | 1.59 points, ±0.77 | 0.99 | 8.0 points, ±2.6 | 1.00 |
| 30 games, about 20 each | 20%, 3% | 590 | 3.38 points, ±1.26 | 1.00 | 17.0 points, ±3.4 | 1.00 |
| 60 games, about 8 each | 10%, 2% | 476 | 1.28 points, ±0.61 | 0.96 | 8.0 points, ±2.9 | 1.00 |
| 60 games, about 8 each | 20%, 3% | 476 | 2.71 points, ±0.84 | 1.00 | 17.0 points, ±3.8 | 1.00 |

More background change (amended after review 2). The rows above assume 0.5% background change per direction. Real verifiers may change many outcomes outside the encounters (for example by rejecting working `go to` actions), and encounter states may carry such changes too. The same scenarios with more background (per direction, outside encounters and, where stated, inside them):

| Encounter scenario (harm 10%, help 2%) | Background outside / inside encounters | Overall: half-width | Overall power | Conditional: true size, half-width | Conditional power |
| --- | --- | --- | --- | --- | --- |
| 15 games, about 15 each | 2% / 0% | ±0.80 | 0.14 | 7.7 points, ±3.9 | 0.82 |
| 15 games, about 15 each | 5% / 0% | ±1.15 | 0.10 | 7.7 points, ±3.9 | 0.82 |
| 15 games, about 15 each | 2% / 2% | ±0.83 | 0.16 | 8.1 points, ±4.7 | 0.67 |
| 15 games, about 15 each | 5% / 5% | ±1.18 | 0.10 | 8.1 points, ±5.6 | 0.49 |
| 30 games, about 8 each | 2% / 0% | ±0.78 | 0.22 | 8.0 points, ±4.1 | 0.91 |
| 30 games, about 8 each | 5% / 0% | ±1.12 | 0.09 | 8.0 points, ±4.1 | 0.91 |
| 30 games, about 8 each | 2% / 2% | ±0.81 | 0.18 | 7.8 points, ±4.8 | 0.69 |
| 30 games, about 8 each | 5% / 5% | ±1.16 | 0.10 | 7.8 points, ±5.6 | 0.53 |
| 30 games, about 20 each | 2% / 0% | ±0.94 | 0.89 | 8.0 points, ±2.6 | 1.00 |
| 30 games, about 20 each | 5% / 0% | ±1.21 | 0.56 | 8.0 points, ±2.6 | 1.00 |
| 30 games, about 20 each | 2% / 2% | ±1.00 | 0.82 | 8.1 points, ±3.0 | 1.00 |
| 30 games, about 20 each | 5% / 5% | ±1.31 | 0.51 | 8.1 points, ±3.6 | 0.96 |

The overall estimand's true size is unchanged by symmetric background (about 0.6 points in the first eight rows and 1.6 in the last four) but its interval widens quickly, so with a few percent of background change its power falls to 0.1–0.2 in the archive-suggested scenarios. The conditional estimand is untouched by background outside encounters and loses power only to background inside them.

Readings. With the encounters the archive suggests (about 200–250 states in 15–30 games) and a modest per-encounter effect (net 8 points), the overall primary has power of only about 0.3–0.5, while the conditional estimand has about 0.8–0.9; if the lists add almost no encounters beyond rank 1, neither can detect anything, which is what the futility rule anticipates. The overall estimand reaches high power only when encounters are spread over many games or the per-encounter effect is large, and less if verifiers change many outcomes elsewhere. A null on the overall estimand is therefore weak evidence by itself, and the conditional estimand is the readout for whether the bias matters where it can act. The same script prints the sign-flip floor: 7 games with a non-zero net difference are needed for p below 0.025 and 6 for p below 0.05.

## Cost

Estimates, not timings; updated after review 1 for the second list, the drift diagnostic and the neutral arm.

- **Agent sampling** (`decisionsim_agent_sample`, GPU, vLLM and the ALFWorld server): per state, two requests of 20 completions and one greedy completion, on prompts of about 1–4 thousand tokens with replies of about 50–100 tokens; one replay walk per game. About 3–5 GPU-hours on an A40 or L40S, in four shards (4 h jobs).
- **Labelling** (`decisionsim_agent_label`, CPU and the ALFWorld server, no GPU): one fresh replay per distinct candidate of the two lists plus one scene scan per game. With 4–9 distinct candidates per state, 12,000–27,000 replays; at about 1,300 replays per CPU-hour, 9–21 CPU-hours, in eight shards that restart the server (it leaks memory on every reset).
- **Verifier scoring** (`decisionsim_agent_score_<model>`, GPU), only after the futility rule: every distinct candidate under four arms for the priority models and three for Qwen (single-candidate states: blind only). That is 45,000–100,000 single-token passes per priority model and 35,000–80,000 for Qwen; at 0.15–0.25 s per pass (histories to 30 steps), 2–7 GPU-hours per priority model and 1.5–5.5 for Qwen, 5.5–19.5 for the three, in two shards per model (6 h jobs).
- **Total:** about 9–25 GPU-hours (central estimate 15) and 9–21 CPU-hours; under the futility rule, sampling alone (3–5 GPU-hours) may be all the GPU this stage uses. A cut-off job resumes from its saved work.

## Risks

- **Sampling quality and seeding.** An engineering gate, fixed now and independent of any outcome (amended): the sampling run's smoke job (two train games, `decisionsim_agent_sample_smoke`, with the seed check on) must show format errors in under 25% of the samples of each list and no replay mismatch, at every smoke state whose first 20 replies are not all identical, a second request with another seed must return a different sequence of replies (a direct check that the per-request seed drives the draws), and at least one smoke state must be non-degenerate, so the check can show something. Otherwise the run stops for a design amendment before the census. Two readouts are required with the gate (amended after review 2): the share of replies cut off at 256 tokens in each list, and the fresh greedy agreement (with the 90% drift rule). The smoke run also repeats the primary request with its own seed and reports whether the replies are reproduced; this is not gated, because vLLM repeats a seeded request only on the same hardware, version and batch, and the saved replies are the data either way.
- **Nondeterminism.** vLLM sampling is reproducible only on the same hardware and version; the saved replies are the data, and a resume never resamples a saved state.
- **The census is weighted toward long failing episodes,** and their late states are often loops; the game-weighted, steps-0–11 and first-occurrence readouts show how much this matters.
- **Rank 1 comes from the archived run,** not from a new greedy call. It is the agent's real action at that state, and its replay check ties the state to the episode; the samples are drawn now, on whatever GPU the job gets, and the fresh greedy completion measures how far the served agent has drifted.
- **Server errors on free-form actions** are dropped and counted; a large count would be reported as a limitation.
- **Swapped verifier coverage** is smaller than the census (no swap object in some games); the decomposition uses only states with one.
- **The blind prompt keeps task lines in its examples** (caveat above).
- **Long prompts.** Agent prompts at step 30 fit vLLM's 8,192-token limit in RQ3; verifier prompts are shorter than the agent's.

## How Stage 1b feeds Stage 2

Stage 2 (not designed here) would branch real episodes at decision points where the aware and blind verifiers choose different actions, continue each branch, and measure the end-to-end cost. Stage 1b supplies, per model, the list of disagreement states with their harm, help, both-fail and both-succeed labels, the harm and help rates, and both estimands with their intervals. A natural gate for Stage 2, to be fixed in its own design, is that a priority model reaches "toward failure, attributable to the real task" here, with harm mostly from regime candidates and enough harm states to branch from (for example at least 30 in at least 15 games); a null here would make Stage 2 unnecessary for this agent, and "toward success" would turn Stage 2 into a test of a benefit.

## Open questions for the user

1. The task-blind definition now carries Stage 1a's guards and its output is identical to Stage 1a's `blind_history` on the census prompts (checked by review 1 on 8,919 prompts); whether to import Stage 1a's function after the branches merge.
2. Whether to score the neutral arm for Qwen3-8B too (skipped, as instructed).

## Facts added before any data of this stage (dry run, 2026-11-20)

The model-free dry run (`decisionsim_agent_dryrun`, no model and no server) rebuilt every one of the 2,973 states from the archived episodes, with each game's opening observation taken from the item collection, and labelled the rank-1 candidate (the greedy action) from its recorded reply. It found no state whose history lacked exactly one task line (rerun after review 1 with the guarded blind deletion: no problem either), and it reproduced the re-pick table above exactly. Three facts it adds:

- The greedy action alone is a regime candidate at 61 states (2.1%), in only 15 of the 134 games, at a median step of 22; none is at step 0, 2 are at steps 1–11 (0.15% of those states) and 59 at steps 12–29 (3.9%). Rank 1 therefore already contributes about 2 points of encounter (an upper bound: no shared-spot exclusion), and the encounters are concentrated in a few long failing episodes, which the game-clustered intervals account for.
- With the strict, RQ2-like requirement that the history never acted on the goal object, the greedy action is a regime candidate at only 4 states (0.13%, 4 games). By the time this agent proposes a failing action on the goal object, it has usually already handled that object. Prediction 1's strict range was therefore lowered from 3–15% to 0.5–5%; the main encounter range (8–30%) is unchanged.
- The greedy action is a progress event at 10.8% of states under the first progress definition, which sets the scale for the progress readout.

## Amendments after review 1, before any data (2026-11-21)

Review 1 passed the engineering with fixes and the science with fixes, the science fixes to be made before any sampling. Every change below was made before any data of this stage existed: no list had been sampled, no candidate labelled, no verifier score computed. Counts quoted were rechecked by the model-free dry run and the precision script on the archive.

**Science.**

1. **Sampling.** The primary list now uses Qwen's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20, min-p 0), 20 samples; temperature 1 untruncated is a pre-registered sensitivity list sampled in the same job, sharing labels and scores through a cache keyed by (state, normalised action). Original: "One request to vLLM with $N = 20$ completions at temperature $T = 1.0$, top-p 1.0, no top-k limit". Reason: the primary should describe the agent as it would be sampled in practice; the untempered tail is a sensitivity. A further sensitivity keeps only rank 1 and candidates sampled at least twice. The smoke gate's "no state whose 20 replies are all identical" was replaced by a direct seed check (a second seed must give a different reply sequence at non-degenerate smoke states).
2. **Conditional estimand.** $\Delta_{\text{invalid}}^{E}$ on the encounter states (a model-free label fixed before any verifier score) is a pre-registered key secondary with its own Holm correction over the two priority models, reported next to the primary. Reason: encounters are rare and concentrated, so the overall estimand is weakly powered for the question of whether the bias matters where it can act.
3. **Precision redone** with a committed script (`scripts/decisionsim_agent_precision.py`): encounters concentrated in 15–60 games weighted toward long failing games, a symmetric background, Holm's first threshold, overall and conditional power, and the sign-flip floor $2^{1-g}$ (at least 7 games with a non-zero net difference for p below 0.025). The first version's table assumed changes spread evenly over all games and overstated the power of the overall estimand.
4. **Rank-1 encounter figure corrected** to 2.1% (61 determined goal-object failures), an upper bound since the shared-spot exclusion could not be applied. Original: "so rank 1 alone gives about 3.6% encounter states".
5. **Null wording qualified.** An overall bound of about one point per decision corresponds to about $1/e$ points among encounter states; the conditional estimand is named as the readout for that question. Original: "a null would then bound the harm below about one point of decisions" and "A null with an interval inside about ±1 percentage point is reported as a bound on practical harm".
6. **Attribution in the verdict,** as in Stage 1a: "toward failure, attributable / not attributable to the real task" (aware minus swapped positive and at least half of a positive aware minus blind). Pre-declared that harm driven mostly by non-regime actions (for example `go to` loops) is not evidence for the main finding. Original: a single "toward failure" category.
7. **Loop sensitivity:** a failing action counts only at the first state of its game where it is listed; every test reports the number of games with a non-zero net difference, and the primary reports a leave-one-game-out range.
8. **Blind-prompt caveat** added (examples keep task lines, the query does not), and a **neutral arm** ("explore the room") for the two priority models as a secondary that never decides; skipped for Qwen3-8B.
9. **Inert-verifier readouts:** fallback rate and share choosing rank 1, per arm.
10. **Scope:** claims are for susceptible verifiers; the mismatch with Stage 1a's co-primary pair (Qwen3-8B, Llama-3.1-8B) is stated.
11. **Progress** is descriptive only; the transformation must match the task, repeats are dropped, and turning on the lamp counts only while holding the goal object. Original: "heats, cools or cleans an instance of the goal object", with no repeat or lamp condition.
12. **Predictions** left unchanged.
13. **Fresh greedy diagnostic:** one temperature-0 completion per state; agreement with the archived action and states where the greedy action was never sampled are reported; agreement below 90% means the lists are described as from a drifted sampler. Rank 1 stays the archived action.
14. **Cheaper path and futility rule:** sampling and labelling first; if the reachable encounter states fall in fewer than 7 games, the primary cannot reach significance, which is stated in advance; the encounter rates are then the result and verifier scoring is decided at that point. At states with a single distinct candidate only the blind arm is scored. (The futility rule was corrected after review 2; see below.)

**Engineering.**

15. The verdict reads the Holm-adjusted p (a test now checks raw p 0.03 and 0.5 giving Holm 0.06 and "no reliable difference" with the interval above zero).
16. The sampling-time history text is saved with each state, and labelling requires every replay to reproduce it; a state where any replay differs is unusable.
17. The blind deletion carries Stage 1a's guards (the deleted line equals the real task line and lies in the first observation before the first action); its output is unchanged.
18. The Mistral gate rule is pre-declared (p = 1 in both Holm families if it fails).
19. vLLM runs with `--generation-config vllm`, every sampling parameter is sent explicitly, and each shard records the vLLM version and the request settings.
20. k-cuts apply after candidates the server could not execute are dropped.
21. Tests now kill the reviewer's surviving mutants (R3, R4, R5, R10, R11, R13, R15, R17, R23, R24, R26, R28, R29; R30 is equivalent); they and mutants of the new code are in `scripts/mutants_decisionsim_agent.py`, with their killing tests in `testing.md`.

## Amendments after review 2, before any data (2026-11-21)

Review 2 passed the amended design with fixes and allowed the sampling and labelling smoke run. Every change below was made before any census list was sampled, any census label read or any verifier scored; the smoke run (two train games, no verifier) was cleared for submission at the same time and reads only the engineering gate.

1. **Futility rule corrected (must fix).** It now counts the games holding an encounter state whose primary list also has a working candidate, a necessary condition for any conditional difference or regime harm, and its claim is about the conditional estimand and regime harm only. Original (review 1): "if the reachable encounter states of the primary list fall in fewer than 7 games, the primary cannot reach significance". Reason: a verifier that rejects a working rank-1 action can choose a regime candidate ranked below it, so states with no reachable encounter can still differ (review 2's 8-game counterexample, now a test), and the overall primary can differ at non-encounter states. The reachable count stays as a descriptive readout.
2. **Conditional-estimand tests (must fix).** A synthetic test where the overall and conditional p-values, decompositions and state sets differ checks that each block reads its own.
3. **Seed check.** A same-seed repeat of the primary request is added and reported (not gated); the gate also needs at least one non-degenerate smoke state.
4. **Verdict text.** A "toward failure" verdict whose harm comes mostly from non-regime actions carries "(not evidence for the main finding: harm mostly from non-regime actions)"; tests at a regime share of exactly one half and at an interval whose lower bound is exactly zero.
5. **Precision.** Rows with 2% and 5% background change per direction, outside and inside encounter states, and every power is called an upper bound; the script is committed.
6. **Run order enforced.** Scoring refuses to start until the model-free analysis (`decisionsim_agent_encounter`) has finished on the same label run, unless given a logged `run_order_override`.
7. **Smoke readouts.** The share of replies cut off at 256 tokens and the fresh greedy agreement are reported with the smoke gate.
8. **Bookkeeping.** The labelling job script's replay-count comment is corrected for the two lists (12,000–27,000 replays), and the number of sampled actions that normalisation changed is recorded per list.
9. **Tests and mutants.** Tests for review 2's surviving mutants (F4, A5, B7, L5, S6; A14, L6, S1, S2, S5, S7, S10; S3; A2, A4; B4, B8, A1, A11); review 2's mutants (adapted) and mutants of the new code are in `scripts/mutants_decisionsim_agent.py`, with their killing tests in `testing.md`.

