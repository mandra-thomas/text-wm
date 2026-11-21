# Stage 1b: verification of the agent's own ranked candidates (design, 2026-11-20)

Exploratory. This note was written and committed before any data of this stage existed: no agent candidate list has been sampled, no candidate has been labelled by replay, and no verifier has scored any candidate, so no decision of this stage could have been computed. It was amended once, still before any data of this stage, with facts from the model-free dry run (last section). Every number quoted below comes from archived runs that were finished before this design (the RQ3 episodes at `text-wm-fix11/runs/rq3`, commit ddeabc2, and the collection at `text-wm-fix17`), recomputed on CPU without any model. Code: `wf/decisionsim_agent.py`; runs `decisionsim_agent_sample`, `decisionsim_agent_label`, `decisionsim_agent_score_<model>`, `decisionsim_agent_analysis` and the model-free `decisionsim_agent_dryrun`.

## Where this sits

The main finding (RQ2, replicated in the yes/no format) is that a prompted world model's predicted success for an action whose failure the history already settles rises when the agent's task names the action's object. Stage 1a (`decisionsim_design.md`, branch `decisionsim`) asks whether that bias changes decisions over fixed candidate menus built by our generators at 761 collected states. Those menus are mostly goal-object failures against a few random working actions, so Stage 1a measures **vulnerability**: how a verifier behaves when the dangerous actions are put in front of it.

Stage 1b asks the complementary question, which is the one practical-impact claims rest on: **how often does a real agent put such actions in front of the verifier, and when it does, does telling the verifier the task make the agent act on a failing action more often than not telling it?** The candidates here are the agent's own, sampled from its own policy at the states it actually visited. Stage 1b is therefore primary for any statement about practical impact, and Stage 1a is the mechanism-level complement.

## Question

At each decision point of a real ALFWorld agent, the agent proposes a ranked list of its own candidate actions and a world-model verifier screens them: the agent acts on its highest-ranked candidate that the verifier predicts will succeed. The primary question is whether the verifier that sees the agent's real task line (task-aware) leads the agent to a failing action more often than the same verifier with the task line removed (task-blind). The estimand is the difference in the share of decisions whose chosen action fails,

$$
\Delta_{\text{invalid}} = \mathbb{E}_s\left[\mathbf{1}\{a^{\text{aware}}_s \text{ fails}\} - \mathbf{1}\{a^{\text{blind}}_s \text{ fails}\}\right],
$$

where the expectation runs over the agent's decision points and $a^{v}_s$ is the action chosen at state $s$ with verifier $v$. The secondary question is the **encounter rate**: the share of decision points whose candidate list contains at least one action of the kind RQ2 studies (a failure the history determines, acting on the task's object).

## What this is and what it is not

It is a one-step decision study on real agent states with the agent's real candidate distribution. It is not an end-to-end measure: nothing is executed after the choice, so a failed action's cost (one wasted step, "Nothing happens.") is not followed into the rest of the episode, and recovery is not modelled. That is Stage 2's job (branching at disagreements, out of scope here; see "How Stage 1b feeds Stage 2"). It is also one agent (the RQ3 Qwen3-8B ReAct agent, thinking off), one environment (ALFWorld through the AgentGym server), one verifier prompt format (the yes/no feasibility prompt), and one selection rule.

## States

**Source.** The 134 episodes of condition "none" in the confirmatory RQ3 run (`rq3`, four shards, commit ddeabc2; archived at `text-wm-runs-archive/text-wm-fix11/runs/rq3`). In those episodes the Qwen3-8B ReAct agent decoded greedily, saw its real task, its history and the admissible actions, and saw no predictions; the episode cap was 30 steps (`STEP_CAP`). 63 of the 134 episodes were won; 73 hit the cap. Each episode records, per step, the agent's action and the server's real reply, so every state is reconstructable by replaying the recorded actions from reset.

**Why these episodes and not the item collection.** The collection that Stage 1a uses stopped at 12 steps and ran its broad generator only at even steps or after a failing proposal, which selects odd-step states on the outcome of the agent's action. The RQ3 episodes have neither problem: they run to 30 steps (the agent's first-attempt goal-object failures occur at a median step of 20, `exploratory_results.md` section 7), and nothing in them was generated conditionally on an outcome.

**Sampling.** A census: every decision point of every "none" episode, that is, the state before each of the agent's 2,973 actions, in all 134 games. No state is selected on any outcome, and there is no step-parity schedule, so the odd/even artefact of the collection cannot arise. Games are dealt to shards by their position in the sorted list of games, so the assignment does not depend on any outcome either.

**Replay check.** Each game is replayed from reset on the real server along its recorded actions. After every step the server's reply must equal the episode's recorded reply. If it does not, the states after the mismatch are dropped and counted (`replay_mismatch`); the earlier states stay. RQ2's full verification reproduced every one of 12,987 collected items, so the expected count is zero.

**Weighting.** The primary estimand weighs every decision equally. Long failing episodes (the 73 that hit the cap) therefore carry more weight than short successful ones, which is what "the agent's decisions" means for this agent. Two pre-declared sensitivity readouts change the weighting: each game weighted equally (the mean of per-game rates), and only the first 12 steps (comparable with Stage 1a's states).

## Candidate generation and ranking

**Options considered.** Three ways to obtain the agent's ranked candidates were considered.

- Ask the agent for a ranked list. This changes the agent's prompt and therefore its behaviour, so the list would describe a different agent; models also tend to pad such lists with plausible but never-chosen actions, and the order is not calibrated to anything.
- Score a fixed set of actions (the admissible list plus free-form ones) by the agent's log-probability. ALFWorld's admissible-commands list holds almost only working actions, so the failing candidates would have to be written by us, which reintroduces Stage 1a's generator-chosen menu. There is also a technical problem: the agent writes a thought before its action, and the log-probability of an action given one particular thought is close to 0 or close to minus infinity depending on whether the thought already named it, so it does not measure the agent's preference. Marginalising over thoughts requires sampling anyway.
- Sample complete replies from the agent and rank the distinct actions by how often they are sampled. Every candidate is the agent's own, including its free-form actions outside the admissible list (which is where its failures come from: the greedy agent's 671 failing actions include 482 `go to` actions, mostly to where it already is, and 108 actions on the task's object). The frequency of an action among the samples is a Monte Carlo estimate of the agent's marginal probability of that action, integrated over its thoughts, which is exactly the "agent's own probability" a ranking should use.

**Choice: sampling, ranked by frequency, with the greedy action first.** The procedure at each state:

1. The prompt is the RQ3 agent's prompt for condition "none", built by the same function (`agent_react_messages`) from the replayed state: the current observation, the history of earlier observations and actions, and the admissible actions. The agent always sees its real task.
2. **Rank 1 is the greedy action**: the action the agent actually took at this state in the archived episode (in condition "none" this is its greedy proposal; no step of the 134 episodes had a format error). This makes the no-verifier configuration exactly the real agent, and keeps the state sequence on the agent's own path.
3. **Samples.** One request to vLLM with $N = 20$ completions at temperature $T = 1.0$, top-p 1.0, no top-k limit, at most 256 new tokens, thinking off (as in RQ3), with a per-state seed, the first 8 hex digits of `sha256("20260924|decisionsim_agent|<game>|<step>")` read as an integer. $T = 1$ with no truncation is the model's own distribution; Qwen's recommended non-thinking settings ($T = 0.7$, top-p 0.8, top-k 20) cut exactly the tail where free-form failures live, so the lists here are broader than a tempered sampler's and narrower than an exhaustive menu. The sampled replies are saved; the saved list is the data (vLLM sampling repeats only on the same hardware and version).
4. **Parsing.** Each reply is read with the RQ3 parser (`parse_react_action`). A reply without an "Action:" line is a format error: it is counted and contributes no candidate (it is not mapped to `look`, because the agent did not propose `look`). Actions are normalised by collapsing whitespace, lower-casing and dropping a trailing full stop; the normalised string is the candidate and the one executed.
5. **Deduplication and ranking.** Distinct normalised actions other than the greedy action are ranked by their count among the 20 samples, most frequent first; equal counts are ordered by a seeded hash of the action, the smallest of `sha256("20260924|<game>|<step>|<action>")` (the tie rule of Stage 1a, blind to outcomes).
6. **Length.** The list keeps rank 1 and at most 7 sampled actions ($k_{\max} = 8$). The number of distinct actions cut off is recorded per state; with 20 samples, a cut-off action was sampled at most twice.

## Outcome labelling

Each candidate is executed once on a fresh replay of its state on the real server (reset, the recorded actions, then the candidate), as the item collection did (`replay`, `step_counterfactual`). A candidate **fails** when the reply is exactly "Nothing happens."; any other reply is a success. A candidate whose step raises a server error is dropped and counted. Rank 1's replayed reply must equal the episode's recorded reply; a mismatch drops the state and is counted.

Every candidate is also labelled with:

- whether it acts on the task's object (`names_object`, whole word, with the goal object parsed from the task line);
- whether a failure is **determined by the history** (`context_determined_label`, the RQ2 rules, applied to the world model's history text of the state: the opening observation, then each action and reply), with the deciding fact (`determination_reason`);
- whether it is at a shared spot (`at_shared_spot`, from the scene scan, never shown to a model);
- whether it is in the state's admissible list (free-form or not), its verb, its rank and sample count;
- whether the history acts on the goal object (`interacted_objects`), so the strict, RQ2-like subset can be formed;
- a **progress event** flag (below).

A **regime candidate** is a failure that acts on the task's object, is determined by the history and is not at a shared spot: the kind of action RQ2 scores. A **strict regime candidate** additionally has a history that does not act on the goal object, as RQ2's population required. An **encounter state** has at least one regime candidate in its list.

**Progress.** ALFWorld's expert is not reachable through the AgentGym server (`on_expert_path` is None in every RQ3 step), so agreement with an expert's next action cannot be computed, and the code base has no PDDL distance. A descriptive progress event is defined from the candidate's real reply instead: the candidate succeeds and either wins the game, picks up an instance of the goal object, moves an instance of the goal object to a receptacle of the task's target type, heats, cools or cleans an instance of the goal object, or reveals an instance of the goal object that the history has not mentioned before. It is secondary and descriptive.

**Swap object.** For attribution, each state gets one swap object, drawn with a per-state seed (`20260924|decisionsim_agent|<game>|<step>`) by the rules of the collection's swapped goals (`swapped_goal`, `swap_feasible`): an object type in the room, other than the goal, that could do the task by the game's own flags, is not named by any candidate at the state, is not acted on in the history, and has no instance at any candidate's receptacle at reset. States without one (expected mostly in the nine games where the collection found no swap object) have no swapped verifier.

## Verifiers

Every verifier is the yes/no feasibility prompt of the earlier runs (`rq1v`'s `yn_*` arms: the world-model system text, the three frozen few-shot examples of set 1, the history, the action and the fixed question "will this action succeed, or will the environment reply "Nothing happens."?"). The score is the log-odds of the single tokens "Yes" and "No",

$$
L = \log P(\text{Yes}) - \log P(\text{No}),
$$

rounded to 1/64 nat (the bf16 resolution) before any comparison; the verifier **accepts** a candidate when the rounded $L$ is above zero. Only the task line of the query's history differs between verifiers; the few-shot examples are identical in every arm, as in every earlier run.

- **Task-aware:** the history as it is, with the agent's real task line.
- **Task-blind:** the same prompt with the query's task line deleted (the line and the newline before it, so the history reads as the room description followed by the first action). Nothing else changes; a test checks that the blind prompt equals the aware prompt with exactly that line removed.
- **Swapped (attribution):** the real task line with the goal object replaced by the state's swap object (`make_swapped_task_line`).

Two configurations need no model: **none** (the agent's rank-1 action, that is, the real greedy agent) and **simulator oracle** (the verifier knows every candidate's true outcome). The agent always knows its real task in every configuration; only the verifier's prompt changes.

**Models.** Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 are the priority verifiers, because under the aligned task they turn many determined failures to Yes (wave A: 16.5% and 37.5% of failing items flip from No to Yes between the swapped and aligned task, against 0.2% for Qwen3-8B; `families_results.md`). Qwen3-8B (thinking off) is the contrast. All load at the pinned revisions in `wf.constants.REVISIONS`. Prompts are built exactly as before: `sizes.arm_messages` for Qwen and Llama, `families.family_messages` for Mistral (its native template placement, as in wave A), and Mistral gets wave A's Yes/No tokenization gate before any score.

**Competence rule (fixed now).** A verifier's decision results are interpretable only if, on this stage's candidates under the task-blind prompt, its AUC (accepting successes over failures, by $L$) is at least 0.70, it accepts at most 95% of failures and at least 5% of successes, the rule of wave A. A model that fails is reported but not interpreted.

## Selection rule

For a verifier $v$, the chosen action $a^{v}_s$ is the highest-ranked candidate that $v$ accepts. If $v$ accepts none, the agent acts on its rank-1 action (the verifier abstains, and the agent behaves as without one). The fallback rate is reported per verifier. The none configuration always chooses rank 1. The oracle chooses the highest-ranked candidate that truly succeeds, and rank 1 if none does.

## Outcomes and statistics

**Primary (per priority model).** $\Delta_{\text{invalid}}$ over all states, with a 95% game-clustered percentile bootstrap (2,000 resamples of games, seed 20260924) and a two-sided whole-game sign-flip permutation p (1,000 permutations, seed 20260924), the tests used throughout the study. The two priority models are Holm-corrected at 0.05; Qwen3-8B is a contrast and is not tested. A result is called "toward failure" when the Holm-adjusted p is below 0.05 and the interval is above zero, "toward success" when the p is below 0.05 and the interval is below zero, and "no reliable difference" otherwise. A null with an interval inside about ±1 percentage point is reported as a bound on practical harm, not as absence of an effect.

**Secondary and descriptive (per model unless marked).**

- The chosen-failure rate $F_v$ under none, task-aware, task-blind, swapped and oracle; the share of chosen failures that are determined and that are regime candidates.
- The share of states where the aware and blind choices differ, split into harm (aware fails, blind succeeds), help (aware succeeds, blind fails), both fail and both succeed. Harm minus help equals $\Delta_{\text{invalid}}$ exactly; the analysis checks this identity. Harm is split by the kind of action the aware verifier let through (regime candidate, other goal-object failure, `go to`, other).
- **Attribution.** On the states with a swap object, the decomposition

$$
F_{\text{aware}} - F_{\text{blind}} = (F_{\text{aware}} - F_{\text{swapped}}) + (F_{\text{swapped}} - F_{\text{blind}}),
$$

where the first part is what naming the real task's object adds over a task of the same form about another object, and the second is what showing any task of that form does. Both parts get intervals and permutation p-values; the identity is checked.
- **Encounter (model-free).** The share of states with at least one regime candidate in the list, and with at least one strict regime candidate, each with a game bootstrap interval; by rank, by step band (0, 1–11, 12–29), by whether the candidate is free-form; and the share of states where a regime candidate is **reachable**, that is, ranked above the first candidate that truly succeeds (a verifier that accepts it would choose it before any working action).
- $\Delta_{\text{invalid}}$ restricted to encounter states and to non-encounter states (descriptive; the two parts, weighted by their shares, add up to the whole).
- Verifier value: $F_{\text{none}} - F_v$ for each verifier, and the gap to the oracle $F_v - F_{\text{oracle}}$.
- Per-candidate acceptance rates by outcome, goal relevance and determination, per arm, and the aware-minus-blind shift in acceptance and in mean raw $L$; the AUC per arm.
- Progress: the share of chosen actions that are progress events, per configuration, and the aware-minus-blind difference with its interval.
- Sensitivity, pre-declared: game-weighted $\Delta_{\text{invalid}}$; the first 12 steps only; lists cut to $k = 2$ and $k = 4$; the fallback replaced by "the highest-$L$ candidate" (whether the abstain rule matters).
- Bookkeeping: format errors per sample, distinct actions per state, lists cut by $k_{\max}$, replay mismatches, candidates dropped for server errors, states without a swap.

## Predictions (logged before any data of this stage)

1. **Encounter.** At least one regime candidate appears in the list at between 8% and 30% of states (central guess 15%); most regime candidates are free-form (outside the admissible list); a regime candidate is reachable at under half of the encounter states. The strict, RQ2-like encounter rate is much lower: 0.5–5% (amended before any data of this stage; the first version said 3–15%, see the last section).
2. **Qwen3-8B (contrast).** $|\Delta_{\text{invalid}}|$ below 0.5 points, interval including zero: it accepts almost no determined goal-object failure under the aligned task (0.2%).
3. **Llama-3.1-8B.** A positive point estimate between 0.2 and 1.5 points, concentrated in regime candidates the aware verifier accepts; it may not reach significance (see "Precision").
4. **Mistral-7B-v0.3.** A positive point estimate between 0.5 and 3 points, and the largest decision-change rate of the three; its task-blind verifier will already accept many failures, so its verifier value over none will be the smallest of the three and may be negative.
5. **Attribution.** For Llama and Mistral, the aware-minus-swapped part is positive and is most of the aware-minus-blind difference. The swapped-minus-blind part can have either sign: a stated task lowers acceptance of task-irrelevant actions (section 7 of `exploratory_results.md`), which includes the agent's failing `go to` loops (help) as well as its working `go to` moves (fewer accepted, more fallbacks down the list).
6. **Verifier value.** For Qwen and Llama, every verifier lowers the chosen-failure rate below the greedy agent's 22.6%; the oracle's rate is below 3% (a state where no listed candidate works is rare).

## What each result would and would not show

- **Toward failure** for a priority model: at real decision points of a real agent, telling a verify-then-act agent's world model the task makes it act on failing actions more often, at the rate estimated. It would not show a loss of task success (one wasted step can be recovered), would not transfer automatically to agents that search more broadly or use the world model's imagined outcomes over several steps, and is tied to this prompt format and selection rule. It is what would justify Stage 2.
- **No reliable difference with a narrow interval:** the bias that RQ2 and Stage 1a measure rarely reaches this agent's decisions under verification, because the agent rarely proposes the vulnerable actions where a verifier's acceptance decides the choice. Together with a high Stage 1a vulnerability this would say the risk is real in scores but small in this deployment; it would not rule out harm for planners that rank by predicted success, generate synthetic experience, or roll out imagined trajectories.
- **Toward success:** the task line helps the verifier at real decisions, for example by rejecting the agent's own looping `go to` actions; the swapped decomposition says whether that comes from the real task or from any task.
- **Encounter rate alone (model-free):** how often a sampling agent proposes the vulnerable actions at all; this is a fact about the agent, independent of any verifier.
- **A verifier that fails the competence rule:** its numbers describe that model only and support no claim.

## Precision (from archived data, no model)

**Expected encounter.** The greedy agent's own action is a goal-object failure at 108 of the 2,973 states (3.6%; 61 of them determined, in 20 games), so rank 1 alone gives about 3.6% encounter states. For the sampled alternatives, the closest archived evidence is what the same agent chose instead when told that its first choice would fail, in the other RQ3 conditions (dropped predictions, the agent's re-picks, recomputed on CPU):

| Situation in RQ3 | Re-picks | Re-pick fails | Re-pick is a goal-object failure | Determined goal-object failure |
| --- | --- | --- | --- | --- |
| Failing proposal shown as failing, perfect foresight | 248 | 20% | 12.5% | 4.0% |
| Failing proposal shown as failing, corrupted condition's true steps | 1,268 | 18% | 10.6% | 3.8% |
| Working proposal shown as failing (false failure) | 86 | 28% | 15.1% | 7.0% |
| Failing proposal, "not valid" verdict only | 244 | 18% | 4.9% | 2.5% |

So the agent's second choice is a goal-object failure about 5–15% of the time, and a determined one about 2.5–7%. If $m$ distinct sampled alternatives each had that chance independently, the share of states with at least one goal-object failure would be about 13–30% for $m = 2$ and 22–50% for $m = 4$, and with at least one determined one about 7–15% and 12–27% (rank 1 included in both). The independence assumption overstates the count when alternatives repeat one idea, and the re-pick prompt (with "Expected outcome: Nothing happens.") is not the sampling prompt, so these are rough; they put the expected number of encounter states (determined goal-object failures) somewhere between about 200 and 800 of 2,973.

**Precision of the primary estimate.** A simulation with the actual 134 games and their state counts (2,973), with per-game rates varying strongly (gamma-distributed multipliers with shape 1), gives the 95% interval's half-width and the chance that the interval is above zero with permutation p below 0.05:

| Share of states whose outcome changes | Harm share of the changes | True $\Delta_{\text{invalid}}$ | Interval half-width | Power |
| --- | --- | --- | --- | --- |
| 1% | 65% | 0.3 points | 0.35 points | 0.30 |
| 1% | 80% | 0.6 points | 0.36 points | 0.89 |
| 2% | 65% | 0.6 points | 0.51 points | 0.58 |
| 2% | 80% | 1.2 points | 0.54 points | 1.00 |
| 5% | 65% | 1.5 points | 0.82 points | 0.94 |
| 10% | 65% | 3.0 points | 1.19 points | 1.00 |

A $\Delta_{\text{invalid}}$ of about one point is detected reliably; one of a few tenths of a point is not, but a null would then bound the harm below about one point of decisions. The predicted effects for Llama (0.2–1.5 points) straddle that line; this is why a null is informative and why the study is not designed to guarantee a positive result.

## Cost

- **Agent sampling** (`decisionsim_agent_sample`, GPU, vLLM and the ALFWorld server): 2,973 requests of 20 completions on prompts of about 1–4 thousand tokens and replies of about 50–100 tokens, one replay walk per game. Estimated 2–3 GPU-hours on an A40 or L40S, in four shards.
- **Labelling** (`decisionsim_agent_label`, CPU and the ALFWorld server, no GPU): one fresh replay per candidate plus one scene scan per game. With 3–6 candidates per state, 9,000–18,000 replays; at the collection verifier's rate (about 1,300 replays per CPU-hour) that is 7–14 CPU-hours, in parts that restart the server (it leaks memory on every reset).
- **Verifier scoring** (`decisionsim_agent_score_<model>`, GPU): every candidate under the aware and blind prompts and, where a swap object exists, the swapped one: 25,000–54,000 single-token forward passes per model. Prompts are longer than Stage 1a's (histories to 30 steps), so about 0.15–0.25 s per pass: 1–4 GPU-hours per model, 3.5–11 for the three.
- **Total:** about 6–14 GPU-hours (central estimate 9) and 7–14 CPU-hours. The sbatch limits are set at three times the central estimate; a cut-off job resumes from its saved work.

## Risks

- **Sampling quality at $T = 1$.** Untruncated sampling can produce malformed replies. An engineering gate, fixed now and independent of any outcome: the sampling run's smoke job (two train games) must show format errors in under 25% of samples, no replay mismatch, and no state whose 20 replies are all identical (which would mean the per-request seed collapsed the samples); otherwise the run stops for a design amendment before the census.
- **Nondeterminism.** vLLM sampling is reproducible only on the same hardware and version; the saved replies are the data, and a resume never resamples a saved state.
- **The census is weighted toward long failing episodes,** and their late states are often loops; the game-weighted and first-12-steps readouts show how much this matters.
- **Rank 1 comes from the archived run,** not from a new greedy call. It is the agent's real action at that state, and its replay check ties the state to the episode; the samples are drawn now, on whatever GPU the job gets.
- **Server errors on free-form actions** are dropped and counted; a large count would be reported as a limitation.
- **Swapped verifier coverage** is smaller than the census (no swap object in some games); the decomposition uses only states with one.
- **Blind prompt differs from Stage 1a's.** Stage 1a is adding a literal task-blind arm at the same time; the two definitions should be the same before either is scored (open question for the user).
- **Long prompts.** Agent prompts at step 30 fit vLLM's 8,192-token limit in RQ3; verifier prompts are shorter than the agent's.

## How Stage 1b feeds Stage 2

Stage 2 (not designed here) would branch real episodes at decision points where the aware and blind verifiers choose different actions, continue each branch, and measure the end-to-end cost. Stage 1b supplies, per model, the list of disagreement states with their harm, help, both-fail and both-succeed labels, the harm and help rates, and $\Delta_{\text{invalid}}$ with its interval. A natural gate for Stage 2, to be fixed in its own design, is that a priority model reaches "toward failure" here and has enough harm states to branch from (for example at least 30 in at least 15 games); a null here would make Stage 2 unnecessary for this agent, and "toward success" would turn Stage 2 into a test of a benefit.

## Open questions for the user

1. The task-blind definition (query task line deleted, examples unchanged) should match Stage 1a's literal task-blind arm.
2. Whether to also score the neutral task line ("explore the room"), which would cost about a third more scoring and would link to Stage 1a's baseline.
3. Whether $T = 1$ (the model's own distribution) or Qwen's recommended non-thinking settings ($T = 0.7$, top-p 0.8, top-k 20) better represents "a sampling agent"; the design uses $T = 1$.

## Facts added before any data of this stage (dry run, 2026-11-20)

The model-free dry run (`decisionsim_agent_dryrun`, no model and no server) rebuilt every one of the 2,973 states from the archived episodes, with each game's opening observation taken from the item collection, and labelled the rank-1 candidate (the greedy action) from its recorded reply. It found no state whose history lacked exactly one task line, and it reproduced the re-pick table above exactly. Three facts it adds:

- The greedy action alone is a regime candidate at 61 states (2.1%), in only 15 of the 134 games, at a median step of 22; none is at step 0, 2 are at steps 1–11 (0.15% of those states) and 59 at steps 12–29 (3.9%). Rank 1 therefore already contributes about 2 points of encounter, and the encounters are concentrated in a few long failing episodes, which the game-clustered intervals account for.
- With the strict, RQ2-like requirement that the history never acted on the goal object, the greedy action is a regime candidate at only 4 states (0.13%, 4 games). By the time this agent proposes a failing action on the goal object, it has usually already handled that object. Prediction 1's strict range was therefore lowered from 3–15% to 0.5–5%; the main encounter range (8–30%) is unchanged.
- The greedy action is a progress event at 10.8% of states, which sets the scale for the progress readout.

