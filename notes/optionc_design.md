# Option C: a closed-loop agent with a world-model verifier (design, 2026-11-22)

**Status.** Written and committed before any Option C data existed: no Option C episode has been played, no step-0 check score exists, and no code for this stage had been written when this note was committed. Every number below comes from runs finished before this note (the RQ3 episodes at `text-wm-fix11/runs/rq3`, commit ddeabc2; Stage 1a's frozen results and its exploratory diagnostics) or from `scripts/optionc_precision.py`, which reads only the archived RQ3 episodes. Nothing runs until an independent review passes and the design is frozen at an annotated git tag. Review 1 (`notes/optionc_review1.md`) found one must-fix and nine should-fixes; every change made for them, before any run, is marked "(amended after review 1)" in the text and listed with the original wording in "Amendments after review 1, before any run" at the end. Code (to follow this note): `wf/optionc.py`; runs `optionc_step0_<model>`, `optionc_step0_analysis`, `optionc_<model>_<readout>[_argmax]`, `optionc_analysis`, `optionc_smoke`.

## Where this sits

The study's main finding (RQ2, replicated in the yes/no format) is that a prompted world model's predicted success for an action whose failure the history already settles rises when the agent's task names the action's object. Three decision-level stages follow it:

- **Stage 1a** (`decisionsim_design.md` and `decisionsim_results.md`, branch `decisionsim`; frozen at tag `decisionsim-stage1a-frozen-2026-11-21`) asked, over fixed candidate menus at 728 collected states, whether a verifier that sees the task chooses failing actions more often than a task-blind one. Its pre-registered gate returned **"worth doing"**, decided by Qwen3-8B: under rule 2 (verify the agent's proposal; if rejected, take the best-scored menu alternative), the task-aware verifier's chosen action failed at 23.9% of states against 4.4% task-blind and 5.8% with no verifier. Llama-3.1-8B showed no reliable change. What the gate licenses, in its own words, is that "a closed-loop agent study is worth running"; it does not license any claim about task success.
- **The exploratory diagnostics** (`decisionsim_dx.md`, branch `decisionsim-dx`; post hoc) found that Qwen's harm needs two ingredients together. First, *any* concrete task (the real one or one about another object, but not "explore the room") makes the yes/no verifier answer No to about 93% of the agent's working first moves, most likely because Qwen reads the question as "will this action accomplish the task?" (an untested reading). Second, naming the object lifts failing goal-object actions to the top of the fallback ranking. Rejection alone causes no harm; all of the attribution to the real object lies in the fallback ranking. Under a rule that switches only to an alternative the verifier itself accepts, the difference falls to +4.1 points [2.2, 6.1] and is no longer attributable to the real object; with a fallback free of goal-object actions it is +3.2.
- **Stage 1b** (`decisionsim_agent_design.md`, this branch's parent) studies the agent's own sampled candidates at its own states, one step at a time. Its smoke run is pending; no Stage 1b data exist.

Option C is the closed-loop study the gate licensed: the agent plays whole episodes and a verifier screens each of its actions before execution. (Amended after review 1.) It is not a closed-loop copy of the gate's test: the gate was decided on rule 2 (switch to the best-scored alternative, whatever its score), while Option C's primary rule switches only to an alternative the verifier accepts. The readout is the gate's; the rule is Stage 1b's and the one a sensible user of a verifier would choose. The argmax run (Tier 2) is the continuity check with the gate's rule, at one seed (power about 0.3 to 0.4 at 6 points).

## Question

**Does a world-model verifier that sees the agent's task make an ALFWorld agent succeed less often than the same verifier without the task?** Secondary: does it make the agent execute more failing actions, take more steps, or loop more; and is any difference due to naming the task's object? (Amended after review 1.) In the loop the swapped verifier is not a clean "any concrete task" control: it is told a wrong task, and any concrete task lowers the score of working actions on objects other than its own by about 15 nats (`decisionsim_dx.md`, section 6), which here are the real goal's progress steps. Attribution is therefore read at matched states and gated by an obstruction check ("Inference").

Let $y_{g,a,s} \in \{0, 1\}$ be whether the episode of game $g$ under verifier arm $a$ and sampling seed $s$ is won within the step cap, and $\bar{y}_{g,a}$ its mean over the $S$ seeds. The primary estimand is the game-paired difference in success between the task-aware and the task-blind verifier,

$$
\Delta_{\text{success}} = \frac{1}{G} \sum_{g=1}^{G} \left( \bar{y}_{g,\text{aware}} - \bar{y}_{g,\text{blind}} \right),
$$

over the $G = 134$ games. A negative value means the task-aware verifier costs task success.

## What this is and what it is not

It is an end-to-end measure for one agent, one environment, one verifier prompt format per readout and two pre-registered decision rules: the episodes run to the end, so a failing action's real cost (a wasted step, a loop, a missed goal) and the agent's recovery are both included. It is not a measure for agents that plan over several imagined steps, rank many candidates by predicted success, or train on imagined experience; the verifier here only screens one action at a time. It does not separate world-model belief from question reading unless the observation-prediction readout is run (see "Readouts" and "The step-0 check").

## Agent and environment (reused from RQ3)

- **Agent:** the RQ3 Qwen3-8B ReAct agent, thinking off, its prompt built by `agent_react_messages` from the current observation, the history of earlier observations and executed actions, and the admissible actions. Its proposal is its greedy reply (temperature 0, at most 256 new tokens), read by `parse_react_action`; a reply without an "Action:" line is a format error and becomes `look`, as in RQ3. The agent is served by vLLM with `--generation-config vllm`.
- **Environment:** ALFWorld through the AgentGym server; the 134 solvable `valid_unseen` games, the same games in every arm; episodes capped at 30 steps (`STEP_CAP`), as in RQ3.
- **What the agent sees:** only its own history of executed actions and real observations. The verifier is invisible to it: no prediction, verdict or rejected proposal is ever written into its prompt. This is RQ3's "dropped" format, which RQ3 found did not degrade the agent (keeping predictions in the history did: 32.8% against 58.2% success with perfect predictions).
- **What the verifier sees:** the world model's history of the state (the opening observation with the task line, then each executed action and its reply, `history_text_from_state`), the action, and its arm's task line. Unlike RQ2's 12-step items, the history runs to the current step.

## Candidates: the agent's own alternatives

Alternatives are sampled from the agent exactly as in Stage 1b (`decisionsim_agent.AgentSampler`, `rank_candidates`): from the agent's own unmodified prompt at the current state, $N = 20$ completions at Qwen's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20, min-p 0, repetition penalty 1, presence and frequency penalties 0, sent explicitly with each request). Each reply is parsed by the RQ3 parser and normalised (whitespace collapsed, lower case, no trailing full stop); format errors give no candidate. **Rank 1 is the agent's greedy proposal**; the other distinct actions follow by their count among the 20 samples, ties broken by Stage 1a's seeded hash, at most 8 candidates in all ($k_{\max} = 8$). A normalised alternative is the string executed.

Sampling is **lazy**: alternatives are drawn only when the verifier rejects the proposal, so a verifier that accepts the proposal costs no sampling. The request seed is the first 8 hex digits of `sha256("20260924|optionc|<seed>|<game>|<agent prompt hash>")`, so it depends on the sampling seed $s$, the game and the exact prompt, and on nothing that differs between verifier arms.

**Common random numbers.** Within a game, every agent reply (greedy or sampled), every verifier score and every counterfactual outcome is cached under the exact prompt (or prefix and action) that produced it, and reused by every later episode of that game in the same run. Two arms therefore play the identical episode until the first step at which their verifiers choose differently; from then on they differ only because of that choice. This removes vLLM's batch-level nondeterminism from the comparison and makes the pairing as tight as the design allows. The cache is saved to disk as it grows, so a resumed job continues with the same replies.

## Verifier arms

All arms use the same verifier model, readout and decision rule within a run; only the task line in the query's history differs, as in Stages 1a and 1b. The few-shot examples (set 1) are identical in every arm.

- **(a) None:** no verifier; the agent's greedy proposal is executed (this is the RQ3 no-foresight agent).
- **(b) Task-blind:** the query's task line deleted (the line and the newline before it), with Stage 1b's guards (`decisionsim_agent.blind_messages`: exactly one task line, equal to the real one, inside the first observation before the first action, never the first line). Caveat as in the earlier stages: the few-shot examples keep their own task lines.
- **(c) Task-aware:** the real task line.
- **(d) Swapped task (a wrong task; amended after review 1):** the real task line with its object replaced by one swap object per game (`make_swapped_task_line`). The swap object is drawn once at the start of the game with the seed `20260924|optionc|<game>` by the collection's rules (`swapped_goal` with `swap_feasible`): an object type present in the room (from the scene scan, never shown to a model), other than the goal, that could do the task by the game's own flags. One object per game, not per state, because the task line must stay fixed through an episode. A game without one has no swapped arm; the attribution contrast uses the games with one. How often the swap object is later named by an executed action is reported.
- **(e) Neutral (optional, Tier 3):** "Your task is to: explore the room." It never decides anything.

## Readouts

**Primary: the yes/no question** of the earlier runs (`rq1v`'s `yn_*` prompts: "Question: will this action succeed, or will the environment reply "Nothing happens."? Answer Yes if it succeeds and No if nothing happens."), scored as the log-odds of the single tokens "Yes" and "No",

$$
L_{\text{yn}} = \log P(\text{Yes}) - \log P(\text{No}).
$$

**Secondary: an anchored observation-prediction readout.** The world-model prompt that asks for the next observation (`rq1`'s plain prompt, ending "Next observation:"), read at the first token of the reply. Every ALFWorld failure reply is "Nothing happens.", so the probability that the reply does not begin with the token "Nothing" is the model's own predicted probability of success:

$$
L_{\text{obs}} = \log\bigl(1 - P(\text{Nothing})\bigr) - \log P(\text{Nothing}).
$$

The pair readout "You" against "Nothing" (the first tokens of a typical success and of the failure, the `L_first` of the earlier runs) is computed from the same forward pass and reported,

$$
L_{\text{You}} = \log P(\text{You}) - \log P(\text{Nothing}),
$$

but it does not decide: in the 134 archived no-foresight episodes, 239 of the 2,302 working replies (10.4%) do not begin with "You" (every one an `examine` reply beginning "On", "There's", "This" or "The", plus one `help`), so the pair readout would score those working actions as if the model expected neither outcome.

For both readouts the score is rounded to 1/64 nat before any comparison, and the verifier **accepts** an action when the rounded score is above zero.

**Why yes/no is primary.** The gate was decided on this readout (with rule 2, not the primary rule here; amended after review 1); changing the readout would change the verifier, and a result could not be tied to the gate or compared with Stages 1a and 1b. Its known weakness (the step-0 collapse under any concrete task) is handled by design rather than by changing the question: the swapped arm separates "any concrete task" from "naming the real object", the primary decision rule never switches to an action the verifier itself rejects, and step-0 overrides and abstentions are logged. **Why the observation readout is still run:** if the collapse is a misreading of the question, the yes/no result says something about prompt wording rather than about what a world model predicts, and the anchored readout is the direct test of that. It is a pre-registered secondary readout for the primary verifier model, gated on its competence in the step-0 check.

## Decision rules

- **Primary: switch only to an accepted alternative.** The verifier scores the agent's proposal. If it accepts, the proposal is executed. If it rejects, alternatives are sampled and scored in rank order, and the first one the verifier accepts is executed. If none is accepted, or there is no alternative (all 20 samples equal the proposal or are format errors), the proposal is executed anyway (the verifier abstains). This is Stage 1b's selection rule and the variant that the diagnostics recommend; it never executes an action the verifier itself predicts to fail in place of the agent's own choice.
- **Secondary: Stage 1a's rule 2 (argmax fallback).** As above, but on rejection every alternative is scored and the one with the highest rounded score is executed, whatever its sign; ties go to Stage 1a's seeded hash; with no alternative the proposal is executed. This is the rule the gate was decided on.

Both rules apply identically in every verifier arm. A rule is a property of the run, so each rule has its own episodes.

## Models

- **Qwen3-8B** (thinking off; the agent's own model, loaded separately as the verifier) is the primary verifier: it carries the Stage 1a effect, and its mechanism is the one the diagnostics describe.
- **Llama-3.1-8B-Instruct** is the contrast: in Stage 1a its naming effect was strong but it showed no any-task collapse and no reliable change in decisions, so little effect is expected.
- **Gemma-2-9B-it** is secondary (Tier 3): in Stage 1a it showed Qwen's two-part mechanism at smaller scale (+8.9 points, "worth doing" post hoc).

All load at the pinned revisions in `wf.constants.REVISIONS`, through the existing scorer (`LogoddsScorer`) and prompt builders (`sizes.arm_messages` for Qwen and Llama, `families.family_messages` for Gemma, with its system text folded into the first turn). Gemma gets wave A's Yes/No tokenization gate before any score; every model gets an equivalent gate for the observation readout ("Nothing" and "You" single tokens, and the template's rendering of the reply "Nothing happens." beginning with that "Nothing" token after every gated prompt).

## The step-0 check (pre-registered; runs before any episode)

**Question.** Is Qwen's step-0 collapse specific to the yes/no question (the task-success reading), or does a concrete task also change what Qwen predicts the next observation to be?

**States and candidates.** Stage 1a's 134 step-0 states (one per game; `decisionsim.build_states` on the collection `rq12_transitions_s0..s2`, restricted to step 0), with Stage 1a's full candidate sets (the agent's step-0 proposal, every one of which works, random admissible actions, mostly working `go to`, and failing goal-object actions) and their real outcomes, and Stage 1a's per-state swap object. These are the same states and actions on which the collapse was found.

**Arms and readouts.** Task-aware, task-blind, swapped and neutral, each under both readouts, for Qwen3-8B, Llama-3.1-8B and (Tier 3) Gemma-2-9B. The yes/no prompts under the aware, swapped and neutral arms are Stage 1a's prompts, so their scores also check that this pipeline reproduces Stage 1a's (reported as the largest difference after rounding; the GPU may differ).

**Measures** (per model, readout and arm): the acceptance rate of the 134 working step-0 proposals and of all working step-0 `go to` candidates; the median score of each; the share of states where no candidate is accepted; the AUC separating working from failing candidates; the naming lift, the mean aware-minus-swapped score on failing goal-object candidates; and the any-task drop, the mean aware-minus-blind and swapped-minus-blind score on working `go to` candidates. Game-clustered intervals as elsewhere.

**Classification, fixed now (amended after review 1).** For each readout, the gap is the aware-minus-blind difference in accepting the 134 working step-0 proposals, paired by state, with its 95% game-bootstrap interval. A readout *collapses* when the interval's upper end is below −30 points; it shows *no gap* when the whole interval lies between −10 and +10 points; otherwise it is *unclear*. The blind arm of a readout is *competent* when it passes the wave A competence rule on all step-0 candidates. For each model, checked in this order:

- *not classifiable (the task-blind yes/no verifier is not competent);*
- *no collapse to explain:* the yes/no readout shows no gap;
- *mixed (the yes/no interval does not settle it):* the yes/no readout is unclear;
- *not classifiable (the observation readout is not competent):* yes/no collapses, but the observation readout's blind arm fails the competence rule;
- *question-specific (supports the task-success reading):* yes/no collapses and the observation readout shows no gap;
- *not question-specific:* both readouts collapse (a stated task changes the predicted observation itself, so the yes/no result is a world-model effect, not a misreading);
- *mixed (the observation interval does not settle it):* otherwise.

Every share used here is reported with its interval; near the thresholds the standard error of a share of 134 proposals is about 3.5 points.

**What the check decides.** (1) The observation readout's closed-loop episodes (Tier 2) are run only if, for Qwen3-8B, the observation readout passes the wave A competence rule at these states under the blind arm (AUC at least 0.70, accepting at most 95% of failures and at least 5% of successes); otherwise they are not run and the failure is reported. (2) The classification fixes how the yes/no closed-loop result is described: under "question-specific", any aware-minus-blind difference at step 0 is described as a consequence of how the yes/no question is read, not of what the model predicts; under "not question-specific", as a world-model effect. (3) The pre-declared reallocation below. The check does not change the decision rule or any threshold.

**Pre-declared reallocation (amended after review 1).** If the classification for Qwen3-8B is "question-specific" and its observation readout is competent, Tier 1's three seeds for the aware and blind arms may be moved to the observation-readout run, which then becomes the primary run (its config's `seeds_by_arm`, and `primary_run` in `optionc_analysis`); yes/no then runs at one seed as a secondary. The decision depends only on the step-0 check, never on an episode, and the analysis refuses an observation-readout primary run unless the step-0 analysis shows that condition.

## Episodes

At each step $t$ of a game under an arm, until the episode is won or 30 steps have been taken:

1. The agent's prompt is built and its greedy reply obtained (from the cache when an earlier episode of the game reached the same prompt); the parsed action is the proposal.
2. Arm (a) executes the proposal. Otherwise the verifier scores the proposal under its arm's prompt; the rule then accepts it or samples and scores alternatives as above. In the aware arm (amended after review 1), the proposal and every alternative the aware verifier scored are also scored under the blind and swapped prompts (the swapped one only where the game has a swap object). These matched-state scores are logged and never acted on.
3. The chosen action is executed on the live environment and its real reply recorded.
4. If the chosen action is not the proposal (an **override**), the proposal is executed once on a private replay of the state (`replay`, `step_counterfactual` on a sandbox env id) and its real reply recorded, so each override can be classified by the true outcomes of both actions. This is the only counterfactual replay; nothing else is replayed.
5. The history gets the observation and the executed action.

Every step record holds: the proposal and its raw reply, whether it was a format error, the verifier's score of the proposal (all readout quantities logged, the rounded score used), accept or reject, the sampled candidate list with counts (when sampled), every alternative score computed, the chosen action, the reason (accepted proposal, accepted alternative, abstained, argmax), the real reply, whether it failed, the proposal's counterfactual reply at overrides, whether the chosen action and the proposal act on the task's object, and whether the executed (action, reply) pair already occurred earlier in the episode. Episodes are saved one at a time; a resumed job skips saved episodes.

**Order within a job.** For each game: the no-verifier episode first (its greedy replies seed the cache), then, seed by seed, every verifier arm of the run that runs at that seed (`seeds_by_arm`; amended after review 1), in the fixed order blind, aware, swapped (and neutral where run). A shard is a set of whole games, dealt by position in the sorted game list. The ALFWorld server leaks memory on every reset, so a job stops after a fixed number of new episodes and the sbatch loop restarts the server and resumes, as in RQ3.

## Outcomes

Per episode: success (won within 30 steps); steps taken; the RQ3 score $S$ (minus the number of steps, $-31$ for an episode not won); invalid actions executed (real reply "Nothing happens."); overrides, and among them **override harm** (the proposal would have worked and the chosen action failed) and **override help** (the proposal would have failed and the chosen action worked); abstentions (proposal rejected, no accepted alternative); step-0 override (the first executed action is not the agent's proposal); **repeats** (executed action and real reply identical to an earlier step of the same episode, which gives the agent nothing new); the longest run of consecutive rejected proposals; format errors; and (amended after review 1) **override harm minus help** and **invalid actions among the first 10 steps**.

**Primary outcome:** success, the estimand above (aware minus blind), Qwen3-8B, yes/no readout, primary rule.

**Key secondary outcomes** (amended after review 1; same model, readout and rule, aware minus blind): override harm minus help per episode, the verifier's own net contribution to failures, read from the counterfactual replay; and invalid actions among the first 10 steps, which is far less entangled with winning. Invalid actions per episode stays a secondary outcome, but it largely restates success: in the archived no-foresight episodes a lost episode executes 8.8 failing actions in 30 steps against 0.75 in 13.4 for a won one, so a 3-point success loss alone moves it by about 0.24.

**Secondary and descriptive** (per run): success and invalid actions for every pair of arms, including verifier value (blind minus none, aware minus none); steps to success among won episodes (descriptive; conditional on success, so not paired); $S$ paired by game; overrides, override harm and help, abstentions, step-0 overrides, repeats and rejection runs per episode; the step of first divergence between the aware and blind episodes of a game and seed, and the share of (game, seed) pairs that never diverge; per-step acceptance rates by the proposal's true outcome and whether it acts on the task's object; the same readouts for the secondary rule, the Llama-3.1-8B contrast, the observation readout and Gemma where run; and the drift check below.

**Drift check (reported, not gated; amended after review 1).** The no-verifier arm is the RQ3 no-foresight agent. Reported against the archived RQ3 episodes: the agreement of the 134 first actions (expected near 100%), the quartiles of the first step at which an episode departs from its archived one, and the paired by-game success difference with its interval. The share of games reproduced action for action is reported as a reproduction rate, not a test: over about 22 steps on another GPU and vLLM build, one flipped near-tie token changes the rest of an episode. A missing archive is recorded in the metrics, not fatal. Comparisons between Option C arms are unaffected, since all are played by the same served agent.

## Inference

Every comparison is paired by game: each game's difference of seed means, $\bar{y}_{g,a} - \bar{y}_{g,b}$ (and likewise for counts). Intervals are 95% game-bootstrap percentile intervals (2,000 resamples of games, seed 20260924); p-values are two-sided whole-game sign-flip permutations of the per-game differences (1,000 permutations, seed 20260924), the tests used throughout the study. The no-verifier arm is seed-free (greedy, no sampling), so it enters as a single episode per game. The attribution contrast uses the games with a swap object.

**Verdict (primary, Qwen3-8B, yes/no, primary rule; amended after review 1).** Write $D_{xy}$ for the mean over games of the per-game difference $\bar{y}_{g,x} - \bar{y}_{g,y}$. Harm needs the 95% interval of the primary difference entirely below zero and a permutation p below 0.05. Help needs the interval entirely above zero and p below 0.05. Otherwise the verdict is **no reliable difference**, reported with its interval as a bound on the success cost.

A harm verdict carries one attribution label. First, the **obstruction check**, from fields every step already logs: for each verifier arm, the share of the agent's *working* proposals on the real goal object that the verifier accepted (and, reported beside it, the share of *failing* ones, which is the naming lift seen in the loop). If the swapped verifier accepts working real-goal proposals more than 10 points less often than the aware one (the margin is fixed now), or if either arm has no such proposal, the label is **attribution not identifiable in the loop (the swapped verifier obstructs the real task)**, and no "any concrete task" reading is allowed. Otherwise attribution is read on the games with a swap object. With the cost $c = D_{\text{blind,aware}}$ there and the recovery $r = D_{\text{swapped,aware}}$, the threshold is half the cost:

$$
r \ge \tfrac{1}{2} c .
$$

The labels, in this order: **attribution unresolved** if $c$ is not positive on the swap-object games, or if the 95% interval of $r$ contains the threshold; **attributable to the real task** if $r$ is at or above the threshold; **not attributable to the real task** otherwise (a tolerance of $10^{-12}$, as in Stage 1a). Key secondaries get the same labels with the sign reversed where more is worse.

The decomposition on the games with a swap object is exact and checked:

$$
\bar{y}_{\text{aware}} - \bar{y}_{\text{blind}} = (\bar{y}_{\text{aware}} - \bar{y}_{\text{swapped}}) + (\bar{y}_{\text{swapped}} - \bar{y}_{\text{blind}}).
$$

**Matched-state attribution (amended after review 1; descriptive).** On the identical states of the aware episodes, each of the aware, blind and swapped prompts is read for how often it would accept the failing goal-object proposals, reject the working proposals, reject the working goal-object proposals, and accept goal-object alternatives (whose outcomes are unknown, since only the chosen action is executed); aware minus blind and aware minus swapped are paired by step with game intervals. This is the decision-level attribution, free of the obstruction confound, and it needs no extra episode.

**Multiplicity.** There is one primary test. The key secondaries (override harm minus help, invalid actions in the first 10 steps) and the secondary invalid actions per episode get the same categories with the sign reversed (more is harm) and their own p, uncorrected, labelled secondary. The secondary rule, the Llama contrast, the observation readout and Gemma each get the same verdict computed the same way, labelled secondary and uncorrected; none of them can change the primary verdict. Everything else is descriptive. There is no interim look: the analysis refuses to run until every shard of a run is complete, at one commit and one setting.

## Predictions (logged before any data)

These are honest central guesses, not targets; the design does not depend on them, and a null is informative.

1. **Step-0 check.** Qwen's yes/no readout reproduces the collapse (aware acceptance of working proposals under 20%, blind at least 80%). Under the observation readout, the aware arm accepts at least 80% of working proposals (no collapse), so the classification is "question-specific"; I hold this at about 60% confidence. The naming lift on failing goal-object actions stays positive under both readouts (aware minus swapped above 2 nats). Llama shows no collapse under either readout.
2. **Step 0 in the loop.** Under the primary rule, the aware Qwen verifier rejects at least 80% of first proposals but overrides at most 15% of them: at most step-0 states it accepts no alternative either, so it abstains and the proposal stands. The blind verifier overrides at most 12% (mainly `look` proposals swapped for an accepted `go to`).
3. **Verifier value.** The blind Qwen verifier raises success over no verifier by 0 to 8 points (central +3) and lowers invalid actions per episode (central −1), mostly by rejecting the agent's `go to` loops; this is not certain, because rejection can also send the agent to a less useful accepted alternative.
4. **Primary.** The aware-minus-blind success difference for Qwen (yes/no, primary rule) is between −8 and +2 points, central −2; the most likely verdict is "no reliable difference" (about 60%), then "harm, not attributable" (about 25%). Invalid actions per episode: aware minus blind between 0 and +1, central +0.3.
5. **Attribution (reworded after review 1).** In the loop the swapped verifier is told a wrong task, and it will reject the real goal's working progress steps that the aware verifier accepts: I expect the obstruction check to fire (swapped accepting working goal-object proposals more than 10 points less often than aware), so success-level attribution will most likely be "not identifiable in the loop". At matched states, I expect the aware prompt to accept failing goal-object proposals more often than the swapped one (the naming lift, a few points of such proposals) and both concrete-task prompts to reject working proposals more often than the blind one.
6. **Secondary rule (argmax).** The aware-minus-blind difference is more negative than under the primary rule, between −10 and −2 points (central −5); at step 0 the aware verifier overrides at least 80% of proposals, mostly to another working `go to`, so step-0 harm is small and the cost comes later.
7. **Llama-3.1-8B.** The aware-minus-blind success difference is at most 3 points either way, few overrides; verdict "no reliable difference".
8. **Observation readout (if run).** A smaller aware-minus-blind difference than the yes/no readout's, in success and in invalid actions.

## What each outcome would and would not license

- **Harm, attributable** (primary): telling a yes/no world-model verifier the task makes this agent fail its task more often, and naming the task's object is what does it. It would not license claims for other readouts, agents that do not screen single actions, or other models; with the step-0 check's classification it says whether the mechanism is a reading of the question or a change in predicted observations.
- **Harm, not attributable (reworded after review 1):** a task-aware verifier costs task success, and a verifier told a wrong task costs nearly as much, with the obstruction check clear. It is evidence that stating a task to a yes/no verifier is harmful in practice, not evidence that the object-naming bias of RQ2 reaches task success. It is not a statement about "any task": the swapped arm is one particular wrong task.
- **Harm, attribution not identifiable in the loop, or unresolved:** a task-aware verifier costs task success; whether naming the object is what does it is not answered at the level of success. The matched-state readouts say how the three prompts differ on the same decisions.
- **Help:** the task helps the verifier screen this agent's actions (for example by rejecting its looping `go to` actions); the swapped and none comparisons say whether it is the real task or any task, and whether a verifier is worth having at all.
- **No reliable difference:** the decision-level harm of Stage 1a does not show up as a success difference this design can detect. Its interval bounds the cost: with three seeds the 95% half-width is about 3 to 4.5 points (see "Precision"), so effects of the size the diagnostics predict (a few points) cannot be excluded, and the invalid-action and override readouts say whether harm occurs at the decision level without reaching success. It would not show that the bias is harmless for planners that rank many candidates or roll out imagined trajectories.
- **Either rule or model failing its gate or competence:** its numbers describe that configuration only and support no claim.

## Precision (archived RQ3 episodes only; `scripts/optionc_precision.py`)

**Archived facts.** In RQ3's no-foresight condition the agent won 63 of 134 games (47.0%) in 2,973 steps (22.2 per episode; 13.4 per won episode); 22.6% of its actions failed (5.0 per episode, SD 5.9); 31% of its steps repeated an earlier (action, reply) pair. Its 134 first actions were 121 `go to` and 13 `look`. Across RQ3's nine conditions with dropped predictions, two thirds of the games were always or never won: success in this agent is mostly determined by the game, and a verifier can change it only in the remaining third. Between the no-foresight and perfect-foresight conditions, which diverged in 114 games, success differed in 25 games (5 lost, 20 won); between two corrupted seeds of one arm, which diverged in 24 to 30 games, it differed in 5 or 6.

**Model.** Game $g$ succeeds with propensity $p_g$ (blind arm) or $p_g (1 - \delta / \bar{p})$ (aware arm), episodes independent given $p_g$; $S$ seeds averaged within a game; detection needs the sign-flip $p < 0.05$ and an interval excluding zero (300 replications). Two noise scenarios: $p_g$ as observed over RQ3's nine dropped conditions (two independent episodes of a game disagree 9.4% of the time), and $p_g$ shrunk toward one half until they disagree 20% of the time (close to the 19-24% between RQ3's no-foresight and perfect-foresight episodes, which also carry a real effect). The model ignores the shared prefix of paired arms (conservative) and between-game variation in the effect (optimistic); seeds count as independent draws, which overstates their value when the verifier rarely rejects, because then seeds share almost every step.

| True success loss | Power, 1 seed | Power, 2 seeds | Power, 3 seeds |
|---|---|---|---|
| 2 points | 0.07 / 0.06 | 0.13 / 0.08 | 0.17 / 0.11 |
| 4 points | 0.20 / 0.14 | 0.37 / 0.19 | 0.54 / 0.38 |
| 6 points | 0.40 / 0.19 | 0.63 / 0.43 | 0.85 / 0.60 |
| 8 points | 0.51 / 0.39 | 0.87 / 0.68 | 0.98 / 0.89 |
| 10 points | 0.74 / 0.51 | 0.98 / 0.86 | 1.00 / 0.96 |
| 95% half-width (no effect) | 5.2 / 7.5 | 3.7 / 5.3 | 3.0 / 4.4 |

Each cell gives the empirical scenario first and the chaotic one second; half-widths are in points of success. Under no effect the test's false-positive rate was 0.01 to 0.05.

**Invalid actions per episode.** The per-game difference between two conditions that diverge in most games has SD about 5.0 (no foresight against perfect foresight). Assuming half its variance is between games and half within (an assumption), the 95% half-width is about 0.85, 0.73 and 0.69 invalid actions per episode with one, two and three seeds, and the difference detectable with power 0.8 about 1.2, 1.05 and 1.0. Stage 1a's switch-only-to-accepted variant (+4.1 points per decision on its menus) would, over about 25 steps, give about one more invalid action per episode if it transferred undiminished; with the agent's own alternatives it should be smaller.

**Reading.** With all 134 games and three seeds, the primary comparison detects a success loss of about 6 to 8 points with power 0.8 and bounds a null to about ±3 to ±4.5 points; effects of 2 to 4 points, the size the diagnostics predict under the primary rule, are not reliably detectable with these games. Adding seeds beyond three gains little (the between-game part does not shrink); adding games would (the 140 solvable `valid_seen` games roughly halve the variance; see "Open questions"). The primary run therefore uses all 134 games and three seeds, and the design states in advance that "no reliable difference" is the most likely outcome and what its interval would and would not exclude.

## Cost

Estimates, not timings; the smoke run measures them. Assumptions: RQ3 ran at about 30 GPU-minutes per game for about 13.5 played episodes, about 2.2 minutes per episode (`rq3-runtime-pace`); a greedy agent call takes about 2 s, an environment step 0.1–0.3 s, a verifier forward pass 0.2–0.4 s on a 2–4 thousand token prompt, a 20-sample request about 3 s, a counterfactual replay about 2–3 s. From Stage 1a, the blind Qwen verifier would reject about 30% of this agent's proposals and the aware one about 40% (about 93% at step 0). That gives about 4.8 s per step for the blind arm and 5.3 s for the aware arm, about 2–2.5 minutes per 25-step episode, close to RQ3's pace. With 30% overhead (model loading, server restarts, slower nodes), one verifier arm over 134 games costs about 6 GPU-hours for its first seed; later seeds reuse every cached step until the first sampled alternative is chosen, estimated at 4 GPU-hours each. The no-verifier arm costs about 2.5 GPU-hours (seed-free). The argmax rule scores every alternative at a rejection, about 10% more.

Amended after review 1: matched-state scoring adds about 2 forward passes per aware step plus 2 per alternative the aware verifier scored, about 1 s per step, roughly 20% of an aware arm. The table is the committed allocation, the user's decision of 2026-11-22 (a budget of about 70 to 80 GPU-hours, the reviewer's plan); every row is set by config alone (`seeds_by_arm`, the arms list, and which runs are submitted).

| Tier | Runs | GPU-hours (range) |
|---|---|---|
| Before any tier | smoke (two train games, every arm, both rules) about 1; step-0 check, Qwen3-8B and Llama-3.1-8B, about 14,000 passes per model, about 2 | 3 (2–4) |
| 1 (primary) | Qwen3-8B, yes/no, primary rule: no verifier (seed-free) 2.5; blind at three seeds 14; aware at three seeds with matched-state scoring 17; swapped at one seed (descriptive) 6 | 40 (27–58) |
| 2 (pre-registered secondary) | Qwen3-8B argmax rule, blind, aware and swapped at one seed, 21 (14–30); Qwen3-8B observation readout, primary rule, one seed, only if its step-0 competence passes, 19 (13–28) | 40 (27–58) |
| 3 (optional) | Llama-3.1-8B, primary rule, one seed, 19 (13–28); Gemma-2-9B, one seed, 19, after a memory smoke of about 0.5; neutral arm, about 6 per seed; `valid_seen` games, about twice any row | 6–40 each |
| Reallocation option | if the step-0 check is question-specific: aware and blind at three seeds on the observation readout instead of yes/no | +16 to Tier 2, −16 from Tier 1 |

Tiers 1 and 2 come to about 80 GPU-hours (54–116), plus about 3 before them. Each run's config carries its upper estimate (`cost_upper_gpu_hours`: 58, 30, 28, 28 and 30), which the cost rule reads. Each job runs vLLM (the agent, `--gpu-memory-utilization 0.50`; 0.45 in the first version of this note, see the build facts below) and the verifier (Hugging Face, bf16) on one GPU, so it needs at least 48 GB (L40S or A40; an 80 GB A100 would also do); jobs are sharded by game (four shards per run) and chained with resume jobs of 6 hours. The smoke run's measured time per step fixes the chain length before the census. The cost rule is code (amended after review 1): every run but the smoke projects its own GPU-hours from the smoke's seconds per episode of each arm, with no credit for the cache, and refuses to start if the projection is above 1.5 times its `cost_upper_gpu_hours`. It reads timings, never outcomes. vLLM keeps `--no-enable-prefix-caching` (greedy determinism, as in RQ3); if its engine copies a 4,000-token prompt into each of the 20 samples, the requests queue rather than fail, and the smoke's sampling seconds and the cost rule catch it.

## Order of runs and gates

Amended after review 1: every gate below is enforced in code. Each episode run except the smoke refuses to start unless every smoke shard has reported with its smoke gate passed, the step-0 analysis has reported, and the cost rule holds; a Gemma run also needs a passed memory smoke. An `order_override` with a reason lets a run start anyway and is saved in its `order_gate.json`.

1. **Smoke** (`optionc_smoke`, two train games, Qwen3-8B yes/no, every arm, both rules, one seed). Engineering gate, fixed now: no crash; format errors under 5% of proposals; the verifier and agent fit together in GPU memory; a rerun of a finished game from its cache reproduces its episodes without new model calls; time per step recorded. Outcomes are not read.
2. **Step-0 check** (`optionc_step0_qwen3-8b`, `optionc_step0_llama3.1-8b`, then `optionc_step0_analysis`): its classification and the observation readout's competence are read and reported before any census episode.
3. **Tier 1** (`optionc_qwen3-8b_yn`, four shards), then `optionc_analysis` once every shard is complete.
4. **Tier 2** (`optionc_qwen3-8b_yn_argmax`; `optionc_qwen3-8b_obs` if gated in), each analysed when complete.
5. **Tier 3** (optional): `optionc_llama3.1-8b_yn`; `optionc_memsmoke_gemma2-9b-it`, then `optionc_gemma2-9b-it_yn`.

## Risks and caveats

- **Underpowered for small success effects** (see "Precision"); the invalid-action and override readouts carry the decision-level picture.
- **One agent, one environment, one prompt format per readout, two rules.** The agent is greedy and rarely proposes the failures RQ2 tests (its first fresh goal-object failure comes at a median step of 20, in 20 of 134 episodes), so the object-naming route has little to act on, which is a fact about this agent, not about world models.
- **The verifier's history grows to 30 steps,** longer than RQ2's items; the task line recedes with it, which Stage 1a found weakens the any-task collapse.
- **The swap object is fixed per game,** so the agent may later act on it; how often is reported.
- **The swapped arm obstructs** (amended after review 1): a verifier told a wrong task rejects the real goal's progress steps, so success-level attribution may not be identifiable; the matched-state readouts carry attribution.
- **Cross-run comparisons are not common-random-number pairs:** each run has its own cache, so the Tier 2 and 3 runs compared with Tier 1's no-verifier episodes are paired by game only.
- **The blind prompt keeps task lines in its examples** (as in every earlier stage).
- **Shared GPU.** The verifier's numbers come from the same scorer as earlier runs, but on whatever GPU a job gets; the step-0 check reports agreement with Stage 1a's scores.
- **Sampling reproducibility.** vLLM's seeded sampling repeats only on the same hardware, version and batch; the saved cache is the data.

## Open questions for the user

1. **Budget** (decided by the user on 2026-11-22, after review 1): about 70 to 80 GPU-hours, the reviewer's plan, committed as the configuration (see the amendments, item 11).
2. **More games.** Adding the 140 solvable `valid_seen` games would roughly halve the variance of the primary comparison (power 0.8 at about 4 to 6 points with three seeds) for about twice the cost. They are not part of RQ3, so they have no archived no-foresight episodes for the drift check.
3. **Readout.** Yes/no is primary for continuity with the gate. If you would rather make the anchored observation readout primary (the stronger world-model claim, but with no gate behind it), the step-0 check can still run first and the swap is a one-line config change before the freeze.
4. **Order with Stage 1b.** Option C does not depend on Stage 1b's results, but Stage 1b's encounter rates would sharpen the predictions; running the step-0 check and the smoke while Stage 1b's census runs is the suggested order.

## Facts added with the build, before any data (2026-11-22)

The code (`wf/optionc.py`) was written after this note and before any Option C data; nothing below changes a rule, threshold, arm, outcome or prediction.

- **GPU share.** vLLM is held to half the GPU (`--gpu-memory-utilization 0.50`, not 0.45), because the 20-sample requests need room for about 5,000 new tokens of KV cache on top of a 3,000-4,000 token prompt; the verifier gets the other 24 GB (Gemma-2-9B's 18.5 GB of weights plus a 256,000-word vocabulary's logits is the tightest fit). `pace/optionc.sbatch` asks for an L40S or A40.
- **Gates in code.** The observation readout's episode run refuses to start unless `optionc_step0_analysis` has finished and found the readout competent for the run's model (or an `obs_gate_override` with a reason is set, which is saved in `obs_gate.json`). Every verifier run checks the tokenization before its first episode, on each game's first-state prompt under every arm: wave A's Yes/No gate for the yes/no readout (enforced for Gemma, recorded for Qwen and Llama, as in Stage 1a) and the first-token gate for the observation readout (enforced for every model).
- **Smoke cache check.** After the smoke episodes, every episode of the first game is replayed from the saved cache with every model replaced by a stand-in that fails if called; the episodes must come out identical. This checks the common-random-numbers machinery and the environment's determinism together.
- **Mock scorer.** The project's mock scorer gives "Yes" and "No" the same log-probability, so every mock yes/no score would round to zero; mock runs (tests and dry runs only) use a hash of the prompt instead, accepting about two thirds of prompts.
- **Dry run.** `python -m wf.runs optionc_smoke --mock` and `optionc_smoke_analysis --mock` (mock agent, scorer and simulator, two games, every arm, both rules) ran end to end, with the cache check and the smoke gate passing.

## Amendments after review 1, before any run (2026-11-22)

Review 1 (`notes/optionc_review1.md`, at 65a18d1) passed the engineering and found one science must-fix and nine should-fixes. Every change below was made before any Option C run of any kind: no smoke, step-0 check or episode exists. Where a rule changed, the original wording is quoted.

1. **M1, the swapped arm obstructs the real task.** In the loop, the swapped verifier rejects working actions on the real goal object (any concrete task lowers a working `take` of another object by about 15 nats), and every archived won episode contains such actions (mean 2.8, the first at a median step of 6), while failing goal-object actions occur in only 20 of 134 episodes. So "swapped recovers half of blind's advantage" does not measure "any task". Changes: (a) a pre-registered obstruction check with a 10-point margin, from logged fields, which labels a harm "attribution not identifiable in the loop" when it fires; (b) prediction 5, the "not attributable" licence and the Question reworded so that swapped means a wrong task; (c) matched-state scoring added, every aware step also scored under the blind and swapped prompts and never acted on, analysed descriptively by prompt on identical states, at about 20% of an aware arm's cost and no extra episode. Original prediction 5: "The aware-minus-swapped difference is near zero (within ±2 points of success): what a concrete task does in the loop comes mostly from showing any task, as the diagnostics found at later steps." Original licence: "a task-aware verifier costs task success, but a verifier told a task about another object costs nearly as much. This is evidence that stating a concrete task to a yes/no verifier is harmful in practice".
2. **S1, key secondaries.** Override harm minus help per episode and invalid actions in the first 10 steps are the key secondaries; invalid actions per episode stays a secondary, stated to largely restate success. Original: "Key secondary outcomes (same model, readout and rule): invalid actions per episode (aware minus blind), and the attribution of a success difference (below)."
3. **S2, attribution unresolved.** Attribution is read from the interval of the recovery, not its point estimate: "unresolved" when that interval contains half the cost, or when the cost is not positive on the swap-object games (previously "not attributable"). Original: attribution needed the blind verifier's advantage over the aware one to be positive and the swapped verifier's advantage over the aware one to be at least half of it, both as point estimates.
4. **S3, step-0 labels.** The classification now reads each readout's paired aware-minus-blind gap from its interval (collapse below −30 points, no gap within ±10) with the blind arm's competence, so a 70%/70% observation readout is "question-specific" (it was "mixed") and a yes/no collapse relative to a blind arm that accepts under 80% is a collapse (it was "no collapse to explain"). Original: "the step-0 collapse is present when fewer than 50% of working proposals are accepted under the aware arm while at least 80% are accepted under the blind arm", with "question-specific" needing at least 80% observation-readout acceptance under the aware arm within 10 points of blind.
5. **S4, continuity.** The note no longer calls Option C the gate's test in a closed loop: the gate used rule 2, the primary rule is accepted-only, and the argmax run is the continuity check. Original: "Option C is the closed-loop test the gate licensed".
6. **S5, gates in code.** Every episode run except the smoke refuses to start without a passed smoke (cache check and smoke gate, every shard), a reported step-0 analysis and the cost rule (projection from the smoke at most 1.5 times the config's `cost_upper_gpu_hours`); an `order_override` is logged. Original: the gates and the cost rule were prose.
7. **S6, Gemma's memory.** The scorer can keep only the last position's logits (`logits_to_keep`, or `num_logits_to_keep`); each verifier run adopts it only if, on up to 8 gate prompts, every rounded score equals the full pass's (`last_only_check.json`). Yes/no scores are read through the same method (numerically the same computation as `score_one_token_pair`). Gemma runs also need `optionc_memsmoke_gemma2-9b-it` (a 30-step prompt, every arm and readout, beside vLLM) to have passed. A CPU test compares the two paths on a tiny model where torch is installed.
8. **S7, drift.** First-action agreement, the first-divergence quartiles and the paired success difference replace the 90% action-for-action rule, which is kept only as a reproduction rate; a missing archive is recorded, not fatal. Original: "if fewer than 90% of games reproduce it, the agent is described as drifted".
9. **S8, sampling throughput.** Prefix caching stays off (greedy determinism; whether the installed vLLM shares an `n = 20` request's prompt KV without it is unchecked); episodes record their sampling seconds and the cost rule catches queueing.
10. **S9, tests.** Tests now kill the reviewer's R1, R2, R4, R5, R6, R7, R8 and R10, and also R3 and R9 (a game without a swap object; a failing `go to` at step 0); see `notes/testing.md`.
11. **Tiers and budget: the user's decision (2026-11-22).** The user set the budget at about 70 to 80 GPU-hours and chose the reviewer's plan, which is the committed configuration: Tier 1, Qwen3-8B yes/no with the primary rule, aware, blind and no verifier at three seeds (the no-verifier arm never samples, so its one episode per game serves all three), the swapped arm at one seed (descriptive), and matched-state scoring of every aware step; Tier 2, the argmax rule and the observation readout, the latter gated on the step-0 check; Llama-3.1-8B in an optional Tier 3; no neutral arm, no `valid_seen` games; yes/no primary, with the pre-declared reallocation to the observation readout under a question-specific step-0 check. Allocation stays config-only: seeds per arm (`seeds_by_arm`), the swapped arm on or off with its own seeds, and which runs are submitted. Original Tier 1: all three verifier arms at three seeds; Llama in Tier 2; about 100 GPU-hours for Tiers 1 and 2.
12. **Smaller points.** The analysis checks that the primary run is Qwen3-8B, accepted-only, with aware and blind arms, on yes/no (or the observation readout under the reallocation rule). The sbatch status line no longer ends the job under `pipefail` when no STATUS.md exists.


## Reallocation exercised (2026-11-22, after the step-0 check and before any census episode)

The step-0 analysis (`optionc_step0_analysis`, job 7246046, at the freeze tag `optionc-design-frozen-2026-11-22`) classified Qwen3-8B as "question-specific (supports the task-success reading)". On the yes/no readout, the aware arm accepts 82 points fewer of the 134 working step-0 proposals than the blind arm (95% game-bootstrap interval −88 to −75; a collapse). On the observation readout the gap is exactly zero in every game ("no gap"). The blind observation readout passes the wave A competence rule (AUC 1.00, accepts 19% of failures and 100% of successes). The pre-declared condition for the reallocation therefore holds, and it is exercised as the committed plan (the user's decision of 2026-11-22) states. No episode of any census run existed when this was decided; the smoke's two train games were read for the engineering gate only.

- `optionc_qwen3-8b_obs` is the primary run: no verifier (seed-free), blind and aware at three seeds, swapped at one, matched-state scoring; `cost_upper_gpu_hours` 58 (Tier 1's former bound).
- `optionc_qwen3-8b_yn` is the pre-registered yes/no secondary at one seed per verifier arm; `cost_upper_gpu_hours` 28.
- The no-verifier arm moved from the yes/no run to the observation run, because `optionc_analysis` takes the reference episodes (and the drift check's episodes) from the primary run. The arm uses no verifier, so its episodes do not depend on the readout. Total cost is unchanged.
- `optionc_analysis` lists both runs with `primary_run: optionc_qwen3-8b_obs`; `primary_problems` checks the step-0 condition, as before.
- Per the step-0 rules, the yes/no result is described as a consequence of how the yes/no question is read, not of what the model predicts. Llama-3.1-8B's observation readout is not competent (its blind arm accepts every failing step-0 candidate), so it has no observation-readout run.
- Smoke timing: 1.26 s per step (the design assumed 4.8–5.3 s), so each shard of either run is expected to finish in one 6-hour job.

Only configuration changed (allocation is config-only by design); the test that pins the configuration was updated to match. No code, threshold, decision rule or readout definition changed.
