# Option C: a closed-loop agent with a world-model verifier (design, 2026-11-22)

**Status.** Written and committed before any Option C data existed: no Option C episode has been played, no step-0 check score exists, and no code for this stage had been written when this note was committed. Every number below comes from runs finished before this note (the RQ3 episodes at `text-wm-fix11/runs/rq3`, commit ddeabc2; Stage 1a's frozen results and its exploratory diagnostics) or from `scripts/optionc_precision.py`, which reads only the archived RQ3 episodes. Nothing runs until an independent review passes and the design is frozen at an annotated git tag. Code (to follow this note): `wf/optionc.py`; runs `optionc_step0_<model>`, `optionc_step0_analysis`, `optionc_<model>_<readout>[_argmax]`, `optionc_analysis`, `optionc_smoke`.

## Where this sits

The study's main finding (RQ2, replicated in the yes/no format) is that a prompted world model's predicted success for an action whose failure the history already settles rises when the agent's task names the action's object. Three decision-level stages follow it:

- **Stage 1a** (`decisionsim_design.md` and `decisionsim_results.md`, branch `decisionsim`; frozen at tag `decisionsim-stage1a-frozen-2026-11-21`) asked, over fixed candidate menus at 728 collected states, whether a verifier that sees the task chooses failing actions more often than a task-blind one. Its pre-registered gate returned **"worth doing"**, decided by Qwen3-8B: under rule 2 (verify the agent's proposal; if rejected, take the best-scored menu alternative), the task-aware verifier's chosen action failed at 23.9% of states against 4.4% task-blind and 5.8% with no verifier. Llama-3.1-8B showed no reliable change. What the gate licenses, in its own words, is that "a closed-loop agent study is worth running"; it does not license any claim about task success.
- **The exploratory diagnostics** (`decisionsim_dx.md`, branch `decisionsim-dx`; post hoc) found that Qwen's harm needs two ingredients together. First, *any* concrete task (the real one or one about another object, but not "explore the room") makes the yes/no verifier answer No to about 93% of the agent's working first moves, most likely because Qwen reads the question as "will this action accomplish the task?" (an untested reading). Second, naming the object lifts failing goal-object actions to the top of the fallback ranking. Rejection alone causes no harm; all of the attribution to the real object lies in the fallback ranking. Under a rule that switches only to an alternative the verifier itself accepts, the difference falls to +4.1 points [2.2, 6.1] and is no longer attributable to the real object; with a fallback free of goal-object actions it is +3.2.
- **Stage 1b** (`decisionsim_agent_design.md`, this branch's parent) studies the agent's own sampled candidates at its own states, one step at a time. Its smoke run is pending; no Stage 1b data exist.

Option C is the closed-loop test the gate licensed: the agent plays whole episodes and a verifier screens each of its actions before execution.

## Question

**Does a world-model verifier that sees the agent's task make an ALFWorld agent succeed less often than the same verifier without the task?** Secondary: does it make the agent execute more failing actions, take more steps, or loop more; and is any difference due to naming the task's object (real task against a task about another object) or to showing any concrete task?

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
- **(d) Swapped task:** the real task line with its object replaced by one swap object per game (`make_swapped_task_line`). The swap object is drawn once at the start of the game with the seed `20260924|optionc|<game>` by the collection's rules (`swapped_goal` with `swap_feasible`): an object type present in the room (from the scene scan, never shown to a model), other than the goal, that could do the task by the game's own flags. One object per game, not per state, because the task line must stay fixed through an episode. A game without one has no swapped arm; the attribution contrast uses the games with one. How often the swap object is later named by an executed action is reported.
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

**Why yes/no is primary.** Option C tests what Stage 1a's gate licensed, and the gate was decided on this verifier; changing the readout would change the verifier, and a result could not be tied to the gate or compared with Stages 1a and 1b. Its known weakness (the step-0 collapse under any concrete task) is handled by design rather than by changing the question: the swapped arm separates "any concrete task" from "naming the real object", the primary decision rule never switches to an action the verifier itself rejects, and step-0 overrides and abstentions are logged. **Why the observation readout is still run:** if the collapse is a misreading of the question, the yes/no result says something about prompt wording rather than about what a world model predicts, and the anchored readout is the direct test of that. It is a pre-registered secondary readout for the primary verifier model, gated on its competence in the step-0 check.

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

**Classification, fixed now.** For a model and readout, the step-0 collapse is present when fewer than 50% of working proposals are accepted under the aware arm while at least 80% are accepted under the blind arm. For each model:

- *question-specific (supports the task-success reading):* the yes/no readout collapses, while under the observation readout at least 80% of working proposals are accepted under the aware arm and that rate is within 10 points of the blind arm's;
- *not question-specific:* both readouts collapse (a stated task changes the predicted observation itself, so the yes/no result is a world-model effect, not a misreading);
- *no collapse to explain:* the yes/no readout does not collapse;
- *mixed:* any other pattern.

**What the check decides.** (1) The observation readout's closed-loop episodes (Tier 2) are run only if, for Qwen3-8B, the observation readout passes the wave A competence rule at these states under the blind arm (AUC at least 0.70, accepting at most 95% of failures and at least 5% of successes); otherwise they are not run and the failure is reported. (2) The classification fixes how the yes/no closed-loop result is described: under "question-specific", any aware-minus-blind difference at step 0 is described as a consequence of how the yes/no question is read, not of what the model predicts; under "not question-specific", as a world-model effect. The check does not change the primary readout, the decision rule or any threshold.

## Episodes

At each step $t$ of a game under an arm, until the episode is won or 30 steps have been taken:

1. The agent's prompt is built and its greedy reply obtained (from the cache when an earlier episode of the game reached the same prompt); the parsed action is the proposal.
2. Arm (a) executes the proposal. Otherwise the verifier scores the proposal under its arm's prompt; the rule then accepts it or samples and scores alternatives as above.
3. The chosen action is executed on the live environment and its real reply recorded.
4. If the chosen action is not the proposal (an **override**), the proposal is executed once on a private replay of the state (`replay`, `step_counterfactual` on a sandbox env id) and its real reply recorded, so each override can be classified by the true outcomes of both actions. This is the only counterfactual replay; nothing else is replayed.
5. The history gets the observation and the executed action.

Every step record holds: the proposal and its raw reply, whether it was a format error, the verifier's score of the proposal (all readout quantities logged, the rounded score used), accept or reject, the sampled candidate list with counts (when sampled), every alternative score computed, the chosen action, the reason (accepted proposal, accepted alternative, abstained, argmax), the real reply, whether it failed, the proposal's counterfactual reply at overrides, whether the chosen action and the proposal act on the task's object, and whether the executed (action, reply) pair already occurred earlier in the episode. Episodes are saved one at a time; a resumed job skips saved episodes.

**Order within a job.** For each game: the no-verifier episode first (its greedy replies seed the cache), then, seed by seed, every verifier arm of the run in the fixed order blind, aware, swapped (and neutral where run). A shard is a set of whole games, dealt by position in the sorted game list. The ALFWorld server leaks memory on every reset, so a job stops after a fixed number of new episodes and the sbatch loop restarts the server and resumes, as in RQ3.

## Outcomes

Per episode: success (won within 30 steps); steps taken; the RQ3 score $S$ (minus the number of steps, $-31$ for an episode not won); invalid actions executed (real reply "Nothing happens."); overrides, and among them **override harm** (the proposal would have worked and the chosen action failed) and **override help** (the proposal would have failed and the chosen action worked); abstentions (proposal rejected, no accepted alternative); step-0 override (the first executed action is not the agent's proposal); **repeats** (executed action and real reply identical to an earlier step of the same episode, which gives the agent nothing new); the longest run of consecutive rejected proposals; format errors.

**Primary outcome:** success, the estimand above (aware minus blind), Qwen3-8B, yes/no readout, primary rule.

**Key secondary outcomes** (same model, readout and rule): invalid actions per episode (aware minus blind), and the attribution of a success difference (below).

**Secondary and descriptive** (per run): success and invalid actions for every pair of arms, including verifier value (blind minus none, aware minus none); steps to success among won episodes (descriptive; conditional on success, so not paired); $S$ paired by game; overrides, override harm and help, abstentions, step-0 overrides, repeats and rejection runs per episode; the step of first divergence between the aware and blind episodes of a game and seed, and the share of (game, seed) pairs that never diverge; per-step acceptance rates by the proposal's true outcome and whether it acts on the task's object; the same readouts for the secondary rule, the Llama-3.1-8B contrast, the observation readout and Gemma where run; and the drift check below.

**Drift check (reported, not gated).** The no-verifier arm is the RQ3 no-foresight agent. The share of games whose no-verifier episode reproduces the archived RQ3 episode action for action, and the success rate against RQ3's 47.0% (63 of 134), are reported; if fewer than 90% of games reproduce it, the agent is described as drifted from RQ3's, and comparisons with RQ3 numbers carry that caveat. Comparisons between Option C arms are unaffected, since all are played by the same served agent.

## Inference

Every comparison is paired by game: each game's difference of seed means, $\bar{y}_{g,a} - \bar{y}_{g,b}$ (and likewise for counts). Intervals are 95% game-bootstrap percentile intervals (2,000 resamples of games, seed 20260924); p-values are two-sided whole-game sign-flip permutations of the per-game differences (1,000 permutations, seed 20260924), the tests used throughout the study. The no-verifier arm is seed-free (greedy, no sampling), so it enters as a single episode per game. The attribution contrast uses the games with a swap object.

**Verdict (primary, Qwen3-8B, yes/no, primary rule).** Write $D_{xy}$ for the mean over games of the per-game difference $\bar{y}_{g,x} - \bar{y}_{g,y}$. Harm needs the 95% interval of the primary difference entirely below zero and a permutation p below 0.05. Attribution to the real task is read on the games with a swap object and needs

$$
D_{\text{blind,aware}} > 0 \quad \text{and} \quad D_{\text{swapped,aware}} \ge \tfrac{1}{2} D_{\text{blind,aware}},
$$

with a tolerance of $10^{-12}$ on the second comparison, as in Stage 1a: the swapped verifier must succeed more often than the aware one by at least half of the blind verifier's advantage. The categories, checked in this order:

- **harm, attributable to the real task:** harm, and the attribution condition holds;
- **harm, not attributable to the real task:** harm, and the attribution condition does not hold (a concrete task about another object costs nearly as much);
- **help:** the interval is entirely above zero and the permutation p is below 0.05;
- **no reliable difference:** otherwise, reported with its interval as a bound on the success cost.

The decomposition on the games with a swap object is exact and checked:

$$
\bar{y}_{\text{aware}} - \bar{y}_{\text{blind}} = (\bar{y}_{\text{aware}} - \bar{y}_{\text{swapped}}) + (\bar{y}_{\text{swapped}} - \bar{y}_{\text{blind}}).
$$

**Multiplicity.** There is one primary test. The key secondary (invalid actions) gets the same categories with the sign reversed (more invalid actions is harm) and its own p, uncorrected, labelled secondary. The secondary rule, the Llama contrast, the observation readout and Gemma each get the same verdict computed the same way, labelled secondary and uncorrected; none of them can change the primary verdict. Everything else is descriptive. There is no interim look: the analysis refuses to run until every shard of a run is complete, at one commit and one setting.

## Predictions (logged before any data)

These are honest central guesses, not targets; the design does not depend on them, and a null is informative.

1. **Step-0 check.** Qwen's yes/no readout reproduces the collapse (aware acceptance of working proposals under 20%, blind at least 80%). Under the observation readout, the aware arm accepts at least 80% of working proposals (no collapse), so the classification is "question-specific"; I hold this at about 60% confidence. The naming lift on failing goal-object actions stays positive under both readouts (aware minus swapped above 2 nats). Llama shows no collapse under either readout.
2. **Step 0 in the loop.** Under the primary rule, the aware Qwen verifier rejects at least 80% of first proposals but overrides at most 15% of them: at most step-0 states it accepts no alternative either, so it abstains and the proposal stands. The blind verifier overrides at most 12% (mainly `look` proposals swapped for an accepted `go to`).
3. **Verifier value.** The blind Qwen verifier raises success over no verifier by 0 to 8 points (central +3) and lowers invalid actions per episode (central −1), mostly by rejecting the agent's `go to` loops; this is not certain, because rejection can also send the agent to a less useful accepted alternative.
4. **Primary.** The aware-minus-blind success difference for Qwen (yes/no, primary rule) is between −8 and +2 points, central −2; the most likely verdict is "no reliable difference" (about 60%), then "harm, not attributable" (about 25%). Invalid actions per episode: aware minus blind between 0 and +1, central +0.3.
5. **Attribution.** The aware-minus-swapped difference is near zero (within ±2 points of success): what a concrete task does in the loop comes mostly from showing any task, as the diagnostics found at later steps.
6. **Secondary rule (argmax).** The aware-minus-blind difference is more negative than under the primary rule, between −10 and −2 points (central −5); at step 0 the aware verifier overrides at least 80% of proposals, mostly to another working `go to`, so step-0 harm is small and the cost comes later.
7. **Llama-3.1-8B.** The aware-minus-blind success difference is at most 3 points either way, few overrides; verdict "no reliable difference".
8. **Observation readout (if run).** A smaller aware-minus-blind difference than the yes/no readout's, in success and in invalid actions.

## What each outcome would and would not license

- **Harm, attributable** (primary): telling a yes/no world-model verifier the task makes this agent fail its task more often, and naming the task's object is what does it. It would not license claims for other readouts, agents that do not screen single actions, or other models; with the step-0 check's classification it says whether the mechanism is a reading of the question or a change in predicted observations.
- **Harm, not attributable:** a task-aware verifier costs task success, but a verifier told a task about another object costs nearly as much. This is evidence that stating a concrete task to a yes/no verifier is harmful in practice, not evidence that the object-naming bias of RQ2 reaches task success. The diagnostics predict this pattern.
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

| Tier | Runs | GPU-hours (range) |
|---|---|---|
| Step-0 check | Qwen3-8B and Llama-3.1-8B, four arms, two readouts, about 1,750 candidates: about 14,000 passes per model | 2 (1–3) |
| 1 (primary) | Qwen3-8B, yes/no, primary rule: none + blind, aware, swapped, three seeds | 45 (30–65) |
| 2 (pre-registered secondary) | Qwen3-8B argmax rule, one seed; Llama-3.1-8B primary rule, one seed; Qwen3-8B observation readout, primary rule, one seed (only if the step-0 check's competence rule passes); three arms each | 54 (36–80) |
| 3 (optional) | Gemma-2-9B primary rule, one seed (+1 step-0 check); neutral arm for Qwen yes/no, one seed; more seeds for Tier 2; `valid_seen` games | 6–20 each |

Tiers 1 and 2 come to about 100 GPU-hours (70–145). Each job runs vLLM (the agent, `--gpu-memory-utilization 0.50`; 0.45 in the first version of this note, see the build facts below) and the verifier (Hugging Face, bf16) on one GPU, so it needs at least 48 GB (L40S or A40; an 80 GB A100 would also do); jobs are sharded by game (four shards per run) and chained with resume jobs of 6 hours. The smoke run's measured time per step fixes the chain length before the census; if it projects Tier 1 at more than 1.5 times the upper estimate, the run stops for an amendment of the size (a cost rule, read from timings, never from outcomes).

## Order of runs and gates

1. **Smoke** (`optionc_smoke`, two train games, Qwen3-8B yes/no, every arm, both rules, one seed). Engineering gate, fixed now: no crash; format errors under 5% of proposals; the verifier and agent fit together in GPU memory; a rerun of a finished game from its cache reproduces its episodes without new model calls; time per step recorded. Outcomes are not read.
2. **Step-0 check** (`optionc_step0_qwen3-8b`, `optionc_step0_llama3.1-8b`, then `optionc_step0_analysis`): its classification and the observation readout's competence are read and reported before any census episode.
3. **Tier 1** (`optionc_qwen3-8b_yn`, four shards), then `optionc_analysis` once every shard is complete.
4. **Tier 2** (`optionc_qwen3-8b_yn_argmax`, `optionc_llama3.1-8b_yn`, `optionc_qwen3-8b_obs` if gated in), each analysed when complete.

## Risks and caveats

- **Underpowered for small success effects** (see "Precision"); the invalid-action and override readouts carry the decision-level picture.
- **One agent, one environment, one prompt format per readout, two rules.** The agent is greedy and rarely proposes the failures RQ2 tests (its first fresh goal-object failure comes at a median step of 20, in 20 of 134 episodes), so the object-naming route has little to act on, which is a fact about this agent, not about world models.
- **The verifier's history grows to 30 steps,** longer than RQ2's items; the task line recedes with it, which Stage 1a found weakens the any-task collapse.
- **The swap object is fixed per game,** so the agent may later act on it; how often is reported.
- **The blind prompt keeps task lines in its examples** (as in every earlier stage).
- **Shared GPU.** The verifier's numbers come from the same scorer as earlier runs, but on whatever GPU a job gets; the step-0 check reports agreement with Stage 1a's scores.
- **Sampling reproducibility.** vLLM's seeded sampling repeats only on the same hardware, version and batch; the saved cache is the data.

## Open questions for the user

1. **Budget.** Tiers 1 and 2 are about 100 GPU-hours. A cheaper plan keeps Tier 1 at two seeds (about 35 GPU-hours; power 0.8 at about 8 to 10 points instead of 6 to 8) or drops the observation-readout episodes.
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

