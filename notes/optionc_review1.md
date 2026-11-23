# Option C, independent review 1 (2026-11-22)

**What was reviewed.** Branch `optionc` at 65a18d1: the design note `notes/optionc_design.md` and `scripts/optionc_precision.py` (committed at 23e0fb0, before any code), the build `wf/optionc.py`, `tests/test_optionc.py`, `scripts/mutants_optionc.py`, the `wf/config/optionc_*.yaml` configs, `pace/optionc.sbatch` and `pace/optionc_step0.sbatch`, and the additions to `wf/runs.py`, `wf/scoring/logodds.py` (`next_token_logprobs`) and `scripts/mutation_check.py`. For context I read Stage 1a's exploratory diagnostics (`notes/decisionsim_dx.md` on `main`) and the archived RQ3 episodes (`text-wm-runs-archive/text-wm-fix11/runs/rq3`, read-only, CPU only).

**What was run.** No GPU job, no game-server episode and no ICE change. On the laptop, in the project venv: the full test suite once; ten reviewer mutants of my own under the strict rule (killed only if pytest exits 1 naming a failing test); `scripts/optionc_precision.py` on the archived RQ3 episodes (200 replications); and three small read-only counts on the same archive, quoted where used.

**Verdict.** The code does what the note says, and I found no engineering defect that blocks the freeze. One science problem does block it. In a closed loop, the swapped-task arm is not a clean "any concrete task" control. The attribution rule, prediction 5 and the licensing text for "harm, not attributable" all treat it as one. The fix is cheap and needs no new arm: make the attribution read depend on a pre-registered diagnostic that is already logged, and reword. With that fixed, and preferably the should-fix items too, the design can be frozen. The full suite passes (924 passed). However, none of my ten mutants was killed: the tests do not guard the cache keys behind common random numbers, the gates on the real-scorer path or several logged step fields (S9). That is a should-fix before the freeze tag, not a defect in the current code. On ordering: freeze the rules and analysis first. Then run the smoke and the step-0 check, which read no episode outcomes, before deciding how much of Tiers 1 and 2 to buy. That decision should follow the cost rule and the pre-declared observation-readout gate, nothing else.

## Must-fix (blocks the freeze)

### M1. In the loop, the swapped verifier rejects the real task's progress steps, so the attribution contrast is confounded

**Where.** `notes/optionc_design.md:146-157` (the attribution condition and categories), `:175` (prediction 5), `:183` (what "harm, not attributable" licenses); `wf/optionc.py:862-870` (`attributable`) and `:947-951` (verdicts built on the decomposition).

**The problem.** The attribution rule asks whether the swapped verifier recovers at least half of the blind verifier's advantage over the aware one,

$$
D_{\text{swapped,aware}} \ge \tfrac{1}{2} D_{\text{blind,aware}},
$$

and the design reads the decomposition

$$
\bar{y}_{\text{aware}} - \bar{y}_{\text{blind}} = (\bar{y}_{\text{aware}} - \bar{y}_{\text{swapped}}) + (\bar{y}_{\text{swapped}} - \bar{y}_{\text{blind}})
$$

as "naming the real object" plus "showing any concrete task". On Stage 1a's fixed menus that reading was reasonable, because the outcome there was whether the chosen action *fails*. If a verifier swapped a working progress step for a working `go to`, that counted as no harm. In a closed loop, the same swap costs steps and can cost the game.

The diagnostics show that this swap is what a swapped Qwen verifier will do. Stage 1a's diagnostics (`decisionsim_dx.md`, section 6) report that any concrete task lowers the yes/no score of a *working `take` of an object other than the task's* by about 15 nats at later steps. The figures are −15.0 for the real task minus blind and −15.1 for swapped minus blind. Under the swapped task line, the real goal object is exactly such an "other object". So the swapped verifier should push down the agent's working `take`, `put`, `heat` and similar actions on the real goal, while the aware verifier does not.

The archived RQ3 no-foresight episodes show where those actions sit. Every one of the 63 won episodes contains at least one working action on the goal object (mean 2.8). In won episodes, the first such action comes at a median step of 6. Failing goal-object actions, the only route by which the RQ2 naming bias can act, occur in just 20 of the 134 episodes.

So in the loop, the aware-minus-swapped success difference has two parts with opposite signs. The first is the naming bias making aware accept failing goal-object actions, which hurts aware; it is rare. The second is the swapped verifier rejecting the real goal's progress steps and switching to accepted alternatives, which hurts swapped; it is common and falls on exactly the steps that decide success.

**Failure scenario.** Suppose the aware verifier costs 4 points against blind and all of that cost really comes from naming. The swapped arm loses, say, 6 points through the obstruction channel. Then the swapped verifier does worse than the aware one, the attribution condition fails, and the verdict is "harm, not attributable to the real task". The pre-registered reading of that verdict would then follow: "a verifier told a task about another object costs nearly as much", plus prediction 5's "comes mostly from showing any task". Both statements would be false, and the result would be logged as confirming the diagnostics. The same confound runs through `verdict_invalid`, which uses the same decomposition. The bias points towards the outcome the note predicts, which is what makes this a freeze issue rather than a wording nit.

**Fix (cheap, no new arm).** Three changes would do it:

- Add a pre-registered **obstruction check** from data the step records already hold. For each verifier arm, take the share of steps at which the proposal acts on the real goal object and works (fields `proposal_targets_goal` and `proposal_failed`) and is accepted. Do the same for failing real-goal-object proposals; this second share is the naming lift observed in the loop.
- Make the attribution category conditional on the check. If the swapped arm accepts working real-goal-object proposals at a materially lower rate than the aware arm, say more than 10 points lower (fix the margin now), then the success-level attribution is reported as **"not identifiable in the loop (the swapped verifier obstructs the real task)"** and no "any concrete task" reading is allowed.
- Reword prediction 5 and the "harm, not attributable" bullet so that "swapped" means "a wrong task", not "any task".

Optionally, and only if the user wants an attribution claim at all, the clean version is a matched-state design rather than another episode arm: at every step of the aware episodes, also score the proposal (and any sampled alternatives) under the swapped and blind prompts without acting on the result. That turns attribution into a decision-level comparison at identical states, like Stage 1b, for extra verifier forward passes but no extra episodes. It is a scope change, so I list it as an option, not a requirement.

A consequence for the budget follows. If the swapped arm's success contrast is descriptive, it does not need three seeds (see the open questions below).

## Should-fix

**S1. The key secondary, invalid actions per episode, mostly restates success** (`notes/optionc_design.md:136, 205`; `wf/optionc.py:951`). In the archived no-foresight episodes, a lost episode executes 8.8 failing actions in 30 steps, against 0.75 in 13.4 steps for a won one. That gap comes from length, from looping, and from a per-step failure rate of 29% in lost episodes against 6% in won ones. A 3-point success loss therefore moves invalid actions per episode by about $0.03 \times 8 \approx 0.24$ through game outcome alone, which is most of the predicted +0.3. Making it co-primary would add a correlated second test, not an independent decision-level measure (see the open-question discussion). Two better key secondaries are recorded already and need only to be named:

- **Override harm minus override help per episode.** Overrides where the proposal would have worked and the chosen action failed, minus the reverse. This is the verifier's own contribution to failures, read from the counterfactual replay.
- **Invalid actions among the first 10 steps.** Before most episodes end, this is far less entangled with winning.

**S2. Attribution is decided on point estimates of an underpowered contrast** (`wf/optionc.py:862-870`). The aware-minus-swapped estimate has roughly the same standard error as the primary, about 1.5 to 2.3 points with three seeds. With a true cost of 6 points, the "half" line sits at 3 points, so the category is close to a coin flip unless the truth is near 0% or 100% attributable. Add an **"attribution unresolved"** category for when the aware-minus-swapped interval contains both 0 and the full cost. Also use it when harm is significant on all 134 games but the cost is not positive on the swap-object subset; the current code calls that case "not attributable".

**S3. The step-0 classification labels some regions wrongly** (`wf/optionc.py:730-749`; note `:107-112`). The classification is a well-defined, exhaustive function, but two branches misname what they find.

- If the yes/no blind arm accepts under 80% of working proposals, the result is "no collapse to explain", even with aware at 5%. It should read "not classifiable (the task-blind yes/no verifier itself rejects 20% or more)". For Qwen this is unlikely: blind accepted 90% in Stage 1a, and the 10% it rejected were all the `look` proposals.
- "Question-specific" needs observation-readout aware acceptance of at least 80% *in absolute terms*. The observation readout's threshold (accept when $P(\text{Nothing}) < 0.5$) has never been calibrated. Suppose that readout accepts, say, 70% of working step-0 proposals under both blind and aware. The task then changes nothing, which is the question-specific signature, yet the label is "mixed". I suggest defining "question-specific" as the yes/no collapse plus an observation-readout aware-minus-blind gap within 10 points, with the observation readout passing its blind-arm competence rule.

Report game-clustered intervals next to every share used here (they are computed already). With 134 proposals the standard error near 80% is about 3.5 points, so a 79% against 80% decision is noise.

**S4. The primary rule is not the gate's rule, and the note should say so plainly** (`notes/optionc_design.md:13, 80, 85`). The note keeps yes/no "for continuity with the gate" but makes the primary decision rule accepted-only, while the gate was decided on rule 2 (argmax). I think accepted-only is the right primary: it is the rule a sensible user of a verifier would choose, and it is Stage 1b's rule. But "Option C is the closed-loop test the gate licensed" overstates continuity. The argmax run is the continuity check, at one seed (power about 0.3 to 0.4 at 6 points).

**S5. Gates the note calls gates are not enforced in code.** The smoke gate (`wf/optionc.py:600-605`) is written to `metrics.json` but nothing reads it. Tier 1 does not refuse to start without a passed smoke or a reported step-0 analysis, and the cost rule ("stop if Tier 1 projects above 1.5 times the upper estimate") exists only in prose. The observation gate shows the pattern (`obs_gate_check`). A ten-line equivalent for the episode runs, refusing unless `optionc_smoke` shows `smoke_gate.passed` and `optionc_step0_analysis` is REPORTED (with a logged override), would make the order of runs hold mechanically.

**S6. Gemma's memory headroom is untested** (`pace/optionc.sbatch:47`; `wf/scoring/logodds.py:285, 319`). Both readouts run the full forward pass and keep logits for every position. For Qwen3-8B (vocabulary 152k) on a 4,000 to 5,000 token verifier prompt, that is about 1.2 to 1.5 GB in bf16 on top of 16.4 GB of weights. This fits in the roughly 22 GB left to the verifier once vLLM takes 0.50 of an L40S or A40, which report about 45 GB, not 48. Llama-3.1-8B is similar. Gemma-2-9B has 18.5 GB of weights, a 256k vocabulary (2.5 GB of logits at 5,000 tokens, with extra copies from its logit soft-capping) and eager attention, so it may not fit on the longest histories. The smoke run tests only Qwen. Before any Gemma run (Tier 3), either pass `logits_to_keep=1` in the two scorer calls (it changes nothing in the last row's logits up to kernel-level rounding; check this against Stage 1a's scores with the existing agreement check) or run a Gemma memory smoke on a 30-step prompt.

**S7. The drift check's 90% rule will almost certainly fire, and a missing archive crashes the analysis** (`wf/optionc.py:957-966, 1010-1014`). An exact action-for-action match over about 22 steps on a different GPU type and vLLM build is fragile: one flipped near-tie token on an `Action:` line changes the rest of the episode. Expect "drifted" for reasons that say nothing about the agent being different. The informative parts are these:

- the 134 first actions (agreement should be near 100%);
- the distribution of the first diverging step;
- the paired by-game success difference against RQ3's 47.0%, with its interval.

Report those and describe the 90% figure as a reproduction rate, not a test. Separately, `episode_files` raises `SystemExit` if `rq3_source` is not under `runs/`. That happens after every contrast has been computed but before `metrics.json` is written, so a missing link loses the whole analysis output. Wrap the drift step so that its failure is recorded, not fatal.

**S8. Sampling throughput is probably lower than the cost note assumes** (`pace/_common.sh:216`, `pace/optionc.sbatch:47`). The jobs run vLLM with `--no-enable-prefix-caching` (for determinism, which is right). As far as I know, vLLM's V1 engine splits an `n = 20` request into 20 child requests that share the prompt's KV cache only through prefix caching (I have not checked this against the installed version). If so, each child holds its own copy of a prompt of about 4,000 tokens: about 80,000 tokens of KV cache, against roughly 30,000 to 40,000 that fit in the 0.50 share after the weights. The request will queue rather than fail, so this is a cost item, not a correctness one. The smoke run's seconds per step will show it, and the cost rule is the right guard.

## Notes

- **Common random numbers are sound.** I traced the cache keys:
  - greedy replies are keyed by the exact agent prompt;
  - sample lists by the request seed and the prompt, where the seed depends only on the sampling seed, the game and the prompt;
  - scores by model, readout and the full verifier messages;
  - counterfactual replies by the executed prefix and the action.

  ALFWorld is deterministic given an action sequence. So two arms of one game and seed play identical steps until the first step at which they execute different actions. A disagreement over accept/reject that still executes the same action, such as an abstention, does not count as divergence. This holds within a run, and therefore within a shard, since games are dealt whole. It does not hold *across* runs: the argmax, Llama and observation runs are compared with the primary run's no-verifier episodes, but they have their own caches, so those cross-run contrasts are not common-random-number pairs. Seeds of one arm coincide wherever the verifier never samples. That is fine, because inference resamples games, not seeds.
- **The verifier is invisible to the agent.** The agent's history holds only executed actions and real observations (`wf/optionc.py:356`). Its samples come from its own unmodified prompt, and no score, verdict or rejected proposal enters any agent prompt. The agent does see an executed alternative as its own past action, which is inherent to the design.
- **Switch rule, fallback, abstention logging and resume** behave as specified, and the tests cover them. Some details:
  - Accepted-only scans alternatives in rank order and stops at the first accepted one.
  - Argmax picks the highest-scored alternative, never the proposal, with ties broken by the seeded hash.
  - "abstained" and "no_alternative" are both logged, and both count as abstentions.
  - Resume repairs truncated JSONL, refuses a changed commit or config, and continues mid-game from the cache.
  - A fresh run deletes the old cache.
- **The observation readout is computed correctly.** The complement is a log-sum-exp over every vocabulary entry except "Nothing", in float32, so it stays finite even when $P(\text{Nothing})$ rounds to one. The first-token gate checks that the chat template's reply begins with that exact token.
- **Precision reproduces.** Rerunning `scripts/optionc_precision.py` with 200 replications gives power at three seeds of 0.56 / 0.40 at 4 points, 0.88 / 0.64 at 6 and 0.98 / 0.87 at 8 (empirical / chaotic scenario), and null half-widths of 3.0 / 4.4 points. These are within simulation noise of the note's table. The invalid-action numbers match exactly: SD 4.99, and differences of 1.21, 1.05 and 0.99 per episode detectable with power 0.8 at one, two and three seeds.
- **Smaller points.**
  - `optionc_analysis` trusts `primary_run` to be the Qwen yes/no accepted-only run without checking its config.
  - In `pace/optionc.sbatch:58`, the `status=$(ls … | grep … | head -1)` line exits the job under `set -euo pipefail` if no STATUS.md exists. The effect is harmless, since the job just ends, but the final `rc` line is not printed.
  - The tokenization gate reruns on every server restart, which costs a few dozen resets each time; negligible.

## The open questions

**Whether invalid actions should be co-primary.** No. The reasons, in order of weight:

1. The question the study asks is about task success. Invalid actions were not chosen because they answer it better, but because they look more detectable. Choosing an outcome for its power after seeing the power analysis is a forking path in spirit, even before data.
2. They would not rescue the study. The note's own prediction is +0.3 per episode, against a smallest detectable difference of about 1.0 at three seeds. The predicted effect is a third of what can be detected, about the same ratio as for success (2 to 4 points predicted against 6 to 8 detectable).
3. As S1 shows, invalid actions per episode largely restate success: a lost episode has about 8 more failed actions than a won one. Two co-primaries that move together buy little and cost a multiplicity correction on the success test.

Keep success as the single primary. Name override harm minus help, and early-step invalid actions, as the decision-level key secondaries. State plainly in advance that the most likely primary outcome is an interval of about ±3 to ±4.5 points, and that this interval is the result: it bounds how much of Stage 1a's 19.5-point decision-level harm survives a realistic rule and a real agent. A bound like "the accept-only Qwen verifier costs at most about 5 points of success" is a useful, publishable statement, and it is what the design can deliver.

**(1) Budget.** Run the smoke and the step-0 check first, about 3 to 5 GPU-hours. They read no outcomes. The step-0 check alone answers the most interesting open scientific question (is the collapse a misreading of the yes/no question?), and the smoke measures the real cost per step. Then:

- **Tier 1:** keep three seeds for the aware and blind arms. Going from two to three seeds narrows the interval from 3.7 to 3.0 points and raises power at 6 points from 0.64 to 0.88, for about 8 GPU-hours.
- **Swapped arm:** given M1, run it at one seed as a descriptive wrong-task arm unless the matched-state shadow scoring is adopted. That saves about 8 GPU-hours.
- **Tier 2:** run the argmax rule (the gate's rule) and the observation readout (if its gate passes). Move Llama-3.1-8B to Tier 3. Stage 1a and the diagnostics show it rejects only about 10% of first moves and has no any-task collapse, so a closed-loop null is close to certain and adds little.

That comes to about 70 to 80 GPU-hours. If the hard cap is about 35, keep Tier 1's aware and blind arms at three seeds with swapped at one, and defer all of Tier 2, rather than cutting the primary to two seeds.

**(2) Adding the 140 `valid_seen` games.** Not now. More games is the only lever that substantially narrows the interval, but even doubled, the detectable effect (about 4 to 6 points) stays above the 2 to 4 points predicted, at about twice the cost and with no archived episodes for the drift check. No validity problem requires it. If it is ever wanted, decide before Tier 1 and pool by design. Adding it after seeing Tier 1 would be optional stopping; at that point it could only run as a separate, separately analysed replication.

**(3) The anchored observation readout as primary.** No. Its competence is unknown until the step-0 check runs, and a primary readout that might fail its own competence rule would leave the study without a primary. Yes/no also carries the effect being followed up. The right use of the observation readout is the one designed: the step-0 check classifies the collapse, and the observation-readout episodes test the world-model reading. One pre-registerable refinement is acceptable if the budget allows. If the step-0 check comes back "question-specific" and the observation readout is competent, give the observation-readout run the primary run's three seeds (a pre-declared reallocation that depends only on the step-0 check, never on episode outcomes).

**(4) The neutral arm.** Leave it out (or keep it optional in Tier 3). Its only job in this design would be to check that deleting the task line does not make the blind prompt odd. Stage 1a already showed it does not: on the same states, task-blind, task-blind with the examples stripped, and neutral gave 4.4%, 4.1% and 3.4% failure, and 90% and 90% acceptance of working first moves.

## Tests and mutants

**Full suite.** In the project venv (`.venv/bin/python -m pytest -q -x -p no:cacheprovider`, run from this worktree at 65a18d1): **924 passed, 7 skipped, 0 failed** (33 minutes on a shared laptop). The cache test the builder fixed passes when it runs after the other modules.

**Reviewer mutants.** Ten mutants of my own (the runner was a scratch script; the mutants are listed below in enough detail to recreate them), each applied to a copy of the worktree. `tests/test_optionc.py` was run first, then the full suite if the mutant survived. Strict rule: killed only if pytest exits 1 and names a failing test.

Not one of the ten was killed. Each line gives the mutant and what it would break.

- **R1.** The sample cache key drops the request seed (`f"{rs}|{psha}"` becomes `psha`). Seeds 2 and 3 would silently reuse seed 1's samples, so the three seeds would be one.
- **R2.** The counterfactual cache key drops the prefix. A proposal that recurs at a later state would get the earlier state's counterfactual outcome, so override harm and help would be misclassified.
- **R3.** The decomposition uses every game instead of the swap-object games. Equivalent in the tests, because every mock game has a swap object.
- **R4.** Gemma's yes/no tokenization gate is no longer enforced in the episode runs. The gates never run in tests, because the mock scorer skips them.
- **R5.** The observation gate accepts a step-0 analysis whose STATUS is not REPORTED.
- **R6.** Abstentions stop counting the "no_alternative" case.
- **R7.** `proposal_targets_goal` records the chosen action instead of the proposal. This is the field the M1 diagnostic and `step_rates` read.
- **R8.** `step0_rejected` becomes true for every no-verifier episode.
- **R9.** The step-0 any-task drop and `go to` acceptance include failing `go to` actions. Probably equivalent on real step-0 data, where `go to` hardly ever fails.
- **R10.** The smoke cache check compares only `success`, not the whole episode. The existing test catches a cache that triggers a model call, but not a replay that differs silently.

Every survivor ran the full suite (924 passed each time); none was a timeout or a collection error. R9 is likely equivalent on real data and R3 only matters on real data, where some games have no swap object. The other eight are real behavioural changes the suite cannot see.

The pattern is consistent. The builder's 80 mutants mostly target logic the tests exercise directly. Three areas are left uncovered: the cache keys behind common random numbers (R1, R2), the gates on the real-scorer path (R4, R5), and the descriptive step fields (R6, R7, R8, R10). This is **should-fix before the freeze tag** (S9 below). The code is correct as written, so nothing produced so far is wrong. But a later edit could break seed independence or the counterfactual classification, and every test would still pass.

**S9 (from the mutants).** Add tests that kill R1, R2, R4, R5, R7 and R10. Specifically:

- two seeds of one arm, with a sampler that returns different actions per seed, must choose differently;
- the same proposal at two different prefixes, with different true outcomes, must be classified by its own prefix;
- a non-mock stand-in scorer whose tokenization gate fails must stop a Gemma yes/no run, and the same for an observation run;
- an observation gate with metrics present but a STATUS line that is not REPORTED must refuse;
- a step where the verifier overrides a goal-object proposal with a non-goal alternative must record `proposal_targets_goal` true and `chosen_targets_goal` false;
- the cache check must flag a replay whose trace differs while success does not.

Then add these ten to `scripts/mutants_optionc.py`.
