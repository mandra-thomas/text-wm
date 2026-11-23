# Decision simulation, Stage 1b: why the null? Exposure, inertia and transfer (exploratory, post hoc, 2026-11-22)

**Status.** Exploratory and post hoc. The frozen Stage 1b design (tag `decisionsim-stage1b-frozen-2026-11-22`, commit `2a05707`), its estimands and its verdicts ([decisionsim_agent_results.md](decisionsim_agent_results.md): "no reliable difference" for Llama-3.1-8B and Mistral-7B-v0.3, overall and on the encounter states) are unchanged by anything here. No model is run and nothing is scored; the note re-reads the saved labels and yes/no scores (local copies of `decisionsim_agent_label` and `decisionsim_agent_score_<model>` from `text-wm-fix37`, each file's SHA-256 checked against ICE). The deep-dive [decisionsim_agent_dx.md](decisionsim_agent_dx.md) already covers some of the same ground; this note is organised around one question: which of three explanations of the null do the data support?

## Step 0: are the data needed already available? (checked before anything below was pre-registered)

**Yes, for every state and every model; no new scoring is needed.** The check (on the 2,973 usable states, all of them; no score is NaN):

- **Scores.** At every state whose primary list holds more than one candidate (1,684 states, 5,151 candidates), every candidate has a yes/no score under the task-aware and task-blind arms, under the swapped arm wherever the state has a swap object (1,454 of those states), and under the neutral arm for Llama and Mistral (Qwen's neutral arm was never scored, by design). The same holds for the temperature-1 list (2,314 multi-candidate states, 9,082 candidates). Every candidate of every list has a task-blind score.
- **Single-candidate states.** The scorer scored only the task-blind arm at a state whose two lists together hold one distinct action. Of the 1,289 primary-list states with a single candidate, 640 also have task-aware scores (because the temperature-1 list added candidates there) and 649 do not. Because a single-candidate state forces every verifier and every rule to choose that one candidate, aware and blind (and forced choice) cannot differ there. What it implies for each statistic: exposure (statistic 1) counts all 2,973 states; candidate-level and ranking statistics (2a, 3a–3c) and the fallback rate use the multi-candidate states only, so that the aware–blind comparison is always paired on the same candidates and the population is defined by the list rather than by which states happened to be scored; final-action disagreement and the failure-rate difference (2c, 3d, 4) are computed over all states, with single-candidate states contributing an exact zero; the rule classification (5) and the forced-choice diagnostic are defined on multi-candidate states, and their differences are also expressed per decision over all states (single-candidate states contributing zero).
- **Frozen selections.** The frozen analysis saved the aware and blind choice only at the states where they differ (its Stage 2 list), not at every state. The selections are therefore recomputed from the saved scores with the frozen module's own `choose` function at tag `2a05707`, unchanged. The deep-dive's independent recompute already showed these recomputed selections reproduce every frozen failure rate, difference and harm/help count exactly; this note repeats that check (statistic 4).

## Summary (added after the analyses ran; details in the sections after the pre-registered list)

The three explanations act at different points of the decision chain, they multiply rather than compete, and each has its own evidence:

- **Low exposure is real and by itself caps the overall effect.** Only 4.3% of states contain a vulnerable (regime) candidate, 3.2% contain one together with a working alternative, and 0.24% are strict RQ2-like states; none occur at step 0. Even if the bias transferred fully on the encounter states, the overall per-decision effect would be about 0.9 points for Llama and 0.4 for Mistral, which is inside or at the edge of the frozen overall intervals.
- **Transfer into rankings is not poor; transfer through this selection rule is.** For Llama, the task moves 94% of candidate scores by more than the bf16 grid. Of the regime candidates the task-blind verifier rejects, the task-aware verifier accepts enough to raise acceptance by 20 points (59% to 79%). In a forced choice with no abstention, ranking by score, that shift becomes a large failure-rate difference on the encounter states: +23.6 points [+6.3, +38.5] for Llama and +11.8 [+3.9, +20.7] for Mistral, with 87–88% of the forced-choice harm from regime candidates. The frozen rule takes the highest-*ranked* accepted candidate, so these shifts are almost always pre-empted by the agent's own first choice. On the encounter states, Llama's net of 26 harm states under forced choice becomes 6 under the frozen rule, and Mistral's net of 13 becomes 0.
- **Verifier inertia explains Mistral and part of Llama, in the form of blanket acceptance, not abstention.** Mistral accepts 98% of all candidates under either prompt, so both verifiers choose rank 1 at 97% of multi-candidate states. Its scores move by more than 1 nat for 39% of candidates, but only 1.6% of accept/reject decisions flip. Llama abstains at 2.6% of states and is not blanket-accepting (it rejects about 40% of failing first choices), but it accepts 80–83% of the agent's first choices, and that is enough for the rule's pre-emption. Abstention inertia applies only to Qwen (29%), and there it creates the harm rather than hiding it.

What the data cannot separate: for Llama and Mistral, the rule's pre-emption and the verifier's high acceptance of rank 1 are the same event (both verifiers choose rank 1), so "inertia" and "the rule blocks transfer" are two descriptions of one layer. The forced-choice comparison also changes the baseline (its blind arm fails less often than the frozen blind arm), so its difference is partly a lost benefit rather than added harm. All of this rests on 110 multi-candidate encounter states in 22 games.

## Pre-registered analysis list and forced-choice rule (committed before any of it was computed)

Everything below is exploratory. The frozen results and the deep-dive had been read; the list is fixed against the readouts it names, not blind to the headline numbers. Nothing is corrected for multiple comparisons. All intervals are 95% percentile intervals from 2,000 resamples of whole games (seed 20260924; the frozen module's `_boot`), and every p-value is the frozen whole-game sign-flip permutation (1,000 flips), descriptive only. Models: Llama-3.1-8B and Mistral-7B-v0.3 (the priority verifiers) and Qwen3-8B (contrast, descriptive), on the primary list; the temperature-1 list as a sensitivity for statistics 1, 4, 5 and the forced-choice diagnostic.

**Populations** (named as in the frozen design): all usable states; the multi-candidate states (more than one runnable candidate in the list); the **encounter states** (at least one regime candidate in the list: a failure that acts on the task's object, is determined by the history and is not at a shared spot); the **encounter states that also list a working candidate** (the futility rule's population); and the **strict** encounter states (an encounter state whose history never acts on the goal object).

1. **Exposure.** The share of all states that are encounter states; that are encounter states which also list a working candidate; that are strict encounter states. Each with a game-clustered interval, its number of games, and by step band.
2. **Verifier behaviour**, task-aware and task-blind side by side, at multi-candidate states. (a) Acceptance rate (rounded log-odds above zero) on all candidates and split by working against failing, regime against non-regime, and rank 1 against alternatives (each split also crossed with working/failing for rank 1 and alternatives). (b) Fallback (abstention) rate: the verifier accepts no candidate. (c) The aware–blind final-action disagreement rate, over all states and over multi-candidate states.
3. **The decision chain**, aware against blind, at multi-candidate states. (a) Scores: the share of candidates whose raw yes/no log-odds differs by more than 1/64 nat and by more than 1 nat; the distribution of aware minus blind raw log-odds (quantiles 5, 25, 50, 75, 95%, and mean), overall and for regime, other failing and working candidates. (b) Acceptance: the share of candidates whose accept/reject decision flips, separately blind-reject to aware-accept and blind-accept to aware-reject, overall and by the same groups. (c) Rankings: the share of states whose set of accepted candidates differs (equivalently, whose order of accepted candidates differs, since accepted candidates are taken in the agent's order; this equals the share of states with at least one flip), and the share whose order of candidates by rounded log-odds (descending, ties by the agent's rank) differs. (d) Selections: the share of states whose chosen action differs. **Funnel**, as state counts and shares of all states: all states; multi-candidate; any candidate with a score change above 1/64 nat; above 1 nat; any acceptance flip; score order differs; chosen action differs; chosen outcome differs (harm or help); net (harm minus help). The same funnel on the encounter states, and a regime-specific funnel: encounter states at multi-candidate states; with a regime candidate whose acceptance flips toward accept under the task; at which that regime candidate becomes the aware choice; at which the blind choice works (regime harm).
4. **The paired aware–blind failure-rate difference** with the frozen rule, on all states, on the encounter states, and on the encounter states that also list a working candidate. The first two recompute the frozen primary and key secondary estimands and must equal the frozen output exactly (estimate, harm and help counts, number of states); the third is new (it equals the second rescaled, since an encounter state without a working candidate cannot differ).
5. **When the rule lets the scores decide.** Each multi-candidate state is put in one class: (A) both verifiers fall back to rank 1; (B) only the aware verifier falls back; (C) only the blind verifier falls back; (D) neither falls back and both choose rank 1; (E) neither falls back and both choose the same alternative (a sixth class the five-way list omits; needed for a partition); (F) neither falls back and they choose different candidates. Per class: the count, the failure rate of the aware and the blind choice, harm and help, and the class's contribution to the overall aware–blind difference (its harm minus help divided by all 2,973 states), with the share of the total that it carries.
6. **By game and by step band** (step 0, steps 1–11, steps 12–29). For the key quantities (exposure, fallback rates, candidate flip rate, final-action disagreement, the failure-rate difference, the class shares of statistic 5, and the forced-choice difference): the value per step band with a game-clustered interval; and the per-game spread (minimum, quartiles, maximum over games with at least one relevant state, and the number of games with a non-zero value).

**Forced-choice diagnostic (separate; not the frozen estimand; changes no verdict).** On the frozen candidate lists, each verifier condition must pick exactly one candidate, with no fallback. **Rule, fixed now:** choose the candidate with the highest yes/no log-odds under that condition, after rounding to the 1/64-nat grid; break ties by the agent's rank (the lower rank wins). Applied at multi-candidate states only; at a single-candidate state every condition picks the one candidate, so those states are excluded from the contrast (and, when a difference is expressed per decision over all states, contribute zero). Conditions: task-aware, task-blind, swapped (states with a swap object), neutral (Llama and Mistral). Readouts, paired by state, with game-clustered intervals: the failure rate of the selected action under each condition, next to the agent's rank-1 action and the oracle (the highest-ranked candidate that works, as in the frozen design); forced aware minus forced blind, and forced aware minus forced swapped on the swap states; each overall (multi-candidate states, and per decision over all states), on the encounter states, on the encounter states that also list a working candidate, and by step band; harm and help counts and the share of forced-choice harm states whose aware choice is a regime candidate. Its purpose is to show whether the score bias transfers into rankings when the rule cannot abstain.

**Conclusion, fixed in form now.** The three explanations are evaluated separately, not as alternatives: (i) **low exposure** (statistic 1, 6), (ii) **verifier inertia**, meaning abstention or blanket acceptance so that the verifier rarely decides (statistics 2b, 5), and (iii) **poor transfer** of the score bias into rankings and selections (statistics 2a, 3, forced choice). For each: what the data distinguish, with numbers, and what they cannot distinguish; then how the layers of the funnel combine.

**Code and tests.** `scripts/decisionsim_agent_exposure.py` (uses the frozen module's `choose`, `fell_back`, `regime`, `encounter`, `views` and statistics, unchanged) writes `notes/decisionsim_agent_exposure_metrics.json`. `tests/test_decisionsim_agent_exposure.py` checks the class partition, the forced-choice rule (grid rounding and the rank tie-break), the funnel counts and a small synthetic data set with a known injected bias.

## Results

Numbers are from [decisionsim_agent_exposure_metrics.json](decisionsim_agent_exposure_metrics.json). Percent or percentage points unless marked; brackets are 95% game-clustered bootstrap intervals; p-values are descriptive. "Aware" is the verifier shown the real task line, "blind" the same prompt with that line deleted.

### 1. Exposure (model-free)

| Population, share of the 2,973 states | Primary list | Games | Temperature-1 list | Games |
|---|---|---|---|---|
| Encounter states (a regime candidate is listed) | 4.34 [2.24, 6.81] | 22 | 5.75 [3.32, 8.60] | 31 |
| Encounter states that also list a working candidate | 3.23 [1.56, 5.26] | 19 | 5.18 [2.81, 7.89] | 30 |
| Strict encounter states | 0.24 [0.07, 0.45] | 6 | 0.61 [0.18, 1.19] | 9 |
| Multi-candidate states (any verifier can matter) | 56.6 [53.1, 59.9] | 127 | 77.8 [75.1, 80.3] | 133 |

By step band (primary list), the encounter share is 0 at step 0, 0.45% [0.08, 0.89] at steps 1–11 and 8.2% [4.2, 12.6] at steps 12–29; with a working candidate it is 0, 0.45% and 6.0% [2.9, 9.5]. Per game, the median game has no encounter state at all; 22 of 134 games have one, and in the most exposed game 47% of its states are encounter states.

### 2. Verifier behaviour, aware against blind

**Acceptance** at the 1,684 multi-candidate states (5,151 candidates; the paired difference is aware minus blind on the same candidates).

| Candidates (number) | Llama aware / blind (difference) | Mistral aware / blind (difference) | Qwen aware / blind (difference) |
|---|---|---|---|
| All (5,151) | 84.1 / 85.8 (−1.7) | 97.8 / 97.8 (0.0) | 44.5 / 58.0 (−13.5) |
| Working (4,057) | 89.0 / 91.5 (−2.5) | 98.9 / 99.5 (−0.7) | 51.9 / 68.6 (−16.7) |
| Failing (1,094) | 65.7 / 64.6 (+1.1) | 93.7 / 91.1 (+2.6) | 17.2 / 18.8 (−1.7) |
| Regime (140) | 78.6 / 58.6 (+20.0 [+8.6, +31.4]) | 90.7 / 87.9 (+2.9 [0.0, +7.8]) | 23.6 / 20.7 (+2.9 [−0.4, +8.1]) |
| Non-regime (5,011) | 84.2 / 86.6 (−2.3) | 98.0 / 98.0 (−0.1) | 45.1 / 59.1 (−14.0) |
| Rank 1, working (1,192) | 88.4 / 91.8 (−3.4) | 98.1 / 99.7 (−1.6) | 54.4 / 71.7 (−17.4) |
| Rank 1, failing (492) | 58.9 / 61.4 (−2.4) | 95.5 / 94.5 (+1.0) | 13.8 / 14.6 (−0.8) |
| Alternatives, working (2,865) | 89.3 / 91.4 (−2.1) | 99.2 / 99.5 (−0.3) | 50.9 / 67.3 (−16.4) |
| Alternatives, failing (602) | 71.3 / 67.3 (+4.0) | 92.2 / 88.4 (+3.8) | 19.9 / 22.3 (−2.3) |

**Fallback (abstention)** at multi-candidate states: Llama 2.6% [1.4, 3.9] aware against 1.6% [0.9, 2.5] blind; Mistral 0.2% [0.1, 0.5] against 0.0%; Qwen 29.4% [24.6, 34.5] against 14.5% [10.6, 18.7]. By step band (step 0, steps 1–11, steps 12–29) the aware rates are 1.5, 1.1 and 3.4% (Llama), 6.2, 0.0 and 0.0% (Mistral) and 98.5, 21.5 and 29.2% (Qwen). Per game, Llama's aware verifier abstains at all in 21 of 127 games and Mistral's in 4; Qwen's abstains in 115, with a median game rate of 25%.

**Final-action disagreement**, aware against blind, over all states: Llama 6.2% [4.9, 7.8], Mistral 0.9% [0.6, 1.4], Qwen 9.8% [8.5, 11.2] (over multi-candidate states 10.9%, 1.7% and 17.2%). By step band it is 5.2, 3.4 and 8.8% for Llama, 9.7, 0.7 and 0.4% for Mistral (almost all of Mistral's at step 0, between two working first moves), and 5.2, 7.7 and 12.0% for Qwen. Per game, Llama disagrees somewhere in 76 of 134 games (median game 3.3%, maximum 36.7%), Mistral in 24.

### 3. Where task-induced changes are lost: the decision chain

**Candidate level** (5,151 candidates at multi-candidate states). The aware-minus-blind raw log-odds difference is written as a shift in nats.

| | Llama | Mistral | Qwen |
|---|---|---|---|
| Score shift larger than 1/64 nat | 94.0% | 95.5% | 97.6% |
| Score shift larger than 1 nat | 29.6% | 39.3% | 80.2% |
| Median shift (5th to 95th percentile), all candidates | −0.13 (−2.0 to +1.6) | −0.63 (−3.1 to +1.0) | −2.75 (−29.3 to +4.0) |
| Median shift, regime candidates | +1.13 (−0.1 to +2.9) | +0.50 (−0.4 to +1.9) | +1.00 (−2.5 to +10.0) |
| Accept/reject flips, blind reject to aware accept | 3.7% | 0.8% | 1.7% |
| Accept/reject flips, blind accept to aware reject | 5.4% | 0.8% | 15.2% |
| Regime candidates flipping to accept / to reject | 20.0% / 0.0% | 2.9% / 0.0% | 4.3% / 1.4% |

**State level: the funnel.** Counts of the 2,973 states at which the task changes each level of the chain (a single-candidate state changes nothing). The order is the order of the chain; levels are not all nested, but a changed choice always had an acceptance flip and a changed outcome always had a changed choice (checked: zero exceptions for every model).

| Level reached | Llama | Mistral | Qwen |
|---|---|---|---|
| All states | 2,973 | 2,973 | 2,973 |
| Multi-candidate | 1,684 | 1,684 | 1,684 |
| Some candidate's score moves more than 1/64 nat | 1,681 | 1,681 | 1,684 |
| Some candidate's score moves more than 1 nat | 1,022 | 1,101 | 1,651 |
| Score order of candidates differs | 675 | 544 | 645 |
| Some accept/reject decision flips (the accepted set differs) | 409 | 73 | 643 |
| Chosen action differs | 184 | 28 | 290 |
| Chosen action's outcome differs | 91 | 11 | 108 |
| Harm / help | 48 / 43 | 8 / 3 | 88 / 20 |
| Net (harm minus help) | +5 | +5 | +68 |

On the encounter states (129; 110 multi-candidate):

| Level reached | Llama | Mistral | Qwen |
|---|---|---|---|
| Multi-candidate encounter states | 110 | 110 | 110 |
| Some score moves more than 1 nat | 89 | 54 | 107 |
| Score order differs | 75 | 55 | 59 |
| Some accept/reject decision flips | 37 | 12 | 46 |
| Chosen action differs | 10 | 2 | 31 |
| Outcome differs | 8 | 2 | 24 |
| Harm / help (net) | 7 / 1 (+6) | 1 / 1 (0) | 22 / 2 (+20) |

**Regime-specific funnel.** For Llama, a regime candidate flips from rejected (blind) to accepted (aware) at 27 of the 110 multi-candidate encounter states; at 9 of them it becomes the aware choice, and at 7 of those the blind choice works (these are all 7 regime harm states). At the other 18 an earlier-ranked candidate was accepted first. Separately, at 35 states both verifiers accept and choose a regime candidate (almost always the agent's own first choice), so the task changes nothing there. For Mistral the counts are 4, 0 and 0, with both verifiers choosing a regime candidate at 39 states. For Qwen they are 6, 5 and 4; its other 11 regime harm states come through abstention (6) or through rejecting a working candidate ranked above a regime candidate that both verifiers accept (5).

Reading the funnel. Under the task, almost every candidate's score moves, and for Llama and Mistral about a third move by more than 1 nat. For Llama, about a quarter of multi-candidate states (409) see an accept/reject flip. The selection rule then passes on less than half of those (184), and only half of the changed choices change the outcome (91). Harm and help then nearly cancel (48 against 43). For Mistral the chain stops earlier: its scores move, but its acceptance is near 98% under both prompts, so only 73 states see any flip. For the regime candidates specifically, the loss happens between "accepted" and "chosen", at the rule.

### 4. The paired aware–blind failure-rate difference (frozen rule)

| Population | Llama | Mistral | Qwen |
|---|---|---|---|
| All states (2,973) | +0.17 [−0.92, +1.27], p 0.83 | +0.17 [−0.09, +0.47], p 0.39 | +2.29 [+1.36, +3.33], p 0.001 |
| Encounter states (129) | +4.65 [−1.16, +12.50], p 0.48 | 0.00 [−2.46, +2.05], p 1.0 | +15.50 [+8.82, +23.26], p 0.002 |
| Encounter states that also list a working candidate (96) | +6.25 [−2.41, +15.65], p 0.49 | 0.00 [−3.45, +2.67], p 1.0 | +20.83 [+11.86, +31.18], p 0.005 |

**Check against the frozen output.** For all three models, the all-states and encounter-state rows match the frozen `metrics.json` exactly: estimate, interval, p-value, number of states, and harm and help counts (`matches_frozen` in the metrics file). The third row is new. It equals the second rescaled by 129/96, because an encounter state with no working candidate cannot differ. By step band, the frozen-rule difference is 0 at step 0 for every model; at steps 1–11 it is +0.07 (Llama), +0.37 (Mistral) and +1.26 (Qwen), and at steps 12–29 it is +0.27, 0.00 and +3.41. Per game, Llama's difference is non-zero in 41 games (from −33 to +20 points per game), Mistral's in 8 and Qwen's in 43.

### 5. When the rule lets the scores decide

Classes of the 1,684 multi-candidate states. Contribution is the class's harm minus help divided by all 2,973 states, so the six contributions add up to the overall difference (checked).

| Class | Llama: states, aware / blind failure, harm / help, contribution | Mistral: same | Qwen: same |
|---|---|---|---|
| A. Both fall back to rank 1 | 16; 75 / 75; 0 / 0; 0.00 | 0 | 223; 56 / 56; 0 / 0; 0.00 |
| B. Only aware falls back | 27; 67 / 11; 16 / 1; +0.50 | 4; 0 / 0; 0 / 0; 0.00 | 272; 27 / 4; 69 / 5; +2.15 |
| C. Only blind falls back | 11; 64 / 36; 3 / 0; +0.10 | 0 | 22; 23 / 32; 3 / 5; −0.07 |
| D. Neither falls back, both choose rank 1 | 1,288 (76.5%); 21 / 21; 0 / 0; 0.00 | 1,630 (96.8%); 28 / 28; 0 / 0; 0.00 | 693 (41.2%); 9 / 9; 0 / 0; 0.00 |
| E. Neither falls back, both choose the same alternative | 181; 13 / 13; 0 / 0; 0.00 | 22; 9 / 9; 0 / 0; 0.00 | 328; 4 / 4; 0 / 0; 0.00 |
| F. Neither falls back, different choices | 161; 21 / 29; 29 / 42; −0.44 | 28; 32 / 14; 8 / 3; +0.17 | 146; 11 / 7; 16 / 10; +0.20 |
| Overall difference | +0.17 | +0.17 | +2.29 |

On the encounter states the classes are: Llama D 92 of 110, F 8 (5 harm, 1 help), C 4 (2 harm), E 5, B 1; Mistral D 105, E 3, F 2 (1 harm, 1 help); Qwen A 48, B 23 (13 harm, 2 help), D 13, E 13, F 9 (7 harm), C 4 (2 harm). By step band, Llama's class D share is 88% at step 0 (57 of 65), 84% at steps 1–11 and 72% at steps 12–29; Mistral's is 74%, 97% and 98%. Qwen's step-0 states are all class B (64 of 65): its aware verifier abstains and the agent's working first move goes ahead. In this table, verifier scores decide the action only in classes B, C and F (and in E, where they agree on an alternative). In classes B, C and F together, that is 2% of multi-candidate states for Mistral and 12% for Llama, against 26% for Qwen. Class D, where the verifier only confirms the agent, is what makes the priority verifiers nearly inert under this rule.

### 6. Distributions by game and step band

Step bands for every quantity are reported in the sections above. In short, every quantity that could carry the bias is zero at step 0 except Qwen's abstention and Mistral's step-0 switches between working first moves. Exposure, disagreement and harm concentrate at steps 12–29. Per-game spreads (minimum, quartiles, maximum, games non-zero) for exposure, fallback, candidate flip rate, disagreement, the frozen-rule difference, the class shares and the forced-choice difference are in the metrics file. Two of them matter for the reading. Llama's per-game candidate flip rate has a median of 7.5% (quartiles 2.9% and 14.1%, 105 of 127 games non-zero), so the task changes acceptances broadly across games, not in a few. Its per-game frozen-rule difference is zero in 93 of 134 games and spread around zero in the rest.

### Forced-choice diagnostic (exploratory; not the frozen estimand; changes no verdict)

Each condition picks the single candidate with the highest rounded log-odds (ties to the agent's better rank), with no abstention, at multi-candidate states. The references are the agent's own first choice and the oracle (the highest-ranked working candidate).

**Failure rate of the forced choice**, multi-candidate states (1,684) and encounter states (110 multi-candidate):

| Selector | Llama, all multi / encounter | Mistral, all multi / encounter | Qwen, all multi / encounter |
|---|---|---|---|
| Agent's rank 1 | 29.2 / 63.6 | 29.2 / 63.6 | 29.2 / 63.6 |
| Forced aware | 11.9 / 67.3 | 9.4 / 60.9 | 8.5 / 49.1 |
| Forced blind | 10.8 / 43.6 | 8.6 / 49.1 | 6.5 / 36.4 |
| Forced swapped (swap states) | 9.8 / 40.0 | 8.4 / 51.3 | 8.1 / 37.5 |
| Forced neutral | 10.3 / 37.3 | 8.9 / 40.9 | not scored |
| Frozen rule, aware / blind | 21.4 / 21.1 and 63.6 / 58.2 | 28.2 / 27.9 and 61.8 / 61.8 | 17.6 / 13.5 and 65.5 / 47.3 |
| Oracle | 2.4 / 12.7 | 2.4 / 12.7 | 2.4 / 12.7 |

**Paired differences** (points):

| Contrast | Llama | Mistral | Qwen |
|---|---|---|---|
| Aware − blind, multi-candidate states | +1.13 [−0.73, +3.08], p 0.27 | +0.83 [−0.11, +1.87], p 0.13 | +1.96 [+0.89, +3.13], p 0.002 |
| the same per decision over all 2,973 states | +0.64 [−0.38, +1.82] | +0.47 [−0.07, +1.06] | +1.11 [+0.50, +1.80] |
| Aware − blind, encounter states | **+23.6 [+6.3, +38.5]**, p 0.03; 32 harm / 6 help in 13 games | **+11.8 [+3.9, +20.7]**, p 0.02; 15 / 2 in 9 games | +12.7 [+2.0, +25.2], p 0.07; 18 / 4 |
| Aware − blind, encounter states with a working candidate | +27.1 [+7.9, +42.9] | +13.5 [+4.7, +23.8] | +14.6 [+2.1, +28.7] |
| Aware − swapped, encounter states (80) | +22.5 [+9.5, +38.0], p 0.01 | +10.0 [+1.2, +25.4], p 0.10 | +7.5 [+1.1, +16.7], p 0.12 |
| Swapped − blind, encounter states (80) | +1.3 [−14.9, +14.3] | +2.5 [−13.6, +13.6] | 0.0 [−11.1, +8.9] |
| Aware − swapped, all multi-candidate swap states | +1.24 [+0.13, +2.49] | +0.41 [−0.31, +1.27] | +0.07 [−0.70, +0.89] |
| Share of forced-choice harm from regime candidates: all multi / encounter | 41% / 88% | 45% / 87% | 38% / 100% |
| Aware − blind, steps 1–11 / 12–29 (all multi) | +1.1 / +1.2 | +0.6 / +1.0 | +1.6 / +2.2 |

Step 0 contributes nothing (no failing candidate is listed there).

**Robustness (added after the pre-registered list).** Leaving out any one game keeps the encounter-state forced-choice difference between +19.6 and +28.9 points for Llama, between +10.0 and +14.4 for Mistral, and between +9.2 and +15.8 for Qwen. Llama's harm states are spread over 11 games, the largest share being 7 in `valid_unseen/150`.

**Temperature-1 list (sensitivity).** The encounter-state forced-choice difference is +20.7 [+5.3, +35.4] for Llama (164 states, 31 games, p 0.03) and +14.6 [+6.3, +24.0] for Mistral (p 0.004). Aware − swapped is +15.0 and +16.7. On all multi-candidate states it is +0.56 [−0.75, +2.04] for Llama and +1.47 [+0.64, +2.36] for Mistral. The frozen-rule differences on this list are +0.13 and +0.24 overall, and +4.09 and 0.00 on its encounter states, as in the frozen sensitivity output. The rule classes look the same as on the primary list (class D covers 76% of multi-candidate states for Llama and 97% for Mistral).

**What the forced choice shows.** When the verifier must rank and cannot abstain or defer to the agent's order, the task-aware verifier picks failing goal-object actions where the task-blind one picks working actions. This happens on the encounter states, for both priority models, and is attributable to naming the real object (aware minus swapped is about as large as aware minus blind; swapped minus blind is near zero). This is the RQ2 bias reaching rankings. Two cautions apply. First, the forced-choice baseline is not the frozen one: the forced blind verifier fails at 43.6% of Llama's encounter states, against 58.2% under the frozen rule. Much of the forced-choice difference is therefore the aware verifier losing the improvement the blind one makes by re-ranking, and only partly harm beyond the agent's own choice: the forced aware verifier is roughly level with the agent's first choice there (67.3% against 63.6%). Second, these are uncorrected, exploratory comparisons on 110 states in 22 games.

## Conclusion: the three explanations

The overall null can be written as a product of layers. If $e$ is the share of decisions where a vulnerable candidate is listed with an alternative the verifier could pick instead, and $\delta$ is the per-state effect of the bias on those decisions under a given selection rule, then the per-decision effect is approximately

$$
\Delta_{\text{per decision}} \approx e \cdot \delta ,
$$

plus whatever the task does at non-encounter states (for the priority models, near zero and of either sign). The three explanations act on different factors: exposure sets $e$, while inertia and transfer set $\delta$ under the frozen rule.

**(i) Low exposure.** *What the data show:* 110 of 2,973 states (3.7%) are multi-candidate encounter states; 96 (3.2%) list a working alternative; none occur at step 0, and they fall in 22 games (19 with a working alternative). Even with the forced-choice effect, the per-decision effect would be about $0.037 \times 23.6 \approx 0.9$ points for Llama and $0.037 \times 11.8 \approx 0.4$ points for Mistral. The measured forced-choice per-decision values are +0.64 and +0.47, because non-encounter states add some noise of either sign. Both lie inside Llama's frozen overall interval [−0.92, +1.27] and at the edge of Mistral's [−0.09, +0.47]. Exposure alone therefore makes the frozen overall estimand nearly blind to the bias in this agent, whatever the rule. *What they cannot show:* how $\delta$ would behave at higher exposure (for example with an agent that proposes goal-object failures earlier or more often). The within-encounter rates are the only evidence, and they come from 22 games.

**(ii) Verifier inertia.** *What the data show:* abstention inertia does not explain the priority nulls. Llama's aware verifier abstains at 2.6% of multi-candidate states and Mistral's at 0.2%. Acceptance inertia does. Mistral accepts 97.8% of candidates under both prompts, its accept/reject decisions flip for 1.6% of candidates although 39% of its scores move by more than 1 nat, and both of its verifiers choose the agent's first choice at 96.8% of multi-candidate states (105 of 110 encounter states). For Mistral, inertia is sufficient to explain the null: the verifier rarely decides anything. Llama is not blanket-accepting (it rejects about 40% of failing first choices and flips 9.1% of accept/reject decisions under the task), but both its verifiers still end on the agent's first choice at 76.5% of multi-candidate states (92 of 110 encounter states). Qwen shows the other kind of inertia, abstention (29.4%), and that is where its frozen-rule harm comes from (class B carries +2.15 of its +2.29 points). *What they cannot show:* for Llama, whether the deference to rank 1 is a property of the verifier or of the rule. Under this rule, accepting the agent's first choice ends the decision, so high acceptance of working first choices (88–92%) and the rule's ordering are the same event.

**(iii) Poor transfer of the prediction bias into rankings.** *What the data show:* the bias transfers into scores (Llama's median shift on regime candidates is +1.1 nats, against −0.1 on other candidates), into acceptances (+20.0 points on regime candidates, 27 encounter states with a regime candidate flipping to accept) and into score rankings (forced choice +23.6 points on encounter states, 88% of that harm from regime candidates, attributable to naming the object). It does not transfer into the frozen rule's selections: of the 27 flips, 9 change the choice and 7 cause harm. On the encounter states, the net shrinks from 26 harm states under forced choice to 6 under the frozen rule (Llama) and from 13 to 0 (Mistral). Poor transfer is therefore not a property of the verifier's rankings. It is a property of the selection rule: "first accepted in the agent's order" pre-empts lower-ranked accepted candidates. *What they cannot show:* how much of the forced-choice difference would survive under a rule that is not forced choice and not the frozen rule (for example, rank by score but keep the agent's choice unless an alternative scores much higher). Also, the forced-choice comparison has its own baseline (its blind arm improves on the agent much more than the frozen blind arm does), so it bounds how far the bias can reach rankings, not the harm under any deployed rule.

**How they combine.** For Llama, three steps account for the null:

- *Exposure* takes a ranking-level effect of about +24 points per encounter state down to about 0.9 points per decision.
- *The rule's pre-emption* (verifier deference to rank 1 within the frozen rule) removes about three quarters of the remaining net on the encounter states (26 to 6 net states, about 0.2 points per decision).
- *Harm and help elsewhere* cancel (net −1 state outside the encounter states), leaving +0.17.

For Mistral, exposure and inertia combine: the ranking-level effect (+12 points per encounter state, about 0.4 per decision) is removed entirely by the rule, because both verifiers accept the agent's first choice at 105 of 110 encounter states. The small overall +0.17 comes from non-regime switches.

None of the three alone explains the null for Llama: exposure caps the effect below detectability, but the rule would cut it by about three quarters even at full exposure, and the verifier's scores show the bias plainly. The answer to "why the null?" is therefore "low exposure times a rule that lets the agent's own first choice pre-empt the verifier's biased acceptances", not "the verifier is unbiased". This is a statement about this agent and this rule. A planner that ranks by predicted success would carry the bias into decisions wherever goal-object failures are listed.
