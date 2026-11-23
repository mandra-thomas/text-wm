# Decision simulation, Stage 1b: robustness and mechanism (exploratory, post hoc, 2026-11-22)

**Status.** Everything in this note is exploratory and post hoc. The frozen Stage 1b verdicts (tag `decisionsim-stage1b-frozen-2026-11-22` on `2a05707`; [decisionsim_agent_results.md](decisionsim_agent_results.md)) were read before this note was written, and nothing here changes them. No model was run; the note re-reads saved labels and yes/no scores. The analysis list below was committed (`f1824b4`) before the script that computes it existed. Numbers are in [decisionsim_agent_dx_metrics.json](decisionsim_agent_dx_metrics.json), written by `scripts/decisionsim_agent_dx.py` from local copies of the label and score runs (every copied file's SHA-256 matched the file on ICE; the hashes are in the metrics file under `meta`). Stage 1a's numbers are quoted from `notes/decisionsim_dx_metrics.json` on main.

## Pre-registered analysis list (written before any analysis below was computed)

The list below was committed before the script that computes it was written. The frozen analysis output (`metrics.json` of `decisionsim_agent_analysis`, job 7245683) had been read, so the list is not blind to the headline numbers; it is fixed only against the readouts it names. Each item says what it asks. Nothing is corrected for multiple comparisons, and no item can change a verdict.

1. **Independent recompute.** Rebuild every decision from the labels and the scores with code that does not call the frozen module's decision or statistics functions (its own rule, rounding to 1/64 nat, bootstrap and sign-flip permutation), for all three models, and check the chosen-failure rates, the overall and conditional aware-minus-blind differences, and the harm and help counts against the frozen output.
2. **Per-game stability.** Per model, the net (harm minus help) of each game, overall and on the encounter states: how many games are positive, negative and zero, the largest single-game contributions, and the share of harm carried by the five heaviest games. The same difference within each of the six task types (pick, pick two, clean, heat, cool, examine), and with each task type left out.
3. **Leave one game out.** The range of the overall and of the conditional difference with each game left out in turn, per model (the frozen output gives the overall range for each model; this adds the conditional range for every model and a check of the overall one).
4. **Encounter states only.** The 129 encounter states described: step, list length, number and ranks of regime candidates, whether rank 1 is itself a regime candidate, whether a working candidate is listed and above or below the first regime candidate; per model, what the aware and blind verifiers chose there (a regime candidate, another failure, or a working action), and every state where they differ, listed with its game, step, both actions and outcome. The same for the strict encounter states.
5. **Regime candidates the aware verifier accepted.** Per model, the 159 regime candidates of the primary list followed through the selection rule: accepted by the aware verifier, by the blind one, by the swapped one; of those the aware verifier accepted, how many it then chose and, for those it did not, what it chose instead (a higher-ranked accepted candidate, working or failing); and whether the blind verifier's choice at the same state worked. Split by rank 1 against alternatives.
6. **Why Qwen abstains, and why the priority models do not.** Per model and arm, the abstention (fallback) rate by step band (step 0, steps 1–11, steps 12–29), and split by whether the list holds any working candidate (abstention where every candidate fails costs nothing). For abstaining states, the scores of working rank-1 actions and working alternatives under the aware, swapped and blind arms (medians and the median shift), by action verb. For Qwen, the harm states by path, and how many are a "lost rescue": rank 1 fails, the blind verifier rejected it and chose a working alternative, and the aware verifier rejected that alternative too. The neutral arm where scored.
7. **Matched step bands against Stage 1a.** The aware-minus-blind difference per model at step 0, steps 1–11, steps 12–29 and at the even steps 0–10 (Stage 1a's states), next to Stage 1a's step-0 and later-step numbers ([decisionsim_dx.md](decisionsim_dx.md) and its metrics). The two ingredients of Stage 1a's mechanism checked here: (i) the step-0 Yes rate on the agent's working first moves under the aware, swapped and blind arms; (ii) whether the agent's step-0 lists hold any failing goal-object action for a fallback to pick. Stage 1a's fallback rule (the highest-scored candidate when the verifier accepts nothing) applied to Stage 1b's lists at step 0 and later.
8. **Precision.** For each priority model and estimand, which effect sizes the 95% intervals exclude, in percentage points per decision and per encounter state; the effect sizes Stage 1a measured (Qwen +19.5 points, Llama +1.1, and Qwen's later-step +7.6) placed against these bounds; the sign-flip floor of each test (the smallest p it could reach given its number of games with a non-zero net difference).

## Summary

**The setting.** At each of the 2,973 decision points of 134 real ALFWorld episodes, the Qwen3-8B ReAct agent's own candidate list (its actual action first, then the other actions it samples, by frequency) is screened by a yes/no verifier, and the agent acts on the highest-ranked candidate the verifier accepts, or on its own first choice if the verifier accepts none (the verifier "abstains"). The frozen result is "no reliable difference" between a verifier that sees the task and one that does not, for both susceptible verifiers (Llama-3.1-8B and Mistral-7B-v0.3), overall and on the encounter states, where the vulnerable kind of action is in the list (a failing action on the task's object whose failure the history determines, called a "regime candidate"). The contrast model, Qwen3-8B, does show a task-aware verifier choosing failing actions more often.

**The short answer.** The nulls are robust, and the reason for them is visible in the data: the bias is present in the priority verifiers' scores but almost never decides which action is taken. Qwen's effect is real but runs through a different mechanism from the one RQ2 measures.

1. **The decisions reproduce exactly.** Code that shares no decision or statistics function with the frozen module gives the same failure rates, differences, harm and help counts and state sets for all three models; its intervals differ from the frozen ones only by resampling noise (at most 0.4 points) (section 1).
2. **The priority nulls are stable.** No game and no task type carries them: leaving out any one game moves Llama's overall difference between −0.03 and +0.51 points and Mistral's between +0.07 and +0.20. Llama's changes are spread over 41 games that net out (22 positive, 19 negative); Mistral changes its choice in only 8 games (sections 2 and 3).
3. **Llama's encounter-state estimate rests on one game.** Five of its seven harm states on encounter states are one repeated action in one game (`valid_unseen/150`); without that game the conditional difference is +0.9 points instead of +4.7 (section 3).
4. **The bias is in Llama's scores but is blocked by the rule.** Llama's task-aware verifier accepts 110 of the 140 regime candidates at multi-candidate states, against 82 task-blind. But 68 of the 73 regime alternatives it accepts sit below a candidate it accepted first, so they are never chosen; and where the regime candidate is the agent's own first choice, the task-blind verifier usually accepts it too (32 of 42), so the extra Yes changes nothing. That leaves 7 harm states (section 5).
5. **At encounter states both verifiers mostly fail anyway.** At 54 of the 129 encounter states (Llama; 58 for Mistral) both verifiers choose the regime candidate. 33 encounter states list no working action at all, including all 7 strict encounters, so no verifier could have done better there (section 4).
6. **Why Qwen abstains and the others do not.** Under any concrete task, Qwen's yes/no scores for working actions fall by about 39 nats at step 0 and by about 2.5 to 5.5 nats later, from task-blind medians of about +7 to +17, so many cross zero; the swapped task does the same. Llama's and Mistral's scores for working actions move by about a nat or less after step 0 and stay positive. Qwen's harm is almost all "lost rescues": the agent's first choice fails, the task-blind verifier rejects it and picks a working alternative, and the task-aware verifier rejects that alternative too and so abstains back to the failing first choice (69 of 88 harm states) (section 6).
7. **Stage 1a's mechanism, checked ingredient by ingredient.** The first ingredient (a blanket No to working first moves at step 0 under any concrete task) reproduces for Qwen almost exactly (4% and 2% Yes under the real and swapped task, against 90% task-blind; Stage 1a: 7%, 6% and 90%). The second ingredient (a fallback menu holding failing goal-object actions) is absent: the agent's step-0 lists hold no failing candidate at all, and the rule abstains instead of falling back. After step 0, Stage 1b's Qwen difference on Stage 1a's own step band is +1.3 points against Stage 1a's +7.6 (section 7).
8. **Precision.** Llama's overall interval excludes harm above 1.3 points per decision and Mistral's above 0.5; on encounter states, Mistral's excludes harm above 2.1 points per encounter state, Llama's only above 12.5. Stage 1a's effects for Qwen (+19.5) and Mistral (+3.3) are excluded by either model's overall interval; Stage 1a's Llama effect (+1.1) is not excluded by Llama's (section 8).

**What this means.** The answer to "does the bias reach this agent's decisions?" is no for both susceptible verifiers, and the reason is the agent and the rule, not an absence of bias in the verifier. Llama shows the RQ2 signature in its scores (about 22 points more Yes on regime candidates under the real task), but the agent rarely lists such an action where the verifier would pick it. For Llama the bound where the vulnerability can act is loose, so "harmless where it applies" is not supported for Llama. For Mistral it is tight, but Mistral's verifier is so permissive that it changes few decisions under any prompt.

## 1. Independent recompute

The script's `item1` rebuilds every list from the labels (normalising and hashing actions itself), applies the rule with its own rounding to 1/64 nat, and computes intervals with its own game bootstrap (2,000 resamples, numpy) and its own sign-flip permutation (10,000 flips).

| | Llama-3.1-8B | Mistral-7B-v0.3 | Qwen3-8B |
|---|---|---|---|
| Chosen-failure rate (%): none, aware, blind | 22.57, 18.16, 18.00 | 22.57, 21.96, 21.80 | 22.57, 15.98, 13.69 |
| Overall difference, recomputed (points) | +0.17 [−0.98, +1.32] | +0.17 [−0.10, +0.49] | +2.29 [+1.40, +3.29] |
| Overall p (10,000 flips) | 0.82 | 0.38 | 0.0001 |
| Overall harm / help | 48 / 43 | 8 / 3 | 88 / 20 |
| Conditional difference, recomputed (129 states) | +4.65 [−1.15, +12.41] | 0.00 [−2.52, +2.15] | +15.50 [+8.70, +22.88] |
| Conditional p (10,000 flips) | 0.51 | 1.00 | 0.0015 |
| Conditional harm / help | 7 / 1 | 1 / 1 | 22 / 2 |

Every point estimate, count and state set equals the frozen output exactly; the intervals and p-values differ only by resampling (for example, Llama's frozen overall interval is [−0.92, +1.27] and Qwen's frozen conditional interval [+8.8, +23.3]).

## 2. Per-game stability

The net of a game is its harm states minus its help states (aware against blind).

| | Llama | Mistral | Qwen |
|---|---|---|---|
| Games with a positive / negative / zero net (134 games) | 22 / 19 / 93 | 5 / 3 / 126 | 36 / 7 / 91 |
| Games with any harm state | 26 | 5 | 40 |
| Most harm states in one game | 6 | 3 | 6 |
| Share of harm in the five heaviest games | 50% | 100% | 32% |
| Largest single-game nets | −10, +6, +6, +5, −5 | +3, +2 | +6, +6, +6, +5 |
| Encounter states: games positive / negative / zero (22 games) | 3 / 1 / 18 | 1 / 1 / 20 | 10 / 0 / 12 |

Llama changes decisions in many games, in both directions, and they cancel; its largest single contribution is a game where the task helps (net −10 in `valid_unseen/93`). Mistral changes decisions in so few games that no test could find much. Qwen's harm is spread (no game holds more than 6 of its 88 harm states) and its game nets are mostly positive.

**By task type** (aware minus blind, points; the type is parsed from the task line as in Stage 1a's deep-dive):

| Task type (games) | Llama | Mistral | Qwen |
|---|---|---|---|
| Pick (24) | +0.83 [0.00, +1.77] | 0.00 | +0.83 [0.00, +1.85] |
| Pick two (17) | −2.00 [−4.58, +0.20] | +0.20 | +4.00 [+1.99, +6.28] |
| Clean (31) | −0.53 [−3.78, +2.41] | −0.13 | +1.73 [0.00, +3.29] |
| Heat (23) | +1.63 [−0.93, +4.55] | −0.20 | +3.67 [+1.03, +6.65] |
| Cool (21) | +0.18 [−0.92, +1.29] | +0.18 | +1.29 [−0.38, +3.79] |
| Examine (18) | +2.15 [−0.81, +6.16] | +1.54 [−0.37, +3.83] | +2.15 [−0.87, +6.27] |
| Leaving one type out: range | −0.12 to +0.61 | 0.00 to +0.27 | +1.94 to +2.51 |

No task type gives Llama or Mistral an interval wholly above zero; 6 of Mistral's 8 harm states are in examine tasks. Qwen is positive in all six types.

## 3. Leave one game out

| Range with each game left out (points) | Llama | Mistral | Qwen |
|---|---|---|---|
| Overall (frozen estimate) | −0.03 to +0.51 (+0.17) | +0.07 to +0.20 (+0.17) | +2.11 to +2.41 (+2.29) |
| Game that moves it most | `valid_unseen/93` (to +0.51) | `valid_unseen/13` (to +0.07) | `valid_unseen/145` (to +2.11) |
| Conditional on encounter (frozen estimate) | +0.86 to +5.47 (+4.65) | −0.87 to +0.81 (0.00) | +13.71 to +17.39 (+15.50) |
| Game that moves it most | `valid_unseen/150` (to +0.86) | `valid_unseen/190` (to −0.87) | `valid_unseen/190` (to +17.39) |

The overall nulls survive the removal of any game. Llama's conditional estimate does not: one game supplies most of it. In `valid_unseen/150` the agent keeps proposing `move apple 2 to garbagecan 1`, a determined failure (it is not at the garbage can), and the task-aware Llama verifier lets it through at steps 13, 14, 15, 16 and 20, where the task-blind verifier picks a working action. This is one decision repeated in a loop, which the frozen first-occurrence sensitivity also discounts (Llama there: −0.32 points [−0.62, 0.00]).

## 4. Encounter states

**What the 129 encounter states look like.** They lie in 22 games, very unevenly: six games hold 13 or 14 each (81 states), and eleven games hold one or two. Six are at steps 1–11 and 123 at steps 12–29 (median step 22); none is at step 0. 19 have a single candidate, which is then a regime candidate the agent takes whatever the verifier says. The agent's own first choice is a regime candidate at 61 of them (including those 19). 96 list at least one working candidate and 33 list none. At 55 the first working candidate is ranked above the first regime candidate, so a verifier could choose the regime candidate only by rejecting the working one; at the other 74 the regime candidate comes first ("reachable"), and 33 of those have no working candidate at all.

**What the verifiers chose there.** Counts of the 129 states by the kind of action each verifier chose: a regime candidate, another failing action, or a working action.

| Aware chose / blind chose | Llama | Mistral | Qwen |
|---|---|---|---|
| regime / regime | 54 | 58 | 58 |
| other failure / other failure | 28 | 28 | 10 |
| working / working | 39 | 41 | 36 |
| regime / working (harm) | 7 | 1 | 15 |
| other failure / working (harm) | 0 | 0 | 7 |
| working / regime or other failure (help) | 1 | 1 | 2 |
| regime / other failure | 0 | 0 | 1 |

At encounter states the verifiers usually fail together: both choose a failing action at 82 states for Llama and 86 for Mistral, about two thirds of the encounter states, because the agent's own first choice is the failing action and both verifiers accept it, or because nothing listed works.

**Every state where Llama's two verifiers differ** (10 states). There are 7 harm states, in which the task-aware verifier chose a regime candidate and the task-blind one a working action: `take tomato 3 from sinkbasin 1` in `valid_unseen/146` (step 25), `move apple 2 to garbagecan 1` in `valid_unseen/150` (steps 13, 14, 15, 16, 20) and `move soapbar 1 to cabinet 1` in `valid_unseen/187` (step 20). There is one help state (`valid_unseen/60`, step 12: the aware verifier kept the working `go to shelf 5` where the blind verifier took `move pencil 1 to shelf 5`) and there are two states where both choices work. In 5 of the 7 harm states the task-aware verifier accepted the regime candidate, which the task-blind verifier rejected (path ii); in the other 2 the task-blind verifier rejected everything and abstained to a working first choice (path iv). **Mistral's** two differences are one harm (`valid_unseen/190`, step 17: `move soapbar 1 to garbagecan 1` chosen over `inventory`) and one help (`valid_unseen/114`, step 17). **Qwen's** 31 differences are listed in the metrics file. Of its 15 regime harm states, 9 are states where the task-aware verifier accepted the regime candidate (5 games, 4 of the states in `valid_unseen/187`), and 6 are abstentions that returned a regime candidate at rank 1, which Qwen's task-aware verifier had itself rejected.

**Strict encounter states** (the RQ2-like subset, 7 states in 6 games). None lists a working candidate, so every verifier, aware or blind, chooses a failing action at all 7, and no difference is possible there.

## 5. Regime candidates through the selection rule

Each of the 159 regime candidates in the primary list, followed through the rule. 19 sit at single-candidate states and are taken whatever the verifier says; the counts below are for the 140 at multi-candidate states (42 at rank 1, 98 alternatives).

| | Llama | Mistral | Qwen |
|---|---|---|---|
| Accepted by the task-aware verifier | 110 (79%) | 127 (91%) | 33 (24%) |
| Accepted by the task-blind verifier | 82 (59%) | 123 (88%) | 29 (21%) |
| Accepted by the aware verifier only / by the blind verifier only | 28 / 0 | 4 / 0 | 6 / 2 |
| Accepted by the swapped verifier, of the 93 with a swap object (aware on the same 93) | 55 (aware 70) | 76 (aware 80) | 10 (aware 18) |
| Rank 1: the aware verifier accepts and chooses it / the blind verifier chooses it | 37 / 34 | 39 / 39 | 10 / 32 |
| Alternatives: the aware verifier accepts / chooses | 73 / 5 | 88 / 1 | 23 / 17 |
| Accepted alternatives not chosen because a higher-ranked candidate was accepted first: working / other failure / regime | 36 / 20 / 12 | 44 / 25 / 18 | 5 / 0 / 1 |
| Regime harm (aware chose the regime candidate, blind chose a working action) | 7 (3 at rank 1, 4 alternatives) | 1 | 15 (9 accepted, 6 by abstention) |

For Llama the RQ2 bias is visible at the level of acceptance: under the real task it accepts 28 regime candidates the task-blind verifier rejects, and never the reverse; the swapped task accepts fewer than the real one (55 against 70 on the same candidates), so naming the real object is what does it. The rule then neutralises most of it. An accepted regime alternative is chosen only if every candidate ranked above it is rejected, and Llama accepts the agent's own first choice nearly always, so 68 of its 73 accepted regime alternatives are never chosen. When the regime candidate is the agent's own first choice, the task-blind verifier also accepts it (34 of the 37 the aware verifier chooses), so the task changes nothing.

Qwen is the opposite case: it rejects most regime candidates under every arm, and its task-aware verifier ends up "choosing" 28 rank-1 regime candidates that it rejected, by abstaining. Its task-blind verifier chooses 32 rank-1 regime candidates, so on regime candidates at rank 1 the two arms mostly agree.

## 6. Why Qwen abstains, and why the priority models do not

**Abstention by step band** (percent of multi-candidate states where the verifier accepts nothing).

| Step band (states) | Llama aware / blind / swapped / neutral | Mistral aware / blind / swapped / neutral | Qwen aware / blind / swapped |
|---|---|---|---|
| Step 0 (65) | 1.5 / 0.0 / 3.8 / 0.0 | 6.2 / 0.0 / 7.5 / 0.0 | 98.5 / 0.0 / 100.0 |
| Steps 1–11 (548) | 1.1 / 0.5 / 2.4 / 0.9 | 0.0 / 0.0 / 0.2 / 0.0 | 21.5 / 5.3 / 21.9 |
| Steps 12–29 (1,071) | 3.4 / 2.2 / 4.2 / 3.0 | 0.0 / 0.0 / 0.0 / 0.0 | 29.2 / 20.2 / 31.4 |

Abstention is harmless when the agent's first choice works, and also when nothing in the list works (40 of the multi-candidate states list no working action). It is costly when the first choice fails and a working candidate is listed. That happens at 170 states for Qwen's task-aware verifier (22 at steps 1–11, 148 at steps 12–29), against 103 for its task-blind verifier, 28 and 15 for Llama's two verifiers, and none for Mistral's.

**Why: the scores of working actions.** Median raw score $L$ in nats of the agent's working candidates at multi-candidate states, task-aware against task-blind, and the median paired shift (aware minus blind).

| Working candidates (number) | Llama aware / blind (shift) | Mistral aware / blind (shift) | Qwen aware / blind (shift) |
|---|---|---|---|
| Rank 1, step 0 (65) | +2.4 / +4.8 (−1.0) | +2.5 / +10.2 (−7.6) | −20.3 / +20.3 (−39.0) |
| Rank 1, steps 1–11 (480) | +2.8 / +3.0 (−0.3) | +5.5 / +7.1 (−1.3) | +7.3 / +17.0 (−4.3) |
| Rank 1, steps 12–29 (647) | +2.1 / +2.4 (−0.1) | +5.0 / +5.8 (−0.6) | +2.0 / +8.3 (−2.8) |
| Alternatives, step 0 (95) | +1.3 / +4.8 (−2.5) | +2.6 / +9.3 (−5.3) | −24.0 / +18.0 (−33.3) |
| Alternatives, steps 1–11 (858) | +3.0 / +3.1 (−0.3) | +5.3 / +6.8 (−1.1) | +3.0 / +15.5 (−5.5) |
| Alternatives, steps 12–29 (1,912) | +2.1 / +2.3 (−0.1) | +4.6 / +5.3 (−0.5) | +1.0 / +6.8 (−2.5) |

For Qwen the swapped task gives nearly the same scores as the real task (median aware-minus-swapped shift +1.3 nats on rank 1 at step 0, 0.0 at steps 1–11 and +0.3 at steps 12–29), so it is the presence of a concrete task, not the object it names, that lowers Qwen's scores for working actions. The fall is largest at step 0 (about 39 nats, the same size as in Stage 1a) and persists, smaller, at every later step. There it starts from lower task-blind scores, so 37% to 48% of working actions after step 0 score at or below zero under the real task, against 20% to 37% task-blind. Llama's scores for working actions sit only 2 to 3 nats above zero, but the task moves them by a quarter of a nat or less after step 0, so they stay positive. Mistral's step-0 shift is sizeable (−7.6 nats) but starts from about +10, so most first moves remain accepted (74% at multi-candidate states), and its later shifts are about one nat from a base of +5 to +7.

**How the harm and help states arise** (aware against blind, all 2,973 states). "The failing side" is the verifier whose chosen action fails.

| Route | Qwen harm | Qwen help | Llama harm | Llama help | Mistral harm | Mistral help |
|---|---|---|---|---|---|---|
| The failing side abstained and took the agent's failing first choice; the other chose a working alternative | 69 | 5 | 16 | 0 | 0 | 0 |
| The failing side accepted the agent's failing first choice; the other chose a working alternative | 4 | 6 | 17 | 34 | 6 | 2 |
| The failing side chose a failing alternative | 15 (9 regime) | 9 | 15 (4 regime) | 9 | 2 (1 regime) | 1 |

The first row is the "lost rescue": the task-blind verifier rescues the agent from one of its failing repeats by accepting a working alternative, and the task-aware verifier, scoring working actions lower, rejects that alternative too and abstains. It is 69 of Qwen's 88 harm states. For Llama the routes roughly cancel: the task makes it let through 17 failing first choices that the blind verifier stops, but also stop 34 that the blind verifier lets through.

## 7. Matched step bands against Stage 1a

Aware minus blind, points. Stage 1a's states are the even steps 0–10 of the item collection (728 states); its later band is the even steps 2–10. Stage 1a used rule 2 (fall back to the best-scored candidate of a fixed menu); its deep-dive also computed a variant that switches only to an alternative the verifier itself accepts, the closest Stage 1a analogue of Stage 1b's rule.

| | Llama | Mistral | Qwen |
|---|---|---|---|
| Stage 1a, step 0 (rule 2) | +3.0 [+0.7, +6.0] | +15.7 [+9.7, +21.6] | +72.4 [+64.9, +79.9] |
| Stage 1a, even steps 2–10 (rule 2) | +0.7 [−0.7, +2.3] | +0.5 [0.0, +1.3] | +7.6 [+5.0, +10.5] |
| Stage 1a, even steps 2–10 (switch only to an accepted alternative) | +0.7 | +0.5 | +4.4 [+2.1, +6.9] |
| Stage 1b, step 0 (134 states) | 0.0 | 0.0 | 0.0 |
| Stage 1b, even steps 2–10 (614 states) | +0.16 [−0.49, +0.83] | +0.65 [0.00, +1.50] | +1.30 [+0.33, +2.49] |
| Stage 1b, steps 1–11 (1,344 states) | +0.07 [−0.53, +0.75] | +0.37 [0.00, +0.85] | +1.27 [+0.44, +2.29] |
| Stage 1b, steps 12–29 (1,495 states) | +0.27 [−1.97, +2.45] | 0.00 [−0.32, +0.33] | +3.41 [+1.75, +5.18] |
| Stage 1b, even steps 2–10, fallback to the highest-scored candidate | +0.16 | +0.65 | +0.49 [−0.32, +1.34] |

**Ingredient (i), the step-0 blanket No.** Yes on the agent's working first move:

| Arm | Llama, Stage 1a / 1b | Mistral, Stage 1a / 1b | Qwen, Stage 1a / 1b |
|---|---|---|---|
| Real task | 89.6% / 88.6% | 84.3% / 80.0% | 7.5% / 3.8% |
| Swapped task | 84.8% / 79.1% | 78.4% / 70.3% | 5.6% / 2.2% |
| Task-blind | 100% / 100% | 100% / 100% | 89.6% / 90.3% |

It reproduces for every model on the agent's own first moves. After step 0, Qwen's Yes on working first choices under the real task is 63% at steps 1–11 and 53% at steps 12–29, against 80% and 63% task-blind: the same ingredient, weaker.

**Ingredient (ii), failing goal-object actions to fall back on.** The agent's 134 step-0 lists hold 229 candidates: 185 `go to`, 43 `look` and 1 `inventory`, every one of which works. There is no failing candidate at step 0 at all, let alone a goal-object one. And the rule does not fall back to the best-scored candidate; it abstains to the agent's own first choice, which at step 0 always works.

**Reading.** Stage 1a's step-0 harm needs both ingredients, and the agent supplies only the first, so step-0 harm is zero for every model. After step 0, Stage 1a's Qwen effect on its own band (+7.6 points) shrinks to +1.3 on the agent's lists over the same steps; Stage 1a's switch-only variant had already brought it to +4.4. The remaining Stage 1b effect grows with step (+3.4 points at steps 12–29), where the agent's failing repeats are common and abstention returns them. For Llama and Mistral the Stage 1a effects were almost entirely at step 0 and do not reappear later in either stage.

## 8. Precision

The 95% intervals of the frozen output, read as bounds. The encounter rate is $e = 0.0434$. If all harm sat at encounter states, a per-decision bound $b$ would correspond to a per-encounter-state bound of

$$
\frac{b}{e},
$$

and a per-encounter-state bound $c$ to a per-decision bound of $c \cdot e$.

| | Llama | Mistral |
|---|---|---|
| Overall: interval (points per decision) | [−0.92, +1.27] | [−0.09, +0.47] |
| Harm excluded above, per decision | 1.27 points (about 38 net states of 2,973) | 0.47 points (about 14 net states) |
| The same bound if all harm sat at encounter states | about 29 points per encounter state | about 11 points per encounter state |
| Smallest p a sign flip could give (games with a net change) | below 0.001 (41) | 0.008 (8) |
| Conditional: interval (points per encounter state) | [−1.2, +12.5] | [−2.5, +2.1] |
| Harm excluded above, per encounter state | 12.5 points | 2.1 points |
| The same bound per decision | about 0.54 points | about 0.09 points |
| Smallest p a sign flip could give (games with a net change) | 0.125 (4) | 0.5 (2) |

**Against effects measured before.** Stage 1a's Qwen effect (+19.5 points per decision), its later-step effect (+7.6), its switch-only variant (+4.1 overall) and Stage 1a's Mistral effect (+3.3) all lie above both priority models' overall upper bounds, so effects of those sizes are excluded for this agent. Stage 1a's Llama effect (+1.1, itself not significant) lies inside Llama's overall interval and is not excluded; it is excluded for Mistral. On encounter states, the design's "modest" scenario (a net 8 points per encounter state) lies inside Llama's interval and is not excluded for Llama, while it is excluded for Mistral; Qwen's observed +15.5 lies above both upper bounds. Llama's conditional test could not have reached significance with the changes it saw: with 4 games changing, the smallest attainable p is 0.125, so its null reflects how few games changed at all as much as how large the changes were.

**What would tighten it.** For Llama the conditional interval is wide because the encounter states sit in 22 games and changes in 4. More episodes, or an agent whose failing goal-object proposals are spread over more games, would narrow it. The overall interval is already narrow enough to exclude every effect size Stage 1a found except Llama's own.

## What this changes

- **Keep:** the frozen verdicts and the reading "the bias rarely reaches this agent's decisions" for both priority models. They reproduce exactly and do not depend on one game, one task type, the list length, the sampling temperature or the loop handling.
- **Add:** the mechanism of the null. The bias is present in Llama's acceptances (it accepts 28 regime candidates only under the real task, none only without it), but the "highest-ranked accepted, else rank 1" rule and the agent's lists keep it from deciding: accepted regime alternatives sit below accepted first choices, and regime first choices are accepted under both prompts.
- **Qualify:** "harmless where the vulnerability can act" holds within about 2 points per encounter state for Mistral but is not shown for Llama (up to 12.5 points, with a point estimate carried by one looping game).
- **For a closed-loop study:** a verifier that sees the task changes this agent's outcomes mainly through any-task rejection (Qwen's lost rescues), not through object naming. A closed loop built to test the RQ2 bias would need candidates that place goal-object failures above working actions, or a rule that ranks by predicted success, as Stage 1a's menus did.
