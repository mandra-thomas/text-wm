# TextCraft: how competent is Llama-3.1-8B as a world model, and what does it do instead? (exploratory, post hoc)

Everything in this note is exploratory and post hoc. It was written on 2026-11-21, after the confirmatory TextCraft results ([textcraft_results.md](textcraft_results.md)), the Llama diagnostics ([textcraft_llama_dx.md](textcraft_llama_dx.md)) and the cross-environment comparison (`notes/textcraft_vs_alfworld.md` on branch `xenv-compare`) had been read. It uses only scores that already exist; nothing was scored for it. It changes none of the confirmatory verdicts.

*Summary: to be written with the results.*

## What this note asks

The confirmatory TextCraft run found that Llama-3.1-8B-Instruct separates working from failing crafts only weakly under the neutral goal line (AUC 0.68 with the usefulness failures, 0.61 on the main failures alone), and that it ranks "used up" failures above real successes (AUC 0.31). This note asks five things, with Qwen3-8B (thinking off) alongside as the reference throughout:

1. The full competence profile: how well each model separates successes from failures, on both reply scales, by failure class, role and history length; how often it predicts success; how calibrated its probabilities are; and whether it passes the competence rule the model-families study fixed in advance.
2. What Llama does instead of tracking the inventory: which simple rule, computable from the prompt text, its scores follow.
3. Whether its scores respond to the quantities that decide the outcome (how many units are missing, how long ago an input was used up, how many units are spare).
4. How competence relates to the goal effect, item by item and stratum by stratum.
5. Whether the incompetence is specific to TextCraft, by comparison with Llama in ALFWorld and with Qwen in TextCraft.

It ends with what the incompetence does and does not explain about the TextCraft results, and how the paper should describe the Llama result.

## Heuristics and fit measures (fixed 2026-11-21, before any fit to either model was computed)

This section was committed on its own before any heuristic was compared with any model's scores, so that the set of candidate explanations is not chosen after seeing which one fits. At the time of writing, what was known about the models' scores was what the earlier notes report (overall and per-class AUCs, the role pattern, the used-up AUC of 0.31). No heuristic had been computed on any item.

### Items and notation

The items are the 1,500 main failures and the 600 determined successes of the confirmatory run (`wf/textcraft/data/items.jsonl`, populations `main` and `success`; 2,100 items from 247 goal pairs; the successes come from 167 of them). Every test action is a craft, `craft K Y using c1 n1, c2 n2, ...`, with inputs $(c_j, n_j)$: $c_j$ units of item $n_j$. The outcome is success exactly when every input is held in at least its count just before the action.

Every heuristic reads only what the prompt shows: the displayed recipe lines, the history's (action, reply) pairs and the action. In the history:

- an *obtaining step* for an item X is a reply `Got N X` (N units obtained) or `Crafted K minecraft:X` (K units obtained, the underscores read as spaces);
- a *consuming step* for X is a history craft whose action lists X as an input (the history's crafts always succeed, so the listed units were removed);
- an item is *mentioned* at a step when its name appears there as an item: the object of a `get`, an input or the output of a `craft`, the object of a `Got` or `Crafted` reply, or an entry of an `Inventory:` reply.

The held amount $h_j$ of an input is the text-only checker's accounting (`wf/textcraft/checker.account`), the same accounting that defines the outcome.

### The heuristics

Each heuristic predicts success or failure for an item. Several also have a graded score, used for rank correlations; higher means "more likely to succeed".

- **H1, name seen.** Success when every input's name is mentioned somewhere in the history. It ignores counts and consumption.
- **H2, gross obtained (ignores consumption).** Success when, for every input, the total number of units obtained over the history (the sum of N and K over its obtaining steps) is at least $c_j$. It counts what came in and ignores what went out. Graded score: the smallest, over the inputs, of the obtained total divided by $c_j$, capped at 1.
- **H3, obtained recently.** Success when every input was obtained in one of the last $k$ history steps, with $k = 3$ as the main setting and $k = 1$ and $k = 6$ reported as sensitivity. It ignores counts and consumption.
- **H4, mentioned recently.** Success when every input's name is mentioned in one of the last 3 history steps, a mention as the input of a craft (its consumption) included. Graded score: minus the number of steps since the least recently mentioned input was last mentioned (an input never mentioned counts as the history length plus one).
- **H5, long history.** Success when the history has at least 4 steps (the lower edge of the middle length bin). Graded score: the history length.
- **H6, true accounting (the environment's rule).** Success when $h_j \ge c_j$ for every input. This is the outcome itself, so its accuracy is 1 by construction. Graded score: the smallest, over the inputs, of $h_j / c_j$, capped at 1.
- **H7, held at all (tracks consumption, ignores counts).** Success when $h_j \ge 1$ for every input.
- **H8, last event is a gain.** Success when, for every input, its most recent obtaining or consuming step is an obtaining step (an input with neither counts as failing). It notices a consumption only if it is the input's last event, and ignores counts.
- **H9, not a top item.** Failure when the action's output is a top item of the displayed list, that is, an item that no displayed line takes as an input (a `planks` or `wool` input counts as taking each of its members); success otherwise. In this design the top items are exactly the two goals of the pair, so under the neutral line, which names neither, this is "crafting a finished goal fails".
- **H10, always fail.** Predicts failure for every item. It has no graded score.
- **H2 ∧ H9 (one combination, fixed now).** Success when H2 predicts success and the output is not a top item.

### How a heuristic is compared with the truth and with a model

All comparisons use the neutral goal line ("Goal: explore the crafting commands."), the two-candidate log-odds L as the main scale and the yes/no log-odds as the second. Intervals are 95% percentile intervals from 2,000 resamples of whole goal pairs, as in the frozen analysis. There are no p-values.

1. **Validity.** The AUC of the heuristic's prediction (or graded score) against the true outcome, with ties counted one half, over all 2,100 items and for each failure class against all successes. This says how good a world model the heuristic would be.
2. **Tracking AUC (the main fit measure).** For a binary heuristic, the AUC of the model's L between the items the heuristic calls a success and those it calls a failure: how well the model's ranking follows the heuristic. For H10 it is undefined. For the true rule H6 it equals the model's competence AUC, so each heuristic's tracking AUC can be set against the truth's.
3. **Rank correlation.** For heuristics with a graded score, the Spearman correlation between that score and L.
4. **Decision agreement.** The share of items on which the model's decision (success when L > 0) agrees with the heuristic, and Cohen's kappa, which corrects that share for agreement expected by chance from the two base rates. Because Llama predicts failure for most items, raw agreement favours H10; kappa is the comparable figure.
5. **Within the truth.** The tracking AUC computed separately among the true failures and among the true successes. A model that follows the truth and not the heuristic has a tracking AUC near 0.5 within each; a model that follows the heuristic has one above 0.5 within the failures, where the heuristic and the truth disagree.
6. **Joint regression.** A least-squares fit of L on an intercept, the true outcome (0 or 1) and the heuristic's prediction (0 or 1), with pair-bootstrap intervals for the two slopes. It says how much of the model's separation goes with the truth once the heuristic is held fixed, and the reverse.

**Reading, fixed now.** The best-fitting heuristic is the one with the highest tracking AUC for Llama over all 2,100 items (for graded heuristics, the highest Spearman correlation is reported beside it). A heuristic *explains* Llama's used-up result if (a) it does not call used-up failures less likely to succeed than real successes (its own AUC, used up against successes, is at or below 0.5), and (b) among used-up failures and successes together, Llama's L follows it more closely than it follows the truth (higher tracking AUC than H6 on that subset). The same fits are computed for Qwen, as a model that does track the state. As a secondary check, H2 and H6 are also fitted on the exploratory inventory-shown prompt (`textcraft_dx_prompts`, 500 items), where the inventory before the action is on screen.

*Results follow in a later commit.*
