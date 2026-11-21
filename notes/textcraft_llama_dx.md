# TextCraft: why Llama-3.1-8B shows a negative goal effect (exploratory diagnostics)

Everything in this note is exploratory and post hoc. It was written after the confirmatory TextCraft results ([textcraft_results.md](textcraft_results.md), [textcraft_metrics.json](textcraft_metrics.json)) had been read, and none of it changes them. The frozen design is [textcraft_design.md](textcraft_design.md).

## Plan for the GPU diagnostics (written 2026-11-20, before they ran)

The CPU analysis of the existing scores (results below) leaves two questions open that the scores cannot answer. Two small runs on ICE answer them, at a total cost under one GPU-hour. Both run from a new checkout (`text-wm-fix26`) at this branch, never in `text-wm-fix24`, and write to new run ids; the confirmatory runs are not touched.

**1. Zero-shot scoring (`textcraft_dx_zeroshot`, both models).** The verified prompt with its three in-context examples removed: system text, then the item's turn, nothing else changed. It scores 150 seeded main couples (300 items) and 100 seeded success couples (200 items) under the aligned, swapped and neutral goal lines, about 3,000 scores. Each is compared with the verified prompt's scores on the same items, by pair-clustered bootstrap, with no p-value.

- *If Llama's negative ΔL, and its concentration in the goal role, persist without examples*, the examples are not needed for the reversal. Pattern copying from the examples is then not its cause.
- *If it vanishes or turns positive*, the examples create it. The examples-on-goal variant already showed that the off-goal example does not, so the cause would lie in the other two examples. A further run permuting or replacing them would be needed.
- Losing the examples may also loosen format adherence (Llama's mass on the two first tokens is already only 0.8 to 0.9). The first-token mass is reported per condition, so that a change in ΔL can be told apart from a model that stops answering in the format.

**2. Free generation and first-token alternatives (`textcraft_dx_generate`, Llama only).** Greedy generation, at most 40 new tokens, and the ten most probable first tokens with their probabilities, under the verified prompt (examples included). It covers 60 seeded goal-role main couples, 30 ingredient main couples and 30 goal-role success couples (240 items) under all three goal lines, 720 generations.

- It shows what Llama predicts when free: a success reply, the failure reply, or something else. It also shows what takes the 10 to 20% of first-token mass that neither candidate gets.
- *If, for goal-role failures, the swapped line turns the generated reply from "Could not find enough items" to "Crafted" much more often than the aligned line does*, the negative ΔL is a change in the predicted outcome and not an artefact of teacher forcing two fixed strings.
- *If the generations are mostly some third text* (a goal message, an inventory, a refusal), and that text depends on the goal line, the two-candidate score reflects a format effect, and the reversal should be read as one.

Neither run is a test, and no outcome of either changes the confirmatory reading.

**3. Object-goal line on ingredient actions (`textcraft_dx_objectgoal`, both models; added 2026-11-20 after the CPU analysis below, before it ran).** Llama's negative ΔL sits in the goal role, where the aligned line names the action's object verbatim *and* the action completes the task. On ALFWorld, Llama's effect was almost all "mention": naming the action's object raised predicted success. TextCraft has no item where the goal line names the action's object without the action completing it. The goal-role reversal could therefore be a completion effect (a craft that finishes the task looks unlikely), or a negative mention effect. This run scores ingredient actions (an action crafting Y, a direct ingredient of the aligned goal G) under a third goal line, `Goal: craft Y.`: the action's own object made the goal, whose recipe lines are already on screen because they are part of G's tree. It scores 150 seeded ingredient main couples (300 items) and 100 seeded ingredient success couples (200 items) as the two-candidate score and as the yes/no question, about 2,000 scores. The object-goal line is compared with the verified aligned (`Goal: craft G.`) and swapped (`Goal: craft P.`) scores of the same items.

- *If L under "craft Y" is below L under "craft G" for Llama*, making the action's object the goal lowers predicted success, as on the goal role: completion caution, not tied to the top items of the trees.
- *If it is above*, naming the action's object raises predicted success, as on ALFWorld (mention). The goal-role reversal would then be specific to the top items, or to something else about the goal role.
- The same contrast on successes says whether any shift is a change in judgement of failures only, or a level shift for every craft of the named object.
