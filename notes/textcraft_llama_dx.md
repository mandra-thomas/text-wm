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

What it can and cannot separate (corrected before the run): the object-goal line names the action's object *and* makes the action complete the goal, so it carries the same two features as the goal role and cannot tell mention from completion. It asks a narrower question: does the goal-role reversal follow whichever displayed item is made the goal, or is it tied to the two trees' top items?

- *If L under "craft Y" is below L under "craft G" for Llama*, completing a named goal lowers predicted success for any item, as on the goal role. The reversal is about the action finishing the named task, with mention, if it acts at all, outweighed.
- *If it is above*, naming and completing raises predicted success on ingredients (the ALFWorld-like direction). The goal-role reversal would then be tied to the top items, for example to how plausible it looks that the whole task is finished.
- The same contrast on successes says whether any shift is a change in judgement of failures only, or a level shift for every craft of the named object.

**4. Prompt cleanliness and explicit inventory (`textcraft_dx_prompts`, both models; added 2026-11-20 at the reviewers' request, before it ran).** The verified system text reproduces AgentGym's instructions to the *agent* ("Every round I will give you an observation, you have to respond an action…", the reminders about quantities, and "get an object from the inventory or the environment"). A world model reading agent instructions might confuse its role, and the "from the inventory or the environment" phrase hints at a hidden store. Three prompt variants are scored on 150 seeded main couples and 100 seeded success couples (500 items), under the aligned, swapped and neutral lines, both models, two-candidate scores, about 9,000 scores:

- *clean*: the system text keeps the world-model framing and states only the environment's rules and the output format: the craft command format, that an action is `get`, `inventory` or `craft` with one of the displayed commands, and that a generic ingredient such as "planks" may be filled by a specific kind. Every agent-directed sentence is removed, and so is "from the inventory or the environment". It does not state that the inventory starts empty (that was the empty-start variant).
- *inventory*: the verified prompt, with one more history step before the action, `> inventory` followed by the environment's exact inventory reply at that point (rendered from the text-only checker's accounting, which reproduces the environment's reply format and order; a test checks this against the environment). The action, the goal lines and the candidates do not change. The state that decides the outcome is then on screen.
- *clean + inventory*: both changes.

Reading, in this order. First competence: the neutral-line AUC of L between successes and failures, overall and per class and role, against the verified prompt on the same items. Only if a variant raises Llama's competence is its ΔL read as "the goal effect of a model that has the state". Then ΔL, per role, against the verified prompt.

- *If Llama's competence rises with the inventory shown and a positive ΔL appears*, the negative effect depends on the model reconstructing the state from the history. The goal line would then act where the state is uncertain, as on ALFWorld.
- *If competence rises and the goal-role effect stays negative*, the reversal is about the goal role itself (task completion), not about state reconstruction.
- *If the clean prompt changes ΔL a lot*, the agent instructions in the system text shape the effect, and the confirmatory prompt is one reasonable prompt among others, not neutral.

**5. Free generation and feasibility, extended (replaces the scope of run 2; added 2026-11-20, before run 2 started).** Run 2 is widened to a seeded sample, drawn with its own seeds and so held out from the other diagnostics' draws, of 60 goal-role and 30 ingredient main couples and 30 goal-role and 30 ingredient success couples (300 items). Under each of the three lines Llama (a) generates the next observation greedily and (b) separately answers the yes/no question ("Will this action succeed? Answer Yes or No.") by greedy generation of at most 4 tokens. Each is judged against the simulator's true reply (stored with the item): an observation as success, the exact failure reply, another failure text or other, and an answer as Yes, No or other. The comparison is aligned against swapped on the same items: the share of correct judgements, and the share predicting success, on failures and on successes separately.

**6. Terminal against intermediate crafting (reviewer's test C; designed 2026-11-20, partly run).** The existing goal and ingredient populations differ in more than role: in the goal × never-obtained cell the missing input is itself a craftable intermediate in 161 of 252 items, against 61 of 248 in the ingredient × never-obtained cell, and goal crafts have more short inputs. A balanced test holds the craft operation fixed and changes only whether it completes the task:

- *Intermediate to terminal* is run 3: an ingredient craft of Y under "craft G" (intermediate) and under "craft Y" (terminal), the same action, history and recipe lines.
- *Terminal to intermediate* would score a goal craft of G under "craft G" (terminal) and under "craft Z", where Z is an item of the database whose recipe consumes G. Z's recipe lines (and those of its other inputs) would be added to the displayed list, as the environment would show them for goal Z. Only items whose added lines share no craftable with the partner's tree would qualify. It needs the recipe database to build the extra lines, a new item builder with its own checks (replay, the text-only checker, the prompt diff), and about 500 items under 2 lines for both models, about 2,000 scores or 15 GPU-minutes. It changes the displayed list as well as the goal line, which is its main weakness. It is not run in this round; run 3 answers the first direction.

In both directions the terminal line also names the action's object, so neither can separate task completion from naming the object: TextCraft's one-item goal format has no line that names an item without making it the goal. As a CPU complement, goal and ingredient items are compared after matching on class, length bin, whether the missing inputs are craftable, and the number of short inputs.

**7. Clustering by recipe family (CPU; families defined 2026-11-20 before any effect was computed on them).** Goal pairs may share recipe structure across pairs (for example, many trees use sticks or planks), so pair-clustered intervals may be too narrow. Two family definitions, fixed now:

- *Shared-craftable components*: two goal pairs are linked when their displayed lists share any craftable item (the output of a line), and a family is a connected component.
- *Primary material*: each pair's family is the fetchable input that appears on the most lines of its displayed list (ties broken alphabetically).

If the first definition puts more than half of the 229 main pairs into one component, it is reported but not used for an interval, because one cluster holding most of the data gives no usable interval. The interval for the second is computed in every case. The Llama ΔL and its goal-role part are reported with intervals that resample whole families.

## Data and method

The scores are those of the confirmatory run `textcraft_score` (51,200 records, ICE jobs 7204417–7204420 at `5a3702a`) and of the two exploratory prompt variants of design section 13, D10 (`textcraft_score_examples_on_goal`, job 7206341, 1,600 records; `textcraft_score_empty_start`, job 7206342, 2,400 records; both at `5a3702a`). They were copied off the cluster and read with `scripts/textcraft_llama_dx.py`. The variants were read the way the frozen analysis's `exploratory_variants` reads them: every record's item hash was recomputed with the variant's own system text and examples (no stale record in either run, none missing), and each variant's ΔL is compared with the verified prompt's ΔL on the same items. The confirmatory output `runs/textcraft_analysis` was not rerun or touched.

Every estimate below is a mean over items with a 95% percentile interval from 2,000 resamples of whole goal pairs, as in the frozen analysis. There are no p-values. Throughout, L is the log-odds of the success reply against the failure reply, and the goal effect of an item is

$$\Delta L_i = L_i(\text{aligned}) - L_i(\text{swapped}).$$

"Aligned minus neutral" and "swapped minus neutral" are the same difference with the neutral line ("Goal: explore the crafting commands.") in place of one of the goal lines. The *goal role* is an action that crafts the aligned goal itself, and an *ingredient* action crafts a direct ingredient of it. Same-history couples, the classes (used up, too few, never obtained) and the length bins are those of the design. Many cuts were looked at, so a single interval that excludes zero is weak evidence on its own; the findings below rest on patterns that recur across cuts.

## Findings from the existing scores (CPU)

### Independent recompute

The per-item scores were re-read from the raw scoring records with code that shares nothing with `wf/textcraft/analysis.py` except the items file (a different bootstrap implementation, numpy resampling with weights per pair). Llama's ΔL comes out at −0.19 [−0.29, −0.09] over 1,500 items in 229 pairs, against −0.193 [−0.298, −0.092] in the metrics; Qwen's at +0.94 [+0.60, +1.27], against +0.94 [+0.63, +1.24]. The role × class cells reproduce the metrics to two decimals (Llama goal × never obtained −0.75, goal × too few −0.52, ingredient × never obtained −0.11, ingredient × too few +0.27, ingredient × used up −0.02).

Llama's per-pair mean ΔL is negative in 66% of the 229 pairs, with quartiles −0.79, −0.29 and +0.16. Leaving out any one pair moves the overall ΔL only between −0.21 and −0.18, and leaving out the ten most negative pairs leaves −0.12 [−0.22, −0.03]. The L distributions under the three lines overlap almost completely (medians −1.17 aligned, −1.36 swapped, −2.03 neutral; 5th to 95th percentiles about −8 to +4). The first-token mass, the probability Llama puts on the two candidates' first tokens ("Craft" and "Could", the same pair on every main item), averages 0.84 aligned, 0.83 swapped and 0.82 neutral, with 6–8% of items below one half. It does not differ between aligned and swapped on the goal role (difference +0.00), and ΔL is the same on items with high and low mass. Qwen's mass is 1.000.

### Where the effect sits: the goal role, all of it

| Cut (main failures) | Llama ΔL | Qwen ΔL | Items, pairs |
|---|---|---|---|
| Goal role, all | −0.63 [−0.79, −0.46] | +1.05 [+0.50, +1.62] | 510, 160 |
| Ingredient role, all | +0.03 [−0.07, +0.13] | +0.88 [+0.48, +1.31] | 990, 162 |
| Goal × never obtained | −0.75 [−0.93, −0.58] | −0.45 [−0.87, −0.04] | 252, 103 |
| Goal × too few | −0.52 [−0.77, −0.25] | +2.52 [+1.58, +3.57] | 252, 83 |
| Ingredient × never obtained | −0.11 [−0.29, +0.06] | +1.64 [+1.03, +2.27] | 248, 93 |
| Ingredient × too few | +0.27 [+0.01, +0.51] | +2.28 [+1.32, +3.39] | 248, 46 |
| Ingredient × used up | −0.02 [−0.11, +0.08] | −0.20 [−0.57, +0.17] | 494, 95 |

Weighted by item counts, the goal role contributes −0.21 of Llama's −0.19 and the ingredient role +0.02. On the 990 ingredient items Llama shows no goal effect at all. The negative primary is a goal-role effect diluted by a null elsewhere.

The goal × never-obtained cell is negative for both models, which makes it the place to start. Its failure reply is always the same kind as elsewhere ("Could not find enough items to craft minecraft:…"), and its success twin is "Crafted K minecraft:…". In only 5 of its 252 items does a missing input appear anywhere in the partner's tree, and in 1 item a missing input's name appears in the history. Every missing input is on screen, as an input of the goal's own recipe line. What distinguishes the cell is recipe structure: in 161 of its 252 items every missing input is itself craftable (an intermediate whose recipe is displayed), against 61 of 248 in the ingredient × never-obtained cell, whose missing inputs are mostly base items fetched with `get`. That structural difference does not account for the role difference for Llama. Within items whose missing inputs are all craftable, the goal role gives −0.82 [−1.03, −0.62] and the ingredient role −0.27 [−0.58, +0.08]. Within items whose missing inputs are all base items, the goal role gives −0.38 [−0.80, +0.04] and the ingredient role −0.09 [−0.31, +0.15]. Llama's effect in this cell is settled at the first token (first-token ΔL −0.75, prefix ΔL −0.74) and appears on the yes/no question too (−0.46 [−0.57, −0.34]). For Qwen the two scales disagree here: two-candidate −0.45, yes/no +1.73 [+1.20, +2.26].

### Matched goal against ingredient (a CPU stand-in for the reviewer's test C)

Goal and ingredient items were matched on failure class (too few, never obtained), length bin, whether the missing inputs are craftable, base or mixed, and the number of short inputs (one, or two or more). In the 19 cells holding at least five items of each role, the goal role's ΔL minus the ingredient role's, averaged with the smaller count as weight, is −0.80 for Llama (95% interval by pair resampling −1.05 to −0.53; 14 of 19 cells negative) and −0.82 for Qwen (−1.69 to +0.09; 13 of 19 negative). The goal-role penalty relative to ingredients is about the same size in both models. What differs is the level: Qwen's ingredient effect is near +1 and Llama's near 0, so Qwen's goal role stays positive and Llama's goes negative.

### Against the explanations

**Competence floor (explanation 4): contradicted.** The overall competence figure (neutral AUC 0.68) hides a large difference between roles. On goal-role crafts Llama discriminates well: the neutral-line AUC between goal-role successes and goal-role failures is 0.84 (successes' mean L −1.67, failures' −5.53). On ingredients it is 0.58. Under the neutral line Llama correctly predicts failure for 99% of goal × never-obtained items and 96% of goal × too-few items. The reversal sits exactly where Llama is competent, and is null where it is not (ingredient × used up, competence AUC far below one half: ΔL −0.02). Splitting by whether Llama's neutral prediction was right gives the same: goal × never obtained −0.76 [−0.94, −0.58] on the 249 items it gets right. A floor account predicts the opposite pattern. What competence does explain is a different fact: on the ingredient role, where Llama does not discriminate, there is also no effect.

**"Swapped is not neutral" (explanation 3): partly right as a description, wrong as a mechanism.** The statement "swapped raises L more than aligned" is entirely a goal-role fact. On ingredients the two lines raise L by the same amount over the neutral line (never obtained +1.18 and +1.30; too few +1.32 and +1.05; used up −0.83 and −0.81). On goal-role failures the aligned line stays at the neutral level (+0.17 on never obtained, +0.16 on too few) and the swapped line raises L (+0.92, +0.68). The specific mechanism proposed, that the swapped goal's displayed tree contains the action's item, cannot apply: a goal-role action's object is never in the partner's tree, and its missing inputs are in the partner's tree in only 17 of 510 goal-role items. History fit does not explain it either: on tie couples, whose history favours neither goal, goal × never obtained gives −0.79 [−1.18, −0.39] and goal × too few −1.11 [−1.56, −0.67]. "Relative to neutral" also depends on the readout: on the yes/no question both goal lines *lower* Llama's "Yes" log-odds against neutral on the goal role (aligned −0.76, swapped −0.46), so the pre-registered label "more careful" is relative to the swapped line only.

**Few-shot pattern copying (explanation 1), from the scores: not supported.** Example 2 is an ingredient craft (sticks) failing for a never-obtained craftable input (acacia planks) after 2 steps. Items that resemble it most, ingredient crafts that fail for never-obtained craftable inputs at 1–3 steps, show no reversal: −0.12 [−0.53, +0.30] (23 items). Goal-role items of the same description show −1.25 [−1.71, −0.79] (46 items). Items whose missing input is a kind of planks give −0.37 [−0.61, −0.12], mostly goal-role items (93 of 136). None of the examples crafts a goal, so the role that carries the reversal is the one the examples never show.

**Lexical or reply-format effect (explanation 6): not supported as the main cause.** The goal-role effect is decided at the first token (first-token ΔL −0.55 of −0.63; prefix-only ΔL −0.59, name remainder small), where the two candidates are the same "Craft" and "Could" in every condition. The yes/no question, whose candidates share nothing with the reply texts, shows the same sign on the goal role (−0.30 [−0.41, −0.18]). Word overlap between goal and action short of identity does nothing (ingredient with overlap +0.05 [−0.11, +0.20]). What the scores cannot exclude is a prompt-side effect of the goal's name appearing verbatim in the action; that is inseparable from the goal role in TextCraft.

**Heterogeneity or artefact (explanation 7): not supported.** The goal-role effect is negative in 73% of the 160 pairs that have goal-role items (median pair −0.67) and in 68% of goal-role items. Leaving out any one pair moves the goal-role mean only between −0.65 and −0.61. It is present at every pair depth (depth 1 −0.99, depth 2 −0.38, depth 3 −0.72), at every prompt length (shortest third −0.81, middle −0.37, longest −0.49), and on items with high and low first-token mass.

**Clustering by recipe family.** Both family definitions were fixed before any effect was computed on them. Linking pairs whose lists share any craftable gives 22 components, the largest holding 208 of the 229 pairs, so by the rule fixed in advance it is reported but gives no usable interval. Grouping pairs by their primary material gives 50 families (the largest, acacia logs, holds 80 pairs). Resampling these families, Llama's ΔL is −0.19 [−0.36, −0.11] and its goal-role part −0.63 [−0.80, −0.49]; Qwen's ΔL is +0.94 [+0.73, +1.31]. The intervals still exclude zero.

### Timing and progress (explanation 2)

On goal-role failures, Llama's reversal is largest when the history is short and holds none of the goal's inputs, and it fades as inputs accumulate. By the share of the goal recipe's inputs held (the `completion` fit measure), ΔL is −0.85 [−1.04, −0.66] at none held, −0.75 [−1.00, −0.50] below a half, and −0.27 [−0.55, +0.02] at a half or more. Among items holding some inputs, by length bin, it is −1.17 [−1.50, −0.82] at 1–3 steps, −0.49 [−0.80, −0.19] at 4–8 and +0.02 [−0.25, +0.27] at 9–12. Qwen shows the same gradient, shifted up: −0.16, +0.27 and +2.93 by completion, and −0.61 at 1–3 steps with none held. On goal-role *successes* Llama's aligned line lowers L against neutral at 1–3 steps (−1.07 [−1.43, −0.70]) and raises it at 9–12 (+0.81 [+0.50, +1.13]); the swapped line does nearly the same (−0.73, +0.67), so ΔL on goal-role successes stays near zero (−0.06 [−0.37, +0.26]).

This fits a reading in which a goal line makes both models judge a craft of the goal by how far the task seems to have progressed, rather than only by the inventory. Crafting the goal after a few steps with nothing toward it looks premature and is judged less likely to succeed than the same craft under the other goal line; with inputs gathered it looks plausible. For Qwen this progress term rides on a large positive goal effect, for Llama on none. It is a description that fits the gradients, not a demonstrated mechanism: completion and length are correlated with the history's content, and the same gradient would follow from other accounts that make the goal-role penalty shrink as the history fills.

### Practical size

On a probability scale Llama's effect is negligible. The mean predicted chance of success differs by +0.38 points [−0.57, +1.28] between aligned and swapped over all main failures, by −1.90 points [−3.12, −0.83] on the goal role and by +1.55 points [+0.37, +2.71] on ingredients. On goal-role failures Llama predicts success (L above zero) for 3.3% of items under the aligned line and 4.9% under the swapped line. Between the two lines 2.7% of items flip from failure to success and 1.2% the other way. On the yes/no question its "Yes" share on goal-role failures is 0.0% under both lines (mean probability of "Yes" 0.9% and 1.1%). The reversal is a change of confidence inside a failure judgement Llama already makes, not a change of decision. For Qwen, by contrast, the share of goal-role successes answered "Yes" falls from 65% under the aligned line to 14% under the swapped line, which is a change of decision.

## Findings from the prompt variants and the new runs (GPU)

All runs below finished with every record present and no stale record (each record's item hash recomputed for its own prompt). The new runs ran at commit `b589e5a` (zero-shot, ICE job 7206794; object goal, job 7206854, after a fast-forward to `610b458` before it started) and `f74c262` (generation, job 7207017; prompt variants, job 7207018), from the new checkouts `text-wm-fix26` and `text-wm-fix27`. Each is read against the verified prompt's scores on the same items.

### Examples on goal (design D10)

The third prompt example ("get 2 jungle logs" succeeding) was put back under the goal whose tree holds jungle logs, so that no example acts off its goal. On 200 seeded main couples (400 items, 130 pairs), Llama's ΔL is −0.24 [−0.39, −0.10] with these examples and −0.19 [−0.34, −0.05] with the verified ones. The per-item difference, variant minus verified, is −0.05 [−0.08, −0.02], and on the goal role −0.09 [−0.15, −0.03]. For Qwen it is +0.72 against +0.75 (difference −0.03 [−0.19, +0.13]). The off-goal example does not produce Llama's reversal. Removing it makes the reversal very slightly larger, the opposite of what design D9's bias would predict if it were acting on Llama.

### Empty start (design D10)

One sentence stating that the inventory starts empty was added to the system text. On 300 seeded main couples (600 items, 168 pairs), Llama's ΔL is −0.11 [−0.23, +0.01] with the sentence and −0.18 [−0.30, −0.06] without. The difference is +0.07 [+0.04, +0.11], and on the goal role −0.49 [−0.72, −0.26] against −0.63 [−0.86, −0.41] (difference +0.14 [+0.07, +0.21]). For Qwen it is +1.00 against +1.04 (difference −0.04 [−0.13, +0.05]). The hidden-state account predicts that stating the empty start removes a positive, evidence-driven part of ΔL, so ΔL should fall. It does not fall for either model: Qwen's effect does not depend on doubt about the starting inventory, and Llama's reversal shrinks by about a fifth on the goal role but stays clearly negative there. Over all items, Llama's interval with the sentence reaches zero (upper end +0.01), so the overall negative is not robust to this prompt change, while the goal-role negative is.

### Zero-shot (run 1)

The verified prompt without its three examples was scored on 150 main and 100 success couples. Without examples neither model works as a world model: the neutral-line AUC between successes and failures falls from 0.64 to 0.43 for Llama and from 0.87 to 0.43 for Qwen on these items, and Llama's probability mass on the two candidates' first tokens falls from 0.86 to 0.25 (Qwen's from 1.00 to 0.64). A ΔL measured zero-shot is the goal line's pull on a model that does not discriminate, so it says little about a world model. With that limit:

| Population and role | Llama zero-shot | Llama verified | Qwen zero-shot | Qwen verified |
|---|---|---|---|---|
| Failures, all (300 items, 102 pairs) | −0.42 [−0.68, −0.16] | −0.28 [−0.46, −0.09] | +2.11 [+1.64, +2.60] | +0.62 [+0.09, +1.21] |
| Failures, goal role (110) | −1.01 [−1.55, −0.41] | −0.90 [−1.22, −0.58] | +3.46 [+2.49, +4.46] | +1.03 [−0.03, +2.15] |
| Failures, ingredient role (190) | −0.08 [−0.26, +0.10] | +0.07 [−0.08, +0.24] | +1.33 [+0.91, +1.75] | +0.38 [−0.22, +1.01] |
| Successes, goal role (102) | +0.65 [+0.17, +1.15] | −0.01 [−0.47, +0.49] | +3.65 [+2.70, +4.58] | +3.22 [+1.63, +4.87] |

Llama's goal-role reversal does not need the examples. Qwen's goal effect is larger without them, and its negative goal × never-obtained cell turns strongly positive (+4.18 [+2.79, +5.51]); for Qwen the examples, and the discrimination they bring, hold the goal effect down.

### Free generation and free yes/no answers (runs 2 and 5)

On a held-out seeded sample (60 goal-role and 30 ingredient main couples, 30 goal-role and 30 ingredient success couples; 300 items), Llama generated the next observation and, separately, answered the yes/no question in free text, under each line.

| Items | Line | Observation predicts success | Free answer "Yes" |
|---|---|---|---|
| Goal-role failures (120) | aligned / swapped / neutral | 2 / 5 / 4 | 0 / 0 / 1 |
| Goal-role successes (60) | aligned / swapped / neutral | 11 / 10 / 14 | 2 / 0 / 3 |
| Ingredient failures (60) | aligned / swapped / neutral | 34 / 33 / 33 | 3 / 5 / 8 |
| Ingredient successes (60) | aligned / swapped / neutral | 41 / 39 / 36 | 7 / 7 / 9 |

On goal-role failures the generated observation is the exact failure reply for 95% (aligned), 92% (swapped) and 95% (neutral) of items, and the paired difference in the share predicting success, aligned minus swapped, is −0.03 [−0.06, 0.00]. On goal-role successes Llama also predicts failure, for about 80% of items under every line. The generations confirm that the two-candidate scores reflect what Llama actually predicts, and that it predicts failure for nearly every craft of a goal, whichever line is shown. On ingredients its generations say success about as often for failures as for successes (34 of 60 against 41 of 60 under the aligned line), as the low ingredient AUC implied. Its free yes/no answers are "No" for nearly everything, successes included, which is why the yes/no readout sits near its floor. Some ingredient generations are a third text ("You already have 4 acacia planks."), and the tokens that take the 10% of first-token mass neither candidate gets are mostly "You", "Got", "Cannot", "Not", "Already" and "Inventory"; that mass is 0.10 under the aligned and the swapped lines and 0.12 under the neutral line, so it does not differ between the two lines that define ΔL.

### Object goal on ingredient actions (run 3)

An ingredient action crafting Y was scored under "Goal: craft Y." and compared with the verified aligned ("craft G"), swapped ("craft P") and neutral lines on 150 main and 100 success ingredient couples.

| Items | Llama: object goal − aligned | Llama: object goal − neutral | Qwen: object goal − aligned | Qwen: object goal − neutral |
|---|---|---|---|---|
| Failures (300 items, 98 pairs) | −1.40 [−1.62, −1.20] | −1.03 [−1.27, −0.80] | −1.23 [−1.94, −0.45] | +0.28 [−0.31, +0.85] |
| Failures, never obtained (86) | −1.84 [−2.34, −1.39] | | −4.00 [−5.79, −2.38] | |
| Failures, too few (78) | −1.57 [−2.02, −1.05] | | +0.01 [−1.55, +1.67] | |
| Failures, used up (136) | −1.03 [−1.23, −0.82] | | −0.19 [−0.89, +0.55] | |
| Successes (200 items, 81 pairs) | −1.74 [−2.04, −1.46] | −0.82 [−1.14, −0.50] | +0.80 [−0.11, +1.71] | +1.93 [+1.31, +2.56] |
| Yes/no, failures | −0.79 [−0.92, −0.66] | −1.25 [−1.39, −1.11] | +2.18 [+1.19, +3.18] | −0.08 [−0.89, +0.74] |
| Yes/no, successes | −0.92 [−1.08, −0.77] | −1.03 [−1.19, −0.87] | +5.37 [+4.46, +6.28] | +0.79 [−0.03, +1.71] |

For Llama, making the action's own object the goal lowers its predicted success on every class, on successes as much as on failures, on both scales, and below the neutral line. By length bin the drop on failures is −2.81 [−3.37, −2.25] at 1–3 steps, −1.11 at 4–8 and −1.00 at 9–12. Discrimination falls slightly (AUC 0.71 to 0.65), and the share of ingredient successes Llama predicts as successes falls from 79% to 51%. This is the goal-role pattern again, larger, on items that are not top items. It follows whichever item is named as the goal. It is a level shift against crafting the named goal, not a sharper judgement of failures.

For Qwen the same line acts differently: it lowers predicted success sharply where the inputs were never obtained (−4.00; yes/no −4.34) and otherwise leaves it or raises it (successes +0.80, yes/no +5.37), and its AUC rises slightly (0.87 to 0.89). Qwen reads "craft Y" as a prompt to check Y's inputs, and the check finds names that never appeared. That matches Qwen's negative goal × never-obtained cell, and its large positive effect where inputs were gathered.

The run cannot say whether naming the object or making it the terminal craft drives Llama's drop, because the line does both (plan 6).

### Clean world-model prompt and explicit inventory (run 4; the reviewers' test B)

On 150 main and 100 success couples (500 items, 108 pairs on the failures), both models were scored under three variants of the prompt and the three goal lines. Competence was read first, as planned. The table gives the change in the neutral-line AUC against the verified prompt on the same items, with intervals from resampling pairs.

| Variant | Llama AUC change, all | Llama, goal role | Qwen AUC change, all | Qwen, goal role |
|---|---|---|---|---|
| Clean system text | −0.016 [−0.031, −0.003] | −0.035 [−0.062, −0.005] | +0.002 [−0.009, +0.013] | −0.014 [−0.029, +0.000] |
| Inventory step | +0.088 [+0.063, +0.113] | +0.066 [+0.032, +0.100] | +0.099 [+0.073, +0.130] | +0.056 [+0.020, +0.093] |
| Clean + inventory | +0.074 [+0.045, +0.105] | +0.045 [+0.004, +0.081] | +0.096 [+0.069, +0.126] | +0.051 [+0.014, +0.089] |

On these items the verified prompt's AUC is 0.54 for Llama and 0.84 for Qwen. Showing the inventory raises them to 0.62 and 0.94. The clean system text leaves Qwen's competence unchanged and lowers Llama's slightly. Showing the inventory helps Llama on every class but does not fix its worst one: used-up failures are still ranked above successes (AUC 0.35, from 0.24).

ΔL on the failures, under each variant:

| Failures | Verified | Clean | Inventory | Clean + inventory |
|---|---|---|---|---|
| Llama, all (300) | −0.30 [−0.51, −0.11] | −0.17 [−0.33, −0.01] | −0.26 [−0.44, −0.10] | −0.09 [−0.24, +0.06] |
| Llama, goal role (100) | −0.98 [−1.34, −0.64] | −0.59 [−0.87, −0.31] | −0.87 [−1.22, −0.54] | −0.51 [−0.81, −0.24] |
| Llama, ingredient (200) | +0.04 [−0.14, +0.23] | +0.04 [−0.13, +0.20] | +0.05 [−0.11, +0.20] | +0.13 [−0.03, +0.28] |
| Qwen, all (300) | +1.15 [+0.60, +1.69] | +1.52 [+0.98, +2.05] | +1.12 [+0.54, +1.68] | +1.42 [+0.87, +1.97] |
| Qwen, goal role (100) | +1.37 [+0.16, +2.58] | +3.03 [+1.76, +4.33] | +1.86 [+0.58, +3.22] | +3.10 [+1.89, +4.39] |
| Qwen, ingredient (200) | +1.05 [+0.40, +1.73] | +0.77 [+0.19, +1.34] | +0.74 [+0.19, +1.34] | +0.58 [+0.06, +1.12] |

The per-item differences against the verified prompt, on the goal role, are: Llama clean +0.39 [+0.24, +0.55], inventory +0.11 [−0.09, +0.29], clean + inventory +0.47 [+0.26, +0.68]; Qwen clean +1.66 [+1.30, +2.01], inventory +0.49 [−0.16, +1.17], clean + inventory +1.73 [+1.06, +2.46]. On the ingredient role no variant moves Llama (all within ±0.1), and the clean text lowers Qwen a little (−0.28 [−0.59, +0.02]).

Reading by the plan:

- *Inventory shown (test B).* Both models' competence rises clearly, and neither model's goal effect changes much. Llama's goal-role effect stays negative (−0.87), and no positive effect appears on its ingredients (+0.05). Qwen's overall effect is unchanged (+1.12 against +1.15) although its AUC reaches 0.94. This is the plan's second outcome: the reversal does not depend on Llama reconstructing the state from the history. It is about the goal role. The same holds for Qwen's positive effect, which also survives a fully readable state. On the decision scale the shown inventory makes no difference to the goal-role failures either: Llama predicts success for 3% of them under the aligned line and 4% under the swapped line.
- *Clean system text.* The agent instructions are not neutral. Removing them moves the goal-role effect up in both models, by +0.39 for Llama and +1.66 for Qwen, and leaves the ingredient role about where it was. Llama's goal-role effect shrinks by about 40% but stays negative. Its overall effect on these failures becomes −0.17 [−0.33, −0.01], and with the inventory also shown −0.09 [−0.24, +0.06].

### Which prompt later runs should use (a recommendation, for the owner to decide)

The clean system text is the better default for later TextCraft runs, including other model families, with three qualifications.

The arguments for it: it is a world-model prompt, not an agent prompt. It matches the ALFWorld world-model system text, which is a framing sentence with no agent instructions (`wf/scoring/prompts.py`, `WM_SYSTEM`). It keeps Qwen's competence unchanged and costs Llama little (−0.016 AUC). It removes the one phrase ("from the inventory or the environment") that hinted at a hidden store. And it changes the goal-role effect by a lot in both models, which shows that the agent instructions were shaping the measured effect rather than leaving it alone.

The qualifications:

1. The confirmatory TextCraft result stands for the prompt it was frozen with. A run under the clean prompt measures a related but different quantity, and should be reported as such, not as a re-estimate of the confirmatory number.
2. These diagnostic runs did not run the tokenization and golden-prompt checks that the design requires before scoring (validation standard, step 7). The clean text and the inventory step should pass `wf/textcraft/tokcheck.py` for every model before any run that will be reported.
3. The explicit inventory step is not a good default. It raises competence a lot, but it changes the question from "does the goal line shift a world model that must track state from the history" to "does it shift one that is shown the state". It is better kept as a separate arm, where it is the cleanest test of whether an effect depends on state reconstruction.

### Criteria for a robust negative

The reviewers listed five criteria for calling Llama's negative effect robust. Against each:

1. *Persists under more than one reasonable prompt.* For the goal role, yes: it is negative under the verified prompt, the examples-on-goal prompt, the empty-start prompt, the clean prompt, the inventory prompt and the clean + inventory prompt (from −0.49 to −0.98 on their samples), and zero-shot. For the overall ΔL, no: its interval reaches zero under the empty-start prompt (−0.11 [−0.23, +0.01]) and under the clean + inventory prompt (−0.09 [−0.24, +0.06]), and barely excludes it under the clean prompt (−0.17 [−0.33, −0.01]).
2. *Not driven by a misunderstanding of the initial inventory.* Met. Stating the empty start shrinks the goal-role effect by about a fifth (+0.14), and showing the inventory does not change it significantly (+0.11 [−0.09, +0.29]).
3. *Holds in a competent subset.* Met. The goal role is where Llama discriminates best (AUC 0.84 under the neutral line, 0.88 with the inventory shown), and the effect is −0.76 on the 249 goal × never-obtained items whose neutral prediction is correct.
4. *Meaningfully negative on an outcome scale.* Not met. The mean predicted chance of success moves by +0.38 points overall and −1.9 points on the goal role. The share of goal-role failures predicted as success is 3.3% against 4.9%, generated observations predict success for 2 against 5 of 120 goal-role failures, and the free yes/no answer is "No" for all of them under both lines.
5. *Reproduced by an independent recompute.* Met (−0.19 [−0.29, −0.09]).

So the overall negative primary is real under the frozen prompt but is neither robust to reasonable prompt changes nor of practical size. The goal-role negative is robust on the log-odds scale and on yes/no, across prompts, pairs and recipe families, but it is a change of confidence inside a failure judgement Llama already makes.

## Assessment

### What most likely explains the Llama result

Llama's negative ΔL is the sum of two things. Neither is specific to Llama in kind, but their balance is.

1. **A terminal-craft penalty.** When the action crafts the item the goal line names, Llama judges it less likely to succeed than the same craft under another goal line. The penalty is largest early in the game, with none of the goal's inputs held. It follows whichever item is named as the goal: an ingredient made the goal drops by −1.40 on failures and −1.74 on successes. It is present without examples, with the empty start stated, with the inventory shown and with the clean prompt (smaller there). Qwen shows a penalty of the same size relative to its own ingredient effect (matched goal minus ingredient −0.82 against Llama's −0.80) and a negative goal × never-obtained cell. So the shared cell the coordinator pointed to is the place where this penalty is not masked.
2. **No positive goal effect on intermediate actions.** On ingredient crafts Llama's ΔL is +0.03 [−0.07, +0.13], under every prompt variant tried. Qwen's is about +0.9. This is what differs most from ALFWorld. There, Llama's effect was almost entirely "mention", the task line naming the action's object (2×2: mention +1.04, goal-directed −0.18). In TextCraft no ingredient action's object is named by the goal line, and Llama's TextCraft usefulness-only readout (−0.16 [−0.43, +0.08]) is close to ALFWorld's goal-directed part. The only TextCraft actions whose object the goal names are the goal crafts, where the terminal penalty pushes the other way. For Qwen the separate cross-environment comparison (`notes/textcraft_vs_alfworld.md`, branch `xenv-compare`) finds that most of the ALFWorld two-candidate effect sat in reply text TextCraft does not have, so the "smaller in TextCraft" pattern has a format component for Qwen as well. That note was not repeated here.

With a positive ingredient effect near +1 and the same goal-role penalty, Qwen's total stays positive. With an ingredient effect near zero, Llama's goes negative.

Confidence. The decomposition is well supported: the goal role carries all of the effect, the ingredient role is null, and the penalty reappears on ingredients made the goal. These hold across pairs, families, prompts and readouts. The mechanism behind the penalty is not established. "Completing a named goal looks premature, especially early" fits the length and progress gradients and the object-goal run. A negative effect of naming the object verbatim would fit them too, because in TextCraft's one-item goal format naming an item and making it the terminal craft always go together. The account of the missing ingredient effect via mention is an inference from ALFWorld's 2×2, not a TextCraft test.

### The explanations, in short

| Explanation | Verdict | Main evidence |
|---|---|---|
| 1. Few-shot pattern copying | Not supported | The off-goal example moves ΔL by −0.05 (the wrong way). The items most like example 2 show no reversal. The reversal persists zero-shot. No example crafts a goal. |
| 2. Goal-completion caution | Supported as a description; mechanism open | All of the effect is in the goal role, it is largest early with nothing held, it reappears when an ingredient is made the goal, and Qwen shares it. Naming and terminal status cannot be separated in TextCraft. |
| 3. Swapped is not neutral, second task on screen | The description is right for the goal role only; the proposed mechanism is not | On ingredients both lines raise L equally over neutral. The partner's tree never holds a goal craft's object, and holds its missing inputs in 17 of 510 goal-role items. Ties show the same effect. |
| 4. Competence floor | Not the cause of the reversal | The reversal sits where Llama is competent (goal role, AUC 0.84) and is unchanged when the inventory raises competence. It does fit the absence of any effect on ingredients. |
| 5. Hidden state | Not supported in TextCraft; the ALFWorld side is open | Empty start: Qwen −0.04, Llama +0.07. A shown inventory changes neither model's ΔL. In ALFWorld, Llama's effect was largest where the receptacle had been seen without the object (2.8), not where the location was hidden (1.5). |
| 6. Lexical or reply format | Not supported as the main cause | The effect is decided at the first token ("Craft" against "Could", the same in every condition), appears on yes/no, and shows up in free generation. The leftover first-token mass is equal under aligned and swapped. |
| 7. Heterogeneity or artefact | Not supported | 73% of goal pairs are negative. Leaving out any one pair moves the estimate by at most 0.02. Clustering by recipe family keeps the interval below zero. |
| 8. Agent instructions in the system text | Found: they shape the effect | The clean text raises the goal-role effect by +0.39 (Llama) and +1.66 (Qwen). |

### What remains undecided, and what would decide it

- **Naming against completion.** No TextCraft goal line can name an item without making it the goal. Two runs would come closest. (a) The parent-goal arm of plan 6 makes a goal craft intermediate by naming an item that consumes it; it needs a small builder with its own replay and checker checks, and about 15 GPU-minutes. (b) A cheaper check needs no GPU: the ALFWorld scores, where the final "put" step completes the task and earlier steps do not, can show whether Llama's ALFWorld effect also dips on the completing step. The loaders on `xenv-compare` would make this short.
- **Timing against content.** The length and progress gradients are correlational. Padding histories with steps that change nothing (repeated inventory look-ups before the first `get`) would vary length at fixed content. It costs about 2,000 scores.
- **Why Llama has no ingredient effect.** The mention account predicts that a goal line naming the ingredient without making it the goal would raise Llama's L. TextCraft's format cannot express that without leaving the environment's template, so this is best tested in ALFWorld's existing 2×2 design rather than here.
- **The prompt for later runs** is a decision for the owner (recommendation above). The tokenization checks are needed first.

## Runs and code

| Run | ICE job | Commit | Checkout | GPU time | Records |
|---|---|---|---|---|---|
| `textcraft_dx_zeroshot` | 7206794 | `b589e5a` | `text-wm-fix26` | 17 min | 3,000 |
| `textcraft_dx_objectgoal` | 7206854 | `610b458` | `text-wm-fix26` | 22 min | 2,000 |
| `textcraft_dx_generate` | 7207017 | `f74c262` | `text-wm-fix27` | 12 min | 900 |
| `textcraft_dx_prompts` | 7207018 | `f74c262` | `text-wm-fix27` | 47 min | 9,000 |

About 98 GPU-minutes in all. An earlier generation job (7206795) was cancelled 6 seconds after it started, when its scope was widened by plan 5; a queued resume job (7207529) was cancelled unused. The confirmatory checkout `text-wm-fix24` and its runs were only read. Code: `wf/textcraft/dx.py` (the four runs), `wf/config/textcraft_dx_*.yaml`, `pace/textcraft_dx.sbatch`, `scripts/textcraft_llama_dx.py` (pair-clustered estimates), with tests in `tests/test_textcraft_dx.py` (17, mock mode). The per-cut analysis scripts were one-off and are not in the repository; every number above can be recomputed from the scoring records with the helpers in `scripts/textcraft_llama_dx.py`.
