# Does ALFWorld place a task's object differently once its existence is known? A CPU check of assumption A1′ (2026-11-25)

**Status.** Post hoc, CPU only, no model scored. Script `scripts/a1prime_check.py`; every number below is in [a1prime_metrics.json](a1prime_metrics.json). It is the first step of the natural bridge ([bridge_design.md](bridge_design.md)), whose existence arm it bears on.

## 1. The question

When the task line names an object ("put some saltshaker on cabinet"), it tells a reader at least two things about the room: that a saltshaker exists, and possibly where it is or how many there are, if the game generator places a task's object differently from other objects. An earlier exploratory test (`exploratory_results.md` §3) found that once a sentence stating the object's existence is added to both the aligned and the swapped prompt, the goal effect on the first answer word all but disappears (Qwen3-8B −0.16 nats, Llama-3.1-8B 0.07, against 2.42 and 0.66 without the sentence). Whether that collapse is what a correct world model would show depends on an assumption, called A1′ here: **once the object's existence is known, where it is (which receptacle, whether inside a closed one, how many instances) does not depend on which task was drawn.** If A1′ holds, existence is the only thing the task line legitimately tells the model about the room, and settling existence should remove every legitimate goal effect. If A1′ fails, a model that knew how the games were generated would keep a residual effect after existence is settled, equal to what the task says about location.

Formally, write $E$ for the object's existence and $W$ for the rest of its placement. The legitimate change in the log-probability of a state under task $g$ against task $g'$ splits as

$$\log\frac{P(E, W \mid h, g)}{P(E, W \mid h, g')} = \log\frac{P(E \mid h, g)}{P(E \mid h, g')} + \log\frac{P(W \mid E, h, g)}{P(W \mid E, h, g')}.$$

Stating existence removes only the first term. A1′ says the second term is zero.

## 2. Data

ALFWorld's own game files (the copy on ICE under `~/.cache/alfworld`, byte-identical to the release, `repro/alfworld_data_sha256.txt`): 5,234 game files, of which 4,652 are training games and 195 `valid_unseen` games (134 of them solvable). Each file holds the PDDL initial state. From it the script reads every object's type, which receptacle holds it, and whether that receptacle can be opened. No initial state marks any receptacle as open, so an openable receptacle is closed at the start. From the game's `traj_data.json` it reads the task type, the task's object, its target receptacle type, and the floor plan. AI2-THOR numbers floor plans by room: 1–30 kitchens, 201–230 living rooms, 301–330 bedrooms, 401–430 bathrooms.

All games are used, solvable or not, because the question is about how the generator places objects, not whether a planner can solve the task (3,551 of the training games carry the solvable flag).

## 3. Generator level: the task's object against the same type in other games of the same floor plan

**Comparison.** For each floor plan and object type, compare the games in which that type is the task's object with the games of the same floor plan in which the type is present but the task names another object. Holding the floor plan fixed means both groups have the same receptacles. Each statistic below is the mean, over the task-object occurrences, of the task game's value minus the mean of the same floor plan's other games for that type. Intervals are 95% bootstraps over floor plans (2,000 resamples). The p-value comes from shuffling, within each floor plan and type, which games carry the "task object" label (2,000 shuffles; 500 for the distribution distance). The shuffle treats games as exchangeable within a floor plan and type and is less conservative than the bootstrap, which resamples whole floor plans; where they disagree, the bootstrap is the one to rely on.

**Training games** (959 floor-plan-and-type strata from 108 floor plans; 4,652 task-object rows against 41,915 rows of the same types in other games):

| Placement property | Task's object | Same type, task names another object | Within floor plan, task minus other [95% CI] |
|---|---|---|---|
| Share of instances inside a closed receptacle | 12.3% | 22.6% | −9.1 points [−10.5, −7.8] |
| Number of instances in the room | 2.25 | 1.88 | +0.35 [+0.31, +0.39] |
| Share in the appliance the task uses (microwave for heat, fridge for cool, sink basin for clean; 342 strata) | 0.0% | 7.1% | −8.3 points [−9.5, −7.3] |
| Share already in the task's target receptacle type | 9.8% | 11.6% | −1.8 points [−4.0, +0.5] |

The distribution over receptacle types differs too. The total variation distance between the task object's receptacle types and the same type's in other games, averaged over strata, is 0.262, against 0.216 when the labels are shuffled (95th percentile of the shuffles 0.221, p = 0.002). Pooled over strata, the task's object is less often in a fridge (4.1% against 8.2%), a drawer (3.6% against 6.6%) or a cabinet (6.3% against 8.1%), and slightly more often on open surfaces such as a bathtub basin, desk or dresser.

**`valid_unseen` games**, where far fewer floor plans hold both kinds of game (41 strata from 4 floor plans), point the same way: closed receptacle −17.4 points [−26.5, −3.4], instances +0.23 [+0.04, +0.50], task appliance −11.5 points [−13.9, −11.2].

**Reading.** A1′ fails as an exact statement. ALFWorld's generator gives the task's object more instances, puts it less often inside closed receptacles, never starts it in the appliance its task uses, and spreads it over receptacle types differently from other objects of its type, all with existence held fixed. These are measured properties of the generated games; the check does not depend on reading the generator's code.

## 4. What this means for the actions RQ2 scores

A failed assumption says nothing yet about size. The decision-relevant event for most RQ2 items is whether the object is at the receptacle the action names. So the script computes, for every RQ2 primary item that takes the task's object from a receptacle (2,449 of the 4,666), how much the task changes the log-probability of that event once existence is known.

For room type $c$, object type $o$ and receptacle type $r$, let $q$ be the mean, over training games of that room type, of the fraction of type-$r$ receptacles that hold an $o$. It is computed three ways: $q_T$ over games whose task object is $o$ (task-type specific where at least 30 such games exist, pooled otherwise); $q_E$ over games where $o$ exists but the task names another object; and $q_N$ over all games whose task names another object, counting an absent $o$ as zero. The room type of each item comes from its receptacles (a fridge only in kitchens, a toilet only in bathrooms, a bed only in bedrooms), a rule that agrees with the floor-plan number in all 5,234 games. With 0.005 added to each $q$, the item's three indices are

$$\text{location given existence} = \log\frac{q_T}{q_E}, \qquad \text{existence} = \log\frac{q_E}{q_N}, \qquad \text{total} = \log\frac{q_T}{q_N},$$

so the total is the sum of the other two, as in the decomposition of section 1. A fourth index covers the instance number the action names (for "saltshaker 2", whether a second instance exists): the log of the share of task games with at least that many instances over the share of games where the type exists but is not the task's.

Under rule uncertainty these indices are also the legitimate log-odds effects. Suppose a model keeps a small probability $\varepsilon$ that some rule it cannot see lets the action succeed whenever the object is at the receptacle. Then its success probability is about $\varepsilon q$, and the legitimate change in log-odds between two tasks is

$$\Delta_{\text{legit}} \approx \log\frac{q_T}{q_E},$$

independent of $\varepsilon$, once existence is settled. The probability-scale change, about $\varepsilon\,(q_T - q_E)$, is negligible.

| Items | Location given existence: median; mean [95% CI over games]; 5th to 95th percentile | Existence: median; mean | Instance number: median; mean |
|---|---|---|---|
| All 2,449 take items | +0.06; −0.29 [−0.44, −0.16]; −3.77 to +0.56 | 0.00; +0.08 | 0.00; +0.15 |
| 905 whose receptacle type the history never names | +0.20; +0.07 [−0.01, +0.14]; −0.58 to +0.67 | 0.00; +0.09 | 0.00; +0.005 |
| 242 at step 0 | +0.12; −0.02 [−0.13, +0.09]; −0.75 to +0.66 | 0.00; +0.10 | 0.00; +0.004 |

The location index is negative on 42% of take items and above 0.5 nats on 10%. Its long negative tail comes from actions that take the object from the very appliance its task uses, where the generator never puts it: there the task line is legitimate evidence *against* success. The existence index has a median of zero because the generator places almost every compatible object type in kitchens and bathrooms, so naming the object adds little; its 95th percentile is about 0.78 nats.

The intervals are over RQ2's games and treat the training-game frequencies as known; the uncertainty of those frequencies is not propagated. The indices are type-level (object type and receptacle type, not the named instance at the named receptacle), and they describe ALFWorld's generator, which no model has seen; a model's own beliefs about where task objects are could differ.

## 5. What this implies for reading the existence collapse as legitimate screening

1. **A1′ does not hold, so the correct reference after existence is settled is not exactly zero.** A model that knew the generator would keep a goal effect that depends on the item: positive where the task's object tends to be on open surfaces at the named receptacle type, strongly negative where the action takes it from its task's own appliance.
2. **That legitimate residual is small.** On typical take items it is a few tenths of a nat or less in log-probability, and so in log-odds under rule uncertainty (median 0.06 over all take items, 0.20 where the receptacle type was never named), and its mean over all take items is slightly negative. A residual close to zero on the first token, as the earlier test found, is therefore consistent with legitimate screening; the failure of A1′ does not make that collapse suspicious.
3. **It does not license a large residual.** A mean residual goal effect much above about half a nat after existence is settled cannot be attributed to location or instance information under this generator: fewer than an eighth of take items have a location index above 0.5 nats, and the mean index is not positive. Such a residual needs another explanation: the object's reference, a role effect, or provenance.
4. **For the bridge.** The existence arm's "persists" reading (`bridge_design.md` §10) is read against this: a residual on items where the model accepts existence is excess over location information only where it clearly exceeds these indices; the index is reported beside it.
5. **Limits.** The indices cover take actions only. For moving, heating, cooling and cleaning, the deciding fact is what the agent holds or where it is, and the task's information about the object's placement bears on those failures only indirectly. The check concerns ALFWorld's generator; whether a model's learned prior resembles it is a separate question that natural data cannot answer.

## 6. How to reproduce

On a machine with ALFWorld's data (standard library only):

    python3 scripts/a1prime_check.py extract --data ~/.cache/alfworld --out a1prime_games.jsonl

Then, with numpy and this repository's environment, and the RQ2 collection (`rq12_transitions_s0`–`s2`, in the RQ2 reproducibility archive):

    python scripts/a1prime_check.py analyze --games a1prime_games.jsonl --transitions <folder with the three collection runs> --out notes/a1prime_metrics.json --items-out notes/a1prime_items.jsonl

The second file holds the per-item indices (2,449 take items), which the natural bridge's analysis reads. The extraction of all 5,234 files writes a 59 MB file (sha256 `cb20bdd46944cb69…`); the analysis is deterministic (seed 20260924) and takes about 15 minutes on a laptop, almost all of it the shuffle tests.
