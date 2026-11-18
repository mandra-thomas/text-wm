# Prompt example sets 2 and 3: drafts for review (2026-11-17)

**Decided 2026-11-18.** Set 3 is approved as drafted. Set 2 is redrawn because of the verb cue (check 3 below) and the creditcard move (check 4). The redrawn set 2, frozen in `wf/scoring/prompts.py` and `notes/fewshot_sets_frozen.json`, is:

- goal-relevant success: "clean some soapbar and put it in cabinet", at countertop 1 with a soapbar 2 shown; `take soapbar 2 from countertop 1` gives "You pick up the soapbar 2 from the countertop 1." (train/1213)
- goal-irrelevant contents failure: "clean some potato and put it in microwave", the open fridge shows a cup, an egg and a potato; `take apple 1 from fridge 1` gives "Nothing happens." (train/1183)
- goal-relevant location failure: "put some creditcard on armchair", the agent is at drawer 1; `take creditcard 1 from sofa 1` gives "Nothing happens." (train/403)
- goal-irrelevant success: "find two creditcard and put them in coffeetable", the agent is at the closed drawer 1; `open drawer 1` gives "You open the drawer 1. The drawer 1 is open. In it, you see a watch 1." (train/2128)

The drafts below are kept for the record.

Drawn by rule and a fixed seed (20260924) from Pilot A train transitions; every string is logged server output; no pilot score was used. Set 1's games (train/700, train/270, train/2332) are excluded and no game is used twice. Each set has four examples: a goal-relevant success, a goal-irrelevant success, a location failure and a contents failure. In set 2 the location failure is goal-relevant; in set 3 the contents failure is. Every outcome follows from the two-step history shown: failures pass the context-determined check, and for successes the history shows every precondition the action needs. Histories are the task line plus the last two steps, as in set 1. The order within each set was shuffled by the same seed.

Set 1 (filed, kept for the primary analysis) has one goal-irrelevant success and two goal-relevant failures, so goal relevance predicts the outcome in it. See notes/deviations.md.

## What to check (added 2026-11-18)

In each block, the last `>` line is the example's action and the line after it is the outcome the model is shown. For each set:

1. **The outcome follows from what is shown.** Would a careful person, reading only the task line and the two earlier steps, know the action's outcome? Failures should be forced: the object is not listed where the agent stands, or the agent is somewhere else. Successes should have every precondition visible.
2. **The labels are right.** Goal-relevant means the action names the task's object or target receptacle. A location failure means the agent is not at the place the action names. A contents failure means the object is not there.
3. **No accidental cue predicts the outcome.** Within a set, nothing besides the state should separate successes from failures: not the action verb, the length, or goal relevance. Set 2 has one such cue: both failures are `take` and neither success is, so the set could teach "take fails". Set 3 does not have it, because one of its successes is a `take`.
4. **Nothing odd that a model might copy.** Check for malformed commands such as `take desklamp from desk 1` with no number (set 3, example 1, in the history), and for nonsense moves such as moving the creditcard back where it came from (set 2, example 2).

Reply per set: approve, or redraw (say which check failed).

## Set 2

### Example 1: goal-irrelevant failure (contents) (train/745, t_002940)

```
Your task is to: put a spraybottle in garbagecan.
> go to garbagecan 1
You arrive at garbagecan 1. On the garbagecan 1, you see nothing.
> go to countertop 1
You arrive at countertop 1. On the countertop 1, you see a candle 3, a candle 2, a cloth 2, a soapbar 1, a soapbottle 2, a soapbottle 1, and a spraybottle 1.
> take candle 1 from countertop 1
Nothing happens.
```

### Example 2: goal-relevant success (train/403, t_000591)

```
Your task is to: put some creditcard on armchair.
> go to diningtable 1
You arrive at diningtable 1. On the diningtable 1, you see a box 1, a creditcard 2, and a remotecontrol 1.
> take creditcard 2 from diningtable 1
You pick up the creditcard 2 from the diningtable 1.
> move creditcard 2 to diningtable 1
You move the creditcard 2 to the diningtable 1.
```

### Example 3: goal-relevant failure (location) (train/1770, t_002179)

```
Your task is to: heat some cup and put it in countertop.
> go to cabinet 8
You arrive at cabinet 8. The cabinet 8 is closed.
> go to microwave 1
You arrive at microwave 1. The microwave 1 is closed.
> take cup 1 from cabinet 3
Nothing happens.
```

### Example 4: goal-irrelevant success (train/258, t_002771)

```
Your task is to: put some bread on microwave.
> go to fridge 1
You arrive at fridge 1. The fridge 1 is closed.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, a bowl 3, a egg 2, a mug 3, a mug 2, and a tomato 2.
> go to countertop 2
You arrive at countertop 2. On the countertop 2, you see a bowl 1, a bread 1, a butterknife 1, a fork 1, a soapbottle 1, a spatula 2, and a winebottle 1.
```

## Set 3

### Example 1: goal-relevant success (train/49, t_001924)

```
Your task is to: examine the cd with the desklamp.
> go to desk 1
You arrive at desk 1. On the desk 1, you see a alarmclock 1, a book 1, a bowl 2, a cd 3, a cd 2, a cd 1, a keychain 3, a keychain 2, a pencil 3, and a pencil 2.
> take desklamp from desk 1
Nothing happens.
> take cd 1 from desk 1
You pick up the cd 1 from the desk 1.
```

### Example 2: goal-relevant failure (contents) (train/2165, t_002384)

```
Your task is to: find two handtowel and put them in toilet.
> go to handtowelholder 2
You arrive at handtowelholder 2. On the handtowelholder 2, you see a handtowel 2.
> go to toilet 1
You arrive at toilet 1. On the toilet 1, you see a candle 2, a dishsponge 2, and a soapbottle 1.
> take handtowel 1 from toilet 1
Nothing happens.
```

### Example 3: goal-irrelevant success (train/336, t_001426)

```
Your task is to: put a cellphone in armchair.
> open drawer 1
You open the drawer 1. The drawer 1 is open. In it, you see nothing.
> go to drawer 2
You arrive at drawer 2. The drawer 2 is closed.
> go to drawer 11
You arrive at drawer 11. The drawer 11 is closed.
```

### Example 4: goal-irrelevant failure (location) (train/1670, t_001798)

```
Your task is to: cool some tomato and put it in countertop.
> go to fridge 1
You arrive at fridge 1. The fridge 1 is closed.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, a apple 1, a bread 2, a egg 1, a mug 1, a pan 1, a plate 3, and a potato 2.
> take potato 2 from cabinet 12
Nothing happens.
```
