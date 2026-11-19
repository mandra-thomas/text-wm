# RQ3 confirmatory result (2026-11-19)

Runs: `rq3` (4 shards) and `rq3_analysis`, both at ddeabc2. Full numbers are in `rq3_metrics.json`, a copy of the analysis run's metrics.json.

## What was tested

H3 asked whether wrong predictions do more harm when the agent keeps them in its history than when it drops them, and whether this differs between false successes and false failures. The prediction was that false successes cost more when kept.

Setup:

- A Qwen3-8B ReAct agent, decoding greedily, played all 134 `valid_unseen` games.
- Before committing to each proposed action, the agent saw a predicted outcome. The predictions came from a copy of the environment with errors injected:
  - **False success:** a failing proposal gets the real success text. Injected with probability 0.75 per opportunity, at most 3 per episode.
  - **False failure:** a working proposal gets "Nothing happens.". Each false-failure episode was yoked to receive as many errors as its false-success twin.
- **Score per episode:** S is minus the step count, capped at 30. A failed episode scores −31.
- **Kept or dropped:** the predictions were either kept in the agent's history or dropped from it.

The test statistic is the interaction

$$I = [S_{FS} - S_{FF}]_{\text{kept}} - [S_{FS} - S_{FF}]_{\text{dropped}},$$

which H3 predicted to be negative. The primary set is the game–schedule units where both error types were delivered: 160 units in 61 games. An interim look at 112 games came before this final analysis.

## Result

**H3 is not supported.**

- On the primary set, I = +0.94 steps (95% game-bootstrap CI −0.20 to 2.14). The estimate has the opposite sign to the prediction and the interval includes zero.
- Over all 402 units, I = +0.24 (CI −0.45 to 0.93).
- The interim look had suggested significance. It did not hold.

| Condition (134 games) | Success | Errors delivered per episode |
| --- | --- | --- |
| No foresight | 47.0% | – |
| Perfect foresight, dropped | 58.2% | 0 |
| Perfect foresight, kept | 32.8% | 0 |
| Valid-or-not verdict only, dropped | 60.4% | 0 |
| False successes, dropped / kept | 59.0% / 35.1% | 1.15 / 1.60 |
| False failures, dropped / kept | 56.0% / 34.8% | 0.85 / 1.33 |

## Why the experiment could not answer the question

- **Keeping predictions hurt even when they were perfect.** With perfect predictions kept in the history, success fell from 58.2% to 32.8%, below the 47.0% with no foresight at all. The agent repeated actions in 82 of 134 episodes with kept predictions, against 24 with dropped ones. This looks like a history-format artifact, not an effect of the errors.
- **Floor effect.** 72–89% of the primary set's episodes failed.
- **The error matching fell short.** Episodes received 1.8–2.2 false failures against 2.4–2.5 false successes.
- **A false success was cheap to correct.** The valid-or-not oracle did as well as perfect foresight, so the next real observation corrected a false success.
- **Above all, the errors were injected.** They were not the world models' own predictions, so the experiment does not test what the goal leak found in RQ2 costs an agent.

## Checks

- **Determinism.** 332 of 342 copied false-failure episodes replayed identically; the 10 that differ come from 3 games.
- **Fallbacks.** Where a yoked false failure had nothing to yoke to, the baseline episode was reused: 36 times in the dropped condition and 119 in the kept condition.

Both checks are listed in the metrics file.
