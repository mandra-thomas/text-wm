# RQ2 confirmatory result (2026-11-18)

Scoring: run `rq2_score`, 4 shards at ddeabc2. Shards 0–2 ran on A40s and shard 3 on an L40S; each item's aligned and swapped prompts were scored in the same job on the same GPU. Analysis: run `rq2_analysis` at ddeabc2. Full numbers are in `rq2_confirmatory_metrics.json` (a copy of the run's metrics.json).

Items: 4,666 main-test items from 1,033 states in 125 of the 134 unseen games. All passed full replay verification. The preregistered minimum was met, and no primary scores are missing. Every item is a failure that the context determines: the true next observation is "Nothing happens."

## Preregistered verdict

| model | ΔL (nats) | 95% CI | Holm p | verdict |
|---|---|---|---|---|
| Qwen3-8B, few-shot | 6.12 | 5.48 to 6.75 | 0.002 | supported (strong) |
| Llama-3.1-8B, few-shot | 1.65 | 1.51 to 1.78 | 0.002 | supported (strong) |

The p-values are at the floor set by 1,000 game-level permutations (raw p = 1/1001, doubled by Holm).

## Secondary measures

| model | Δp (pp) | ΔL first token (nats) | mean P(success) aligned / swapped | written success shift (pp) |
|---|---|---|---|---|
| Qwen3-8B | 5.9 (5.1 to 6.8) | 2.52 (2.26 to 2.79) | 0.076 / 0.017 | 8.8 (7.5 to 10.3) |
| Llama-3.1-8B | 16.4 (14.6 to 18.3) | 0.68 (0.62 to 0.76) | 0.407 / 0.243 | 27.4 (24.6 to 30.1) |

By history length, Qwen's effect is concentrated where the agent has not acted yet. With history length 0 (514 items), ΔL is 11.0 nats and Δp is 42 pp. At lengths 1–3, 4–7 and 8+, ΔL is 5.2 to 6.0 nats but Δp is only about 1.4 pp. Llama's effect is flat across lengths: ΔL is 1.5 to 1.7 nats and Δp is 13 to 18 pp.

## Object-mention 2×2 (exploratory)

The 2×2 covers 1,076 items whose action is move, heat, cool or clean; take items cannot be built. It has four task lines. Aligned is the true goal, which the action advances. Swapped is the true goal with another object. Mention names the action's object, but in a goal the action does not advance: another transformation, or another destination. Fourth applies the same change to the swapped line.

The two parts sum exactly to aligned minus swapped on these items. The goal-directed part is (aligned − mention) − (swapped − fourth). The mention part is mention − fourth.

| model | goal-directed (nats) | mention (nats) |
|---|---|---|
| Qwen3-8B | 0.72 (0.05 to 1.38), p = 0.04 | 6.93 (6.04 to 7.77), p = 0.001 |
| Llama-3.1-8B | −0.18 (−0.31 to −0.05), p = 0.01 | 1.04 (0.87 to 1.23), p = 0.001 |

On these items, almost all of the aligned-versus-swapped difference comes from naming the action's object in the task line. Very little comes from the action advancing the goal.
