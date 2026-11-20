# RQ1, H1 and secondary RQ2 results (2026-11-18)

Scoring: run `rq1_score`, 8 parts at de40795, on L40S, A40 and A100 GPUs. Every item's prompts for a model were scored in the same job. Analysis: run `rq1_analysis_v2` at 28b4d30, which re-reads the saved written predictions with the final parser version `3c25d40cc71529e2` (2026-11-20). Full numbers are in `rq1_v2_metrics.json` (a copy of that run's metrics.json, with a provenance block). The first analysis, `rq1_analysis` at de40795 with parser version `8639849027611cd2`, is kept as `rq1_metrics.json`; it is superseded for every number that depends on written predictions, and identical for every score. The design is in `deviations.md`, "Decided 2026-11-18, after the confirmatory RQ2 result". None of it is preregistered.

**Parser status (updated 2026-11-20).** Every number based on written predictions (A, H1, the false-success rates) depends on the parser that reads them. The parser went through six audit rounds against a blind rater: the first two samples were used to develop it, and rounds three to six each failed the rule fixed before the audits (at least 95% agreement on no effect against effect, and no row read in the opposite direction). Rounds five and six passed the agreement bar (95.5% and 96.6%) but each had rows read in the opposite direction. The audits were ended on 2026-11-19 without the parser passing its rule, and the written-prediction results are reported as validated at 95–97% agreement on stratified hard cases, with the no-opposite rule unmet (`deviations.md`, "Sixth parser audit").

The numbers in this note use the final parser version `3c25d40cc71529e2` (updated 2026-11-20 from `rq1_v2_metrics.json`). Against the version the first analysis ran with (`8639849027611cd2`), it moves 110 of the 68,340 predictions to the other side: 70 unreadable become failures, 6 unreadable become successes, and 34 change between success and failure. (The decision log's 93 is the fifth version's count against the same baseline.) The effect on this note is small: no verdict changes, A from the history alone is unchanged to three decimals for both models, and the largest changes are in the full-state rows and the Qwen example sets 2 and 3. Every changed number is listed in `deviations.md`, "Changes from the 2026-11-20 audit". Scores (ΔL, Δp) do not depend on the parser and are unchanged.

L is the log-odds of the success text against "Nothing happens.", and the "P(success)" and "Δp" below are means of s = σ(L). s is not the model's probability that the action succeeds; see `rq2_results.md`.

Items: written predictions on 6,527 true failures and 307 true successes from 132 unseen games. Scores on the same 4,666 main-test items as the confirmatory RQ2 run, from 125 games. No predictions or scores are missing.

## RQ1: the lean toward predicting success

A is the false-success rate minus the false-failure rate. The false-success rate is the share of failing actions for which the written prediction states an effect. The false-failure rate is the share of working actions for which it states no effect. A prediction counts as a failure prediction only if it says nothing happens. Partial effects (process only, or a different action or object) count on the success side.

| model and prompt | A | 95% CI | false successes | false failures | A, partial effects excluded | A, partial effects as failures |
|---|---|---|---|---|---|---|
| Qwen3-8B, history only (set 1) | 0.193 | 0.168 to 0.218 | 22.6% | 3.3% | 0.187 | 0.186 |
| Qwen3-8B, full state (set 1) | 0.284 | 0.258 to 0.312 | 31.4% | 3.0% | 0.268 | 0.259 |
| Llama-3.1-8B, history only (set 1) | 0.767 | 0.734 to 0.797 | 77.7% | 1.0% | 0.678 | 0.401 |
| Llama-3.1-8B, full state (set 1) | 0.666 | 0.639 to 0.695 | 68.9% | 2.3% | 0.586 | 0.421 |
| Word2World Llama-3.1-8B, full state | 0.098 | 0.084 to 0.111 | 10.1% | 0.3% | 0.097 | 0.097 |
| Word2World Qwen2.5-7B, full state | −0.032 | −0.061 to −0.004 | 3.7% | 6.8% | −0.032 | −0.032 |

A across example sets 1, 2 and 3: Qwen 0.19, 0.30 and 0.32 (SD 0.066); Llama 0.77, 0.77 and 0.88 (SD 0.068).

Qwen's false-success rate depends strongly on why the action fails. It is 10% when the agent is away from the receptacle, 10% when the agent does not hold the object, 43% when its hands are full, and 68% when the agent never went to the receptacle. Llama's is 68% to 99% for every reason except "already there" (going to the receptacle the agent is at, 41 items), where it is 100%. With the full state, Qwen's rate for "agent is away from the receptacle" rises from 10% to 30%. The layout shows where the object is, and Qwen predicts the take works without checking that the agent is there.

## H1

- **H1a: |A| is larger from the history alone than with the full state.**
  - Qwen: −0.091 (CI −0.119 to −0.064), so **not supported**, and reversed. The full state makes Qwen more optimistic.
  - Llama: +0.101 (CI 0.058 to 0.141), so **supported**.
- **H1b: A(Llama) − A(Qwen) > 0**, history only, set 1: +0.574 (CI 0.534 to 0.613), so **supported**.

At the final parser every H1 verdict is the same as at the earlier version; the estimates moved by at most 0.002.

## Secondary RQ2 effects (scores, not provisional)

ΔL is the mean over items of L(aligned) − L(swapped), in nats; Δp is in percentage points. Each comparison below is the first prompt minus the second, on the same items. Every p is the game-level permutation p (0.001 is the floor at 1,000 permutations).

| comparison | Qwen ΔL (CI) | Qwen Δp | Llama ΔL (CI) | Llama Δp |
|---|---|---|---|---|
| Set 1, aligned vs swapped | 6.10 (5.48 to 6.73) | 5.8 | 1.64 (1.50 to 1.79) | 16.4 |
| Set 2, aligned vs swapped | 8.71 (7.89 to 9.53) | 11.8 | 2.21 (1.98 to 2.44) | 23.1 |
| Set 3, aligned vs swapped | 6.20 (5.25 to 7.10) | 8.0 | 1.65 (1.46 to 1.85) | 19.1 |
| Full state, aligned vs swapped | 2.07 (1.61 to 2.55) | 2.4 | 0.47 (0.35 to 0.60) | 5.8 |
| "Ignore the goal" note, aligned vs swapped | 5.55 (4.98 to 6.12) | 6.4 | 1.69 (1.54 to 1.83) | 16.4 |
| Reduction from the note (difference in differences) | 0.56 (0.40 to 0.73) | | −0.04 (−0.10 to 0.01), p 0.14 | |
| Aligned vs neutral task ("explore the room") | 8.66 (7.97 to 9.35) | 6.8 | 2.84 (2.66 to 3.06) | 27.4 |
| Swapped vs neutral task | 2.56 (2.24 to 2.89) | 1.0 | 1.20 (1.03 to 1.38) | 11.0 |

SD of ΔL across the three example sets: Qwen 1.48, Llama 0.33.

The Word2World models, full state as trained, with aligned vs swapped goals: Llama −0.28 (−0.42 to −0.15, p 0.001); Qwen −0.03 (−0.21 to 0.15, p 0.74).

Mean two-candidate score s = σ(L) (logged as "P(success)") under each prompt, Qwen / Llama:

| prompt | Qwen | Llama |
|---|---|---|
| aligned | 0.075 | 0.408 |
| swapped | 0.017 | 0.244 |
| neutral task | 0.007 | 0.134 |
| aligned + note | 0.098 | 0.410 |
| swapped + note | 0.034 | 0.247 |
| aligned, full state | 0.184 | 0.343 |
| swapped, full state | 0.161 | 0.285 |

## Extra-mention control (exploratory)

The control adds two prompts, each naming both objects:
- **aligned+**: the aligned task, plus "There is a {swapped object} somewhere in the room."
- **swapped+**: the swapped task, plus "There is a {action's object} somewhere in the room."

The three parts below sum exactly to ΔL. "Remove other-object sentence" is aligned − aligned+, "goal role" is aligned+ − swapped+, and "mention" is swapped+ − swapped.

| subset (items) | model | remove other-object sentence | goal role, both named | mention of the action's object |
|---|---|---|---|---|
| all (4,666) | Qwen | 1.09 | −0.99 (−1.42 to −0.55) | 6.00 (5.37 to 6.58) |
| all (4,666) | Llama | 0.06 | 0.59 (0.43 to 0.74) | 0.99 (0.86 to 1.12) |
| take (2,449) | Qwen | 1.00 | −0.59 (−1.02 to −0.14) | 4.78 |
| take (2,449) | Llama | 0.12 | 1.00 (0.73 to 1.23) | 1.20 |
| not take (2,217) | Qwen | 1.19 | −1.43 (−2.13 to −0.78) | 7.35 |
| not take (2,217) | Llama | −0.01 | 0.15 (0.03 to 0.26) | 0.76 |
| history length 0 (514) | Qwen | 3.59 | 2.83 (2.08 to 3.57) | 4.55 |
| history length 0 (514) | Llama | 0.38 | 0.10 (0.02 to 0.19) | 1.04 |
| history length 1+ (4,152) | Qwen | 0.78 | −1.46 (−1.89 to −1.03) | 6.18 |
| history length 1+ (4,152) | Llama | 0.02 | 0.65 (0.48 to 0.82) | 0.98 |

In Δp, the goal role is +1.2 pp for Qwen and +6.1 pp for Llama (+11.9 pp on take items).

## Reading

- Both prompted models lean toward predicting success, Llama far more. It states an effect for 78% of actions that do nothing, so H1b holds by a wide margin. The fine-tuned Word2World models barely lean (Llama) or lean slightly the other way (Qwen).
- Giving the hidden state reduces Llama's lean (H1a holds) but increases Qwen's. In both, it cuts the goal effect by about 70%, without removing it: by 66% for Qwen (CI 59% to 73%) and 71% for Llama (64% to 78%), from `audit_metrics.json`, `ratios.<model>.full_layout_share_removed`.
- The RQ2 effect holds with every example set, and above the "strong" threshold each time. Telling the model to ignore the goal removes 9% of Qwen's effect (CI 7% to 12%) and none of Llama's (−3%, CI −6% to 1%), from `audit_metrics.json`, `ratios.<model>.ignore_note_share_removed`.
- **Superseded (2026-11-19):** the next point's reading that the effect comes from the object being *named* was overturned by the object-free goals in `exploratory_results.md`, section 1. With the object only described, 61% (Qwen) and 92% (Llama) of the effect remains, so the task acts through the object's meaning. The point is kept as written at the time.
- Most of the RQ2 effect comes from the action's object being named in the prompt. For Qwen, naming it in a plain sentence reproduces almost the whole effect. Once both objects are named, making it the goal lowers the log-odds, except on an episode's first step. For Llama, the goal role is about a third of the effect, concentrated on take actions, which the 2×2 could not cover. Both controls agree that mention dominates. They disagree on the sign of the small goal-specific part on move, heat, cool and clean items.
