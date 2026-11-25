# Wishful Foresight: does the agent's goal leak into an LLM world model?

**New to the project? Start with [notes/GUIDE.md](notes/GUIDE.md)**, a reader's guide to how the experiments are set up and scored.

Language-model agents increasingly consult a **world model**, a language model asked what the environment will reply to an action, before they act. That world model is often shown the agent's task.

In a goal-independent environment, an action whose failure is already settled by what the agent has seen should be predicted to fail, whatever the task says. This project tests whether it is. The study is preregistered and runs on ALFWorld (text household tasks). It compares two prompted models, Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, and also scores Word2World's fine-tuned ALFWorld world models. Later, exploratory runs extend the ALFWorld test to eleven models from four developers ([notes/cross_model_summary.md](notes/cross_model_summary.md)) and to two further environments.

## The test

Each test item is an action that fails, and whose failure follows from the history. An example: `take mug 1 from cabinet 3` when the agent has not gone to cabinet 3. The environment's reply is "Nothing happens."

For each item:

- We teacher-force two candidate replies: the action's success text and "Nothing happens."
- We compute L, the log-odds of the first over the second.
- We do this under two task lines that differ only in one object. The **aligned** task is the episode's real task, for example "cool some mug and put it in cabinet". The **swapped** task uses another object in the room, for example "cool some plate and put it in cabinet".

The goal effect **ΔL** is the mean of L(aligned) − L(swapped), in nats. A correct world model gives ΔL = 0.

| Question | What it measures |
| --- | --- |
| RQ2 (primary, preregistered) | ΔL on determined failures |
| RQ1 | Direction of errors in written predictions: false-success rate minus false-failure rate (A) |
| RQ3 | Cost to an agent of injected false successes against false failures, with predictions dropped or kept |

## Results recorded in this repository

**Audit and reruns (2026-11-20).** A documentation and rigor audit led to seven CPU analysis reruns at 28b4d30. No verdict changed. The RQ2 result is the same with the shared-spot items (none was excluded by that rule alone), the RQ1 numbers are now at the final parser, and the headline ratios have intervals. Every changed number, old and new, is listed in [notes/deviations.md](notes/deviations.md), "Changes from the 2026-11-20 audit".

**RQ2, confirmatory.** Details are in [notes/rq2_results.md](notes/rq2_results.md). The test used 4,666 replay-verified failing actions from 125 of the 134 unseen games. Both models pass the preregistered test, whose threshold was 0.5 nats:

| Model | ΔL | 95% CI | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 6.12 nats | 5.48 to 6.75 | 0.002 | supported (strong) |
| Llama-3.1-8B-Instruct | 1.65 nats | 1.51 to 1.78 | 0.002 | supported (strong) |

The two models differ in where the effect lands:

- **Qwen:** in the study's thinking-off format, concentrated where the agent has not acted yet. That concentration comes from the format's empty think block and disappears without it (see the format note below).
- **Llama:** spread across the episode.

*(Updated 2026-11-23, exploratory follow-up; the verdicts above are unchanged.)* About half of Qwen3-8B's 6.12 nats depends on the prompt format. With thinking off, Qwen3's chat template puts an empty `<think></think>` block before the answer. Rescored without it on all 4,666 items, the effect is 3.31 nats (2.95 to 3.64), and a plain-text prompt with no chat template gives the same value on a 300-item subsample (3.26 against 3.28). The standardized effect (1.02 against 1.05) and the share of items moved toward success (87.9% against 88.7%) do not change, so the block stretches the scale of Qwen's scores rather than making it more goal-sensitive. Qwen's step-0 concentration (11.0 nats) exists only with the block: without it, step 0 gives 2.73 nats against 3.3 to 3.6 at later steps. Details in [notes/qwen_effect_audit_pipeline.md](notes/qwen_effect_audit_pipeline.md), section 7.

**RQ1, H1 and secondary RQ2 conditions.** Details are in [notes/rq1_results.md](notes/rq1_results.md).

- **Lean toward success.** Both prompted models lean toward predicting success, Llama far more (A = 0.77 against 0.19).
- **H1.** H1b holds. H1a holds for Llama and is reversed for Qwen.
- **The goal effect under other conditions:**
  - it holds with each of three few-shot example sets;
  - an instruction to ignore the goal removes 9% of Qwen's effect (95% CI 7% to 12%) and none of Llama's;
  - showing the hidden state cuts it by about 70% without removing it (66% for Qwen, 71% for Llama);
  - the fine-tuned Word2World Qwen model shows none, and the fine-tuned Llama model a small reversed effect (−0.28 nats, CI −0.42 to −0.15).
- **Caveat.** The written-prediction numbers rest on a parser that never met its prespecified audit rule. The audits ended after six rounds with 95–97% agreement on hard cases, but with some replies read in the opposite direction. The note uses the final parser version, which reads 110 of the 68,340 predictions differently from the first analysis; no verdict changes. See the note's "Parser status".
- **Superseded reading.** The note's reading that "most of the RQ2 effect comes from the action's object being named" was overturned by the exploratory runs below: with object-free task descriptions, 61% (Qwen) and 92% (Llama) of the effect remains.

**RQ3, confirmatory.** Details are in [notes/rq3_results.md](notes/rq3_results.md). H3 is not supported: the interaction is +0.94 steps (95% CI −0.20 to 2.14), opposite in sign to the prediction. The experiment could not test the question well:

- keeping even perfect predictions in the agent's history lowered success below having no foresight at all;
- most episodes in the primary set failed;
- its errors were injected, not the world models' own;
- *(revised 2026-11-25)* the agent's prompt included ALFWorld's admissible-action list, and oracle foresight that only said whether an action was valid did as well as perfect foresight (60.4% against 58.2% success), so the null means "not detected in this setup".

**Exploratory follow-ups.** Details are in [notes/exploratory_results.md](notes/exploratory_results.md). None was preregistered; each design was logged before its data.

- **Not only the object's name.** Most of the effect survives task descriptions that never name the object (61% for Qwen, 95% CI 54% to 68%; 92% for Llama, 83% to 103%). But for Qwen, merely stating that the object is somewhere in the room reproduces nearly all of it.
- **Other formats and tasks.** It replicates when the model answers a fixed yes/no question. It is unchanged when the aligned task is not the exact one the agent pursued.
- **Discrimination.** On our candidates it lowers how well the models rank working actions above failing ones: a large drop for Llama, a small one for Qwen. All failing candidates act on the goal's object while most working ones do not, so this reflects the task promoting actions on its object over unrelated working actions.
- **Practical size depends on model and format.**
  - Llama's predicted success moves by 13–40 points.
  - In the study's thinking-off format, Qwen's moves by 42 points before the agent has acted and little after, because Qwen stays confident that these actions fail. Both the step-0 peak and how far Qwen's scores sit from the decision boundary depend on the empty think block of that format (see the format note under RQ2).
- **Instructions.** An instruction to ignore the goal barely helps.
- **Reasoning.** *(revised 2026-11-25)* Whether Qwen3 thinking mode reduces the effect depends on the scale, and is not determined. In nats it does not: after reasoning, the aligned-vs-swapped effect is 4.01 nats (95% CI 2.87 to 5.12) on 1,000 items, and without the empty `<think></think>` block that Qwen3's thinking-off template places before the answer the same items give 3.33 nats (2.94 to 3.70). On the standardized scale (the mean item shift divided by the SD of the item shifts) it falls: 0.25 (0.18 to 0.32) after reasoning, against 1.02 with the empty block and 1.10 without it on the same items, and the share of items moved toward success falls from 88% to 61% (post hoc, recomputed from the archived `thinking_probe` run; the no-reasoning values are in `notes/qwen_effect_audit_followup_metrics.json`, `thinking_probe_items`). Part of that fall is sampling noise: each task line samples its own reasoning trace, which widens the spread of item shifts, so the standardized effect after reasoning is biased down by an unknown amount. So neither "reasoning does not reduce the effect" nor "reasoning reduces it" is shown. The comparison also crosses scoring engines (vLLM after reasoning, Hugging Face without). The reading of 2026-11-23 said reasoning "does not reduce the effect", from the nats alone. *(Reading revised 2026-11-23, exploratory follow-up.)* An earlier reading said thinking mode narrows the gap by a third (34%, 95% CI 16% to 52%) without reducing the lift over a neutral task. It compared against the thinking-off answer, whose empty block about doubles Qwen's effect in nats (6.06 on these items), so it compared a filled think block with an empty one. The comparisons with the neutral task have no baseline without the block and are not read. The probe's numbers stand; only their reading changes.
- **Mechanism.** The dissociation test of "the task as evidence that its object exists" gave mixed readouts and needs a confirmatory test.
- **Not missing information.** In the mechanism diagnostic, stating the facts that decide the failure barely reduces the effect, and stating the environment's rules *increases* it. The effect is fully present where the model answers questions about the state correctly. So a failure to track the state does not explain it. What does is open: the diagnostic's note read the rest as a task-relevance bias at answer time, but the later mechanism battery (status table below) found no single account, and for Qwen a bare mention of the object, even with an instruction to ignore it, reproduces the whole effect.
- **Model size (Qwen3 0.6B to 32B).**
  - The effect is present at every size.
  - The logged prediction that it would not grow with size failed over the full range, because the two smallest models can't do the task.
  - Among the competent sizes, the log-odds effect shrinks with size but the practical size does not. Asked whether a failing action will succeed, Qwen3-32B answers "Yes" on 11.6% of them under the real task, against 3.8% under the swapped one (its mean probability of "Yes", mean σ of the yes/no score, is 0.150 against 0.058).
  - *(Updated 2026-11-23, exploratory follow-up.)* The shrinking in nats is mostly the thinking-off block's scale inflation, which differs by size and is not monotone in it (plain text keeps 31%, 53% and 23% of the with-block effect for 4B, 8B and 14B on 300 items). Standardized, the effect is flat with the block (0.97, 1.02 and 1.00); in plain text it stays flat for 4B and 8B (1.05, 1.15) but not for 14B (0.72). Qwen3-32B was not rescored.

## Status (2026-11-23)

| Stage | Run(s) | State |
| --- | --- | --- |
| Pilots A and B | `pilotA_*`, `pilotB_*` | Done (2026-11-17) |
| Item collection, 134 `valid_unseen` games | `rq12_transitions_s0..s2` | Done; every item replay-verified |
| RQ2 confirmatory (preregistered) | `rq2_score`, `rq2_analysis` | Done; results in `notes/rq2_results.md` |
| RQ1, H1 and secondary RQ2 conditions | `rq1_score`, `rq1_analysis` | Done; results in `notes/rq1_results.md` (parser audits ended without a pass; see the note) |
| RQ3 confirmatory | `rq3`, `rq3_analysis` | Done; results in `notes/rq3_results.md` (H3 not supported) |
| Object-free goals, working actions | `rq1x_score`, `rq1x_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Qwen3 thinking mode | `thinking_probe`, `thinking_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Dissociation: object exists / located / absent | `rq1d_score`, `rq1d_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Counterfactual aligned task, yes/no feasibility | `rq1v_score`, `rq1v_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Mechanism diagnostic: state inference or judgement | `mech_score`, `mech_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Mechanism battery: alternative task lines that separate candidate accounts | `mech2_score`, `mech2_analysis`; CPU script `scripts/mech2_dx.py` | Done (exploratory; design frozen before scoring at tag `mechanism2-design-frozen-2026-11-22`, commit `6eab8af`). Frozen verdict for Qwen3-8B, Llama-3.1-8B and Mistral-7B (secondary): "no single account; mixture or an account not on the list". Every content account (task as evidence, salience, wishful prediction, policy blending, word overlap) is contradicted in both formats; position is untested. For Qwen, adding "Ignore the {obj}." to the swapped task reproduces the whole effect (r 1.03), the described object keeps most of it (r 0.84), and explicit claims about the object's location move predictions by about twice the effect in the stated direction (r −2.26). The user's hope barely matters. Results in `notes/mechanism2_results.md`, robustness deep-dive in `notes/mechanism2_dx.md`, design in `notes/mechanism2_design.md` |
| Model-size series, Qwen3 0.6B to 32B | `size_score_<model>`, `size_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Audit analyses and reruns (CPU) | `audit_analysis`, `rq2_shared_analysis`, `rq1_analysis_v2`, `rq1x_analysis_v2`, `rq1v_analysis_v2`, `thinking_analysis_v2`, `size_analysis_v2` | Done (2026-11-20); changes listed in `notes/deviations.md` |
| Forensic audit of Qwen3-8B's effect size (exploratory) | none (stored scores only); scripts `scripts/qwen_audit/` | Done (analysis list and interpretation rule committed before computing, commit `62be254`). No pipeline fault: no missing, duplicate or stale score, prompts differ only in the task line, and rescoring on the same GPU type reproduces every score. The effect is broad (88% of items move toward success; median item 5.35 nats), but its size in nats is "mostly measurement" by the fixed rule: 59% of the 6.12 nats (3.60) is the success sentence's tail, and divided by the SD of each model's item-level shifts Qwen3-8B's effect is 1.02 against Llama-3.1-8B's 0.85 (two-candidate), 0.75 against 0.68 (first token) and 0.88 against 1.12 (yes/no). Standardized, the Qwen3 size series is flat (0.90, 1.02, 0.96, 0.86 for 4B to 32B) in the study's thinking-off format; in plain text that holds for 4B and 8B but not 14B (see below). The confirmatory RQ2 result is unchanged. The second, independent audit (`notes/qwen_effect_audit_pipeline.md`, 2026-11-23) re-implemented the Qwen scoring on GPU: a scorer written from scratch reproduces the stored scores exactly on the same GPU type, and vLLM and float32 weights agree, so no bug produced the number. About half of the effect's size depends on one prompt-format detail, the empty `<think></think>` block that Qwen3's chat template places before the answer with thinking off. Its follow-up (exploratory, plan committed before scoring at `8810f24`) rescored all 4,666 items without the block: 3.31 nats against 6.12 (ratio 0.54), with the standardized effect (1.02 against 1.05) and the share of items moved (87.9% against 88.7%) unchanged, so the block stretches the scale rather than adding goal sensitivity. The render without the block is Qwen's template with thinking on, where the two answers get under 0.1% of the next-token probability, so it is not a natural alternative; plain text (no chat template, answers keep at least 90%) is the comparison to rely on, and it gives the same halving (3.26 against 3.28 on 300 items). The step-0 concentration exists only with the block (10.99 nats at step 0; 2.73 without). In plain text the size series stays flat when standardized for 4B and 8B (1.05, 1.15) but not for 14B (0.72); 32B was not rescored. The thinking-mode reduction compares a filled think block with an empty one; against the no-block level (3.33 on the probe's items) reasoning (4.01) shows no reduction in nats, but the standardized effect (0.25 against 1.10) and the share of items moved (61% against 88%) fall, partly through sampling noise in the reasoning traces, so whether reasoning reduces the effect is not determined *(revised 2026-11-25; see "Reasoning" above)*. In every format tried the effect stays positive and 2 to 5 times Llama-3.1-8B's. Results in `notes/qwen_effect_audit_forensic.md` and `notes/qwen_effect_audit_pipeline.md` |
| Model families, wave A (Gemma-2-9B, Mistral-7B-v0.3, Llama-3.2-1B/3B) | `family_score_<model>`, `families_analysis` | Done (exploratory); results in `notes/families_results.md`, design in `notes/families_design.md` |
| Decision simulation, Stage 1a (fixed candidate menus): does the bias change which action a verifier picks? | `decisionsim_score_<model>`, `decisionsim_analysis`; CPU scripts `scripts/decisionsim_dx.py`, `scripts/decisionsim_dx_frozen.py` | Done; all five models scored (design frozen before scoring at tag `decisionsim-stage1a-frozen-2026-11-21`, commit `eeeb7dd`). The gate for a closed-loop study, frozen at that tag, reads "worth doing", decided by Qwen3-8B: a verifier shown the real task makes the chosen action fail at 23.9% of states, against 5.8% with no verifier and 4.4% with a task-blind verifier (aligned minus task-blind +19.5 points [17.2, 22.3]), mostly at step 0. Llama-3.1-8B: no reliable change; the three secondary models, applied post hoc, also reach "worth doing" but never decide the gate. Mechanism (exploratory): harm needs two parts together, any concrete task making Qwen's yes/no verifier reject working first moves, and naming the object ranking failing goal-object actions first in the fallback. Under natural changes of rule or menu the effect keeps its direction but shrinks to 3–6 points. *(Updated 2026-11-23, exploratory follow-up: the step-0 harm is format-sensitive. At the 134 step-0 states, Qwen's real-task verifier accepts 7.5% of the agent's first moves with Qwen3's empty thinking-off block before the answer, against 90% task-blind, a step-0 harm of 71.6 points; in plain text it accepts 26% against 98%, a harm of 38.8 points [30.6, 47.0]. The block roughly doubles the step-0 harm but does not create it, and the real-versus-swapped gap persists in plain text (39 against 4 points). Frozen verdicts unchanged; `notes/qwen_effect_audit_pipeline.md` §7.2 C.)* Results in `notes/decisionsim_results.md`, exploratory analysis in `notes/decisionsim_dx.md`, design in `notes/decisionsim_design.md` |
| Model families, wave B and post-training (Qwen3.5-9B, Ministral-3-8B, Gemma 4 E4B and 31B, Qwen3.8-27B; base against instruct) | branch `families-waveb` | Design frozen at tag `families-waveb-design-frozen-2026-11-22` (commit `f6bbffb`) after one review and its fixes, with node-local weights verified against committed hash manifests. Environment check (Mistral-7B and Qwen3-8B rescored in the new software stack and compared with their earlier scores) **not passed** (job 7247973): Mistral passes every criterion; Qwen3-8B fails only the yes/no per-score criterion (mean absolute difference 0.248 nats against 0.07), while its goal effect moves 0.026 nats and tokens are identical. A float64 diagnostic (branch `families-waveb-envfix`, job 7248092) found the cause is not bf16 rounding: before rounding, the newer stack still moves Qwen's yes/no readout logits by a median of 0.44% (about the bf16 unit roundoff, 2⁻⁸), and its large readout logits turn that into about 0.2 nats per score. That is about seven times the change measured across GPU types in the old stack (0.035 nats per score, 91% of scores identical; 30% identical across the stack change). Decided 2026-11-23, before any wave B score exists: every older model used in a wave B comparison is rescored in the new stack, so all comparisons sit in one environment. *(revised 2026-11-25)* The amendment is written and re-reviewed (review 2 and its fixes on branch `families-waveb-serve2-rescore`); a re-freeze is in progress. No wave B model has been scored |
| Second environment: TextCraft | `textcraft_score`, `textcraft_analysis` | Done (confirmatory for the frozen test; design frozen before scoring at tag `textcraft-design-frozen-2026-11-20`, commit `5a3702a`). Confirmed for Qwen3-8B (ΔL +0.94 nats [0.63, 1.24]; about a third to a half of ALFWorld's effect on reply-anchored readouts), small negative effect for Llama-3.1-8B (−0.19 [−0.30, −0.09]) with weak competence, so it does not replicate across both models. Results in `notes/textcraft_results.md`, design in `notes/textcraft_design.md` |
| TextCraft follow-ups (exploratory, post hoc) | `textcraft_dx_zeroshot`, `textcraft_dx_generate`, `textcraft_dx_objectgoal`, `textcraft_dx_prompts`; CPU scripts `scripts/textcraft_*.py`, `scripts/xenv/` | Done; none changes the confirmatory verdict. Llama diagnostics (`notes/textcraft_llama_dx.md`) and competence (`notes/textcraft_llama_competence.md`); comparison with ALFWorld on every shared scale (`notes/textcraft_vs_alfworld.md`); Qwen robustness (`notes/textcraft_qwen_robustness.md`) and competence (`notes/textcraft_qwen_competence.md`). Summary in `notes/textcraft_results.md`, "Follow-up analyses" |
| TextCraft round 2 (clean prompt, five models, lift and penalty decomposition frozen before scoring) | `textcraft_r2_score_<model>`, `textcraft_r2_analysis` (code at tag `textcraft-r2-design-frozen-2026-11-23`, branch `textcraft-r2`) | Done (confirmatory for the 50-test family; frozen at that tag, commit `4ccf84b`; scored at that commit on L40S, ICE jobs 7247899–7247918, analysis job 7247919, no score missing). The within-item penalty (naming and completing the goal against making the same craft an ingredient of the goal) is confirmed below zero in all three models that discriminate on goal crafting: Qwen3-8B −1.02 [−1.62, −0.45], Llama-3.1-8B −1.68 [−1.94, −1.42], Gemma-2-9B −1.27 [−1.48, −1.06]; across-model reading "penalty general across the competent models"; no rejection rule (R1, R2, R5) fires. Intermediate lift: Qwen3-8B confirmed above zero (+0.67), Llama-3.1-8B equivalent to zero (+0.095; does not discriminate on ingredients), Gemma-2-9B confirmed below zero (−0.30). Readings: Llama-3.1-8B "completion penalty, naming constrained" (measurable effect only), Gemma-2-9B "naming / partial-completion penalty", Qwen3-8B "penalty present, mechanism unresolved". Added empty look-ups do not change the penalty beyond the margin where readable (Llama-3.1, Gemma). Descriptive: R3 fires for Qwen3-8B (within-item penalty and between-item role gap disagree; marginal, no familywise control); central question for Llama-3.1-8B "penalty with inventory-consistent tracking". Mistral-7B and Llama-3.2-3B discriminate for no contrast (Yes to 99.9% and 96.9% of failures), so their effects are measurable only. Results in `notes/textcraft_r2_results.md`, design in `notes/textcraft_r2_design.md` |
| TextCraft round 3 (contemporary models: Qwen3.5-9B, Ministral-3-8B, Gemma-4-12B; larger-model extension; exploratory Qwen3.5 thinking arm) | branch `textcraft-r3` | Design frozen at tag `textcraft-r3-design-frozen-2026-11-23` (commit `0d6f460`) after two independent reviews; design content complete before round 2's analysis was read. Smoke runs (2026-11-23): Qwen3.5-9B and Ministral-3-8B pass every gate; Gemma-4-12B fails the attention-agreement gate (eager and SDPA differ by 0.22 nats on yes/no), so the pre-registered fallback assigns the Gemma slot to Gemma-4-E4B (its smoke queued); the Qwen3.5 thinking arm's start check (padded against alone, 0.111 against 0.10) stopped it before generating, under diagnosis. Core scoring of Qwen3.5-9B and Ministral-3-8B and the environment checks running; nothing read |
| Third environment: τ-bench retail | `tau_build`, `tau_verify`, `tau_score`, `tau_analysis`, `tau_contamination` | Done. Qwen3-8B first-token effect +3.98 nats [3.61, 4.35], replicates the ALFWorld direction but is read as a general task-consistency bias: a working call on the same order moves more than the failing call (fail minus witness −5.47 [−5.88, −5.05]). Llama-3.1-8B +0.47 [0.41, 0.54], small and failure-specific by the frozen reading table, though the exploratory follow-up finds the failure-specific part small (+0.09) and confined to two cells. Joint verdict: not a replication across both models. The design was pushed before scoring (commit `c737cd6`); the tag `taubench-design-frozen-2026-11-21` was created after scoring and before analysis. Results in `notes/tau_results.md`, design in `notes/tau_design.md` |
| τ-bench follow-up (exploratory, post hoc) | CPU script `scripts/tau_dx.py` | Done; changes no confirmatory verdict. Independent recompute, class and cell structure, competence, yes/no and full-reply readouts, practical size and cross-environment comparison: `notes/tau_dx.md` |
| Interpretability pilot (Qwen3-8B, yes/no) | `wf/interp/`; runs `interp_pilot_stage0`, `interp_pilot_patch`, `interp_pilot_ablate`, `interp_pilot_analysis` | Done (exploratory calibration run; by design it gives no hypothesis verdict). Design frozen at tag `interp-design-frozen-2026-11-22` (commit `7ade3b9`); analysis job 7295359 at `c73147e` after an analysis-only deviation (tags `interp-deviation-analysis-fix-2026-11-23`, `-2026-11-24`). Reproduction gates pass (yes/no 2.782 against the logged 2.783 nats). *(revised 2026-11-25, to match the independent audit in section 13 of the results note; no reported number changes.)* The audit's classification: replication of the behavioural effect achieved; causal localization by token group and layer partial; computational mechanism and semantic interpretation not achieved. Patching finds the task's effect at the action line's object word by layers 11–18 and at the answer position from layers 19–22. That route is largely forced by prompt order (any account must pass through the action's object token), no edge of it was tested, and the single-group shares are not additive, so it shows where the difference is used, not a flow or what it means. The history's patching share is low over all items (0.031 [0.010, 0.066]) mainly because most histories do not name the object; on later items whose history does, it is 0.40 [0.23, 0.72]. Those items are 7.0% of later failures and 6.2% of all failures (the "3%" given here before was their weight in the patching sample, which keeps only items with an effect of at least 1 nat). Aligned-against-swapped patching cannot detect the history being used as evidence, so the low share does not show that the history is unused. Removing one direction at the answer position: the later-step reading "after arrival" holds only with percentile intervals (layer 24's lower bound 0.566 against a threshold of 0.6) and is "inconclusive" with normal or basic bootstrap intervals; step 0 and pooled are inconclusive. The 0.44 cosine between the task direction and the neutral success direction is confounded with object mention (88% of determined successes name the object in their history, against 7% of later failures) and was read against the wrong null, so it is not evidence for a generic success feature; the two intervention rows for that question are empty by construction. A stage-0 control first left out of the note: after step 0, adding "There is a {obj} somewhere in the room." to the swapped task raises the yes/no log-odds by 2.26 nats, against 2.29 for making the object the goal (at step 0, 2.21 against 7.70); the sentence asserts that the object exists, so it cannot separate a lexical account from the task as evidence that the object exists. Results: `notes/interp_results.md` (section 13 for the audit) |
| Decision simulation, Stage 1b (candidates generated and ranked by the agent; primary for practical impact) | runs `decisionsim_agent_sample`, `decisionsim_agent_label`, `decisionsim_agent_encounter`, `decisionsim_agent_score_<model>`, `decisionsim_agent_analysis`; CPU script `scripts/decisionsim_agent_dx.py` | Done (design frozen before any census label was read at tag `decisionsim-stage1b-frozen-2026-11-22`, commit `2a05707`). Frozen verdicts: Llama-3.1-8B and Mistral-7B "no reliable difference" overall (+0.17 points [−0.92, +1.27] and [−0.09, +0.47]) and on encounter states (+4.65 [−1.16, +12.50]; 0.00 [−2.46, +2.05]); neither verifier abstains enough to count as inert, so the pre-declared reading is "the bias rarely reaches this agent's decisions". The agent's own lists contain a determined goal-object failure at 4.3% of states [2.2, 6.8], never at step 0. Qwen3-8B (contrast, descriptive) +2.29 [+1.36, +3.33], mostly from rejecting working alternatives under any task, not from naming the object. Deep-dive (exploratory): the nulls are stable; Llama's encounter estimate rests on one looping game; the RQ2 bias is visible in Llama's acceptances but the selection rule pre-empts it. *(revised 2026-11-25)* The agent's prompt included ALFWorld's admissible-action list, and every failing candidate it proposed lay outside that list, so these nulls mean "not detected in this setup", not that the bias is harmless for agents in general. Results in `notes/decisionsim_agent_results.md`, deep-dive in `notes/decisionsim_agent_dx.md`, design in `notes/decisionsim_agent_design.md` |
| τ-bench retail, model families (Mistral-7B, Llama-3.2-3B; Gemma-2-9B not run, its context window is too short) | runs `tau_family_token_check`, `tau_family_score_smoke`, `tau_family_smoke_analysis`, `tau_family_score_<model>`, `tau_family_placement_mistral-7b-v0.3`, `tau_family_analysis` (code on branch `tau-families`) | Done (confirmatory per-model family of two; design frozen before any score at tag `tau-families-design-frozen-2026-11-23`, commit `a078a8d`). Llama-3.2-3B: "replicates (ALFWorld direction)", +0.99 nats [0.90, 1.08] on the first token, every gate passed; its working call moves as much (+1.07), so "a shift shared by working and failing calls (working calls move more)", Qwen3-8B's label; 82% of items move; both failure classes positive. Mistral-7B: "not interpretable (fails the families competence rule on the first-token readout)" (AUC 0.58 against 0.70; discrimination 0.36 nats against 1); unread estimate +0.23 [0.18, 0.27]; exploratory placement check: format-dependent (+0.53 with the system text in the first turn). Neither model is interpretable on yes/no. Results in `notes/tau_families_results.md`, metrics `notes/tau_families_metrics.json`, design in `notes/tau_families_design.md` |
| τ-bench authorization experiment: does part of the effect reflect a belief that the backend refuses unrequested writes? | runs `tau_auth_score_smoke`, `tau_auth_smoke_analysis`, `tau_auth_score`, `tau_auth_analysis` (code on branch `tau-auth`) | Done (exploratory; design frozen before any score at tag `tau-auth-design-frozen-2026-11-22`, commit `f70b285`). 192 items, three descriptions of the backend: the confirmatory prompt, one stating and showing that the task never changes the output (task-blind), and one with a request rule (positive control). Qwen3-8B passes both gates (task-blind comprehension 24/24 on all four questions; positive control +3.38 nats [+3.01, +3.77]); its request component (what a write request adds over a bare question, +2.30 nats [+1.87, +2.74]) is not removed by the task-blind description (−0.09 [−0.42, +0.26]; +2.39 persists), nor is its naming component (−0.08 removed, +1.95 persists): both "relevance-like (reading (a))", reading "(a) throughout", although its answer to the rule question flips to correct. Llama-3.1-8B fails both gates (task-blind comprehension question K2 0/24; positive control +0.30 [+0.19, +0.40], below 1 nat), so both categories are "inconclusive (gate)"; descriptively its request component grows (+0.54 to +0.74). Results in `notes/tau_auth_results.md`, metrics `notes/tau_auth_metrics.json`, design in `notes/tau_auth_design.md` |
| Closed-loop agent with a world-model verifier (Option C): does a verifier shown the agent's task lower its success? | runs `optionc_smoke`, `optionc_step0_<model>`, `optionc_step0_analysis`, `optionc_qwen3-8b_obs`, `optionc_qwen3-8b_yn`, `optionc_analysis` (code on branch `optionc`) | Done (exploratory; design frozen before anything ran at tag `optionc-design-frozen-2026-11-22`, commit `54cb71e`). Step-0 check: Qwen3-8B 'question-specific (supports the task-success reading)' (yes/no: the task-aware verifier accepts 82 points fewer of the agent's working first moves than the blind one, [−88, −75]; observation readout: no gap in any game, competent), so the pre-declared reallocation was exercised before any census episode (commit `2a86683`): the observation readout became the primary run (no verifier, blind and aware at three seeds, swapped at one) and yes/no a one-seed secondary. Primary (observation readout, accepted-only rule): success 'no reliable difference', aware minus blind −1.2 points [−3.0, 0.0] (p 0.23; only 3 of 134 games differ, 84% of seed pairs play identical episodes); both key secondaries and invalid actions 'no reliable difference'; obstruction check clear. Secondary (yes/no, one seed): success 'no reliable difference', −4.5 points [−11.2, +2.2]; invalid actions 'harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)', +1.12 per episode [+0.33, +1.96], with the same label for invalid actions in the first 10 steps and override harm minus help, through more abstentions and fewer rescues rather than more overrides (worse than the task-blind verifier, still better than none). Matched states: the real task raises acceptance of failing goal-object proposals under both readouts (+27.5 points observation, +3.3 yes/no, against blind). Argmax rule (yes/no readout; when the verifier rejects the proposal, the highest-scored sampled alternative is executed whatever its sign; jobs 7253142–7253145, submitted after the Tier 1 analysis; analysis job 7295370 at `01cad81`): success 'no reliable difference', −0.7 points [−9.0, +7.5], with 29 of 134 games differing; invalid actions per episode 'no reliable difference', −0.07 [−0.85, +0.71]; invalid actions in the first 10 steps 'harm, attribution not identifiable in the loop (the swapped verifier obstructs the real task)', +0.23 [+0.10, +0.37]. The design's prediction for this rule (a loss of 2 to 10 points) is not supported. The Llama loop was not run. *(revised 2026-11-25)* The agent's prompt included ALFWorld's admissible-action list, so these nulls mean "not detected in this setup". The primary rests on 3 discordant games (smallest attainable p 0.25), and the design could detect only success losses of roughly 6 to 11 points, so smaller costs are not excluded. The first runs used 12.3 GPU-hours. Results in `notes/optionc_results.md`. *(The verifier uses Qwen3's thinking-off format with the empty think block, as Stage 1a did; see the results note, section 10.3.)* |
| ScienceWorld | not in this repository | Paused. A pre-specified feasibility probe found almost no determined failures on the task's own object (99.6% of the few found were one generic parser reply), so it cannot replicate the main test; a set of control items on unrelated objects is available |

Analyses need no GPU: run `python -m wf.runs <run>_analysis` once every scoring shard reports `shard complete`. Run output is not tracked by git, so a fresh clone has none; the data live on ICE (see [notes/reproducing.md](notes/reproducing.md)). Real runs refuse mock files and work saved at another commit.

Two cautions for reading the numbers. The two-candidate score s = σ(L), sometimes logged as "P(success)" or "Δp", is not the model's probability that the action succeeds; it compares one success wording with "Nothing happens.". And the game-bootstrap 95% intervals are slightly narrow: in simulation they covered the true value 93–95% of the time ([notes/testing.md](notes/testing.md)). The preregistered verdicts also rest on the permutation test, which was calibrated.

## Documents in this repository

- [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md): the implementation spec for the pilots.
- [notes/deviations.md](notes/deviations.md): the dated decision log. It records every change since the study was filed, and each exploratory design, logged before its data. It also holds the parser audits and what the write-up must mention. Where it and the protocol disagree, it is current.
- [notes/rq2_results.md](notes/rq2_results.md) and [notes/rq1_results.md](notes/rq1_results.md): confirmatory results, with metrics in [notes/rq2_confirmatory_metrics.json](notes/rq2_confirmatory_metrics.json), [notes/rq2_shared_metrics.json](notes/rq2_shared_metrics.json) (shared-spot sensitivity) and [notes/rq1_v2_metrics.json](notes/rq1_v2_metrics.json) (RQ1 at the final parser; it supersedes `notes/rq1_metrics.json`).
- [notes/rq3_results.md](notes/rq3_results.md): the agent experiment (RQ3), with metrics in [notes/rq3_metrics.json](notes/rq3_metrics.json).
- [notes/cross_model_summary.md](notes/cross_model_summary.md): every model scored so far, read together (eleven in ALFWorld, five in the decision simulation), with what still rests on the two reference models.
- Other environments and studies, each with its design note and a metrics file next to the results: [notes/families_results.md](notes/families_results.md) (model families), [notes/textcraft_results.md](notes/textcraft_results.md) (TextCraft, with links to the exploratory follow-up notes), [notes/tau_results.md](notes/tau_results.md) (τ-bench retail, with the exploratory follow-up [notes/tau_dx.md](notes/tau_dx.md) the authorization experiment [notes/tau_auth_results.md](notes/tau_auth_results.md) and the model-families extension [notes/tau_families_results.md](notes/tau_families_results.md)) and [notes/decisionsim_results.md](notes/decisionsim_results.md) (decision simulation, Stage 1a, with the exploratory follow-up [notes/decisionsim_dx.md](notes/decisionsim_dx.md)), and [notes/optionc_results.md](notes/optionc_results.md) (the closed-loop verifier study, Option C, with its design note and review).
- [notes/exploratory_results.md](notes/exploratory_results.md): the exploratory follow-ups, with metrics in `notes/rq1x_v2_metrics.json`, `notes/thinking_v2_metrics.json`, `notes/rq1d_metrics.json`, `notes/rq1v_v2_metrics.json`, `notes/mech_metrics.json`, `notes/size_v2_metrics.json` and `notes/audit_metrics.json`. The `_v2` files are the 2026-11-20 reruns and supersede the files of the same name without it, which are kept.
- [notes/testing.md](notes/testing.md): test suite, mutation check, item verification and calibration of the statistics.
- [notes/reproducing.md](notes/reproducing.md): software and model versions, where the data are, which run, commit and GPU produced each result, how to rerun everything, and licences.
- [notes/extensions.md](notes/extensions.md): the pre-pilot list of extensions, archived.

The filed proposal and preregistration are not in this repository.

## How the stages fit together

1. **Tier 0** checks that the machinery works. It answers no research question.
2. **Pilot A** builds items from 50 train games, then scores them with Qwen3-8B and Llama-3.1-8B-Instruct (few-shot) and the fine-tuned Word2World model. It reports A, ΔL, Δp and the first-token measures, applies the scoring stop rule, and estimates the items needed for power. In the end the confirmatory run scored every eligible item (4,666, against a preregistered minimum of 1,000), so that estimate did not limit the study. `PILOT_RULE_GOAL_CHANNEL` is advisory.
3. **Pilot B** runs in order: `pilotB_headroom` (does perfect foresight help?), `pilotB_consumption` (does the agent use predictions?), `pilotB_eligibility` (sets the error-injection rates) and `pilotB_retention_minirun` (chooses the retained format). `pilotB_consumption` is the gate: perfect predictions must change the agent's action on at least 10% of steps, the proposal's own pilot check. `PILOT_RULE_HEADROOM` is advisory since 2026-11-17 and never stops the job; `PILOT_RULE_RETAINED_FORMAT` applies after the mini-run.
4. **`runs/PILOT_REPORT.md`** collects the results, using `python -m wf.tools.assemble_pilot_report`.
5. **Tier 3** was an optional stage, allowed only if Pilots A and B finished cleanly. It was never run (no `t3_*` run exists), and nothing depends on it.
6. **The confirmatory study** ran on `valid_unseen`, with its design fixed by the pilot rules: item collection, item verification, RQ2, RQ1 and RQ3, then the exploratory runs. None of them reads pilot output.

Pilot A and Pilot B refuse to run without `runs/FILED.txt`, and that marker must not be a mock filing.

## Run locally

No GPU, models or ALFWorld server needed.

```bash
pip install -e ".[dev]"
pytest -q
bash pace/run_mock_pipeline.sh     # Tier 0 and the pilots, in order, on the simulator
python -m wf.runs t0_smoke --mock  # one run
python scripts/mutation_check.py   # breaks the code on purpose; the tests must notice
```

Mock runs use `wf/env/sim.py`, which reproduces the AgentGym server's behaviour behind the real client. They write only to `runs_mock/` or a temp folder and refuse any other `RUNS_DIR`. Mock records are marked, real runs refuse them, and cluster jobs refuse to start in mock mode.

A passing mock run shows the logic holds together. It says nothing about the real server or real models, and only ICE can show that. `run_mock_pipeline.sh` covers only Tier 0 and the pilots; the confirmatory and exploratory runners are exercised on mock data by the test suite instead.

## Run on PACE ICE

One-time setup is in [pace/SETUP_ICE.md](pace/SETUP_ICE.md): copy `pace/pace.env.example` to `pace/pace.env`, fill in your account and paths, then install Word2World, the environments and the models. `pace/pace.env` holds your Hugging Face token, so never commit it.

Run these in order from the repo root on ICE:

```bash
bash pace/submit.sh t0_all.sbatch         # Tier 0
bash pace/submit.sh verify_models.sbatch  # known-answer checks and the human review file
bash pace/submit.sh pilotA.sbatch
bash pace/submit.sh pilotB.sbatch
```

`submit.sh` adds the account, QOS, partition and GPU request from `pace.env`, so use it for GPU jobs. Leave `PACE_GRES` empty (or `gpu:1`): a typed request such as `gpu:A100:1` overrides each script's own "any of L40S, A40 or A100" setting. `verify_env.sbatch` is the exception, because it runs on CPU:

```bash
sbatch pace/verify_env.sbatch                                # all real-server checks
sbatch --export=ALL,VERIFY_ONLY=games pace/verify_env.sbatch # just the real-game check
```

Tier 0 and Pilot A stop vLLM before HF scoring, because the two cannot share the GPU. Pilot B uses only vLLM.

`pilotB_headroom` no longer stops on the headroom rule; it stops (exit 1) only if keeping predictions in the history is materially worse than dropping them. `pace/pilotB_recheck.sbatch` is no longer used.

### Confirmatory and exploratory runs

These have their own scripts; their order, settings and costs are in [notes/reproducing.md](notes/reproducing.md). In short: `rq12_collect.sbatch` collects the items (one job per shard, `RQ12_RUN=rq12_transitions_s0` and so on); `verify_items.sbatch` replays every item on CPU; `rq2_score.sbatch` runs whichever scoring run `RQ2_RUN` names (RQ2, RQ1 and every exploratory scoring run), one game shard per job (`WF_SHARD`, `WF_N_SHARDS`); and `rq3.sbatch` runs RQ3. Qwen3-32B needs an A100 with 80 GB.

### Getting code onto ICE

To get new code onto ICE, clone or pull from GitHub (`main`). Jobs started on 2026-11-17 used checkouts made from git bundles under `/storage/ice1/2/1/gmat3/` (`text-wm-fix4`, `text-wm-fix5`); their commits are on `main`. Each checkout needs its own `pace/pace.env` naming its own `PROJECT_ROOT`.

### After a timeout or crash

Resubmit the same job with `WF_RESUME=1`:

```bash
WF_RESUME=1 bash pace/submit.sh pilotA.sbatch
```

- Work is saved as it finishes: one episode, score, generation or built game at a time. You can read partial results while a job runs.
- A resume skips runs that finished at the current commit with unchanged inputs. A rule stop is never skipped.
- A resume refuses work saved by a different commit, uncommitted code, changed settings or changed input files, so one result never mixes versions. Resubmitting without `WF_RESUME` restarts the whole job and deletes each run's saved work.
- Only `WF_RESUME=1` resumes. Jobs use `--no-requeue`, because a requeue would restart them without it.
- `t0_replay`, `t0_parser` and `t0_expert_labels` take minutes, so they restart instead.
- The verification tools keep partial results but do not resume. Resubmitting one overwrites its output. `export_review` saves items to `runs/review_items.jsonl` as it goes and writes `HUMAN_REVIEW.md` at the end.

## Where results go

Each run writes to `runs/<run_id>/`. A run split across jobs by game writes one folder per part, `runs/<run_id>/shard_<i>_of_<n>/` (for example `runs/rq2_score/shard_00_of_04/`), and its analysis reads them all. Each folder holds:

| File | Contents |
| --- | --- |
| `STATUS.md` | PASS or FAIL against a target, or REPORTED where a run has no target or outputs a rule. Its first line says `BELOW TARGET` or `STOP` when it applies. Missing means the run has not finished or crashed. |
| `REPORT.md`, `metrics.json` | Results, where the run produces them |
| `git_sha`, `config.json`, `inputs.json` | The commit, settings, and hashes of the files read |
| `episodes.jsonl`, `transitions.jsonl`, `items.jsonl`, `scoring.jsonl`, `generations.jsonl`, `observations.jsonl` | Saved work, one record per line |
| `games_done.jsonl` | Games fully saved. A game counts as finished only once it is listed here. |

Verification output sits directly in `runs/` and is not tracked by git:

| File | From |
| --- | --- |
| `verify_real_env.json`, `verify_games.json` | Real-server checks |
| `verify_scoring.json`, `verify_agent.json` | Known-answer model checks |
| `HUMAN_REVIEW.md`, `review_items.jsonl` | Real items and episodes for a person to read |

Logs are in `logs/`, named by job id.

## Repo map

| Path | Purpose |
| --- | --- |
| `wf/runs.py` | One function per run, plus gates and resume. Start reading here: `python -m wf.runs <run_id>` |
| `wf/rq2.py` | Confirmatory RQ2 scoring and analysis, with the object-mention 2×2 |
| `wf/rq1.py` | RQ1, H1 and the secondary RQ2 conditions |
| `wf/rq3.py` | RQ3 agent experiment (statistics in `wf/analysis/rq3_stats.py`) |
| `wf/rq1x.py`, `wf/rq1d.py`, `wf/rq1v.py` | Exploratory: object-free goals and working actions; dissociation; counterfactual task and yes/no question |
| `wf/mech.py`, `wf/sizes.py`, `wf/thinking.py` | Exploratory: mechanism diagnostic; Qwen3 size series; thinking mode |
| `wf/families.py`, `wf/decisionsim.py` | Exploratory: model families; offline decision simulation, Stage 1a |
| `wf/textcraft/` | TextCraft replication: text-only checker, item build and verification, prompts, scoring, analysis; `dx.py` holds the exploratory diagnostic runs |
| `wf/tau/` | τ-bench retail replication: port of the retail tools (`retail.py`, upstream files in `vendor/`), checker, item builder, prompts, scoring and analysis (`pipeline.py`) |
| `wf/util.py` | Run folders, saving and resume helpers, mock guards |
| `wf/constants.py` | Preregistered constants and engineering thresholds |
| `wf/config/` | Settings for each run |
| `wf/env/` | Server client (`server.py`), simulator (`sim.py`), replay (`replay.py`), scene scan (`scene.py`), expert labels (`expert.py`) |
| `wf/agent/` | ReAct loop (`react.py`) and foresight conditions (`foresight.py`) |
| `wf/data/` | Goal parsing (`goals.py`), transitions and alternatives (`transitions.py`), success twins (`candidates.py`) |
| `wf/scoring/` | Prompts (`prompts.py`), log-odds scoring (`logodds.py`), observation parser (`parser.py`) |
| `wf/corruption/` | Which steps get a wrong prediction (`schedule.py`) |
| `wf/analysis/` | Statistics (`stats.py`, `rq3_stats.py`) and reports (`report.py`) |
| `wf/tools/` | Real-server and model verifiers, full item verification (`verify_items.py`), the blind parser audit (`parser_audit.py`), review export, report assembly, environment record |
| `pace/` | Job scripts and ICE setup |
| `tests/` | Unit, regression, adversarial and crash-and-resume tests |
| `scripts/` | Mutation check, statistics calibration, few-shot set drawing, RQ1 prompt check, vLLM smoke test |
| `notes/` | Results, decision log, metrics copies, testing and reproduction notes |

## Things worth knowing

- **What "ALFWorld" means here.** ALFWorld as served by the AgentGym server shipped with Word2World (`e8dc240`), with Word2World's renamed actions (`put` is `move`; there is a `help` action) and data download, and alfworld 0.3.3. Only solvable games are used, 134 of the 195 in AgentGym's `valid_unseen` list, and the environment ends every episode at 50 steps. Every inadmissible command gets "Nothing happens.". Details are in [notes/rq2_results.md](notes/rq2_results.md), "Methods".
- **Some receptacles share a spot.** From `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. Items affected by this carry `shared_location` and are left out of the main tests: Pilot A's, and RQ1's and RQ2's on the confirmatory data. Pilot A also scored them for a sensitivity check. On the confirmatory data no item was excluded by this rule alone, so the result with them is the same ([notes/rq2_results.md](notes/rq2_results.md), "Shared-spot sensitivity"). Read that section of [notes/deviations.md](notes/deviations.md) before analysing or writing up results.
- **Solvable games only.** The split sizes are 2,179 train, 140 `valid_seen` and 134 `valid_unseen`. AgentGym's mappings also list unsolvable games that crash on reset, and `wf/env/server.py` filters them out. See protocol §5.1.
- **The server keeps one live game per id.** Finished games refuse further steps, and ALFWorld ends every episode at 50 steps. Counterfactual replays run on a separate sandbox id, so they never touch the live episode.
- **Steps are slow by design.** Every step replays the game from a reset, in every condition, which costs about 1.7 s before any model time.
- **The real server leaks memory on every reset.** It reached about 20 GB after 4,281 resets over 30 games, or about 4.5 MB per reset, without getting slower. Jobs have 64 GB; the first Pilot A job (50 games plus scoring) peaked near that limit without failing. The later long jobs restarted the server instead: item collection every 6 games, RQ3 every 40 episodes (`max_episodes_per_job` in `wf/config/rq3.yaml`), and item verification ran in parts of about 1,300 items.
- **Expert labels are unavailable on the real server.** `on_expert_path` stays null rather than being approximated.

## Model families, wave A (done, 2026-11-20)

*(Updated 2026-11-23: this section was headed "in progress"; wave A is scored and analysed, with results in [notes/families_results.md](notes/families_results.md).)*

Gemma 2 9B, Mistral 7B v0.3 and Llama 3.2 1B and 3B are scored with the size series' prompts, one run per model (`family_score_<tag>`, `pace/family_score.sbatch`), and `families_analysis` sets them beside Qwen3-8B and Llama-3.1-8B. A tokenization gate checks every item through the scorer's own encoding before any score, and a small exploratory run (`family_placement_mistral-7b-v0.3`, `pace/family_placement.sbatch`) checks whether Mistral's goal effect depends on where its template puts the system text. The yes/no format is the primary cross-family comparison, for models that pass a competence gate fixed before data. Its one primary metric is the share of failures the goal moves toward Yes, per model and in seven planned paired comparisons (difference, and equivalence within 5 points), Holm-corrected as one family of fixed size. *(revised 2026-11-25)* Option C's step-0 check found Qwen3-8B partly reading this anchored yes/no question as asking about task success ("question-specific (supports the task-success reading)"), so the share may include a task-progress judgement; the two-candidate effect, which does not use the question, is positive for every model. The goal-to-evidence ratio is secondary: across families it mostly reflects how far each model pushes its log-odds. Design, gate results and the three amendments made before any score was read: [notes/families_design.md](notes/families_design.md).
