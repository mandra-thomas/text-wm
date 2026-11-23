# Deviations after filing (filed 2026-11-05)

Changes made on `fix/audit-bugs` after the filing marker. No preregistered threshold was changed. Report these in the final writeup.

## Still open (updated 2026-11-20)

- **Closed 2026-11-20: shared-spot sensitivity on confirmatory data.** No item was excluded by the shared-spot rule alone, so the result with them is the confirmatory result (`rq2_results.md`, "Shared-spot sensitivity").
- **Closed 2026-11-20: logged goal effect on working actions.** Computed after the fix (`rq1x_v2_metrics.json`, `rq1v_v2_metrics.json`); reported in `exploratory_results.md`, sections 1 and 4.
- **Closed 2026-11-20: metrics at the final parser.** `rq1_v2_metrics.json` and `thinking_v2_metrics.json` were made with the final parser `3c25d40cc71529e2` and supersede `rq1_metrics.json` and `thinking_metrics.json` for written-prediction numbers.
- **Closed 2026-11-20: exclusion flow.** In `rq2_results.md`, "Exclusion flow" (`audit_metrics.json`).
- **Public data archive.** The run data (items, scores, generations, episodes) are on ICE and in the owner's local copy; a public archive is planned (`reproducing.md`).

## Decided or found 2026-11-22

- **Model families, wave B: review 1 and its fixes, before any wave B score** (branch `families-waveb`; [families_waveb_review1.md](families_waveb_review1.md); design note section 18). No wave B weight had been downloaded and no wave B model scored.
  - *A crash found before any run.* In transformers 5, `apply_chat_template(tokenize=True)` returns a BatchEncoding, which the scorer passed to `torch.cat` and to the model; every chat-format score would have stopped the job. The scorer now takes the ids out of it (transformers 4 unchanged), and a serve2-only test scores end to end through tiny random-weight models of the three wave B classes with the pinned tokenizers.
  - *B1's basis corrected.* Qwen3-4B is competent but its share toward Yes is 0.49 with a yes/no effect of 2.27 nats, so "every interpretable model shows it" holds for the log-odds effect, not the share. A share that is not supported now reads "effect carried by a minority of items (as Qwen3-4B)" when the yes/no effect's CI is above zero, and "no detectable effect" only when both are null. The prediction is unchanged.
  - *Route A (node-local weights) chosen,* with a committed manifest of every file's size and hash per model (from Hub metadata, no token), a disk-space check, a hash check of every downloaded file, a refusal of any folder on `/storage`, and the hashes saved in each shard.
  - *Gates added or tightened:* the environment check's GPU type pinned (A40 for Mistral 7B, L40S for Qwen3-8B); no wave B number written anywhere unless the environment check passed; GPU and job memory checked before loading; an eager-against-SDPA agreement check at the start of each Gemma 4 job; the CPU gate tool runs the check variants and exits 1 on any failure.
  - *Pre-registered additions:* a 300-item check of Qwen3.8 27B with `preserve_thinking=False` (its template otherwise writes empty think blocks into the examples), read by the one-fifth rule; B3's readout on the two-candidate item share if the base model fails only the yes/no coverage criterion; the run order (environment check, the four small models, Qwen3.8 with its check, Gemma 4 31B last).
  - *Wording:* generation pairs carry the Qwen3 size spread of the share (0.49 / 0.89 / 0.70 / 0.71 at 4B / 8B / 14B / 32B), and a "differs" is not read as a generation effect; B2 can fail only on a clear negative difference; the outcome table says "goal-congruent shift" and does not read the share as separating wishful prediction from task relevance.
  - *Gemma pair kept as designed.* Gemma-4-12B-it exists and fits the GPU tier, but uses the 31B template (an empty thought channel ends its generation prompt) and another architecture, so Gemma 4 E4B stays, with its size confound stated.

## Decided or found 2026-11-21

- **TextCraft follow-ups (exploratory).** These are the cross-environment comparison (`xenv-compare`) and the Llama diagnostics (`textcraft-llama-dx`); summary in `textcraft_results.md`, "Follow-up analyses". Every diagnostic GPU run had a dated plan committed before its data. The runs used new checkouts on the cluster and left the confirmatory runs untouched.
- **Clean TextCraft prompt adopted for later runs.** The world-model system text without the original agent's instructions becomes the default for all later TextCraft runs, once its tokenization and golden-prompt checks pass for every model. The confirmatory result stays reported under the prompt it was frozen with.
- **TextCraft round 2 planned (branch `textcraft-r2`).** It is a pre-registered test of the two-component account (a lift on intermediate crafts and a penalty on crafting the named goal that shrinks as inputs accumulate), run under the clean prompt.
  - Models: Gemma-2-9B, Mistral-7B and Llama-3.2-3B, with Qwen3-8B and Llama-3.1-8B rescored.
  - Arms that separate naming the object from completing the task: a parent goal, and a two-item goal line if it can be made faithful to the environment.
  - A padding arm separates timing from history content.
  - The design is frozen at a tag and reviewed before any score.
- **τ-bench smoke run recorded before the full run.** Format passes for both models, and the projection fits the time limit (largest shard 3.1 h of 8). Qwen's projected half-width exceeds the 0.5-nat equivalence margin, so a Qwen result that is not positive will read as inconclusive, not as no effect (`tau_design.md` on branch `taubench`). The items were rebuilt and verified at the commit that records this, and full scoring was submitted.
- **Decision simulation, fixed menus.** Branch `decisionsim`. Stage 1a adds a literal task-blind verifier (the task sentence deleted from the first observation; the prompt examples keep theirs), a no-verifier baseline (the agent's proposal fails at 5.8% of even-step states) and a simulator oracle. The gate for a closed-loop study is now evaluated against the task-blind verifier; the neutral version is reported as a secondary verdict that never decides. A confirming review is running.
- **Decision simulation, agent-generated candidates.** Branch `decisionsim-agent`, built, not run.
  - Candidates: at each of the 2,973 decisions of the RQ3 no-foresight episodes, rank 1 is the agent's actual action, and ranks 2 to 8 come from 20 samples at temperature 1.
  - Verifiers: none, task-aware, task-blind, oracle and swapped.
  - Primary measure: the failure rate of the chosen action, task-aware minus task-blind.
  - The agent's own greedy action falls in the RQ2 regime at only 61 states (2.1%, in 15 games), and under the strict RQ2 definition at 4.
  - The design is under review.
- **Follow-up designs revised again before any score.**
  - *Mechanism battery, review 3.* A mention component, computed from arms already scored, separates "task as evidence" from "salience plus normal reading of explicit claims". A contrast counts as consistent only when the prediction it confirms excludes zero. A negation placebo arm is added. A completing-versus-intermediate split of the aligned effect is added.
  - *Interpretability pilot, review 2.* See the 2026-11-20 entry; now in a confirming review.
- **Test suite.** TextCraft's recipe file is Mojang game data and is not committed. In a clone without it, TextCraft tests that error are now skipped with the rebuild instruction (`tests/conftest.py`); with the file present nothing is skipped.

## Decided or found 2026-11-20, later (replications and follow-up designs)

None of this changes a preregistered threshold or the RQ2 result.

- **TextCraft: frozen, scored, merged.** The design and analysis code were frozen before scoring at the annotated tag `textcraft-design-frozen-2026-11-20` (commit `5a3702a`). The tag message records the test, the item hashes, the design history and what was seen before the freeze: one format-only smoke run, with no aligned-against-swapped difference computed. Full scoring (51,200 scores) and the analysis ran at that commit. Result: confirmed for Qwen3-8B, a small negative effect for Llama-3.1-8B, so not a replication across both models (`textcraft_results.md`). The branch was merged into the main line after scoring. Two exploratory prompt variants (design section 13, D10) were scored afterwards. An exploratory investigation of the Llama result is in progress on branch `textcraft-llama-dx`.
- **Mistral placement check read.** Moving Mistral's system text to the first turn changes its goal effect by +0.30 nats [0.13, 0.48] on the yes/no question and +0.02 [−0.15, 0.20] on the two-candidate scale. Both are below the logged one-fifth threshold, so placement is reported as not moving the effect (`families_results.md`).
- **τ-bench (branch `taubench`).** Review 3 added a validity gate: do the prompts and labels concern only whether the backend executes the call, rather than whether the call is appropriate or allowed by policy? The labels are task-invariant: the simulated backend never sees the task, and every failure reply comes from a listed rule. The wording was changed before any score:
  - the yes/no question asks whether the backend executes the call without an error, instead of whether it "succeeds";
  - a null on failing calls is read together with the working "witness" call;
  - the reading of the prompt variant without the "backend does not see the task" sentence is narrowed;
  - the instruction no longer frames the task as an input to the prediction.

  Review 4 confirmed these with wording-only fixes. The full token check passed on the final prompts for both models (1,413 items each, no problems). The items were rebuilt and verified on the cluster at the final commit, and the rebuild reproduces the pinned hashes. A format and timing smoke run is in progress.
- **Follow-up designs revised before any score.** Each change below was made in response to an independent review, before any data.
  - **Mechanism battery (`mechanism2`).** Reviews 1 and 2.
    - Two contrasts were made into minimal pairs, because the desire lines had dropped the household task their comparators kept.
    - An affirmed-existence arm was added. It pairs with the negated one and differs only in "no" against "a".
    - Prediction cells that favoured one account were made symmetric.
  - **Interpretability pilot (`interp`).** Reviews 1 and 2.
    - The last layer is excluded from the ablation summary, because there the removal is complete by construction.
    - The test of a generic success feature removes the readout component before comparing directions.
    - Substantive thresholds stay as written; the pilot sets only sample sizes.
    - Step-0 quotas were added.
  - **Decision simulation (`decisionsim`).** Review 1.
    - State-level rates use even steps only, because of the collector's schedule (`exploratory_results.md`, section 7).
    - Scores are rounded to 1/64 nat before comparison.
    - The swapped task is scored, so the object-naming effect can be separated from the relevance penalty.
    - The gate for a closed-loop agent study tests a verifier that checks the agent's own proposal. It is no longer a maximum over a menu whose make-up guarantees a positive result.
- **Decision studies staged.** The question "does the prediction shift change decisions?" is studied offline in stages.
  - **Fixed candidate menus** at collected states measure whether the shift *can* change a choice. This adds a literal task-blind verifier, a no-verifier baseline and a simulator-oracle verifier.
  - **Candidates generated and ranked by an agent** measure how often it happens in a real decision distribution. This stage is primary for claims about practical impact.
  - **Branching the simulator where verifiers disagree** is designed only after the first two show enough disagreements.
  - A closed-loop comparison of agents is out of scope. No stage is designed to guarantee a harmful result.
- **Facts found while designing these studies** are in `exploratory_results.md`, section 7: the agent's natural failure rate, the step-0 concentration of the yes/no effect, the relevance penalty of a neutral comparison, the 1/64-nat steps of the scores, and the collector's odd-step schedule.

## Decided or found 2026-11-20 (documentation and rigor audit)

### Changes from the 2026-11-20 audit

On 2026-11-20 seven analyses ran on ICE at 28b4d30 (checkout `text-wm-fix20`, CPU only), and their metrics were copied into `notes/`:

- `audit_metrics.json` (run `audit_analysis`): the exclusion flow, the 788 seen-without-object items with and without instance-number items, the share of failures answered Yes in the size series, intervals for the headline ratios, first-token coverage and ties, items per game and near-duplicates, and twins lost to a winning plan;
- `rq2_shared_metrics.json` (run `rq2_shared_analysis`): the shared-spot sensitivity;
- `rq1_v2_metrics.json` (run `rq1_analysis_v2`): supersedes `rq1_metrics.json`, because it reads the written predictions with the final parser `3c25d40cc71529e2`; scores are unchanged;
- `rq1x_v2_metrics.json` and `rq1v_v2_metrics.json` (runs `rq1x_analysis_v2`, `rq1v_analysis_v2`): supersede `rq1x_metrics.json` and `rq1v_metrics.json`, because they compute the logged goal effect on working actions that the first analyses left empty; every other number is identical;
- `thinking_v2_metrics.json` (run `thinking_analysis_v2`): supersedes `thinking_metrics.json`, because it reads the answers with the final parser; the scores (L) are unchanged;
- `size_v2_metrics.json` (run `size_analysis_v2`): supersedes `size_metrics.json` after the NaN fix in the flips and the slope bootstrap; every number is identical.

The old files are kept, so that the numbers quoted before 2026-11-20 can still be traced. Each new file has a provenance block (analysis commit, scoring-run commits, input hashes, parser version).

Every number in the public notes that changed, old → new:

| Number | Old | New | File (source) |
| --- | --- | --- | --- |
| RQ2 mean s, Qwen aligned / swapped | 0.076 / 0.017 (no source) | 0.075 / 0.017 | `rq2_results.md` (`rq1_v2_metrics.json`, `rq2_secondary.mean_p_success`) |
| RQ2 mean s, Llama aligned / swapped | 0.407 / 0.243 (no source) | 0.408 / 0.244 | `rq2_results.md` (same) |
| RQ1 Qwen history-only A, CI lower bound | 0.167 | 0.168 | `rq1_results.md` (`rq1_v2_metrics.json`) |
| RQ1 Qwen history-only A, partial effects as failures | 0.185 | 0.186 | `rq1_results.md` |
| RQ1 Qwen full-state A (CI) | 0.285 (0.259 to 0.312) | 0.284 (0.258 to 0.312) | `rq1_results.md` |
| RQ1 Qwen full-state false-success rate | 31.5% | 31.4% | `rq1_results.md` |
| RQ1 Qwen full-state A, partial effects excluded | 0.269 | 0.268 | `rq1_results.md` |
| RQ1 Llama full-state A (CI) | 0.668 (0.640 to 0.696) | 0.666 (0.639 to 0.695) | `rq1_results.md` |
| RQ1 Llama full-state false-success rate | 69.1% | 68.9% | `rq1_results.md` |
| RQ1 Llama full-state A, partial excluded / as failures | 0.587 / 0.420 | 0.586 / 0.421 | `rq1_results.md` |
| RQ1 Qwen A for example sets 2 and 3 (SD over sets) | 0.31 and 0.33 (0.075) | 0.30 and 0.32 (0.066) | `rq1_results.md` |
| H1a, Qwen (CI) | −0.092 (−0.120 to −0.065) | −0.091 (−0.119 to −0.064); verdict unchanged | `rq1_results.md` |
| H1a, Llama (CI) | +0.099 (0.057 to 0.140) | +0.101 (0.058 to 0.141); verdict unchanged | `rq1_results.md` |
| Full state cuts the goal effect | "about 70%", no interval | 66% [59%, 73%] Qwen, 71% [64%, 78%] Llama (wording kept) | `rq1_results.md`, README (`audit_metrics.json`, `ratios`) |
| Ignore-goal note removes | 9% / none | 9% [7%, 12%] / −3% [−6%, 1%] | `rq1_results.md`, `exploratory_results.md`, README (`audit_metrics.json`) |
| Object-free goal keeps | 61% / 92% | 61% [54%, 68%] / 92% [83%, 103%] | `exploratory_results.md`, README (`audit_metrics.json`) |
| Object-free goal keeps, check-passed items, Llama | 84% (no source) | 85% [76%, 95%] (84.5%) | `exploratory_results.md` (`audit_metrics.json`) |
| Thinking mode removes | "a third" | 34% [16%, 52%] (keeps 66% [48%, 84%]) | `exploratory_results.md`, README (`audit_metrics.json`) |
| Thinking written shift | 4.7 [1.6, 8.3] (one-off re-parse) | 4.7 [1.5, 7.8] | `exploratory_results.md` (`thinking_v2_metrics.json`) |
| Goal effect on working actions, two-candidate | not computed (NaN) | Qwen 1.94 [1.34, 2.62], Llama 0.16 [0.01, 0.31] nats | `exploratory_results.md` (`rq1x_v2_metrics.json`) |
| Goal effect on working actions, yes/no | not computed (NaN) | Qwen 2.44 [1.43, 3.52], Llama 0.98 [0.73, 1.24] nats | `exploratory_results.md` (`rq1v_v2_metrics.json`) |
| 788 seen-without-object items, Qwen / Llama | 4.48 [3.57, 5.41] / 2.82 [2.47, 3.18] (post hoc) | 4.49 [3.60, 5.45] / 2.81 [2.46, 3.14] | `exploratory_results.md` (`audit_metrics.json`) |
| Mechanism, Qwen plain CI lower bound | 5.51 | 5.50 | `exploratory_results.md` (`mech_metrics.json`, double rounding) |
| Mechanism, Llama plain CI lower bound | 1.50 | 1.49 | same |
| Mechanism, state question main − twin, Qwen estimate | −0.34 | −0.33 | same |
| Mechanism, state question main − twin, Llama CI lower bound | 0.32 | 0.31 | same |
| Mechanism, Spearman, Qwen | −0.08 | −0.07 | same |
| Size slope, two-candidate | +0.52 | +0.51 | `exploratory_results.md` (`size_metrics.json`, double rounding) |
| Size, 1.7B score shift | 15.6 | 15.5 | same |
| Size, 1.7B AUC under the aligned task | 0.79 | 0.78 | same |
| Qwen3-32B on failing actions, aligned / swapped | 15% / 6% "of the time" (a mean probability) | answers Yes on 11.6% / 3.8% (mean σ 15.0% / 5.8%) | `exploratory_results.md`, README (`audit_metrics.json`, `size_yes_share`) |
| RQ3 false successes per episode | 2.4–2.5 | 2.3–2.5 | `rq3_results.md` (`rq3_metrics.json`) |

**No verdict or claim changes meaning.** The H1 verdicts are the same at the final parser, and the confirmatory RQ2 result is the same with the shared-spot items. Two readings gain a qualifier. The goal effect on working actions exists but is much smaller than on failures (about a third for Qwen and a tenth for Llama in nats). Among the 788 seen-without-object items, Qwen shows almost no effect on the 83 that differ only by instance number.


Five independent read-only reviews of the code and documents at e7ca63b, each covering one area: environment and items, scoring and prompts, statistics, staleness, and reproducibility. Nothing below changes a preregistered threshold or a reported confirmatory number. The entries record what the reviews found that was missing from this log.

**The 12-step cap on item histories (not preregistered).** The agent that produced the RQ1 and RQ2 histories stopped after 12 steps (`step_cap: 12`), so every item's history holds at most 11 actions. Origin: `wf/config/pilotA_transitions.yaml` has had `step_cap: 12` since the first commit (`e2c1f1f`, 2026-10-28). The confirmatory collection used the Pilot A collector ("Confirmatory RQ1 and RQ2 item collection" below), and commit `425542e` (2026-11-17) set the same value in `rq12_transitions*.yaml`. The collector's own default is also 12. `STEP_CAP = 30` (decision 8) governs the RQ3, Pilot B and Tier 0 agents, not item collection, and `PILOT_PROTOCOL.md` gives only the 30. The preregistration names no cap, and no written reason for 12 was found; it was a Pilot A collection setting carried over. Effect: the items cover the first 12 steps of each episode only, so later-episode states are not represented, and the history-length breakdowns stop at 11.

**Shared-spot rule: the "report both ways" commitment is not yet met.** The rule (2026-11-06, below) was applied to the confirmatory data: shared-spot items are left out of RQ2's population and RQ1's items. But they were never scored, so the promised result with them put back does not exist for any confirmatory run. A scoring run for these items has now been added; its result will be reported next to the main one in `rq2_results.md`, whichever way it falls. **Update, same day:** the analysis (`rq2_shared_metrics.json`) found that no item is excluded by this rule alone. Of the 6,528 determined failures with a twin and a swap, one is at a shared spot, and its history acts on a goal object, so it is outside the population for that reason too. The population with shared-spot items put back is the same 4,666 items, and ΔL is the same: 6.12 and 1.65 nats. The commitment is met.

**Logged readouts that came out empty.** The goal effect on working actions, logged for `rq1x` (2026-11-19, "With the failures already scored, this gives the goal effect on actions that work") and computed as `goal_effect_on_successes` in `rq1x` and `yes_no.goal_effect_successes` in `rq1v`, is NaN with 0 items in both metrics files. Cause: the shared pairing function always keeps only items whose true reply is "Nothing happens.", so successes were silently dropped; `require_agnostic=False` did not lift that filter. The notes report instead a post hoc subset of 37 matched successes, marked post hoc. The bug is fixed in code; rerunning the two analyses (CPU only) is pending, and both readouts will be reported as logged. **Update, same day:** rerun as `rq1x_analysis_v2` and `rq1v_analysis_v2`. On the 292 determined successes, the two-candidate goal effect is 1.94 nats [1.34, 2.62] for Qwen and 0.16 [0.01, 0.31] for Llama; on the yes/no question it is 2.44 [1.43, 3.52] and 0.98 [0.73, 1.24]. Reported in `exploratory_results.md`, sections 1 and 4. Closed.

**GPU types.** Three corrections to earlier entries.

- The 2026-11-17 entry "GPU type for confirmatory jobs" said scores could differ across GPU types "only by floating-point rounding". That was an expectation; the measurement a day later ("Scores across GPU types") found an SD of 0.55 nats (Qwen) and 0.17 (Llama) in an item's aligned-minus-swapped L. The later entry is the one that holds.
- "The aligned and swapped versions of an item are always scored in the same job on the same GPU" holds for an uninterrupted job. A resume skips work by (model, item, condition), so if a job was cut off between an item's two conditions, the second was scored by the resume job, possibly on another GPU type. How often this happened is not recorded.
- Some exploratory contrasts pair a score from `rq1_score` with one from `rq1x_score` (for example the named aligned task against a described task), which ran as different jobs on possibly different GPUs. This adds noise, not bias.

**Model revisions.** No run passed a `revision=` when loading a model, so each used whatever the Hugging Face cache held. Each model has exactly one snapshot in the ICE caches, so these are the revisions every run used, recorded after the fact:

| Model | Snapshot |
| --- | --- |
| Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B | 8d4f069c4bf5a739763664897b26ca17c161f0da |
| X1AOX1A/WorldModel-Alfworld-Llama3.1-8B | 8f089d2126f0406a296747041ebf33b902837cd3 |
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 |
| Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 |

The code now pins these revisions (a `REVISIONS` map, added 2026-11-20), so a change on the Hub (including to a chat template) cannot silently change a prompt. Software versions are in `reproducing.md`.

**Parser change counts.** Re-reading the 68,340 `rq1_score` predictions on 2026-11-20 confirmed the counts in the fifth and sixth audit entries below, and that both are against the version the analysis used (`8639849027611cd2`): 93 change side at the fifth version (`e118530da85de6a0`) and 110 at the final one (`3c25d40cc71529e2`): 70 unreadable become failures, 6 unreadable become successes, and 34 move between success and failure.

**Documentation.** The reviews also found stale or missing text in the public notes. Fixed on 2026-11-20: a methods section in `rq2_results.md` (what "ALFWorld" means here, how histories, alternatives, twins and swaps were made, the determination and shared-spot rules); the parser status in `rq1_results.md`; definitions of yes-share and of the two flip measures in `exploratory_results.md`; the item-verification results in `testing.md`; a new `reproducing.md` (versions, data, which run and commit produced each result, licences); and the README. Numbers that needed reruns (stale metrics, rounding, ratio intervals) were updated after the reruns the same day; see "Changes from the 2026-11-20 audit" above.

**Twins lost to a winning plan (limitation).** For tasks that only require one goal object at the target, the setup plan for `take <goal> from <target>` wins the game, so the action never runs and has no twin. Counted from the recorded fields (`audit_metrics.json`, `twin_plan_wins`): 164 such failures in 22 games, of which 127 would otherwise be in RQ2's population. RQ2 therefore contains no take of the goal object from the task's own target receptacle. The count is a signature match, not a replayed confirmation.

**Instance-number items.** Of the 788 items whose receptacle was seen without the object, 83 were seen holding another instance of the same type (`audit_metrics.json`, `seen_without_object`). The effect without them is 4.97 nats for Qwen and 2.90 for Llama; on the 83 alone, 0.46 [−1.35, 1.91] and 2.09. `docs/milestone_checkin.tex`, cited in two entries below, is a course document that is not in the public repository.

## Closed 2026-11-20: items open after the pilots

Each item is kept as written, with what became of it first.

- **Closed, partly done.** The every-step generator for contents, open-state and holding failures was built for the confirmatory collection ("Confirmatory RQ1 and RQ2 item collection" below), and the full-state condition for the few-shot models was built and scored in `rq1_score` (`rq1_results.md`). H1 was not rerun on a subset of 150 to 200 such failures, and no matched false-success against false-failure comparison was built; `rq1_results.md` reports the false-success rate by failure reason instead. *As written:* **Decision 6, remaining parts.** Agreed, not built: a generator quota of 150 to 200 failures decided by observed contents or holding state, H1 run on those items (with the pooled number also reported, and the change of carrying items disclosed), and false-success versus false-failure comparisons matched on the deciding fact. H1 also needs the full-state condition for the few-shot models, which the code does not have yet. None of this blocks the pilots.
- **Closed, unchanged.** No loop guard was added; RQ3 ran with the same greedy agent, and its repeated actions are reported in `rq3_results.md`. *As written:* **Agent stuck loops (observed 2026-11-17, nothing changed).** With greedy decoding, Qwen3-8B sometimes repeats an action that does nothing ("go to drawer 1" while already there) until the step limit. A replay of `train/791` (trace job 7136513) showed the prompts are built as intended: the oracle prediction for the repeated action correctly said "Nothing happens", and the agent took it anyway. In the first 85 Pilot B headroom episodes, loops of 5 or more steps appear in 4 of 22 episodes with no prediction, 4 of 21 with oracle predictions kept in the history, 0 of 21 with oracle predictions dropped, and 0 of 21 with the validity verdict only. Success is 10 of 21 in both retention arms. A loop guard or sampling would change the agent, so that is a team decision, not a fix.
- **Closed.** Llama stayed co-primary as decided. Its written-prediction shift is in `rq2_results.md`. The parser audits it waited for ended on 2026-11-19 without the parser passing its rule ("Sixth parser audit" below; parser status in `rq1_results.md`). *As written:* **Llama scoring stop rule fired (Pilot A, 2026-11-17).** For Llama-3.1-8B-Instruct the sign of L agrees with its written prediction on 741 of 1,138 determined failures (65%, below the 90% line of decision 13); Qwen3-8B agrees on 91%. Llama's ΔL is 1.36 nats (CI 1.08 to 1.68, p = 0.001). Its first-token mass is lower (median 0.86 against >0.99), and some of its failure sentences also start with "You", so ΔL_first is a weaker check for Llama. **Decided 2026-11-17:** Llama stays co-primary in the preregistered analysis, with the failed check disclosed as a measurement issue. Its written predictions, re-read with the fixed parser, also shift toward success under the aligned goal (+17.5 points, CI 12.0 to 23.6), so they are reported as the behavioural check on its ΔL; that reading waits for the parser audit.
- **Closed.** The parser was fixed and audited in four more rounds (third to sixth, below). It was never frozen as passing: the audits ended on 2026-11-19 with the no-opposite rule unmet. *As written:* **Parser drops free-form failure predictions (Pilot A, 2026-11-17).** Written predictions such as "You can't take the mug 1 from cabinet 3 because it is empty." or "There is no tomato 1 in the environment." are classed `unparseable` and left out of A. That removes failure predictions selectively: 364 of Llama's 475 unparseable replies and 99 of Qwen3's 151 are worded as failures. Counting them as failures (a rough regex, not the fix) moves Llama's A from 0.99 to about 0.87 and Qwen3's from 0.22 to about 0.20; Word2World has none. Fix before the confirmatory RQ1 run and check it against the 100-item parser hand check. **Fixed in code, awaiting the audit:** model text now goes through `parse_prediction` (environment observations keep `parse_observation`). Failure wording (can't, cannot, there is no, you don't see/have, you have no, already holding, "you try to … but") is `noop_paraphrase` and counts as a failure; success described in other words ("You place …", "You wash …") is `success_paraphrase` and counts as a success; anything else stays `unparseable`. `pilotA_reparse` recomputes every Ŷ_gen measure from Pilot A's saved text, before and after, without running a model. The blind audit (`python -m wf.tools.parser_audit`) samples 100 few-shot predictions, over-sampling the hard cases, and hides the parser's label and the true outcome. It passes if the parser matches the reader on at least 95% of rows the reader labels S or F (the parser's "?" counts as a miss) and no failure-worded reply is read as a success. These criteria were fixed before the audit was read. The parser is frozen once it passes, before any confirmatory data are scored.
- **Closed.** Decided as cap and yoke ("RQ3 error matching" below); the shortfall that remained is reported in `rq3_results.md`. *As written:* **Injected errors per episode are not matched in the retained cells (Pilot B mini-run, 2026-11-17).** The proposal matches the two error types on errors per episode. Calibration (`pilotB_eligibility`) set one rate per arm from oracle episodes, and the discarded cells came out close (2.9 false successes, 3.2 false failures per episode), but the retained cells did not (4.5 against 2.1). Retained false-success episodes offer more eligible steps, most likely because the agent repeats failing actions. **Decided 2026-11-17:** cap and yoke, see "Decided 2026-11-17, after the pilots".
- **Closed.** Not triggered; RQ3 ran on ALFWorld. *As written:* **ScienceWorld fallback.** No longer triggered: `PILOT_RULE_HEADROOM` is advisory (see the headroom decision), and RQ3 stays on ALFWorld.

## Decided 2026-11-18, after the confirmatory RQ2 result

Everything in this section was decided after the confirmatory RQ2 scores had been seen (notes/rq2_results.md), so none of it is preregistered. The RQ1 measure, H1's predictions and the third audit's rules were fixed earlier (2026-11-17) and are unchanged.

**Example sets 2 and 3 frozen.** Set 3 is approved as drafted. Its first example's history holds a malformed command ("take desklamp from desk 1", no number), which is real server output and is left in. Set 2 is redrawn, because both of its failures were takes and neither success was, so the verb alone predicted the outcome. The redraw keeps set 3 and set 1's games out and uses its own seed ("20260924|set2-redraw"). One of its two successes must be a take, chosen by the seed. Histories may hold no malformed command and no failed walk ("go to" answered "Nothing happens."). A success may not undo the step before it. The first redraw had a failed walk in a history, which is why that rule was added; the second draw is the one used. Both sets are in `wf/scoring/prompts.py` and `notes/fewshot_sets_frozen.json` (with the replaced draft), drawn by `scripts/draw_fewshot_sets.py`.

**Third parser audit: failed against the blind rater; parser fixed.** 158 rows from the confirmatory RQ2 predictions, none overlapping the first two samples. The blind rater (another model instance, given only the rows and the category definitions) and the parser agreed on 92.9% of decided rows on no effect against effect, below the 95% bar, with 3 rows read in opposite directions. The pass rule is not met, so the RQ2 written-prediction shift stays provisional. The misses were three patterns, each a parser error: "You are carrying the cd 2." or "You are holding the knife 1." after a take, which says the take worked (8 rows, previously unparseable); a tried action followed by a reason it cannot happen ("The egg 1 is not dirty.", "already at room temperature", 2 rows, previously process only); and "The action '…' is not valid." (1 row, previously a different effect). An earlier test had recorded "You are holding the knife 1." as unparseable, following audit 2's provisional "?" label; under the final definitions it is a stated success, and the test is changed. The fix changes 18 of 25,270 real readings (3 of 6,606 in Pilot A and 15 of 18,664 in the confirmatory RQ2 run), and each one was read. As the rules require, the changed parser gets a fresh (fourth) sample, drawn from the new run's predictions below. The blind rater and the user both label that sample; the third sample is not labelled further, because the audit tool refuses to score a sample against a changed parser.

**Parser, fifth version, and a fifth blind sample (2026-11-19).** The fourth audit's misses are fixed as patterns, with a regression test for each: "already in/on" for a move, "You still hold …", "You are carrying the X." as the whole reply to a move, "The desklamp 1 is on." for `use`, "not heatable/coolable/cleanable", the action's own result denied ("The cup 1 is not hot." for a heat), "still closed". The thinking-mode replies add refusals the parser had never seen ("The pan is not found.", "… is not present here.", "You are not at shelf 2.", "The fridge 1 does not contain …"). Two first drafts of these rules were too broad ("You are carrying the egg 1. The egg 1 is now in the fridge 1." became no effect, and "not cool" was read as a refusal for any action); both were narrowed before the sample was drawn. Effect on real readings, each changed reading read: 93 of the 68,340 `rq1_score` predictions change side (0.14%), 1 of the 18,664 confirmatory RQ2 predictions, and 218 of the 5,000 thinking-mode answers (4.4%, nearly all refusals previously read as effects). A fresh sample of 155 rows (version `e118530da85de6a0`) was drawn from the `rq1_score` predictions excluding every text in the earlier samples and the development texts. Only 4 unparseable rows remain to draw. Pass rule unchanged: at least 95% agreement on no effect against effect on rows the blind rater decides, and no row in the opposite direction.

**Fifth parser audit: failed on the no-opposite rule; parser fixed; sixth sample (2026-11-19).** The blind rater (a fresh model instance, given only the rows and the category definitions) decided all 154 rows. Agreement on no effect against effect was 95.5% (147 of 154), above the 95% bar, but 3 rows were read in the opposite direction, so the audit fails as the rule is written. The three: "The action is invalid. You must have a tomato in your inventory to heat it." (read as a different effect), "… The pan 1 is still normal." after a cool (process only), and "The egg 1 is not clean." for a heat (different effect; the rater read a failed precondition). The four rows the parser left unparseable (counted as misses) were "You drop the potato 2.", "You are not carrying anything. The book 1 is not on desk 2.", "You have moved the pencil 1 to the desk 1." and "cabinet 5 is closed." for a close. Fixed: invalid-action and "you must have/hold" wording, "still normal/dirty/unchanged", and any state of the object denied for a heat, cool or clean ("is not hot/cool/clean"). Every changed real reading was read: 110 of 68,340 `rq1_score` predictions change side, 3 of 18,664 confirmatory RQ2 predictions, 219 of 5,000 thinking-mode answers. A sixth sample is drawn at version `3c25d40cc71529e2`, excluding the fifth sample's texts as well as the earlier ones. This is the second fix-and-resample round in one day; every round's result is reported.

**Sixth parser audit: failed on the no-opposite rule; no further rounds (2026-11-19).** Version `3c25d40cc71529e2`, 150 fresh rows; the blind rater (a fresh model instance) decided 148. Agreement on no effect against effect: 96.6% (143 of 148), above the 95% bar; 5 rows opposite, so the audit fails as written. The five: "You find no lettuce 1." and "You find no lettuce in the fridge 1." for a clean (rater no effect, parser different effect); "You are carrying a potato 1. The potato 1 is dirty. You can clean it with the sinkbasin 1." for a clean (rater no effect); "You are already at the fridge 1. The fridge 1 is open. In it, you see …" for a move and "You are facing the microwave 1. The microwave 1 is closed." for a take (rater a different effect, parser no effect; the rater's own notes say both could be no effect). **Decided:** no seventh round. Each round fixes the previous sample's misses and the next sample finds a new rare phrasing, and repeated fix-and-resample rounds on one day risk tuning the parser to the rater. The written-prediction results are reported as validated at 95–97% agreement on stratified hard cases, with the no-opposite rule unmet. Weighting each stratum's disagreement rate in the sixth sample by its share of the 68,340 `rq1_score` predictions estimates that about 0.9% of all predictions are read on the wrong side; the exact "Nothing happens." stratum (46% of predictions) is taken as error-free. Re-parsed with this version: RQ1's A is unchanged to two decimals (Qwen 0.19, Llama 0.77; H1 verdicts unchanged); thinking-mode written success rates become 18.7% aligned, 13.8% swapped and 6.2% neutral (were 21.0%, 15.5%, 9.8%), and the aligned-vs-swapped shift 4.7 points (CI 1.6 to 8.3; was 5.4).

**Exploratory validation runs: counterfactual aligned task and yes/no feasibility (decided 2026-11-19, after outside reviews of the draft, before any of these scores).** `wf/rq1v.py`, runs `rq1v_score` and `rq1v_analysis`, few-shot Qwen3-8B and Llama-3.1-8B, example set 1, history only. Not preregistered.

- **Counterfactual aligned task.** The aligned task is always the episode's real task, so the history fits it, and the swapped prompt may read as less coherent. A cheap check on existing scores found Llama's goal-role part (both objects named) at 0.10 nats at step 0 and 0.65 after, the pattern a coherence confound would produce. Here the real task's destination is replaced by another destination type present in the room (chosen by a hash of the item), on take, heat, cool and clean actions, which advance the task whatever its destination; move actions are left out. The swapped counterpart replaces the object in that line, and the extra-mention pair is rebuilt on these lines. Prediction: if the real task's fit with the history drives part of the effect, the counterfactual-task effect and its goal-role part are smaller than the real-task ones on the same items, most of all after step 0; if not, they match.
- **Yes/no feasibility.** L compares an action-specific success text with "Nothing happens.", so its level is not comparable across different actions, which the top-candidate analysis relies on. The same prompt ends instead with a fixed question ("will this action succeed, or will the environment reply "Nothing happens."? Answer Yes … No …") and the model scores "Yes" against "No". Scored on the 4,666 failures and the 292 determined successes under the aligned, swapped and neutral tasks. Reported: the goal effect on this scale, AUC, and the top-candidate rate (aligned against neutral) at the same states. The top-candidate result in the draft stands only if it holds here.

**Exploratory: mechanism diagnostic, state inference against judgement (decided 2026-11-19, after a meeting with Prof. Chao Zhang and two outside reviews of the plan, before any of these scores).** `wf/mech.py`, runs `mech_score` and `mech_analysis`, few-shot Qwen3-8B and Llama-3.1-8B, example set 1, history only, on RQ2's 4,666 primary items. Not preregistered. Two accounts are tested. **(a) State inference:** the model does not recover the facts that decide the failure, and falls back on the task. **(b) Judgement:** it recovers them, but judges the outcome by how it would solve the task.

The arms are:

- **Prediction (two-candidate L, aligned and swapped tasks).** Four versions:
  - plain;
  - with the deciding facts as the history establishes them (`history_facts`: location, held object, open and closed receptacles, last-seen contents), stated before "Next observation:";
  - with a fixed statement of the environment's rules;
  - with both.
- **Yes/no questions under the aligned, swapped and neutral tasks.** Each replaces "Next observation:" and is scored "Yes" against "No" from one forward pass.
  - The deciding state fact. The correct answer is always No, and a twin question whose correct answer is Yes guards against a model that always says No.
  - Whether every condition the action needs is met now (correct: No).
  - On a fixed 1,000-item subsample: the rule behind the failure, with a twin.
- **The agent's own preference.** The ReAct agent prompt is rebuilt from the item's history with each task line, asks for the action only, and scores "Action: <action>" against "Action: look". No admissible-action list is shown, because items do not store one.

**Readouts, written before scoring:**

1. **Do the stated facts close the gap?** The goal effect with facts, against plain. A cut of at least 75% favours (a); a cut below 25% favours (b).
2. **Does the effect remain where the state is known?** The plain goal effect on items whose state question and twin are both answered correctly under the neutral task, against those answered wrongly. If the known-state items keep at least 75% of the overall effect, that favours (b).
3. **Does the goal move the belief itself?** The goal effect (aligned minus swapped) on the state question's log-odds of Yes. A positive effect means the goal moves the belief toward the fact that would let the action work, which is state inference corrupted by the goal.
4. **Does the evaluator read its own policy?** The Spearman correlation between the shift in the agent's preference and the ΔL of each item, with a game-bootstrap interval, overall and within failure reason. An interval above 0.3 is read as the evaluator reading its own policy.
5. **Rule knowledge.** Accuracy of the rule question and its twin under the neutral task, and the goal's effect on it.

No forced step-by-step reasoning arm is included; free reasoning was covered by the thinking-mode run. Two post hoc looks at the confirmatory scores came before this design, and both are reported as post hoc:

- **Effect by history length, room size, and actions that do or do not advance the task.** Qwen's effect is largest at step 0 (11.0 nats) and much smaller later. After step 0 both models rise mildly with history length: Qwen 5.3, 5.3, 5.8 and 5.7 nats, Llama 1.5, 1.7, 1.7 and 1.8. Room-size bins are uneven, and the largest holds only 11 games. Within verbs, Qwen's effect is larger for actions that advance the task on cool, heat and move, but not on clean; Llama's only on heat. Here, advancing is the same as the task and the action sharing a word, so this split cannot rule out word overlap.
- **Regression on the fraction of receptacles still unobserved.** With history length, log room size and failure reason as controls, Qwen's coefficient is negative (−13.7, CI −19.7 to −6.1) and Llama's is near zero. The sign depends on the specification. With the absolute number of unobserved receptacles instead, Qwen's coefficient is positive (+0.24, CI 0.13 to 0.38, controlling for history), and so is Llama's (+0.04, CI 0.01 to 0.07). The measure is collinear with room size (r = 0.92) and history length. So these data cannot separate hidden-state uncertainty from room size or game differences, and no conclusion is drawn from them.

(The two descriptions above were corrected the same day, before any score of this run existed, after an independent check showed the first wording overstated both results.)

**Corrections to the design after an independent code check (2026-11-19, before any score of this run existed).** The check verified the answer key of every state question on all 4,666 items against the starting layout. It found three things in the rules material, now fixed:

- **Acting from the same spot.** The rules statement said the agent can act only on the receptacle it is at. ALFWorld's PDDL allows any receptacle at the same spot, and the data contain such successes. The statement now says so, and the "away" rule question asks about a receptacle "in another part of the room".
- **Putting into a closed receptacle.** The statement omitted that this fails. It now says so.
- **The move and heat/cool/clean twin questions.** Their "Yes" answers needed qualifying: the receptacle must be open or not openable and able to hold the object, and the object must be of a kind that can be heated, cooled or cleaned.

The analysis now also computes three readouts in the form logged above:

- the share of the effect removed, as a ratio with its own game-bootstrap interval (readout 1);
- the known-state items' effect as a share of the overall effect, with an interval (readout 2);
- intervals for the within-reason correlations (readout 4).

It also reports the goal's effect on the twin questions as a control for a general shift toward "Yes" (readout 3). The agent-preference score is read from its full L only: both of its candidates begin with "Action", so the first-token term is zero.

**Exploratory: model-size series, Qwen3 dense (decided 2026-11-19, same meeting, before any of these scores).** `wf/sizes.py`, runs `size_score_<model>` (one per size, sharded by game) and `size_analysis`. Models: Qwen3-0.6B, 1.7B, 4B, 8B, 14B and 32B instruct, with thinking off. Each is scored with the RQ2 prompt (example set 1, history only) on the 4,666 primary items and the determined successes. Arms: aligned, swapped and neutral tasks, as two-candidate L and as the yes/no question. Qwen3-8B is rescored in this run, so every size shares code and settings; its confirmatory numbers stand as filed.

Reported per size:

- the goal effect, overall, at step 0 and later, and on failures decided by contents or open state;
- flips;
- the mean two-candidate score and yes-share on failures and successes, as the model's own accuracy;
- AUC under the aligned and neutral tasks, on both scales.

**Prediction, written before scoring:** the goal effect stays positive at every size, and the slope of ΔL against log parameters is not positive (larger models are not more swayed). This is a direction, not a test with a threshold. Base models and the 30B mixture-of-experts model are not in this run. The new weights are about 109 GB (Qwen3-8B is already cached). Space was freed so all sizes fit in the scratch quota together. The parameter counts used for the slope are the models' actual counts (0.75, 2.03, 4.02, 8.19, 14.77 and 32.76 billion), not their nominal names.

**Amendment to the dissociation test's readout (2026-11-19, before any of its scores were read; prompted by an outside review).** The logged prediction ("the aligned-vs-swapped effect shrinks once the shared sentence settles existence") cannot separate the accounts. The extra-mention result already shows that the exist sentence added to the swapped prompt reproduces nearly all of Qwen's effect, so the shared-sentence effect will shrink whichever account is true. A salience account (any mention of the object makes success more likely) also predicts shrinkage, since the sentence makes the object equally prominent in both prompts. The readouts are now levels, not the shared-sentence effect:

- **Primary: the denial sentence on the swapped prompt,** L(swapped + absent) − L(swapped). If the task acts as evidence that its object exists, denying existence lowers L (negative). If any mention of the object raises L (salience), it rises (positive). The denial contradicts a solvable task and an action that names the object, so it is reported with that caveat.
- **Secondary: the true-location sentence on the swapped prompt, split by whether the stated location matches the action's receptacle.** Where it differs (the object is elsewhere), the evidence account predicts a sharp fall in L and the salience account a rise. Where it matches (most location failures), the sentence supports the object being there, a rise is expected under both, and the split settles nothing.
- **Reported, not decisive:** the shared-sentence effects (aligned + S vs swapped + S) for all three sentences, and the exist sentence on the swapped prompt (a replication of the extra-mention control).

True information (exist, located) and counterfactual information (absent) are reported separately. The analysis code gains the location-matches split (`location_matches_action`, `location_differs_from_action`); the scoring is unchanged.

**Exploratory: dissociation test of the "task as evidence of existence" account (decided 2026-11-19, after the thinking-mode results and two outside reviews, before any of these scores).** `wf/rq1d.py`, runs `rq1d_score` and `rq1d_analysis`, few-shot Qwen3-8B and Llama-3.1-8B, example set 1, history only, on RQ2's primary items that have one known location for the action's object instance (the history never touches the goal object, so its scene location still holds). Each arm appends one sentence about the action's object to BOTH the aligned and the swapped task line, so each pair still differs only in the goal: *exist* ("There is a mug somewhere in the room."), *located* (its true location, "The mug 1 is at cabinet 3."), *absent* ("There is no mug in this room.", counterfactual). The plain aligned and swapped prompts are rescored in the same job. Predictions, written before scoring: if the goal acts mainly as evidence that its object exists and is available, the aligned-vs-swapped effect shrinks sharply under all three sentences, most under *located*; if it acts through what the agent wants, it persists. Reported for all items, by history length (0 vs 1+) and by failure reason, with how far each sentence moves the swapped and aligned prompts on its own. Not preregistered.

**Exploratory: Qwen3 thinking mode (decided 2026-11-19, before any thinking-mode output).** All Qwen3 results so far use thinking disabled, as preregistered. `wf/thinking.py`, runs `thinking_probe` and `thinking_analysis`. Qwen3-8B with thinking enabled, few-shot example set 1, history only, on a fixed random 1,000 of RQ2's 4,666 items (seed string `20261119|thinking`), under the aligned, swapped, neutral ("explore the room"), and both object-free goals. Generation with vLLM at Qwen's recommended thinking settings (temperature 0.6, top-p 0.95, top-k 20), a fixed seed per prompt, at most 4,096 tokens; a reply that never ends its reasoning is counted and left out. **Changed after the 10-item smoke run (50 generations), which was read for format only:** 28 of 50 replies reached 4,096 tokens without ending their reasoning, mostly looping over the task. Leaving them out would drop most items and select the easy ones. Now the reasoning is capped at 3,072 tokens and, if it has not ended, closed (`</think>`) and the model answers (budget forcing); such replies are flagged `forced`, and every result is reported on all replies and on unforced ones only. Readings: the written answer by the current parser; L after reasoning (log P(o+) − log P(o−) at the answer position, given the model's own reasoning, scored with vLLM prompt log-probabilities); and whether the reasoning mentions "goal"/"task"/"objective" or the action's object. The thinking-off scores for the same items come from `rq1_score` and `rq1x_score`. Sampling noise differs between the two goals of an item, which adds variance but no bias to the paired differences.

**Exploratory follow-up: object-free goals, working actions (decided 2026-11-19, after the RQ1 results, before any of these scores).** `wf/rq1x.py`, runs `rq1x_score` and `rq1x_analysis`, few-shot Qwen3-8B and Llama-3.1-8B only, example set 1, history only. Not preregistered.

- **Object-free goal.** The extra-mention control showed most of ΔL comes from the action's object being named. To test whether the goal acts through meaning without the name, the aligned and swapped task lines keep their template, and the goal object's name is replaced by a fixed description (`DESCRIPTIONS`, e.g. mug: "thick drinking vessel with a handle for hot drinks"). Descriptions were written before any score, and a test checks that none contains an ALFWorld object or receptacle name or any part of its own object's compound name. Estimand: L(aligned, described) − L(swapped, described), on RQ2's population. The action line still names the object, as in every condition.
- **Manipulation check.** Each model is asked once per object to name it from its description (greedy, 16 tokens). The object-free effect is reported on all items and on the items where the model named both the goal and the swapped object correctly.
- **Working actions.** The determined successes are scored under the aligned, swapped and neutral goals. With the failures already scored, this gives the goal effect on actions that work, and how well L separates working from failing actions (AUC, game-resampled CI) under each goal. If the goal raises L by the same amount on both, it adds bias without information.
- **Free splits of `rq1_score`.** ΔL, the mention part and the goal role, by whether an earlier observation already names the action's object, and by failure reason (reasons with at least 50 items).

**Fourth parser audit: failed against the blind rater (2026-11-18).** 175 rows drawn from the `rq1_score` predictions with the parser at version `8639849027611cd2`, excluding every text in the earlier samples. The blind rater (another model instance, given only the rows and the category definitions) left 1 row undecided. On the other 174, the parser matched it on no effect against effect on 82.8%, below the 95% bar, with 5 rows read in opposite directions. The misses are patterns the parser does not yet read, not ambiguous text:

- Replies the parser left unparseable, which the rater read: "The mug 1 is already on desk 1." for a move (no effect, 6 rows); "You still hold the egg 2." for a take (no effect, 6 rows); "You are carrying the book 1." for a move (no effect, 2 rows); "The desklamp 1 is on." for `use desklamp` (success, 4 rows). These are 24 of the 25 rows in the unparseable stratum, which the tool counts as misses by rule.
- Opposite rows: "You attempt to heat the plate 1 … The plate 1 is not heatable." and "… The cup 1 is not hot." were read as process only, but say the action fails (3 rows); "The fridge 1 is still closed." for a take was read as a different effect (1 row); "You open the fridge 1. … Nothing happens." was read as no effect while the rater read the opening as an effect (1 row, a judgement call).

How much this matters: across all 68,340 predictions of the run, the parser leaves 98 unparseable (0.14%), and the patterns behind the opposite rows occur in about 20 predictions. The stratified sample puts 25 of its 175 rows on that 0.14%, so the agreement rate describes the hard cases, not the population. Under the rule, the written-prediction results (RQ1's A, H1, and RQ2's written shift) stay provisional. The rule requires a parser fix and a fresh fifth sample; the user does not label the fourth.

**RQ1, H1 and the secondary RQ2 conditions: one more scoring run (`wf/rq1.py`, `rq1_score`).** Items are the same verified `valid_unseen` collection. Written predictions (greedy, up to 64 tokens) cover every determined item: the 6,527 determined failures not at a shared spot, plus the 307 determined successes. Teacher-forced scores cover RQ2's population (4,666 items).

- **Few-shot models (Qwen3-8B, Llama-3.1-8B-Instruct).**
  - Written predictions under the aligned goal, with example set 1 history only, sets 2 and 3 history only, and set 1 with the full state.
  - Scores under the aligned and swapped goals for each of those four prompt variants.
  - Scores for the two mitigations and the extra-mention control (below).
- **Word2World checkpoints (Qwen2.5-7B and Llama-3.1-8B, fine-tuned).** Full state only, as trained: written predictions under the aligned goal, and scores under the aligned and swapped goals. The Llama checkpoint had never been scored before.
- **Full-state prompt for the few-shot models.** Each user turn, the examples included, opens with "Hidden state (where every object is at the start of the episode; the agent cannot see this):" and the game's starting layout. This is the Word2World context's "Objects on Receptacles" section only. The rest of that context repeats the task line, so it would show the true goal in the swapped condition; a test checks this. The example games' layouts come from the same Word2World files, matched by each example's task line.
- **Ignore-the-goal mitigation.** Before "Next observation:", the query turn adds: "(The agent's goal does not change what happens. Predict the observation from the history and the action alone.)" It is reported as ΔL under the instruction and as the reduction in ΔL against no instruction, a difference-in-differences with a game CI and permutation p.
- **No-goal mitigation.** The existing "explore the room" task line. It is reported against the aligned and swapped goals, with the mean predicted success under each.
- **Extra-mention control (new, exploratory).** The 2×2 cannot be built for take items, which are half of the population, and it showed the effect is mostly the task line naming the action's object. The new control works for every item and adds one sentence to the task line. "Aligned plus" is the aligned goal followed by "There is a mug somewhere in the room." naming the swapped goal's object. "Swapped plus" is the swapped goal followed by the same sentence naming the action's object. Both prompts then name both objects, and each object's existence is implied in both. The goal role is the only difference, so aligned plus minus swapped plus is the goal effect with mentions balanced. Swapped plus minus swapped is the effect of naming the action's object in a sentence that is not a goal. Reported overall, for take and non-take items, and for items with and without history.
- **H1 as tested.**
  - **First prediction:** "the lean is larger from history alone than with the full state". Since the sign of A is not predicted, it is read as |A| history only minus |A| full state, with a game-resampled CI that keeps each game's predictions in both conditions together. It is supported if the CI is above zero. The signed difference is also reported.
  - **Second prediction:** Llama leans more optimistic than Qwen. It is read as A for Llama minus A for Qwen (set 1, history only), with the same paired CI, and supported if above zero.
  - **Reporting:** A is reported for each example set with the SD across sets, and both sensitivity rules are kept.
- **Size and sharding.** About 20 GPU-hours in 8 game shards; records are saved one by one and resumed after a time limit. The analysis waits for every shard and one commit, and re-reads every prediction with the current parser.

**Items whose task matches an example's task.** For 49 of the 4,666 RQ2 items, the game's task line is word-for-word the task line of one of example set 1's examples (91 items for set 2, 66 for set 3). That example's history then shows the item's true task under both goals, identically, so it is not a leak of the item's goal. It was equally true in the confirmatory RQ2 run. Found by `scripts/check_rq1_prompts.py`, which builds every new prompt on the real items. The same script found that no swapped, no-goal or ignore-goal query turn shows the item's true task line, and that every full-state turn carries its layout.

**RQ2 scoring fix for the fine-tuned prompt.** In `wf/rq2.py` the 2×2's mention and fourth lines were always built with the few-shot prompt. The confirmatory RQ2 run scored only the few-shot models, so no result is affected. The task-line override now goes through `build_messages(..., task_line=...)`, which uses each model's own prompt. A test checks the Word2World prompt carries the override.

## Decided 2026-11-17, after the pilots (before any confirmatory data)

These were decided after Pilot A and Pilot B results existed, on train games only. No confirmatory (`valid_unseen`) data had been collected or scored. The preregistered primary test (ΔL per co-primary model, game bootstrap, game-level permutation, Holm) is unchanged.

**RQ3 size.** RQ3 runs on the 134 solvable `valid_unseen` games instead of 274 games. This is the proposal's own contingency, applied as written: "if RQ3 lags at the milestone it shrinks to the 134 unseen games". RQ3 lagged at the milestone (2026-10-29) and still lags. All cells use the same games, so every comparison is paired by game.

**RQ3 error matching: cap and yoke.** The proposal matches the two error types "on errors per episode and step type". The pilot calibration (one rate per arm) matched the counts when predictions were dropped but not when they were kept (4.5 false successes against 2.1 false failures per episode), because false successes make the agent retry failing actions, and each retry was another chance to inject. A fixed count per episode would not work either: under perfect foresight with predictions dropped, the median episode offers one chance for a false success and 38% offer none.

- **False-success arm.** A chance is a step where the agent proposes an action that will fail and the environment has a success text for it. At each chance, inject with probability 0.75, decided by a seed per (game, schedule), until 3 have been delivered. The cap of 3 is `CORRUPTIONS_PER_EPISODE_TARGET`, set in the pilot code's first commit (`e2c1f1f`, 2026-10-28); it is not in the proposal, which specifies three error *schedules*, not three errors. Checked against the pilot's delivered distribution: among false-success episodes that received any error, 67% (dropped) and 79% (kept) received 3 or more, and the most was 17. So the cap binds for most episodes and stops a stuck agent from attracting unlimited errors.
- **Schedules.** Three seeds per game, shared across both retention modes. They differ in which chances are injected. With probability 1 the three schedules would be identical, because runs repeat exactly with the prefix cache off.
- **False-failure arm, yoked.** For the same game, schedule and retention mode, the false-failure episode gets exactly d errors, where d is the number its false-success partner received. The i-th false failure goes at the first working proposal with the same verb, at or after the step of the i-th false success. If none exists, the first working proposal of any verb at or after that step is used, and the fallback is logged.
- **Records.** Every episode logs planned and delivered errors and each fallback. Each cell reports the number of episodes that fell short. Cost per delivered error and the delivered count are secondary analyses.

**RQ3 primary analysis set.** The primary set is the (game, schedule) units whose false-success episode received at least one error in both retention modes. With d = 0, an episode is identical to perfect foresight (and its yoked partner gets none), so membership depends only on behaviour before any injection, which makes it a pre-treatment selection. The interaction I is computed on this set with a game-clustered CI. All units are reported as a secondary analysis. From the pilot's perfect-foresight episodes, about 48% of games qualify at probability 0.75 (52% at 1.0, 41% at 0.5), which is about 64 of 134 games. The pilot's per-game SD of I in qualifying games is 5.05 steps (13 games). Counting one schedule per game, the SE is about 0.63 steps and the 80%-power minimum detectable effect is about 1.8 steps (two-sided, 0.05); the three schedules can only lower this. The pilot's estimate in that set is +3.0 steps (H3 predicts a negative I).

**RQ3 pilot interaction, recorded before confirmatory data.** The mini-run gave I = +1.7 steps (95% CI −0.3 to +3.6), the opposite sign to H3, on 30 games. H3 is not revised. Two notes for the write-up:

1. In the kept cells the false-success arm received more than twice as many errors as the false-failure arm and still did as well, so the imbalance probably biased the pilot toward H3, making its positive I conservative. This is not clean, because episodes going badly also offer more chances to inject, so error count and difficulty are entangled.
2. **Exploratory explanation, written now so it cannot be fitted afterwards.** In the kept format, the real observation sits directly after the wrong prediction in the history ("Expected outcome: You pick up the mug 1." then "Observation: Nothing happens."). The agent can see the world model was wrong and may discount it. H3 assumed kept predictions make false beliefs persist; in this format the contradiction is kept too.

**Rule R margin.** `PILOT_RULE_RETAINED_FORMAT` did not fire (retention moved the false-success arm by 1.3 steps against a margin of 1.0, with an SE of about 1). The rule is followed, so the trace-line kept format stays. It was close, and the format should not be described as validated.

**Labels for conditions added to RQ1 and RQ2.**

- **Full-state condition for the few-shot models.** Pre-specified in the proposal ("each item is scored history-only (primary) and full-state", and H1's predictions), not in the filed preregistration. It had not been built when the pilots ran. Hidden state means Word2World's starting hidden state for the game (its trained input); the current state follows from it plus the history.
- **Object-mention control.** Exploratory, added after the pilot RQ2 results. It asks whether the aligned goal's effect is goal-directed or comes from the action's object being named. ALFWorld has no task that names an object without requiring the agent to pick it up, so the control keeps the swapped goal and adds one neutral sentence naming the action's object ("There is a mug somewhere in this room."). Comparing it with the plain swapped goal isolates mention; comparing the aligned goal with it isolates goal relevance. Proposed design, open to team veto before scoring.

**Confirmatory RQ1 and RQ2 scoring: models, conditions and the decision rule (decided 2026-11-17).**

- **Word2World Llama checkpoint.** `X1AOX1A/WorldModel-Alfworld-Llama3.1-8B` is scored too, at the lowest queue priority, full-state only (its trained input). Hugging Face metadata gives its base model as `meta-llama/Llama-3.1-8B` (the base model, not Instruct); the Qwen checkpoint's base is `Qwen/Qwen2.5-7B` (also base). The prompted co-primary Llama is `Llama-3.1-8B-Instruct`, so the fine-tuned and prompted Llamas differ in post-training as well as fine-tuning. Holm stays across the two co-primary models only (Qwen3-8B and Llama-3.1-8B-Instruct, history-only).
- **Prompt example sets.** Two more sets are drafted from train games only, with every string taken from logged server output, and fixed before any confirmatory item is scored. They are not chosen using any pilot score. Rules for every set: goal relevance must not predict the outcome (each set includes goal-relevant failures and goal-irrelevant successes); the sets match on success and failure counts and on determination reasons, with at least one location failure and one contents failure each. **Audit of set 1 (the filed set, used in both pilots):** it fails the first rule. Its two goal-relevant examples are both failures and its one goal-irrelevant example is a success, so relevance predicts the outcome perfectly, in the direction against H2. Whether set 1 changes is open; the drafts go to the team for review first.
- **Mitigation conditions (secondary).** "Ignore the goal": the aligned goal plus an instruction that the environment does not depend on the task. "No goal": the task line removed entirely. This differs from the existing `absent` condition, which states a neutral task ("explore the room"). No goal splits ΔL into how much the aligned goal raises the log-odds (aligned minus no goal) and how much the swapped goal lowers it (swapped minus no goal).
- **Object-mention control.** The neutral-sentence design is withdrawn: it does not match the aligned goal's surface form or position. Proposed instead: a task line of the same template that names the action's object but whose success the action does not advance. For `move` actions the destination differs; for `heat`, `cool` and `clean` the transformation differs. It cannot be built for `take` actions, because every task naming an object is advanced by picking it up. Those are 446 of Pilot A's 920 primary items, so the control would cover about half. Awaiting team approval of one filled-in item before it is built.
- **Verdict rule, interpreted before confirmatory data.** Per co-primary model, checked in this order: ΔL ≤ 0 or a 95% game-bootstrap CI including zero gives "not supported", including 0 < ΔL < 0.5 with a CI including zero. A positive ΔL below 0.5 with a CI excluding zero gives "weak (not support)". ΔL ≥ 0.5 with a CI excluding zero and a Holm-adjusted p ≥ 0.01 gives "not supported (fails significance threshold)"; this includes 0.01 ≤ p < 0.05. Otherwise "supported", and "supported (strong)" at ΔL ≥ 1.0. The p is the game-level permutation p (all of a game's items flip together, 1,000 permutations), Holm-adjusted across the two co-primary models. The overall H2 verdict is "supported" if at least one co-primary model is supported, as the preregistration says. Per-model and overall verdicts are both reported.
- **RQ3 runner.** Built in parallel in a separate git worktree by a second agent, following the RQ3 entries above. The parser and `wf/constants.py` are read-only for it; a needed change to a shared module is flagged, not made. It is merged only after review, the full test suite and a train-game smoke run.

**RQ1 definitions, the third parser audit, example sets and the object-mention estimand (decided 2026-11-17, before confirmatory scoring).**

- **RQ1 primary definition.** A written prediction is a *failure* prediction if and only if it predicts no effect: "Nothing happens." or an equivalent ("You can't …", "There is no …", "The microwave 1 is closed. You must open it first.", "The apple 1 is already clean."). Anything else predicts *an effect* and counts on the success side of A. Reasons: it is the distinction the environment itself draws (it answers "Nothing happens." to every action that fails), it is what matters to an agent deciding whether to try the action, and it is the distinction the first-token measure q_first reads ("You" against "Nothing"). Among predicted effects the report gives the breakdown: *requested success* (the action as asked works), *process only* (the action is under way but no result is stated, e.g. "You submerge the plate 1 in the sinkbasin 1." for a clean action), and *different effect* (another action or object, e.g. Llama's "You arrive at cabinet 3." for a take action). Sensitivity analyses: A with process-only and different-effect predictions excluded, and A with them counted as failures. Predictions that commit to nothing stay `unparseable` and are excluded, with their count reported. This replaces the provisional "?" treatment of walk-there and process-only replies used in the second audit.
- **Third parser audit.** 150 to 200 rows, not overlapping the first two samples, stratified so that clean, heat and cool actions and Llama's walk-there replies are well covered. The blind rater gets the final category definitions above. Pre-specified scoring: the rater labels each row no effect, requested success, process only, different effect, or "?" (commits to nothing). The pass criterion is on the primary binary (no effect against effect): the parser must match the rater on at least 95% of rows the rater does not mark "?", where the parser's `unparseable` counts as a miss, and no row may be read in the opposite direction. Rows the rater marks "?" are excluded from the criterion and their number is reported. Agreement on the four-way breakdown is reported but is not part of the pass rule. The first two samples (99 and 98 rows) were used to develop the parser and are not evidence for it.
- **Example sets.** Set 1 stays exactly as filed for the primary analysis. Disclosure: its two goal-relevant examples are failures and its goal-irrelevant example is a success, so it shows the model that goal-relevant actions fail. That pushes ΔL down, which is conservative for H2. ΔL is reported separately for each example set, not only as the SD across sets. Sets 2 and 3 are in `notes/fewshot_sets_review.md` for team review before they are frozen.
- **Object-mention control: 2×2 approved.** Four task lines per item, same template, same position: *aligned* (names the action's object; the action advances the goal), *mention* (names the action's object; the action does not advance the goal), *swapped* (another object; aligned's verb and destination), *fourth* (another object; mention's verb and destination). Goal-directed estimand: (aligned − mention) − (swapped − fourth). Taking the difference again removes the effect of the task verb itself; for example, "heat" appears in the success text "You heat the cup 1 …". Mention effect: mention − fourth. The mention cell uses a task type the action is merely irrelevant to, not one it works against: for heat and cool actions the mention task is "clean some X …", for clean actions it is "heat some X …", and for move actions it is the same template with a destination of another receptacle type in the scene. Heat against cool, where the action would work against the goal, is not used. Coverage is move, heat, cool and clean items; take items cannot be built and are listed as a limitation.

**RQ2 confirmatory scoring, as built (2026-11-17, before any unseen item is scored).** `wf/rq2.py` scores the preregistered population (determined failures that name the goal object, goal-agnostic history, not at a shared spot) from all three collection shards, under the aligned and swapped goals, history only, prompt example set 1, for Qwen3-8B and Llama-3.1-8B-Instruct, sharded by game and saved per score. Every item in the population is scored; the preregistered minimum (1,000 items from 200 states) is checked by the analysis, which refuses to report until every scoring shard is complete and all ran at one commit. Verdicts follow the logged precedence, with Holm across both co-primary models. Also scored in the same run, as secondary: the written predictions under both goals (the behavioural check, labelled provisional until the third parser audit passes) and the exploratory object-mention 2×2. The 2×2 is built only where the logged rules apply: heat and cool items whose aligned task is that transformation (mention task: clean), clean items whose aligned task is cleaning (mention task: heat), and move items whose destination type is the goal's receptacle (mention task: the same template with another receptacle type in the room, chosen by a seed per item). Other items, including every take item, are not covered, and the report gives the number covered. Prioritised for 2026-11-19; RQ1, H1, example sets 2 and 3 and the mitigations follow.

**Collection stopped and restarted after an adversarial review (2026-11-17, evening).** The first unseen-game collection (commit `d15e933`) was stopped after about 25 minutes and 11 games. Two independent reviews of the new code, plus checks on the live jobs, found:

- open-state failures (opening what is open, closing what is closed) were never generated, so decision 6's every-step guarantee did not hold;
- the every-step generator also produced "move the goal object to every receptacle of the target type", which are decided by location, not holding, and multiplied the number of resets;
- the game server's memory reached 25 to 34 GB after 2 to 6 games per job (the job limit is 64 GB), and games took about 10 minutes each;
- the state just before a winning step was never built (inherited from Pilot A);
- failed `go to` actions were labelled `contents_or_open_state`;
- a missing or wrong hidden state left the run passing, with only a list in the report.

Changes (commit after `d15e933`): the every-step generator now covers contents, open state and holding at the agent's own receptacle only, including every instance of the goal object and appliances with any number; the winning action is kept; `go to` failures are labelled `already_there` or `location_other`; a missing hidden state, a hidden state that does not list every object the live scan found, or any mismatch in an end-of-run replay of 40 saved items (history and true outcome must reproduce exactly) fails the run; a run folder is locked so two jobs cannot write to one shard; and the job restarts the game server every 6 games while the model server stays up. The stopped run's partial data is kept in `text-wm-fix8/old_runs_d15e933/` for an old-against-new comparison on the same states. No confirmatory item from the stopped run is used.

**Prediction parser, fourth version (2026-11-17, evening).** An adversarial review of `parse_prediction` and a pass over all 6,606 real Pilot A predictions found misreadings in both directions: no-effect replies such as "The potato 1 is not in the world." and "You see nothing." counted as effects; arrivals or openings that mention something absent ("You arrive at countertop 1. There is no mug 1 here.") at risk of counting as no effect; and the wrong action counted as the requested one ("You pick up the mug 1 …" for a move, "You put the cup 1 in the microwave 1. The microwave 1 starts to heat the cup 1." as a stated result). The parser now normalises contractions and prefixes; reads effect actions (go to, open, close, examine, use, inventory, look) by their opening sentence; requires the action's own verb, object, instance and receptacle for a requested success; and does not count purpose or attempt phrases ("to cool it", "starts to heat") as stated results. 130 of the 6,606 pilot readings change; each was read. The third audit sample is drawn only after this version is frozen.

**Confirmatory RQ1 and RQ2 item collection.** 134 `valid_unseen` games with the Pilot A collector, extended so that each item saves the game's hidden state (the Word2World context text and its SHA-256), the scene's object locations and receptacles, and its `determination_reason`. Every failure decided by observed contents, open state or holding state is generated at every visited step, with no subsampling (decision 6). Other alternatives keep Pilot A's sampling (every second step plus every invalid proposal).

## Decided 2026-11-17

Made before any real Pilot A or Pilot B result existed. Pilot A on the old code produced no results (its folders hold only config files).

| Decision | Outcome |
| --- | --- |
| 4. Goal-channel rule | Advisory. Evaluated as filed in Δp and also in nats (fires when every scored model's ΔL upper CI bound is below 0.5). The preregistered RQ2 test runs and is reported whichever way it falls. Δp stays, because the preregistration requires it. |
| 5. RQ1 measure | A uses Ŷ_gen only, the parsed free generation, as the proposal says. The first-token log-odds, logit(q_first), is reported alongside ΔL as a secondary measure. |
| 6. `context_determined` | Kept, with disclosure. Every Pilot A result is also split by the fact that decides each failure (`by_reason`, `false_success_by_reason`). See "Still open" for the rest. |
| 10. Pilot rule provenance | "Pre-specified, not preregistered". See below. |
| 13. Scoring gate | Continue with disclosure. Tier 0 on `d4fa809` failed at 87/100 and stays FAIL on record. The stop rule, written before resubmitting: Pilot A stops if sign(L) and Ŷ_gen agree on fewer than 90% of its determined true failures, for any few-shot model. Pilot B never waited on this gate in code; the queue dependency is removed. |
| 14. vLLM prefix cache | Off (`--no-enable-prefix-caching`) for every agent run. Scoring runs in HF transformers, not vLLM, and already repeats exactly. |
| Pilot A size | 50 games, up from 30, with chained resume jobs so a time limit loses nothing. |
| Llama-3.1-8B-Instruct | Scored in Pilot A itself (replaces decision 7's later run). The model is gated; if its weights cannot be loaded, Pilot A records it as not scored and stays unfinished, and a resume scores only Llama. |
| Headroom rule (H) | Advisory, decided while `pilotB_headroom` was running (27 of 50 games in, success about flat). Reported in success points and in steps to completion; never stops the job; no 100-game recheck. The gate is the proposal's check, `pilotB_consumption`. See below. |

**Decision 5 in detail.** Write q for the model's probability that the action succeeds in any wording. Since a failure always reads "Nothing happens.", L ≈ logit(q) − c, where c = −log P(o+ | success) ≥ 0 is a wording penalty that grows with every detail of the success text the model cannot guess. So the sign of L reads as "predicts success" only when c ≈ 0, which fails for `go to` texts that list unseen contents: on Tier 0's 100 train items, all 12 successes where the sign of L disagreed with the model's own text were such `go to` items. On context-determined items, the population A uses, A came out 0.24 from Ŷ_gen and 0.22 from the sign of L, but with only 9 determined successes. Each score now also saves the first token of each text, its decoded form (to confirm what follows the chat template), their combined probability `first_mass`, and `c_hat = L_first − L`; Pilot A reports the mean change in c_hat between goals, since ΔL = Δlogit(q_first) − Δc.

**Headroom rule in detail.** The proposal's pilot check is that "perfect foresight changes the agent's actions; if not, RQ3 moves to ScienceWorld". The pilot code (`e2c1f1f`, 2026-10-28) instead required perfect foresight to raise ALFWorld success by at least 5 points over no foresight. That requirement is in neither the proposal nor the preregistration, and it conflicts with the proposal's premise that foresight often fails to help, even when correct, because agents misuse it [2]. We now follow the filed check. `pilotB_consumption` gates the later stages: the share of steps where the perfect prediction changes the agent's action must reach `CONSUMPTION_MIN` (0.10). With 27 games of the headroom run done, it was 23% (oracle predictions dropped from the history after each step), 34% (oracle predictions kept in the history) and 19% (valid / not valid verdict only). The success gain stays in every report as an advisory number, together with the paired gain in steps to completion (`oracle_steps_gain`). The 100-game recheck is skipped because the main RQ3 runs measure headroom on 274 games. The headroom episodes were collected at `28a396d`; later code accepts them through `accept_commits` in the consumption and eligibility configs. The retained-format parser fix (explicit "Action:" line) is in `28a396d`, so it covers every headroom episode and the retention mini-run.

**Decision 10 in detail.** All three rules and their thresholds (`FUTILITY_EST_PP`, `FUTILITY_UPPER_PP`, `HEADROOM_MIN_PP`, `RETENTION_MARGIN`) are in commit `e2c1f1f`, made 2026-10-28 12:48:36 and pushed to GitHub at 12:48:43 by the local reflog (confirm on GitHub's activity page). The first ICE job of any kind started at 13:02 that day. `docs/milestone_checkin.tex` (a course document, not in the public repository) names all three rules without thresholds. Suggested wording: "G, H and R were fixed in the project repository (commit `e2c1f1f`, pushed 2026-10-28) before any pilot data existed. They were not part of the filed preregistration. H and R are feasibility gates that select designs; G governs emphasis only, and the preregistered RQ2 test is reported regardless." Rule R reads the retention effect within the false-success (favorable) arm, in steps averaged per game; if it fires, the retained format changes to state persistence, and RQ3 is not dropped. Pilot games come from `train`, so none enter the confirmatory tests on the validation splits.

**Agent action parser.** The parser took the last line of the agent's reply whenever no line began with "Action:". In the retained condition the history shows "Proposed action:" and "Expected outcome:" lines, so a reply copying that pattern could turn "Expected outcome: ..." into the action and count as an invalid proposal. This fits, but has not been shown to cause, the doubled invalid-proposal rate under retained perfect foresight (0.37 vs 0.19 on 20 games). The parser now requires an explicit "Action:" line. A reply without one is a format error: the agent looks around (`look`), the step is flagged (`proposal_format_error`, `executed_format_error`, `under_true_format_error`), it never counts as an invalid proposal or as decision-critical, and Pilot B reports the format-error rate per cell. Every step now saves the agent's raw text (`proposal_raw`, `executed_raw`). Fixed before Pilot B, because rule R reads the retained arm.

**Tool caches.** The ICE home quota (30 GB) was full, mostly a 20 GB uv download cache written by this project's setup on 2026-10-28; it was cleared after confirming no environment links into it. Jobs now keep every tool cache (vLLM, triton, torch inductor, uv, pip, XDG) under the checkout on `/storage`; Hugging Face already used `/storage`.

**Placebo spread, for sample size.** The Qwen3 placebo spread of 4.56 nats is the per-item SD of L under one irrelevant goal minus L under another, over 137 true failures from 20 train games, unclustered. The per-item values were not saved, so ρ cannot be estimated from it. Pilot A estimates ρ on the paired ΔL (`icc_L`) and reports the items needed for 80% power at true effects of 0.5, 0.75 and 1.0 (`n_for_80pct_power`, two-sided α = 0.005, 134 games). Plan on 0.75: the preregistered criterion needs the point estimate itself to reach 0.5, so a true effect of exactly 0.5 meets it about half the time at any n.

## Decided 2026-11-06

| Decision | Outcome |
| --- | --- |
| 1. Mini-run | Held until the calibration fixes below are in |
| 2. Pilot B memory | 128 GB, because the server may reach the old 64 GB limit |
| 3. vLLM prefix cache | A test with the cache off ran alongside; see decision 14 on 2026-11-17 |
| 7. Llama-3.1-8B-Instruct | Superseded on 2026-11-17: Llama is scored in Pilot A itself |
| 8. `STEP_CAP` | Stays 30. A cap of 20 would have lost 2 of the 6 real Qwen3 wins seen, at steps 24 and 25 |
| 9. Admissible list | Stays for now; revisit with the invalid rate under oracle runs |
| 11. `ELIGIBLE_SET_TARGET` | No fallback. If no size up to 5,000 reaches SE ≤ 1.5 it reports null, and the team decides then |
| 12. Pilot A | Held until the fixes below are in, so nothing is scored twice |

## Read before analysing or writing up: changes on 2026-11-06

All of these came from four independent code reviews and were made before any real Pilot A or Pilot B result existed.

**Swapped goals are now achievable.** The protocol asks for a swapped goal of the same task type, but the code took any object in the room, so 53% of main-test failures in a sample of 204 real `valid_unseen` items had an impossible swap, such as `put a hot pot in cabinet`. RQ2 compared an achievable goal against an impossible one about half the time. A swap must now pass the game's own flags in that game's `initial_state.pddl`: pickupable, heatable, coolable or cleanable as the task needs, accepted by the target receptacle (`canContain`), two instances for pick-two tasks, and not the target itself. The rule accepted 717 of 721 real goals in their own games; the other 4 are pick-two tasks whose hidden-state text lists one instance. When no object qualifies, the item has no swap and leaves Pilot A (`swap_feasible` false, reason `no_swap_object`). No substitute is used.

**Closed arrivals are no longer "determined".** `The drawer 2 is closed.` counted as decidable from history even when the history never said so. These were 22 of 28 determined successes in the same 204 items, so A's false-failure rate was mostly this one case. A `go to` success is now determined only if the history already shows the receptacle closed. Other success labels are unchanged, under decision 6.

**A leaves out excluded successes.** `shared_location` successes had stayed in A while the matching failures were removed, and unparseable successes were counted too. On planted data this moved A from −0.01 to −0.30. Both are now out, exclusions are counted on both sides, and the sensitivity check puts flagged items back on both sides. A's real-data size is unknown.

**Few-shot examples are real strings.** The three examples were hand-written and matched no real text: none of 295 real `go to` observations stop at the name, and all 64 real `open` observations say `The X is open.`. That also broke the protocol's rule that observations come from the environment. They are now logged server text from `t0_smoke` on train games 700, 270 and 2332, still one success and two failures. This edits a frozen prompt. The effect on ΔL should be small, because both conditions share the prompt, but it has not been measured. `t0_scoring_agreement` must be rerun.

**RQ3 error rates are matched between arms.** Rates were calibrated on `none` and oracle episodes pooled, counted invalid proposals that cannot be corrupted (19% of real invalid proposals were `done`), and let each arm aim for 3 errors alone. On simulated data with no true effect, unequal error counts gave a false H and an I whose CI excluded zero. Calibration now uses oracle episodes only, counts only steps with a success twin, and gives both arms one target: the lower of 3 and what each can reach. `corruption_mix` shows which action types get corrupted. Real invalid proposals are 55% `go to`, which may rarely mislead the agent, so read that mix before interpreting H.

**Objects can be in two receptacles.** ALFWorld listed at least one object in two places in 366 of 832 sampled games, and picking one up clears only one listing. `Scene.locations` now holds every listing. No outcome label was wrong because of this; the old scan picked one arbitrarily.

**Success text can no longer contradict the item.** A twin from reset could say `In it, you see a toiletpaper 1.` while the agent held that object. `open` and `go to` items whose receptacle the agent has changed now have no twin (`twin_contradicts_state`). This applies to corruptions too.

**The Pilot A gate counts the main test.** ≥500 items and ≥80 states now refer to goal-agnostic, goal-naming, non-shared items. `eligible_failure_determined` still shows the broader count.

**Headroom can stop a job.** It reports `REPORTED`, never PASS. A borderline or fired rule exits 3, which stops the job, and a resume does not skip it. `pace/pilotB_recheck.sbatch` runs the recheck and then the later stages, which refuse to start until the recheck clears the rule.

**Agent repeatability.** Agreement was 0.850 and 0.757 in two Tier 0 runs, against a 0.95 target. Prompts were byte-identical at every divergence, so this is not our code. The protocol's advice to add seeding cannot work, because the agent picks greedily at temperature 0, and that wording is corrected. In a 3-way comparison, first and second runs of a prompt differed on 7 of 194 prompts and later runs on 0 of 83, which points at vLLM's prefix cache without proving it (p = 0.08). All 10 paired games in the first run ended the same way. Treat this as noise in RQ3, not bias.

**Things the write-up should say.** Foresight had never run with the real agent model before Pilot B. Long success texts can make L look pessimistic, which matters for A more than ΔL. The fine-tuned models learned a goal signal only through a `[SUCCESS]` marker we do not score, so a null result for them may be partly structural. 23 of 117 "goal-agnostic" failures show the goal object in an observation, because the rule reads actions only, as filed.

## Read before analysing or writing up: shared locations

**Decided 2026-11-06, before any real Pilot A scores existed.** Only item counts and label checks had been seen, so the choice could not have been shaped by results. Mention this rule in the methods section and report both versions of the result.

**What we found.** ALFWorld places some receptacles at the same spot. From `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. `context_determined` assumed each receptacle had its own spot, so "the agent is elsewhere" counted as proof of failure. Nearly all failure labels rest on that reasoning.

| Evidence | Result |
| --- | --- |
| Game files with at least one shared spot (sample of 400 train games) | 296 of 400 (74%) |
| Receptacles at a shared spot in that sample | 1,786 of 9,527 (19%) |
| "Certain to fail" claims that were wrong on the real server, before the fix | 6 of 1,053 (0.6%) over 30 games, all fitting this pattern; 3 traced to it directly |
| Failure labels resting on location | 480 of 506 after 15 games |
| Detection test: `go to` is not offered for a receptacle at the same spot | Matched a direct test on all 2,164 ordered pairs in 8 games (20 shared) |
| Scene-scan misses in those 8 games explained by shared spots | 13 of 13 |

**What we changed.**

- The scene scan records which receptacles share a spot, and it now looks inside all of them. Before this, it skipped them and missed 34 of 873 objects (3.9%) over 30 games.
- Every item has a `shared_location` field: true when the action's receptacle shares a spot with the one the agent is at.
- Flagged items are still built and scored, but they are left out of the main test: the RQ2 goal effect, RQ1's A, and the 500-item and 80-state gate.
- `pilotA_score` also reports each model with those items put back, as a sensitivity check.

**Where to find it.**

| What | Where |
| --- | --- |
| Main results | `reports.<model>.goal_agnostic`, `A`, `A_ci` in `runs/pilotA_score/metrics.json` |
| Sensitivity check | `reports.<model>.with_shared_location` in the same file, also summarised in `STATUS.md` |
| Item counts | `n_main_test` and `n_shared_location` in `pilotA_transitions`; `exclusions` lists `shared_location` |
| Per-item flag | `shared_location` in `transitions.jsonl` |

**How to report it.**

- State the main result, then the sensitivity result. If they disagree, say so and treat the finding as depending on this choice.
- RQ1's A is more exposed than RQ2's ΔL. ΔL compares one item under two goals, so a doubtful item affects both sides. A counts a success prediction on such an item as a false success.
- The fix is partial, and the write-up should say so. The history rarely shows which receptacles share a spot (`look` can), so "determined" still means determined for someone who knows the layout. Flagged items were identified using environment knowledge that the model never sees. See decision 6 above.
- Confirmatory items need the same rule. It is not yet applied anywhere beyond Pilot A, because the confirmatory build does not exist.
- **Update 2026-11-20.** The confirmatory collection carries the flag, and RQ1 and RQ2 leave flagged items out ("RQ2 confirmatory scoring, as built" above). The flagged items were not scored, so the second version has not been reported yet; see "Shared-spot rule" under 2026-11-20 above.

## Read before analysing or writing up: `valid_unseen` prompts

**Found 2026-11-06 by a crash in `t0_ckpt_accuracy`, and fixed before any confirmatory work.** World-model prompts were looked up by position in the file, but each row's `id` is the game's global index and the rows have gaps. On `train` that made no difference: all 2,420 rows line up. On `valid_unseen` only 3 of 150 rows line up, so a fine-tuned model saw another game's hidden state, with the correct task line written over it. Out-of-range games crashed, and the rest gave no error.

- **Not affected:** Pilot A and Pilot B, which use `train`.
- **Would have been affected:** RQ2 and RQ1 on `valid_unseen`, for the fine-tuned models.
- **Invalid:** every `t0_ckpt_accuracy` figure from before this fix, including the one in `runs/t0_ckpt_accuracy` and in the milestone documents.
- **Now:** a prompt is found by `id`, and a run stops if the prompt is missing or does not contain the game's task line. All 34 `valid_unseen` games from the failed run match with the fix.

## Protocol changes

| Area | Before | After |
| --- | --- | --- |
| Permutation test | Single items flip | Whole games flip, as the prereg requires (§4.9 said "within pairs") |
| Primary Pilot A population | All eligible failures | Only actions naming the goal object, since the aligned goal must be one the action would advance. Others stay scored. |
| Items scored | Failures only | Also true successes under `aligned`, needed for A |
| `absent` etc. on fine-tuned models | Invented prompt | Trained chat format with hidden state, so these are full-state, as the proposal says |
| `committed` / `candidate` wording | After "Next observation:" | Before it, and in the system message for fine-tuned models |
| Scene objects | Fixed list | Read from the environment by visiting every receptacle on a private env id |
| Success twins | Earlier prefixes plus four fixed setups | Planned from reset using that scene |
| Corruption draws | Hash included the arm | Shared across arms (§4.8) |
| `t0_scoring_agreement`, `t0_ckpt_accuracy` | Fixed or first-step items | Items from seeded walks; no LLM needed, so they stay in Phase B |
| `t0_replay` | Scripted actions | 20 logged states × 3 actions, plus replay-vs-live agreement |
| `t0_ckpt_accuracy` match | Exact or same class | Exact only |
| Parser classes | Four plus unparseable | Adds `info` for inventory, look, help, examine |
| Agent prompt | Prediction with no action named | `Proposed action:` above `Expected outcome:`, explained in the system prompt |
| Pilot A main test | All eligible failures naming the goal object | Also leaves out `shared_location` items, which stay scored for a sensitivity check (see above) |
| Scene scan | Not applicable | Restarts the game before ALFWorld's 50-step limit (`ENV_STEP_LIMIT`), which also caps any future `STEP_CAP` at 50 |
| ALFWorld server | Fixed port, no health check | Job-chosen port and a health check, since jobs can share a node |
| Headroom "materially negative" | Any negative difference | Whole CI below zero |
| Saving results | Not specified | Added to the protocol's rules: every long job saves each finished unit and resumes without rework or mixed commits |
| `STATUS.md` | Reported-only runs said PASS | PASS means a target was met; others say `REPORTED`, and results under a target say `BELOW TARGET` |
| `pilotB` job | 6 h, 64 GB | 14 h, 128 GB: about 10–12 h estimated from real episode times |
| `pilotA_transitions` | 120 games, 4 h | 30 games, 6 h: 30 real games gave 818 eligible items against a 500 target, and fewer games halve the time and server memory |

The text of `PILOT_PROTOCOL.md` was brought up to date with these changes on 2026-11-06. No preregistered value changed.

## Bug fixes (no design change intended)

Foresight on a private env id; task and history in every condition; invalid actions executed rather than rewritten; no-twin steps not counted as corruptions; goal parser; swapped task line changes only the object; `context_determined` from history facts; `history_goal_agnostic` reads actions only; `delta_p` filtering; A computable; room description kept in history; no truncation of the admissible list; trace `targets_goal_object`; 100 distinct agreement items; `t0_parser` reads the dataset and fails loudly; receptacle lookup no longer matches `to` inside `potato` or `tomato`; world-model prompts looked up by each row's `id`, and checked against the game's task line, instead of by file position; `use` and `slice` have their own action type; contents are filed under the receptacle the line names; a move no longer creates contents for an unseen receptacle; non-finite scores stop the run; the score cache includes the prompt; a missing swap raises an error; `recovery` needs a positive oracle gain; the retention rule compares the same games; fine-tuned generation stops at every end-of-turn token and allows 256 tokens; every episode, score, generation and built game saved as it finishes, with `--resume` on all long runs; a resume skips runs that finished at the same commit with unchanged inputs, and refuses changed commits, uncommitted code, changed settings or changed input files. Scores carry an item hash. A run refuses a dependency that did not finish, unless its config names the commit under `accept_commits`. Mock runs stay out of real folders, mock records are marked and refused by real runs, and `runs/*/STATUS.md`, `REPORT.md` and `PILOT_REPORT.md` are no longer tracked. The report refuses missing, stale or mock runs. Verification jobs exit nonzero on failure.

## Tier 0 results invalidated

Job 6685206 ran the old code, so the Tier 0 claims in `docs/milestone_checkin.tex` (a course document, not in the public repository) need replacing. Every `t0_ckpt_accuracy` figure before 2026-11-06 is invalid. Tier 0 must be rerun on the commit that holds the changes above: `t0_scoring_agreement` uses the new few-shot examples, and items now use the new swaps and labels. `runs/` in older checkouts still holds old mock output, which real runs refuse.

## GPU type for confirmatory jobs (2026-11-17)

All 32 L40S GPUs on ICE were taken when the confirmatory collection and RQ2 scoring were queued. GPU jobs now take the first free L40S, A40 or A100 (all run the models in bfloat16 with at least 40 GB). The 16 GB V100s are excluded: an 8B model does not fit in 16-bit precision, and V100s have no bfloat16. The collector and RQ3 jobs log the GPU they got. This cannot change item correctness, because every collected item is replayed on the real server before scoring. Log-probabilities can differ across GPU types only by floating-point rounding, far below the 0.5-nat threshold. **(Superseded by "Scores across GPU types (2026-11-18)" below, which measured larger differences; see also 2026-11-20 above.)**

## Scorer precision (2026-11-18)

The RQ2 smoke run on real items showed first-token log-probabilities that were exact bfloat16 values (−4.5, −7.75), with exact ties between candidates. The scorer took `log_softmax` of the model's bfloat16 logits without converting them to float32, so every per-token log-probability was rounded to about 3 significant digits. That is up to 0.016 nats for a token near −8. The rounding is the same in every condition and averages out, so it cannot create or hide a 0.5-nat ΔL. It did make exact ties in the first-token measures, and first-token probability sums slightly above 1. From commit db3eb56 the scorer converts to float32 first. The pilots used the old scorer. Confirmatory RQ2 scoring uses the new one. The model's own logits stay bfloat16.

## Server ports on shared nodes (2026-11-18)

Three of four verification jobs on one node failed to bind their own ALFWorld server and used a fourth job's server instead. A server takes seconds to bind, and the readiness check took any answer on the port as its own. The collection was not affected: no collection job's server or vLLM ever failed to bind, according to their logs. Each job now picks a free port from an offset derived from its job id. A server counts as ready only when the process listening on the port is the job's own process or a child of it. On a failed start, the job tries another port. `pace/port_race_test.sbatch` starts three servers at once from the same first port; on ICE each ended up on its own port, owned by its own process.

## Scores across GPU types (2026-11-18)

The RQ2 smoke items were scored twice: on an A100 with the bfloat16-rounding scorer, and on an A40 with the float32 scorer. Most per-item differences are larger than rounding can explain, so they come from the GPU type: bfloat16 forward passes differ across hardware. The change in an item's aligned-minus-swapped L had SD 0.55 nats for Qwen3 and 0.17 for Llama. The item-to-item SD of L is 10.1 and 1.9, so the GPU adds under 1% to the variance. The mean change was −0.09 and −0.03 nats, 1.3 and 1.4 standard errors from zero over 65 items, so there is no sign of bias. The aligned and swapped versions of an item are scored in the same job on the same GPU, unless a job was cut off between them and a resume finished the item (2026-11-20 above). The confirmatory scoring jobs may land on different GPU types; each job's log records its GPU. Individual scores reproduce only on the same GPU type.

## TextCraft round 2: amendments before the freeze (2026-11-21)

Before any round-2 score, and before the design was frozen, the round-2 design (`notes/textcraft_r2_design.md`, first committed at 4656ec0 and built at f8dd14e) was amended by the owner's decisions and by review 1. All changes are in its section 14.

- **Competence.** It is read per contrast: the yes/no neutral-line AUC is required to be at least 0.70 in each contrast's own action categories, and the other wave-A criteria hold on the full population. It no longer withholds estimates. The first version used one overall gate, which is still reported. These rules were chosen with knowledge of round-1 data, and the note says so.
- **Parent goal.** Three subsets (strict, word overlap only, whole name contained) are reported beside the pooled P. A partial-naming reading rule was first added, then withdrawn the same day; the direct comparison of strict against overlapping parents replaces it.
- **Margin.** The equivalence margin is min(0.3 × SD, 0.4 nats), with each test's power reported.
- **Mechanism readings.** These are tightened: no completion-only reading unless naming is constrained; the assumptions behind the two-item split are stated; padding is read as sensitivity to added steps.
- **Controls.** The ingredient controls of the naming, completion and timing contrasts are class-matched (too few, never obtained).
- **Predictions.** The ALFWorld-based predictions were corrected (the goal-directed part maps to completion, not to the lift).
- **New arms.** Two arms were added to the parent records (the yes/no question under the neutral line; the two-item line on the extended list), so each model scores 25,743 instead of 23,988.
- **Statistics and provenance.** Family tests use 20,000 resamples, and the analysis checks provenance (one commit, pinned revisions).

The records were rebuilt and re-verified, and the ICE tokenization check was rerun on them. None of this affects the confirmatory TextCraft result, which stays as reported.

The same day, also before any round-2 score, the round-2 yes/no question was anchored to TextCraft's replies: "will the environment reply that this action succeeded ("Crafted ..." or "Got ...") or that it failed ("Could not find ...")?", parallel to ALFWorld's "Nothing happens." question. Round 1's question ("Will this action succeed?") can be read as task success: Qwen's Yes on working goal crafts fell from 73% to 14% under the swapped line. Round-2 yes/no scores, and the competence gates built on them, are therefore a different quantity from round 1's, and every round-1 yes/no figure in the design note came from the unanchored question (design section 14.Q).


On 2026-11-22, after review 2 and still before any round-2 score, the analysis gained a coded rejection list (R1–R5, each with the competence it needs) and a descriptive mapping for the central question (mechanism, poor state reasoning, combination, undecided). R3 (P against the role gap) fires only when the matched role gap confirms the unmatched one. Readings now state when a contrast is not competent. The analysis also refuses scores for arms a record does not have. No record or prompt changed (design section 15).


At sign-off (2026-11-22), before any round-2 score, the owner's final corrections were applied (design section 15.F):

- R3 is a descriptive consistency diagnostic outside the 50-test family;
- R4 is a descriptive comparison with a paired interval, not a rejection;
- R5 rejects only the predicted attenuation of the penalty with added steps;
- competence is described as predictive discrimination, not inventory-rule competence;
- the central question's answers have neutral labels;
- the analysis reports which comparisons are confirmatory.

A statistical audit (section 15.A) found no error in the Holm, TOST, margin, clustering or identity code. Records, prompts, populations, sample sizes and the model set did not change.

On 2026-11-22, an independent pre-freeze audit (design section 15.G) re-derived the round-2 statistics and found them correct, and found that nothing defining the experiment had changed. It asked for wording fixes: R3 and R5 in 15.R now match 15.F, and the central question uses the neutral labels. It also asked for two hardening checks, both added: the analysis refuses any model set other than the five, and the timing gates are listed as gates in the inference map. Tests were added for its surviving mutants. No rule, record, prompt, population, sample size or model changed.
