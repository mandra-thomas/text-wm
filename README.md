# Wishful Foresight: does the agent's goal leak into an LLM world model?

Language-model agents increasingly consult a **world model**, a language model asked what the environment will reply to an action, before they act. That world model is often shown the agent's task.

In a goal-independent environment, an action whose failure is already settled by what the agent has seen should be predicted to fail, whatever the task says. This project tests whether it is. The study is preregistered and runs on ALFWorld (text household tasks). It compares two prompted models, Qwen3-8B (thinking off) and Llama-3.1-8B-Instruct, and also scores Word2World's fine-tuned ALFWorld world models.

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

**RQ2, confirmatory.** Details are in [notes/rq2_results.md](notes/rq2_results.md). The test used 4,666 replay-verified failing actions from 125 of the 134 unseen games. Both models pass the preregistered test, whose threshold was 0.5 nats:

| Model | ΔL | 95% CI | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 6.12 nats | 5.48 to 6.75 | 0.002 | supported (strong) |
| Llama-3.1-8B-Instruct | 1.65 nats | 1.51 to 1.78 | 0.002 | supported (strong) |

The two models differ in where the effect lands:

- **Qwen:** concentrated where the agent has not acted yet.
- **Llama:** spread across the episode.

**RQ1, H1 and secondary RQ2 conditions.** Details are in [notes/rq1_results.md](notes/rq1_results.md).

- **Lean toward success.** Both prompted models lean toward predicting success, Llama far more (A = 0.77 against 0.19).
- **H1.** H1b holds. H1a holds for Llama and is reversed for Qwen.
- **The goal effect under other conditions:**
  - it holds with each of three few-shot example sets;
  - an instruction to ignore the goal removes 9% of Qwen's effect and none of Llama's;
  - showing the hidden state cuts it by about 70% without removing it;
  - the fine-tuned Word2World Qwen model shows none, and the fine-tuned Llama model a small reversed effect (−0.28 nats, CI −0.42 to −0.15).
- **Caveat.** The written-prediction numbers are provisional, because the prediction parser has not met its prespecified audit rule. The decision log records audits three to six.
- **Superseded reading.** The note's reading that "most of the RQ2 effect comes from the action's object being named" was overturned by the exploratory runs below: with object-free task descriptions, 61% (Qwen) and 92% (Llama) of the effect remains.

**RQ3, confirmatory.** Details are in [notes/rq3_results.md](notes/rq3_results.md). H3 is not supported: the interaction is +0.94 steps (95% CI −0.20 to 2.14), opposite in sign to the prediction. The experiment could not test the question well:

- keeping even perfect predictions in the agent's history lowered success below having no foresight at all;
- most episodes in the primary set failed;
- its errors were injected, not the world models' own.

**Exploratory follow-ups.** Details are in [notes/exploratory_results.md](notes/exploratory_results.md). None was preregistered; each design was logged before its data.

- **Not only the object's name.** Most of the effect survives task descriptions that never name the object (61% for Qwen, 92% for Llama). But for Qwen, merely stating that the object is somewhere in the room reproduces nearly all of it.
- **Other formats and tasks.** It replicates when the model answers a fixed yes/no question. It is unchanged when the aligned task is not the exact one the agent pursued.
- **Discrimination.** On our candidates it lowers how well the models rank working actions above failing ones: a large drop for Llama, a small one for Qwen. All failing candidates act on the goal's object while most working ones do not, so this reflects the task promoting actions on its object over unrelated working actions.
- **Practical size depends on model and format.**
  - Llama's predicted success moves by 13–40 points.
  - Qwen's moves by 42 points before the agent has acted, and little after, because Qwen stays confident that these actions fail.
- **Instructions.** An instruction to ignore the goal barely helps.
- **Reasoning.** Qwen3 thinking mode narrows the aligned-vs-swapped gap by a third, but does not reduce the lift over a neutral task.
- **Mechanism.** The dissociation test of "the task as evidence that its object exists" gave mixed readouts and needs a confirmatory test.

## Status (2026-11-19)

| Stage | Run(s) | State |
| --- | --- | --- |
| Pilots A and B | `pilotA_*`, `pilotB_*` | Done (2026-11-17) |
| Item collection, 134 `valid_unseen` games | `rq12_transitions_s0..s2` | Done; every item replay-verified |
| RQ2 confirmatory (preregistered) | `rq2_score`, `rq2_analysis` | Done; results in `notes/rq2_results.md` |
| RQ1, H1 and secondary RQ2 conditions | `rq1_score`, `rq1_analysis` | Done; results in `notes/rq1_results.md` (written-prediction numbers provisional) |
| RQ3 confirmatory | `rq3`, `rq3_analysis` | Done; results in `notes/rq3_results.md` (H3 not supported) |
| Object-free goals, working actions | `rq1x_score`, `rq1x_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Qwen3 thinking mode | `thinking_probe`, `thinking_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Dissociation: object exists / located / absent | `rq1d_score`, `rq1d_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |
| Counterfactual aligned task, yes/no feasibility | `rq1v_score`, `rq1v_analysis` | Done (exploratory); results in `notes/exploratory_results.md` |

Analyses need no GPU: run `python -m wf.runs <run>_analysis` once every scoring shard reports `shard complete`. Run output is not tracked by git, so a fresh clone has none. Real runs refuse mock files and work saved at another commit.

## Documents in this repository

- [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md): the implementation spec for the pilots.
- [notes/deviations.md](notes/deviations.md): the dated decision log. It records every change since the study was filed, and each exploratory design, logged before its data. It also holds the parser audits and what the write-up must mention. Where it and the protocol disagree, it is current.
- [notes/rq2_results.md](notes/rq2_results.md) and [notes/rq1_results.md](notes/rq1_results.md): confirmatory results, with metrics in [notes/rq2_confirmatory_metrics.json](notes/rq2_confirmatory_metrics.json) and [notes/rq1_metrics.json](notes/rq1_metrics.json).
- [notes/rq3_results.md](notes/rq3_results.md): the agent experiment (RQ3), with metrics in [notes/rq3_metrics.json](notes/rq3_metrics.json).
- [notes/exploratory_results.md](notes/exploratory_results.md): the exploratory follow-ups, with metrics in `notes/rq1x_metrics.json`, `notes/thinking_metrics.json`, `notes/rq1d_metrics.json` and `notes/rq1v_metrics.json`.
- [notes/testing.md](notes/testing.md): test suite, mutation check and calibration of the statistics.

The filed proposal and preregistration are not in this repository.

## How the stages fit together

1. **Tier 0** checks that the machinery works. It answers no research question.
2. **Pilot A** builds items from 50 train games, then scores them with Qwen3-8B and Llama-3.1-8B-Instruct (few-shot) and the fine-tuned Word2World model. It reports A, ΔL, Δp and the first-token measures, applies the scoring stop rule, and sets the study's sample size. `PILOT_RULE_GOAL_CHANNEL` is advisory.
3. **Pilot B** runs in order: `pilotB_headroom` (does perfect foresight help?), `pilotB_consumption` (does the agent use predictions?), `pilotB_eligibility` (sets the error-injection rates) and `pilotB_retention_minirun` (chooses the retained format). `pilotB_consumption` is the gate: perfect predictions must change the agent's action on at least 10% of steps, the proposal's own pilot check. `PILOT_RULE_HEADROOM` is advisory since 2026-11-17 and never stops the job; `PILOT_RULE_RETAINED_FORMAT` applies after the mini-run.
4. **`runs/PILOT_REPORT.md`** collects the results, using `python -m wf.tools.assemble_pilot_report`.
5. **Tier 3** is optional and allowed only if Pilots A and B finish cleanly. Nothing depends on it.
6. **The confirmatory study** runs on `valid_unseen`, with its design fixed by the pilot rules.

Pilot A and Pilot B refuse to run without `runs/FILED.txt`, and that marker must not be a mock filing.

## Run locally

No GPU, models or ALFWorld server needed.

```bash
pip install -e ".[dev]"
pytest -q
bash pace/run_mock_pipeline.sh     # every run, in order, on the simulator
python -m wf.runs t0_smoke --mock  # one run
python scripts/mutation_check.py   # breaks the code on purpose; the tests must notice
```

Mock runs use `wf/env/sim.py`, which reproduces the AgentGym server's behaviour behind the real client. They write only to `runs_mock/` or a temp folder and refuse any other `RUNS_DIR`. Mock records are marked, real runs refuse them, and cluster jobs refuse to start in mock mode.

A passing mock run shows the logic holds together. It says nothing about the real server or real models, and only ICE can show that.

## Run on PACE ICE

One-time setup is in [pace/SETUP_ICE.md](pace/SETUP_ICE.md): copy `pace/pace.env.example` to `pace/pace.env`, fill in your account and paths, then install Word2World, the environments and the models. `pace/pace.env` holds your Hugging Face token, so never commit it.

Run these in order from the repo root on ICE:

```bash
bash pace/submit.sh t0_all.sbatch         # Tier 0
bash pace/submit.sh verify_models.sbatch  # known-answer checks and the human review file
bash pace/submit.sh pilotA.sbatch
bash pace/submit.sh pilotB.sbatch
```

`submit.sh` adds the account, QOS, partition and GPU from `pace.env`, so use it for GPU jobs. `verify_env.sbatch` is the exception, because it runs on CPU:

```bash
sbatch pace/verify_env.sbatch                                # all real-server checks
sbatch --export=ALL,VERIFY_ONLY=games pace/verify_env.sbatch # just the real-game check
```

Tier 0 and Pilot A stop vLLM before HF scoring, because the two cannot share the GPU. Pilot B uses only vLLM.

`pilotB_headroom` no longer stops on the headroom rule; it stops (exit 1) only if keeping predictions in the history is materially worse than dropping them. `pace/pilotB_recheck.sbatch` is no longer used.

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

Each run writes to `runs/<run_id>/`:

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
| `wf/constants.py` | Preregistered constants and engineering thresholds |
| `wf/config/` | Settings for each run |
| `wf/env/` | Server client (`server.py`), simulator (`sim.py`), replay (`replay.py`), scene scan (`scene.py`), expert labels (`expert.py`) |
| `wf/agent/` | ReAct loop (`react.py`) and foresight conditions (`foresight.py`) |
| `wf/data/` | Goal parsing (`goals.py`), transitions and alternatives (`transitions.py`), success twins (`candidates.py`) |
| `wf/scoring/` | Prompts (`prompts.py`), log-odds scoring (`logodds.py`), observation parser (`parser.py`) |
| `wf/corruption/` | Which steps get a wrong prediction (`schedule.py`) |
| `wf/analysis/` | Statistics (`stats.py`) and reports (`report.py`) |
| `wf/tools/` | Real-server and model verifiers, review export, report assembly, environment record |
| `pace/` | Job scripts and ICE setup |
| `tests/` | Unit, regression, adversarial and crash-and-resume tests |
| `scripts/` | Mutation check and vLLM smoke test |

## Things worth knowing

- **Some receptacles share a spot.** From `cabinet 1` the agent can act on `cabinet 2`, and `go to cabinet 2` does nothing. Items affected by this carry `shared_location`, stay scored, and are left out of Pilot A's main test. Read that section of [notes/deviations.md](notes/deviations.md) before analysing or writing up results.
- **Solvable games only.** The split sizes are 2,179 train, 140 `valid_seen` and 134 `valid_unseen`. AgentGym's mappings also list unsolvable games that crash on reset, and `wf/env/server.py` filters them out. See protocol §5.1.
- **The server keeps one live game per id.** Finished games refuse further steps, and ALFWorld ends every episode at 50 steps. Counterfactual replays run on a separate sandbox id, so they never touch the live episode.
- **Steps are slow by design.** Every step replays the game from a reset, in every condition, which costs about 1.7 s before any model time.
- **The real server leaks memory on every reset.** It reached about 20 GB after 4,281 resets over 30 games, or about 4.5 MB per reset, without getting slower. Jobs have 64 GB; the first Pilot A job (50 games plus scoring) peaked near that limit without failing, but the full RQ3 run, at roughly 100,000 resets, needs a fix or periodic restarts.
- **Expert labels are unavailable on the real server.** `on_expert_path` stays null rather than being approximated.
