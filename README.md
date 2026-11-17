# Wishful Foresight — Pilot

Does a language-model world simulator predict what an agent wants to happen instead of what will happen, and does that hurt the agent? This repo holds the staged pilot for that study, run on ALFWorld through the Word2World AgentGym server. The confirmatory study is not built yet.

| Question | What it measures |
| --- | --- |
| RQ1 | Direction of errors: false-success rate minus false-failure rate (A) |
| RQ2 | Goal effect: change in log-odds of success between the agent's real goal and a swapped goal (ΔL, in nats) |
| RQ3 | Cost to the agent: false successes against false failures, with predictions discarded or retained |

Design documents:

- [cs8803_dlt_proposal.pdf](cs8803_dlt_proposal.pdf) and [cs8803_dlt_preregistration.pdf](cs8803_dlt_preregistration.pdf): the study.
- [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md): the implementation spec. Its time estimates predate the fixes below.
- [notes/deviations.md](notes/deviations.md): every change since filing, the team decisions (most made 2026-11-17), what is still open, and what the write-up must mention. Where this and the protocol disagree, this is current.
- [notes/extensions.md](notes/extensions.md): ideas parked until the pilot report exists.

## Status (2026-11-17)

An audit on 2026-11-05 found the original harness passed on mocks but broke on the real server. The fixes and the 2026-11-17 decisions are merged into `main`. The pilots ran on ICE on 2026-11-17: Pilot A and Pilot B's headroom stage at `28a396d`, Pilot B's later stages at `9fb5e0c`.

| Stage | Code | Real results |
| --- | --- | --- |
| Real-server checks | Done | Pass |
| Tier 0: infrastructure | Done | Scoring gate failed at 87/100 on `d4fa809`; continued with disclosure and a stop rule (decision 13) |
| Known-answer model checks | Done | Done; human review file awaits a reader |
| Pilot A: RQ1 and RQ2 signal, sample size | Done | Qwen3-8B and the fine-tuned Word2World model scored; Llama-3.1-8B scoring |
| Pilot B: headroom, consumption, calibration, retention | Done | Headroom, consumption and calibration done; retention mini-run running |
| Tier 3: optional extras | Not built | None |
| Confirmatory RQ1, RQ2, RQ3 | Not built | None |

Pilot results so far, on 50 train games. They are pilot numbers, not the confirmatory test.

- **RQ2 (goal effect).** For Qwen3-8B, history only, on 920 failing items from 49 games, ΔL is 7.1 nats (95% CI 5.9 to 8.3, permutation p = 0.001), against the preregistered 0.5. About half of that is the first word alone (3.6 nats). Its own written predictions say "success" 12.8 points more often under the aligned goal. The fine-tuned Word2World model, which sees the full hidden state, shows no goal effect (−0.2 nats, CI −0.43 to 0.05).
- **RQ1 (direction).** Qwen3-8B, history only, leans optimistic: A = 0.23 (CI 0.17 to 0.29), with 27% false successes and 3% false failures. Word2World with full state leans slightly pessimistic: A = −0.08. The two differ in model as well as in state, so H1 still needs the full-state condition for the few-shot models, which is not built.
- **RQ3 (agent).** Perfect foresight does not raise the agent's win rate (54% against 56% with no predictions, CI for the gain −14 to +10 points). This is consistent with the proposal's premise that agents misuse foresight. Perfect predictions change the agent's action on 28% of steps, which passes the proposal's pilot check. Both error types can be injected at 3 per episode.
- **Checks.** The Pilot A scoring stop rule did not fire (sign of L and written prediction agree on 91% of determined failures for Qwen3-8B). The goal-channel rule did not fire.

Run output is no longer tracked by git, so a fresh clone has none. Older checkouts may still hold old mock files under `runs/`, which real runs refuse. `python -m wf.tools.assemble_pilot_report` refuses to write a report from missing, stale or mock runs; add `--accept-commit RUN=SHA` to keep a run from an earlier commit, which the report then shows.

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
