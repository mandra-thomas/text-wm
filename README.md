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
- [notes/deviations.md](notes/deviations.md): every change since filing, the four open team decisions, and what the write-up must mention. Where this and the protocol disagree, this is current.
- [notes/extensions.md](notes/extensions.md): ideas parked until the pilot report exists.

## Status (2026-11-05)

An audit found the original harness passed on mocks but broke on the real server. The fixes are on `fix/audit-bugs`, and the first real runs of the fixed code are in progress.

| Stage | Code | Real results |
| --- | --- | --- |
| Real-server checks | Done | Client, sandbox and winning episodes pass; games check in progress |
| Tier 0: infrastructure | Done | Rerun in progress |
| Known-answer model checks | Done | Queued after Tier 0 |
| Pilot A: RQ2 signal, sample size | Done | None yet |
| Pilot B: headroom, consumption, calibration, retention | Done | None yet |
| Tier 3: optional extras | Not built | None |
| Confirmatory RQ1, RQ2, RQ3 | Not built | None |

Run output is no longer tracked by git, so a fresh clone has none. Older checkouts may still hold old mock files under `runs/`, which real runs refuse. `python -m wf.tools.assemble_pilot_report` refuses to write a report from missing, stale or mock runs; add `--accept-commit RUN=SHA` to keep a run from an earlier commit, which the report then shows.

## How the stages fit together

1. **Tier 0** checks that the machinery works. It answers no research question.
2. **Pilot A** builds at least 500 items from at least 80 states, then scores them. `PILOT_RULE_GOAL_CHANNEL` decides whether RQ2 stays in the headline, and the run sets the study's sample size.
3. **Pilot B** runs in order: `pilotB_headroom` (does perfect foresight help?), `pilotB_consumption` (does the agent use predictions?), `pilotB_eligibility` (sets the error-injection rates) and `pilotB_retention_minirun` (chooses the retained format). `PILOT_RULE_HEADROOM` and `PILOT_RULE_RETAINED_FORMAT` apply here. A borderline headroom result needs `pilotB_headroom_recheck` first.
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

If `pilotB_headroom` stops with exit code 3, read its `STATUS.md`. A borderline result needs `bash pace/submit.sh pilotB_recheck.sbatch`, which runs the 100-game recheck and then the later stages.

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
- **The real server leaks memory on every reset.** It reached about 20 GB after 4,281 resets over 30 games, or about 4.5 MB per reset, without getting slower. Jobs have 64 GB, which Pilot A's 30 games fit comfortably, but the full RQ3 run, at roughly 100,000 resets, needs a fix or periodic restarts.
- **Expert labels are unavailable on the real server.** `on_expert_path` stays null rather than being approximated.
