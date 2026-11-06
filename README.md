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
- [notes/deviations.md](notes/deviations.md): every change since filing, and six open team decisions. Where this and the protocol disagree, this is current.
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

Do not cite anything tracked under `runs/` yet. `runs/pilot*` and `runs/PILOT_REPORT.md` are old mock output, and `runs/t0_*` came from the code before the fixes.

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

Mock runs use `wf/env/sim.py`, which reproduces the AgentGym server's behaviour behind the real client. Their output goes to `runs_mock/`, so they cannot overwrite real results, and every `STATUS.md` and `REPORT.md` is labelled as mock.

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

The GPU jobs start vLLM for the agent, then stop it before HF scoring, because the two cannot share the GPU.

### After a timeout or crash

Resubmit the same job with `WF_RESUME=1`:

```bash
WF_RESUME=1 bash pace/submit.sh pilotA.sbatch
```

- Work is saved as it finishes: one episode, score, generation or built game at a time. You can read partial results while a job runs.
- A resume skips runs that already passed at the current commit.
- A resume refuses to continue work saved by a different commit, so one result never mixes code versions. Rerun without `WF_RESUME` to start that run again.
- `t0_replay`, `t0_parser` and `t0_expert_labels` take minutes, so they restart instead.
- The verification tools keep partial results but do not resume. Resubmitting one overwrites its output.

## Where results go

Each run writes to `runs/<run_id>/`:

| File | Contents |
| --- | --- |
| `STATUS.md` | PASS or FAIL against the run's gate. Missing means the run has not finished. |
| `REPORT.md`, `metrics.json` | Results, where the run produces them |
| `git_sha`, `config.json` | The commit and settings used |
| `episodes.jsonl`, `transitions.jsonl`, `items.jsonl`, `scoring.jsonl`, `generations.jsonl`, `observations.jsonl` | Saved work, one record per line |
| `games_done.jsonl` | Games fully saved. A game counts as finished only once it is listed here. |

Verification output sits directly in `runs/` and is not tracked by git:

| File | From |
| --- | --- |
| `verify_real_env.json`, `verify_games.json` | Real-server checks |
| `verify_scoring.json`, `verify_agent.json` | Known-answer model checks |
| `HUMAN_REVIEW.md` | Real items and episodes for a person to read |

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

- **Solvable games only.** The split sizes are 2,179 train, 140 `valid_seen` and 134 `valid_unseen`. AgentGym's mappings also list unsolvable games that crash on reset, and `wf/env/server.py` filters them out. See protocol §5.1.
- **The server keeps one live game per id.** Finished games refuse further steps, and ALFWorld ends every episode at 50 steps. Counterfactual replays run on a separate sandbox id, so they never touch the live episode.
- **Steps are slow by design.** Every step replays the game from a reset, in every condition, which costs about 1.7 s before any model time.
- **The real server leaks about 2.6 MB per reset.** That is fine for the pilots, but the full RQ3 run, at roughly 100,000 resets, needs a fix or periodic restarts.
- **Expert labels are unavailable on the real server.** `on_expert_path` stays null rather than being approximated.
