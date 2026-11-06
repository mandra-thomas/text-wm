# Wishful Foresight — Pilot

Preliminary experiments for goal-conditioned bias in text world models (ALFWorld / Word2World AgentGym).

**Law:** [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md)  
**Remote execution:** PACE ICE — see [pace/SETUP_ICE.md](pace/SETUP_ICE.md)

Split sizes in the protocol (`valid_seen` 140, `valid_unseen` 134, train solvable pool) are games with `solvable: true` in ALFWorld PDDL. AgentGym’s mapping JSONs also list unsolvable trials that crash on reset; `wf/env/server.py` filters them out — see protocol §5.1.

## Quick start (mock / unit)

```bash
pip install -e ".[dev]"
pytest -q
bash pace/run_mock_pipeline.sh   # every run, in order, on the simulator
```

Mock runs use `wf/env/sim.py`, which reproduces the AgentGym server's semantics behind the real client. Their output goes to `runs_mock/`, so they cannot overwrite real results, and every `STATUS.md` and `REPORT.md` is labelled as mock.

## PACE

1. Sync this repo to ICE project storage.
2. Copy `pace/pace.env.example` → `pace/pace.env` and fill account/paths.
3. Follow `pace/SETUP_ICE.md` (Word2World, models, vLLM smoke).
4. `bash pace/submit.sh t0_all.sbatch`
5. After filing prereg: set `RETENTION_MARGIN` (already `1.0` in `wf/constants.py`), write `runs/FILED.txt`, then:
   - `bash pace/submit.sh pilotA.sbatch`
   - `bash pace/submit.sh pilotB.sbatch`
6. After a wall-clock timeout or crash, resubmit the same job with `sbatch --export=ALL,WF_RESUME=1 pace/<job>.sbatch`. Work is saved as it finishes: one episode, one score or one built game at a time. A resume skips runs that already passed at this commit and refuses to continue work saved by a different commit. Partial results can be read while a job runs.

Changes made after filing are listed in [notes/deviations.md](notes/deviations.md).

## Package layout

See protocol §2 (`wf/`, `runs/`, `tests/`, `pace/`).
