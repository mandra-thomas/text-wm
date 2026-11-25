Inputs for rerunning the project's preregistered headline result, RQ2, on a CPU in a few minutes.

**What it reproduces.** `python -m wf.runs rq2_analysis` recomputes `notes/rq2_confirmatory_metrics.json`. The test measures the goal effect ΔL: the mean, over failing actions whose failure follows from the history, of the log-odds of the action's success reply against "Nothing happens.", with the agent's real task minus the same with one object swapped. The result is 6.121 nats (95% CI 5.48 to 6.75) for Qwen3-8B and 1.646 (1.51 to 1.78) for Llama-3.1-8B-Instruct, both "supported (strong)" against the preregistered 0.5-nat threshold, on 4,666 items from 125 ALFWorld games.

**How.** Follow the README section "Reproduce the headline result (CPU, about 3 minutes)" at this tag. In short: clone, check out `repro-rq2-2026-11-25`, install `repro/requirements-rq2.txt` in Python 3.11, download `rq2-headline-inputs.tar.gz`, check it against `repro/SHA256SUMS`, unpack it into `runs/`, run `python -m wf.runs rq2_analysis`, then `python repro/compare_rq2_metrics.py runs/rq2_analysis/metrics.json notes/rq2_confirmatory_metrics.json`.

**Assets.**

- `rq2-headline-inputs.tar.gz` (3.5 MB; 73 MB unpacked). It contains:
  - `rq12_transitions_s0`, `_s1`, `_s2/transitions.jsonl`: the 12,987 collected and replay-verified items, collected at commit `f8d20c1`.
  - `rq2_score/shard_00_of_04` to `shard_03_of_04`: both models' per-item scores (`scoring.jsonl`) and short written predictions (`generations.jsonl`), scored at `ddeabc2`.
  - Each run folder's `config.json`, `git_sha` and `STATUS.md`.
  - `rq2_release/`: a manifest with every file's SHA-256, both as shipped and in the original run folder; a readme; the data licences; and the upstream MIT licence texts.
- `SHA256SUMS`: the archive's SHA-256, also committed in the repository as `repro/SHA256SUMS`.

**Differences from the original run folders.**

- **One field removed.** Each item's `hidden_state` is left out. It is a starting-layout text copied verbatim from Word2World's data download, which declares no licence, and the analysis never reads it. The release keeps its SHA-256 (`hidden_state_sha256`).
- **Files the analysis does not read left out.** These include the shards' `inputs.json`, which names cluster paths.

`repro/build_rq2_archive.py` builds the archive deterministically from the run folders.

**Licences.** The code is MIT. The item text comes from ALFWorld, ALFRED and TextWorld (all MIT) through AgentGym's ALFWorld server (MIT). Scores and predictions come from Qwen3-8B (Apache-2.0) and Llama-3.1-8B-Instruct (Llama 3.1 Community License). See `DATA_LICENSES.md`.

**Tested.** On 2026-11-25 a fresh clone following the README exactly reproduced every value in the committed metrics file. All but one float were bit-identical; that one, an intra-class correlation, differs in its 17th significant digit. The analysis took about a minute on a laptop CPU.
