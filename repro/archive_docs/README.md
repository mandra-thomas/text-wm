# RQ2 headline inputs (release `repro-rq2-2026-11-25`)

These files are the inputs of the project's preregistered headline test, RQ2. Given them, `python -m wf.runs rq2_analysis` recomputes `notes/rq2_confirmatory_metrics.json` on a CPU in a few minutes. That file holds ΔL = 6.121 nats for Qwen3-8B and 1.646 for Llama-3.1-8B-Instruct, both "supported (strong)", from 4,666 items in 125 games.

Unpack the archive into the repository's `runs/` folder. The repository README, section "Reproduce the headline result", gives the exact commands.

| Path | What it is |
| --- | --- |
| `rq12_transitions_s0`, `_s1`, `_s2` | The item collection, one folder per game shard: 12,987 items collected at commit `f8d20c1` and replay-verified in full against the game server. The analysis selects the 4,666 preregistered items from them. The `hidden_state` field is removed; see `DATA_LICENSES.md` |
| `rq2_score/shard_00_of_04` to `shard_03_of_04` | Per-item scores of both models under the aligned and swapped task lines, with the exploratory mention conditions (`scoring.jsonl`), and their short written predictions (`generations.jsonl`), scored at commit `ddeabc2` |
| `rq2_release/MANIFEST.json` | Every file's SHA-256 here and in the original run folder, record counts, the item verification summary, and what was left out and why |
| `rq2_release/DATA_LICENSES.md`, `rq2_release/licenses/` | Upstream licences and their notices |

The files are copied unchanged from the confirmatory run folders, apart from the removed field. A `transitions.jsonl` therefore has a different SHA-256 from the original that the verification records name; the manifest lists both.
