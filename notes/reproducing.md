# Reproducing the results (2026-11-20; sections 2, 5 and 7 updated 2026-11-25)

This note says what the study ran on, where its data are, which run produced each reported number, and how to run everything again. Code, configs, seeds and commits are all in this repository. The run data are not: the collected items, scores, written predictions and agent episodes live outside git (`runs/**` is ignored). The exception is the headline result. The inputs of the preregistered RQ2 analysis are public as the GitHub release `repro-rq2-2026-11-25`, and the README's section "Reproduce the headline result" reruns it on a CPU.

There are two levels of reproduction:

- **Re-running the analyses** from the saved scores needs no GPU and gives the reported numbers, apart from the known parser change in RQ1 (below). For RQ2 anyone can do this from the public release; for the other analyses it needs the run data, which are not public.
- **Re-running everything from scratch** gives a new, statistically equivalent data set, not the same one. The agent's greedy rollouts are not bit-identical across GPUs, so a new collection produces different histories and items, and scores differ slightly across GPU types. Exact reproduction of the reported numbers requires the saved items.

## 1. What you need

### Software

Recorded on 2026-11-20 from the environments installed on ICE, which every run used. The torch, transformers and vLLM versions match `runs/env.json`, written by the Tier 0 job (6685206, commit `d49710c`) before the confirmatory runs. `pyproject.toml` gives only lower bounds, so install these versions to match the study.

| Component | Version |
| --- | --- |
| Python (both environments) | 3.11 (3.11.18 in `runs/env.json`) |
| **Serve environment** (models; `SERVE_ENV`) | |
| torch | 2.6.0 (built for CUDA 12.4) |
| transformers | 4.51.3 |
| vLLM | 0.8.5 |
| tokenizers | 0.21.4 |
| accelerate | 1.2.1 |
| xformers | 0.0.29.post2 |
| triton | 3.2.0 |
| numpy, scipy, pyyaml | 2.2.6, 1.17.1, 6.0.3 |
| **ALFWorld server environment** (`Word2World/uv_alfworld`) | |
| alfworld | 0.3.3 |
| TextWorld | 1.7.0 |
| agentenv_alfworld (AgentGym's server) | 0.0.1 |
| torch, numpy, fastapi (as listed in that environment) | 2.15.0, 2.4.6, 0.162.0 |
| **Word2World** checkout | `e8dc240269fb222de3c242b09c70bfbf0b4ac1c9` |

The CUDA driver version was not recorded. `pace/fix_env.sbatch` is the script that pinned vLLM 0.8.5 and alfworld 0.3.3 in these environments; `pace/SETUP_ICE.md` has the rest of the setup.

### A second model environment: `serve2` (added 2026-11-23)

Models that need transformers 5 cannot run in the serve environment above (transformers 4.51.3). Families wave B (Qwen3.5, Ministral 3, Gemma 4, Qwen3.8; branch `families-waveb`) is scored in a second environment, `serve2`, and later TextCraft rounds that use such models (the round 3 proposal on branch `textcraft-r3`) plan to use it too. Everything earlier, including the confirmatory runs and wave A, keeps the serve environment above. The facts below are from `notes/families_waveb_design.md` on branch `families-waveb`.

| Component | Version |
| --- | --- |
| Location on ICE | `<storage root>/envs/serve2` (on ICE, `/storage/ice1/2/1/gmat3/envs/serve2`), built 2026-11-20 with uv by `envs/build_serve2.sbatch` |
| Python | 3.11.18 |
| torch | 2.14.1 (built for CUDA 12.6) |
| transformers | 5.23.0 |
| tokenizers, huggingface-hub, accelerate | 0.23.3, 2.4.0, 1.16.0 |
| jinja2, mistral-common, sentencepiece, protobuf, safetensors | 3.1.6, 1.13.1, 0.2.3, 7.36.2, 0.8.0 |
| numpy, scipy | 2.3.5, 1.17.1 |

The full list (80 packages) is `envs/serve2_freeze.txt`; it pins exact versions but not hashes. A scoring job selects `serve2` explicitly (`pace/family_score_serve2.sbatch` on that branch) and refuses to start if its Python has transformers below 5. transformers 5 reads the model cache from `HF_HUB_CACHE`, not `TRANSFORMERS_CACHE`, and returns a `BatchEncoding` from `apply_chat_template`, which the wave B scorer unwraps; the wave B design note lists the six differences that matter. No wave B score is read unless an environment check (two wave A models rescored in `serve2` and compared with their earlier scores) passes.

### The environment and its data

"ALFWorld" here is the AgentGym ALFWorld server shipped with Word2World, not the original ALFWorld release. Its game data come from Word2World's `scripts/download_data/download_data.py`. That data renames `put` to `move` and adds a `help` action, so the stock ALFWorld download does not work with this code. No version or checksum of the downloaded data was recorded. The fine-tuned models' prompts, and the full-state condition's starting layouts, come from Word2World's `data/init_contexts/alfworld/wm_instruct_*.json` in the same download. The Hugging Face dataset `X1AOX1A/LLMasWorldModels` is used only by the Tier 0 parser check.

The study uses the 134 solvable `valid_unseen` games. AgentGym lists 195; the other 61 crash on reset and `wf/env/server.py` leaves them out. The game list is not committed: it is the server's listing of solvable games, shuffled with seed 20260924 (`wf/runs.py`), so it is fixed given the same data.

### Models

| Model | Role | Snapshot used |
| --- | --- | --- |
| `Qwen/Qwen3-8B` | Agent (all agent runs) and co-primary world model; thinking off unless stated | `b968826d9c46dd6066d109eabc6255188de91218` |
| `meta-llama/Llama-3.1-8B-Instruct` | Co-primary world model (gated) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| `X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B` | Word2World fine-tuned world model | `8d4f069c4bf5a739763664897b26ca17c161f0da` |
| `X1AOX1A/WorldModel-Alfworld-Llama3.1-8B` | Word2World fine-tuned world model | `8f089d2126f0406a296747041ebf33b902837cd3` |
| `Qwen/Qwen3-0.6B` | Size series | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `Qwen/Qwen3-1.7B` | Size series | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `Qwen/Qwen3-4B` | Size series | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `Qwen/Qwen3-14B` | Size series | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| `Qwen/Qwen3-32B` | Size series | `9216db5781bf21249d130ec9da846c4624c16137` |

The runs did not pin revisions. These snapshots were read from the ICE model caches on 2026-11-20; each model has exactly one snapshot there, so it is the one every run loaded. From 2026-11-20 the code pins them. Weights are loaded in bfloat16, and the scorer takes log-probabilities in float32 from commit `db3eb56` on (all confirmatory and exploratory scoring).

## 2. Where the data are

Each batch of runs was made in its own checkout on ICE, under `/storage/ice1/2/1/gmat3/text-wm-fix<N>/`, so that a resume never mixed commits. Each run's output is in that checkout's `runs/<run>/`, or `runs/<run>/shard_<i>_of_<n>/` for runs split by game. Every run folder holds `git_sha` (the commit), `config.json`, `inputs.json` (hashes of the files it read) and its saved work. Later checkouts hold copies of the collection so they could score it. All 19 checkouts' `runs/` folders together are a little under 2 GB.

*(Updated 2026-11-23: the count of 19 was the count on 2026-11-20. Since then the rule has stayed the same, one fresh numbered checkout per frozen study or batch of runs, so the count has grown. The size above is for those 19 and was not remeasured.)* The checkouts the notes name, with what they hold:

- `text-wm-fix4`, `fix5`: the first jobs of 2026-11-17, made from git bundles (README);
- `fix8`: the stopped first collection's partial data (`deviations.md`);
- `fix9`: the item collection; `fix11` to `fix20`: the confirmatory and exploratory runs in the table below;
- `fix21`: the wave A family scores (`family_score_*`); `fix23`: the wave A analysis, the placement check and the copies of its inputs;
- `fix24`: TextCraft round 1's scoring; `fix26`, `fix27`: its exploratory diagnostic runs;
- `fix29`: Stage 1b's census labelling (its freeze tag); `fix30`: Stage 1a's decision-simulation scores; `fix36`: the mechanism battery's scoring and analysis; `fix37`: Stage 1b's model-free analysis, scoring and analysis;
- `fix39`: wave B's checks at its freeze (tag message);
- `text-wm-qwenaudit`: the forensic audit's prompt check (`qwen_effect_audit_forensic.md`), which modified no other checkout.

Some numbers in between (for example τ-bench's and Option C's checkouts) are not named in the notes. The RQ3 episodes are also archived as `text-wm-runs-archive/text-wm-fix11/runs/rq3` (`decisionsim_agent_design.md`).

ICE course storage is not permanent. *(Corrected 2026-11-25.)* An earlier version of this paragraph said the owner keeps a full copy outside this repository. That was true only on 2026-11-20. The off-cluster copy then covered the checkouts `text-wm-fix` to `fix19`, with a SHA-256 manifest. On 2026-11-25 a copy of the later checkouts was in progress, without a manifest yet. Until it is finished and checksummed, the run data of `fix20` onward, which covers every result after 2026-11-20, exist only on ICE. The off-cluster copy is not public.

**What is public.** The GitHub release [`repro-rq2-2026-11-25`](https://github.com/mandra-thomas/text-wm/releases/tag/repro-rq2-2026-11-25) holds the inputs of the headline analysis, `rq2_analysis`, as one 3.5 MB archive, `rq2-headline-inputs.tar.gz`, whose SHA-256 is committed in `repro/SHA256SUMS`. The archive holds:

- the 12,987 collected items, from the three `rq12_transitions_s*` runs;
- the four `rq2_score` shards, with both co-primary models' scores and written predictions;
- a manifest.

It was built from fix11 by `repro/build_rq2_archive.py`, which is deterministic: rebuilding from the same run folders gives the same bytes. Two things differ from the run folders:

- **One field removed.** Each item's `hidden_state`, the text copied verbatim from Word2World's data download, is left out because that download declares no licence ([DATA_LICENSES.md](../DATA_LICENSES.md)). `rq2_analysis` does not read it.
- **Files the analysis does not read left out.** The scoring shards' `inputs.json`, which names cluster paths, is one of them. The manifest records the original SHA-256 of every file the archive holds.

No other run's data are public.

## 3. Which run produced each result

This table covers the ALFWorld studies run up to 2026-11-20. For every reported result, including the later studies (families, TextCraft, τ-bench, the decision simulations, the mechanism battery, Option C and the interpretability pilot), the README's section "Reproduce the headline result" has a short table of the tag or commit to check out, and whether that code is on `main`. Each later study's results note names its runs, jobs and commits.

Commits are read from each run folder's `git_sha`. Every metrics file in `notes/` is byte-identical to the `metrics.json` of the analysis run named here. "GPU" is what each run's jobs could get; the type each job actually got is printed in its job log on ICE (`logs/`), which is not copied here.

| Result | Runs | Commit | Metrics file | Checkout | GPU |
| --- | --- | --- | --- | --- | --- |
| The collected items: 12,987 in all, 4,666 in RQ2's population | `rq12_transitions_s0`, `_s1`, `_s2` (config `rq12_transitions_s<k>.yaml`), verified by `verify_items` | `f8d20c1` | none; counts are in `rq2_confirmatory_metrics.json` (`items`), verification in each shard's `VERIFY_part_<i>_of_4.json` | fix9 (copies in fix11 to fix19) | any of L40S, A40, A100; verification on CPU |
| RQ2 verdict, secondary measures, object-mention 2×2 (`rq2_results.md`) | `rq2_score` (4 shards), `rq2_analysis` | `ddeabc2`, both | `rq2_confirmatory_metrics.json` | fix11 | shards 0–2 A40, shard 3 L40S |
| RQ1, H1, secondary RQ2 conditions, Word2World models, extra-mention control (`rq1_results.md`; also `exploratory_results.md` §1) | `rq1_score` (8 shards), `rq1_analysis` | `de40795`, both; parser `8639849027611cd2` | `rq1_metrics.json` | fix12 | L40S, A40 and A100 across shards; per shard not recorded here |
| RQ3 (`rq3_results.md`) | `rq3` (4 shards), `rq3_analysis` | `ddeabc2`, both | `rq3_metrics.json` (with a provenance block: commit and episode hashes) | fix11 | any of L40S, A40, A100 |
| Object-free goals, working actions (`exploratory_results.md` §1) | `rq1x_score` (8 shards), `rq1x_analysis` (also reads `rq1_score`) | `39722d6`, both | `rq1x_metrics.json` | fix13 | not recorded here |
| Thinking mode (§2) | `thinking_probe` (one job), `thinking_analysis` (also reads `rq1_score`, `rq1x_score`) | `adb6528`, both | `thinking_metrics.json` | fix14 | not recorded here |
| Dissociation (§3) | `rq1d_score` (8 shards), `rq1d_analysis` | scoring `a3e1580`, analysis `2155b8d` | `rq1d_metrics.json` | fix15 (scoring), fix17 (analysis) | not recorded here |
| Counterfactual task, yes/no question (§4) | `rq1v_score` (8 shards), `rq1v_analysis` | `2155b8d`, both | `rq1v_metrics.json` | fix17 | not recorded here |
| Mechanism diagnostic (§5) | `mech_score` (8 shards), `mech_analysis` | `10a61a7`, both | `mech_metrics.json` | fix19 | any of L40S, A40, A100 |
| Qwen3 size series (§6) | `size_score_qwen3-<size>` (1 shard for 0.6B, 1.7B, 4B; 2 for 8B, 14B and 32B), `size_analysis` | scoring `8bfd84c`, analysis `10a61a7` | `size_metrics.json` | fix18 | any for 0.6B–14B; 32B on A100-80GB only |
| RQ1 and H1 at the final parser (supersedes `rq1_metrics.json` for written predictions) | `rq1_analysis_v2` (reads `rq1_score`) | analysis `28b4d30`; scores `de40795`; parser `3c25d40cc71529e2` | `rq1_v2_metrics.json` | fix20 | CPU |
| Goal effect on working actions (`exploratory_results.md` §1, §4) | `rq1x_analysis_v2`, `rq1v_analysis_v2` (read `rq1_score`, `rq1x_score`, `rq1v_score`) | analysis `28b4d30`; scores `de40795`, `39722d6`, `2155b8d` | `rq1x_v2_metrics.json`, `rq1v_v2_metrics.json` | fix20 | CPU |
| Thinking mode at the final parser (§2) | `thinking_analysis_v2` | analysis `28b4d30`; probe `adb6528`; parser `3c25d40cc71529e2` | `thinking_v2_metrics.json` | fix20 | CPU |
| Size series after the NaN fix (§6; numbers identical) | `size_analysis_v2` | analysis `28b4d30`; scores `8bfd84c` | `size_v2_metrics.json` | fix20 | CPU |
| Exclusion flow, items per game, first-token coverage, twins lost (`rq2_results.md`); ratio intervals, share answered Yes, instance-number items (`exploratory_results.md`) | `audit_analysis` (reads the collection, `rq2_score`, `rq1_score`, `rq1x_score`, `rq1v_score`, `size_score_*`) | `28b4d30` | `audit_metrics.json` | fix20 | CPU |
| Shared-spot sensitivity (`rq2_results.md`) | `rq2_shared_analysis` (reads the collection and `rq2_score`; `rq2_shared_score` had no items to score) | `28b4d30` | `rq2_shared_metrics.json` | fix20 | CPU |
| Numbers marked **post hoc** in the notes | none; one-off scripts on the scores above | unknown | none | — | — |

Notes on the table:

- **RQ1 at the final parser.** `rq1_results.md` now quotes `rq1_analysis_v2` (fix20, 28b4d30). An earlier second `rq1_analysis`, at `c3c4c3d` with the same final parser, exists in fix16; its metrics were never copied into `notes/`. Re-reading the run's 68,340 predictions with the final parser moves 110 of them to the other side; see `rq1_results.md`, "Parser status".
- **Thinking mode's written shift.** The 4.7-point written shift in `exploratory_results.md` is from `thinking_v2_metrics.json`, at the final parser. `thinking_metrics.json` holds the earlier parser's 5.4.
- **Superseded metrics files.** `rq1_metrics.json`, `rq1x_metrics.json`, `rq1v_metrics.json`, `thinking_metrics.json` and `size_metrics.json` are kept, byte-identical to their first analysis runs, so earlier quotes can be traced; the `_v2` files supersede them (`deviations.md`, "Changes from the 2026-11-20 audit"). The 2026-11-20 runs were also checked locally before ICE, with identical results.
- **The 32B run** used 2 shards (`WF_N_SHARDS=2`), overriding its config's `n_shards: 4`.
- **Post hoc numbers.** The scripts that made them are not in the repository; folding them into the pipeline is planned.

## 4. How the runs were submitted on ICE

Every job sources `pace/_common.sh`, which reads `pace/pace.env` (account, paths, token), refuses mock mode, keeps tool caches under the checkout, and checks that `PROJECT_ROOT` is the checkout the script sits in. GPU jobs go through `pace/submit.sh`, which adds the account, QOS, partition, mail and GPU request from `pace.env` and passes the rest of the environment to the job. Leave `PACE_GRES` empty or `gpu:1`, so the scripts' own "any of L40S, A40 or A100" request applies. All real runs refuse to start without `runs/FILED.txt`, which is in the repository.

1. **Item collection** (`pace/rq12_collect.sbatch`, GPU, 4-hour jobs, 64 GB). One job per shard:

   ```bash
   for k in 0 1 2; do RQ12_RUN=rq12_transitions_s$k bash pace/submit.sh rq12_collect.sbatch; done
   ```

   The job starts vLLM with the agent model and restarts the leaking ALFWorld server every 6 games, looping until the shard is done or time runs out. Resubmit an unfinished shard with `WF_RESUME=1`.
2. **Item verification** (`pace/verify_items.sbatch`, CPU partition, plain `sbatch`). Four parts per shard, each with a fresh server:

   ```bash
   for k in 0 1 2; do for i in 0 1 2 3; do
     sbatch --export=ALL,VERIFY_RUNS=rq12_transitions_s$k,VERIFY_PART=$i/4 pace/verify_items.sbatch
   done; done
   ```

   Scoring refuses a collection shard without a full pass on its exact file.
3. **Scoring runs** (`pace/rq2_score.sbatch`, GPU, HF transformers, no game server). This one script runs any scoring run named in `RQ2_RUN`, one game shard per job: `rq2_score`, `rq1_score`, `rq1x_score`, `rq1d_score`, `rq1v_score`, `mech_score` and `size_score_<model>`.

   ```bash
   for i in 0 1 2 3; do RQ2_RUN=rq2_score WF_SHARD=$i WF_N_SHARDS=4 bash pace/submit.sh rq2_score.sbatch; done
   ```

   `WF_N_SHARDS` must match across a run's shards and overrides the config's `n_shards`. The script's limit is 2 hours, enough for an RQ2 shard; longer runs were submitted with plain `sbatch`, the account options and a longer `--time` (about 2.5 GPU-hours per eighth of `rq1_score`, so 4-hour jobs). The long scoring jobs had resume jobs queued behind them, which run only if the job before fails or times out:

   ```bash
   sbatch <account options> --dependency=afternotok:<job id> \
     --export=ALL,RQ2_RUN=rq1_score,WF_SHARD=0,WF_N_SHARDS=8,WF_RESUME=1 pace/rq2_score.sbatch
   ```

   The owner's helper scripts that queued these chains were on ICE and are not in the repository. A resume refuses work saved at another commit, so do not pull new commits into a checkout while its shards are running.
4. **Qwen3-32B** needs an A100 with 80 GB. On ICE the `gpu-a100` constraint also matches 40 GB cards, so its jobs must request the 80 GB cards explicitly. The other sizes fit on any of the three types.
5. **Thinking mode** (`thinking_probe`) runs vLLM inside the job (no server) on one GPU, over 1,000 items and five task lines. There is no dedicated job script, and its submission line was not recorded. Any one-GPU job that runs `python -m wf.runs thinking_probe` works, for example `rq2_score.sbatch` with `RQ2_RUN=thinking_probe` and a longer `--time`.
6. **RQ3** (`pace/rq3.sbatch`, GPU, 4-hour jobs). One job per shard; vLLM stays up and the game server restarts every 40 episodes:

   ```bash
   for i in 0 1 2 3; do WF_SHARD=$i WF_N_SHARDS=4 bash pace/submit.sh rq3.sbatch; done
   ```

   Resubmit an unfinished shard with `WF_RESUME=1` and the same `WF_SHARD`.
7. **Analyses** need no GPU and no server: `python -m wf.runs <run>_analysis` on any CPU node, once every shard's `STATUS.md` says `shard complete`. They refuse incomplete runs and shards made at different commits.

**Using a new collection.** Each scoring config trusts the collection made at `f8d20c1` (`accept_commits` in `wf/config/*_score.yaml`). Items collected at another commit are refused until that commit is added there.

## 5. Running without SLURM

The job scripts also run as plain shell scripts on any Linux machine with a suitable GPU: `_common.sh` uses `local` where it would use the SLURM job id, and the current directory where it would use the submit directory. For example:

```bash
RQ2_RUN=rq2_score WF_SHARD=0 WF_N_SHARDS=4 bash pace/rq2_score.sbatch
```

Set the variables that `pace/pace.env.example` lists (`SERVE_ENV`, `WORD2WORLD_ROOT`, `ALFWORLD_DATA`, `HF_HOME`, `RUNS_DIR`, `HF_TOKEN` for Llama) in `pace/pace.env` or the environment. `module load` is tried and its failure ignored. You can also call `python -m wf.runs <run>` directly. Scoring runs need only the models and the items. Collection and RQ3 also need the ALFWorld server (Word2World's `uv_alfworld/bin/alfworld`) and a vLLM server with the agent model, started with the prefix cache off. Point the code at them with `ALFWORLD_BASE_URL` and `VLLM_BASE_URL`; the job scripts start both and do this for you.

Without a GPU, `pip install -e ".[dev]"`, `pytest -q` and the mock runs (README, "Run locally") check the logic on a simulator. Run `python -m wf.textcraft.freeze --fetch` first, or the TextCraft tests skip. Without a GPU you can also rerun the headline analysis on the real data: README, "Reproduce the headline result". On 2026-11-25 that took about a minute on a laptop CPU from a fresh clone, and it reproduced `rq2_confirmatory_metrics.json` exactly, apart from one float in its 17th significant digit.

## 6. From scratch, in order, with rough costs

The pilots ran on `train` games and fixed the design, which is now in the configs. No confirmatory or exploratory run reads pilot output, so the pilots are not needed to reproduce the reported results.

| Step | What | Cost | Source of the estimate |
| --- | --- | --- | --- |
| 0 | Setup (`pace/SETUP_ICE.md`): environments, data, about 170 GB of model weights with the size series | an afternoon | — |
| 1 | Tier 0 and model checks (optional) | about 1 GPU-hour for Tier 0 | `runs/env.json` (57 minutes) |
| 2 | Item collection, 3 shards | roughly 22 GPU-hours or less | 134 games at about 10 minutes each, the rate measured on the stopped first collection (`deviations.md`); the final collection's time is not recorded |
| 3 | Item verification, 12 parts | about 12 CPU-hours | `verify_items.sbatch`: about 1 hour per 1,300 items |
| 4 | `rq2_score`, then `rq2_analysis` | about 4 GPU-hours | `rq2_score.sbatch`: about 1 hour per quarter |
| 5 | `rq1_score`, then `rq1_analysis` | about 20 GPU-hours | `rq2_score.sbatch` comment and `deviations.md` |
| 6 | `rq3`, then `rq3_analysis` | about 65 GPU-hours | about 16 hours per shard over 4 shards, from the job records; not recorded in the repository |
| 7 | Exploratory scoring: `rq1x_score`, `rq1d_score`, `rq1v_score`, `mech_score` (8 shards each), `thinking_probe`, the six `size_score_<model>` runs; then their analyses | not recorded in the repository | the size series' weights are about 109 GB beyond Qwen3-8B (`deviations.md`) |
| 8 | All analyses | minutes each, CPU | — |

`rq1x_analysis` and `thinking_analysis` also read `rq1_score`, so run step 5 before them.

## 7. Licences

*(Updated 2026-11-25.)* The project's code is under the MIT licence ([LICENSE](../LICENSE)). The upstream licences, what of each is in the repository or the public RQ2 release, and the notices they require are in [DATA_LICENSES.md](../DATA_LICENSES.md), with copies of the MIT texts in `licenses/`. On 2026-11-25 each licence was read from its upstream file, not from metadata alone:

| Component | Licence | How checked |
| --- | --- | --- |
| ALFWorld | MIT | `LICENSE` in `alfworld/alfworld` |
| ALFRED (ALFWorld's scenes) | MIT | `LICENSE` in `askforalfred/alfred` |
| TextWorld | MIT (GitHub labels it "Other" because the file starts with a version line) | `LICENSE.txt` in `microsoft/TextWorld` |
| AgentGym (ALFWorld server, TextCraft) | MIT | `LICENSE` in `WooooDyy/AgentGym` |
| τ²-bench (retail data) | MIT | `wf/tau/vendor/LICENSE` |
| Qwen3 models | Apache-2.0 | `LICENSE` on the Qwen3-8B model card |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License; gated, access approved by Meta on request. Its redistribution terms cover the weights, code and documentation, not outputs | `models/llama3_1/LICENSE` in `meta-llama/llama-models` |
| Word2World fine-tuned checkpoints | declared "other", with no name or text given | Hugging Face metadata; nothing of them is redistributed |
| Word2World code | none: the repository has no licence file | GitHub API (no licence file) |
| `X1AOX1A/LLMasWorldModels` dataset (Word2World's data download) | none declared on the dataset card | Hugging Face card and metadata. Its own text, the items' `hidden_state` field, is therefore left out of the public release |

## 8. Known limits

- **GPU type.** bfloat16 forward passes differ across GPU types. On 65 smoke items, an item's aligned-minus-swapped L changed with SD 0.55 nats (Qwen) and 0.17 (Llama) between an A100 and an A40, with no sign of bias (`deviations.md`, "Scores across GPU types"). Individual scores repeat exactly only on the same GPU type. An item's two conditions are scored in one job, except where a resume finished an item cut off between them, and some exploratory contrasts pair `rq1_score` with `rq1x_score`, which ran as separate jobs.
- **RQ1's parser.** `rq1_results.md` uses parser version `8639849027611cd2`. Re-running `rq1_analysis` at the current commit uses `3c25d40cc71529e2` and reads 110 of 68,340 predictions differently, so its written-prediction numbers will not match to the last digit.
- **vLLM is not bit-deterministic.** The agent decodes greedily, but its rollouts can still differ across runs, GPUs and batch sizes. In Tier 0, step-level agreement between two runs was 0.85 and 0.76, before the prefix cache was turned off. In RQ3, 10 of 342 episodes that should have matched exactly did not. The thinking-mode replies are sampled with a fixed seed per prompt, but vLLM sampling still depends on hardware and version. New collections and thinking runs are therefore new samples.
- **Versions recorded after the fact.** The model snapshots and the package versions above were read on 2026-11-20 from the environments and caches the runs used, not logged by each run. Each model had one snapshot in the cache, so the snapshots are certain; the package versions are those installed now, and no change to the environments after the runs is recorded.
- **No game data checksum.** The ALFWorld data downloaded by Word2World's script carry no recorded version or checksum.
