# PACE ICE one-time setup (Wishful Foresight)

Do this on an ICE **compute** node (interactive GPU or CPU job), not the login node, for installs and model downloads. The versions the study actually ran with are listed in `notes/reproducing.md`; install those if you want to reproduce its numbers.

## 1. Project layout

```bash
# Example — replace with your own storage root. The study's checkouts were under /storage/ice1/2/1/<user>/.
export PROJECT_ROOT=/storage/ice1/<path>/$USER/text-wm-pilot
mkdir -p "$PROJECT_ROOT"/{hfcache,runs,.envs,third_party,logs}
cd "$PROJECT_ROOT"
# sync this repo here (git clone, or rsync from a laptop)
cp pace/pace.env.example pace/pace.env
# edit pace.env: PACE_ACCOUNT, paths, HF_TOKEN. Leave PACE_GRES empty or "gpu:1" (see §7).
```

Each checkout needs its own `pace/pace.env` naming its own `PROJECT_ROOT`; jobs stop if the two disagree.

## 2. Word2World + ALFWorld data

```bash
set -a; source pace/pace.env; set +a
cd "$PROJECT_ROOT/third_party"
git clone https://github.com/X1AOX1A/Word2World.git
cd Word2World
git checkout e8dc240269fb222de3c242b09c70bfbf0b4ac1c9   # the commit the study used (runs/env.json)
bash scripts/env_setup/uv_agentgym_rl.sh
bash scripts/env_setup/uv_alfworld.sh
# Remove any pre-existing ALFWorld data older than 0.4.0 first: the code expects ALFWorld's 0.4.0 action
# wording (put -> move, plus help), which Word2World's download supplies as ALFWorld's own files. Then:
python scripts/download_data/download_data.py
# Confirm server script exists:
ls scripts/env_server/start_alfworld.sh
# Data usually lands in ~/.cache/alfworld (or set ALFWORLD_DATA).
# Expect solvable counts: train≈2179, valid_seen=140, valid_unseen=134
# (AgentGym mappings are larger; unsolvable PDDL is excluded by wf/env/server.py — PILOT_PROTOCOL §5.1).
```

`pace/fix_env.sbatch` rebuilds the ALFWorld server environment (`uv_alfworld`) with `alfworld==0.3.3` and upgrades vLLM to 0.8.5 in the serve environment. The study's environments went through it.

## 3. Serve env (vLLM + transformers)

```bash
module load anaconda3/2023.03   # the module the job scripts load
conda create -y -p "$SERVE_ENV" python=3.11
conda activate "$SERVE_ENV"
pip install "vllm==0.8.5" "transformers==4.51.3" "torch==2.6.0" accelerate datasets pyyaml scipy
# AgentGym packages may live in WORD2WORLD uv envs — keep AGENTGYM_ENV separate if needed.
```

## 4. Prefetch models & dataset

```bash
conda activate "$SERVE_ENV"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE
# Llama-3.1-8B-Instruct is gated: accept Meta's licence on Hugging Face, then put a token with
# access in pace/pace.env as HF_TOKEN=... (jobs read it at start; never commit pace.env).
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "Qwen/Qwen3-8B",                              # agent and co-primary world model
    "meta-llama/Llama-3.1-8B-Instruct",           # co-primary world model (gated; needs HF_TOKEN)
    "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B",     # Word2World fine-tuned checkpoints
    "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B",
    # Qwen3 size series (exploratory); about 109 GB together
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
]:
    print("fetch", repo)
    snapshot_download(repo)   # add revision=... to match the study; snapshots are in notes/reproducing.md
snapshot_download("X1AOX1A/LLMasWorldModels", repo_type="dataset")
PY
```

Check the storage quota before fetching the size series.

## 5. vLLM thinking-off smoke

```bash
conda activate "$SERVE_ENV"
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --no-enable-prefix-caching \
  --port "$VLLM_PORT" &
sleep 60
python scripts/smoke_vllm_thinking.py --base-url "http://127.0.0.1:${VLLM_PORT}/v1"
# Expect: no <think> blocks in 20 samples. Kill the server when done.
```

## 6. GPU ownership (one GPU per job)

Tier 0 / pilots must not co-reside vLLM (agent) and HF transformers (WM scoring) on one GPU — HF will CPU-offload and generation becomes ~1 min/item. `pace/t0_all.sbatch` and `pace/pilotA.sbatch` time-share: vLLM for agent work, then stop vLLM, then HF scoring. Pilot B, item collection and RQ3 keep vLLM for the whole job (agent only). RQ2, RQ1 and the exploratory scoring runs use HF transformers only, except the thinking-mode probe, which runs vLLM inside the job.

`t0_ckpt_accuracy` and the fine-tuned models read `$WORD2WORLD_ROOT/data/init_contexts/alfworld/wm_instruct_*.json` (from `download_data.py`), or the directory in `WF_INIT_CONTEXTS`.

## 7. Submit runs

Submit GPU jobs through `pace/submit.sh`, from the repo root. It adds the account, QOS, partition, mail and GPU request from `pace.env`:

```bash
cd "$PROJECT_ROOT"
bash pace/submit.sh t0_all.sbatch          # Tier 0 suite
# After runs/FILED.txt exists:
bash pace/submit.sh pilotA.sbatch
bash pace/submit.sh pilotB.sbatch
```

The confirmatory and exploratory jobs, their order and their settings are in `notes/reproducing.md`. CPU-only jobs (`verify_env.sbatch`, `verify_items.sbatch`) name the CPU partition themselves and are submitted with plain `sbatch`.

**GPU type.** The job scripts ask for one GPU of any of the L40S, A40 or A100 types (`--gres=gpu:1` with a constraint); each job's log records the one it got. `submit.sh` passes `PACE_GRES` as `--gres`, so a typed value such as `gpu:A100:1` overrides that and pins every job to A100s: leave `PACE_GRES` empty or set it to `gpu:1`. Qwen3-32B is the exception: it needs an A100 with **80 GB**, and on ICE the `gpu-a100` constraint also matches 40 GB cards, so request the 80 GB cards explicitly for those jobs.

## Notes

- Any of **L40S, A40 or A100** works for the 8B models (all run bfloat16 with at least 40 GB). V100s do not: no bfloat16, and 16 GB is too small. Scores differ slightly across GPU types (`notes/deviations.md`, "Scores across GPU types").
- Keep weights under `$HF_HOME` on project/scratch storage. Job scripts also keep every tool cache under the checkout, because home is a 30 GB quota.
- ALFWorld stays up for the whole job (CPU/localhost). vLLM and HF scoring take turns on the GPU (see §6).
