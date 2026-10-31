# PACE ICE one-time setup (Wishful Foresight pilot)

Do this on an ICE **compute** node (interactive GPU or CPU job), not the login node, for installs and model downloads.

## 1. Project layout

```bash
# Example — replace with your project storage root
export PROJECT_ROOT=/storage/coda1/p-XXX/$USER/text-wm-pilot
mkdir -p "$PROJECT_ROOT"/{hfcache,runs,.envs,third_party,logs}
cd "$PROJECT_ROOT"
# sync this repo here (git clone or rsync from laptop)
cp pace/pace.env.example pace/pace.env
# edit pace.env: PACE_ACCOUNT, paths, HF_TOKEN
```

## 2. Word2World + ALFWorld data

```bash
set -a; source pace/pace.env; set +a
cd "$PROJECT_ROOT/third_party"
git clone https://github.com/X1AOX1A/Word2World.git
cd Word2World
# record SHA later in runs/env.json
bash scripts/env_setup/uv_agentgym_rl.sh
bash scripts/env_setup/uv_alfworld.sh
# Remove any pre-existing stock ALFWorld data first, then:
python scripts/download_data/download_data.py
# Confirm server script exists:
ls scripts/env_server/start_alfworld.sh
# Data usually lands in ~/.cache/alfworld (or set ALFWORLD_DATA).
# Expect solvable counts: train≈2179, valid_seen=140, valid_unseen=134
# (AgentGym mappings are larger; unsolvable PDDL is excluded by wf/env/server.py — PILOT_PROTOCOL §5.1).
```

## 3. Serve env (vLLM + transformers)

```bash
module load anaconda3 cuda/12.1   # adjust to ICE module list
conda create -y -p "$SERVE_ENV" python=3.11
conda activate "$SERVE_ENV"
pip install "vllm>=0.6" "transformers>=4.51" torch accelerate datasets pyyaml scipy
# AgentGym packages may live in WORD2WORLD uv envs — keep AGENTGYM_ENV separate if needed.
```

## 4. Prefetch models & dataset

```bash
conda activate "$SERVE_ENV"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE
# huggingface-cli login   # if HF_TOKEN unset and Llama needed later
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "Qwen/Qwen3-8B",
    "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B",
    "X1AOX1A/WorldModel-Alfworld-Llama3.1-8B",
    "X1AOX1A/LLMasWorldModels",
]:
    print("fetch", repo)
    snapshot_download(repo)
# Optional / gated (confirmatory co-primary, not Pilot A):
# snapshot_download("meta-llama/Llama-3.1-8B-Instruct")
PY
```

## 5. vLLM thinking-off smoke

```bash
conda activate "$SERVE_ENV"
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --port "$VLLM_PORT" &
sleep 60
python scripts/smoke_vllm_thinking.py --base-url "http://127.0.0.1:${VLLM_PORT}/v1"
# Expect: no <think> blocks in 20 samples. Kill the server when done.
```

## 6. GPU ownership (1×A100)

Tier 0 / pilots must not co-reside vLLM (agent) and HF transformers (WM scoring) on one GPU — HF will CPU-offload and generation becomes ~1 min/item. `pace/t0_all.sbatch` time-shares: vLLM for agent checks, then stop vLLM, then HF scoring. Pilot A should stop vLLM before `pilotA_score` the same way.

## 7. Submit runs

```bash
cd "$PROJECT_ROOT"
sbatch pace/t0_all.sbatch          # Tier 0 suite
# After runs/FILED.txt exists and RETENTION_MARGIN is set:
sbatch pace/pilotA.sbatch
sbatch pace/pilotB.sbatch
```

## Notes

- Prefer **A100 / L40S / H100**. V100 is incompatible with modern vLLM.
- Keep weights under `$HF_HOME` on project/scratch storage.
- ALFWorld stays up for the whole job (CPU/localhost). vLLM and HF scoring take turns on the GPU (see §6).
