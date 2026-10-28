#!/bin/bash
#SBATCH --job-name=wf-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
set -a
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/pace/pace.env"
set +a
cd "${PROJECT_ROOT}"
mkdir -p logs .envs third_party "${HF_HOME}"

echo "=== SETUP START $(date) on $(hostname) ==="
nvidia-smi || true

# Use conda only — do NOT module-load an older system CUDA; pip torch ships CUDA 12.4 libs.
module load anaconda3/2023.03
eval "$(conda shell.bash hook)"

if [[ ! -d "${SERVE_ENV}" ]]; then
  conda create -y -p "${SERVE_ENV}" python=3.11
fi
conda activate "${SERVE_ENV}"
activate_serve

python - <<'PY'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "=== prefetch models ==="
python - <<'PY'
from huggingface_hub import snapshot_download

for repo in [
    "Qwen/Qwen3-8B",
    "X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B",
]:
    print("fetch model", repo, flush=True)
    snapshot_download(repo)
    print("done", repo, flush=True)

print("fetch dataset X1AOX1A/LLMasWorldModels", flush=True)
try:
    snapshot_download("X1AOX1A/LLMasWorldModels", repo_type="dataset")
    print("done dataset", flush=True)
except Exception as e:
    print("WARN dataset download failed:", e, flush=True)
PY

echo "=== Word2World ==="
if [[ ! -d "${WORD2WORLD_ROOT}/.git" ]]; then
  git clone https://github.com/X1AOX1A/Word2World.git "${WORD2WORLD_ROOT}"
fi
cd "${WORD2WORLD_ROOT}"
echo "Word2World SHA=$(git rev-parse HEAD)"
pip install -q uv || true
if [[ -f scripts/env_setup/uv_agentgym_rl.sh ]]; then
  bash scripts/env_setup/uv_agentgym_rl.sh || echo "WARN: uv_agentgym_rl.sh failed"
fi
if [[ -f scripts/env_setup/uv_alfworld.sh ]]; then
  bash scripts/env_setup/uv_alfworld.sh || echo "WARN: uv_alfworld.sh failed"
fi
if [[ -f scripts/download_data/download_data.py ]]; then
  python scripts/download_data/download_data.py || echo "WARN: download_data failed"
fi

cd "${PROJECT_ROOT}"
python -m wf.tools.write_env_json || true
echo "=== SETUP END $(date) ==="
