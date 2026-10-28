#!/bin/bash
#SBATCH --job-name=wf-fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --output=logs/fix_%j.out
#SBATCH --error=logs/fix_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
set -a
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/pace/pace.env"
set +a
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== FIX START $(date) on $(hostname) ==="
nvidia-smi || true
module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate "${SERVE_ENV}"
activate_serve

echo "=== upgrade vLLM for Qwen3 ==="
pip install -U "pip<25.1"
# Qwen3ForCausalLM needs newer vLLM than 0.6.6
pip install --no-cache-dir "vllm==0.8.5" || pip install --no-cache-dir "vllm>=0.8.5,<0.10"
python - <<'PY'
import vllm, torch
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "=== rebuild uv_alfworld with binary blis ==="
export PATH="$HOME/.local/bin:$PATH"
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "${WORD2WORLD_ROOT}"
rm -rf uv_alfworld
uv venv uv_alfworld --python 3.11
# shellcheck disable=SC1091
source uv_alfworld/bin/activate
export UV_LINK_MODE=copy
uv pip install --upgrade pip
# Prefer wheels to avoid compiling blis against missing Python.h
uv pip install --only-binary=:all: "blis" || uv pip install "blis"
uv pip install "alfworld==0.3.3"
uv pip uninstall -y opencv-python || true
cd AgentGym/agentenv-alfworld
uv pip install -e .
which alfworld
alfworld --help | head -5 || true
deactivate || true

mkdir -p "$HOME/.cache"
if [[ -d "${WORD2WORLD_ROOT}/data/alfworld" ]]; then
  ln -sfn "${WORD2WORLD_ROOT}/data/alfworld" "$HOME/.cache/alfworld"
elif [[ -f "${WORD2WORLD_ROOT}/data/alfworld.zip" ]]; then
  unzip -n "${WORD2WORLD_ROOT}/data/alfworld.zip" -d "$HOME/.cache" || true
fi

cd "${PROJECT_ROOT}"
python -m wf.tools.write_env_json || true
echo "=== FIX END $(date) ==="
