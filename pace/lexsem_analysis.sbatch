#!/bin/bash
#SBATCH --job-name=wf-lexsem-analysis
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/lexsem_analysis_%j.out
#SBATCH --error=logs/lexsem_analysis_%j.err
#
# The frozen lexsem analysis (notes/lexsem_design.md section 13): run once, from a clean checkout at the freeze tag,
# over runs/lexsem. LX_COMMIT must be the tagged commit; the analysis refuses scores made at any other commit.
# LX_STAGE0 optionally names the archived stage-0 index for the descriptive new-minus-archived comparison.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/_common.sh"
activate_serve
echo "=== lexsem analysis START $(date) commit=$(git rev-parse HEAD) ==="
test -z "$(git status --porcelain --untracked-files=no)" || { echo "checkout not clean" >&2; exit 1; }
python -m wf.lexsem_analysis --dir "${RUNS_DIR}/lexsem" --out "${RUNS_DIR}/lexsem/analysis.json" \
  --expect-commit "${LX_COMMIT:?set LX_COMMIT to the tagged commit}" ${LX_STAGE0:+--stage0-index "${LX_STAGE0}"}
echo "=== lexsem analysis END $(date) ==="
