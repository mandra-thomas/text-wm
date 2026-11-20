#!/usr/bin/env bash
# Submit an sbatch script with account, QOS, partition and GPU request from pace.env.
# A script's own #SBATCH --gres and --partition win: they are added from pace.env only for a script that does not
# set them (a script that names a partition, such as the coc-cpu ones, gets no GPU request), and --constraint is
# never added, so a script's own GPU types stand (2026-11-20; before, PACE_GRES="gpu:A100:1" pinned every job
# submitted this way to A100s). Extra sbatch options can follow the script name:
#   bash pace/submit.sh rq2_score.sbatch --export=ALL,RQ2_RUN=rq2_score,WF_SHARD=0,WF_N_SHARDS=4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

TARGET="${1:?usage: submit.sh <script>.sbatch [extra sbatch options]}"
shift
SBATCH_FILE="${SCRIPT_DIR}/${TARGET}"
[[ -f "${SBATCH_FILE}" ]] || { echo "missing ${SBATCH_FILE}"; exit 1; }

script_sets() { grep -Eq "^#SBATCH[[:space:]]+--$1([=[:space:]]|$)" "${SBATCH_FILE}"; }

EXTRA=()
[[ -n "${PACE_ACCOUNT:-}" ]] && EXTRA+=(--account="${PACE_ACCOUNT}")
[[ -n "${PACE_QOS:-}" ]] && EXTRA+=(--qos="${PACE_QOS}")
if [[ -n "${PACE_PARTITION:-}" ]] && ! script_sets partition; then EXTRA+=(--partition="${PACE_PARTITION}"); fi
if [[ -n "${PACE_GRES:-}" ]] && ! script_sets gres && ! script_sets partition; then EXTRA+=(--gres="${PACE_GRES}"); fi
[[ -n "${PACE_MAIL:-}" ]] && EXTRA+=(--mail-user="${PACE_MAIL}" --mail-type=END,FAIL)

mkdir -p "${PROJECT_ROOT}/logs"
sbatch ${EXTRA[@]+"${EXTRA[@]}"} "$@" "${SBATCH_FILE}"  # an empty array is "unbound" under set -u in old bash
