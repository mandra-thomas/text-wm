#!/usr/bin/env bash
# Submit an sbatch script with account/qos/gres from pace.env
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

TARGET="${1:?usage: submit.sh t0_all.sbatch|pilotA.sbatch|pilotB.sbatch}"
SBATCH_FILE="${SCRIPT_DIR}/${TARGET}"
[[ -f "${SBATCH_FILE}" ]] || { echo "missing ${SBATCH_FILE}"; exit 1; }

EXTRA=()
[[ -n "${PACE_ACCOUNT:-}" ]] && EXTRA+=(--account="${PACE_ACCOUNT}")
[[ -n "${PACE_QOS:-}" ]] && EXTRA+=(--qos="${PACE_QOS}")
[[ -n "${PACE_PARTITION:-}" ]] && EXTRA+=(--partition="${PACE_PARTITION}")
[[ -n "${PACE_GRES:-}" ]] && EXTRA+=(--gres="${PACE_GRES}")
[[ -n "${PACE_MAIL:-}" ]] && EXTRA+=(--mail-user="${PACE_MAIL}" --mail-type=END,FAIL)

mkdir -p "${PROJECT_ROOT}/logs"
sbatch "${EXTRA[@]}" "${SBATCH_FILE}"
