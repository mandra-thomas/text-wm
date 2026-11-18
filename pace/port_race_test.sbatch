#!/bin/bash
#SBATCH --job-name=wf-portrace
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --partition=coc-cpu
#SBATCH --output=logs/port_race_%j.out
#SBATCH --error=logs/port_race_%j.err
# Start three ALFWorld servers at once whose job ids map to the same first port (2026-11-18): each must
# end up on its own port with its own process listening. Exit 0 only if so.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source pace/_common.sh
out="logs/port_race_${SLURM_JOB_ID}"
mkdir -p "${out}"
for fake in 1 801 1601; do
  (
    export SLURM_JOB_ID="${fake}" ALFWORLD_PID_FILE="${out}/pid_${fake}"
    if start_alfworld_server; then echo "${ALFWORLD_PORT} $(cat "${ALFWORLD_PID_FILE}")" >"${out}/ok_${fake}"; fi
  ) >"${out}/start_${fake}.log" 2>&1 &
done
wait
cat "${out}"/start_*.log
ok=0
ports=$(cat "${out}"/ok_* 2>/dev/null | cut -d" " -f1 | sort -u | wc -l)
n=$(ls "${out}"/ok_* 2>/dev/null | wc -l)
for f in "${out}"/ok_*; do
  read -r port pid <"${f}"
  _owns_port "${pid}" "${port}" && echo "port ${port}: owned by ${pid}" || { echo "port ${port}: NOT owned by ${pid}"; ok=1; }
done
echo "started=${n} distinct_ports=${ports}"
[[ "${n}" == 3 && "${ports}" == 3 ]] || ok=1
for f in "${out}"/pid_*; do ALFWORLD_PID_FILE="${f}" stop_alfworld_server; done
sleep 3
for f in "${out}"/ok_*; do read -r port pid <"${f}"; _port_in_use "${port}" && { echo "port ${port} still open after stop"; ok=1; }; done
echo "RESULT $([[ ${ok} == 0 ]] && echo PASS || echo FAIL)"
exit "${ok}"
