#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export AER_CONFIG="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"

if [[ ! -f "${AER_CONFIG}" ]]; then
  printf '[ERROR] 未找到配置文件: %s\n' "${AER_CONFIG}" >&2
  exit 1
fi

# 读取 SAVE_DIR、GPU 分配和其他运行参数。
# shellcheck source=/dev/null
source "${AER_CONFIG}"

LOG_DIR="${SAVE_DIR}/run/nohup"
mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ "${1:-}" == "stop" ]]; then
  log "停止 1_run.sh"
  bash "${SCRIPT_DIR}/1_run.sh" stop
  log "停止 2_run.sh"
  bash "${SCRIPT_DIR}/2_run.sh" stop
  exit 0
fi

if [[ "${1:-}" != "" ]]; then
  printf '[ERROR] 未知命令: %s。可选: stop\n' "$1" >&2
  exit 1
fi

timestamp="$(date '+%Y%m%d_%H%M%S')"
setup_log="${LOG_DIR}/setup_${timestamp}.log"
run1_log="${LOG_DIR}/1_run_${timestamp}.log"
run2_log="${LOG_DIR}/2_run_${timestamp}.log"

log "开始配置环境，日志: ${setup_log}"
bash "${SCRIPT_DIR}/run_experiments.sh" setup 2>&1 | tee "${setup_log}"

log "启动 1_run.sh，GPU=${RUN_1_CUDA_VISIBLE_DEVICES}，日志: ${run1_log}"
STOP_RAY_BETWEEN_RUNS=0 nohup bash "${SCRIPT_DIR}/1_run.sh" > "${run1_log}" 2>&1 &
run1_pid="$!"
printf '%s\n' "${run1_pid}" > "${LOG_DIR}/1_run_${timestamp}.pid"
ln -sfn "${run1_log}" "${LOG_DIR}/1_run_latest.log"

log "启动 2_run.sh，GPU=${RUN_2_CUDA_VISIBLE_DEVICES}，日志: ${run2_log}"
STOP_RAY_BETWEEN_RUNS=0 nohup bash "${SCRIPT_DIR}/2_run.sh" > "${run2_log}" 2>&1 &
run2_pid="$!"
printf '%s\n' "${run2_pid}" > "${LOG_DIR}/2_run_${timestamp}.pid"
ln -sfn "${run2_log}" "${LOG_DIR}/2_run_latest.log"

log "已提交两个 nohup 任务: 1_run PID=${run1_pid}, 2_run PID=${run2_pid}"
log "查看日志: tail -f ${LOG_DIR}/1_run_latest.log 或 tail -f ${LOG_DIR}/2_run_latest.log"
log "停止任务: bash ${SCRIPT_DIR}/run_all_nohup.sh stop"
