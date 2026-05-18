#!/usr/bin/env bash

# 单实验脚本进程控制：正常运行时记录入口 PID；stop 时停止该入口及其子进程。

aer_pc_log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

aer_pc_die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

aer_pc_safe_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

aer_pc_abs_path() {
  local path="$1"
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd)"
  printf '%s/%s' "${dir}" "$(basename "${path}")"
}

aer_pc_pid_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

aer_pc_pgid_for_pid() {
  local pid="$1"
  ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]'
}

aer_pc_current_pgid() {
  aer_pc_pgid_for_pid "$$"
}

aer_single_script_capture_env_overrides() {
  local name
  local backup_name

  export AER_SINGLE_SCRIPT_CAPTURED_ENV_NAMES="$*"
  for name in "$@"; do
    if [[ ${!name+x} ]]; then
      backup_name="AER_SINGLE_SCRIPT_ORIGINAL_${name}"
      printf -v "${backup_name}" '%s' "${!name}"
      export "${backup_name}"
    fi
  done
}

aer_single_script_restore_env_overrides() {
  local name
  local backup_name

  for name in ${AER_SINGLE_SCRIPT_CAPTURED_ENV_NAMES:-}; do
    backup_name="AER_SINGLE_SCRIPT_ORIGINAL_${name}"
    if [[ ${!backup_name+x} ]]; then
      printf -v "${name}" '%s' "${!backup_name}"
      export "${name}"
    fi
  done
}

aer_single_script_pid_file() {
  local namespace="$1"
  local script_path="$2"
  local script_name
  script_name="$(aer_pc_safe_name "$(basename "${script_path}")")"
  printf '%s/run/script_pids/%s/%s.pid' "${SAVE_DIR}" "${namespace}" "${script_name}"
}

aer_pc_collect_descendants() {
  local pid="$1"
  local child
  if command -v pgrep >/dev/null 2>&1; then
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
      aer_pc_collect_descendants "${child}"
    done
  fi
  printf '%s\n' "${pid}"
}

aer_pc_group_alive() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] || return 1
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -g "${pgid}" >/dev/null 2>&1
  else
    return 1
  fi
}

aer_pc_wait_until_stopped() {
  local pid="$1"
  local pgid="${2:-}"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -n "${pgid}" ]] && command -v pgrep >/dev/null 2>&1; then
      aer_pc_group_alive "${pgid}" || return 0
    else
      aer_pc_pid_alive "${pid}" || return 0
    fi
    sleep 1
  done
  return 1
}

aer_single_script_register() {
  local namespace="$1"
  local script_path="$2"
  local pid_file
  local old_pid=""
  local old_pgid=""
  local pgid

  pid_file="$(aer_single_script_pid_file "${namespace}" "${script_path}")"
  mkdir -p "$(dirname "${pid_file}")"

  if [[ -s "${pid_file}" ]]; then
    SCRIPT_PID=""
    SCRIPT_PGID=""
    # shellcheck source=/dev/null
    source "${pid_file}" || true
    old_pid="${SCRIPT_PID:-}"
    old_pgid="${SCRIPT_PGID:-}"
    if aer_pc_pid_alive "${old_pid}"; then
      aer_pc_die "脚本已在运行: $(basename "${script_path}") PID=${old_pid} PGID=${old_pgid}。如需停止，请运行: bash ${script_path} stop"
    fi
    aer_pc_log "清理过期运行记录: ${pid_file}"
    rm -f "${pid_file}"
  fi

  pgid="$(aer_pc_pgid_for_pid "$$")"
  {
    printf 'SCRIPT_PID=%q\n' "$$"
    printf 'SCRIPT_PGID=%q\n' "${pgid}"
    printf 'SCRIPT_PATH=%q\n' "${script_path}"
    printf 'SCRIPT_NAMESPACE=%q\n' "${namespace}"
    printf 'STARTED_AT=%q\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } > "${pid_file}"

  export AER_SINGLE_SCRIPT_PID_FILE="${pid_file}"
  aer_pc_log "记录单实验进程: PID=$$ PGID=${pgid} FILE=${pid_file}"
}

aer_single_script_unregister() {
  local pid_file="${AER_SINGLE_SCRIPT_PID_FILE:-}"
  [[ -n "${pid_file}" && -f "${pid_file}" ]] || return 0
  SCRIPT_PID=""
  # shellcheck source=/dev/null
  source "${pid_file}" || true
  if [[ "${SCRIPT_PID:-}" == "$$" ]]; then
    rm -f "${pid_file}"
  fi
}

aer_single_script_stop() {
  local namespace="$1"
  local script_path="$2"
  local pid_file
  local current_pgid
  local descendants

  pid_file="$(aer_single_script_pid_file "${namespace}" "${script_path}")"
  if [[ ! -s "${pid_file}" ]]; then
    aer_pc_log "未找到运行记录，可能没有正在运行的实验: ${pid_file}"
    return 0
  fi

  SCRIPT_PID=""
  SCRIPT_PGID=""
  SCRIPT_PATH=""
  # shellcheck source=/dev/null
  source "${pid_file}" || aer_pc_die "无法读取运行记录: ${pid_file}"

  if ! aer_pc_pid_alive "${SCRIPT_PID:-}"; then
    aer_pc_log "运行记录中的 PID 已不存在，清理记录: ${pid_file}"
    rm -f "${pid_file}"
    return 0
  fi

  current_pgid="$(aer_pc_current_pgid)"
  if [[ -n "${SCRIPT_PGID:-}" && "${SCRIPT_PGID}" == "${SCRIPT_PID}" && "${SCRIPT_PGID}" != "${current_pgid}" ]]; then
    aer_pc_log "停止实验进程组: $(basename "${script_path}") PID=${SCRIPT_PID} PGID=${SCRIPT_PGID}"
    kill -TERM -"${SCRIPT_PGID}" 2>/dev/null || true
    if ! aer_pc_wait_until_stopped "${SCRIPT_PID}" "${SCRIPT_PGID}"; then
      aer_pc_log "进程组未在 10 秒内退出，发送 SIGKILL: PGID=${SCRIPT_PGID}"
      kill -KILL -"${SCRIPT_PGID}" 2>/dev/null || true
    fi
  else
    aer_pc_log "停止实验进程树: $(basename "${script_path}") PID=${SCRIPT_PID}"
    descendants="$(aer_pc_collect_descendants "${SCRIPT_PID}" | sort -rn | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM ${descendants} 2>/dev/null || true
    if ! aer_pc_wait_until_stopped "${SCRIPT_PID}" ""; then
      aer_pc_log "进程树未在 10 秒内退出，发送 SIGKILL: PID=${SCRIPT_PID}"
      descendants="$(aer_pc_collect_descendants "${SCRIPT_PID}" | sort -rn | tr '\n' ' ')"
      # shellcheck disable=SC2086
      kill -KILL ${descendants} 2>/dev/null || true
    fi
  fi

  rm -f "${pid_file}"
  aer_pc_log "停止命令已完成: $(basename "${script_path}")"
}

aer_single_script_init() {
  local namespace="$1"
  local script_path="$2"
  shift 2

  script_path="$(aer_pc_abs_path "${script_path}")"

  if [[ "${1:-}" == "stop" ]]; then
    aer_single_script_stop "${namespace}" "${script_path}"
    exit 0
  fi

  if [[ "${AER_SINGLE_SCRIPT_MANAGED:-0}" != "1" ]]; then
    if command -v setsid >/dev/null 2>&1; then
      export AER_SINGLE_SCRIPT_MANAGED=1
      export AER_SINGLE_SCRIPT_MANAGED_PATH="${script_path}"
      exec setsid bash "${script_path}" "$@"
    fi
    aer_pc_log "未找到 setsid，将使用递归子进程停止作为兜底"
    export AER_SINGLE_SCRIPT_MANAGED=1
  fi

  aer_single_script_register "${namespace}" "${script_path}"
}
