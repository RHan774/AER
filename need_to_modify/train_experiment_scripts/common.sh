#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEED_TO_MODIFY_DIR="$(cd "${THIS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NEED_TO_MODIFY_DIR}/.." && pwd)"
export AER_CONFIG="${AER_CONFIG:-${NEED_TO_MODIFY_DIR}/config.env}"

# shellcheck source=../script_process_control.sh
source "${NEED_TO_MODIFY_DIR}/script_process_control.sh"

AER_SINGLE_SCRIPT_OVERRIDE_NAMES=(
  SAVE_DIR
  WANDB_MODE
  WANDB_PROJECT
  WANDB_ENTITY
  CUDA_VISIBLE_DEVICES
  N_GPUS_PER_NODE
  MODEL_PATH
  EMBEDDING_MODEL_PATH
  SIMILARITY_DEVICE
  SIMILARITY_CUDA_VISIBLE_DEVICES
  SIMILARITY_NUM_PROCESSES
  AFTER_TRAIN_EVAL_METRICS
  AFTER_TRAIN_EVAL_KS
  AFTER_TRAIN_EVAL_SEMANTIC_DEVICE
  AFTER_TRAIN_EVAL_SEMANTIC_BATCH_SIZE
  AFTER_TRAIN_EVAL_SEMANTIC_MAX_LENGTH
  DRY_RUN
  FORCE_RERUN
)
aer_single_script_capture_env_overrides "${AER_SINGLE_SCRIPT_OVERRIDE_NAMES[@]}"

if [[ -f "${AER_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${AER_CONFIG}"
else
  printf '[ERROR] 未找到配置文件: %s\n' "${AER_CONFIG}" >&2
  exit 1
fi
aer_single_script_restore_env_overrides

# 单实验脚本支持 `bash xxx.sh stop` 停止该实验的全部进程。
if [[ -n "${BASH_SOURCE[1]:-}" ]]; then
  aer_single_script_init "train" "${BASH_SOURCE[1]}" "$@"
  trap 'aer_single_script_unregister || true' EXIT
fi

# 只加载函数和配置；run_experiments.sh 被 source 时不会自动执行 main。
# shellcheck source=../run_experiments.sh
source "${NEED_TO_MODIFY_DIR}/run_experiments.sh"
aer_single_script_restore_env_overrides

refresh_run_paths() {
  STATE_DIR="${SAVE_DIR}/run/state"
  LOG_DIR="${SAVE_DIR}/run/train_logs"
  EVAL_LOG_DIR="${SAVE_DIR}/run/eval_logs"
  EVAL_DIR="${SAVE_DIR}/eval"
  TMP_DIR="${SAVE_DIR}/tmp"
}

single_train_cleanup() {
  stop_eval_watcher || true
  wait_eval_bg || true
  aer_single_script_unregister || true
}

single_train_signal_exit() {
  local code="$1"
  trap - INT TERM
  single_train_cleanup
  exit "${code}"
}

trap single_train_cleanup EXIT
trap 'single_train_signal_exit 130' INT
trap 'single_train_signal_exit 143' TERM

run_single_training_experiment() {
  local exp_name="$1"
  local algorithm="$2"
  local tau="$3"
  local total_steps="$4"
  local entropy_coeff="${5:-0.0}"
  local exploration_algorithms="${6:-}"
  local delayed_algorithms="${7:-}"
  local delay_fraction="${8:-1.0}"

  refresh_run_paths
  check_inputs train
  prepare_dirs
  apply_network_env
  run_experiment "${exp_name}" "${algorithm}" "${tau}" "${total_steps}" "${entropy_coeff}" "${exploration_algorithms}" "${delayed_algorithms}" "${delay_fraction}"
  wait_eval_bg
}

require_tau_plan() {
  local algorithm="$1"
  local tau_csv
  tau_csv="$(tau_plan_path "${algorithm}")"
  [[ -s "${tau_csv}" ]] || die "缺少 tau 表: ${tau_csv}。请先运行 baseline naive 校准脚本，或手动准备该文件。"
}
