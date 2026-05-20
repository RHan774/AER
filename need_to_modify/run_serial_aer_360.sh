#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 在这里显式设置要串行运行的 AER 训练实验。
# 格式: "实验名|相似度算法|tau"
# 注意：实验名会直接用于 save/checkpoints/<实验名> 和 save/validation/<实验名>。
###############################################################################
SERIAL_AER_EXPERIMENTS=(
  "token_match-tau0p426176-s360|token_match|0.426176"
  "ngram_overlap-tau0p264776-s360|ngram_overlap|0.264776"
  "ngram_overlap-tau0p356872-s360|ngram_overlap|0.356872"
  "ngram_overlap-tau0p448968-s360|ngram_overlap|0.448968"
  # "aer-semantic_embedding-custom-tau0p050-s360|semantic_embedding|0.050"
)

# 统一训练步数。按需求固定为 360；如确需临时调试，可用环境变量覆盖。
SERIAL_TRAINING_STEPS="${SERIAL_TRAINING_STEPS:-360}"

# 每个验证步都保存 checkpoint，这样最佳验证步一定有 checkpoint 可做全量评测。
SERIAL_TEST_FREQ="${SERIAL_TEST_FREQ:-12}"
SERIAL_SAVE_FREQ="${SERIAL_SAVE_FREQ:-${SERIAL_TEST_FREQ}}"

# 最佳 checkpoint 选择：对所有 val-core/**/acc/mean@16 取均值，均值最高者胜出。
SERIAL_BEST_K="${SERIAL_BEST_K:-16}"
SERIAL_BEST_CKPTS_TO_KEEP="${SERIAL_BEST_CKPTS_TO_KEEP:-3}"

# 后台把训练格式 checkpoint 转成推理格式 checkpoint。
# 训练格式 checkpoint 的保留数量默认沿用 config.env 中的 MAX_ACTOR_CKPT_TO_KEEP/MAX_CRITIC_CKPT_TO_KEEP；
# 如设置 SERIAL_MAX_CKPTS_TO_KEEP，则同时覆盖 actor/critic 的训练格式保留数量。
SERIAL_CONVERT_SAVED_CKPTS_TO_HF="${SERIAL_CONVERT_SAVED_CKPTS_TO_HF:-1}"
SERIAL_DELETE_TRAIN_CKPT_AFTER_HF="${SERIAL_DELETE_TRAIN_CKPT_AFTER_HF:-1}"
SERIAL_CKPT_CONVERT_INTERVAL_SECONDS="${SERIAL_CKPT_CONVERT_INTERVAL_SECONDS:-30}"

# 全量评测输出子目录前缀；最终目录会自动附加 step 和 config.env 中的 FORMAL_EVAL_OUTPUT_SUBDIR。
SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX="${SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX:-formal_best}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export AER_CONFIG="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"

# shellcheck source=script_process_control.sh
source "${SCRIPT_DIR}/script_process_control.sh"

AER_SERIAL_OVERRIDE_NAMES=(
  SAVE_DIR
  WANDB_MODE
  WANDB_PROJECT
  WANDB_ENTITY
  CUDA_VISIBLE_DEVICES
  N_GPUS_PER_NODE
  MAX_ACTOR_CKPT_TO_KEEP
  MAX_CRITIC_CKPT_TO_KEEP
  SERIAL_MAX_CKPTS_TO_KEEP
  SERIAL_MAX_ACTOR_CKPTS_TO_KEEP
  SERIAL_MAX_CRITIC_CKPTS_TO_KEEP
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
  FORMAL_EVAL_GPUS
  FORMAL_EVAL_METRICS
  FORMAL_EVAL_KS
  FORMAL_EVAL_SAMPLES_PER_PROMPT
  FORMAL_EVAL_OUTPUT_SUBDIR
  FORMAL_EVAL_SEMANTIC_DEVICE
  FORMAL_EVAL_SEMANTIC_BATCH_SIZE
  FORMAL_EVAL_SEMANTIC_MAX_LENGTH
  FORMAL_EVAL_ROLLOUT_SAVE_BATCH_SIZE
  FORMAL_EVAL_BACKEND
  FORMAL_EVAL_GPU_MEMORY_UTILIZATION
  FORMAL_EVAL_VLLM_MAX_MODEL_LEN
  FORMAL_EVAL_VLLM_MAX_NUM_SEQS
  FORMAL_EVAL_VLLM_MAX_NUM_BATCHED_TOKENS
  FORMAL_EVAL_MAX_NEW_TOKENS
  FORMAL_EVAL_TEMPERATURE
  FORMAL_EVAL_TOP_P
  FORMAL_EVAL_TOP_K
  FORMAL_EVAL_SEED
  FORMAL_EVAL_FORCE_MERGE
  DRY_RUN
  FORCE_RERUN
)
aer_single_script_capture_env_overrides "${AER_SERIAL_OVERRIDE_NAMES[@]}"

# 只加载函数和配置；run_experiments.sh 被 source 时不会自动执行 main。
# shellcheck source=run_experiments.sh
source "${SCRIPT_DIR}/run_experiments.sh"
aer_single_script_restore_env_overrides

SERIAL_MAX_ACTOR_CKPTS_TO_KEEP="${SERIAL_MAX_ACTOR_CKPTS_TO_KEEP:-${SERIAL_MAX_CKPTS_TO_KEEP:-${MAX_ACTOR_CKPT_TO_KEEP:-0}}}"
SERIAL_MAX_CRITIC_CKPTS_TO_KEEP="${SERIAL_MAX_CRITIC_CKPTS_TO_KEEP:-${SERIAL_MAX_CKPTS_TO_KEEP:-${MAX_CRITIC_CKPT_TO_KEEP:-0}}}"
SERIAL_TRAIN_FORMAT_ACTOR_KEEP="${SERIAL_MAX_ACTOR_CKPTS_TO_KEEP}"
SERIAL_TRAIN_FORMAT_CRITIC_KEEP="${SERIAL_MAX_CRITIC_CKPTS_TO_KEEP}"

# 支持 `bash need_to_modify/run_serial_aer_360.sh stop` 停止当前串行队列及其子进程。
aer_single_script_init "serial_aer_360" "${BASH_SOURCE[0]}" "$@"

SERIAL_CKPT_WATCHER_PID=""
SERIAL_CKPT_WATCHER_STOP_FILE=""

serial_cleanup() {
  stop_checkpoint_watcher || true
  wait_checkpoint_watcher || true
  stop_eval_watcher || true
  wait_eval_bg || true
  aer_single_script_unregister || true
}

serial_signal_exit() {
  local code="$1"
  trap - INT TERM
  serial_cleanup
  exit "${code}"
}

trap serial_cleanup EXIT
trap 'serial_signal_exit 130' INT
trap 'serial_signal_exit 143' TERM

write_config_assignment() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "${key}" "${value}"
}

safe_file_tag() {
  aer_pc_safe_name "$1"
}

serial_validate_nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} 必须是非负整数，当前值: ${value}"
}

serial_hf_model_is_ready() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  if ! compgen -G "${model_dir}/*.safetensors" >/dev/null && ! compgen -G "${model_dir}/*.bin" >/dev/null; then
    return 1
  fi
  [[ -f "${model_dir}/tokenizer.json" || -f "${model_dir}/tokenizer.model" || -f "${model_dir}/vocab.json" ]] || return 1
}

serial_actor_checkpoint_is_complete() {
  local actor_dir="$1"
  [[ -d "${actor_dir}" ]] || return 1
  [[ -f "${actor_dir}/config.json" ]] || return 1
  [[ -f "${actor_dir}/tokenizer.json" || -f "${actor_dir}/tokenizer.model" || -f "${actor_dir}/vocab.json" ]] || return 1

  local rank0_file
  local rank0_name
  local world_size
  rank0_file="$(find "${actor_dir}" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' | head -n 1)"
  [[ -n "${rank0_file}" ]] || return 1
  rank0_name="$(basename "${rank0_file}")"
  [[ "${rank0_name}" =~ ^model_world_size_([0-9]+)_rank_0\.pt$ ]] || return 1
  world_size="${BASH_REMATCH[1]}"

  local rank
  for ((rank = 0; rank < world_size; rank += 1)); do
    [[ -s "${actor_dir}/model_world_size_${world_size}_rank_${rank}.pt" ]] || return 1
  done
}

serial_checkpoint_steps() {
  local ckpt_dir="$1"
  [[ -d "${ckpt_dir}" ]] || return 0

  local path
  local name
  local steps=()
  shopt -s nullglob
  for path in "${ckpt_dir}"/global_step_*; do
    [[ -d "${path}" ]] || continue
    name="$(basename "${path}")"
    [[ "${name}" =~ ^global_step_([0-9]+)$ ]] || continue
    steps+=("${BASH_REMATCH[1]}")
  done
  shopt -u nullglob

  if [[ "${#steps[@]}" -gt 0 ]]; then
    printf '%s\n' "${steps[@]}" | sort -n
  fi
}

serial_train_keep_count() {
  local actor_keep="${SERIAL_TRAIN_FORMAT_ACTOR_KEEP:-0}"
  local critic_keep="${SERIAL_TRAIN_FORMAT_CRITIC_KEEP:-0}"
  if (( critic_keep > actor_keep )); then
    printf '%s' "${critic_keep}"
  else
    printf '%s' "${actor_keep}"
  fi
}

serial_prune_training_checkpoints() {
  local exp_name="$1"
  bool_is_true "${SERIAL_DELETE_TRAIN_CKPT_AFTER_HF}" || return 0

  local keep_count
  keep_count="$(serial_train_keep_count)"
  (( keep_count > 0 )) || return 0

  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local steps=()
  local step
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    steps+=("${step}")
  done < <(serial_checkpoint_steps "${ckpt_dir}")

  local total="${#steps[@]}"
  (( total > keep_count )) || return 0

  local remove_count=$((total - keep_count))
  local idx
  local removed=0
  for ((idx = 0; idx < remove_count; idx += 1)); do
    step="${steps[$idx]}"
    if ! serial_hf_model_is_ready "${ckpt_dir}/global_step_${step}_hf"; then
      continue
    fi
    rm -rf -- "${ckpt_dir}/global_step_${step}"
    removed=$((removed + 1))
  done

  if (( removed > 0 )); then
    log "已删除 ${exp_name} 的 ${removed} 个旧训练格式 checkpoint；保留最近 ${keep_count} 个训练格式 checkpoint"
  fi
}

serial_merge_checkpoint_to_hf() {
  local exp_name="$1"
  local step="$2"
  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local actor_dir="${ckpt_dir}/global_step_${step}/actor"
  local hf_dir="${ckpt_dir}/global_step_${step}_hf"
  local lock_dir="${ckpt_dir}/.merge_global_step_${step}.running"
  local tmp_dir="${hf_dir}.tmp.$$"
  local status=0

  if serial_hf_model_is_ready "${hf_dir}"; then
    return 0
  fi
  if ! serial_actor_checkpoint_is_complete "${actor_dir}"; then
    return 0
  fi
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    return 0
  fi

  log "转换推理格式 checkpoint: ${exp_name} global_step_${step} -> ${hf_dir}"
  rm -rf -- "${tmp_dir}"
  if bool_is_true "${DRY_RUN:-0}"; then
    rm -rf -- "${lock_dir}"
    return 0
  fi

  (
    cd "${VERL_DIR}"
    export CUDA_VISIBLE_DEVICES=""
    python "${VERL_DIR}/scripts/model_merger.py" merge \
      --backend fsdp \
      --local_dir "${actor_dir}" \
      --target_dir "${tmp_dir}"
  ) || status=$?

  if [[ "${status}" -eq 0 ]] && serial_hf_model_is_ready "${tmp_dir}"; then
    rm -rf -- "${hf_dir}"
    mv "${tmp_dir}" "${hf_dir}"
    log "推理格式 checkpoint 已保存: ${hf_dir}"
  else
    rm -rf -- "${tmp_dir}"
    log "推理格式 checkpoint 转换失败: ${exp_name} global_step_${step}, status=${status}"
    status=1
  fi

  rm -rf -- "${lock_dir}"
  return "${status}"
}

serial_scan_checkpoints_once() {
  local exp_name="$1"
  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local step

  [[ -d "${ckpt_dir}" ]] || return 0
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    serial_merge_checkpoint_to_hf "${exp_name}" "${step}" || true
    serial_prune_training_checkpoints "${exp_name}" || true
  done < <(serial_checkpoint_steps "${ckpt_dir}")
}

start_checkpoint_watcher() {
  local exp_name="$1"
  bool_is_true "${SERIAL_CONVERT_SAVED_CKPTS_TO_HF}" || return 0
  bool_is_true "${DRY_RUN:-0}" && return 0
  if [[ -n "${SERIAL_CKPT_WATCHER_PID}" ]] && kill -0 "${SERIAL_CKPT_WATCHER_PID}" 2>/dev/null; then
    return 0
  fi

  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local log_file="${EVAL_LOG_DIR}/${exp_name}.checkpoint_convert.log"
  mkdir -p "${ckpt_dir}" "$(dirname "${log_file}")"

  SERIAL_CKPT_WATCHER_STOP_FILE="${TMP_DIR}/.checkpoint_watcher_stop_${exp_name}"
  rm -f "${SERIAL_CKPT_WATCHER_STOP_FILE}"

  (
    activate_conda
    apply_network_env
    export CUDA_VISIBLE_DEVICES=""
    log "[ckpt-watcher] 启动: ${exp_name}, interval=${SERIAL_CKPT_CONVERT_INTERVAL_SECONDS}s"
    while [[ ! -f "${SERIAL_CKPT_WATCHER_STOP_FILE}" ]]; do
      serial_scan_checkpoints_once "${exp_name}" || true
      sleep "${SERIAL_CKPT_CONVERT_INTERVAL_SECONDS}" || true
    done
    serial_scan_checkpoints_once "${exp_name}" || true
    log "[ckpt-watcher] 退出: ${exp_name}"
  ) >> "${log_file}" 2>&1 &
  SERIAL_CKPT_WATCHER_PID="$!"
  log "启动 checkpoint 转换 watcher (PID=${SERIAL_CKPT_WATCHER_PID}): ${exp_name}, 日志: ${log_file}"
}

stop_checkpoint_watcher() {
  if [[ -n "${SERIAL_CKPT_WATCHER_STOP_FILE}" ]]; then
    touch "${SERIAL_CKPT_WATCHER_STOP_FILE}"
  fi
}

wait_checkpoint_watcher() {
  if [[ -n "${SERIAL_CKPT_WATCHER_PID}" ]]; then
    wait "${SERIAL_CKPT_WATCHER_PID}" 2>/dev/null || true
  fi
  SERIAL_CKPT_WATCHER_PID=""
  SERIAL_CKPT_WATCHER_STOP_FILE=""
}

ensure_hf_checkpoint_ready() {
  local exp_name="$1"
  local step="$2"
  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local hf_dir="${ckpt_dir}/global_step_${step}_hf"
  local lock_dir="${ckpt_dir}/.merge_global_step_${step}.running"
  local waited=0

  serial_hf_model_is_ready "${hf_dir}" && return 0

  while [[ -d "${lock_dir}" && "${waited}" -lt 3600 ]]; do
    serial_hf_model_is_ready "${hf_dir}" && return 0
    sleep 10
    waited=$((waited + 10))
  done

  if ! bool_is_true "${DRY_RUN:-0}"; then
    activate_conda
    apply_network_env
  fi
  serial_merge_checkpoint_to_hf "${exp_name}" "${step}"
  serial_hf_model_is_ready "${hf_dir}" || die "最佳 checkpoint 的推理格式目录不可用: ${hf_dir}"
}

select_best_checkpoint_step() {
  local exp_name="$1"
  local metrics_csv="${EVAL_DIR}/${exp_name}/train_log/train_metrics.csv"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"
  local best_env="${EVAL_DIR}/${exp_name}/best_step_mean${SERIAL_BEST_K}.env"

  if [[ ! -s "${metrics_csv}" && -s "${log_file}" ]]; then
    export_train_log "${exp_name}" "${log_file}"
  fi
  [[ -s "${metrics_csv}" ]] || die "缺少训练指标 CSV，无法选择最佳 step: ${metrics_csv}"

  mkdir -p "$(dirname "${best_env}")"
  python - "${metrics_csv}" "${ckpt_dir}" "${SERIAL_BEST_K}" "${best_env}" "${SERIAL_BEST_CKPTS_TO_KEEP}" <<'PY'
import csv
import math
import re
import shlex
import sys
from pathlib import Path

metrics_csv, ckpt_dir, raw_k, best_env, raw_keep_count = sys.argv[1:]
k = str(int(raw_k))
keep_count = int(raw_keep_count)
if keep_count < 1:
    raise SystemExit(f"SERIAL_BEST_CKPTS_TO_KEEP 必须大于等于 1，当前值: {keep_count}")
ckpt_dir = Path(ckpt_dir)


def hf_model_is_ready(path):
    if not (path / "config.json").is_file():
        return False
    has_weight = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    has_tokenizer = any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    return has_weight and has_tokenizer


def to_float(value):
    if value in (None, ""):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


with open(metrics_csv, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

mean_suffix = f"/mean@{k}"
mean_pattern = re.compile(rf"^val-core/.+/acc{re.escape(mean_suffix)}$")
mean_cols = [name for name in fieldnames if mean_pattern.match(name)]

if not mean_cols:
    raise SystemExit(f"{metrics_csv} 缺少 val-core/**/acc/mean@{k} 列；请确认 VAL_KWARGS_N={k}")

candidates = []
for row in rows:
    step_value = to_float(row.get("step"))
    if step_value is None:
        continue
    step = int(step_value)
    actor_dir = ckpt_dir / f"global_step_{step}" / "actor"
    hf_dir = ckpt_dir / f"global_step_{step}_hf"
    if not actor_dir.is_dir() and not hf_model_is_ready(hf_dir):
        continue

    mean_values = [to_float(row.get(col)) for col in mean_cols]
    mean_values = [value for value in mean_values if value is not None]
    if not mean_values:
        continue

    mean_acc = sum(mean_values) / len(mean_values)
    score = mean_acc
    candidates.append((score, step, mean_acc, len(mean_values)))

if not candidates:
    raise SystemExit(f"{metrics_csv} 中没有同时具备训练格式/HF checkpoint 和 val-core/**/acc/mean@{k} 的 step")

candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
score, step, mean_acc, mean_count = candidates[0]
top_steps = [str(candidate[1]) for candidate in candidates[:keep_count]]
env_lines = {
    "BEST_STEP": str(step),
    "BEST_SCORE": f"{score:.12g}",
    "BEST_MEAN_ACC": f"{mean_acc:.12g}",
    "BEST_MEAN_COLUMN_COUNT": str(mean_count),
    "BEST_TOP_STEPS": " ".join(top_steps),
}
with open(best_env, "w", encoding="utf-8") as f:
    for key, value in env_lines.items():
        f.write(f"{key}={shlex.quote(value)}\n")
print(step)
PY
  # shellcheck source=/dev/null
  source "${best_env}"
  log "最佳 checkpoint: ${exp_name} step=${BEST_STEP}, val-core acc mean@${SERIAL_BEST_K}=${BEST_MEAN_ACC}, score=${BEST_SCORE}, top${SERIAL_BEST_CKPTS_TO_KEEP}=${BEST_TOP_STEPS}"
}

prune_checkpoints_to_best_steps() {
  local exp_name="$1"
  local keep_steps="$2"
  local ckpt_dir="${SAVE_DIR}/checkpoints/${exp_name}"

  [[ -d "${ckpt_dir}" ]] || die "checkpoint 目录不存在，无法清理: ${ckpt_dir}"
  [[ -n "${keep_steps}" ]] || die "缺少要保留的最佳 checkpoint step 列表: ${exp_name}"

  local -A keep_step_set=()
  local step
  for step in ${keep_steps}; do
    [[ "${step}" =~ ^[0-9]+$ ]] || die "非法 checkpoint step: ${step}"
    keep_step_set["${step}"]=1
  done

  local removed_count=0
  local kept_count=0
  local ckpt_path
  local ckpt_name
  for ckpt_path in "${ckpt_dir}"/global_step_*; do
    [[ -d "${ckpt_path}" ]] || continue
    ckpt_name="$(basename "${ckpt_path}")"
    [[ "${ckpt_name}" =~ ^global_step_([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"

    if [[ -n "${keep_step_set[${step}]:-}" ]]; then
      kept_count=$((kept_count + 1))
      continue
    fi

    rm -rf -- "${ckpt_path}"
    removed_count=$((removed_count + 1))
  done

  log "checkpoint 清理完成: ${exp_name}, 保留=${kept_count}(${keep_steps}), 删除=${removed_count}, 目录=${ckpt_dir}"
}

write_formal_eval_config() {
  local exp_name="$1"
  local step="$2"
  local config_path="$3"
  local base_output_subdir="${FORMAL_EVAL_OUTPUT_SUBDIR:-formal_pass${FORMAL_EVAL_SAMPLES_PER_PROMPT:-128}_sharded}"
  local output_subdir="${SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX}_step${step}_${base_output_subdir}"

  {
    printf 'source %q\n' "${AER_CONFIG}"
    write_config_assignment "TOTAL_TRAINING_STEPS" "${SERIAL_TRAINING_STEPS}"
    write_config_assignment "FORMAL_EVAL_EXPERIMENT_NAMES" "${exp_name}"
    write_config_assignment "FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES" ""
    write_config_assignment "FORMAL_EVAL_INCLUDE_BASELINE_NAIVE" "0"
    write_config_assignment "FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY" "0"
    write_config_assignment "FORMAL_EVAL_INCLUDE_GAMMA_SEARCH" "0"
    write_config_assignment "FORMAL_EVAL_MAIN_ALGORITHMS" ""
    write_config_assignment "MAIN_SIMILARITY_ALGORITHMS" ""
    write_config_assignment "FORMAL_EVAL_CHECKPOINT_STEP" "${step}"
    write_config_assignment "FORMAL_EVAL_OUTPUT_SUBDIR" "${output_subdir}"
    write_config_assignment "FORMAL_EVAL_GPUS" "${FORMAL_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES}}"
    write_config_assignment "FORMAL_EVAL_METRICS" "${FORMAL_EVAL_METRICS}"
    write_config_assignment "FORMAL_EVAL_KS" "${FORMAL_EVAL_KS}"
    write_config_assignment "FORMAL_EVAL_SAMPLES_PER_PROMPT" "${FORMAL_EVAL_SAMPLES_PER_PROMPT}"
    write_config_assignment "FORMAL_EVAL_SEMANTIC_DEVICE" "${FORMAL_EVAL_SEMANTIC_DEVICE:-}"
    write_config_assignment "FORMAL_EVAL_SEMANTIC_BATCH_SIZE" "${FORMAL_EVAL_SEMANTIC_BATCH_SIZE}"
    write_config_assignment "FORMAL_EVAL_SEMANTIC_MAX_LENGTH" "${FORMAL_EVAL_SEMANTIC_MAX_LENGTH}"
    write_config_assignment "FORMAL_EVAL_ROLLOUT_SAVE_BATCH_SIZE" "${FORMAL_EVAL_ROLLOUT_SAVE_BATCH_SIZE}"
    write_config_assignment "FORMAL_EVAL_BACKEND" "${FORMAL_EVAL_BACKEND}"
    write_config_assignment "FORMAL_EVAL_GPU_MEMORY_UTILIZATION" "${FORMAL_EVAL_GPU_MEMORY_UTILIZATION}"
    write_config_assignment "FORMAL_EVAL_VLLM_MAX_MODEL_LEN" "${FORMAL_EVAL_VLLM_MAX_MODEL_LEN}"
    write_config_assignment "FORMAL_EVAL_VLLM_MAX_NUM_SEQS" "${FORMAL_EVAL_VLLM_MAX_NUM_SEQS}"
    write_config_assignment "FORMAL_EVAL_VLLM_MAX_NUM_BATCHED_TOKENS" "${FORMAL_EVAL_VLLM_MAX_NUM_BATCHED_TOKENS}"
    write_config_assignment "FORMAL_EVAL_MAX_NEW_TOKENS" "${FORMAL_EVAL_MAX_NEW_TOKENS}"
    write_config_assignment "FORMAL_EVAL_TEMPERATURE" "${FORMAL_EVAL_TEMPERATURE}"
    write_config_assignment "FORMAL_EVAL_TOP_P" "${FORMAL_EVAL_TOP_P}"
    write_config_assignment "FORMAL_EVAL_TOP_K" "${FORMAL_EVAL_TOP_K}"
    write_config_assignment "FORMAL_EVAL_SEED" "${FORMAL_EVAL_SEED}"
    write_config_assignment "FORMAL_EVAL_FORCE_MERGE" "${FORMAL_EVAL_FORCE_MERGE}"
    write_config_assignment "DRY_RUN" "${DRY_RUN:-0}"
    write_config_assignment "FORCE_RERUN" "${FORCE_RERUN:-0}"
  } > "${config_path}"
}

run_formal_eval_for_best_step() {
  local exp_name="$1"
  local step="$2"
  local config_path="${TMP_DIR}/serial_formal_$(safe_file_tag "${exp_name}")_step${step}.env"

  write_formal_eval_config "${exp_name}" "${step}" "${config_path}"
  log "开始全量评测最佳 checkpoint: ${exp_name} global_step_${step}"
  AER_CONFIG="${config_path}" bash "${SCRIPT_DIR}/run_eval_formal_checkpoints.sh"
}

parse_experiment_spec() {
  local spec="$1"
  local -n out_name_ref="$2"
  local -n out_algorithm_ref="$3"
  local -n out_tau_ref="$4"

  IFS='|' read -r out_name_ref out_algorithm_ref out_tau_ref <<< "${spec}"
  [[ -n "${out_name_ref}" ]] || die "实验配置缺少实验名: ${spec}"
  [[ -n "${out_algorithm_ref}" ]] || die "实验配置缺少相似度算法: ${spec}"
  [[ -n "${out_tau_ref}" ]] || die "实验配置缺少 tau: ${spec}"
}

run_serial_experiment() {
  local spec="$1"
  local exp_name=""
  local algorithm=""
  local tau=""

  parse_experiment_spec "${spec}" exp_name algorithm tau
  log "串行队列启动训练: exp=${exp_name}, algorithm=${algorithm}, tau=${tau}, steps=${SERIAL_TRAINING_STEPS}"
  start_checkpoint_watcher "${exp_name}"

  local saved_max_actor="${MAX_ACTOR_CKPT_TO_KEEP:-0}"
  local saved_max_critic="${MAX_CRITIC_CKPT_TO_KEEP:-0}"
  if bool_is_true "${SERIAL_CONVERT_SAVED_CKPTS_TO_HF}"; then
    # 由后台转换器在 HF checkpoint 落盘后清理训练格式 checkpoint，避免训练进程先轮转掉尚未转换的旧 step。
    MAX_ACTOR_CKPT_TO_KEEP=0
    MAX_CRITIC_CKPT_TO_KEEP=0
  fi
  run_experiment "${exp_name}" "${algorithm}" "${tau}" "${SERIAL_TRAINING_STEPS}" "0.0" "" "" "1.0"
  MAX_ACTOR_CKPT_TO_KEEP="${saved_max_actor}"
  MAX_CRITIC_CKPT_TO_KEEP="${saved_max_critic}"
  wait_eval_bg

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN=1，跳过最佳 checkpoint 选择和全量评测: ${exp_name}"
    stop_checkpoint_watcher
    wait_checkpoint_watcher
    return 0
  fi

  select_best_checkpoint_step "${exp_name}"
  stop_checkpoint_watcher
  wait_checkpoint_watcher
  ensure_hf_checkpoint_ready "${exp_name}" "${BEST_STEP}"
  run_formal_eval_for_best_step "${exp_name}" "${BEST_STEP}"
}

main() {
  if [[ "${#SERIAL_AER_EXPERIMENTS[@]}" -eq 0 ]]; then
    die "SERIAL_AER_EXPERIMENTS 为空。请在脚本开头按 '实验名|相似度算法|tau' 添加至少一个实验。"
  fi

  TOTAL_TRAINING_STEPS="${SERIAL_TRAINING_STEPS}"
  TEST_FREQ="${SERIAL_TEST_FREQ}"
  SAVE_FREQ="${SERIAL_SAVE_FREQ}"
  serial_validate_nonnegative_int "SERIAL_MAX_ACTOR_CKPTS_TO_KEEP" "${SERIAL_MAX_ACTOR_CKPTS_TO_KEEP}"
  serial_validate_nonnegative_int "SERIAL_MAX_CRITIC_CKPTS_TO_KEEP" "${SERIAL_MAX_CRITIC_CKPTS_TO_KEEP}"
  MAX_ACTOR_CKPT_TO_KEEP="${SERIAL_MAX_ACTOR_CKPTS_TO_KEEP}"
  MAX_CRITIC_CKPT_TO_KEEP="${SERIAL_MAX_CRITIC_CKPTS_TO_KEEP}"
  SERIAL_TRAIN_FORMAT_ACTOR_KEEP="${MAX_ACTOR_CKPT_TO_KEEP}"
  SERIAL_TRAIN_FORMAT_CRITIC_KEEP="${MAX_CRITIC_CKPT_TO_KEEP}"
  VAL_KWARGS_N="${SERIAL_BEST_K}"
  RUN_EVAL_AFTER_TRAIN=1

  check_inputs train
  prepare_dirs
  apply_network_env

  log "串行 AER 训练配置: steps=${TOTAL_TRAINING_STEPS}, test_freq=${TEST_FREQ}, save_freq=${SAVE_FREQ}, train_ckpt_keep(actor=${SERIAL_TRAIN_FORMAT_ACTOR_KEEP}, critic=${SERIAL_TRAIN_FORMAT_CRITIC_KEEP}), best_top=${SERIAL_BEST_CKPTS_TO_KEEP}"
  log "checkpoint 推理格式转换: enabled=${SERIAL_CONVERT_SAVED_CKPTS_TO_HF}, delete_train_after_hf=${SERIAL_DELETE_TRAIN_CKPT_AFTER_HF}, interval=${SERIAL_CKPT_CONVERT_INTERVAL_SECONDS}s"
  log "训练期间 CPU 评测指标: metrics=${AFTER_TRAIN_EVAL_METRICS}, ks=${AFTER_TRAIN_EVAL_KS}, semantic_device=${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}"

  local spec
  for spec in "${SERIAL_AER_EXPERIMENTS[@]}"; do
    run_serial_experiment "${spec}"
  done

  log "串行 AER 训练与最佳 checkpoint 全量评测全部完成"
}

main "$@"
