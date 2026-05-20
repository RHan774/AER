#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 在这里显式设置要串行运行的 AER 训练实验。
# 格式: "实验名|相似度算法|tau"
# 注意：实验名会直接用于 ${PRIMUS_OUTPUT_DIR}/checkpoints/<实验名> 和 ${PRIMUS_OUTPUT_DIR}/validation/<实验名>。
###############################################################################
SERIAL_AER_EXPERIMENTS=(
  "semantic_embedding-tau0p07242576-s360|semantic_embedding|0.07242576"
  "simhash-tau0p15517985-s360|simhash|0.15517985"
  "simhash-tau0p1258215-s360|simhash|0.1258215"
)

# 统一训练步数。按需求固定为 360；如确需临时调试，可用环境变量覆盖。
SERIAL_TRAINING_STEPS="${SERIAL_TRAINING_STEPS:-360}"

# 每个验证步都保存 checkpoint，这样最佳验证步一定有 checkpoint 可做全量评测。
SERIAL_TEST_FREQ="${SERIAL_TEST_FREQ:-12}"
SERIAL_SAVE_FREQ="${SERIAL_SAVE_FREQ:-${SERIAL_TEST_FREQ}}"
SERIAL_MAX_CKPTS_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP:-}"

# 最佳 checkpoint 选择：对所有 val-core/**/acc/mean@16 取均值，均值最高者胜出。
SERIAL_BEST_K="${SERIAL_BEST_K:-16}"

# 全量评测输出子目录前缀；最终目录会自动附加 step 和 config.env 中的 FORMAL_EVAL_OUTPUT_SUBDIR。
SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX="${SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX:-formal_best}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export AER_CONFIG="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"
SERIAL_GPU_PROFILE_PREFIX="RUN_2"

# shellcheck source=script_process_control.sh
source "${SCRIPT_DIR}/script_process_control.sh"

AER_SERIAL_OVERRIDE_NAMES=(
  DATA_DIR
  INFERENCE_CKPT_DIR
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
  STOP_RAY_BETWEEN_RUNS
  DRY_RUN
  FORCE_RERUN
)
aer_single_script_capture_env_overrides "${AER_SERIAL_OVERRIDE_NAMES[@]}"

# 只加载函数和配置；run_experiments.sh 被 source 时不会自动执行 main。
# shellcheck source=run_experiments.sh
source "${SCRIPT_DIR}/run_experiments.sh"
aer_single_script_restore_env_overrides

# 支持 `bash need_to_modify/2_run.sh stop` 停止当前串行队列及其子进程。
aer_single_script_init "serial_aer_360" "${BASH_SOURCE[0]}" "$@"

serial_cleanup() {
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

formal_eval_output_subdir_for_step() {
  local step="$1"
  local base_output_subdir="${FORMAL_EVAL_OUTPUT_SUBDIR:-formal_pass${FORMAL_EVAL_SAMPLES_PER_PROMPT:-128}_sharded}"
  printf '%s_step%s_%s' "${SERIAL_FORMAL_OUTPUT_SUBDIR_PREFIX}" "${step}" "${base_output_subdir}"
}

count_csv_items() {
  local raw="$1"
  local trimmed="${raw#[}"
  trimmed="${trimmed%]}"

  local -a items=()
  local item
  local count=0
  IFS=',' read -r -a items <<< "${trimmed}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s' "${count}"
}

apply_serial_gpu_profile() {
  local prefix="$1"
  local name
  local profile_name
  local backup_name
  local value

  for name in CUDA_VISIBLE_DEVICES N_GPUS_PER_NODE SIMILARITY_CUDA_VISIBLE_DEVICES SIMILARITY_NUM_PROCESSES FORMAL_EVAL_GPUS; do
    backup_name="AER_SINGLE_SCRIPT_ORIGINAL_${name}"
    if [[ ${!backup_name+x} ]]; then
      continue
    fi

    profile_name="${prefix}_${name}"
    if [[ ${!profile_name+x} ]]; then
      value="${!profile_name}"
      if [[ -n "${value}" ]]; then
        printf -v "${name}" '%s' "${value}"
        export "${name}"
      fi
    fi
  done
}

ensure_serial_gpu_config() {
  local script_label
  local train_gpu_count
  local formal_gpus
  local formal_gpu_count
  script_label="$(basename "${BASH_SOURCE[0]}")"
  train_gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES:-}")"
  formal_gpus="${FORMAL_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
  formal_gpu_count="$(count_csv_items "${formal_gpus}")"

  if [[ "${train_gpu_count}" -lt 1 ]]; then
    die "${script_label} 至少需要 1 张训练 GPU，但 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<空>} 没有解析到有效 GPU。"
  fi
  if [[ "${formal_gpu_count}" -lt 1 ]]; then
    die "${script_label} 至少需要 1 张正式评测 GPU，但 FORMAL_EVAL_GPUS=${formal_gpus:-<空>} 没有解析到有效 GPU。"
  fi

  if [[ ! "${N_GPUS_PER_NODE:-}" =~ ^[0-9]+$ ]]; then
    die "N_GPUS_PER_NODE 必须是整数，当前值: ${N_GPUS_PER_NODE:-<空>}"
  fi
  if [[ "${N_GPUS_PER_NODE}" -ne "${train_gpu_count}" ]]; then
    die "${script_label} 的 N_GPUS_PER_NODE=${N_GPUS_PER_NODE} 与训练 GPU 数 ${train_gpu_count} 不一致；请同步设置 ${SERIAL_GPU_PROFILE_PREFIX}_N_GPUS_PER_NODE=${train_gpu_count}。"
  fi

  if [[ "${SIMILARITY_DEVICE:-}" == cuda* ]]; then
    if [[ ! "${SIMILARITY_NUM_PROCESSES:-}" =~ ^[0-9]+$ ]]; then
      die "SIMILARITY_NUM_PROCESSES 必须是整数，当前值: ${SIMILARITY_NUM_PROCESSES:-<空>}"
    fi

    if [[ -n "${SIMILARITY_CUDA_VISIBLE_DEVICES:-}" && "${SIMILARITY_CUDA_VISIBLE_DEVICES}" != "null" ]]; then
      local similarity_gpu_count
      similarity_gpu_count="$(count_csv_items "${SIMILARITY_CUDA_VISIBLE_DEVICES}")"
      if [[ "${similarity_gpu_count}" -lt 1 ]]; then
        die "${script_label} 至少需要 1 张相似度计算 GPU，但 SIMILARITY_CUDA_VISIBLE_DEVICES=${SIMILARITY_CUDA_VISIBLE_DEVICES} 没有解析到有效 GPU。"
      fi
      if [[ "${SIMILARITY_NUM_PROCESSES}" -ne "${similarity_gpu_count}" ]]; then
        die "${script_label} 的 SIMILARITY_NUM_PROCESSES=${SIMILARITY_NUM_PROCESSES} 与相似度 GPU 数 ${similarity_gpu_count} 不一致。"
      fi
    elif [[ "${SIMILARITY_NUM_PROCESSES}" -ne "${train_gpu_count}" ]]; then
      die "${script_label} 未单独设置 SIMILARITY_CUDA_VISIBLE_DEVICES 时，SIMILARITY_NUM_PROCESSES=${SIMILARITY_NUM_PROCESSES} 应与训练 GPU 数 ${train_gpu_count} 一致。"
    fi
  fi
}

select_best_checkpoint_step() {
  local exp_name="$1"
  local metrics_csv="${EVAL_DIR}/${exp_name}/train_log/train_metrics.csv"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local inference_ckpt_dir="${INFERENCE_CKPT_DIR}/${exp_name}"
  local best_env="${EVAL_DIR}/${exp_name}/best_step_mean${SERIAL_BEST_K}.env"

  if [[ ! -s "${metrics_csv}" && -s "${log_file}" ]]; then
    export_train_log "${exp_name}" "${log_file}"
  fi
  [[ -s "${metrics_csv}" ]] || die "缺少训练指标 CSV，无法选择最佳 step: ${metrics_csv}"

  mkdir -p "$(dirname "${best_env}")"
  python - "${metrics_csv}" "${inference_ckpt_dir}" "${SERIAL_BEST_K}" "${best_env}" <<'PY'
import csv
import math
import re
import shlex
import sys
from pathlib import Path

metrics_csv, inference_ckpt_dir, raw_k, best_env = sys.argv[1:]
k = str(int(raw_k))
inference_ckpt_dir = Path(inference_ckpt_dir)


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


def hf_model_is_ready(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        return False
    return any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))


with open(metrics_csv, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

mean_suffix = f"/mean@{k}"
# 数据源名称可能包含斜杠，例如 math-ai/math500，因此这里从右侧识别 acc 指标段。
mean_pattern = re.compile(rf"^val-core/.+/acc{re.escape(mean_suffix)}$")
mean_cols = [name for name in fieldnames if mean_pattern.match(name)]

if not mean_cols:
    val_core_cols = [name for name in fieldnames if name.startswith("val-core/")]
    preview = ", ".join(val_core_cols[:8]) if val_core_cols else "<无 val-core 列>"
    raise SystemExit(f"{metrics_csv} 缺少 val-core/**/acc/mean@{k} 列；请确认 VAL_KWARGS_N={k}。现有 val-core 列示例: {preview}")

candidates = []
for row in rows:
    step_value = to_float(row.get("step"))
    if step_value is None:
        continue
    step = int(step_value)
    inference_dir = inference_ckpt_dir / f"global_step_{step}"
    if not hf_model_is_ready(inference_dir):
        continue

    mean_values = [to_float(row.get(col)) for col in mean_cols]
    mean_values = [value for value in mean_values if value is not None]
    if not mean_values:
        continue

    mean16 = sum(mean_values) / len(mean_values)
    score = mean16
    candidates.append((score, step, mean16, len(mean_values)))

if not candidates:
    raise SystemExit(f"{metrics_csv} 中没有同时具备推理 checkpoint 和 val-core/**/acc/mean@{k} 的 step")

candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
score, step, mean16, mean_count = candidates[0]
env_lines = {
    "BEST_STEP": str(step),
    "BEST_SCORE": f"{score:.12g}",
    "BEST_MEAN16": f"{mean16:.12g}",
    "BEST_MEAN_COLUMN_COUNT": str(mean_count),
}
with open(best_env, "w", encoding="utf-8") as f:
    for key, value in env_lines.items():
        f.write(f"{key}={shlex.quote(value)}\n")
print(step)
PY
  # shellcheck source=/dev/null
  source "${best_env}"
  log "最佳推理 checkpoint: ${exp_name} step=${BEST_STEP}, val-core mean@${SERIAL_BEST_K}=${BEST_MEAN16}, score=${BEST_SCORE}"
}

write_formal_eval_config() {
  local exp_name="$1"
  local step="$2"
  local config_path="$3"
  local formal_log_root="${4:-}"
  local output_subdir
  output_subdir="$(formal_eval_output_subdir_for_step "${step}")"

  {
    write_config_assignment "PRIMUS_OUTPUT_DIR" "${SAVE_DIR}"
    write_config_assignment "SAVE_DIR" "${SAVE_DIR}"
    write_config_assignment "DATA_DIR" "${DATA_DIR}"
    write_config_assignment "INFERENCE_CKPT_DIR" "${INFERENCE_CKPT_DIR}"
    printf 'source %q\n' "${AER_CONFIG}"
    write_config_assignment "PRIMUS_OUTPUT_DIR" "${SAVE_DIR}"
    write_config_assignment "SAVE_DIR" "${SAVE_DIR}"
    write_config_assignment "DATA_DIR" "${DATA_DIR}"
    write_config_assignment "INFERENCE_CKPT_DIR" "${INFERENCE_CKPT_DIR}"
    write_config_assignment "TOTAL_TRAINING_STEPS" "${SERIAL_TRAINING_STEPS}"
    write_config_assignment "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES}"
    write_config_assignment "STOP_RAY_BETWEEN_RUNS" "${STOP_RAY_BETWEEN_RUNS:-1}"
    write_config_assignment "FORMAL_EVAL_EXPERIMENT_NAMES" "${exp_name}"
    write_config_assignment "FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES" ""
    write_config_assignment "FORMAL_EVAL_EXPLICIT_ONLY" "1"
    write_config_assignment "FORMAL_EVAL_INCLUDE_BASELINE_NAIVE" "0"
    write_config_assignment "FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY" "0"
    write_config_assignment "FORMAL_EVAL_INCLUDE_GAMMA_SEARCH" "0"
    write_config_assignment "FORMAL_EVAL_MAIN_ALGORITHMS" ""
    write_config_assignment "FORMAL_EVAL_CHECKPOINT_STEP" "${step}"
    write_config_assignment "FORMAL_EVAL_OUTPUT_SUBDIR" "${output_subdir}"
    if [[ -n "${formal_log_root}" ]]; then
      write_config_assignment "FORMAL_EVAL_LOG_ROOT" "${formal_log_root}"
    fi
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

write_formal_eval_status() {
  local status_file="$1"
  local status="$2"
  local exp_name="$3"
  local step="$4"
  local output_dir="$5"
  local log_root="$6"
  local config_path="$7"
  local exit_code="${8:-}"

  mkdir -p "$(dirname "${status_file}")"
  {
    write_config_assignment "STATUS" "${status}"
    write_config_assignment "EXPERIMENT_NAME" "${exp_name}"
    write_config_assignment "BEST_STEP" "${step}"
    write_config_assignment "OUTPUT_DIR" "${output_dir}"
    write_config_assignment "FORMAL_EVAL_LOG_ROOT" "${log_root}"
    write_config_assignment "FORMAL_EVAL_MASTER_LOG" "${log_root}/master.log"
    write_config_assignment "FORMAL_EVAL_CONFIG" "${config_path}"
    write_config_assignment "UPDATED_AT" "$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ -n "${exit_code}" ]]; then
      write_config_assignment "EXIT_CODE" "${exit_code}"
    fi
  } > "${status_file}"
}

run_formal_eval_for_best_step() {
  local exp_name="$1"
  local step="$2"
  local config_path="${TMP_DIR}/serial_formal_$(safe_file_tag "${exp_name}")_step${step}.env"
  local output_subdir
  local output_dir
  local log_root
  local status_file

  output_subdir="$(formal_eval_output_subdir_for_step "${step}")"
  output_dir="${EVAL_DIR}/${exp_name}/${output_subdir}"
  log_root="${SAVE_DIR}/run/formal_eval_logs/serial_$(date '+%Y%m%d_%H%M%S')_$(safe_file_tag "${exp_name}")_step${step}"
  status_file="${EVAL_DIR}/${exp_name}/formal_best_step${step}.status.env"

  write_formal_eval_config "${exp_name}" "${step}" "${config_path}" "${log_root}"
  write_formal_eval_status "${status_file}" "running" "${exp_name}" "${step}" "${output_dir}" "${log_root}" "${config_path}"
  log "开始全量评测最佳 checkpoint: ${exp_name} global_step_${step}，状态: ${status_file}"
  if FORMAL_EVAL_LOG_ROOT="${log_root}" AER_CONFIG="${config_path}" bash "${SCRIPT_DIR}/run_eval_formal_checkpoints.sh"; then
    write_formal_eval_status "${status_file}" "succeeded" "${exp_name}" "${step}" "${output_dir}" "${log_root}" "${config_path}" "0"
    log "完成全量评测最佳 checkpoint: ${exp_name} global_step_${step}，结果: ${output_dir}"
  else
    local exit_code="$?"
    write_formal_eval_status "${status_file}" "failed" "${exp_name}" "${step}" "${output_dir}" "${log_root}" "${config_path}" "${exit_code}"
    log "全量评测失败: ${exp_name} global_step_${step}，退出码=${exit_code}，主日志: ${log_root}/master.log"
    return "${exit_code}"
  fi
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
  run_experiment "${exp_name}" "${algorithm}" "${tau}" "${SERIAL_TRAINING_STEPS}" "0.0" "" "" "1.0"
  wait_eval_bg

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN=1，跳过最佳 checkpoint 选择和全量评测: ${exp_name}"
    return 0
  fi

  select_best_checkpoint_step "${exp_name}"
  run_formal_eval_for_best_step "${exp_name}" "${BEST_STEP}"
}

main() {
  if [[ "${#SERIAL_AER_EXPERIMENTS[@]}" -eq 0 ]]; then
    die "SERIAL_AER_EXPERIMENTS 为空。请在脚本开头按 '实验名|相似度算法|tau' 添加至少一个实验。"
  fi

  TOTAL_TRAINING_STEPS="${SERIAL_TRAINING_STEPS}"
  TEST_FREQ="${SERIAL_TEST_FREQ}"
  SAVE_FREQ="${SERIAL_SAVE_FREQ}"
  if [[ -n "${SERIAL_MAX_CKPTS_TO_KEEP}" ]]; then
    MAX_ACTOR_CKPT_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP}"
    MAX_CRITIC_CKPT_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP}"
  fi
  VAL_KWARGS_N="${SERIAL_BEST_K}"
  RUN_EVAL_AFTER_TRAIN=1

  apply_serial_gpu_profile "${SERIAL_GPU_PROFILE_PREFIX}"
  check_inputs train
  ensure_serial_gpu_config
  prepare_dirs
  apply_network_env

  log "串行 AER 训练配置: steps=${TOTAL_TRAINING_STEPS}, test_freq=${TEST_FREQ}, save_freq=${SAVE_FREQ}, training_state_actor_keep=${MAX_ACTOR_CKPT_TO_KEEP}, training_state_critic_keep=${MAX_CRITIC_CKPT_TO_KEEP}"
  log "路径配置: output=${SAVE_DIR}, data=${DATA_DIR}, inference_ckpt=${INFERENCE_CKPT_DIR}"
  log "GPU运行配置: profile=${SERIAL_GPU_PROFILE_PREFIX}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, SIMILARITY_CUDA_VISIBLE_DEVICES=${SIMILARITY_CUDA_VISIBLE_DEVICES:-}, SIMILARITY_NUM_PROCESSES=${SIMILARITY_NUM_PROCESSES}, FORMAL_EVAL_GPUS=${FORMAL_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES}}"
  log "训练期间 CPU 评测指标: metrics=${AFTER_TRAIN_EVAL_METRICS}, ks=${AFTER_TRAIN_EVAL_KS}, semantic_device=${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}"

  local spec
  for spec in "${SERIAL_AER_EXPERIMENTS[@]}"; do
    run_serial_experiment "${spec}"
  done

  log "串行 AER 训练与最佳 checkpoint 全量评测全部完成"
}

main "$@"
