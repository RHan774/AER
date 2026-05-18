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
SERIAL_MAX_CKPTS_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP:-0}"

# 最佳 checkpoint 选择：对所有 val-core/*/{acc|reward}/mean@16 取均值，均值最高者胜出。
SERIAL_BEST_K="${SERIAL_BEST_K:-16}"
SERIAL_BEST_CKPTS_TO_KEEP="${SERIAL_BEST_CKPTS_TO_KEEP:-3}"

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

# 支持 `bash need_to_modify/run_serial_aer_360.sh stop` 停止当前串行队列及其子进程。
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
mean_pattern = re.compile(rf"^val-core/[^/]+/(acc|reward){re.escape(mean_suffix)}$")
mean_cols = [name for name in fieldnames if mean_pattern.match(name)]

if not mean_cols:
    raise SystemExit(f"{metrics_csv} 缺少 val-core/*/{{acc|reward}}/mean@{k} 列；请确认 VAL_KWARGS_N={k}")

candidates = []
for row in rows:
    step_value = to_float(row.get("step"))
    if step_value is None:
        continue
    step = int(step_value)
    actor_dir = ckpt_dir / f"global_step_{step}" / "actor"
    if not actor_dir.is_dir():
        continue

    mean_values = [to_float(row.get(col)) for col in mean_cols]
    mean_values = [value for value in mean_values if value is not None]
    if not mean_values:
        continue

    mean16 = sum(mean_values) / len(mean_values)
    score = mean16
    candidates.append((score, step, mean16, len(mean_values)))

if not candidates:
    raise SystemExit(f"{metrics_csv} 中没有同时具备 checkpoint 和 val-core/*/{{acc|reward}}/mean@{k} 的 step")

candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
score, step, mean16, mean_count = candidates[0]
top_steps = [str(candidate[1]) for candidate in candidates[:keep_count]]
env_lines = {
    "BEST_STEP": str(step),
    "BEST_SCORE": f"{score:.12g}",
    "BEST_MEAN16": f"{mean16:.12g}",
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
  log "最佳 checkpoint: ${exp_name} step=${BEST_STEP}, val-core mean@${SERIAL_BEST_K}=${BEST_MEAN16}, score=${BEST_SCORE}, 保留前${SERIAL_BEST_CKPTS_TO_KEEP}=${BEST_TOP_STEPS}"
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
  run_experiment "${exp_name}" "${algorithm}" "${tau}" "${SERIAL_TRAINING_STEPS}" "0.0" "" "" "1.0"
  wait_eval_bg

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN=1，跳过最佳 checkpoint 选择和全量评测: ${exp_name}"
    return 0
  fi

  select_best_checkpoint_step "${exp_name}"
  prune_checkpoints_to_best_steps "${exp_name}" "${BEST_TOP_STEPS}"
  run_formal_eval_for_best_step "${exp_name}" "${BEST_STEP}"
}

main() {
  if [[ "${#SERIAL_AER_EXPERIMENTS[@]}" -eq 0 ]]; then
    die "SERIAL_AER_EXPERIMENTS 为空。请在脚本开头按 '实验名|相似度算法|tau' 添加至少一个实验。"
  fi

  TOTAL_TRAINING_STEPS="${SERIAL_TRAINING_STEPS}"
  TEST_FREQ="${SERIAL_TEST_FREQ}"
  SAVE_FREQ="${SERIAL_SAVE_FREQ}"
  MAX_ACTOR_CKPT_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP}"
  MAX_CRITIC_CKPT_TO_KEEP="${SERIAL_MAX_CKPTS_TO_KEEP}"
  VAL_KWARGS_N="${SERIAL_BEST_K}"
  RUN_EVAL_AFTER_TRAIN=1

  check_inputs train
  prepare_dirs
  apply_network_env

  log "串行 AER 训练配置: steps=${TOTAL_TRAINING_STEPS}, test_freq=${TEST_FREQ}, save_freq=${SAVE_FREQ}, max_ckpts=${MAX_ACTOR_CKPT_TO_KEEP}, best_ckpts_to_keep=${SERIAL_BEST_CKPTS_TO_KEEP}"
  log "训练期间 CPU 评测指标: metrics=${AFTER_TRAIN_EVAL_METRICS}, ks=${AFTER_TRAIN_EVAL_KS}, semantic_device=${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}"

  local spec
  for spec in "${SERIAL_AER_EXPERIMENTS[@]}"; do
    run_serial_experiment "${spec}"
  done

  log "串行 AER 训练与最佳 checkpoint 全量评测全部完成"
}

main "$@"
