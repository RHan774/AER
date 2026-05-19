#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEED_TO_MODIFY_DIR="$(cd "${THIS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NEED_TO_MODIFY_DIR}/.." && pwd)"
export AER_CONFIG="${AER_CONFIG:-${NEED_TO_MODIFY_DIR}/config.env}"

# shellcheck source=../script_process_control.sh
source "${NEED_TO_MODIFY_DIR}/script_process_control.sh"

AER_SINGLE_SCRIPT_OVERRIDE_NAMES=(
  DATA_DIR
  INFERENCE_CKPT_DIR
  WANDB_MODE
  WANDB_PROJECT
  WANDB_ENTITY
  CUDA_VISIBLE_DEVICES
  MODEL_PATH
  EMBEDDING_MODEL_PATH
  DRY_RUN
  FORCE_RERUN
  FORMAL_EVAL_CHECKPOINT_STEP
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
  aer_single_script_init "formal_eval" "${BASH_SOURCE[1]}" "$@"
  trap 'aer_single_script_unregister || true' EXIT
fi

# 只加载函数和配置；run_eval_formal_checkpoints.sh 被 source 时不会自动执行 main。
# shellcheck source=../run_eval_formal_checkpoints.sh
source "${NEED_TO_MODIFY_DIR}/run_eval_formal_checkpoints.sh"
aer_single_script_restore_env_overrides

single_formal_cleanup() {
  aer_single_script_unregister || true
}

single_formal_signal_exit() {
  local code="$1"
  trap - INT TERM
  single_formal_cleanup
  exit "${code}"
}

trap single_formal_cleanup EXIT
trap 'single_formal_signal_exit 130' INT
trap 'single_formal_signal_exit 143' TERM

refresh_formal_paths() {
  DATA_DIR="${DATA_DIR:-${REPO_ROOT}/save/data}"
  INFERENCE_CKPT_DIR="${INFERENCE_CKPT_DIR:-${SAVE_DIR}/inference_checkpoints}"
  EVAL_DIR="${SAVE_DIR}/eval"
}

init_single_formal_eval() {
  refresh_formal_paths
  apply_runtime_env

  EVAL_CHECKPOINT_STEP="${FORMAL_EVAL_CHECKPOINT_STEP:-${EVAL_CHECKPOINT_STEP:-${TOTAL_TRAINING_STEPS:-640}}}"
  EVAL_GPUS="${FORMAL_EVAL_GPUS:-${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES}}}"
  split_csv "${EVAL_GPUS}" GPU_IDS

  EVAL_RERUN_METRICS="${FORMAL_EVAL_METRICS:-${EVAL_RERUN_METRICS:-pass@k,first@1,distinct-2,self-bleu,semantic-cosine,equational-diversity}}"
  EVAL_RERUN_KS="${FORMAL_EVAL_KS:-${EVAL_RERUN_KS:-1,2,4,8,16,32,64,128}}"
  EVAL_SAMPLES_PER_PROMPT="${FORMAL_EVAL_SAMPLES_PER_PROMPT:-${EVAL_SAMPLES_PER_PROMPT:-128}}"
  EVAL_OUTPUT_SUBDIR="${FORMAL_EVAL_OUTPUT_SUBDIR:-${EVAL_OUTPUT_SUBDIR:-formal_pass${EVAL_SAMPLES_PER_PROMPT}_sharded}}"
  EVAL_SEMANTIC_DEVICE="${FORMAL_EVAL_SEMANTIC_DEVICE:-${EVAL_SEMANTIC_DEVICE:-$(logical_cuda_devices "${#GPU_IDS[@]}")}}"
  EVAL_SEMANTIC_BATCH_SIZE="${FORMAL_EVAL_SEMANTIC_BATCH_SIZE:-${EVAL_SEMANTIC_BATCH_SIZE:-128}}"
  EVAL_SEMANTIC_MAX_LENGTH="${FORMAL_EVAL_SEMANTIC_MAX_LENGTH:-${EVAL_SEMANTIC_MAX_LENGTH:-4096}}"
  EVAL_ROLLOUT_SAVE_BATCH_SIZE="${FORMAL_EVAL_ROLLOUT_SAVE_BATCH_SIZE:-${EVAL_ROLLOUT_SAVE_BATCH_SIZE:-8}}"
  EVAL_BACKEND="${FORMAL_EVAL_BACKEND:-${EVAL_BACKEND:-vllm}}"
  EVAL_GPU_MEMORY_UTILIZATION="${FORMAL_EVAL_GPU_MEMORY_UTILIZATION:-${EVAL_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.9}}}"
  EVAL_VLLM_MAX_MODEL_LEN="${FORMAL_EVAL_VLLM_MAX_MODEL_LEN:-${EVAL_VLLM_MAX_MODEL_LEN:-$((${MAX_PROMPT_LENGTH:-2048} + ${MAX_RESPONSE_LENGTH:-4096}))}}"
  EVAL_VLLM_MAX_NUM_SEQS="${FORMAL_EVAL_VLLM_MAX_NUM_SEQS:-${EVAL_VLLM_MAX_NUM_SEQS:-256}}"
  EVAL_VLLM_MAX_NUM_BATCHED_TOKENS="${FORMAL_EVAL_VLLM_MAX_NUM_BATCHED_TOKENS:-${EVAL_VLLM_MAX_NUM_BATCHED_TOKENS:-65536}}"
  EVAL_MAX_NEW_TOKENS="${FORMAL_EVAL_MAX_NEW_TOKENS:-${EVAL_MAX_NEW_TOKENS:-${MAX_RESPONSE_LENGTH:-4096}}}"
  EVAL_TEMPERATURE="${FORMAL_EVAL_TEMPERATURE:-${EVAL_TEMPERATURE:-${VAL_KWARGS_TEMPERATURE:-0.6}}}"
  EVAL_TOP_P="${FORMAL_EVAL_TOP_P:-${EVAL_TOP_P:-${VAL_KWARGS_TOP_P:-0.95}}}"
  EVAL_TOP_K="${FORMAL_EVAL_TOP_K:-${EVAL_TOP_K:-${VAL_KWARGS_TOP_K:-20}}}"
  EVAL_SEED="${FORMAL_EVAL_SEED:-${EVAL_SEED:-42}}"
  EVAL_FORCE_MERGE="${FORMAL_EVAL_FORCE_MERGE:-${EVAL_FORCE_MERGE:-0}}"

  VAL_FILES=(
    "${DATA_DIR}/math-ai/math500/test_repeated.parquet"
    "${DATA_DIR}/math-ai/amc23/test_repeated.parquet"
    "${DATA_DIR}/math-ai/aime24/test_repeated.parquet"
    "${DATA_DIR}/math-ai/aime25/test_repeated.parquet"
  )

  activate_conda
  apply_runtime_env
}

run_single_formal_eval() {
  local exp_name="$1"
  init_single_formal_eval
  FORMAL_EXPS=("${exp_name}")
  print_status
  printf 'FORMAL_EXPERIMENT:\n  - %s\n' "${exp_name}"
  preflight
  download_embedding_model_if_needed

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN=1，预检完成，不执行合并、推理和评估"
    return 0
  fi

  run_one_experiment "${exp_name}"
}

require_tau_plan() {
  local algorithm="$1"
  local tau_csv
  tau_csv="$(tau_plan_path "${algorithm}")"
  [[ -s "${tau_csv}" ]] || die "缺少 tau 表: ${tau_csv}。请先运行训练校准脚本，或手动准备该文件。"
}
