#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  printf '[ERROR] 未找到配置文件: %s\n' "${CONFIG_FILE}" >&2
  exit 1
fi

# 读取本地配置；脚本后续所有路径和评测参数都从这里派生。
source "${CONFIG_FILE}"

AER_DIR="${REPO_ROOT}/verl/recipe/aer"
VERL_DIR="${REPO_ROOT}/verl"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/save/data}"
INFERENCE_CKPT_DIR="${INFERENCE_CKPT_DIR:-${SAVE_DIR}/inference_checkpoints}"
EVAL_DIR="${SAVE_DIR}/eval"
LOG_ROOT="${FORMAL_EVAL_LOG_ROOT:-${SAVE_DIR}/run/formal_eval_logs/$(date '+%Y%m%d_%H%M%S')}"
MASTER_LOG="${LOG_ROOT}/master.log"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

bool_is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

on_error() {
  local exit_code="$?"
  local line_no="$1"
  printf '[ERROR] 脚本在第 %s 行失败，退出码 %s。主日志: %s\n' "${line_no}" "${exit_code}" "${MASTER_LOG}" >&2
  exit "${exit_code}"
}

trap 'on_error ${LINENO}' ERR

apply_runtime_env() {
  export HF_ENDPOINT="${HF_ENDPOINT:-}"
  export HF_HOME="${SAVE_DIR}/hf_home"
  export HF_DATASETS_CACHE="${SAVE_DIR}/hf_datasets_cache"
  export HUGGINGFACE_HUB_CACHE="${SAVE_DIR}/hf_hub_cache"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export TOKENIZERS_PARALLELISM=true
  export NCCL_DEBUG=WARN
  export VLLM_LOGGING_LEVEL=WARN
  export RAY_TMPDIR="${RAY_TMPDIR:-${HOME}/rt}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-32}"
  export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-32}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  export WANDB_PROJECT="${WANDB_PROJECT:-AER}"

  mkdir -p "${RAY_TMPDIR}" "${EVAL_DIR}"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
  fi
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    export PIP_INDEX_URL
  fi
  if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    export PIP_EXTRA_INDEX_URL
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
  fi
}

activate_conda() {
  command -v conda >/dev/null 2>&1 || die "未找到 conda，无法进入与训练相同的环境 ${CONDA_ENV_NAME:-aer}"
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-aer}"
}

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  local idx
  IFS=',' read -r -a out_ref <<< "${raw}"
  for idx in "${!out_ref[@]}"; do
    out_ref[$idx]="$(printf '%s' "${out_ref[$idx]}" | xargs)"
  done
}

join_by_comma() {
  local first=1
  local item
  for item in "$@"; do
    if [[ "${first}" -eq 1 ]]; then
      printf '%s' "${item}"
      first=0
    else
      printf ',%s' "${item}"
    fi
  done
}

logical_cuda_devices() {
  local count="$1"
  local devices=()
  local idx
  for ((idx = 0; idx < count; idx += 1)); do
    devices+=("cuda:${idx}")
  done
  join_by_comma "${devices[@]}"
}

metric_list_contains() {
  local metrics="$1"
  local needle="$2"
  [[ ",${metrics}," == *",${needle},"* || "${metrics}" == "all" ]]
}

append_metric_if_missing() {
  local metrics="$1"
  local metric="$2"
  if metric_list_contains "${metrics}" "${metric}"; then
    printf '%s' "${metrics}"
  else
    printf '%s,%s' "${metrics}" "${metric}"
  fi
}

tau_tag() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

gamma_tag() {
  tau_tag "$1"
}

baseline_naive_exp_name() {
  printf 'baseline-naive-calib-tau0-s%s' "${CALIBRATION_STEPS}"
}

baseline_entropy_exp_name() {
  printf 'baseline-entropy-mean-tau0-s%s' "${TOTAL_TRAINING_STEPS}"
}

tau_plan_path() {
  printf '%s/tau_plan_%s.csv' "${EVAL_DIR}" "$1"
}

gamma_best_env_path() {
  printf '%s/gamma_best_%s.env' "${EVAL_DIR}" "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}"
}

gamma_search_exp_name() {
  local algorithm="$1"
  local gamma="$2"
  local tau="$3"
  printf 'gamma-search-%s-g%s-tau%s-s%s' "${algorithm}" "$(gamma_tag "${gamma}")" "$(tau_tag "${tau}")" "${TOTAL_TRAINING_STEPS}"
}

main_aer_exp_name() {
  local algorithm="$1"
  local gamma="$2"
  local tau="$3"
  printf 'aer-%s-g%s-tau%s-s%s' "${algorithm}" "$(gamma_tag "${gamma}")" "$(tau_tag "${tau}")" "${TOTAL_TRAINING_STEPS}"
}

read_tau_value() {
  local tau_csv="$1"
  local column="$2"
  python - "$tau_csv" "$column" <<'PY'
import csv
import sys

path, column = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    row = next(csv.DictReader(f))
value = row.get(column)
if value is None or value == "":
    raise SystemExit(f"{path} 缺少列 {column} 或该列为空")
print(value)
PY
}

read_tau_for_gamma() {
  local algorithm="$1"
  local gamma="$2"
  local tau_csv
  tau_csv="$(tau_plan_path "${algorithm}")"
  python - "${tau_csv}" "${algorithm}" "${gamma}" <<'PY'
import csv
import math
import sys

path, algorithm, gamma = sys.argv[1:]
target = float(gamma)
with open(path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("algorithm") == algorithm and math.isclose(float(row["gamma"]), target, rel_tol=0.0, abs_tol=1e-12):
            print(row["tau"])
            break
    else:
        raise SystemExit(f"{path} 中没有 algorithm={algorithm}, gamma={gamma}")
PY
}

resolve_gamma_best() {
  if [[ -n "${GAMMA_BEST:-}" && "${GAMMA_BEST}" != "auto" ]]; then
    printf '%s' "${GAMMA_BEST}"
    return 0
  fi

  local gamma_env
  gamma_env="$(gamma_best_env_path)"
  [[ -s "${gamma_env}" ]] || die "GAMMA_BEST=auto 但未找到 ${gamma_env}，请先运行训练脚本完成 gamma 搜索，或在 config.env 中手动设置 GAMMA_BEST。"
  # shellcheck source=/dev/null
  source "${gamma_env}"
  printf '%s' "${GAMMA_BEST}"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  {
    printf '\n[%s] CMD:' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %q' "$@"
    printf '\n'
  } >> "${log_file}"

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN: 跳过命令，详见 ${log_file}"
    return 0
  fi

  "$@" > >(tee -a "${log_file}") 2> >(tee -a "${log_file}" >&2)
}

hf_model_is_ready() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  if ! compgen -G "${model_dir}/*.safetensors" >/dev/null && ! compgen -G "${model_dir}/*.bin" >/dev/null; then
    return 1
  fi
  [[ -f "${model_dir}/tokenizer.json" || -f "${model_dir}/tokenizer.model" || -f "${model_dir}/vocab.json" ]] || return 1
}

inference_checkpoint_path() {
  local exp_name="$1"
  local step="$2"
  printf '%s/%s/global_step_%s' "${INFERENCE_CKPT_DIR}" "${exp_name}" "${step}"
}

stop_ray_if_needed() {
  bool_is_true "${STOP_RAY_BETWEEN_RUNS:-1}" || return 0
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

hf_download() {
  local repo_id="$1"
  local revision="$2"
  local local_dir="$3"
  local repo_type="${4:-model}"

  mkdir -p "${local_dir}"
  log "下载/验证 ${repo_type}: ${repo_id} -> ${local_dir}"

  if command -v huggingface-cli >/dev/null 2>&1; then
    run_logged "${LOG_ROOT}/download.log" huggingface-cli download "${repo_id}" --revision "${revision}" --local-dir "${local_dir}" --repo-type "${repo_type}"
  elif command -v hf >/dev/null 2>&1; then
    run_logged "${LOG_ROOT}/download.log" hf download "${repo_id}" --revision "${revision}" --local-dir "${local_dir}"
  else
    run_logged "${LOG_ROOT}/download.log" python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${repo_id}', revision='${revision}', repo_type='${repo_type}', local_dir='${local_dir}')"
  fi
}

download_embedding_model_if_needed() {
  if ! metric_list_contains "${EVAL_RERUN_METRICS}" "semantic-cosine"; then
    return 0
  fi
  if [[ -d "${EMBEDDING_MODEL_PATH}" ]]; then
    return 0
  fi
  bool_is_true "${DOWNLOAD_MODELS:-0}" || die "评估包含 semantic-cosine，但 EMBEDDING_MODEL_PATH 不存在且 DOWNLOAD_MODELS!=1: ${EMBEDDING_MODEL_PATH}"
  hf_download "${EMBEDDING_MODEL_REPO}" "${EMBEDDING_MODEL_REVISION:-main}" "${EMBEDDING_MODEL_PATH}" "model"
}

generate_tau_plan_if_needed() {
  local algorithm="$1"
  local tau_csv="${EVAL_DIR}/tau_plan_${algorithm}.csv"
  local calib_exp="calib-${algorithm}-tau0-s${CALIBRATION_STEPS}"
  local calib_log="${SAVE_DIR}/run/logs/${calib_exp}.log"

  if [[ -s "${tau_csv}" ]]; then
    return 0
  fi
  if [[ ! -s "${calib_log}" ]]; then
    return 1
  fi

  log "tau_plan 不存在，尝试从校准日志生成: ${algorithm}"
  run_logged "${LOG_ROOT}/tau_plan_${algorithm}.log" \
    python "${AER_DIR}/eval/evaluate_aer.py" tau-plan \
      --input "${calib_log}" \
      --algorithm "${algorithm}" \
      --output "${tau_csv}"
}

add_experiment() {
  local exp_name="$1"
  local existing
  [[ -n "${exp_name}" ]] || return 0
  for existing in "${FORMAL_EXPS[@]:-}"; do
    if [[ "${existing}" == "${exp_name}" ]]; then
      return 0
    fi
  done
  FORMAL_EXPS+=("${exp_name}")
}

discover_formal_experiments_from_dirs() {
  local algorithm="$1"
  local pattern="${SAVE_DIR}/checkpoints/aer-${algorithm}-"*"s${TOTAL_TRAINING_STEPS}"
  local path
  shopt -s nullglob
  for path in ${pattern}; do
    [[ -d "${path}" ]] || continue
    add_experiment "$(basename "${path}")"
  done
  shopt -u nullglob
}

collect_formal_experiments() {
  FORMAL_EXPS=()

  local exp_name
  for exp_name in ${FORMAL_EVAL_EXPERIMENT_NAMES:-}; do
    add_experiment "${exp_name}"
  done

  if bool_is_true "${FORMAL_EVAL_INCLUDE_BASELINE_NAIVE:-1}"; then
    add_experiment "$(baseline_naive_exp_name)"
  fi
  if bool_is_true "${FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY:-1}"; then
    add_experiment "$(baseline_entropy_exp_name)"
  fi

  local gamma_best=""
  if [[ "${FORMAL_EVAL_INCLUDE_GAMMA_SEARCH:-best}" != "0" || -n "${FORMAL_EVAL_MAIN_ALGORITHMS:-}" ]]; then
    gamma_best="$(resolve_gamma_best)"
  fi

  if [[ "${FORMAL_EVAL_INCLUDE_GAMMA_SEARCH:-best}" == "best" ]]; then
    local tau
    tau="$(read_tau_for_gamma "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma_best}")"
    add_experiment "$(gamma_search_exp_name "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma_best}" "${tau}")"
  elif [[ "${FORMAL_EVAL_INCLUDE_GAMMA_SEARCH:-best}" == "all" ]]; then
    local gamma
    for gamma in ${GAMMA_LIST}; do
      local tau
      tau="$(read_tau_for_gamma "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma}")"
      add_experiment "$(gamma_search_exp_name "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma}" "${tau}")"
    done
  fi

  local algorithm
  for algorithm in ${FORMAL_EVAL_MAIN_ALGORITHMS:-${MAIN_SIMILARITY_ALGORITHMS:-}}; do
    local tau
    tau="$(read_tau_for_gamma "${algorithm}" "${gamma_best}")"
    if [[ "${algorithm}" == "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" ]] && bool_is_true "${FORMAL_EVAL_REUSE_GAMMA_SEARCH_FOR_TARGET:-${REUSE_GAMMA_SEARCH_FOR_TARGET:-1}}"; then
      add_experiment "$(gamma_search_exp_name "${algorithm}" "${gamma_best}" "${tau}")"
    else
      add_experiment "$(main_aer_exp_name "${algorithm}" "${gamma_best}" "${tau}")"
    fi
  done

  for exp_name in ${FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES:-}; do
    add_experiment "${exp_name}"
  done

  if [[ "${#FORMAL_EXPS[@]}" -eq 0 ]]; then
    die "未配置任何正式评测实验。请检查 FORMAL_EVAL_* 配置。"
  fi
}

collect_explicit_formal_experiments() {
  FORMAL_EXPS=()

  local exp_name
  for exp_name in ${FORMAL_EVAL_EXPERIMENT_NAMES:-}; do
    add_experiment "${exp_name}"
  done
  for exp_name in ${FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES:-}; do
    add_experiment "${exp_name}"
  done

  if [[ "${#FORMAL_EXPS[@]}" -eq 0 ]]; then
    die "FORMAL_EVAL_EXPLICIT_ONLY=1 但未配置 FORMAL_EVAL_EXPERIMENT_NAMES。"
  fi
}

print_status() {
  cat <<EOF
REPO_ROOT=${REPO_ROOT}
CONFIG_FILE=${CONFIG_FILE}
SAVE_DIR=${SAVE_DIR}
DATA_DIR=${DATA_DIR}
INFERENCE_CKPT_DIR=${INFERENCE_CKPT_DIR}
LOG_ROOT=${LOG_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EVAL_GPUS=${EVAL_GPUS}
EVAL_CHECKPOINT_STEP=${EVAL_CHECKPOINT_STEP}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
FORMAL_EVAL_INCLUDE_BASELINE_NAIVE=${FORMAL_EVAL_INCLUDE_BASELINE_NAIVE:-1}
FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY=${FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY:-1}
FORMAL_EVAL_INCLUDE_GAMMA_SEARCH=${FORMAL_EVAL_INCLUDE_GAMMA_SEARCH:-best}
FORMAL_EVAL_MAIN_ALGORITHMS=${FORMAL_EVAL_MAIN_ALGORITHMS:-${MAIN_SIMILARITY_ALGORITHMS:-}}
TARGET_SIMILARITY_FOR_GAMMA_SEARCH=${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}
GAMMA_BEST=${GAMMA_BEST}
EVAL_RERUN_METRICS=${EVAL_RERUN_METRICS}
EVAL_RERUN_KS=${EVAL_RERUN_KS}
EVAL_SAMPLES_PER_PROMPT=${EVAL_SAMPLES_PER_PROMPT}
EVAL_SEMANTIC_DEVICE=${EVAL_SEMANTIC_DEVICE}
EVAL_OUTPUT_SUBDIR=${EVAL_OUTPUT_SUBDIR}
EOF
}

preflight() {
  local missing=()
  local exp_name
  local file

  if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    missing+=("Bash 版本过低，脚本需要 wait -n，要求 bash >= 4.3，当前为 ${BASH_VERSION}")
  fi

  [[ -d "${VERL_DIR}" ]] || missing+=("缺少 verl 目录: ${VERL_DIR}")
  [[ -d "${AER_DIR}" ]] || missing+=("缺少 AER recipe 目录: ${AER_DIR}")
  [[ -f "${AER_DIR}/eval/eval_from_model.py" ]] || missing+=("缺少模型推理评估入口: ${AER_DIR}/eval/eval_from_model.py")
  [[ -f "${AER_DIR}/eval/eval_from_jsonl.py" ]] || missing+=("缺少 JSONL 汇总评估入口: ${AER_DIR}/eval/eval_from_jsonl.py")

  if ! python -c "import pandas, torch, transformers, vllm, verl" >/dev/null 2>&1; then
    missing+=("当前 conda 环境缺少评估必需 Python 依赖，请先确认已按 run_experiments.sh setup 安装 verl、vllm、pandas、torch、transformers")
  fi
  if metric_list_contains "${EVAL_RERUN_METRICS}" "equational-diversity" && ! python - "${VERL_DIR}" "${EVAL_RERUN_METRICS}" <<'PY' >/dev/null 2>&1
import sys

sys.path.insert(0, sys.argv[1])
from recipe.aer.eval.metrics.registry import parse_metric_names
from recipe.aer.eval.metrics.equational_diversity import equational_diversity

metrics = parse_metric_names(sys.argv[2])
if "equational-diversity" not in metrics:
    raise SystemExit("缺少 equational-diversity")
equational_diversity(["$x=1$", "$y=2$"])
PY
  then
    missing+=("当前评估代码无法解析或执行 equational-diversity，请检查 metrics.registry 与 metrics.equational_diversity")
  fi

  for file in "${VAL_FILES[@]}"; do
    [[ -f "${file}" ]] || missing+=("缺少验证数据: ${file}")
  done

  if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
    missing+=("至少需要 1 张 GPU 做正式模型推理，但 EVAL_GPUS 为空")
  fi

  if metric_list_contains "${EVAL_RERUN_METRICS}" "semantic-cosine"; then
    if [[ -z "${EMBEDDING_MODEL_PATH:-}" ]]; then
      missing+=("评估包含 semantic-cosine，但 EMBEDDING_MODEL_PATH 为空")
    elif [[ ! -d "${EMBEDDING_MODEL_PATH}" ]] && ! bool_is_true "${DOWNLOAD_MODELS:-0}"; then
      missing+=("评估包含 semantic-cosine，但 embedding 模型目录不存在且 DOWNLOAD_MODELS!=1: ${EMBEDDING_MODEL_PATH}")
    fi
    if ! python -c "import sentence_transformers" >/dev/null 2>&1; then
      missing+=("评估包含 semantic-cosine，但当前 conda 环境缺少 sentence-transformers")
    fi
    if [[ -z "${EVAL_SEMANTIC_DEVICE:-}" ]]; then
      missing+=("评估包含 semantic-cosine，但 EVAL_SEMANTIC_DEVICE 为空")
    fi
  fi

  for exp_name in "${FORMAL_EXPS[@]}"; do
    local inference_dir
    inference_dir="$(inference_checkpoint_path "${exp_name}" "${EVAL_CHECKPOINT_STEP}")"
    hf_model_is_ready "${inference_dir}" || missing+=("缺少可用于推理的 checkpoint: ${inference_dir}")
  done

  if [[ "${#missing[@]}" -ne 0 ]]; then
    printf '[ERROR] 预检失败，未开始合并或推理。请先处理以下问题：\n' >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

ensure_inference_checkpoint_ready() {
  local exp_name="$1"
  local inference_dir
  inference_dir="$(inference_checkpoint_path "${exp_name}" "${EVAL_CHECKPOINT_STEP}")"
  hf_model_is_ready "${inference_dir}" || die "缺少可用于推理的 checkpoint: ${inference_dir}"
}

eval_output_dir_for_exp() {
  local exp_name="$1"
  printf '%s/%s/%s' "${EVAL_DIR}" "${exp_name}" "${EVAL_OUTPUT_SUBDIR}"
}

eval_is_complete() {
  local output_dir="$1"
  [[ -s "${output_dir}/validation_summary.csv" && -s "${output_dir}/validation_per_prompt.csv" ]]
}

run_sharded_inference() {
  local exp_name="$1"
  local model_path="$2"
  local output_dir="$3"
  local exp_log_dir="${LOG_ROOT}/${exp_name}"
  local num_shards="${#GPU_IDS[@]}"
  local shard_index
  local pids=()
  local rollout_paths=()

  mkdir -p "${output_dir}" "${exp_log_dir}"

  for shard_index in "${!GPU_IDS[@]}"; do
    local gpu="${GPU_IDS[$shard_index]}"
    local log_path="${exp_log_dir}/shard_${shard_index}-of-${num_shards}.log"
    local shard_rollout_name
    printf -v shard_rollout_name "model_rollout_shard_%05d-of-%05d.jsonl" "${shard_index}" "${num_shards}"
    rollout_paths+=("${output_dir}/${shard_rollout_name}")

    log "启动 ${exp_name} shard ${shard_index}/${num_shards} on GPU ${gpu}，日志: ${log_path}"
    (
      cd "${AER_DIR}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      printf '[%s] CUDA_VISIBLE_DEVICES=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${CUDA_VISIBLE_DEVICES}"
      printf '[%s] CMD: python eval/eval_from_model.py --model-path %q --output-dir %q --skip-eval\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${model_path}" "${output_dir}"
      if bool_is_true "${DRY_RUN:-0}"; then
        exit 0
      fi
      python eval/eval_from_model.py \
        --model-path "${model_path}" \
        --val-files "${VAL_FILES[@]}" \
        --output-dir "${output_dir}" \
        --metrics "${EVAL_RERUN_METRICS}" \
        --ks "${EVAL_RERUN_KS}" \
        --samples-per-prompt "${EVAL_SAMPLES_PER_PROMPT}" \
        --semantic-model "${EMBEDDING_MODEL_PATH}" \
        --semantic-batch-size "${EVAL_SEMANTIC_BATCH_SIZE}" \
        --semantic-max-length "${EVAL_SEMANTIC_MAX_LENGTH}" \
        --rollout-save-batch-size "${EVAL_ROLLOUT_SAVE_BATCH_SIZE}" \
        --backend "${EVAL_BACKEND}" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
        --vllm-max-model-len "${EVAL_VLLM_MAX_MODEL_LEN}" \
        --vllm-max-num-seqs "${EVAL_VLLM_MAX_NUM_SEQS}" \
        --vllm-max-num-batched-tokens "${EVAL_VLLM_MAX_NUM_BATCHED_TOKENS}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --top-p "${EVAL_TOP_P}" \
        --top-k "${EVAL_TOP_K}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
        --seed "${EVAL_SEED}" \
        --step "${EVAL_CHECKPOINT_STEP}" \
        --num-shards "${num_shards}" \
        --shard-index "${shard_index}" \
        --skip-eval
    ) > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  local remaining="${#pids[@]}"
  while [[ "${remaining}" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
      log "检测到 ${exp_name} 的一个 shard 失败，继续等待其他 shard 结束"
    fi
    remaining=$((remaining - 1))
    log "${exp_name} 已有一个 shard 结束，剩余 ${remaining} 个"
  done

  if [[ "${failed}" -ne 0 ]]; then
    die "${exp_name} 至少一个 shard 失败，请检查 ${exp_log_dir}/shard_*-of-${num_shards}.log"
  fi

  if bool_is_true "${DRY_RUN:-0}"; then
    return 0
  fi

  local missing_rollouts=()
  local rollout_path
  for rollout_path in "${rollout_paths[@]}"; do
    [[ -s "${rollout_path}" ]] || missing_rollouts+=("${rollout_path}")
  done

  if [[ "${#missing_rollouts[@]}" -ne 0 ]]; then
    printf '[ERROR] %s 所有 shard 进程已结束，但缺少以下 rollout 文件，停止汇总评估：\n' "${exp_name}" >&2
    printf '  - %s\n' "${missing_rollouts[@]}" >&2
    exit 1
  fi

  log "${exp_name} 所有 shard 推理完成"
}

run_jsonl_eval() {
  local exp_name="$1"
  local output_dir="$2"
  local num_shards="${#GPU_IDS[@]}"
  local rollout_paths=()
  local shard_index

  for shard_index in "${!GPU_IDS[@]}"; do
    local shard_rollout_name
    printf -v shard_rollout_name "model_rollout_shard_%05d-of-%05d.jsonl" "${shard_index}" "${num_shards}"
    rollout_paths+=("${output_dir}/${shard_rollout_name}")
  done

  log "汇总评估 ${exp_name}，指标: ${EVAL_RERUN_METRICS}，semantic-cosine 设备: ${EVAL_SEMANTIC_DEVICE}"
  (
    cd "${AER_DIR}"
    export CUDA_VISIBLE_DEVICES="${EVAL_GPUS}"
    run_logged "${LOG_ROOT}/${exp_name}/jsonl_eval.log" \
      python eval/eval_from_jsonl.py \
        --input "${rollout_paths[@]}" \
        --output-dir "${output_dir}" \
        --metrics "${EVAL_RERUN_METRICS}" \
        --ks "${EVAL_RERUN_KS}" \
        --semantic-model "${EMBEDDING_MODEL_PATH}" \
        --semantic-device "${EVAL_SEMANTIC_DEVICE}" \
        --semantic-batch-size "${EVAL_SEMANTIC_BATCH_SIZE}" \
        --semantic-max-length "${EVAL_SEMANTIC_MAX_LENGTH}"
  )
}

run_one_experiment() {
  local exp_name="$1"
  local output_dir
  local model_path

  output_dir="$(eval_output_dir_for_exp "${exp_name}")"
  if eval_is_complete "${output_dir}" && ! bool_is_true "${FORCE_RERUN:-0}"; then
    log "评估结果已存在，跳过 ${exp_name}: ${output_dir}"
    return 0
  fi

  model_path="$(inference_checkpoint_path "${exp_name}" "${EVAL_CHECKPOINT_STEP}")"
  ensure_inference_checkpoint_ready "${exp_name}"
  stop_ray_if_needed
  run_sharded_inference "${exp_name}" "${model_path}" "${output_dir}"
  run_jsonl_eval "${exp_name}" "${output_dir}"
  stop_ray_if_needed
  log "完成 ${exp_name}: ${output_dir}"
}

main() {
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
  if bool_is_true "${FORMAL_EVAL_EXPLICIT_ONLY:-0}"; then
    collect_explicit_formal_experiments
  else
    collect_formal_experiments
  fi
  print_status
  printf 'FORMAL_EXPERIMENTS:\n'
  printf '  - %s\n' "${FORMAL_EXPS[@]}"
  preflight
  download_embedding_model_if_needed

  if bool_is_true "${DRY_RUN:-0}"; then
    log "DRY_RUN=1，预检完成，不执行合并、推理和评估"
    return 0
  fi

  local exp_name
  for exp_name in "${FORMAL_EXPS[@]}"; do
    run_one_experiment "${exp_name}"
  done

  log "所有正式实验第 ${EVAL_CHECKPOINT_STEP} 步 checkpoint 的重推理评估已完成。主日志: ${MASTER_LOG}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
