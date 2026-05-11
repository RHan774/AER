#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"
EXAMPLE_CONFIG="${SCRIPT_DIR}/config.example.env"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
else
  echo "[WARN] 未找到 ${CONFIG_FILE}，临时使用示例配置。正式运行请先复制并填写 config.env。"
  # shellcheck source=/dev/null
  source "${EXAMPLE_CONFIG}"
fi

AER_DIR="${REPO_ROOT}/verl/recipe/aer"
VERL_DIR="${REPO_ROOT}/verl"
STATE_DIR="${SAVE_DIR}/run/state"
LOG_DIR="${SAVE_DIR}/run/logs"
EVAL_DIR="${SAVE_DIR}/eval"
TMP_DIR="${SAVE_DIR}/tmp"

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

prepare_dirs() {
  mkdir -p "${SAVE_DIR}" "${STATE_DIR}" "${LOG_DIR}" "${EVAL_DIR}" "${TMP_DIR}"
  mkdir -p "${SAVE_DIR}/checkpoints" "${SAVE_DIR}/validation" "${SAVE_DIR}/data" "${SAVE_DIR}/models"
}

apply_network_env() {
  export HF_ENDPOINT="${HF_ENDPOINT:-}"
  export HF_HOME="${SAVE_DIR}/hf_home"
  export HF_DATASETS_CACHE="${SAVE_DIR}/hf_datasets_cache"
  export HUGGINGFACE_HUB_CACHE="${SAVE_DIR}/hf_hub_cache"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export TOKENIZERS_PARALLELISM=true
  export NCCL_DEBUG=WARN
  export VLLM_LOGGING_LEVEL=WARN
  export RAY_TMPDIR="${RAY_TMPDIR:-${HOME}/rt}"
  mkdir -p "${RAY_TMPDIR}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS}"
  export TORCH_NUM_THREADS="${TORCH_NUM_THREADS}"
  export WANDB_MODE="${WANDB_MODE}"
  export WANDB_PROJECT="${WANDB_PROJECT}"

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
  command -v conda >/dev/null 2>&1 || die "未找到 conda，请先安装 Miniconda/Anaconda。"
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
}

prepare_local_wheel_links() {
  local wheel_link_dir="/tmp/aer_wheels_${USER:-user}"
  mkdir -p "${wheel_link_dir}"
  find "${VERL_DIR}" -maxdepth 1 -type f -name '*.whl' -exec ln -sf {} "${wheel_link_dir}/" \;
  printf '%s' "${wheel_link_dir}"
}

make_env_file_without_prefix() {
  local output_file="$1"
  local find_links="$2"
  awk -v find_links="${find_links}" '
    /flash-attn==/ { next }
    /flashinfer-python==/ { next }
    /^prefix:/ { next }
    { print }
    $0 ~ /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ {
      print "      - --find-links=" find_links
    }
  ' "${REPO_ROOT}/environment.yml" > "${output_file}"
}

setup_env() {
  prepare_dirs
  apply_network_env
  command -v conda >/dev/null 2>&1 || die "未找到 conda，请先安装 Miniconda/Anaconda。"
  eval "$(conda shell.bash hook)"

  local env_file="${TMP_DIR}/environment.no-prefix.yml"
  local wheel_link_dir
  wheel_link_dir="$(prepare_local_wheel_links)"
  make_env_file_without_prefix "${env_file}" "${wheel_link_dir}"

  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    log "创建 conda 环境 ${CONDA_ENV_NAME}"
    conda env create -n "${CONDA_ENV_NAME}" -f "${env_file}"
  else
    log "conda 环境 ${CONDA_ENV_NAME} 已存在，跳过创建"
  fi

  conda activate "${CONDA_ENV_NAME}"
  
  log "单独安装 flash-attn 和 flashinfer 以避免 torch 依赖报错"
  python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --find-links="${wheel_link_dir}"
  python -m pip install flashinfer-python==0.2.2.post1+cu124torch2.6 -i https://flashinfer.ai/whl/cu124/torch2.6 --find-links="${wheel_link_dir}"

  log "安装本地 verl/AER 代码为 editable 包"
  python -m pip install --no-deps -e "${VERL_DIR}"

  install_optional_deps_if_needed
  force_match_current_env_versions
  setup_wandb
}

install_optional_deps_if_needed() {
  bool_is_true "${INSTALL_OPTIONAL_DEPS}" || return 0

  local deps=("scikit-learn==1.7.2")
  case " ${EXPERIMENT_ALGORITHMS} " in
    *" levenshtein "*) deps+=("rapidfuzz") ;;
  esac
  case " ${EXPERIMENT_ALGORITHMS} " in
    *" tfidf_cosine "*) ;;
  esac
  if [[ " ${EXPERIMENT_ALGORITHMS} " == *" semantic_embedding "* ]] || [[ " ${EVAL_METRICS:-}" == *"semantic-cosine"* ]]; then
    deps+=("sentence-transformers")
  fi

  if [[ ${#deps[@]} -gt 0 ]]; then
    log "安装当前实验队列需要的可选依赖: ${deps[*]}"
    python -m pip install "${deps[@]}"
  fi
}

force_match_current_env_versions() {
  bool_is_true "${MATCH_CURRENT_ENV_AFTER_CREATE:-1}" || return 0

  log "强制覆盖少量包版本，使新服务器环境与当前服务器保持一致"
  python -m pip install --no-deps --force-reinstall \
    fsspec==2026.3.0 \
    opentelemetry-api==1.40.0 \
    opentelemetry-sdk==1.40.0 \
    opentelemetry-semantic-conventions==0.61b0 \
    opentelemetry-exporter-prometheus==0.61b0
}

setup_wandb() {
  if [[ "${WANDB_MODE}" == "online" ]]; then
    [[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_MODE=online 但 WANDB_API_KEY 为空。请在 config.env 中填写，或把 WANDB_MODE 改成 offline。"
    log "登录 wandb"
    python -m wandb login --relogin "${WANDB_API_KEY}"
  else
    log "WANDB_MODE=${WANDB_MODE}，不执行 wandb 在线登录"
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
    huggingface-cli download "${repo_id}" --revision "${revision}" --local-dir "${local_dir}" --repo-type "${repo_type}"
  elif command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" --revision "${revision}" --local-dir "${local_dir}"
  else
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${repo_id}', revision='${revision}', repo_type='${repo_type}', local_dir='${local_dir}')"
  fi
}

download_models() {
  bool_is_true "${DOWNLOAD_MODELS}" || return 0
  activate_conda
  apply_network_env

  hf_download "${POLICY_MODEL_REPO}" "${POLICY_MODEL_REVISION}" "${MODEL_PATH}" "model"

  if bool_is_true "${DOWNLOAD_EMBEDDING_MODEL:-0}" || [[ " ${EXPERIMENT_ALGORITHMS} " == *" semantic_embedding "* ]] || [[ " ${EVAL_METRICS:-}" == *"semantic-cosine"* ]]; then
    hf_download "${EMBEDDING_MODEL_REPO}" "${EMBEDDING_MODEL_REVISION}" "${EMBEDDING_MODEL_PATH}" "model"
  fi
}

prepare_one_dataset() {
  local source="$1"
  local expected_file="$2"

  if [[ -f "${expected_file}" ]] && ! bool_is_true "${FORCE_RERUN}"; then
    log "已存在 ${expected_file}，跳过数据准备 ${source}"
    return 0
  fi

  mkdir -p "${SAVE_DIR}/data/${source}"
  log "准备数据集 ${source}"
  python "${AER_DIR}/src/data_preparation.py" \
    --data_dir "${DATA_REPO_PREFIX}" \
    --save_dir "${SAVE_DIR}/data" \
    --n_repeat "${DATA_REPEAT}" \
    --data_source "${source}"
}

prepare_data() {
  bool_is_true "${PREPARE_DATA}" || return 0
  activate_conda
  apply_network_env

  prepare_one_dataset "DigitalLearningGmbH/MATH-lighteval" "${SAVE_DIR}/data/DigitalLearningGmbH/MATH-lighteval/train.parquet"
  prepare_one_dataset "math-ai/math500" "${SAVE_DIR}/data/math-ai/math500/test_repeated.parquet"
  prepare_one_dataset "math-ai/amc23" "${SAVE_DIR}/data/math-ai/amc23/test_repeated.parquet"
  prepare_one_dataset "math-ai/aime24" "${SAVE_DIR}/data/math-ai/aime24/test_repeated.parquet"
  prepare_one_dataset "math-ai/aime25" "${SAVE_DIR}/data/math-ai/aime25/test_repeated.parquet"
}

run_smoke_tests() {
  bool_is_true "${RUN_SMOKE_TESTS}" || return 0
  activate_conda
  apply_network_env
  cd "${AER_DIR}"

  log "运行轻量测试：评测工具与默认相似度算法"
  python tests/test_evaluate_aer.py
  python tests/test_similarity.py --algorithm token_match
  python tests/test_similarity.py --algorithm ngram_overlap --n "${SIMILARITY_N}"
  python tests/test_similarity.py --algorithm simhash --n "${SIMILARITY_N}"
}

stop_ray_if_needed() {
  bool_is_true "${STOP_RAY_BETWEEN_RUNS}" || return 0
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

val_files_override() {
  printf "['%s','%s','%s','%s']" \
    "${SAVE_DIR}/data/math-ai/math500/test_repeated.parquet" \
    "${SAVE_DIR}/data/math-ai/amc23/test_repeated.parquet" \
    "${SAVE_DIR}/data/math-ai/aime24/test_repeated.parquet" \
    "${SAVE_DIR}/data/math-ai/aime25/test_repeated.parquet"
}

tau_tag() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

trainer_logger_override() {
  if [[ "${WANDB_MODE}" == "disabled" || "${WANDB_MODE}" == "off" ]]; then
    printf '[console]'
  else
    printf '[console,wandb]'
  fi
}

ray_cpu_override_arg() {
  if [[ -n "${RAY_NUM_CPUS:-}" ]]; then
    printf 'ray_init.num_cpus=%s' "${RAY_NUM_CPUS}"
  fi
}

run_experiment() {
  local exp_name="$1"
  local algorithm="$2"
  local tau="$3"
  local total_steps="$4"
  local entropy_coeff="${5:-0.0}"
  local marker="${STATE_DIR}/${exp_name}.done"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local val_files
  local trainer_logger
  local ray_cpu_arg

  if [[ -f "${marker}" ]] && ! bool_is_true "${FORCE_RERUN}"; then
    log "实验已完成，跳过: ${exp_name}"
    return 0
  fi

  val_files="$(val_files_override)"
  trainer_logger="$(trainer_logger_override)"
  ray_cpu_arg="$(ray_cpu_override_arg)"

  log "启动实验 ${exp_name}: algorithm=${algorithm}, tau=${tau}, entropy_coeff=${entropy_coeff}, steps=${total_steps}"

  local cmd=(
    python -m recipe.aer.src.main_ppo
    "trainer.resume_mode=auto"
    "trainer.resume_from_path=''"
    "data.train_files='${SAVE_DIR}/data/DigitalLearningGmbH/MATH-lighteval/train.parquet'"
    "data.val_files=${val_files}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.filter_overlong_prompts=False"
    "data.truncation=right"
    "actor_rollout_ref.model.path='${MODEL_PATH}'"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum"
    "actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_KWARGS_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_KWARGS_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_KWARGS_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_KWARGS_N}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "reward_model.reward_manager=aer"
    "algorithm.adv_estimator=grpo"
    "algorithm.tau=${tau}"
    "algorithm.similarity_algorithm=${algorithm}"
    "algorithm.similarity_params.n=${SIMILARITY_N}"
    "algorithm.similarity_params.model_name='${EMBEDDING_MODEL_PATH}'"
    "algorithm.similarity_params.batch_size=${SIMILARITY_BATCH_SIZE}"
    "algorithm.similarity_params.max_length=${SIMILARITY_MAX_LENGTH}"
    "algorithm.similarity_params.device=${SIMILARITY_DEVICE}"
    "algorithm.similarity_params.num_processes=${SIMILARITY_NUM_PROCESSES}"
    "trainer.total_epochs=100"
    "trainer.total_training_steps=${total_steps}"
    "trainer.project_name='${WANDB_PROJECT}'"
    "trainer.experiment_name='${exp_name}'"
    "trainer.logger=${trainer_logger}"
    "trainer.rollout_data_dir=''"
    "trainer.validation_data_dir='${SAVE_DIR}/validation/${exp_name}'"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP}"
    "trainer.default_local_dir='${SAVE_DIR}/checkpoints/${exp_name}'"
  )

  if [[ -n "${ray_cpu_arg}" ]]; then
    cmd+=("${ray_cpu_arg}")
  fi

  printf '\n[%s] CMD:' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
  printf ' %q' "${cmd[@]}" >> "${log_file}"
  printf '\n' >> "${log_file}"

  if bool_is_true "${DRY_RUN}"; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  stop_ray_if_needed
  activate_conda
  apply_network_env
  cd "${VERL_DIR}"

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  touch "${marker}"
  export_train_log "${exp_name}" "${log_file}"
  eval_jsonl_if_available "${exp_name}" "${total_steps}"
  stop_ray_if_needed
}

export_train_log() {
  local exp_name="$1"
  local log_file="$2"

  if [[ -s "${log_file}" ]]; then
    log "导出训练日志指标: ${exp_name}"
    cd "${AER_DIR}"
    python eval/evaluate_aer.py train-log \
      --input "${log_file}" \
      --output-dir "${EVAL_DIR}/${exp_name}/train_log" || true
  fi
}

eval_jsonl_if_available() {
  local exp_name="$1"
  local expected_step="${2:-}"
  bool_is_true "${RUN_EVAL_AFTER_TRAIN}" || return 0

  local input_dir="${SAVE_DIR}/validation/${exp_name}"
  local input_path="${input_dir}"
  if [[ ! -d "${input_dir}" ]]; then
    log "未找到 validation JSONL 目录，跳过离线评测: ${input_dir}"
    return 0
  fi

  if bool_is_true "${EVAL_LAST_STEP_ONLY:-1}"; then
    if [[ -n "${expected_step}" && -f "${input_dir}/${expected_step}.jsonl" ]]; then
      input_path="${input_dir}/${expected_step}.jsonl"
    else
      input_path="$(
        python - "${input_dir}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
numeric_paths = []
fallback_paths = []
for path in root.rglob("*.jsonl"):
    fallback_paths.append(path)
    try:
        numeric_paths.append((int(path.stem), str(path), path))
    except ValueError:
        pass

if numeric_paths:
    print(max(numeric_paths)[2])
elif fallback_paths:
    print(sorted(fallback_paths)[-1])
else:
    sys.exit(1)
PY
      )" || {
        log "未找到 validation JSONL 文件，跳过离线评测: ${input_dir}"
        return 0
      }
    fi
    log "从最后一步 validation JSONL 计算评测指标: ${exp_name}, input=${input_path}"
  else
    log "从全部 validation JSONL 计算评测指标: ${exp_name}, input=${input_path}"
  fi

  cd "${AER_DIR}"
  python eval/eval_from_jsonl.py \
    --input "${input_path}" \
    --output-dir "${EVAL_DIR}/${exp_name}/jsonl" \
    --metrics "${EVAL_METRICS}" \
    --ks "${EVAL_KS}" \
    --semantic-model "${EMBEDDING_MODEL_PATH}" \
    --semantic-device "${SIMILARITY_DEVICE}" || true
}

generate_tau_plan() {
  local algorithm="$1"
  local calib_exp="$2"
  local calib_log="${LOG_DIR}/${calib_exp}.log"
  local tau_csv="${EVAL_DIR}/tau_plan_${algorithm}.csv"

  [[ -s "${calib_log}" ]] || die "校准日志不存在，无法生成 tau: ${calib_log}"

  log "生成 tau 计划: ${algorithm}"
  cd "${AER_DIR}"
  python eval/evaluate_aer.py tau-plan \
    --input "${calib_log}" \
    --algorithm "${algorithm}" \
    --output "${tau_csv}"
}

read_tau_value() {
  local algorithm="$1"
  local column="$2"
  local tau_csv="${EVAL_DIR}/tau_plan_${algorithm}.csv"

  python -c "import csv; p='${tau_csv}'; c='${column}'; row=next(csv.DictReader(open(p, encoding='utf-8'))); print(row[c])"
}

run_aer_queue() {
  local algorithm
  for algorithm in ${EXPERIMENT_ALGORITHMS}; do
    local calib_exp="calib-${algorithm}-tau0-s${CALIBRATION_STEPS}"
    run_experiment "${calib_exp}" "${algorithm}" "0" "${CALIBRATION_STEPS}" "0.0"
    generate_tau_plan "${algorithm}" "${calib_exp}"

    local label
    for label in low mid high; do
      local column="tau_${label}"
      local tau
      local tau_name
      local exp_name
      tau="$(read_tau_value "${algorithm}" "${column}")"
      tau_name="$(tau_tag "${tau}")"
      exp_name="aer-${algorithm}-${label}-tau${tau_name}-s${TOTAL_TRAINING_STEPS}"
      run_experiment "${exp_name}" "${algorithm}" "${tau}" "${TOTAL_TRAINING_STEPS}" "0.0"
    done
  done
}

coeff_tag() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g; s/e-/em/g; s/e+/ep/g'
}

run_entropy_baselines() {
  [[ -n "${ENTROPY_BASELINE_COEFFS:-}" ]] || return 0

  local coeff
  for coeff in ${ENTROPY_BASELINE_COEFFS}; do
    local tag
    tag="$(coeff_tag "${coeff}")"
    run_experiment "baseline-entropy-grpo-c${tag}-s${TOTAL_TRAINING_STEPS}" "token_match" "0" "${TOTAL_TRAINING_STEPS}" "${coeff}"
  done
}

check_inputs() {
  local command="${1:-all}"
  [[ -d "${VERL_DIR}" ]] || die "未找到 verl 目录: ${VERL_DIR}"
  [[ -f "${REPO_ROOT}/environment.yml" ]] || die "未找到 environment.yml"
  if [[ "${command}" =~ ^(setup|train|all)$ && "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]] && ! bool_is_true "${DRY_RUN:-0}"; then
    die "请先复制 config.example.env 为 config.env，并填写 WANDB_API_KEY。"
  fi
}

print_status() {
  cat <<EOF
REPO_ROOT=${REPO_ROOT}
SAVE_DIR=${SAVE_DIR}
CONFIG_FILE=${CONFIG_FILE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EXPERIMENT_ALGORITHMS=${EXPERIMENT_ALGORITHMS}
CALIBRATION_STEPS=${CALIBRATION_STEPS}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
WANDB_MODE=${WANDB_MODE}
WANDB_PROJECT=${WANDB_PROJECT}
MODEL_PATH=${MODEL_PATH}
EOF
}

main() {
  local command="${1:-all}"
  check_inputs "${command}"
  prepare_dirs
  apply_network_env

  case "${command}" in
    status)
      print_status
      ;;
    setup)
      setup_env
      ;;
    assets)
      download_models
      prepare_data
      ;;
    test)
      run_smoke_tests
      ;;
    train)
      run_entropy_baselines
      run_aer_queue
      ;;
    all)
      setup_env
      download_models
      prepare_data
      run_smoke_tests
      run_entropy_baselines
      run_aer_queue
      ;;
    *)
      die "未知命令: ${command}。可选: status/setup/assets/test/train/all"
      ;;
  esac
}

main "$@"
