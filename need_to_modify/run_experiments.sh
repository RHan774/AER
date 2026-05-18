#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${AER_CONFIG:-${SCRIPT_DIR}/config.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
else
  printf '[ERROR] 未找到配置文件: %s\n' "${CONFIG_FILE}" >&2
  exit 1
fi

AER_DIR="${REPO_ROOT}/verl/recipe/aer"
VERL_DIR="${REPO_ROOT}/verl"
STATE_DIR="${SAVE_DIR}/run/state"
LOG_DIR="${SAVE_DIR}/run/train_logs"
EVAL_LOG_DIR="${SAVE_DIR}/run/eval_logs"
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
  mkdir -p "${SAVE_DIR}" "${STATE_DIR}" "${LOG_DIR}" "${EVAL_LOG_DIR}" "${EVAL_DIR}" "${TMP_DIR}"
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
  local configured_algorithms=" ${BASELINE_SIMILARITY_ALGORITHM:-} ${CALIBRATION_METRIC_ALGORITHMS:-} ${TARGET_SIMILARITY_FOR_GAMMA_SEARCH:-} ${MAIN_SIMILARITY_ALGORITHMS:-} ${EXTRA_SIMILARITY_ALGORITHMS:-} "
  case "${configured_algorithms}" in
    *" levenshtein "*) deps+=("rapidfuzz") ;;
  esac
  case "${configured_algorithms}" in
    *" tfidf_cosine "*) ;;
  esac
  if [[ "${configured_algorithms}" == *" semantic_embedding "* ]] || [[ " ${AFTER_TRAIN_EVAL_METRICS:-} ${FORMAL_EVAL_METRICS:-} " == *"semantic-cosine"* ]]; then
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

  local configured_algorithms=" ${CALIBRATION_METRIC_ALGORITHMS:-} ${TARGET_SIMILARITY_FOR_GAMMA_SEARCH:-} ${MAIN_SIMILARITY_ALGORITHMS:-} ${EXTRA_SIMILARITY_ALGORITHMS:-} "
  if bool_is_true "${DOWNLOAD_EMBEDDING_MODEL:-0}" || [[ "${configured_algorithms}" == *" semantic_embedding "* ]] || [[ " ${AFTER_TRAIN_EVAL_METRICS:-} ${FORMAL_EVAL_METRICS:-} " == *"semantic-cosine"* ]]; then
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

first_word() {
  local item
  for item in $1; do
    printf '%s' "${item}"
    return 0
  done
}

hydra_list_from_string() {
  local raw="${1:-}"
  local result="["
  local sep=""
  local item
  for item in ${raw}; do
    result+="${sep}${item}"
    sep=","
  done
  result+="]"
  printf '%s' "${result}"
}

metric_list_contains() {
  local metrics="$1"
  local needle="$2"
  [[ ",${metrics}," == *",${needle},"* || "${metrics}" == "all" ]]
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

gamma_tag() {
  tau_tag "$1"
}

coeff_tag() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g; s/e-/em/g; s/e+/ep/g'
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

# 后台 watcher：训练期间轮询 validation 目录，每发现新 JSONL 就立即评测该步。
EVAL_WATCHER_PID=""
EVAL_WATCHER_STOP_FILE=""

start_eval_watcher() {
  local exp_name="$1"
  bool_is_true "${RUN_EVAL_AFTER_TRAIN}" || return 0

  local input_dir="${SAVE_DIR}/validation/${exp_name}"
  local output_dir="${EVAL_DIR}/${exp_name}/jsonl"
  local eval_log_file="${EVAL_LOG_DIR}/${exp_name}.log"
  mkdir -p "${input_dir}" "${output_dir}"

  EVAL_WATCHER_STOP_FILE="${TMP_DIR}/.eval_watcher_stop_${exp_name}"
  rm -f "${EVAL_WATCHER_STOP_FILE}"

  (
    cd "${AER_DIR}"
    while [[ ! -f "${EVAL_WATCHER_STOP_FILE}" ]]; do
      local jsonl_file
      for jsonl_file in "${input_dir}"/*.jsonl; do
        [[ -f "${jsonl_file}" ]] || continue
        local step_name
        step_name="$(basename "${jsonl_file}" .jsonl)"
        local step_output_dir="${output_dir}/${step_name}"
        [[ -s "${step_output_dir}/validation_summary.csv" ]] && continue
        log "[watcher] 评测 ${exp_name} step ${step_name}"
        python eval/eval_from_jsonl.py \
          --input "${jsonl_file}" \
          --output-dir "${step_output_dir}" \
          --metrics "${AFTER_TRAIN_EVAL_METRICS}" \
          --ks "${AFTER_TRAIN_EVAL_KS}" \
          --semantic-model "${EMBEDDING_MODEL_PATH}" \
          --semantic-device "${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}" \
          --semantic-batch-size "${AFTER_TRAIN_EVAL_SEMANTIC_BATCH_SIZE:-32}" \
          --semantic-max-length "${AFTER_TRAIN_EVAL_SEMANTIC_MAX_LENGTH:-1024}" || true
      done
      sleep 30
    done
    # 停止信号后做最后一轮扫描
    for jsonl_file in "${input_dir}"/*.jsonl; do
      [[ -f "${jsonl_file}" ]] || continue
      local step_name
      step_name="$(basename "${jsonl_file}" .jsonl)"
      local step_output_dir="${output_dir}/${step_name}"
      [[ -s "${step_output_dir}/validation_summary.csv" ]] && continue
      log "[watcher] 最终评测 ${exp_name} step ${step_name}"
      python eval/eval_from_jsonl.py \
        --input "${jsonl_file}" \
        --output-dir "${step_output_dir}" \
        --metrics "${AFTER_TRAIN_EVAL_METRICS}" \
        --ks "${AFTER_TRAIN_EVAL_KS}" \
        --semantic-model "${EMBEDDING_MODEL_PATH}" \
        --semantic-device "${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}" \
        --semantic-batch-size "${AFTER_TRAIN_EVAL_SEMANTIC_BATCH_SIZE:-32}" \
        --semantic-max-length "${AFTER_TRAIN_EVAL_SEMANTIC_MAX_LENGTH:-1024}" || true
    done
  ) >> "${eval_log_file}" 2>&1 &
  EVAL_WATCHER_PID="$!"
  EVAL_BG_PIDS+=("${EVAL_WATCHER_PID}")
  log "启动后台评测 watcher (PID=${EVAL_WATCHER_PID}): ${exp_name}, 日志: ${eval_log_file}"
}

stop_eval_watcher() {
  if [[ -n "${EVAL_WATCHER_STOP_FILE}" ]]; then
    touch "${EVAL_WATCHER_STOP_FILE}"
  fi
  EVAL_WATCHER_PID=""
  EVAL_WATCHER_STOP_FILE=""
}

run_experiment() {
  local exp_name="$1"
  local algorithm="$2"
  local tau="$3"
  local total_steps="$4"
  local entropy_coeff="${5:-0.0}"
  local exploration_algorithms="${6:-}"
  local delayed_algorithms="${7:-}"
  local delay_fraction="${8:-1.0}"
  local marker="${STATE_DIR}/${exp_name}.done"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local val_files
  local trainer_logger
  local ray_cpu_arg
  local exploration_list
  local delayed_list

  if [[ -f "${marker}" ]] && ! bool_is_true "${FORCE_RERUN}"; then
    log "实验已完成，跳过: ${exp_name}"
    return 0
  fi

  val_files="$(val_files_override)"
  trainer_logger="$(trainer_logger_override)"
  ray_cpu_arg="$(ray_cpu_override_arg)"
  exploration_list="$(hydra_list_from_string "${exploration_algorithms}")"
  delayed_list="$(hydra_list_from_string "${delayed_algorithms}")"

  log "启动实验 ${exp_name}: algorithm=${algorithm}, tau=${tau}, entropy_coeff=${entropy_coeff}, steps=${total_steps}, extra_metrics=${exploration_list}"

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
    "algorithm.exploration_metric_algorithms=${exploration_list}"
    "algorithm.exploration_metric_delayed_algorithms=${delayed_list}"
    "algorithm.exploration_metric_delay_fraction=${delay_fraction}"
    "algorithm.similarity_params.n=${SIMILARITY_N}"
    "algorithm.similarity_params.normalize_method=${SIMILARITY_NORMALIZE_METHOD:-max}"
    "algorithm.similarity_params.max_features=${SIMILARITY_MAX_FEATURES:-1000}"
    "algorithm.similarity_params.min_df=${SIMILARITY_MIN_DF:-1}"
    "algorithm.similarity_params.max_df=${SIMILARITY_MAX_DF:-1.0}"
    "algorithm.similarity_params.ngram_range=${SIMILARITY_NGRAM_RANGE:-[1,2]}"
    "algorithm.similarity_params.hash_bits=${SIMILARITY_HASH_BITS:-64}"
    "algorithm.similarity_params.use_counts=${SIMILARITY_USE_COUNTS:-true}"
    "algorithm.similarity_params.calibrate_random=${SIMILARITY_CALIBRATE_RANDOM:-true}"
    "algorithm.similarity_params.model_name='${EMBEDDING_MODEL_PATH}'"
    "algorithm.similarity_params.batch_size=${SIMILARITY_BATCH_SIZE}"
    "algorithm.similarity_params.max_length=${SIMILARITY_MAX_LENGTH}"
    "algorithm.similarity_params.tail_tokens=${SIMILARITY_TAIL_TOKENS:-1024}"
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
  if [[ "${SIMILARITY_DEVICE}" == cuda* && -n "${SIMILARITY_CUDA_VISIBLE_DEVICES:-}" && "${SIMILARITY_CUDA_VISIBLE_DEVICES}" != "null" ]]; then
    cmd+=("algorithm.similarity_params.cuda_visible_devices=${SIMILARITY_CUDA_VISIBLE_DEVICES}")
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

  # 训练期间后台 watcher 实时评测每一步 validation JSONL（纯 CPU，不占 GPU）。
  start_eval_watcher "${exp_name}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  touch "${marker}"
  # 通知 watcher 训练已结束，让它做最后一轮扫描后自行退出。
  stop_eval_watcher
  export_train_log "${exp_name}" "${log_file}"
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
      --output-dir "${EVAL_DIR}/${exp_name}/train_log" \
      --all-keys || true
  fi
}

eval_jsonl_if_available() {
  local exp_name="$1"
  local expected_step="${2:-}"
  bool_is_true "${RUN_EVAL_AFTER_TRAIN}" || return 0

  local input_dir="${SAVE_DIR}/validation/${exp_name}"
  if [[ ! -d "${input_dir}" ]]; then
    log "未找到 validation JSONL 目录，跳过离线评测: ${input_dir}"
    return 0
  fi

  local output_dir="${EVAL_DIR}/${exp_name}/jsonl"
  mkdir -p "${output_dir}"

  # 找出所有尚未评测的 JSONL 文件并逐个评测。
  local jsonl_file
  while IFS= read -r jsonl_file; do
    [[ -n "${jsonl_file}" ]] || continue
    local step_name
    step_name="$(basename "${jsonl_file}" .jsonl)"
    local step_output_dir="${output_dir}/${step_name}"
    if [[ -s "${step_output_dir}/validation_summary.csv" ]]; then
      continue
    fi
    log "评测 ${exp_name} step ${step_name}"
    cd "${AER_DIR}"
    python eval/eval_from_jsonl.py \
      --input "${jsonl_file}" \
      --output-dir "${step_output_dir}" \
      --metrics "${AFTER_TRAIN_EVAL_METRICS}" \
      --ks "${AFTER_TRAIN_EVAL_KS}" \
      --semantic-model "${EMBEDDING_MODEL_PATH}" \
      --semantic-device "${AFTER_TRAIN_EVAL_SEMANTIC_DEVICE:-cpu}" \
      --semantic-batch-size "${AFTER_TRAIN_EVAL_SEMANTIC_BATCH_SIZE:-32}" \
      --semantic-max-length "${AFTER_TRAIN_EVAL_SEMANTIC_MAX_LENGTH:-1024}" || true
  done < <(find "${input_dir}" -maxdepth 1 -name '*.jsonl' -type f | sort)
}

generate_tau_plan() {
  local calib_exp
  local metrics_csv
  calib_exp="$(baseline_naive_exp_name)"
  metrics_csv="${EVAL_DIR}/${calib_exp}/train_log/train_metrics.csv"

  if [[ ! -s "${metrics_csv}" && -s "${LOG_DIR}/${calib_exp}.log" ]]; then
    export_train_log "${calib_exp}" "${LOG_DIR}/${calib_exp}.log"
  fi
  [[ -s "${metrics_csv}" ]] || die "校准指标不存在，无法生成 tau: ${metrics_csv}"

  log "根据 T0 最小探索奖励生成 tau 表: ${CALIBRATION_METRIC_ALGORITHMS}"
  python - "${metrics_csv}" "${CALIBRATION_METRIC_ALGORITHMS}" "${GAMMA_LIST}" "${EVAL_DIR}" "${TAU_PRECISION:-6}" <<'PY'
import csv
import sys
from pathlib import Path

metrics_csv, algorithms_raw, gammas_raw, eval_dir_raw, precision_raw = sys.argv[1:]
algorithms = [item for item in algorithms_raw.split() if item]
gammas = [float(item) for item in gammas_raw.split() if item]
precision = int(precision_raw)
eval_dir = Path(eval_dir_raw)
eval_dir.mkdir(parents=True, exist_ok=True)

with open(metrics_csv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"校准指标为空: {metrics_csv}")

for index, algorithm in enumerate(algorithms):
    column = f"metric/exploration reward/{algorithm}"
    if column not in rows[0] and index == 0:
        column = "metric/exploration reward"
    values = [float(row[column]) for row in rows if row.get(column) not in (None, "")]
    if not values:
        raise SystemExit(f"{metrics_csv} 中没有 {algorithm} 的探索奖励列")
    min_reward = min(values)
    output = eval_dir / f"tau_plan_{algorithm}.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "min_exploration_reward", "gamma", "tau", "source_metric", "n_points"])
        writer.writeheader()
        for gamma in gammas:
            writer.writerow({
                "algorithm": algorithm,
                "min_exploration_reward": min_reward,
                "gamma": gamma,
                "tau": round(min_reward * gamma, precision),
                "source_metric": column,
                "n_points": len(values),
            })
    print(f"已写入 {output}")
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

ensure_tau_plan() {
  local algorithm
  for algorithm in ${CALIBRATION_METRIC_ALGORITHMS}; do
    if [[ ! -s "$(tau_plan_path "${algorithm}")" ]]; then
      if bool_is_true "${DRY_RUN:-0}"; then
        return 0
      fi
      generate_tau_plan
      return 0
    fi
  done
}

select_gamma_best() {
  local algorithm="${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}"
  local output_env
  output_env="$(gamma_best_env_path)"

  log "根据训练后 CPU 评测结果选择 gamma_best: ${algorithm}"
  python - "$(tau_plan_path "${algorithm}")" "${algorithm}" "${GAMMA_LIST}" "${EVAL_DIR}" "${TOTAL_TRAINING_STEPS}" "${GAMMA_SELECTION_PRIMARY_METRIC:-correct_rate}" "${GAMMA_SELECTION_TIEBREAK_METRIC:-distinct_2}" "${output_env}" <<'PY'
import csv
import math
import sys
from pathlib import Path

tau_csv, algorithm, gammas_raw, eval_dir_raw, steps, primary_metric, tie_metric, output_env = sys.argv[1:]
eval_dir = Path(eval_dir_raw)

def tag(value):
    return str(value).replace("-", "m").replace(".", "p")

def tau_for(gamma):
    target = float(gamma)
    with open(tau_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("algorithm") == algorithm and math.isclose(float(row["gamma"]), target, rel_tol=0.0, abs_tol=1e-12):
                return row["tau"]
    raise SystemExit(f"{tau_csv} 中没有 gamma={gamma}")

def summary_row(exp_name):
    base = eval_dir / exp_name / "jsonl"
    # 优先查找按步存放的子目录（watcher 模式），取最大步数的结果。
    step_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    ) if base.is_dir() else []
    if step_dirs:
        path = step_dirs[-1] / "validation_summary.csv"
    else:
        path = base / "validation_summary.csv"
    if not path.exists():
        raise SystemExit(f"缺少 gamma 评测结果，无法自动选择 gamma_best: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for preferred in ("AVG_DATASETS", "all"):
        for row in rows:
            if row.get("data_source") == preferred:
                return row
    if rows:
        return rows[0]
    raise SystemExit(f"评测结果为空: {path}")

def number(row, *keys, default=float("-inf")):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return default

best = None
for gamma in [item for item in gammas_raw.split() if item]:
    tau = tau_for(gamma)
    exp_name = f"gamma-search-{algorithm}-g{tag(gamma)}-tau{tag(tau)}-s{steps}"
    row = summary_row(exp_name)
    primary = number(row, primary_metric, "correct_rate", "first@1", "pass@8", "pass@4", "pass@1")
    tie = number(row, tie_metric, "distinct_2")
    self_bleu_bonus = -number(row, "self_bleu4", default=float("inf"))
    candidate = ((primary, tie, self_bleu_bonus), gamma, tau, exp_name)
    if best is None or candidate[0] > best[0]:
        best = candidate

_, gamma, tau, exp_name = best
Path(output_env).write_text(
    f"GAMMA_BEST={gamma}\nGAMMA_BEST_TAU={tau}\nGAMMA_BEST_EXPERIMENT={exp_name}\n",
    encoding="utf-8",
)
print(f"gamma_best={gamma}, tau={tau}, experiment={exp_name}")
PY
}

resolve_gamma_best() {
  if [[ -n "${GAMMA_BEST:-}" && "${GAMMA_BEST}" != "auto" ]]; then
    printf '%s' "${GAMMA_BEST}"
    return 0
  fi
  if bool_is_true "${DRY_RUN:-0}"; then
    first_word "${GAMMA_LIST}"
    return 0
  fi
  local gamma_env
  gamma_env="$(gamma_best_env_path)"
  if [[ ! -s "${gamma_env}" ]]; then
    select_gamma_best
  fi
  # shellcheck source=/dev/null
  source "${gamma_env}"
  printf '%s' "${GAMMA_BEST}"
}

run_baseline_naive() {
  bool_is_true "${RUN_BASELINE_NAIVE:-1}" || return 0
  run_experiment "$(baseline_naive_exp_name)" "${BASELINE_SIMILARITY_ALGORITHM:-token_match}" "0" "${CALIBRATION_STEPS}" "0.0" "${CALIBRATION_METRIC_ALGORITHMS}" "${CALIBRATION_DELAYED_ALGORITHMS:-}" "${CALIBRATION_DELAY_FRACTION:-1.0}"
  bool_is_true "${DRY_RUN:-0}" || generate_tau_plan
}

run_baseline_entropy() {
  bool_is_true "${RUN_BASELINE_ENTROPY:-1}" || return 0
  run_experiment "$(baseline_entropy_exp_name)" "${BASELINE_SIMILARITY_ALGORITHM:-token_match}" "0" "${TOTAL_TRAINING_STEPS}" "${ENTROPY_BASELINE_COEFF:-0.0}" ""
}

run_gamma_search() {
  bool_is_true "${RUN_GAMMA_SEARCH:-1}" || return 0
  ensure_tau_plan

  local gamma
  for gamma in ${GAMMA_LIST}; do
    local tau
    if bool_is_true "${DRY_RUN:-0}" && [[ ! -s "$(tau_plan_path "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}")" ]]; then
      tau="0"
    else
      tau="$(read_tau_for_gamma "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma}")"
    fi
    run_experiment "$(gamma_search_exp_name "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${gamma}" "${tau}")" "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" "${tau}" "${TOTAL_TRAINING_STEPS}" "0.0" ""
  done

  if [[ "${GAMMA_BEST:-auto}" == "auto" ]] && ! bool_is_true "${DRY_RUN:-0}"; then
    select_gamma_best
  fi
}

run_main_aer() {
  bool_is_true "${RUN_MAIN_AER:-1}" || return 0
  ensure_tau_plan

  local gamma_best
  gamma_best="$(resolve_gamma_best)"
  local algorithm
  for algorithm in ${MAIN_SIMILARITY_ALGORITHMS}; do
    local tau
    if bool_is_true "${DRY_RUN:-0}" && [[ ! -s "$(tau_plan_path "${algorithm}")" ]]; then
      tau="0"
    else
      tau="$(read_tau_for_gamma "${algorithm}" "${gamma_best}")"
    fi
    if [[ "${algorithm}" == "${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}" ]] && bool_is_true "${REUSE_GAMMA_SEARCH_FOR_TARGET:-1}"; then
      log "主实验 ${algorithm} 复用 gamma 搜索 run: $(gamma_search_exp_name "${algorithm}" "${gamma_best}" "${tau}")"
      continue
    fi
    run_experiment "$(main_aer_exp_name "${algorithm}" "${gamma_best}" "${tau}")" "${algorithm}" "${tau}" "${TOTAL_TRAINING_STEPS}" "0.0" ""
  done
}

run_extra_aer() {
  bool_is_true "${RUN_EXTRA_AER:-0}" || return 0
  [[ -n "${EXTRA_SIMILARITY_ALGORITHMS:-}" ]] || return 0
  ensure_tau_plan

  local gamma_best
  gamma_best="$(resolve_gamma_best)"
  local algorithm
  for algorithm in ${EXTRA_SIMILARITY_ALGORITHMS}; do
    local tau
    tau="$(read_tau_for_gamma "${algorithm}" "${gamma_best}")"
    run_experiment "$(main_aer_exp_name "${algorithm}" "${gamma_best}" "${tau}")" "${algorithm}" "${tau}" "${TOTAL_TRAINING_STEPS}" "0.0" ""
  done
}

EVAL_BG_PIDS=()

wait_eval_bg() {
  if [[ "${#EVAL_BG_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  log "等待 ${#EVAL_BG_PIDS[@]} 个后台 CPU 评测完成..."
  local pid
  for pid in "${EVAL_BG_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  EVAL_BG_PIDS=()
}

run_training_queue() {
  run_baseline_naive
  run_baseline_entropy
  # gamma_best 自动选择依赖 gamma search 的评测结果，需等待评测完成。
  wait_eval_bg
  run_gamma_search
  wait_eval_bg
  run_main_aer
  run_extra_aer
  wait_eval_bg
}

check_inputs() {
  local command="${1:-all}"
  [[ -d "${VERL_DIR}" ]] || die "未找到 verl 目录: ${VERL_DIR}"
  [[ -f "${REPO_ROOT}/environment.yml" ]] || die "未找到 environment.yml"
  if [[ "${command}" =~ ^(setup|train|all)$ && "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]] && ! bool_is_true "${DRY_RUN:-0}"; then
    die "请在 ${CONFIG_FILE} 中填写 WANDB_API_KEY，或把 WANDB_MODE 改成 offline/disabled。"
  fi
}

print_status() {
  cat <<EOF
REPO_ROOT=${REPO_ROOT}
SAVE_DIR=${SAVE_DIR}
CONFIG_FILE=${CONFIG_FILE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
RUN_BASELINE_NAIVE=${RUN_BASELINE_NAIVE}
RUN_BASELINE_ENTROPY=${RUN_BASELINE_ENTROPY}
RUN_GAMMA_SEARCH=${RUN_GAMMA_SEARCH}
RUN_MAIN_AER=${RUN_MAIN_AER}
CALIBRATION_METRIC_ALGORITHMS=${CALIBRATION_METRIC_ALGORITHMS}
CALIBRATION_STEPS=${CALIBRATION_STEPS}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
TARGET_SIMILARITY_FOR_GAMMA_SEARCH=${TARGET_SIMILARITY_FOR_GAMMA_SEARCH}
GAMMA_LIST=${GAMMA_LIST}
GAMMA_BEST=${GAMMA_BEST}
MAIN_SIMILARITY_ALGORITHMS=${MAIN_SIMILARITY_ALGORITHMS}
SIMILARITY_DEVICE=${SIMILARITY_DEVICE}
SIMILARITY_CUDA_VISIBLE_DEVICES=${SIMILARITY_CUDA_VISIBLE_DEVICES:-}
SIMILARITY_NUM_PROCESSES=${SIMILARITY_NUM_PROCESSES}
AFTER_TRAIN_EVAL_METRICS=${AFTER_TRAIN_EVAL_METRICS}
AFTER_TRAIN_EVAL_KS=${AFTER_TRAIN_EVAL_KS}
WANDB_MODE=${WANDB_MODE}
WANDB_PROJECT=${WANDB_PROJECT}
MODEL_PATH=${MODEL_PATH}
EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH}
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
      run_training_queue
      ;;
    all)
      setup_env
      download_models
      prepare_data
      run_smoke_tests
      run_training_queue
      ;;
    *)
      die "未知命令: ${command}。可选: status/setup/assets/test/train/all"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
