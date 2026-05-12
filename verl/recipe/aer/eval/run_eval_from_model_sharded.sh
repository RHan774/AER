#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AER_DIR}"

# 4 卡并行评测：每张卡启动一个 vLLM 实例，避免小模型 TP=4 的通信开销。
# 如需换卡或换路径，可以通过环境变量覆盖下面的默认值。
MODEL_PATH="${MODEL_PATH:-../../../save/checkpoints/levenshtein-tau0.155-offpolicy/global_step_240_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-../../../save/eval/levenshtein-tau0.155-offpolicy/rerun_pass128_sharded}"
GPUS="${GPUS:-4,5,6,7}"

METRICS="${METRICS:-pass@k,first@1,distinct-2,self-bleu,semantic-cosine,equational-diversity}"
KS="${KS:-1,2,4,8,16,32,64,128}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-128}"
ROLLOUT_SAVE_BATCH_SIZE="${ROLLOUT_SAVE_BATCH_SIZE:-8}"
SEMANTIC_DEVICE="${SEMANTIC_DEVICE:-cuda:4,cuda:5,cuda:6,cuda:7}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-128}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"

VAL_FILES=(
  "../../../save/data/math-ai/math500/test_repeated.parquet"
  "../../../save/data/math-ai/amc23/test_repeated.parquet"
  "../../../save/data/math-ai/aime24/test_repeated.parquet"
  "../../../save/data/math-ai/aime25/test_repeated.parquet"
)

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NUM_SHARDS="${#GPU_IDS[@]}"
mkdir -p "${OUTPUT_DIR}"

pids=()
rollout_paths=()
for shard_index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$shard_index]}"
  log_path="${OUTPUT_DIR}/shard_${shard_index}-of-${NUM_SHARDS}.log"
  printf -v shard_rollout_name "model_rollout_shard_%05d-of-%05d.jsonl" "${shard_index}" "${NUM_SHARDS}"
  rollout_paths+=("${OUTPUT_DIR}/${shard_rollout_name}")
  echo "启动 shard ${shard_index}/${NUM_SHARDS} on GPU ${gpu}, 日志: ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python eval/eval_from_model.py \
    --model-path "${MODEL_PATH}" \
    --val-files "${VAL_FILES[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --metrics "${METRICS}" \
    --ks "${KS}" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
    --semantic-batch-size "${SEMANTIC_BATCH_SIZE}" \
    --rollout-save-batch-size "${ROLLOUT_SAVE_BATCH_SIZE}" \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --vllm-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed 42 \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard_index}" \
    --skip-eval \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
remaining="${#pids[@]}"
while [[ "${remaining}" -gt 0 ]]; do
  if ! wait -n; then
    failed=1
    echo "检测到一个 shard 失败，继续等待其它 shard 收尾..." >&2
  fi
  remaining=$((remaining - 1))
  echo "已有一个 shard 结束，剩余 ${remaining} 个"
done

if [[ "${failed}" -ne 0 ]]; then
  echo "至少一个 shard 失败，请先检查 ${OUTPUT_DIR}/shard_*-of-${NUM_SHARDS}.log" >&2
  exit 1
fi

missing_rollouts=()
for rollout_path in "${rollout_paths[@]}"; do
  if [[ ! -s "${rollout_path}" ]]; then
    missing_rollouts+=("${rollout_path}")
  fi
done

if [[ "${#missing_rollouts[@]}" -ne 0 ]]; then
  echo "所有 shard 进程已退出，但缺少以下 rollout 文件，停止汇总评测:" >&2
  printf '  %s\n' "${missing_rollouts[@]}" >&2
  exit 1
fi

echo "所有 ${NUM_SHARDS} 个 shard 均已完成，开始汇总评测..."
python eval/eval_from_jsonl.py \
  --input "${rollout_paths[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --metrics "${METRICS}" \
  --ks "${KS}" \
  --semantic-device "${SEMANTIC_DEVICE}"

echo "并行评测完成: ${OUTPUT_DIR}"
