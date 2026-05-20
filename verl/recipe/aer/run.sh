# 在 run.sh 开头添加，指定使用 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 数据保存目录（必须先定义，因为后面的变量依赖它）
save_dir="../../../save"

# 该实验运行 naive-GRPO baseline；tau=0 时探索奖励权重始终为 0，只把探索奖励作为指标记录。
tau=0
entropy_coeff=0.0
# 训练兼容字段，tau=0 时不会影响实际 reward；探索奖励指标由 exploration_metric_algorithms 控制。
similarity_algorithm="token_match"
exploration_metric_algorithms="[token_match,ngram_overlap,semantic_embedding,simhash]"
# 只把耗时较大的额外指标放到最后 10% 训练步计算；主训练奖励算法不受影响。
exploration_metric_delayed_algorithms="[simhash,semantic_embedding]"
exploration_metric_delay_fraction=0.10
similarity_n=3
resume_mode="auto" # auto;resume_path;disable
resume_from_path=""

train_batch_size=128
ppo_mini_batch_size=32
# bs=128 后不能继续假设 300~400 step 收敛；先用 naive GRPO 做收敛扫描。
total_training_steps=360
# 每 save_freq step 保存一次；当前和 test_freq 对齐，保证每个验证点都有可恢复 checkpoint。
save_freq=12
test_freq=12
experiment_name="baseline-naive-grpo-tau${tau}-all-similarity-metrics"
# experiment_name="baseline-entropy_coeff${entropy_coeff}"
max_actor_ckpt_to_keep=8
max_critic_ckpt_to_keep=8

# modify: 原来是1，改为到16
val_kwargs_n=16


# data
train_files="${save_dir}/data/DigitalLearningGmbH/MATH-lighteval/train.parquet"
# 当前 test_repeated.parquet 的 repeat 次数为 1；验证阶段的 repeated sampling 由 val_kwargs_n 控制。
val_files="['${save_dir}/data/math-ai/math500/test_repeated.parquet', '${save_dir}/data/math-ai/amc23/test_repeated.parquet', '${save_dir}/data/math-ai/aime24/test_repeated.parquet', '${save_dir}/data/math-ai/aime25/test_repeated.parquet']"
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
# # modify: 原来是32，增加到64加速训练
# train_batch_size=64
filter_overlong_prompts=False
truncation="right"

# actor_rollout_ref
# model
model_path="/data/models/Qwen/Qwen3-4B-Base"
enable_gradient_checkpointing=True
use_remove_padding=True
# actor
# ppo_mini_batch_size保持32，老师说不需要调大
use_dynamic_bsz=True
max_token_len_per_gpu=16384
loss_agg_mode="seq-mean-token-sum"
ppo_epochs=1
use_kl_loss=False
# # modify：调整clip_ratio应对off-policy: 0.2 → 0.15
# clip_ratio=0.15
ulysses_sequence_parallel_size=1
lr=1e-6
weight_decay=0.0
offload=False
# rollout
temperature=1.0
top_p=1.0
# modify: 原来是0.7，增加到0.9
gpu_memory_utilization=0.85
tensor_model_parallel_size=1
rollout_n=16
val_kwargs_temperature=0.6
val_kwargs_top_p=0.95
val_kwargs_top_k=20
# # modify: 原来是1，改为到8
# val_kwargs_n=8
do_sample=True

# reward_model
reward_manager="aer"
# # 主要修改的超参数
# tau=0.55

# add: 相似度计算算法选择
# 可选: token_match, ngram_overlap, char_ngram, levenshtein, tfidf_cosine, semantic_embedding, simhash, compression_ratio, rouge_l
# - token_match: Token 精确匹配（原有方法，最快，~1ms）
# - ngram_overlap: N-gram 重叠度（~5ms，推荐 n=3）
# - char_ngram: 字符级 N-gram（~10ms，对数学符号鲁棒）
# - levenshtein: 编辑距离（~50ms，RapidFuzz 加速）
# - tfidf_cosine: TF-IDF 余弦相似度（~20ms）
# - semantic_embedding: 语义嵌入（~500ms，最佳语义理解，需安装 sentence-transformers，支持 32K 长文本）
# - compression_ratio: 压缩比相似度（~5ms，基于 arXiv:2403.00553 论文，快速多样性评估）
# - rouge_l: ROUGE-L 相似度（~30ms，基于最长公共子序列，对顺序敏感）
# similarity_algorithm="tfidf_cosine"
# add: N-gram 的 n 值（用于 ngram_overlap, char_ngram）

# add: 语义嵌入模型（用于 semantic_embedding）
# 可选: Qwen/Qwen3-Embedding-0.6B (默认，32K 长文本), Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B
#      all-MiniLM-L6-v2 (最快，256 token 限制), all-mpnet-base-v2 (512 token 限制)
similarity_model="/data/models/Qwen/Qwen3-Embedding-0.6B"
# GPU 上 Qwen3-Embedding-0.6B 仍只编码 response 尾部 1024 token，控制显存和耗时。
similarity_batch_size=32
similarity_max_length=4096
similarity_tail_tokens=4096
# add: 语义嵌入使用训练外的物理 GPU。训练用 0,1,2,3，embedding worker 用 4,5,6,7。
similarity_device="cuda"
similarity_cuda_visible_devices="[4,5,6,7]"
# add: 多进程并行数（用于 embedding 计算）
# 4 个进程会按顺序分配到 similarity_cuda_visible_devices 中的 4 张 GPU。
similarity_num_processes=4

# algorithm
adv_estimator="grpo"

# trainer
# modify: 原来是1000 ，改为10 epochs（不用管，收敛了看step就行）
total_epochs=100
project_name="AER"
# rollout_data_dir="${save_dir}/rollout"
# validation_data_dir="${save_dir}/validation"
# validation JSONL 是离线评测的输入，默认保留；rollout 训练样本体积更大，按需打开。
rollout_data_dir=""
validation_data_dir="${save_dir}/validation/${experiment_name}"
nnodes=1
n_gpus_per_node=4
# # modify: 原来是-1，改为50 (50个step保存一次)
# save_freq=50
# # modify: 原来是50，改为10 (10个step测试一次)
# test_freq=10
# 训练 state checkpoint 仍由 max_actor_ckpt_to_keep / max_critic_ckpt_to_keep 控制保留数量；
# 每个保存步会额外导出一份独立的 HuggingFace 推理 checkpoint，不参与训练 state 的清理。
default_local_dir="${save_dir}/checkpoints/${experiment_name}"
inference_checkpoint_dir="${save_dir}/inference_checkpoints/${experiment_name}"

# debug0: Ray集群问题:
# requests.exceptions.HTTPError: 502 Server Error: Bad Gateway for url: http://127.0.0.1:8265/api/version
# 不再启动 Ray，main_ppo 中用本地Ray集群，这里就不需要启动 Ray 了
# 原代码如下：
# nohup ray job submit --no-wait \
#     --runtime-env-json='{
#         "working_dir": "'${PWD}'",
#         "env_vars": {
#           "HF_ENDPOINT": "https://hf-mirror.com",
#         }
#     }' \
#     -- python -m recipe.aer.src.main_ppo \
export HF_ENDPOINT="https://hf-mirror.com"
export TOKENIZERS_PARALLELISM="true"
export NCCL_DEBUG="WARN"
export VLLM_LOGGING_LEVEL="WARN"
export RAY_TMPDIR=~/ray_tmp

# add: 优化 CPU 线程数配置（用于 embedding 计算）
# 服务器有 224 个逻辑核心（2×56核×2超线程）
# 4 个 embedding 进程各 48 线程，约使用 192 个逻辑核，给 Ray/vLLM 留余量。
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export VECLIB_MAXIMUM_THREADS=48
# PyTorch CPU 线程数
export TORCH_NUM_THREADS=48

    # 注释掉 clip_ratio 和 kl_loss_coef，使用配置文件默认值
    # actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    # actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \

nohup /data/ruanruihan/.conda/envs/aer/bin/python -m recipe.aer.src.main_ppo \
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_from_path}" \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.checkpoint.contents="[model,optimizer,extra,hf_model]" \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_kwargs_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_kwargs_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_kwargs_top_k} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_kwargs_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample} \
    reward_model.reward_manager=${reward_manager} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.tau=${tau} \
    algorithm.similarity_algorithm=${similarity_algorithm} \
    algorithm.exploration_metric_algorithms="${exploration_metric_algorithms}" \
    algorithm.exploration_metric_delayed_algorithms="${exploration_metric_delayed_algorithms}" \
    algorithm.exploration_metric_delay_fraction=${exploration_metric_delay_fraction} \
    algorithm.similarity_params.n=${similarity_n} \
    algorithm.similarity_params.model_name=${similarity_model} \
    algorithm.similarity_params.batch_size=${similarity_batch_size} \
    algorithm.similarity_params.max_length=${similarity_max_length} \
    algorithm.similarity_params.tail_tokens=${similarity_tail_tokens} \
    algorithm.similarity_params.device=${similarity_device} \
    algorithm.similarity_params.cuda_visible_devices="${similarity_cuda_visible_devices}" \
    algorithm.similarity_params.num_processes=${similarity_num_processes} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.validation_data_dir="${validation_data_dir}" \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.max_actor_ckpt_to_keep="${max_actor_ckpt_to_keep}" \
    trainer.max_critic_ckpt_to_keep="${max_critic_ckpt_to_keep}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.inference_checkpoint_dir="${inference_checkpoint_dir}" > baseline_naive_grpo_tau0_all_similarity_metrics_log.txt 2>&1 &
