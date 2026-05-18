# AER 新实验运行说明

这个目录用于一键配置环境、运行新实验计划，并对训练好的 checkpoint 做正式完整评测。默认训练超参对齐 `verl/recipe/aer/run.sh`，实验队列改为 T0/T1/gamma-search/main AER，也提供了每个实验可单独运行、单独改参数的脚本。

## 1. 最少操作

先检查并修改 `need_to_modify/config.env`：

```bash
bash need_to_modify/run_experiments.sh status
```

重点确认：

```bash
WANDB_API_KEY=""
MODEL_PATH="/data/models/Qwen/Qwen3-4B"
EMBEDDING_MODEL_PATH="/data/models/Qwen/Qwen3-Embedding-0.6B"
CUDA_VISIBLE_DEVICES="0,1,2,3"
SIMILARITY_DEVICE="cuda"
SIMILARITY_CUDA_VISIBLE_DEVICES="[4,5,6,7]"
SIMILARITY_NUM_PROCESSES=4
```

一键配置环境、下载模型和数据、跑测试、跑全部训练实验：

```bash
nohup bash need_to_modify/run_experiments.sh all > master.log 2>&1 &
tail -f master.log
```

## 2. 常用命令

```bash
# 只安装环境
bash need_to_modify/run_experiments.sh setup
# 只下载模型和数据
bash need_to_modify/run_experiments.sh assets
# 只跑轻量测试（确认环境没问题）
bash need_to_modify/run_experiments.sh test
# **常用**：只运行实验
bash need_to_modify/run_experiments.sh train
```

## 3. 单实验脚本

如果不想跑整条队列，可以使用拆分后的独立脚本。每个脚本开头都有一段“可单独修改的实验参数”，可以直接在脚本内改，也可以用环境变量临时覆盖。

训练脚本在 `need_to_modify/train_experiment_scripts/`，每个训练脚本都会启动完整训练，并在训练期间对每个 validation step 的 JSONL 结果启动纯 CPU watcher 评测：

```bash
# tau 校准，同时生成 save/eval/tau_plan_<algorithm>.csv
bash need_to_modify/train_experiment_scripts/00_baseline_naive_calib.sh

# 单个 gamma 搜索训练
bash need_to_modify/train_experiment_scripts/03_gamma_search_ngram_overlap_g1p2.sh

# gamma 搜索结束后自动选择最佳 gamma
bash need_to_modify/train_experiment_scripts/06_select_gamma_best.sh

# 单个主 AER 训练
bash need_to_modify/train_experiment_scripts/09_aer_semantic_embedding_best_gamma.sh
```

正式评测脚本在 `need_to_modify/formal_eval_scripts/`，每个脚本只评测一个训练实验，评测指标仍使用独立的 `FORMAL_EVAL_*` 配置：

```bash
bash need_to_modify/formal_eval_scripts/00_eval_baseline_naive.sh
bash need_to_modify/formal_eval_scripts/06_eval_gamma_search_best.sh
bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh
```

常用覆盖方式：

```bash
DRY_RUN=1 bash need_to_modify/train_experiment_scripts/02_gamma_search_ngram_overlap_g1p1.sh
GAMMA=1.3 bash need_to_modify/train_experiment_scripts/07_aer_token_match_best_gamma.sh
FORMAL_EVAL_CHECKPOINT_STEP=504 bash need_to_modify/formal_eval_scripts/07_eval_aer_token_match_best_gamma.sh
```

停止某个单实验时，在同一个脚本后加 `stop`。脚本会停止该实验入口进程和它启动的子进程，包括训练进程、CPU watcher、正式评测 shard 等：

```bash
bash need_to_modify/train_experiment_scripts/03_gamma_search_ngram_overlap_g1p2.sh stop
bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh stop
```

## 4. 训练队列

完整计划中的标准队列：

1. `baseline-naive-calib-tau0-s504`：tau=0，记录 `CALIBRATION_METRIC_ALGORITHMS` 的探索奖励，并生成 tau 表。
2. `baseline-entropy-mean-tau0-s504`：熵正则 mean baseline。
3. `gamma-search-ngram_overlap-g{gamma}-tau{tau}-s504`：默认搜索 `1.1 1.2 1.3 1.4`。
4. `aer-token_match-g{best}-tau{tau}-s504`、`aer-ngram_overlap...`、`aer-semantic_embedding...`：主 AER 实验。
5. `aer-simhash-g{best}-tau{tau}-s504`：补充 AER 实验。

在 `config.env` 中控制具体运行哪些实验：

```bash
RUN_BASELINE_NAIVE=1
RUN_BASELINE_ENTROPY=1
RUN_GAMMA_SEARCH=1
RUN_MAIN_AER=1
CALIBRATION_METRIC_ALGORITHMS="token_match ngram_overlap semantic_embedding simhash"
TARGET_SIMILARITY_FOR_GAMMA_SEARCH="ngram_overlap"
GAMMA_LIST="1.1 1.2 1.3 1.4"
GAMMA_BEST="auto"
MAIN_SIMILARITY_ALGORITHMS="token_match ngram_overlap semantic_embedding"
RUN_EXTRA_AER=1
EXTRA_SIMILARITY_ALGORITHMS="simhash"
```

`GAMMA_BEST=auto` 时，脚本会在 gamma 搜索全部跑完并完成训练后 CPU 评测后，写出 `save/eval/gamma_best_<algorithm>.env`。如果已经人工选好 gamma，可直接设置 `GAMMA_BEST="1.2"`。

## 5. semantic_embedding 设备

训练阶段 `semantic_embedding` 支持 CPU、单 GPU 或多 GPU：

```bash
# CPU
SIMILARITY_DEVICE="cpu"
SIMILARITY_CUDA_VISIBLE_DEVICES=""
SIMILARITY_NUM_PROCESSES=4

# 单 GPU
SIMILARITY_DEVICE="cuda"
SIMILARITY_CUDA_VISIBLE_DEVICES="[4]"
SIMILARITY_NUM_PROCESSES=1

# 多 GPU
SIMILARITY_DEVICE="cuda"
SIMILARITY_CUDA_VISIBLE_DEVICES="[4,5,6,7]"
SIMILARITY_NUM_PROCESSES=4
```

训练期间后台 watcher 实时评测每一步 validation JSONL（纯 CPU，与训练并行不占 GPU）：

```bash
AFTER_TRAIN_EVAL_METRICS="pass@k,first@1,distinct-2,self-bleu,equational-diversity"
AFTER_TRAIN_EVAL_KS="1,2,4,8"
AFTER_TRAIN_EVAL_SEMANTIC_DEVICE="cpu"
```

## 6. 正式完整评测

正式评测使用独立的 `FORMAL_EVAL_*` 配置，不和训练后轻量评测共用同一组指标。

```bash
nohup bash need_to_modify/run_eval_formal_checkpoints.sh > formal_eval.log 2>&1 &
tail -f formal_eval.log
```

默认评测 naive baseline、entropy baseline、最佳 gamma-search run，以及 `FORMAL_EVAL_MAIN_ALGORITHMS` 对应的主 AER 实验。补充实验可写入 `FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES`，或直接运行 `formal_eval_scripts/10_eval_aer_simhash_best_gamma.sh`。常用配置：

```bash
FORMAL_EVAL_METRICS="pass@k,first@1,distinct-2,self-bleu,semantic-cosine,equational-diversity"
FORMAL_EVAL_KS="1,2,4,8,16,32,64,128"
FORMAL_EVAL_SAMPLES_PER_PROMPT=128
FORMAL_EVAL_GPUS="0,1,2,3"
FORMAL_EVAL_CHECKPOINT_STEP="${TOTAL_TRAINING_STEPS}"  # 默认 504
```

也可以手动指定实验名：

```bash
FORMAL_EVAL_EXPERIMENT_NAMES="baseline-naive-calib-tau0-s504 aer-token_match-g1p2-tau0p123456-s504"
FORMAL_EVAL_MAIN_ALGORITHMS=""
FORMAL_EVAL_INCLUDE_GAMMA_SEARCH=0
```

## 7. 输出位置

| 路径 | 内容 |
|---|---|
| `save/checkpoints/<exp>/` | FSDP checkpoint |
| `save/validation/<exp>/` | 训练中保存的 validation JSONL |
| `save/eval/tau_plan_<algorithm>.csv` | 由 T0 最小探索奖励生成的 tau 表 |
| `save/eval/gamma_best_<algorithm>.env` | 自动选择出的最佳 gamma |
| `save/eval/<exp>/train_log/` | 训练日志导出的指标 |
| `save/eval/<exp>/jsonl/<step>/` | 训练期间实时 CPU 评测（每步一个子目录） |
| `save/eval/<exp>/<FORMAL_EVAL_OUTPUT_SUBDIR>/` | 正式完整评测 |
| `save/run/train_logs/<exp>.log` | 每个训练实验 stdout/stderr |
| `save/run/eval_logs/<exp>.log` | 训练期间后台评测 watcher 日志 |
| `save/run/script_pids/<kind>/<script>.pid` | 单实验脚本运行记录，供 `bash <script> stop` 使用 |
| `need_to_modify/eval_logs/<timestamp>/` | 正式完整评测主日志、合并日志、推理日志和指标日志 |
| `need_to_modify/train_experiment_scripts/` | 每个训练实验的独立入口 |
| `need_to_modify/formal_eval_scripts/` | 每个正式评测实验的独立入口 |

中断后重新执行同一命令即可；已完成实验由 `save/run/state/<exp>.done` 跳过，未完成实验使用 `trainer.resume_mode=auto` 续跑。
