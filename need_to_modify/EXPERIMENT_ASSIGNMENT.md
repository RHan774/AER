# 实验分工

## 默认负责内容

该目录现在按 `新实验计划.md` 组织完整队列，配置集中写在 `need_to_modify/config.env`。当前配置适合在已有 tau 表或已有 baseline checkpoint 后继续跑 gamma/main AER；如果从零复现实验，请打开对应 `RUN_*` 开关，或直接运行拆分后的单实验脚本。

| 阶段 | 默认实验 | 配置项 |
|---|---|---|
| T0 tau 校准 | `baseline-naive-calib-tau0-s504`，同时记录 `token_match`、`ngram_overlap`、`semantic_embedding`、`simhash` 探索奖励 | `RUN_BASELINE_NAIVE`、`CALIBRATION_METRIC_ALGORITHMS=...` |
| T1 熵 baseline | `baseline-entropy-mean-tau0-s504` | `RUN_BASELINE_ENTROPY`、`ENTROPY_BASELINE_COEFF=0.01` |
| T2-T5 gamma 搜索 | `gamma-search-ngram_overlap-g{gamma}-tau{tau}-s504` | `RUN_GAMMA_SEARCH=1`、`GAMMA_LIST="1.1 1.2 1.3 1.4"` |
| T6-T8 主 AER | `token_match`、`ngram_overlap`、`semantic_embedding` 复用 `gamma_best` | `RUN_MAIN_AER=1`、`MAIN_SIMILARITY_ALGORITHMS=...` |
| T9 补充 AER | `simhash` 复用 `gamma_best` | `RUN_EXTRA_AER=1`、`EXTRA_SIMILARITY_ALGORITHMS="simhash"` |
| 完整评测 | baseline、最佳 gamma-search、主 AER checkpoint 的 Pass@K 与多样性指标；补充 AER 可用单脚本或 `FORMAL_EVAL_EXTRA_EXPERIMENT_NAMES` 加入 | `run_eval_formal_checkpoints.sh`、`FORMAL_EVAL_*` |

默认训练到 `504` step，每 `12` step 验证一次，每 `24` step 保存一次，对齐当前 `verl/recipe/aer/run.sh` 的核心训练配置。

## 单实验脚本分工

训练实验已经拆到 `need_to_modify/train_experiment_scripts/`。这些脚本复用 `run_experiments.sh` 的函数，因此包含完整训练命令、断点续跑、训练期间每个验证步的 CPU watcher 评测；每个脚本开头都可以单独改 `ALGORITHM`、`GAMMA`、`TAU`、`TOTAL_STEPS` 等参数。

运行中的单实验可用同一个脚本加 `stop` 停止，例如 `bash need_to_modify/train_experiment_scripts/03_gamma_search_ngram_overlap_g1p2.sh stop`。停止逻辑优先停止独立进程组，兜底递归停止 pid 树。

| 脚本 | 用途 |
|---|---|
| `00_baseline_naive_calib.sh` | T0 tau 校准，训练结束后生成 tau 表 |
| `01_baseline_entropy_mean.sh` | T1 熵 baseline |
| `02`-`05_gamma_search_ngram_overlap_*.sh` | 单个 gamma 搜索训练 |
| `06_select_gamma_best.sh` | 根据训练期间 CPU 评测结果写出 `gamma_best_<algorithm>.env` |
| `07`-`10_aer_*_best_gamma.sh` | 单个主 AER 或补充 AER 训练 |

正式评测实验已经拆到 `need_to_modify/formal_eval_scripts/`。这些脚本复用 `run_eval_formal_checkpoints.sh` 的合并 checkpoint、分片推理和 JSONL 指标评测逻辑；每个脚本开头都可以单独改 `EXPERIMENT_NAME`、`GAMMA`、`TAU`，也可以用 `FORMAL_EVAL_*` 覆盖评测参数。

运行中的正式评测也可用同一个脚本加 `stop` 停止，例如 `bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh stop`。

| 脚本 | 用途 |
|---|---|
| `00_eval_baseline_naive.sh` | 评测 T0 baseline |
| `01_eval_baseline_entropy_mean.sh` | 评测 T1 baseline |
| `02`-`05_eval_gamma_search_ngram_overlap_*.sh` | 评测单个 gamma-search checkpoint |
| `06_eval_gamma_search_best.sh` | 评测自动选择出的最佳 gamma-search checkpoint |
| `07`-`10_eval_aer_*_best_gamma.sh` | 评测单个主 AER 或补充 AER checkpoint |

## 需要人工确认

| 项目 | 说明 |
|---|---|
| `WANDB_API_KEY` | 不应提交到 Git，本地填写或把 `WANDB_MODE` 改成 `offline/disabled` |
| `MODEL_PATH` | 默认 `/data/models/Qwen/Qwen3-4B`，按服务器实际路径修改 |
| `EMBEDDING_MODEL_PATH` | semantic embedding 训练和 semantic-cosine 评测需要 |
| `SIMILARITY_*` | 控制 semantic_embedding 使用 CPU、单 GPU 或多 GPU |
| `CALIBRATION_DELAYED_ALGORITHMS` | 默认 `semantic_embedding`，T0 中延后到最后 10% 步才记录的耗时算法 |
| `GAMMA_BEST` | 默认 `auto`，也可根据 gamma 搜索评测结果人工指定 |
| `FORMAL_EVAL_*` | 正式完整评测专用配置，不与训练期间 CPU watcher 评测共用 |

## 可选扩展

增加补充算法时，先把算法加入校准列表，保证能从同一次 T0 baseline 取到 `min_exploration_reward`：

```bash
CALIBRATION_METRIC_ALGORITHMS="token_match ngram_overlap semantic_embedding simhash"
RUN_EXTRA_AER=1
EXTRA_SIMILARITY_ALGORITHMS="simhash"
```

如果只想评测已有 checkpoint，不跑默认自动收集的实验，可在 `config.env` 中手动指定：

```bash
FORMAL_EVAL_EXPERIMENT_NAMES="baseline-naive-calib-tau0-s504 aer-token_match-g1p2-tau0p123456-s504"
FORMAL_EVAL_INCLUDE_BASELINE_NAIVE=0
FORMAL_EVAL_INCLUDE_BASELINE_ENTROPY=0
FORMAL_EVAL_INCLUDE_GAMMA_SEARCH=0
FORMAL_EVAL_MAIN_ALGORITHMS=""
```

每个训练实验仍会占满 `N_GPUS_PER_NODE` 张训练 GPU，脚本默认顺序执行；semantic embedding 的额外设备由 `SIMILARITY_CUDA_VISIBLE_DEVICES` 单独控制。
