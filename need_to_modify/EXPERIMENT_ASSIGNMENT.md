# 实验分工

## 默认负责内容

该目录现在按 `新实验计划.md` 组织完整队列，默认写入 `need_to_modify/config.env`：

| 阶段 | 默认实验 | 配置项 |
|---|---|---|
| T0 tau 校准 | `baseline-naive-calib-tau0-s504`，同时记录 `token_match`、`ngram_overlap`、`semantic_embedding` 探索奖励 | `RUN_BASELINE_NAIVE=1`、`CALIBRATION_METRIC_ALGORITHMS=...` |
| T1 熵 baseline | `baseline-entropy-mean-tau0-s504` | `RUN_BASELINE_ENTROPY=1`、`ENTROPY_BASELINE_COEFF=5e-4` |
| T2-T5 gamma 搜索 | `gamma-search-ngram_overlap-g{gamma}-tau{tau}-s504` | `RUN_GAMMA_SEARCH=1`、`GAMMA_LIST="1.1 1.2 1.3 1.4"` |
| T6-T8 主 AER | `token_match`、`ngram_overlap`、`semantic_embedding` 复用 `gamma_best` | `RUN_MAIN_AER=1`、`MAIN_SIMILARITY_ALGORITHMS=...` |
| 完整评测 | baseline、最佳 gamma-search、主 AER checkpoint 的 Pass@K 与多样性指标 | `run_eval_formal_checkpoints.sh`、`FORMAL_EVAL_*` |

默认训练到 `504` step，每 36 step 验证和保存一次，对齐当前 `verl/recipe/aer/run.sh`。

## 需要人工确认

| 项目 | 说明 |
|---|---|
| `WANDB_API_KEY` | 不应提交到 Git，本地填写或把 `WANDB_MODE` 改成 `offline/disabled` |
| `MODEL_PATH` | 默认 `/data/models/Qwen/Qwen3-4B`，按服务器实际路径修改 |
| `EMBEDDING_MODEL_PATH` | semantic embedding 训练和 semantic-cosine 评测需要 |
| `SIMILARITY_*` | 控制 semantic_embedding 使用 CPU、单 GPU 或多 GPU |
| `CALIBRATION_DELAYED_ALGORITHMS` | 默认 `semantic_embedding`，T0 中延后到最后 10% 步才记录的耗时算法 |
| `GAMMA_BEST` | 默认 `auto`，也可根据 gamma 搜索评测结果人工指定 |

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
