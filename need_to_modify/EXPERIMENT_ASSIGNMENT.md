# 实验分工

## 该服务器默认负责

该服务器适合跑中途不需要人工判断的固定队列。默认队列已经写入 `config.example.env`：

| 类型 | 实验 | 原因 |
|---|---|---|
| AER 校准 | `ngram_overlap`、`simhash`、`levenshtein`、`semantic_embedding` 的 `tau=0` 校准 | 每个算法先跑 72 step，脚本可自动从 `metric/exploration reward` 生成 tau |
| AER 正式实验 | `ngram_overlap` 的 `tau_low/tau_mid/tau_high` | 轻量、无需额外模型、能检验 token n-gram 探索奖励 |
| AER 正式实验 | `simhash` 的 `tau_low/tau_mid/tau_high` | 轻量、低内存、适合长输出近重复惩罚 |
| AER 正式实验 | `levenshtein` 的 `tau_low/tau_mid/tau_high` | 编辑距离定义清晰，可作为字符级相似度对照 |
| AER 校准与正式实验 | `semantic_embedding` 的 `tau_low/tau_mid/tau_high` | 使用 embedding 模型直接度量语义相似度，作为语义探索奖励主对照 |

默认每个正式实验训练到 `TOTAL_TRAINING_STEPS=240`，每 12 step 验证一次，每 24 step 保存一次。若你这边 naive GRPO 还没有确定最终 `S_final`，启动前可以把 `TOTAL_TRAINING_STEPS` 改成 `320`，后续统一从 wandb/validation/checkpoint 里选同一个 step 比较。

## 原服务器继续负责

| 类型 | 实验 | 原因 |
|---|---|---|
| naive GRPO | 当前正在跑的 naive 扫描/正式 baseline | 决定统一 `S_final`，需要根据收敛情况判断 |
| entropy baseline | 当前另一台服务器上的熵正则 baseline | 你能更方便地根据 collapse、长度、Pass@K 判断是否调整系数 |
| 二阶段调参 | 根据前面结果追加 tau 或改训练步数 | 需要看 wandb 后人工决策 |
| 重型/敏感算法 | `semantic_embedding`、必要时 `levenshtein` | 可能受 CPU、内存、embedding 模型下载与速度影响，更适合可随时调整的机器 |

## 可选扩展

如果该服务器时间充足，可以在 `config.env` 中调整：

```bash
EXPERIMENT_ALGORITHMS="ngram_overlap simhash token_match levenshtein semantic_embedding"
TOTAL_TRAINING_STEPS=320
```

如果你希望该服务器补齐熵正则系数网格，可填写：

```bash
ENTROPY_BASELINE_COEFFS="5e-4 2e-3"
```

不要同时开启太多扩展。每个训练都会占满 4 张 GPU，脚本默认顺序执行。
