# AER 服务器运行说明

这个目录用于把 AER 项目交给在 4 张 GPU 服务器上跑固定实验队列。默认只需要改一个配置文件，然后执行一个脚本。

## 1. 需要做的最少操作

> 如果服务器上未激活conda环境，需要先激活：
> ```bash
> conda init
> source ~/.bashrc
> ```

```bash
## 检查当前配置，如果有需要修改的，需要修改`need_to_modify/config.env`
bash need_to_modify/run_experiments.sh status
```

至少检查：

```bash
REPO_ROOT= # 当前项目的路径
SAVE_DIR= # 数据（数据集、模型、checkpoints、测试时生成的响应等）保存路径
CONFIG_FILE= # 应指向 AER/need_to_modify/config.env
CUDA_VISIBLE_DEVICES=0,1,2,3 # ！重要！使用GPU序号
# EXPERIMENT_ALGORITHMS=ngram_overlap simhash levenshtein semantic_embedding
# CALIBRATION_STEPS=72
# TOTAL_TRAINING_STEPS=240
# WANDB_MODE=online
# WANDB_PROJECT=AER
MODEL_PATH= # 模型下载路径，默认在${SAVE_DIR}下
```

如果模型和数据想放到其它磁盘，只改：

```bash
SAVE_DIR="/path/to/large_disk/aer_save"
```

然后运行：

```bash
nohup bash need_to_modify/run_experiments.sh all > master.log 2>&1 &
tail -f master.log
```

脚本会按顺序完成：创建/复用 conda 环境、安装本地代码、配置镜像和 wandb、下载模型、准备 parquet 数据、跑轻量测试、跑实验队列、导出训练日志和 validation JSONL 评测结果。

## 2. 默认实验队列

服务器默认跑：

1. `ngram_overlap`：72 step 校准，自动生成 `tau_low/tau_mid/tau_high`，再跑三档正式实验。
2. `simhash`：72 step 校准，自动生成 `tau_low/tau_mid/tau_high`，再跑三档正式实验。
3. `levenshtein`：72 step 校准，自动生成 `tau_low/tau_mid/tau_high`，再跑三档正式实验。
4. `semantic_embedding`：使用本地 `Qwen3-Embedding-0.6B` 在训练阶段计算探索奖励，72 step 校准后跑三档正式实验。

对应配置：

```bash
EXPERIMENT_ALGORITHMS="ngram_overlap simhash levenshtein semantic_embedding"
CALIBRATION_STEPS=72
TOTAL_TRAINING_STEPS=240
```

如果主 baseline 的最终比较步数可能是 320，启动前把 `TOTAL_TRAINING_STEPS=320`。

## 3. 常用命令

只查看配置：

```bash
bash need_to_modify/run_experiments.sh status
```

只搭环境：

```bash
bash need_to_modify/run_experiments.sh setup
```

只下载模型和准备数据：

```bash
bash need_to_modify/run_experiments.sh assets
```

只跑轻量测试：

```bash
bash need_to_modify/run_experiments.sh test
```

只跑训练队列：

```bash
bash need_to_modify/run_experiments.sh train
```

## 4. 输出位置

默认都在 `${SAVE_DIR}` 下：

| 路径 | 内容 |
|---|---|
| `save/models/` | Qwen policy 模型和 semantic-cosine 评测用 embedding 模型 |
| `save/data/` | 训练/验证 parquet |
| `save/checkpoints/<exp>/` | FSDP checkpoint |
| `save/validation/<exp>/` | 训练中保存的 validation JSONL |
| `save/eval/<exp>/train_log/` | 从训练日志导出的曲线 CSV/JSON |
| `save/eval/<exp>/jsonl/` | 从 validation JSONL 计算的 Pass@K、多样性指标 |
| `save/run/logs/<exp>.log` | 每个实验的完整 stdout/stderr |

wandb run name 与本地 `<exp>` 一致。

## 5. 出错后续跑

脚本每个实验结束后会写 marker：

```bash
save/run/state/<exp>.done
```

如果中途断掉，重新执行同一个命令即可。未完成实验会用 `trainer.resume_mode=auto` 从已有 checkpoint 继续，已完成实验会跳过。

如果确实要重跑全部实验：

```bash
FORCE_RERUN=1 bash need_to_modify/run_experiments.sh train
```

## 6. 注意

默认 `STOP_RAY_BETWEEN_RUNS=1`，脚本会在每个实验前后执行 `ray stop --force`，避免 Ray 旧进程影响下一轮训练。服务器如果还跑着别人的 Ray 任务，需要先把该项改成 `0`。

Ray 的临时目录默认使用 `RAY_TMPDIR=${HOME}/rt`，放在用户目录下且路径足够短。不要把它设成很长的项目路径，否则 Ray 的 Unix socket 路径可能超过系统限制并在启动时报 `AF_UNIX path length cannot exceed 107 bytes`。

默认会跑训练阶段的 `semantic_embedding` 相似度算法，并在 assets 阶段下载 embedding 模型。它比 `ngram_overlap`、`simhash`、`levenshtein` 更吃 CPU/内存；如果该服务器资源不足，可以从 `EXPERIMENT_ALGORITHMS` 中临时移除 `semantic_embedding`，但保留 `DOWNLOAD_EMBEDDING_MODEL=1` 仍可用于训练结束后的 `semantic-cosine` 离线评测。