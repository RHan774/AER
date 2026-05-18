# 训练单实验脚本

这个目录中的脚本用于单独运行一个训练实验。每个脚本都会调用 `../run_experiments.sh` 中的完整训练逻辑，并包含训练期间每个 validation step 的 CPU watcher 评测。

推荐顺序：

1. `00_baseline_naive_calib.sh`：生成 tau 表。
2. `02`-`05_gamma_search_ngram_overlap_*.sh`：逐个运行 gamma 搜索。
3. `06_select_gamma_best.sh`：根据训练期间 CPU 评测结果选择最佳 gamma。
4. `07`-`10_aer_*_best_gamma.sh`：逐个运行主 AER 或补充 AER。

每个脚本开头都有“可单独修改的实验参数”。也可以用环境变量临时覆盖，例如：

```bash
DRY_RUN=1 bash need_to_modify/train_experiment_scripts/02_gamma_search_ngram_overlap_g1p1.sh
GAMMA=1.3 bash need_to_modify/train_experiment_scripts/07_aer_token_match_best_gamma.sh
TOTAL_STEPS=504 bash need_to_modify/train_experiment_scripts/09_aer_semantic_embedding_best_gamma.sh
```

停止某个正在运行的训练实验时，在同一个脚本后加 `stop`：

```bash
bash need_to_modify/train_experiment_scripts/03_gamma_search_ngram_overlap_g1p2.sh stop
```

停止命令会读取 `save/run/script_pids/train/<script>.pid`，优先停止该脚本的独立进程组，兜底递归停止子进程。

如果直接运行 gamma 或 AER 脚本，需要确保 `save/eval/tau_plan_<algorithm>.csv` 已存在；如果 `GAMMA=auto` 或未指定，需要确保 `save/eval/gamma_best_<algorithm>.env` 已存在。
