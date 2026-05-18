# 正式评测单实验脚本

这个目录中的脚本用于单独评测一个训练好的 checkpoint。每个脚本都会调用 `../run_eval_formal_checkpoints.sh` 中的完整正式评测逻辑，包括 checkpoint 合并、分片推理和 JSONL 指标评测。

常用入口：

```bash
bash need_to_modify/formal_eval_scripts/00_eval_baseline_naive.sh
bash need_to_modify/formal_eval_scripts/06_eval_gamma_search_best.sh
bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh
```

每个脚本开头都有“可单独修改的评测参数”。也可以用环境变量覆盖 `FORMAL_EVAL_*`：

```bash
DRY_RUN=1 bash need_to_modify/formal_eval_scripts/06_eval_gamma_search_best.sh
FORMAL_EVAL_CHECKPOINT_STEP=504 bash need_to_modify/formal_eval_scripts/07_eval_aer_token_match_best_gamma.sh
FORMAL_EVAL_METRICS="pass@k,first@1,distinct-2,self-bleu" bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh
```

停止某个正在运行的正式评测时，在同一个脚本后加 `stop`：

```bash
bash need_to_modify/formal_eval_scripts/09_eval_aer_semantic_embedding_best_gamma.sh stop
```

停止命令会读取 `save/run/script_pids/formal_eval/<script>.pid`，优先停止该脚本的独立进程组，兜底递归停止子进程。

如果评测 gamma-search 或 AER 脚本，需要确保对应 tau 表和 `gamma_best_<algorithm>.env` 已存在，或者在脚本/环境变量中手动指定 `TAU` 与 `GAMMA`。
