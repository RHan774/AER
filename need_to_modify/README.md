# 实验运行说明

> 一键运行：
> ```bash
> bash need_to_modify/run_all_nohup.sh
> ```

### 1. 修改配置
先检查并修改 `need_to_modify/config.env` 顶部：

```bash
bash need_to_modify/run_experiments.sh status
```

重点确认：

```bash
PRIMUS_OUTPUT_DIR="/path/to/primus/output"
RUN_1_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
RUN_1_N_GPUS_PER_NODE=8
RUN_1_SIMILARITY_CUDA_VISIBLE_DEVICES="[0,1,2,3,4,5,6,7]"
RUN_1_SIMILARITY_NUM_PROCESSES=8
RUN_1_FORMAL_EVAL_GPUS="0,1,2,3,4,5,6,7"
RUN_2_CUDA_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
RUN_2_N_GPUS_PER_NODE=8
RUN_2_SIMILARITY_CUDA_VISIBLE_DEVICES="[8,9,10,11,12,13,14,15]"
RUN_2_SIMILARITY_NUM_PROCESSES=8
RUN_2_FORMAL_EVAL_GPUS="8,9,10,11,12,13,14,15"
WANDB_API_KEY=""
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Base"
EMBEDDING_MODEL_PATH="/data/models/Qwen/Qwen3-Embedding-0.6B"
DATA_DIR="/home/ruanruihan/AER/save/data"
SIMILARITY_DEVICE="cuda"
```

`PRIMUS_OUTPUT_DIR` 用于保存 checkpoints、validation、eval、日志和临时文件；`DATA_DIR` 只放 parquet 数据集，默认就是仓库的 `save/data`。

checkpoint 目录说明：

```bash
${PRIMUS_OUTPUT_DIR}/checkpoints              # 训练 state checkpoint，受 MAX_ACTOR_CKPT_TO_KEEP / MAX_CRITIC_CKPT_TO_KEEP 控制
${PRIMUS_OUTPUT_DIR}/inference_checkpoints    # 推理用 HuggingFace checkpoint，每个保存步都会保留
```

### 2. 配置环境
```bash
bash need_to_modify/run_experiments.sh setup
```

### 3. 运行实验
```bash
mkdir -p ${PRIMUS_OUTPUT_DIR}/run/nohup
nohup bash need_to_modify/1_run.sh > ${PRIMUS_OUTPUT_DIR}/run/nohup/1_run.log 2>&1 &
nohup bash need_to_modify/2_run.sh > ${PRIMUS_OUTPUT_DIR}/run/nohup/2_run.log 2>&1 &
```

也可以用一条命令完成配置环境并同时 nohup 启动两个实验队列：

```bash
bash need_to_modify/run_all_nohup.sh
```

### 4. 查看实验进展
```bash
tail -f ${PRIMUS_OUTPUT_DIR}/run/nohup/1_run_latest.log
tail -f ${PRIMUS_OUTPUT_DIR}/run/nohup/2_run_latest.log
tail -f ${PRIMUS_OUTPUT_DIR}/run/nohup/1_run.log
tail -f ${PRIMUS_OUTPUT_DIR}/run/nohup/2_run.log
```

### 5. 终止实验
```bash
bash need_to_modify/1_run.sh stop
bash need_to_modify/2_run.sh stop
bash need_to_modify/run_all_nohup.sh stop
```
