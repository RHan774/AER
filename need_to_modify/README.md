# 实验运行说明

### 1. 修改配置
先检查并修改 `need_to_modify/config.env`：

```bash
bash need_to_modify/run_experiments.sh status
```

重点确认：

```bash
WANDB_API_KEY=""
MODEL_PATH="/data/models/Qwen/Qwen3-4B-Base"
EMBEDDING_MODEL_PATH="/data/models/Qwen/Qwen3-Embedding-0.6B"
CUDA_VISIBLE_DEVICES="0,1,2,3"
SIMILARITY_DEVICE="cuda"
SIMILARITY_CUDA_VISIBLE_DEVICES="[4,5,6,7]"
SIMILARITY_NUM_PROCESSES=4
```

### 2. 配置环境
```bash
# 只安装环境
bash need_to_modify/run_experiments.sh setup
# 只下载模型和数据
bash need_to_modify/run_experiments.sh assets
# 只跑轻量测试（确认环境没问题）
bash need_to_modify/run_experiments.sh test
```

### 3. 运行实验
```bash
nohup bash need_to_modify/run_serial_aer_360.sh > master.log 2>&1 &
```

### 4. 查看实验进展
```bash
tail -f master.log
```

### 5. 终止实验
```bash
bash need_to_modify/run_serial_aer_360.sh stop
```