#!/bin/bash
################################################################################
# 单次执行监控检查（不启动守护进程）
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/aer_monitor.py"

echo "=========================================="
echo "   AER训练监控 - 单次检查"
echo "=========================================="
echo ""

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate aer

# 执行单次检查
python "${MONITOR_SCRIPT}"

echo ""
echo "检查完成"
echo "查看完整日志: cat ${SCRIPT_DIR}/logs/monitor_$(date +%Y%m%d).log"
