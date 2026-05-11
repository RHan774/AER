#!/bin/bash
################################################################################
# 停止AER训练监控守护进程
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/aer_monitor.py"

echo "=========================================="
echo "   AER训练监控 - 停止脚本"
echo "=========================================="
echo ""

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate aer

# 使用Python脚本停止
python "${MONITOR_SCRIPT}" --stop

echo ""
echo "如需重新启动，运行: bash ${SCRIPT_DIR}/start.sh"
