#!/bin/bash
################################################################################
# 启动AER训练监控守护进程
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/aer_monitor.py"

echo "=========================================="
echo "   AER训练监控 - 启动脚本"
echo "=========================================="
echo ""

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate aer

# 检查监控脚本是否存在
if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
    echo "错误: 监控脚本不存在: ${MONITOR_SCRIPT}"
    exit 1
fi

# 检查是否已在运行
if [[ -f "${SCRIPT_DIR}/.daemon_pid" ]]; then
    OLD_PID=$(cat "${SCRIPT_DIR}/.daemon_pid")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "监控守护进程已在运行 (PID: ${OLD_PID})"
        echo "如需重启，请先运行: bash stop.sh"
        exit 1
    else
        echo "清理旧的PID文件..."
        rm -f "${SCRIPT_DIR}/.daemon_pid"
    fi
fi

# 移除停止标志
rm -f "${SCRIPT_DIR}/.stop_flag"

echo "启动配置:"
echo "  检查间隔: 5分钟"
echo "  日志目录: ${SCRIPT_DIR}/logs/"
echo "  崩溃报告: ${SCRIPT_DIR}/crash_reports/"
echo ""

# 启动守护进程
python "${MONITOR_SCRIPT}" --daemon

echo ""
echo "监控守护进程已启动"
echo "查看日志: tail -f ${SCRIPT_DIR}/logs/monitor_$(date +%Y%m%d).log"
echo "停止监控: bash ${SCRIPT_DIR}/stop.sh"
