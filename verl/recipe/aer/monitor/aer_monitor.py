#!/usr/bin/env python3
"""
AER训练自动监控和恢复脚本
每30分钟检查一次训练状态，自动检测并修复常见错误

使用方法:
    python aer_monitor.py              # 单次检查
    python aer_monitor.py --daemon     # 作为守护进程运行（每5分钟）
    python aer_monitor.py --stop       # 停止守护进程
"""

import os
import sys
import re
import subprocess
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import signal

# ==================== 配置 ====================
# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
AER_DIR = SCRIPT_DIR.parent
REPO_DIR = AER_DIR.parent

LOG_FILE = AER_DIR / "log.txt"
RUN_SCRIPT = AER_DIR / "run.sh"
MONITOR_LOG_DIR = SCRIPT_DIR / "logs"
CRASH_DIR = SCRIPT_DIR / "crash_reports"
PID_FILE = SCRIPT_DIR / ".daemon_pid"
STOP_FILE = SCRIPT_DIR / ".stop_flag"
COUNTER_FILE = SCRIPT_DIR / ".monitor_counter"

CONDA_ENV = "aer"
CHECK_INTERVAL = 300  # 5分钟

# 确保目录存在
MONITOR_LOG_DIR.mkdir(exist_ok=True)
CRASH_DIR.mkdir(exist_ok=True)


# ==================== 日志函数 ====================
def log_message(msg, level="INFO", also_print=True):
    """写入监控日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"

    # 写入每日监控日志
    daily_log = MONITOR_LOG_DIR / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
    with open(daily_log, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

    if also_print:
        print(log_msg, flush=True)

    return log_msg


def get_counter():
    """获取监控计数器"""
    if COUNTER_FILE.exists():
        try:
            return int(COUNTER_FILE.read_text().strip())
        except:
            return 0
    return 0


def increment_counter():
    """增加计数器"""
    counter = get_counter() + 1
    COUNTER_FILE.write_text(str(counter))
    return counter


# ==================== 进程管理 ====================
def is_process_running():
    """检查main_ppo进程是否在运行"""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "main_ppo"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            pids = [line.split()[0] for line in result.stdout.strip().split("\n") if line.strip()]
            if pids:
                return True, pids
        return False, []
    except Exception as e:
        log_message(f"检查进程时出错: {e}", "ERROR")
        return False, []


def start_training_process():
    """启动训练进程"""
    log_message("正在启动训练进程...", "INFO")

    try:
        # 激活conda环境并运行
        cmd = f"""
        cd {AER_DIR} && \
        source ~/.bashrc && \
        conda activate {CONDA_ENV} && \
        nohup bash run.sh >> log.txt 2>&1 &
        echo $! > {SCRIPT_DIR}/.training_pid
        """

        subprocess.run(cmd, shell=True, timeout=30)
        time.sleep(5)

        running, pids = is_process_running()
        if running:
            log_message(f"训练进程已启动，PID: {pids}", "INFO")
            return True
        else:
            log_message("训练进程启动后未检测到main_ppo进程", "WARN")
            return False

    except Exception as e:
        log_message(f"启动训练进程时出错: {e}", "ERROR")
        return False


def stop_training_process():
    """停止训练进程"""
    log_message("正在停止训练进程...", "INFO")
    try:
        subprocess.run(["pkill", "-f", "main_ppo"], timeout=30)
        time.sleep(2)
        subprocess.run(["pkill", "-9", "-f", "main_ppo"], timeout=10)
        log_message("训练进程已停止", "INFO")
    except Exception as e:
        log_message(f"停止训练进程时出错: {e}", "ERROR")


# ==================== 日志分析 ====================
def read_log_tail(lines=500):
    """读取日志文件的最后N行"""
    if not LOG_FILE.exists():
        return ""
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(LOG_FILE)],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except Exception as e:
        log_message(f"读取日志文件时出错: {e}", "ERROR")
        return ""


def analyze_log(log_content):
    """分析日志内容，检测错误类型"""
    errors = {
        "oom": False,
        "nccl_error": False,
        "ray_error": False,
        "file_error": False,
        "normal_exit": False,
        "other_errors": []
    }

    # OOM错误检测
    oom_patterns = [
        r"Cuda failure.*'out of memory'",
        r"RuntimeError: CUDA out of memory",
        r"torch\.cuda\.OutOfMemoryError",
        r"CUDA out of memory"
    ]

    for pattern in oom_patterns:
        if re.search(pattern, log_content, re.IGNORECASE):
            errors["oom"] = True
            break

    # NCCL错误
    if re.search(r"NCCL error|DistBackendError", log_content):
        errors["nccl_error"] = True

    # Ray错误
    if re.search(r"ray\.|actor|placement group|ConnectionRefusedError|requests\.exceptions\.HTTPError.*502", log_content, re.IGNORECASE):
        errors["ray_error"] = True

    # 文件错误
    if re.search(r"FileNotFoundError|OSError.*No such file", log_content):
        errors["file_error"] = True

    # 正常结束
    if re.search(r"Training completed|All epochs finished|Training finished successfully", log_content):
        errors["normal_exit"] = True

    # 提取错误信息
    error_patterns = [
        (r"KeyError:\s+'(\w+)'", "KeyError", "缺少必要的配置或数据字段"),
        (r"AttributeError:\s+'(\w+)'", "AttributeError", "属性访问错误"),
        (r"ValueError:\s+(.+)", "ValueError", "参数值错误"),
        (r"PermissionDenied:\s+(.+)", "PermissionError", "文件权限错误"),
    ]

    for pattern, err_type, desc in error_patterns:
        matches = re.findall(pattern, log_content)
        if matches:
            for match in matches:
                errors["other_errors"].append({
                    "type": err_type,
                    "message": f"{desc}: {match}",
                    "detail": str(match)
                })

    return errors


# ==================== 错误处理 ====================
def get_current_gpu_memory_utilization():
    """获取run.sh中的gpu_memory_utilization值"""
    try:
        content = RUN_SCRIPT.read_text()
        match = re.search(r"gpu_memory_utilization=([\d.]+)", content)
        if match:
            return float(match.group(1))
    except Exception as e:
        log_message(f"读取gpu_memory_utilization时出错: {e}", "ERROR")
    return 0.75


def update_gpu_memory_utilization(current_value, decrement=0.02):
    """更新run.sh中的gpu_memory_utilization参数"""
    new_value = round(max(0.5, current_value - decrement), 2)

    if new_value == current_value:
        log_message(f"gpu_memory_utilization已达到最小值 {current_value}", "WARN")
        return False, current_value

    # 创建备份
    backup_file = RUN_SCRIPT.with_suffix(f".sh.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(RUN_SCRIPT, backup_file)
    log_message(f"已创建备份: {backup_file.name}", "INFO")

    try:
        content = RUN_SCRIPT.read_text()
        new_content = re.sub(
            f"gpu_memory_utilization={current_value}",
            f"gpu_memory_utilization={new_value}",
            content
        )
        RUN_SCRIPT.write_text(new_content)
        log_message(f"gpu_memory_utilization: {current_value} -> {new_value}", "INFO")
        return True, new_value

    except Exception as e:
        log_message(f"更新gpu_memory_utilization时出错: {e}", "ERROR")
        shutil.copy2(backup_file, RUN_SCRIPT)
        return False, current_value


def create_crash_report(error_type, details, actions_taken="", suggested_fix=""):
    """创建崩溃报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = CRASH_DIR / f"{error_type}_{timestamp}.md"

    report_content = f"""# {error_type} 错误报告

**时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**日志文件**: `{LOG_FILE.relative_to(REPO_DIR)}`
**运行脚本**: `{RUN_SCRIPT.relative_to(REPO_DIR)}`

## 错误详情

```
{details}
```

## 已执行操作

{actions_taken}

## 建议修复方案

{suggested_fix}

---

此报告由 `aer_monitor.py` 自动生成
"""

    report_file.write_text(report_content, encoding="utf-8")
    log_message(f"已创建崩溃报告: {report_file.name}", "INFO")
    return report_file


def handle_oom_error():
    """处理OOM错误"""
    log_message(">>> 处理OOM错误", "WARN")

    current_value = get_current_gpu_memory_utilization()
    log_message(f"当前gpu_memory_utilization: {current_value}", "INFO")

    if current_value <= 0.5:
        log_message("gpu_memory_utilization已过低，需要手动检查", "ERROR")
        create_crash_report(
            "OOM_CannotFix",
            f"gpu_memory_utilization已降至{current_value}，但仍然OOM",
            "已停止自动调整",
            "建议: 1) 减小train_batch_size 2) 减小max_token_len_per_gpu 3) 检查GPU内存占用"
        )
        return False

    # 停止进程
    stop_training_process()
    time.sleep(2)

    # 降低gpu_memory_utilization
    success, new_value = update_gpu_memory_utilization(current_value, decrement=0.02)
    if success:
        actions = f"""
1. 停止旧进程
2. 备份run.sh
3. gpu_memory_utilization: {current_value} -> {new_value}
4. 重启训练
"""
        create_crash_report(
            "OOM_AutoFixed",
            f"CUDA Out of Memory",
            actions,
            f"已自动降低GPU内存利用率到 {new_value}，将在5分钟后检查恢复情况"
        )
        return start_training_process()

    return False


def handle_nccl_error():
    """处理NCCL错误"""
    log_message(">>> 处理NCCL错误", "WARN")

    details = """
NCCL通信错误，通常由以下原因引起：
1. GPU间通信失败
2. 网络问题（多机训练时）
3. NCCL版本不兼容
4. GPU内存不足导致的通信失败
"""

    suggested_fix = """
1. 检查GPU状态: nvidia-smi
2. 尝试减少GPU数量
3. 检查NCCL版本: python -c "import torch; print(torch.cuda.nccl.version())"
4. 如果是单机多卡，尝试设置: export NCCL_P2P_DISABLE=1
5. 清理Ray缓存后重启
"""

    create_crash_report("NCCL_Error", details, "已记录，需人工处理", suggested_fix)
    log_message("NCCL错误需要人工处理", "WARN")
    return False


def handle_ray_error():
    """处理Ray错误"""
    log_message(">>> 处理Ray错误，尝试自动修复", "WARN")

    # 清理Ray缓存
    ray_dirs = ["~/ray_tmp", "/tmp/ray", "/tmp/ray_session_latest"]
    cleaned = []

    for ray_dir in ray_dirs:
        ray_path = Path(ray_dir).expanduser()
        if ray_path.exists():
            try:
                shutil.rmtree(ray_path)
                cleaned.append(str(ray_dir))
                log_message(f"已清理: {ray_dir}", "INFO")
            except Exception as e:
                log_message(f"清理失败 {ray_dir}: {e}", "WARN")

    if cleaned:
        actions = f"清理了以下Ray缓存目录:\n" + "\n".join(f"- {d}" for d in cleaned)
        create_crash_report("Ray_AutoFixed", "Ray相关错误，已清理缓存", actions, "重启训练")
        return start_training_process()

    create_crash_report("Ray_Error", "Ray错误但无法自动清理缓存", "", "手动执行: ray stop && rm -rf /tmp/ray && ray start")
    return False


def handle_file_error(error_info):
    """处理文件错误"""
    log_message(">>> 处理文件错误", "WARN")

    details = f"""文件错误: {error_info.get('message', 'Unknown')}
检查路径是否存在且可访问
"""

    create_crash_report("File_Error", details, "", "检查文件路径和权限")
    return False


def handle_other_error(errors):
    """处理其他类型的错误"""
    log_message(">>> 处理其他错误", "WARN")

    all_details = []
    for err in errors["other_errors"]:
        all_details.append(f"- {err['type']}: {err['message']}")

    details = "\n".join(all_details)
    create_crash_report("Other_Error", details, "", "需要根据具体错误类型手动处理")
    return False


# ==================== 主监控循环 ====================
def main_monitor():
    """执行一次监控检查"""
    log_message("=" * 50)
    log_message("开始监控检查", "INFO")

    counter = increment_counter()
    log_message(f"监控计数: #{counter}", "INFO")

    # 检查进程状态
    running, pids = is_process_running()

    # 读取日志
    log_content = read_log_tail()
    errors = analyze_log(log_content)

    # 检查正常结束
    if errors.get("normal_exit"):
        log_message("检测到训练正常结束", "INFO")
        create_crash_report("Training_Completed", "训练已正常完成", "", "监控将停止")
        if STOP_FILE.exists():
            STOP_FILE.unlink()
        return "stop"

    if running:
        log_message(f"进程正常运行 (PID: {pids[0] if pids else 'unknown'})", "INFO")

        # 检查日志是否长时间未更新
        if LOG_FILE.exists():
            log_age = time.time() - LOG_FILE.stat().st_mtime
            if log_age > 3600:  # 1小时
                log_message(f"警告: 日志文件已{log_age//60}分钟未更新", "WARN")
    else:
        log_message("进程未运行，检查是否崩溃...", "WARN")

        # 检测错误类型并处理
        if errors["oom"]:
            handle_oom_error()
        elif errors["nccl_error"]:
            handle_nccl_error()
        elif errors["ray_error"]:
            handle_ray_error()
        elif errors["file_error"]:
            handle_file_error({})
        elif errors["other_errors"]:
            handle_other_error(errors)
        elif log_content and ("Error" in log_content or "error" in log_content):
            # 有错误但未匹配到已知模式
            error_lines = []
            for line in log_content.split("\n"):
                if "Error" in line or "error" in line or "Exception" in line:
                    error_lines.append(line)
            error_text = "\n".join(error_lines[-10:]) if error_lines else "未知错误"
            create_crash_report("Unknown_Error", error_text, "", "需要人工分析日志")
        else:
            log_message("进程未运行且无错误日志", "INFO")

    log_message("监控检查完成", "INFO")
    log_message("=" * 50)
    return "continue"


def daemon_loop():
    """守护进程循环"""
    log_message("启动守护进程模式", "INFO")
    log_message(f"检查间隔: {CHECK_INTERVAL}秒", "INFO")

    # 写入PID
    PID_FILE.write_text(str(os.getpid()))

    # 设置信号处理
    def signal_handler(signum, frame):
        log_message(f"收到信号 {signum}，准备退出...", "INFO")
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    consecutive_failures = 0
    max_failures = 5

    try:
        while not STOP_FILE.exists():
            result = main_monitor()

            if result == "stop":
                log_message("训练正常结束，守护进程退出", "INFO")
                break

            # 等待下一次检查
            for _ in range(CHECK_INTERVAL):
                if STOP_FILE.exists():
                    break
                time.sleep(1)

            consecutive_failures = 0  # 重置失败计数

    except Exception as e:
        log_message(f"守护进程异常: {e}", "ERROR")
        import traceback
        log_message(traceback.format_exc(), "ERROR")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()
        log_message("守护进程已停止", "INFO")


def stop_daemon():
    """停止守护进程"""
    if not PID_FILE.exists():
        print("未找到运行中的守护进程")
        return False

    pid = int(PID_FILE.read_text())
    print(f"正在停止守护进程 (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
        # 创建停止标志
        STOP_FILE.touch()

        for _ in range(10):
            time.sleep(1)
            if not PID_FILE.exists():
                break
            try:
                os.kill(pid, 0)  # 检查进程是否存在
            except OSError:
                break
        else:
            print("优雅终止失败，强制终止...")
            os.kill(pid, signal.SIGKILL)

        print("守护进程已停止")
        return True

    except OSError as e:
        print(f"停止失败: {e}")
        return False


# ==================== 主入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AER训练自动监控和恢复")
    parser.add_argument("--daemon", action="store_true", help="作为守护进程运行")
    parser.add_argument("--stop", action="store_true", help="停止守护进程")

    args = parser.parse_args()

    if args.stop:
        stop_daemon()
    elif args.daemon:
        # 检查是否已有实例运行
        if PID_FILE.exists():
            old_pid = int(PID_FILE.read_text())
            try:
                os.kill(old_pid, 0)
                print(f"守护进程已在运行 (PID: {old_pid})")
                print("使用 --stop 停止现有进程")
                sys.exit(1)
            except OSError:
                PID_FILE.unlink()

        daemon_loop()
    else:
        # 单次检查
        main_monitor()
