"""训练日志解析与 tau 计划生成。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import strip_ansi, write_csv, write_json

DEFAULT_TRAIN_KEYS = [
    "metric/exploration reward",
    "metric/weight",
    "metric/acc reward",
    "actor/entropy_loss",
    "response_length/clip_ratio",
    "response_length/mean",
]

TAU_WINDOWS = [
    ("E_1_24", "tau_1", 1, 24),
    ("E_25_48", "tau_2", 25, 48),
    ("E_49_72", "tau_3", 49, 72),
]


def parse_metric_value(text: str) -> float | None:
    """解析日志中的数值，兼容 `np.float64(0.1)` 形式。"""

    match = re.match(r"\s*(?:np\.float64\()?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_train_log(path: str | Path) -> list[dict[str, Any]]:
    """解析训练日志中形如 `step:12 - key:value` 的指标行。"""

    rows_by_step: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean_line = strip_ansi(line)
            step_match = re.search(r"\bstep\s*:\s*(\d+)", clean_line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            row = rows_by_step.setdefault(step, {"step": step})
            for part in clean_line.split(" - "):
                if ":" not in part:
                    continue
                key, value = part.rsplit(":", 1)
                key = key.strip()
                if key.endswith("step") or key == "step":
                    continue
                metric_value = parse_metric_value(value)
                if metric_value is not None:
                    row[key] = metric_value

    return [rows_by_step[step] for step in sorted(rows_by_step)]


def coerce_float(value: Any) -> float | None:
    """将 JSON/CSV/日志中的数值统一转成 float。"""

    if value is None or value == "":
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return parse_metric_value(str(value))


def load_train_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取训练指标，兼容原始日志、已导出的 JSON 和 CSV。"""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"训练指标 JSON 必须是列表: {path}")
        return [dict(row) for row in data if isinstance(row, dict)]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return parse_train_log(path)


def export_train_log(input_path: str, output_dir: str | Path, keys: list[str] | None = None, all_keys: bool = False) -> None:
    """导出训练动态指标 CSV/JSON。"""

    rows = parse_train_log(input_path)
    if not rows:
        raise ValueError(f"没有在日志中解析到 step 指标: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_keys:
        field_set: set[str] = {"step"}
        for row in rows:
            field_set.update(row.keys())
        fieldnames = ["step", *sorted(field_set - {"step"})]
    else:
        selected = keys or DEFAULT_TRAIN_KEYS
        fieldnames = ["step", *[key for key in selected if any(key in row for row in rows)]]

    write_csv(output_dir / "train_metrics.csv", rows, fieldnames=fieldnames)
    write_json(output_dir / "train_metrics.json", rows)


def mean_ignore_none(values: list[float]) -> float | None:
    """空列表返回 None 的均值函数。"""

    return float(np.mean(values)) if values else None


def values_in_window(values: list[tuple[int, float]], start: int, end: int) -> list[float]:
    """取闭区间 step 内的指标值。"""

    return [value for step, value in values if start <= step <= end]


def metric_points(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    """从训练指标行中提取指定 key 的 step 序列。"""

    points: list[tuple[int, float]] = []
    for row in rows:
        step = coerce_float(row.get("step"))
        value = coerce_float(row.get(key))
        if step is None or value is None:
            continue
        points.append((int(step), value))
    return sorted(points)


def round_tau(value: float | None, precision: int) -> float | None:
    """tau 统一保留指定小数，并避免负值。"""

    if value is None:
        return None
    return round(max(float(value), 0.0), precision)


def make_tau_plan(input_path: str, algorithm: str, precision: int = 3) -> dict[str, Any]:
    """根据 tau=0 校准日志生成 tau 候选值。"""

    rows = load_train_rows(input_path)
    values = metric_points(rows, "metric/exploration reward")
    if not values:
        raise ValueError("日志中没有 metric/exploration reward，无法生成 tau 计划")

    row: dict[str, Any] = {"algorithm": algorithm}
    aliases = [("E_early", "tau_low"), ("E_mid", "tau_mid"), ("E_late", "tau_high")]
    for (stat_name, tau_name, start, end), (stat_alias, tau_alias) in zip(TAU_WINDOWS, aliases):
        window_values = values_in_window(values, start, end)
        mean_value = mean_ignore_none(window_values)
        if mean_value is None:
            raise ValueError(f"step {start}-{end} 没有 metric/exploration reward，无法生成 {tau_name}")
        tau = round_tau(mean_value, precision)
        row[stat_name] = mean_value
        row[tau_name] = tau
        row[stat_alias] = mean_value
        row[tau_alias] = tau
    return row


def export_tau_plan(input_path: str, algorithm: str, output_path: str | Path, precision: int = 3) -> dict[str, Any]:
    """生成并写出 tau 计划。"""

    row = make_tau_plan(input_path=input_path, algorithm=algorithm, precision=precision)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        write_json(output_path, row)
    else:
        fieldnames = [
            "algorithm",
            "E_1_24",
            "E_25_48",
            "E_49_72",
            "E_early",
            "E_mid",
            "E_late",
            "tau_1",
            "tau_2",
            "tau_3",
            "tau_low",
            "tau_mid",
            "tau_high",
        ]
        write_csv(output_path, [row], fieldnames=fieldnames)
    return row


def dumps_tau_plan(row: dict[str, Any]) -> str:
    """格式化 tau 计划用于命令行打印。"""

    return json.dumps(row, ensure_ascii=False, indent=2)
