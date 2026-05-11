"""AER 评测输入输出工具。"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """去掉日志中的 ANSI 颜色控制符。"""

    return ANSI_RE.sub("", text)


def ensure_output_dir(path: str | Path) -> Path:
    """创建输出目录并返回 Path。"""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_float(value: Any, digits: int = 6) -> str:
    """稳定格式化浮点数；None 和 NaN 输出为空。"""

    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """写 CSV 文件，保证空结果也有表头。"""

    if fieldnames is None:
        field_set: set[str] = set()
        for row in rows:
            field_set.update(row.keys())
        fieldnames = sorted(field_set)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row.get(key)) for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    """写 JSON 文件，保留中文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 Markdown 表格，方便复制到实验记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_float(row.get(key)) for key in fieldnames) + " |\n")


def jsonl_sort_key(path: Path) -> tuple[int, str]:
    """优先按 step 文件名排序，例如 12.jsonl、24.jsonl。"""

    try:
        return int(path.stem), str(path)
    except ValueError:
        return sys.maxsize, str(path)


def collect_jsonl_paths(inputs: Iterable[str]) -> list[Path]:
    """收集 JSONL 文件，支持文件、目录和 glob。"""

    paths: list[Path] = []
    for item in inputs:
        raw_path = Path(item)
        matched = [Path(match) for match in glob.glob(item)] if any(ch in item for ch in "*?[]") else []
        candidates = matched if matched else [raw_path]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.rglob("*.jsonl"), key=jsonl_sort_key))
            elif candidate.is_file() and candidate.suffix == ".jsonl":
                paths.append(candidate)

    unique_paths = sorted(set(paths), key=jsonl_sort_key)
    if not unique_paths:
        raise FileNotFoundError(f"没有找到 JSONL 文件: {list(inputs)}")
    return unique_paths


def infer_step_from_path(path: Path) -> int | None:
    """从 JSONL 文件名推断 step。"""

    try:
        return int(path.stem)
    except ValueError:
        return None


def load_jsonl_records(inputs: list[str]) -> list[dict[str, Any]]:
    """读取已有 validation/rollout JSONL。"""

    records: list[dict[str, Any]] = []
    for path in collect_jsonl_paths(inputs):
        file_step = infer_step_from_path(path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
                record.setdefault("step", file_step)
                record["_source_file"] = str(path)
                record["_line_no"] = line_no
                records.append(record)

    if not records:
        raise ValueError("JSONL 文件为空，无法评测")
    return records


def get_data_source(record: dict[str, Any]) -> str:
    """读取数据集名称；旧 JSONL 没有 data_source 时归为 all。"""

    for key in ("data_source", "dataset", "source"):
        value = record.get(key)
        if value:
            return str(value)
    return "all"
