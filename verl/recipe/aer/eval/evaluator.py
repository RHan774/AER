"""统一评测聚合逻辑。"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np

from .io_utils import get_data_source, write_csv, write_json, write_markdown_table
from .metrics.distinct import distinct_n
from .metrics.pass_at_k import compute_pass_metrics
from .metrics.registry import should_compute_semantic
from .metrics.self_bleu import self_bleu
from .metrics.semantic_cosine import SemanticEmbeddingCache, average_pairwise_cosine


def mean_ignore_none(values) -> float | None:
    """忽略 None/NaN 后求均值。"""

    clean = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(numeric_value):
            clean.append(numeric_value)
    return float(np.mean(clean)) if clean else None


def prompt_hash(prompt: str) -> str:
    """生成稳定 prompt 短哈希，用于定位 per-prompt 结果。"""

    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def valid_group_id(value: Any) -> str | None:
    """返回可用于分组的题目标识。"""

    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def get_prompt_group_key(record: dict[str, Any], prompt: str) -> str:
    """优先按题目 ID 分组，旧 JSONL 再退化到 prompt 文本。"""

    for key in ("unique_id", "prompt_id", "id"):
        group_id = valid_group_id(record.get(key))
        if group_id is not None:
            return f"id:{group_id}"
    return f"prompt:{prompt}"


def display_prompt_id(group_key: str, prompt: str) -> str:
    """输出 CSV 中可读且不会过长的题目标识。"""

    if group_key.startswith("id:"):
        group_id = group_key[3:]
        return group_id if len(group_id) <= 160 else prompt_hash(group_id)
    return prompt_hash(prompt)


def build_prompt_groups(records: list[dict[str, Any]]) -> dict[tuple[Any, str, str], list[dict[str, Any]]]:
    """按 step、数据集、prompt 分组。"""

    groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        prompt = str(record.get("input", ""))
        key = (record.get("step"), get_data_source(record), get_prompt_group_key(record, prompt))
        groups[key].append(record)
    return groups


def evaluate_records(
    records: list[dict[str, Any]],
    metrics: list[str],
    ks: list[int],
    correct_threshold: float,
    semantic_model: str = "",
    semantic_device: str = "cpu",
    semantic_batch_size: int = 32,
    semantic_max_length: int = 4096,
    prompt_preview_chars: int = 120,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """对已带 `input/output/score` 的记录计算所有指定指标。

    Args:
        records: 一行一个输出样本，至少包含 input 和 output；Pass@K 需要 score/acc/reward。
        metrics: 需要计算的指标名，来自 metrics.registry。
        ks: Pass@K 的 k 列表。
        correct_threshold: score/acc/reward 大于等于该值视为正确。
        semantic_model: semantic-cosine 使用的 embedding 模型；未设置时跳过。
    """

    if should_compute_semantic(metrics) and not semantic_model:
        raise ValueError("请求 semantic-cosine 时必须提供 --semantic-model")

    semantic_cache: SemanticEmbeddingCache | None = None
    if should_compute_semantic(metrics):
        semantic_cache = SemanticEmbeddingCache(
            model_name=semantic_model,
            device=semantic_device,
            batch_size=semantic_batch_size,
            max_length=semantic_max_length,
        )
        semantic_cache.encode_missing([str(record.get("output", "")) for record in records])

    groups = build_prompt_groups(records)
    per_prompt_rows: list[dict[str, Any]] = []
    bucket_rows: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)

    for (step, data_source, group_key), group_records in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])):
        prompt = str(group_records[0].get("input", ""))
        outputs = [str(record.get("output", "")) for record in group_records]
        row: dict[str, Any] = {
            "step": step,
            "data_source": data_source,
            "prompt_id": display_prompt_id(group_key, prompt),
            "prompt_hash": prompt_hash(prompt),
            "prompt": prompt.replace("\n", "\\n")[:prompt_preview_chars],
        }

        if "pass@k" in metrics or "first@1" in metrics:
            pass_metrics = compute_pass_metrics(group_records, ks=ks, threshold=correct_threshold)
            row.update(pass_metrics)
        else:
            row["n_samples"] = len(group_records)

        if "distinct-2" in metrics:
            row["distinct_2"] = distinct_n(outputs, n=2)
        if "self-bleu" in metrics:
            row["self_bleu4"] = self_bleu(outputs, max_order=4)
        if semantic_cache is not None:
            row["semantic_cosine"] = average_pairwise_cosine(semantic_cache.get_many(outputs))

        per_prompt_rows.append(row)
        bucket_rows[(step, data_source)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (step, data_source), rows in sorted(bucket_rows.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        summary_rows.append(summarize_prompt_rows(step=step, data_source=data_source, rows=rows, metrics=metrics, ks=ks))
    summary_rows.extend(build_avg_dataset_rows(summary_rows, metrics=metrics, ks=ks))

    summary_fieldnames = build_summary_fieldnames(metrics=metrics, ks=ks)
    prompt_fieldnames = build_prompt_fieldnames(metrics=metrics, ks=ks)
    return summary_rows, per_prompt_rows, summary_fieldnames, prompt_fieldnames


def summarize_prompt_rows(step: Any, data_source: str, rows: list[dict[str, Any]], metrics: list[str], ks: list[int]) -> dict[str, Any]:
    """把 per-prompt 指标聚合为某个 step/数据集的 summary。"""

    summary: dict[str, Any] = {
        "step": step,
        "data_source": data_source,
        "n_prompts": len(rows),
        "samples_per_prompt_mean": mean_ignore_none(row.get("n_samples") for row in rows),
    }
    if "pass@k" in metrics or "first@1" in metrics:
        summary["correct_rate"] = mean_ignore_none(row.get("correct_rate") for row in rows)
        summary["first@1"] = mean_ignore_none(row.get("first@1") for row in rows)
        for k in ks:
            summary[f"pass@{k}"] = mean_ignore_none(row.get(f"pass@{k}") for row in rows)
    if "distinct-2" in metrics:
        summary["distinct_2"] = mean_ignore_none(row.get("distinct_2") for row in rows)
    if "self-bleu" in metrics:
        summary["self_bleu4"] = mean_ignore_none(row.get("self_bleu4") for row in rows)
    if "semantic-cosine" in metrics:
        summary["semantic_cosine"] = mean_ignore_none(row.get("semantic_cosine") for row in rows)
    return summary


def build_avg_dataset_rows(summary_rows: list[dict[str, Any]], metrics: list[str], ks: list[int]) -> list[dict[str, Any]]:
    """额外生成每个 step 的数据集简单均值。"""

    step_to_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("data_source") != "all":
            step_to_rows[row.get("step")].append(row)

    avg_rows: list[dict[str, Any]] = []
    for step, rows in sorted(step_to_rows.items(), key=lambda item: str(item[0])):
        if len(rows) < 2:
            continue
        avg_row: dict[str, Any] = {
            "step": step,
            "data_source": "AVG_DATASETS",
            "n_prompts": sum(int(row.get("n_prompts", 0)) for row in rows),
            "samples_per_prompt_mean": mean_ignore_none(row.get("samples_per_prompt_mean") for row in rows),
        }
        for key in metric_value_keys(metrics=metrics, ks=ks):
            avg_row[key] = mean_ignore_none(row.get(key) for row in rows)
        avg_rows.append(avg_row)
    return avg_rows


def metric_value_keys(metrics: list[str], ks: list[int]) -> list[str]:
    """返回 summary 中需要聚合的指标列。"""

    keys: list[str] = []
    if "pass@k" in metrics or "first@1" in metrics:
        keys.extend(["correct_rate", "first@1"])
        keys.extend([f"pass@{k}" for k in ks])
    if "distinct-2" in metrics:
        keys.append("distinct_2")
    if "self-bleu" in metrics:
        keys.append("self_bleu4")
    if "semantic-cosine" in metrics:
        keys.append("semantic_cosine")
    return keys


def build_summary_fieldnames(metrics: list[str], ks: list[int]) -> list[str]:
    """summary CSV 的列顺序。"""

    return ["step", "data_source", "n_prompts", "samples_per_prompt_mean", *metric_value_keys(metrics=metrics, ks=ks)]


def build_prompt_fieldnames(metrics: list[str], ks: list[int]) -> list[str]:
    """per-prompt CSV 的列顺序。"""

    fields = ["step", "data_source", "prompt_id", "prompt_hash", "n_samples"]
    if "pass@k" in metrics or "first@1" in metrics:
        fields.extend(["n_correct", "correct_rate", "first@1"])
        fields.extend([f"pass@{k}" for k in ks])
    if "distinct-2" in metrics:
        fields.append("distinct_2")
    if "self-bleu" in metrics:
        fields.append("self_bleu4")
    if "semantic-cosine" in metrics:
        fields.append("semantic_cosine")
    fields.append("prompt")
    return fields


def write_evaluation_outputs(output_dir, summary_rows, per_prompt_rows, summary_fieldnames, prompt_fieldnames) -> None:
    """写出评测结果的 CSV/JSON/Markdown。"""

    write_csv(output_dir / "validation_summary.csv", summary_rows, fieldnames=summary_fieldnames)
    write_json(output_dir / "validation_summary.json", summary_rows)
    write_markdown_table(output_dir / "validation_summary.md", summary_rows, fieldnames=summary_fieldnames)
    write_csv(output_dir / "validation_per_prompt.csv", per_prompt_rows, fieldnames=prompt_fieldnames)
