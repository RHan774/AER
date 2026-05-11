"""评测指标注册与命令行解析。"""

from __future__ import annotations

DEFAULT_METRICS = ["pass@k", "first@1", "distinct-2", "self-bleu", "semantic-cosine"]

ALIASES = {
    "pass": "pass@k",
    "pass@k": "pass@k",
    "pass_at_k": "pass@k",
    "pass-at-k": "pass@k",
    "passatk": "pass@k",
    "first": "first@1",
    "first@1": "first@1",
    "distinct": "distinct-2",
    "distinct-2": "distinct-2",
    "distinct2": "distinct-2",
    "self_bleu": "self-bleu",
    "self-bleu": "self-bleu",
    "self-bleu-4": "self-bleu",
    "semantic": "semantic-cosine",
    "semantic_cosine": "semantic-cosine",
    "semantic-cosine": "semantic-cosine",
}


def parse_metric_names(raw_metrics: str | None) -> list[str]:
    """解析 `--metrics` 参数。

    Args:
        raw_metrics: 逗号分隔指标名；为空或 `all` 表示默认全测。

    Returns:
        去重且顺序稳定的指标名列表。
    """

    if raw_metrics is None or raw_metrics.strip() == "" or raw_metrics.strip().lower() == "all":
        return list(DEFAULT_METRICS)

    metrics: list[str] = []
    for item in raw_metrics.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key == "all":
            return list(DEFAULT_METRICS)
        canonical = ALIASES.get(key)
        if canonical is None:
            available = ", ".join(DEFAULT_METRICS)
            raise ValueError(f"未知评测指标: {item}。可用指标: {available}")
        if canonical not in metrics:
            metrics.append(canonical)
    return metrics


def should_compute_pass_at_k(metrics: list[str]) -> bool:
    """是否需要进行多样本 Pass@K 评测。"""

    return "pass@k" in metrics


def should_compute_semantic(metrics: list[str]) -> bool:
    """是否需要加载 embedding 模型计算 semantic-cosine。"""

    return "semantic-cosine" in metrics
