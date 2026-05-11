"""Pass@K 与正确性相关指标。"""

from __future__ import annotations

from typing import Any


def pass_at_k_unbiased(n_samples: int, n_correct: int, k: int) -> float | None:
    """计算 repeated sampling 场景下的 pass@k 无偏估计。

    Args:
        n_samples: 同一道题实际生成的样本数。
        n_correct: 其中验证正确的样本数。
        k: pass@k 的 k。

    Returns:
        若 `k > n_samples` 则返回 None；否则返回至少抽到一个正确答案的估计。
    """

    if k <= 0 or n_samples <= 0 or k > n_samples:
        return None
    if n_correct <= 0:
        return 0.0
    if n_samples - n_correct < k:
        return 1.0

    product = 1.0
    for value in range(n_samples - n_correct + 1, n_samples + 1):
        product *= 1.0 - k / value
    return 1.0 - product


def to_float(value: Any) -> float | None:
    """尽量把 JSON 字段转成浮点数，失败则返回 None。"""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def is_correct(record: dict[str, Any], threshold: float) -> bool:
    """从 acc、score 或 reward 字段判断一个输出是否正确。"""

    for key in ("acc", "score", "reward"):
        value = to_float(record.get(key))
        if value is not None:
            return value >= threshold
    return False


def compute_pass_metrics(records: list[dict[str, Any]], ks: list[int], threshold: float) -> dict[str, Any]:
    """计算同一道题的一组输出上的正确性指标。"""

    correct_flags = [is_correct(record, threshold=threshold) for record in records]
    n_samples = len(records)
    n_correct = int(sum(correct_flags))
    metrics: dict[str, Any] = {
        "n_samples": n_samples,
        "n_correct": n_correct,
        "correct_rate": n_correct / n_samples if n_samples else None,
        "first@1": float(correct_flags[0]) if correct_flags else None,
    }
    for k in ks:
        metrics[f"pass@{k}"] = pass_at_k_unbiased(n_samples=n_samples, n_correct=n_correct, k=k)
    return metrics
