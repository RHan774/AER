"""Distinct-n 多样性指标。"""

from __future__ import annotations

from .text import ngrams, tokenize


def distinct_n(texts: list[str], n: int = 2) -> float | None:
    """计算一组输出的 Distinct-n。

    Distinct-2 = unique bigrams / total bigrams。值越高表示词面多样性越高。
    """

    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(tokenize(text), n))
    if not all_ngrams:
        return None
    return len(set(all_ngrams)) / len(all_ngrams)
