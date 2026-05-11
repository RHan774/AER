"""Self-BLEU 多样性指标。"""

from __future__ import annotations

import math

import numpy as np

from .text import ngrams, tokenize


def modified_precision(candidate: list[str], references: list[list[str]], order: int, smooth: float) -> float:
    """计算 BLEU 的 clipped n-gram precision。

    使用加一平滑，避免长阶 n-gram 没有重合时 BLEU 直接变成 0。
    """

    cand_ngrams = ngrams(candidate, order)
    if not cand_ngrams:
        return 1.0

    cand_counts: dict[tuple[str, ...], int] = {}
    for gram in cand_ngrams:
        cand_counts[gram] = cand_counts.get(gram, 0) + 1

    max_ref_counts: dict[tuple[str, ...], int] = {}
    for ref in references:
        ref_counts: dict[tuple[str, ...], int] = {}
        for gram in ngrams(ref, order):
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        for gram, count in ref_counts.items():
            max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)

    clipped = sum(min(count, max_ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    return (clipped + smooth) / (len(cand_ngrams) + smooth)


def bleu_score(candidate: list[str], references: list[list[str]], max_order: int = 4, smooth: float = 1.0) -> float | None:
    """计算单个候选相对多个参考的 BLEU 分数。"""

    if not candidate or not references:
        return None
    ref_lengths = [len(ref) for ref in references if ref]
    if not ref_lengths:
        return None

    precisions = [modified_precision(candidate, references, order, smooth) for order in range(1, max_order + 1)]
    geo_mean = math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / max_order)
    cand_len = len(candidate)
    closest_ref_len = min(ref_lengths, key=lambda ref_len: (abs(ref_len - cand_len), ref_len))
    brevity_penalty = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - closest_ref_len / max(cand_len, 1))
    return brevity_penalty * geo_mean


def self_bleu(texts: list[str], max_order: int = 4) -> float | None:
    """计算同一题多输出之间的 Self-BLEU。

    值越低，表示同题输出之间越不重复；只有一个输出时无法计算。
    """

    if len(texts) < 2:
        return None

    tokenized = [tokenize(text) for text in texts]
    scores: list[float] = []
    for idx, candidate in enumerate(tokenized):
        references = [tokens for ref_idx, tokens in enumerate(tokenized) if ref_idx != idx]
        score = bleu_score(candidate, references, max_order=max_order)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else None
