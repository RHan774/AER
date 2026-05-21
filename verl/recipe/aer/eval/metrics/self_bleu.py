"""Self-BLEU 多样性指标。"""

from __future__ import annotations

import math
from collections import Counter

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


def _ngram_count_tables(tokenized: list[list[str]], max_order: int) -> list[list[Counter[tuple[str, ...]]]]:
    """预先缓存每个回答、每个阶数的 n-gram 计数。"""

    return [[Counter(ngrams(tokens, order)) for tokens in tokenized] for order in range(1, max_order + 1)]


def _reference_top_counts(count_tables: list[Counter[tuple[str, ...]]]) -> dict[tuple[str, ...], tuple[int, int, int]]:
    """记录每个 n-gram 在全部回答中的最大、次大计数和最大计数出现次数。"""

    top_counts: dict[tuple[str, ...], tuple[int, int, int]] = {}
    for counts in count_tables:
        for gram, count in counts.items():
            max_count, second_count, max_count_frequency = top_counts.get(gram, (0, 0, 0))
            if count > max_count:
                top_counts[gram] = (count, max_count, 1)
            elif count == max_count:
                top_counts[gram] = (max_count, second_count, max_count_frequency + 1)
            elif count > second_count:
                top_counts[gram] = (max_count, count, max_count_frequency)
    return top_counts


def _modified_precision_from_cache(
    candidate_counts: Counter[tuple[str, ...]],
    reference_top_counts: dict[tuple[str, ...], tuple[int, int, int]],
    smooth: float,
) -> float:
    """用缓存的全局 top 计数计算排除当前回答后的 clipped precision。"""

    ngram_total = sum(candidate_counts.values())
    if ngram_total == 0:
        return 1.0

    clipped = 0
    for gram, count in candidate_counts.items():
        max_count, second_count, max_count_frequency = reference_top_counts.get(gram, (0, 0, 0))
        if count == max_count and max_count_frequency == 1:
            reference_limit = second_count
        else:
            reference_limit = max_count
        clipped += min(count, reference_limit)
    return (clipped + smooth) / (ngram_total + smooth)


def _closest_reference_length(candidate_idx: int, tokenized: list[list[str]]) -> int | None:
    """返回排除当前回答后的 BLEU brevity penalty 参考长度。"""

    cand_len = len(tokenized[candidate_idx])
    ref_lengths = [len(tokens) for idx, tokens in enumerate(tokenized) if idx != candidate_idx and tokens]
    if not ref_lengths:
        return None
    return min(ref_lengths, key=lambda ref_len: (abs(ref_len - cand_len), ref_len))


def self_bleu(texts: list[str], max_order: int = 4) -> float | None:
    """计算同一题多输出之间的 Self-BLEU。

    值越低，表示同题输出之间越不重复；只有一个输出时无法计算。
    """

    if len(texts) < 2:
        return None

    tokenized = [tokenize(text) for text in texts]
    count_tables_by_order = _ngram_count_tables(tokenized, max_order=max_order)
    reference_top_counts_by_order = [_reference_top_counts(count_tables) for count_tables in count_tables_by_order]

    scores: list[float] = []
    for idx, candidate in enumerate(tokenized):
        if not candidate:
            continue
        closest_ref_len = _closest_reference_length(idx, tokenized)
        if closest_ref_len is None:
            continue

        precisions = [
            _modified_precision_from_cache(
                candidate_counts=count_tables[idx],
                reference_top_counts=reference_top_counts,
                smooth=1.0,
            )
            for count_tables, reference_top_counts in zip(count_tables_by_order, reference_top_counts_by_order)
        ]
        geo_mean = math.exp(sum(math.log(max(precision, 1e-12)) for precision in precisions) / max_order)
        cand_len = len(candidate)
        brevity_penalty = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - closest_ref_len / max(cand_len, 1))
        scores.append(brevity_penalty * geo_mean)
    return float(np.mean(scores)) if scores else None
