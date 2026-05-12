"""Equational Diversity 公式多样性指标。"""

from __future__ import annotations

from collections import Counter
import re

import numpy as np


BRACKET_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
PAREN_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
DOLLAR_DISPLAY_RE = re.compile(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", re.DOTALL)
DOLLAR_INLINE_RE = re.compile(r"(?<![\\$])\$(?!\$)(.*?)(?<![\\$])\$(?!\$)", re.DOTALL)


def normalize_formula(formula: str) -> str:
    """规范化公式字符串，避免纯空白差异影响集合去重。"""

    return " ".join((formula or "").strip().split())


def extract_formulas(text: str) -> set[str]:
    """按论文附录 C.3 的 LaTeX 分隔符规则抽取公式集合。

    覆盖 ``\\[...\\]``、``\\(...\\)``、``$...$`` 三类模式，并额外兼容
    常见的 ``$$...$$`` display math。返回集合以对同一回答内部公式去重。
    """

    formulas: set[str] = set()
    for pattern in (BRACKET_DISPLAY_RE, PAREN_INLINE_RE, DOLLAR_DISPLAY_RE, DOLLAR_INLINE_RE):
        for match in pattern.findall(text or ""):
            formula = normalize_formula(match)
            if formula:
                formulas.add(formula)
    return formulas


def per_response_equational_diversity(texts: list[str]) -> list[float]:
    """计算同题每个回答的 ED 分数。

    对回答 ``o_i``，令 ``F(o_i)`` 为其公式集合，``F_-i`` 为其它回答公式集合并集。
    若 ``F(o_i)`` 非空，``ED(o_i)=|F(o_i) \\ F_-i| / |F(o_i)|``；否则为 0。
    """

    formula_sets = [extract_formulas(text) for text in texts]
    formula_counts: Counter[str] = Counter()
    for formulas in formula_sets:
        formula_counts.update(formulas)

    scores: list[float] = []
    for formulas in formula_sets:
        if not formulas:
            scores.append(0.0)
            continue
        unique_count = sum(1 for formula in formulas if formula_counts[formula] == 1)
        scores.append(unique_count / len(formulas))
    return scores


def equational_diversity(texts: list[str]) -> float | None:
    """计算同一题多输出之间的平均 Equational Diversity。

    值越高，表示每个回答中越多公式没有出现在同题其它回答里。
    """

    if not texts:
        return None
    scores = per_response_equational_diversity(texts)
    return float(np.mean(scores)) if scores else None
