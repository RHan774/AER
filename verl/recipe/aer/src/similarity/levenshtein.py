# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""基于 RapidFuzz 的归一化 Levenshtein 相似度。"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any

from rapidfuzz.distance import Levenshtein as RapidFuzzLevenshtein


class LevenshteinSimilarity(BatchSimilarityComputer):
    """基于归一化 Levenshtein 距离计算相似度。

    当前实现直接对齐数学定义，而不是使用 ``fuzz.ratio`` 之类并不等价的近似指标。

    Args:
        normalize_method: 归一化方式，可选 ``"max"``、``"avg"`` 或 ``"min"``。
            相似度计算形式为 ``1 - distance / denom``。

    说明：
        仅使用 ``rapidfuzz`` 的加速实现，不再保留纯 Python 回退路径。
        比较前固定做 NFC Unicode 规范化，避免视觉等价的组合字符被当作不同字符。
    """

    def __init__(self, normalize_method: str = "max", **kwargs):
        normalize_method = normalize_method.lower()
        if normalize_method not in {"max", "avg", "min"}:
            raise ValueError("normalize_method must be one of: max, avg, min")
        super().__init__(normalize_method=normalize_method, **kwargs)
        self.normalize_method = normalize_method

    def _denominator(self, len1: int, len2: int) -> float:
        if self.normalize_method == "max":
            return float(max(len1, len2))
        if self.normalize_method == "avg":
            return (len1 + len2) / 2.0
        return float(min(len1, len2))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)

        if text1 == text2:
            return 1.0

        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0

        if self.normalize_method == "max":
            return float(RapidFuzzLevenshtein.normalized_similarity(text1, text2))

        denom = self._denominator(len1, len2)
        if denom <= 0:
            return 0.0

        distance = RapidFuzzLevenshtein.distance(text1, text2)
        return max(0.0, 1.0 - (distance / denom))

    def _cached_similarity_fn(self) -> Callable[[str, str], float]:
        cache: dict[tuple[str, str], float] = {}

        def cached_similarity(text1: str, text2: str) -> float:
            key = (text1, text2) if text1 <= text2 else (text2, text1)
            if key not in cache:
                cache[key] = self._compute_similarity(key[0], key[1])
            return cache[key]

        return cached_similarity

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, texts, device, self._cached_similarity_fn())
