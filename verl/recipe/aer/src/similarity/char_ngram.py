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

"""字符级 n-gram 相似度。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


class CharNGramSimilarity(BatchSimilarityComputer):
    """基于字符级 n-gram 集合计算相似度。

    Args:
        n: 字符 n-gram 窗口大小，必须为正数。
        metric: 相似度度量方式，支持 ``"jaccard"`` 或 ``"dice"``。

    示例：
        >>> computer = CharNGramSimilarity(n=4, metric="jaccard")
        >>> matrix = computer.compute(data, tokenizer)
    """

    def __init__(self, n: int = 4, metric: str = "jaccard", **kwargs):
        if n < 1:
            raise ValueError("n must be >= 1")
        metric = metric.lower()
        if metric not in {"jaccard", "dice"}:
            raise ValueError(f"unsupported metric: {metric}")
        super().__init__(n=n, metric=metric, **kwargs)
        self.n = n
        self.metric = metric

    def _extract_char_ngrams(self, text: str) -> set[str]:
        if not text:
            return set()
        if len(text) < self.n:
            return {text}
        return {text[i : i + self.n] for i in range(len(text) - self.n + 1)}

    def _compute_similarity(self, set1: set[str], set2: set[str]) -> float:
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        if self.metric == "jaccard":
            union = len(set1) + len(set2) - intersection
            return intersection / union if union > 0 else 0.0
        return 2 * intersection / (len(set1) + len(set2))

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        char_ngrams = [self._extract_char_ngrams(text) for text in texts]
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, char_ngrams, device, self._compute_similarity)
