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

"""Token 级计数型 n-gram Jaccard 相似度。"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


class NGramOverlapSimilarity(BatchSimilarityComputer):
    """基于 token n-gram 计数计算 multiset Jaccard 相似度。

    Args:
        n: token n-gram 的窗口大小，必须为正整数。

    说明：
        空 response 或长度小于 ``n`` 的 response 不产生 n-gram，因此相似度为 0。
        重复 n-gram 会按出现次数计入交并比，避免重复片段被集合去重后低估。
    """

    def __init__(self, n: int = 3, **kwargs):
        if n < 1:
            raise ValueError("n must be >= 1")
        super().__init__(n=n, **kwargs)
        self.n = n

    def _extract_ngrams(self, tokens: list[int]) -> Counter[tuple[int, ...]]:
        if len(tokens) < self.n:
            return Counter()
        n = self.n
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def _compute_jaccard(counter1: Counter[tuple[int, ...]], counter2: Counter[tuple[int, ...]]) -> float:
        if not counter1 or not counter2:
            return 0.0
        if counter1 is counter2 or counter1 == counter2:
            return 1.0

        if len(counter1) > len(counter2):
            counter1, counter2 = counter2, counter1
        intersection = sum(min(count, counter2.get(ngram, 0)) for ngram, count in counter1.items())
        if intersection == 0:
            return 0.0
        union = counter1.total() + counter2.total() - intersection
        return intersection / union

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        token_lists = self._get_response_token_lists(data)
        ngram_sets = [self._extract_ngrams(tokens) for tokens in token_lists]
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, ngram_sets, device, self._compute_jaccard)
