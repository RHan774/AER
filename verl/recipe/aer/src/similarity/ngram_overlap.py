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

from typing import TYPE_CHECKING, Any

import numpy as np
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

    _BIT_WIDTH = 18

    def __init__(self, n: int = 3, **kwargs):
        if n < 1:
            raise ValueError("n must be >= 1")
        super().__init__(n=n, **kwargs)
        self.n = n
        self._shifts = np.arange(n - 1, -1, -1, dtype=np.int64) * self._BIT_WIDTH

    def _extract_sorted_ngrams(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
        """返回 (sorted_keys, counts, total_count)。"""
        if len(tokens) < self.n:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0
        arr = np.asarray(tokens, dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, self.n)
        keys = (windows << self._shifts).sum(axis=1)
        sorted_keys = np.sort(keys)
        unique_keys, counts = np.unique(sorted_keys, return_counts=True)
        return unique_keys, counts, int(counts.sum())

    @staticmethod
    def _jaccard_sorted(
        keys1: np.ndarray, counts1: np.ndarray, total1: int,
        keys2: np.ndarray, counts2: np.ndarray, total2: int,
    ) -> float:
        if total1 == 0 or total2 == 0:
            return 0.0
        common = np.intersect1d(keys1, keys2, assume_unique=True)
        if len(common) == 0:
            return 0.0
        idx1 = np.searchsorted(keys1, common)
        idx2 = np.searchsorted(keys2, common)
        intersection = int(np.minimum(counts1[idx1], counts2[idx2]).sum())
        if intersection == 0:
            return 0.0
        union = total1 + total2 - intersection
        return intersection / union

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        token_lists = self._get_response_token_lists(data)

        max_token = max((max(tl) for tl in token_lists if tl), default=0)
        if max_token >= (1 << self._BIT_WIDTH):
            raise ValueError(
                f"Token value {max_token} exceeds {self._BIT_WIDTH}-bit encoding limit. "
                f"vocab_size must be < {1 << self._BIT_WIDTH}."
            )

        ngram_data = [self._extract_sorted_ngrams(tokens) for tokens in token_lists]

        device = data.batch["responses"].device
        batch_size = len(token_lists)
        similarity_matrix = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)

        for group in self._get_group_indices(data):
            g_size = len(group)
            group_sim = np.zeros((g_size, g_size), dtype=np.float64)

            for i in range(g_size):
                keys_i, counts_i, total_i = ngram_data[group[i]]
                group_sim[i, i] = 1.0 if total_i > 0 else 0.0
                for j in range(i + 1, g_size):
                    keys_j, counts_j, total_j = ngram_data[group[j]]
                    sim = self._jaccard_sorted(keys_i, counts_i, total_i, keys_j, counts_j, total_j)
                    group_sim[i, j] = sim
                    group_sim[j, i] = sim

            group_index = torch.as_tensor(group, device=device, dtype=torch.long)
            group_tensor = torch.tensor(group_sim, device=device, dtype=torch.float32)
            similarity_matrix[group_index.unsqueeze(1), group_index.unsqueeze(0)] = group_tensor

        return similarity_matrix
