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

"""ROUGE-L 相似度。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


class RougeLSimilarity(BatchSimilarityComputer):
    """计算 ROUGE-L F-beta 相似度。

    Args:
        use_char_level: 是否直接使用字符级序列，而不是空格分词后的 token 序列。
        beta: F-beta 中的权重参数，必须大于 0。
        tokenize_by_space: 当 ``use_char_level`` 为 false 时，若为 true 则按空格分词；
            否则移除空白字符后按字符级比较。

    说明：
        LCS 使用滚动动态规划数组实现，因此空间复杂度为 ``O(min(n, m))``。
    """

    def __init__(
        self,
        use_char_level: bool = False,
        beta: float = 1.0,
        tokenize_by_space: bool = True,
        **kwargs,
    ):
        if beta <= 0:
            raise ValueError("beta must be > 0")
        super().__init__(
            use_char_level=use_char_level,
            beta=beta,
            tokenize_by_space=tokenize_by_space,
            **kwargs,
        )
        self.use_char_level = use_char_level
        self.beta = beta
        self.beta_squared = beta * beta
        self.tokenize_by_space = tokenize_by_space

    def _tokenize(self, text: str) -> list[str]:
        if self.use_char_level:
            return list(text)
        if self.tokenize_by_space:
            return text.split()
        stripped = [char for char in text if not char.isspace()]
        return stripped or list(text)

    @staticmethod
    def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
        if len(seq1) < len(seq2):
            seq1, seq2 = seq2, seq1

        dp = [0] * (len(seq2) + 1)
        for i in range(1, len(seq1) + 1):
            prev = 0
            for j in range(1, len(seq2) + 1):
                current = dp[j]
                if seq1[i - 1] == seq2[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = current
        return dp[-1]

    def _compute_f_score(self, precision: float, recall: float) -> float:
        if precision == 0.0 or recall == 0.0:
            return 0.0
        denominator = recall + self.beta_squared * precision
        return ((1 + self.beta_squared) * precision * recall) / denominator if denominator > 0 else 0.0

    def _compute_similarity(self, seq1: list[str], seq2: list[str]) -> float:
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        if seq1 == seq2:
            return 1.0

        lcs_len = self._lcs_length(seq1, seq2)
        precision = lcs_len / len(seq2)
        recall = lcs_len / len(seq1)
        return self._compute_f_score(precision, recall)

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        tokenized = [self._tokenize(text) for text in texts]
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, tokenized, device, self._compute_similarity)
