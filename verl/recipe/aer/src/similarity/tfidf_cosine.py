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

"""基于 scikit-learn 的 TF-IDF 余弦相似度。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any

class TFIDFCosineSimilarity(BatchSimilarityComputer):
    """基于 TF-IDF 向量计算余弦相似度。

    Args:
        max_features: 最大词表大小。设为 ``None`` 时保留完整词表。
        min_df: 词项保留所需的最小文档频率。
        max_df: 词项保留所需的最大文档频率比例。
        ngram_range: n-gram 范围，例如 ``(1, 2)``。

    说明：
        仅使用 ``scikit-learn`` 的标准 TF-IDF 向量化实现。
    """

    def __init__(
        self,
        max_features: int | None = 1000,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range=(1, 2),
        **kwargs,
    ):
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
            raise ValueError("ngram_range must be a tuple/list of length 2")
        if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n")
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        if not 0 < max_df <= 1:
            raise ValueError("max_df must be in (0, 1]")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be >= 1 when provided")

        super().__init__(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            **kwargs,
        )
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

    def _compute_with_sklearn(self, texts: list[str], data: DataProto, device: torch.device) -> torch.Tensor:
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)
        similarity_matrix = torch.as_tensor(cosine_sim, device=device, dtype=torch.float32)
        return self._apply_group_mask(similarity_matrix, self._get_group_mask(data, device))

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        device = data.batch["responses"].device
        return self._compute_with_sklearn(texts, data, device)
