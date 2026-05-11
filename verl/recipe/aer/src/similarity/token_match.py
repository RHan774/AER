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

"""
Token 精确匹配相似度计算器

这是原有的相似度计算方法，作为基准算法保留
"""

from typing import Any

import torch

try:
    from verl import DataProto
except ImportError:
    DataProto = Any

from .base import SimilarityComputer


class TokenMatchSimilarity(SimilarityComputer):
    """
    Token 精确匹配相似度计算器

    通过计算两个 token 序列的精确匹配比例来衡量相似度

    算法原理：
        1. 对于每一对响应，逐位置比较 token 是否相同
        2. 使用长度归一化：sqrt(L_i * L_j)
        3. 只计算同一 UID 组内的相似度

    公式：
        similarity(i, j) = sum(token_i^k == token_j^k) / sqrt(L_i * L_j)
        其中 L_i 和 L_j 分别是响应 i 和 j 的有效长度

    优点：
        - 计算速度最快
        - 易于理解和调试
        - 无需额外依赖

    缺点：
        - 只考虑精确匹配，无法捕捉语义相似性
        - 对噪声敏感，单个 token 差异会导致相似度下降

    适用场景：
        - 需要快速基准对比
        - Token 序列质量较高，噪声较少
    """

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        """
        计算 token 精确匹配相似度矩阵

        Args:
            data: DataProto 对象，包含：
                - batch["responses"]: token 序列矩阵
                - batch["attention_mask"]: 注意力掩码
                - non_tensor_batch["uid"]: 分组 ID
            tokenizer: 分词器（此算法不需要，忽略）

        Returns:
            torch.Tensor: 相似度矩阵，shape 为 [batch_size, batch_size]
        """
        token_matrix = data.batch["responses"]
        response_length = token_matrix.size(1)
        response_mask = data.batch["attention_mask"][:, -response_length:].to(dtype=torch.bool)
        valid_lengths = response_mask.sum(dim=1).to(dtype=torch.float32)
        batch_size = token_matrix.size(0)
        device = token_matrix.device

        similarity_matrix = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)

        for group in self._get_group_indices(data):
            group_index = torch.as_tensor(group, device=device, dtype=torch.long)
            group_tokens = token_matrix.index_select(0, group_index)
            group_mask = response_mask.index_select(0, group_index)
            group_lengths = valid_lengths.index_select(0, group_index)

            # 只在同 UID 组内广播比较，避免构造整批 B*B*L 的巨大临时张量。
            pair_matches = group_tokens.unsqueeze(1).eq(group_tokens.unsqueeze(0))
            pair_valid_mask = group_mask.unsqueeze(1) & group_mask.unsqueeze(0)
            overlap = (pair_matches & pair_valid_mask).sum(dim=2).to(dtype=torch.float32)

            norm_factor = torch.sqrt(group_lengths.unsqueeze(1) * group_lengths.unsqueeze(0))
            group_similarity = torch.where(
                norm_factor > 0,
                overlap / norm_factor,
                torch.zeros_like(overlap),
            )
            similarity_matrix[group_index.unsqueeze(1), group_index.unsqueeze(0)] = group_similarity

        return similarity_matrix
