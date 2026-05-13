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

"""AER 奖励计算中使用的相似度基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


_SIMILARITY_CACHE_KEY = "_aer_similarity_cache"


class SimilarityComputer(ABC):
    """两两相似度计算接口。

    所有实现都应返回形状为 ``[batch_size, batch_size]`` 的矩阵，其中
    ``matrix[i, j]`` 表示第 ``i`` 个 response 与第 ``j`` 个 response 的相似度。
    相似度通常应位于 ``[0, 1]``，且不同 UID 组之间的相似度应为 0。
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _get_batch_cache(self, data: DataProto) -> dict[str, Any]:
        """返回当前 batch 内可复用的相似度预处理缓存。"""

        meta_info = getattr(data, "meta_info", None)
        if isinstance(meta_info, dict):
            cache = meta_info.get(_SIMILARITY_CACHE_KEY)
            if cache is None:
                cache = {}
                meta_info[_SIMILARITY_CACHE_KEY] = cache
            return cache

        cache = getattr(data, _SIMILARITY_CACHE_KEY, None)
        if cache is None:
            cache = {}
            try:
                setattr(data, _SIMILARITY_CACHE_KEY, cache)
            except Exception:
                return {}
        return cache

    @abstractmethod
    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        """计算一个 batch 的两两相似度矩阵。"""
        raise NotImplementedError

    def _get_group_mask(self, data: DataProto, device: torch.device) -> torch.Tensor:
        """返回同一 UID 组内为 1 的稠密掩码矩阵。"""
        cache = self._get_batch_cache(data)
        cache_key = ("group_mask", str(device))
        if cache_key in cache:
            return cache[cache_key]

        ids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
        _, inverse_indices = np.unique(ids, return_inverse=True)
        index_tensor = torch.as_tensor(inverse_indices, device=device, dtype=torch.long)
        group_mask = (index_tensor.unsqueeze(0) == index_tensor.unsqueeze(1)).float()
        cache[cache_key] = group_mask
        return group_mask

    def _get_group_indices(self, data: DataProto) -> list[list[int]]:
        """按 UID 返回样本索引分组。

        返回顺序与输入 batch 顺序保持稳定，从而保证结果矩阵可复现。
        """

        cache = self._get_batch_cache(data)
        if "group_indices" in cache:
            return cache["group_indices"]

        groups: dict[Any, list[int]] = {}
        for idx, uid in enumerate(np.asarray(data.non_tensor_batch["uid"], dtype=object).tolist()):
            groups.setdefault(uid, []).append(idx)
        group_indices = list(groups.values())
        cache["group_indices"] = group_indices
        return group_indices

    def _apply_group_mask(self, similarity_matrix: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
        """将不同 UID 组之间的相似度置零。"""
        return similarity_matrix * group_mask

    def _build_groupwise_matrix(
        self,
        data: DataProto,
        items: Sequence[Any],
        device: torch.device,
        similarity_fn: Callable[[Any, Any], float],
    ) -> torch.Tensor:
        """仅在 UID 组内计算相似度。

        这样可以避免先做整批 ``O(B^2)`` 计算、再被 group mask 直接清零的无效开销。
        """

        batch_size = len(items)
        similarity_matrix = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)

        for group in self._get_group_indices(data):
            group_size = len(group)
            if group_size == 0:
                continue

            group_values = [[0.0] * group_size for _ in range(group_size)]
            for offset, i in enumerate(group):
                item_i = items[i]
                group_values[offset][offset] = float(similarity_fn(item_i, item_i))
                for local_j, j in enumerate(group[offset + 1 :], start=offset + 1):
                    sim = float(similarity_fn(item_i, items[j]))
                    group_values[offset][local_j] = sim
                    group_values[local_j][offset] = sim

            group_index = torch.as_tensor(group, device=device, dtype=torch.long)
            group_tensor = torch.tensor(group_values, device=device, dtype=torch.float32)
            similarity_matrix[group_index.unsqueeze(1), group_index.unsqueeze(0)] = group_tensor

        return similarity_matrix

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.kwargs})"


class BatchSimilarityComputer(SimilarityComputer):
    """面向文本类相似度算法的辅助基类。"""

    def _get_response_mask(self, data: DataProto) -> torch.Tensor:
        """返回与 ``batch['responses']`` 对齐的布尔有效掩码。"""
        cache = self._get_batch_cache(data)
        if "response_mask" in cache:
            return cache["response_mask"]

        responses = data.batch["responses"]
        response_length = responses.size(1)
        response_mask = data.batch["attention_mask"][:, -response_length:].to(dtype=torch.bool)
        cache["response_mask"] = response_mask
        return response_mask

    def _get_valid_response_lengths(self, data: DataProto) -> list[int]:
        """返回每个样本的有效 response 长度。"""
        cache = self._get_batch_cache(data)
        if "valid_response_lengths" in cache:
            return cache["valid_response_lengths"]

        response_mask = self._get_response_mask(data)
        valid_lengths = response_mask.sum(dim=1).to(dtype=torch.long).tolist()
        cache["valid_response_lengths"] = valid_lengths
        return valid_lengths

    def _get_response_token_lists(self, data: DataProto) -> list[list[int]]:
        """提取每个样本的有效 response token 序列。

        示例：
            >>> token_lists = computer._get_response_token_lists(data)
            >>> token_lists[0][:3]
            [151644, 198, 1234]
        """

        responses = data.batch["responses"]
        cache = self._get_batch_cache(data)
        if "response_token_lists" in cache:
            return cache["response_token_lists"]

        valid_lengths = self._get_valid_response_lengths(data)
        token_lists = [responses[idx, :length].tolist() for idx, length in enumerate(valid_lengths)]
        cache["response_token_lists"] = token_lists
        return token_lists

    def _decode_responses(self, data: DataProto, tokenizer) -> list[str]:
        """将有效 response token 解码为字符串。

        Args:
            data: 提供 ``batch`` 与 ``non_tensor_batch`` 的 batch 对象。
            tokenizer: 实现了 ``decode(ids, skip_special_tokens=True)`` 的分词器。

        Returns:
            按 batch 顺序返回解码后的 response 文本列表。

        说明：
            空 response 会被解码为空字符串。
            每个样本只解码一次，结果应由具体算法的预处理逻辑复用。
        """

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__} 需要 tokenizer 才能解码文本")

        cache = self._get_batch_cache(data)
        cache_key = ("decoded_responses", id(tokenizer))
        if cache_key in cache:
            return cache[cache_key]

        token_lists = self._get_response_token_lists(data)
        texts = [tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else "" for token_ids in token_lists]
        cache[cache_key] = texts
        return texts
