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

"""基于 token n-gram 的 SimHash 近重复相似度。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)


class SimHashSimilarity(BatchSimilarityComputer):
    """用 SimHash 指纹近似衡量 token n-gram 相似度。

    Args:
        n: token n-gram 的窗口大小。
        hash_bits: 指纹位数，固定为 64 位。
        use_counts: 是否保留重复 n-gram 的权重。
        calibrate_random: 是否把随机指纹的期望相似度从 0.5 校准到 0。

    说明：
        SimHash 常用于大规模近重复检测。它将每个 response 压缩为固定长度指纹，
        两两比较只需要一次异或和 bit count，适合 rollout 数较多或 response 很长时
        作为比精确 n-gram Jaccard 更快的探索奖励近似。
    """

    def __init__(
        self,
        n: int = 3,
        hash_bits: int = 64,
        use_counts: bool = True,
        calibrate_random: bool = True,
        **kwargs,
    ):
        if n < 1:
            raise ValueError("n must be >= 1")
        if hash_bits < 8 or hash_bits > 64:
            raise ValueError("hash_bits must be in [8, 64]")
        super().__init__(
            n=n,
            hash_bits=hash_bits,
            use_counts=use_counts,
            calibrate_random=calibrate_random,
            **kwargs,
        )
        self.n = n
        self.hash_bits = hash_bits
        self.use_counts = use_counts
        self.calibrate_random = calibrate_random
        self._bit_positions = np.arange(hash_bits, dtype=np.uint64)

    def _fingerprint_from_tokens(self, tokens: list[int]) -> tuple[int, bool]:
        """计算单条 response 的 SimHash fingerprint（全 numpy 向量化）。"""
        if not tokens:
            return 0, False
        if len(tokens) < self.n:
            arr = np.array(tokens, dtype=np.uint64).reshape(1, -1)
            hashes = _FNV_OFFSET ^ arr[:, 0]
            for col in range(arr.shape[1]):
                hashes ^= arr[:, col]
                hashes *= _FNV_PRIME
        else:
            arr = np.asarray(tokens, dtype=np.uint64)
            windows = np.lib.stride_tricks.sliding_window_view(arr, self.n)
            hashes = np.full(len(windows), _FNV_OFFSET, dtype=np.uint64)
            for col in range(self.n):
                hashes ^= windows[:, col]
                hashes *= _FNV_PRIME

        if self.use_counts:
            unique_hashes, weights = np.unique(hashes, return_counts=True)
            weights = weights.astype(np.int64)
        else:
            unique_hashes = np.unique(hashes)
            weights = np.ones(len(unique_hashes), dtype=np.int64)

        bits = ((unique_hashes[:, None] >> self._bit_positions) & np.uint64(1)).astype(np.int64)
        scores = ((bits * 2 - 1) * weights[:, None]).sum(axis=0)

        if self.hash_bits == 64:
            fingerprint = int(
                np.packbits((scores >= 0).astype(np.uint8)[::-1], bitorder="big").view(np.uint64)[0]
            )
        else:
            fingerprint = 0
            for bit in range(self.hash_bits):
                if scores[bit] >= 0:
                    fingerprint |= 1 << bit
        return fingerprint, True

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        token_lists = self._get_response_token_lists(data)

        batch_size = len(token_lists)
        fingerprints = np.zeros(batch_size, dtype=np.uint64)
        has_features_arr = np.zeros(batch_size, dtype=bool)
        for idx, tokens in enumerate(token_lists):
            fp, valid = self._fingerprint_from_tokens(tokens)
            fingerprints[idx] = np.uint64(fp)
            has_features_arr[idx] = valid

        device = data.batch["responses"].device
        similarity_matrix = torch.zeros((batch_size, batch_size), device=device, dtype=torch.float32)

        for group in self._get_group_indices(data):
            gi = np.array(group)
            group_fp = fingerprints[gi]
            group_valid = has_features_arr[gi]

            xor_matrix = group_fp[:, None] ^ group_fp[None, :]
            bits = (xor_matrix[:, :, None] >> self._bit_positions) & np.uint64(1)
            distances = bits.sum(axis=2).astype(np.float64)

            if self.calibrate_random:
                group_sim = np.maximum(0.0, 1.0 - 2.0 * distances / self.hash_bits)
            else:
                group_sim = 1.0 - distances / self.hash_bits

            valid_mask = group_valid[:, None] & group_valid[None, :]
            group_sim *= valid_mask

            group_index = torch.as_tensor(group, device=device, dtype=torch.long)
            group_tensor = torch.tensor(group_sim, device=device, dtype=torch.float32)
            similarity_matrix[group_index.unsqueeze(1), group_index.unsqueeze(0)] = group_tensor

        return similarity_matrix
