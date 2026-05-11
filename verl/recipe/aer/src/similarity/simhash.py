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

import zlib
from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


class SimHashSimilarity(BatchSimilarityComputer):
    """用 SimHash 指纹近似衡量 token n-gram 相似度。

    Args:
        n: token n-gram 的窗口大小。
        hash_bits: 指纹位数，支持 8 到 64 位。
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
        self._hash_mask = (1 << hash_bits) - 1

    def _extract_features(self, tokens: list[int]) -> dict[tuple[int, ...], int]:
        if not tokens:
            return {}
        if len(tokens) < self.n:
            return {tuple(tokens): 1}

        features: dict[tuple[int, ...], int] = {}
        n = self.n
        for start in range(len(tokens) - n + 1):
            feature = tuple(tokens[start : start + n])
            if self.use_counts:
                features[feature] = features.get(feature, 0) + 1
            else:
                features[feature] = 1
        return features

    @staticmethod
    def _hash_feature(feature: tuple[int, ...]) -> int:
        payload = ",".join(str(token) for token in feature).encode("utf-8")
        low = zlib.crc32(payload)
        high = zlib.crc32(payload, 0x9E3779B9)
        return (high << 32) | low

    def _fingerprint(self, features: dict[tuple[int, ...], int]) -> tuple[int, bool]:
        if not features:
            return 0, False

        scores = [0] * self.hash_bits
        for feature, weight in features.items():
            hashed = self._hash_feature(feature) & self._hash_mask
            for bit in range(self.hash_bits):
                if hashed & (1 << bit):
                    scores[bit] += weight
                else:
                    scores[bit] -= weight

        fingerprint = 0
        for bit, score in enumerate(scores):
            if score >= 0:
                fingerprint |= 1 << bit
        return fingerprint, True

    def _compute_similarity(self, item1: tuple[int, bool], item2: tuple[int, bool]) -> float:
        fingerprint1, has_features1 = item1
        fingerprint2, has_features2 = item2
        if not has_features1 or not has_features2:
            return 0.0
        if fingerprint1 == fingerprint2:
            return 1.0

        distance = (fingerprint1 ^ fingerprint2).bit_count()
        if self.calibrate_random:
            return max(0.0, 1.0 - (2.0 * distance / self.hash_bits))
        return 1.0 - (distance / self.hash_bits)

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        token_lists = self._get_response_token_lists(data)
        fingerprints = [self._fingerprint(self._extract_features(tokens)) for tokens in token_lists]
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, fingerprints, device, self._compute_similarity)
