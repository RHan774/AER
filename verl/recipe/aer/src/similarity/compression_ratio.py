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

"""基于压缩的相似度。"""

from __future__ import annotations

import gzip
import zlib
from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from verl import DataProto
else:
    DataProto = Any


class CompressionRatioSimilarity(BatchSimilarityComputer):
    """计算受 NCD 启发的对称压缩相似度。

    给定压缩后字节长度 ``C(x)`` 和 ``C(y)``，相似度定义为

    ``sim(x, y) = 1 - (C(xy) - min(C(x), C(y))) / max(C(x), C(y))``

    其中 ``C(xy)`` 通过 ``x || sep || y`` 与 ``y || sep || x`` 两种拼接方式
    的压缩长度平均值得到。结果可选裁剪到 ``[0, 1]``。

    Args:
        compression_type: 压缩算法类型，支持 ``"gzip"`` 或 ``"zlib"``。
        normalize: 是否将结果裁剪到 ``[0, 1]``。

    说明：
        单条文本的压缩长度会被缓存，因此真正需要重复压缩的只有拼接后的样本对。
    """

    SEPARATOR = b" || "

    def __init__(self, compression_type: str = "gzip", normalize: bool = True, **kwargs):
        compression_type = compression_type.lower()
        if compression_type == "gzip":
            compressor = gzip.compress
        elif compression_type == "zlib":
            compressor = zlib.compress
        else:
            raise ValueError("compression_type must be 'gzip' or 'zlib'")

        super().__init__(compression_type=compression_type, normalize=normalize, **kwargs)
        self.compression_type = compression_type
        self.normalize = normalize
        self._compress = compressor

    def _compressed_size(self, payload: bytes) -> int:
        return len(self._compress(payload))

    def _compute_similarity(self, item1: tuple[bytes, int], item2: tuple[bytes, int]) -> float:
        bytes1, size1 = item1
        bytes2, size2 = item2

        if not bytes1 and not bytes2:
            return 1.0
        if not bytes1 or not bytes2:
            return 0.0
        if bytes1 == bytes2:
            return 1.0

        c_xy = self._compressed_size(bytes1 + self.SEPARATOR + bytes2)
        c_yx = self._compressed_size(bytes2 + self.SEPARATOR + bytes1)
        combined_size = (c_xy + c_yx) / 2.0
        denominator = max(size1, size2)
        similarity = 1.0 - ((combined_size - min(size1, size2)) / denominator)

        if self.normalize:
            similarity = max(0.0, min(1.0, similarity))
        return similarity

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        encoded = [text.encode("utf-8") for text in texts]
        cached_sizes = [self._compressed_size(payload) for payload in encoded]
        items = list(zip(encoded, cached_sizes, strict=True))
        device = data.batch["responses"].device
        return self._build_groupwise_matrix(data, items, device, self._compute_similarity)
