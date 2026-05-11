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

"""语义嵌入相似度。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from verl import DataProto
else:
    DataProto = Any
    SentenceTransformer = Any


_MODEL_CACHE: dict[tuple[str, str, int], SentenceTransformer] = {}


class SemanticEmbeddingSimilarity(BatchSimilarityComputer):
    """基于句向量嵌入计算余弦相似度。

    Args:
        model_name: sentence-transformers 兼容模型名称。
        batch_size: 编码批大小。
        max_length: 嵌入模型配置的最大序列长度。
        device: 计算设备，支持 ``"cpu"`` 或 ``"cuda"``。

    说明：
        编码前会对重复文本去重，避免同一 response 重复进入嵌入模型。
        相似度通过归一化后的点积计算，只在同 UID 组内做矩阵乘法。
        点积值会由 ``[-1, 1]`` 线性归一化到 ``[0, 1]``。
    """

    RECOMMENDED_MODELS = {
        "fast": "all-MiniLM-L6-v2",
        "balanced": "all-mpnet-base-v2",
        "quality": "stsb-roberta-large",
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "long_text": "Qwen/Qwen3-Embedding-0.6B",
        "long_text_4b": "Qwen/Qwen3-Embedding-4B",
        "long_text_8b": "Qwen/Qwen3-Embedding-8B",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
        max_length: int = 4096,
        device: str = "cpu",
        **kwargs,
    ):
        # 兼容旧配置；当前实现不再对同一个 SentenceTransformer 实例做线程并发。
        kwargs.pop("num_processes", None)
        kwargs.pop("deduplicate", None)
        kwargs.pop("clamp", None)
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or start with 'cuda'")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")

        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            **kwargs,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model = None

    def _load_model(self):
        cache_key = (self.model_name, self.device, self.max_length)
        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            return self.model

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_length
        _MODEL_CACHE[cache_key] = model
        self.model = model
        return model

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        if self.model is None:
            self._load_model()

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        return embeddings

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        unique_texts: list[str] = []
        inverse_indices: list[int] = []
        index_by_text: dict[str, int] = {}
        for text in texts:
            unique_idx = index_by_text.get(text)
            if unique_idx is None:
                unique_idx = len(unique_texts)
                index_by_text[text] = unique_idx
                unique_texts.append(text)
            inverse_indices.append(unique_idx)

        unique_embeddings = self._encode_batch(unique_texts)
        inverse_tensor = torch.as_tensor(inverse_indices, device=unique_embeddings.device, dtype=torch.long)
        return unique_embeddings.index_select(0, inverse_tensor)

    def _build_groupwise_similarity_matrix(self, data: DataProto, embeddings: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        batch_size = embeddings.size(0)
        similarity_matrix = torch.zeros((batch_size, batch_size), device=target_device, dtype=torch.float32)

        for group in self._get_group_indices(data):
            group_index = torch.as_tensor(group, device=target_device, dtype=torch.long)
            group_embeddings = embeddings.index_select(0, group_index)
            group_similarity = group_embeddings @ group_embeddings.t()
            group_similarity = ((group_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
            similarity_matrix[group_index.unsqueeze(1), group_index.unsqueeze(0)] = group_similarity

        return similarity_matrix

    def compute(self, data: DataProto, tokenizer=None) -> torch.Tensor:
        texts = self._decode_responses(data, tokenizer)
        target_device = data.batch["responses"].device
        if not texts:
            return torch.zeros((0, 0), device=target_device, dtype=torch.float32)

        embeddings = self._encode_texts(texts)
        if embeddings.device != target_device:
            embeddings = embeddings.to(target_device)
        return self._build_groupwise_similarity_matrix(data, embeddings, target_device)
