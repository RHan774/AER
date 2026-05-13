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

import atexit
import math
import multiprocessing as mp
import os
import queue
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .base import BatchSimilarityComputer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from verl import DataProto
else:
    DataProto = Any
    SentenceTransformer = Any


_MODEL_CACHE: dict[tuple[str, str, int, str | None], SentenceTransformer] = {}
_POOL_CACHE: dict[tuple[str, str, int, int, str | None], _EmbeddingWorkerPool] = {}


def _normalize_cuda_visible_devices(cuda_visible_devices: str | list[int | str] | tuple[int | str, ...] | None) -> list[str]:
    """规范化 semantic embedding 专用的物理 GPU 列表。"""

    if cuda_visible_devices is None:
        return []
    if isinstance(cuda_visible_devices, str):
        return [device.strip() for device in cuda_visible_devices.split(",") if device.strip()]
    return [str(device).strip() for device in cuda_visible_devices if str(device).strip()]


def _format_cuda_visible_devices(devices: list[str]) -> str | None:
    """把 GPU 列表转成可放入缓存键和环境变量的字符串。"""

    return ",".join(devices) if devices else None


def _semantic_embedding_worker(
    input_queue,
    output_queue,
    model_name: str,
    device: str,
    max_length: int,
    cuda_visible_device: str | None,
):
    if cuda_visible_device:
        # 每个 embedding worker 只看见一张专用物理 GPU，避免占用训练 GPU。
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        if device == "cuda" or device.startswith("cuda:"):
            device = "cuda:0"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length

    while True:
        item = input_queue.get()
        if item is None:
            break

        job_id, chunk_idx, texts, batch_size = item
        try:
            with torch.inference_mode():
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            output_queue.put((job_id, chunk_idx, embeddings, None))
        except Exception as exc:
            output_queue.put((job_id, chunk_idx, None, repr(exc)))


class _EmbeddingWorkerPool:
    def __init__(self, model_name: str, device: str, max_length: int, num_processes: int, cuda_visible_devices: list[str] | None = None):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.num_processes = num_processes
        self.cuda_visible_devices = cuda_visible_devices or []

        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        process_specs = []
        for worker_idx in range(num_processes):
            cuda_visible_device = self.cuda_visible_devices[worker_idx % len(self.cuda_visible_devices)] if self.cuda_visible_devices else None
            process = ctx.Process(
                target=_semantic_embedding_worker,
                args=(
                    self.input_queue,
                    self.output_queue,
                    model_name,
                    device,
                    max_length,
                    cuda_visible_device,
                ),
                daemon=True,
            )
            process_specs.append((process, cuda_visible_device))

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for process, cuda_visible_device in process_specs:
                if cuda_visible_device:
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
                process.start()
                self.processes.append(process)
        finally:
            if original_cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    def encode(self, texts: list[str], batch_size: int, chunk_size: int | None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if chunk_size is None:
            chunk_size = max(batch_size, math.ceil(len(texts) / self.num_processes))
        chunk_size = max(1, chunk_size)

        job_id = uuid.uuid4().hex
        chunks = [texts[start : start + chunk_size] for start in range(0, len(texts), chunk_size)]
        for chunk_idx, chunk in enumerate(chunks):
            self.input_queue.put((job_id, chunk_idx, chunk, batch_size))

        results: list[np.ndarray | None] = [None] * len(chunks)
        pending = len(chunks)
        while pending > 0:
            try:
                result_job_id, chunk_idx, embeddings, error = self.output_queue.get(timeout=3600)
            except queue.Empty as exc:
                raise RuntimeError("Timed out waiting for semantic embedding worker output") from exc

            if result_job_id != job_id:
                continue
            if error is not None:
                raise RuntimeError(f"Semantic embedding worker failed: {error}")

            results[chunk_idx] = embeddings
            pending -= 1

        return np.concatenate(results, axis=0)

    def close(self):
        for _ in self.processes:
            self.input_queue.put(None)

        for process in self.processes:
            if process.is_alive():
                process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()

        self.input_queue.close()
        self.output_queue.close()


def _cleanup_worker_pools():
    for pool in list(_POOL_CACHE.values()):
        try:
            pool.close()
        except Exception:
            pass
    _POOL_CACHE.clear()


atexit.register(_cleanup_worker_pools)


class SemanticEmbeddingSimilarity(BatchSimilarityComputer):
    """基于句向量嵌入计算余弦相似度。

    Args:
        model_name: sentence-transformers 兼容模型名称。
        batch_size: 编码批大小。
        max_length: 嵌入模型配置的最大序列长度。
        device: 计算设备，支持 ``"cpu"`` 或 ``"cuda"``。
        num_processes: 分片编码进程数，1 表示单进程。
        tail_tokens: 只编码每个 response 尾部的 token 数；``None`` 表示编码完整 response。
        chunk_size: 多进程时每个任务分片的文本条数，默认按进程数自动均分。
        cuda_visible_devices: semantic embedding 专用的物理 GPU 列表；多进程时每个 worker 只看见其中一张 GPU。

    说明：
        编码前会对重复文本去重，避免同一 response 重复进入嵌入模型。
        ``num_processes > 1`` 时使用持久 worker pool，每个 worker 加载一份 embedding 模型。
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
        num_processes: int = 1,
        tail_tokens: int | None = 1024,
        chunk_size: int | None = None,
        cuda_visible_devices: str | list[int | str] | tuple[int | str, ...] | None = None,
        **kwargs,
    ):
        kwargs.pop("deduplicate", None)
        kwargs.pop("clamp", None)
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or start with 'cuda'")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if tail_tokens is not None and tail_tokens < 1:
            raise ValueError("tail_tokens must be >= 1 or None")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 or None")
        normalized_cuda_visible_devices = _normalize_cuda_visible_devices(cuda_visible_devices)
        if normalized_cuda_visible_devices and not device.startswith("cuda"):
            raise ValueError("cuda_visible_devices 只应在 device 为 cuda 时设置")

        effective_max_length = min(max_length, tail_tokens) if tail_tokens is not None else max_length
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            num_processes=num_processes,
            tail_tokens=tail_tokens,
            chunk_size=chunk_size,
            cuda_visible_devices=_format_cuda_visible_devices(normalized_cuda_visible_devices),
            **kwargs,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.effective_max_length = effective_max_length
        self.device = device
        self.num_processes = num_processes
        self.tail_tokens = tail_tokens
        self.chunk_size = chunk_size
        self.cuda_visible_devices = normalized_cuda_visible_devices
        self.cuda_visible_devices_key = _format_cuda_visible_devices(normalized_cuda_visible_devices)
        self.model = None

    def _load_model(self):
        cache_key = (self.model_name, self.device, self.effective_max_length, self.cuda_visible_devices_key)
        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            return self.model

        from sentence_transformers import SentenceTransformer

        model_device = self.device
        if self.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices_key or ""
            if model_device == "cuda" or model_device.startswith("cuda:"):
                model_device = "cuda:0"
        model = SentenceTransformer(self.model_name, device=model_device)
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.effective_max_length
        _MODEL_CACHE[cache_key] = model
        self.model = model
        return model

    def _load_worker_pool(self) -> _EmbeddingWorkerPool:
        cache_key = (self.model_name, self.device, self.effective_max_length, self.num_processes, self.cuda_visible_devices_key)
        if cache_key not in _POOL_CACHE:
            _POOL_CACHE[cache_key] = _EmbeddingWorkerPool(
                model_name=self.model_name,
                device=self.device,
                max_length=self.effective_max_length,
                num_processes=self.num_processes,
                cuda_visible_devices=self.cuda_visible_devices,
            )
        return _POOL_CACHE[cache_key]

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        if self.num_processes > 1:
            pool = self._load_worker_pool()
            embeddings = pool.encode(texts=texts, batch_size=self.batch_size, chunk_size=self.chunk_size)
            return torch.from_numpy(embeddings).to(dtype=torch.float32)

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

    def _decode_tail_responses(self, data: DataProto, tokenizer) -> list[str]:
        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__} 需要 tokenizer 才能解码文本")

        cache = self._get_batch_cache(data)
        cache_key = ("decoded_tail_responses", id(tokenizer), self.tail_tokens)
        if cache_key in cache:
            return cache[cache_key]

        responses = data.batch["responses"]
        valid_lengths = self._get_valid_response_lengths(data)

        texts = []
        for idx, length in enumerate(valid_lengths):
            if length <= 0:
                texts.append("")
                continue

            start = 0 if self.tail_tokens is None else max(0, length - self.tail_tokens)
            token_ids = responses[idx, start:length].detach().cpu().tolist()
            texts.append(tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else "")
        cache[cache_key] = texts
        return texts

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
        texts = self._decode_tail_responses(data, tokenizer)
        target_device = data.batch["responses"].device
        if not texts:
            return torch.zeros((0, 0), device=target_device, dtype=torch.float32)

        embeddings = self._encode_texts(texts)
        if embeddings.device != target_device:
            embeddings = embeddings.to(target_device)
        return self._build_groupwise_similarity_matrix(data, embeddings, target_device)
