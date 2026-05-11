"""语义余弦相似度指标。"""

from __future__ import annotations

import numpy as np


def average_pairwise_cosine(vectors: list[np.ndarray]) -> float | None:
    """计算同一题多个输出 embedding 的两两余弦相似度均值。

    值越低表示语义空间中越分散。若少于两个输出则无法计算。
    """

    n_vectors = len(vectors)
    if n_vectors < 2:
        return None
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    vector_sum = matrix.sum(axis=0, dtype=np.float64)
    off_diagonal_sum = float(np.dot(vector_sum, vector_sum) - n_vectors)
    mean_cosine = off_diagonal_sum / (n_vectors * (n_vectors - 1))
    return float(np.clip(mean_cosine, -1.0, 1.0))


class SemanticEmbeddingCache:
    """按唯一文本缓存 sentence-transformers embedding。

    评测大 Pass@K 时，同一答案可能重复出现。先去重再编码可以显著减少
    embedding 模型推理次数。
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32, max_length: int | None = 4096):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._cache: dict[str, np.ndarray] = {}

    def encode_missing(self, texts: list[str]) -> None:
        """只编码缓存中不存在的文本。"""

        missing_texts = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if not missing_texts:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("计算 semantic-cosine 需要安装 sentence-transformers") from exc

        try:
            model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
        except TypeError:
            model = SentenceTransformer(self.model_name, device=self.device)

        if self.max_length is not None and hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_length

        embeddings = model.encode(
            missing_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for idx, text in enumerate(missing_texts):
            self._cache[text] = embeddings[idx]

    def get_many(self, texts: list[str]) -> list[np.ndarray]:
        """返回一组文本的 embedding；调用前会自动补齐缓存。"""

        self.encode_missing(texts)
        return [self._cache[text] for text in texts]
