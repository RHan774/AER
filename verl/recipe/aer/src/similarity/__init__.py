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

"""AER 探索奖励中使用的相似度算法集合。"""

from typing import Dict, Type

from .base import SimilarityComputer
from .char_ngram import CharNGramSimilarity
from .compression_ratio import CompressionRatioSimilarity
from .levenshtein import LevenshteinSimilarity
from .ngram_overlap import NGramOverlapSimilarity
from .rouge_l import RougeLSimilarity
from .semantic_embedding import SemanticEmbeddingSimilarity
from .simhash import SimHashSimilarity
from .tfidf_cosine import TFIDFCosineSimilarity
from .token_match import TokenMatchSimilarity

SIMILARITY_REGISTRY: Dict[str, Type[SimilarityComputer]] = {
    "token_match": TokenMatchSimilarity,
    "ngram_overlap": NGramOverlapSimilarity,
    "char_ngram": CharNGramSimilarity,
    "levenshtein": LevenshteinSimilarity,
    "tfidf_cosine": TFIDFCosineSimilarity,
    "semantic_embedding": SemanticEmbeddingSimilarity,
    "simhash": SimHashSimilarity,
    "compression_ratio": CompressionRatioSimilarity,
    "rouge_l": RougeLSimilarity,
}


def get_similarity_computer(name: str, **kwargs) -> SimilarityComputer:
    """
    根据算法名称创建相似度计算器。

    Args:
        name: 算法名称，支持以下选项：
            - "token_match": Token 精确匹配
            - "ngram_overlap": N-gram 重叠度
            - "char_ngram": 字符级 N-gram
            - "levenshtein": 编辑距离
            - "tfidf_cosine": TF-IDF 余弦相似度
            - "semantic_embedding": 语义嵌入相似度
            - "simhash": SimHash 近重复相似度
            - "compression_ratio": 压缩比相似度（快速多样性评估）
            - "rouge_l": ROUGE-L 相似度（基于最长公共子序列）
        **kwargs: 算法特定的参数

    Returns:
        SimilarityComputer: 相似度计算器实例

    Raises:
        ValueError: 当 ``name`` 未知时抛出。
    """
    if name not in SIMILARITY_REGISTRY:
        available = list(SIMILARITY_REGISTRY.keys())
        raise ValueError(f"未知的相似度算法: '{name}'。可用的算法: {available}")

    return SIMILARITY_REGISTRY[name](**kwargs)


def list_available_algorithms() -> list:
    """
    列出当前环境下可用的相似度算法。
    """
    return sorted(SIMILARITY_REGISTRY.keys())


__all__ = [
    "SimilarityComputer",
    "TokenMatchSimilarity",
    "NGramOverlapSimilarity",
    "CharNGramSimilarity",
    "LevenshteinSimilarity",
    "TFIDFCosineSimilarity",
    "CompressionRatioSimilarity",
    "RougeLSimilarity",
    "SimHashSimilarity",
    "get_similarity_computer",
    "list_available_algorithms",
    "SIMILARITY_REGISTRY",
    "SemanticEmbeddingSimilarity",
]
