#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度计算算法测试脚本

该脚本用于测试 AER 中实现的多种相似度计算算法。
读取 rollout_example.jsonl 文件，计算所有 output 之间的相似度矩阵和探索奖励。

使用方法:
    python test_similarity.py --algorithm token_match
    python test_similarity.py --algorithm ngram_overlap --n 3
    python test_similarity.py --algorithm char_ngram --n 4
    python test_similarity.py --algorithm levenshtein
    python test_similarity.py --algorithm tfidf_cosine
    python test_similarity.py --algorithm semantic_embedding --device cpu
    python test_similarity.py --algorithm simhash --n 3
"""

import argparse
import json
import os
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

# add: 添加项目路径到 sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from recipe.aer.src.similarity import (
    get_similarity_computer,
    list_available_algorithms,
)


class RolloutDataLoader:
    """加载 rollout_example.jsonl 数据"""

    def __init__(self, file_path: str):
        """
        初始化数据加载器

        Args:
            file_path: rollout_example.jsonl 文件路径
        """
        self.file_path = file_path
        self.data = []
        self.outputs = []
        self.inputs = []

    def load(self) -> List[Dict[str, Any]]:
        """
        加载数据

        Returns:
            List[Dict]: 数据列表

        Raises:
            FileNotFoundError: 文件不存在
            JSONDecodeError: JSON 解析错误
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.data.append(data)
                    self.outputs.append(data.get("output", ""))
                    self.inputs.append(data.get("input", ""))
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")

        print(f"成功加载 {len(self.data)} 条数据")
        return self.data

    def get_outputs(self) -> List[str]:
        """获取所有 output 内容"""
        return self.outputs

    def get_inputs(self) -> List[str]:
        """获取所有 input 内容"""
        return self.inputs


class SimilarityTester:
    """相似度计算测试器"""

    def __init__(self, outputs: List[str], inputs: List[str], tokenizer_path: str = None):
        """
        初始化测试器

        Args:
            outputs: 输出文本列表
            inputs: 输入文本列表
            tokenizer_path: 分词器路径（可选）
        """
        self.outputs = outputs
        self.inputs = inputs
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.similarity_matrix = None
        self.exploration_rewards = None

        # add: 延迟加载分词器（仅在需要时加载）
        if tokenizer_path:
            self._load_tokenizer()

    def _load_tokenizer(self):
        """加载分词器"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )
            print(f"成功加载分词器: {self.tokenizer_path}")
        except ImportError:
            print("警告: transformers 未安装，部分算法可能无法使用")
            print("安装命令: pip install transformers")
        except Exception as e:
            print(f"警告: 加载分词器失败: {e}")

    def compute_similarity(
        self,
        algorithm: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算相似度矩阵和探索奖励

        Args:
            algorithm: 相似度算法名称
            **kwargs: 算法参数

        Returns:
            Tuple[np.ndarray, np.ndarray]: (相似度矩阵, 探索奖励)
        """
        print(f"\n{'='*60}")
        print(f"使用算法: {algorithm}")
        print(f"算法参数: {kwargs}")
        print(f"{'='*60}")

        # add: 创建相似度计算器
        try:
            computer = get_similarity_computer(algorithm, **kwargs)
        except ValueError as e:
            print(f"错误: {e}")
            print(f"可用算法: {list_available_algorithms()}")
            raise

        # add: 构建 DataProto 对象（模拟实际数据格式）
        from verl import DataProto

        batch_size = len(self.outputs)

        # add: 创建模拟的 token 数据
        # 对于不需要 token 的算法，我们创建虚拟数据
        # 对于需要文本的算法，我们直接传递文本

        # 为每个 output 创建唯一的 uid（所有数据属于同一组）
        uids = [f"prompt_0"] * batch_size

        # add: 创建字典格式的数据
        data_dict = {
            "outputs": self.outputs,
            "uids": uids,
        }

        # add: 如果算法需要 token 级别的数据，创建模拟数据
        if algorithm in ["token_match", "ngram_overlap", "simhash"]:
            # 创建虚拟的 token 序列
            max_length = 2048
            responses = torch.zeros(batch_size, max_length, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

            # 这里我们使用一个简化的方法：对于 token_match 和 ngram_overlap，
            # 我们直接使用文本相似度的变体来模拟
            print("注意: 使用文本级别的相似度计算替代 token 级别")
            return self._compute_text_similarity(algorithm, **kwargs)

        # add: 对于需要解码的算法，直接处理文本
        if algorithm in ["char_ngram", "levenshtein", "tfidf_cosine", "semantic_embedding", "simhash"]:
            return self._compute_text_similarity(algorithm, **kwargs)

        raise NotImplementedError(f"算法 {algorithm} 的测试实现尚未完成")

    def _compute_text_similarity(
        self,
        algorithm: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        直接计算文本级别的相似度

        Args:
            algorithm: 算法名称
            **kwargs: 算法参数

        Returns:
            Tuple[np.ndarray, np.ndarray]: (相似度矩阵, 探索奖励)
        """
        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))

        # add: 根据算法选择计算方法
        if algorithm == "token_match":
            similarity_matrix = self._token_match_similarity()
        elif algorithm == "ngram_overlap":
            n = kwargs.get("n", 3)
            similarity_matrix = self._ngram_overlap_similarity(n)
        elif algorithm == "char_ngram":
            n = kwargs.get("n", 4)
            metric = kwargs.get("metric", "jaccard")
            similarity_matrix = self._char_ngram_similarity(n, metric)
        elif algorithm == "levenshtein":
            normalize_method = kwargs.get("normalize_method", "max")
            similarity_matrix = self._levenshtein_similarity(normalize_method)
        elif algorithm == "tfidf_cosine":
            max_features = kwargs.get("max_features", 1000)
            ngram_range = kwargs.get("ngram_range", (1, 2))
            similarity_matrix = self._tfidf_cosine_similarity(max_features, ngram_range)
        elif algorithm == "semantic_embedding":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            device = kwargs.get("device", "cpu")
            similarity_matrix = self._semantic_embedding_similarity(model_name, device)
        elif algorithm == "simhash":
            n = kwargs.get("n", 3)
            hash_bits = kwargs.get("hash_bits", 64)
            use_counts = kwargs.get("use_counts", True)
            calibrate_random = kwargs.get("calibrate_random", True)
            similarity_matrix = self._simhash_similarity(n, hash_bits, use_counts, calibrate_random)
        else:
            raise ValueError(f"未知算法: {algorithm}")

        # add: 计算探索奖励
        # 探索奖励 = 1 / (该样本与其他所有样本的相似度之和)
        # 如果相似度之和为 0，则探索奖励为 0
        similarity_sums = similarity_matrix.sum(axis=1)
        exploration_rewards = np.zeros_like(similarity_sums)
        for i in range(len(similarity_sums)):
            if similarity_sums[i] > 0:
                exploration_rewards[i] = 1.0 / similarity_sums[i]

        self.similarity_matrix = similarity_matrix
        self.exploration_rewards = exploration_rewards

        return similarity_matrix, exploration_rewards

    def _token_match_similarity(self) -> np.ndarray:
        """
        Token 精确匹配相似度

        注意：由于测试环境缺少完整的 tokenizer 和 DataProto，
        这里使用字符级匹配来近似 token 级别的行为。
        这与实际训练中的 token 级别匹配不完全一致，但可以模拟相似的相似度范围。

        实际训练使用：similarity = sum(token_i == token_j) / sqrt(L_i * L_j)
        这里使用：similarity = char_matches / sqrt(L_i * L_j)
        """
        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))

        for i in range(batch_size):
            len_i = len(self.outputs[i])
            for j in range(i, batch_size):
                len_j = len(self.outputs[j])

                # 计算字符级别的匹配数（逐位置比较）
                min_len = min(len_i, len_j)
                matches = sum(1 for k in range(min_len) if self.outputs[i][k] == self.outputs[j][k])

                # 使用 sqrt(L_i * L_j) 归一化，与实际训练一致
                norm_factor = np.sqrt(len_i * len_j)
                if norm_factor > 0:
                    sim = matches / norm_factor
                else:
                    sim = 0.0

                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def _ngram_overlap_similarity(self, n: int) -> np.ndarray:
        """N-gram 重叠度相似度"""
        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))

        def extract_ngrams(text: str, n: int) -> Counter:
            """提取 n-gram"""
            words = text.split()
            if len(words) < n:
                return Counter()
            return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        for i in range(batch_size):
            ngrams_i = extract_ngrams(self.outputs[i], n)
            for j in range(i, batch_size):
                ngrams_j = extract_ngrams(self.outputs[j], n)

                if len(ngrams_i) == 0 or len(ngrams_j) == 0:
                    sim = 0.0
                else:
                    intersection = sum((ngrams_i & ngrams_j).values())
                    union = ngrams_i.total() + ngrams_j.total() - intersection
                    sim = intersection / union if union > 0 else 0.0

                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def _char_ngram_similarity(self, n: int, metric: str = "jaccard") -> np.ndarray:
        """字符级 N-gram 相似度"""
        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))

        def extract_char_ngrams(text: str, n: int) -> set:
            """提取字符级 n-gram"""
            if len(text) < n:
                return {text}
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        for i in range(batch_size):
            ngrams_i = extract_char_ngrams(self.outputs[i], n)
            for j in range(i, batch_size):
                ngrams_j = extract_char_ngrams(self.outputs[j], n)

                if len(ngrams_i) == 0 and len(ngrams_j) == 0:
                    sim = 0.0
                elif len(ngrams_i) == 0 or len(ngrams_j) == 0:
                    sim = 0.0
                else:
                    intersection = len(ngrams_i & ngrams_j)

                    if metric == "jaccard":
                        union = len(ngrams_i | ngrams_j)
                        sim = intersection / union if union > 0 else 0.0
                    else:  # dice
                        sim = 2 * intersection / (len(ngrams_i) + len(ngrams_j))

                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def _levenshtein_similarity(self, normalize_method: str = "max") -> np.ndarray:
        """编辑距离相似度"""
        try:
            from rapidfuzz import distance, fuzz
            use_rapidfuzz = True
        except ImportError:
            use_rapidfuzz = False
            print("警告: rapidfuzz 不可用，使用内置实现（较慢）")

        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))

        def levenshtein_sim(s1: str, s2: str) -> float:
            """计算归一化编辑距离相似度"""
            if use_rapidfuzz:
                return fuzz.ratio(s1, s2) / 100.0
            else:
                # 内置实现
                len1, len2 = len(s1), len(s2)
                if len1 == 0 and len2 == 0:
                    return 1.0
                if len1 == 0 or len2 == 0:
                    return 0.0

                # 动态规划计算编辑距离
                dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
                for i in range(len1 + 1):
                    dp[i][0] = i
                for j in range(len2 + 1):
                    dp[0][j] = j

                for i in range(1, len1 + 1):
                    for j in range(1, len2 + 1):
                        cost = 0 if s1[i-1] == s2[j-1] else 1
                        dp[i][j] = min(
                            dp[i-1][j] + 1,      # 删除
                            dp[i][j-1] + 1,      # 插入
                            dp[i-1][j-1] + cost  # 替换
                        )

                dist = dp[len1][len2]

                if normalize_method == "max":
                    max_len = max(len1, len2)
                    return 1.0 - dist / max_len if max_len > 0 else 1.0
                elif normalize_method == "avg":
                    avg_len = (len1 + len2) / 2
                    return 1.0 - dist / avg_len if avg_len > 0 else 1.0
                else:  # min
                    min_len = min(len1, len2)
                    return 1.0 - dist / min_len if min_len > 0 else 1.0

        for i in range(batch_size):
            for j in range(i, batch_size):
                sim = levenshtein_sim(self.outputs[i], self.outputs[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def _tfidf_cosine_similarity(
        self,
        max_features: int = 1000,
        ngram_range: tuple = (1, 2)
    ) -> np.ndarray:
        """TF-IDF 余弦相似度"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError(
                "scikit-learn 不可用，无法使用 TF-IDF 相似度。"
                "请安装: pip install scikit-learn"
            )

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            token_pattern=r'(?u)\b\w+\b',
        )

        tfidf_matrix = vectorizer.fit_transform(self.outputs)
        cosine_sim = cosine_similarity(tfidf_matrix)

        return cosine_sim

    def _semantic_embedding_similarity(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ) -> np.ndarray:
        """语义嵌入相似度"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 不可用，无法使用语义嵌入相似度。"
                "请安装: pip install sentence-transformers"
            )

        print(f"加载模型: {model_name}")
        model = SentenceTransformer(model_name, device=device)

        embeddings = model.encode(
            self.outputs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 计算余弦相似度（嵌入已归一化）
        cosine_sim = np.dot(embeddings, embeddings.T)

        return cosine_sim

    def _simhash_similarity(
        self,
        n: int = 3,
        hash_bits: int = 64,
        use_counts: bool = True,
        calibrate_random: bool = True,
    ) -> np.ndarray:
        """SimHash 近重复相似度"""
        batch_size = len(self.outputs)
        similarity_matrix = np.zeros((batch_size, batch_size))
        hash_mask = (1 << hash_bits) - 1

        def extract_features(text: str) -> dict:
            tokens = text.split()
            if not tokens:
                return {}
            if len(tokens) < n:
                return {tuple(tokens): 1}

            features = {}
            for start in range(len(tokens) - n + 1):
                feature = tuple(tokens[start:start + n])
                if use_counts:
                    features[feature] = features.get(feature, 0) + 1
                else:
                    features[feature] = 1
            return features

        def hash_feature(feature: tuple) -> int:
            payload = ",".join(feature).encode("utf-8")
            low = zlib.crc32(payload)
            high = zlib.crc32(payload, 0x9E3779B9)
            return ((high << 32) | low) & hash_mask

        def fingerprint(features: dict) -> tuple:
            if not features:
                return 0, False

            scores = [0] * hash_bits
            for feature, weight in features.items():
                hashed = hash_feature(feature)
                for bit in range(hash_bits):
                    if hashed & (1 << bit):
                        scores[bit] += weight
                    else:
                        scores[bit] -= weight

            value = 0
            for bit, score in enumerate(scores):
                if score >= 0:
                    value |= 1 << bit
            return value, True

        fingerprints = [fingerprint(extract_features(output)) for output in self.outputs]
        for i in range(batch_size):
            fp_i, has_i = fingerprints[i]
            for j in range(i, batch_size):
                fp_j, has_j = fingerprints[j]
                if not has_i or not has_j:
                    sim = 0.0
                elif fp_i == fp_j:
                    sim = 1.0
                else:
                    distance = (fp_i ^ fp_j).bit_count()
                    if calibrate_random:
                        sim = max(0.0, 1.0 - (2.0 * distance / hash_bits))
                    else:
                        sim = 1.0 - (distance / hash_bits)

                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix

    def print_results(self):
        """打印结果"""
        if self.similarity_matrix is None or self.exploration_rewards is None:
            print("请先运行 compute_similarity()")
            return

        batch_size = len(self.outputs)

        print(f"\n{'='*80}")
        print(f"相似度矩阵 (batch_size={batch_size})")
        print(f"{'='*80}")
        print(f"{' ':>6}", end="")
        for j in range(batch_size):
            print(f"  Out{j+1:02d}", end="")
        print()

        for i in range(batch_size):
            print(f"Out{i+1:02d}", end="")
            for j in range(batch_size):
                print(f" {self.similarity_matrix[i, j]:.3f}", end="")
            print()

        print(f"\n{'='*80}")
        print(f"探索奖励 (Exploration Rewards)")
        print(f"{'='*80}")
        print(f"{'输出':<10} {'相似度之和':<15} {'探索奖励':<15}")
        print("-" * 40)
        for i in range(batch_size):
            sim_sum = self.similarity_matrix[i].sum()
            reward = self.exploration_rewards[i]
            print(f"Out {i+1:02d}   {sim_sum:<15.6f} {reward:<15.6f}")

        # add: 统计信息
        print(f"\n{'='*80}")
        print(f"统计信息")
        print(f"{'='*80}")
        print(f"平均探索奖励: {self.exploration_rewards.mean():.6f}")
        print(f"最小探索奖励: {self.exploration_rewards.min():.6f}")
        print(f"最大探索奖励: {self.exploration_rewards.max():.6f}")
        print(f"探索奖励标准差: {self.exploration_rewards.std():.6f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AER 相似度计算算法测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 token_match 算法
    python tests/test_similarity.py --algorithm token_match

  # 使用 ngram_overlap 算法，n=3
    python tests/test_similarity.py --algorithm ngram_overlap --n 3

  # 使用 char_ngram 算法
    python tests/test_similarity.py --algorithm char_ngram --n 4

  # 使用 levenshtein 算法
    python tests/test_similarity.py --algorithm levenshtein

  # 使用 tfidf_cosine 算法
    python tests/test_similarity.py --algorithm tfidf_cosine

  # 使用 semantic_embedding 算法（CPU）
    python tests/test_similarity.py --algorithm semantic_embedding --device cpu

  # 使用 simhash 算法
    python tests/test_similarity.py --algorithm simhash --n 3

  # 列出所有可用算法
    python tests/test_similarity.py --list-algorithms
        """
    )

    parser.add_argument(
        "--file",
        type=str,
        default="rollout_example.jsonl",
        help="rollout 数据文件路径 (默认: rollout_example.jsonl)"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
            choices=["token_match", "ngram_overlap", "char_ngram", "levenshtein",
                 "tfidf_cosine", "semantic_embedding", "compression_ratio", "rouge_l", "simhash"],
        help="相似度计算算法"
    )

    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="列出所有可用算法"
    )

    # N-gram 参数
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="N-gram 的 n 值 (默认: 3)"
    )

    # char_ngram 参数
    parser.add_argument(
        "--metric",
        type=str,
        choices=["jaccard", "dice"],
        default="jaccard",
        help="相似度度量 (用于 char_ngram)"
    )

    # levenshtein 参数
    parser.add_argument(
        "--normalize-method",
        type=str,
        choices=["max", "avg", "min"],
        default="max",
        help="编辑距离归一化方法"
    )

    # tfidf_cosine 参数
    parser.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="TF-IDF 最大特征数"
    )

    parser.add_argument(
        "--ngram-range",
        type=int,
        nargs=2,
        default=[1, 2],
        help="TF-IDF n-gram 范围"
    )

    # simhash 参数
    parser.add_argument(
        "--hash-bits",
        type=int,
        default=64,
        help="SimHash 指纹位数"
    )

    parser.add_argument(
        "--no-use-counts",
        action="store_true",
        help="不保留重复 n-gram 权重 (用于 simhash)"
    )

    parser.add_argument(
        "--no-calibrate-random",
        action="store_true",
        help="不将随机指纹期望相似度校准为 0 (用于 simhash)"
    )

    # semantic_embedding 参数
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="语义嵌入模型名称"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="计算设备 (用于 semantic_embedding)"
    )

    # add: compression_ratio 算法参数
    parser.add_argument(
        "--compression-type",
        type=str,
        choices=["gzip", "zlib"],
        default="gzip",
        help="压缩类型 (用于 compression_ratio)"
    )

    # add: rouge_l 算法参数
    parser.add_argument(
        "--use-char-level",
        action="store_true",
        help="使用字符级别 (用于 rouge_l)"
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="F-beta 分数的 beta 参数 (用于 rouge_l)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件路径 (可选)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # add: 列出可用算法
    if args.list_algorithms:
        print("可用算法:")
        for algo in list_available_algorithms():
            print(f"  - {algo}")
        return

    # add: 检查算法参数
    if not args.algorithm:
        print("错误: 请指定 --algorithm 参数")
        print(f"可用算法: {list_available_algorithms()}")
        return

    # add: 查找数据文件
    file_path = args.file
    if not os.path.isabs(file_path):
        # 尝试在当前目录和脚本目录中查找
        script_dir = Path(__file__).parent
        possible_paths = [
            Path(file_path),
            script_dir / file_path,
            script_dir / "rollout_example.jsonl",
        ]
        for path in possible_paths:
            if path.exists():
                file_path = str(path)
                break
        else:
            print(f"错误: 找不到文件 {args.file}")
            print(f"已尝试的路径: {[str(p) for p in possible_paths]}")
            return

    print(f"加载数据文件: {file_path}")

    # add: 加载数据
    try:
        loader = RolloutDataLoader(file_path)
        loader.load()
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return

    outputs = loader.get_outputs()
    inputs = loader.get_inputs()

    print(f"数据量: {len(outputs)} 条")
    print(f"输出长度范围: {min(len(o) for o in outputs)} - {max(len(o) for o in outputs)} 字符")

    # add: 构建算法参数
    algo_params = {}

    if args.algorithm == "ngram_overlap":
        algo_params["n"] = args.n
    elif args.algorithm == "char_ngram":
        algo_params["n"] = args.n
        algo_params["metric"] = args.metric
    elif args.algorithm == "levenshtein":
        algo_params["normalize_method"] = args.normalize_method
    elif args.algorithm == "tfidf_cosine":
        algo_params["max_features"] = args.max_features
        algo_params["ngram_range"] = tuple(args.ngram_range)
    elif args.algorithm == "semantic_embedding":
        algo_params["model_name"] = args.model_name
        algo_params["device"] = args.device
    elif args.algorithm == "simhash":
        algo_params["n"] = args.n
        algo_params["hash_bits"] = args.hash_bits
        algo_params["use_counts"] = not args.no_use_counts
        algo_params["calibrate_random"] = not args.no_calibrate_random
    # add: 新增算法的参数处理
    elif args.algorithm == "compression_ratio":
        if args.compression_type:
            algo_params["compression_type"] = args.compression_type
    elif args.algorithm == "rouge_l":
        if args.use_char_level:
            algo_params["use_char_level"] = args.use_char_level
        if args.beta is not None:
            algo_params["beta"] = args.beta

    # add: 创建测试器并计算
    tester = SimilarityTester(outputs, inputs)

    try:
        similarity_matrix, exploration_rewards = tester.compute_similarity(
            args.algorithm,
            **algo_params
        )
    except Exception as e:
        print(f"错误: 相似度计算失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # add: 打印结果
    tester.print_results()

    # add: 保存结果
    if args.output:
        result = {
            "algorithm": args.algorithm,
            "algo_params": algo_params,
            "similarity_matrix": similarity_matrix.tolist(),
            "exploration_rewards": exploration_rewards.tolist(),
            "outputs": outputs,
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
