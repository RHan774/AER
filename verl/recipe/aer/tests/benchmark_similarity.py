#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度计算算法对比测试脚本

该脚本用于对比不同相似度计算算法在相同数据上的表现。
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加项目路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# 添加当前脚本目录，确保可导入同目录下的 test_similarity.py
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from recipe.aer.src.similarity import list_available_algorithms
import test_similarity


def run_all_algorithms(
    file_path: str = "rollout_example.jsonl"
) -> Dict[str, Dict[str, Any]]:
    """
    运行所有可用算法并记录结果

    Args:
        file_path: 数据文件路径

    Returns:
        Dict: 每个算法的结果
    """
    # 加载数据
    loader = test_similarity.RolloutDataLoader(file_path)
    loader.load()

    outputs = loader.get_outputs()
    inputs = loader.get_inputs()

    algorithms = list_available_algorithms()

    # 定义每种算法的参数
    algo_configs = {
        "token_match": {},
        "ngram_overlap": {"n": 3},
        "char_ngram": {"n": 4, "metric": "jaccard"},
        "levenshtein": {"normalize_method": "max"},
        "tfidf_cosine": {"max_features": 1000, "ngram_range": (1, 2)},
        # add: 新增算法配置
        "compression_ratio": {"compression_type": "gzip"},
        "rouge_l": {"beta": 1.0},
        "simhash": {"n": 3, "hash_bits": 64},
        # "semantic_embedding": {"model_name": "all-MiniLM-L6-v2", "device": "cpu"},
    }

    results = {}

    for algo in algorithms:
        if algo not in algo_configs:
            continue

        print(f"\n{'='*80}")
        print(f"测试算法: {algo}")
        print(f"{'='*80}")

        tester = test_similarity.SimilarityTester(outputs, inputs)

        try:
            start_time = time.time()
            similarity_matrix, exploration_rewards = tester.compute_similarity(
                algo,
                **algo_configs[algo]
            )
            elapsed_time = time.time() - start_time

            results[algo] = {
                "similarity_matrix": similarity_matrix,
                "exploration_rewards": exploration_rewards,
                "elapsed_time": elapsed_time,
                "mean_reward": exploration_rewards.mean(),
                "std_reward": exploration_rewards.std(),
                "min_reward": exploration_rewards.min(),
                "max_reward": exploration_rewards.max(),
            }

            print(f"计算时间: {elapsed_time:.3f} 秒")

        except Exception as e:
            print(f"算法 {algo} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    return results


def print_comparison(results: Dict[str, Dict[str, Any]]):
    """打印对比结果"""
    print(f"\n{'='*80}")
    print(f"算法对比结果")
    print(f"{'='*80}")

    print(f"\n{'算法':<20} {'平均探索奖励':<15} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'计算时间(s)':<12}")
    print("-" * 85)

    for algo, result in results.items():
        print(
            f"{algo:<20} "
            f"{result['mean_reward']:<15.6f} "
            f"{result['std_reward']:<12.6f} "
            f"{result['min_reward']:<12.6f} "
            f"{result['max_reward']:<12.6f} "
            f"{result['elapsed_time']:<12.3f}"
        )

    # add: 相对性能分析
    print(f"\n{'='*80}")
    print(f"相对性能分析 (以 token_match 为基准)")
    print(f"{'='*80}")

    if "token_match" in results:
        baseline = results["token_match"]["mean_reward"]

        print(f"\n{'算法':<20} {'相对探索奖励':<20} {'相对计算时间':<20}")
        print("-" * 50)

        for algo, result in results.items():
            rel_reward = (result["mean_reward"] / baseline - 1) * 100
            rel_time = (result["elapsed_time"] / results["token_match"]["elapsed_time"] - 1) * 100
            print(
                f"{algo:<20} "
                f"{rel_reward:>+6.1f}%{'':<13} "
                f"{rel_time:>+6.1f}%"
            )


def save_comparison(results: Dict[str, Dict[str, Any]], output_file: str):
    """保存对比结果"""
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": list(results.keys()),
        "results": {
            algo: {
                "mean_reward": float(result["mean_reward"]),
                "std_reward": float(result["std_reward"]),
                "min_reward": float(result["min_reward"]),
                "max_reward": float(result["max_reward"]),
                "elapsed_time": float(result["elapsed_time"]),
                "exploration_rewards": result["exploration_rewards"].tolist(),
            }
            for algo, result in results.items()
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    # 查找数据文件
    file_path = "rollout_example.jsonl"
    script_dir = Path(__file__).parent

    if not os.path.exists(file_path):
        possible_paths = [
            script_dir / file_path,
            script_dir / "rollout_example.jsonl",
        ]
        for path in possible_paths:
            if path.exists():
                file_path = str(path)
                break
        else:
            print(f"错误: 找不到文件 {file_path}")
            return

    print(f"使用数据文件: {file_path}")

    # 运行所有算法
    results = run_all_algorithms(file_path)

    # 打印对比结果
    if results:
        print_comparison(results)

        # 保存结果
        output_file = script_dir / "similarity_benchmark_results.json"
        save_comparison(results, str(output_file))
    else:
        print("没有成功运行的算法")


if __name__ == "__main__":
    main()
