#!/usr/bin/env python3
"""入口一：直接读取已有 validation/rollout JSONL 进行评测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.aer.eval.evaluator import evaluate_records, write_evaluation_outputs
from recipe.aer.eval.io_utils import ensure_output_dir, load_jsonl_records
from recipe.aer.eval.metrics.registry import parse_metric_names


def parse_ks(raw_ks: str) -> list[int]:
    """解析逗号分隔的 k 列表。"""

    return [int(value) for value in raw_ks.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="从已有 JSONL 输出计算 AER 评测指标")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 文件、目录或 glob")
    parser.add_argument("--output-dir", required=True, help="评测结果输出目录")
    parser.add_argument("--metrics", default="all", help="评测指标，逗号分隔；默认 all")
    parser.add_argument("--ks", default="1,2,4,8", help="Pass@K 的 k 列表，逗号分隔")
    parser.add_argument("--correct-threshold", type=float, default=0.5, help="score/acc 大于等于该值视为正确")
    parser.add_argument("--semantic-model", default="/data/models/Qwen/Qwen3-Embedding-0.6B", help="semantic-cosine 使用的 embedding 模型路径")
    parser.add_argument("--semantic-device", default="cpu", help="semantic-cosine 编码设备")
    parser.add_argument("--semantic-batch-size", type=int, default=32, help="semantic-cosine 编码 batch size")
    parser.add_argument("--semantic-max-length", type=int, default=4096, help="semantic-cosine 编码最大长度")
    parser.add_argument("--prompt-preview-chars", type=int, default=120, help="per-prompt CSV 中保留的 prompt 预览长度")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    metrics = parse_metric_names(args.metrics)
    ks = parse_ks(args.ks)
    records = load_jsonl_records(args.input)
    output_dir = ensure_output_dir(args.output_dir)

    summary_rows, per_prompt_rows, summary_fields, prompt_fields = evaluate_records(
        records=records,
        metrics=metrics,
        ks=ks,
        correct_threshold=args.correct_threshold,
        semantic_model=args.semantic_model,
        semantic_device=args.semantic_device,
        semantic_batch_size=args.semantic_batch_size,
        semantic_max_length=args.semantic_max_length,
        prompt_preview_chars=args.prompt_preview_chars,
    )
    write_evaluation_outputs(output_dir, summary_rows, per_prompt_rows, summary_fields, prompt_fields)
    print(f"已写入 {output_dir / 'validation_summary.csv'}")
    print(f"已写入 {output_dir / 'validation_per_prompt.csv'}")


if __name__ == "__main__":
    main()
