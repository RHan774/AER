#!/usr/bin/env python3
"""入口二：从模型 checkpoint/HF 目录重新推理并评测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.aer.eval.evaluator import evaluate_records, write_evaluation_outputs
from recipe.aer.eval.inference import rollout_and_verify, write_rollout_jsonl
from recipe.aer.eval.io_utils import ensure_output_dir
from recipe.aer.eval.metrics.registry import parse_metric_names, should_compute_pass_at_k


def parse_ks(raw_ks: str) -> list[int]:
    """解析逗号分隔的 k 列表。"""

    return [int(value) for value in raw_ks.split(",") if value.strip()]


def infer_samples_per_prompt(metrics: list[str], ks: list[int], explicit_samples_per_prompt: int | None = None) -> int:
    """根据指标决定每题 rollout 次数。

    默认保持实验设计口径：只有请求 Pass@K 时才按 `max(--ks)` 多次推理。
    若用户显式指定 `--samples-per-prompt`，以用户指定值为准。
    """

    if explicit_samples_per_prompt is not None:
        if explicit_samples_per_prompt <= 0:
            raise ValueError("--samples-per-prompt 必须大于 0")
        return explicit_samples_per_prompt
    if should_compute_pass_at_k(metrics):
        return max(ks) if ks else 1
    return 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="从模型重新推理并计算 AER 评测指标")
    parser.add_argument("--model-path", required=True, help="可直接加载的 HF 模型目录，或已 merge 的 checkpoint")
    parser.add_argument("--val-files", nargs="+", required=True, help="验证 parquet 路径，支持多个文件")
    parser.add_argument("--output-dir", required=True, help="评测结果输出目录")
    parser.add_argument("--metrics", default="all", help="评测指标，逗号分隔；默认 all")
    parser.add_argument("--ks", default="1,2,4,8", help="Pass@K 的 k 列表，逗号分隔")
    parser.add_argument("--samples-per-prompt", type=int, default=None, help="每题生成样本数；默认由 metrics 和 ks 自动推断")
    parser.add_argument("--prompt-key", default="prompt", help="parquet 中的 prompt 字段名")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 道题，用于调试")
    parser.add_argument("--no-deduplicate", action="store_true", help="不按 unique_id/prompt 去重")
    parser.add_argument("--step", type=int, default=None, help="写入 JSONL 的 step 标记，可填 checkpoint step")
    parser.add_argument("--seed", type=int, default=42, help="推理采样随机种子")
    parser.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto", help="推理后端")
    parser.add_argument("--batch-size", type=int, default=8, help="HF 后端展开样本后的 batch size")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="vLLM tensor parallel size")
    parser.add_argument("--dtype", default="bfloat16", help="模型推理 dtype")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU 显存利用率")
    parser.add_argument("--trust-remote-code", action="store_true", help="加载模型时允许 remote code")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p 采样")
    parser.add_argument("--top-k", type=int, default=20, help="top-k 采样；小于等于 0 表示关闭")
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
    args.samples_per_prompt = infer_samples_per_prompt(metrics, ks, args.samples_per_prompt)
    output_dir = ensure_output_dir(args.output_dir)

    records = rollout_and_verify(args)
    rollout_path = output_dir / "model_rollout.jsonl"
    write_rollout_jsonl(
        rollout_path,
        records,
        metadata={
            "model_path": args.model_path,
            "val_files": args.val_files,
            "metrics": metrics,
            "ks": ks,
            "samples_per_prompt": args.samples_per_prompt,
            "backend": args.backend,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
    )

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
    print(f"每题生成样本数: {args.samples_per_prompt}")
    print(f"已写入 {rollout_path}")
    print(f"已写入 {output_dir / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()
