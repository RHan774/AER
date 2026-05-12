#!/usr/bin/env python3
"""入口二：从模型 checkpoint/HF 目录重新推理并评测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.aer.eval.eval_from_jsonl import evaluate_jsonl_inputs
from recipe.aer.eval.inference import rollout_and_verify
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


def build_rollout_metadata(args, metrics: list[str], ks: list[int]) -> dict:
    """构造 rollout 元信息，用于续跑时校验输出文件是否匹配当前参数。"""

    return {
        "model_path": args.model_path,
        "val_files": args.val_files,
        "prompt_key": args.prompt_key,
        "limit": args.limit,
        "deduplicate": not args.no_deduplicate,
        "step": args.step,
        "metrics": metrics,
        "ks": ks,
        "samples_per_prompt": args.samples_per_prompt,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "rollout_save_batch_size": args.rollout_save_batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_max_model_len": args.vllm_max_model_len,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }


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
    parser.add_argument("--rollout-save-batch-size", type=int, default=8, help="每生成多少道题就追加保存一次 rollout；越小断点越细，越大吞吐更高")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="vLLM tensor parallel size")
    parser.add_argument("--dtype", default="bfloat16", help="模型推理 dtype")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU 显存利用率")
    parser.add_argument("--vllm-max-model-len", type=int, default=None, help="vLLM 最大上下文长度；建议设为 prompt+response 上限")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=None, help="vLLM 最大并发序列数")
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=None, help="vLLM 每轮调度的最大 batched tokens")
    parser.add_argument("--trust-remote-code", action="store_true", help="加载模型时允许 remote code")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="top-p 采样")
    parser.add_argument("--top-k", type=int, default=20, help="top-k 采样；小于等于 0 表示关闭")
    parser.add_argument("--correct-threshold", type=float, default=0.5, help="score/acc 大于等于该值视为正确")
    parser.add_argument("--semantic-model", default="/data/models/Qwen/Qwen3-Embedding-0.6B", help="semantic-cosine 使用的 embedding 模型路径")
    parser.add_argument("--semantic-device", default="cuda", help="semantic-cosine 编码设备；多卡可用逗号分隔，如 cuda:4,cuda:5")
    parser.add_argument("--semantic-batch-size", type=int, default=32, help="semantic-cosine 编码 batch size")
    parser.add_argument("--semantic-max-length", type=int, default=4096, help="semantic-cosine 编码最大长度")
    parser.add_argument("--prompt-preview-chars", type=int, default=120, help="per-prompt CSV 中保留的 prompt 预览长度")
    parser.add_argument("--num-shards", type=int, default=1, help="按题目切成多少个分片，用于多进程并行推理")
    parser.add_argument("--shard-index", type=int, default=0, help="当前进程负责的分片编号，范围 [0, num_shards)")
    parser.add_argument("--skip-eval", action="store_true", help="只写 rollout JSONL，不立即聚合指标；多分片并行时推荐开启")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    metrics = parse_metric_names(args.metrics)
    ks = parse_ks(args.ks)
    args.samples_per_prompt = infer_samples_per_prompt(metrics, ks, args.samples_per_prompt)
    output_dir = ensure_output_dir(args.output_dir)

    rollout_name = "model_rollout.jsonl" if args.num_shards == 1 else f"model_rollout_shard_{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
    rollout_path = output_dir / rollout_name
    rollout_metadata = build_rollout_metadata(args, metrics, ks)
    rollout_and_verify(args, rollout_path=rollout_path, metadata=rollout_metadata)

    if args.skip_eval:
        print(f"每题生成样本数: {args.samples_per_prompt}")
        print(f"已写入 {rollout_path}")
        print("已跳过指标聚合；多分片完成后请用 eval_from_jsonl.py 汇总所有 shard JSONL")
        return

    eval_output_dir = output_dir if args.num_shards == 1 else ensure_output_dir(output_dir / f"shard_{args.shard_index:05d}-of-{args.num_shards:05d}")
    evaluate_jsonl_inputs(
        input_paths=[str(rollout_path)],
        output_dir=eval_output_dir,
        metrics=metrics,
        ks=ks,
        correct_threshold=args.correct_threshold,
        semantic_model=args.semantic_model,
        semantic_device=args.semantic_device,
        semantic_batch_size=args.semantic_batch_size,
        semantic_max_length=args.semantic_max_length,
        prompt_preview_chars=args.prompt_preview_chars,
    )
    print(f"每题生成样本数: {args.samples_per_prompt}")
    print(f"已写入 {rollout_path}")
    print(f"已写入 {eval_output_dir / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()
