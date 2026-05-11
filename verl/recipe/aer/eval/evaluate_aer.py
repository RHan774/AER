#!/usr/bin/env python3
"""兼容入口：保留旧命令，同时委托给新的模块化评测实现。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.aer.eval import eval_from_jsonl
from recipe.aer.eval.train_log import DEFAULT_TRAIN_KEYS, dumps_tau_plan, export_tau_plan, export_train_log


def build_parser() -> argparse.ArgumentParser:
    """构造兼容旧用法的命令行参数。"""

    parser = argparse.ArgumentParser(description="AER 实验评测工具兼容入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validation", help="等价于 eval_from_jsonl.py")
    validation.add_argument("--input", nargs="+", required=True, help="JSONL 文件、目录或 glob")
    validation.add_argument("--output-dir", required=True, help="输出目录")
    validation.add_argument("--metrics", default="all", help="评测指标，逗号分隔；默认 all")
    validation.add_argument("--ks", default="1,2,4,8", help="Pass@K 的 k 列表")
    validation.add_argument("--correct-threshold", type=float, default=0.5, help="score/acc 大于等于该值视为正确")
    validation.add_argument("--semantic-model", default="/data/models/Qwen/Qwen3-Embedding-0.6B", help="semantic-cosine 使用的 embedding 模型")
    validation.add_argument("--semantic-device", default="cpu", help="semantic-cosine 编码设备")
    validation.add_argument("--semantic-batch-size", type=int, default=32, help="semantic-cosine 编码 batch size")
    validation.add_argument("--semantic-max-length", type=int, default=4096, help="semantic-cosine 编码最大长度")
    validation.add_argument("--prompt-preview-chars", type=int, default=120, help="prompt 预览长度")

    train_log = subparsers.add_parser("train-log", help="解析训练日志曲线")
    train_log.add_argument("--input", required=True, help="训练日志路径，例如 log.txt")
    train_log.add_argument("--output-dir", required=True, help="输出目录")
    train_log.add_argument("--keys", default=",".join(DEFAULT_TRAIN_KEYS), help="导出的指标 key，逗号分隔")
    train_log.add_argument("--all-keys", action="store_true", help="导出所有数值指标")

    tau_plan = subparsers.add_parser("tau-plan", help="根据校准日志生成 tau 候选值")
    tau_plan.add_argument("--input", required=True, help="tau=0 校准训练日志")
    tau_plan.add_argument("--algorithm", required=True, help="相似度算法名")
    tau_plan.add_argument("--output", required=True, help="输出 CSV 或 JSON")
    tau_plan.add_argument("--precision", type=int, default=3, help="tau 小数位")
    return parser


def main() -> None:
    """CLI 入口。"""

    args = build_parser().parse_args()
    if args.command == "validation":
        eval_from_jsonl.main(
            [
                "--input",
                *args.input,
                "--output-dir",
                args.output_dir,
                "--metrics",
                args.metrics,
                "--ks",
                args.ks,
                "--correct-threshold",
                str(args.correct_threshold),
                "--semantic-model",
                args.semantic_model,
                "--semantic-device",
                args.semantic_device,
                "--semantic-batch-size",
                str(args.semantic_batch_size),
                "--semantic-max-length",
                str(args.semantic_max_length),
                "--prompt-preview-chars",
                str(args.prompt_preview_chars),
            ]
        )
        return

    if args.command == "train-log":
        keys = [key for key in args.keys.split(",") if key.strip()]
        export_train_log(input_path=args.input, output_dir=args.output_dir, keys=keys, all_keys=args.all_keys)
        print(f"已写入 {Path(args.output_dir) / 'train_metrics.csv'}")
        return

    if args.command == "tau-plan":
        row = export_tau_plan(
            input_path=args.input,
            algorithm=args.algorithm,
            output_path=args.output,
            precision=args.precision,
        )
        print(f"已写入 {args.output}")
        print(dumps_tau_plan(row))
        return


if __name__ == "__main__":
    main()
