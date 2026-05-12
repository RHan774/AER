#!/usr/bin/env python3
"""单独从 JSONL 计算 Equational Diversity。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.aer.eval.evaluator import evaluate_records
from recipe.aer.eval.io_utils import ensure_output_dir, load_jsonl_records, write_csv, write_json, write_markdown_table


METRIC_NAME = "equational-diversity"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="从 save/eval 下的 rollout JSONL 单独计算 Equational Diversity")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 文件、目录或 glob，例如 save/eval/.../rerun_pass128_sharded")
    parser.add_argument("--output-dir", default=None, help="输出目录；默认写到 JSONL 所在目录")
    parser.add_argument("--prompt-preview-chars", type=int, default=120, help="per-prompt CSV 中保留的 prompt 预览长度")
    return parser


def infer_output_dir(records: list[dict[str, Any]], explicit_output_dir: str | None) -> Path:
    """推断输出目录，默认跟随输入 JSONL 目录。"""

    if explicit_output_dir:
        return ensure_output_dir(explicit_output_dir)

    source_dirs = {Path(str(record["_source_file"])).parent for record in records if record.get("_source_file")}
    if len(source_dirs) == 1:
        return ensure_output_dir(next(iter(source_dirs)))
    return ensure_output_dir("save/eval/equational_diversity")


def write_equational_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    per_prompt_rows: list[dict[str, Any]],
    summary_fieldnames: list[str],
    prompt_fieldnames: list[str],
) -> None:
    """写出独立 ED 评测文件，避免覆盖常规 validation_summary。"""

    write_csv(output_dir / "equational_diversity_summary.csv", summary_rows, fieldnames=summary_fieldnames)
    write_json(output_dir / "equational_diversity_summary.json", summary_rows)
    write_markdown_table(output_dir / "equational_diversity_summary.md", summary_rows, fieldnames=summary_fieldnames)
    write_csv(output_dir / "equational_diversity_per_prompt.csv", per_prompt_rows, fieldnames=prompt_fieldnames)


def evaluate_equational_diversity(
    input_paths: list[str],
    output_dir: str | None = None,
    prompt_preview_chars: int = 120,
) -> Path:
    """读取 JSONL 并写出 ED 评测结果。"""

    records = load_jsonl_records(input_paths)
    resolved_output_dir = infer_output_dir(records, output_dir)
    summary_rows, per_prompt_rows, summary_fields, prompt_fields = evaluate_records(
        records=records,
        metrics=[METRIC_NAME],
        ks=[],
        correct_threshold=0.5,
        prompt_preview_chars=prompt_preview_chars,
    )
    write_equational_outputs(resolved_output_dir, summary_rows, per_prompt_rows, summary_fields, prompt_fields)
    return resolved_output_dir


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    output_dir = evaluate_equational_diversity(
        input_paths=args.input,
        output_dir=args.output_dir,
        prompt_preview_chars=args.prompt_preview_chars,
    )
    print(f"已写入 {output_dir / 'equational_diversity_summary.csv'}")
    print(f"已写入 {output_dir / 'equational_diversity_per_prompt.csv'}")


if __name__ == "__main__":
    main()
