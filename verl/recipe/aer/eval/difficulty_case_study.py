#!/usr/bin/env python3
"""按题目难度分组评估，并生成 case study 候选。

难度分组只依赖 baseline 的 per-prompt 正确率，避免用待评估方法本身定义难度：

- easy: baseline correct_rate >= easy_threshold
- medium: 0 < baseline correct_rate < easy_threshold
- hard: baseline correct_rate == 0

命令：
cd "/data/ruanruihan/adaptive exploration reward"
conda run -n aer python verl/recipe/aer/eval/difficulty_case_study.py \
  --baseline save/eval/baseline-naive-offpolicy/rerun_pass128_sharded/validation_per_prompt.csv \
  --experiment "semantic_embedding-tau0.065=save/eval/semantic_embedding-tau0.065-offpolicy/rerun_pass128_sharded/validation_per_prompt.csv" \
  --jsonl "baseline=save/eval/baseline-naive-offpolicy/rerun_pass128_sharded/*.jsonl" \
  --jsonl "semantic_embedding-tau0.065=save/eval/semantic_embedding-tau0.065-offpolicy/rerun_pass128_sharded/*.jsonl" \
  --output-dir save/eval/difficulty_case_study"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


DEFAULT_METRICS = [
    "correct_rate",
    "first@1",
    "pass@1",
    "pass@2",
    "pass@4",
    "pass@8",
    "distinct_2",
    "self_bleu4",
    "semantic_cosine",
]


def ensure_output_dir(path: str | Path) -> Path:
    """创建输出目录。"""

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prompt_hash(prompt: str) -> str:
    """生成稳定 prompt 短哈希。"""

    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def get_data_source(record: dict[str, Any]) -> str:
    """读取数据集名称；旧 JSONL 没有 data_source 时归为 all。"""

    for key in ("data_source", "dataset", "source"):
        value = record.get(key)
        if value:
            return str(value)
    return "all"


def format_cell(value: Any, digits: int = 6) -> str:
    """稳定格式化 CSV/Markdown 单元格。"""

    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in fieldnames})


def write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 Markdown 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(key)) for key in fieldnames) + " |\n")


def collect_jsonl_paths(inputs: list[str]) -> list[Path]:
    """收集 JSONL 文件，支持文件、目录和 glob。"""

    paths: list[Path] = []
    for item in inputs:
        matched = [Path(match) for match in glob.glob(item)] if any(ch in item for ch in "*?[]") else []
        candidates = matched if matched else [Path(item)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.rglob("*.jsonl")))
            elif candidate.is_file() and candidate.suffix == ".jsonl":
                paths.append(candidate)
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError(f"没有找到 JSONL 文件: {inputs}")
    return unique_paths


def load_jsonl_records(inputs: list[str]) -> list[dict[str, Any]]:
    """读取 JSONL。"""

    records: list[dict[str, Any]] = []
    for path in collect_jsonl_paths(inputs):
        try:
            file_step: int | None = int(path.stem)
        except ValueError:
            file_step = None
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
                record.setdefault("step", file_step)
                records.append(record)
    return records


def parse_named_path(raw: str) -> tuple[str, str]:
    """解析 name=path 形式参数。"""

    if "=" not in raw:
        raise ValueError(f"参数必须是 name=path 形式: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"参数必须是 name=path 形式: {raw}")
    return name, path


def parse_metrics(raw: str) -> list[str]:
    """解析指标列。"""

    if raw.strip().lower() == "default":
        return DEFAULT_METRICS
    return [item.strip() for item in raw.split(",") if item.strip()]


def to_float(value: Any) -> float | None:
    """宽松转换为 float。"""

    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def mean_ignore_none(values: list[float | None]) -> float | None:
    """忽略空值求均值。"""

    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV。"""

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_step(rows: list[dict[str, str]]) -> str | None:
    """返回 CSV 中最大的数字 step；若没有数字 step，返回 None。"""

    numeric_steps: list[tuple[int, str]] = []
    for row in rows:
        raw_step = str(row.get("step", "")).strip()
        if raw_step == "":
            continue
        try:
            numeric_steps.append((int(float(raw_step)), raw_step))
        except ValueError:
            continue
    if not numeric_steps:
        return None
    return max(numeric_steps, key=lambda item: item[0])[1]


def filter_step(rows: list[dict[str, str]], step: str | None) -> list[dict[str, str]]:
    """按 step 筛选；step 为 None 时不筛选。"""

    if step is None:
        return rows
    return [row for row in rows if str(row.get("step", "")).strip() == str(step)]


def prompt_key(row: dict[str, str]) -> tuple[str, str]:
    """per-prompt CSV 的稳定题目 key。"""

    return (str(row.get("data_source", "all")), str(row.get("prompt_hash") or row.get("prompt_id") or row.get("prompt")))


def difficulty_group(baseline_correct_rate: float | None, easy_threshold: float) -> str:
    """根据 baseline correct_rate 分组。"""

    value = baseline_correct_rate if baseline_correct_rate is not None else 0.0
    if value >= easy_threshold:
        return "easy"
    if value > 0:
        return "medium"
    return "hard"


def summarize_group(
    experiment: str,
    group_name: str,
    rows: list[dict[str, str]],
    metrics: list[str],
) -> dict[str, Any]:
    """聚合一个实验在一个难度组上的指标。"""

    summary: dict[str, Any] = {
        "experiment": experiment,
        "difficulty": group_name,
        "n_prompts": len(rows),
    }
    for metric in metrics:
        summary[metric] = mean_ignore_none([to_float(row.get(metric)) for row in rows])
    return summary


def add_delta_rows(summary_rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    """添加相对 baseline 的 delta 列。"""

    baseline_by_group: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        if row["experiment"] == "baseline":
            baseline_by_group[row["difficulty"]] = row

    rows_with_delta: list[dict[str, Any]] = []
    for row in summary_rows:
        result = dict(row)
        baseline = baseline_by_group.get(row["difficulty"])
        for metric in metrics:
            value = row.get(metric)
            base_value = baseline.get(metric) if baseline else None
            if value is None or base_value is None:
                result[f"delta_{metric}"] = None
            else:
                result[f"delta_{metric}"] = value - base_value
        rows_with_delta.append(result)
    return rows_with_delta


def group_raw_records(records: list[dict[str, Any]], step: str | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """按 data_source + prompt_hash 分组原始 JSONL。"""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if step is not None and str(record.get("step", "")).strip() != str(step):
            continue
        prompt = str(record.get("input", ""))
        groups[(get_data_source(record), prompt_hash(prompt))].append(record)
    return groups


def record_score(record: dict[str, Any]) -> float | None:
    """读取单条输出的正确性分数。"""

    for key in ("acc", "score", "reward"):
        value = to_float(record.get(key))
        if value is not None:
            return value
    return None


def is_correct(record: dict[str, Any], threshold: float) -> bool:
    """判断单条输出是否正确。"""

    score = record_score(record)
    return score is not None and score >= threshold


def compact_text(text: str, max_chars: int) -> str:
    """压缩过长文本，保留 Markdown 可读性。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def format_outputs(records: list[dict[str, Any]], threshold: float, max_outputs: int, max_chars: int) -> str:
    """格式化若干输出样本。"""

    if not records:
        return "_未提供原始 JSONL，无法展示输出。_"

    correct_records = [record for record in records if is_correct(record, threshold)]
    wrong_records = [record for record in records if not is_correct(record, threshold)]
    selected = correct_records[:max_outputs]
    if len(selected) < max_outputs:
        selected.extend(wrong_records[: max_outputs - len(selected)])

    blocks: list[str] = []
    for index, record in enumerate(selected, start=1):
        mark = "correct" if is_correct(record, threshold) else "wrong"
        score = record_score(record)
        score_text = "" if score is None else f", score={score:.3f}"
        output = compact_text(str(record.get("output", "")), max_chars=max_chars)
        blocks.append(f"{index}. [{mark}{score_text}]\n\n```text\n{output}\n```")
    return "\n\n".join(blocks)


def metric_delta(exp_row: dict[str, str], base_row: dict[str, str], metric: str) -> float | None:
    """计算单题指标差值。"""

    exp_value = to_float(exp_row.get(metric))
    base_value = to_float(base_row.get(metric))
    if exp_value is None or base_value is None:
        return None
    return exp_value - base_value


def choose_case_candidates(
    baseline_rows: dict[tuple[str, str], dict[str, str]],
    experiment_rows: dict[tuple[str, str], dict[str, str]],
    easy_threshold: float,
    limit_per_type: int,
) -> dict[str, list[tuple[tuple[str, str], dict[str, str], dict[str, str], float]]]:
    """挑选 case study 候选。"""

    buckets: dict[str, list[tuple[tuple[str, str], dict[str, str], dict[str, str], float]]] = defaultdict(list)
    common_keys = sorted(set(baseline_rows) & set(experiment_rows))

    for key in common_keys:
        base = baseline_rows[key]
        exp = experiment_rows[key]
        base_correct = to_float(base.get("correct_rate")) or 0.0
        exp_correct = to_float(exp.get("correct_rate")) or 0.0
        group = difficulty_group(base_correct, easy_threshold=easy_threshold)
        correct_delta = exp_correct - base_correct
        distinct_delta = metric_delta(exp, base, "distinct_2") or 0.0
        self_bleu_delta = metric_delta(exp, base, "self_bleu4") or 0.0
        semantic_delta = metric_delta(exp, base, "semantic_cosine") or 0.0
        diversity_gain = distinct_delta - self_bleu_delta - semantic_delta

        if group == "hard" and exp_correct > 0:
            buckets["hard_to_solved"].append((key, base, exp, exp_correct + diversity_gain))
        if group == "medium" and correct_delta > 0:
            buckets["medium_accuracy_gain"].append((key, base, exp, correct_delta + 0.2 * diversity_gain))
        if exp_correct >= max(base_correct - 0.01, 0.0) and diversity_gain > 0:
            buckets["diversity_without_accuracy_loss"].append((key, base, exp, diversity_gain))
        if correct_delta < -0.125:
            buckets["failure_or_tradeoff"].append((key, base, exp, -correct_delta + max(diversity_gain, 0.0)))

    selected: dict[str, list[tuple[tuple[str, str], dict[str, str], dict[str, str], float]]] = {}
    for bucket, items in buckets.items():
        selected[bucket] = sorted(items, key=lambda item: item[3], reverse=True)[:limit_per_type]
    return selected


def write_case_markdown(
    output_path: Path,
    primary_name: str,
    cases: dict[str, list[tuple[tuple[str, str], dict[str, str], dict[str, str], float]]],
    raw_groups: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    correct_threshold: float,
    max_outputs: int,
    max_chars: int,
) -> None:
    """写 case study 候选 Markdown。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Case Study Candidates\n\n")
        f.write(f"Primary experiment: `{primary_name}`\n\n")
        for bucket, items in cases.items():
            f.write(f"## {bucket}\n\n")
            if not items:
                f.write("_没有找到满足条件的候选。_\n\n")
                continue
            for case_index, (key, base, exp, score) in enumerate(items, start=1):
                prompt = exp.get("prompt") or base.get("prompt") or ""
                f.write(f"### {bucket} #{case_index}\n\n")
                f.write(f"- data_source: `{key[0]}`\n")
                f.write(f"- prompt_hash: `{key[1]}`\n")
                f.write(f"- candidate_score: `{score:.6f}`\n")
                f.write(f"- baseline correct_rate/pass@8/distinct_2/self_bleu4/semantic_cosine: `{base.get('correct_rate','')}` / `{base.get('pass@8','')}` / `{base.get('distinct_2','')}` / `{base.get('self_bleu4','')}` / `{base.get('semantic_cosine','')}`\n")
                f.write(f"- {primary_name} correct_rate/pass@8/distinct_2/self_bleu4/semantic_cosine: `{exp.get('correct_rate','')}` / `{exp.get('pass@8','')}` / `{exp.get('distinct_2','')}` / `{exp.get('self_bleu4','')}` / `{exp.get('semantic_cosine','')}`\n\n")
                f.write("Prompt preview:\n\n")
                f.write(f"```text\n{compact_text(prompt, max_chars=max_chars)}\n```\n\n")
                f.write("Baseline outputs:\n\n")
                f.write(format_outputs(raw_groups.get("baseline", {}).get(key, []), threshold=correct_threshold, max_outputs=max_outputs, max_chars=max_chars))
                f.write("\n\n")
                f.write(f"{primary_name} outputs:\n\n")
                f.write(format_outputs(raw_groups.get(primary_name, {}).get(key, []), threshold=correct_threshold, max_outputs=max_outputs, max_chars=max_chars))
                f.write("\n\n")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="AER 难度分组评估与 case study 候选生成")
    parser.add_argument("--baseline", required=True, help="baseline 的 validation_per_prompt.csv")
    parser.add_argument("--experiment", action="append", required=True, help="实验结果，格式 name=validation_per_prompt.csv；可重复")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--step", default="latest", help="评估 step；默认 latest。设为 all 表示不筛 step")
    parser.add_argument("--easy-threshold", type=float, default=0.75, help="baseline correct_rate 达到该值视为 easy")
    parser.add_argument("--metrics", default="default", help="聚合指标列，逗号分隔；默认 default")
    parser.add_argument("--primary", default="", help="用于 case study 的主实验名；默认取第一个 experiment")
    parser.add_argument("--jsonl", action="append", default=[], help="可选原始 JSONL，格式 name=路径/目录/glob；name 需为 baseline 或实验名")
    parser.add_argument("--correct-threshold", type=float, default=0.5, help="score/acc/reward 大于等于该值视为正确")
    parser.add_argument("--case-limit", type=int, default=3, help="每类 case study 最多输出多少个候选")
    parser.add_argument("--case-max-outputs", type=int, default=3, help="每个实验每道题最多展示多少条输出")
    parser.add_argument("--case-max-chars", type=int, default=1200, help="prompt/output 最多展示字符数")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""

    args = build_parser().parse_args(argv)
    output_dir = ensure_output_dir(args.output_dir)
    metrics = parse_metrics(args.metrics)

    baseline_all_rows = load_csv_rows(args.baseline)
    if args.step == "latest":
        step = latest_step(baseline_all_rows)
    elif args.step == "all":
        step = None
    else:
        step = args.step

    baseline_rows_list = filter_step(baseline_all_rows, step)
    baseline_rows = {prompt_key(row): row for row in baseline_rows_list}
    experiment_paths = [parse_named_path(raw) for raw in args.experiment]
    if not experiment_paths:
        raise ValueError("至少需要一个 --experiment")

    experiments: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for name, path in experiment_paths:
        rows = filter_step(load_csv_rows(path), step)
        experiments[name] = {prompt_key(row): row for row in rows}

    row_buckets: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for key, base_row in baseline_rows.items():
        group = difficulty_group(to_float(base_row.get("correct_rate")), easy_threshold=args.easy_threshold)
        row_buckets["baseline"][group].append(base_row)
        for name, rows_by_key in experiments.items():
            if key in rows_by_key:
                row_buckets[name][group].append(rows_by_key[key])

    summary_rows: list[dict[str, Any]] = []
    for experiment_name in ["baseline", *experiments.keys()]:
        for group in ["easy", "medium", "hard"]:
            summary_rows.append(summarize_group(experiment_name, group, row_buckets[experiment_name][group], metrics))

    summary_rows = add_delta_rows(summary_rows, metrics)
    fieldnames = ["experiment", "difficulty", "n_prompts", *metrics, *[f"delta_{metric}" for metric in metrics]]
    write_csv(output_dir / "difficulty_group_summary.csv", summary_rows, fieldnames=fieldnames)
    write_markdown_table(output_dir / "difficulty_group_summary.md", summary_rows, fieldnames=fieldnames)

    primary = args.primary or experiment_paths[0][0]
    if primary not in experiments:
        raise ValueError(f"--primary 不在 experiment 列表中: {primary}")

    raw_groups: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for raw in args.jsonl:
        name, path = parse_named_path(raw)
        raw_groups[name] = group_raw_records(load_jsonl_records([path]), step=step)

    cases = choose_case_candidates(
        baseline_rows=baseline_rows,
        experiment_rows=experiments[primary],
        easy_threshold=args.easy_threshold,
        limit_per_type=args.case_limit,
    )
    write_case_markdown(
        output_path=output_dir / "case_study_candidates.md",
        primary_name=primary,
        cases=cases,
        raw_groups=raw_groups,
        correct_threshold=args.correct_threshold,
        max_outputs=args.case_max_outputs,
        max_chars=args.case_max_chars,
    )

    print(f"使用 step: {step if step is not None else 'all'}")
    print(f"已写入 {output_dir / 'difficulty_group_summary.csv'}")
    print(f"已写入 {output_dir / 'difficulty_group_summary.md'}")
    print(f"已写入 {output_dir / 'case_study_candidates.md'}")


if __name__ == "__main__":
    main()
