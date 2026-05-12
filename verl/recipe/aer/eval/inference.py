"""从模型重新推理并生成可评测 JSONL。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io_utils import write_json
from .math_verify_utils import verify_math_response


def parse_val_files(raw_files: list[str]) -> list[Path]:
    """解析验证集路径参数。

    支持两种形式：
    1. 多个路径：`--val-files a.parquet b.parquet`
    2. Hydra 风格列表字符串：`--val-files "['a.parquet','b.parquet']"`
    """

    if len(raw_files) == 1:
        raw = raw_files[0].strip()
        if raw.startswith("[") and raw.endswith("]"):
            import ast

            return [Path(item) for item in ast.literal_eval(raw)]
    return [Path(item) for item in raw_files]


def normalize_chat_prompt(prompt: Any) -> list[dict[str, str]] | str:
    """把 parquet 中的 prompt 字段转为 chat messages 或纯字符串。"""

    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict):
                messages.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
        if messages:
            return messages
    return str(prompt)


def is_valid_id(value: Any) -> bool:
    """判断 parquet 字段是否是可用题目标识。"""

    if value is None:
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null"}


def build_example_id(row, prompt: list[dict[str, str]] | str) -> str:
    """为模型重推理构造稳定且尽量紧凑的题目 ID。"""

    unique_id = row.get("unique_id", None)
    if is_valid_id(unique_id):
        return str(unique_id)

    data_source = str(row.get("data_source", "unknown"))
    extra_info = row.get("extra_info", {}) or {}
    if isinstance(extra_info, dict) and is_valid_id(extra_info.get("index", None)):
        split = str(extra_info.get("split", "test"))
        return f"{data_source}:{split}:{extra_info['index']}"

    return json.dumps(prompt, ensure_ascii=False, sort_keys=True)


def prompt_to_text(prompt: list[dict[str, str]] | str, tokenizer=None) -> str:
    """将 chat prompt 转成模型可直接生成的文本。"""

    if isinstance(prompt, str):
        return prompt
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    # tokenizer 不支持 chat_template 时使用保守拼接，保证脚本仍可运行。
    rendered = []
    for message in prompt:
        rendered.append(f"{message['role']}\n{message['content']}")
    rendered.append("assistant\n")
    return "\n".join(rendered)


def load_eval_dataset(
    val_files: list[str],
    prompt_key: str = "prompt",
    limit: int | None = None,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """读取验证 parquet，并按题目去重。

    `test_repeated.parquet` 或手工拼接文件可能包含重复题目。这里默认按
    `unique_id`、`extra_info.index` 或 prompt 文本去重，避免 Pass@K 的
    rollout 次数被重复文件意外放大。
    """

    import pandas as pd

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in parse_val_files(val_files):
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            prompt = normalize_chat_prompt(row[prompt_key])
            dedup_key = build_example_id(row, prompt)
            if deduplicate and dedup_key in seen:
                continue
            seen.add(dedup_key)

            reward_model = row.get("reward_model", {}) or {}
            ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
            examples.append(
                {
                    "unique_id": dedup_key,
                    "data_source": str(row.get("data_source", "unknown")),
                    "prompt": prompt,
                    "ground_truth": str(ground_truth),
                }
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def shard_examples(examples: list[dict[str, Any]], shard_index: int = 0, num_shards: int = 1) -> list[dict[str, Any]]:
    """按题目维度做确定性分片，用于多进程单卡并行评测。"""

    if num_shards <= 0:
        raise ValueError("--num-shards 必须大于 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index 必须满足 0 <= shard_index < num_shards")
    if num_shards == 1:
        return examples
    return [example for idx, example in enumerate(examples) if idx % num_shards == shard_index]


def build_vllm_output_generator(
    model_path: str,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    trust_remote_code: bool,
    seed: int | None,
) -> Callable[[list[str]], list[list[str]]]:
    """加载一次 vLLM，并返回可重复调用的批量生成函数。"""

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    llm = LLM(**llm_kwargs)
    sampling_kwargs: dict[str, Any] = {
        "n": samples_per_prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k if top_k > 0 else -1,
    }
    if seed is not None:
        sampling_kwargs["seed"] = seed
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        # 旧版 vLLM 可能不支持 seed 参数，此时保留其它采样参数。
        sampling_kwargs.pop("seed", None)
        sampling_params = SamplingParams(**sampling_kwargs)

    def generate_batch(prompts: list[str]) -> list[list[str]]:
        outputs = llm.generate(prompts, sampling_params)
        return [[candidate.text for candidate in request_output.outputs] for request_output in outputs]

    return generate_batch


def generate_with_vllm(
    model_path: str,
    prompts: list[str],
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    trust_remote_code: bool,
    seed: int | None,
) -> list[list[str]]:
    """使用 vLLM 批量生成；适合大模型和较大 Pass@K。"""

    generate_batch = build_vllm_output_generator(
        model_path=model_path,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    return generate_batch(prompts)


def build_hf_output_generator(
    model_path: str,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    dtype: str,
    trust_remote_code: bool,
    seed: int | None,
) -> Callable[[list[str]], list[list[str]]]:
    """加载一次 HuggingFace 模型，并返回可重复调用的批量生成函数。

    这是 vLLM 不可用时的兼容后端。为了节省显存，按展开后的样本维度分批
    生成，即每个 prompt 重复 `samples_per_prompt` 次。
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    if seed is not None:
        try:
            from transformers import set_seed

            set_seed(seed)
        except Exception:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if top_k > 0:
        generation_kwargs["top_k"] = top_k
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    def generate_batch(prompts: list[str]) -> list[list[str]]:
        expanded: list[tuple[int, str]] = []
        for prompt_idx, prompt in enumerate(prompts):
            for _ in range(samples_per_prompt):
                expanded.append((prompt_idx, prompt))

        grouped_outputs: list[list[str]] = [[] for _ in prompts]
        for start in range(0, len(expanded), batch_size):
            batch = expanded[start : start + batch_size]
            batch_prompts = [item[1] for item in batch]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            # 对 decoder-only 模型，generate 返回的是 padding 后输入 + 新 token。
            # 因此切掉统一的 padding 后输入长度，而不是每行有效长度。
            prompt_token_width = encoded["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(**encoded, **generation_kwargs)

            for row_idx, (prompt_idx, _) in enumerate(batch):
                output_ids = generated[row_idx, prompt_token_width:]
                grouped_outputs[prompt_idx].append(tokenizer.decode(output_ids, skip_special_tokens=True))

        return grouped_outputs

    return generate_batch


def generate_with_hf(
    model_path: str,
    prompts: list[str],
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    dtype: str,
    trust_remote_code: bool,
    seed: int | None,
) -> list[list[str]]:
    """使用 HuggingFace transformers 生成。"""

    generate_batch = build_hf_output_generator(
        model_path=model_path,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        batch_size=batch_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    return generate_batch(prompts)


def build_output_generator(
    model_path: str,
    samples_per_prompt: int,
    backend: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    trust_remote_code: bool,
    seed: int | None,
) -> tuple[str, Callable[[list[str]], list[list[str]]]]:
    """根据 backend 加载模型，并返回实际后端名与批量生成函数。"""

    selected_backend = backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            selected_backend = "vllm"
        except ImportError:
            selected_backend = "hf"

    if selected_backend == "vllm":
        return selected_backend, build_vllm_output_generator(
            model_path=model_path,
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    if selected_backend == "hf":
        return selected_backend, build_hf_output_generator(
            model_path=model_path,
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            batch_size=batch_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    raise ValueError(f"未知推理后端: {backend}")


def generate_outputs(
    model_path: str,
    prompt_texts: list[str],
    samples_per_prompt: int,
    backend: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    trust_remote_code: bool,
    seed: int | None,
) -> list[list[str]]:
    """根据 backend 调用对应推理实现。"""

    _, generate_batch = build_output_generator(
        model_path=model_path,
        samples_per_prompt=samples_per_prompt,
        backend=backend,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    return generate_batch(prompt_texts)


ROLL_OUT_RESUME_KEYS = [
    "model_path",
    "val_files",
    "prompt_key",
    "limit",
    "deduplicate",
    "step",
    "samples_per_prompt",
    "backend",
    "batch_size",
    "tensor_parallel_size",
    "dtype",
    "trust_remote_code",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "gpu_memory_utilization",
    "vllm_max_model_len",
    "vllm_max_num_seqs",
    "vllm_max_num_batched_tokens",
    "num_shards",
    "shard_index",
]


def load_rollout_metadata(path: str | Path) -> dict[str, Any] | None:
    """读取 rollout 元信息；不存在时返回 None。"""

    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_rollout_metadata(path: str | Path, metadata: dict[str, Any] | None) -> None:
    """续跑前校验已有 rollout 是否由兼容参数生成。"""

    if metadata is None:
        return
    path = Path(path)
    if not path.exists():
        return
    existing_metadata = load_rollout_metadata(path)
    if existing_metadata is None:
        print(f"发现已有 rollout 但没有元信息文件，将仅按 unique_id/sample_index 续跑: {path}")
        return

    mismatches = []
    for key in ROLL_OUT_RESUME_KEYS:
        if key in existing_metadata and existing_metadata.get(key) != metadata.get(key):
            mismatches.append((key, existing_metadata.get(key), metadata.get(key)))
    if not mismatches:
        return

    details = "; ".join(f"{key}: existing={old!r}, current={new!r}" for key, old, new in mismatches[:8])
    raise ValueError(f"已有 rollout 元信息与当前推理参数不一致，为避免混用旧结果已停止续跑。请更换 --output-dir 或确认后删除旧 JSONL。差异: {details}")


def normalize_sample_index(value: Any) -> int | None:
    """解析样本下标，非法值返回 None。"""

    try:
        sample_index = int(value)
    except (TypeError, ValueError):
        return None
    return sample_index if sample_index >= 0 else None


def rollout_record_key(record: dict[str, Any]) -> tuple[str, int] | None:
    """返回用于续跑去重的记录键。"""

    unique_id = record.get("unique_id")
    sample_index = normalize_sample_index(record.get("sample_index"))
    if not is_valid_id(unique_id) or sample_index is None:
        return None
    return str(unique_id), sample_index


def load_existing_rollout_records(path: str | Path) -> list[dict[str, Any]]:
    """读取已有 rollout JSONL。"""

    path = Path(path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("rb") as f:
        line_no = 0
        while True:
            line_start = f.tell()
            raw_line = f.readline()
            if not raw_line:
                break
            line_no += 1
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                remaining = f.read()
                if remaining.strip():
                    raise ValueError(f"{path}:{line_no} 不是合法 UTF-8，无法安全续跑: {exc}") from exc
                with path.open("rb+") as repair_file:
                    repair_file.truncate(line_start)
                    repair_file.flush()
                    os.fsync(repair_file.fileno())
                print(f"检测到 rollout 末尾存在未写完的 UTF-8 行，已截断到上一条完整记录: {path}")
                break
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                remaining = f.read()
                if remaining.strip():
                    raise ValueError(f"{path}:{line_no} 不是合法 JSON，无法安全续跑: {exc}") from exc
                with path.open("rb+") as repair_file:
                    repair_file.truncate(line_start)
                    repair_file.flush()
                    os.fsync(repair_file.fileno())
                print(f"检测到 rollout 末尾存在未写完的 JSON 行，已截断到上一条完整记录: {path}")
                break
    return records


def select_current_rollout_records(
    existing_records: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    samples_per_prompt: int,
) -> tuple[list[dict[str, Any]], dict[str, set[int]], int, int]:
    """从已有 JSONL 中筛出当前分片需要的记录，并返回已完成样本下标。"""

    expected_ids = {str(example["unique_id"]) for example in examples}
    seen_keys: set[tuple[str, int]] = set()
    done_samples: dict[str, set[int]] = {unique_id: set() for unique_id in expected_ids}
    current_records: list[dict[str, Any]] = []
    ignored_count = 0
    duplicate_count = 0

    for record in existing_records:
        key = rollout_record_key(record)
        if key is None:
            ignored_count += 1
            continue
        unique_id, sample_index = key
        if unique_id not in expected_ids or sample_index >= samples_per_prompt:
            ignored_count += 1
            continue
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)
        done_samples[unique_id].add(sample_index)
        current_records.append(record)

    return current_records, done_samples, ignored_count, duplicate_count


def missing_sample_indices(example: dict[str, Any], done_samples: dict[str, set[int]], samples_per_prompt: int) -> list[int]:
    """返回某道题仍需补齐的 sample_index。"""

    unique_id = str(example["unique_id"])
    done = done_samples.get(unique_id, set())
    return [sample_index for sample_index in range(samples_per_prompt) if sample_index not in done]


def append_rollout_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """追加写入一批 rollout，并 fsync 确保批次落盘。"""

    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as check_file:
            check_file.seek(-1, os.SEEK_END)
            needs_leading_newline = check_file.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_leading_newline:
            f.write("\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_rollout_records_for_example(
    args,
    example: dict[str, Any],
    prompt_text: str,
    outputs: list[str],
    required_sample_indices: set[int],
) -> list[dict[str, Any]]:
    """将单题输出转为 rollout 记录，只保留续跑缺失的样本。"""

    if len(outputs) < args.samples_per_prompt:
        raise ValueError(f"模型只为题目 {example['unique_id']} 生成了 {len(outputs)} 个样本，少于期望的 {args.samples_per_prompt} 个")

    records: list[dict[str, Any]] = []
    for sample_idx, output in enumerate(outputs[: args.samples_per_prompt]):
        if sample_idx not in required_sample_indices:
            continue
        score_info = verify_math_response(output, example["ground_truth"])
        records.append(
            {
                "step": args.step,
                "data_source": example["data_source"],
                "unique_id": example["unique_id"],
                "sample_index": sample_idx,
                "input": prompt_text,
                "output": output,
                "ground_truth": example["ground_truth"],
                **score_info,
            }
        )
    return records


def rollout_and_verify(args, rollout_path: str | Path | None = None, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """读取验证集、调用模型推理，并用 math_verify 打分。

    传入 `rollout_path` 时启用增量写入和断点续跑：每个题目批次生成后立即追加到
    JSONL；重新运行时按 `unique_id + sample_index` 跳过已有样本。
    """

    examples = load_eval_dataset(
        val_files=args.val_files,
        prompt_key=args.prompt_key,
        limit=args.limit,
        deduplicate=not args.no_deduplicate,
    )
    examples = shard_examples(
        examples=examples,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not examples:
        raise ValueError("验证集为空，无法推理评测")

    if rollout_path is not None:
        validate_rollout_metadata(rollout_path, metadata)
        existing_records = load_existing_rollout_records(rollout_path)
        records, done_samples, ignored_count, duplicate_count = select_current_rollout_records(
            existing_records=existing_records,
            examples=examples,
            samples_per_prompt=args.samples_per_prompt,
        )
        if ignored_count or duplicate_count:
            raise ValueError(
                f"已有 rollout 文件包含 {ignored_count} 条不属于当前分片/参数的记录和 {duplicate_count} 条重复样本。"
                "由于后续会通过 eval_from_jsonl.py 读取整个 JSONL，为避免评测混入异常记录，请更换 --output-dir 或清理旧文件后重跑。"
            )
        expected_records = len(examples) * args.samples_per_prompt
        completed_prompts = sum(1 for example in examples if not missing_sample_indices(example, done_samples, args.samples_per_prompt))
        if existing_records:
            print(f"发现已有 rollout: {rollout_path}")
            print(f"当前分片已完成 {completed_prompts}/{len(examples)} 道题，已可复用 {len(records)}/{expected_records} 条样本")

        todo_examples = [example for example in examples if missing_sample_indices(example, done_samples, args.samples_per_prompt)]
        if not todo_examples:
            if metadata is not None:
                write_json(Path(rollout_path).with_suffix(".meta.json"), metadata)
            return records

        if metadata is not None:
            write_json(Path(rollout_path).with_suffix(".meta.json"), metadata)

        prompt_batch_size = int(getattr(args, "rollout_save_batch_size", 8))
        if prompt_batch_size <= 0:
            raise ValueError("--rollout-save-batch-size 必须大于 0")

        # 先加载 tokenizer 渲染 chat_template，确保 vLLM 与 HF 后端使用同一 prompt 格式。
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        todo_prompt_texts = [prompt_to_text(example["prompt"], tokenizer=tokenizer) for example in todo_examples]
        selected_backend, generate_batch = build_output_generator(
            model_path=args.model_path,
            samples_per_prompt=args.samples_per_prompt,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
        )
        print(f"继续推理 {len(todo_examples)} 道题，后端: {selected_backend}，每 {prompt_batch_size} 道题追加保存一次")

        total_batches = (len(todo_examples) + prompt_batch_size - 1) // prompt_batch_size
        for batch_idx, start in enumerate(range(0, len(todo_examples), prompt_batch_size), start=1):
            batch_examples = todo_examples[start : start + prompt_batch_size]
            batch_prompt_texts = todo_prompt_texts[start : start + prompt_batch_size]
            grouped_outputs = generate_batch(batch_prompt_texts)
            if len(grouped_outputs) != len(batch_examples):
                raise ValueError(f"模型返回了 {len(grouped_outputs)} 组输出，但当前批次有 {len(batch_examples)} 道题")

            batch_records: list[dict[str, Any]] = []
            for example, prompt_text, outputs in zip(batch_examples, batch_prompt_texts, grouped_outputs):
                unique_id = str(example["unique_id"])
                required_indices = set(missing_sample_indices(example, done_samples, args.samples_per_prompt))
                example_records = build_rollout_records_for_example(
                    args=args,
                    example=example,
                    prompt_text=prompt_text,
                    outputs=outputs,
                    required_sample_indices=required_indices,
                )
                batch_records.extend(example_records)
                done_samples.setdefault(unique_id, set()).update(record["sample_index"] for record in example_records)

            append_rollout_jsonl(rollout_path, batch_records)
            records.extend(batch_records)
            print(f"已保存 rollout 批次 {batch_idx}/{total_batches}: 新增 {len(batch_records)} 条，累计 {len(records)}/{expected_records} 条")

        if len(records) != expected_records:
            raise ValueError(f"推理结束后 rollout 仍未补齐: 当前 {len(records)} 条，期望 {expected_records} 条")
        return records

    # 先加载 tokenizer 渲染 chat_template，确保 vLLM 与 HF 后端使用同一 prompt 格式。
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    prompt_texts = [prompt_to_text(example["prompt"], tokenizer=tokenizer) for example in examples]
    grouped_outputs = generate_outputs(
        model_path=args.model_path,
        prompt_texts=prompt_texts,
        samples_per_prompt=args.samples_per_prompt,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )

    records: list[dict[str, Any]] = []
    for example, prompt_text, outputs in zip(examples, prompt_texts, grouped_outputs):
        for sample_idx, output in enumerate(outputs):
            score_info = verify_math_response(output, example["ground_truth"])
            records.append(
                {
                    "step": args.step,
                    "data_source": example["data_source"],
                    "unique_id": example["unique_id"],
                    "sample_index": sample_idx,
                    "input": prompt_text,
                    "output": output,
                    "ground_truth": example["ground_truth"],
                    **score_info,
                }
            )
    return records


def write_rollout_jsonl(path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """写出重新推理得到的逐样本 JSONL 和元信息。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(path.with_suffix(".meta.json"), metadata)
