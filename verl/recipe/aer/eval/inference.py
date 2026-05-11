"""从模型重新推理并生成可评测 JSONL。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    trust_remote_code: bool,
    seed: int | None,
) -> list[list[str]]:
    """使用 vLLM 批量生成；适合大模型和较大 Pass@K。"""

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
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
    outputs = llm.generate(prompts, sampling_params)
    return [[candidate.text for candidate in request_output.outputs] for request_output in outputs]


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
    """使用 HuggingFace transformers 生成。

    这是 vLLM 不可用时的兼容后端。为了节省显存，按展开后的样本维度分批
    生成，即每个 prompt 重复 `samples_per_prompt` 次。
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    expanded: list[tuple[int, str]] = []
    for prompt_idx, prompt in enumerate(prompts):
        for _ in range(samples_per_prompt):
            expanded.append((prompt_idx, prompt))

    grouped_outputs: list[list[str]] = [[] for _ in prompts]
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
    trust_remote_code: bool,
    seed: int | None,
) -> list[list[str]]:
    """根据 backend 调用对应推理实现。"""

    selected_backend = backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            selected_backend = "vllm"
        except ImportError:
            selected_backend = "hf"

    if selected_backend == "vllm":
        return generate_with_vllm(
            model_path=model_path,
            prompts=prompt_texts,
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    if selected_backend == "hf":
        return generate_with_hf(
            model_path=model_path,
            prompts=prompt_texts,
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


def rollout_and_verify(args) -> list[dict[str, Any]]:
    """读取验证集、调用模型推理，并用 math_verify 打分。"""

    examples = load_eval_dataset(
        val_files=args.val_files,
        prompt_key=args.prompt_key,
        limit=args.limit,
        deduplicate=not args.no_deduplicate,
    )
    if not examples:
        raise ValueError("验证集为空，无法推理评测")

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
