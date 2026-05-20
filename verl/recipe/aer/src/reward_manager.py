# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Union

import torch
from math_verify import parse, verify

from verl import DataProto

# add: 导入相似度计算模块
from .similarity import get_similarity_computer, list_available_algorithms

_SIMILARITY_CACHE_KEY = "_aer_similarity_cache"
_TEXT_DECODING_SIMILARITY_ALGORITHMS = {
    "char_ngram",
    "levenshtein",
    "tfidf_cosine",
    "semantic_embedding",
    "compression_ratio",
    "rouge_l",
}
_SEMANTIC_EMBEDDING_EXPLORATION_SCALE = 10.0


@lru_cache(maxsize=65536)
def _parse_golden_answer_cached(golden_answer: str):
    """缓存标准答案解析结果；同一道题会在多次 rollout/epoch 中重复出现。"""

    return parse("\\boxed{" + golden_answer + "}")


class _CachedDecodeTokenizer:
    """在一次 reward 计算内缓存 tokenizer.decode 结果。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache: dict[tuple[tuple[int, ...], tuple[Any, ...], tuple[tuple[str, Any], ...]], str] = {}

    def decode(self, token_ids, *args, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_key = tuple(int(token_id) for token_id in token_ids.detach().cpu().tolist())
        else:
            token_key = tuple(int(token_id) for token_id in token_ids)
        key = (token_key, tuple(repr(arg) for arg in args), tuple(sorted((name, repr(value)) for name, value in kwargs.items())))
        decoded = self.cache.get(key)
        if decoded is None:
            decoded = self.tokenizer.decode(list(token_key), *args, **kwargs)
            self.cache[key] = decoded
        return decoded


def _needs_decode_cache(main_algorithm: str, active_metric_algorithms) -> bool:
    """只有实际运行文本类相似度时才启用 decode 缓存。"""

    if main_algorithm in _TEXT_DECODING_SIMILARITY_ALGORITHMS:
        return True
    return any(algorithm in _TEXT_DECODING_SIMILARITY_ALGORITHMS for algorithm in active_metric_algorithms)


def _valid_eval_id(value: Any) -> str | None:
    """把数据字段转成可写入评测 JSONL 的题目标识。"""

    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _build_prompt_id_from_lazy_prompt(non_tensor_batch: dict[str, Any], data_source: str, get_prompt_str) -> str:
    """优先用数据字段生成 ID；只有兜底哈希时才解码 prompt。"""

    for key in ("unique_id", "id"):
        value = _valid_eval_id(non_tensor_batch.get(key))
        if value is not None:
            return value

    extra_info = non_tensor_batch.get("extra_info", {}) or {}
    if isinstance(extra_info, dict):
        index = _valid_eval_id(extra_info.get("index"))
        if index is not None:
            split = _valid_eval_id(extra_info.get("split")) or "test"
            return f"{data_source}:{split}:{index}"

    prompt_digest = hashlib.sha1(get_prompt_str().encode("utf-8")).hexdigest()[:12]
    return f"{data_source}:prompt:{prompt_digest}"


def _build_prompt_id(non_tensor_batch: dict[str, Any], data_source: str, prompt_str: str) -> str:
    """优先使用数据集 ID，其次使用 extra_info，最后用 prompt 哈希兜底。"""

    return _build_prompt_id_from_lazy_prompt(non_tensor_batch, data_source, lambda: prompt_str)


def _get_non_tensor_item(non_tensor_batch: dict[str, Any], key: str, idx: int, default: Any = None) -> Any:
    """从 non_tensor_batch 中按行读取单个字段。"""

    if key not in non_tensor_batch:
        return default
    return non_tensor_batch[key][idx]


def _build_prompt_id_from_batch(non_tensor_batch: dict[str, Any], idx: int, data_source: str, get_prompt_str) -> str:
    """不构造 DataProtoItem，直接从 batch 级非 tensor 字段生成 prompt id。"""

    for key in ("unique_id", "id"):
        value = _valid_eval_id(_get_non_tensor_item(non_tensor_batch, key, idx))
        if value is not None:
            return value

    extra_info = _get_non_tensor_item(non_tensor_batch, "extra_info", idx, {}) or {}
    if isinstance(extra_info, dict):
        index = _valid_eval_id(extra_info.get("index"))
        if index is not None:
            split = _valid_eval_id(extra_info.get("split")) or "test"
            return f"{data_source}:{split}:{index}"

    prompt_digest = hashlib.sha1(get_prompt_str().encode("utf-8")).hexdigest()[:12]
    return f"{data_source}:prompt:{prompt_digest}"


def _get_ground_truth_from_batch(non_tensor_batch: dict[str, Any], idx: int) -> Any:
    """读取第 idx 条样本的标准答案。"""

    reward_model = _get_non_tensor_item(non_tensor_batch, "reward_model", idx)
    return reward_model["ground_truth"]


def _prepare_reward_batch_views(data: DataProto):
    """预取 reward 循环会反复访问的 batch 视图和有效长度。"""

    prompts = data.batch["prompts"]
    responses = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]
    prompt_length = prompts.shape[-1]
    valid_prompt_lengths = attention_mask[:, :prompt_length].sum(dim=1).to(dtype=torch.long).tolist()
    valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=1).to(dtype=torch.long).tolist()
    return prompts, responses, prompt_length, valid_prompt_lengths, valid_response_lengths


def _clear_similarity_batch_cache(data: DataProto) -> None:
    """清理只在 reward 计算期间使用的相似度预处理缓存。"""

    meta_info = getattr(data, "meta_info", None)
    if isinstance(meta_info, dict):
        meta_info.pop(_SIMILARITY_CACHE_KEY, None)
    if hasattr(data, _SIMILARITY_CACHE_KEY):
        try:
            delattr(data, _SIMILARITY_CACHE_KEY)
        except Exception:
            pass


def _score_with_extracted_answer(
    golden_answer: str,
    extracted_golden_answer,
    extracted_answer,
) -> Dict[str, float | str]:
    """使用已解析的标准答案和预测答案计算数学正确性奖励。"""

    if len(extracted_golden_answer) == 0:
        print(f"fail to extract golden answer {golden_answer}")
        reward = 0.0
        result = {
            "score": reward,
            "acc": reward,
            "pred": ""
        }
        return result

    if len(extracted_answer) == 0:
        # print(f"fail to extract answer: {solution}")
        reward = 0.0
        result = {
            "score": reward,
            "acc": reward,
            "pred": "",
        }
        return result
    # Reward 1 if the answer is equivalent to the golden answer, 0 otherwise.
    reward = float(verify(extracted_golden_answer[0], extracted_answer[0]))
    result = {
        "score": reward,
        "acc": reward,
        "pred": str(extracted_answer[-1]),
    }
    return result


def _score_with_extracted_golden_answer(
    solution: str,
    golden_answer: str,
    extracted_golden_answer,
) -> Dict[str, float | str]:
    """使用已解析的标准答案计算数学正确性奖励。"""

    return _score_with_extracted_answer(
        golden_answer,
        extracted_golden_answer,
        parse(solution[-512:]),
    )


def _score_math_response_with_cached_golden_answer(
    solution: str,
    ground_truth: str,
) -> Dict[str, float | str]:
    """复用标准答案解析结果，response 仍按原逻辑逐条解析。"""

    extracted_golden_answer = _parse_golden_answer_cached(ground_truth)
    return _score_with_extracted_golden_answer(solution, ground_truth, extracted_golden_answer)


def math_accuracy_reward(solution: str, golden_answer: str) -> Dict[str, float | str]:
    """Reward function that checks whether the answer is equivalent to the golden answer."""
    extracted_golden_answer = _parse_golden_answer_cached(golden_answer)
    return _score_with_extracted_golden_answer(solution, golden_answer, extracted_golden_answer)


def compute_score(data_source: str, solution: str, ground_truth: str) -> Dict[str, float | str]:
    return math_accuracy_reward(solution, ground_truth)


_DEFAULT_COMPUTE_SCORE = compute_score


def _compute_score_with_optional_cache(
    data_source: str,
    solution: str,
    ground_truth: str,
) -> Dict[str, float | str]:
    """默认数学奖励走缓存；若外部替换 compute_score，则保留原扩展行为。"""

    if compute_score is _DEFAULT_COMPUTE_SCORE:
        return _score_math_response_with_cached_golden_answer(solution, ground_truth)
    return compute_score(data_source=data_source, solution=solution, ground_truth=ground_truth)


class RLRewardManager:
    def __init__(self, tokenizer, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key


    def __call__(self, data: DataProto, return_dict: bool = True, num_examine: int = 1) -> Union[torch.Tensor, Dict[str, Any]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        tokenizer = self.tokenizer
        prompts, responses, prompt_length, valid_prompt_lengths, valid_response_lengths = _prepare_reward_batch_views(data)
        non_tensor_batch = data.non_tensor_batch
        rewards = []

        for i in range(len(data)):
            prompt_ids = prompts[i]
            valid_prompt_length = valid_prompt_lengths[i]
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str: str | None = None

            def get_prompt_str(valid_prompt_ids=valid_prompt_ids) -> str:
                nonlocal prompt_str
                if prompt_str is None:
                    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                return prompt_str

            response_ids = responses[i]
            valid_response_length = valid_response_lengths[i]
            valid_response_ids = response_ids[:valid_response_length]

            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = _get_ground_truth_from_batch(non_tensor_batch, i)
            data_source = _get_non_tensor_item(non_tensor_batch, self.reward_fn_key, i)
            # 评测脚本需要数据集来源和标准答案来做分数据集统计与错误分析。
            reward_extra_info["data_source"].append(str(data_source))
            reward_extra_info["ground_truth"].append(str(ground_truth))
            prompt_id = _build_prompt_id_from_batch(non_tensor_batch, i, str(data_source), get_prompt_str)
            reward_extra_info["prompt_id"].append(prompt_id)
            reward_extra_info["unique_id"].append(prompt_id)

            score = _compute_score_with_optional_cache(data_source, response_str, ground_truth)

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(float(reward))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[data source]", data_source)
                print("[prompt]", get_prompt_str())
                print("[response]", response_str)
                print("[ground truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if rewards:
            row_index = torch.arange(len(rewards), device=reward_tensor.device, dtype=torch.long)
            col_index = torch.as_tensor(valid_response_lengths, device=reward_tensor.device, dtype=torch.long) - 1
            reward_tensor[row_index, col_index] = torch.as_tensor(rewards, device=reward_tensor.device, dtype=reward_tensor.dtype)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


def compute_token_similarity(data: DataProto) -> torch.Tensor:
    """
    原 Token 精确匹配相似度计算函数（保留用于向后兼容）

    # modify: 此函数已被 similarity.token_match.TokenMatchSimilarity 类替代
    # 新代码建议使用 get_similarity_computer("token_match") 或相似度模块中的其他算法
    # 保留此函数以确保向后兼容性

    计算基于 token 精确匹配的相似度矩阵
    """
    return get_similarity_computer("token_match").compute(data)


def _to_plain_config(value: Any) -> Any:
    """把 OmegaConf 配置转换成普通 Python 容器，便于传给各算法。"""

    if hasattr(value, "items"):
        return {key: _to_plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ListConfig":
        return [_to_plain_config(item) for item in value]
    return value


def _normalize_metric_algorithms(algorithms: Any) -> list[str]:
    """规范化需要额外记录探索奖励指标的算法列表。"""

    if algorithms is None:
        return []
    if isinstance(algorithms, str):
        algorithms = algorithms.strip()
        if algorithms.startswith("[") and algorithms.endswith("]"):
            algorithms = algorithms[1:-1]
        raw_algorithms = [item.strip().strip("\"'") for item in algorithms.split(",") if item.strip()]
    else:
        raw_algorithms = [str(item).strip() for item in algorithms if str(item).strip()]

    normalized: list[str] = []
    for algorithm in raw_algorithms:
        if algorithm.lower() == "all":
            normalized.extend(list_available_algorithms())
        else:
            normalized.append(algorithm)

    deduped: list[str] = []
    seen = set()
    for algorithm in normalized:
        if algorithm not in seen:
            deduped.append(algorithm)
            seen.add(algorithm)
    return deduped


def _normalize_metric_delay_fraction(value: Any) -> float:
    """规范化额外指标延后计算比例。"""

    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("exploration_metric_delay_fraction must be in (0, 1]")
    return fraction


def _compute_exploration_values(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """由相似度矩阵计算每条 response 的探索奖励标量。"""

    similarity_sum = similarity_matrix.sum(-1)
    exploration_values = torch.zeros_like(similarity_sum)
    positive_mask = similarity_sum > 0
    exploration_values[positive_mask] = similarity_sum[positive_mask].reciprocal()
    return exploration_values


def _scale_selected_exploration_values(algorithm: str, exploration_values: torch.Tensor) -> torch.Tensor:
    """对被选中的相似度算法应用探索奖励倍率。"""

    if algorithm == "semantic_embedding":
        return exploration_values * _SEMANTIC_EMBEDDING_EXPLORATION_SCALE
    return exploration_values


class AERRewardManager:
    # add: 添加相似度算法参数支持
    def __init__(
        self,
        tokenizer,
        reward_fn_key="data_source",
        similarity_algorithm="token_match",
        similarity_params=None,
        exploration_metric_algorithms=None,
        exploration_metric_delayed_algorithms=None,
        exploration_metric_delay_fraction: float = 1.0,
    ):
        """
        AER 奖励管理器，支持多种相似度计算算法

        Args:
            tokenizer: 分词器
            reward_fn_key: 奖励函数键
            similarity_algorithm: 相似度算法名称，可选：
                - "token_match": Token 精确匹配（默认，原有方法）
                - "ngram_overlap": N-gram 重叠度
                - "char_ngram": 字符级 N-gram
                - "levenshtein": 编辑距离
                - "tfidf_cosine": TF-IDF 余弦相似度
                - "semantic_embedding": 语义嵌入相似度
                - "simhash": SimHash 近重复相似度
            similarity_params: 算法特定参数字典
            exploration_metric_algorithms: 额外记录探索奖励指标的算法列表；设为 ["all"] 时使用所有已注册算法
            exploration_metric_delayed_algorithms: 只在最后阶段记录的额外指标算法列表；不会影响主训练奖励算法
            exploration_metric_delay_fraction: 最后多少比例训练步开始记录延后指标，1.0 表示全程记录
        """
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key
        self.similarity_algorithm = similarity_algorithm
        self.similarity_params = _to_plain_config(similarity_params or {})
        self.exploration_metric_algorithms = _normalize_metric_algorithms(exploration_metric_algorithms)
        self.exploration_metric_delayed_algorithms = set(
            _normalize_metric_algorithms(exploration_metric_delayed_algorithms)
        )
        self.exploration_metric_delay_fraction = _normalize_metric_delay_fraction(exploration_metric_delay_fraction)

        # add: 创建相似度计算器
        self.exploration_metric_computers = {
            algorithm: get_similarity_computer(algorithm, **self.similarity_params)
            for algorithm in self.exploration_metric_algorithms
        }
        if similarity_algorithm in self.exploration_metric_computers:
            self.similarity_computer = self.exploration_metric_computers[similarity_algorithm]
        else:
            self.similarity_computer = get_similarity_computer(similarity_algorithm, **self.similarity_params)

    def _should_compute_exploration_metric(
        self,
        algorithm: str,
        current_step: int | None,
        total_training_steps: int | None,
    ) -> bool:
        """判断额外指标当前 step 是否需要计算。"""

        if algorithm == self.similarity_algorithm:
            return True
        if algorithm not in self.exploration_metric_delayed_algorithms:
            return True
        if current_step is None or total_training_steps is None:
            return True

        total_steps = int(total_training_steps)
        if total_steps <= 0:
            return True

        final_steps = max(1, math.ceil(total_steps * self.exploration_metric_delay_fraction))
        first_delayed_step = max(1, total_steps - final_steps + 1)
        return int(current_step) >= first_delayed_step

    def _get_active_exploration_metric_computers(
        self,
        current_step: int | None,
        total_training_steps: int | None,
    ):
        """返回当前 step 需要实际计算的额外指标。"""

        return {
            algorithm: computer
            for algorithm, computer in self.exploration_metric_computers.items()
            if self._should_compute_exploration_metric(algorithm, current_step, total_training_steps)
        }

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = True,
        num_examine: int = 1,
        current_step: int | None = None,
        total_training_steps: int | None = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        reward_tensor_acc = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_exploration = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        if current_step is None or total_training_steps is None:
            meta_info = getattr(data, "meta_info", {}) or {}
            current_step = meta_info.get("_aer_global_step", current_step)
            total_training_steps = meta_info.get("_aer_total_training_steps", total_training_steps)
        active_metric_computers = self._get_active_exploration_metric_computers(
            current_step=current_step,
            total_training_steps=total_training_steps,
        )
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        tokenizer = _CachedDecodeTokenizer(self.tokenizer) if _needs_decode_cache(self.similarity_algorithm, active_metric_computers) else self.tokenizer
        prompts, responses, prompt_length, valid_prompt_lengths, valid_response_lengths = _prepare_reward_batch_views(data)
        non_tensor_batch = data.non_tensor_batch
        acc_rewards = []

        exploration_values_by_algorithm = {
            algorithm: _compute_exploration_values(computer.compute(data, tokenizer=tokenizer))
            for algorithm, computer in active_metric_computers.items()
        }
        if self.similarity_algorithm in exploration_values_by_algorithm:
            exploration_values = exploration_values_by_algorithm[self.similarity_algorithm]
        else:
            exploration_values = _compute_exploration_values(
                self.similarity_computer.compute(data, tokenizer=tokenizer)
            )
        exploration_values = _scale_selected_exploration_values(self.similarity_algorithm, exploration_values)
        if self.similarity_algorithm in exploration_values_by_algorithm:
            exploration_values_by_algorithm[self.similarity_algorithm] = exploration_values
        reward_exploration_metrics_by_algorithm = {
            algorithm: algorithm_values.mean().item()
            for algorithm, algorithm_values in exploration_values_by_algorithm.items()
        }

        for i in range(len(data)):
            prompt_ids = prompts[i]
            valid_prompt_length = valid_prompt_lengths[i]
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str: str | None = None

            def get_prompt_str(valid_prompt_ids=valid_prompt_ids) -> str:
                nonlocal prompt_str
                if prompt_str is None:
                    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                return prompt_str

            response_ids = responses[i]
            valid_response_length = valid_response_lengths[i]
            valid_response_ids = response_ids[:valid_response_length]

            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = _get_ground_truth_from_batch(non_tensor_batch, i)
            data_source = _get_non_tensor_item(non_tensor_batch, self.reward_fn_key, i)
            # 评测脚本需要数据集来源和标准答案来做分数据集统计与错误分析。
            reward_extra_info["data_source"].append(str(data_source))
            reward_extra_info["ground_truth"].append(str(ground_truth))
            prompt_id = _build_prompt_id_from_batch(non_tensor_batch, i, str(data_source), get_prompt_str)
            reward_extra_info["prompt_id"].append(prompt_id)
            reward_extra_info["unique_id"].append(prompt_id)

            score = _compute_score_with_optional_cache(data_source, response_str, ground_truth)

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
                
            acc_rewards.append(float(reward))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[data source]", data_source)
                print("[prompt]", get_prompt_str())
                print("[response]", response_str)
                print("[ground truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if acc_rewards:
            row_index = torch.arange(len(acc_rewards), device=reward_tensor_acc.device, dtype=torch.long)
            col_index = torch.as_tensor(valid_response_lengths, device=reward_tensor_acc.device, dtype=torch.long) - 1
            reward_tensor_acc[row_index, col_index] = torch.as_tensor(acc_rewards, device=reward_tensor_acc.device, dtype=reward_tensor_acc.dtype)
            reward_tensor_exploration[row_index, col_index] = exploration_values.to(device=reward_tensor_exploration.device, dtype=reward_tensor_exploration.dtype)

        if return_dict:
            _clear_similarity_batch_cache(data)
            return {
                "reward_tensor_acc": reward_tensor_acc,
                "reward_tensor_exploration": reward_tensor_exploration,
                "reward_exploration_metrics_by_algorithm": reward_exploration_metrics_by_algorithm,
                "reward_extra_info": reward_extra_info,
            }
        else:
            _clear_similarity_batch_cache(data)
            return reward_tensor_acc
