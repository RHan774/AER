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
import torch

from typing import Any, Dict, Union
from collections import defaultdict
from verl import DataProto
from math_verify import parse, verify

# add: 导入相似度计算模块
from .similarity import get_similarity_computer


def _valid_eval_id(value: Any) -> str | None:
    """把数据字段转成可写入评测 JSONL 的题目标识。"""

    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _build_prompt_id(non_tensor_batch: dict[str, Any], data_source: str, prompt_str: str) -> str:
    """优先使用数据集 ID，其次使用 extra_info，最后用 prompt 哈希兜底。"""

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

    prompt_digest = hashlib.sha1(prompt_str.encode("utf-8")).hexdigest()[:12]
    return f"{data_source}:prompt:{prompt_digest}"


def math_accuracy_reward(solution: str, golden_answer: str) -> Dict[str, float | str]:
    """Reward function that checks whether the answer is equivalent to the golden answer."""
    extracted_golden_answer = parse("\\boxed{" + golden_answer + "}")
    if len(extracted_golden_answer) == 0:
        print(f"fail to extract golden answer {golden_answer}")
        reward = 0.0
        result = {
            "score": reward,
            "acc": reward,
            "pred": ""
        }
        return result
    
    extracted_answer = parse(solution[-512:])
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


def compute_score(data_source: str, solution: str, ground_truth: str) -> Dict[str, float | str]:
    return math_accuracy_reward(solution, ground_truth)


class RLRewardManager:
    def __init__(self, tokenizer, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key


    def __call__(self, data: DataProto, return_dict: bool = True, num_examine: int = 1) -> Union[torch.Tensor, Dict[str, Any]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            # 评测脚本需要数据集来源和标准答案来做分数据集统计与错误分析。
            reward_extra_info["data_source"].append(str(data_source))
            reward_extra_info["ground_truth"].append(str(ground_truth))
            prompt_id = _build_prompt_id(data_item.non_tensor_batch, str(data_source), prompt_str)
            reward_extra_info["prompt_id"].append(prompt_id)
            reward_extra_info["unique_id"].append(prompt_id)

            score = compute_score(
                data_source=data_source,
                solution=response_str,
                ground_truth=ground_truth,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[data source]", data_source)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

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


class AERRewardManager:
    # add: 添加相似度算法参数支持
    def __init__(
        self,
        tokenizer,
        reward_fn_key="data_source",
        similarity_algorithm="token_match",
        similarity_params=None
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
        """
        self.tokenizer = tokenizer
        self.reward_fn_key = reward_fn_key
        self.similarity_algorithm = similarity_algorithm
        self.similarity_params = similarity_params or {}

        # add: 创建相似度计算器
        self.similarity_computer = get_similarity_computer(
            similarity_algorithm,
            **self.similarity_params
        )


    def __call__(self, data: DataProto, return_dict: bool = True, num_examine: int = 1) -> Union[torch.Tensor, Dict[str, Any]]:
        reward_tensor_acc = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor_exploration = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # modify: 使用配置的相似度计算算法
        # 原代码：similarity = compute_token_similarity(data)
        # 新代码：支持多种相似度算法，通过 similarity_algorithm 参数选择
        # add: 传递 tokenizer 以支持需要解码文本的算法（如 char_ngram, levenshtein, tfidf_cosine, semantic_embedding）
        similarity = self.similarity_computer.compute(data, tokenizer=self.tokenizer)
        similarity = similarity.sum(-1)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            # 评测脚本需要数据集来源和标准答案来做分数据集统计与错误分析。
            reward_extra_info["data_source"].append(str(data_source))
            reward_extra_info["ground_truth"].append(str(ground_truth))
            prompt_id = _build_prompt_id(data_item.non_tensor_batch, str(data_source), prompt_str)
            reward_extra_info["prompt_id"].append(prompt_id)
            reward_extra_info["unique_id"].append(prompt_id)

            score = compute_score(
                data_source=data_source,
                solution=response_str,
                ground_truth=ground_truth,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
                
            exploration_reward = 1.0/similarity[i].item() if (similarity[i].item() > 0) else 0.0
            
            reward_tensor_acc[i, valid_response_length - 1] = reward
            reward_tensor_exploration[i, valid_response_length - 1] = exploration_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < num_examine:
                already_print_data_sources[data_source] += 1
                print("[data source]", data_source)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor_acc": reward_tensor_acc,
                "reward_tensor_exploration": reward_tensor_exploration,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor_acc
