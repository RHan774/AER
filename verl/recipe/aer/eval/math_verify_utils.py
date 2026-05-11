"""数学答案验证工具。"""

from __future__ import annotations

from typing import Any


def verify_math_response(solution: str, ground_truth: str) -> dict[str, Any]:
    """用 math_verify 判断模型输出是否匹配标准答案。

    与训练 reward 保持同一逻辑：标准答案外层补 `\\boxed{}`，模型输出只取
    末尾 512 字符进行解析，以降低长 CoT 中间内容对最终答案抽取的干扰。
    """

    from math_verify import parse, verify

    extracted_golden_answer = parse("\\boxed{" + str(ground_truth) + "}")
    if len(extracted_golden_answer) == 0:
        return {"score": 0.0, "acc": 0.0, "pred": ""}

    extracted_answer = parse(str(solution)[-512:])
    if len(extracted_answer) == 0:
        return {"score": 0.0, "acc": 0.0, "pred": ""}

    reward = float(verify(extracted_golden_answer[0], extracted_answer[0]))
    return {"score": reward, "acc": reward, "pred": str(extracted_answer[-1])}
