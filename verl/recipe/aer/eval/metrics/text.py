"""文本切分与 n-gram 工具。

评测指标共用这一套轻量分词规则，保证 Distinct-2 与 Self-BLEU 的
token 口径一致。这里不依赖外部分词器，避免评测脚本受模型 tokenizer
变化影响。
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")


def tokenize(text: str) -> list[str]:
    """将文本切成英文/数字片段、中文单字和数学符号。

    该规则对数学推理输出比较稳健：LaTeX 命令、数字、符号不会被全部丢弃；
    同时没有外部依赖，便于在训练服务器上直接复现。
    """

    return TOKEN_RE.findall(text or "")


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """生成 n-gram；当文本长度不足时返回空列表。"""

    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
