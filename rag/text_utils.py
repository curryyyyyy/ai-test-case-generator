from __future__ import annotations

import re


# 连续汉字片段 / 其他词片段（汉字优先，以便单独按 bigram 处理）
_CJK_OR_WORD_RE = re.compile(r"[\u4e00-\u9fff]+|\w+")
_CJK_ONLY_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """统一的词法切分：非中文按词切分，连续中文按相邻二字组（bigram）切分。

    为什么中文要切 bigram：
    早期实现用 `\\w+|[\\u4e00-\\u9fff]`，而 `\\w` 本身就能匹配汉字，于是连续无空格的
    中文会被整体吞成一个超长 token（例如"登录支持短信验证码登录方式"整体算一个词），
    导致查询词"短信"永远无法命中——BM25 / 词重叠打分对中文事实上失效。

    此前 BM25 与 Rerank 各维护了一份同样的分词逻辑，行为不一致且缺陷相同，
    这里收敛为唯一实现，供检索（rag.retriever）与重排（rag.reranker）共用。

    Args:
        text: 待切分文本。

    Returns:
        token 列表（保留顺序与重复，便于统计词频）。
    """
    tokens: list[str] = []
    for segment in _CJK_OR_WORD_RE.findall(str(text).lower()):
        if _CJK_ONLY_RE.fullmatch(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


def tokenize_unique(text: str) -> set[str]:
    """去重后的 token 集合，供只需要判断重合度的场景（如 lite 重排）使用。"""
    return {token for token in tokenize(text) if token.strip()}
