"""检索链路各步骤的策略接口。

重构前 retriever.py 把「扩展 query / 向量召回 / BM25 召回 / 融合 / 去重 / 重排」
全写在一个模块里，每一步都是模块级私有函数并用 import 直连具体实现，
想换掉任意一环（模型式 query 扩展、换向量库、换融合算法）都得动 retriever 本体。

这里把每步抽成 Protocol + 注册表：retriever 只依赖接口，
替换实现只需 register_* + 传 strategy 名，不用改检索主流程。

七个步骤与对应协议（默认实现见 defaults.py）：

    expander    -> QueryExpander      多 query 扩展
    vector      -> VectorSearcher     向量召回
    bm25        -> SparseSearcher     稀疏/BM25 召回
    fusion      -> ResultFuser        多路结果融合
    dedup       -> ResultDeduper      去重
    reranker    -> RerankStrategy     重排
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rag.schemas import RetrievedChunk


@runtime_checkable
class QueryExpander(Protocol):
    """把一个 query 扩展成多个等价 query（用于提升召回率）。"""

    name: str

    def expand(self, query: str, max_queries: int) -> list[str]:
        ...


@runtime_checkable
class VectorSearcher(Protocol):
    """向量（语义）召回。"""

    name: str

    def search(
        self,
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
        where_filter: dict[str, Any],
    ) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class SparseSearcher(Protocol):
    """稀疏（关键词/BM25）召回。"""

    name: str

    def search(
        self,
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
        where_filter: dict[str, Any],
    ) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class ResultFuser(Protocol):
    """把多路召回结果融合成一路（默认 RRF）。"""

    name: str

    def fuse(
        self,
        result_sets: list[list[RetrievedChunk]],
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
    ) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class ResultDeduper(Protocol):
    """去重（同一 chunk 可能被多路召回命中）。"""

    name: str

    def dedup(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class RerankStrategy(Protocol):
    """重排，返回 (结果, 元信息字典)。"""

    name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
        **options: Any,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        ...
