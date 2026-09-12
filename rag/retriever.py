from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from rag.config import (
    BM25_TOP_K,
    ENABLE_RERANK,
    FETCH_K,
    HYBRID_SEARCH_ENABLED,
    MAX_CONTEXT_CHARS,
    MULTI_QUERY_ENABLED,
    PER_QUERY_TOP_K,
    QUERY_COUNT,
    RERANK_CANDIDATE_POOL,
    RERANK_FINAL_TOP_N,
    RERANK_MODE,
    RETRIEVER_TOP_K,
    SEARCH_TYPE,
)
from rag.schemas import Citation, RetrievedChunk


# 策略实现在 rag/strategy_defaults 中注册；此处 import 触发注册（必须保留）。
from rag import strategy_defaults as _strategy_defaults  # noqa: F401


@dataclass
class RetrievalMeta:
    expanded_queries: list[str]
    pre_dedup_count: int
    post_dedup_count: int
    rerank_mode: str
    rerank_enabled: bool
    rerank_latency_ms: int
    rerank_degraded: bool
    rerank_degraded_reason: str


def _build_where_filter(
    doc_id: str,
    doc_type: str,
    extra_filter: dict[str, str] | None = None,
) -> dict[str, Any]:
    merged_filter: dict[str, str] = {"doc_type": doc_type}
    if doc_type == "requirement":
        merged_filter["doc_id"] = doc_id
    if extra_filter:
        merged_filter.update(extra_filter)
    if len(merged_filter) == 1:
        where_filter: dict[str, Any] = merged_filter
    else:
        where_filter = {
            "$and": [{key: value} for key, value in merged_filter.items()]
        }
    return where_filter


def build_chunk(
    doc: Any,
    score: float,
    query_text: str,
    doc_id: str,
    doc_type: str,
) -> RetrievedChunk:
    metadata = getattr(doc, "metadata", {}) or {}
    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id", "")),
        doc_id=str(metadata.get("doc_id", doc_id)),
        doc_type=str(metadata.get("doc_type", doc_type)),
        source_name=str(metadata.get("source_name", "")),
        section_path=str(metadata.get("section_path", "ROOT")),
        text=str(getattr(doc, "page_content", "")),
        score=float(score),
        query=query_text,
    )


# 兼容旧引用（策略实现内部统一用 build_chunk）。
_build_chunk = build_chunk


# BM25 实现已迁至 rag/strategy_defaults.BM25Searcher。
# 缓存失效入口保留在此：rag/ingest.py 在语料写入后调用它。
_BM25_CACHE_LOCK = threading.Lock()


def invalidate_bm25_cache() -> None:
    """语料写入后调用，避免检索命中过期索引。"""
    from rag.strategy_defaults import DEFAULT_BM25_SEARCHER

    DEFAULT_BM25_SEARCHER.invalidate()


def _search_once(
    query_text: str,
    doc_id: str,
    doc_type: str,
    k: int,
    extra_filter: dict[str, str] | None = None,
    vector_strategy: str | None = None,
    sparse_strategy: str | None = None,
    fusion_strategy: str | None = None,
) -> list[RetrievedChunk]:
    """单条 query 的召回：向量 + 稀疏，命中多路时用融合策略合并。"""
    where_filter = _build_where_filter(
        doc_id=doc_id,
        doc_type=doc_type,
        extra_filter=extra_filter,
    )

    vector_searcher = _resolve_strategy(
        "vector",
        vector_strategy,
        search_type=SEARCH_TYPE,
        fetch_k=FETCH_K,
    )
    vector_results = vector_searcher.search(
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=k,
        where_filter=where_filter,
    )
    if not HYBRID_SEARCH_ENABLED:
        return vector_results

    sparse_searcher = _resolve_strategy("bm25", sparse_strategy)
    sparse_results = sparse_searcher.search(
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=max(k, BM25_TOP_K),
        where_filter=where_filter,
    )
    if not sparse_results:
        return vector_results

    fuser = _resolve_strategy("fusion", fusion_strategy)
    return fuser.fuse(
        result_sets=[vector_results, sparse_results],
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=max(k, BM25_TOP_K),
    )


def _resolve_strategy(kind: str, name: str | None, **kwargs: Any) -> Any:
    """按名字取策略；未指定时用注册表里的默认实现。

    策略可替换是这一步重构的核心：换 query 扩展器、换向量库、换融合算法，
    都只需要注册新实现并在这里传名字，检索主流程不用改动。
    """
    from rag import strategy_registry

    if name is None:
        available = strategy_registry.available(kind)
        if not available:
            raise ValueError(f"未注册任何 {kind} 策略")
        name = available[0]
    return strategy_registry.get(kind, name, **kwargs)


def _dedup_keep_best(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return _resolve_strategy("dedup", None).dedup(chunks)


def retrieve_context_with_meta(
    query: str,
    doc_id: str,
    doc_type: str = "requirement",
    top_k: int = 5,
    multi_query: bool | None = None,
    enable_rerank: bool | None = None,
    extra_filter: dict[str, str] | None = None,
    rerank_mode: str | None = None,
    expander_strategy: str | None = None,
    vector_strategy: str | None = None,
    sparse_strategy: str | None = None,
    fusion_strategy: str | None = None,
    reranker_strategy: str | None = None,
) -> tuple[list[RetrievedChunk], RetrievalMeta]:
    # 运行时指定的 rerank 模式优先，未指定时回落到 rag/config.py 的常量。
    # 不能去改全局常量：Streamlit 是多会话共享进程，改常量会跨会话串扰。
    effective_rerank_mode = RERANK_MODE if rerank_mode is None else rerank_mode
    query_text = str(query).strip()
    if not query_text:
        return (
            [],
            RetrievalMeta(
                expanded_queries=[],
                pre_dedup_count=0,
                post_dedup_count=0,
                rerank_mode=(
                    "disabled" if enable_rerank is False else effective_rerank_mode
                ),
                rerank_enabled=bool(enable_rerank) if enable_rerank is not None else ENABLE_RERANK,
                rerank_latency_ms=0,
                rerank_degraded=False,
                rerank_degraded_reason="",
            ),
        )

    final_top_k = top_k if top_k > 0 else RETRIEVER_TOP_K
    mq_enabled = MULTI_QUERY_ENABLED if multi_query is None else multi_query
    rerank_enabled = ENABLE_RERANK if enable_rerank is None else enable_rerank

    if mq_enabled:
        expander = _resolve_strategy("expander", expander_strategy)
        expanded_queries = expander.expand(query_text, max_queries=QUERY_COUNT)
    else:
        expanded_queries = [query_text]

    all_candidates: list[RetrievedChunk] = []
    per_query_k = PER_QUERY_TOP_K if mq_enabled else final_top_k
    for expanded in expanded_queries:
        all_candidates.extend(
            _search_once(
                query_text=expanded,
                doc_id=doc_id,
                doc_type=doc_type,
                k=per_query_k,
                extra_filter=extra_filter,
                vector_strategy=vector_strategy,
                sparse_strategy=sparse_strategy,
                fusion_strategy=fusion_strategy,
            )
        )

    pre_dedup_count = len(all_candidates)
    deduped = _dedup_keep_best(all_candidates)
    post_dedup_count = len(deduped)

    candidate_pool = deduped[:RERANK_CANDIDATE_POOL]
    rerank_strategy = _resolve_strategy("reranker", reranker_strategy)
    selected, rerank_meta = rerank_strategy.rerank(
        query=query_text,
        candidates=candidate_pool,
        top_n=min(RERANK_FINAL_TOP_N, final_top_k),
        enable_rerank=rerank_enabled,
        mode=effective_rerank_mode,
    )

    if not rerank_enabled:
        selected = deduped[:final_top_k]
    elif not selected:
        selected = deduped[:final_top_k]

    meta = RetrievalMeta(
        expanded_queries=expanded_queries,
        pre_dedup_count=pre_dedup_count,
        post_dedup_count=post_dedup_count,
        rerank_mode=rerank_meta["rerank_mode"],
        rerank_enabled=rerank_meta["rerank_enabled"],
        rerank_latency_ms=rerank_meta["rerank_latency_ms"],
        rerank_degraded=rerank_meta["rerank_degraded"],
        rerank_degraded_reason=rerank_meta["rerank_degraded_reason"],
    )
    return selected[:final_top_k], meta


# 说明：原这里是直调 rag.reranker.rerank 的逻辑，现已改为 reranker 策略。
# 若需要绕过策略，请向 strategy_registry 注册自定义 RerankStrategy，
# 而不是在检索主流程里加分支。


def retrieve_context(
    query: str,
    doc_id: str,
    doc_type: str = "requirement",
    top_k: int = 5,
) -> list[RetrievedChunk]:
    chunks, _meta = retrieve_context_with_meta(
        query=query,
        doc_id=doc_id,
        doc_type=doc_type,
        top_k=top_k,
    )
    return chunks


def retrieve_testcase_context_with_meta(
    query: str,
    top_k: int = 5,
    multi_query: bool | None = None,
    enable_rerank: bool | None = None,
    module: str = "",
    test_type: str = "",
    priority: str = "",
    rerank_mode: str | None = None,
    reranker_strategy: str | None = None,
) -> tuple[list[RetrievedChunk], RetrievalMeta]:
    extra_filter: dict[str, str] = {}
    if module:
        extra_filter["module"] = module
    if test_type:
        extra_filter["test_type"] = test_type
    if priority:
        extra_filter["priority"] = priority
    return retrieve_context_with_meta(
        query=query,
        doc_id="",
        doc_type="testcase",
        top_k=top_k,
        multi_query=multi_query,
        enable_rerank=enable_rerank,
        extra_filter=extra_filter if extra_filter else None,
        rerank_mode=rerank_mode,
        reranker_strategy=reranker_strategy,
    )


def format_retrieved_context(
    chunks: list[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if not chunks:
        return ""

    lines: list[str] = []
    current_len = 0
    for chunk in chunks:
        line = f"[{chunk.chunk_id}] [{chunk.section_path}] {chunk.text}".strip()
        if not line:
            continue
        next_len = current_len + len(line) + 1
        if next_len > max_chars:
            break
        lines.append(line)
        current_len = next_len
    return "\n".join(lines)


def build_citations(chunks: list[RetrievedChunk], limit: int = 5) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for chunk in chunks[:limit]:
        citation = Citation(
            chunk_id=chunk.chunk_id,
            section_path=chunk.section_path,
            source_name=chunk.source_name,
            score=chunk.score,
        )
        citations.append(citation.model_dump())
    return citations
