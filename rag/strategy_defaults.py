"""检索策略的默认实现注册。

把现有实现（规则式 query 扩展、Chroma 向量召回、自建 BM25、RRF 融合、
按 chunk_id 去重、reranker 包装）适配成 strategies.py 里的协议，
并注册到 strategy_registry。

适配层刻意保持极薄：实际逻辑仍在 query_expander / retriever / reranker
里，避免重构过程中行为发生变化。
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

from rag.config import (
    BM25_TOP_K,
    ENABLE_RERANK,
    RERANK_CROSS_ENCODER_LOCAL_FILES_ONLY,
    RERANK_CROSS_ENCODER_MODEL,
    RERANK_MODE,
    RERANK_TIMEOUT_MS,
    RRF_K,
)
from rag.query_expander import expand_query as _expand_query
from rag.reranker import rerank as _rerank
from rag.schemas import RetrievedChunk
from rag.strategy_registry import register
from rag.text_utils import tokenize
from rag.store import get_vector_store


# ---------------------------------------------------------------- query 扩展


class RuleBasedExpander:
    """现有规则式扩展（同义词替换 + 意图补充）。"""

    name = "rule"

    def expand(self, query: str, max_queries: int) -> list[str]:
        return _expand_query(query, max_queries=max_queries)


# ---------------------------------------------------------------- 向量召回


class ChromaSearcher:
    """基于 Chroma 的向量召回（MMR 或相似度）。"""

    name = "chroma"

    def __init__(self, search_type: str = "mmr", fetch_k: int = 20) -> None:
        self.search_type = search_type
        self.fetch_k = fetch_k

    def search(
        self,
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
        where_filter: dict[str, Any],
    ) -> list[RetrievedChunk]:
        from rag.retriever import build_chunk

        vector_store = get_vector_store()
        if self.search_type == "mmr":
            docs = vector_store.max_marginal_relevance_search(
                query=query_text,
                k=k,
                fetch_k=self.fetch_k,
                filter=where_filter,
            )
            raw_results: list[tuple[Any, float]] = [(doc, 0.0) for doc in docs]
        else:
            raw_results = vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=k,
                filter=where_filter,
            )
        return [
            build_chunk(doc, score, query_text, doc_id, doc_type)
            for doc, score in raw_results
        ]


# ---------------------------------------------------------------- BM25 召回


class BM25Searcher:
    """自建倒排索引的 BM25 召回，索引按 where 条件缓存。"""

    name = "bm25"

    K1 = 1.5
    B = 0.75
    CACHE_TTL_SECONDS = 300.0

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _build_index(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        tokenized = [tokenize(str(text)) for text in documents]
        doc_lengths = [len(tokens) for tokens in tokenized]
        doc_count = len(doc_lengths)
        total_len = sum(doc_lengths)
        if doc_count == 0 or total_len <= 0:
            return None

        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, tokens in enumerate(tokenized):
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            for token, freq in term_freq.items():
                postings.setdefault(token, []).append((doc_index, freq))

        return {
            "documents": list(documents),
            "metadatas": list(metadatas),
            "doc_lengths": doc_lengths,
            "avg_doc_len": total_len / doc_count,
            "postings": postings,
            "doc_freq": {term: len(items) for term, items in postings.items()},
            "created_at": time.time(),
        }

    def _get_index(self, where_filter: dict[str, Any]) -> dict[str, Any] | None:
        cache_key = json.dumps(where_filter, sort_keys=True, default=str)
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None and now - cached["created_at"] < self.CACHE_TTL_SECONDS:
                return cached

        # 构建索引是重活，刻意不持锁，避免阻塞并发查询。
        payload = get_vector_store().get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        if not documents or not metadatas:
            with self._lock:
                self._cache.pop(cache_key, None)
            return None

        index = self._build_index(documents, metadatas)
        with self._lock:
            if index is None:
                self._cache.pop(cache_key, None)
            else:
                self._cache[cache_key] = index
        return index

    def _score(self, index: dict[str, Any], query_terms: list[str]) -> dict[int, float]:
        doc_count = len(index["doc_lengths"])
        scores: dict[int, float] = {}
        for term in query_terms:
            items = index["postings"].get(term)
            if not items:
                continue
            df = index["doc_freq"].get(term, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for doc_index, freq in items:
                doc_len = index["doc_lengths"][doc_index]
                numerator = freq * (self.K1 + 1)
                denominator = freq + self.K1 * (
                    1 - self.B + self.B * doc_len / index["avg_doc_len"]
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * numerator / denominator
        return scores

    def search(
        self,
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
        where_filter: dict[str, Any],
    ) -> list[RetrievedChunk]:
        from rag.retriever import build_chunk

        query_terms = tokenize(query_text)
        if not query_terms:
            return []

        index = self._get_index(where_filter)
        if index is None:
            return []

        ranked = sorted(
            self._score(index, query_terms).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[RetrievedChunk] = []
        for doc_index, score in ranked[:k]:
            if score <= 0:
                continue
            doc = _BM25Doc(
                page_content=str(index["documents"][doc_index]),
                metadata=index["metadatas"][doc_index] or {},
            )
            results.append(build_chunk(doc, score, query_text, doc_id, doc_type))
        return results


class _BM25Doc:
    """供 build_chunk 消费的轻量文档对象。"""

    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


# ---------------------------------------------------------------- 融合


class RRFFuser:
    """Reciprocal Rank Fusion：按名次倒数求和融合多路结果。"""

    name = "rrf"

    def __init__(self, rrf_k: int = RRF_K) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        result_sets: list[list[RetrievedChunk]],
        query_text: str,
        doc_id: str,
        doc_type: str,
        k: int,
    ) -> list[RetrievedChunk]:
        fused_scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        for result_set in result_sets:
            for rank, chunk in enumerate(result_set, start=1):
                fused_scores[chunk.chunk_id] = fused_scores.get(
                    chunk.chunk_id, 0.0
                ) + 1.0 / (self.rrf_k + rank)
                chunks.setdefault(chunk.chunk_id, chunk)

        ranked = sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )
        return [
            _replace_chunk(
                chunks[chunk_id],
                score=float(score),
                query=query_text,
                doc_id=doc_id,
                doc_type=doc_type,
            )
            for chunk_id, score in ranked[:k]
        ]


def _replace_chunk(
    chunk: RetrievedChunk,
    score: float,
    query: str,
    doc_id: str,
    doc_type: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id or doc_id,
        doc_type=chunk.doc_type or doc_type,
        source_name=chunk.source_name,
        section_path=chunk.section_path,
        text=chunk.text,
        score=score,
        query=query,
    )


# ---------------------------------------------------------------- 去重


class ChunkIdDeduper:
    """按 chunk_id 去重，保留得分最高的一条。"""

    name = "chunk_id"

    def dedup(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        best: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
        return sorted(best.values(), key=lambda item: item.score, reverse=True)


# ---------------------------------------------------------------- 重排


class RerankerStrategy:
    """包装 rag.reranker.rerank，统一成协议返回 (结果, 元信息)。"""

    name = "default"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
        **options: Any,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        result = _rerank(
            query=query,
            candidates=candidates,
            enable_rerank=options.get("enable_rerank", ENABLE_RERANK),
            mode=options.get("mode", RERANK_MODE),
            cross_encoder_model=options.get(
                "cross_encoder_model", RERANK_CROSS_ENCODER_MODEL
            ),
            cross_encoder_local_files_only=options.get(
                "cross_encoder_local_files_only",
                RERANK_CROSS_ENCODER_LOCAL_FILES_ONLY,
            ),
            timeout_ms=options.get("timeout_ms", RERANK_TIMEOUT_MS),
            final_top_n=top_n,
        )
        return result.items, {
            "rerank_mode": result.rerank_mode,
            "rerank_enabled": result.rerank_enabled,
            "rerank_latency_ms": result.rerank_latency_ms,
            "rerank_degraded": result.degraded,
            "rerank_degraded_reason": result.degraded_reason,
        }


# ---------------------------------------------------------------- 注册


def register_defaults() -> None:
    """注册所有默认策略（幂等，重复调用会覆盖同名项）。"""
    register("expander", "rule", RuleBasedExpander)
    register("vector", "chroma", ChromaSearcher)
    register("bm25", "bm25", BM25Searcher)
    register("fusion", "rrf", RRFFuser)
    register("dedup", "chunk_id", ChunkIdDeduper)
    register("reranker", "default", RerankerStrategy)


register_defaults()


# BM25 缓存失效：ingest 写入语料后需要清缓存，
# 注册表按名字拿到的可能是新实例，所以这里保留一个模块级默认实例供复用。
DEFAULT_BM25_SEARCHER = BM25Searcher()
