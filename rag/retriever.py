from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import threading
import time
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
    RRF_K,
    RERANK_CANDIDATE_POOL,
    RERANK_CROSS_ENCODER_LOCAL_FILES_ONLY,
    RERANK_CROSS_ENCODER_MODEL,
    RERANK_FINAL_TOP_N,
    RERANK_MODE,
    RERANK_TIMEOUT_MS,
    RETRIEVER_TOP_K,
    SEARCH_TYPE,
)
from rag.query_expander import expand_query
from rag.reranker import rerank
from rag.schemas import Citation, RetrievedChunk
from rag.store import get_vector_store


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


def _build_chunk(
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


def _vector_search_once(
    query_text: str,
    doc_id: str,
    doc_type: str,
    k: int,
    extra_filter: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    vector_store = get_vector_store()
    raw_results: list[tuple[Any, float]]
    where_filter = _build_where_filter(
        doc_id=doc_id,
        doc_type=doc_type,
        extra_filter=extra_filter,
    )

    if SEARCH_TYPE == "mmr":
        docs = vector_store.max_marginal_relevance_search(
            query=query_text,
            k=k,
            fetch_k=FETCH_K,
            filter=where_filter,
        )
        raw_results = [(doc, 0.0) for doc in docs]
    else:
        raw_results = vector_store.similarity_search_with_relevance_scores(
            query=query_text,
            k=k,
            filter=where_filter,
        )

    results: list[RetrievedChunk] = []
    for doc, score in raw_results:
        results.append(_build_chunk(doc, score, query_text, doc_id, doc_type))
    return results


# 连续汉字片段 / 其他词片段（汉字优先，以便单独按 bigram 处理）
_CJK_OR_WORD_RE = re.compile(r"[\u4e00-\u9fff]+|\w+")
_CJK_ONLY_RE = re.compile(r"[\u4e00-\u9fff]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    """BM25 分词：非中文按词切分，连续中文按相邻二字组（bigram）切分。

    原实现用 `\\w+|[\\u4e00-\\u9fff]`，由于 `\\w` 本身匹配汉字，连续无空格的中文
    会被整体吞成一个超长 token，例如“登录支持短信验证码登录方式”整体作为一个词，
    导致查询词“短信”永远无法命中——BM25 对中文事实上失效，混合检索里的
    BM25 分支等于空转。这里改为对中文生成 bigram，使子串查询可以命中。
    """
    tokens: list[str] = []
    for segment in _CJK_OR_WORD_RE.findall(text.lower()):
        if _CJK_ONLY_RE.fullmatch(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


BM25_K1 = 1.5
BM25_B = 0.75
# BM25 索引缓存有效期（秒）。语料发生写入时会由 ingest 主动失效，
# TTL 只作为兜底，防止极端情况下长期命中过期索引。
BM25_CACHE_TTL_SECONDS = 300.0


@dataclass
class _BM25Document:
    """供 _build_chunk 消费的轻量文档对象。"""

    page_content: str
    metadata: dict[str, Any]


@dataclass
class _BM25Index:
    documents: list[str]
    metadatas: list[dict[str, Any]]
    doc_lengths: list[int]
    avg_doc_len: float
    # 倒排索引：term -> [(doc_index, term_freq)]
    postings: dict[str, list[tuple[int, int]]]
    doc_freq: dict[str, int]
    created_at: float


# 缓存键为过滤条件，不同 doc_id / doc_type / 额外过滤各自持有独立索引。
_BM25_CACHE: dict[str, _BM25Index] = {}
_BM25_CACHE_LOCK = threading.Lock()


def invalidate_bm25_cache() -> None:
    """语料写入后调用，避免检索命中过期索引。"""
    with _BM25_CACHE_LOCK:
        _BM25_CACHE.clear()


def _build_bm25_index(
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> _BM25Index | None:
    tokenized_docs = [_tokenize_for_bm25(str(text)) for text in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    doc_count = len(doc_lengths)
    if doc_count == 0:
        return None

    total_len = sum(doc_lengths)
    if total_len <= 0:
        return None

    postings: dict[str, list[tuple[int, int]]] = {}
    for doc_index, tokens in enumerate(tokenized_docs):
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        for token, freq in term_freq.items():
            postings.setdefault(token, []).append((doc_index, freq))

    return _BM25Index(
        documents=list(documents),
        metadatas=list(metadatas),
        doc_lengths=doc_lengths,
        avg_doc_len=total_len / doc_count,
        postings=postings,
        doc_freq={term: len(items) for term, items in postings.items()},
        created_at=time.time(),
    )


def _get_bm25_index(where_filter: dict[str, Any]) -> _BM25Index | None:
    """获取（必要时构建）BM25 索引。

    原实现每次查询都全量拉语料并重新分词，一次完整流水线会触发约 30 次
    全库扫描；这里改为复用缓存索引，仅对命中的倒排表打分。
    """
    cache_key = json.dumps(where_filter, sort_keys=True, default=str)
    now = time.time()

    with _BM25_CACHE_LOCK:
        cached = _BM25_CACHE.get(cache_key)
        if cached is not None and now - cached.created_at < BM25_CACHE_TTL_SECONDS:
            return cached

    # 构建索引是重活，刻意不持锁，避免阻塞并发查询。
    vector_store = get_vector_store()
    payload = vector_store.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )
    documents = payload.get("documents", []) or []
    metadatas = payload.get("metadatas", []) or []
    if not documents or not metadatas:
        with _BM25_CACHE_LOCK:
            _BM25_CACHE.pop(cache_key, None)
        return None

    index = _build_bm25_index(documents, metadatas)
    with _BM25_CACHE_LOCK:
        if index is None:
            _BM25_CACHE.pop(cache_key, None)
        else:
            _BM25_CACHE[cache_key] = index
    return index


def _score_bm25_index(
    index: _BM25Index,
    query_terms: list[str],
) -> dict[int, float]:
    doc_count = len(index.doc_lengths)
    scores: dict[int, float] = {}
    for term in query_terms:
        items = index.postings.get(term)
        if not items:
            continue
        df = index.doc_freq.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for doc_index, freq in items:
            doc_len = index.doc_lengths[doc_index]
            numerator = freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (
                1 - BM25_B + BM25_B * doc_len / index.avg_doc_len
            )
            scores[doc_index] = scores.get(doc_index, 0.0) + idf * numerator / denominator
    return scores


def _bm25_search_once(
    query_text: str,
    doc_id: str,
    doc_type: str,
    k: int,
    extra_filter: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    query_terms = _tokenize_for_bm25(query_text)
    if not query_terms:
        return []

    where_filter = _build_where_filter(
        doc_id=doc_id,
        doc_type=doc_type,
        extra_filter=extra_filter,
    )
    index = _get_bm25_index(where_filter)
    if index is None:
        return []

    scores = _score_bm25_index(index, query_terms)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results: list[RetrievedChunk] = []
    for doc_index, score in ranked[:k]:
        if score <= 0:
            continue
        doc = _BM25Document(
            page_content=str(index.documents[doc_index]),
            metadata=index.metadatas[doc_index] or {},
        )
        results.append(_build_chunk(doc, score, query_text, doc_id, doc_type))
    return results


def _fuse_ranked_results(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    query_text: str,
    doc_id: str,
    doc_type: str,
    k: int,
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for rank, chunk in enumerate(vector_results, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        chunks[chunk.chunk_id] = chunk
    for rank, chunk in enumerate(bm25_results, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        if chunk.chunk_id not in chunks:
            chunks[chunk.chunk_id] = chunk

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    results: list[RetrievedChunk] = []
    for chunk_id, score in ranked[:k]:
        chunk = chunks[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id or doc_id,
                doc_type=chunk.doc_type or doc_type,
                source_name=chunk.source_name,
                section_path=chunk.section_path,
                text=chunk.text,
                score=float(score),
                query=query_text,
            )
        )
    return results


def _search_once(
    query_text: str,
    doc_id: str,
    doc_type: str,
    k: int,
    extra_filter: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    vector_results = _vector_search_once(
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=k,
        extra_filter=extra_filter,
    )
    if not HYBRID_SEARCH_ENABLED:
        return vector_results

    bm25_results = _bm25_search_once(
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=max(k, BM25_TOP_K),
        extra_filter=extra_filter,
    )
    if not bm25_results:
        return vector_results

    return _fuse_ranked_results(
        vector_results=vector_results,
        bm25_results=bm25_results,
        query_text=query_text,
        doc_id=doc_id,
        doc_type=doc_type,
        k=max(k, BM25_TOP_K),
    )


def _dedup_keep_best(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def retrieve_context_with_meta(
    query: str,
    doc_id: str,
    doc_type: str = "requirement",
    top_k: int = 5,
    multi_query: bool | None = None,
    enable_rerank: bool | None = None,
    extra_filter: dict[str, str] | None = None,
) -> tuple[list[RetrievedChunk], RetrievalMeta]:
    query_text = str(query).strip()
    if not query_text:
        return (
            [],
            RetrievalMeta(
                expanded_queries=[],
                pre_dedup_count=0,
                post_dedup_count=0,
                rerank_mode="disabled" if enable_rerank is False else RERANK_MODE,
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
        expanded_queries = expand_query(query_text, max_queries=QUERY_COUNT)
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
            )
        )

    pre_dedup_count = len(all_candidates)
    deduped = _dedup_keep_best(all_candidates)
    post_dedup_count = len(deduped)

    candidate_pool = deduped[:RERANK_CANDIDATE_POOL]
    rerank_result = rerank(
        query=query_text,
        candidates=candidate_pool,
        enable_rerank=rerank_enabled,
        mode=RERANK_MODE,
        cross_encoder_model=RERANK_CROSS_ENCODER_MODEL,
        cross_encoder_local_files_only=RERANK_CROSS_ENCODER_LOCAL_FILES_ONLY,
        timeout_ms=RERANK_TIMEOUT_MS,
        final_top_n=min(RERANK_FINAL_TOP_N, final_top_k),
    )
    selected = rerank_result.items

    if not rerank_enabled:
        selected = deduped[:final_top_k]
    elif not selected:
        selected = deduped[:final_top_k]

    meta = RetrievalMeta(
        expanded_queries=expanded_queries,
        pre_dedup_count=pre_dedup_count,
        post_dedup_count=post_dedup_count,
        rerank_mode=rerank_result.rerank_mode,
        rerank_enabled=rerank_result.rerank_enabled,
        rerank_latency_ms=rerank_result.rerank_latency_ms,
        rerank_degraded=rerank_result.degraded,
        rerank_degraded_reason=rerank_result.degraded_reason,
    )
    return selected[:final_top_k], meta


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
