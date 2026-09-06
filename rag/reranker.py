from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import time
from typing import Any

from rag.text_utils import tokenize_unique


# 本地无模型缓存时 cross_encoder 会去 HuggingFace 下载，国内网络基本必失败，
# 因此提供 API 版 rerank 作为替代：只要有个兼容 /rerank 的服务就能用。
# 协议采用 Cohere 风格（Cohere / Jina / 硅基流动 / vLLM / 火山方舟均兼容）。
API_RERANK_PATH = "/rerank"
API_RERANK_DEFAULT_TIMEOUT_MS = 15000


@dataclass
class RerankResult:
    items: list
    rerank_mode: str
    rerank_enabled: bool
    rerank_latency_ms: int
    degraded: bool
    degraded_reason: str


def _score(query: str, text: str, base_score: float) -> float:
    # 复用检索侧的统一分词器（含中文 bigram 修复），
    # 否则重排的重合度打分对中文恒为 0，lite 重排等于失效。
    q = tokenize_unique(query)
    t = tokenize_unique(text)
    if not q or not t:
        return base_score
    overlap = len(q.intersection(t))
    return base_score + overlap / max(len(q), 1)


def _slice_candidates(candidates: list, final_top_n: int) -> list:
    return candidates[:final_top_n]


def _rerank_lite(
    query: str,
    candidates: list,
    final_top_n: int,
) -> list:
    ranked = sorted(
        candidates,
        key=lambda item: _score(
            query=query,
            text=getattr(item, "text", ""),
            base_score=float(getattr(item, "score", 0.0)),
        ),
        reverse=True,
    )
    return _slice_candidates(ranked, final_top_n)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str, local_files_only: bool) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        local_files_only=local_files_only,
    )


def _rerank_cross_encoder(
    query: str,
    candidates: list,
    final_top_n: int,
    model_name: str,
    local_files_only: bool,
    timeout_ms: int,
) -> RerankResult:
    start = time.time()
    cross_encoder = _load_cross_encoder(model_name, local_files_only)
    load_and_prepare_ms = int((time.time() - start) * 1000)
    if timeout_ms > 0 and load_and_prepare_ms > timeout_ms:
        raise TimeoutError("cross-encoder load exceeded timeout")

    if not candidates:
        return RerankResult(
            items=[],
            rerank_mode="cross_encoder",
            rerank_enabled=True,
            rerank_latency_ms=load_and_prepare_ms,
            degraded=False,
            degraded_reason="",
        )

    pairs = [(query, str(getattr(item, "text", ""))) for item in candidates]
    scores = cross_encoder.predict(pairs)
    total_latency_ms = int((time.time() - start) * 1000)
    if timeout_ms > 0 and total_latency_ms > timeout_ms:
        raise TimeoutError("cross-encoder predict exceeded timeout")

    ranked = sorted(
        zip(candidates, scores, strict=False),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    items = [item for item, _score_value in ranked[:final_top_n]]
    if not items:
        raise ValueError("cross-encoder returned empty items")

    return RerankResult(
        items=items,
        rerank_mode="cross_encoder",
        rerank_enabled=True,
        rerank_latency_ms=total_latency_ms,
        degraded=False,
        degraded_reason="",
    )


def _rerank_api(
    query: str,
    candidates: list,
    final_top_n: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout_ms: int,
) -> RerankResult:
    """调用外部 /rerank 服务做重排（Cohere 风格协议）。

    失败时直接抛异常，由 rerank() 统一降级到 lite，不在内部吞异常，
    这样前端能看到真实的降级原因。
    """
    import httpx

    if not base_url:
        raise ValueError("未配置 RERANK_API_BASE_URL，无法使用 API rerank")
    if not model:
        raise ValueError("未配置 RERANK_API_MODEL，无法使用 API rerank")

    if not candidates:
        return RerankResult(
            items=[],
            rerank_mode="api",
            rerank_enabled=True,
            rerank_latency_ms=0,
            degraded=False,
            degraded_reason="",
        )

    endpoint = base_url.rstrip("/") + API_RERANK_PATH
    payload = {
        "model": model,
        "query": query,
        "documents": [str(getattr(item, "text", "")) for item in candidates],
        "top_n": final_top_n,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout_seconds = (
        timeout_ms / 1000 if timeout_ms > 0 else API_RERANK_DEFAULT_TIMEOUT_MS / 1000
    )
    start = time.time()
    response = httpx.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    latency_ms = int((time.time() - start) * 1000)

    if response.status_code != 200:
        raise RuntimeError(
            f"rerank API 返回 {response.status_code}: {response.text[:200]}"
        )

    body = response.json()
    # 不同厂商字段命名不同：Cohere/硅基流动用 results，部分国内服务用 data。
    raw_results = body.get("results") or body.get("data") or []
    if not raw_results:
        raise ValueError("rerank API 返回空结果")

    scored: list[tuple[float, int]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score", item.get("score", 0.0))
        if not isinstance(index, int) or index < 0 or index >= len(candidates):
            continue
        scored.append((float(score), index))

    if not scored:
        raise ValueError("rerank API 结果中无可用的 index/score")

    scored.sort(key=lambda pair: pair[0], reverse=True)
    items = [candidates[index] for _score_value, index in scored[:final_top_n]]

    return RerankResult(
        items=items,
        rerank_mode="api",
        rerank_enabled=True,
        rerank_latency_ms=latency_ms,
        degraded=False,
        degraded_reason="",
    )


def rerank(
    query: str,
    candidates: list,
    enable_rerank: bool = True,
    mode: str = "lite",
    cross_encoder_model: str = "",
    cross_encoder_local_files_only: bool = True,
    timeout_ms: int = 0,
    final_top_n: int = 5,
) -> RerankResult:
    if final_top_n <= 0:
        return RerankResult(
            items=[],
            rerank_mode="disabled" if not enable_rerank else mode,
            rerank_enabled=enable_rerank,
            rerank_latency_ms=0,
            degraded=False,
            degraded_reason="",
        )

    if not enable_rerank:
        return RerankResult(
            items=_slice_candidates(candidates, final_top_n),
            rerank_mode="disabled",
            rerank_enabled=False,
            rerank_latency_ms=0,
            degraded=False,
            degraded_reason="",
        )

    start = time.time()
    try:
        if mode == "api":
            return _rerank_api(
                query=query,
                candidates=candidates,
                final_top_n=final_top_n,
                base_url=os.getenv("RERANK_API_BASE_URL", "").strip(),
                api_key=(
                    os.getenv("RERANK_API_KEY", "").strip()
                    or os.getenv("OPENAI_API_KEY", "")
                ),
                model=os.getenv("RERANK_API_MODEL", "").strip(),
                timeout_ms=timeout_ms,
            )

        if mode == "cross_encoder":
            return _rerank_cross_encoder(
                query=query,
                candidates=candidates,
                final_top_n=final_top_n,
                model_name=cross_encoder_model,
                local_files_only=cross_encoder_local_files_only,
                timeout_ms=timeout_ms,
            )

        ranked = _rerank_lite(
            query=query,
            candidates=candidates,
            final_top_n=final_top_n,
        )
        latency_ms = int((time.time() - start) * 1000)
        return RerankResult(
            items=ranked,
            rerank_mode="lite",
            rerank_enabled=True,
            rerank_latency_ms=latency_ms,
            degraded=False,
            degraded_reason="",
        )
    except Exception as exc:
        degraded_reason = f"{type(exc).__name__}: {exc}".strip()

        # cross_encoder / api 失败时退到 lite：lite 零依赖且一定有结果，
        # 比直接原样截断更能保住相关性。lite 自身再失败，才只能截断。
        if mode in ("cross_encoder", "api"):
            try:
                fallback_items = _rerank_lite(
                    query=query,
                    candidates=candidates,
                    final_top_n=final_top_n,
                )
            except Exception as lite_exc:  # noqa: BLE001 - 兜底也必须兜住
                return RerankResult(
                    items=_slice_candidates(candidates, final_top_n),
                    rerank_mode=mode,
                    rerank_enabled=True,
                    rerank_latency_ms=int((time.time() - start) * 1000),
                    degraded=True,
                    degraded_reason=(
                        f"{degraded_reason}；lite 兜底也失败: "
                        f"{type(lite_exc).__name__}: {lite_exc}"
                    ),
                )
            return RerankResult(
                items=fallback_items,
                rerank_mode="lite",
                rerank_enabled=True,
                rerank_latency_ms=int((time.time() - start) * 1000),
                degraded=True,
                degraded_reason=degraded_reason,
            )

        return RerankResult(
            items=_slice_candidates(candidates, final_top_n),
            rerank_mode=mode,
            rerank_enabled=True,
            rerank_latency_ms=int((time.time() - start) * 1000),
            degraded=True,
            degraded_reason=degraded_reason,
        )
