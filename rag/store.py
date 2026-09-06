import os
import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)

# 项目根目录，用于定位 .env，避免因进程 CWD 不同而加载不到配置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embedding fallback to keep RAG usable offline."""

    def __init__(self, dimensions: int = LOCAL_EMBEDDING_DIM) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+|[\u4e00-\u9fff]", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            slot = int(digest, 16) % self.dimensions
            vector[slot] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _build_openai_embeddings() -> OpenAIEmbeddings:
    # 显式指定 .env 路径，避免依赖进程 CWD 导致读不到 key 而静默降级。
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    # Embedding 允许走独立的 base_url：聊天可能用厂商的专用端点（如 Coding Plan 的
    # /api/coding/v3），而 embedding 只能用标准 /api/v3，两者必须能分开配。
    # 未配置时回落到 OPENAI_BASE_URL，保持原有行为。
    embedding_base = (
        os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )
    return OpenAIEmbeddings(
        model=os.getenv("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=embedding_base,
    )


def _build_local_embeddings() -> Embeddings:
    """本地 sentence-transformers 向量模型。

    当远程 embedding 不可用（未开通、欠费、网络受限）时，
    用本地中文模型（如 BAAI/bge-small-zh-v1.5）保住真实语义检索，
    避免退到 LocalHashEmbeddings 那种字面哈希。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = os.getenv("RAG_LOCAL_EMBEDDING_MODEL", "").strip()
    if not model_name:
        raise ValueError("未配置 RAG_LOCAL_EMBEDDING_MODEL")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # 归一化后余弦相似度与点积等价，与远程 embedding 的度量方式一致。
        encode_kwargs={"normalize_embeddings": True},
    )


class FallbackEmbeddings(Embeddings):
    """按优先级依次尝试多个 embedding，全部失败才退到本地哈希向量。

    降级状态必须是「类级别」共享的：若为实例属性，每次 get_vector_store()
    新建实例都会丢失该状态，导致同一 collection 里混入不同维度的向量，
    使相似度计算完全失真甚至维度不匹配报错。

    注意：切换 provider 同样会改变向量空间。已入库的向量是用旧 provider
    算出来的，换了 provider 之后必须清空 data/chroma 重建，否则检索结果
    是「两种向量空间混算」的垃圾结果。
    """

    _using_fallback = False
    _active_index = 0
    _active_label = ""

    def __init__(
        self,
        providers: List[tuple[str, Embeddings]],
        fallback: Embeddings,
    ) -> None:
        self.providers = list(providers)
        self.fallback = fallback

    @classmethod
    def _mark_degraded(cls, exc: Exception) -> None:
        if cls._using_fallback:
            return
        cls._using_fallback = True
        logger.warning(
            "所有 embedding 方案均不可用，已永久降级为本地哈希向量"
            "（检索质量严重下降）：%s: %s",
            type(exc).__name__,
            exc,
        )

    @classmethod
    def _switch_provider(
        cls,
        index: int,
        from_label: str,
        to_label: str,
        exc: Exception,
    ) -> None:
        cls._active_index = index
        cls._active_label = to_label
        logger.warning(
            "embedding 方案 %s 不可用，切换到 %s。注意：向量空间已改变，"
            "已入库数据需清空 data/chroma 重建，否则检索结果失真。原因：%s: %s",
            from_label,
            to_label,
            type(exc).__name__,
            exc,
        )

    def _embed_documents_with(
        self,
        provider: Embeddings,
        texts: List[str],
    ) -> List[List[float]]:
        return provider.embed_documents(texts)

    def _embed_query_with(
        self,
        provider: Embeddings,
        text: str,
    ) -> List[float]:
        return provider.embed_query(text)

    def _run(
        self,
        operation: str,
        texts: List[str] | None = None,
        text: str | None = None,
    ) -> Any:
        """从当前生效的 provider 开始依次尝试，直到成功或全部失败。"""
        if FallbackEmbeddings._using_fallback:
            provider = self.fallback
            return (
                provider.embed_documents(texts or [])
                if operation == "documents"
                else provider.embed_query(text or "")
            )

        last_error: Exception | None = None
        from_index = FallbackEmbeddings._active_index
        for offset in range(len(self.providers)):
            index = (from_index + offset) % len(self.providers)
            label, provider = self.providers[index]
            try:
                result = (
                    self._embed_documents_with(provider, texts or [])
                    if operation == "documents"
                    else self._embed_query_with(provider, text or "")
                )
                if index != from_index:
                    FallbackEmbeddings._switch_provider(
                        index,
                        self.providers[from_index][0],
                        label,
                        last_error or RuntimeError("unknown"),
                    )
                else:
                    FallbackEmbeddings._active_label = label
                return result
            except Exception as exc:  # noqa: BLE001 - 逐个降级，失败即换下一个
                last_error = exc

        FallbackEmbeddings._mark_degraded(last_error or RuntimeError("no provider"))
        provider = self.fallback
        return (
            provider.embed_documents(texts or [])
            if operation == "documents"
            else provider.embed_query(text or "")
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._run("documents", texts=texts)

    def embed_query(self, text: str) -> List[float]:
        return self._run("query", text=text)


def is_embedding_degraded() -> bool:
    """是否曾降级为本地哈希 embedding（跨实例共享状态）。"""
    return FallbackEmbeddings._using_fallback


def active_embedding_label() -> str:
    """当前实际生效的 embedding 方案，供 UI 展示。"""
    if FallbackEmbeddings._using_fallback:
        return "local-hash（兜底）"
    return FallbackEmbeddings._active_label or "unknown"


def _build_embeddings() -> Embeddings:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    builders = {
        "remote": ("remote", _build_openai_embeddings),
        "local": ("local", _build_local_embeddings),
    }
    # 默认远程优先、本地兜底；可用 RAG_EMBEDDING_PROVIDER 调整顺序，
    # 例如 local,remote 表示优先用本地模型（离线、无费用）。
    order = os.getenv("RAG_EMBEDDING_PROVIDER", "remote,local")

    providers: List[tuple[str, Embeddings]] = []
    for name in [item.strip().lower() for item in order.split(",") if item.strip()]:
        if name not in builders:
            logger.warning("未知的 embedding 方案: %s（可选 remote/local）", name)
            continue
        label, builder = builders[name]
        try:
            providers.append((label, builder()))
        except Exception as exc:  # noqa: BLE001 - 构造失败就跳过该方案
            logger.warning("embedding 方案 %s 初始化失败，已跳过：%s", label, exc)

    if not providers:
        return LocalHashEmbeddings()

    FallbackEmbeddings._active_label = providers[0][0]
    return FallbackEmbeddings(providers=providers, fallback=LocalHashEmbeddings())


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """返回全局唯一的向量库实例。

    必须单例化：否则每次检索都会新建 Chroma 与 Embeddings 客户端，
    既浪费连接资源，也会让降级状态（见 FallbackEmbeddings）无法保持。
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_build_embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )
