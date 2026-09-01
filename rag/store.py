import os
import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List

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


class FallbackEmbeddings(Embeddings):
    """主 embedding 不可用时降级到本地哈希向量。

    降级状态必须是「类级别」共享的：若为实例属性，每次 get_vector_store()
    新建实例都会丢失该状态，导致同一 collection 里混入 256 维（本地）与
    1536 维（OpenAI）两种向量，使相似度计算完全失真甚至维度不匹配报错。
    """

    _using_fallback = False

    def __init__(self, primary: Embeddings, fallback: Embeddings) -> None:
        self.primary = primary
        self.fallback = fallback

    @classmethod
    def _mark_degraded(cls, exc: Exception) -> None:
        if cls._using_fallback:
            return
        cls._using_fallback = True
        logger.warning(
            "Embedding 服务不可用，已永久降级为本地哈希向量（检索质量严重下降）：%s: %s",
            type(exc).__name__,
            exc,
        )

    def _embed_documents_primary(self, texts: List[str]) -> List[List[float]]:
        return self.primary.embed_documents(texts)

    def _embed_query_primary(self, text: str) -> List[float]:
        return self.primary.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if FallbackEmbeddings._using_fallback:
            return self.fallback.embed_documents(texts)
        try:
            return self._embed_documents_primary(texts)
        except Exception as exc:
            FallbackEmbeddings._mark_degraded(exc)
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if FallbackEmbeddings._using_fallback:
            return self.fallback.embed_query(text)
        try:
            return self._embed_query_primary(text)
        except Exception as exc:
            FallbackEmbeddings._mark_degraded(exc)
            return self.fallback.embed_query(text)


def is_embedding_degraded() -> bool:
    """是否曾降级为本地哈希 embedding（跨实例共享状态）。"""
    return FallbackEmbeddings._using_fallback


def _build_embeddings() -> Embeddings:
    try:
        primary = _build_openai_embeddings()
        return FallbackEmbeddings(primary=primary, fallback=LocalHashEmbeddings())
    except Exception:
        return LocalHashEmbeddings()


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
