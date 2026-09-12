"""检索策略注册表。

提供「按名字取策略」的统一入口，并给出明确的未注册报错，
避免各处直接 import 具体实现类。
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = {
    "expander": {},
    "vector": {},
    "bm25": {},
    "fusion": {},
    "dedup": {},
    "reranker": {},
}

_KINDS = tuple(_REGISTRY.keys())


def _ensure_kind(kind: str) -> dict[str, Any]:
    if kind not in _REGISTRY:
        raise ValueError(f"未知的策略类型: {kind}（可选 {', '.join(_KINDS)}）")
    return _REGISTRY[kind]


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    """注册一个策略工厂；同名会覆盖（便于测试或替换实现）。"""
    _ensure_kind(kind)[name] = factory


def get(kind: str, name: str, **kwargs: Any) -> Any:
    """按名字实例化策略。"""
    bucket = _ensure_kind(kind)
    factory = bucket.get(name)
    if factory is None:
        raise ValueError(
            f"未注册的 {kind} 策略: {name}（可选 {', '.join(sorted(bucket))}）"
        )
    return factory(**kwargs)


def available(kind: str) -> list[str]:
    return sorted(_ensure_kind(kind))
