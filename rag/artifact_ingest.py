"""落盘产物回流向量库：让本轮产出成为下一轮的知识资产。

qa-skills 的流水线产物（需求模型 / 测试策略 / 用例 markmap / 测试报告）落盘后，
如果只躺在磁盘上，下一轮生成就还得从零开始。这里把它们重新切块入库，
形成闭环：

    markmap 用例 → 入库（doc_type=testcase）
    → 下一轮 generate_cases 双路检索时作为「历史用例写法参考」被召回
    → 新人写用例的口径逐步收敛到项目既有风格

分类型入库的考量：
- **用例 markmap 入 testcase 库**：与历史测试用例同库，会被用例生成阶段的
  第二路检索命中（retrieve_testcase_context_with_meta 不过滤 doc_id）
- **需求模型 / 策略 / 报告入 artifact 库**：独立命名空间，不干扰
  requirement 检索（requirement 按 doc_id 严格过滤当前文档，
  混入历史产物会让"当前需求的依据"变得不纯）

回流是幂等的：同一 source_name 的旧 chunk 会被先删后插，重复执行不会膨胀。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from rag.ingest import index_document
from rag.retriever import invalidate_bm25_cache
from rag.store import get_vector_store
from utils.artifacts import get_artifacts_dir
from utils.document_parser.md_parser import parse_markdown


# 产物文件名 → 入库 doc_type
ARTIFACT_DOC_TYPES: dict[str, str] = {
    "测试用例_markmap.md": "testcase",
    "需求模型.md": "artifact",
    "测试策略.md": "artifact",
}


def _resolve_doc_type(file_name: str) -> str:
    """按文件名判定产物类型；报告与回归清单带日期，按前缀匹配。"""
    if file_name in ARTIFACT_DOC_TYPES:
        return ARTIFACT_DOC_TYPES[file_name]
    if file_name.startswith("测试报告"):
        return "artifact"
    if file_name.startswith("回归清单"):
        return "artifact"
    if file_name.startswith("手动执行记录"):
        return "artifact"
    # 未知产物一律进 artifact 库，宁可检索不到也不污染 requirement/testcase。
    return "artifact"


def _drop_existing(source_name: str, doc_type: str) -> int:
    """删除同名产物的旧 chunk，保证回流幂等。"""
    vector_store = get_vector_store()
    try:
        existing: dict[str, Any] = vector_store.get(
            where={"$and": [{"source_name": source_name}, {"doc_type": doc_type}]}
        )
    except Exception:
        # 元数据查询在部分 Chroma 版本/降级 embedding 下可能失败，
        # 查不到就当作没有历史，直接入库（重复至多影响召回权重）。
        return 0

    ids = existing.get("ids") if isinstance(existing, dict) else None
    if not ids:
        return 0

    try:
        vector_store.delete(ids=list(ids))
    except Exception:
        return 0
    return len(ids)


def index_artifact_file(
    file_path: str | Path,
    project_name: str = "",
    doc_type: str | None = None,
) -> int:
    """把单个落盘产物写入向量库，返回入库 chunk 数。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到产物文件: {path}")
    if path.suffix.lower() != ".md":
        # 只回流 Markdown 产物：schema.yaml 是机读层，检索它没意义。
        return 0

    resolved_doc_type = doc_type or _resolve_doc_type(path.name)
    # 同名产物在不同项目下会撞车，用 项目名/文件名 作为 source_name 隔离。
    source_name = f"{project_name}/{path.name}" if project_name else path.name

    _drop_existing(source_name, resolved_doc_type)

    structured_doc = parse_markdown(path.read_text(encoding="utf-8")).to_dict()
    count = index_document(
        doc_id=f"artifact_{uuid.uuid4().hex[:8]}",
        source_name=source_name,
        structured_doc=structured_doc,
        doc_type=resolved_doc_type,
        extra_metadata={"module": project_name},
    )
    invalidate_bm25_cache()
    return count


def index_project_artifacts(project_name: str) -> dict[str, int]:
    """批量回流某项目的全部 Markdown 产物，返回 {文件名: chunk 数}。"""
    artifacts_dir = get_artifacts_dir(project_name)
    results: dict[str, int] = {}

    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        try:
            results[path.name] = index_artifact_file(
                path, project_name=project_name
            )
        except Exception as exc:
            # 单个产物回流失败不应影响其他产物，记录 0 便于前端提示。
            results[path.name] = 0
            print(f"[artifact_ingest] 回流失败 {path.name}: {exc}")

    return results
