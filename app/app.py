import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import time
import uuid
from typing import Any, Callable, TypeVar

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 在导入期就加载 .env，确保任何缓存资源构建前配置已就绪。
# 若把 load_dotenv 放在 @st.cache_resource 函数内，缓存只执行一次，
# 进程启动后修改 .env 将无法重新加载，必须重启进程。
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# 应用入口的 sys.path 引导：让项目根下的 rag / skills / utils / workflow 包可被导入。
# 只需根目录；绝不能把 workflow/ 目录本身加进来，否则 `import workflow`
# 会命中 workflow/workflow.py 模块而遮蔽 workflow 包，进而造成循环导入。
# 执行 `pip install -e .` 后下面这段会自动跳过（其他 CLI 脚本各自也带同样引导）。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.document_parser import parse_bytes, supported_suffixes, supports
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, RERANK_MODE, RETRIEVER_TOP_K
from rag.ingest import index_document
from rag.ingest import index_testcase_knowledge_file
from rag.store import is_embedding_degraded
from workflow import nodes
from workflow.checkpoint_store import new_thread_id
from workflow.workflow import create_workflow


# 阶段定义集中在 app/phases.py（单一事实来源），这里只做导入。
import app.phases as phases_module  # noqa: E402
from app.phases import (  # noqa: E402
    PHASE_CASE,
    PHASE_DOWNLOAD,
    PHASE_OUTLINE,
    PHASE_REQUIREMENT,
    PHASE_TEST_POINTS,
    PHASE_UPLOAD,
    artifact_ready,
    get_phase,
    invoke_count as phase_invoke_count,
    phase_order,
    rebuild_registry,
)

# 单次大模型请求超时（秒），避免网络抖动把会话永久挂住。
LLM_REQUEST_TIMEOUT_SECONDS = 120.0
# 默认聊天模型；可用 .env 的 OPENAI_MODEL 覆盖，切换模型无需改代码。
DEFAULT_CHAT_MODEL = "Qwen/Qwen3.5-35B-A3B"
# 单个阶段任务的整体等待上限（秒）。大模型生成用例可能较慢，
# 因此给一个较宽松的兜底值，超时后放弃等待并释放 UI。
TASK_TIMEOUT_SECONDS = 900.0

T = TypeVar("T")

PHASE_ORDER: list[tuple[str, str]] = phase_order()

# 测试点/大纲表格的可选值。schema 里的 Literal 与之对应，
# 调整时需要同时改 workflow/schemas.py 的枚举定义。
TEST_TYPE_OPTIONS = ["功能", "性能", "安全", "兼容性"]
PRIORITY_OPTIONS = ["P0", "P1", "P2", "P3"]


def _default_state() -> dict[str, Any]:
    return {
        "document": "",
        "structured_doc": {},
        "doc_id": "",
        "requirement_analysis": "",
        "test_points": [],
        "test_outline": [],
        "modified_outline": [],
        "test_cases": [],
        "modified_test_cases": [],
        "retrieval_logs": [],
        "excel_output_path": "",
    }


@st.cache_resource
def get_graph_and_llm() -> tuple[Any, ChatOpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY，请先在项目根目录 .env 配置。")

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        # 不设置超时会导致单次网络抖动就把整个 Streamlit 会话永久挂住。
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=2,
    )
    return create_workflow(), llm


def _ensure_session() -> None:
    if "phase" not in st.session_state:
        st.session_state.phase = PHASE_UPLOAD
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = new_thread_id()
    if "source_document" not in st.session_state:
        st.session_state.source_document = ""
    if "source_structured_doc" not in st.session_state:
        st.session_state.source_structured_doc = {}
    if "source_file_name" not in st.session_state:
        st.session_state.source_file_name = ""
    if "source_doc_id" not in st.session_state:
        st.session_state.source_doc_id = ""
    if "requirement_editor_text" not in st.session_state:
        # Use a sentinel so an empty user edit isn't mistaken for "uninitialized".
        st.session_state.requirement_editor_text = None
    if "test_points_table" not in st.session_state:
        st.session_state.test_points_table = []
    if "outline_table" not in st.session_state:
        st.session_state.outline_table = []
    if "test_cases_table" not in st.session_state:
        st.session_state.test_cases_table = []
    if "excel_output_path" not in st.session_state:
        st.session_state.excel_output_path = ""
    if "enable_multi_query" not in st.session_state:
        st.session_state.enable_multi_query = True
    if "enable_rerank" not in st.session_state:
        st.session_state.enable_rerank = True
    # 以下是「高级参数」。默认值一律取自 rag/config.py，
    # 保证不碰侧边栏时行为与改动前完全一致。
    if "retriever_top_k" not in st.session_state:
        st.session_state.retriever_top_k = RETRIEVER_TOP_K
    if "rerank_mode" not in st.session_state:
        st.session_state.rerank_mode = RERANK_MODE
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = CHUNK_OVERLAP
    if "extra_instruction" not in st.session_state:
        st.session_state.extra_instruction = ""
    if "export_format" not in st.session_state:
        st.session_state.export_format = "excel"


def _build_config(llm: ChatOpenAI) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "llm": llm,
            "enable_multi_query": st.session_state.enable_multi_query,
            "enable_rerank": st.session_state.enable_rerank,
            # 高级参数走同一条旁路通道，节点侧用 _get_config_value 读取。
            "retriever_top_k": st.session_state.retriever_top_k,
            "rerank_mode": st.session_state.rerank_mode,
            "extra_instruction": st.session_state.extra_instruction,
            "export_format": st.session_state.export_format,
        }
    }


def _parse_uploaded_document(uploaded_file: Any) -> tuple[str, dict[str, Any]]:
    """解析上传的文档，返回 (文档标识, 结构化内容)。

    格式分派全部交给 utils.document_parser.registry：二进制格式的临时文件
    读写与清理也在注册表里统一处理，这里不再有 per-format 分支。
    """
    data = uploaded_file.getvalue()
    structured = parse_bytes(data, uploaded_file.name).to_dict()

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".md":
        # 纯文本格式保留原文，供后续按需展示。
        return data.decode("utf-8", errors="ignore"), structured

    return f"{suffix.lstrip('.').upper()}:{uploaded_file.name}", structured


def _get_state_values(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    snapshot = graph.get_state(config)
    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        return values
    return {}


def _cases_to_table_rows(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in test_cases:
        row = dict(case)
        steps = row.get("steps", [])
        if isinstance(steps, list):
            row["steps"] = "\n".join(str(item) for item in steps)
        else:
            row["steps"] = str(steps)
        rows.append(row)
    return rows


def _test_points_to_table_rows(test_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in test_points:
        rows.append(
            {
                "name": str(point.get("name", "")),
                "test_type": str(point.get("test_type", "功能")),
                "priority": str(point.get("priority", "P1")),
            }
        )
    return rows


def _normalize_test_points_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "test_type": str(row.get("test_type", "功能")).strip() or "功能",
                "priority": str(row.get("priority", "P1")).strip() or "P1",
            }
        )
    return normalized


def _outline_to_table_rows(test_outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in test_outline:
        module_name = str(item.get("module_name", "")).strip()
        points = item.get("test_points", [])
        if not isinstance(points, list):
            continue

        for point in points:
            if not isinstance(point, dict):
                continue
            rows.append(
                {
                    "module_name": module_name,
                    "name": str(point.get("name", "")),
                    "test_type": str(point.get("test_type", "功能")),
                    "priority": str(point.get("priority", "P1")),
                }
            )
    return rows


def _normalize_outline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        module_name = str(row.get("module_name", "")).strip()
        point_name = str(row.get("name", "")).strip()
        if not module_name or not point_name:
            continue

        point = {
            "name": point_name,
            "test_type": str(row.get("test_type", "功能")).strip() or "功能",
            "priority": str(row.get("priority", "P1")).strip() or "P1",
        }
        grouped.setdefault(module_name, []).append(point)

    outline: list[dict[str, Any]] = []
    for module_name, points in grouped.items():
        outline.append({"module_name": module_name, "test_points": points})
    return outline


def _editor_data_to_rows(editor_data: Any) -> list[dict[str, Any]]:
    if isinstance(editor_data, list):
        return [dict(item) for item in editor_data]

    to_dict = getattr(editor_data, "to_dict", None)
    if callable(to_dict):
        try:
            records = editor_data.to_dict(orient="records")
            return [dict(item) for item in records]
        except TypeError:
            pass

    return []


def _normalize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        case = dict(row)
        steps_text = str(case.get("steps", ""))
        case["steps"] = [line.strip() for line in steps_text.splitlines() if line.strip()]
        normalized.append(case)
    return normalized


# 注入表格转换函数并构建阶段表。放在这里是因为 _to_table_rows 系列
# 定义在上方，避免 app.py 与 phases.py 出现循环导入。
rebuild_registry(
    {
        "test_points": _test_points_to_table_rows,
        "outline": _outline_to_table_rows,
        "cases": _cases_to_table_rows,
    }
)
PHASES = phases_module.PHASES
# 需要逐次 invoke 回放的阶段（上传/下载阶段不参与回放）。
_REPLAY_ORDER: list[str] = [
    key for key, _ in PHASE_ORDER if get_phase(key).invoke_count > 0
]


def _reset_flow() -> None:
    st.session_state.phase = PHASE_UPLOAD
    st.session_state.thread_id = new_thread_id()
    st.session_state.source_document = ""
    st.session_state.source_structured_doc = {}
    st.session_state.source_file_name = ""
    st.session_state.source_doc_id = ""
    st.session_state.requirement_editor_text = None
    st.session_state.test_points_table = []
    st.session_state.outline_table = []
    st.session_state.test_cases_table = []
    st.session_state.excel_output_path = ""


def _run_with_progress(
    task_label: str,
    fn: Callable[[], T],
    timeout: float = TASK_TIMEOUT_SECONDS,
) -> T:
    """Run a blocking task with visible progress and elapsed time."""
    status = st.status(f"{task_label}中...", expanded=True)
    progress = st.progress(0, text=f"{task_label}（预估进度）")
    start = time.time()
    progress_value = 5

    # 不使用 with ThreadPoolExecutor(...)：其 __exit__ 会 shutdown(wait=True) 死等，
    # 一旦任务挂住就会永久占用 Streamlit 脚本线程、并阻塞进程退出。
    # 这里改为手动管理，配合超时与 wait=False 释放。
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn)
        while not future.done():
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"{task_label}超时：已等待 {int(elapsed)}s，超过上限 {int(timeout)}s。"
                )
            progress_value = min(progress_value + 2, 92)
            progress.progress(
                progress_value,
                text=f"{task_label}（预估进度） - 已等待 {int(elapsed)}s",
            )
            time.sleep(0.2)

        remaining = max(timeout - (time.time() - start), 1.0)
        result = future.result(timeout=remaining)

        total = time.time() - start
        progress.progress(100, text=f"{task_label}完成")
        status.update(
            label=f"{task_label}完成，耗时 {total:.1f}s",
            state="complete",
            expanded=False,
        )
        return result
    except Exception:
        status.update(label=f"{task_label}失败", state="error", expanded=True)
        raise
    finally:
        # 不等待仍在运行的任务：避免挂住脚本线程，也避免非守护线程拖住进程退出。
        executor.shutdown(wait=False, cancel_futures=True)


def _prime_editors_from_values(values: dict[str, Any], phase: str) -> None:
    """按阶段配置表把工作流状态同步到编辑器控件。"""
    spec = PHASES.get(phase)
    if spec is None or spec.prime is None:
        return
    spec.prime(values)


def _phase_artifact_ready(values: dict[str, Any], phase: str) -> bool:
    """判断工作流是否已生成目标阶段的产出物。"""
    return artifact_ready(phase, values)


def _replay_to_phase(graph: Any, llm: ChatOpenAI, target_phase: str) -> None:
    if not st.session_state.source_structured_doc:
        raise ValueError("缺少已上传文档，请先上传并开始生成。")

    invoke_count = phase_invoke_count(target_phase)
    labels = [PHASES[key].rerun_label for key in _REPLAY_ORDER]
    st.session_state.thread_id = new_thread_id()
    config = _build_config(llm)

    init_state = _default_state()
    init_state["document"] = st.session_state.source_document
    init_state["structured_doc"] = st.session_state.source_structured_doc
    init_state["doc_id"] = st.session_state.source_doc_id

    # 工作流编译时除首个节点外均设置了 interrupt_before。
    # 这意味着每次 graph.invoke 只执行到被打断节点之前（不含该节点）并在中断边界返回。
    # 要到达目标阶段，必须持续用 invoke(None, config) 恢复；每次恢复会执行被中断的节点
    # 并推进到下一个边界。最多 invoke invoke_count 次，一旦目标产出物就绪即提前结束。
    target_ready = False
    for index in range(invoke_count):
        payload = init_state if index == 0 else None
        _run_with_progress(labels[index], lambda p=payload: graph.invoke(p, config))
        values = _get_state_values(graph, config)
        if _phase_artifact_ready(values, target_phase):
            target_ready = True
            break

    if not target_ready:
        # 兜底：若最后一个节点处于中断状态，再恢复一次确保执行。
        _run_with_progress(labels[-1], lambda: graph.invoke(None, config))

    values = _get_state_values(graph, config)
    _prime_editors_from_values(values, target_phase)
    st.session_state.phase = target_phase


def _rerun_current_phase(graph: Any, llm: ChatOpenAI, phase: str) -> None:
    """Re-run generation for current phase based on existing checkpoint state."""
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    spec = PHASES.get(phase)
    if spec is None or not spec.node_name:
        raise ValueError(f"不支持的阶段重生成: {phase}")

    node_fn = getattr(nodes, spec.node_name, None)
    if node_fn is None:
        raise ValueError(f"节点未实现: {spec.node_name}")

    updates = _run_with_progress(
        spec.rerun_label,
        lambda: node_fn(values, config),
    )

    graph.update_state(config, updates)
    refreshed = _get_state_values(graph, config)
    _prime_editors_from_values(refreshed, phase)
    st.session_state.phase = phase


def _render_upload_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("1. 上传需求文档")
    uploaded_file = st.file_uploader(
        f"支持 {' / '.join(suffix.lstrip('.') for suffix in supported_suffixes())}",
        type=[suffix.lstrip(".") for suffix in supported_suffixes()],
    )

    if st.button("开始生成", type="primary"):
        if uploaded_file is None:
            st.warning("请先上传文档。")
            return

        try:
            doc_id = f"web_{uuid.uuid4().hex}"
            document, structured_doc = _parse_uploaded_document(uploaded_file)
            indexed_chunks = index_document(
                doc_id=doc_id,
                source_name=uploaded_file.name,
                structured_doc=structured_doc,
                chunk_size=int(st.session_state.chunk_size),
                chunk_overlap=int(st.session_state.chunk_overlap),
            )
            st.session_state.source_document = document
            st.session_state.source_structured_doc = structured_doc
            st.session_state.source_file_name = uploaded_file.name
            st.session_state.source_doc_id = doc_id
            st.info(f"文档入库完成，chunk 数：{indexed_chunks}")
            _replay_to_phase(graph, llm, PHASE_REQUIREMENT)
            st.rerun()
        except Exception as exc:
            st.error(f"启动流程失败：{exc}")


def _render_advanced_settings() -> None:
    """侧边栏「高级参数」面板。

    这些参数原本是 rag/config.py 里的常量，调一次要改代码、重启进程。
    这里把它们提到运行时：检索类参数下一次生成即生效，切块类参数只在入库时生效。
    """
    with st.expander("高级参数", expanded=False):
        st.session_state.retriever_top_k = int(
            st.number_input(
                "检索返回条数 top_k",
                min_value=1,
                max_value=50,
                value=int(st.session_state.retriever_top_k),
                step=1,
                help="每个阶段检索最多保留的片段数，改完下一次生成立即生效。",
            )
        )
        st.session_state.rerank_mode = st.selectbox(
            "Rerank 模式",
            options=["cross_encoder", "api", "lite"],
            index=["cross_encoder", "api", "lite"].index(st.session_state.rerank_mode)
            if st.session_state.rerank_mode in ("cross_encoder", "api", "lite")
            else 0,
            help=(
                "cross_encoder：本地交叉编码器，更准但依赖模型缓存；"
                "api：调用外部 /rerank 服务（需在 .env 配置 RERANK_API_BASE_URL / "
                "RERANK_API_MODEL）；lite：关键词重合度打分，零依赖。"
                "前两者失败时自动降级为 lite。"
            ),
        )
        st.session_state.chunk_size = int(
            st.number_input(
                "切块大小（字符）",
                min_value=100,
                max_value=2000,
                value=int(st.session_state.chunk_size),
                step=50,
            )
        )
        # overlap 必须小于 size，否则 ingest._chunk_text 会强制按比例缩小。
        max_overlap = max(int(st.session_state.chunk_size) - 1, 0)
        st.session_state.chunk_overlap = int(
            st.number_input(
                "切块重叠（字符）",
                min_value=0,
                max_value=max_overlap,
                value=min(int(st.session_state.chunk_overlap), max_overlap),
                step=10,
            )
        )
        st.caption(
            "切块参数仅在「上传文档 / 入库知识库」时生效，"
            "已入库内容不会重新切分，改动后需重新上传。"
        )
        st.session_state.extra_instruction = st.text_area(
            "全局附加要求",
            value=st.session_state.extra_instruction,
            height=90,
            placeholder="例：用例必须包含数据校验步骤；异常分支优先级高于正常流程。",
            help="会追加到每个阶段 Prompt 的末尾，不改代码即可约束生成风格。",
        )
        st.session_state.export_format = st.selectbox(
            "导出格式",
            options=["excel", "csv"],
            index=0 if st.session_state.export_format == "excel" else 1,
            help=(
                "excel：带样式的 xlsx；csv：UTF-8 BOM，可直接导入 TestLink / "
                "Jira / 禅道等平台。"
            ),
        )


def _render_testcase_kb_uploader() -> None:
    st.subheader("测试用例知识库入库")
    suffix_labels = " / ".join(
        suffix.lstrip(".") for suffix in supported_suffixes()
    )
    uploaded_files = st.file_uploader(
        f"上传历史测试用例（{suffix_labels}，可多选）",
        type=[suffix.lstrip(".") for suffix in supported_suffixes()],
        accept_multiple_files=True,
        key="testcase_kb_uploader",
    )
    module = st.text_input("module（可选）", key="testcase_kb_module")
    test_type = st.selectbox(
        "test_type（可选）",
        options=["", "功能", "性能", "安全", "兼容性"],
        index=0,
        key="testcase_kb_test_type",
    )
    priority = st.selectbox(
        "priority（可选）",
        options=["", "P0", "P1", "P2", "P3"],
        index=0,
        key="testcase_kb_priority",
    )

    if st.button("入库测试用例", key="index_testcase_kb_btn"):
        if not uploaded_files:
            st.warning("请先上传至少一个测试用例文件。")
            return

        results: list[tuple[str, bool, str]] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            for uploaded in uploaded_files:
                # 文件名来自客户端，必须取 basename 净化，否则 "../" 或绝对路径
                # 会让写入逃出临时目录（CWE-22 路径穿越）。
                file_name = Path(str(uploaded.name)).name
                if not supports(file_name):
                    allowed = " / ".join(suffix.lstrip(".") for suffix in supported_suffixes())
                    results.append((file_name, False, f"仅支持 {allowed}"))
                    continue

                tmp_path = tmp_root / file_name
                # 纵深防御：确认解析后的路径仍在临时目录内。
                if not tmp_path.resolve().is_relative_to(tmp_root.resolve()):
                    results.append((file_name, False, "非法文件名"))
                    continue

                tmp_path.write_bytes(uploaded.getvalue())
                try:
                    chunks = index_testcase_knowledge_file(
                        file_path=tmp_path,
                        module=module.strip(),
                        test_type=test_type.strip(),
                        priority=priority.strip(),
                        chunk_size=int(st.session_state.chunk_size),
                        chunk_overlap=int(st.session_state.chunk_overlap),
                    )
                    results.append((file_name, True, f"成功，chunks={chunks}"))
                except Exception as exc:
                    results.append((file_name, False, f"失败：{exc}"))

        st.markdown("**入库结果**")
        for file_name, ok, message in results:
            if ok:
                st.success(f"{file_name}: {message}")
            else:
                st.error(f"{file_name}: {message}")


def _render_retrieval_evidence(values: dict[str, Any], phase: str, title: str) -> None:
    logs = values.get("retrieval_logs", [])
    if not isinstance(logs, list) or not logs:
        return

    latest = None
    for log in reversed(logs):
        if isinstance(log, dict) and str(log.get("phase", "")) == phase:
            latest = log
            break
    if latest is None:
        return

    with st.expander(title, expanded=False):
        st.caption(f"Query: {latest.get('query', '')}")
        expanded_queries = latest.get("expanded_queries", [])
        if isinstance(expanded_queries, list) and expanded_queries:
            st.caption(f"Expanded Queries: {len(expanded_queries)}")
            for query in expanded_queries[:3]:
                st.code(str(query), language="text")
        pre_dedup_count = latest.get("pre_dedup_count")
        post_dedup_count = latest.get("post_dedup_count")
        rerank_mode = latest.get("rerank_mode", "unknown")
        rerank_enabled = latest.get("rerank_enabled")
        rerank_latency_ms = latest.get("rerank_latency_ms", 0)
        rerank_degraded = latest.get("rerank_degraded", False)
        rerank_degraded_reason = str(latest.get("rerank_degraded_reason", "")).strip()
        if pre_dedup_count is not None and post_dedup_count is not None:
            st.caption(
                f"候选数: pre_dedup={pre_dedup_count}, post_dedup={post_dedup_count}"
            )
        st.caption(
            f"Rerank: mode={rerank_mode}, enabled={rerank_enabled}, "
            f"latency={rerank_latency_ms}ms, degraded={rerank_degraded}"
        )
        if rerank_degraded_reason:
            st.caption(f"Rerank fallback reason: {rerank_degraded_reason}")

        citations = latest.get("citations", [])
        if isinstance(citations, list) and citations:
            st.markdown("**Citations**")
            for idx, citation in enumerate(citations[:3], start=1):
                if not isinstance(citation, dict):
                    continue
                st.markdown(
                    f"{idx}. `{citation.get('chunk_id', '')}` | "
                    f"`{citation.get('section_path', '')}` | "
                    f"`{citation.get('source_name', '')}` | "
                    f"score={citation.get('score', 0.0)}"
                )

        hits = latest.get("hits", [])
        if not isinstance(hits, list) or not hits:
            st.write("未检索到依据片段。")
            return
        for idx, hit in enumerate(hits[:3], start=1):
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id", ""))
            doc_type = str(hit.get("doc_type", ""))
            section_path = str(hit.get("section_path", ""))
            source_name = str(hit.get("source_name", ""))
            score = hit.get("score", 0.0)
            preview = str(hit.get("text_preview", "")).strip()
            st.markdown(
                f"{idx}. `{chunk_id}` | `{doc_type}` | `{section_path}` | `{source_name}` | score={score}"
            )
            if preview:
                st.caption(preview)
            full_text = str(hit.get("text_full", "")).strip()
            if full_text:
                with st.expander(f"查看全文 #{idx}", expanded=False):
                    st.text(full_text)


def _render_phase_actions(
    graph: Any,
    llm: ChatOpenAI,
    config: dict[str, Any],
    current_phase: str,
    advance_label: str,
    advance_prompt: str,
    advance_payload: dict[str, Any],
    advance_error_prefix: str,
    require_message: str = "",
    require_ok: bool = True,
    after_advance: Callable[..., None] | None = None,
) -> None:
    """渲染「通过并进入下一阶段 / 重新生成本阶段」这组通用操作。

    重构前这四个页面各写了一遍几乎相同的按钮逻辑（校验、update_state、
    invoke、同步下一页控件、切阶段、报错），改一处交互要改四遍。
    现在统一在这里实现，各页面只描述差异。
    """
    col1, col2 = st.columns(2)

    if col1.button(advance_label, type="primary"):
        try:
            if require_message and not require_ok:
                raise ValueError(require_message)

            graph.update_state(config, advance_payload)
            final_state = _run_with_progress(
                advance_prompt, lambda: graph.invoke(None, config)
            )

            if after_advance is not None:
                after_advance(graph, config, final_state)

            next_phase = _next_phase(current_phase)
            next_values = _get_state_values(graph, config)
            _prime_editors_from_values(next_values, next_phase)
            st.session_state.phase = next_phase
            st.rerun()
        except Exception as exc:
            st.error(f"{advance_error_prefix}：{exc}")

    if col2.button(f"重新生成{PHASES[current_phase].label}"):
        try:
            _rerun_current_phase(graph, llm, current_phase)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")


def _next_phase(current_phase: str) -> str:
    """按阶段顺序取下一个阶段。"""
    keys = [key for key, _ in PHASE_ORDER]
    index = keys.index(current_phase)
    if index + 1 >= len(keys):
        return current_phase
    return keys[index + 1]


def _render_requirement_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("2. 审核需求分析")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    if st.session_state.requirement_editor_text is None:
        st.session_state.requirement_editor_text = str(
            values.get("requirement_analysis", "")
        )

    st.caption("AI 生成需求分析")
    st.text_area("编辑需求分析", key="requirement_editor_text", height=280)

    _render_phase_actions(
        graph=graph,
        llm=llm,
        config=config,
        current_phase=PHASE_REQUIREMENT,
        advance_label="通过并提取测试点",
        advance_prompt="正在提取测试点",
        advance_payload={
            "requirement_analysis": st.session_state.requirement_editor_text.strip()
        },
        advance_error_prefix="提取测试点失败",
    )

    _render_retrieval_evidence(values, "analyze_requirement", "本阶段依据片段")


def _render_test_points_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("3. 审核测试点")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    if not st.session_state.test_points_table:
        st.session_state.test_points_table = _test_points_to_table_rows(
            values.get("test_points", [])
        )

    st.caption("AI 生成测试点")
    edited_data = st.data_editor(
        st.session_state.test_points_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("测试点", required=True),
            "test_type": st.column_config.SelectboxColumn(
                "类型", options=TEST_TYPE_OPTIONS, required=True
            ),
            "priority": st.column_config.SelectboxColumn(
                "优先级", options=PRIORITY_OPTIONS, required=True
            ),
        },
    )

    test_points = _normalize_test_points_rows(_editor_data_to_rows(edited_data))
    _render_phase_actions(
        graph=graph,
        llm=llm,
        config=config,
        current_phase=PHASE_TEST_POINTS,
        advance_label="通过并生成测试大纲",
        advance_prompt="正在生成测试大纲",
        advance_payload={"test_points": test_points},
        advance_error_prefix="生成测试大纲失败",
        require_message="至少保留一个有效测试点。",
        require_ok=bool(test_points),
    )

    _render_retrieval_evidence(values, "extract_test_points", "本阶段依据片段")


def _render_outline_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("4. 审核测试大纲")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    if not st.session_state.outline_table:
        st.session_state.outline_table = _outline_to_table_rows(
            values.get("test_outline", [])
        )

    st.caption("AI 生成测试大纲")
    edited_data = st.data_editor(
        st.session_state.outline_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "module_name": st.column_config.TextColumn("模块", required=True),
            "name": st.column_config.TextColumn("测试点", required=True),
            "test_type": st.column_config.SelectboxColumn(
                "类型", options=TEST_TYPE_OPTIONS, required=True
            ),
            "priority": st.column_config.SelectboxColumn(
                "优先级", options=PRIORITY_OPTIONS, required=True
            ),
        },
    )

    modified_outline = _normalize_outline_rows(_editor_data_to_rows(edited_data))
    _render_phase_actions(
        graph=graph,
        llm=llm,
        config=config,
        current_phase=PHASE_OUTLINE,
        advance_label="通过并生成测试用例",
        advance_prompt="正在生成测试用例",
        advance_payload={"modified_outline": modified_outline},
        advance_error_prefix="生成测试用例失败",
        require_message="至少保留一个有效模块与测试点。",
        require_ok=bool(modified_outline),
    )

    _render_retrieval_evidence(values, "generate_outline", "本阶段依据片段")


def _render_case_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("5. 审核测试用例")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    edited_data = st.data_editor(
        st.session_state.test_cases_table,
        width="stretch",
        num_rows="dynamic",
    )

    modified_cases = _normalize_cases(_editor_data_to_rows(edited_data))
    _render_phase_actions(
        graph=graph,
        llm=llm,
        config=config,
        current_phase=PHASE_CASE,
        advance_label=f"通过并导出 {st.session_state.export_format.upper()}",
        advance_prompt="正在导出文件",
        advance_payload={"modified_test_cases": modified_cases},
        advance_error_prefix="导出失败",
        require_message="至少保留一条有效测试用例。",
        require_ok=bool(modified_cases),
        after_advance=_capture_export_path,
    )

    _render_retrieval_evidence(values, "generate_cases_requirement", "需求依据片段")
    _render_retrieval_evidence(values, "generate_cases_testcase", "历史用例参考片段")


def _capture_export_path(graph: Any, config: dict[str, Any], final_state: Any) -> None:
    """导出节点返回的路径可能只在 final_state 里，也可能只落到了图上。"""
    export_path = ""
    if isinstance(final_state, dict):
        export_path = str(final_state.get("excel_output_path", ""))
    if not export_path:
        export_path = str(_get_state_values(graph, config).get("excel_output_path", ""))
    st.session_state.excel_output_path = export_path


# 导出格式 -> 下载时的 MIME 类型。
EXPORT_MIME_TYPES: dict[str, str] = {
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv; charset=utf-8",
}


def _render_download_page() -> None:
    st.subheader("6. 下载测试用例")
    excel_output_path = st.session_state.excel_output_path

    if excel_output_path and Path(excel_output_path).exists():
        output_path = Path(excel_output_path)
        # MIME 按实际后缀取，避免导出 CSV 时仍按 Excel 类型下载。
        suffix = output_path.suffix.lower()
        export_format = (
            "csv" if suffix == ".csv" else "excel"
        )
        st.success(f"已生成文件：{output_path}")
        st.download_button(
            label=f"下载 {export_format.upper()} 文件",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime=EXPORT_MIME_TYPES[export_format],
            type="primary",
        )
    else:
        st.warning("未找到导出的文件，请返回上一步重新导出。")

    if st.button("生成新的测试用例"):
        _reset_flow()
        st.rerun()


def _render_phase_nav(phase: str) -> None:
    labels = [label for _, label in PHASE_ORDER]
    phase_index_map = {phase_name: idx for idx, (phase_name, _) in enumerate(PHASE_ORDER)}
    current_index = phase_index_map.get(phase, 0)

    cols = st.columns(len(labels))
    for idx, col in enumerate(cols):
        if idx < current_index:
            col.success(f"已完成\n{idx + 1}. {labels[idx]}")
        elif idx == current_index:
            col.info(f"当前\n{idx + 1}. {labels[idx]}")
        else:
            col.caption(f"待执行\n{idx + 1}. {labels[idx]}")


def main() -> None:
    st.set_page_config(page_title="AI 测试用例生成器", layout="wide")
    st.title("AI 测试用例生成器")

    _ensure_session()
    with st.sidebar:
        st.subheader("RAG 设置")
        # 降级必须显式告知用户，否则检索质量会无声崩塌且无人察觉。
        if is_embedding_degraded():
            st.error(
                "⚠️ Embedding 服务不可用，已降级为本地哈希向量，"
                "检索结果严重失真。请检查 OPENAI_API_KEY / 网络后重启应用。"
            )
        st.session_state.enable_multi_query = st.toggle(
            "启用 Multi-Query",
            value=st.session_state.enable_multi_query,
        )
        st.session_state.enable_rerank = st.toggle(
            "启用 Rerank",
            value=st.session_state.enable_rerank,
        )
        _render_advanced_settings()
        st.divider()
        _render_testcase_kb_uploader()

    try:
        graph, llm = get_graph_and_llm()
    except Exception as exc:
        st.error(f"初始化失败：{exc}")
        st.stop()

    phase = st.session_state.phase
    _render_phase_nav(phase)
    st.divider()

    # 阶段 -> 渲染函数的映射表。新增阶段时只需在这里加一行，
    # 不再需要维护一条 if-elif 链。
    renderer = _PHASE_RENDERERS.get(phase)
    if renderer is None:
        _reset_flow()
        st.rerun()
        return

    renderer(graph, llm)


def _render_upload(view_graph: Any, view_llm: ChatOpenAI) -> None:
    _render_upload_page(view_graph, view_llm)


def _render_download(view_graph: Any, view_llm: ChatOpenAI) -> None:
    _ = view_graph, view_llm
    _render_download_page()


# 渲染函数签名统一为 (graph, llm)，便于按表分发。
_PHASE_RENDERERS: dict[str, Callable[[Any, ChatOpenAI], None]] = {
    PHASE_UPLOAD: _render_upload,
    PHASE_REQUIREMENT: _render_requirement_page,
    PHASE_TEST_POINTS: _render_test_points_page,
    PHASE_OUTLINE: _render_outline_page,
    PHASE_CASE: _render_case_page,
    PHASE_DOWNLOAD: _render_download,
}


if __name__ == "__main__":
    main()
