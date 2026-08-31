"""Streamlit 前端入口：驱动 qa-skills 八阶段流水线。

界面与工作流的对应关系（每个阶段产出后 interrupt，人工确认再继续）：

    上传文档 → 需求分析 → 测试策略 → 测试点 → 测试大纲 → 测试用例
    → 用例审查 → 执行回收 → Bug 分析 → 回归清单 → 测试报告

界面只做两件事：把人工修订写回 state，以及推进工作流。
所有阶段产出同时落盘到 artifacts/{项目名}/，会话中断后凭产物可续跑。
"""

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

from utils.document_parser.docx_parser import parse_docx
from utils.document_parser.md_parser import parse_markdown
from rag.artifact_ingest import index_project_artifacts
from rag.ingest import index_document
from rag.ingest import index_testcase_knowledge_file
from rag.store import is_embedding_degraded
from workflow.checkpoint_store import new_thread_id
from workflow.workflow import create_workflow
from execution.result_parser import (
    build_records_from_cases,
    normalize_records,
    parse_execution_output,
    render_execution_records_markdown,
    summarize_records,
)
from execution.scaffolds import build_api_scaffold, build_e2e_scaffold
from utils.artifacts import (
    dated_file_name,
    record_artifact,
    sanitize_project_name,
    write_artifact,
)


PHASE_UPLOAD = "upload"
PHASE_REQUIREMENT = "requirement_review"
PHASE_STRATEGY = "strategy_review"
PHASE_TEST_POINTS = "test_points_review"
PHASE_OUTLINE = "outline_review"
PHASE_CASE = "case_review"
PHASE_REVIEW = "case_audit"
PHASE_EXECUTION = "execution"
PHASE_BUGS = "bug_review"
PHASE_REGRESSION = "regression_review"
PHASE_REPORT = "report"

# 单次大模型请求超时（秒），避免网络抖动把会话永久挂住。
LLM_REQUEST_TIMEOUT_SECONDS = 120.0
# 单个阶段任务的整体等待上限（秒）。大模型生成用例可能较慢，
# 因此给一个较宽松的兜底值，超时后放弃等待并释放 UI。
TASK_TIMEOUT_SECONDS = 900.0

T = TypeVar("T")

PHASE_ORDER: list[tuple[str, str]] = [
    (PHASE_UPLOAD, "上传文档"),
    (PHASE_REQUIREMENT, "需求分析"),
    (PHASE_STRATEGY, "测试策略"),
    (PHASE_TEST_POINTS, "测试点提取"),
    (PHASE_OUTLINE, "测试大纲"),
    (PHASE_CASE, "测试用例"),
    (PHASE_REVIEW, "用例审查"),
    (PHASE_EXECUTION, "执行回收"),
    (PHASE_BUGS, "Bug 分析"),
    (PHASE_REGRESSION, "回归清单"),
    (PHASE_REPORT, "测试报告"),
]


def _default_state() -> dict[str, Any]:
    return {
        "document": "",
        "structured_doc": {},
        "doc_id": "",
        "project_name": "",
        "requirement_analysis": "",
        "test_strategy": {},
        "modified_test_strategy": {},
        "test_points": [],
        "test_outline": [],
        "modified_outline": [],
        "test_cases": [],
        "modified_test_cases": [],
        "review_findings": [],
        "review_summary": "",
        "execution_mode": "manual",
        "execution_records": [],
        "bug_items": [],
        "bug_summary": "",
        "regression_items": [],
        "regression_summary": "",
        "test_report": "",
        "test_report_conclusion": "",
        "test_report_basis": "",
        "schema_validation_passed": True,
        "schema_validation_output": "",
        "retrieval_logs": [],
        "artifacts_dir": "",
        "artifact_files": [],
        "excel_output_path": "",
    }


@st.cache_resource
def get_graph_and_llm() -> tuple[Any, ChatOpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未检测到 OPENAI_API_KEY，请先在项目根目录 .env 配置。")

    base_url = os.getenv("OPENAI_BASE_URL")
    llm = ChatOpenAI(
        model="Qwen/Qwen3.5-35B-A3B",
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
    if "project_name" not in st.session_state:
        st.session_state.project_name = ""
    if "requirement_editor_text" not in st.session_state:
        # Use a sentinel so an empty user edit isn't mistaken for "uninitialized".
        st.session_state.requirement_editor_text = None
    if "strategy_risks_table" not in st.session_state:
        st.session_state.strategy_risks_table = []
    if "strategy_summary_text" not in st.session_state:
        st.session_state.strategy_summary_text = ""
    if "test_points_table" not in st.session_state:
        st.session_state.test_points_table = []
    if "outline_table" not in st.session_state:
        st.session_state.outline_table = []
    if "test_cases_table" not in st.session_state:
        st.session_state.test_cases_table = []
    if "execution_table" not in st.session_state:
        st.session_state.execution_table = []
    if "excel_output_path" not in st.session_state:
        st.session_state.excel_output_path = ""
    if "enable_multi_query" not in st.session_state:
        st.session_state.enable_multi_query = True
    if "enable_rerank" not in st.session_state:
        st.session_state.enable_rerank = True


def _build_config(llm: ChatOpenAI) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "llm": llm,
            "enable_multi_query": st.session_state.enable_multi_query,
            "enable_rerank": st.session_state.enable_rerank,
        }
    }


def _parse_uploaded_document(uploaded_file: Any) -> tuple[str, dict[str, Any]]:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".md":
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        structured = parse_markdown(raw_text).to_dict()
        return raw_text, structured

    if suffix == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)

        try:
            structured = parse_docx(tmp_path).to_dict()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return f"DOCX:{uploaded_file.name}", structured

    raise ValueError("仅支持 docx 或 md 文件。")


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
        tags = row.get("tags", [])
        row["tags"] = "、".join(str(item) for item in tags) if isinstance(tags, list) else str(tags or "")
        rows.append(row)
    return rows


def _normalize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        case = dict(row)
        steps_text = str(case.get("steps", ""))
        case["steps"] = [line.strip() for line in steps_text.splitlines() if line.strip()]
        tags_text = str(case.get("tags", "")).strip()
        case["tags"] = [
            tag.strip() for tag in tags_text.replace("，", "、").split("、") if tag.strip()
        ]
        normalized.append(case)
    return normalized


def _test_points_to_table_rows(test_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in test_points:
        rows.append(
            {
                "name": str(point.get("name", "")),
                "test_type": str(point.get("test_type", "功能")),
                "priority": str(point.get("priority", "P1")),
                "risk_ref": str(point.get("risk_ref", "")),
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
                "risk_ref": str(row.get("risk_ref", "")).strip(),
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
                    "risk_ref": str(point.get("risk_ref", "")),
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
            "risk_ref": str(row.get("risk_ref", "")).strip(),
        }
        grouped.setdefault(module_name, []).append(point)

    outline: list[dict[str, Any]] = []
    for module_name, points in grouped.items():
        outline.append({"module_name": module_name, "test_points": points})
    return outline


def _strategy_to_risk_rows(test_strategy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for risk in (test_strategy or {}).get("risk_map", []):
        evidence = risk.get("evidence", {}) if isinstance(risk, dict) else {}
        rows.append(
            {
                "id": str(risk.get("id", "")),
                "feature": str(risk.get("feature", "")),
                "dimension": str(risk.get("dimension", "")),
                "impact": int(risk.get("impact", 1) or 1),
                "likelihood": int(risk.get("likelihood", 1) or 1),
                "level": str(risk.get("level", "Medium")),
                "source": str(evidence.get("source", "")) if isinstance(evidence, dict) else "",
                "rationale": str(risk.get("rationale", "")),
            }
        )
    return rows


def _risk_rows_to_strategy(
    base_strategy: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: str,
) -> dict[str, Any]:
    """把界面上编辑过的风险表回写为策略结构。

    只回写 Risk Map 与摘要：功能域/类型域的十轴决策在界面上不提供编辑
    （那需要更强的交互），人工如需调整可直接改落盘的 `测试策略.md`。
    """
    strategy = dict(base_strategy or {})
    risk_map: list[dict[str, Any]] = []
    for row in rows:
        feature = str(row.get("feature", "")).strip()
        if not feature:
            continue
        impact = int(row.get("impact", 1) or 1)
        likelihood = int(row.get("likelihood", 1) or 1)
        risk_map.append(
            {
                "id": str(row.get("id", "")).strip(),
                "feature": feature,
                "dimension": str(row.get("dimension", "")).strip(),
                "impact": impact,
                "likelihood": likelihood,
                "level": str(row.get("level", "Medium")).strip(),
                "evidence": {
                    "level": "E1",
                    "source": str(row.get("source", "")).strip(),
                    "confidence": "medium",
                    "status": "inference",
                },
                "rationale": str(row.get("rationale", "")).strip(),
            }
        )
    strategy["risk_map"] = risk_map
    strategy["summary"] = summary
    return strategy


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


def _reset_flow() -> None:
    st.session_state.phase = PHASE_UPLOAD
    st.session_state.thread_id = new_thread_id()
    st.session_state.source_document = ""
    st.session_state.source_structured_doc = {}
    st.session_state.source_file_name = ""
    st.session_state.source_doc_id = ""
    st.session_state.requirement_editor_text = None
    st.session_state.strategy_risks_table = []
    st.session_state.strategy_summary_text = ""
    st.session_state.test_points_table = []
    st.session_state.outline_table = []
    st.session_state.test_cases_table = []
    st.session_state.execution_table = []
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
    if phase == PHASE_REQUIREMENT:
        st.session_state.requirement_editor_text = str(values.get("requirement_analysis", ""))
    if phase == PHASE_STRATEGY:
        strategy = values.get("modified_test_strategy") or values.get("test_strategy") or {}
        st.session_state.strategy_risks_table = _strategy_to_risk_rows(strategy)
        st.session_state.strategy_summary_text = str(strategy.get("summary", ""))
    if phase == PHASE_TEST_POINTS:
        st.session_state.test_points_table = _test_points_to_table_rows(values.get("test_points", []))
    if phase == PHASE_OUTLINE:
        st.session_state.outline_table = _outline_to_table_rows(values.get("test_outline", []))
    if phase == PHASE_CASE:
        st.session_state.test_cases_table = _cases_to_table_rows(values.get("test_cases", []))
    if phase == PHASE_EXECUTION:
        st.session_state.execution_table = build_records_from_cases(
            values.get("modified_test_cases")
            if values.get("modified_test_cases") is not None
            else values.get("test_cases", []),
            values.get("execution_records", []),
        )


def _phase_artifact_ready(values: dict[str, Any], phase: str) -> bool:
    """判断工作流是否已生成目标阶段的产出物。"""
    if phase == PHASE_REQUIREMENT:
        return bool(values.get("requirement_analysis"))
    if phase == PHASE_STRATEGY:
        return bool(values.get("test_strategy"))
    if phase == PHASE_TEST_POINTS:
        return bool(values.get("test_points"))
    if phase == PHASE_OUTLINE:
        return bool(values.get("test_outline"))
    if phase == PHASE_CASE:
        return bool(values.get("test_cases"))
    if phase == PHASE_REVIEW:
        # 审查可能得出"零问题"，此时 review_summary 有值即为完成。
        return bool(values.get("review_findings")) or bool(values.get("review_summary"))
    if phase == PHASE_EXECUTION:
        return bool(values.get("excel_output_path"))
    if phase == PHASE_BUGS:
        return bool(values.get("bug_items")) or bool(values.get("bug_summary"))
    if phase == PHASE_REGRESSION:
        return bool(values.get("regression_items")) or bool(values.get("regression_summary"))
    if phase == PHASE_REPORT:
        return bool(values.get("test_report"))
    return False


# 各阶段推进所需的 invoke 次数（工作流在目标节点前 interrupt）。
_INVOKE_COUNT: dict[str, int] = {
    PHASE_REQUIREMENT: 1,
    PHASE_STRATEGY: 2,
    PHASE_TEST_POINTS: 3,
    PHASE_OUTLINE: 4,
    PHASE_CASE: 5,
    PHASE_REVIEW: 6,
    PHASE_EXECUTION: 7,
    PHASE_BUGS: 8,
    PHASE_REGRESSION: 9,
    PHASE_REPORT: 10,
}

_PHASE_LABELS: dict[str, str] = {
    PHASE_REQUIREMENT: "正在生成需求分析",
    PHASE_STRATEGY: "正在生成测试策略",
    PHASE_TEST_POINTS: "正在提取测试点",
    PHASE_OUTLINE: "正在生成测试大纲",
    PHASE_CASE: "正在生成测试用例",
    PHASE_REVIEW: "正在审查测试用例",
    PHASE_EXECUTION: "正在导出 Excel",
    PHASE_BUGS: "正在分析 Bug",
    PHASE_REGRESSION: "正在生成回归清单",
    PHASE_REPORT: "正在生成测试报告",
}


def _replay_to_phase(graph: Any, llm: ChatOpenAI, target_phase: str) -> None:
    if not st.session_state.source_structured_doc:
        raise ValueError("缺少已上传文档，请先上传并开始生成。")

    invoke_count = _INVOKE_COUNT[target_phase]
    st.session_state.thread_id = new_thread_id()
    config = _build_config(llm)

    init_state = _default_state()
    init_state["document"] = st.session_state.source_document
    init_state["structured_doc"] = st.session_state.source_structured_doc
    init_state["doc_id"] = st.session_state.source_doc_id
    init_state["project_name"] = st.session_state.project_name

    # 工作流编译时除首个节点外均设置了 interrupt_before。
    # 这意味着每次 graph.invoke 只执行到被打断节点之前（不含该节点）并在中断边界返回。
    # 要到达目标阶段，必须持续用 invoke(None, config) 恢复；每次恢复会执行被中断的节点
    # 并推进到下一个边界。最多 invoke invoke_count 次，一旦目标产出物就绪即提前结束。
    target_ready = False
    for index in range(invoke_count):
        label = _labels_for_replay(target_phase)[index]
        payload = init_state if index == 0 else None
        _run_with_progress(label, lambda p=payload: graph.invoke(p, config))
        values = _get_state_values(graph, config)
        if _phase_artifact_ready(values, target_phase):
            target_ready = True
            break

    if not target_ready:
        # 兜底：若最后一个节点处于中断状态，再恢复一次确保执行。
        _run_with_progress(
            _PHASE_LABELS[target_phase], lambda: graph.invoke(None, config)
        )

    values = _get_state_values(graph, config)
    _prime_editors_from_values(values, target_phase)
    st.session_state.phase = target_phase


def _labels_for_replay(target_phase: str) -> list[str]:
    order = [
        PHASE_REQUIREMENT,
        PHASE_STRATEGY,
        PHASE_TEST_POINTS,
        PHASE_OUTLINE,
        PHASE_CASE,
        PHASE_REVIEW,
        PHASE_EXECUTION,
        PHASE_BUGS,
        PHASE_REGRESSION,
        PHASE_REPORT,
    ]
    target_index = order.index(target_phase)
    return [_PHASE_LABELS[phase] for phase in order[: target_index + 1]]


def _rerun_current_phase(graph: Any, llm: ChatOpenAI, phase: str) -> None:
    """Re-run generation for current phase based on existing checkpoint state."""
    from workflow.nodes import (
        analyze_requirement_node,
        bug_analysis_node,
        extract_test_points_node,
        generate_cases_node,
        generate_outline_node,
        regression_node,
        review_cases_node,
        test_report_node,
        test_strategy_node,
    )

    config = _build_config(llm)
    values = _get_state_values(graph, config)

    rerunners: dict[str, Callable[[], dict[str, Any]]] = {
        PHASE_REQUIREMENT: lambda: analyze_requirement_node(values, config),
        PHASE_STRATEGY: lambda: test_strategy_node(values, config),
        PHASE_TEST_POINTS: lambda: extract_test_points_node(values, config),
        PHASE_OUTLINE: lambda: generate_outline_node(values, config),
        PHASE_CASE: lambda: generate_cases_node(values, config),
        PHASE_REVIEW: lambda: review_cases_node(values, config),
        PHASE_BUGS: lambda: bug_analysis_node(values, config),
        PHASE_REGRESSION: lambda: regression_node(values, config),
        PHASE_REPORT: lambda: test_report_node(values, config),
    }
    if phase not in rerunners:
        raise ValueError(f"不支持的阶段重生成: {phase}")

    updates = _run_with_progress(
        f"正在重新{_PHASE_LABELS[phase].replace('正在', '')}",
        rerunners[phase],
    )
    graph.update_state(config, updates)
    refreshed = _get_state_values(graph, config)
    _prime_editors_from_values(refreshed, phase)
    st.session_state.phase = phase


def _advance(graph: Any, llm: ChatOpenAI, values: dict[str, Any], phase: str) -> dict[str, Any]:
    """推进到下一阶段并刷新编辑器数据。"""
    config = _build_config(llm)
    _run_with_progress(_PHASE_LABELS[phase], lambda: graph.invoke(None, config))
    next_values = _get_state_values(graph, config)
    _prime_editors_from_values(next_values, phase)
    _ = values
    return next_values


# ---------------------------------------------------------------------------
# 页面渲染
# ---------------------------------------------------------------------------


def _render_testcase_kb_uploader() -> None:
    st.subheader("测试用例知识库入库")
    uploaded_files = st.file_uploader(
        "上传历史测试用例（md/docx，可多选）",
        type=["md", "docx"],
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
                suffix = Path(file_name).suffix.lower()
                if suffix not in {".md", ".docx"}:
                    results.append((file_name, False, "仅支持 md/docx"))
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


def _render_artifact_panel(values: dict[str, Any]) -> None:
    """展示落盘产物清单与回流入口。"""
    artifacts_dir = str(values.get("artifacts_dir", "") or "")
    artifact_files = values.get("artifact_files", []) or []
    if not artifact_files:
        return

    with st.expander(f"落盘产物（{len(artifact_files)} 个）", expanded=False):
        if artifacts_dir:
            st.caption(f"目录：{artifacts_dir}")
        for file_path in artifact_files:
            path = Path(str(file_path))
            if path.exists():
                st.download_button(
                    label=f"下载 {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    key=f"dl_{path.name}_{abs(hash(str(path))) % 100000}",
                )
            else:
                st.caption(f"{path.name}（文件已被清理）")

        if st.button("回流产物到知识库", key="ingest_artifacts"):
            try:
                project_name = str(values.get("project_name", "") or "default")
                results = index_project_artifacts(project_name)
                st.success("回流完成：" + "、".join(f"{k}({v})" for k, v in results.items()))
            except Exception as exc:
                st.error(f"回流失败：{exc}")


def _render_upload_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("1. 上传需求文档")
    project_name_input = st.text_input(
        "项目名（决定落盘目录 artifacts/{项目名}/）",
        value=st.session_state.project_name,
        placeholder="例如：营销券系统",
    )
    uploaded_file = st.file_uploader("支持 docx / md", type=["docx", "md"])

    if st.button("开始生成", type="primary"):
        if uploaded_file is None:
            st.warning("请先上传文档。")
            return

        try:
            project_name = sanitize_project_name(
                project_name_input or Path(str(uploaded_file.name)).stem
            )
            st.session_state.project_name = project_name

            doc_id = f"web_{uuid.uuid4().hex}"
            document, structured_doc = _parse_uploaded_document(uploaded_file)
            indexed_chunks = index_document(
                doc_id=doc_id,
                source_name=uploaded_file.name,
                structured_doc=structured_doc,
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


def _render_requirement_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("2. 审核需求分析（需求模型）")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    requirement_analysis = str(values.get("requirement_analysis", ""))

    if st.session_state.requirement_editor_text is None:
        st.session_state.requirement_editor_text = requirement_analysis

    st.caption("AI 生成的需求模型（十字段 Schema，可直接编辑）")
    st.text_area("编辑需求模型", key="requirement_editor_text", height=320)

    col1, col2 = st.columns(2)
    if col1.button("通过并生成测试策略", type="primary"):
        try:
            graph.update_state(
                config,
                {"requirement_analysis": st.session_state.requirement_editor_text.strip()},
            )
            _advance(graph, llm, values, PHASE_STRATEGY)
            st.session_state.phase = PHASE_STRATEGY
            st.rerun()
        except Exception as exc:
            st.error(f"生成测试策略失败：{exc}")

    if col2.button("重新生成需求分析"):
        try:
            _rerun_current_phase(graph, llm, PHASE_REQUIREMENT)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_retrieval_evidence(values, "analyze_requirement", "本阶段依据片段")
    _render_artifact_panel(values)


def _render_strategy_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("3. 审核测试策略（Risk Map + 两域范围）")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    strategy = values.get("modified_test_strategy") or values.get("test_strategy") or {}

    if not st.session_state.strategy_risks_table:
        st.session_state.strategy_risks_table = _strategy_to_risk_rows(strategy)
        st.session_state.strategy_summary_text = str(strategy.get("summary", ""))

    st.caption("Risk Map：风险等级 = Impact × Likelihood，无证据的评级视为无效")
    edited_risks = st.data_editor(
        st.session_state.strategy_risks_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "id": st.column_config.TextColumn("编号", required=True),
            "feature": st.column_config.TextColumn("功能点", required=True),
            "dimension": st.column_config.TextColumn("维度"),
            "impact": st.column_config.NumberColumn("Impact", min_value=1, max_value=5),
            "likelihood": st.column_config.NumberColumn("Likelihood", min_value=1, max_value=5),
            "level": st.column_config.SelectboxColumn(
                "等级", options=["Critical", "High", "Medium", "Low"], required=True
            ),
            "source": st.column_config.TextColumn("证据来源（章节）"),
            "rationale": st.column_config.TextColumn("评级理由"),
        },
    )

    st.text_area("策略摘要", key="strategy_summary_text", height=100)

    # 范围决策只读展示：十轴决策的编辑建议直接改落盘的 测试策略.md。
    for scope_key, title in (
        ("functional_scope", "功能域范围（六轴）"),
        ("type_scope", "类型域范围（十轴）"),
    ):
        items = strategy.get(scope_key, []) or []
        if not items:
            continue
        with st.expander(title, expanded=False):
            for item in items:
                depth = item.get("depth", "")
                decision = item.get("decision", "")
                depth_text = f" / {depth}" if decision != "exclude" else ""
                st.markdown(
                    f"- **{item.get('axis', '')}**：{decision}{depth_text}｜{item.get('rationale', '')}"
                )

    col1, col2 = st.columns(2)
    if col1.button("通过并提取测试点", type="primary"):
        try:
            risk_rows = _editor_data_to_rows(edited_risks)
            updated_strategy = _risk_rows_to_strategy(
                strategy, risk_rows, st.session_state.strategy_summary_text
            )
            graph.update_state(config, {"modified_test_strategy": updated_strategy})
            _advance(graph, llm, values, PHASE_TEST_POINTS)
            st.session_state.phase = PHASE_TEST_POINTS
            st.rerun()
        except Exception as exc:
            st.error(f"提取测试点失败：{exc}")

    if col2.button("重新生成测试策略"):
        try:
            _rerun_current_phase(graph, llm, PHASE_STRATEGY)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_retrieval_evidence(values, "test_strategy", "本阶段依据片段")
    _render_artifact_panel(values)


def _render_test_points_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("4. 审核测试点")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    if not st.session_state.test_points_table:
        st.session_state.test_points_table = _test_points_to_table_rows(values.get("test_points", []))

    st.caption("AI 生成测试点（risk_ref 关联 Risk Map 中的风险编号）")
    edited_data = st.data_editor(
        st.session_state.test_points_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("测试点", required=True),
            "test_type": st.column_config.SelectboxColumn(
                "类型", options=["功能", "性能", "安全", "兼容性"], required=True
            ),
            "priority": st.column_config.SelectboxColumn(
                "优先级", options=["P0", "P1", "P2", "P3"], required=True
            ),
            "risk_ref": st.column_config.TextColumn("关联风险"),
        },
    )

    col1, col2 = st.columns(2)
    if col1.button("通过并生成测试大纲", type="primary"):
        try:
            test_points = _normalize_test_points_rows(_editor_data_to_rows(edited_data))
            if not test_points:
                raise ValueError("至少保留一个有效测试点。")

            graph.update_state(config, {"test_points": test_points})
            _advance(graph, llm, values, PHASE_OUTLINE)
            st.session_state.phase = PHASE_OUTLINE
            st.rerun()
        except Exception as exc:
            st.error(f"生成测试大纲失败：{exc}")

    if col2.button("重新生成测试点"):
        try:
            _rerun_current_phase(graph, llm, PHASE_TEST_POINTS)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_retrieval_evidence(values, "extract_test_points", "本阶段依据片段")


def _render_outline_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("5. 审核测试大纲")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    if not st.session_state.outline_table:
        st.session_state.outline_table = _outline_to_table_rows(values.get("test_outline", []))

    st.caption("模块划分决定用例编号（TC-{模块号}-{序号}）")
    edited_data = st.data_editor(
        st.session_state.outline_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "module_name": st.column_config.TextColumn("模块", required=True),
            "name": st.column_config.TextColumn("测试点", required=True),
            "test_type": st.column_config.SelectboxColumn(
                "类型", options=["功能", "性能", "安全", "兼容性"], required=True
            ),
            "priority": st.column_config.SelectboxColumn(
                "优先级", options=["P0", "P1", "P2", "P3"], required=True
            ),
            "risk_ref": st.column_config.TextColumn("关联风险"),
        },
    )

    col1, col2 = st.columns(2)
    if col1.button("通过并生成测试用例", type="primary"):
        try:
            modified_outline = _normalize_outline_rows(_editor_data_to_rows(edited_data))
            if not modified_outline:
                raise ValueError("至少保留一个有效模块与测试点。")

            graph.update_state(config, {"modified_outline": modified_outline})
            _advance(graph, llm, values, PHASE_CASE)
            st.session_state.phase = PHASE_CASE
            st.rerun()
        except Exception as exc:
            st.error(f"生成测试用例失败：{exc}")

    if col2.button("重新生成测试大纲"):
        try:
            _rerun_current_phase(graph, llm, PHASE_OUTLINE)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_retrieval_evidence(values, "generate_outline", "本阶段依据片段")


def _render_case_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("6. 审核测试用例")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    edited_data = st.data_editor(
        st.session_state.test_cases_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "case_id": st.column_config.TextColumn("用例编号", required=True),
            "directory": st.column_config.TextColumn("模块"),
            "case_level": st.column_config.SelectboxColumn(
                "优先级", options=["P0", "P1", "P2", "P3"], required=True
            ),
            "test_point": st.column_config.TextColumn("测试点"),
            "precondition": st.column_config.TextColumn("前置条件"),
            "steps": st.column_config.TextColumn("测试步骤（每行一步）"),
            "expected_result": st.column_config.TextColumn("预期结果"),
            "risk_ref": st.column_config.TextColumn("关联风险"),
            "evidence_source": st.column_config.TextColumn("文档依据"),
            "test_data": st.column_config.TextColumn("测试数据"),
            "tags": st.column_config.TextColumn("标签（、分隔）"),
        },
    )

    col1, col2 = st.columns(2)
    if col1.button("通过并进入用例审查", type="primary"):
        try:
            modified_cases = _normalize_cases(_editor_data_to_rows(edited_data))
            graph.update_state(config, {"modified_test_cases": modified_cases})
            _advance(graph, llm, values, PHASE_REVIEW)
            st.session_state.phase = PHASE_REVIEW
            st.rerun()
        except Exception as exc:
            st.error(f"用例审查失败：{exc}")

    if col2.button("重新生成测试用例"):
        try:
            _rerun_current_phase(graph, llm, PHASE_CASE)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_retrieval_evidence(values, "generate_cases_requirement", "需求依据片段")
    _render_retrieval_evidence(values, "generate_cases_testcase", "历史用例参考片段")
    _render_artifact_panel(values)


def _render_review_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("7. 用例审查结果")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    findings = values.get("review_findings", []) or []
    summary = str(values.get("review_summary", "")).strip()

    if summary:
        st.info(summary)

    if findings:
        st.caption("审查发现（不自动改写用例，由你决定是否采纳）")
        st.data_editor(
            findings,
            width="stretch",
            disabled=True,
            column_config={
                "case_id": st.column_config.TextColumn("用例"),
                "category": st.column_config.TextColumn("类别"),
                "description": st.column_config.TextColumn("问题"),
                "suggestion": st.column_config.TextColumn("建议"),
            },
        )
    else:
        st.success("未发现问题。")

    if st.button("返回修改用例"):
        st.session_state.phase = PHASE_CASE
        st.rerun()

    col1, col2 = st.columns(2)
    if col1.button("通过并导出产物", type="primary"):
        try:
            _advance(graph, llm, values, PHASE_EXECUTION)
            st.session_state.phase = PHASE_EXECUTION
            st.rerun()
        except Exception as exc:
            st.error(f"导出失败：{exc}")

    if col2.button("重新审查"):
        try:
            _rerun_current_phase(graph, llm, PHASE_REVIEW)
            st.rerun()
        except Exception as exc:
            st.error(f"重新审查失败：{exc}")

    _render_retrieval_evidence(values, "review_cases", "本阶段依据片段")
    _render_artifact_panel(values)


def _render_execution_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("8. 执行回收")
    config = _build_config(llm)
    values = _get_state_values(graph, config)

    passed = bool(values.get("schema_validation_passed", True))
    validation_output = str(values.get("schema_validation_output", "")).strip()
    with st.expander("Schema 校验结果", expanded=not passed):
        if passed:
            st.success("markmap ↔ schema.yaml 一致性校验通过")
        else:
            st.warning("校验未通过（不影响导出，但下游消费前建议修正）")
        if validation_output:
            st.code(validation_output, language="text")

    st.caption("粘贴 pytest 输出或 JUnit XML 自动解析；也可直接在下表手工录入结果")
    uploaded_report = st.file_uploader(
        "上传执行输出（txt / xml，可选）", type=["txt", "xml", "log"]
    )
    raw_output = st.text_area("或粘贴执行输出", height=140, key="execution_raw_output")

    if st.button("解析执行输出"):
        text = raw_output or ""
        if uploaded_report is not None:
            text = uploaded_report.getvalue().decode("utf-8", errors="ignore")
        parsed = parse_execution_output(text)
        if not parsed:
            st.warning("未从输出中解析出带 TC 编号的结果，请检查格式或手工录入。")
        else:
            cases = (
                values.get("modified_test_cases")
                if values.get("modified_test_cases") is not None
                else values.get("test_cases", [])
            )
            st.session_state.execution_table = build_records_from_cases(cases, parsed)
            st.success(f"解析到 {len(parsed)} 条结果，已合并到下表。")

    if not st.session_state.execution_table:
        cases = (
            values.get("modified_test_cases")
            if values.get("modified_test_cases") is not None
            else values.get("test_cases", [])
        )
        st.session_state.execution_table = build_records_from_cases(
            cases, values.get("execution_records", [])
        )

    edited = st.data_editor(
        st.session_state.execution_table,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "case_id": st.column_config.TextColumn("用例", disabled=True),
            "title": st.column_config.TextColumn("名称", disabled=True),
            "priority": st.column_config.TextColumn("优先级", disabled=True),
            "status": st.column_config.SelectboxColumn(
                "结果", options=["通过", "失败", "阻塞", "未执行"], required=True
            ),
            "triage": st.column_config.SelectboxColumn(
                "失败分流", options=["", "A", "B1", "B2", "C", "D", "U"]
            ),
            "evidence": st.column_config.TextColumn("证据"),
            "note": st.column_config.TextColumn("备注"),
        },
    )

    # 脚本骨架生成：给出可下载的执行脚手架，由使用者在真实环境中补齐并运行。
    with st.expander("生成执行脚本骨架", expanded=False):
        st.caption("生成的是可运行骨架，接口与定位细节留 TODO 由你补齐后再执行")
        cases_for_scaffold = (
            values.get("modified_test_cases")
            if values.get("modified_test_cases") is not None
            else values.get("test_cases", [])
        )
        col_api, col_e2e = st.columns(2)
        if col_api.button("生成 API 脚本（pytest + requests）"):
            files = build_api_scaffold(cases_for_scaffold)
            st.session_state.scaffold_files = files
        if col_e2e.button("生成 E2E 脚本（Playwright）"):
            files = build_e2e_scaffold(cases_for_scaffold)
            st.session_state.scaffold_files = files

        for name, content in (st.session_state.get("scaffold_files") or {}).items():
            st.download_button(
                label=f"下载 {name}",
                data=content,
                file_name=Path(name).name,
                key=f"scaffold_{name}",
            )

    col1, col2 = st.columns(2)
    if col1.button("保存结果并进行 Bug 分析", type="primary"):
        try:
            records = normalize_records(_editor_data_to_rows(edited))
            project_name = str(values.get("project_name", "") or "default")
            markdown = render_execution_records_markdown(
                records, project_name=project_name, mode="manual"
            )
            record_path = write_artifact(
                project_name, dated_file_name("手动执行记录"), markdown
            )

            graph.update_state(
                config,
                {
                    "execution_records": records,
                    "artifact_files": record_artifact(
                        values.get("artifact_files", []), record_path
                    ),
                },
            )
            _advance(graph, llm, values, PHASE_BUGS)
            st.session_state.phase = PHASE_BUGS
            st.rerun()
        except Exception as exc:
            st.error(f"保存执行结果失败：{exc}")

    if col2.button("跳过执行，直接出报告"):
        try:
            graph.update_state(config, {"execution_mode": "skip"})
            _advance(graph, llm, values, PHASE_BUGS)
            st.session_state.phase = PHASE_BUGS
            st.rerun()
        except Exception as exc:
            st.error(f"推进失败：{exc}")

    summary = summarize_records(_editor_data_to_rows(edited))
    st.caption(
        f"当前统计：共 {summary['total']} 条，通过 {summary['通过']}，"
        f"失败 {summary['失败']}，阻塞 {summary['阻塞']}，未执行 {summary['未执行']}"
    )
    _render_artifact_panel(values)


def _render_bugs_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("9. Bug 分析")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    bug_items = values.get("bug_items", []) or []
    bug_summary = str(values.get("bug_summary", "")).strip()

    if bug_summary:
        st.info(bug_summary)

    if bug_items:
        st.data_editor(
            bug_items,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "id": st.column_config.TextColumn("编号"),
                "title": st.column_config.TextColumn("描述"),
                "severity": st.column_config.SelectboxColumn(
                    "严重程度", options=["S0", "S1", "S2"], required=True
                ),
                "status": st.column_config.SelectboxColumn(
                    "状态",
                    options=["新建", "已修复待验证", "已验证关闭", "不予修复"],
                    required=True,
                ),
                "related_case": st.column_config.TextColumn("发现用例"),
                "reproduce_steps": st.column_config.TextColumn("复现步骤"),
                "expected_behavior": st.column_config.TextColumn("预期行为"),
                "actual_behavior": st.column_config.TextColumn("实际行为"),
                "root_cause": st.column_config.TextColumn("根因分析"),
                "impact_scope": st.column_config.TextColumn("影响范围"),
                "severity_basis": st.column_config.TextColumn("定级依据"),
                "fix_suggestion": st.column_config.TextColumn("修复建议"),
                "regression_suggestion": st.column_config.TextColumn("回归建议"),
            },
        )
    else:
        st.caption("未确认任何 Bug（失败可能已被归为环境/用例问题，或本轮无执行结果）。")

    col1, col2 = st.columns(2)
    if col1.button("通过并生成回归清单", type="primary"):
        try:
            _advance(graph, llm, values, PHASE_REGRESSION)
            st.session_state.phase = PHASE_REGRESSION
            st.rerun()
        except Exception as exc:
            st.error(f"生成回归清单失败：{exc}")

    if col2.button("重新分析 Bug"):
        try:
            _rerun_current_phase(graph, llm, PHASE_BUGS)
            st.rerun()
        except Exception as exc:
            st.error(f"重新分析失败：{exc}")

    _render_retrieval_evidence(values, "bug_analysis", "本阶段依据片段")


def _render_regression_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("10. 回归清单")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    items = values.get("regression_items", []) or []
    regression_summary = str(values.get("regression_summary", "")).strip()

    if regression_summary:
        st.info(regression_summary)

    if items:
        st.data_editor(
            items,
            width="stretch",
            disabled=True,
            column_config={
                "case_id": st.column_config.TextColumn("用例"),
                "title": st.column_config.TextColumn("名称"),
                "level": st.column_config.TextColumn("级别"),
                "reason": st.column_config.TextColumn("依据"),
            },
        )
    else:
        st.caption("无回归触发源（无 Bug 或变更说明）。")

    col1, col2 = st.columns(2)
    if col1.button("通过并生成测试报告", type="primary"):
        try:
            _advance(graph, llm, values, PHASE_REPORT)
            st.session_state.phase = PHASE_REPORT
            st.rerun()
        except Exception as exc:
            st.error(f"生成测试报告失败：{exc}")

    if col2.button("重新生成回归清单"):
        try:
            _rerun_current_phase(graph, llm, PHASE_REGRESSION)
            st.rerun()
        except Exception as exc:
            st.error(f"重新生成失败：{exc}")

    _render_artifact_panel(values)


def _render_report_page(graph: Any, llm: ChatOpenAI) -> None:
    st.subheader("11. 测试报告")
    config = _build_config(llm)
    values = _get_state_values(graph, config)
    report = str(values.get("test_report", "")).strip()
    conclusion = str(values.get("test_report_conclusion", "")).strip()
    basis = str(values.get("test_report_basis", "")).strip()

    if conclusion:
        if conclusion == "通过":
            st.success(f"总体结论：{conclusion}｜{basis}")
        elif conclusion == "不通过":
            st.error(f"总体结论：{conclusion}｜{basis}")
        else:
            st.warning(f"总体结论：{conclusion}｜{basis}")

    if report:
        st.markdown(report)
        st.download_button(
            label="下载测试报告（Markdown）",
            data=report,
            file_name=dated_file_name("测试报告"),
            mime="text/markdown",
            type="primary",
        )
    else:
        st.warning("尚未生成报告内容。")

    excel_output_path = str(values.get("excel_output_path", "")).strip()
    if excel_output_path and Path(excel_output_path).exists():
        output_path = Path(excel_output_path)
        st.download_button(
            label="下载测试用例 Excel",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    _render_artifact_panel(values)

    if st.button("生成新的测试用例"):
        _reset_flow()
        st.rerun()


def _render_phase_nav(phase: str) -> None:
    labels = [label for _, label in PHASE_ORDER]
    phase_index_map = {phase_name: idx for idx, (phase_name, _) in enumerate(PHASE_ORDER)}
    current_index = phase_index_map.get(phase, 0)

    # 阶段较多，横向排不下时按接近当前阶段的窗口展示。
    window_start = max(current_index - 3, 0)
    window_end = min(window_start + 7, len(labels))
    window_start = max(window_end - 7, 0)

    cols = st.columns(window_end - window_start)
    for offset, idx in enumerate(range(window_start, window_end)):
        col = cols[offset]
        if idx < current_index:
            col.success(f"已完成\n{idx + 1}. {labels[idx]}")
        elif idx == current_index:
            col.info(f"当前\n{idx + 1}. {labels[idx]}")
        else:
            col.caption(f"待执行\n{idx + 1}. {labels[idx]}")


def main() -> None:
    st.set_page_config(page_title="AI 测试用例生成器", layout="wide")
    st.title("AI 测试用例生成器（qa-skills 流水线）")

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
        if st.session_state.project_name:
            st.caption(f"当前项目：{st.session_state.project_name}")
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

    if phase == PHASE_UPLOAD:
        _render_upload_page(graph, llm)
    elif phase == PHASE_REQUIREMENT:
        _render_requirement_page(graph, llm)
    elif phase == PHASE_STRATEGY:
        _render_strategy_page(graph, llm)
    elif phase == PHASE_TEST_POINTS:
        _render_test_points_page(graph, llm)
    elif phase == PHASE_OUTLINE:
        _render_outline_page(graph, llm)
    elif phase == PHASE_CASE:
        _render_case_page(graph, llm)
    elif phase == PHASE_REVIEW:
        _render_review_page(graph, llm)
    elif phase == PHASE_EXECUTION:
        _render_execution_page(graph, llm)
    elif phase == PHASE_BUGS:
        _render_bugs_page(graph, llm)
    elif phase == PHASE_REGRESSION:
        _render_regression_page(graph, llm)
    elif phase == PHASE_REPORT:
        _render_report_page(graph, llm)
    else:
        _reset_flow()
        st.rerun()


if __name__ == "__main__":
    main()
