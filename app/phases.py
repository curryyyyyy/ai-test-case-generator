"""工作流阶段的单一事实来源。

重构前「一个阶段」的信息散落在 app.py 的八处：常量、PHASE_ORDER、
invoke_count_map、进度文案、_phase_artifact_ready、_prime_editors、
_rerun_current_phase 分支、main 里的 if-elif 分发。
新增或调整阶段必须八处同步改，漏一处就会出现「页面能进但拿不到数据」
这类难查的问题。

现在阶段相关的一切都集中在本文件的一张表里，
app.py 只负责“按表渲染”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# 阶段标识常量（保持既有取值，避免历史 checkpoint 失效）
PHASE_UPLOAD = "upload"
PHASE_REQUIREMENT = "requirement_review"
PHASE_TEST_POINTS = "test_points_review"
PHASE_OUTLINE = "outline_review"
PHASE_CASE = "case_review"
PHASE_DOWNLOAD = "download"


@dataclass(frozen=True)
class PhaseSpec:
    """一个阶段的全部元信息。

    attributes:
        key: 阶段标识，存于 session_state.phase
        label: 顶部进度条上显示的名字
        artifact_key: 判断该阶段产出物是否就绪的状态字段
        editor_key: 人工审核结果在 session_state 中的键（无编辑器则为空）
        prime: 把工作流状态同步到编辑器控件的函数
        to_rows: 把产出物转成表格行的函数
        from_rows: 把表格行还原成结构化数据的函数
        node_name: 重生成时调用的节点函数名（无则不可重生成）
        rerun_label: 重生成时的进度提示
        invoke_count: 从头回放到本阶段需要的 invoke 次数
        retrieval_log_key: 展示检索依据时使用的日志 key
    """

    key: str
    label: str
    artifact_key: str = ""
    editor_key: str = ""
    prime: Callable[[dict[str, Any]], None] | None = None
    node_name: str = ""
    rerun_label: str = ""
    invoke_count: int = 0
    retrieval_log_key: str = ""


def _prime_text(editor_key: str, artifact_key: str) -> Callable[[dict[str, Any]], None]:
    def _prime(values: dict[str, Any]) -> None:
        import streamlit as st

        st.session_state[editor_key] = str(values.get(artifact_key, ""))

    return _prime


def _prime_first(artifact_key: str, editor_key: str, to_rows: Callable[[list[Any]], list[dict[str, Any]]]) -> Callable[[dict[str, Any]], None]:
    def _prime(values: dict[str, Any]) -> None:
        import streamlit as st

        st.session_state[editor_key] = to_rows(values.get(artifact_key, []) or [])

    return _prime


# 表格转换函数在下方定义后会注入；这里先用占位引用，避免循环导入。
PHASES: dict[str, PhaseSpec] = {}


@dataclass(frozen=True)
class WorkflowPhase:
    """按顺序排列的阶段序列（含无产出的上传/下载阶段）。"""

    order: tuple[str, ...] = (
        PHASE_UPLOAD,
        PHASE_REQUIREMENT,
        PHASE_TEST_POINTS,
        PHASE_OUTLINE,
        PHASE_CASE,
        PHASE_DOWNLOAD,
    )


def phase_order() -> list[tuple[str, str]]:
    """返回 [(阶段标识, 显示名)]，供进度条渲染。"""
    return [(key, PHASES[key].label) for key in WorkflowPhase().order]


def get_phase(key: str) -> PhaseSpec:
    spec = PHASES.get(key)
    if spec is None:
        raise ValueError(f"未知阶段: {key}")
    return spec


def artifact_ready(phase_key: str, values: dict[str, Any]) -> bool:
    """判断工作流是否已生成目标阶段的产出物。"""
    spec = PHASES.get(phase_key)
    if spec is None or not spec.artifact_key:
        return False
    return bool(values.get(spec.artifact_key))


def invoke_count(phase_key: str) -> int:
    return get_phase(phase_key).invoke_count


def replay_labels() -> list[str]:
    """回放过程中逐次 invoke 的进度文案，按阶段顺序生成。"""
    return [
        PHASES[key].rerun_label
        for key in WorkflowPhase().order
        if PHASES[key].invoke_count > 0
    ]


def rebuild_registry(
    to_rows_map: dict[str, Callable[[list[Any]], list[dict[str, Any]]]],
) -> None:
    """注入表格转换函数并构建阶段表。

    放在函数里是为了让 app.py 的表格转换函数定义完之后再建表，
    避免 app.py 与 phases.py 之间出现循环导入。
    """
    PHASES.clear()
    PHASES.update(
        {
            PHASE_UPLOAD: PhaseSpec(key=PHASE_UPLOAD, label="上传文档"),
            PHASE_REQUIREMENT: PhaseSpec(
                key=PHASE_REQUIREMENT,
                label="需求分析",
                artifact_key="requirement_analysis",
                editor_key="requirement_editor_text",
                prime=_prime_text("requirement_editor_text", "requirement_analysis"),
                node_name="analyze_requirement_node",
                rerun_label="正在生成需求分析",
                invoke_count=1,
                retrieval_log_key="analyze_requirement",
            ),
            PHASE_TEST_POINTS: PhaseSpec(
                key=PHASE_TEST_POINTS,
                label="测试点提取",
                artifact_key="test_points",
                editor_key="test_points_table",
                prime=_prime_first("test_points", "test_points_table", to_rows_map["test_points"]),
                node_name="extract_test_points_node",
                rerun_label="正在提取测试点",
                invoke_count=2,
                retrieval_log_key="extract_test_points",
            ),
            PHASE_OUTLINE: PhaseSpec(
                key=PHASE_OUTLINE,
                label="测试大纲",
                artifact_key="test_outline",
                editor_key="outline_table",
                prime=_prime_first("test_outline", "outline_table", to_rows_map["outline"]),
                node_name="generate_outline_node",
                rerun_label="正在生成测试大纲",
                invoke_count=3,
                retrieval_log_key="generate_outline",
            ),
            PHASE_CASE: PhaseSpec(
                key=PHASE_CASE,
                label="测试用例",
                artifact_key="test_cases",
                editor_key="test_cases_table",
                prime=_prime_first("test_cases", "test_cases_table", to_rows_map["cases"]),
                node_name="generate_cases_node",
                rerun_label="正在生成测试用例",
                invoke_count=4,
                retrieval_log_key="generate_cases_requirement",
            ),
            PHASE_DOWNLOAD: PhaseSpec(key=PHASE_DOWNLOAD, label="下载结果"),
        }
    )
