from pathlib import Path
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    analyze_requirement_node,
    export_excel_node,
    extract_test_points_node,
    generate_cases_node,
    generate_outline_node,
)
from .state import TestCaseState


# Checkpoint 落盘位置（data/ 已被 .gitignore 忽略）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"

workflow = StateGraph(TestCaseState)

workflow.add_node("analyze_requirement_node", analyze_requirement_node)
workflow.add_node("extract_test_points_node", extract_test_points_node)
workflow.add_node("generate_outline_node", generate_outline_node)
workflow.add_node("generate_cases_node", generate_cases_node)
workflow.add_node("export_excel_node", export_excel_node)

workflow.add_edge(START, "analyze_requirement_node")
workflow.add_edge("analyze_requirement_node", "extract_test_points_node")
workflow.add_edge("extract_test_points_node", "generate_outline_node")
workflow.add_edge("generate_outline_node", "generate_cases_node")
workflow.add_edge("generate_cases_node", "export_excel_node")
workflow.add_edge("export_excel_node", END)


def _build_checkpointer() -> SqliteSaver:
    """构建基于 SQLite 的 checkpointer。

    MemorySaver 会把所有会话的检查点堆在进程内存里且永不释放，
    进程重启即全部丢失；改用 SQLite 可持久化并解除内存压力。

    这里自行持有连接而非用 SqliteSaver.from_conn_string()：后者是上下文管理器，
    退出时会关闭连接，不适合长期存活的应用。连接方式与 LangGraph 内部一致
    （check_same_thread=False 以支持跨线程访问），并额外设置 busy timeout
    降低并发写冲突概率。
    """
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(CHECKPOINT_DB_PATH),
        check_same_thread=False,
        timeout=30.0,
    )
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def create_workflow() -> Any:
    """创建并返回编译后的 LangGraph 工作流。"""
    return workflow.compile(
        checkpointer=_build_checkpointer(),
        interrupt_before=[
            "extract_test_points_node",
            "generate_outline_node",
            "generate_cases_node",
            "export_excel_node",
        ],
    )
