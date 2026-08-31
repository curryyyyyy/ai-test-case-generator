"""LangGraph 工作流定义：qa-skills 八阶段流水线。

节点顺序即 qa/SKILL.md 的流水线顺序：

    1 需求理解 → 2 测试策略 → 3 用例编写（测试点 → 大纲 → 用例）
    → 4 用例审查 → 5 执行 → 6 Bug 分析 → 7 回归 → 8 收尾报告

每个「AI 产出」之后都设 `interrupt_before`，把控制权交回人工审核：
AI 生成 → 人审/修订 → 继续。这不是全自动黑盒，而是人机协作流水线。

阶段 5（执行）本身不是 LLM 节点：执行结果由外部录入（自动化脚本运行或手动
执行记录），因此它的边界体现在 `bug_analysis_node` 之前的中断点上——
用户在界面上提交执行结果后，才继续走 Bug 分析与后续阶段。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .checkpoint_store import build_checkpointer
from .nodes import (
    analyze_requirement_node,
    bug_analysis_node,
    export_excel_node,
    extract_test_points_node,
    generate_cases_node,
    generate_outline_node,
    regression_node,
    review_cases_node,
    test_report_node,
    test_strategy_node,
)
from .state import TestCaseState


workflow = StateGraph(TestCaseState)

workflow.add_node("analyze_requirement_node", analyze_requirement_node)
workflow.add_node("test_strategy_node", test_strategy_node)
workflow.add_node("extract_test_points_node", extract_test_points_node)
workflow.add_node("generate_outline_node", generate_outline_node)
workflow.add_node("generate_cases_node", generate_cases_node)
workflow.add_node("review_cases_node", review_cases_node)
workflow.add_node("export_excel_node", export_excel_node)
workflow.add_node("bug_analysis_node", bug_analysis_node)
workflow.add_node("regression_node", regression_node)
workflow.add_node("test_report_node", test_report_node)

workflow.add_edge(START, "analyze_requirement_node")
workflow.add_edge("analyze_requirement_node", "test_strategy_node")
workflow.add_edge("test_strategy_node", "extract_test_points_node")
workflow.add_edge("extract_test_points_node", "generate_outline_node")
workflow.add_edge("generate_outline_node", "generate_cases_node")
workflow.add_edge("generate_cases_node", "review_cases_node")
workflow.add_edge("review_cases_node", "export_excel_node")
workflow.add_edge("export_excel_node", "bug_analysis_node")
workflow.add_edge("bug_analysis_node", "regression_node")
workflow.add_edge("regression_node", "test_report_node")
workflow.add_edge("test_report_node", END)


# 每个阶段产出前中断，等待人工审核或外部输入（执行结果）。
INTERRUPT_BEFORE = [
    "test_strategy_node",
    "extract_test_points_node",
    "generate_outline_node",
    "generate_cases_node",
    "review_cases_node",
    "export_excel_node",
    "bug_analysis_node",
    "regression_node",
    "test_report_node",
]


def create_workflow() -> Any:
    """创建并返回编译后的 LangGraph 工作流。"""
    return workflow.compile(
        checkpointer=build_checkpointer(),
        interrupt_before=INTERRUPT_BEFORE,
    )
