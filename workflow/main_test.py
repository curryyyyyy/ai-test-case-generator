"""工作流验证脚本：演示「中断 → 人工干预 → 恢复」的完整语义。

依赖真实 LLM 调用（读取项目根目录 .env 的 OPENAI_API_KEY），
与 README「13. 常见问题 Q5」的说明一致：没有 API Key / 网络受限时会在 LLM 阶段失败。
"""

import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from workflow.state import TestCaseState
from workflow.workflow import create_workflow


def build_initial_state() -> TestCaseState:
    """构造用于首次运行的初始状态（字段与 app/app.py 的 _default_state 对齐）。"""
    return {
        "document": "这是用于演示的原始文档内容。",
        "structured_doc": {
            "title": "Mock需求文档",
            "modules": ["登录", "下单"],
            "version": "0.1",
        },
        "doc_id": "cli_demo_001",
        "project_name": "命令行演示项目",
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


def build_llm_from_env() -> ChatOpenAI:
    """从 .env 加载并初始化 LLM。"""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "未检测到 OPENAI_API_KEY，请在项目根目录 .env 中配置。"
            f"当前读取路径: {env_path}"
        )

    base_url = os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model="Qwen/Qwen3.5-35B-A3B",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def show_snapshot(graph, config, step: str) -> None:
    snapshot = graph.get_state(config)
    next_nodes = getattr(snapshot, "next", [])
    print(f"[{step}] 挂起于 -> {list(next_nodes) if next_nodes else 'END'}")


def main() -> None:
    """验证工作流中断、人工干预和恢复执行流程。"""
    graph = create_workflow()
    llm = build_llm_from_env()

    config = {
        "configurable": {
            "thread_id": "test_thread_001",
            "llm": llm,
        }
    }
    initial_state = build_initial_state()

    print("=== 1) 首次 invoke：执行需求分析后停在「测试策略」前 ===")
    graph.invoke(initial_state, config)
    show_snapshot(graph, config, "step1")

    print("\n=== 2) 逐阶段推进到「测试用例」产出，每个中断点可人工干预 ===")
    for index in range(4):
        result = graph.invoke(None, config)
        print(f"    invoke #{index + 1} 完成，当前产出："
              f"requirement_analysis={len(str(result.get('requirement_analysis', '')))}字符"
              if index == 0 else f"    invoke #{index + 1} 完成")
        show_snapshot(graph, config, f"step2-{index + 1}")
        pprint({k: v for k, v in result.items() if k in {
            "test_strategy", "test_points", "test_outline", "test_cases"}})

    print("\n=== 3) 模拟人工干预：修改用例（如把优先级全部调低） ===")
    snapshot = graph.get_state(config)
    current_cases = (snapshot.values or {}).get("test_cases") or []
    modified = []
    for case in current_cases:
        case = dict(case)
        case["case_level"] = "P2"
        modified.append(case)
    graph.update_state(config, {"modified_test_cases": modified})
    print(f"    已写入 {len(modified)} 条人工修改后的用例")

    print("\n=== 4) 继续执行到 END：审查 → 导出（落盘 + 校验）→ 后续阶段 ===")
    final_state = graph.invoke(None, config)
    show_snapshot(graph, config, "final")

    if isinstance(final_state, dict):
        print("\n关键字段验证：")
        pprint(
            {
                "excel_output_path": final_state.get("excel_output_path"),
                "schema_validation_passed": final_state.get("schema_validation_passed"),
                "artifacts_dir": final_state.get("artifacts_dir"),
                "test_report": (
                    f"{len(str(final_state.get('test_report', '')))}字符"
                    if final_state.get("test_report") else "（跳过执行时为占位）"
                ),
            }
        )


if __name__ == "__main__":
    main()
