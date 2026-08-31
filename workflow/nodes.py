"""LangGraph 工作流节点实现。

节点按 qa-skills 的八阶段流水线组织。每个节点的通用套路是：

    构造 query → RAG 检索 → 调用 skill → 落盘产物 → 返回结构化结果

「检索先行」不是装饰：每个阶段的产出都要能追溯到需求文档的片段，
检索日志会随 state 传到前端展示（query / 候选数 / rerank 状态 / 引用）。

落盘遵循 qa-skills 的「文件即流水线状态」：阶段间传递的是产物路径，
会话中断后凭落盘产物即可续跑。
"""

import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import Any, cast

# 注意：本文件**不要**加 `from __future__ import annotations`。
# 那会让节点签名的注解退化为字符串，LangGraph 在 add_node 时无法识别
# `config: RunnableConfig | None` 而逐个节点告警。

# 项目根目录（导出产物落盘位置）。
# 注意：这里不再做 sys.path 注入——路径统一由入口（app/app.py）或
# 包安装（pip install -e .）保证，避免各处散落的路径操作互相干扰。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from rag.retriever import (
    build_citations,
    format_retrieved_context,
    retrieve_testcase_context_with_meta,
    retrieve_context_with_meta,
)
from .state import TestCaseState
from utils.artifacts import (
    dated_file_name,
    get_artifacts_dir,
    record_artifact,
    write_artifact,
)
from utils.case_ids import assign_smoke_numbers, normalize_case_ids
from utils.excel_exporter.excel_exporter import export_test_cases_to_excel
from utils.exporters.markmap_exporter import render_markmap
from utils.exporters.schema_exporter import (
    build_case_schema,
    dump_schema_yaml,
    validate_schema,
)

from skills.test_design_skills import (  # noqa: E402
    analyze_requirement_skill,
    bug_analysis_skill,
    extract_test_points_skill,
    generate_cases_skill,
    generate_outline_skill,
    regression_skill,
    render_test_strategy,
    review_cases_skill,
    test_report_skill,
    test_strategy_skill,
)


# 检索日志保留上限。日志会随每次“重新生成”线性增长并全部进入 checkpoint，
# 不设上限会让 state 膨胀到拖垮内存与序列化开销。
RETRIEVAL_LOG_MAX_ENTRIES = 10
# 只保留前 N 条命中的全文。前端“查看全文”也只展示前 3 条，
# 其余仅留预览即可，可显著降低单条日志体积。
RETRIEVAL_LOG_FULL_TEXT_HITS = 3

MARKMAP_FILE = "测试用例_markmap.md"
SCHEMA_FILE = "测试用例.schema.yaml"


def _build_log_hits(chunks: list[Any]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_type": getattr(chunk, "doc_type", ""),
                "section_path": chunk.section_path,
                "source_name": chunk.source_name,
                "score": chunk.score,
                "text_preview": chunk.text[:300],
                "text_full": chunk.text if index < RETRIEVAL_LOG_FULL_TEXT_HITS else "",
            }
        )
    return hits


def _append_retrieval_log(
    state: TestCaseState,
    phase: str,
    query: str,
    expanded_queries: list[str],
    pre_dedup_count: int,
    post_dedup_count: int,
    chunks: list[Any],
    citations: list[dict[str, Any]],
    latency_ms: int,
    rerank_mode: str,
    rerank_enabled: bool,
    rerank_latency_ms: int,
    rerank_degraded: bool,
    rerank_degraded_reason: str,
) -> list[dict[str, Any]]:
    logs = list(state.get("retrieval_logs", []))
    logs.append(
        {
            "phase": phase,
            "query": query,
            "expanded_queries": expanded_queries,
            "top_k": len(chunks),
            "pre_dedup_count": pre_dedup_count,
            "post_dedup_count": post_dedup_count,
            "hits": _build_log_hits(chunks),
            "citations": citations,
            "latency_ms": latency_ms,
            "rerank_mode": rerank_mode,
            "rerank_enabled": rerank_enabled,
            "rerank_latency_ms": rerank_latency_ms,
            "rerank_degraded": rerank_degraded,
            "rerank_degraded_reason": rerank_degraded_reason,
        }
    )
    # 只保留最近的若干条，避免日志无界增长。
    return logs[-RETRIEVAL_LOG_MAX_ENTRIES:]


def _truncate_for_query(text: str, limit: int = 600) -> str:
    plain = str(text).strip()
    if len(plain) <= limit:
        return plain
    return plain[:limit]


def _build_retrieval_for_phase(
    state: TestCaseState,
    config: RunnableConfig | None,
    phase: str,
    query: str,
    doc_type: str = "requirement",
) -> tuple[str, list[dict[str, Any]], int]:
    doc_id = str(state.get("doc_id", "")).strip()
    if not doc_id:
        return "", list(state.get("retrieval_logs", [])), 0

    configurable = {}
    if config is not None:
        configurable = config.get("configurable", {})
    enable_multi_query = configurable.get("enable_multi_query", None)
    enable_rerank = configurable.get("enable_rerank", None)

    start = time.time()
    chunks, meta = retrieve_context_with_meta(
        query=query,
        doc_id=doc_id,
        doc_type=doc_type,
        multi_query=enable_multi_query,
        enable_rerank=enable_rerank,
    )
    elapsed_ms = int((time.time() - start) * 1000)
    citations = build_citations(chunks, limit=5)
    logs = _append_retrieval_log(
        state=state,
        phase=phase,
        query=query,
        expanded_queries=meta.expanded_queries,
        pre_dedup_count=meta.pre_dedup_count,
        post_dedup_count=meta.post_dedup_count,
        chunks=chunks,
        citations=citations,
        latency_ms=elapsed_ms,
        rerank_mode=meta.rerank_mode,
        rerank_enabled=meta.rerank_enabled,
        rerank_latency_ms=meta.rerank_latency_ms,
        rerank_degraded=meta.rerank_degraded,
        rerank_degraded_reason=meta.rerank_degraded_reason,
    )
    return format_retrieved_context(chunks), logs, elapsed_ms


def _get_llm_from_config(config: RunnableConfig | None) -> BaseChatModel:
    """Get LLM instance from LangGraph runtime config."""
    if config is None:
        raise ValueError("缺少 config，无法获取 LLM 实例。")

    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    if llm is None:
        raise ValueError(
            "请在 config['configurable']['llm'] 中传入可用的 ChatModel 实例。"
        )

    return cast(BaseChatModel, llm)


# ---------------------------------------------------------------------------
# 状态取值辅助：人工修订优先于 AI 原产
# ---------------------------------------------------------------------------


def _project_name(state: TestCaseState) -> str:
    return str(state.get("project_name", "")).strip() or "default"


def _resolve_strategy(state: TestCaseState) -> dict[str, Any] | None:
    """取生效的测试策略：人工修订优先。

    策略是 dict，空 dict 表示未修订，用 truthy 判断即可（不存在"清空策略"这种
    合法的人审结果——策略至少要有十轴决策）。
    """
    modified = state.get("modified_test_strategy")
    if modified:
        return cast(dict[str, Any], modified)
    strategy = state.get("test_strategy")
    return cast(dict[str, Any] | None, strategy) or None


def _resolve_cases(state: TestCaseState) -> list[dict[str, Any]]:
    """取生效的用例集：人工修订优先。

    这里必须用 `is None` 而不是 `or`——空列表是合法的人工审核结果
    （用户删掉了所有用例），用 or 会静默回退到 AI 原产，使人审形同虚设。
    """
    modified = state.get("modified_test_cases")
    if modified is not None:
        return cast(list[dict[str, Any]], modified)
    return cast(list[dict[str, Any]], state.get("test_cases", []))


def _write_artifacts(
    state: TestCaseState,
    files: dict[str, str],
) -> tuple[list[str], str]:
    """把多个产物写入落盘目录，返回（更新后的产物清单，落盘目录）。"""
    project_name = _project_name(state)
    artifacts_dir = get_artifacts_dir(project_name)
    artifact_files = list(state.get("artifact_files", []) or [])

    for file_name, content in files.items():
        path = write_artifact(project_name, file_name, content)
        artifact_files = record_artifact(artifact_files, path)

    return artifact_files, str(artifacts_dir)


# ---------------------------------------------------------------------------
# 阶段 1：需求理解
# ---------------------------------------------------------------------------


def analyze_requirement_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """调用需求分析 Skill，产出需求模型并落盘 `需求模型.md`。"""
    print("--- 执行需求分析 Node ---")
    llm = _get_llm_from_config(config)
    query = (
        "请提取需求目标与成功标准、功能范围与非目标、角色权限、"
        "业务规则、状态流转、异常场景、依赖与不明确项"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="analyze_requirement",
        query=query,
    )
    requirement_analysis = asyncio.run(
        analyze_requirement_skill(
            llm=llm,
            structured_doc=state["structured_doc"],
            retrieved_context=retrieved_context,
            project_name=_project_name(state),
        )
    )

    artifact_files, artifacts_dir = _write_artifacts(
        state, {"需求模型.md": requirement_analysis}
    )
    return {
        "requirement_analysis": requirement_analysis,
        "retrieval_logs": retrieval_logs,
        "artifact_files": artifact_files,
        "artifacts_dir": artifacts_dir,
    }


# ---------------------------------------------------------------------------
# 阶段 2：测试策略
# ---------------------------------------------------------------------------


def test_strategy_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """基于需求模型生成测试策略（Risk Map + 两域范围决策）。"""
    print("--- 执行测试策略 Node ---")
    llm = _get_llm_from_config(config)
    query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请检索与风险相关的描述：数据一致性、权限控制、边界约束、状态流转、"
        "资金与资损、并发与性能、外部依赖、兼容性与迁移"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="test_strategy",
        query=query,
    )
    strategy = asyncio.run(
        test_strategy_skill(
            llm=llm,
            requirement_model=state["requirement_analysis"],
            retrieved_context=retrieved_context,
        )
    )

    strategy_markdown = render_test_strategy(strategy, _project_name(state))
    artifact_files, artifacts_dir = _write_artifacts(
        state, {"测试策略.md": strategy_markdown}
    )
    return {
        "test_strategy": strategy,
        "retrieval_logs": retrieval_logs,
        "artifact_files": artifact_files,
        "artifacts_dir": artifacts_dir,
    }


# ---------------------------------------------------------------------------
# 阶段 3：测试点 / 大纲 / 用例
# ---------------------------------------------------------------------------


def extract_test_points_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """调用测试点提取 Skill，产出测试点列表（挂 risk_ref）。"""
    print("--- 执行测试点提取 Node ---")
    llm = _get_llm_from_config(config)
    query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请围绕正常流程、边界值、异常场景、权限控制和数据校验提取可测试点"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="extract_test_points",
        query=query,
    )
    test_points = asyncio.run(
        extract_test_points_skill(
            llm=llm,
            requirement_analysis=state["requirement_analysis"],
            retrieved_context=retrieved_context,
            test_strategy=_resolve_strategy(state),
        )
    )
    return {
        "test_points": [test_point.model_dump() for test_point in test_points],
        "retrieval_logs": retrieval_logs,
    }


def generate_outline_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """基于测试点生成测试大纲（模块划分决定后续用例编号）。"""
    print("--- 执行测试大纲生成 Node ---")
    llm = _get_llm_from_config(config)
    query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请根据业务模块、用户路径、状态流转和异常分支整理测试结构\n"
        f"测试点摘要:\n{_truncate_for_query(str(state['test_points']))}"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="generate_outline",
        query=query,
    )
    test_outline = asyncio.run(
        generate_outline_skill(
            llm=llm,
            requirement_analysis=state["requirement_analysis"],
            test_points=state["test_points"],
            retrieved_context=retrieved_context,
        )
    )
    return {
        "test_outline": [outline.model_dump() for outline in test_outline],
        "retrieval_logs": retrieval_logs,
    }


def generate_cases_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """基于测试大纲生成测试用例，并完成编号规范化与 markmap/schema 双轨落盘。"""
    print("--- 执行测试用例生成 Node ---")
    llm = _get_llm_from_config(config)

    outline_for_generation = state["modified_outline"]
    if not outline_for_generation:
        outline_for_generation = state["test_outline"]

    requirement_query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请检索与该功能点相关的主流程、前置条件、状态变化、校验规则和异常约束\n"
        f"测试大纲摘要:\n{_truncate_for_query(str(outline_for_generation))}"
    )
    testcase_query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请检索相似功能的测试步骤写法、前置条件表达、预期结果描述和异常场景覆盖方式\n"
        f"测试大纲摘要:\n{_truncate_for_query(str(outline_for_generation))}"
    )
    requirement_context, retrieval_logs, requirement_latency_ms = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="generate_cases_requirement",
        query=requirement_query,
        doc_type="requirement",
    )
    testcase_start = time.time()
    testcase_chunks, testcase_meta = retrieve_testcase_context_with_meta(
        query=testcase_query,
        multi_query=(
            config.get("configurable", {}).get("enable_multi_query", None)
            if config is not None
            else None
        ),
        enable_rerank=(
            config.get("configurable", {}).get("enable_rerank", None)
            if config is not None
            else None
        ),
    )
    testcase_elapsed_ms = int((time.time() - testcase_start) * 1000)
    testcase_context = format_retrieved_context(testcase_chunks)
    testcase_citations = build_citations(testcase_chunks, limit=5)
    retrieval_logs = _append_retrieval_log(
        state={**state, "retrieval_logs": retrieval_logs},
        phase="generate_cases_testcase",
        query=testcase_query,
        expanded_queries=testcase_meta.expanded_queries,
        pre_dedup_count=testcase_meta.pre_dedup_count,
        post_dedup_count=testcase_meta.post_dedup_count,
        chunks=testcase_chunks,
        citations=testcase_citations,
        latency_ms=testcase_elapsed_ms,
        rerank_mode=testcase_meta.rerank_mode,
        rerank_enabled=testcase_meta.rerank_enabled,
        rerank_latency_ms=testcase_meta.rerank_latency_ms,
        rerank_degraded=testcase_meta.rerank_degraded,
        rerank_degraded_reason=testcase_meta.rerank_degraded_reason,
    )
    combined_context = requirement_context
    if testcase_context:
        combined_context = (
            f"{requirement_context}\n\n"
            "<testcase_reference>\n"
            "以下为历史测试用例写法参考，仅用于覆盖与表达方式参考，严禁逐句照抄。\n"
            f"{testcase_context}\n"
            "</testcase_reference>"
        ).strip()

    strategy = _resolve_strategy(state)
    test_cases = asyncio.run(
        generate_cases_skill(
            llm=llm,
            requirement_analysis=state["requirement_analysis"],
            outline_for_generation=outline_for_generation,
            retrieved_context=combined_context,
            test_strategy=strategy,
        )
    )

    # 编号规范化：保证 TC 首段与模块号一致、序号连续，
    # 让界面 / Excel / markmap / schema.yaml 四处共用同一套编号。
    normalized = normalize_case_ids([case.model_dump() for case in test_cases])
    normalized = assign_smoke_numbers(normalized)

    # 双轨落盘：markmap 是人工维护源，schema.yaml 由它单向抽取。
    markmap_content = render_markmap(
        normalized,
        project_name=_project_name(state),
        requirement_analysis=state["requirement_analysis"],
        strategy=strategy,
    )
    schema_content = dump_schema_yaml(
        build_case_schema(
            normalized,
            project_name=_project_name(state),
            strategy=strategy,
            source_markmap=MARKMAP_FILE,
        )
    )
    artifact_files, artifacts_dir = _write_artifacts(
        state, {MARKMAP_FILE: markmap_content, SCHEMA_FILE: schema_content}
    )

    return {
        "test_cases": normalized,
        "retrieval_logs": retrieval_logs,
        "artifact_files": artifact_files,
        "artifacts_dir": artifacts_dir,
    }


# ---------------------------------------------------------------------------
# 阶段 4：用例审查
# ---------------------------------------------------------------------------


def review_cases_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """对人工确认后的用例集做独立审查，产出审查发现。

    审查是事后、独立的环节：写时自审无法替代，本节点也不改写用例本身，
    只输出发现交由人裁决（qa-skills 的检查点纪律：不替用户做决定）。
    """
    print("--- 执行用例审查 Node ---")
    llm = _get_llm_from_config(config)
    cases = _resolve_cases(state)
    if not cases:
        return {"review_findings": [], "review_summary": "无用例可审查。"}

    query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请检索用于核对用例覆盖率与预期正确性的规则：业务约束、边界、异常、权限"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="review_cases",
        query=query,
    )
    findings, summary = asyncio.run(
        review_cases_skill(
            llm=llm,
            requirement_analysis=state["requirement_analysis"],
            test_cases=cases,
            retrieved_context=retrieved_context,
        )
    )
    return {
        "review_findings": [finding.model_dump() for finding in findings],
        "review_summary": summary,
        "retrieval_logs": retrieval_logs,
    }


# ---------------------------------------------------------------------------
# 阶段 5-8：执行结果落盘 / Bug 分析 / 回归 / 报告
# ---------------------------------------------------------------------------


def export_excel_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """导出测试用例为 Excel 文件，并对 markmap/schema 做一次一致性校验。"""
    print("--- 执行 Excel 导出 Node ---")
    _ = config

    cases_for_export = _resolve_cases(state)
    if not cases_for_export:
        raise ValueError("没有可导出的测试用例：人工审核结果为空，请至少保留一条用例。")

    output_dir = PROJECT_ROOT / "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"test_cases_{timestamp}.xlsx"

    excel_output_path = export_test_cases_to_excel(
        test_cases=cases_for_export,
        output_file_path=output_file,
    )

    # 门禁：markmap 与 schema 的一致性 + Critical/High 风险覆盖。
    # 校验失败只记录不阻断——它是质量信号，不应让用户卡在导出环节。
    artifacts_dir = Path(
        state.get("artifacts_dir", "") or get_artifacts_dir(_project_name(state))
    )
    markmap_path = artifacts_dir / MARKMAP_FILE
    schema_path = artifacts_dir / SCHEMA_FILE
    strategy_path = artifacts_dir / "测试策略.md"
    validation_passed = True
    validation_output = "（跳过校验：未找到 markmap 或 schema 产物）"
    if markmap_path.exists() and schema_path.exists():
        validation_passed, validation_output = validate_schema(
            markmap_path=markmap_path,
            schema_path=schema_path,
            strategy_path=strategy_path if strategy_path.exists() else None,
        )

    return {
        "excel_output_path": excel_output_path,
        "schema_validation_passed": validation_passed,
        "schema_validation_output": validation_output,
    }


def bug_analysis_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """对执行失败做 Bug 分析，产出 Bug 条目（阶段 6）。

    无执行结果时不硬编 Bug——缺输入时如实返回空列表，
    让前端引导用户先录入执行结果。
    """
    print("--- 执行 Bug 分析 Node ---")
    records = list(state.get("execution_records", []) or [])
    failed = [
        record
        for record in records
        if str(record.get("status", "")) in {"失败", "阻塞"}
    ]
    if not failed:
        return {
            "bug_items": [],
            "bug_summary": "无失败记录，跳过 Bug 分析。",
        }

    llm = _get_llm_from_config(config)
    query = (
        f"{_truncate_for_query(state['requirement_analysis'])}\n"
        "请检索用于判定预期行为的业务规则与异常处理约定"
    )
    retrieved_context, retrieval_logs, _ = _build_retrieval_for_phase(
        state=state,
        config=config,
        phase="bug_analysis",
        query=query,
    )
    bugs, summary = asyncio.run(
        bug_analysis_skill(
            llm=llm,
            execution_records=failed,
            requirement_analysis=state["requirement_analysis"],
            retrieved_context=retrieved_context,
        )
    )
    return {
        "bug_items": [bug.model_dump() for bug in bugs],
        "bug_summary": summary,
        "retrieval_logs": retrieval_logs,
    }


def regression_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """基于 Bug 与修复说明生成分级回归清单（阶段 7）。"""
    print("--- 执行回归分析 Node ---")
    bug_items = list(state.get("bug_items", []) or [])
    if not bug_items:
        return {
            "regression_items": [],
            "regression_summary": "无 Bug 与变更说明，无回归触发源。",
        }

    llm = _get_llm_from_config(config)
    items, summary = asyncio.run(
        regression_skill(
            llm=llm,
            bug_items=bug_items,
            test_cases=_resolve_cases(state),
            requirement_analysis=state["requirement_analysis"],
        )
    )

    lines: list[str] = [
        f"# 回归清单 — {_project_name(state)}",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 清单",
        "",
        "| 用例 | 名称 | 级别 | 依据 |",
        "|------|------|------|------|",
    ]
    for item in items:
        lines.append(
            f"| {item.case_id} | {item.title} | **{item.level}** | {item.reason} |"
        )
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append(summary)

    artifact_files, artifacts_dir = _write_artifacts(
        state, {dated_file_name("回归清单"): "\n".join(lines)}
    )
    return {
        "regression_items": [item.model_dump() for item in items],
        "regression_summary": summary,
        "artifact_files": artifact_files,
        "artifacts_dir": artifacts_dir,
    }


def test_report_node(
    state: TestCaseState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """汇总各阶段产物生成测试报告（阶段 8）。"""
    print("--- 执行测试报告 Node ---")
    llm = _get_llm_from_config(config)
    report = asyncio.run(
        test_report_skill(
            llm=llm,
            project_name=_project_name(state),
            requirement_analysis=state["requirement_analysis"],
            test_strategy=_resolve_strategy(state),
            test_cases=_resolve_cases(state),
            execution_records=list(state.get("execution_records", []) or []),
            bug_items=list(state.get("bug_items", []) or []),
            regression_items=list(state.get("regression_items", []) or []),
        )
    )

    report_markdown = str(report.get("report", "")).strip()
    machine_summary = str(report.get("machine_summary", "")).strip()
    if machine_summary:
        report_markdown = f"{report_markdown}\n\n## 机读摘要\n\n```yaml\n{machine_summary}\n```\n"

    artifact_files, artifacts_dir = _write_artifacts(
        state, {dated_file_name("测试报告"): report_markdown}
    )
    return {
        "test_report": report_markdown,
        "test_report_conclusion": str(report.get("conclusion", "")),
        "test_report_basis": str(report.get("conclusion_basis", "")),
        "artifact_files": artifact_files,
        "artifacts_dir": artifacts_dir,
    }
