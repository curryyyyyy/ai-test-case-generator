"""Prompt 加载、LLM 调用与结构化输出封装。

对应 qa-skills 的阶段划分，每个函数是一个阶段的执行入口：

    阶段1 analyze_requirement_skill   → 需求模型（渲染为 Markdown 供人审）
    阶段2 test_strategy_skill         → Risk Map + 两域范围决策
    阶段3 extract_test_points_skill   → 测试点（挂 risk_ref）
    阶段3 generate_outline_skill      → 测试大纲（决定 TC 模块号）
    阶段3 generate_cases_skill        → 测试用例（case-format 硬约束）
    阶段4 review_cases_skill          → 用例审查发现
    阶段6 bug_analysis_skill          → Bug 条目
    阶段7 regression_skill            → 回归清单
    阶段8 test_report_skill           → 测试报告

结构化输出走三层兜底：`with_structured_output` → JSON 文本解析 → LLM 修复 JSON。
弱模型与不兼容接口下，第二三层是主要通路，不要随意删除。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from workflow.schemas import (
    BugItem,
    RegressionItem,
    ReviewFinding,
    RiskItem,
    ScopeDecision,
    TestCase,
    TestOutline,
    TestPoint,
)


# ---------------------------------------------------------------------------
# 输出契约
# ---------------------------------------------------------------------------


class RoleItem(BaseModel):
    """需求模型中的角色条目。"""

    role: str = Field(description="角色名")
    can: str = Field(description="该角色能做什么")
    cannot: str = Field(description="该角色不能做什么")


class RuleItem(BaseModel):
    """需求模型中的业务规则，强制带文档依据。"""

    rule: str = Field(description="规则描述")
    evidence_source: str = Field(description="依据：需求文档的章节标题")


class OpenQuestion(BaseModel):
    """待澄清问题，必须给建议选项而不是开放式提问。"""

    question: str = Field(description="问题")
    suggestion: str = Field(description="建议选项或建议取值，便于用户一句话裁决")


class RequirementModelOutput(BaseModel):
    """需求分析结构化输出，字段对齐 qa-skills requirement-analysis 的十字段 Schema。"""

    goal: str = Field(description="需求目标（一句话）+ 成功标准")
    scope_in: list[str] = Field(default_factory=list, description="功能范围内")
    scope_out: list[str] = Field(default_factory=list, description="明确的非目标")
    roles: list[RoleItem] = Field(default_factory=list, description="角色与权限")
    inputs: list[str] = Field(default_factory=list, description="输入：来源、格式、约束")
    outputs: list[str] = Field(default_factory=list, description="输出：去向、格式、消费方")
    states: list[str] = Field(
        default_factory=list,
        description="状态与流转，句式：状态A --事件--> 状态B",
    )
    rules: list[RuleItem] = Field(default_factory=list, description="业务规则，每条带依据")
    exceptions: list[str] = Field(default_factory=list, description="异常场景与预期行为")
    dependencies: list[str] = Field(default_factory=list, description="上下游依赖")
    open_questions: list[OpenQuestion] = Field(default_factory=list, description="待澄清问题")


class TestStrategyOutput(BaseModel):
    """测试策略结构化输出：Risk Map + 功能域六轴 + 类型域十轴。"""

    risk_map: list[RiskItem] = Field(description="风险清单，每条必须带证据")
    functional_scope: list[ScopeDecision] = Field(description="功能域六轴决策")
    type_scope: list[ScopeDecision] = Field(description="类型域十轴决策，十轴全答")
    summary: str = Field(description="策略摘要：覆盖重点、已知盲区、待裁决事项")


class RequirementAnalysisOutput(BaseModel):
    """兼容旧版：需求分析文本输出。"""

    requirement_analysis: str = Field(description="需求分析报告")


class TestPointListOutput(BaseModel):
    """测试点列表结构化输出。"""

    test_points: list[TestPoint] = Field(description="根据需求分析提取出的测试点列表")


class TestOutlineListOutput(BaseModel):
    """测试大纲列表结构化输出。"""

    test_outline: list[TestOutline] = Field(description="按模块组织的测试大纲列表")


class TestCaseListOutput(BaseModel):
    """测试用例列表结构化输出。"""

    test_cases: list[TestCase] = Field(description="结构化测试用例列表")


class ReviewFindingsOutput(BaseModel):
    """用例审查输出。"""

    findings: list[ReviewFinding] = Field(description="审查发现列表")
    summary: str = Field(description="审查结论摘要")


class BugListOutput(BaseModel):
    """Bug 分析输出。"""

    bugs: list[BugItem] = Field(description="确认的 Bug 条目")
    summary: str = Field(description="分析摘要：确认数、排除数及理由")


class RegressionListOutput(BaseModel):
    """回归清单输出。"""

    items: list[RegressionItem] = Field(description="回归清单条目")
    summary: str = Field(description="摘要：三级数量、覆盖模块、不在范围的原因")


class TestReportOutput(BaseModel):
    """测试报告输出。"""

    report: str = Field(description="完整 Markdown 报告正文")
    machine_summary: str = Field(description="机读摘要片段（YAML 文本）")
    conclusion: str = Field(description="总体结论：通过 / 有条件通过 / 不通过")
    conclusion_basis: str = Field(description="一句话结论依据")


ModelT = TypeVar("ModelT", bound=BaseModel)
PROMPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 渲染：结构化产物 → 人读 Markdown（落盘与前端展示）
# ---------------------------------------------------------------------------


def render_requirement_model(model: RequirementModelOutput, project_name: str = "") -> str:
    """把结构化需求模型渲染为 Markdown（落盘 `需求模型.md`）。

    渲染是**单向派生**：结构化对象是抽取产物，Markdown 是其可读视图。
    """
    title = f"# 需求模型：{project_name}\n" if project_name else "# 需求模型\n"
    lines: list[str] = [title]

    lines.append("## 1. 目标与成功标准\n")
    lines.append(model.goal.strip() or "（未抽取到目标描述）")
    lines.append("")

    lines.append("## 2. 功能范围\n")
    lines.append("### 2.1 范围内\n")
    if model.scope_in:
        lines.extend(f"- {item}" for item in model.scope_in)
    else:
        lines.append("- （无）")
    lines.append("")
    lines.append("### 2.2 非目标（明确不做）\n")
    if model.scope_out:
        lines.extend(f"- {item}" for item in model.scope_out)
    else:
        lines.append("- （文档未显式声明非目标）")
    lines.append("")

    lines.append("## 3. 角色与权限\n")
    if model.roles:
        lines.append("| 角色 | 能做什么 | 不能做什么 |")
        lines.append("|------|---------|-----------|")
        for role in model.roles:
            lines.append(f"| {role.role} | {role.can} | {role.cannot} |")
    else:
        lines.append("（未抽取到角色信息）")
    lines.append("")

    lines.append("## 4. 输入与输出\n")
    if model.inputs:
        lines.append("**输入**\n")
        lines.extend(f"- {item}" for item in model.inputs)
    if model.outputs:
        lines.append("\n**输出**\n")
        lines.extend(f"- {item}" for item in model.outputs)
    if not model.inputs and not model.outputs:
        lines.append("（未抽取到输入输出信息）")
    lines.append("")

    lines.append("## 5. 状态与流转\n")
    if model.states:
        lines.extend(f"- {item}" for item in model.states)
    else:
        lines.append("- （未抽取到状态流转）")
    lines.append("")

    lines.append("## 6. 业务规则\n")
    if model.rules:
        lines.append("| # | 规则 | 依据（章节） | 证据等级 |")
        lines.append("|---|------|-------------|---------|")
        for index, rule in enumerate(model.rules, start=1):
            # 本阶段无代码仓库，规则依据一律为文档证据 E1。
            lines.append(f"| R{index} | {rule.rule} | {rule.evidence_source} | E1 |")
    else:
        lines.append("（未抽取到业务规则）")
    lines.append("")

    lines.append("## 7. 异常场景\n")
    if model.exceptions:
        lines.extend(f"- {item}" for item in model.exceptions)
    else:
        lines.append("- （未抽取到异常场景）")
    lines.append("")

    lines.append("## 8. 依赖\n")
    if model.dependencies:
        lines.extend(f"- {item}" for item in model.dependencies)
    else:
        lines.append("- （未抽取到依赖）")
    lines.append("")

    lines.append("## 9. 待澄清问题（open_questions）\n")
    if model.open_questions:
        lines.append("| # | 问题 | 建议 | 裁决 |")
        lines.append("|---|------|------|------|")
        for index, item in enumerate(model.open_questions, start=1):
            lines.append(f"| Q{index} | {item.question} | {item.suggestion} | 待裁决 |")
    else:
        lines.append("需求已足够清晰，无需澄清。")
    lines.append("")

    return "\n".join(lines)


def render_test_strategy(strategy: dict[str, Any], project_name: str = "") -> str:
    """把结构化测试策略渲染为 Markdown（落盘 `测试策略.md`）。"""
    title = f"# 测试策略：{project_name}\n" if project_name else "# 测试策略\n"
    lines: list[str] = [title]

    lines.append(
        "> **范围假设声明**：本策略限于系统级黑盒（UI / API / 手动 / 专项移交）。"
        "单元与集成测试属开发侧职责，本策略的评级与覆盖结论以该层已有保障为前提。\n"
    )

    lines.append("## 1. 策略摘要\n")
    lines.append(str(strategy.get("summary", "")).strip() or "（无摘要）")
    lines.append("")

    risk_map = strategy.get("risk_map", [])
    lines.append("## 2. Risk Map\n")
    if risk_map:
        lines.append(
            "| 编号 | 功能点 | 维度 | Impact | Likelihood | 等级 | 依据 | 说明 |"
        )
        lines.append("|------|--------|------|--------|-----------|------|------|------|")
        for risk in risk_map:
            evidence = risk.get("evidence", {}) if isinstance(risk, dict) else {}
            source = ""
            if isinstance(evidence, dict):
                source = str(evidence.get("source", ""))
            lines.append(
                f"| {risk.get('id', '')} | {risk.get('feature', '')} | "
                f"{risk.get('dimension', '')} | {risk.get('impact', '')} | "
                f"{risk.get('likelihood', '')} | **{risk.get('level', '')}** | "
                f"{source} | {risk.get('rationale', '')} |"
            )
    else:
        lines.append("（未识别到风险条目）")
    lines.append("")

    def _render_scope(scope_title: str, scope_key: str) -> None:
        lines.append(f"## {scope_title}\n")
        items = strategy.get(scope_key, [])
        if not items:
            lines.append("（无决策条目）\n")
            return
        lines.append("| 轴 | 决策 | 深度 | 理由 | 信号 | 关联风险 |")
        lines.append("|---|------|------|------|------|---------|")
        for item in items:
            signals = item.get("signals", []) if isinstance(item, dict) else []
            risk_refs = item.get("risk_refs", []) if isinstance(item, dict) else []
            lines.append(
                f"| `{item.get('axis', '')}` | {item.get('decision', '')} | "
                f"{item.get('depth', '')} | {item.get('rationale', '')} | "
                f"{'；'.join(str(s) for s in signals)} | "
                f"{'、'.join(str(r) for r in risk_refs)} |"
            )
        lines.append("")

    _render_scope("3. 功能域范围", "functional_scope")
    _render_scope("4. 类型域范围（十轴全答）", "type_scope")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 基础设施：Prompt 加载与结构化输出兜底
# ---------------------------------------------------------------------------


def _load_prompt_template(file_name: str) -> str:
    prompt_path = PROMPT_DIR / file_name
    return prompt_path.read_text(encoding="utf-8").strip()


def _extract_text_content(content: Any) -> str:
    """兼容不同模型返回格式，提取纯文本内容。"""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()

    return str(content).strip()


def _extract_json_object(raw_text: str) -> str:
    """从模型输出中提取 JSON 对象文本。"""
    text = raw_text.strip()

    code_block_match = re.search(
        r"```(?:json)?\s*(\{[\s\S]*\})\s*```",
        text,
        re.IGNORECASE,
    )
    if code_block_match:
        return code_block_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def _sanitize_json_text(json_text: str) -> str:
    """对常见脏输出做轻量清洗，尽量不改动合法 JSON。"""
    sanitized = json_text.strip()
    if not sanitized:
        return sanitized

    sanitized = sanitized.replace("\u201c", '"').replace("\u201d", '"')
    sanitized = sanitized.replace("\u2018", "'").replace("\u2019", "'")
    sanitized = sanitized.replace("\ufeff", "")

    lines = sanitized.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        if '"' not in line or ":" not in line:
            cleaned_lines.append(line)
            continue

        colon_index = line.find(":")
        first_quote_after_colon = line.find('"', colon_index)
        last_quote = line.rfind('"')
        if first_quote_after_colon == -1 or last_quote <= first_quote_after_colon:
            cleaned_lines.append(line)
            continue

        value = line[first_quote_after_colon + 1 : last_quote]
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        cleaned_lines.append(
            f"{line[:first_quote_after_colon + 1]}{escaped_value}{line[last_quote:]}"
        )

    return "\n".join(cleaned_lines)


async def _repair_json_with_llm(
    llm: BaseChatModel,
    schema: type[ModelT],
    broken_json_text: str,
) -> str:
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    repair_prompt = (
        "你是 JSON 修复器。"
        "请将下面的内容修复为一个严格合法的 JSON 对象。"
        "不要输出任何解释、不要使用 Markdown 代码块。"
        "必须满足给定 JSON Schema。"
        f"\nJSON Schema:\n{schema_text}"
        f"\n待修复内容:\n{broken_json_text}"
    )
    response = await llm.ainvoke([HumanMessage(content=repair_prompt)])
    repaired_raw = _extract_text_content(getattr(response, "content", response))
    return _extract_json_object(repaired_raw)


async def _invoke_json_fallback(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    schema: type[ModelT],
    inputs: dict[str, Any],
    skill_name: str,
) -> ModelT:
    """当结构化输出不兼容时，降级为 JSON 文本输出并手动解析。"""
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages = prompt.format_messages(**inputs)
    messages.append(
        HumanMessage(
            content=(
                "<output_format>"
                "请仅输出一个合法 JSON 对象，不要输出额外解释，不要使用 Markdown 代码块。"
                f"输出必须满足以下 JSON Schema: {schema_text}"
                "</output_format>"
            )
        )
    )

    response = await llm.ainvoke(messages)
    raw_text = _extract_text_content(getattr(response, "content", response))
    json_text = _extract_json_object(raw_text)
    sanitized_json_text = _sanitize_json_text(json_text)

    try:
        return schema.model_validate_json(sanitized_json_text)
    except ValidationError:
        try:
            return schema.model_validate(json.loads(sanitized_json_text))
        except Exception as exc:
            try:
                repaired_json_text = await _repair_json_with_llm(
                    llm=llm,
                    schema=schema,
                    broken_json_text=sanitized_json_text,
                )
                try:
                    return schema.model_validate_json(repaired_json_text)
                except ValidationError:
                    return schema.model_validate(json.loads(repaired_json_text))
            except Exception as repair_exc:
                raise ValueError(
                    f"{skill_name} 降级 JSON 解析失败。原始输出: {raw_text}"
                ) from repair_exc


async def _invoke_structured_output(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    schema: type[ModelT],
    inputs: dict[str, Any],
    skill_name: str,
) -> ModelT:
    """优先走 with_structured_output，失败时自动降级。"""
    structured_llm = llm.with_structured_output(
        schema,
        method="json_mode",
        include_raw=True,
    )
    chain = prompt | structured_llm

    try:
        response = await chain.ainvoke(inputs)
    except Exception:
        return await _invoke_json_fallback(
            llm=llm,
            prompt=prompt,
            schema=schema,
            inputs=inputs,
            skill_name=skill_name,
        )

    if isinstance(response, schema):
        return response

    if not isinstance(response, dict):
        return await _invoke_json_fallback(
            llm=llm,
            prompt=prompt,
            schema=schema,
            inputs=inputs,
            skill_name=skill_name,
        )

    parsed = response.get("parsed")
    if parsed is None:
        return await _invoke_json_fallback(
            llm=llm,
            prompt=prompt,
            schema=schema,
            inputs=inputs,
            skill_name=skill_name,
        )

    return cast(ModelT, parsed)


# ---------------------------------------------------------------------------
# 阶段 1：需求理解
# ---------------------------------------------------------------------------


async def analyze_requirement_skill(
    llm: BaseChatModel,
    structured_doc: dict[str, Any],
    retrieved_context: str = "",
    project_name: str = "",
) -> str:
    """分析结构化文档，生成需求模型并渲染为 Markdown。

    返回 Markdown 文本而非结构对象：需求模型在前端以文本框人工编辑，
    文本即权威源（结构化对象只在生成时用作抽取中间态）。
    """
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("analyze_requirement_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=RequirementModelOutput,
        inputs={
            "structured_doc": json.dumps(
                structured_doc,
                ensure_ascii=False,
                indent=2,
            ),
            "retrieved_context": retrieved_context,
        },
        skill_name="analyze_requirement_skill",
    )

    return render_requirement_model(result, project_name=project_name).strip()


# ---------------------------------------------------------------------------
# 阶段 2：测试策略
# ---------------------------------------------------------------------------


async def test_strategy_skill(
    llm: BaseChatModel,
    requirement_model: str,
    retrieved_context: str = "",
) -> dict[str, Any]:
    """基于需求模型生成测试策略（Risk Map + 两域范围决策）。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("test_strategy_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=TestStrategyOutput,
        inputs={
            "requirement_model": requirement_model,
            "retrieved_context": retrieved_context,
        },
        skill_name="test_strategy_skill",
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# 阶段 3：测试点 / 大纲 / 用例
# ---------------------------------------------------------------------------


def _format_test_strategy_for_prompt(test_strategy: dict[str, Any] | None) -> str:
    """把策略压缩成 prompt 可消费的紧凑文本。"""
    if not test_strategy:
        return "（无测试策略，按需求自行判断范围与优先级）"

    lines: list[str] = []
    risk_map = test_strategy.get("risk_map", [])
    if risk_map:
        lines.append("Risk Map：")
        for risk in risk_map:
            evidence = risk.get("evidence", {})
            source = evidence.get("source", "") if isinstance(evidence, dict) else ""
            lines.append(
                f"- {risk.get('id', '')} [{risk.get('level', '')}] "
                f"{risk.get('feature', '')}（{risk.get('dimension', '')}）"
                f"｜依据: {source}｜{risk.get('rationale', '')}"
            )
    else:
        lines.append("Risk Map：（空）")

    for key, label in (
        ("functional_scope", "功能域范围"),
        ("type_scope", "类型域范围"),
    ):
        items = test_strategy.get(key, [])
        if not items:
            continue
        lines.append(f"{label}：")
        for item in items:
            depth = item.get("depth", "")
            decision = item.get("decision", "")
            depth_part = f"/{depth}" if decision != "exclude" else ""
            risk_refs = item.get("risk_refs", [])
            risk_part = f"｜风险: {'、'.join(risk_refs)}" if risk_refs else ""
            lines.append(
                f"- {item.get('axis', '')}: {decision}{depth_part}"
                f"｜{item.get('rationale', '')}{risk_part}"
            )

    summary = str(test_strategy.get("summary", "")).strip()
    if summary:
        lines.append(f"策略摘要：{summary}")

    return "\n".join(lines)


async def extract_test_points_skill(
    llm: BaseChatModel,
    requirement_analysis: str,
    retrieved_context: str = "",
    test_strategy: dict[str, Any] | None = None,
) -> list[TestPoint]:
    """基于需求模型与测试策略提取测试点列表。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("extract_test_points_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=TestPointListOutput,
        inputs={
            "requirement_analysis": requirement_analysis,
            "test_strategy": _format_test_strategy_for_prompt(test_strategy),
            "retrieved_context": retrieved_context,
        },
        skill_name="extract_test_points_skill",
    )
    return result.test_points


async def generate_outline_skill(
    llm: BaseChatModel,
    requirement_analysis: str,
    test_points: list[dict[str, Any]],
    retrieved_context: str = "",
) -> list[TestOutline]:
    """基于测试点生成分模块测试大纲。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("generate_outline_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=TestOutlineListOutput,
        inputs={
            "requirement_analysis": requirement_analysis,
            "test_points": json.dumps(
                test_points,
                ensure_ascii=False,
                indent=2,
            ),
            "retrieved_context": retrieved_context,
        },
        skill_name="generate_outline_skill",
    )
    return result.test_outline


async def generate_cases_skill(
    llm: BaseChatModel,
    requirement_analysis: str,
    outline_for_generation: list[dict[str, Any]],
    retrieved_context: str = "",
    test_strategy: dict[str, Any] | None = None,
) -> list[TestCase]:
    """基于测试大纲生成结构化测试用例。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("generate_cases_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=TestCaseListOutput,
        inputs={
            "requirement_analysis": requirement_analysis,
            "test_strategy": _format_test_strategy_for_prompt(test_strategy),
            "outline": json.dumps(
                outline_for_generation,
                ensure_ascii=False,
                indent=2,
            ),
            "retrieved_context": retrieved_context,
        },
        skill_name="generate_cases_skill",
    )
    return result.test_cases


# ---------------------------------------------------------------------------
# 阶段 4：用例审查
# ---------------------------------------------------------------------------


async def review_cases_skill(
    llm: BaseChatModel,
    requirement_analysis: str,
    test_cases: list[dict[str, Any]],
    retrieved_context: str = "",
) -> tuple[list[ReviewFinding], str]:
    """审查用例集，返回（发现列表，结论摘要）。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("review_cases_skill.md")
    )

    # 只传入用例的审查相关字段，避免把完整正文塞满上下文。
    slim_cases = [
        {
            "case_id": case.get("case_id", ""),
            "directory": case.get("directory", ""),
            "case_level": case.get("case_level", ""),
            "test_point": case.get("test_point", ""),
            "precondition": case.get("precondition", ""),
            "steps": case.get("steps", []),
            "expected_result": case.get("expected_result", ""),
            "risk_ref": case.get("risk_ref", ""),
        }
        for case in test_cases
    ]

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=ReviewFindingsOutput,
        inputs={
            "requirement_analysis": requirement_analysis,
            "test_cases": json.dumps(slim_cases, ensure_ascii=False, indent=2),
            "retrieved_context": retrieved_context,
        },
        skill_name="review_cases_skill",
    )
    return result.findings, result.summary


# ---------------------------------------------------------------------------
# 阶段 6：Bug 分析
# ---------------------------------------------------------------------------


async def bug_analysis_skill(
    llm: BaseChatModel,
    execution_records: list[dict[str, Any]],
    requirement_analysis: str,
    retrieved_context: str = "",
) -> tuple[list[BugItem], str]:
    """对执行失败做 Bug 分析，返回（Bug 条目，分析摘要）。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("bug_analysis_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=BugListOutput,
        inputs={
            "execution_records": json.dumps(
                execution_records, ensure_ascii=False, indent=2
            ),
            "requirement_analysis": requirement_analysis,
            "retrieved_context": retrieved_context,
        },
        skill_name="bug_analysis_skill",
    )
    return result.bugs, result.summary


# ---------------------------------------------------------------------------
# 阶段 7：回归清单
# ---------------------------------------------------------------------------


async def regression_skill(
    llm: BaseChatModel,
    bug_items: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    requirement_analysis: str,
) -> tuple[list[RegressionItem], str]:
    """基于 Bug 与修复说明生成分级回归清单。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("regression_skill.md")
    )

    slim_cases = [
        {
            "case_id": case.get("case_id", ""),
            "directory": case.get("directory", ""),
            "case_level": case.get("case_level", ""),
            "test_point": case.get("test_point", ""),
            "risk_ref": case.get("risk_ref", ""),
        }
        for case in test_cases
    ]

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=RegressionListOutput,
        inputs={
            "bug_items": json.dumps(bug_items, ensure_ascii=False, indent=2),
            "test_cases": json.dumps(slim_cases, ensure_ascii=False, indent=2),
            "requirement_analysis": requirement_analysis,
        },
        skill_name="regression_skill",
    )
    return result.items, result.summary


# ---------------------------------------------------------------------------
# 阶段 8：测试报告
# ---------------------------------------------------------------------------


async def test_report_skill(
    llm: BaseChatModel,
    project_name: str,
    requirement_analysis: str,
    test_strategy: dict[str, Any] | None,
    test_cases: list[dict[str, Any]],
    execution_records: list[dict[str, Any]],
    bug_items: list[dict[str, Any]],
    regression_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总各阶段产物生成测试报告。"""
    prompt = ChatPromptTemplate.from_template(
        _load_prompt_template("test_report_skill.md")
    )

    result = await _invoke_structured_output(
        llm=llm,
        prompt=prompt,
        schema=TestReportOutput,
        inputs={
            "project_name": project_name,
            "requirement_analysis": requirement_analysis,
            "test_strategy": _format_test_strategy_for_prompt(test_strategy),
            "test_cases": json.dumps(
                [
                    {
                        "case_id": case.get("case_id", ""),
                        "directory": case.get("directory", ""),
                        "case_level": case.get("case_level", ""),
                        "test_point": case.get("test_point", ""),
                        "risk_ref": case.get("risk_ref", ""),
                    }
                    for case in test_cases
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "execution_records": json.dumps(
                execution_records, ensure_ascii=False, indent=2
            ),
            "bug_items": json.dumps(bug_items, ensure_ascii=False, indent=2),
            "regression_items": json.dumps(
                regression_items, ensure_ascii=False, indent=2
            ),
        },
        skill_name="test_report_skill",
    )
    return result.model_dump()
