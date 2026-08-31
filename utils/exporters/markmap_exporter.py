"""把结构化用例渲染为 qa-skills 的 markmap 用例文件。

markmap 是**给人看、给人执行**的交付物，也是唯一人工维护源
（schema.yaml 由它单向抽取）。格式硬约束见 `core/case-format.md`：

- 文件头部导读四件套（功能简介 / 环境账号 / 术语表 / 图例）——零上下文新人能开工
- 正文零代码内部——正文禁出现 `文件:行`、SDK 符号、错误码常量
- 编号 `TC-{模块号}-{序号}`，P0 追加（SMOKE-n）
- 四段式（有 UI）/ 协作五段式（无 UI 后端功能）
- 附录与正文物理隔离

渲染前需先经 `utils.case_ids` 规范化编号，保证编号首段与模块号一致。
"""

from __future__ import annotations

from typing import Any

from utils.case_ids import group_cases_by_module


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _format_steps(steps: Any) -> str:
    """把步骤列表压成规范里的单行编号串：`1. 进入… 2. 填写…`。"""
    if isinstance(steps, str):
        return steps.strip()
    items = [_as_text(item) for item in (steps or [])]
    items = [item for item in items if item]
    return " ".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _summarize(text: str, limit: int = 300) -> str:
    """截取需求描述的前若干字符作为功能简介（不截断到半个句子之外）。"""
    plain = _as_text(text).replace("\n", " ")
    if len(plain) <= limit:
        return plain
    return plain[:limit].rstrip() + "…"


def _is_dev_collab(case: dict[str, Any]) -> bool:
    """判定执行模型：步骤里出现「请开发执行」即为测试无法独立执行的协作用例。"""
    steps = case.get("steps", [])
    joined = " ".join(str(item) for item in steps) if isinstance(steps, list) else str(steps)
    return "请开发执行" in joined


def _render_guide(
    project_name: str,
    requirement_summary: str,
) -> list[str]:
    """渲染导读区四件套。

    环境与账号这类信息本项目无从得知，按规范**不省略**——列 TODO 并写明找谁拿。
    """
    lines: list[str] = []
    lines.append(f"# {project_name} 测试用例\n")

    lines.append("## 导读\n")
    lines.append("### 功能简介\n")
    lines.append(
        requirement_summary or "TODO：补充功能简介（由需求文档第一章提取）"
    )
    lines.append("")

    lines.append("### 测试环境与账号\n")
    lines.append("| 角色 | 入口地址 | 测试账号 | 说明 |")
    lines.append("|------|---------|---------|------|")
    # 不用花括号占位：那会被 schema 校验器判定为用例占位符数据（可执行性红线）。
    lines.append("| TODO | TODO：向项目负责人索取入口地址 | TODO：向项目负责人索取账号 | — |")
    lines.append("")

    lines.append("### 术语表\n")
    lines.append("| 术语 | 说明 |")
    lines.append("|------|------|")
    lines.append("| TODO | 正文中出现的非通用词，逐条登记 |")
    lines.append("")

    lines.append("### 图例\n")
    lines.append("- `[P0]` 冒烟级：主流程核心路径，失败则功能不可用；`[P1]` 常规；`[P2]` 边界")
    lines.append("- `（SMOKE-n）` 冒烟集序号，便于快速抽取冒烟用例")
    lines.append("- `[需真机]` / `[需Mock]` / `[需专业环境]` 为可测试性标注，见附录")
    lines.append("- `[并发]` / `[可靠]` / `[安全]` / `[兼容]` / `[迁移]` / `[集成]` / `[国际化]` 为类型域轴标签")
    lines.append("")
    return lines


def _render_case(case: dict[str, Any]) -> list[str]:
    """渲染单条用例（四段式 / 协作五段式二选一）。"""
    case_id = _as_text(case.get("case_id"))
    name = _as_text(case.get("test_point")) or "未命名用例"
    level = _as_text(case.get("case_level")) or "P1"
    smoke = _as_text(case.get("smoke"))

    head = f"- **{case_id} {name}** [{level}]"
    if smoke:
        head += f"（{smoke}）"
    tags = case.get("tags", []) or []
    if tags:
        rendered_tags = " ".join(
            tag if str(tag).startswith("[") else f"[{tag}]" for tag in tags
        )
        head += f" {rendered_tags}"

    lines = [head]

    precondition = _as_text(case.get("precondition"))
    if precondition:
        lines.append(f"  - 前置条件: {precondition}")

    steps = _format_steps(case.get("steps", []))
    # 协作用例的标注由步骤原文自带（"请开发执行"），此处只补段落名。
    step_label = "操作步骤"
    if _is_dev_collab(case):
        step_label = "操作步骤（请开发执行）"
    lines.append(f"  - {step_label}: {steps}")

    expected = _as_text(case.get("expected_result"))
    expected_label = "预期结果（请开发反馈）" if _is_dev_collab(case) else "预期结果"
    lines.append(f"  - {expected_label}: {expected}")

    if _is_dev_collab(case):
        lines.append("  - 验证方式: 开发查库 / 查日志后反馈实际值")

    test_data = _as_text(case.get("test_data"))
    if test_data:
        lines.append(f"  - 测试数据: {test_data}")

    return lines


def _render_risk_appendix(
    strategy: dict[str, Any] | None,
    test_cases: list[dict[str, Any]],
) -> list[str]:
    """渲染风险覆盖对照：暴露零覆盖的 Critical/High 风险（证据链最后一环）。"""
    risk_map = (strategy or {}).get("risk_map", []) if isinstance(strategy, dict) else []
    if not risk_map:
        return []

    covered: dict[str, list[str]] = {}
    for case in test_cases:
        risk_ref = _as_text(case.get("risk_ref"))
        if risk_ref:
            covered.setdefault(risk_ref, []).append(_as_text(case.get("case_id")))

    lines: list[str] = []
    lines.append("### 风险覆盖对照\n")
    lines.append("| 风险 | 等级 | 覆盖用例 | 状态 |")
    lines.append("|------|------|---------|------|")
    for risk in risk_map:
        risk_id = _as_text(risk.get("id"))
        level = _as_text(risk.get("level"))
        case_ids = covered.get(risk_id, [])
        if case_ids:
            status = "已覆盖"
            case_text = "、".join(case_ids)
        elif level in {"Critical", "High"}:
            status = "**零覆盖（门禁失败）**"
            case_text = "—"
        else:
            status = "未覆盖"
            case_text = "—"
        lines.append(f"| {risk_id} | {level} | {case_text} | {status} |")
    lines.append("")
    return lines


def _render_testability_appendix(test_cases: list[dict[str, Any]]) -> list[str]:
    """渲染可测试性说明：带标注的用例必须落到可行动（工具/平台 + 找谁）。"""
    annotated = [case for case in test_cases if case.get("tags")]
    if not annotated:
        return []

    lines: list[str] = []
    lines.append("### 可测试性说明\n")
    lines.append("| 用例 | 标注 | 替代方案 |")
    lines.append("|------|------|---------|")
    for case in annotated:
        tags = "、".join(
            tag if str(tag).startswith("[") else f"[{tag}]" for tag in case["tags"]
        )
        lines.append(
            f"| {_as_text(case.get('case_id'))} | {tags} | "
            "TODO：向对应负责人确认具体工具与入口 |"
        )
    lines.append("")
    return lines


def render_markmap(
    test_cases: list[dict[str, Any]],
    project_name: str = "项目",
    requirement_analysis: str = "",
    strategy: dict[str, Any] | None = None,
) -> str:
    """把用例列表渲染为 markmap 格式的 Markdown 文本。"""
    lines: list[str] = []
    lines.extend(_render_guide(project_name, _summarize(requirement_analysis)))

    lines.append("---\n")
    for module_no, module_name, cases in group_cases_by_module(test_cases):
        lines.append(f"## {module_no}. {module_name}\n")
        for case in cases:
            lines.extend(_render_case(case))
        lines.append("")

    lines.append("---\n")
    lines.append("## 附录（开发技术核查清单，非测试执行项）\n")
    appendix = _render_risk_appendix(strategy, test_cases)
    appendix.extend(_render_testability_appendix(test_cases))
    if not appendix:
        appendix.append("（无代码级核查项：本轮为纯文档模式，未接入代码仓库）\n")
    lines.extend(appendix)

    return "\n".join(lines).rstrip() + "\n"
