"""LangGraph 全局状态定义。

字段按 qa-skills 八阶段流水线组织（见 qa/SKILL.md）：

    0 探索（旁路，本项目不做） → 1 需求理解 → 2 测试策略 → 3 用例编写
    → 4 用例审查 → 5 执行 → 6 Bug 分析 → 7 回归 → 8 收尾报告

每个阶段既有「AI 产出」也有「人工修订产出」（modified_* 前缀）。
人工修订优先于 AI 原产的判定一律 `is None` 而非 `or`——空列表是合法的人工
结果（用户删光了所有用例），用 or 会静默回退到 AI 原产，使人审形同虚设。
"""

from typing import Any, TypedDict


class TestCaseState(TypedDict):
    """LangGraph 全局状态：用于在多阶段协作流程中共享测试生成上下文。"""

    # ---------------- 输入与解析 ----------------

    # 原始文档内容（例如 PRD、需求说明、接口文档等未结构化文本）
    document: str

    # 结构化文档（由前期解析模块输出，通常为分层字典结构）
    structured_doc: dict

    # 当前上传文档 ID（用于 RAG 过滤）
    doc_id: str

    # 项目名，决定落盘产物目录（artifacts/{project_name}/）
    project_name: str

    # ---------------- 阶段 1：需求理解 ----------------

    # 需求分析报告（对业务目标、范围、约束、风险点的分析结论）
    requirement_analysis: str

    # ---------------- 阶段 2：测试策略 ----------------

    # 测试策略：{"risk_map": [...], "functional_scope": [...], "type_scope": [...], "summary": str}
    test_strategy: dict[str, Any]

    # 人工修订后的测试策略
    modified_test_strategy: dict[str, Any]

    # ---------------- 阶段 3：测试点与大纲 ----------------

    # 测试点列表（从需求中提取出的可验证要点，每一项为一个字典）
    test_points: list[dict]

    # 测试大纲（按模块/场景组织的测试设计框架，每一项为一个字典）
    test_outline: list[dict]

    # 人工修改后的测试大纲（人工评审与调整后的大纲结果）
    modified_outline: list[dict]

    # ---------------- 阶段 3/4：测试用例 ----------------

    # 生成的测试用例（模型根据大纲产出的测试用例集合，每一项为一个字典）
    test_cases: list[dict]

    # 人工修改后的测试用例（人工补充、修订、确认后的最终用例集合）
    modified_test_cases: list[dict]

    # ---------------- 阶段 4：用例审查 ----------------

    # 审查发现（ReviewFinding 字典列表）；空列表表示审查未发现问题
    review_findings: list[dict]

    # 审查结论摘要
    review_summary: str

    # ---------------- 阶段 5：执行 ----------------

    # 执行策略：manual（手动）/ api（接口脚本）/ e2e（UI 脚本）/ skip（跳过）
    execution_mode: str

    # 执行记录（ExecutionRecord 字典列表）：TC 编号 × 结果
    execution_records: list[dict]

    # ---------------- 阶段 6：Bug 分析 ----------------

    # Bug 条目（BugItem 字典列表）
    bug_items: list[dict]

    # Bug 分析摘要（确认数、排除数及理由）
    bug_summary: str

    # ---------------- 阶段 7：回归 ----------------

    # 回归清单（RegressionItem 字典列表）
    regression_items: list[dict]

    # 回归摘要（三级数量、覆盖模块、不在范围的原因）
    regression_summary: str

    # ---------------- 阶段 8：收尾报告 ----------------

    # 测试报告正文（Markdown）
    test_report: str

    # 总体结论：通过 / 有条件通过 / 不通过
    test_report_conclusion: str

    # 一句话结论依据
    test_report_basis: str

    # ---------------- 质量门禁 ----------------

    # markmap ↔ schema.yaml 校验是否通过（含 Critical/High 风险覆盖门禁）
    schema_validation_passed: bool

    # 校验器原始输出，供前端展示
    schema_validation_output: str

    # ---------------- 落盘与可观测性 ----------------

    # 各节点检索日志（用于可观测性与前端展示）
    retrieval_logs: list[dict]

    # 落盘产物目录（绝对路径字符串）
    artifacts_dir: str

    # 已落盘的产物文件路径列表（绝对路径字符串）
    artifact_files: list[str]

    # 导出的 Excel 文件路径
    excel_output_path: str
