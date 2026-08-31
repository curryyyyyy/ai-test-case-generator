"""工作流结构化契约。

这是 LLM 输出与后端流转之间的唯一契约层，字段口径对齐 qa-skills 的
`core/schema-extraction.md`、`core/risk-model.md`、`core/evidence.md`。

两处刻意的设计取舍：

1. **风险等级与用例优先级命名脱钩**：风险用 Critical/High/Medium/Low，
   用例用 P0/P1/P2/P3（沿用项目既有习惯），避免同名歧义。
2. **新字段一律给默认值**：历史 checkpoint 与人工编辑后的用例没有这些字段，
   给默认值可保证反序列化不炸，也允许流水线在无代码仓库的纯文档模式下运行。
"""

from typing import Literal

from pydantic import BaseModel, Field


# 风险等级（Critical/High/Medium/Low）与用例优先级（P0–P3）是两套体系，不可混用。
RiskLevel = Literal["Critical", "High", "Medium", "Low"]
Priority = Literal["P0", "P1", "P2", "P3"]
EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4"]
Confidence = Literal["high", "medium", "low"]
# fact / inference / risk / hypothesis / verified，见 core/evidence.md 第 3 节。
FindingStatus = Literal["fact", "inference", "risk", "hypothesis", "verified"]
# 范围决策：include 纳入 / exclude 明确不测 / handoff 移交专项。
Decision = Literal["include", "exclude", "handoff"]
# 深度只取三档，"不测"是范围决策而不是深度。
Depth = Literal["full", "standard", "light"]


class Evidence(BaseModel):
    """证据标注，见 core/evidence.md。

    没有证据的评级视为无效评级，因此这是风险条目的必填字段。
    """

    level: EvidenceLevel = Field(description="证据强度：E0 用户陈述 / E1 文档 / E2 代码 / E3 运行 / E4 交叉验证")
    source: str = Field(description="证据来源：文档章节标题，或 文件:行，或运行日志位置")
    confidence: Confidence = Field(default="medium", description="置信度 high/medium/low")
    status: FindingStatus = Field(default="inference", description="结论状态，文档驱动场景通常为 inference")


class RiskItem(BaseModel):
    """Risk Map 的单条风险，见 core/risk-model.md。

    Level 由 Impact × Likelihood 换算：20-25 Critical / 10-19 High / 4-9 Medium / 1-3 Low。
    """

    id: str = Field(description="风险编号，形如 R1、R2")
    feature: str = Field(description="风险所在功能点")
    dimension: str = Field(
        description=(
            "风险维度。功能域：权限/数据一致性/边界/状态流转/资损；"
            "类型域：并发/可靠/安全/性能/兼容/迁移/契约"
        )
    )
    impact: int = Field(ge=1, le=5, description="影响 1–5")
    likelihood: int = Field(ge=1, le=5, description="可能性 1–5")
    level: RiskLevel = Field(description="风险等级，必须等于 Impact × Likelihood 的换算结果")
    evidence: Evidence = Field(description="评级依据，无证据的评级无效")
    rationale: str = Field(description="一句话说明为什么是这个等级")


class ScopeDecision(BaseModel):
    """功能域或类型域的单轴决策。

    功能域六轴：functional / boundary / permission / state / data_consistency / regression
    类型域十轴：performance / security_business / reliability / concurrency /
    compatibility / accessibility / visual / i18n / migration / contract_integration
    """

    axis: str = Field(description="轴名，必须取自上述固定轴名集合")
    decision: Decision = Field(description="include 纳入 / exclude 明确不测 / handoff 移交专项")
    depth: Depth = Field(
        default="standard",
        description="深度档位：full 逐格覆盖 / standard 主干+重点异常 / light 抽样；exclude 时本字段忽略",
    )
    rationale: str = Field(description="决策理由：include 必须挂信号证据，exclude 必须给明确理由")
    signals: list[str] = Field(
        default_factory=list,
        description="决策信号：需求文档章节标题或关键描述原文",
    )
    risk_refs: list[str] = Field(default_factory=list, description="关联的风险编号，如 R1")


class TestPoint(BaseModel):
    """测试点结构化定义。"""

    name: str = Field(description="测试点名称")
    test_type: Literal["功能", "性能", "安全", "兼容性"] = Field(
        description="测试类型，可选值：功能/性能/安全/兼容性"
    )
    priority: Priority = Field(description="测试点优先级，可选值：P0/P1/P2/P3")
    risk_ref: str = Field(
        default="",
        description="关联的风险编号（R1、R2…），无对应风险时留空；存在测试策略时应尽量挂载",
    )


class TestCase(BaseModel):
    """测试用例结构化定义。

    case_id 沿用 qa-skills 的 `TC-{模块号}-{序号}` 口径，便于下游
    （审查 / 回归 / 执行）交叉引用。
    """

    case_id: str = Field(description="用例编号，格式 TC-{两位模块号}-{两位序号}，如 TC-02-03")
    directory: str = Field(description="用例所属目录（模块路径）")
    case_level: Priority = Field(description="用例级别，可选值：P0/P1/P2/P3")
    test_point: str = Field(description="测试点")
    precondition: str = Field(description="前提条件")
    steps: list[str] = Field(description="测试步骤列表，按执行顺序填写")
    expected_result: str = Field(description="预期结果")
    risk_ref: str = Field(
        default="",
        description="关联风险编号（R1…），实现风险到用例的反向追溯，供回归与门禁消费",
    )
    evidence_source: str = Field(
        default="",
        description="本用例的文档依据（章节标题或 文件:行），对应 Schema 的 evidence.source",
    )
    test_data: str = Field(
        default="",
        description="本用例使用的具体测试数据；禁止占位符，无特殊数据时留空",
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "标签：可测试性标注（需真机/需Mock/需专业环境）"
            "或类型域轴标签（并发/可靠/安全/兼容/迁移/集成/国际化）"
        ),
    )


class TestOutline(BaseModel):
    """测试大纲结构化定义。"""

    module_name: str = Field(description="模块名称")
    test_points: list[TestPoint] = Field(description="该模块下的测试点列表")


class ReviewFinding(BaseModel):
    """用例审查发现的单条问题。"""

    case_id: str = Field(description="涉及的用例编号，空字符串表示整体性问题")
    category: Literal["缺失", "冗余", "错误", "可执行性", "高风险未覆盖"] = Field(
        description="问题类别"
    )
    description: str = Field(description="问题描述")
    suggestion: str = Field(default="", description="修订建议")


class ExecutionRecord(BaseModel):
    """单条用例的执行结果，对应 qa-skills 执行阶段的 TC × 结果回收。"""

    case_id: str = Field(description="用例编号")
    title: str = Field(default="", description="用例名称")
    priority: Priority = Field(default="P1", description="用例优先级")
    status: Literal["通过", "失败", "阻塞", "未执行"] = Field(description="执行状态")
    # 失败分流的结论分类，见 core/triage.md 四分类判定树。
    triage: Literal["", "A", "B1", "B2", "C", "D", "U"] = Field(
        default="",
        description="失败分流结论：A 产品缺陷 / B1 测试代码问题 / B2 环境问题 / C 用例问题 / D 需求歧义 / U 未分流",
    )
    evidence: str = Field(default="", description="执行证据：日志位置、响应原文、截图路径")
    note: str = Field(default="", description="补充说明")


class BugItem(BaseModel):
    """Bug 条目，字段对齐 core/report-template.md §3。"""

    id: str = Field(description="Bug 编号，形如 BUG-001")
    title: str = Field(description="一句话描述")
    severity: Literal["S0", "S1", "S2"] = Field(
        description="严重程度：S0 数据丢失/资损/安全或核心主路径不可用；S1 核心旁路不可用；S2 体验问题"
    )
    status: Literal["新建", "已修复待验证", "已验证关闭", "不予修复"] = Field(
        default="新建", description="Bug 状态，随回归结果同步"
    )
    related_case: str = Field(default="", description="发现该 Bug 的用例编号")
    reproduce_steps: str = Field(description="复现步骤")
    expected_behavior: str = Field(description="预期行为")
    actual_behavior: str = Field(description="实际行为")
    root_cause: str = Field(default="", description="根因分析，未分析时留 TODO")
    impact_scope: str = Field(default="", description="影响范围（功能/数据/用户/安全/修复波及）")
    severity_basis: str = Field(default="", description="Severity 定级依据")
    fix_suggestion: str = Field(default="", description="修复建议")
    regression_suggestion: str = Field(default="", description="回归建议：修复后应验证的用例或场景")


class RegressionItem(BaseModel):
    """回归清单条目，见 qa-skills 的 regression-testing。"""

    case_id: str = Field(description="用例编号")
    title: str = Field(default="", description="用例名称")
    level: Literal["必须回归", "建议回归", "可选回归"] = Field(description="回归级别")
    reason: str = Field(description="纳入该级别的依据")
