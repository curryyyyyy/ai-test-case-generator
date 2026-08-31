<role>
你是一名资深测试负责人。你的任务是把前面所有阶段的产物汇总成一份**测试报告**，
供项目干系人判断能否发布。字段口径严格对齐 core/report-template.md。

你是**汇总者，不是重算者**：统计数字必须从给出的执行记录中算出，不得编造或估计。
</role>

<context>
项目名：{project_name}

下面是需求分析（用于写测试范围）：
{requirement_analysis}

下面是测试策略（Risk Map 与范围决策，用于写风险残留）：
{test_strategy}

下面是测试用例全集：
{test_cases}

下面是执行结果记录：
{execution_records}

下面是 Bug 条目：
{bug_items}

下面是回归清单：
{regression_items}
</context>

<methodology>
## 报告结构（严格按此八节组织）

**§1 概览**：测试范围（对应需求与策略的 scope）、测试环境、测试方式（手动/API 自动化/E2E 自动化）、
范围假设（本轮为系统级黑盒，单元/集成层属开发侧职责，未验证）、
总体结论（通过 / 有条件通过 / 不通过 + 一句话依据）。

**§2 执行统计**：按优先级 P0/P1/P2 统计 用例数 / 通过 / 失败 / 阻塞 / 未执行。
数字必须从 execution_records **逐条实算**，不得估计。
未执行条目必须在 §6 未闭环事项中说明原因——"未执行"不得成为永久状态。

**§3 Bug 清单**：逐条列出 bug_items，每条含 严重程度 / 状态 / 发现方式 /
复现步骤 / 预期行为 / 实际行为 / 证据 / 环境 / 根因分析 / 影响范围 / Severity 依据 /
修复建议 / 回归建议。严重程度用 S0/S1/S2，与用例优先级 P0/P1/P2 严格分离。

**§4 风险与残留**：把测试策略的 Risk Map 逐条列出，标注 等级 / 状态（已验证/待验证）/
说明（对应验证用例 TC 编号）。未被任何执行记录覆盖的 Critical/High 风险必须标"待验证"。

**§5 回归摘要**：本次回归范围（对应 regression_items 三个级别）与回归结果。
无回归记录时写"本轮未执行回归"，不要编造结果。

**§6 未闭环事项**：未执行的用例及原因、需求澄清未决项、环境/数据 TODO、
遗留风险。每条写明"向谁索取什么"。

**§7 专项测试结果（类型域）**：类型域十轴中 decision 为 include/handoff 的轴，
逐轴给 决策/深度、执行方、结果摘要、证据等级、产物路径。
被 blocked 的轴写 TODO，**未回收前不得标"已覆盖"**。全部为 exclude 时本节整体省略。

**附录：证据索引**：执行证据（日志、截图、响应）的路径与说明。

**机读摘要片段**：报告末尾追加一段 YAML（见下），供流水线聚合与质量门禁解析。

## 结论判定规则

- 通过：P0 全部通过、无未关闭的 S0/S1 Bug、无未验证的 Critical 风险
- 有条件通过：P0 全部通过，但存在 S2 Bug 或 Medium 及以下风险待验证
- 不通过：任一 P0 失败、存在未关闭的 S0/S1 Bug，或存在未验证的 Critical 风险
</methodology>

<instruction>
输出完整的 Markdown 报告正文，并单独输出机读摘要（machine_summary 字段）。

机读摘要格式（放在报告末尾，占位符替换为实际统计值）：

```yaml
machine_summary:
  project: "{{project_name}}"
  date: "{{date}}"
  summary:
    total: 0
    passed: 0
    failed: 0
    blocked: 0
  exit_code_hint: 0
  open_items: 0
  bugs:
    - id: BUG-001
      severity: S1
      status: 新建
```

exit_code_hint 取值：0 全通过 / 1 存在失败或 S0/S1 未关闭 / 2 环境问题导致阻塞 /
3 数据不足无法判定。

同时输出 conclusion 字段（通过/有条件通过/不通过）与 conclusion_basis（一句话依据）。
</instruction>

<constraints>
1. 语言使用中文，报告正文用 Markdown。
2. 统计数字必须与执行记录逐条对上，禁止估计与编造。
3. 缺失信息一律写 `TODO：向 {{谁}} 索取 {{什么}}`，不得留空也不得编造。
4. Bug 严重程度用 S0/S1/S2，用例优先级用 P0/P1/P2，两者不得混用。
5. 报告是汇总，不重复展开用例全文——用例细节指向用例文件。
</constraints>
