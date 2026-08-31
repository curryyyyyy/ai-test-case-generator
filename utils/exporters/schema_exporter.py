"""从用例单向抽取 qa-skills 的 `测试用例.schema.yaml`。

双轨原则（core/schema-extraction.md）：markmap 是唯一人工维护源，Schema 是由它
**单向抽取**的机读元数据层，作为 Skill 间流转接口（test-case-review / regression-testing /
执行层消费）。markmap 改动后必须重新抽取，两轨不一致以 markmap 为准。

YAML 序列化优先用 PyYAML；环境未安装时退化为内置的手写序列化器，
保证零第三方依赖也能产出合法 YAML（转义纪律是这里最容易翻车的地方：
双引号值内裸放引号会让整份文件解析失败，中断下游全部消费）。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.case_ids import group_cases_by_module

try:  # pragma: no cover - 取决于运行环境是否装有 PyYAML
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_SCRIPT = PROJECT_ROOT / "core" / "scripts" / "validate_schema.py"

# schema-extraction.md 定义的 type 枚举。
CASE_TYPE_VOCAB = {
    "functional",
    "boundary",
    "exception",
    "permission",
    "regression",
    "state",
    "data",
    "reliability",
    "concurrency",
    "security",
    "compatibility",
}

_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("concurrency", ("并发", "竞态", "同时操作", "并行")),
    ("reliability", ("重试", "幂等", "恢复", "容错", "超时重", "降级")),
    ("security", ("越权", "注入", "篡改", "敏感信息", "鉴权")),
    ("compatibility", ("兼容", "多端", "分辨率", "浏览器")),
    ("permission", ("权限", "角色", "可见性")),
    ("state", ("状态", "流转", "生命周期")),
    ("data", ("一致性", "脏数据", "残留", "读回")),
    ("boundary", ("边界", "极值", "超限", "上限", "下限", "最大", "最小", "长度", "空值")),
    ("exception", ("异常", "失败", "错误", "中断", "超时", "拒绝", "非法")),
]

_TAG_TO_TYPE = {
    "并发": "concurrency",
    "可靠": "reliability",
    "安全": "security",
    "兼容": "compatibility",
}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def infer_case_type(case: dict[str, Any]) -> str:
    """推断用例在 Schema 中的 type。

    优先用类型域轴标签（来自测试策略），其次按名称关键词匹配，
    兜底 functional。
    """
    for tag in case.get("tags", []) or []:
        mapped = _TAG_TO_TYPE.get(_as_text(tag).strip("[]"))
        if mapped:
            return mapped

    text = f"{_as_text(case.get('test_point'))} {_as_text(case.get('expected_result'))}"
    for case_type, keywords in _TYPE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return case_type
    return "functional"


def _normalize_tags(case: dict[str, Any]) -> list[str]:
    """把标签统一成 `[xxx]` 形态，并补上非功能类型的中文标签。

    词表外标签在校验器里只告警不报错，保留原类型语义比静默丢弃更有价值。
    """
    tags: list[str] = []
    for tag in case.get("tags", []) or []:
        tag_text = _as_text(tag).strip("[]")
        if tag_text:
            tags.append(f"[{tag_text}]")

    test_type = _as_text(case.get("test_type"))
    if test_type and test_type != "功能" and f"[{test_type}]" not in tags:
        tags.append(f"[{test_type}]")
    return tags


def _is_dev_collab(case: dict[str, Any]) -> bool:
    steps = case.get("steps", [])
    joined = " ".join(str(item) for item in steps) if isinstance(steps, list) else str(steps)
    return "请开发执行" in joined


def _split_lines(value: Any) -> list[str]:
    """把可能的多行文本拆成列表，供 steps / expected 等列表字段使用。"""
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    text = _as_text(value)
    if not text:
        return []
    return [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]


def build_case_schema(
    test_cases: list[dict[str, Any]],
    project_name: str = "项目",
    strategy: dict[str, Any] | None = None,
    source_markmap: str = "测试用例_markmap.md",
) -> dict[str, Any]:
    """构建三层形态的 Schema：meta / modules / cases。"""
    strategy_ref = "测试策略.md" if strategy else None

    modules: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for module_no, module_name, module_cases in group_cases_by_module(test_cases):
        module_label = f"{module_no}. {module_name}"
        modules.append({"module": module_label, "shared_preconditions": []})

        for case in module_cases:
            dev_collab = _is_dev_collab(case)
            execution_model = "dev-collab" if dev_collab else "ui"
            # 协作五段式无法用 UI 自动化；有可测试性标注的用例至多 partial。
            tags = _normalize_tags(case)
            has_special_env = any(
                tag in {"[需真机]", "[需Mock]", "[需专业环境]"} for tag in tags
            )
            if dev_collab:
                supported, framework = "partial", "api"
            elif has_special_env:
                supported, framework = "partial", "manual"
            else:
                supported, framework = "yes", "playwright"

            evidence_source = _as_text(case.get("evidence_source"))
            cases.append(
                {
                    "id": _as_text(case.get("case_id")),
                    "title": _as_text(case.get("test_point")),
                    "module": module_label,
                    "priority": _as_text(case.get("case_level")) or "P1",
                    "type": infer_case_type(case),
                    "execution_model": execution_model,
                    "smoke": _as_text(case.get("smoke")) or None,
                    "preconditions": _split_lines(case.get("precondition")),
                    "steps": _split_lines(case.get("steps")),
                    "expected": _split_lines(case.get("expected_result")),
                    "test_data": {},
                    "risk_ref": _as_text(case.get("risk_ref")) or None,
                    # 纯文档模式无代码仓库，code_refs 一律为空——
                    # 编造指涉属幻觉证据，比留空危害更大。
                    "code_refs": [],
                    "evidence": {
                        "level": "E1" if evidence_source else "E0",
                        "source": evidence_source or "需求文档",
                        "confidence": "medium",
                        "status": "inference",
                    },
                    "tags": tags,
                    "automation": {"supported": supported, "framework": framework},
                    "status": "active",
                }
            )

    meta: dict[str, Any] = {
        "source_markmap": source_markmap,
        "project": project_name,
    }
    if strategy_ref:
        meta["strategy_ref"] = strategy_ref

    return {"meta": meta, "modules": modules, "cases": cases}


# ---------------------------------------------------------------------------
# YAML 序列化：PyYAML 优先，无依赖时退化到手写序列化器
# ---------------------------------------------------------------------------


def _quote(value: str) -> str:
    """双引号标量：转义反斜杠与双引号，这是 YAML 转义纪律的核心。"""
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _needs_block(value: str) -> bool:
    """判断是否需要退化为块标量。

    含换行、双引号、冒号空格、首尾空白或过长的值用块标量更安全。
    """
    if not value:
        return False
    return (
        "\n" in value
        or '"' in value
        or ": " in value
        or value != value.strip()
        or len(value) > 80
    )


def _dump_scalar(value: Any, indent: int) -> str:
    """渲染单个标量的值部分（不含键名）。"""
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).replace("\r", "")
    if not text:
        return '""'
    if not _needs_block(text):
        return _quote(text)

    # 块标量：缩进对齐，空行用 . 占位防止被折叠吞掉。
    pad = " " * (indent + 2)
    lines = [line.rstrip() for line in text.split("\n")]
    body = "\n".join(f"{pad}{line}" if line else f"{pad}." for line in lines)
    return ">-\n" + body


def _dump_mapping_fallback(data: dict[str, Any], indent: int = 0) -> list[str]:
    """手写 YAML 映射序列化（PyYAML 不可用时的兜底）。"""
    pad = " " * indent
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(_dump_mapping_fallback(value, indent + 2))
            continue
        if isinstance(value, list):
            if not value:
                lines.append(f"{pad}{key}: []")
                continue
            lines.append(f"{pad}{key}:")
            lines.extend(_dump_list_fallback(value, indent + 2))
            continue
        lines.append(f"{pad}{key}: {_dump_scalar(value, indent)}")
    return lines


def _dump_list_fallback(items: list[Any], indent: int) -> list[str]:
    """手写 YAML 序列序列化。"""
    pad = " " * indent
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            # 序列项本身是映射：首键与 `- ` 同行，其余键对齐到键名列。
            rendered = _dump_mapping_fallback(item, indent + 2)
            if not rendered:
                continue
            first = rendered[0].lstrip()
            lines.append(f"{pad}- {first}")
            lines.extend(rendered[1:])
            continue
        scalar = _dump_scalar(item, indent)
        if scalar.startswith(">-\n"):
            lines.append(f"{pad}- >-")
            lines.extend(scalar.split("\n")[1:])
        else:
            lines.append(f"{pad}- {scalar}")
    return lines


def dump_schema_yaml(schema: dict[str, Any]) -> str:
    """把 Schema 序列化为 YAML 文本。"""
    if _HAS_YAML:  # pragma: no cover - 有 PyYAML 时走标准库路径
        return yaml.safe_dump(
            schema,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            # 宽到不会自动折行，避免长文本被插入换行破坏块结构。
            width=10**6,
        )

    lines: list[str] = ["# 由 markmap 单向抽取，请勿手工编辑（改动请改 markmap 后重新抽取）"]
    lines.extend(_dump_mapping_fallback(schema))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 校验：调用 qa-skills 的 validate_schema.py
# ---------------------------------------------------------------------------


def validate_schema(
    markmap_path: str | Path,
    schema_path: str | Path,
    strategy_path: str | Path | None = None,
) -> tuple[bool, str]:
    """调用 core/scripts/validate_schema.py 校验 markmap 与 schema 的一致性。

    返回 (是否通过, 输出文本)。校验脚本缺失时不视为失败——
    它是质量门禁，不应阻断主流程。
    """
    if not VALIDATE_SCRIPT.exists():
        return True, "（未找到校验脚本，跳过校验）"

    command = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        str(markmap_path),
        str(schema_path),
    ]
    if strategy_path and Path(strategy_path).exists():
        command.extend(["--strategy", str(strategy_path)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # 校验器自身异常不应阻断主流程
        return True, f"（校验脚本执行异常，已跳过：{exc}）"

    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output.strip()
