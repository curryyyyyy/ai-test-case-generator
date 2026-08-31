"""执行结果回收：把外部运行输出归一化成 ExecutionRecord。

支持三种输入来源：
1. pytest 文本输出（`-v` 或 `--tb=no -q` 的行式结果）
2. JUnit XML（`pytest --junitxml` 或多数 CI 的通用报告格式）
3. 界面上手工录入的 TC × 结果表（手动执行路径）

归一化后统一落盘为 `手动执行记录_{日期}.md`（或执行分报告），
作为报告 §2 执行统计与 §3 Bug 清单的数据来源——
qa-skills 明确要求这一步：否则"未执行"会变成永远填不上的空洞。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from xml.etree import ElementTree

# test 函数名里 TC 编号用下划线（Python 标识符不允许连字符），先还原。
_UNDERSCORE_TC = re.compile(r"TC_(\d+)_(\d+)")
_TC = re.compile(r"TC-\d+-\d+")
# pytest -v 的行式结果：`path::test_name PASSED  [ 10%]`
_PYTEST_LINE = re.compile(
    r"(TC-\d+-\d+).*?\b(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASSED)\b",
    re.IGNORECASE,
)

_STATUS_MAP = {
    "PASSED": "通过",
    "FAILED": "失败",
    "ERROR": "阻塞",
    "SKIPPED": "未执行",
    "XFAIL": "通过",
    "XPASSED": "失败",
}

VALID_STATUSES = ("通过", "失败", "阻塞", "未执行")


def _normalize_status(raw: Any) -> str:
    """把各种写法归一化到四态，无法识别的一律记为未执行（不猜测）。"""
    text = str(raw or "").strip().upper()
    if text in _STATUS_MAP:
        return _STATUS_MAP[text]
    if text in {"通过", "PASS", "OK", "成功"}:
        return "通过"
    if text in {"失败", "FAIL", "NG"}:
        return "失败"
    if text in {"阻塞", "BLOCKED", "BLOCK"}:
        return "阻塞"
    return "未执行"


def parse_pytest_output(output: str) -> list[dict[str, Any]]:
    """解析 pytest 文本输出，返回 ExecutionRecord 列表。

    只认带 TC 编号的行——没有编号的用例无法与流水线对齐，宁可丢弃。
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line in str(output or "").splitlines():
        normalized_line = _UNDERSCORE_TC.sub(r"TC-\1-\2", line)
        match = _PYTEST_LINE.search(normalized_line)
        if not match:
            continue

        case_id = match.group(1)
        if case_id in seen:
            continue
        seen.add(case_id)
        records.append(
            {
                "case_id": case_id,
                "title": "",
                "priority": "P1",
                "status": _normalize_status(match.group(2)),
                "triage": "",
                "evidence": "pytest 输出",
                "note": "",
            }
        )
    return records


def parse_junit_xml(xml_text: str) -> list[dict[str, Any]]:
    """解析 JUnit XML 报告，返回 ExecutionRecord 列表。"""
    try:
        root = ElementTree.fromstring(str(xml_text or ""))
    except ElementTree.ParseError:
        return []

    records: list[dict[str, Any]] = []
    for testcase in root.iter("testcase"):
        raw_name = f"{testcase.get('classname', '')} {testcase.get('name', '')}"
        normalized_name = _UNDERSCORE_TC.sub(r"TC-\1-\2", raw_name)
        match = _TC.search(normalized_name)
        if not match:
            continue

        if testcase.find("failure") is not None:
            status = "失败"
        elif testcase.find("error") is not None:
            status = "阻塞"
        elif testcase.find("skipped") is not None:
            status = "未执行"
        else:
            status = "通过"

        records.append(
            {
                "case_id": match.group(0),
                "title": str(testcase.get("name", "")),
                "priority": "P1",
                "status": status,
                "triage": "",
                "evidence": "JUnit XML 报告",
                "note": "",
            }
        )
    return records


def parse_execution_output(text: str) -> list[dict[str, Any]]:
    """自动判别输入格式（XML / pytest 文本）并解析。"""
    stripped = str(text or "").strip()
    if not stripped:
        return []
    if stripped.startswith("<"):
        return parse_junit_xml(stripped)
    return parse_pytest_output(stripped)


def normalize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """归一化手工录入或表格编辑得到的执行记录。

    输入行的 case_id 为空的行直接丢弃——没有 TC 编号无法与用例对齐。
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        records.append(
            {
                "case_id": case_id,
                "title": str(row.get("title", "")).strip(),
                "priority": str(row.get("priority", "P1")).strip() or "P1",
                "status": _normalize_status(row.get("status", "未执行")),
                "triage": str(row.get("triage", "")).strip(),
                "evidence": str(row.get("evidence", "")).strip(),
                "note": str(row.get("note", "")).strip(),
            }
        )
    return records


def build_records_from_cases(
    test_cases: list[dict[str, Any]],
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """以用例集为基数生成执行记录表（未录结果的补"未执行"）。

    这样界面上直接呈现完整的 TC × 结果矩阵，用户只填结果列，
    不会出现"漏填的用例在统计里凭空消失"。
    """
    provided = {
        str(record.get("case_id", "")): record for record in (records or [])
    }
    result: list[dict[str, Any]] = []
    for case in test_cases:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            continue
        existing = provided.get(case_id)
        result.append(
            {
                "case_id": case_id,
                "title": str(case.get("test_point", "")),
                "priority": str(case.get("case_level", "P1")),
                "status": _normalize_status(
                    existing.get("status", "未执行") if existing else "未执行"
                ),
                "triage": str(existing.get("triage", "")) if existing else "",
                "evidence": str(existing.get("evidence", "")) if existing else "",
                "note": str(existing.get("note", "")) if existing else "",
            }
        )
    return result


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按优先级统计执行结果，供报告 §2 与界面概览使用。"""
    summary: dict[str, Any] = {
        "total": len(records),
        "通过": 0,
        "失败": 0,
        "阻塞": 0,
        "未执行": 0,
        "by_priority": {},
    }
    by_priority: dict[str, dict[str, int]] = {}

    for record in records:
        status = _normalize_status(record.get("status", "未执行"))
        summary[status] = int(summary.get(status, 0)) + 1

        priority = str(record.get("priority", "P1"))
        bucket = by_priority.setdefault(
            priority, {"用例数": 0, "通过": 0, "失败": 0, "阻塞": 0, "未执行": 0}
        )
        bucket["用例数"] += 1
        bucket[status] = int(bucket.get(status, 0)) + 1

    summary["by_priority"] = by_priority
    return summary


def render_execution_records_markdown(
    records: list[dict[str, Any]],
    project_name: str = "",
    mode: str = "manual",
) -> str:
    """把执行记录渲染为落盘 Markdown（手动执行记录 / 执行分报告）。"""
    mode_label = {
        "manual": "手动执行记录",
        "api": "API 自动化执行报告",
        "e2e": "E2E 自动化执行报告",
    }.get(mode, "执行记录")

    summary = summarize_records(records)
    lines: list[str] = [
        f"# {project_name} {mode_label} — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}｜执行方式：{mode}",
        "",
        "## 执行统计",
        "",
        "| 优先级 | 用例数 | 通过 | 失败 | 阻塞 | 未执行 |",
        "|--------|--------|------|------|------|--------|",
    ]

    for priority in sorted(summary["by_priority"]):
        bucket = summary["by_priority"][priority]
        lines.append(
            f"| {priority} | {bucket['用例数']} | {bucket['通过']} | "
            f"{bucket['失败']} | {bucket['阻塞']} | {bucket['未执行']} |"
        )

    lines.extend(
        [
            "",
            f"合计：{summary['total']} 条，通过 {summary['通过']}，"
            f"失败 {summary['失败']}，阻塞 {summary['阻塞']}，未执行 {summary['未执行']}",
            "",
            "## 明细",
            "",
            "| 用例 | 名称 | 优先级 | 结果 | 失败分流 | 证据 | 备注 |",
            "|------|------|--------|------|---------|------|------|",
        ]
    )

    for record in records:
        lines.append(
            f"| {record.get('case_id', '')} | {record.get('title', '')} | "
            f"{record.get('priority', 'P1')} | {record.get('status', '未执行')} | "
            f"{record.get('triage', '')} | {record.get('evidence', '')} | "
            f"{record.get('note', '')} |"
        )

    lines.append("")
    return "\n".join(lines)
