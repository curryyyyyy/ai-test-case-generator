"""测试用例导出字段的单一定义来源。

Excel / CSV / 其他导出器共享这里的字段顺序与取值规则，
避免出现「Excel 多了某列、CSV 少了某列」这类不一致。
新增导出格式时只需复用这里的定义，不必再抄一份。
"""

from __future__ import annotations

from typing import Any

# (字段名, 表头文案)。顺序即导出列顺序。
TESTCASE_HEADERS: list[tuple[str, str]] = [
    ("case_id", "用例ID"),
    ("directory", "模块"),
    ("test_point", "功能点"),
    ("case_level", "优先级"),
    ("precondition", "前置条件"),
    ("steps", "测试步骤"),
    ("expected_result", "预期结果"),
]


def normalize_cell_value(field_name: str, test_case: dict[str, Any]) -> str:
    """取出字段并转成单元格文本。

    steps 是列表，用换行连接；其余一律转字符串，
    避免 None 在 Excel / CSV 里表现不一致。
    """
    raw_value = test_case.get(field_name, "")
    if field_name == "steps":
        if isinstance(raw_value, list):
            return "\n".join(str(item) for item in raw_value)
        return str(raw_value)
    return "" if raw_value is None else str(raw_value)
