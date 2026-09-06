"""CSV 导出器。

用于把用例导入 TestLink / Jira / 禅道等平台，或直接用表格工具查看。
Excel 里的换行在 CSV 中必须保留，所以统一用 CRLF 包裹含换行的字段，
否则多行步骤会被截断成多行记录。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from utils.exporters.testcase_fields import TESTCASE_HEADERS, normalize_cell_value


def export_test_cases_to_csv(
    test_cases: list[dict[str, Any]],
    output_file_path: str | Path,
) -> str:
    """将测试用例列表导出为 CSV（UTF-8 BOM，Excel 直接打开不乱码）并返回路径。"""
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig 带 BOM：否则 Excel 打开中文会乱码。
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([header for _field, header in TESTCASE_HEADERS])
        for test_case in test_cases:
            writer.writerow(
                [
                    normalize_cell_value(field_name, test_case)
                    for field_name, _header in TESTCASE_HEADERS
                ]
            )

    return str(output_path.resolve())
