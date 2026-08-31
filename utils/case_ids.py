"""用例编号规范化与模块分组。

qa-skills 的硬约束（core/case-format.md §3 + core/schema-extraction.md）：
TC 编号形如 `TC-{两位模块号}-{两位序号}`，且**编号首段必须与所属一级模块编号一致**
（校验器据此交叉核对，不一致即报错）。

LLM 生成的编号常常与模块分组脱节（例如模块顺序重排后编号未同步），
因此在用例产出后统一规范化一次，让「界面展示 / Excel / markmap / schema.yaml」
四处共用同一套编号，避免多轨编号互相打架。
"""

from __future__ import annotations

from typing import Any


def _case_sort_key(case: dict[str, Any]) -> tuple[int, str]:
    """组内排序：P0 在前，其次按原编号（保留 LLM 的步骤组织顺序）。"""
    level = str(case.get("case_level", "P1"))
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return (order.get(level, 1), str(case.get("case_id", "")))


def group_cases_by_module(
    test_cases: list[dict[str, Any]],
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """按 directory 分组，返回 [(模块号, 模块名, 用例列表)]。

    模块号从 1 开始，按模块首次出现的顺序分配，用例在组内按优先级排序。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    module_order: list[str] = []

    for case in test_cases:
        module_name = str(case.get("directory", "")).strip() or "未分类"
        if module_name not in grouped:
            grouped[module_name] = []
            module_order.append(module_name)
        grouped[module_name].append(case)

    result: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, module_name in enumerate(module_order, start=1):
        cases = sorted(grouped[module_name], key=_case_sort_key)
        result.append((index, module_name, cases))
    return result


def normalize_case_ids(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按模块分组重排 case_id，保证编号首段与模块号一致、序号连续。

    返回新的列表，不原地修改入参（state 中的用例可能被多处引用）。
    """
    normalized: list[dict[str, Any]] = []
    for module_no, _module_name, cases in group_cases_by_module(test_cases):
        for seq, case in enumerate(cases, start=1):
            new_case = dict(case)
            new_case["case_id"] = f"TC-{module_no:02d}-{seq:02d}"
            normalized.append(new_case)
    return normalized


def assign_smoke_numbers(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给 P0 用例分配冒烟序号，返回带 smoke 字段的新列表。

    冒烟集用于快速抽取：P0 且执行模型为 ui 的用例按顺序编 SMOKE-1、SMOKE-2…
    """
    result: list[dict[str, Any]] = []
    smoke_seq = 0
    for case in test_cases:
        new_case = dict(case)
        if str(case.get("case_level", "")) == "P0":
            smoke_seq += 1
            new_case["smoke"] = f"SMOKE-{smoke_seq}"
        else:
            new_case["smoke"] = ""
        result.append(new_case)
    return result
