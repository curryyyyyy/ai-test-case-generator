"""从用例 Schema 生成 API / E2E 测试脚本骨架。

生成物是**脚手架**而不是成品：用例正文是业务语言，缺少接口签名与元素定位，
这些只能由熟悉被测系统的人补齐。骨架里把能自动带上的都带上——TC 编号、
用例标题、前置、步骤、预期、优先级、风险关联——并把缺的部分显式标 TODO。

对齐 qa-skills 的 api-testing / automated-e2e-testing 工程约定：
- API：一个模块一个文件，test 名沿用 TC 编号，统一 Client 封装
- E2E：Page Object 交互，test 名沿用 TC 编号，页面定位留 TODO
"""

from __future__ import annotations

import re
from typing import Any

from utils.case_ids import group_cases_by_module


def _safe_identifier(text: str) -> str:
    """把中文标题压成可安全用作 Python 标识符后缀的 ASCII 串。"""
    ascii_part = re.sub(r"[^0-9a-zA-Z]+", "_", str(text or "")).strip("_")
    return ascii_part[:40]


def _escape_docstring(text: str) -> str:
    """转义 docstring 中会提前闭合字符串的内容。"""
    return str(text or "").replace("\\", "\\\\").replace('"""', "'''")


def _case_docstring(case: dict[str, Any]) -> str:
    """把用例要素内联成 test 函数的 docstring，让脚本自带用例上下文。"""
    steps = case.get("steps", [])
    if isinstance(steps, list):
        steps_text = "\n".join(f"    {index}. {step}" for index, step in enumerate(steps, 1))
    else:
        steps_text = f"    {steps}"

    lines = [
        f"{case.get('case_id', '')} {case.get('test_point', '')} "
        f"[{case.get('case_level', 'P1')}]",
        "",
        f"前置条件: {case.get('precondition', '') or '（无）'}",
        "操作步骤:",
        steps_text or "    （无）",
        f"预期结果: {case.get('expected_result', '')}",
    ]
    risk_ref = str(case.get("risk_ref", "")).strip()
    if risk_ref:
        lines.append(f"关联风险: {risk_ref}")
    return _escape_docstring("\n".join(lines))


def render_client_module() -> str:
    """生成 API 测试的统一请求封装（对齐 api-testing 的脚手架约定）。"""
    return '''"""统一请求封装：日志、超时、鉴权头。

由 ai-test-case-generator 自动生成，可按项目既定技术栈替换。
"""
from __future__ import annotations

import os
from typing import Any

import requests


class Client:
    """requests.Session 不支持 base_url，必须显式拼接。"""

    def __init__(self, base_url: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.s = requests.Session()
        if token:
            self.s.headers.update({"Authorization": f"Bearer {token}"})

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", 10)
        return self.s.request(method, url, **kwargs)

    def get(self, path: str, **kw: Any) -> requests.Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> requests.Response:
        return self.request("POST", path, **kw)

    def put(self, path: str, **kw: Any) -> requests.Response:
        return self.request("PUT", path, **kw)

    def delete(self, path: str, **kw: Any) -> requests.Response:
        return self.request("DELETE", path, **kw)


def login(user: str, password: str) -> str:
    """按项目实际登录接口实现——占位，勿直接照抄。"""
    raise NotImplementedError(
        f"TODO: 实现登录取 token 的逻辑（当前入参 user={user!r}）"
    )


def base_url() -> str:
    value = os.getenv("BASE_URL", "")
    if not value:
        raise RuntimeError("TODO：设置环境变量 BASE_URL 指向被测环境")
    return value
'''


def render_conftest() -> str:
    """生成 conftest.py：提供 client fixture。"""
    return '''"""pytest fixture：从环境变量读取被测环境配置，不硬编码。"""
from __future__ import annotations

import os

import pytest

from common.client import Client, base_url, login


@pytest.fixture(scope="session")
def client() -> Client:
    token = ""
    user = os.getenv("TEST_USER", "")
    password = os.getenv("TEST_PASSWORD", "")
    if user and password:
        token = login(user, password)
    return Client(base_url(), token)
'''


def render_api_module(module_no: int, module_name: str, cases: list[dict[str, Any]]) -> str:
    """生成一个模块的 API 测试文件。"""
    body: list[str] = [
        '"""自动生成的 API 测试骨架。',
        "",
        f"模块：{module_no}. {module_name}",
        "用例来源：测试用例_markmap.md（schema.yaml 由它单向抽取）",
        "函数名沿用 TC 编号，执行结果可直接回收进流水线（Bug 分析 / 回归 / 报告）。",
        '"""',
        "from __future__ import annotations",
        "",
        "import pytest",
        "",
    ]

    for case in cases:
        case_id = str(case.get("case_id", ""))
        ident = case_id.replace("-", "_") or "case"
        suffix = _safe_identifier(case.get("test_point", ""))
        func_name = f"test_{ident}" + (f"_{suffix}" if suffix else "")
        body.append("")
        body.append(f"def {func_name}(client):")
        body.append(f'    """{_case_docstring(case)}"""')
        body.append("    # TODO：依据用例步骤补充接口调用与断言")
        body.append("    # 示例：resp = client.post('/api/xxx', json={...})；assert resp.status_code == 200")
        body.append(
            '    pytest.skip("TODO：接口细节未配置，补齐调用后删除本行")'
        )

    return "\n".join(body) + "\n"


def render_e2e_module(module_no: int, module_name: str, cases: list[dict[str, Any]]) -> str:
    """生成一个模块的 Playwright 测试文件。"""
    body: list[str] = [
        '"""自动生成的 E2E 测试骨架（Playwright）。',
        "",
        f"模块：{module_no}. {module_name}",
        "工程约定见 core/ 下 automated-e2e-testing 的 references：",
        "正式测试中禁止裸写定位器，页面交互一律封装进 Page Object。",
        '"""',
        "from __future__ import annotations",
        "",
        "import pytest",
        "from playwright.sync_api import Page",
        "",
        "",
        "def _goto(page: Page, path: str) -> None:",
        '    """统一入口跳转：base_url 由 pytest-base-url 或环境变量提供。"""',
        "    page.goto(path)",
        "",
    ]

    for case in cases:
        case_id = str(case.get("case_id", ""))
        ident = case_id.replace("-", "_") or "case"
        suffix = _safe_identifier(case.get("test_point", ""))
        func_name = f"test_{ident}" + (f"_{suffix}" if suffix else "")
        body.append("")
        body.append(f"def {func_name}(page: Page):")
        body.append(f'    """{_case_docstring(case)}"""')
        body.append("    # TODO：补充页面入口路径与元素定位（封装为 Page Object 方法）")
        body.append("    # 示例：_goto(page, '/marketing/coupon')；page.get_by_role('button', name='创建').click()")
        body.append('    pytest.skip("TODO：页面定位未配置，补齐后删除本行")')

    return "\n".join(body) + "\n"


def render_requirements_api() -> str:
    return "requests\npytest\n"


def render_requirements_e2e() -> str:
    return "pytest\npytest-playwright\nplaywright\n"


def build_api_scaffold(test_cases: list[dict[str, Any]]) -> dict[str, str]:
    """生成 API 测试骨架的全部文件（相对路径 → 内容）。

    只纳入 Schema 判定为可自动化的用例：ui 模型走 E2E，dev-collab 走 api/手动，
    带 [需真机]/[需专业环境] 标注的用例不生成脚本（生成了也跑不通）。
    """
    files: dict[str, str] = {
        "api-tests/conftest.py": render_conftest(),
        "api-tests/common/client.py": render_client_module(),
        "api-tests/requirements.txt": render_requirements_api(),
    }
    for module_no, module_name, cases in group_cases_by_module(test_cases):
        selected = [
            case
            for case in cases
            if not _has_blocking_tag(case)
            and case.get("execution_model", "ui") in {"dev-collab", "ui"}
        ]
        if not selected:
            continue
        safe_module = _safe_identifier(module_name) or f"module{module_no}"
        file_path = f"api-tests/test_{module_no:02d}_{safe_module}.py"
        files[file_path] = render_api_module(module_no, module_name, selected)
    return files


def build_e2e_scaffold(test_cases: list[dict[str, Any]]) -> dict[str, str]:
    """生成 E2E 测试骨架的全部文件。"""
    files: dict[str, str] = {
        "e2e-tests/requirements.txt": render_requirements_e2e(),
    }
    for module_no, module_name, cases in group_cases_by_module(test_cases):
        selected = [
            case
            for case in cases
            if not _has_blocking_tag(case)
            and case.get("execution_model", "ui") == "ui"
            and str(case.get("case_level", "")) in {"P0", "P1"}
        ]
        if not selected:
            continue
        safe_module = _safe_identifier(module_name) or f"module{module_no}"
        file_path = f"e2e-tests/test_{module_no:02d}_{safe_module}.py"
        files[file_path] = render_e2e_module(module_no, module_name, selected)
    return files


def _has_blocking_tag(case: dict[str, Any]) -> bool:
    """带这些标注的用例无法在标准环境执行，不生成脚本。"""
    blocking = {"[需真机]", "[需专业环境]"}
    return any(str(tag) in blocking for tag in case.get("tags", []) or [])
