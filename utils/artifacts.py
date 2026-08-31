"""落盘产物管理——流水线的持久化状态。

qa-skills 的核心约定是「文件即流水线状态」：阶段之间传递的是**产物文件路径**，
不是会话内转述；任何会话中断后，凭落盘产物即可续跑。本模块负责这份状态的落地，
产物目录结构与 qa-skills 的命名约定保持一致：

```text
artifacts/{项目名}/
├─ 需求模型.md                 # 阶段 1
├─ 测试策略.md                 # 阶段 2
├─ 测试用例_markmap.md         # 阶段 3（唯一人工维护源）
├─ 测试用例.schema.yaml        # 由 markmap 单向抽取的机读层
├─ 测试用例.xlsx               # 项目原有的 Excel 交付物
├─ 回归清单_{日期}.md          # 阶段 7
└─ 测试报告_{日期}.md          # 阶段 8
```

双轨原则：markmap 是给人看与人工维护的唯一源，schema.yaml 由它单向抽取；
两者不一致时以 markmap 为准，Schema 永远可重新生成。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# 文件名非法字符（含路径分隔符，防 CWE-22 路径穿越）。
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_project_name(project_name: str) -> str:
    """净化项目名，使其可安全用作目录名。

    项目名可能来自上传文件名或用户输入，必须去掉路径分隔符与控制字符，
    否则 `../` 或绝对路径会让写盘逃出 artifacts 目录。
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(project_name or "").strip())
    cleaned = cleaned.strip(" .")
    # 空名或纯点号（".", ".."）会给一个稳定的默认名，避免拼出意外路径。
    if not cleaned or set(cleaned) == {"."}:
        return "default"
    return cleaned[:80]


def get_artifacts_dir(project_name: str) -> Path:
    """返回项目的落盘目录（不存在则创建）。"""
    safe_name = sanitize_project_name(project_name)
    target = ARTIFACTS_ROOT / safe_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def dated_file_name(base_name: str, extension: str = "md") -> str:
    """生成带日期的产物文件名，如 `测试报告_20260831.md`。

    报告类产物追加不覆盖（新版本另存），历史版本得以保留。
    """
    return f"{base_name}_{datetime.now().strftime('%Y%m%d')}.{extension.lstrip('.')}"


def write_artifact(project_name: str, file_name: str, content: str) -> str:
    """写入一个落盘产物，返回其绝对路径字符串。"""
    target_dir = get_artifacts_dir(project_name)
    # 文件名同样来自调用方，净化后再拼，杜绝路径穿越。
    safe_file_name = _UNSAFE_CHARS.sub("_", Path(file_name).name)
    if not safe_file_name or set(safe_file_name) == {"."}:
        raise ValueError(f"非法的产物文件名: {file_name!r}")

    target_path = target_dir / safe_file_name
    target_path.write_text(content, encoding="utf-8")
    return str(target_path.resolve())


def list_artifacts(project_name: str) -> list[str]:
    """列出项目已落盘的产物（按修改时间倒序），供前端展示与断点续跑。"""
    target_dir = get_artifacts_dir(project_name)
    files = [path for path in target_dir.iterdir() if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path.resolve()) for path in files]


def record_artifact(artifact_files: list[Any] | None, new_path: str) -> list[str]:
    """把新产物路径并入产物清单，去重且保持顺序。"""
    existing = [str(item) for item in (artifact_files or [])]
    resolved = str(new_path)
    if resolved not in existing:
        existing.append(resolved)
    return existing


def read_artifact(file_path: str) -> str:
    """读取产物内容；文件不存在时返回空串而不是抛错（产物可能已被手工清理）。"""
    path = Path(file_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")
