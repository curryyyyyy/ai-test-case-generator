"""文档解析器注册表。

重构前「按后缀选解析器」这套逻辑在四个地方各写了一份：
    app/app.py、rag/ingest.py、rag/eval_offline.py、rag/index_testcase_kb.py
新增一种格式（如 PDF）要同步改 7 处，漏一处就会出现「上传能过、入库报错」
这类不一致。

现在统一收敛到这里：注册一次，上传、入库、批量脚本、离线评测全部自动支持。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from utils.document_parser.docx_parser import parse_docx
from utils.document_parser.md_parser import parse_markdown
from utils.document_parser.parser import DocumentSection


@dataclass(frozen=True)
class ParserSpec:
    """一种文档格式的解析说明。

    attributes:
        suffix: 归一化后的后缀（含点，小写），如 ".md"
        label: 给用户看的格式名，用于报错提示
        parse_text: 从原始文本解析，仅纯文本格式（如 md）需要提供
        parse_path: 从文件路径解析，二进制格式（如 docx）必须提供
        binary: 是否为二进制格式。上传阶段据此决定是否需要落临时文件
    """

    suffix: str
    label: str
    parse_text: Callable[[str], DocumentSection] | None = None
    parse_path: Callable[[Path], DocumentSection] | None = None
    binary: bool = False

    def parse_bytes(self, data: bytes) -> DocumentSection:
        """直接解析内存中的文件内容，避免每种格式各写一遍临时文件逻辑。"""
        if self.parse_text is not None:
            return self.parse_text(data.decode("utf-8", errors="ignore"))
        if self.parse_path is not None:
            raise ValueError(
                f"{self.label} 为二进制格式，请改用 parse_file() 传入路径"
            )
        raise ValueError(f"{self.label} 未配置任何解析方式")


_PARSERS: dict[str, ParserSpec] = {}


def register_parser(spec: ParserSpec) -> None:
    """注册一种文档格式；同后缀重复注册会覆盖（便于测试或替换实现）。"""
    _PARSERS[spec.suffix.lower()] = spec


def normalize_suffix(value: str | Path) -> str:
    """把文件名、路径或裸后缀归一化为小写后缀（含点）。

    注意不能直接用 Path(x).suffix：对裸后缀 ".docx"，pathlib 会把它当成
    隐藏文件名（stem 为空、suffix 为空），导致 "注册 .docx 却查不到 .docx"。
    这里对以点开头且不含分隔符的输入直接按后缀处理。
    """
    text = str(value).strip()
    if not text:
        return ""

    if text.startswith(".") and not any(sep in text for sep in ("/", "\\")):
        return text.lower()

    return Path(text).suffix.lower()


def supports(suffix: str | Path) -> bool:
    return normalize_suffix(suffix) in _PARSERS


def get_parser(suffix: str | Path) -> ParserSpec:
    """取后缀对应的解析器，不存在时抛出带可选格式的明确错误。"""
    normalized = normalize_suffix(suffix)
    spec = _PARSERS.get(normalized)
    if spec is None:
        raise ValueError(
            f"仅支持 {', '.join(supported_suffixes())}，当前文件: {suffix}"
        )
    return spec


def supported_suffixes() -> list[str]:
    """所有已注册后缀，按注册顺序返回（用于 UI 白名单与错误提示）。"""
    return list(_PARSERS.keys())


def supported_labels() -> list[str]:
    return [spec.label for spec in _PARSERS.values()]


def parse_file(path: str | Path, suffix: str | None = None) -> DocumentSection:
    """按后缀解析磁盘上的文件。"""
    file_path = Path(path)
    spec = get_parser(suffix or file_path.name)
    if spec.parse_path is None:
        if spec.parse_text is None:
            raise ValueError(f"{spec.label} 未配置任何解析方式")
        return spec.parse_text(file_path.read_text(encoding="utf-8"))
    return spec.parse_path(file_path)


def parse_bytes(data: bytes, file_name: str) -> DocumentSection:
    """按文件名后缀解析内存中的文件内容。"""
    spec = get_parser(file_name)
    if spec.binary:
        # 二进制格式统一在这里落临时文件并清理，调用方不必再关心。
        import tempfile

        suffix = spec.suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(data)
            tmp_path = Path(tmp_file.name)
        try:
            assert spec.parse_path is not None
            return spec.parse_path(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return spec.parse_bytes(data)


register_parser(
    ParserSpec(
        suffix=".md",
        label="Markdown",
        parse_text=parse_markdown,
        binary=False,
    )
)
register_parser(
    ParserSpec(
        suffix=".docx",
        label="Word",
        parse_path=parse_docx,
        binary=True,
    )
)
