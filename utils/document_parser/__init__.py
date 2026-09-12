from .parser import DocumentSection, section_to_dict
from .md_parser import parse_markdown
from .docx_parser import parse_docx
from .registry import (
    ParserSpec,
    get_parser,
    parse_bytes,
    parse_file,
    register_parser,
    supported_labels,
    supported_suffixes,
    supports,
)

__all__ = [
    "DocumentSection",
    "ParserSpec",
    "get_parser",
    "parse_bytes",
    "parse_docx",
    "parse_file",
    "parse_markdown",
    "register_parser",
    "section_to_dict",
    "supported_labels",
    "supported_suffixes",
    "supports",
]
