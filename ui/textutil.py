"""Safe text rendering helpers (no HTML is ever rendered)."""

from __future__ import annotations

import re

_MD = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$:])")


def md_escape(text: str) -> str:
    return _MD.sub(r"\\\1", text)


def highlight(text: str, start: int, end: int, context: int = 120) -> str:
    """Python slices by code point, matching the API's offset unit (no UTF-16 conversion)."""
    a, b = max(0, start - context), min(len(text), end + context)
    return (
        md_escape(text[a:start])
        + ":orange-background["
        + md_escape(text[start:end])
        + "]"
        + md_escape(text[end:b])
    )
