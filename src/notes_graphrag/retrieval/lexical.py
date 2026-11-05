"""Lucene query escaping for user input (parameterisation alone is not enough)."""

from __future__ import annotations

import re

_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def escape_lucene(query: str) -> str:
    escaped = _SPECIAL.sub(r"\\\1", query)
    # Boolean operator words become plain terms when lower-cased and quoted away.
    return " ".join(f'"{w}"' if w in {"AND", "OR", "NOT"} else w for w in escaped.split())
