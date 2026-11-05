"""Deterministic chunker: chunks are exact slices of the revision text.

Offsets are zero-based, half-open Unicode code point ranges.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass

from notes_graphrag.core.errors import PayloadTooLarge

CHUNKER_VERSION = "chunker-v1"
_NS = uuid.UUID("6f1c1d0e-6a53-4c2b-9d0a-2f3c8f0d7b11")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate (about 3.5 chars per token)."""
    return math.ceil(len(text) / 3.5) if text else 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ordinal: int
    start_char: int
    end_char: int
    text: str
    token_count: int


def _units(text: str, max_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        while e - s > max_chars:
            cut = text.rfind(" ", s + max_chars // 2, s + max_chars)
            cut = cut + 1 if cut != -1 else s + max_chars
            out.append((s, cut))
            s = cut
        out.append((s, e))
    return out


def chunk_text(
    text: str,
    revision_id: str,
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 60,
    max_chunks: int = 128,
) -> list[Chunk]:
    max_chars = int(target_tokens * 3.5)
    units = _units(text, max_chars)
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j = i
        while j + 1 < len(units) and (
            estimate_tokens(text[units[i][0] : units[j + 1][1]]) <= target_tokens
        ):
            j += 1
        start, end = units[i][0], units[j][1]
        chunks.append(_make(text, revision_id, len(chunks), start, end))
        if len(chunks) > max_chunks:
            raise PayloadTooLarge("too many chunks", code="TOO_MANY_CHUNKS")
        if j + 1 >= len(units):
            break
        nxt = j + 1
        while (
            nxt - 1 > i and estimate_tokens(text[units[nxt - 1][0] : units[j][1]]) <= overlap_tokens
        ):
            nxt -= 1
        i = max(nxt, i + 1)
    return chunks


def _make(text: str, revision_id: str, ordinal: int, start: int, end: int) -> Chunk:
    cid = uuid.uuid5(_NS, f"{revision_id}|{ordinal}|{start}|{end}|{CHUNKER_VERSION}")
    piece = text[start:end]
    return Chunk(str(cid), ordinal, start, end, piece, estimate_tokens(piece))
