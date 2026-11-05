"""Input validation (architecture section 6.1)."""

from __future__ import annotations

from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import PayloadTooLarge, UnsupportedMedia, ValidationFailed

SUPPORTED_SUFFIXES = (".txt", ".md")


def decode_text(raw: bytes, settings: Settings) -> str:
    """Strictly decode UTF-8 and normalise line endings once."""
    if len(raw) > settings.max_note_bytes:
        raise PayloadTooLarge("note exceeds byte limit")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValidationFailed("input is not valid UTF-8", code="INVALID_ENCODING") from exc
    return normalize_text(text, settings)


def normalize_text(text: str, settings: Settings) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("﻿"):
        text = text[1:]
    if not text.strip():
        raise ValidationFailed("note is empty", code="EMPTY_NOTE")
    if len(text) > settings.max_note_chars:
        raise PayloadTooLarge("note exceeds character limit")
    return text


def validate_title(title: str, settings: Settings) -> str:
    title = title.strip()
    if not title:
        raise ValidationFailed("title is empty", code="EMPTY_TITLE")
    if len(title) > settings.max_title_chars:
        raise ValidationFailed("title too long", code="TITLE_TOO_LONG")
    return title


def validate_filename(name: str) -> None:
    if not name.lower().endswith(SUPPORTED_SUFFIXES):
        raise UnsupportedMedia("unsupported file type")
