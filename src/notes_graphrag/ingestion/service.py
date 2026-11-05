"""Note service: idempotent create/replace/delete (architecture section 7)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.chunking import CHUNKER_VERSION
from notes_graphrag.ingestion.entity_resolution import NORMALIZATION_VERSION
from notes_graphrag.ingestion.extraction import SCHEMA_VERSION
from notes_graphrag.ingestion.validation import normalize_text, validate_title
from notes_graphrag.persistence.memory import Job, MemoryStore, Note, Revision

PROMPT_VERSION = "extract_v1"


def pipeline_fingerprint(settings: Settings) -> str:
    parts = [
        CHUNKER_VERSION,
        SCHEMA_VERSION,
        PROMPT_VERSION,
        NORMALIZATION_VERSION,
        settings.embedding_profile,
        settings.extraction_model,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def utcnow() -> datetime:
    return datetime.now(UTC)


class NoteService:
    def __init__(
        self, store: MemoryStore, settings: Settings, clock: Callable[[], datetime] = utcnow
    ) -> None:
        self.store, self.settings, self.clock = store, settings, clock

    def _payload_hash(self, title: str, text: str) -> str:
        return hashlib.sha256(json.dumps([title, text], ensure_ascii=False).encode()).hexdigest()

    def create(self, title: str, text: str, idempotency_key: str) -> tuple[Note, Revision, Job]:
        return self._submit("POST", "/notes", None, title, text, idempotency_key)

    def replace(
        self, note_id: str, title: str, text: str, idempotency_key: str
    ) -> tuple[Note, Revision, Job]:
        return self._submit("PUT", f"/notes/{note_id}", note_id, title, text, idempotency_key)

    def delete(self, note_id: str) -> None:
        self.store.delete_note(note_id, self.clock())

    def _submit(
        self, method: str, route: str, note_id: str | None, title: str, text: str, key: str
    ) -> tuple[Note, Revision, Job]:
        title = validate_title(title, self.settings)
        text = normalize_text(text, self.settings)
        note, rev, job, _ = self.store.submit(
            rkey=self.store.request_key(method, route, key),
            payload_hash=self._payload_hash(title, text),
            note_id=note_id,
            title=title,
            text=text,
            fingerprint=pipeline_fingerprint(self.settings),
            now=self.clock(),
        )
        return note, rev, job
