"""Note service: idempotent create, replace, reindex and delete (architecture section 7)."""

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
from notes_graphrag.persistence.base import Store
from notes_graphrag.persistence.models import Submission

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
        self, store: Store, settings: Settings, clock: Callable[[], datetime] = utcnow
    ) -> None:
        self.store, self.settings, self.clock = store, settings, clock

    def _rkey(self, method: str, route: str, key: str) -> str:
        return hashlib.sha256(
            f"{self.store.workspace_id}|{method}|{route}|{key}".encode()
        ).hexdigest()

    @staticmethod
    def _payload_hash(title: str, text: str) -> str:
        return hashlib.sha256(json.dumps([title, text], ensure_ascii=False).encode()).hexdigest()

    def _submit(
        self, method: str, route: str, note_id: str | None, title: str, text: str, key: str
    ) -> Submission:
        title = validate_title(title, self.settings)
        text = normalize_text(text, self.settings)
        return self.store.submit(
            rkey=self._rkey(method, route, key),
            payload_hash=self._payload_hash(title, text),
            note_id=note_id,
            title=title,
            text=text,
            fingerprint=pipeline_fingerprint(self.settings),
            now=self.clock(),
        )

    def create(self, title: str, text: str, idempotency_key: str) -> Submission:
        return self._submit("POST", "/notes", None, title, text, idempotency_key)

    def replace(self, note_id: str, title: str, text: str, idempotency_key: str) -> Submission:
        return self._submit("PUT", f"/notes/{note_id}", note_id, title, text, idempotency_key)

    def reindex(self, note_id: str, idempotency_key: str) -> Submission:
        from notes_graphrag.core.errors import NotFound

        note = self.store.get_note(note_id)
        if note is None or note.deleted_at is not None:
            raise NotFound("note not found")
        cur = self.store.get_revision(note.desired_revision_id)
        assert cur is not None
        return self._submit(
            "POST", f"/notes/{note_id}/reindex", note_id, cur.title, cur.text, idempotency_key
        )

    def retry(self, job_id: str) -> None:
        self.store.retry_job(job_id, self.clock())

    def delete(self, note_id: str) -> None:
        self.store.delete_note(note_id, self.clock())
