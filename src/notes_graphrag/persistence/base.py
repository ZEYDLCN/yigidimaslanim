"""Storage contract shared by the in-memory reference store and the Neo4j store.

Both implementations must pass the same contract tests (tests/contract).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from notes_graphrag.domain.state_machines import Stage
from notes_graphrag.persistence.models import (
    GateStatus,
    Job,
    Note,
    ProbeRecord,
    Revision,
    StagedChunk,
    Submission,
)

REQUIRED_INDEXES = ("chunk_text_v1", "entity_names_v1", "chunk_embedding_v1")


class Store(Protocol):
    workspace_id: str

    # notes and idempotent submission
    def submit(
        self,
        *,
        rkey: str,
        payload_hash: str,
        note_id: str | None,
        title: str,
        text: str,
        fingerprint: str,
        now: datetime,
    ) -> Submission: ...
    def get_note(self, note_id: str) -> Note | None: ...
    def get_revision(self, revision_id: str) -> Revision | None: ...
    def list_notes(
        self, after: tuple[datetime, str] | None, limit: int
    ) -> tuple[list[Note], bool]: ...
    def delete_note(self, note_id: str, now: datetime) -> None: ...

    # jobs
    def get_job(self, job_id: str) -> Job | None: ...
    def job_for_revision(self, revision_id: str) -> Job | None: ...
    def retry_job(self, job_id: str, now: datetime) -> Job: ...
    def acquire(self, now: datetime, lease_seconds: int) -> Job | None: ...
    def record_progress(
        self, job_id: str, token: str, now: datetime, stage: Stage, chunks_total: int
    ) -> None: ...
    def requeue(
        self, job_id: str, token: str, now: datetime, available_at: datetime, code: str
    ) -> bool: ...
    def fail(self, job_id: str, token: str, now: datetime, code: str) -> bool: ...

    # staging and publication (fenced by lease token)
    def stage_chunk(self, job_id: str, token: str, now: datetime, staged: StagedChunk) -> None: ...
    def staged_chunk_ids(self, revision_id: str) -> set[str]: ...
    def publish(
        self,
        job_id: str,
        token: str,
        now: datetime,
        expected_chunk_ids: list[str],
        embedding_profile: str,
        staged_profile: str,
    ) -> bool: ...

    # worker heartbeat and workspace state
    def heartbeat(self, now: datetime) -> None: ...
    def last_heartbeat(self) -> datetime | None: ...
    def graph_version(self) -> int: ...

    # capability probes (architecture 2.2, 14.2): recorded results, bound to a configuration
    def record_probe(
        self, fingerprint: str, passed: bool, results: list[dict[str, Any]], now: datetime
    ) -> None: ...
    def latest_probe(self) -> ProbeRecord | None: ...

    # shared inference slot
    def ensure_gate(self, endpoint_id: str) -> None: ...
    def gate_status(self, endpoint_id: str) -> GateStatus: ...
    def gate_try_acquire(
        self, endpoint_id: str, owner: str, token: str, now: datetime, lease_seconds: int
    ) -> bool: ...
    def gate_heartbeat(
        self, endpoint_id: str, token: str, now: datetime, lease_seconds: int
    ) -> None: ...
    def gate_release(self, endpoint_id: str, token: str, target: str) -> None: ...

    # retention (architecture 7.2, 7.3)
    def purge(
        self, now: datetime, content_retention_days: float, idempotency_retention_days: float
    ) -> dict[str, int]: ...

    # operations
    def health(self) -> dict[str, Any]: ...
    def schema_version(self) -> int: ...
