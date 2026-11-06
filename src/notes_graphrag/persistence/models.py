"""Immutable snapshots returned by every Store implementation.

Callers never mutate store state directly; they receive these snapshots and call store methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from notes_graphrag.domain.state_machines import JobStatus, RevisionStatus, Stage
from notes_graphrag.ingestion.chunking import Chunk
from notes_graphrag.ingestion.extraction import ValidatedClaim

GATE_IDLE = "IDLE"
GATE_BUSY = "BUSY"
GATE_RECOVERING = "RECOVERING"
PROBE_OWNER = "recovery-probe"


@dataclass(frozen=True)
class Note:
    note_id: str
    title: str
    created_at: datetime
    desired_revision_id: str
    active_revision_id: str | None = None
    deleted_at: datetime | None = None


@dataclass(frozen=True)
class Revision:
    revision_id: str
    note_id: str
    title: str
    text: str
    content_hash: str
    fingerprint: str
    status: RevisionStatus


@dataclass(frozen=True)
class Job:
    job_id: str
    note_id: str
    revision_id: str
    payload_hash: str
    status: JobStatus
    stage: Stage | None = None
    attempt: int = 0
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    available_at: datetime | None = None
    error_code: str | None = None
    chunks_total: int = 0
    chunks_done: int = 0


@dataclass(frozen=True)
class StagedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]
    claims: tuple[ValidatedClaim, ...]


@dataclass(frozen=True)
class Submission:
    note: Note
    revision: Revision
    job: Job
    replayed: bool


@dataclass(frozen=True)
class ProbeRecord:
    fingerprint: str
    passed: bool
    recorded_at: datetime
    results: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class GateStatus:
    state: str
    owner_id: str | None
    lease_expires_at: datetime | None
