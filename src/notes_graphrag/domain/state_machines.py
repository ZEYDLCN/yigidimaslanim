"""Job and revision lifecycles (architecture sections 6.6 and 7.1)."""

from __future__ import annotations

from enum import StrEnum

from notes_graphrag.core.errors import Conflict


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"


class Stage(StrEnum):
    CHUNKING = "CHUNKING"
    EXTRACTING = "EXTRACTING"
    EMBEDDING = "EMBEDDING"
    PERSISTING = "PERSISTING"
    PUBLISHING = "PUBLISHING"


class RevisionStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"


J = JobStatus
R = RevisionStatus

JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    J.QUEUED: frozenset({J.RUNNING, J.CANCELLED, J.SUPERSEDED}),
    J.RUNNING: frozenset({J.SUCCEEDED, J.FAILED, J.SUPERSEDED, J.CANCELLED, J.QUEUED}),
    J.FAILED: frozenset({J.QUEUED}),
    J.SUCCEEDED: frozenset(),
    J.SUPERSEDED: frozenset(),
    J.CANCELLED: frozenset(),
}

REVISION_TRANSITIONS: dict[RevisionStatus, frozenset[RevisionStatus]] = {
    R.PENDING: frozenset({R.PROCESSING, R.CANCELLED, R.SUPERSEDED}),
    R.PROCESSING: frozenset({R.READY, R.FAILED, R.CANCELLED, R.SUPERSEDED, R.PENDING}),
    R.READY: frozenset({R.SUPERSEDED, R.CANCELLED}),
    R.FAILED: frozenset({R.PENDING, R.CANCELLED}),
    R.SUPERSEDED: frozenset(),
    R.CANCELLED: frozenset(),
}


def check_job(old: JobStatus, new: JobStatus) -> None:
    if new not in JOB_TRANSITIONS[old]:
        raise Conflict(f"illegal job transition {old} -> {new}", code="ILLEGAL_TRANSITION")


def check_revision(old: RevisionStatus, new: RevisionStatus) -> None:
    if new not in REVISION_TRANSITIONS[old]:
        raise Conflict(f"illegal revision transition {old} -> {new}", code="ILLEGAL_TRANSITION")
