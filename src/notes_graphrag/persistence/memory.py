"""In-memory reference repository.

Implements the lifecycle semantics of architecture sections 6.6/7 (idempotency,
lease fencing, atomic publication, tombstones) so the logic can be tested without
Neo4j. It is NOT the production store; the Neo4j repository must pass the same tests.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from notes_graphrag.core.errors import (
    IdempotencyKeyReused,
    NoteDeleted,
    NotFound,
    StaleLease,
    ValidationFailed,
)
from notes_graphrag.domain.state_machines import (
    JobStatus,
    RevisionStatus,
    Stage,
    check_job,
    check_revision,
)
from notes_graphrag.ingestion.chunking import Chunk
from notes_graphrag.ingestion.entity_resolution import (
    KnownEntity,
    new_entity_id,
    normalize_name,
    resolve,
)
from notes_graphrag.ingestion.extraction import ValidatedClaim


@dataclass
class Note:
    note_id: str
    title: str
    created_at: datetime
    desired_revision_id: str
    active_revision_id: str | None = None
    deleted_at: datetime | None = None


@dataclass
class Revision:
    revision_id: str
    note_id: str
    title: str
    text: str
    content_hash: str
    fingerprint: str
    status: RevisionStatus = RevisionStatus.PENDING


@dataclass
class Job:
    job_id: str
    note_id: str
    revision_id: str
    request_key: str
    payload_hash: str
    status: JobStatus = JobStatus.QUEUED
    stage: Stage | None = None
    attempt: int = 0
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    available_at: datetime | None = None
    error_code: str | None = None
    chunks_total: int = 0
    chunks_done: int = 0


@dataclass
class StagedChunk:
    chunk: Chunk
    embedding: list[float]
    claims: list[ValidatedClaim]


@dataclass
class KeyRecord:
    note_id: str
    revision_id: str
    job_id: str
    payload_hash: str


@dataclass
class StoredEntity:
    known: KnownEntity


@dataclass
class StoredClaim:
    claim: ValidatedClaim
    subject_id: str
    object_id: str
    revision_id: str
    chunk_id: str


@dataclass
class MemoryStore:
    workspace_id: str = "default"
    graph_version: int = 0
    notes: dict[str, Note] = field(default_factory=dict)
    revisions: dict[str, Revision] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    keys: dict[str, KeyRecord] = field(default_factory=dict)
    staged: dict[str, dict[str, StagedChunk]] = field(default_factory=dict)  # revision -> chunk
    entities: dict[str, KnownEntity] = field(default_factory=dict)
    claims: dict[str, StoredClaim] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- request creation (note + revision + job in one atomic step) ----
    def request_key(self, method: str, route: str, key: str) -> str:
        return hashlib.sha256(f"{self.workspace_id}|{method}|{route}|{key}".encode()).hexdigest()

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
    ) -> tuple[Note, Revision, Job, bool]:
        """Returns (note, revision, job, replayed)."""
        with self.lock:
            rec = self.keys.get(rkey)
            if rec is not None:
                if rec.payload_hash != payload_hash:
                    raise IdempotencyKeyReused("idempotency key reused with different payload")
                note = self.notes[rec.note_id]
                if note.deleted_at is not None:
                    raise NoteDeleted("note was deleted")
                return note, self.revisions[rec.revision_id], self.jobs[rec.job_id], True

            chash = hashlib.sha256(text.encode()).hexdigest()
            if note_id is None:
                note_id = str(uuid.uuid4())
                note = Note(note_id, title, now, "")
                self.notes[note_id] = note
            else:
                note = self._live_note(note_id)
                cur = self.revisions[note.desired_revision_id]
                if (
                    cur.content_hash == chash
                    and cur.fingerprint == fingerprint
                    and cur.title == title
                    and cur.status != RevisionStatus.FAILED
                ):
                    job = next(j for j in self.jobs.values() if j.revision_id == cur.revision_id)
                    self.keys[rkey] = KeyRecord(note_id, cur.revision_id, job.job_id, payload_hash)
                    return note, cur, job, True

            rev = Revision(str(uuid.uuid4()), note_id, title, text, chash, fingerprint)
            self.revisions[rev.revision_id] = rev
            old = note.desired_revision_id
            note.desired_revision_id = rev.revision_id
            note.title = title
            if old:
                self._supersede_pending(old)
            job = Job(
                str(uuid.uuid4()), note_id, rev.revision_id, rkey, payload_hash, available_at=now
            )
            self.jobs[job.job_id] = job
            self.keys[rkey] = KeyRecord(note_id, rev.revision_id, job.job_id, payload_hash)
            return note, rev, job, False

    def _supersede_pending(self, revision_id: str) -> None:
        rev = self.revisions[revision_id]
        if rev.status in (RevisionStatus.PENDING, RevisionStatus.PROCESSING):
            # Running jobs notice the change at their next fenced write.
            for j in self.jobs.values():
                if j.revision_id == revision_id and j.status == JobStatus.QUEUED:
                    check_job(j.status, JobStatus.SUPERSEDED)
                    j.status = JobStatus.SUPERSEDED
                    check_revision(rev.status, RevisionStatus.SUPERSEDED)
                    rev.status = RevisionStatus.SUPERSEDED

    def _live_note(self, note_id: str) -> Note:
        note = self.notes.get(note_id)
        if note is None:
            raise NotFound("note not found")
        if note.deleted_at is not None:
            raise NoteDeleted("note was deleted")
        return note

    def delete_note(self, note_id: str, now: datetime) -> None:
        with self.lock:
            note = self._live_note(note_id)
            note.deleted_at = now
            note.active_revision_id = None
            self.graph_version += 1
            for j in self.jobs.values():
                if j.note_id == note_id and j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    j.status = JobStatus.CANCELLED
            for r in self.revisions.values():
                if r.note_id == note_id and r.status not in (
                    RevisionStatus.CANCELLED,
                    RevisionStatus.SUPERSEDED,
                ):
                    r.status = RevisionStatus.CANCELLED

    # ---- worker side ----
    def acquire(self, now: datetime, lease_seconds: int) -> Job | None:
        from datetime import timedelta

        with self.lock:
            for j in sorted(self.jobs.values(), key=lambda x: x.available_at or now):
                expired = j.status == JobStatus.RUNNING and (
                    j.lease_expires_at is not None and j.lease_expires_at <= now
                )
                if expired:
                    j.status = JobStatus.QUEUED
                if j.status == JobStatus.QUEUED and (j.available_at or now) <= now:
                    check_job(j.status, JobStatus.RUNNING)
                    j.status = JobStatus.RUNNING
                    j.attempt += 1
                    j.lease_token = str(uuid.uuid4())
                    j.lease_expires_at = now + timedelta(seconds=lease_seconds)
                    rev = self.revisions[j.revision_id]
                    if rev.status in (RevisionStatus.PENDING, RevisionStatus.FAILED):
                        rev.status = RevisionStatus.PROCESSING
                    return j
            return None

    def _fenced(self, job_id: str, token: str, now: datetime) -> Job:
        job = self.jobs[job_id]
        if (
            job.status != JobStatus.RUNNING
            or job.lease_token != token
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        ):
            raise StaleLease("lease lost")
        return job

    def stage_chunk(self, job_id: str, token: str, now: datetime, staged: StagedChunk) -> None:
        with self.lock:
            job = self._fenced(job_id, token, now)
            self.staged.setdefault(job.revision_id, {})[staged.chunk.chunk_id] = staged
            job.chunks_done = len(self.staged[job.revision_id])

    def staged_for(self, revision_id: str) -> dict[str, StagedChunk]:
        return dict(self.staged.get(revision_id, {}))

    def publish(
        self,
        job_id: str,
        token: str,
        now: datetime,
        expected_chunk_ids: list[str],
        embedding_profile: str,
        staged_profile: str,
    ) -> bool:
        """Atomic publication. Returns False if the revision was superseded."""
        with self.lock:
            job = self._fenced(job_id, token, now)
            rev = self.revisions[job.revision_id]
            note = self.notes[job.note_id]
            if note.deleted_at is not None:
                job.status = JobStatus.CANCELLED
                return False
            if note.desired_revision_id != rev.revision_id:
                job.status = JobStatus.SUPERSEDED
                rev.status = RevisionStatus.SUPERSEDED
                return False
            staged = self.staged.get(rev.revision_id, {})
            if set(staged) != set(expected_chunk_ids):
                raise ValidationFailed("incomplete staging", code="INCOMPLETE_STAGING")
            if embedding_profile != staged_profile:
                raise ValidationFailed("embedding profile mismatch", code="PROFILE_MISMATCH")

            for sc in staged.values():
                for c in sc.claims:
                    ctx = {c.subject.name, c.object.name}
                    ids = []
                    for ent in (c.subject, c.object):
                        found = resolve(
                            ent.name, ent.kind, ctx - {ent.name}, list(self.entities.values())
                        )
                        if found is None:
                            found = new_entity_id()
                            self.entities[found] = KnownEntity(found, ent.kind, ent.name)
                        self.entities[found].context |= ctx - {ent.name}
                        ids.append(found)
                    self.claims[c.claim_id] = StoredClaim(
                        c, ids[0], ids[1], rev.revision_id, sc.chunk.chunk_id
                    )

            if note.active_revision_id:
                prev = self.revisions[note.active_revision_id]
                check_revision(prev.status, RevisionStatus.SUPERSEDED)
                prev.status = RevisionStatus.SUPERSEDED
            note.active_revision_id = rev.revision_id
            rev.status = RevisionStatus.READY
            job.status = JobStatus.SUCCEEDED
            self.graph_version += 1
            return True

    # ---- read side: always filtered through the active revision ----
    def active_chunks(self) -> list[tuple[Note, Revision, StagedChunk]]:
        out = []
        with self.lock:
            for n in self.notes.values():
                if n.deleted_at is None and n.active_revision_id:
                    rev = self.revisions[n.active_revision_id]
                    if rev.status == RevisionStatus.READY:
                        out += [
                            (n, rev, sc) for sc in self.staged.get(rev.revision_id, {}).values()
                        ]
        return out

    def active_claims(self) -> list[StoredClaim]:
        active = {rev.revision_id for _, rev, _ in self.active_chunks()}
        return [c for c in self.claims.values() if c.revision_id in active]

    def entity_by_name(self, name: str) -> list[KnownEntity]:
        key = normalize_name(name)
        return [e for e in self.entities.values() if key in e.names()]
