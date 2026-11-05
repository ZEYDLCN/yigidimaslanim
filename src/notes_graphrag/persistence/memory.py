"""In-memory Store with the same semantics as the Neo4j store.

Used for demos and fast tests. It is not durable: all state is lost when the process exits.
Every read returns an immutable snapshot.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from notes_graphrag.core.errors import (
    Conflict,
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
from notes_graphrag.ingestion.entity_resolution import KnownEntity, new_entity_id, resolve
from notes_graphrag.ingestion.extraction import ValidatedClaim
from notes_graphrag.persistence.base import REQUIRED_INDEXES
from notes_graphrag.persistence.models import (
    GATE_BUSY,
    GATE_IDLE,
    GATE_RECOVERING,
    PROBE_OWNER,
    GateStatus,
    Job,
    Note,
    Revision,
    StagedChunk,
    Submission,
)


@dataclass
class MemClaim:
    claim_id: str
    revision_id: str
    chunk_id: str
    subject_id: str
    object_id: str
    claim: ValidatedClaim


@dataclass
class MemGate:
    state: str = GATE_IDLE
    owner: str | None = None
    token: str | None = None
    expires: datetime | None = None


@dataclass
class MemoryStore:
    workspace_id: str = "default"
    notes: dict[str, Note] = field(default_factory=dict)
    revisions: dict[str, Revision] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    # request_key -> (payload_hash, note_id, revision_id, job_id)
    idem: dict[str, tuple[str, str, str, str]] = field(default_factory=dict)
    staged: dict[str, dict[str, StagedChunk]] = field(default_factory=dict)
    entities: dict[str, KnownEntity] = field(default_factory=dict)
    claims: dict[str, MemClaim] = field(default_factory=dict)
    gates: dict[str, MemGate] = field(default_factory=dict)
    graph_ver: int = 0
    heartbeat_at: datetime | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- notes and submission -------------------------------------------------------------
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
    ) -> Submission:
        chash = hashlib.sha256(text.encode()).hexdigest()
        with self.lock:
            hit = self.idem.get(rkey)
            if hit is not None:
                ph, nid, rid, jid = hit
                if ph != payload_hash:
                    raise IdempotencyKeyReused("idempotency key reused with different payload")
                note = self.notes[nid]
                if note.deleted_at is not None:
                    raise NoteDeleted("note was deleted")
                return Submission(note, self.revisions[rid], self.jobs[jid], replayed=True)

            if note_id is None:
                note = Note(str(uuid.uuid4()), title, now, desired_revision_id="")
            else:
                note = self._live_note(note_id)
                cur = self.revisions[note.desired_revision_id]
                if (
                    cur.content_hash == chash
                    and cur.fingerprint == fingerprint
                    and cur.title == title
                    and cur.status != RevisionStatus.FAILED
                ):
                    job = self.job_for_revision(cur.revision_id)
                    assert job is not None
                    self.idem[rkey] = (payload_hash, note.note_id, cur.revision_id, job.job_id)
                    return Submission(note, cur, job, replayed=True)
                self._supersede_queued(note.desired_revision_id)

            rev = Revision(
                str(uuid.uuid4()),
                note.note_id,
                title,
                text,
                chash,
                fingerprint,
                RevisionStatus.PENDING,
            )
            job = Job(
                str(uuid.uuid4()),
                note.note_id,
                rev.revision_id,
                payload_hash,
                JobStatus.QUEUED,
                available_at=now,
            )
            self.revisions[rev.revision_id] = rev
            self.jobs[job.job_id] = job
            note = replace(note, title=title, desired_revision_id=rev.revision_id)
            self.notes[note.note_id] = note
            self.idem[rkey] = (payload_hash, note.note_id, rev.revision_id, job.job_id)
            return Submission(note, rev, job, replayed=False)

    def _supersede_queued(self, revision_id: str) -> None:
        if not revision_id:
            return
        rev = self.revisions[revision_id]
        if rev.status not in (RevisionStatus.PENDING, RevisionStatus.PROCESSING):
            return
        job = self.job_for_revision(revision_id)
        if job is not None and job.status == JobStatus.QUEUED:
            check_job(job.status, JobStatus.SUPERSEDED)
            check_revision(rev.status, RevisionStatus.SUPERSEDED)
            self.jobs[job.job_id] = replace(job, status=JobStatus.SUPERSEDED)
            self.revisions[revision_id] = replace(rev, status=RevisionStatus.SUPERSEDED)

    def _live_note(self, note_id: str) -> Note:
        note = self.notes.get(note_id)
        if note is None:
            raise NotFound("note not found")
        if note.deleted_at is not None:
            raise NoteDeleted("note was deleted")
        return note

    def get_note(self, note_id: str) -> Note | None:
        with self.lock:
            return self.notes.get(note_id)

    def get_revision(self, revision_id: str) -> Revision | None:
        with self.lock:
            return self.revisions.get(revision_id)

    def list_notes(self, after: tuple[datetime, str] | None, limit: int) -> tuple[list[Note], bool]:
        with self.lock:
            rows = sorted(
                (n for n in self.notes.values() if n.deleted_at is None),
                key=lambda n: (n.created_at, n.note_id),
            )
            if after is not None:
                rows = [n for n in rows if (n.created_at, n.note_id) > after]
            return rows[:limit], len(rows) > limit

    def delete_note(self, note_id: str, now: datetime) -> None:
        with self.lock:
            note = self._live_note(note_id)
            self.notes[note_id] = replace(note, deleted_at=now, active_revision_id=None)
            self.graph_ver += 1
            for j in list(self.jobs.values()):
                if j.note_id == note_id and j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    self.jobs[j.job_id] = replace(
                        j, status=JobStatus.CANCELLED, lease_token=None, lease_expires_at=None
                    )
            for r in list(self.revisions.values()):
                if r.note_id == note_id and r.status not in (
                    RevisionStatus.CANCELLED,
                    RevisionStatus.SUPERSEDED,
                ):
                    self.revisions[r.revision_id] = replace(r, status=RevisionStatus.CANCELLED)

    # ---- jobs -------------------------------------------------------------------------------
    def get_job(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def job_for_revision(self, revision_id: str) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs.values() if j.revision_id == revision_id), None)

    def retry_job(self, job_id: str, now: datetime) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise NotFound("job not found")
            note = self.notes[job.note_id]
            if note.deleted_at is not None:
                raise NoteDeleted("note was deleted")
            if note.desired_revision_id != job.revision_id:
                raise Conflict("job was superseded by a newer revision", code="JOB_SUPERSEDED")
            check_job(job.status, JobStatus.QUEUED)
            rev = self.revisions[job.revision_id]
            check_revision(rev.status, RevisionStatus.PENDING)
            self.jobs[job_id] = replace(
                job,
                status=JobStatus.QUEUED,
                attempt=0,
                error_code=None,
                available_at=now,
                lease_token=None,
                lease_expires_at=None,
            )
            self.revisions[rev.revision_id] = replace(rev, status=RevisionStatus.PENDING)
            return self.jobs[job_id]

    def acquire(self, now: datetime, lease_seconds: int) -> Job | None:
        with self.lock:
            candidates = [
                j
                for j in self.jobs.values()
                if (j.status == JobStatus.QUEUED and (j.available_at or now) <= now)
                or (
                    j.status == JobStatus.RUNNING
                    and j.lease_expires_at is not None
                    and j.lease_expires_at <= now
                )
            ]
            if not candidates:
                return None
            j = min(candidates, key=lambda x: (x.available_at or now, x.job_id))
            token = str(uuid.uuid4())
            new = replace(
                j,
                status=JobStatus.RUNNING,
                attempt=j.attempt + 1,
                lease_token=token,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            self.jobs[j.job_id] = new
            rev = self.revisions[j.revision_id]
            if rev.status == RevisionStatus.PENDING:
                self.revisions[rev.revision_id] = replace(rev, status=RevisionStatus.PROCESSING)
            return new

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

    def record_progress(
        self, job_id: str, token: str, now: datetime, stage: Stage, chunks_total: int
    ) -> None:
        with self.lock:
            job = self._fenced(job_id, token, now)
            self.jobs[job_id] = replace(job, stage=stage, chunks_total=chunks_total)

    def requeue(
        self, job_id: str, token: str, now: datetime, available_at: datetime, code: str
    ) -> bool:
        with self.lock:
            job = self.jobs[job_id]
            if job.status != JobStatus.RUNNING or job.lease_token != token:
                return False
            check_job(job.status, JobStatus.QUEUED)
            self.jobs[job_id] = replace(
                job,
                status=JobStatus.QUEUED,
                lease_token=None,
                lease_expires_at=None,
                available_at=available_at,
                error_code=code,
            )
            return True

    def fail(self, job_id: str, token: str, now: datetime, code: str) -> bool:
        with self.lock:
            job = self.jobs[job_id]
            if job.status != JobStatus.RUNNING or job.lease_token != token:
                return False
            check_job(job.status, JobStatus.FAILED)
            self.jobs[job_id] = replace(
                job,
                status=JobStatus.FAILED,
                error_code=code,
                lease_token=None,
                lease_expires_at=None,
            )
            rev = self.revisions[job.revision_id]
            if rev.status == RevisionStatus.PROCESSING:
                check_revision(rev.status, RevisionStatus.FAILED)
                self.revisions[rev.revision_id] = replace(rev, status=RevisionStatus.FAILED)
            return True

    # ---- staging and publication ------------------------------------------------------------
    def stage_chunk(self, job_id: str, token: str, now: datetime, staged: StagedChunk) -> None:
        with self.lock:
            job = self._fenced(job_id, token, now)
            bucket = self.staged.setdefault(job.revision_id, {})
            bucket[staged.chunk.chunk_id] = staged
            self.jobs[job_id] = replace(job, chunks_done=len(bucket))

    def staged_chunk_ids(self, revision_id: str) -> set[str]:
        with self.lock:
            return set(self.staged.get(revision_id, {}))

    def publish(
        self,
        job_id: str,
        token: str,
        now: datetime,
        expected_chunk_ids: list[str],
        embedding_profile: str,
        staged_profile: str,
    ) -> bool:
        with self.lock:
            job = self._fenced(job_id, token, now)
            rev = self.revisions[job.revision_id]
            note = self.notes[job.note_id]
            if note.deleted_at is not None:
                self.jobs[job_id] = replace(job, status=JobStatus.CANCELLED, lease_token=None)
                return False
            if note.desired_revision_id != rev.revision_id:
                self.jobs[job_id] = replace(job, status=JobStatus.SUPERSEDED, lease_token=None)
                if rev.status in (RevisionStatus.PENDING, RevisionStatus.PROCESSING):
                    self.revisions[rev.revision_id] = replace(rev, status=RevisionStatus.SUPERSEDED)
                return False

            staged = self.staged.get(rev.revision_id, {})
            if set(staged) != set(expected_chunk_ids):
                raise ValidationFailed("incomplete staging", code="INCOMPLETE_STAGING")
            if embedding_profile != staged_profile:
                raise ValidationFailed("embedding profile mismatch", code="PROFILE_MISMATCH")

            for sc in sorted(staged.values(), key=lambda s: s.chunk.ordinal):
                for vc in sc.claims:
                    ids = []
                    for ent in (vc.subject, vc.object):
                        ctx = {vc.subject.name, vc.object.name} - {ent.name}
                        found = resolve(ent.name, ent.kind, ctx, list(self.entities.values()))
                        if found is None:
                            found = new_entity_id()
                            self.entities[found] = KnownEntity(found, ent.kind, ent.name)
                        self.entities[found].context |= ctx
                        ids.append(found)
                    self.claims[vc.claim_id] = MemClaim(
                        vc.claim_id, rev.revision_id, sc.chunk.chunk_id, ids[0], ids[1], vc
                    )

            prev_id = note.active_revision_id
            if prev_id and prev_id != rev.revision_id:
                prev = self.revisions[prev_id]
                check_revision(prev.status, RevisionStatus.SUPERSEDED)
                self.revisions[prev_id] = replace(prev, status=RevisionStatus.SUPERSEDED)
            check_revision(rev.status, RevisionStatus.READY)
            self.revisions[rev.revision_id] = replace(rev, status=RevisionStatus.READY)
            self.notes[note.note_id] = replace(note, active_revision_id=rev.revision_id)
            self.jobs[job_id] = replace(
                job,
                status=JobStatus.SUCCEEDED,
                stage=Stage.PUBLISHING,
                lease_token=None,
                lease_expires_at=None,
                chunks_done=len(staged),
            )
            self.graph_ver += 1
            return True

    # ---- workspace state --------------------------------------------------------------------
    def heartbeat(self, now: datetime) -> None:
        with self.lock:
            self.heartbeat_at = now

    def last_heartbeat(self) -> datetime | None:
        with self.lock:
            return self.heartbeat_at

    def graph_version(self) -> int:
        with self.lock:
            return self.graph_ver

    # ---- shared inference slot ----------------------------------------------------------------
    def ensure_gate(self, endpoint_id: str) -> None:
        with self.lock:
            self.gates.setdefault(endpoint_id, MemGate())

    def gate_status(self, endpoint_id: str) -> GateStatus:
        with self.lock:
            g = self.gates.setdefault(endpoint_id, MemGate())
            return GateStatus(g.state, g.owner, g.expires)

    def gate_try_acquire(
        self, endpoint_id: str, owner: str, token: str, now: datetime, lease_seconds: int
    ) -> bool:
        with self.lock:
            g = self.gates.setdefault(endpoint_id, MemGate())
            if g.state == GATE_BUSY and g.expires is not None and g.expires <= now:
                # The holder vanished or timed out: the server may still be computing.
                g.state, g.owner, g.token, g.expires = GATE_RECOVERING, None, None, None
                return False
            allowed = g.state == GATE_RECOVERING if owner == PROBE_OWNER else g.state == GATE_IDLE
            if not allowed:
                return False
            g.state = GATE_BUSY
            g.owner = owner
            g.token = token
            g.expires = now + timedelta(seconds=lease_seconds)
            return True

    def gate_heartbeat(
        self, endpoint_id: str, token: str, now: datetime, lease_seconds: int
    ) -> None:
        with self.lock:
            g = self.gates.setdefault(endpoint_id, MemGate())
            if g.state == GATE_BUSY and g.token == token:
                g.expires = now + timedelta(seconds=lease_seconds)

    def gate_release(self, endpoint_id: str, token: str, target: str) -> None:
        if target not in (GATE_IDLE, GATE_RECOVERING):
            raise ValueError("invalid gate target")
        with self.lock:
            g = self.gates.setdefault(endpoint_id, MemGate())
            if g.state == GATE_BUSY and g.token == token:
                g.state, g.owner, g.token, g.expires = target, None, None, None

    # ---- operations -------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        from notes_graphrag.persistence.migrations import latest_version

        return {
            "reachable": True,
            "schema_version": latest_version(),
            "indexes_online": None,
            "required_indexes": list(REQUIRED_INDEXES),
            "persistence": "memory (non-durable)",
        }

    def schema_version(self) -> int:
        from notes_graphrag.persistence.migrations import latest_version

        return latest_version()
