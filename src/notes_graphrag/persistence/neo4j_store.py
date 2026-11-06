"""Neo4j implementation of the Store contract (architecture sections 5-7).

Locking discipline:
- Every fenced write first performs a SET on the coordinating node. That takes the Neo4j write
  lock before the WHERE check runs, so the check reads committed state that cannot change until
  the transaction ends.
- Lock order is Workspace -> Note -> IngestionJob, which prevents deadlocks between publication
  (which touches all three) and deletion (Note -> Job).
- No LLM or embedding call happens inside a transaction.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar

from neo4j import Driver, ManagedTransaction
from neo4j.exceptions import ConstraintError, Neo4jError, ServiceUnavailable, SessionExpired

from notes_graphrag.core.errors import (
    Conflict,
    IdempotencyKeyReused,
    NoteDeleted,
    NotFound,
    StaleLease,
    ValidationFailed,
)
from notes_graphrag.domain.ontology import EntityKind
from notes_graphrag.domain.state_machines import (
    JobStatus,
    RevisionStatus,
    Stage,
    check_job,
    check_revision,
)
from notes_graphrag.ingestion.entity_resolution import (
    KnownEntity,
    new_entity_id,
    normalize_name,
    resolve,
)
from notes_graphrag.persistence.base import REQUIRED_INDEXES
from notes_graphrag.persistence.migrations import latest_version
from notes_graphrag.persistence.models import (
    GATE_BUSY,
    GATE_IDLE,
    GATE_RECOVERING,
    PROBE_OWNER,
    GateStatus,
    Job,
    Note,
    ProbeRecord,
    Revision,
    StagedChunk,
    Submission,
)
from notes_graphrag.persistence.serialize import (
    dump_claims,
    load_claims,
    opt_dt,
    req_dt,
    to_iso,
)

T = TypeVar("T")
TRANSIENT_ERRORS = (ServiceUnavailable, SessionExpired, Neo4jError)


def _note(d: Any) -> Note:
    d = dict(d)
    return Note(
        note_id=d["note_id"],
        title=d["title"],
        created_at=req_dt(d["created_at"]),
        desired_revision_id=d["desired_revision_id"],
        active_revision_id=d.get("active_revision_id"),
        deleted_at=opt_dt(d.get("deleted_at")),
    )


def _rev(d: Any) -> Revision:
    d = dict(d)
    return Revision(
        revision_id=d["revision_id"],
        note_id=d["note_id"],
        title=d["title"],
        text=d["text"],
        content_hash=d["content_hash"],
        fingerprint=d["fingerprint"],
        status=RevisionStatus(d["status"]),
    )


def _job(d: Any) -> Job:
    d = dict(d)
    return Job(
        job_id=d["job_id"],
        note_id=d["note_id"],
        revision_id=d["revision_id"],
        payload_hash=d["payload_hash"],
        status=JobStatus(d["status"]),
        stage=Stage(d["stage"]) if d.get("stage") else None,
        attempt=int(d.get("attempt") or 0),
        lease_token=d.get("lease_token"),
        lease_expires_at=opt_dt(d.get("lease_expires_at")),
        available_at=opt_dt(d.get("available_at")),
        error_code=d.get("error_code"),
        chunks_total=int(d.get("chunks_total") or 0),
        chunks_done=int(d.get("chunks_done") or 0),
    )


class Neo4jStore:
    def __init__(self, driver: Driver, database: str, workspace_id: str) -> None:
        self._driver = driver
        self._db = database
        self.workspace_id = workspace_id
        self._exec_write(
            "MERGE (w:Workspace {workspace_id: $ws}) ON CREATE SET w.graph_version = 0",
            ws=workspace_id,
        )

    def close(self) -> None:
        self._driver.close()

    # ---- transaction helpers ------------------------------------------------------------------
    def _read(self, fn: Callable[[ManagedTransaction], T]) -> T:
        with self._driver.session(database=self._db) as s:
            return s.execute_read(fn)

    def _write(self, fn: Callable[[ManagedTransaction], T]) -> T:
        with self._driver.session(database=self._db) as s:
            return s.execute_write(fn)

    def _rows(self, q: str, **p: Any) -> list[dict[str, Any]]:
        return self._read(lambda tx: tx.run(q, **p).data())

    def _exec_write(self, q: str, **p: Any) -> list[dict[str, Any]]:
        return self._write(lambda tx: tx.run(q, **p).data())

    # ---- notes and submission -----------------------------------------------------------------
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
        # Two identical requests can race past the read and collide on the unique idempotency
        # record. The loser retries and then replays the winner's result (architecture 7.2).
        for attempt in range(3):
            try:
                return self._submit_once(
                    rkey=rkey,
                    payload_hash=payload_hash,
                    note_id=note_id,
                    title=title,
                    text=text,
                    fingerprint=fingerprint,
                    now=now,
                )
            except ConstraintError:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def _submit_once(
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
        ws = self.workspace_id
        chash = hashlib.sha256(text.encode()).hexdigest()
        ts = to_iso(now)

        def fn(tx: ManagedTransaction) -> Submission:
            # Replays use the ids stored on the record, never relationships (migration 003).
            hit = tx.run(
                """
                MATCH (i:IdempotencyRecord {workspace_id: $ws, request_key: $rkey})
                MATCH (n:Note {workspace_id: $ws, note_id: i.note_id})
                RETURN i.payload_hash AS ph, i.revision_id AS rid, i.job_id AS jid, n
                """,
                ws=ws,
                rkey=rkey,
            ).data()
            if hit:
                row = hit[0]
                if row["ph"] != payload_hash:
                    raise IdempotencyKeyReused("idempotency key reused with different payload")
                note = _note(row["n"])
                if note.deleted_at is not None:
                    raise NoteDeleted("note was deleted")
                rev = tx.run(
                    "MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid}) RETURN r",
                    ws=ws,
                    rid=row["rid"],
                ).data()[0]["r"]
                job = tx.run(
                    "MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid}) RETURN j",
                    ws=ws,
                    jid=row["jid"],
                ).data()[0]["j"]
                return Submission(note, _rev(rev), _job(job), replayed=True)

            if note_id is not None:
                # Lock the note first, then read its committed desired revision.
                locked = tx.run(
                    "MATCH (n:Note {workspace_id: $ws, note_id: $nid}) SET n.lock_touch = $ts "
                    "RETURN n",
                    ws=ws,
                    nid=note_id,
                    ts=ts,
                ).data()
                if not locked or _note(locked[0]["n"]).deleted_at is not None:
                    raise NotFound("note not found")
                note = _note(locked[0]["n"])
                cur_rows = tx.run(
                    """
                    MATCH (n:Note {workspace_id: $ws, note_id: $nid})-[:HAS_REVISION]->
                          (r:NoteRevision {revision_id: n.desired_revision_id})
                    MATCH (j:IngestionJob)-[:BUILDS]->(r)
                    RETURN r, j
                    """,
                    ws=ws,
                    nid=note_id,
                ).data()
                cur = _rev(cur_rows[0]["r"])
                if (
                    cur.content_hash == chash
                    and cur.fingerprint == fingerprint
                    and cur.title == title
                    and cur.status != RevisionStatus.FAILED
                ):
                    tx.run(
                        """
                        MATCH (n:Note {workspace_id: $ws, note_id: $nid})
                        MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})
                        MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                        CREATE (i:IdempotencyRecord {workspace_id: $ws, request_key: $rkey,
                                                     payload_hash: $ph, created_at: $ts,
                                                     note_id: $nid, revision_id: $rid,
                                                     job_id: $jid})
                        CREATE (i)-[:FOR_NOTE]->(n), (i)-[:FOR_REVISION]->(r),
                               (i)-[:FOR_JOB]->(j)
                        """,
                        ws=ws,
                        nid=note_id,
                        rid=cur.revision_id,
                        jid=_job(cur_rows[0]["j"]).job_id,
                        rkey=rkey,
                        ph=payload_hash,
                        ts=ts,
                    ).consume()
                    return Submission(note, cur, _job(cur_rows[0]["j"]), replayed=True)
                if cur.status in (RevisionStatus.PENDING, RevisionStatus.PROCESSING):
                    tx.run(
                        """
                        MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})
                              <-[:BUILDS]-(j:IngestionJob {status: 'QUEUED'})
                        SET j.status = 'SUPERSEDED', r.status = 'SUPERSEDED'
                        """,
                        ws=ws,
                        rid=cur.revision_id,
                    ).consume()
                nid = note.note_id
            else:
                nid = str(uuid.uuid4())

            rid = str(uuid.uuid4())
            jid = str(uuid.uuid4())
            tx.run(
                """
                MERGE (n:Note {workspace_id: $ws, note_id: $nid})
                ON CREATE SET n.created_at = $ts
                SET n.title = $title, n.desired_revision_id = $rid, n.updated_at = $ts
                CREATE (r:NoteRevision {workspace_id: $ws, revision_id: $rid, note_id: $nid,
                                        title: $title, text: $text, content_hash: $chash,
                                        fingerprint: $fp, status: 'PENDING', created_at: $ts})
                CREATE (n)-[:HAS_REVISION]->(r)
                CREATE (j:IngestionJob {workspace_id: $ws, job_id: $jid, request_key: $rkey,
                                        payload_hash: $ph, note_id: $nid, revision_id: $rid,
                                        status: 'QUEUED', attempt: 0, available_at: $ts,
                                        chunks_total: 0, chunks_done: 0})
                CREATE (j)-[:FOR_NOTE]->(n), (j)-[:BUILDS]->(r)
                CREATE (i:IdempotencyRecord {workspace_id: $ws, request_key: $rkey,
                                             payload_hash: $ph, created_at: $ts,
                                             note_id: $nid, revision_id: $rid, job_id: $jid})
                CREATE (i)-[:FOR_NOTE]->(n), (i)-[:FOR_REVISION]->(r), (i)-[:FOR_JOB]->(j)
                """,
                ws=ws,
                nid=nid,
                rid=rid,
                jid=jid,
                title=title,
                text=text,
                chash=chash,
                fp=fingerprint,
                rkey=rkey,
                ph=payload_hash,
                ts=ts,
            ).consume()
            row = tx.run(
                """
                MATCH (n:Note {workspace_id: $ws, note_id: $nid})
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                RETURN n, r, j
                """,
                ws=ws,
                nid=nid,
                rid=rid,
                jid=jid,
            ).data()[0]
            return Submission(_note(row["n"]), _rev(row["r"]), _job(row["j"]), replayed=False)

        return self._write(fn)

    def get_note(self, note_id: str) -> Note | None:
        rows = self._rows(
            "MATCH (n:Note {workspace_id: $ws, note_id: $nid}) RETURN n",
            ws=self.workspace_id,
            nid=note_id,
        )
        return _note(rows[0]["n"]) if rows else None

    def get_revision(self, revision_id: str) -> Revision | None:
        rows = self._rows(
            "MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid}) RETURN r",
            ws=self.workspace_id,
            rid=revision_id,
        )
        return _rev(rows[0]["r"]) if rows else None

    def list_notes(self, after: tuple[datetime, str] | None, limit: int) -> tuple[list[Note], bool]:
        ac = to_iso(after[0]) if after else None
        aid = after[1] if after else None
        rows = self._rows(
            """
            MATCH (n:Note {workspace_id: $ws})
            WHERE n.deleted_at IS NULL
              AND ($ac IS NULL OR n.created_at > $ac OR (n.created_at = $ac AND n.note_id > $aid))
            RETURN n ORDER BY n.created_at, n.note_id LIMIT $lim
            """,
            ws=self.workspace_id,
            ac=ac,
            aid=aid,
            lim=limit + 1,
        )
        notes = [_note(r["n"]) for r in rows]
        return notes[:limit], len(notes) > limit

    def delete_note(self, note_id: str, now: datetime) -> None:
        ws, ts = self.workspace_id, to_iso(now)

        def fn(tx: ManagedTransaction) -> None:
            locked = tx.run(
                "MATCH (n:Note {workspace_id: $ws, note_id: $nid}) SET n.lock_touch = $ts RETURN n",
                ws=ws,
                nid=note_id,
                ts=ts,
            ).data()
            if not locked or _note(locked[0]["n"]).deleted_at is not None:
                raise NotFound("note not found")
            tx.run(
                """
                MATCH (n:Note {workspace_id: $ws, note_id: $nid})
                SET n.deleted_at = $ts, n.active_revision_id = null
                WITH n
                OPTIONAL MATCH (n)-[act:ACTIVE_REVISION]->()
                DELETE act
                WITH DISTINCT n
                MATCH (n)<-[:FOR_NOTE]-(j:IngestionJob)
                WHERE j.status IN ['QUEUED', 'RUNNING']
                SET j.status = 'CANCELLED', j.lease_token = null, j.lease_expires_at = null
                WITH DISTINCT n
                MATCH (n)-[:HAS_REVISION]->(r:NoteRevision)
                WHERE NOT r.status IN ['CANCELLED', 'SUPERSEDED']
                SET r.status = 'CANCELLED'
                """,
                ws=ws,
                nid=note_id,
                ts=ts,
            ).consume()
            tx.run(
                "MATCH (w:Workspace {workspace_id: $ws}) SET w.graph_version = w.graph_version + 1",
                ws=ws,
            ).consume()

        self._write(fn)

    # ---- jobs ---------------------------------------------------------------------------------
    def get_job(self, job_id: str) -> Job | None:
        rows = self._rows(
            "MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid}) RETURN j",
            ws=self.workspace_id,
            jid=job_id,
        )
        return _job(rows[0]["j"]) if rows else None

    def job_for_revision(self, revision_id: str) -> Job | None:
        rows = self._rows(
            "MATCH (j:IngestionJob {workspace_id: $ws, revision_id: $rid}) RETURN j LIMIT 1",
            ws=self.workspace_id,
            rid=revision_id,
        )
        return _job(rows[0]["j"]) if rows else None

    def retry_job(self, job_id: str, now: datetime) -> Job:
        ws, ts = self.workspace_id, to_iso(now)

        def fn(tx: ManagedTransaction) -> Job:
            rows = tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})-[:FOR_NOTE]->(n:Note)
                SET n.lock_touch = $ts
                WITH j, n
                MATCH (j)-[:BUILDS]->(r:NoteRevision)
                SET j.lock_touch = $ts
                RETURN j, n, r
                """,
                ws=ws,
                jid=job_id,
                ts=ts,
            ).data()
            if not rows:
                raise NotFound("job not found")
            job, note, rev = _job(rows[0]["j"]), _note(rows[0]["n"]), _rev(rows[0]["r"])
            if note.deleted_at is not None:
                raise NotFound("job not found")
            if job.status != JobStatus.FAILED:
                raise Conflict("only failed jobs can be retried", code="JOB_NOT_RETRYABLE")
            if note.desired_revision_id != rev.revision_id:
                raise Conflict("job was superseded by a newer revision", code="JOB_SUPERSEDED")
            check_job(job.status, JobStatus.QUEUED)
            check_revision(rev.status, RevisionStatus.PENDING)
            tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})
                SET j.status = 'QUEUED', j.attempt = 0, j.error_code = null,
                    j.available_at = $ts, j.lease_token = null, j.lease_expires_at = null,
                    r.status = 'PENDING'
                """,
                ws=ws,
                jid=job_id,
                rid=rev.revision_id,
                ts=ts,
            ).consume()
            return _job(
                tx.run(
                    "MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid}) RETURN j",
                    ws=ws,
                    jid=job_id,
                ).data()[0]["j"]
            )

        return self._write(fn)

    def acquire(self, now: datetime, lease_seconds: int) -> Job | None:
        ws, ts = self.workspace_id, to_iso(now)
        exp = to_iso(now + timedelta(seconds=lease_seconds))

        def fn(tx: ManagedTransaction) -> Job | None:
            candidates = tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws})
                WHERE (j.status = 'QUEUED' AND j.available_at <= $ts)
                   OR (j.status = 'RUNNING' AND j.lease_expires_at <= $ts)
                RETURN j.job_id AS jid ORDER BY j.available_at LIMIT 5
                """,
                ws=ws,
                ts=ts,
            ).data()
            for c in candidates:
                row = tx.run(
                    """
                    MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                    SET j.lock_touch = $ts
                    WITH j
                    WHERE (j.status = 'QUEUED' AND j.available_at <= $ts)
                       OR (j.status = 'RUNNING' AND j.lease_expires_at <= $ts)
                    SET j.status = 'RUNNING', j.attempt = j.attempt + 1,
                        j.lease_token = $token, j.lease_expires_at = $exp
                    WITH j
                    MATCH (r:NoteRevision {workspace_id: $ws, revision_id: j.revision_id})
                    FOREACH (x IN CASE WHEN r.status = 'PENDING' THEN [1] ELSE [] END |
                        SET r.status = 'PROCESSING')
                    RETURN j
                    """,
                    ws=ws,
                    jid=c["jid"],
                    ts=ts,
                    token=str(uuid.uuid4()),
                    exp=exp,
                ).data()
                if row:
                    return _job(row[0]["j"])
            return None

        return self._write(fn)

    def record_progress(
        self, job_id: str, token: str, now: datetime, stage: Stage, chunks_total: int
    ) -> None:
        def fn(tx: ManagedTransaction) -> None:
            rows = tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                SET j.lock_touch = $ts
                WITH j
                WHERE j.status = 'RUNNING' AND j.lease_token = $token AND j.lease_expires_at > $ts
                SET j.stage = $stage, j.chunks_total = $total
                RETURN j.job_id AS jid
                """,
                ws=self.workspace_id,
                jid=job_id,
                ts=to_iso(now),
                token=token,
                stage=stage.value,
                total=chunks_total,
            ).data()
            if not rows:
                raise StaleLease("lease lost")

        self._write(fn)

    def requeue(
        self, job_id: str, token: str, now: datetime, available_at: datetime, code: str
    ) -> bool:
        rows = self._exec_write(
            """
            MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
            SET j.lock_touch = $ts
            WITH j WHERE j.status = 'RUNNING' AND j.lease_token = $token
            SET j.status = 'QUEUED', j.lease_token = null, j.lease_expires_at = null,
                j.available_at = $avail, j.error_code = $code
            RETURN j.job_id AS jid
            """,
            ws=self.workspace_id,
            jid=job_id,
            ts=to_iso(now),
            token=token,
            avail=to_iso(available_at),
            code=code,
        )
        return bool(rows)

    def fail(self, job_id: str, token: str, now: datetime, code: str) -> bool:
        rows = self._exec_write(
            """
            MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
            SET j.lock_touch = $ts
            WITH j WHERE j.status = 'RUNNING' AND j.lease_token = $token
            SET j.status = 'FAILED', j.error_code = $code, j.lease_token = null,
                j.lease_expires_at = null
            WITH j
            MATCH (r:NoteRevision {workspace_id: $ws, revision_id: j.revision_id})
            FOREACH (x IN CASE WHEN r.status = 'PROCESSING' THEN [1] ELSE [] END |
                SET r.status = 'FAILED')
            RETURN j.job_id AS jid
            """,
            ws=self.workspace_id,
            jid=job_id,
            ts=to_iso(now),
            token=token,
            code=code,
        )
        return bool(rows)

    # ---- staging and publication --------------------------------------------------------------
    def stage_chunk(self, job_id: str, token: str, now: datetime, staged: StagedChunk) -> None:
        ch = staged.chunk

        def fn(tx: ManagedTransaction) -> None:
            rows = tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                SET j.lock_touch = $ts
                WITH j WHERE j.status = 'RUNNING' AND j.lease_token = $token
                              AND j.lease_expires_at > $ts
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: j.revision_id})
                MERGE (c:Chunk {workspace_id: $ws, chunk_id: $cid})
                SET c.revision_id = r.revision_id, c.ordinal = $ordinal, c.text = $text,
                    c.start_char = $cstart, c.end_char = $cend, c.token_count = $tokens,
                    c.embedding = $embedding, c.processing_status = 'STAGED',
                    c.staged_claims = $claims
                MERGE (r)-[:HAS_CHUNK]->(c)
                WITH j, r
                MATCH (r)-[:HAS_CHUNK]->(ch:Chunk)
                WITH j, count(ch) AS n
                SET j.chunks_done = n
                RETURN n
                """,
                ws=self.workspace_id,
                jid=job_id,
                ts=to_iso(now),
                token=token,
                cid=ch.chunk_id,
                ordinal=ch.ordinal,
                text=ch.text,
                cstart=ch.start_char,
                cend=ch.end_char,
                tokens=ch.token_count,
                embedding=[float(x) for x in staged.embedding],
                claims=dump_claims(staged.claims),
            ).data()
            if not rows:
                raise StaleLease("lease lost")

        self._write(fn)

    def staged_chunk_ids(self, revision_id: str) -> set[str]:
        rows = self._rows(
            """
            MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.chunk_id AS cid
            """,
            ws=self.workspace_id,
            rid=revision_id,
        )
        return {r["cid"] for r in rows}

    def publish(
        self,
        job_id: str,
        token: str,
        now: datetime,
        expected_chunk_ids: list[str],
        embedding_profile: str,
        staged_profile: str,
    ) -> bool:
        ws, ts = self.workspace_id, to_iso(now)

        def fn(tx: ManagedTransaction) -> bool:
            # 1. Workspace lock (serialises publications; guards graph_version).
            tx.run(
                "MATCH (w:Workspace {workspace_id: $ws}) SET w.lock_touch = $ts",
                ws=ws,
                ts=ts,
            ).consume()
            # 2. Note then job locks, then the fence check.
            rows = tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})-[:FOR_NOTE]->(n:Note)
                SET n.lock_touch = $ts
                WITH j, n
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: j.revision_id})
                SET j.lock_touch = $ts
                RETURN j, n, r
                """,
                ws=ws,
                jid=job_id,
                ts=ts,
            ).data()
            if not rows:
                raise StaleLease("job not found")
            job, note, rev = _job(rows[0]["j"]), _note(rows[0]["n"]), _rev(rows[0]["r"])
            if not (
                job.status == JobStatus.RUNNING
                and job.lease_token == token
                and job.lease_expires_at is not None
                and job.lease_expires_at > now
            ):
                raise StaleLease("lease lost")
            if note.deleted_at is not None:
                tx.run(
                    "MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid}) "
                    "SET j.status = 'CANCELLED', j.lease_token = null",
                    ws=ws,
                    jid=job_id,
                ).consume()
                return False
            if note.desired_revision_id != rev.revision_id:
                tx.run(
                    """
                    MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                    SET j.status = 'SUPERSEDED', j.lease_token = null
                    WITH j
                    MATCH (r:NoteRevision {workspace_id: $ws, revision_id: j.revision_id})
                    WHERE r.status IN ['PENDING', 'PROCESSING']
                    SET r.status = 'SUPERSEDED'
                    """,
                    ws=ws,
                    jid=job_id,
                ).consume()
                return False

            staged_rows = tx.run(
                """
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})-[:HAS_CHUNK]->
                      (c:Chunk)
                RETURN c.chunk_id AS cid, c.ordinal AS ordinal, c.staged_claims AS claims
                ORDER BY c.ordinal
                """,
                ws=ws,
                rid=rev.revision_id,
            ).data()
            if {r["cid"] for r in staged_rows} != set(expected_chunk_ids):
                raise ValidationFailed("incomplete staging", code="INCOMPLETE_STAGING")
            if embedding_profile != staged_profile:
                raise ValidationFailed("embedding profile mismatch", code="PROFILE_MISMATCH")

            # 3. Entity resolution against the workspace's entities (in-transaction snapshot).
            known: dict[str, KnownEntity] = {}
            for e in tx.run("MATCH (e:Entity {workspace_id: $ws}) RETURN e", ws=ws).data():
                d = dict(e["e"])
                known[d["entity_id"]] = KnownEntity(
                    d["entity_id"],
                    EntityKind(d["kind"]),
                    d["canonical_name"],
                    set(d.get("aliases") or []),
                    set(d.get("context") or []),
                )
            new_rows: list[dict[str, Any]] = []
            claim_rows: list[dict[str, Any]] = []
            touched: set[str] = set()
            for sr in staged_rows:
                for vc in load_claims(sr["claims"]):
                    ids: list[str] = []
                    for ent in (vc.subject, vc.object):
                        ctx = {vc.subject.name, vc.object.name} - {ent.name}
                        found = resolve(ent.name, ent.kind, ctx, list(known.values()))
                        if found is None:
                            found = new_entity_id()
                            known[found] = KnownEntity(found, ent.kind, ent.name)
                            new_rows.append(
                                {
                                    "entity_id": found,
                                    "kind": ent.kind.value,
                                    "canonical_name": ent.name,
                                    "normalized_name": normalize_name(ent.name),
                                }
                            )
                        known[found].context |= ctx
                        touched.add(found)
                        ids.append(found)
                    claim_rows.append(
                        {
                            "claim_id": vc.claim_id,
                            "chunk_id": sr["cid"],
                            "subject_id": ids[0],
                            "object_id": ids[1],
                            "revision_id": rev.revision_id,
                            "predicate": vc.predicate,
                            "polarity": vc.polarity.value,
                            "modality": vc.modality.value,
                            "evidence_start": vc.evidence_start,
                            "evidence_end": vc.evidence_end,
                            "evidence_quote": vc.evidence_quote,
                            "temporal_text": vc.temporal_text,
                        }
                    )
            if new_rows:
                tx.run(
                    """
                    UNWIND $rows AS row
                    CREATE (e:Entity {workspace_id: $ws, entity_id: row.entity_id,
                                      kind: row.kind, canonical_name: row.canonical_name,
                                      normalized_name: row.normalized_name, aliases: [],
                                      context: [], created_at: $ts})
                    """,
                    rows=new_rows,
                    ws=ws,
                    ts=ts,
                ).consume()
            ctx_rows = [{"entity_id": e, "context": sorted(known[e].context)} for e in touched]
            if ctx_rows:
                tx.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:Entity {workspace_id: $ws, entity_id: row.entity_id})
                    SET e.context = row.context
                    """,
                    rows=ctx_rows,
                    ws=ws,
                ).consume()
            if claim_rows:
                tx.run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {workspace_id: $ws, chunk_id: row.chunk_id})
                    MATCH (s:Entity {workspace_id: $ws, entity_id: row.subject_id})
                    MATCH (o:Entity {workspace_id: $ws, entity_id: row.object_id})
                    MERGE (cl:Claim {workspace_id: $ws, claim_id: row.claim_id})
                    SET cl.revision_id = row.revision_id, cl.predicate = row.predicate,
                        cl.polarity = row.polarity, cl.modality = row.modality,
                        cl.evidence_start = row.evidence_start,
                        cl.evidence_end = row.evidence_end,
                        cl.evidence_quote = row.evidence_quote,
                        cl.temporal_text = row.temporal_text,
                        cl.validation_status = 'VALIDATED'
                    MERGE (cl)-[:EVIDENCED_BY]->(c)
                    MERGE (cl)-[:SUBJECT]->(s)
                    MERGE (cl)-[:OBJECT]->(o)
                    """,
                    rows=claim_rows,
                    ws=ws,
                ).consume()

            # 4. Atomic activation: the previous active revision is superseded in this same tx.
            tx.run(
                """
                MATCH (n:Note {workspace_id: $ws, note_id: $nid})
                OPTIONAL MATCH (n)-[old:ACTIVE_REVISION]->(prev:NoteRevision)
                WHERE prev.revision_id <> $rid
                FOREACH (p IN CASE WHEN prev IS NULL THEN [] ELSE [prev] END |
                    SET p.status = 'SUPERSEDED')
                DELETE old
                WITH n
                MATCH (r:NoteRevision {workspace_id: $ws, revision_id: $rid})
                SET r.status = 'READY', n.active_revision_id = $rid, n.updated_at = $ts
                MERGE (n)-[:ACTIVE_REVISION]->(r)
                """,
                ws=ws,
                nid=note.note_id,
                rid=rev.revision_id,
                ts=ts,
            ).consume()
            tx.run(
                """
                MATCH (j:IngestionJob {workspace_id: $ws, job_id: $jid})
                SET j.status = 'SUCCEEDED', j.stage = 'PUBLISHING', j.lease_token = null,
                    j.lease_expires_at = null, j.chunks_done = $n
                WITH j
                MATCH (w:Workspace {workspace_id: $ws})
                SET w.graph_version = w.graph_version + 1
                """,
                ws=ws,
                jid=job_id,
                n=len(expected_chunk_ids),
            ).consume()
            return True

        return self._write(fn)

    # ---- workspace state ----------------------------------------------------------------------
    def heartbeat(self, now: datetime) -> None:
        self._exec_write(
            "MATCH (w:Workspace {workspace_id: $ws}) SET w.worker_heartbeat = $ts",
            ws=self.workspace_id,
            ts=to_iso(now),
        )

    def last_heartbeat(self) -> datetime | None:
        rows = self._rows(
            "MATCH (w:Workspace {workspace_id: $ws}) RETURN w.worker_heartbeat AS hb",
            ws=self.workspace_id,
        )
        return opt_dt(rows[0]["hb"]) if rows else None

    def graph_version(self) -> int:
        rows = self._rows(
            "MATCH (w:Workspace {workspace_id: $ws}) RETURN w.graph_version AS v",
            ws=self.workspace_id,
        )
        return int(rows[0]["v"]) if rows else 0

    # ---- capability probes ---------------------------------------------------------------------
    def record_probe(
        self, fingerprint: str, passed: bool, results: list[dict[str, Any]], now: datetime
    ) -> None:
        self._exec_write(
            """
            CREATE (p:CapabilityProbe {workspace_id: $ws, fingerprint: $fp, passed: $passed,
                                       recorded_at: $ts, results: $results})
            """,
            ws=self.workspace_id,
            fp=fingerprint,
            passed=passed,
            ts=to_iso(now),
            results=json.dumps(results, ensure_ascii=False, sort_keys=True),
        )

    def latest_probe(self) -> ProbeRecord | None:
        rows = self._rows(
            """
            MATCH (p:CapabilityProbe {workspace_id: $ws})
            RETURN p ORDER BY p.recorded_at DESC LIMIT 1
            """,
            ws=self.workspace_id,
        )
        if not rows:
            return None
        p = dict(rows[0]["p"])
        results = json.loads(p["results"])
        return ProbeRecord(
            fingerprint=p["fingerprint"],
            passed=bool(p["passed"]),
            recorded_at=req_dt(p["recorded_at"]),
            results=tuple({str(k): str(v) for k, v in r.items()} for r in results),
        )

    # ---- shared inference slot ----------------------------------------------------------------
    def ensure_gate(self, endpoint_id: str) -> None:
        self._exec_write(
            "MERGE (g:InferenceGate {endpoint_id: $ep}) "
            "ON CREATE SET g.state = 'IDLE', g.owner_id = null, g.lease_token = null, "
            "g.lease_expires_at = null",
            ep=endpoint_id,
        )

    def gate_status(self, endpoint_id: str) -> GateStatus:
        rows = self._rows(
            "MATCH (g:InferenceGate {endpoint_id: $ep}) "
            "RETURN g.state AS state, g.owner_id AS owner, g.lease_expires_at AS exp",
            ep=endpoint_id,
        )
        if not rows:
            return GateStatus(GATE_IDLE, None, None)
        r = rows[0]
        return GateStatus(r["state"], r["owner"], opt_dt(r["exp"]))

    def gate_try_acquire(
        self, endpoint_id: str, owner: str, token: str, now: datetime, lease_seconds: int
    ) -> bool:
        exp = to_iso(now + timedelta(seconds=lease_seconds))

        def fn(tx: ManagedTransaction) -> bool:
            row = tx.run(
                "MATCH (g:InferenceGate {endpoint_id: $ep}) SET g.lock_touch = $ts "
                "RETURN g.state AS state, g.lease_expires_at AS exp",
                ep=endpoint_id,
                ts=to_iso(now),
            ).single()
            if row is None:
                raise NotFound("inference gate missing; call ensure_gate first")
            state, expires = row["state"], opt_dt(row["exp"])
            if state == GATE_BUSY and expires is not None and expires <= now:
                tx.run(
                    "MATCH (g:InferenceGate {endpoint_id: $ep}) SET g.state = 'RECOVERING', "
                    "g.owner_id = null, g.lease_token = null, g.lease_expires_at = null",
                    ep=endpoint_id,
                ).consume()
                return False
            allowed = state == GATE_RECOVERING if owner == PROBE_OWNER else state == GATE_IDLE
            if not allowed:
                return False
            tx.run(
                "MATCH (g:InferenceGate {endpoint_id: $ep}) SET g.state = 'BUSY', "
                "g.owner_id = $owner, g.lease_token = $token, g.lease_expires_at = $exp",
                ep=endpoint_id,
                owner=owner,
                token=token,
                exp=exp,
            ).consume()
            return True

        return self._write(fn)

    def gate_heartbeat(
        self, endpoint_id: str, token: str, now: datetime, lease_seconds: int
    ) -> None:
        self._exec_write(
            "MATCH (g:InferenceGate {endpoint_id: $ep}) "
            "WHERE g.lease_token = $token AND g.state = 'BUSY' "
            "SET g.lease_expires_at = $exp",
            ep=endpoint_id,
            token=token,
            exp=to_iso(now + timedelta(seconds=lease_seconds)),
        )

    def gate_release(self, endpoint_id: str, token: str, target: str) -> None:
        if target not in (GATE_IDLE, GATE_RECOVERING):
            raise ValueError("invalid gate target")
        self._exec_write(
            "MATCH (g:InferenceGate {endpoint_id: $ep}) "
            "WHERE g.lease_token = $token AND g.state = 'BUSY' "
            "SET g.state = $target, g.owner_id = null, g.lease_token = null, "
            "g.lease_expires_at = null",
            ep=endpoint_id,
            token=token,
            target=target,
        )

    # ---- retention -----------------------------------------------------------------------------
    def purge(
        self,
        now: datetime,
        content_retention_days: float,
        idempotency_retention_days: float,
    ) -> dict[str, int]:
        ws = self.workspace_id
        content_cutoff = to_iso(now - timedelta(days=content_retention_days))
        idem_cutoff = to_iso(now - timedelta(days=idempotency_retention_days))

        def fn(tx: ManagedTransaction) -> dict[str, int]:
            # Content of deleted notes (revisions, chunks, claims, jobs); the tombstone stays.
            rows = tx.run(
                """
                MATCH (n:Note {workspace_id: $ws})
                WHERE n.deleted_at IS NOT NULL AND n.deleted_at <= $cutoff
                OPTIONAL MATCH (n)-[:HAS_REVISION]->(r:NoteRevision)
                OPTIONAL MATCH (r)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (cl:Claim {workspace_id: $ws})-[:EVIDENCED_BY]->(c)
                WITH n, collect(DISTINCT r) AS revs, collect(DISTINCT c) AS chunks,
                     collect(DISTINCT cl) AS claims
                FOREACH (x IN claims | DETACH DELETE x)
                FOREACH (x IN chunks | DETACH DELETE x)
                FOREACH (x IN revs | DETACH DELETE x)
                WITH n, size(revs) AS nrev, size(claims) AS ncl
                OPTIONAL MATCH (j:IngestionJob {workspace_id: $ws, note_id: n.note_id})
                DETACH DELETE j
                WITH n, nrev, ncl, count(j) AS njobs
                WHERE nrev > 0 OR ncl > 0 OR njobs > 0
                SET n.title = '', n.desired_revision_id = '', n.active_revision_id = null
                RETURN sum(nrev) AS revisions, sum(ncl) AS claims
                """,
                ws=ws,
                cutoff=content_cutoff,
            ).data()
            counts = {
                "revisions": int(rows[0]["revisions"] or 0) if rows else 0,
                "claims": int(rows[0]["claims"] or 0) if rows else 0,
            }
            orphans = tx.run(
                """
                MATCH (e:Entity {workspace_id: $ws})
                WHERE NOT (e)<-[:SUBJECT|OBJECT]-(:Claim)
                DETACH DELETE e
                RETURN count(e) AS c
                """,
                ws=ws,
            ).single()
            counts["entities"] = int(orphans["c"]) if orphans else 0
            idem = tx.run(
                """
                MATCH (i:IdempotencyRecord {workspace_id: $ws})
                WHERE i.created_at < $cutoff
                DETACH DELETE i
                RETURN count(i) AS c
                """,
                ws=ws,
                cutoff=idem_cutoff,
            ).single()
            counts["idempotency"] = int(idem["c"]) if idem else 0
            return counts

        return self._write(fn)

    # ---- operations ---------------------------------------------------------------------------
    def schema_version(self) -> int:
        rows = self._rows("MATCH (m:SchemaMigration) RETURN coalesce(max(m.version), 0) AS v")
        return int(rows[0]["v"])

    def health(self) -> dict[str, Any]:
        try:
            rows = self._rows("SHOW INDEXES YIELD name, state RETURN name, state")
            states = {r["name"]: r["state"] for r in rows}
            return {
                "reachable": True,
                "schema_version": self.schema_version(),
                "indexes_online": all(states.get(n) == "ONLINE" for n in REQUIRED_INDEXES),
                "required_indexes": list(REQUIRED_INDEXES),
                "persistence": "neo4j",
                "expected_schema_version": latest_version(),
            }
        except TRANSIENT_ERRORS:
            return {
                "reachable": False,
                "schema_version": 0,
                "indexes_online": False,
                "required_indexes": list(REQUIRED_INDEXES),
                "persistence": "neo4j",
                "expected_schema_version": latest_version(),
            }
