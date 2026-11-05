from datetime import UTC, datetime, timedelta

import pytest

from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import (
    IdempotencyKeyReused,
    NoteDeleted,
    ProviderUnavailable,
    StaleLease,
)
from notes_graphrag.domain.state_machines import JobStatus, RevisionStatus
from notes_graphrag.ingestion.extraction import ExtractionOutput
from notes_graphrag.ingestion.service import NoteService
from notes_graphrag.ingestion.worker import Worker
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.providers.fakes import FakeChat, FakeEmbeddings, cosine
from notes_graphrag.retrieval.fusion import rrf

TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."


class Clock:
    def __init__(self):
        self.t = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += timedelta(seconds=s)


@pytest.fixture
def env():
    clock, store = Clock(), MemoryStore()
    s = Settings(_env_file=None, embedding_dimensions=16, embedding_profile="fake-v1")
    emb, chat = FakeEmbeddings(16), FakeChat()
    schema = ExtractionOutput.model_json_schema()
    return (
        clock,
        store,
        s,
        NoteService(store, s, clock),
        Worker(store, s, chat, emb, schema, clock),
        chat,
    )


def search(store, emb, q):
    qv = emb.embed([q])[0]
    items = store.active_chunks()
    vec = [c.chunk.chunk_id for _, _, c in sorted(items, key=lambda x: -cosine(qv, x[2].embedding))]
    words = set(q.lower().split())
    lex = [c.chunk.chunk_id for _, _, c in items if words & set(c.chunk.text.lower().split())]
    return rrf([lex, vec])


def test_full_lifecycle(env):
    clock, store, s, svc, worker, chat = env
    note, rev, job = svc.create("Atlas", TEXT, "k1")
    assert store.active_chunks() == []  # nothing searchable before publication
    worker.run_once()
    assert job.status == JobStatus.SUCCEEDED and rev.status == RevisionStatus.READY
    claims = store.active_claims()
    assert {(c.claim.subject.name, c.claim.predicate, c.claim.object.name) for c in claims} == {
        ("Ada", "WORKS_ON", "Project Atlas"),
        ("Project Atlas", "USES", "Neo4j"),
    }
    for c in claims:
        assert TEXT[c.claim.evidence_start : c.claim.evidence_end] == c.claim.evidence_quote
    # shared Project Atlas entity reused across both claims
    assert len(store.entity_by_name("Project Atlas")) == 1
    assert search(store, FakeEmbeddings(16), "which database neo4j")

    # replace: old evidence stays until the new revision is published
    _, rev2, job2 = svc.replace(note.note_id, "Atlas", TEXT.replace("Neo4j", "Postgres"), "k2")
    assert any("Neo4j" in c.claim.object.name for c in store.active_claims())
    worker.run_once()
    names = {c.claim.object.name for c in store.active_claims()}
    assert "Neo4j" not in names and "Postgres" in names
    assert rev.status == RevisionStatus.SUPERSEDED and rev2.status == RevisionStatus.READY

    svc.delete(note.note_id)
    assert store.active_chunks() == [] and store.active_claims() == []
    with pytest.raises(NoteDeleted):
        svc.replace(note.note_id, "t", "x", "k3")


def test_idempotency(env):
    clock, store, s, svc, worker, chat = env
    a = svc.create("T", TEXT, "same")
    b = svc.create("T", TEXT, "same")
    assert a[2].job_id == b[2].job_id and len(store.jobs) == 1
    with pytest.raises(IdempotencyKeyReused):
        svc.create("T", TEXT + " more", "same")
    # delayed retry of a create after deletion cannot resurrect the note
    svc.delete(a[0].note_id)
    with pytest.raises(NoteDeleted):
        svc.create("T", TEXT, "same")


def test_same_content_replace_reuses_revision(env):
    _, store, _, svc, worker, chat = env
    note, rev, job = svc.create("T", TEXT, "a")
    worker.run_once()
    (
        _,
        rev2,
        job2,
    ) = svc.replace(note.note_id, "T", TEXT, "b")
    assert rev2.revision_id == rev.revision_id and job2.job_id == job.job_id
    assert chat.calls == 1


def test_newer_replacement_supersedes_queued_older(env):
    _, store, _, svc, worker, chat = env
    note, rev1, job1 = svc.create("T", TEXT, "a")
    _, rev2, job2 = svc.replace(note.note_id, "T", TEXT + " Ada works on Project Zed.", "b")
    assert job1.status == JobStatus.SUPERSEDED and rev1.status == RevisionStatus.SUPERSEDED
    worker.run_once()
    assert note.active_revision_id == rev2.revision_id


def test_failed_update_keeps_previous_revision(env):
    _, store, s, svc, worker, chat = env
    note, rev1, _ = svc.create("T", TEXT, "a")
    worker.run_once()

    class Broken:
        calls = 0

        def chat_json(self, **kw):
            return "{}garbage"

    bad = Worker(store, s, Broken(), FakeEmbeddings(16), {}, env[0])
    _, rev2, job2 = svc.replace(note.note_id, "T", "Something else entirely.", "b")
    bad.run_once()
    assert job2.status == JobStatus.FAILED and rev2.status == RevisionStatus.FAILED
    assert note.active_revision_id == rev1.revision_id and store.active_claims()


def test_expired_lease_cannot_write_or_publish(env):
    clock, store, s, svc, worker, chat = env
    _, rev, job = svc.create("T", TEXT, "a")
    j = store.acquire(clock(), s.job_lease_seconds)
    old_token = j.lease_token
    clock.advance(s.job_lease_seconds + 1)
    with pytest.raises(StaleLease):
        store.stage_chunk(j.job_id, old_token, clock(), None)  # type: ignore[arg-type]
    j2 = store.acquire(clock(), s.job_lease_seconds)  # reclaimed
    assert j2 is j and j2.lease_token != old_token and j2.attempt == 2
    with pytest.raises(StaleLease):
        store.publish(j.job_id, old_token, clock(), [], s.embedding_profile, "fake-v1")


def test_transient_provider_failure_retries_with_backoff_then_fails(env):
    clock, store, s, svc, _, _ = env

    class Down:
        def chat_json(self, **kw):
            raise ProviderUnavailable("down")

    w = Worker(store, s, Down(), FakeEmbeddings(16), {}, clock)
    _, rev, job = svc.create("T", TEXT, "a")
    w.run_once()
    assert job.status == JobStatus.QUEUED and w.run_once() is None  # backoff not elapsed
    for _ in range(2):
        clock.advance(120)
        w.run_once()
    assert job.status == JobStatus.FAILED and job.attempt == s.job_max_attempts
    assert store.active_chunks() == []


def test_delete_during_processing_blocks_publication(env):
    clock, store, s, svc, worker, chat = env
    note, rev, job = svc.create("T", TEXT, "a")
    j = store.acquire(clock(), s.job_lease_seconds)
    svc.delete(note.note_id)
    with pytest.raises(StaleLease):
        store.publish(j.job_id, j.lease_token, clock(), [], s.embedding_profile, "fake-v1")
    assert store.active_chunks() == []


def test_state_machine_rejects_illegal():
    from notes_graphrag.core.errors import Conflict
    from notes_graphrag.domain.state_machines import check_job

    with pytest.raises(Conflict):
        check_job(JobStatus.SUCCEEDED, JobStatus.RUNNING)
