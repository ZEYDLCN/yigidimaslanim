"""Behavioural contract shared by every Store implementation (architecture section 15.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from notes_graphrag.answering.service import AnswerService
from notes_graphrag.core.errors import (
    Conflict,
    IdempotencyKeyReused,
    NoteDeleted,
    NotFound,
    ProviderUnavailable,
    StaleLease,
)
from notes_graphrag.domain.state_machines import JobStatus, RevisionStatus
from notes_graphrag.persistence.models import GATE_BUSY, GATE_IDLE, GATE_RECOVERING, PROBE_OWNER
from notes_graphrag.retrieval.service import Retriever

NOW = datetime(2026, 1, 1, tzinfo=UTC)
TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."


def drain(k) -> None:
    while k.worker.run_once():
        pass


def ask(k, q: str):
    qa = AnswerService(k.knowledge, Retriever(k.knowledge, k.emb, k.settings), k.chat, k.settings)
    return qa.ask(q, include_graph=True)


def test_lifecycle_create_publish_answer_replace_delete(kit):
    k = kit
    sub = k.notes.create("Atlas", TEXT, "k1")
    assert k.store.get_note(sub.note.note_id).active_revision_id is None
    assert k.knowledge.vector_candidates(k.emb.embed(["neo4j"])[0], 5) == []  # not searchable yet
    drain(k)
    job = k.store.get_job(sub.job.job_id)
    assert job.status == JobStatus.SUCCEEDED and job.chunks_done == job.chunks_total == 1
    note = k.store.get_note(sub.note.note_id)
    assert note.active_revision_id == sub.revision.revision_id
    assert k.store.get_revision(note.active_revision_id).status == RevisionStatus.READY

    claims = {(c.subject, c.predicate, c.object) for c in k.knowledge.claims_for_note(note.note_id)}
    assert claims == {("Ada", "WORKS_ON", "Project Atlas"), ("Project Atlas", "USES", "Neo4j")}
    for c in k.knowledge.claims_for_note(note.note_id):
        rev = k.store.get_revision(c.revision_id)
        assert rev.text[c.evidence_start : c.evidence_end] == c.evidence_quote

    r = ask(k, "Which database does Ada's project use?")
    assert r.status == "answered" and "Neo4j" in r.answer
    assert {c.note_id for c in r.citations} == {note.note_id}
    assert {n["label"] for n in r.graph["nodes"]} >= {"Ada", "Project Atlas", "Neo4j"}

    rep = k.notes.replace(note.note_id, "Atlas", TEXT.replace("Neo4j", "Postgres"), "k2")
    assert (
        "Neo4j" in ask(k, "What does Project Atlas use?").answer
    )  # old revision serves until publication
    drain(k)
    after = ask(k, "What does Project Atlas use?")
    assert "Postgres" in after.answer and "Neo4j" not in after.answer
    assert k.store.get_job(rep.job.job_id).status == JobStatus.SUCCEEDED

    k.notes.delete(note.note_id)
    assert k.store.get_note(note.note_id).deleted_at is not None
    assert ask(k, "What does Project Atlas use?").status == "insufficient_evidence"
    assert k.knowledge.vector_candidates(k.emb.embed(["postgres"])[0], 5) == []
    with pytest.raises(NotFound):  # deleted notes look missing to new requests
        k.notes.replace(note.note_id, "t", "x", "k3")
    with pytest.raises(NotFound):
        k.notes.delete(note.note_id)


def test_idempotent_replay_and_key_conflict(kit):
    k = kit
    a = k.notes.create("T", TEXT, "same")
    b = k.notes.create("T", TEXT, "same")
    assert b.replayed and b.job.job_id == a.job.job_id and b.note.note_id == a.note.note_id
    with pytest.raises(IdempotencyKeyReused):
        k.notes.create("T", TEXT + " more", "same")
    k.notes.delete(a.note.note_id)
    with pytest.raises(NoteDeleted):  # a delayed retry cannot resurrect a deleted note
        k.notes.create("T", TEXT, "same")


def test_same_content_replace_reuses_revision(kit):
    k = kit
    first = k.notes.create("T", TEXT, "a")
    drain(k)
    again = k.notes.replace(first.note.note_id, "T", TEXT, "b")
    assert again.revision.revision_id == first.revision.revision_id
    assert again.job.job_id == first.job.job_id
    assert k.chat.calls == 1  # no duplicate provider calls


def test_reindex_reuses_matching_revision(kit):
    k = kit
    s = k.notes.create("T", TEXT, "a")
    drain(k)
    again = k.notes.reindex(s.note.note_id, "ri")
    assert again.revision.revision_id == s.revision.revision_id


def test_newer_replacement_supersedes_queued_older(kit):
    k = kit
    s1 = k.notes.create("T", TEXT, "a")
    s2 = k.notes.replace(s1.note.note_id, "T", TEXT + " Ada works on Project Zed.", "b")
    assert k.store.get_job(s1.job.job_id).status == JobStatus.SUPERSEDED
    assert k.store.get_revision(s1.revision.revision_id).status == RevisionStatus.SUPERSEDED
    drain(k)
    assert k.store.get_note(s1.note.note_id).active_revision_id == s2.revision.revision_id


def test_failed_update_keeps_previous_revision(kit):
    k = kit
    s1 = k.notes.create("T", TEXT, "a")
    drain(k)

    class Broken:
        def chat_json(self, **kw):
            return "{}garbage"

    k.worker.chat = Broken()
    s2 = k.notes.replace(s1.note.note_id, "T", "Something else entirely.", "b")
    drain(k)
    assert k.store.get_job(s2.job.job_id).status == JobStatus.FAILED
    assert k.store.get_revision(s2.revision.revision_id).status == RevisionStatus.FAILED
    assert k.store.get_note(s1.note.note_id).active_revision_id == s1.revision.revision_id
    assert k.knowledge.claims_for_note(s1.note.note_id)


def test_retry_failed_job_and_reject_superseded(kit):
    k = kit

    class Broken:
        def chat_json(self, **kw):
            return "{}garbage"

    good = k.worker.chat
    k.worker.chat = Broken()
    s1 = k.notes.create("T", TEXT, "a")
    drain(k)
    assert k.store.get_job(s1.job.job_id).error_code == "EXTRACTION_INVALID"
    k.worker.chat = good
    k.notes.retry(s1.job.job_id)
    drain(k)
    assert k.store.get_job(s1.job.job_id).status == JobStatus.SUCCEEDED
    with pytest.raises(Conflict):
        k.notes.retry(s1.job.job_id)  # succeeded jobs are not retryable


def test_expired_lease_is_reclaimed_and_cannot_write(kit):
    k = kit
    s = k.notes.create("T", TEXT, "a")
    j1 = k.store.acquire(NOW, k.settings.job_lease_seconds)
    assert j1 is not None and j1.job_id == s.job.job_id
    later = NOW + timedelta(seconds=k.settings.job_lease_seconds + 1)
    j2 = k.store.acquire(later, k.settings.job_lease_seconds)
    assert j2 is not None and j2.job_id == j1.job_id
    assert j2.attempt == j1.attempt + 1 and j2.lease_token != j1.lease_token
    with pytest.raises(StaleLease):
        k.store.record_progress(j1.job_id, j1.lease_token, later, _stage(), 1)
    assert k.store.fail(j1.job_id, j1.lease_token, later, "X") is False
    with pytest.raises(StaleLease):
        k.store.publish(j1.job_id, j1.lease_token, later, [], "fake-v1", "fake-v1")


def test_transient_failure_requeues_then_fails(kit):
    k = kit

    class Down:
        def chat_json(self, **kw):
            raise ProviderUnavailable("down", code="LM_STUDIO_UNAVAILABLE")

    k.worker.chat = Down()
    k.worker.rng.seed(1)
    s = k.notes.create("T", TEXT, "a")
    t = NOW
    for _ in range(k.settings.job_max_attempts + 2):
        k.worker.clock = lambda t=t: t
        k.worker.run_once()
        t += timedelta(minutes=5)
    job = k.store.get_job(s.job.job_id)
    assert job.status == JobStatus.FAILED and job.attempt == k.settings.job_max_attempts
    assert k.knowledge.vector_candidates(k.emb.embed(["x"])[0], 5) == []


def test_delete_during_processing_blocks_publication(kit):
    k = kit
    s = k.notes.create("T", TEXT, "a")
    j = k.store.acquire(NOW, k.settings.job_lease_seconds)
    k.notes.delete(s.note.note_id)
    # A cancelled job no longer holds a valid lease, so its fenced publish is refused.
    with pytest.raises(StaleLease):
        k.store.publish(j.job_id, j.lease_token, NOW, [], "fake-v1", "fake-v1")
    assert k.store.get_job(j.job_id).status == JobStatus.CANCELLED
    assert k.knowledge.note_active_chunks(s.note.note_id) == 0


def test_list_notes_pagination_is_stable(kit):
    k = kit
    ids = []
    for i in range(5):
        ids.append(k.notes.create(f"T{i}", f"Note number {i}.", f"k{i}").note.note_id)
    seen, cursor = [], None
    while True:
        rows, more = k.store.list_notes(cursor, 2)
        seen += [n.note_id for n in rows]
        if not more:
            break
        cursor = (rows[-1].created_at, rows[-1].note_id)
    assert len(seen) == len(set(seen)) == len(ids) and set(seen) == set(ids)


def test_two_hop_expansion_across_notes(kit):
    k = kit
    k.notes.create("A", "Ada works on Project Atlas.", "a")
    k.notes.create("B", "Project Atlas uses Neo4j.", "b")
    drain(k)
    ada = next(e for e in k.knowledge.entities() if e.name == "Ada")
    claims = {c.predicate for c in k.knowledge.claims_for_entity(ada.entity_id)}
    assert claims == {"WORKS_ON"}
    from notes_graphrag.retrieval.graph import expand

    hops = {(x.claim.predicate, x.hop) for x in expand(k.knowledge, [ada.entity_id])}
    assert hops == {("WORKS_ON", 1), ("USES", 2)}


def test_lexical_and_vector_candidates_are_scoped_to_active_revisions(kit):
    k = kit
    s = k.notes.create("T", TEXT, "a")
    drain(k)
    lex = k.knowledge.lexical_candidates("Neo4j database", 10)
    assert lex and all(k.knowledge.chunk(c).note_id == s.note.note_id for c in lex)
    vec = k.knowledge.vector_candidates(k.emb.embed(["Neo4j"])[0], 10)
    assert vec and vec[0][0] == lex[0]
    k.notes.delete(s.note.note_id)
    assert k.knowledge.lexical_candidates("Neo4j database", 10) == []


def test_heartbeat_and_graph_version_round_trip(kit):
    k = kit
    assert k.store.last_heartbeat() is None
    k.store.heartbeat(NOW)
    assert k.store.last_heartbeat() == NOW
    v = k.store.graph_version()
    k.notes.create("T", TEXT, "a")
    drain(k)
    assert k.store.graph_version() == v + 1


def _stage():
    from notes_graphrag.domain.state_machines import Stage

    return Stage.EXTRACTING


# ---- shared inference slot --------------------------------------------------------------------
def test_gate_is_exclusive_and_expired_holder_enters_recovery(kit):
    k = kit
    ep = "endpoint-test"
    k.store.ensure_gate(ep)
    assert k.store.gate_try_acquire(ep, "a", "t1", NOW, 10) is True
    assert k.store.gate_try_acquire(ep, "b", "t2", NOW, 10) is False
    assert k.store.gate_status(ep).state == GATE_BUSY
    later = NOW + timedelta(seconds=11)
    assert k.store.gate_try_acquire(ep, "b", "t3", later, 10) is False  # crash: now RECOVERING
    assert k.store.gate_status(ep).state == GATE_RECOVERING
    assert k.store.gate_try_acquire(ep, "b", "t4", later, 10) is False  # normal owners blocked
    assert k.store.gate_try_acquire(ep, PROBE_OWNER, "p1", later, 10) is True
    k.store.gate_release(ep, "p1", GATE_IDLE)
    assert k.store.gate_status(ep).state == GATE_IDLE
    assert k.store.gate_try_acquire(ep, "b", "t5", later, 10) is True


def test_gate_release_with_wrong_token_is_ignored(kit):
    k = kit
    ep = "endpoint-release"
    k.store.ensure_gate(ep)
    assert k.store.gate_try_acquire(ep, "a", "t1", NOW, 60)
    k.store.gate_release(ep, "not-mine", GATE_IDLE)
    assert k.store.gate_status(ep).state == GATE_BUSY
    k.store.gate_release(ep, "t1", GATE_RECOVERING)
    assert k.store.gate_status(ep).state == GATE_RECOVERING


def test_gate_heartbeat_extends_lease(kit):
    k = kit
    ep = "endpoint-beat"
    k.store.ensure_gate(ep)
    k.store.gate_try_acquire(ep, "a", "t1", NOW, 10)
    k.store.gate_heartbeat(ep, "t1", NOW + timedelta(seconds=9), 10)
    assert k.store.gate_try_acquire(ep, "b", "t2", NOW + timedelta(seconds=15), 10) is False
    assert k.store.gate_status(ep).state == GATE_BUSY


def test_gated_slot_marks_recovery_on_ambiguous_timeout(kit):
    from notes_graphrag.providers.gate import InferenceGate

    k = kit
    gate = InferenceGate(
        k.store,
        "ep-amb",
        lease_seconds=60,
        wait_seconds=0.05,
        probe=lambda: None,
        poll_seconds=0.01,
    )
    with pytest.raises(ProviderUnavailable):
        with gate.slot("api"):
            raise ProviderUnavailable("t", code="PROVIDER_TIMEOUT")
    assert k.store.gate_status("ep-amb").state == GATE_RECOVERING
    with gate.slot("api"):  # the probe succeeds, so the slot is admitted again
        pass
    assert k.store.gate_status("ep-amb").state == GATE_IDLE


# ---- retention (architecture 7.2 and 7.3) ----------------------------------------------------
def test_purge_removes_deleted_content_but_keeps_replay_protection(kit):
    k = kit
    s = k.notes.create("T", TEXT, "purge-key")
    drain(k)
    k.notes.delete(s.note.note_id)
    later = NOW + timedelta(seconds=1)
    counts = k.store.purge(later, 0, 30)
    assert counts["revisions"] >= 1 and counts["claims"] >= 1
    assert k.store.get_revision(s.revision.revision_id) is None  # source text is gone
    assert k.knowledge.claims_for_note(s.note.note_id) == []
    assert k.knowledge.note_active_chunks(s.note.note_id) == 0
    assert k.store.get_note(s.note.note_id).deleted_at is not None  # tombstone remains
    with pytest.raises(NoteDeleted):  # replay answers NOTE_DELETED; it must not create a note
        k.notes.create("T", TEXT, "purge-key")
    assert k.store.list_notes(None, 100)[0] == []


def test_purge_keeps_live_notes_and_removes_orphan_entities(kit):
    k = kit
    gone = k.notes.create("Gone", "Mert works on Project Zed.", "gone")
    keep = k.notes.create("Keep", TEXT, "keep")
    drain(k)
    assert any(e.name == "Mert" for e in k.knowledge.entities())
    k.notes.delete(gone.note.note_id)
    counts = k.store.purge(NOW + timedelta(seconds=1), 0, 30)
    assert counts["entities"] >= 1
    names = {e.name for e in k.knowledge.entities()}
    assert "Mert" not in names and {"Ada", "Neo4j"} <= names
    assert k.knowledge.claims_for_note(keep.note.note_id)
    assert ask(k, "Which database does Project Atlas use?").status == "answered"


def test_idempotency_records_expire_after_their_retention(kit):
    k = kit
    k.notes.create("A", TEXT, "expiring")
    k.store.purge(NOW + timedelta(days=31), 1, 30)  # the record is older than 30 days
    again = k.notes.create("A", TEXT, "expiring")  # the key is now a new operation
    assert again.replayed is False


def test_purged_note_cannot_be_recreated_through_its_id(kit):
    k = kit
    s = k.notes.create("T", TEXT, "gone-id")
    k.notes.delete(s.note.note_id)
    k.store.purge(NOW + timedelta(seconds=1), 0, 30)
    with pytest.raises(NotFound):
        k.notes.replace(s.note.note_id, "T", "new text", "after-purge")


# ---- capability probe records --------------------------------------------------------------
def test_probe_records_round_trip(kit):
    k = kit
    assert k.store.latest_probe() is None
    k.store.record_probe("fp-1", False, [{"name": "chat", "ok": False, "code": "X"}], NOW)
    later = NOW + timedelta(minutes=5)
    k.store.record_probe("fp-2", True, [{"name": "chat", "ok": True, "code": "OK"}], later)
    latest = k.store.latest_probe()
    assert latest.fingerprint == "fp-2" and latest.passed is True
    assert latest.recorded_at == later
    assert latest.results[0]["name"] == "chat"
