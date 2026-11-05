"""Neo4j-specific behaviour: migrations, concurrency, Turkish full text, processes, and the API.

Run with NEO4J_TEST_URI, NEO4J_TEST_PASSWORD and NEO4J_TEST_WIPE=1 (see tests/support).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neo4j import GraphDatabase
from tests.support.neo4j_env import target_or_skip, wipe_and_migrate

from notes_graphrag.answering.service import AnswerService
from notes_graphrag.api.app import create_app
from notes_graphrag.api.dependencies import build_context
from notes_graphrag.core.config import Settings
from notes_graphrag.domain.state_machines import JobStatus
from notes_graphrag.ingestion.service import NoteService
from notes_graphrag.persistence.migrations import MigrationError, load_migrations, migrate
from notes_graphrag.persistence.neo4j_source import Neo4jKnowledge
from notes_graphrag.persistence.neo4j_store import Neo4jStore
from notes_graphrag.providers.fakes import FakeEmbeddings
from notes_graphrag.retrieval.service import Retriever

DIMS = 64  # matches the demo profile used by the CLI worker and API
PROFILE = "fake-v1"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def db():
    target = target_or_skip()
    driver = GraphDatabase.driver(target.uri, auth=(target.user, target.password))
    wipe_and_migrate(driver, "neo4j", DIMS)
    settings = Settings(
        _env_file=None,
        neo4j_uri=target.uri,
        neo4j_password=target.password,
        neo4j_username=target.user,
        embedding_dimensions=DIMS,
        embedding_profile=PROFILE,
    )
    yield driver, settings
    # The subprocess worker was stopped by the test; clear its heartbeat for the next wipe.
    with driver.session(database="neo4j") as session:
        session.run("MATCH (w:Workspace) REMOVE w.worker_heartbeat").consume()
    driver.close()


def _store(driver, settings):
    return Neo4jStore(driver, "neo4j", settings.workspace_id)


def test_migrations_are_idempotent_and_versioned(db):
    driver, settings = db
    assert migrate(driver, "neo4j", DIMS) == []  # already up to date
    store = _store(driver, settings)
    assert store.schema_version() == len(load_migrations())


def test_edited_applied_migration_is_rejected(db, tmp_path):
    driver, _ = db
    for f in load_migrations():
        (tmp_path / f"{f.version:03d}_{f.name}.cypher").write_text(f.template + "\n// edited\n")
    with pytest.raises(MigrationError, match="edited after being applied"):
        migrate(driver, "neo4j", DIMS, tmp_path)


def test_vector_dimension_mismatch_is_refused_not_reused(db):
    driver, _ = db
    with pytest.raises(MigrationError, match="needs a full reindex"):
        migrate(driver, "neo4j", DIMS * 2)


def test_health_reports_online_indexes_and_schema(db):
    driver, settings = db
    h = _store(driver, settings).health()
    assert h["reachable"] is True
    assert h["indexes_online"] is True
    assert h["schema_version"] == h["expected_schema_version"]


def test_concurrent_acquire_gives_one_owner_per_job(db):
    driver, settings = db
    store = _store(driver, settings)
    emb = FakeEmbeddings(DIMS, PROFILE)
    del emb
    NoteService(store, settings, lambda: NOW).create("T", TEXT, "race-acquire")
    results: list[object] = []
    barrier = threading.Barrier(8)

    def grab() -> None:
        barrier.wait()
        results.append(store.acquire(NOW, settings.job_lease_seconds))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly one owner, got {len(winners)}"


def test_concurrent_identical_submissions_create_exactly_one_job(db):
    driver, settings = db
    store = _store(driver, settings)
    svc = NoteService(store, settings, lambda: NOW)
    ids: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def submit() -> None:
        barrier.wait()
        try:
            ids.append(svc.create("T", TEXT, "race-submit").job.job_id)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(ids)) == 1
    with driver.session(database="neo4j") as s:
        assert s.run("MATCH (j:IngestionJob) RETURN count(j) AS c").single()["c"] == 1


def test_turkish_dotted_i_matches_case_insensitively(db):
    driver, settings = db
    store = _store(driver, settings)
    kn = Neo4jKnowledge(driver, "neo4j", settings.workspace_id)
    svc = NoteService(store, settings, lambda: NOW)
    from notes_graphrag.ingestion.extraction import ExtractionOutput
    from notes_graphrag.ingestion.worker import Worker
    from notes_graphrag.persistence.memory import MemoryStore  # noqa: F401 - import check only
    from notes_graphrag.providers.fakes import DemoChat
    from notes_graphrag.providers.schema import inline_refs

    svc.create(
        "Şehir", "İstanbul ofisinde Ada çalışıyor. Ada Project Atlas üzerinde çalışır.", "tr"
    )
    schema = inline_refs(ExtractionOutput.model_json_schema())
    w = Worker(store, settings, DemoChat(), FakeEmbeddings(DIMS, PROFILE), schema, lambda: NOW)
    while w.run_once():
        pass
    assert kn.lexical_candidates("istanbul", 5), "lowercase query must match İstanbul"
    assert kn.lexical_candidates("İSTANBUL", 5), "uppercase Turkish query must match"


def test_worker_in_separate_process_processes_api_submission(db):
    driver, settings = db
    store = _store(driver, settings)
    svc = NoteService(store, settings, lambda: datetime.now(UTC))
    sub = svc.create("Cross", TEXT, "multiproc")
    env = {
        **os.environ,
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_USERNAME": settings.neo4j_username,
        "NEO4J_PASSWORD": settings.neo4j_password.get_secret_value(),
        "NEO4J_DATABASE": "neo4j",
        "WORKER_POLL_SECONDS": "0.1",
        "PYTHONPATH": str(REPO / "src"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "notes_graphrag.cli", "worker", "--demo"],
        env=env,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 90
        status = None
        while time.monotonic() < deadline:
            status = store.get_job(sub.job.job_id).status
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                break
            time.sleep(0.3)
        assert status == JobStatus.SUCCEEDED, proc.poll()
        assert store.last_heartbeat() is not None  # the separate process heartbeats
        assert store.graph_version() >= 1
        assert Neo4jKnowledge(driver, "neo4j", settings.workspace_id).claims_for_note(
            sub.note.note_id
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_api_on_neo4j_end_to_end_with_stopped_worker_recovery(db):
    driver, settings = db
    ctx = build_context(settings, demo=True, backend="neo4j", embedded_worker=False)
    client = TestClient(create_app(ctx))
    assert client.get("/health/ready").status_code == 200
    r = client.post(
        "/api/v1/notes",
        json={"title": "Atlas", "text": TEXT},
        headers={"Idempotency-Key": "api-neo4j"},
    )
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "QUEUED"
    # Simulate a crashed worker: it acquires the job and never finishes or releases it.
    crashed = ctx.store.acquire(datetime.now(UTC), 1)
    assert crashed is not None and crashed.job_id == job_id
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "RUNNING"
    time.sleep(1.2)  # lease (1s) expires
    while ctx.worker.run_once():
        pass
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "SUCCEEDED"
    q = client.post("/api/v1/questions", json={"question": "What does Project Atlas use?"}).json()
    assert q["status"] == "answered" and "Neo4j" in q["answer"]
    assert AnswerService and Retriever  # imported services exercised through the API path
    client.delete(f"/api/v1/notes/{body['note_id']}")
    gone = client.post("/api/v1/questions", json={"question": "What does Project Atlas use?"})
    assert gone.json()["status"] == "insufficient_evidence"
