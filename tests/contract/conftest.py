"""Store fixtures: the same contract tests run against the memory store and the Neo4j store.

The Neo4j fixture WIPES the database it is pointed at. It only runs when both
NEO4J_TEST_URI and NEO4J_TEST_WIPE=1 are set, so a developer database is never cleared by accident.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.support.neo4j_env import target_or_skip, wipe_and_migrate

from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.extraction import ExtractionOutput
from notes_graphrag.ingestion.service import NoteService
from notes_graphrag.ingestion.worker import Worker
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.persistence.memory_source import MemoryKnowledge
from notes_graphrag.providers.fakes import DemoChat, FakeEmbeddings
from notes_graphrag.providers.schema import inline_refs

TEST_DIMS = 16
NOW = datetime(2026, 1, 1, tzinfo=UTC)
TEST_PROFILE = "fake-v1"


@dataclass
class Kit:
    kind: str
    store: Any
    knowledge: Any
    settings: Settings
    notes: NoteService
    worker: Worker
    emb: FakeEmbeddings
    chat: DemoChat


def make_settings(**over: Any) -> Settings:
    base = {
        "embedding_dimensions": TEST_DIMS,
        "embedding_profile": TEST_PROFILE,
        "neo4j_password": os.environ.get("NEO4J_TEST_PASSWORD", "unused"),
    }
    base.update(over)
    return Settings(_env_file=None, **base)


def _kit(kind: str, store: Any, knowledge: Any, settings: Settings, clock: Any) -> Kit:
    emb, chat = FakeEmbeddings(TEST_DIMS, TEST_PROFILE), DemoChat()
    schema = inline_refs(ExtractionOutput.model_json_schema())
    return Kit(
        kind=kind,
        store=store,
        knowledge=knowledge,
        settings=settings,
        notes=NoteService(store, settings, clock),
        worker=Worker(store, settings, chat, emb, schema, clock),
        emb=emb,
        chat=chat,
    )


@pytest.fixture(params=["memory", "neo4j"])
def kit(request: pytest.FixtureRequest) -> Iterator[Kit]:
    clock = lambda: NOW  # noqa: E731 - fixed clock keeps lease arithmetic deterministic
    if request.param == "memory":
        settings = make_settings()
        mem = MemoryStore(workspace_id="default")
        yield _kit("memory", mem, MemoryKnowledge(mem), settings, clock)
        return

    from neo4j import GraphDatabase

    from notes_graphrag.persistence.neo4j_source import Neo4jKnowledge
    from notes_graphrag.persistence.neo4j_store import Neo4jStore

    target = target_or_skip()
    driver = GraphDatabase.driver(target.uri, auth=(target.user, target.password))
    try:
        wipe_and_migrate(driver, "neo4j", TEST_DIMS)
        settings = make_settings(neo4j_uri=target.uri, neo4j_password=target.password)
        store = Neo4jStore(driver, "neo4j", "default")
        yield _kit("neo4j", store, Neo4jKnowledge(driver, "neo4j", "default"), settings, clock)
    finally:
        with driver.session(database="neo4j") as session:  # tests own this database
            session.run("MATCH (w:Workspace) REMOVE w.worker_heartbeat").consume()
        driver.close()
