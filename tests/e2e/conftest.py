"""End-to-end fixtures: the full HTTP stack (API + worker logic) on both storage backends.

The neo4j variant needs NEO4J_TEST_URI, NEO4J_TEST_PASSWORD and NEO4J_TEST_WIPE=1
(see tests/support).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from notes_graphrag.api.app import create_app
from notes_graphrag.api.dependencies import AppContext, build_context
from notes_graphrag.core.config import Settings

DIMS = 64
PROFILE = "fake-v1"


class Stack:
    def __init__(self, ctx: AppContext, client: TestClient, backend: str) -> None:
        self.ctx = ctx
        self.client = client
        self.backend = backend

    def drain(self) -> int:
        n = 0
        while self.ctx.worker.run_once():
            n += 1
        return n

    def create(self, title: str, text: str, key: str) -> Any:
        return self.client.post(
            "/api/v1/notes",
            json={"title": title, "text": text},
            headers={"Idempotency-Key": key},
        )

    def ask(self, question: str, graph: bool = False) -> Any:
        return self.client.post(
            "/api/v1/questions", json={"question": question, "include_graph": graph}
        )


def _settings(**over: Any) -> Settings:
    return Settings(_env_file=None, embedding_dimensions=DIMS, embedding_profile=PROFILE, **over)


@pytest.fixture(params=["memory", "neo4j"])
def stack(request: pytest.FixtureRequest) -> Iterator[Stack]:
    backend = request.param
    if backend == "memory":
        ctx = build_context(_settings(), demo=True, backend="memory")
        yield Stack(ctx, TestClient(create_app(ctx), raise_server_exceptions=False), backend)
        return

    from neo4j import GraphDatabase
    from tests.support.neo4j_env import target_or_skip, wipe_and_migrate

    target = target_or_skip()
    driver = GraphDatabase.driver(target.uri, auth=(target.user, target.password))
    try:
        wipe_and_migrate(driver, "neo4j", DIMS)
        settings = _settings(
            neo4j_uri=target.uri, neo4j_password=target.password, neo4j_username=target.user
        )
        ctx = build_context(settings, demo=True, backend="neo4j")
        client = TestClient(create_app(ctx), raise_server_exceptions=False)
        yield Stack(ctx, client, backend)
    finally:
        with driver.session(database="neo4j") as session:
            session.run("MATCH (w:Workspace) REMOVE w.worker_heartbeat").consume()
        driver.close()
