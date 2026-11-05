"""Neo4j test environment helpers.

Tests that touch a real database need NEO4J_TEST_URI, NEO4J_TEST_PASSWORD and NEO4J_TEST_WIPE=1.
The wipe clears the target database, so it is never implied: the caller must opt in explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class Neo4jTarget:
    uri: str
    user: str
    password: str


def target_or_skip() -> Neo4jTarget:
    uri = os.environ.get("NEO4J_TEST_URI")
    if not uri:
        pytest.skip("NEO4J_TEST_URI not set; Neo4j tests not run")
    if os.environ.get("NEO4J_TEST_WIPE") != "1":
        pytest.skip("set NEO4J_TEST_WIPE=1 to allow clearing the test database")
    return Neo4jTarget(
        uri=uri,
        user=os.environ.get("NEO4J_TEST_USER", "neo4j"),
        password=os.environ["NEO4J_TEST_PASSWORD"],
    )


def _active_worker_heartbeat(session: Any) -> bool:
    row = session.run(
        "MATCH (w:Workspace) WHERE w.worker_heartbeat IS NOT NULL "
        "RETURN max(w.worker_heartbeat) AS hb"
    ).single()
    if row is None or row["hb"] is None:
        return False
    from datetime import UTC, datetime

    hb = datetime.fromisoformat(row["hb"])
    return (datetime.now(UTC) - hb).total_seconds() < 120


def wipe_and_migrate(driver: Any, database: str, dims: int, directory: Any = None) -> list[int]:
    """Clear the database and apply migrations. Refuses if a worker is actively using it."""
    from notes_graphrag.persistence.migrations import migrate

    with driver.session(database=database) as s:
        if _active_worker_heartbeat(s):
            raise RuntimeError(
                "a worker heartbeat is fresh on this database; stop the running services "
                "before running tests that clear it"
            )
        for row in s.run("SHOW CONSTRAINTS YIELD name RETURN name").data():
            s.run(f"DROP CONSTRAINT `{row['name']}` IF EXISTS").consume()
        for row in s.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name").data():
            s.run(f"DROP INDEX `{row['name']}` IF EXISTS").consume()
        s.run("MATCH (n) DETACH DELETE n").consume()
    return migrate(driver, database, dims, directory)
