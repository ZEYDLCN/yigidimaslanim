"""Versioned Cypher migrations with checksums and a vector-dimension guard (architecture 18.2).

A migration that was already applied must not be edited: its checksum is verified on every run.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j import Driver

DIM_PLACEHOLDER = "{{EMBEDDING_DIMENSIONS}}"
VECTOR_INDEX = "chunk_embedding_v1"
_NAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.cypher$")


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    template: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()


def default_dir() -> Path:
    env = os.environ.get("MIGRATIONS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "migrations"


def load_migrations(directory: Path | None = None) -> list[Migration]:
    d = directory or default_dir()
    out: list[Migration] = []
    for p in sorted(d.glob("*.cypher")):
        m = _NAME.match(p.name)
        if m is None:
            raise MigrationError(f"bad migration file name: {p.name}")
        out.append(Migration(int(m.group(1)), m.group(2), p.read_text("utf-8")))
    if [m.version for m in out] != list(range(1, len(out) + 1)):
        raise MigrationError("migration versions must be contiguous, starting at 001")
    return out


def latest_version(directory: Path | None = None) -> int:
    return max((m.version for m in load_migrations(directory)), default=0)


def statements(template: str, dims: int) -> list[str]:
    body = template.replace(DIM_PLACEHOLDER, str(dims))
    lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _check_vector_dimensions(session: Any, dims: int) -> None:
    rows = session.run(
        "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
        name=VECTOR_INDEX,
    ).data()
    if not rows:
        return
    cfg = rows[0]["options"].get("indexConfig", {})
    existing = cfg.get("vector.dimensions")
    if existing is not None and int(existing) != dims:
        raise MigrationError(
            f"existing {VECTOR_INDEX} has {existing} dimensions but settings require {dims}; "
            "a model/profile change needs a full reindex, not a silent index reuse"
        )


def migrate(driver: Driver, database: str, dims: int, directory: Path | None = None) -> list[int]:
    migs = load_migrations(directory)
    newly: list[int] = []
    with driver.session(database=database) as s:
        s.run(
            "CREATE CONSTRAINT schema_migration_version IF NOT EXISTS "
            "FOR (m:SchemaMigration) REQUIRE m.version IS UNIQUE"
        ).consume()
        applied = {
            r["version"]: r["checksum"]
            for r in s.run(
                "MATCH (m:SchemaMigration) RETURN m.version AS version, m.checksum AS checksum"
            ).data()
        }
        for m in migs:
            if m.version in applied and applied[m.version] != m.checksum:
                raise MigrationError(f"migration {m.version:03d} was edited after being applied")
        _check_vector_dimensions(s, dims)
        for m in migs:
            if m.version in applied:
                continue
            for stmt in statements(m.template, dims):
                s.run(stmt).consume()
            s.run(
                "CREATE (:SchemaMigration {version: $v, name: $n, checksum: $c, applied_at: $ts})",
                v=m.version,
                n=m.name,
                c=m.checksum,
                ts=datetime.now(UTC).isoformat(),
            ).consume()
            newly.append(m.version)
    return newly
