"""Read-side view of the active knowledge base.

Every method returns only data reachable through a non-deleted note's READY active revision.
Implementations (memory, Neo4j) must enforce that scoping inside the query itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    note_id: str
    revision_id: str
    title: str
    start_char: int
    end_char: int
    text: str
    revision_text: str


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    subject_id: str
    subject: str
    predicate: str
    object_id: str
    object: str
    polarity: str
    modality: str
    chunk_id: str
    evidence_quote: str
    evidence_start: int
    evidence_end: int
    temporal_text: str | None
    note_id: str
    revision_id: str
    title: str


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    name: str
    aliases: tuple[str, ...]


class KnowledgeSource(Protocol):
    def graph_version(self) -> int: ...
    def vector_candidates(self, vec: list[float], k: int) -> list[tuple[str, float]]: ...
    def lexical_candidates(self, question: str, k: int) -> list[str]: ...
    def chunks_by_ids(self, ids: list[str]) -> dict[str, ChunkRecord]: ...
    def chunk(self, chunk_id: str) -> ChunkRecord | None: ...
    def entities(self) -> list[EntityRecord]: ...
    def claims_for_entity(self, entity_id: str) -> list[ClaimRecord]: ...
    def claims_for_chunk(self, chunk_id: str) -> list[ClaimRecord]: ...
    def claims_for_note(self, note_id: str) -> list[ClaimRecord]: ...
    def note_active_chunks(self, note_id: str) -> int: ...
