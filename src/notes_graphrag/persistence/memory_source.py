"""KnowledgeSource over MemoryStore. Reads go through the active-revision filter."""

from __future__ import annotations

from notes_graphrag.persistence.memory import MemoryStore, StoredClaim
from notes_graphrag.retrieval.source import ChunkRecord, ClaimRecord, EntityRecord


def _claim_record(store: MemoryStore, sc: StoredClaim) -> ClaimRecord:
    c = sc.claim
    return ClaimRecord(
        claim_id=c.claim_id,
        subject_id=sc.subject_id,
        subject=store.entities[sc.subject_id].canonical_name,
        predicate=c.predicate,
        object_id=sc.object_id,
        object=store.entities[sc.object_id].canonical_name,
        polarity=c.polarity.value,
        modality=c.modality.value,
        chunk_id=sc.chunk_id,
        evidence_quote=c.evidence_quote,
        evidence_start=c.evidence_start,
        evidence_end=c.evidence_end,
        temporal_text=c.temporal_text,
    )


class MemoryKnowledge:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def graph_version(self) -> int:
        return self.store.graph_version

    def chunks(self) -> list[ChunkRecord]:
        with self.store.lock:
            return [
                ChunkRecord(
                    sc.chunk.chunk_id,
                    note.note_id,
                    rev.revision_id,
                    rev.title,
                    sc.chunk.start_char,
                    sc.chunk.end_char,
                    sc.chunk.text,
                    tuple(sc.embedding),
                    rev.text,
                )
                for note, rev, sc in self.store.active_chunks()
            ]

    def chunk(self, chunk_id: str) -> ChunkRecord | None:
        return next((c for c in self.chunks() if c.chunk_id == chunk_id), None)

    def _claims(self) -> list[ClaimRecord]:
        with self.store.lock:
            return [_claim_record(self.store, sc) for sc in self.store.active_claims()]

    def entities(self) -> list[EntityRecord]:
        with self.store.lock:
            ids = {c.subject_id for c in self._claims()} | {c.object_id for c in self._claims()}
            return [
                EntityRecord(
                    i,
                    self.store.entities[i].canonical_name,
                    tuple(sorted(self.store.entities[i].aliases)),
                )
                for i in sorted(ids)
            ]

    def claims_for_entity(self, entity_id: str) -> list[ClaimRecord]:
        return sorted(
            (c for c in self._claims() if entity_id in (c.subject_id, c.object_id)),
            key=lambda c: c.claim_id,
        )

    def claims_for_chunk(self, chunk_id: str) -> list[ClaimRecord]:
        return sorted(
            (c for c in self._claims() if c.chunk_id == chunk_id), key=lambda c: c.claim_id
        )
