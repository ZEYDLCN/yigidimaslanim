"""KnowledgeSource over MemoryStore. Every read is restricted to active READY revisions."""

from __future__ import annotations

import math
import re

from notes_graphrag.domain.state_machines import RevisionStatus
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.persistence.models import Revision, StagedChunk
from notes_graphrag.retrieval.lexical import query_terms, tokens
from notes_graphrag.retrieval.source import ChunkRecord, ClaimRecord, EntityRecord

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(tokens(text))


def _cos(a: list[float] | tuple[float, ...], b: tuple[float, ...]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0 or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


class MemoryKnowledge:
    def __init__(self, store: MemoryStore) -> None:
        self.s = store

    def graph_version(self) -> int:
        return self.s.graph_version()

    def _active(self) -> dict[str, tuple[Revision, StagedChunk]]:
        """chunk_id -> (active revision, staged chunk), for non-deleted notes only."""
        out: dict[str, tuple[Revision, StagedChunk]] = {}
        with self.s.lock:
            for note in self.s.notes.values():
                if note.deleted_at is not None or not note.active_revision_id:
                    continue
                rev = self.s.revisions[note.active_revision_id]
                if rev.status != RevisionStatus.READY:
                    continue
                for cid, sc in self.s.staged.get(rev.revision_id, {}).items():
                    out[cid] = (rev, sc)
        return out

    def _chunk_record(self, cid: str, rev: Revision, sc: StagedChunk) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=cid,
            note_id=rev.note_id,
            revision_id=rev.revision_id,
            title=rev.title,
            start_char=sc.chunk.start_char,
            end_char=sc.chunk.end_char,
            text=sc.chunk.text,
            revision_text=rev.text,
        )

    def vector_candidates(self, vec: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(cid, _cos(vec, sc.embedding)) for cid, (_, sc) in self._active().items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def lexical_candidates(self, question: str, k: int) -> list[str]:
        terms = set(query_terms(question))
        if not terms:
            return []
        hits = []
        for cid, (_, sc) in self._active().items():
            n = len(terms & _tokens(sc.chunk.text))
            if n:
                hits.append((n, cid))
        hits.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in hits[:k]]

    def chunks_by_ids(self, ids: list[str]) -> dict[str, ChunkRecord]:
        active = self._active()
        return {cid: self._chunk_record(cid, *active[cid]) for cid in ids if cid in active}

    def chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self.chunks_by_ids([chunk_id]).get(chunk_id)

    def _claims(self) -> list[ClaimRecord]:
        active = self._active()
        out = []
        with self.s.lock:
            for mc in self.s.claims.values():
                if mc.chunk_id not in active:
                    continue
                rev, _ = active[mc.chunk_id]
                if mc.revision_id != rev.revision_id:
                    continue
                c = mc.claim
                out.append(
                    ClaimRecord(
                        claim_id=c.claim_id,
                        subject_id=mc.subject_id,
                        subject=self.s.entities[mc.subject_id].canonical_name,
                        predicate=c.predicate,
                        object_id=mc.object_id,
                        object=self.s.entities[mc.object_id].canonical_name,
                        polarity=c.polarity.value,
                        modality=c.modality.value,
                        chunk_id=mc.chunk_id,
                        evidence_quote=c.evidence_quote,
                        evidence_start=c.evidence_start,
                        evidence_end=c.evidence_end,
                        temporal_text=c.temporal_text,
                        note_id=rev.note_id,
                        revision_id=rev.revision_id,
                        title=rev.title,
                    )
                )
        return sorted(out, key=lambda c: c.claim_id)

    def entities(self) -> list[EntityRecord]:
        ids = {c.subject_id for c in self._claims()} | {c.object_id for c in self._claims()}
        with self.s.lock:
            rows = [
                EntityRecord(
                    i, self.s.entities[i].canonical_name, tuple(sorted(self.s.entities[i].aliases))
                )
                for i in ids
            ]
        return sorted(rows, key=lambda e: (e.name, e.entity_id))

    def claims_for_entity(self, entity_id: str) -> list[ClaimRecord]:
        return [c for c in self._claims() if entity_id in (c.subject_id, c.object_id)]

    def claims_for_chunk(self, chunk_id: str) -> list[ClaimRecord]:
        return [c for c in self._claims() if c.chunk_id == chunk_id]

    def claims_for_note(self, note_id: str) -> list[ClaimRecord]:
        return [c for c in self._claims() if c.note_id == note_id]

    def note_active_chunks(self, note_id: str) -> int:
        return sum(1 for rev, _ in self._active().values() if rev.note_id == note_id)
