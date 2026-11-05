"""KnowledgeSource over Neo4j. Every query is scoped to READY active revisions of live notes.

Index results are post-filtered, so the overfetch loop widens the index probe until enough
active hits remain or the cap is reached (architecture 8.1).
"""

from __future__ import annotations

from typing import Any

from neo4j import Driver

from notes_graphrag.retrieval.lexical import escape_lucene, query_terms
from notes_graphrag.retrieval.source import ChunkRecord, ClaimRecord, EntityRecord

ACTIVE_CHUNK = """
MATCH (n:Note {workspace_id: $ws})-[:ACTIVE_REVISION]->(r:NoteRevision {status: 'READY'})
      -[:HAS_CHUNK]->(c:Chunk)
WHERE n.deleted_at IS NULL
"""
CLAIM_MATCH = """
MATCH (n:Note {workspace_id: $ws})-[:ACTIVE_REVISION]->(r:NoteRevision {status: 'READY'})
      -[:HAS_CHUNK]->(c:Chunk)<-[:EVIDENCED_BY]-(cl:Claim)
MATCH (cl)-[:SUBJECT]->(s:Entity)
MATCH (cl)-[:OBJECT]->(o:Entity)
WHERE n.deleted_at IS NULL
"""
CHUNK_RETURN = """
RETURN c.chunk_id AS chunk_id, n.note_id AS note_id, r.revision_id AS revision_id,
       r.title AS title, c.start_char AS start_char, c.end_char AS end_char,
       c.text AS text, r.text AS revision_text
"""
CLAIM_RETURN = """
RETURN cl.claim_id AS claim_id, s.entity_id AS subject_id, s.canonical_name AS subject,
       cl.predicate AS predicate, o.entity_id AS object_id, o.canonical_name AS object,
       cl.polarity AS polarity, cl.modality AS modality, c.chunk_id AS chunk_id,
       cl.evidence_quote AS evidence_quote, cl.evidence_start AS evidence_start,
       cl.evidence_end AS evidence_end, cl.temporal_text AS temporal_text,
       n.note_id AS note_id, r.revision_id AS revision_id, r.title AS title
ORDER BY claim_id
"""
PROBE_STEPS = (1, 2, 4)  # multiples of the requested k


def _chunk(d: dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=d["chunk_id"],
        note_id=d["note_id"],
        revision_id=d["revision_id"],
        title=d["title"],
        start_char=d["start_char"],
        end_char=d["end_char"],
        text=d["text"],
        revision_text=d["revision_text"],
    )


def _claim(d: dict[str, Any]) -> ClaimRecord:
    return ClaimRecord(
        claim_id=d["claim_id"],
        subject_id=d["subject_id"],
        subject=d["subject"],
        predicate=d["predicate"],
        object_id=d["object_id"],
        object=d["object"],
        polarity=d["polarity"],
        modality=d["modality"],
        chunk_id=d["chunk_id"],
        evidence_quote=d["evidence_quote"],
        evidence_start=d["evidence_start"],
        evidence_end=d["evidence_end"],
        temporal_text=d.get("temporal_text"),
        note_id=d["note_id"],
        revision_id=d["revision_id"],
        title=d["title"],
    )


class Neo4jKnowledge:
    def __init__(self, driver: Driver, database: str, workspace_id: str) -> None:
        self._driver = driver
        self._db = database
        self.ws = workspace_id

    def _rows(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self._db) as s:
            return s.execute_read(lambda tx: tx.run(cypher, ws=self.ws, **params).data())

    def graph_version(self) -> int:
        rows = self._rows("MATCH (w:Workspace {workspace_id: $ws}) RETURN w.graph_version AS v")
        return int(rows[0]["v"]) if rows else 0

    def vector_candidates(self, vec: list[float], k: int) -> list[tuple[str, float]]:
        hits: list[tuple[str, float]] = []
        for mult in PROBE_STEPS:
            probe = k * mult
            rows = self._rows(
                """
                CALL db.index.vector.queryNodes('chunk_embedding_v1', $probe, $vec)
                YIELD node AS c, score
                MATCH (n:Note {workspace_id: $ws})-[:ACTIVE_REVISION]->
                      (r:NoteRevision {status: 'READY'})-[:HAS_CHUNK]->(c)
                WHERE n.deleted_at IS NULL
                RETURN c.chunk_id AS cid, score ORDER BY score DESC LIMIT $k
                """,
                probe=probe,
                vec=vec,
                k=k,
            )
            # Neo4j's cosine score is (1 + cos) / 2; convert back so thresholds match memory.
            hits = [(r["cid"], 2.0 * float(r["score"]) - 1.0) for r in rows]
            if len(hits) >= k:
                break
        return hits

    def lexical_candidates(self, question: str, k: int) -> list[str]:
        terms = query_terms(question)
        if not terms:
            return []
        q = " ".join(escape_lucene(t) for t in terms)
        hits: list[str] = []
        for mult in PROBE_STEPS:
            probe = k * mult
            rows = self._rows(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_v1', $q, {limit: $probe})
                YIELD node AS c, score
                MATCH (n:Note {workspace_id: $ws})-[:ACTIVE_REVISION]->
                      (r:NoteRevision {status: 'READY'})-[:HAS_CHUNK]->(c)
                WHERE n.deleted_at IS NULL
                RETURN c.chunk_id AS cid, score ORDER BY score DESC LIMIT $k
                """,
                q=q,
                probe=probe,
                k=k,
            )
            hits = [r["cid"] for r in rows]
            if len(hits) >= k:
                break
        return hits

    def chunks_by_ids(self, ids: list[str]) -> dict[str, ChunkRecord]:
        if not ids:
            return {}
        rows = self._rows(ACTIVE_CHUNK + "AND c.chunk_id IN $ids " + CHUNK_RETURN, ids=ids)
        return {r["chunk_id"]: _chunk(r) for r in rows}

    def chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self.chunks_by_ids([chunk_id]).get(chunk_id)

    def entities(self) -> list[EntityRecord]:
        rows = self._rows(
            """
            MATCH (n:Note {workspace_id: $ws})-[:ACTIVE_REVISION]->
                  (r:NoteRevision {status: 'READY'})-[:HAS_CHUNK]->(c:Chunk)
                  <-[:EVIDENCED_BY]-(cl:Claim)-[:SUBJECT|OBJECT]->(e:Entity)
            WHERE n.deleted_at IS NULL
            RETURN DISTINCT e.entity_id AS entity_id, e.canonical_name AS name,
                   coalesce(e.aliases, []) AS aliases
            ORDER BY name, entity_id
            """
        )
        return [EntityRecord(r["entity_id"], r["name"], tuple(sorted(r["aliases"]))) for r in rows]

    def claims_for_entity(self, entity_id: str) -> list[ClaimRecord]:
        rows = self._rows(
            CLAIM_MATCH + "AND (s.entity_id = $eid OR o.entity_id = $eid) " + CLAIM_RETURN,
            eid=entity_id,
        )
        return [_claim(r) for r in rows]

    def claims_for_chunk(self, chunk_id: str) -> list[ClaimRecord]:
        rows = self._rows(CLAIM_MATCH + "AND c.chunk_id = $cid " + CLAIM_RETURN, cid=chunk_id)
        return [_claim(r) for r in rows]

    def claims_for_note(self, note_id: str) -> list[ClaimRecord]:
        rows = self._rows(CLAIM_MATCH + "AND n.note_id = $nid " + CLAIM_RETURN, nid=note_id)
        return [_claim(r) for r in rows]

    def note_active_chunks(self, note_id: str) -> int:
        rows = self._rows(ACTIVE_CHUNK + "AND n.note_id = $nid RETURN count(c) AS k", nid=note_id)
        return int(rows[0]["k"]) if rows else 0
