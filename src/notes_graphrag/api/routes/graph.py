from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok
from notes_graphrag.api.views import decode_cursor, encode_cursor
from notes_graphrag.core.errors import BadRequest, NotFound
from notes_graphrag.ingestion.entity_resolution import normalize_name
from notes_graphrag.retrieval.graph import Limits, expand
from notes_graphrag.retrieval.source import ClaimRecord

MAX_NODES, MAX_EDGES = 100, 150
router = APIRouter(prefix="/api/v1")


@router.get("/entities")
def entities(
    request: Request,
    q: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> JSONResponse:
    src = ctx_of(request).knowledge
    key = normalize_name(q) if q else None
    rows = [e for e in src.entities() if key is None or key in normalize_name(e.name)]
    parts = decode_cursor(cursor, 2)
    if parts:
        rows = [e for e in rows if (e.name, e.entity_id) > (parts[0], parts[1])]
    page = rows[:limit]
    nxt = encode_cursor(page[-1].name, page[-1].entity_id) if len(rows) > limit and page else None
    items = [
        {
            "entity_id": e.entity_id,
            "name": e.name,
            "aliases": list(e.aliases),
            "claim_count": len(src.claims_for_entity(e.entity_id)),
        }
        for e in page
    ]
    return ok(request, {"items": items, "next_cursor": nxt})


def _edges(
    claims: list[ClaimRecord],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], bool]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    limited = len(claims) > MAX_EDGES
    for c in claims[:MAX_EDGES]:
        new = {c.subject_id, c.object_id} - nodes.keys()
        if len(nodes) + len(new) > MAX_NODES:
            limited = True
            continue
        nodes[c.subject_id] = {"id": c.subject_id, "label": c.subject}
        nodes[c.object_id] = {"id": c.object_id, "label": c.object}
        edges.append(
            {
                "id": c.claim_id,
                "source": c.subject_id,
                "target": c.object_id,
                "predicate": c.predicate,
                "polarity": c.polarity,
                "modality": c.modality,
                "evidence": {
                    "note_id": c.note_id,
                    "revision_id": c.revision_id,
                    "title": c.title,
                    "start_char": c.evidence_start,
                    "end_char": c.evidence_end,
                    "quote": c.evidence_quote,
                },
            }
        )
    return nodes, edges, limited


@router.get("/graph")
def graph(
    request: Request,
    entity_id: str | None = None,
    note_id: str | None = None,
    hops: Annotated[int, Query(ge=1, le=2)] = 1,
) -> JSONResponse:
    if bool(entity_id) == bool(note_id):
        raise BadRequest("provide exactly one of entity_id or note_id", code="BAD_GRAPH_QUERY")
    src = ctx_of(request).knowledge
    if entity_id:
        if entity_id not in {e.entity_id for e in src.entities()}:
            raise NotFound("entity not found")
        claims = [
            x.claim
            for x in expand(src, [entity_id], Limits(max_hops=hops, total_claims=MAX_EDGES + 1))
        ]
    else:
        assert note_id is not None
        if src.note_active_chunks(note_id) == 0:
            raise NotFound("note not found or not indexed")
        claims = src.claims_for_note(note_id)
    nodes, edges, limited = _edges(claims)
    return ok(request, {"nodes": list(nodes.values()), "edges": edges, "limited": limited})
