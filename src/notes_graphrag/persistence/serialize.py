"""Conversions between Python values and Neo4j-compatible property values.

Timestamps are fixed-width UTC ISO strings so that lexicographic comparison in Cypher matches
chronological order. Nested extraction output is stored as a JSON string (architecture 2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from notes_graphrag.domain.ontology import EntityKind, Modality, Polarity
from notes_graphrag.ingestion.extraction import ExtractedEntity, ValidatedClaim


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def req_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def opt_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _entity(e: ExtractedEntity) -> dict[str, str]:
    return {"local_id": e.local_id, "name": e.name, "kind": e.kind.value}


def dump_claims(claims: Iterable[ValidatedClaim]) -> str:
    rows = [
        {
            "claim_id": c.claim_id,
            "subject": _entity(c.subject),
            "object": _entity(c.object),
            "predicate": c.predicate,
            "polarity": c.polarity.value,
            "modality": c.modality.value,
            "evidence_start": c.evidence_start,
            "evidence_end": c.evidence_end,
            "evidence_quote": c.evidence_quote,
            "temporal_text": c.temporal_text,
        }
        for c in claims
    ]
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def _load_entity(d: dict[str, Any]) -> ExtractedEntity:
    return ExtractedEntity(local_id=d["local_id"], name=d["name"], kind=EntityKind(d["kind"]))


def load_claims(raw: str | None) -> tuple[ValidatedClaim, ...]:
    if not raw:
        return ()
    out = []
    for d in json.loads(raw):
        out.append(
            ValidatedClaim(
                claim_id=d["claim_id"],
                subject=_load_entity(d["subject"]),
                object=_load_entity(d["object"]),
                predicate=d["predicate"],
                polarity=Polarity(d["polarity"]),
                modality=Modality(d["modality"]),
                evidence_start=d["evidence_start"],
                evidence_end=d["evidence_end"],
                evidence_quote=d["evidence_quote"],
                temporal_text=d["temporal_text"],
            )
        )
    return tuple(out)
