"""Extraction schema and the ordered validation pipeline (architecture 6.3)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from notes_graphrag.domain.ontology import EntityKind, Modality, Polarity, predicate_allows

SCHEMA_VERSION = "1"
_NS = uuid.UUID("0b6e5a4c-7e0b-4f55-8c5e-3b4a1f2d9c21")


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Evidence(Span):
    quote: str = Field(min_length=1, max_length=2000)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=200)
    kind: EntityKind
    mentions: list[Span] = Field(default_factory=list, max_length=50)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_local_id: str
    predicate: str
    object_local_id: str
    polarity: Polarity = Polarity.POSITIVE
    modality: Modality = Modality.ASSERTED
    evidence: Evidence
    temporal_text: str | None = Field(default=None, max_length=200)


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=200)


@dataclass(frozen=True)
class ValidatedClaim:
    claim_id: str
    subject: ExtractedEntity
    object: ExtractedEntity
    predicate: str
    polarity: Polarity
    modality: Modality
    evidence_start: int  # revision-global
    evidence_end: int
    evidence_quote: str
    temporal_text: str | None


@dataclass
class ValidationReport:
    claims: list[ValidatedClaim] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def claim_id(revision_id: str, subj: str, pred: str, obj: str, start: int, end: int) -> str:
    return str(uuid.uuid5(_NS, f"{revision_id}|{subj}|{pred}|{obj}|{start}|{end}"))


def validate_extraction(
    raw_json: str,
    *,
    revision_id: str,
    revision_text: str,
    chunk_start: int,
    chunk_text: str,
) -> ValidationReport:
    """Validate in the documented order. Spans from the model are chunk-local."""
    report = ValidationReport()
    try:
        out = ExtractionOutput.model_validate_json(raw_json)
    except ValueError as exc:
        report.errors.append(f"schema: {exc}")
        return report

    ids = [e.local_id for e in out.entities]
    if len(ids) != len(set(ids)):
        report.errors.append("duplicate local entity ids")
        return report
    by_id = {e.local_id: e for e in out.entities}

    for e in out.entities:
        for m in e.mentions:
            if m.end > len(chunk_text) or m.start >= m.end:
                report.errors.append(f"entity {e.local_id}: mention span out of range")

    for c in out.claims:
        subj, obj = by_id.get(c.subject_local_id), by_id.get(c.object_local_id)
        if subj is None or obj is None:
            report.errors.append("claim references unknown entity")
            continue
        if not predicate_allows(c.predicate, subj.kind, obj.kind):
            report.errors.append(f"claim {c.predicate}: incompatible {subj.kind}->{obj.kind}")
            continue
        ev = c.evidence
        if ev.end > len(chunk_text) or ev.start >= ev.end:
            report.errors.append("claim evidence span out of range")
            continue
        if chunk_text[ev.start : ev.end] != ev.quote:
            report.errors.append("claim evidence quote does not match span")
            continue
        g0, g1 = chunk_start + ev.start, chunk_start + ev.end
        if revision_text[g0:g1] != ev.quote:
            report.errors.append("claim evidence not found at revision offset")
            continue
        # Cheap support check: both entity names must appear in the quote or
        # the chunk (names may be pronoun-resolved only through the chunk).
        if subj.name not in chunk_text or obj.name not in chunk_text:
            report.errors.append("claim entity name absent from source chunk")
            continue
        report.claims.append(
            ValidatedClaim(
                claim_id(revision_id, subj.name, c.predicate, obj.name, g0, g1),
                subj,
                obj,
                c.predicate,
                c.polarity,
                c.modality,
                g0,
                g1,
                ev.quote,
                c.temporal_text,
            )
        )
    return report
