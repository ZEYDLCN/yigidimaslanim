"""Evidence packing: dedupe/merge overlapping passages and respect the token budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.chunking import estimate_tokens
from notes_graphrag.retrieval.source import ChunkRecord, ClaimRecord

PROMPT_OVERHEAD_TOKENS = 700  # system instructions + JSON schema


@dataclass
class EvidenceItem:
    evidence_id: str
    note_id: str
    revision_id: str
    chunk_id: str
    chunk_ids: list[str]
    title: str
    text: str
    start_char: int
    end_char: int
    claim_ids: list[str] = field(default_factory=list)


@dataclass
class EvidencePack:
    items: list[EvidenceItem] = field(default_factory=list)
    claims: list[tuple[ClaimRecord, list[str]]] = field(default_factory=list)
    limited_by_budget: bool = False

    def item(self, evidence_id: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.evidence_id == evidence_id), None)


def evidence_budget(settings: Settings, question: str) -> int:
    """Reserve instructions, schema, question and output before filling evidence."""
    avail = (
        settings.lm_studio_context_tokens
        - settings.answer_max_output_tokens
        - estimate_tokens(question)
        - PROMPT_OVERHEAD_TOKENS
    )
    return max(0, min(settings.max_context_tokens, avail))


@dataclass
class _Group:
    revision_id: str
    note_id: str
    title: str
    revision_text: str
    start: int
    end: int
    chunk_ids: list[str]

    def tokens(self) -> int:
        return estimate_tokens(self.revision_text[self.start : self.end])


def pack_evidence(
    ranked: list[ChunkRecord],
    claims: list[ClaimRecord],
    *,
    budget_tokens: int,
    max_items: int,
) -> EvidencePack:
    groups: list[_Group] = []
    limited = False
    for ch in ranked:  # `ranked` is already ordered by priority
        host = next(
            (
                g
                for g in groups
                if g.revision_id == ch.revision_id
                and ch.start_char < g.end
                and g.start < ch.end_char
            ),
            None,
        )
        used = sum(g.tokens() for g in groups)
        if host is not None:
            if ch.chunk_id in host.chunk_ids:
                continue
            merged = _Group(**{**host.__dict__})
            merged.start = min(host.start, ch.start_char)
            merged.end = max(host.end, ch.end_char)
            merged.chunk_ids = [*host.chunk_ids, ch.chunk_id]
            if used - host.tokens() + merged.tokens() > budget_tokens:
                limited = True
                continue
            groups[groups.index(host)] = merged
            continue
        if len(groups) >= max_items:
            limited = True
            continue
        new = _Group(
            ch.revision_id,
            ch.note_id,
            ch.title,
            ch.revision_text,
            ch.start_char,
            ch.end_char,
            [ch.chunk_id],
        )
        if used + new.tokens() > budget_tokens:
            limited = True
            continue
        groups.append(new)

    pack = EvidencePack(limited_by_budget=limited)
    for n, g in enumerate(groups, 1):
        pack.items.append(
            EvidenceItem(
                f"E{n}",
                g.note_id,
                g.revision_id,
                g.chunk_ids[0],
                g.chunk_ids,
                g.title,
                g.revision_text[g.start : g.end],
                g.start,
                g.end,
            )
        )
    for claim in claims:
        eids = [i.evidence_id for i in pack.items if claim.chunk_id in i.chunk_ids]
        if eids:
            pack.claims.append((claim, eids))
            for i in pack.items:
                if i.evidence_id in eids:
                    i.claim_ids.append(claim.claim_id)
    return pack
