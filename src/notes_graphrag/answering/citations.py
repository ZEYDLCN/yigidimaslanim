"""Answer schema, citation validation and server-side citation construction (section 8.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from notes_graphrag.retrieval.evidence import EvidencePack

MAX_ANSWER_CHARS = 4000
_MARKER = re.compile(r"\s*\[E\d+\]")


class AnswerSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "partial", "insufficient_evidence"]
    sentences: list[AnswerSentence] = Field(default_factory=list, max_length=20)
    missing_information: list[str] = Field(default_factory=list, max_length=10)


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    note_id: str
    revision_id: str
    chunk_id: str
    title: str
    quote: str
    start_char: int
    end_char: int


def validate_answer(raw: str, pack: EvidencePack) -> tuple[AnswerOutput | None, list[str]]:
    try:
        out = AnswerOutput.model_validate_json(raw)
    except ValueError as exc:
        return None, [f"schema: {exc}"]
    errors: list[str] = []
    known = {i.evidence_id for i in pack.items}
    if out.status == "insufficient_evidence":
        if out.sentences:
            errors.append("insufficient_evidence must not contain sentences")
        return (out if not errors else None), errors
    if not out.sentences:
        errors.append("answer has no sentences")
    if out.status == "partial" and not out.missing_information:
        errors.append("partial answer must list missing_information")
    for n, s in enumerate(out.sentences):
        if not s.evidence_ids:
            errors.append(f"sentence {n} has no citation")
        bad = [e for e in s.evidence_ids if e not in known]
        if bad:
            errors.append(f"sentence {n} cites unknown evidence {bad}")
    if sum(len(s.text) for s in out.sentences) > MAX_ANSWER_CHARS:
        errors.append("answer too long")
    return (out if not errors else None), errors


def render(out: AnswerOutput, pack: EvidencePack) -> tuple[str, list[Citation]]:
    """Markers and citation objects come from server-owned metadata only."""
    order: list[str] = []
    parts: list[str] = []
    for s in out.sentences:
        ids = list(dict.fromkeys(s.evidence_ids))
        order += [i for i in ids if i not in order]
        parts.append(_MARKER.sub("", s.text).strip() + " " + "".join(f"[{i}]" for i in ids))
    cites = []
    for eid in order:
        it = pack.item(eid)
        assert it is not None  # guaranteed by validate_answer
        cites.append(
            Citation(
                eid,
                it.note_id,
                it.revision_id,
                it.chunk_id,
                it.title,
                it.text,
                it.start_char,
                it.end_char,
            )
        )
    return " ".join(parts), cites
