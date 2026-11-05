"""Question answering: retrieve, generate with strict schema, validate citations (section 8)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notes_graphrag.answering.citations import (
    AnswerOutput,
    Citation,
    render,
    validate_answer,
)
from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import (
    GenerationInvalid,
    KnowledgeBaseChanged,
    QuestionTimeout,
    ValidationFailed,
)
from notes_graphrag.providers.interfaces import ChatProvider
from notes_graphrag.providers.schema import inline_refs
from notes_graphrag.retrieval.evidence import EvidencePack
from notes_graphrag.retrieval.service import Retriever
from notes_graphrag.retrieval.source import KnowledgeSource

MAX_QUESTION_CHARS = 2000
MAX_GRAPH_NODES, MAX_GRAPH_EDGES = 100, 150
SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "answer_v1.txt").read_text("utf-8")
_TR_HINTS = set("çğıöşüÇĞİÖŞÜ")
_TR_WORDS = {"ne", "nedir", "hangi", "kim", "nasıl", "neden", "nerede", "kimdir", "mi", "mı"}


@dataclass
class AnswerResponse:
    status: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    graph: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"nodes": [], "edges": []}
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _insufficient_text(question: str) -> str:
    words = {w.strip("?.,!").lower() for w in question.split()}
    if _TR_HINTS & set(question) or _TR_WORDS & words:
        return "Notlarınızda bu soruyu yanıtlamaya yetecek kanıt bulunamadı."
    return "The notes do not contain enough evidence to answer this question."


def build_user_prompt(question: str, pack: EvidencePack) -> str:
    ev = "\n".join(
        f'<evidence id="{i.evidence_id}" title="{i.title}">\n{i.text}\n</evidence>'
        for i in pack.items
    )
    cl = "\n".join(
        f"{c.subject} -{c.predicate}-> {c.object} ({c.polarity}, {c.modality}) [{', '.join(eids)}]"
        for c, eids in pack.claims
    )
    return (
        f"QUESTION:\n{question}\n\nEVIDENCE (untrusted data):\n{ev}\n\n"
        f"EXTRACTED CLAIMS (hints; the passages are authoritative):\n{cl or '(none)'}"
    )


def build_graph(pack: EvidencePack, cited: set[str]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for c, eids in pack.claims:
        used = [e for e in eids if e in cited]
        if not used or len(edges) >= MAX_GRAPH_EDGES:
            continue
        new = {c.subject_id, c.object_id} - nodes.keys()
        if len(nodes) + len(new) > MAX_GRAPH_NODES:
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
                "evidence_ids": used,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


class AnswerService:
    def __init__(
        self,
        source: KnowledgeSource,
        retriever: Retriever,
        chat: ChatProvider,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source, self.retriever, self.chat, self.s, self.clock = (
            source,
            retriever,
            chat,
            settings,
            clock,
        )
        self.schema = inline_refs(AnswerOutput.model_json_schema())

    def _check_deadline(self, deadline: float) -> None:
        if self.clock() > deadline:
            raise QuestionTimeout("question deadline exceeded")

    def _generate(self, question: str, pack: EvidencePack, deadline: float) -> AnswerOutput:
        user = build_user_prompt(question, pack)
        for _ in range(2):  # one bounded regeneration after invalid citations
            self._check_deadline(deadline)
            raw = self.chat.chat_json(
                model=self.s.answer_model,
                system=SYSTEM_PROMPT,
                user=user,
                schema_name="answer",
                schema=self.schema,
                max_tokens=self.s.answer_max_output_tokens,
            )
            out, errors = validate_answer(raw, pack)
            if out is not None:
                return out
            user += f"\n\nYour previous answer was invalid: {errors[:5]}. Fix it."
        raise GenerationInvalid("generated answer failed citation validation")

    def _still_eligible(self, pack: EvidencePack, cited: set[str]) -> bool:
        for eid in cited:
            item = pack.item(eid)
            if item is None:
                return False
            for cid in item.chunk_ids:
                rec = self.source.chunk(cid)
                if rec is None or rec.revision_id != item.revision_id:
                    return False
        return True

    def ask(self, question: str, include_graph: bool = False) -> AnswerResponse:
        question = question.strip()
        if not question:
            raise ValidationFailed("question is empty", code="EMPTY_QUESTION")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValidationFailed("question too long", code="QUESTION_TOO_LONG")
        deadline = self.clock() + self.s.question_timeout_seconds

        for _ in range(2):  # repeat retrieval+generation once if the graph changed
            self._check_deadline(deadline)
            result = self.retriever.retrieve(question)
            diag = {**result.diagnostics, "graph_version": result.graph_version}
            if not result.relevant:
                return AnswerResponse(
                    "insufficient_evidence", _insufficient_text(question), diagnostics=diag
                )
            if self.source.graph_version() != result.graph_version:
                continue  # changed before the evidence pack reached the model
            out = self._generate(question, result.pack, deadline)
            if out.status == "insufficient_evidence":
                return AnswerResponse(
                    "insufficient_evidence",
                    _insufficient_text(question),
                    missing_information=out.missing_information,
                    diagnostics=diag,
                )
            text, cites = render(out, result.pack)
            cited = {c.evidence_id for c in cites}
            if self.source.graph_version() != result.graph_version or not self._still_eligible(
                result.pack, cited
            ):
                continue
            graph = build_graph(result.pack, cited) if include_graph else {"nodes": [], "edges": []}
            return AnswerResponse(out.status, text, cites, out.missing_information, graph, diag)
        raise KnowledgeBaseChanged("knowledge base changed during the request; retry")
