"""Hybrid retrieval: lexical + vector (RRF) -> entity seeds -> bounded graph expansion -> pack.

`mode` supports the evaluation baselines (architecture 16.2): "vector", "hybrid", "graph".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.entity_resolution import normalize_name
from notes_graphrag.providers.interfaces import EmbeddingProvider
from notes_graphrag.retrieval.evidence import EvidencePack, evidence_budget, pack_evidence
from notes_graphrag.retrieval.fusion import rrf
from notes_graphrag.retrieval.graph import ExpandedClaim, Limits, expand
from notes_graphrag.retrieval.lexical import tokens
from notes_graphrag.retrieval.source import ClaimRecord, KnowledgeSource
from notes_graphrag.retrieval.vector import QUERY_TEMPLATE_V1, format_query

CANDIDATE_CAP = 40
FUSED_CAP = 20
CHUNK_SEEDS = 5
Mode = Literal["vector", "hybrid", "graph"]


@dataclass
class RetrievalResult:
    pack: EvidencePack
    relevant: bool
    graph_version: int
    expanded: list[ExpandedClaim] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _contains(seq: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n > 0 and any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


class Retriever:
    def __init__(
        self,
        source: KnowledgeSource,
        embeddings: EmbeddingProvider,
        settings: Settings,
        *,
        limits: Limits | None = None,
        query_template: str = QUERY_TEMPLATE_V1,
        min_vector_score: float = 0.5,
        mode: Mode = "graph",
    ) -> None:
        self.source, self.emb, self.s = source, embeddings, settings
        self.limits = limits or Limits()
        self.template = query_template
        self.min_vector_score = min_vector_score
        self.mode = mode

    def link_entities(self, question: str) -> tuple[list[str], list[str]]:
        """Unambiguous question entities, plus ambiguous names (kept as text signals only)."""
        q = tokens(question)
        by_name: dict[str, set[str]] = {}
        for e in self.source.entities():
            for name in (e.name, *e.aliases):
                if _contains(q, tokens(name)):
                    by_name.setdefault(normalize_name(name), set()).add(e.entity_id)
        linked: list[str] = []
        ambiguous: list[str] = []
        for name, ids in sorted(by_name.items()):
            if len(ids) == 1:
                linked.extend(ids)
            else:
                ambiguous.append(name)
        return list(dict.fromkeys(linked)), ambiguous

    def retrieve(self, question: str) -> RetrievalResult:
        version = self.source.graph_version()
        qvec = self.emb.embed([format_query(self.template, question)])[0]
        vec_hits = self.source.vector_candidates(qvec, CANDIDATE_CAP)
        top = vec_hits[0][1] if vec_hits else 0.0
        vec_rank = [cid for cid, _ in vec_hits]
        lex_rank = (
            self.source.lexical_candidates(question, CANDIDATE_CAP) if self.mode != "vector" else []
        )
        fused = rrf([lex_rank, vec_rank], limit=FUSED_CAP)
        linked, ambiguous = self.link_entities(question) if self.mode == "graph" else ([], [])
        relevant = bool(lex_rank) or bool(linked) or top >= self.min_vector_score
        diag: dict[str, Any] = {
            "retrieval_mode": self.mode,
            "lexical_hits": len(lex_rank),
            "vector_top_score": round(top, 4),
            "linked_entities": len(linked),
            "ambiguous_names": ambiguous,
        }
        if not relevant:
            return RetrievalResult(EvidencePack(), False, version, diagnostics=diag)

        expanded: list[ExpandedClaim] = []
        seed_chunks = fused[:CHUNK_SEEDS]
        if self.mode == "graph":
            seeds = list(linked)
            for cid in seed_chunks:
                for cl in self.source.claims_for_chunk(cid):
                    seeds += [cl.subject_id, cl.object_id]
            expanded = expand(self.source, list(dict.fromkeys(seeds)), self.limits)
            expanded.sort(key=lambda x: (x.hop, x.seed_id not in linked, x.claim.claim_id))

        ids = set(fused) | {x.claim.chunk_id for x in expanded}
        by_id = self.source.chunks_by_ids(sorted(ids))

        # Chunk priority = fused rank score + best expansion-path bonus (hop 1: 1.0, hop 2: 0.5).
        score: dict[str, float] = {}
        for rank, cid in enumerate(fused, 1):
            score[cid] = 60.0 / (59 + rank)
        bonus: dict[str, float] = {}
        for x in expanded:
            bonus[x.claim.chunk_id] = max(bonus.get(x.claim.chunk_id, 0.0), 1.0 / x.hop)
        for cid, b in bonus.items():
            score[cid] = score.get(cid, 0.0) + b
        ranked_ids = sorted((c for c in score if c in by_id), key=lambda c: (-score[c], c))

        claims: dict[str, ClaimRecord] = {x.claim.claim_id: x.claim for x in expanded}
        for cid in seed_chunks:
            for cl in self.source.claims_for_chunk(cid):
                claims.setdefault(cl.claim_id, cl)

        pack = pack_evidence(
            [by_id[c] for c in ranked_ids],
            list(claims.values()),
            budget_tokens=evidence_budget(self.s, question),
            max_items=self.limits.final_passages,
        )
        diag |= {
            "evidence_count": len(pack.items),
            "expanded_claims": len(expanded),
            "limited_by_budget": pack.limited_by_budget,
        }
        return RetrievalResult(pack, bool(pack.items), version, expanded, diag)
