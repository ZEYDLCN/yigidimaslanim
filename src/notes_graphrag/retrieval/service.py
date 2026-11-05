"""Hybrid retrieval: lexical + vector (RRF) -> entity seeds -> bounded graph expansion -> pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.entity_resolution import normalize_name
from notes_graphrag.providers.interfaces import EmbeddingProvider
from notes_graphrag.retrieval.evidence import EvidencePack, evidence_budget, pack_evidence
from notes_graphrag.retrieval.fusion import rrf
from notes_graphrag.retrieval.graph import ExpandedClaim, Limits, expand
from notes_graphrag.retrieval.lexical import query_terms, tokens
from notes_graphrag.retrieval.source import ClaimRecord, KnowledgeSource
from notes_graphrag.retrieval.vector import QUERY_TEMPLATE_V1, cosine, format_query

CANDIDATE_CAP = 40
FUSED_CAP = 20
CHUNK_SEEDS = 5


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
        use_graph: bool = True,
    ) -> None:
        self.source, self.emb, self.s = source, embeddings, settings
        self.limits, self.template = limits or Limits(), query_template
        self.min_vector_score, self.use_graph = min_vector_score, use_graph

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
        chunks = self.source.chunks()
        empty = RetrievalResult(EvidencePack(), False, version, diagnostics={"evidence_count": 0})
        if not chunks:
            return empty

        qvec = self.emb.embed([format_query(self.template, question)])[0]
        scored = sorted(
            ((cosine(qvec, c.embedding), c.chunk_id) for c in chunks), key=lambda x: (-x[0], x[1])
        )
        vec_rank = [cid for _, cid in scored[:CANDIDATE_CAP]]
        top_vec = scored[0][0]
        terms = set(query_terms(question))
        hits = sorted(
            ((len(terms & set(tokens(c.text))), c.chunk_id) for c in chunks),
            key=lambda x: (-x[0], x[1]),
        )
        lex_rank = [cid for n, cid in hits[:CANDIDATE_CAP] if n > 0]
        fused = rrf([lex_rank, vec_rank], limit=FUSED_CAP)

        linked, ambiguous = self.link_entities(question)
        relevant = bool(lex_rank) or bool(linked) or top_vec >= self.min_vector_score
        diag: dict[str, Any] = {
            "retrieval_mode": "hybrid_graph" if self.use_graph else "hybrid",
            "candidates": len(chunks),
            "lexical_hits": len(lex_rank),
            "linked_entities": len(linked),
            "ambiguous_names": ambiguous,
        }
        if not relevant:
            return RetrievalResult(EvidencePack(), False, version, diagnostics=diag)

        by_id = {c.chunk_id: c for c in chunks}
        expanded: list[ExpandedClaim] = []
        if self.use_graph:
            seeds = list(linked)
            for cid in fused[:CHUNK_SEEDS]:
                for cl in self.source.claims_for_chunk(cid):
                    seeds += [cl.subject_id, cl.object_id]
            seeds = list(dict.fromkeys(seeds))
            expanded = expand(self.source, seeds, self.limits)
            expanded.sort(key=lambda x: (x.hop, x.seed_id not in linked, x.claim.claim_id))

        # Chunk priority = fused rank score + best expansion-path bonus (hop 1: 1.0, hop 2: 0.5).
        score: dict[str, float] = {}
        for rank, cid in enumerate(fused, 1):
            score[cid] = 60.0 / (59 + rank)
        bonus: dict[str, float] = {}
        for x in expanded:
            bonus[x.claim.chunk_id] = max(bonus.get(x.claim.chunk_id, 0.0), 1.0 / x.hop)
        for cid, b in bonus.items():
            score[cid] = score.get(cid, 0.0) + b
        ranked_ids = sorted(score, key=lambda c: (-score[c], c))
        claims: list[ClaimRecord] = [x.claim for x in expanded]
        for cid in fused[:CHUNK_SEEDS]:
            claims += [c for c in self.source.claims_for_chunk(cid) if c not in claims]

        pack = pack_evidence(
            [by_id[c] for c in ranked_ids if c in by_id],
            claims,
            budget_tokens=evidence_budget(self.s, question),
            max_items=self.limits.final_passages,
        )
        diag |= {
            "evidence_count": len(pack.items),
            "expanded_claims": len(expanded),
            "limited_by_budget": pack.limited_by_budget,
        }
        return RetrievalResult(pack, bool(pack.items), version, expanded, diag)
