"""Bounded claim-level graph expansion (architecture section 8.2).

One semantic hop is subject-claim-object; it is never an unrestricted two-edge traversal.
"""

from __future__ import annotations

from dataclasses import dataclass

from notes_graphrag.retrieval.source import ClaimRecord, KnowledgeSource


@dataclass(frozen=True)
class Limits:
    max_seeds: int = 10
    max_hops: int = 2
    claims_per_entity: int = 15
    total_claims: int = 80
    final_passages: int = 12


@dataclass(frozen=True)
class ExpandedClaim:
    claim: ClaimRecord
    hop: int
    via_entity: str  # entity the claim was discovered from
    seed_id: str  # seed this path originates from


def expand(
    source: KnowledgeSource, seed_ids: list[str], limits: Limits | None = None
) -> list[ExpandedClaim]:
    limits = limits or Limits()
    seeds = list(dict.fromkeys(seed_ids))[: limits.max_seeds]
    visited: set[str] = set(seeds)
    origin: dict[str, str] = {s: s for s in seeds}
    seen_claims: set[str] = set()
    out: list[ExpandedClaim] = []
    frontier = seeds
    for hop in range(1, limits.max_hops + 1):
        nxt: list[str] = []
        for ent in frontier:  # BFS rounds: hop-1 evidence is never starved by hop-2
            taken = 0
            for claim in source.claims_for_entity(ent):
                if len(out) >= limits.total_claims:
                    return out
                if claim.claim_id in seen_claims:
                    continue
                if taken >= limits.claims_per_entity:  # a hub cannot eat the whole budget
                    break
                taken += 1
                seen_claims.add(claim.claim_id)
                out.append(ExpandedClaim(claim, hop, ent, origin[ent]))
                other = claim.object_id if claim.subject_id == ent else claim.subject_id
                if other not in visited:
                    visited.add(other)
                    origin[other] = origin[ent]
                    nxt.append(other)
        frontier = nxt
        if not frontier:
            break
    return out
