"""Reciprocal rank fusion and active-scope filtering (architecture section 8.1)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, limit: int | None = None) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] += 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda i: (-scores[i], i))
    return ordered[:limit] if limit is not None else ordered


def overfetch(
    fetch: Callable[[int], Iterable[str]],
    is_active: Callable[[str], bool],
    *,
    want: int,
    steps: Sequence[int] = (40, 80, 160),
) -> tuple[list[str], bool]:
    """Progressively overfetch until `want` active hits remain.

    Returns (hits, limited) where limited means the cap was reached short of `want`.
    """
    hits: list[str] = []
    for n in steps:
        hits = [h for h in fetch(n) if is_active(h)]
        if len(hits) >= want:
            return hits[:want], False
    return hits[:want], True
