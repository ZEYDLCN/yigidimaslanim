"""Vector scoring and the versioned query preprocessing (part of the embedding profile)."""

from __future__ import annotations

import math
from collections.abc import Sequence

QUERY_TEMPLATE_V1 = (
    "Instruct: Given a question about personal notes, retrieve passages that answer the "
    "question.\nQuery:{question}"
)


def format_query(template: str, question: str) -> str:
    return template.replace("{question}", question)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0 or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
