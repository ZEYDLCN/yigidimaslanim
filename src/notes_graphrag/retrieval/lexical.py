"""Lucene query escaping for user input (parameterisation alone is not enough)."""

from __future__ import annotations

import re

_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def escape_lucene(query: str) -> str:
    escaped = _SPECIAL.sub(r"\\\1", query)
    # Boolean operator words become plain terms when lower-cased and quoted away.
    return " ".join(f'"{w}"' if w in {"AND", "OR", "NOT"} else w for w in escaped.split())


STOPWORDS = frozenset(
    "the and for are was were what which who whom does did how why when where with from that this "
    "have has had use uses used its not you your can will about into than then them they "
    "ile ve bir bu şu için ne nedir hangi kim nasıl neden nerede mi mı mu mü da de ki gibi "
    "olan ise veya ama çok daha en".split()
)


def tokens(text: str) -> list[str]:
    from notes_graphrag.ingestion.entity_resolution import normalize_name

    return re.findall(r"\w+", normalize_name(text))


def query_terms(question: str) -> list[str]:
    seen: dict[str, None] = {}
    for t in tokens(question):
        if len(t) >= 3 and t not in STOPWORDS:
            seen.setdefault(t)
    return list(seen)
