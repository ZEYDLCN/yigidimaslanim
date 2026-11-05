"""Conservative, workspace-scoped entity resolution (architecture 6.4)."""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field

from notes_graphrag.domain.ontology import EntityKind

NORMALIZATION_VERSION = "norm-v1"
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """NFC + str.lower + whitespace collapse.

    str.lower keeps Turkish pairs distinct: I->i, İ->i̇ (i + combining dot), ı->ı.
    No diacritic stripping and no casefold, so identity is never collapsed blindly.
    """
    s = unicodedata.normalize("NFC", name)
    return unicodedata.normalize("NFC", _WS.sub(" ", s.strip()).lower())


@dataclass
class KnownEntity:
    entity_id: str
    kind: EntityKind
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    context: set[str] = field(default_factory=set)  # normalised neighbouring entity names

    def names(self) -> set[str]:
        return {normalize_name(self.canonical_name), *map(normalize_name, self.aliases)}


def resolve(
    name: str,
    kind: EntityKind,
    context: set[str],
    candidates: list[KnownEntity],
) -> str | None:
    """Return an existing entity id, or None meaning "create a new entity".

    Same-kind exact name/alias match only. A single-token name needs context
    overlap to be reused; multiple matches need exactly one context match.
    """
    key = normalize_name(name)
    ctx = {normalize_name(c) for c in context}
    matches = [c for c in candidates if c.kind == kind and key in c.names()]
    if not matches:
        return None
    overlapping = [m for m in matches if ctx & {normalize_name(x) for x in m.context}]
    if len(overlapping) == 1:
        return overlapping[0].entity_id
    if len(overlapping) > 1:
        return None
    if len(matches) == 1 and len(key.split()) >= 2:
        return matches[0].entity_id
    return None


def new_entity_id() -> str:
    return str(uuid.uuid4())
