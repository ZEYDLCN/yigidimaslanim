"""Initial ontology (architecture section 5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EntityKind(StrEnum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    CONCEPT = "Concept"
    PLACE = "Place"
    EVENT = "Event"


class Polarity(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATED = "NEGATED"


class Modality(StrEnum):
    ASSERTED = "ASSERTED"
    POSSIBLE = "POSSIBLE"
    PLANNED = "PLANNED"
    REPORTED = "REPORTED"


ALL_KINDS = frozenset(EntityKind)
K = EntityKind


@dataclass(frozen=True)
class PredicateRule:
    subjects: frozenset[EntityKind]
    objects: frozenset[EntityKind]
    same_kind: bool = False


PREDICATES: dict[str, PredicateRule] = {
    "WORKS_AT": PredicateRule(frozenset({K.PERSON}), frozenset({K.ORGANIZATION})),
    "WORKS_ON": PredicateRule(frozenset({K.PERSON, K.ORGANIZATION}), frozenset({K.PROJECT})),
    "USES": PredicateRule(
        frozenset({K.PERSON, K.ORGANIZATION, K.PROJECT}), frozenset({K.TECHNOLOGY})
    ),
    "CREATED_BY": PredicateRule(
        frozenset({K.PROJECT, K.TECHNOLOGY}), frozenset({K.PERSON, K.ORGANIZATION})
    ),
    "PART_OF": PredicateRule(
        frozenset({K.ORGANIZATION, K.PROJECT, K.CONCEPT}),
        frozenset({K.ORGANIZATION, K.PROJECT, K.CONCEPT}),
        same_kind=True,
    ),
    "DEPENDS_ON": PredicateRule(
        frozenset({K.PROJECT, K.TECHNOLOGY, K.CONCEPT}),
        frozenset({K.PROJECT, K.TECHNOLOGY, K.CONCEPT}),
    ),
    "LOCATED_IN": PredicateRule(
        frozenset({K.ORGANIZATION, K.PLACE, K.EVENT}), frozenset({K.PLACE})
    ),
    "RELATED_TO": PredicateRule(ALL_KINDS, ALL_KINDS),
}


def predicate_allows(predicate: str, subject: EntityKind, obj: EntityKind) -> bool:
    rule = PREDICATES.get(predicate)
    if rule is None or subject not in rule.subjects or obj not in rule.objects:
        return False
    return not (rule.same_kind and subject != obj)
