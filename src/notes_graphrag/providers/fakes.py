"""Deterministic provider fakes for CI. They do not validate LM Studio behaviour."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from notes_graphrag.providers.lmstudio_provider import l2_normalize


class FakeEmbeddings:
    def __init__(self, dimensions: int = 16, profile: str = "fake-v1") -> None:
        self.dimensions = dimensions
        self.profile = profile
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            vec = [0.0] * self.dimensions
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
                vec[h % self.dimensions] += 1.0
            if not any(vec):
                vec[0] = 1.0
            out.append(l2_normalize(vec))
        return out


_SENT = re.compile(r"(\w[\w ]*?) works on (Project \w+)\.")
_USES = re.compile(r"(Project \w+) uses (\w+)\.")


class FakeChat:
    """Rule-based extractor for the running example only."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> str:
        self.calls += 1
        m0 = re.search(r"<<<\n(.*?)\n>>>", user, re.S)
        user = m0.group(1) if m0 else user  # spans are chunk-local
        ents: dict[str, dict[str, Any]] = {}
        claims: list[dict[str, Any]] = []

        def ent(name: str, kind: str) -> str:
            lid = ents.setdefault(
                name, {"local_id": f"e{len(ents) + 1}", "name": name, "kind": kind, "mentions": []}
            )["local_id"]
            return str(lid)

        for m in _SENT.finditer(user):
            s, o = ent(m.group(1), "Person"), ent(m.group(2), "Project")
            claims.append(_claim(s, "WORKS_ON", o, m.start(), m.end(), user))
        for m in _USES.finditer(user):
            s, o = ent(m.group(1), "Project"), ent(m.group(2), "Technology")
            claims.append(_claim(s, "USES", o, m.start(), m.end(), user))
        return json.dumps(
            {"schema_version": "1", "entities": list(ents.values()), "claims": claims}
        )


def _claim(s: str, p: str, o: str, a: int, b: int, text: str) -> dict[str, Any]:
    return {
        "subject_local_id": s,
        "predicate": p,
        "object_local_id": o,
        "polarity": "POSITIVE",
        "modality": "ASSERTED",
        "evidence": {"start": a, "end": b, "quote": text[a:b]},
    }


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


class DemoChat:
    """Demo-only model: rule-based extraction plus an answerer that cites matching evidence.

    It exists so the API/UI can be exercised without LM Studio. It proves nothing about model
    quality.
    """

    def __init__(self) -> None:
        self._extract = FakeChat()
        self.calls = 0

    def chat_text(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        return "ok"

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> str:
        self.calls += 1
        if schema_name != "answer":
            return self._extract.chat_json(
                model=model,
                system=system,
                user=user,
                schema_name=schema_name,
                schema=schema,
                max_tokens=max_tokens,
            )
        from notes_graphrag.retrieval.lexical import query_terms, tokens

        q = re.search(r"QUESTION:\n(.*?)\n\nEVIDENCE", user, re.S)
        terms = set(query_terms(q.group(1) if q else ""))
        sentences = []
        for eid, body in re.findall(r'<evidence id="(E\d+)"[^>]*>\n(.*?)\n</evidence>', user, re.S):
            if terms & set(tokens(body)):
                sentences.append({"text": body.strip()[:300], "evidence_ids": [eid]})
        if not sentences:
            return json.dumps(
                {"status": "insufficient_evidence", "sentences": [], "missing_information": []}
            )
        return json.dumps(
            {"status": "answered", "sentences": sentences[:3], "missing_information": []}
        )
