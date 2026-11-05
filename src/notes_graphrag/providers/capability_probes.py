"""LM Studio capability probes (architecture section 2.2).

Each probe returns a specific diagnostic code instead of a generic failure. Discovery alone
never counts as readiness: chat, strict JSON, embeddings and a retrieval fixture are all run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import AppError
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider
from notes_graphrag.retrieval.vector import QUERY_TEMPLATE_V1, cosine, format_query

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ready": {"type": "boolean"}},
    "required": ["ready"],
    "additionalProperties": False,
}
_CODES = {
    "PROVIDER_AUTH_REJECTED": "AUTH_REJECTED",
    "OUTPUT_TRUNCATED": "OUTPUT_TRUNCATED",
    "NOT_JSON": "SCHEMA_FAILURE",
    "EMBEDDING_DIMENSION_MISMATCH": "DIMENSION_MISMATCH",
    "PROVIDER_TIMEOUT": "TIMEOUT",
    "LM_STUDIO_UNAVAILABLE": "ENDPOINT_UNREACHABLE",
}
FIXTURE_DOCS = [
    "Ada works on Project Atlas, which uses Neo4j as its graph database.",
    "The recipe needs two eggs, flour and a pinch of salt.",
    "Yarın İstanbul'da hava yağmurlu olacak.",
]
FIXTURE_QUERY = "Which database does Ada's project use?"


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    code: str = "OK"
    detail: str = ""


def _run(name: str, fn: Callable[[], str]) -> ProbeResult:
    try:
        return ProbeResult(name, True, "OK", fn())
    except AppError as exc:
        return ProbeResult(name, False, _CODES.get(exc.code, exc.code), exc.message)
    except (ValueError, KeyError, TypeError) as exc:
        return ProbeResult(name, False, "SCHEMA_FAILURE", str(exc)[:200])


def run_probes(
    provider: LMStudioProvider, settings: Settings, query_template: str = QUERY_TEMPLATE_V1
) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    def models() -> str:
        ids = set(provider.list_models())
        missing = [
            m
            for m in (settings.extraction_model, settings.answer_model, settings.embedding_model)
            if not m or m not in ids
        ]
        if missing:
            raise AppError(f"configured model id not served: {missing}", code="MODEL_MISSING")
        return f"{len(ids)} models listed"

    results.append(_run("models", models))
    if not results[0].ok:
        return results  # later probes would only repeat the same failure

    def chat() -> str:
        for prompt in ("Reply with the single word: ok.", "Tek kelimeyle yanıtla: tamam."):
            provider.chat_text(
                model=settings.extraction_model, system="Be brief.", user=prompt, max_tokens=64
            )
        return "english and turkish completions returned"

    def schema() -> str:
        raw = provider.chat_json(
            model=settings.extraction_model,
            system="Return the requested JSON object.",
            user="Set ready to true.",
            schema_name="capability_probe",
            schema=_PROBE_SCHEMA,
            max_tokens=64,
        )
        if json.loads(raw) != {"ready": True}:
            raise AppError("strict schema output did not match", code="SCHEMA_FAILURE")
        return "strict json_schema accepted"

    def embeddings() -> str:
        one = provider.embed(["probe"])
        batch = provider.embed(FIXTURE_DOCS)
        if len(one) != 1 or len(batch) != len(FIXTURE_DOCS):
            raise AppError("embedding count mismatch", code="EMBEDDINGS_UNSUPPORTED")
        return f"dimension {len(one[0])} verified"

    def retrieval() -> str:
        docs = provider.embed(FIXTURE_DOCS)
        (q,) = provider.embed([format_query(query_template, FIXTURE_QUERY)])
        sims = [cosine(q, d) for d in docs]
        if not sims[0] > max(sims[1:]):  # strictly first; a tie proves nothing
            raise AppError("expected passage not ranked first", code="RETRIEVAL_FIXTURE_FAILED")
        return "expected passage ranked first"

    for name, fn in (
        ("chat_bilingual", chat),
        ("structured_output", schema),
        ("embeddings", embeddings),
        ("retrieval_fixture", retrieval),
    ):
        results.append(_run(name, fn))
    return results
