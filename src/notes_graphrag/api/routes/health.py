from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok
from notes_graphrag.ingestion.service import utcnow

router = APIRouter()
WORKER_FRESH_SECONDS = 30


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def ready(request: Request) -> JSONResponse:
    ctx = ctx_of(request)
    checks = {
        "persistence": f"{ctx.mode} (in-memory, non-durable; Neo4j repository not implemented)",
        "embedding_profile": ctx.settings.embedding_profile,
    }
    return JSONResponse({"status": "ready", "checks": checks})


@router.get("/api/v1/system/status")
def system_status(request: Request) -> JSONResponse:
    ctx = ctx_of(request)
    hb = ctx.store.worker_heartbeat
    age = (utcnow() - hb).total_seconds() if hb else None
    return ok(
        request,
        {
            "mode": ctx.mode,
            "graph_version": ctx.store.graph_version,
            "worker": {
                "heartbeat_age_seconds": age,
                "fresh": age is not None and age <= WORKER_FRESH_SECONDS,
            },
            "lm_studio": ctx.lm_status(),
            "models": {
                "extraction": ctx.settings.extraction_model,
                "answer": ctx.settings.answer_model,
                "embedding": ctx.settings.embedding_model,
                "embedding_profile": ctx.settings.embedding_profile,
                "embedding_dimensions": ctx.settings.embedding_dimensions,
            },
            "inference_gate": "not implemented (planned for the reliability phase)",
        },
    )
