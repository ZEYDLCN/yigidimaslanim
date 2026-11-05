from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok

router = APIRouter()
WORKER_FRESH_SECONDS = 30


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def ready(request: Request) -> JSONResponse:
    ctx = ctx_of(request)
    h = ctx.store.health()
    schema_ok = h.get("schema_version") == h.get("expected_schema_version", h.get("schema_version"))
    indexes_ok = h.get("indexes_online") is not False
    is_ready = bool(h.get("reachable")) and schema_ok and indexes_ok
    body = {
        "status": "ready" if is_ready else "not_ready",
        "checks": {
            "persistence": h.get("persistence"),
            "reachable": h.get("reachable"),
            "schema_version": h.get("schema_version"),
            "schema_ok": schema_ok,
            "indexes_online": h.get("indexes_online"),
            "embedding_profile": ctx.settings.embedding_profile,
        },
    }
    return JSONResponse(body, status_code=200 if is_ready else 503)


@router.get("/api/v1/system/status")
def system_status(request: Request) -> JSONResponse:
    ctx = ctx_of(request)
    hb = ctx.store.last_heartbeat()
    age = (datetime.now(UTC) - hb).total_seconds() if hb else None
    gate = ctx.store.gate_status(ctx.settings.lm_studio_instance_id)
    return ok(
        request,
        {
            "mode": ctx.mode,
            "graph_version": ctx.store.graph_version(),
            "worker": {
                "heartbeat_age_seconds": age,
                "fresh": age is not None and age <= WORKER_FRESH_SECONDS,
            },
            "inference_slot": {"state": gate.state, "lease_expires_at": gate.lease_expires_at},
            "lm_studio": ctx.lm_status(),
            "models": {
                "extraction": ctx.settings.extraction_model,
                "answer": ctx.settings.answer_model,
                "embedding": ctx.settings.embedding_model,
                "embedding_profile": ctx.settings.embedding_profile,
                "embedding_dimensions": ctx.settings.embedding_dimensions,
            },
        },
    )
