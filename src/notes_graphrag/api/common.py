from __future__ import annotations

from typing import Any

from fastapi import Header, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.dependencies import AppContext
from notes_graphrag.core.errors import BadRequest


def ctx_of(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def ok(request: Request, body: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse({"request_id": request.state.request_id, **body}, status_code=status)


def idempotency_key(
    key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str:
    if key is None or not key.strip():
        raise BadRequest("Idempotency-Key header is required", code="IDEMPOTENCY_KEY_REQUIRED")
    key = key.strip()
    if len(key) > 200 or not key.isprintable():
        raise BadRequest(
            "Idempotency-Key must be at most 200 printable characters",
            code="IDEMPOTENCY_KEY_INVALID",
        )
    return key
