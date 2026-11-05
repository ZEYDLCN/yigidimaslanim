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
    if not key or not key.strip() or len(key) > 200:
        raise BadRequest("Idempotency-Key header is required", code="IDEMPOTENCY_KEY_REQUIRED")
    return key.strip()
