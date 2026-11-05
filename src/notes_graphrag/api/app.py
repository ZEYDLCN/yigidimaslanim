"""FastAPI application: request IDs, error envelope, routers, in-process worker (dev mode)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from notes_graphrag.api.dependencies import AppContext
from notes_graphrag.api.routes import graph, health, jobs, notes, questions
from notes_graphrag.core.errors import AppError

log = logging.getLogger("notes_graphrag")


from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired  # noqa: E402

from notes_graphrag.ingestion.worker import run_forever  # noqa: E402


def _envelope(
    request: Request, code: str, message: str, status: int, retryable: bool = False
) -> JSONResponse:
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        {"request_id": rid, "error": {"code": code, "message": message, "retryable": retryable}},
        status_code=status,
    )


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        thread = None
        if ctx.embedded_worker:
            ctx.stop.clear()
            thread = threading.Thread(
                target=run_forever,
                args=(ctx.worker, ctx.store, ctx.stop, ctx.settings.worker_poll_seconds),
                daemon=True,
            )
            thread.start()
        yield
        ctx.stop.set()
        if thread:
            thread.join(timeout=5)

    app = FastAPI(title="Notes GraphRAG", lifespan=lifespan)
    app.state.ctx = ctx

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid.uuid4().hex
        start = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        # IDs, timings and outcome only: never note text or credentials.
        log.info(
            "request id=%s %s %s -> %s %.0fms",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.monotonic() - start) * 1000,
        )
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        return _envelope(request, exc.code, exc.message, exc.http_status, exc.retryable)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        return _envelope(request, "VALIDATION_FAILED", f"invalid request fields: {fields}", 422)

    @app.exception_handler(ServiceUnavailable)
    @app.exception_handler(SessionExpired)
    async def database_down(request: Request, exc: Exception) -> JSONResponse:
        log.warning("database unavailable")
        return _envelope(request, "DEPENDENCY_UNAVAILABLE", "database is unavailable", 503, True)

    @app.exception_handler(Neo4jError)
    async def database_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("database error: %s", type(exc).__name__)
        return _envelope(request, "DATABASE_ERROR", "database request failed", 503, True)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _envelope(request, "INTERNAL_ERROR", "internal error", 500)

    for r in (notes.router, jobs.router, questions.router, graph.router, health.router):
        app.include_router(r)
    return app
