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


def worker_loop(ctx: AppContext) -> None:
    while not ctx.stop.is_set():
        try:
            job = ctx.worker.run_once()
        except Exception:  # noqa: BLE001
            log.exception("worker iteration failed")
            job = None
        if job is None:
            ctx.stop.wait(0.2)


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
        if ctx.run_worker:
            ctx.stop.clear()
            thread = threading.Thread(target=worker_loop, args=(ctx,), daemon=True)
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

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _envelope(request, "INTERNAL_ERROR", "internal error", 500)

    for r in (notes.router, jobs.router, questions.router, graph.router, health.router):
        app.include_router(r)
    return app
