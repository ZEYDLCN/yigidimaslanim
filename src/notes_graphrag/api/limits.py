"""Request body limit, enforced while streaming (architecture 6.1)."""

from __future__ import annotations

import uuid
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLarge(Exception):
    """Raised while reading a request body that exceeds the configured limit."""


def too_large_response(scope: Scope, receive: Receive, send: Send) -> Any:
    response = JSONResponse(
        {
            "request_id": uuid.uuid4().hex,
            "error": {
                "code": "PAYLOAD_TOO_LARGE",
                "message": "request body exceeds the size limit",
                "retryable": False,
            },
        },
        status_code=413,
    )
    return response(scope, receive, send)


class BodyLimitMiddleware:
    """Rejects bodies over `max_bytes` with 413, with or without a Content-Length header.

    A declared length over the limit is refused before the body is read. Otherwise the body is
    counted as the route reads it. FastAPI wraps read errors in a 400, so the API's exception
    handler maps a BodyTooLarge cause back to 413 (see api/app.py).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared < 0 or declared > self.max_bytes:
                    await too_large_response(scope, receive, send)
                    return

        seen = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise BodyTooLarge()
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            if not started:
                await too_large_response(scope, receive, send)
            else:
                raise
