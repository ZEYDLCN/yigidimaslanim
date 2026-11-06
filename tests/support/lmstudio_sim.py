"""A stand-in for LM Studio's OpenAI-compatible REST API, for exercising the real HTTP adapter.

This is a SIMULATOR of the request/response contract used by providers/lmstudio_provider.py, not
LM Studio and not a model. Output comes from the deterministic rule-based fakes. Its purpose is to
let the adapter, timeouts, auth handling, the inference gate and the probes run over real sockets,
with faults injected on purpose (slow replies, truncation, malformed JSON, 401, 500, wrong
dimensions). Behaviour is switched at runtime with POST /sim/control.

Run:  uv run python -m tests.support.lmstudio_sim --host 127.0.0.1 --port 1234
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from notes_graphrag.providers.fakes import DemoChat, FakeEmbeddings

CHAT_MODES = {"ok", "slow", "truncated", "malformed", "error", "unavailable", "unauthorized"}
EMBED_MODES = {"ok", "slow", "bad_dims", "error", "unauthorized"}


class SimState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chat = "ok"
        self.chat_delay = 0.0
        self.embed = "ok"
        self.embed_delay = 0.0
        self.calls = {"chat": 0, "embed": 0}
        self.inflight = 0
        self.max_inflight = 0


def create_sim(
    *,
    chat_model: str = "sim-chat",
    embed_model: str = "sim-embed",
    dims: int = 64,
    token: str | None = None,
) -> FastAPI:
    app = FastAPI(title="LM Studio simulator (test support)")
    state = SimState()
    chat_impl = DemoChat()
    embed_impl = FakeEmbeddings(dims, "sim-hash")
    app.state.sim = state

    def auth_ok(authorization: str | None) -> bool:
        if token is None:
            return True
        return authorization == f"Bearer {token}"

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> JSONResponse:
        if not auth_ok(authorization):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": chat_model, "object": "model"},
                    {"id": embed_model, "object": "model"},
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str | None = Header(default=None)) -> Any:
        with state.lock:
            mode, delay = state.chat, state.chat_delay
            state.calls["chat"] += 1
            state.inflight += 1
            state.max_inflight = max(state.max_inflight, state.inflight)
        try:
            return await _chat(request, authorization, mode, delay)
        finally:
            with state.lock:
                state.inflight -= 1

    async def _chat(request: Request, authorization: str | None, mode: str, delay: float) -> Any:
        if not auth_ok(authorization) or mode == "unauthorized":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if mode == "error":
            return JSONResponse({"error": "internal"}, status_code=500)
        if mode == "unavailable":
            return JSONResponse({"error": "unavailable"}, status_code=503)
        body = await request.json()
        if mode == "slow":
            await asyncio.sleep(delay)
        fmt = body.get("response_format")
        schema_name = fmt["json_schema"]["name"] if fmt else None
        if schema_name == "capability_probe":
            content = json.dumps({"ready": True})
        elif fmt is None:
            content = "ok"
        else:
            content = chat_impl.chat_json(
                model=body["model"],
                system="",
                user=body["messages"][-1]["content"],
                schema_name=schema_name,
                schema={},
                max_tokens=body["max_tokens"],
            )
        finish = "stop"
        if mode == "truncated":
            content, finish = content[: max(1, len(content) // 3)], "length"
        if mode == "malformed" and fmt is not None:
            content = "Sure! Here is the JSON you asked for: {not json"
        return {
            "id": "sim",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, authorization: str | None = Header(default=None)) -> Any:
        with state.lock:
            mode, delay = state.embed, state.embed_delay
            state.calls["embed"] += 1
            state.inflight += 1
            state.max_inflight = max(state.max_inflight, state.inflight)
        try:
            return await _embed(request, authorization, mode, delay)
        finally:
            with state.lock:
                state.inflight -= 1

    async def _embed(request: Request, authorization: str | None, mode: str, delay: float) -> Any:
        if not auth_ok(authorization) or mode == "unauthorized":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if mode == "error":
            return JSONResponse({"error": "internal"}, status_code=500)
        if mode == "slow":
            await asyncio.sleep(delay)
        body = await request.json()
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        vecs = embed_impl.embed(texts)
        if mode == "bad_dims":
            vecs = [v + [0.0] for v in vecs]
        return {
            "object": "list",
            "model": body["model"],
            "data": [
                {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)
            ],
        }

    @app.post("/sim/control")
    async def control(request: Request) -> JSONResponse:
        body = await request.json()
        with state.lock:
            if "chat" in body:
                if body["chat"] not in CHAT_MODES:
                    return JSONResponse({"error": "bad chat mode"}, status_code=400)
                state.chat = body["chat"]
            if "embed" in body:
                if body["embed"] not in EMBED_MODES:
                    return JSONResponse({"error": "bad embed mode"}, status_code=400)
                state.embed = body["embed"]
            state.chat_delay = float(body.get("chat_delay", state.chat_delay))
            state.embed_delay = float(body.get("embed_delay", state.embed_delay))
            return JSONResponse(
                {
                    "chat": state.chat,
                    "embed": state.embed,
                    "chat_delay": state.chat_delay,
                    "embed_delay": state.embed_delay,
                }
            )

    @app.get("/sim/state")
    async def sim_state() -> dict[str, Any]:
        with state.lock:
            return {
                "chat": state.chat,
                "embed": state.embed,
                "calls": dict(state.calls),
                "max_inflight": state.max_inflight,
            }

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--dims", type=int, default=int(os.environ.get("SIM_DIMS", "64")))
    args = ap.parse_args()
    token = os.environ.get("SIM_TOKEN") or None
    uvicorn.run(create_sim(dims=args.dims, token=token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
