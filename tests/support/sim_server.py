"""Run the LM Studio simulator on a real socket in a background thread (for tests)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import uvicorn

from tests.support.lmstudio_sim import create_sim


@dataclass
class RunningSim:
    base_url: str
    app: object


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def running_sim(token: str | None = None, dims: int = 64) -> Iterator[RunningSim]:
    app = create_sim(token=token, dims=dims)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base}/sim/state", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        raise RuntimeError("simulator did not start")
    try:
        yield RunningSim(f"{base}/v1", app)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def control(base: str, **body: object) -> dict:
    root = base.removesuffix("/v1")
    r = httpx.post(f"{root}/sim/control", json=body, timeout=5)
    r.raise_for_status()
    return r.json()
