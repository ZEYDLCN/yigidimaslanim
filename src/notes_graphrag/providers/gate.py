"""Shared inference slot (architecture 12.4).

One LM Studio endpoint serves one request at a time across API and worker processes. The slot
lives in the database, so every process sees the same state. A timed-out request leaves the
slot RECOVERING: nothing else is admitted until a small probe proves the server responds again.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from notes_graphrag.core.errors import InferenceBusy, ProviderUnavailable
from notes_graphrag.persistence.base import Store
from notes_graphrag.persistence.models import GATE_IDLE, GATE_RECOVERING, PROBE_OWNER

log = logging.getLogger("notes_graphrag.gate")
AMBIGUOUS_CODE = "PROVIDER_TIMEOUT"


def _now() -> datetime:
    return datetime.now(UTC)


class InferenceGate:
    def __init__(
        self,
        store: Store,
        endpoint_id: str,
        *,
        lease_seconds: int = 180,
        wait_seconds: float = 30,
        probe: Callable[[], None] | None = None,
        clock: Callable[[], datetime] = _now,
        poll_seconds: float = 0.2,
    ) -> None:
        store.ensure_gate(endpoint_id)
        self.store = store
        self.endpoint_id = endpoint_id
        self.lease = lease_seconds
        self.wait = wait_seconds
        self.probe = probe
        self.clock = clock
        self.poll = poll_seconds

    def _probe_once(self) -> bool:
        assert self.probe is not None
        ptoken = str(uuid.uuid4())
        if not self.store.gate_try_acquire(
            self.endpoint_id, PROBE_OWNER, ptoken, self.clock(), self.lease
        ):
            return False
        target = GATE_RECOVERING
        try:
            self.probe()
            target = GATE_IDLE
            log.info("inference endpoint recovered")
        except Exception:  # noqa: BLE001 - any probe failure keeps the slot in recovery
            log.warning("inference recovery probe failed; slot stays in recovery")
        finally:
            self.store.gate_release(self.endpoint_id, ptoken, target)
        return target == GATE_IDLE

    def try_recover(self) -> bool:
        """Probe a RECOVERING slot now, without waiting for a request. True if it cleared."""
        if self.probe is None:
            return False
        if self.store.gate_status(self.endpoint_id).state != GATE_RECOVERING:
            return False
        return self._probe_once()

    def _acquire(self, owner: str) -> str:
        deadline = time.monotonic() + self.wait
        while True:
            token = str(uuid.uuid4())
            now = self.clock()
            if self.store.gate_try_acquire(self.endpoint_id, owner, token, now, self.lease):
                return token
            recovering = self.store.gate_status(self.endpoint_id).state == GATE_RECOVERING
            if self.probe is not None and recovering and self._probe_once():
                continue
            if time.monotonic() >= deadline:
                raise InferenceBusy("inference slot is busy; try again later")
            time.sleep(self.poll)

    def _beat(self, token: str, stop: threading.Event) -> None:
        interval = max(self.lease / 3, 0.05)
        while not stop.wait(interval):
            try:
                self.store.gate_heartbeat(self.endpoint_id, token, self.clock(), self.lease)
            except Exception:  # noqa: BLE001 - a lost heartbeat only lets the lease expire
                log.warning("gate heartbeat failed; lease will expire if this persists")

    @contextmanager
    def slot(self, owner: str) -> Iterator[str]:
        token = self._acquire(owner)
        stop = threading.Event()
        beat = threading.Thread(target=self._beat, args=(token, stop), daemon=True)
        beat.start()
        target = GATE_IDLE
        try:
            yield token
        except ProviderUnavailable as exc:
            if exc.code == AMBIGUOUS_CODE:
                target = GATE_RECOVERING
            raise
        finally:
            stop.set()
            beat.join(timeout=2)
            self.store.gate_release(self.endpoint_id, token, target)


class GatedChat:
    """Chat provider whose every call runs inside the shared inference slot."""

    def __init__(self, inner: Any, gate: InferenceGate, owner: str) -> None:
        self.inner, self.gate, self.owner = inner, gate, owner

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> str:
        with self.gate.slot(self.owner):
            return str(
                self.inner.chat_json(
                    model=model,
                    system=system,
                    user=user,
                    schema_name=schema_name,
                    schema=schema,
                    max_tokens=max_tokens,
                )
            )


class GatedEmbeddings:
    def __init__(self, inner: Any, gate: InferenceGate, owner: str) -> None:
        self.inner, self.gate, self.owner = inner, gate, owner
        self.profile: str = inner.profile
        self.dimensions: int = inner.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self.gate.slot(self.owner):
            return list(self.inner.embed(texts))


class GatedLM:
    """Every model call of an LM Studio provider passes the shared slot, diagnostics included.

    Model discovery is not inference and is not gated. A health check therefore cannot bypass the
    one-request-at-a-time limit (architecture 12.2).
    """

    def __init__(self, inner: Any, gate: InferenceGate, owner: str) -> None:
        self.inner, self.gate, self.owner = inner, gate, owner
        self.profile: str = inner.profile
        self.dimensions: int = inner.dimensions

    def list_models(self) -> list[str]:
        return list(self.inner.list_models())

    def chat_text(self, **kwargs: Any) -> str:
        with self.gate.slot(self.owner):
            return str(self.inner.chat_text(**kwargs))

    def chat_json(self, **kwargs: Any) -> str:
        with self.gate.slot(self.owner):
            return str(self.inner.chat_json(**kwargs))

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self.gate.slot(self.owner):
            return list(self.inner.embed(texts))
