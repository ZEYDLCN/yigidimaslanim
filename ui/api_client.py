"""Thin HTTP client. The UI talks only to the API: no database access, no extraction logic."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any

import httpx


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, request_id: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.status, self.code, self.message, self.request_id = status, code, message, request_id


# Tests replace this to route calls into an in-process app.
CLIENT_FACTORY: Callable[[], httpx.Client] | None = None


class Api:
    def __init__(self, client: httpx.Client | None = None) -> None:
        if client is None:
            client = (
                CLIENT_FACTORY()
                if CLIENT_FACTORY
                else httpx.Client(
                    base_url=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"), timeout=200
                )
            )
        self.c = client

    def _req(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            r = self.c.request(method, path, **kw)
        except httpx.TransportError as exc:
            raise ApiError(0, "API_UNREACHABLE", "The API is not reachable.") from exc
        if r.status_code == 204:
            return {}
        body = r.json()
        if r.status_code >= 400:
            e = body.get("error", {})
            raise ApiError(
                r.status_code,
                e.get("code", "ERROR"),
                e.get("message", ""),
                body.get("request_id", ""),
            )
        return body  # type: ignore[no-any-return]

    @staticmethod
    def new_key() -> str:
        return str(uuid.uuid4())

    def create_note(self, title: str, text: str, key: str) -> dict[str, Any]:
        return self._req(
            "POST",
            "/api/v1/notes",
            json={"title": title, "text": text},
            headers={"Idempotency-Key": key},
        )

    def upload_note(self, title: str, name: str, data: bytes, key: str) -> dict[str, Any]:
        return self._req(
            "POST",
            "/api/v1/notes/upload",
            files={"file": (name, data)},
            data={"title": title},
            headers={"Idempotency-Key": key},
        )

    def replace_note(self, note_id: str, title: str, text: str, key: str) -> dict[str, Any]:
        return self._req(
            "PUT",
            f"/api/v1/notes/{note_id}",
            json={"title": title, "text": text},
            headers={"Idempotency-Key": key},
        )

    def delete_note(self, note_id: str) -> None:
        self._req("DELETE", f"/api/v1/notes/{note_id}")

    def notes(self, cursor: str | None = None) -> dict[str, Any]:
        return self._req("GET", "/api/v1/notes", params={"limit": 50, "cursor": cursor})

    def revision(self, note_id: str, revision_id: str) -> dict[str, Any]:
        return self._req("GET", f"/api/v1/notes/{note_id}/revisions/{revision_id}")

    def retry_job(self, job_id: str) -> dict[str, Any]:
        return self._req("POST", f"/api/v1/jobs/{job_id}/retry")

    def ask(self, question: str, include_graph: bool) -> dict[str, Any]:
        return self._req(
            "POST", "/api/v1/questions", json={"question": question, "include_graph": include_graph}
        )

    def entities(self, q: str | None) -> dict[str, Any]:
        return self._req("GET", "/api/v1/entities", params={"q": q, "limit": 50})

    def graph(self, entity_id: str, hops: int) -> dict[str, Any]:
        return self._req("GET", "/api/v1/graph", params={"entity_id": entity_id, "hops": hops})

    def ready(self) -> dict[str, Any]:
        try:
            r = self.c.get("/health/ready")
        except httpx.TransportError as exc:
            raise ApiError(0, "API_UNREACHABLE", "The API is not reachable.") from exc
        return r.json()  # type: ignore[no-any-return]

    def status(self) -> dict[str, Any]:
        return self._req("GET", "/api/v1/system/status")
