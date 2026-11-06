"""HTTPX adapter for LM Studio's OpenAI-compatible REST API (architecture 2.1, 6.5)."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
from pydantic import SecretStr

from notes_graphrag.core.errors import ProviderBadResponse, ProviderUnavailable

MAX_EMBED_BATCH = 8


def l2_normalize(vec: list[float]) -> list[float]:
    if not all(math.isfinite(x) for x in vec):
        raise ProviderBadResponse("embedding contains non-finite values")
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ProviderBadResponse("embedding has zero norm")
    return [x / norm for x in vec]


class LMStudioProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None = None,
        embedding_model: str = "",
        embedding_dimensions: int = 1024,
        embedding_profile: str = "",
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # Bearer header only when a real token is configured (loopback needs none).
        headers = {}
        if api_key and api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.embedding_model = embedding_model
        self.dimensions = embedding_dimensions
        self.profile = embedding_profile

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                "LM Studio request timed out", code="PROVIDER_TIMEOUT"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable("LM Studio is unreachable") from exc
        if resp.status_code in (401, 403):
            raise ProviderBadResponse("LM Studio rejected the token", code="PROVIDER_AUTH_REJECTED")
        if resp.status_code >= 500:
            raise ProviderUnavailable(f"LM Studio server error {resp.status_code}")
        if resp.status_code >= 400:
            # Provider error bodies are deliberately not echoed (may contain content).
            raise ProviderBadResponse(f"LM Studio rejected request ({resp.status_code})")
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderBadResponse("LM Studio returned non-JSON") from exc
        if not isinstance(data, dict):
            raise ProviderBadResponse("unexpected response envelope")
        return data

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get("/models")
        except httpx.TransportError as exc:
            raise ProviderUnavailable("LM Studio is unreachable") from exc
        if resp.status_code in (401, 403):
            raise ProviderBadResponse("LM Studio rejected the token", code="PROVIDER_AUTH_REJECTED")
        if resp.status_code >= 400:
            raise ProviderUnavailable(f"LM Studio model listing failed ({resp.status_code})")
        try:
            return [m["id"] for m in resp.json()["data"]]
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderBadResponse("malformed model listing") from exc

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
        data = self._post(
            "/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            },
        )
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponse("malformed chat envelope") from exc
        if finish == "length":
            raise ProviderBadResponse("output truncated", code="OUTPUT_TRUNCATED")
        if not isinstance(content, str) or not content.strip():
            raise ProviderBadResponse("missing final answer")
        try:
            json.loads(content)
        except ValueError as exc:
            raise ProviderBadResponse("content is not JSON", code="NOT_JSON") from exc
        return content

    def chat_text(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderBadResponse("malformed chat envelope") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderBadResponse("missing final answer")
        return content

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), MAX_EMBED_BATCH):
            batch = texts[i : i + MAX_EMBED_BATCH]
            data = self._post("/embeddings", {"model": self.embedding_model, "input": batch})
            try:
                items = sorted(data["data"], key=lambda d: d["index"])
                vecs = [[float(x) for x in d["embedding"]] for d in items]
                indices = [d["index"] for d in items]
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderBadResponse("malformed embedding envelope") from exc
            if indices != list(range(len(batch))) or len(vecs) != len(batch):
                raise ProviderBadResponse("embedding count/index mismatch")
            for v in vecs:
                if len(v) != self.dimensions:
                    raise ProviderBadResponse(
                        "embedding dimension mismatch", code="EMBEDDING_DIMENSION_MISMATCH"
                    )
                out.append(l2_normalize(v))
        return out

    def close(self) -> None:
        self._client.close()
