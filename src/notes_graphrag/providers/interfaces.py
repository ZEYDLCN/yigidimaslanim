from __future__ import annotations

from typing import Any, Protocol


class ChatProvider(Protocol):
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
        """Return the raw JSON string of a strict-schema completion."""
        ...


class EmbeddingProvider(Protocol):
    profile: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
