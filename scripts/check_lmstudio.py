"""Probe the configured LM Studio endpoint. Never prints credentials."""

from __future__ import annotations

import sys

from notes_graphrag.core.config import Settings
from notes_graphrag.providers.capability_probes import run_probes
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider


def main() -> int:
    s = Settings()
    provider = LMStudioProvider(
        base_url=s.llm_base_url,
        api_key=s.lm_studio_api_key,
        embedding_model=s.embedding_model,
        embedding_dimensions=s.embedding_dimensions,
        embedding_profile=s.embedding_profile,
        timeout=s.provider_timeout_seconds,
    )
    results = run_probes(provider, s)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL'}] {r.name}: {r.code} {r.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
