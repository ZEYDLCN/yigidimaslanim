"""Application context: wiring of settings, store, providers and services."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from notes_graphrag.answering.service import AnswerService
from notes_graphrag.core.config import Settings
from notes_graphrag.ingestion.extraction import ExtractionOutput
from notes_graphrag.ingestion.service import NoteService
from notes_graphrag.ingestion.worker import Worker
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.persistence.memory_source import MemoryKnowledge
from notes_graphrag.providers.fakes import DemoChat, FakeEmbeddings
from notes_graphrag.providers.interfaces import ChatProvider, EmbeddingProvider
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider
from notes_graphrag.providers.schema import inline_refs
from notes_graphrag.retrieval.service import Retriever
from notes_graphrag.retrieval.source import KnowledgeSource
from notes_graphrag.retrieval.vector import QUERY_TEMPLATE_V1


@dataclass
class AppContext:
    settings: Settings
    store: MemoryStore
    notes: NoteService
    source: KnowledgeSource
    answers: AnswerService
    worker: Worker
    mode: str
    lm_status: Callable[[], dict[str, Any]]
    run_worker: bool = True
    stop: threading.Event = field(default_factory=threading.Event)


def build_context(settings: Settings, *, demo: bool = False, run_worker: bool = True) -> AppContext:
    store = MemoryStore(workspace_id=settings.workspace_id)
    chat: ChatProvider
    emb: EmbeddingProvider
    template = QUERY_TEMPLATE_V1
    lm: LMStudioProvider | None = None
    if demo:
        settings = settings.model_copy(
            update={"embedding_dimensions": 64, "embedding_profile": "fake-v1"}
        )
        chat, emb, template = DemoChat(), FakeEmbeddings(64), "{question}"
    else:
        lm = LMStudioProvider(
            base_url=settings.llm_base_url,
            api_key=settings.lm_studio_api_key,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            embedding_profile=settings.embedding_profile,
            timeout=settings.provider_timeout_seconds,
        )
        chat = emb = lm
        if settings.embedding_base_url != settings.llm_base_url:
            emb = LMStudioProvider(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
                embedding_profile=settings.embedding_profile,
                timeout=settings.provider_timeout_seconds,
            )
    source = MemoryKnowledge(store)
    retriever = Retriever(source, emb, settings, query_template=template)
    schema = inline_refs(ExtractionOutput.model_json_schema())

    def lm_status() -> dict[str, Any]:
        if lm is None:
            return {"mode": "demo-fake", "endpoint_reachable": None}
        out: dict[str, Any] = {
            "mode": "lmstudio",
            "authentication": "configured" if settings.lm_studio_api_key else "not-required",
            "capability_probes": "not run by the API; use scripts/check_lmstudio.py",
        }
        try:
            ids = set(lm.list_models())
        except Exception:  # noqa: BLE001 - report as unreachable without leaking details
            return {**out, "endpoint_reachable": False}
        return {
            **out,
            "endpoint_reachable": True,
            "extraction_model_discoverable": settings.extraction_model in ids,
            "answer_model_discoverable": settings.answer_model in ids,
            "embedding_model_discoverable": settings.embedding_model in ids,
        }

    return AppContext(
        settings=settings,
        store=store,
        notes=NoteService(store, settings),
        source=source,
        answers=AnswerService(source, retriever, chat, settings),
        worker=Worker(store, settings, chat, emb, schema),
        mode="memory-demo" if demo else "memory-lmstudio",
        lm_status=lm_status,
        run_worker=run_worker,
    )
