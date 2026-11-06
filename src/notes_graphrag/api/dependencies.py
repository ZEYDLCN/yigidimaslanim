"""Wiring: settings -> storage -> providers -> inference gate -> services.

`demo=True` swaps LM Studio for deterministic rule-based fakes (no model quality claims).
The storage backend is either the in-memory reference store or Neo4j.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from notes_graphrag.answering.service import AnswerService
from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import DependencyUnavailable
from notes_graphrag.ingestion.extraction import ExtractionOutput
from notes_graphrag.ingestion.service import NoteService
from notes_graphrag.ingestion.worker import Worker
from notes_graphrag.persistence.base import Store
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.persistence.memory_source import MemoryKnowledge
from notes_graphrag.persistence.migrations import latest_version
from notes_graphrag.providers.fakes import DemoChat, FakeEmbeddings
from notes_graphrag.providers.gate import GatedChat, GatedEmbeddings, InferenceGate
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider
from notes_graphrag.providers.schema import inline_refs
from notes_graphrag.retrieval.service import Retriever
from notes_graphrag.retrieval.source import KnowledgeSource
from notes_graphrag.retrieval.vector import QUERY_TEMPLATE_V1

log = logging.getLogger("notes_graphrag.deps")
DEMO_DIMENSIONS = 64
DEMO_PROFILE = "fake-v1"


def effective_settings(settings: Settings, demo: bool) -> Settings:
    if not demo:
        return settings
    return settings.model_copy(
        update={"embedding_dimensions": DEMO_DIMENSIONS, "embedding_profile": DEMO_PROFILE}
    )


@dataclass
class Providers:
    chat: Any
    emb: Any
    probe: Callable[[], None]
    query_template: str
    lm: LMStudioProvider | None = None


def build_providers(settings: Settings, *, demo: bool) -> Providers:
    if demo:
        chat = DemoChat()
        return Providers(
            chat, FakeEmbeddings(DEMO_DIMENSIONS, DEMO_PROFILE), lambda: None, "{question}"
        )
    lm = LMStudioProvider(
        base_url=settings.llm_base_url,
        api_key=settings.lm_studio_api_key,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        embedding_profile=settings.embedding_profile,
        timeout=settings.provider_timeout_seconds,
    )
    emb: Any = lm
    if settings.embedding_base_url != settings.llm_base_url:
        emb = LMStudioProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            embedding_profile=settings.embedding_profile,
            timeout=settings.provider_timeout_seconds,
        )

    def probe() -> None:
        lm.chat_text(model=settings.extraction_model, system="Be brief.", user="ok", max_tokens=4)

    return Providers(lm, emb, probe, QUERY_TEMPLATE_V1, lm)


def build_storage(settings: Settings, backend: str) -> tuple[Store, KnowledgeSource, Any]:
    """Returns (store, knowledge view, closeable resource or None)."""
    if backend == "memory":
        mem = MemoryStore(workspace_id=settings.workspace_id)
        return mem, MemoryKnowledge(mem), None
    if backend != "neo4j":
        raise ValueError(f"unknown store backend: {backend}")
    if not settings.neo4j_password.get_secret_value():
        raise DependencyUnavailable("NEO4J_PASSWORD is not set", code="CONFIG_MISSING")
    from neo4j import GraphDatabase

    from notes_graphrag.persistence.neo4j_source import Neo4jKnowledge
    from notes_graphrag.persistence.neo4j_store import Neo4jStore

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )
    neo = Neo4jStore(driver, settings.neo4j_database, settings.workspace_id)
    knowledge = Neo4jKnowledge(driver, settings.neo4j_database, settings.workspace_id)
    return neo, knowledge, driver


def check_schema(store: Store) -> None:
    applied = store.schema_version()
    expected = latest_version()
    if applied != expected:
        raise RuntimeError(
            f"database schema is version {applied}, expected {expected}; "
            "run `notes-graphrag migrate`"
        )


def build_gate(settings: Settings, store: Store, providers: Providers) -> InferenceGate:
    return InferenceGate(
        store,
        settings.lm_studio_instance_id,
        lease_seconds=settings.job_lease_seconds,
        wait_seconds=settings.inference_slot_wait_seconds,
        probe=providers.probe,
    )


def build_worker(
    settings: Settings, store: Store, providers: Providers, gate: InferenceGate, owner: str
) -> Worker:
    schema = inline_refs(ExtractionOutput.model_json_schema())
    return Worker(
        store,
        settings,
        GatedChat(providers.chat, gate, owner),
        GatedEmbeddings(providers.emb, gate, owner),
        schema,
    )


@dataclass
class AppContext:
    settings: Settings
    store: Store
    knowledge: KnowledgeSource
    notes: NoteService
    answers: AnswerService
    gate: InferenceGate
    worker: Worker
    mode: str
    demo: bool
    lm_status: Callable[[], dict[str, Any]]
    embedded_worker: bool = False
    closeable: Any = None
    stop: threading.Event = field(default_factory=threading.Event)


def build_context(
    settings: Settings,
    *,
    demo: bool = False,
    backend: str = "memory",
    embedded_worker: bool = False,
    check: bool = True,
) -> AppContext:
    s = effective_settings(settings, demo)
    store, knowledge, closeable = build_storage(s, backend)
    if check and backend == "neo4j":
        check_schema(store)
    providers = build_providers(s, demo=demo)
    gate = build_gate(s, store, providers)
    gated_emb = GatedEmbeddings(providers.emb, gate, "api")
    gated_chat = GatedChat(providers.chat, gate, "api")
    retriever = Retriever(knowledge, gated_emb, s, query_template=providers.query_template)
    answers = AnswerService(knowledge, retriever, gated_chat, s)
    worker = build_worker(s, store, providers, gate, f"worker:embedded:{id(store)}")

    def lm_status() -> dict[str, Any]:
        if providers.lm is None:
            return {"mode": "demo-fake", "endpoint_reachable": None}
        out: dict[str, Any] = {
            "mode": "lmstudio",
            "authentication": "configured" if s.lm_studio_api_key else "not-required",
            "capability_probes": "run scripts/check_lmstudio.py",
        }
        try:
            ids = set(providers.lm.list_models())
        except Exception:  # noqa: BLE001 - report unreachable without leaking details
            return {**out, "endpoint_reachable": False}
        return {
            **out,
            "endpoint_reachable": True,
            "extraction_model_discoverable": s.extraction_model in ids,
            "answer_model_discoverable": s.answer_model in ids,
            "embedding_model_discoverable": s.embedding_model in ids,
        }

    return AppContext(
        settings=s,
        store=store,
        knowledge=knowledge,
        notes=NoteService(store, s),
        answers=answers,
        gate=gate,
        worker=worker,
        mode=f"{backend}-{'demo' if demo else 'lmstudio'}",
        demo=demo,
        lm_status=lm_status,
        embedded_worker=embedded_worker,
        closeable=closeable,
    )
