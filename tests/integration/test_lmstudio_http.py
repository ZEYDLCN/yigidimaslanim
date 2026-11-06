"""The real HTTP adapter against the LM Studio contract simulator, over real sockets.

Not LM Studio: see tests/support/lmstudio_sim.py. These cases prove the adapter, timeouts, auth,
probes and the inference gate behave correctly on the wire, with faults injected on purpose.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.support.sim_server import control, running_sim

from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import ProviderBadResponse, ProviderUnavailable
from notes_graphrag.persistence.memory import MemoryStore
from notes_graphrag.persistence.models import GATE_IDLE, GATE_RECOVERING
from notes_graphrag.providers.capability_probes import run_probes
from notes_graphrag.providers.gate import GatedChat, GatedEmbeddings, InferenceGate
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider

CHAT, EMB = "sim-chat", "sim-embed"
SCHEMA = {"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}


def settings(**over) -> Settings:
    base = dict(
        _env_file=None,
        extraction_model=CHAT,
        answer_model=CHAT,
        embedding_model=EMB,
        embedding_dimensions=64,
        embedding_profile="sim-hash-64",
        provider_timeout_seconds=0.5,
    )
    base.update(over)
    return Settings(**base)


def provider(base: str, s: Settings, token: str | None = None) -> LMStudioProvider:
    return LMStudioProvider(
        base_url=base,
        api_key=SecretStr(token) if token else None,
        embedding_model=EMB,
        embedding_dimensions=s.embedding_dimensions,
        embedding_profile=s.embedding_profile,
        timeout=s.provider_timeout_seconds,
    )


def chat_json(p: LMStudioProvider) -> str:
    return p.chat_json(
        model=CHAT,
        system="s",
        user="Set ready to true.",
        schema_name="capability_probe",
        schema=SCHEMA,
        max_tokens=64,
    )


def test_all_probes_pass_over_real_http():
    with running_sim(dims=64) as sim:
        s = settings()
        results = run_probes(provider(sim.base_url, s), s)
        assert [r.name for r in results] == [
            "models",
            "chat_bilingual",
            "structured_output",
            "embeddings",
            "retrieval_fixture",
        ]
        assert all(r.ok for r in results), [(r.name, r.code, r.detail) for r in results]


def test_missing_configured_model_is_reported_by_id():
    with running_sim() as sim:
        s = settings(extraction_model="not-served")
        res = run_probes(provider(sim.base_url, s), s)
        assert res[0].code == "MODEL_MISSING" and "not-served" in res[0].detail


def test_wrong_token_is_auth_rejected_not_generic():
    with running_sim(token="right") as sim:
        s = settings()
        with pytest.raises(ProviderBadResponse) as exc:
            provider(sim.base_url, s, token="wrong").list_models()
        assert exc.value.code == "PROVIDER_AUTH_REJECTED"
        assert run_probes(provider(sim.base_url, s, token="right"), s)[0].ok


def test_no_authorization_header_when_no_token_is_configured():
    with running_sim(token=None) as sim:
        s = settings()
        assert provider(sim.base_url, s).list_models()  # loopback without auth works unchanged


def test_slow_reply_past_the_timeout_is_an_ambiguous_timeout():
    with running_sim() as sim:
        control(sim.base_url, chat="slow", chat_delay=2.0)
        p = provider(sim.base_url, settings(provider_timeout_seconds=0.3))
        with pytest.raises(ProviderUnavailable) as exc:
            chat_json(p)
        assert exc.value.code == "PROVIDER_TIMEOUT"


@pytest.mark.parametrize(
    "mode, code",
    [
        ("truncated", "OUTPUT_TRUNCATED"),
        ("malformed", "NOT_JSON"),
        ("unauthorized", "PROVIDER_AUTH_REJECTED"),
    ],
)
def test_bad_chat_outputs_are_classified(mode, code):
    with running_sim() as sim:
        control(sim.base_url, chat=mode)
        with pytest.raises(ProviderBadResponse) as exc:
            chat_json(provider(sim.base_url, settings()))
        assert exc.value.code == code


def test_server_error_is_retryable_unavailability():
    with running_sim() as sim:
        control(sim.base_url, chat="error")
        with pytest.raises(ProviderUnavailable) as exc:
            chat_json(provider(sim.base_url, settings()))
        assert exc.value.retryable is True


def test_wrong_embedding_dimension_is_rejected_on_the_wire():
    with running_sim(dims=64) as sim:
        control(sim.base_url, embed="bad_dims")
        with pytest.raises(ProviderBadResponse) as exc:
            provider(sim.base_url, settings()).embed(["hello"])
        assert exc.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_batched_embeddings_keep_order_and_unit_norm():
    with running_sim(dims=64) as sim:
        vecs = provider(sim.base_url, settings()).embed([f"note {i}" for i in range(19)])
        assert len(vecs) == 19 and all(len(v) == 64 for v in vecs)
        assert all(abs(sum(x * x for x in v) - 1.0) < 1e-9 for v in vecs)


def test_gate_serialises_concurrent_calls_on_the_wire():
    store = MemoryStore()
    with running_sim(dims=64) as sim:
        s = settings(provider_timeout_seconds=5)
        p = provider(sim.base_url, s)
        control(sim.base_url, embed="slow", embed_delay=0.4)
        gate = InferenceGate(
            store,
            "ep-wire",
            lease_seconds=10,
            wait_seconds=10,
            probe=lambda: p.chat_text(model=CHAT, system="s", user="ok", max_tokens=4),
            poll_seconds=0.02,
        )
        emb = GatedEmbeddings(p, gate, "worker-a")
        errors: list[BaseException] = []

        def go() -> None:
            try:
                emb.embed(["same time"])
            except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert sim.app.state.sim.max_inflight == 1, "the endpoint saw overlapping requests"
        assert store.gate_status("ep-wire").state == GATE_IDLE


def test_timeout_leaves_slot_recovering_and_next_call_probes_first():
    store = MemoryStore()
    with running_sim() as sim:
        s = settings(provider_timeout_seconds=0.3)
        p = provider(sim.base_url, s)
        gate = InferenceGate(
            store,
            "ep-rec",
            lease_seconds=10,
            wait_seconds=5,
            probe=lambda: p.chat_text(model=CHAT, system="s", user="ok", max_tokens=4),
            poll_seconds=0.02,
        )
        chat = GatedChat(p, gate, "api")
        control(sim.base_url, chat="slow", chat_delay=1.5)
        with pytest.raises(ProviderUnavailable):
            chat.chat_json(
                model=CHAT,
                system="s",
                user="Set ready to true.",
                schema_name="capability_probe",
                schema=SCHEMA,
                max_tokens=64,
            )
        assert store.gate_status("ep-rec").state == GATE_RECOVERING
        control(sim.base_url, chat="ok", chat_delay=0.0)
        time.sleep(0.2)
        out = chat.chat_json(
            model=CHAT,
            system="s",
            user="Set ready to true.",
            schema_name="capability_probe",
            schema=SCHEMA,
            max_tokens=64,
        )
        assert '"ready"' in out
        assert store.gate_status("ep-rec").state == GATE_IDLE


def test_probe_fails_while_server_still_down_and_slot_stays_recovering():
    store = MemoryStore()
    with running_sim() as sim:
        s = settings(provider_timeout_seconds=0.3)
        p = provider(sim.base_url, s)
        gate = InferenceGate(
            store,
            "ep-down",
            lease_seconds=10,
            wait_seconds=0.2,
            probe=lambda: p.chat_text(model=CHAT, system="s", user="ok", max_tokens=4),
            poll_seconds=0.02,
        )
        store.ensure_gate("ep-down")
        store.gate_try_acquire("ep-down", "api", "t1", datetime.now(UTC), 10)
        store.gate_release("ep-down", "t1", GATE_RECOVERING)
        control(sim.base_url, chat="unavailable")
        from notes_graphrag.core.errors import InferenceBusy

        with pytest.raises(InferenceBusy):
            with gate.slot("api"):
                pass
        assert store.gate_status("ep-down").state == GATE_RECOVERING


def test_recovering_slot_clears_without_any_request_once_the_server_is_back():
    store = MemoryStore()
    with running_sim() as sim:
        s = settings(provider_timeout_seconds=0.3)
        p = provider(sim.base_url, s)
        gate = InferenceGate(
            store,
            "ep-idle",
            lease_seconds=10,
            wait_seconds=1,
            probe=lambda: p.chat_text(model=CHAT, system="s", user="ok", max_tokens=4),
            poll_seconds=0.02,
        )
        store.ensure_gate("ep-idle")
        store.gate_try_acquire("ep-idle", "api", "t0", datetime.now(UTC), 10)
        store.gate_release("ep-idle", "t0", GATE_RECOVERING)
        assert gate.try_recover() is True  # the server answers, so the worker clears it itself
        assert store.gate_status("ep-idle").state == GATE_IDLE


def test_diagnostic_probes_cannot_bypass_the_shared_slot():
    from notes_graphrag.providers.gate import GatedLM

    store = MemoryStore()
    with running_sim(dims=64) as sim:
        s = settings(provider_timeout_seconds=5)
        p = provider(sim.base_url, s)
        control(sim.base_url, chat="slow", chat_delay=0.3)
        gate = InferenceGate(
            store,
            "ep-probe",
            lease_seconds=10,
            wait_seconds=10,
            probe=lambda: p.chat_text(model=CHAT, system="s", user="ok", max_tokens=4),
            poll_seconds=0.02,
        )
        gated = GatedLM(p, gate, "probe")
        errors: list[BaseException] = []

        def ordinary() -> None:
            try:
                GatedLM(p, gate, "api").chat_text(model=CHAT, system="s", user="ok", max_tokens=4)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=ordinary)
        t.start()
        results = run_probes(gated, s)
        t.join()
        assert not errors and all(r.ok for r in results), [(r.name, r.code) for r in results]
        assert sim.app.state.sim.max_inflight == 1
