"""Streamlit AppTest against the real FastAPI app (in-process, demo providers)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

UI = Path(__file__).parents[2] / "ui"
sys.path.insert(0, str(UI))

from notes_graphrag.api.app import create_app  # noqa: E402
from notes_graphrag.api.dependencies import build_context  # noqa: E402
from notes_graphrag.core.config import Settings  # noqa: E402

NOTE = "Ada works on Project Atlas. Project Atlas uses Neo4j."


@pytest.fixture
def env(monkeypatch):
    import api_client

    ctx = build_context(Settings(_env_file=None), demo=True, run_worker=False)
    client = TestClient(create_app(ctx))
    monkeypatch.setattr(api_client, "CLIENT_FACTORY", lambda: client)
    return ctx


def drain(ctx):
    while ctx.worker.run_once():
        pass


def run():
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_highlight_uses_code_points_and_escapes_markup():
    from textutil import highlight, md_escape

    text = "😀 Ada [x](javascript:alert(1)) <script>"
    start = text.index("Ada")
    out = highlight(text, start, start + 3)
    assert ":orange-background[Ada]" in out
    assert "<script>" not in out and "\\<script\\>" in out and "\\[x\\]\\(javascript" in out
    assert md_escape("a*b") == "a\\*b"


def test_add_note_then_ask_flow(env):
    at = run()
    at.text_input[0].set_value("Atlas")
    at.text_area[0].set_value(NOTE)
    at.button[0].click().run()  # form submit
    assert not at.exception and any("Queued" in s.value for s in at.success)
    assert len(env.store.notes) == 1

    drain(env)
    at.run()
    assert any("ready" in e.label for e in at.expander)

    q = next(t for t in at.text_input if t.label == "Question")
    q.set_value("Which database does Ada's project use?")
    ask = next(b for b in at.button if b.label == "Ask")
    ask.click().run()
    assert not at.exception, at.exception
    assert any("Neo4j" in m.value for m in at.success)
    assert any("Atlas" in m.value or "Neo4j" in m.value for m in at.markdown)


def test_unanswerable_shows_warning_and_api_down_is_reported(env, monkeypatch):
    at = run()
    q = next(t for t in at.text_input if t.label == "Question")
    q.set_value("What is the capital of France?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert any("not contain enough evidence" in w.value for w in at.warning)

    import api_client

    def down():
        import httpx

        def boom(request):
            raise httpx.ConnectError("refused")

        return httpx.Client(base_url="http://x", transport=httpx.MockTransport(boom))

    monkeypatch.setattr(api_client, "CLIENT_FACTORY", down)
    at = run()
    assert any("API_UNREACHABLE" in e.value for e in at.error)
