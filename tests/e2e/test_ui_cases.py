"""Streamlit UI flows against the real API (in-process, demo providers, memory backend)."""

from __future__ import annotations

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

TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."


@pytest.fixture
def env(monkeypatch):
    import api_client

    ctx = build_context(Settings(_env_file=None), demo=True, backend="memory")
    client = TestClient(create_app(ctx), raise_server_exceptions=False)
    monkeypatch.setattr(api_client, "CLIENT_FACTORY", lambda: client)
    return ctx


def drain(ctx) -> None:
    while ctx.worker.run_once():
        pass


def fresh_app() -> AppTest:
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def note_ids(ctx) -> list[str]:
    return [n.note_id for n in ctx.store.list_notes(None, 100)[0]]


def test_hostile_title_is_escaped_everywhere_it_is_shown(env):
    env.notes.create("[click](javascript:alert(1)) <b>bold</b>", TEXT, "hostile")
    drain(env)
    at = fresh_app()
    label = at.expander[0].label
    assert "[click]" not in label  # the raw markdown link would render as a clickable link
    assert "\\[click\\]" in label


def test_corrected_submission_after_a_validation_error_is_not_refused_as_reused(env):
    at = fresh_app()
    at.text_input[0].set_value("Atlas")
    at.text_area[0].set_value("   ")  # rejected by the API: EMPTY_NOTE
    at.button[0].click().run()
    assert any("note is empty" in e.value or "EMPTY_NOTE" in e.value for e in at.error)
    at.text_area[0].set_value(TEXT)  # corrected content, same form
    at.button[0].click().run()
    assert not at.error, [e.value for e in at.error]
    assert len(note_ids(env)) == 1


def test_replace_then_delete_through_the_ui(env):
    created = env.notes.create("Atlas", TEXT, "ui-life")
    drain(env)
    at = fresh_app()
    nid = created.note.note_id
    at.text_area(key=f"txt-{nid}").set_value("Project Atlas uses Postgres.")
    at.button(key=f"rep-{nid}").click().run()
    assert not at.error, [e.value for e in at.error]
    drain(env)
    at.run()
    at.button(key=f"del-{nid}").click().run()
    assert not at.error, [e.value for e in at.error]
    assert env.store.get_note(nid).deleted_at is not None
    assert note_ids(env) == []


def test_failed_job_offers_retry_and_recovers(env):
    class Broken:
        def chat_json(self, **kw):
            return "{}garbage"

    good = env.worker.chat
    env.worker.chat = Broken()
    created = env.notes.create("Broken", TEXT, "ui-retry")
    drain(env)
    env.worker.chat = good
    at = fresh_app()
    assert any("Indexing failed" in w.value for w in at.warning)
    at.button(key=f"retry-{created.note.note_id}").click().run()
    assert not at.error, [e.value for e in at.error]
    drain(env)
    at.run()
    assert any("ready" in e.label for e in at.expander)


def test_unanswerable_question_is_shown_as_a_warning_not_an_answer(env):
    env.notes.create("Atlas", TEXT, "ui-ask")
    drain(env)
    at = fresh_app()
    q = next(t for t in at.text_input if t.label == "Question")
    q.set_value("What is the capital of France?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert any("not contain enough evidence" in w.value for w in at.warning)
    assert not at.success
