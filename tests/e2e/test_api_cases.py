"""Case matrix over the HTTP API. Every case runs against the memory and Neo4j backends."""

from __future__ import annotations

import pytest

TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."
H = {"Idempotency-Key": "case-key"}


def err(r) -> str:
    return r.json()["error"]["code"]


def envelope_ok(r) -> None:
    body = r.json()
    assert body["request_id"] and body["error"]["message"] and "retryable" in body["error"]


# ---- routing and transport --------------------------------------------------------------------
def test_unknown_route_and_wrong_method_use_the_envelope(stack):
    r = stack.client.get("/no/such/route")
    assert r.status_code == 404 and err(r) == "NOT_FOUND"
    envelope_ok(r)
    r = stack.client.patch("/api/v1/notes")
    assert r.status_code == 405 and err(r) == "METHOD_NOT_ALLOWED"
    envelope_ok(r)


def test_malformed_json_and_wrong_types(stack):
    r = stack.client.post(
        "/api/v1/notes", content=b"{bad", headers={**H, "content-type": "application/json"}
    )
    assert r.status_code == 400 and err(r) == "BAD_JSON"
    r = stack.client.post("/api/v1/notes", json={"title": 5, "text": "x"}, headers=H)
    assert r.status_code == 422 and err(r) == "VALIDATION_FAILED"
    envelope_ok(r)


def test_body_over_limit_is_413_with_and_without_content_length(stack):
    big = b'{"title":"t","text":"' + b"a" * 2_200_000 + b'"}'
    r = stack.client.post(
        "/api/v1/notes", content=big, headers={**H, "content-type": "application/json"}
    )
    assert r.status_code == 413 and err(r) == "PAYLOAD_TOO_LARGE"
    envelope_ok(r)

    def chunks():
        yield big[:1_000_000]
        yield big[1_000_000:]

    r = stack.client.post(
        "/api/v1/notes", content=chunks(), headers={**H, "content-type": "application/json"}
    )
    assert r.status_code == 413, "a chunked body without Content-Length is limited too"


# ---- idempotency keys -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key, code",
    [
        (None, "IDEMPOTENCY_KEY_REQUIRED"),
        ("   ", "IDEMPOTENCY_KEY_REQUIRED"),
        ("k" * 201, "IDEMPOTENCY_KEY_INVALID"),
        ("bad\x01key", "IDEMPOTENCY_KEY_INVALID"),
    ],
)
def test_idempotency_key_validation(stack, key, code):
    headers = {} if key is None else {"Idempotency-Key": key}
    r = stack.client.post("/api/v1/notes", json={"title": "t", "text": "x"}, headers=headers)
    assert r.status_code == 400 and err(r) == code


def test_same_key_on_different_route_is_a_different_operation(stack):
    created = stack.create("A", TEXT, "shared").json()
    r = stack.client.post(
        f"/api/v1/notes/{created['note_id']}/reindex", headers={"Idempotency-Key": "shared"}
    )
    assert (
        r.status_code == 202 and r.json()["replayed"] is True
    )  # same content: reused, not duplicated


def test_same_key_different_payload_conflicts(stack):
    stack.create("A", TEXT, "dup")
    r = stack.create("A", TEXT + " More.", "dup")
    assert r.status_code == 409 and err(r) == "IDEMPOTENCY_KEY_REUSED"


# ---- input content ----------------------------------------------------------------------------
def test_text_length_boundaries(stack):
    assert stack.create("t", "a" * 100_000, "len-ok").status_code == 202
    r = stack.create("t", "a" * 100_001, "len-over")
    assert r.status_code == 413 and err(r) == "PAYLOAD_TOO_LARGE"


def test_title_length_boundaries_and_blank_title(stack):
    assert stack.create("a" * 200, "x y", "title-ok").status_code == 202
    assert err(stack.create("a" * 201, "x y", "title-long")) == "TITLE_TOO_LONG"
    assert err(stack.create("   ", "x y", "title-blank")) == "EMPTY_TITLE"


@pytest.mark.parametrize(
    "text, code",
    [
        ("", "EMPTY_NOTE"),
        ("   \n\t  ", "EMPTY_NOTE"),
        ("﻿", "EMPTY_NOTE"),
        ("a\x00b", "CONTROL_CHARACTERS"),
        ("a\x1bb", "CONTROL_CHARACTERS"),
        ("a\x85b", "CONTROL_CHARACTERS"),
    ],
)
def test_rejected_note_content(stack, text, code):
    r = stack.create("t", text, f"bad-{len(text)}-{code}")
    assert r.status_code == 422 and err(r) == code


def test_accepted_unicode_content_keeps_exact_offsets(stack):
    text = (
        "😀 Ada works on Project Atlas. 🚀 Project Atlas uses Neo4j.\n"
        "İstanbul ofisi çalışıyor.\r\nGöz."
    )
    created = stack.create("Unicode", text, "uni").json()
    stack.drain()
    rev = stack.client.get(
        f"/api/v1/notes/{created['note_id']}/revisions/{created['revision_id']}"
    ).json()
    assert rev["text"] == text.replace("\r\n", "\n")  # line endings normalised once
    q = stack.ask("Which database does Project Atlas use?", graph=True).json()
    assert q["status"] == "answered"
    for c in q["citations"]:
        assert (
            rev["text"][c["start_char"] : c["end_char"]] == c["quote"]
            or c["note_id"] != created["note_id"]
        )


def test_markdown_with_html_is_stored_as_text_not_executed(stack):
    payload = "<script>alert('x')</script> Ada works on Project Atlas."
    created = stack.create("md", payload, "html").json()
    rev = stack.client.get(
        f"/api/v1/notes/{created['note_id']}/revisions/{created['revision_id']}"
    ).json()
    assert rev["text"] == payload  # stored verbatim; rendering is the UI's job (escaped)


# ---- upload -----------------------------------------------------------------------------------
def _upload(stack, name, data, key):
    return stack.client.post(
        "/api/v1/notes/upload",
        files={"file": (name, data)},
        data={"title": "Up"},
        headers={"Idempotency-Key": key},
    )


def test_upload_rules(stack):
    assert _upload(stack, "a.md", TEXT.encode(), "u-ok").status_code == 202
    assert _upload(stack, "a.pdf", b"%PDF-1.4", "u-pdf").status_code == 415
    assert (
        _upload(stack, "a.txt", b"\xff\xfe\x00bad", "u-enc").json()["error"]["code"]
        == "INVALID_ENCODING"
    )
    assert _upload(stack, "a.txt", b"", "u-empty").json()["error"]["code"] == "EMPTY_NOTE"
    assert _upload(stack, "a.txt", "﻿Ada works on X.".encode(), "u-bom").status_code == 202
    assert (
        _upload(stack, "../../x.txt", b"hi", "u-name").status_code == 202
    )  # name is never used as a path
    big = _upload(stack, "b.txt", b"a" * (1_048_576 + 1), "u-big")
    assert big.status_code == 413


# ---- note lifecycle as seen through HTTP -------------------------------------------------------
def test_deleted_notes_look_missing_to_new_requests(stack):
    created = stack.create("A", TEXT, "life").json()
    nid, jid = created["note_id"], created["job_id"]
    stack.drain()
    assert stack.client.delete(f"/api/v1/notes/{nid}").status_code == 204
    assert stack.client.delete(f"/api/v1/notes/{nid}").status_code == 404
    assert stack.client.get(f"/api/v1/notes/{nid}").status_code == 404
    assert (
        stack.client.put(
            f"/api/v1/notes/{nid}",
            json={"title": "t", "text": "x"},
            headers={"Idempotency-Key": "p"},
        ).status_code
        == 404
    )
    assert (
        stack.client.post(
            f"/api/v1/notes/{nid}/reindex", headers={"Idempotency-Key": "r"}
        ).status_code
        == 404
    )
    assert stack.client.get(f"/api/v1/jobs/{jid}").status_code == 404
    assert stack.client.post(f"/api/v1/jobs/{jid}/retry").status_code == 404
    assert stack.client.get("/api/v1/graph", params={"note_id": nid}).status_code == 404


def test_replaying_the_original_create_after_delete_is_conflict(stack):
    created = stack.create("A", TEXT, "orig").json()
    stack.client.delete(f"/api/v1/notes/{created['note_id']}")
    r = stack.create("A", TEXT, "orig")
    assert r.status_code == 409 and err(r) == "NOTE_DELETED"


def test_unknown_or_tampered_ids_are_404_not_500(stack):
    for path in [
        "/api/v1/notes/x'%20OR%201=1",
        "/api/v1/notes/../../etc",
        "/api/v1/jobs/%00",
        "/api/v1/notes/nope/revisions/nope",
    ]:
        r = stack.client.get(path)
        assert r.status_code == 404, path
    assert stack.client.post("/api/v1/jobs/nope/retry").status_code == 404


def test_retry_rules_reject_non_failed_jobs(stack):
    created = stack.create("A", TEXT, "retry-rules").json()
    r = stack.client.post(f"/api/v1/jobs/{created['job_id']}/retry")
    assert r.status_code == 409 and err(r) == "JOB_NOT_RETRYABLE"
    stack.drain()
    r = stack.client.post(f"/api/v1/jobs/{created['job_id']}/retry")
    assert r.status_code == 409 and err(r) == "JOB_NOT_RETRYABLE"


def test_revision_must_belong_to_the_note_in_the_path(stack):
    a = stack.create("A", TEXT, "rev-a").json()
    b = stack.create("B", "Grace works on Project Zed.", "rev-b").json()
    stack.drain()
    r = stack.client.get(f"/api/v1/notes/{b['note_id']}/revisions/{a['revision_id']}")
    assert r.status_code == 404


def test_list_pagination_has_no_duplicates_or_gaps(stack):
    ids = [
        stack.create(f"T{i}", f"Note number {i} about things.", f"pg{i}").json()["note_id"]
        for i in range(7)
    ]
    seen, cursor = [], None
    for _ in range(10):
        params = {"limit": 3} | ({"cursor": cursor} if cursor else {})
        page = stack.client.get("/api/v1/notes", params=params).json()
        seen += [n["note_id"] for n in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(seen) == len(set(seen)) == 7 and set(seen) == set(ids)
    assert stack.client.get("/api/v1/notes", params={"limit": 101}).status_code == 422
    assert err(stack.client.get("/api/v1/notes", params={"cursor": "!!!"})) == "INVALID_CURSOR"


# ---- questions and graph ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        'Ada AND (Neo4j OR ) * ~ \\ " [ ] { } ^ : + - !',
        "what is the and of",
        "😀 🚀 ?",
        "Ada'nın projesi nerede?",
        "'; DROP TABLE notes; --",
        "*",
        "NOT AND OR",
    ],
)
def test_hostile_or_odd_questions_never_error(stack, q):
    stack.create("A", TEXT, f"q-{abs(hash(q))}")
    stack.drain()
    r = stack.ask(q)
    assert r.status_code == 200, (q, r.text[:200])
    assert r.json()["status"] in ("answered", "partial", "insufficient_evidence")


def test_question_length_boundaries(stack):
    assert stack.ask("a" * 2000).status_code == 200
    r = stack.ask("a" * 2001)
    assert r.status_code == 422 and err(r) == "QUESTION_TOO_LONG"
    assert err(stack.ask("   ")) == "EMPTY_QUESTION"


def test_unanswerable_question_is_insufficient_evidence_and_cites_nothing(stack):
    stack.create("A", TEXT, "abst")
    stack.drain()
    r = stack.ask("What is the capital of France?").json()
    assert r["status"] == "insufficient_evidence" and r["citations"] == []


def test_answer_on_empty_workspace(stack):
    r = stack.ask("Which database?").json()
    assert r["status"] == "insufficient_evidence"


def test_graph_parameter_rules(stack):
    assert err(stack.client.get("/api/v1/graph")) == "BAD_GRAPH_QUERY"
    assert (
        err(stack.client.get("/api/v1/graph", params={"entity_id": "a", "note_id": "b"}))
        == "BAD_GRAPH_QUERY"
    )
    assert (
        stack.client.get("/api/v1/graph", params={"entity_id": "a", "hops": 3}).status_code == 422
    )
    assert err(stack.client.get("/api/v1/graph", params={"entity_id": "missing"})) == "NOT_FOUND"


def test_entity_search_is_safe_and_paginated(stack):
    stack.create("A", TEXT, "ent")
    stack.drain()
    assert stack.client.get("/api/v1/entities", params={"q": "(["}).status_code == 200
    items = stack.client.get("/api/v1/entities", params={"limit": 1}).json()
    assert len(items["items"]) == 1
    names = {
        e["name"]
        for e in stack.client.get("/api/v1/entities", params={"limit": 50}).json()["items"]
    }
    assert {"Ada", "Project Atlas", "Neo4j"} <= names


def test_graph_edges_carry_exact_evidence(stack):
    stack.create("A", TEXT, "edges")
    stack.drain()
    ents = stack.client.get("/api/v1/entities", params={"q": "Atlas"}).json()["items"]
    g = stack.client.get(
        "/api/v1/graph", params={"entity_id": ents[0]["entity_id"], "hops": 2}
    ).json()
    assert {"Ada", "Project Atlas", "Neo4j"} <= {n["label"] for n in g["nodes"]}
    for e in g["edges"]:
        ev = e["evidence"]
        assert ev["quote"] and ev["end_char"] > ev["start_char"]


def test_system_status_is_json_and_reports_the_slot(stack):
    from datetime import UTC, datetime

    stack.ctx.store.heartbeat(datetime.now(UTC))
    stack.ctx.store.gate_try_acquire(
        stack.ctx.settings.lm_studio_instance_id, "api", "t-status", datetime.now(UTC), 60
    )
    r = stack.client.get("/api/v1/system/status")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["inference_slot"]["state"] == "BUSY"
    assert (
        body["inference_slot"]["lease_expires_at"].endswith("Z")
        or "+" in body["inference_slot"]["lease_expires_at"]
    )
    assert body["worker"]["fresh"] is True
