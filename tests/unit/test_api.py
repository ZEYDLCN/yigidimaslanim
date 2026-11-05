import pytest
from fastapi.testclient import TestClient

from notes_graphrag.api.app import create_app
from notes_graphrag.api.dependencies import build_context
from notes_graphrag.core.config import Settings

NOTE = "Ada works on Project Atlas. Project Atlas uses Neo4j."


@pytest.fixture
def ctx():
    return build_context(Settings(_env_file=None), demo=True, backend="memory")


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx), raise_server_exceptions=False)


def drain(ctx):
    while ctx.worker.run_once():
        pass


def post_note(client, key="k1", text=NOTE, title="Atlas"):
    return client.post(
        "/api/v1/notes", json={"title": title, "text": text}, headers={"Idempotency-Key": key}
    )


def test_create_returns_202_and_replays_idempotently(client, ctx):
    r = post_note(client)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    assert r.headers["X-Request-ID"] == r.json()["request_id"]
    assert post_note(client).json()["job_id"] == r.json()["job_id"]
    conflict = post_note(client, text=NOTE + " more")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert conflict.json()["request_id"]


def test_missing_key_and_validation_errors(client):
    r = client.post("/api/v1/notes", json={"title": "t", "text": "x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "IDEMPOTENCY_KEY_REQUIRED"
    assert post_note(client, text="  ").status_code == 422
    assert (
        client.post("/api/v1/notes", json={}, headers={"Idempotency-Key": "a"}).status_code == 422
    )
    assert client.get("/api/v1/notes/nope").status_code == 404


def test_full_flow_ask_graph_update_delete(client, ctx):
    created = post_note(client).json()
    nid, jid = created["note_id"], created["job_id"]
    assert client.get(f"/api/v1/notes/{nid}").json()["state"] == "processing"
    assert client.get("/health/ready").status_code == 200
    drain(ctx)
    job = client.get(f"/api/v1/jobs/{jid}").json()
    assert job["status"] == "SUCCEEDED" and job["chunks_done"] == job["chunks_total"] == 1
    assert client.get(f"/api/v1/notes/{nid}").json()["state"] == "ready"

    q = client.post(
        "/api/v1/questions",
        json={"question": "Which database does Ada's project use?", "include_graph": True},
    ).json()
    assert q["status"] == "answered" and "Neo4j" in q["answer"] and q["citations"]
    c = q["citations"][0]
    src = client.get(f"/api/v1/notes/{nid}/revisions/{c['revision_id']}").json()
    assert src["text"][c["start_char"] : c["end_char"]] == c["quote"]
    assert q["graph"]["edges"]

    ents = client.get("/api/v1/entities", params={"q": "atlas"}).json()["items"]
    assert [e["name"] for e in ents] == ["Project Atlas"] and ents[0]["claim_count"] == 2
    g = client.get("/api/v1/graph", params={"entity_id": ents[0]["entity_id"]}).json()
    assert {n["label"] for n in g["nodes"]} == {"Ada", "Project Atlas", "Neo4j"}
    assert all(e["evidence"]["quote"] for e in g["edges"]) and not g["limited"]
    assert client.get("/api/v1/graph", params={"note_id": nid}).json()["edges"]
    assert client.get("/api/v1/graph").status_code == 400

    r = client.put(
        f"/api/v1/notes/{nid}",
        json={"title": "Atlas", "text": NOTE.replace("Neo4j", "Postgres")},
        headers={"Idempotency-Key": "k2"},
    )
    assert r.status_code == 202
    # old revision keeps serving until the new one is published
    assert (
        "Neo4j"
        in client.post("/api/v1/questions", json={"question": "Neo4j database?"}).json()["answer"]
    )
    drain(ctx)
    after = client.post(
        "/api/v1/questions", json={"question": "What does Project Atlas use?"}
    ).json()
    assert "Neo4j" not in after["answer"] and "Postgres" in after["answer"]

    assert client.delete(f"/api/v1/notes/{nid}").status_code == 204
    assert client.get(f"/api/v1/notes/{nid}").status_code == 404
    assert client.get(f"/api/v1/notes/{nid}/revisions/{c['revision_id']}").status_code == 404
    assert client.get(f"/api/v1/jobs/{jid}").status_code == 404
    gone = client.post("/api/v1/questions", json={"question": "Which database?"}).json()
    assert gone["status"] == "insufficient_evidence" and gone["citations"] == []
    replay = post_note(client)  # delayed retry of the original create cannot resurrect it
    assert replay.status_code == 409 and replay.json()["error"]["code"] == "NOTE_DELETED"


def test_pagination_stable_order_and_limits(client):
    ids = [
        post_note(client, key=f"k{i}", text=f"Note {i}.", title=f"T{i}").json()["note_id"]
        for i in range(5)
    ]
    page1 = client.get("/api/v1/notes", params={"limit": 2}).json()
    page2 = client.get("/api/v1/notes", params={"limit": 2, "cursor": page1["next_cursor"]}).json()
    page3 = client.get("/api/v1/notes", params={"limit": 2, "cursor": page2["next_cursor"]}).json()
    seen = [n["note_id"] for p in (page1, page2, page3) for n in p["items"]]
    assert seen == ids and page3["next_cursor"] is None
    assert client.get("/api/v1/notes", params={"limit": 101}).status_code == 422
    bad = client.get("/api/v1/notes", params={"cursor": "###"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_CURSOR"


def test_upload_rules(client):
    def up(name, data, key="u1"):
        return client.post(
            "/api/v1/notes/upload",
            files={"file": (name, data)},
            data={"title": "Up"},
            headers={"Idempotency-Key": key},
        )

    assert up("a.md", NOTE.encode()).status_code == 202
    assert up("a.pdf", b"x", "u2").status_code == 415
    bad = up("a.txt", b"\xff\xfe\x00", "u3")
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_ENCODING"
    big = up("b.txt", b"a" * (1_048_576 + 1), "u4")
    assert big.status_code == 413


def test_failed_job_retry_and_state(client, ctx):
    class Broken:
        def chat_json(self, **kw):
            return "{}garbage"

    good_chat = ctx.worker.chat
    ctx.worker.chat = Broken()
    r = post_note(client).json()
    drain(ctx)
    note = client.get(f"/api/v1/notes/{r['note_id']}").json()
    assert note["state"] == "failed" and note["job"]["error_code"] == "EXTRACTION_INVALID"
    ctx.worker.chat = good_chat
    assert client.post(f"/api/v1/jobs/{r['job_id']}/retry").status_code == 202
    drain(ctx)
    assert client.get(f"/api/v1/notes/{r['note_id']}").json()["state"] == "ready"
    again = client.post(f"/api/v1/jobs/{r['job_id']}/retry")  # only FAILED jobs are retryable
    assert again.status_code == 409


def test_reindex_reuses_matching_revision(client, ctx):
    r = post_note(client).json()
    drain(ctx)
    again = client.post(f"/api/v1/notes/{r['note_id']}/reindex", headers={"Idempotency-Key": "ri"})
    assert again.status_code == 202 and again.json()["job_id"] == r["job_id"]
    assert client.post(f"/api/v1/notes/{r['note_id']}/reindex").status_code == 400


def test_system_status_and_unexpected_errors(client, ctx):
    from datetime import UTC, datetime

    ctx.store.heartbeat(datetime.now(UTC))  # the worker loop does this in production
    s = client.get("/api/v1/system/status").json()
    assert s["worker"]["fresh"] and s["lm_studio"]["mode"] == "demo-fake"
    assert "password" not in str(s).lower() and "api_key" not in str(s).lower()
    assert client.get("/health/live").json() == {"status": "ok"}

    def boom(*a, **k):
        raise RuntimeError("secret internals")

    ctx.answers.ask = boom
    r = client.post("/api/v1/questions", json={"question": "x y z"})
    assert r.status_code == 500 and "secret" not in r.text
    assert r.json()["error"]["code"] == "INTERNAL_ERROR" and r.json()["request_id"]
