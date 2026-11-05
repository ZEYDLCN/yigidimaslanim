"""End-to-end smoke test over real HTTP (architecture 15.4). Cleans up only what it created."""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import httpx

NOTE = "Ada works on Project Atlas. Project Atlas uses Neo4j."
QUESTION = "Which database does Ada's project use?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.api_base, timeout=args.timeout)
    run = uuid.uuid4().hex[:8]
    created: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            raise AssertionError(msg)

    def wait(job_id: str) -> dict:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            j = c.get(f"/api/v1/jobs/{job_id}").json()
            if j["status"] in ("SUCCEEDED", "FAILED", "SUPERSEDED", "CANCELLED"):
                return j
            time.sleep(0.3)
        raise AssertionError("job did not reach a terminal state in time")

    try:
        check(c.get("/health/ready").status_code == 200, "readiness")
        key = f"smoke-{run}"
        body = {"title": f"smoke {run}", "text": NOTE}
        r = c.post("/api/v1/notes", json=body, headers={"Idempotency-Key": key}).json()
        created.append(r["note_id"])
        check(wait(r["job_id"])["status"] == "SUCCEEDED", "ingestion succeeded")
        st = c.get("/api/v1/system/status").json()
        check(st["worker"]["fresh"], "fresh worker heartbeat")
        again = c.post("/api/v1/notes", json=body, headers={"Idempotency-Key": key}).json()
        check(
            again["job_id"] == r["job_id"] and again["note_id"] == r["note_id"], "idempotent replay"
        )

        q = c.post("/api/v1/questions", json={"question": QUESTION, "include_graph": True}).json()
        check(q["status"] == "answered" and "Neo4j" in q["answer"], "two-hop answer mentions Neo4j")
        cit = q["citations"][0]
        src = c.get(f"/api/v1/notes/{r['note_id']}/revisions/{cit['revision_id']}").json()
        check(
            src["text"][cit["start_char"] : cit["end_char"]] == cit["quote"],
            "citation offsets exact",
        )

        ents = c.get("/api/v1/entities", params={"q": "Atlas"}).json()["items"]
        g = c.get("/api/v1/graph", params={"entity_id": ents[0]["entity_id"]}).json()
        labels = {n["label"] for n in g["nodes"]}
        check(
            {"Ada", "Project Atlas", "Neo4j"} <= labels,
            "Ada -> Project Atlas -> Neo4j path in graph",
        )

        rep = c.put(
            f"/api/v1/notes/{r['note_id']}",
            headers={"Idempotency-Key": key + "-2"},
            json={"title": body["title"], "text": NOTE.replace("Neo4j", "Postgres")},
        ).json()
        check(wait(rep["job_id"])["status"] == "SUCCEEDED", "replacement published")
        q2 = c.post("/api/v1/questions", json={"question": "What does Project Atlas use?"}).json()
        check("Neo4j" not in q2["answer"] and "Postgres" in q2["answer"], "old evidence gone")

        check(c.delete(f"/api/v1/notes/{r['note_id']}").status_code == 204, "delete")
        created.clear()
        check(c.get(f"/api/v1/notes/{r['note_id']}").status_code == 404, "deleted note unavailable")
        q3 = c.post("/api/v1/questions", json={"question": "What does Project Atlas use?"}).json()
        check(q3["status"] == "insufficient_evidence", "no evidence after delete")
        print("SMOKE PASSED")
        return 0
    except AssertionError as exc:
        print(f"SMOKE FAILED: {exc}")
        return 1
    finally:
        for nid in created:  # cleanup only this run's artifacts
            c.delete(f"/api/v1/notes/{nid}")


if __name__ == "__main__":
    sys.exit(main())
