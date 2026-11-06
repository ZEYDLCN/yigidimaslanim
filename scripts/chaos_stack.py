"""Fault-injection run against the Compose stack (architecture 15.4 and 17 phase 8).

Prerequisites (see docs/operations.md, "Chaos run"):
  * the Compose stack is up in production mode (GRAPHRAG_DEMO=0), with neo4j, api and worker
    running;
  * the LM Studio contract simulator is running on 0.0.0.0:1234 (tests/support/lmstudio_sim.py);
  * the probes were recorded for this configuration.
The script injects faults through the simulator's control endpoint and by stopping, killing or
restarting containers, then checks what the API and the database report. It exits non-zero if any
check fails.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

API = "http://127.0.0.1:8000"
SIM = "http://127.0.0.1:1234"
TEXT = "Ada works on Project Atlas. Project Atlas uses Neo4j."
NEO4J_CONTAINER = "yigidimaslanim-neo4j-1"
WORKER = "yigidimaslanim-worker-1"
API_CONTAINER = "yigidimaslanim-api-1"


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    failures: int = 0

    def check(self, scenario: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((scenario, ("PASS" if ok else "FAIL") + " " + name, detail))
        if not ok:
            self.failures += 1


# A real crash: SIGKILL to the worker process from inside its container. `docker kill` is not used
# because Docker treats it as a manual stop, which restart policies do not recover from.
CRASH_PROGRAM = """
import os, signal
me = os.getpid()
for p in os.listdir("/proc"):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        argv = open(f"/proc/{p}/cmdline", "rb").read().split(b"\\0")
    except OSError:
        continue
    if any(a.endswith(b"notes-graphrag") for a in argv) and b"worker" in argv:
        os.kill(int(p), signal.SIGKILL)
"""


def crash_worker(container: str) -> None:
    sh("docker", "exec", container, "python", "-c", CRASH_PROGRAM)


def restart_count(container: str) -> int:
    return int(sh("docker", "inspect", container, "--format", "{{.RestartCount}}").strip())


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def sim(**body: object) -> dict:
    r = httpx.post(f"{SIM}/sim/control", json=body, timeout=5)
    r.raise_for_status()
    return r.json()


def wait_until(pred: Callable[[], object], timeout: float, step: float = 0.5) -> object:
    deadline = time.monotonic() + timeout
    last: object = None
    while time.monotonic() < deadline:
        try:
            last = pred()
            if last:
                return last
        except httpx.HTTPError:
            pass
        time.sleep(step)
    return last


def create(title: str, text: str, key: str) -> dict:
    r = httpx.post(
        f"{API}/api/v1/notes",
        json={"title": title, "text": text},
        headers={"Idempotency-Key": key},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def job(job_id: str) -> dict:
    return httpx.get(f"{API}/api/v1/jobs/{job_id}", timeout=10).json()


def slot_state() -> str:
    return httpx.get(f"{API}/api/v1/system/status", timeout=10).json()["inference_slot"]["state"]


def ready_status() -> int:
    return httpx.get(f"{API}/health/ready", timeout=10).status_code


def ask(question: str) -> httpx.Response:
    return httpx.post(f"{API}/api/v1/questions", json={"question": question}, timeout=30)


def run(report: Report) -> None:
    tag = str(int(time.time()))

    # ---- baseline: a known answerable note that later scenarios can cite -------------------
    base = create(f"base-{tag}", TEXT, f"chaos-base-{tag}")
    report.check(
        "baseline",
        "note ingested",
        wait_until(lambda: job(base["job_id"])["status"] == "SUCCEEDED", 60) is True,
    )

    # ---- S1: slow provider past the timeout, then recovery ---------------------------------
    sim(chat="slow", chat_delay=8.0)
    s1 = create(f"s1-{tag}", "Mert works on Project Zed.", f"chaos-s1-{tag}")
    saw_recovering = wait_until(lambda: slot_state() == "RECOVERING", 30)
    report.check(
        "S1 slow provider", "inference slot enters RECOVERING after a timeout", bool(saw_recovering)
    )
    sim(chat="ok", chat_delay=0.0)
    final = wait_until(lambda: job(s1["job_id"])["status"] in ("SUCCEEDED", "FAILED"), 90)
    j = job(s1["job_id"])
    report.check(
        "S1 slow provider", "job completes after recovery", j["status"] == "SUCCEEDED", str(final)
    )
    report.check(
        "S1 slow provider",
        "job was retried (attempt >= 2)",
        j["attempt"] >= 2,
        f"attempt={j['attempt']}",
    )
    report.check(
        "S1 slow provider",
        "slot returns to IDLE",
        wait_until(lambda: slot_state() == "IDLE", 30) is not False,
    )

    # ---- S2: provider unavailable: retries run out; questions get a retryable 503 ----------
    sim(chat="unavailable")
    s2 = create(f"s2-{tag}", "Lena works on Project Quartz.", f"chaos-s2-{tag}")
    failed = wait_until(lambda: job(s2["job_id"])["status"] == "FAILED", 120)
    j = job(s2["job_id"])
    report.check("S2 provider down", "job fails after its attempts", bool(failed), j["status"])
    report.check(
        "S2 provider down",
        "failure code is LM_STUDIO_UNAVAILABLE",
        j["error_code"] == "LM_STUDIO_UNAVAILABLE",
        str(j["error_code"]),
    )
    q = ask("Which database does Ada's project use?")
    report.check(
        "S2 provider down",
        "question returns 503 with the envelope",
        q.status_code == 503
        and q.json()["error"]["code"] == "LM_STUDIO_UNAVAILABLE"
        and q.json()["error"]["retryable"] is True,
        f"{q.status_code}",
    )
    sim(chat="ok")
    r = httpx.post(f"{API}/api/v1/jobs/{s2['job_id']}/retry", timeout=10)
    report.check(
        "S2 provider down",
        "retry after recovery is accepted",
        r.status_code == 202,
        str(r.status_code),
    )
    report.check(
        "S2 provider down",
        "retried job succeeds",
        wait_until(lambda: job(s2["job_id"])["status"] == "SUCCEEDED", 90) is True,
    )

    # ---- S3: rejected token: not retried ----------------------------------------------------
    sim(chat="unauthorized")
    s3 = create(f"s3-{tag}", "Omer works on Project Onyx.", f"chaos-s3-{tag}")
    wait_until(lambda: job(s3["job_id"])["status"] in ("FAILED", "SUCCEEDED"), 60)
    j = job(s3["job_id"])
    report.check(
        "S3 rejected token",
        "fails with PROVIDER_AUTH_REJECTED",
        j["status"] == "FAILED" and j["error_code"] == "PROVIDER_AUTH_REJECTED",
        str(j),
    )
    report.check(
        "S3 rejected token",
        "not retried (attempt == 1)",
        j["attempt"] == 1,
        f"attempt={j['attempt']}",
    )
    sim(chat="ok")

    # ---- S4: wrong embedding dimension from the runtime --------------------------------------
    sim(embed="bad_dims")
    s4 = create(f"s4-{tag}", "Ece works on Project Ruby.", f"chaos-s4-{tag}")
    wait_until(lambda: job(s4["job_id"])["status"] in ("FAILED", "SUCCEEDED"), 60)
    j = job(s4["job_id"])
    report.check(
        "S4 bad dimension",
        "fails with EMBEDDING_DIMENSION_MISMATCH",
        j["error_code"] == "EMBEDDING_DIMENSION_MISMATCH",
        str(j["error_code"]),
    )
    sim(embed="ok")

    # ---- S5: worker process killed while a job is in flight ---------------------------------
    sim(chat="slow", chat_delay=2.0)  # under the 3s timeout: the job stays RUNNING for a while
    s5 = create(f"s5-{tag}", "Pelin works on Project Garnet.", f"chaos-s5-{tag}")
    running = wait_until(lambda: job(s5["job_id"])["status"] == "RUNNING", 30)
    report.check("S5 worker killed", "job is RUNNING when the worker dies", bool(running))
    sim(chat="ok", chat_delay=0.0)
    restarts_before = restart_count(WORKER)
    crash_worker(WORKER)
    report.check(
        "S5 worker killed",
        "the supervisor restarted the worker container",
        wait_until(lambda: restart_count(WORKER) > restarts_before, 30) is not False,
        f"restarts {restarts_before} -> {restart_count(WORKER)}",
    )
    j_dead = job(s5["job_id"])
    report.check(
        "S5 worker killed",
        "job is left leased (RUNNING) after the crash",
        j_dead["status"] == "RUNNING",
        j_dead["status"],
    )
    done = wait_until(lambda: job(s5["job_id"])["status"] == "SUCCEEDED", 120)
    j = job(s5["job_id"])
    report.check(
        "S5 worker killed",
        "lease expires and the restarted worker finishes the job",
        bool(done),
        f"status={j['status']} attempt={j['attempt']}",
    )
    report.check(
        "S5 worker killed",
        "job was reclaimed (attempt >= 2)",
        j["attempt"] >= 2,
        f"attempt={j['attempt']}",
    )
    report.check(
        "S5 worker killed",
        "inference slot recovered after the crashed holder",
        wait_until(lambda: slot_state() == "IDLE", 60) is not False,
    )

    # ---- S6: database restart during traffic -------------------------------------------------
    stop_at = time.monotonic() + 45
    statuses: list[int] = []
    bodies: list[str] = []

    def hammer() -> None:
        while time.monotonic() < stop_at:
            try:
                r = httpx.post(
                    f"{API}/api/v1/questions", json={"question": "Which database?"}, timeout=10
                )
                statuses.append(r.status_code)
                if r.status_code >= 400:
                    bodies.append(r.text[:160])
            except httpx.HTTPError:
                statuses.append(0)
            time.sleep(0.2)

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(hammer) for _ in range(3)]
        time.sleep(1.5)
        sh("docker", "restart", NEO4J_CONTAINER)
        for f in futs:
            f.result()
    five_xx = [s for s in statuses if s >= 500 and s != 503]
    report.check(
        "S6 database restart",
        "no 500 responses during the restart",
        not five_xx,
        str(sorted(set(statuses))),
    )
    report.check(
        "S6 database restart",
        "503 envelopes were returned while the database was down",
        any(s == 503 for s in statuses) or any(s == 0 for s in statuses),
        f"distinct={sorted(set(statuses))}",
    )
    report.check(
        "S6 database restart",
        "readiness returns to 200 after the restart",
        wait_until(lambda: ready_status() == 200, 60) is not False,
    )
    after = create(f"s6-{tag}", "Mina works on Project Lapis.", f"chaos-s6-{tag}")
    report.check(
        "S6 database restart",
        "new ingestion works after the restart",
        wait_until(lambda: job(after["job_id"])["status"] == "SUCCEEDED", 90) is True,
    )

    # ---- S7: API restart keeps state ---------------------------------------------------------
    sh("docker", "restart", API_CONTAINER)
    report.check(
        "S7 api restart",
        "readiness after restart",
        wait_until(lambda: ready_status() == 200, 60) is not False,
    )
    report.check(
        "S7 api restart",
        "earlier note still answers questions",
        ask("Which database does Project Atlas use?").json().get("status") == "answered",
    )

    # ---- S8: concurrent burst ------------------------------------------------------------------
    def one(i: int) -> int:
        if i % 3 == 0:
            return httpx.post(
                f"{API}/api/v1/notes",
                json={"title": f"burst-{i}", "text": f"Note {i} about Atlas."},
                headers={"Idempotency-Key": f"chaos-burst-{tag}-{i}"},
                timeout=20,
            ).status_code
        return ask("Which database does Project Atlas use?").status_code

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        codes = list(ex.map(one, range(60)))
    report.check(
        "S8 burst",
        "no 5xx under 60 concurrent requests",
        all(c < 500 for c in codes),
        str(sorted(set(codes))),
    )

    # ---- S9: oversized body over the wire ----------------------------------------------------
    big = b'{"title":"t","text":"' + b"a" * 3_000_000 + b'"}'
    r = httpx.post(
        f"{API}/api/v1/notes",
        content=big,
        headers={"Idempotency-Key": "chaos-big", "content-type": "application/json"},
        timeout=30,
    )
    report.check(
        "S9 oversized body",
        "413 with the envelope",
        r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE",
        str(r.status_code),
    )

    # ---- S10: content never reaches the logs ---------------------------------------------------
    marker = f"SECRET-MARKER-{tag}"
    m = create(f"s10-{tag}", f"{marker} Ada works on Project Atlas.", f"chaos-s10-{tag}")
    wait_until(lambda: job(m["job_id"])["status"] == "SUCCEEDED", 60)
    ask(f"Which database does {marker} Ada's project use?")
    logs = sh("docker", "logs", WORKER) + sh("docker", "logs", API_CONTAINER)
    report.check(
        "S10 logs", "note and question text never appears in container logs", marker not in logs
    )


def main() -> int:
    report = Report()
    started = time.monotonic()
    try:
        run(report)
    except Exception as exc:  # a broken harness is a failure too
        report.check("harness", f"run completed ({type(exc).__name__}: {exc})", False)
    width = max(len(s) for s, _, _ in report.rows) if report.rows else 10
    for scenario, check, detail in report.rows:
        print(
            f"{scenario:<{width}}  {check}"
            + (f"   [{detail}]" if detail and check.startswith("FAIL") else "")
        )
    total = len(report.rows)
    print(f"\n{total - report.failures}/{total} checks passed in {time.monotonic() - started:.0f}s")
    print(json.dumps({"failures": report.failures, "checks": total}))
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
