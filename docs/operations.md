# Operations

## Services

| Service | Role | Started by |
| --- | --- | --- |
| Neo4j 5.26 Community | Authoritative store, vector and full-text indexes | `compose.yaml` (`neo4j`) |
| migrate | Applies Cypher migrations, then exits | `compose.yaml` (`migrate`) |
| api | FastAPI on port 8000 (loopback by default) | `compose.yaml` (`api`) |
| worker | Ingestion loop: leases jobs, extracts, embeds, publishes | `compose.yaml` (`worker`) |
| LM Studio | Chat and embedding models | Host, not Compose |

Run exactly one worker in the MVP. Lease fencing keeps extra workers safe, but scaling is not tested
for throughput.

## First start

```bash
cp .env.example .env            # set NEO4J_PASSWORD; keep .env out of git
scripts/build_wheelhouse.sh     # builds deps and the app wheel on the host (see below)
docker compose up -d --wait neo4j
docker compose up -d migrate     # exits 0 when migrations are applied
docker compose up -d api worker
curl --fail http://127.0.0.1:8000/health/ready
```

`scripts/build_wheelhouse.sh` exists because the image is built offline from the locked
dependencies. The build container does not trust the sandbox TLS interception CA, so fetching
packages inside the image failed. Building the wheelhouse on the host avoids that and makes the image
build reproducible. Run the script again after any dependency or source change.

Set `GRAPHRAG_DEMO=1` to run every service with the deterministic demo providers, for example to
test the stack without LM Studio. Demo mode uses 64-dimensional embeddings, so run it against a
database that was migrated in demo mode. The migration guard refuses a mismatch.

## Schema and the vector dimension guard

- `notes-graphrag migrate` applies versioned migrations once and records a checksum for each.
- An applied migration that has been edited is refused.
- An existing `chunk_embedding_v1` index whose dimension differs from `EMBEDDING_DIMENSIONS` is
  refused. This was checked live: a demo database (64) against the production setting (1024).
  Changing the embedding model or profile requires a full reindex, not an index reuse.
- `api` refuses to start on a schema version that does not match the code.

## Backup and restore (tested)

Neo4j Community has no online dump, so backup stops the database container briefly.

```bash
scripts/neo4j_backup.sh gr-neo4j ./backups
scripts/neo4j_restore.sh <target-container> ./backups
```

Tested on 2026-11-05 with disposable data:

- A marker node was written, then the database dumped (`neo4j.dump`, about 50 KB, 15 s including
  restart).
- A fresh, empty `neo4j:5.26-community` container was restored from that dump in about 16 s.
- After restore: the marker node was present, schema versions were `[1, 2]`, 21 indexes were
  `ONLINE`, the vector index was 64-dimensional, and the API reported `ready` against the restored
  database.

Backups contain note text and embeddings. Store them with the same care as the database.

Recovery time objective is not defined; the measured restore takes under a minute for this data
size. Backup age is not monitored.

## Inference slot and recovery

All LM Studio calls (chat and embedding, from API and worker) pass through one database-backed slot
per endpoint (`InferenceGate`):

- A caller acquires the slot, and a heartbeat keeps the lease alive while the call runs.
- A timeout (`PROVIDER_TIMEOUT`) marks the slot `RECOVERING`. Nothing else is admitted until a small
  probe call succeeds.
- A holder that disappears (crashed process) leaves an expired lease. The next caller moves the slot to
  `RECOVERING` and the probe path runs. A crashed holder is therefore never silently overlapped with a
  new request.
- A busy slot returns `503 INFERENCE_BUSY` to interactive callers after `INFERENCE_SLOT_WAIT_SECONDS`.

Check the slot with `GET /api/v1/system/status` (`inference_slot.state`).

Limit: the probe is a timed heuristic. It proves the endpoint answers, not that the previous GPU job
has actually stopped. Only the live LM Studio test can show that, and it was not run here.

## Failure behaviour (verified in tests)

| Failure | Behaviour | Test |
| --- | --- | --- |
| Worker crashes mid-job | Lease expires; next acquire reclaims; attempt counter increments; the old token cannot write | `test_expired_lease_is_reclaimed_and_cannot_write` (both stores), `test_api_on_neo4j_end_to_end_with_stopped_worker_recovery` |
| Two workers race for one job | Exactly one acquires | `test_concurrent_acquire_gives_one_owner_per_job` |
| Two identical submissions race | One job; the loser retries and replays | `test_concurrent_identical_submissions_create_exactly_one_job` |
| Transient provider error | Requeued with backoff, then failed after `JOB_MAX_ATTEMPTS` | `test_transient_failure_requeues_then_fails` |
| Note deleted during processing | Publication refused; job cancelled | `test_delete_during_processing_blocks_publication` |
| Previous revision on failed update | Still served | `test_failed_update_keeps_previous_revision` |
| Separate worker process | Processes an API submission and heartbeats | `test_worker_in_separate_process_processes_api_submission` |
| Database unavailable | 503 `DEPENDENCY_UNAVAILABLE`, retryable | error handler in `api/app.py` (not separately tested) |

## Test database safety

Destructive Neo4j tests clear the database they point at, so they require
`NEO4J_TEST_WIPE=1`. They also refuse to run while a worker heartbeat is fresh (under 120 seconds).
Stop the running services first. CI uses its own throwaway service container.

## Readiness is backed by evidence

`/health/ready` returns 200 only when the database answers, the schema matches the code, the
required indexes are online, and, in production mode, a capability probe passed recently for
this exact configuration. The probe records its result and a fingerprint of the models,
dimension, profile, context length and query template. A changed setting makes an earlier result
inapplicable, and so does an older result than `CAPABILITY_PROBE_MAX_AGE_HOURS` (default 24).

```bash
notes-graphrag probe --record     # runs the probes through the shared inference slot and stores the result
```

Demo mode cannot be probed and is reported as exempt. Readiness in demo mode says so explicitly.

## Retention

- Deleted notes keep a tombstone. Their revisions, chunks, claims and jobs are removed after
  `DELETED_CONTENT_RETENTION_DAYS` (default 1). Entities left without claims are removed too.
- Idempotency records are kept for `IDEMPOTENCY_RETENTION_DAYS` (default 30). Within that period a
  replayed request for a deleted note still answers `409 NOTE_DELETED` and never creates a new
  note. After it, the same key is a new operation.
- The worker runs this maintenance about every `MAINTENANCE_INTERVAL_SECONDS` (default 3600). It is
  idempotent. Counts are logged when anything was removed.

## Request limits

- Bodies larger than `MAX_REQUEST_BYTES` (default 2 MiB) get `413 PAYLOAD_TOO_LARGE`, whether or not
  the client sent a Content-Length. Upload limits (1 MiB of file content) apply on top of that.
- Notes may not contain control characters except tab, line feed and carriage return
  (`422 CONTROL_CHARACTERS`). NUL is never accepted.
- Every failure, including unknown routes, wrong methods and malformed JSON, uses the same error
  envelope with a `request_id`.

## Known limits (be explicit with users)

- Persistence is Neo4j; the in-memory store is for demos and tests only and loses data on exit.
- No authentication. Keep the API on loopback or behind an authenticated gateway.
- Idempotency records are kept indefinitely. The 30-day retention in architecture 7.2 is not
  implemented.
- Full-text search does not stem (see decision 0001).
- The relevance gate is permissive: one shared common word can make unrelated passages count as
  evidence (evaluation q16). Tightening it needs a held-out set, not tuning on this one.
- Entity resolution is conservative and simple: it has no manual merge or split.
- Live LM Studio, model quality, and the 16 GiB/GPU latency targets were not measured here.
