# Notes → Knowledge Graph (GraphRAG + Neo4j + LM Studio)

Turns personal notes (Turkish or English) into a source-backed knowledge graph. Questions are
answered from retrieved passages with citations; the graph shows how entities connect.

- Architecture: the design document (sections 1–21).
- Phases and honest status: `docs/DEVELOPMENT_PLAN.md`.
- Running and recovering the system: `docs/operations.md`.
- Decisions: `docs/decisions/`.

## Quick start (no LM Studio needed: demo providers)

```bash
export NEO4J_PASSWORD='choose-a-strong-password'   # or put it in .env (never commit .env)
scripts/build_wheelhouse.sh                        # locked dependencies + app wheel, on the host
GRAPHRAG_DEMO=1 docker compose up -d --wait neo4j
GRAPHRAG_DEMO=1 docker compose up -d migrate
GRAPHRAG_DEMO=1 docker compose up -d api worker
uv run python scripts/smoke.py --api-base http://127.0.0.1:8000
```

Demo providers are deterministic rule-based fakes. They prove the plumbing, not model quality.

## With LM Studio

Start LM Studio on the host with the chat and embedding models loaded, set the model IDs and
`LLM_BASE_URL`/`EMBEDDING_BASE_URL` in `.env`, then run `uv run python scripts/check_lmstudio.py`
before starting services. Compose containers reach the host through `host.docker.internal`. The
live path has **not** been run in this repository's development environment.

## Local development without containers

```bash
uv sync --group dev --group ui
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest                    # unit + in-memory contract tests; Neo4j tests skip without env
```

Neo4j tests (contract and integration) need a disposable database:

```bash
export NEO4J_TEST_URI=bolt://127.0.0.1:7687 NEO4J_TEST_PASSWORD=... NEO4J_TEST_WIPE=1
uv run pytest
```

`NEO4J_TEST_WIPE=1` clears that database. The tests refuse to run while a worker heartbeat is fresh.

## Layout

```
src/notes_graphrag/
  api/            FastAPI app, routes, dependency wiring
  ingestion/      validation, chunker, extraction schema, entity resolution, worker, note service
  persistence/    Store contract, memory store, Neo4j store and knowledge view, migrations
  providers/      LM Studio adapter, capability probes, inference gate, demo fakes
  retrieval/      lexical/vector candidates, RRF, bounded graph expansion, evidence packing
  answering/      structured answers, citation validation, rendering
migrations/       versioned Cypher (checksummed; applied once)
ui/               Streamlit: Notes, Ask, Graph, System (talks only to the API)
evaluation/       small labelled set and runner (demo results are not model quality)
scripts/          smoke test, LM Studio probe, backup/restore, wheelhouse, evaluation
```

## Verified

- 96 tests pass: unit tests, one contract suite run against both the memory store and Neo4j 5.26,
  Neo4j integration tests (migrations, concurrency, Turkish full-text, separate worker process,
  API end-to-end).
- The containerised stack (Neo4j, migrate, API, worker) passes the 12-check smoke test.
- Backup and restore were exercised on disposable data (`docs/operations.md`).

## Not verified here

Live LM Studio behaviour (structured output, embedding quality, latency, Turkish extraction) and
any evaluation of model quality. See the status list in `docs/DEVELOPMENT_PLAN.md`.

## Security notes

No authentication: keep the API on loopback or behind an authenticated gateway. Note text and
embeddings live in Neo4j and in backups. Logs carry IDs, timings and status codes, not note text.
