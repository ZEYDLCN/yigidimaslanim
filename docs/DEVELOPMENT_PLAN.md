# Development plan and status

Source: the architecture document (Notes → Knowledge Graph). Phases follow its section 17.

## Status

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 0 Bootstrap | Package, lock, config, Compose, CI | Done | `uv.lock`, `compose.yaml`, CI workflow |
| 1 Persistence and notes | Store contract, Neo4j repository, idempotency, leases | Done (Neo4j 5.26.32) | 23 contract tests, each run on both stores (46 cases) |
| 2 Text ingestion | Chunker, validation, staged publication | Done | chunker and publication tests; case matrix in `tests/e2e` |
| 3 Real extraction | LM Studio adapter, schema validation, entity resolution | Adapter verified over HTTP against the contract simulator; live model not run | `tests/integration/test_lmstudio_http.py` (16 tests) |
| 4 Search | Vector and full-text indexes, active filter, RRF | Done (Neo4j indexes) | scoped candidate tests; Turkish case folding test |
| 5 Graph retrieval | Entity linking, bounded expansion, evidence packing | Done | expansion limit tests; 2-hop contract test |
| 6 Answers | Structured answers, citation validation, abstention, graph version | Done with demo model | answer and citation tests |
| 7 Interface | FastAPI, Streamlit (Notes, Ask, Graph, System) | Done | API case matrix (both stores); Streamlit flow tests |
| 8 Reliability | Inference gate, lease fencing, restart recovery, retention, backup | Done; live LM Studio recovery not run | chaos run 28/28; mutation checks; measured backup and restore |
| 9 Evaluation | Dataset, baselines, live smoke, operations guide | Dataset and harness done; live run not possible here | `evaluation/`, `scripts/evaluate.py`, `docs/operations.md` |

## Verification summary (2026-11-06)

- **Test suite:** 215 tests pass against Neo4j 5.26.32 Community: unit, one contract suite run on
  both stores, the HTTP case matrix on both stores, Streamlit flows, the LM Studio contract over real
  sockets, Neo4j integration tests (migrations, concurrency, Turkish full text, separate worker
  process, API end to end).
- **Chaos run against the Compose stack** (`scripts/chaos_stack.py`, production mode, simulator as
  the provider): 28 of 28 checks pass. Covers slow provider and slot recovery, provider down,
  rejected token, wrong dimension, worker crash with lease reclaim, database restart during traffic,
  API restart, a 60-request burst, oversized body, and no note text in logs.
- **Mutation checks:** five deliberately broken rules (lease fencing, deleted-note visibility, body
  limit, control characters, purge and replay protection) and the gate's admission control. Each is
  caught by at least one test. Restoring the code makes the tests pass again.
- **Backup and restore:** re-verified on the final schema (migrations 1 to 4).
- **Evaluation:** demo providers only. Results validate plumbing, not model quality.

## Definition of done (architecture 17.2), honest status

- Installs from the committed lockfile: yes (`uv sync --frozen`).
- Schema setup repeatable and checks index compatibility: yes (migrations with checksums and a
  dimension guard).
- LM Studio serves the configured models from the actual caller: **not verified** (no LM Studio here).
- Structured output, Turkish/English, embedding dimension, retrieval probes on recorded profiles:
  **the probe logic and recording are verified against the simulator; the live probe is not run.**
- Concurrent API/worker inference within memory limits: the single-slot guarantee is verified on the
  wire with the simulator; GPU memory limits were **not measured**.
- Worker resumes durable work after a restart: yes (lease reclaim; chaos S5).
- Notes created, replaced, deleted without stale evidence: yes.
- Every claim traceable to an exact passage in an immutable revision: yes (offset check on read).
- Turkish and English extraction pass the evaluation cases: **not measured with a real model.**
- Hybrid search and graph traversal respect scope and limits: yes.
- Answers have valid citations or state insufficient evidence: yes (citation validation).
- Contradictions, uncertainty, negation preserved: prompt and data model support it; **not evaluated
  with a real model.**
- UI exposes progress, failures, evidence, source passages: yes (Streamlit flow tests and a screenshot
  from an earlier session). Markdown in note titles is escaped.
- Unit, contract, integration and deterministic end-to-end tests pass: yes (215).
- Real LM Studio smoke test passes: **not run.**
- Evaluation report identifies dataset, models, prompts, counts, limits: partly (demo providers only).
- Logs and history contain no credentials or unintended note content: verified in chaos S10; no
  secrets were committed.
- Backup and restore exercised on disposable data: yes, on the final schema.
- Tested installation and startup instructions saved: yes (`docs/operations.md`, `README.md`).

## Open items that need a real machine

1. Start LM Studio with the chosen chat and embedding models, set the model IDs and dimension in `.env`,
   run `notes-graphrag probe --record`, then `scripts/smoke.py` against the production stack.
2. Run `scripts/evaluate.py` with live providers (not implemented yet: the script runs demo providers).
3. Measure GPU memory and warm p95 latency on the target hardware (architecture 16.3).
