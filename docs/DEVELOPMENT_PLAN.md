# Development plan and status

Source: the architecture document (Notes → Knowledge Graph). Phases follow its section 17.

## Status

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 0 Bootstrap | Package, lock, config, Compose, CI | Done | `uv.lock`, `compose.yaml`, CI workflow |
| 1 Persistence and notes | Store contract, Neo4j repository, idempotency, leases | Done (Neo4j) | 18 contract tests on both stores |
| 2 Text ingestion | Chunker, validation, staged publication | Done | chunker and publication tests |
| 3 Real extraction | LM Studio adapter, schema validation, entity resolution | Adapter and validation done; live model not run | HTTP contract tests with mock transport; `scripts/check_lmstudio.py` needs a live server |
| 4 Search | Vector and full-text indexes, active filter, RRF | Done (Neo4j indexes) | scoped candidate contract tests; Turkish case folding test |
| 5 Graph retrieval | Entity linking, bounded expansion, evidence packing | Done | expansion limit tests; 2-hop contract test |
| 6 Answers | Structured answers, citation validation, abstention, graph version | Done with demo model | answer and citation tests |
| 7 Interface | FastAPI, Streamlit (Notes, Ask, Graph, System) | Done | API tests, Streamlit AppTest, screenshots from the earlier session, container smoke test |
| 8 Reliability | Inference gate, lease fencing, restart recovery, backup | Done; live LM Studio recovery not run | gate contract tests; measured backup and restore |
| 9 Evaluation | Dataset, baselines, live smoke, operations guide | Partly: dataset and harness done; live run not possible here | `evaluation/`, `scripts/evaluate.py`, `docs/operations.md` |

## Definition of done (architecture 17.2), honest status

- Installs from the committed lockfile: yes (`uv sync --frozen`).
- Schema setup repeatable and checks index compatibility: yes.
- LM Studio serves the configured models from the actual caller: **not verified** (no LM Studio here).
- Structured output, Turkish/English, embedding dimension, retrieval probes on recorded profiles:
  **not verified live**.
- Concurrent API/worker inference within memory limits, gate recovery: gate logic verified; memory
  limits and real GPU behaviour **not measured**.
- Worker resumes durable work after restart: yes (lease reclaim, separate process).
- Notes created, replaced, deleted without stale evidence: yes.
- Every claim traceable to an exact passage in an immutable revision: yes (offset check on read).
- Turkish and English extraction pass the evaluation cases: **not measured with a real model**.
- Hybrid search and graph traversal respect scope and limits: yes.
- Answers have valid citations or state insufficient evidence: yes (citation validation).
- Contradictions, uncertainty, negation preserved: prompt and data model support it; **not
  evaluated** with a real model.
- UI exposes progress, failures, evidence, source passages: yes (Streamlit, verified by screenshot).
- Unit, contract, integration and deterministic end-to-end tests pass: yes (96 tests).
- Real LM Studio smoke test passes: **not run**.
- Evaluation report identifies dataset, models, prompts, counts, limits: partly (demo providers only).
- Logs and history contain no credentials or unintended note content: logs carry IDs and timings
  only; no secrets were committed (checked for the test password).
- Backup and restore exercised on disposable data: yes.
- Tested installation and startup instructions saved: yes (`docs/operations.md`, `README.md`).
