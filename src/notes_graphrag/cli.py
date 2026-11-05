"""Command line entry points.

migrate                 apply Neo4j schema migrations (refuses a mismatched vector dimension)
serve [--demo]          run the HTTP API (add --embedded-worker for a single-process setup)
worker [--demo]         run an ingestion worker process against Neo4j
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from notes_graphrag.core.config import Settings


def _logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _demo(settings: Settings, flag: bool) -> bool:
    return flag or settings.graphrag_demo


def cmd_migrate(settings: Settings, demo: bool) -> int:
    from neo4j import GraphDatabase

    from notes_graphrag.api.dependencies import effective_settings
    from notes_graphrag.core.errors import DependencyUnavailable
    from notes_graphrag.persistence.migrations import migrate

    s = effective_settings(settings, _demo(settings, demo))
    if not s.neo4j_password.get_secret_value():
        raise DependencyUnavailable("NEO4J_PASSWORD is not set", code="CONFIG_MISSING")
    driver = GraphDatabase.driver(
        s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password.get_secret_value())
    )
    try:
        applied = migrate(driver, s.neo4j_database, s.embedding_dimensions)
    finally:
        driver.close()
    print(f"applied migrations: {applied or 'none (already up to date)'}")
    return 0


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from notes_graphrag.api.app import create_app
    from notes_graphrag.api.dependencies import build_context

    ctx = build_context(
        settings,
        demo=_demo(settings, args.demo),
        backend=args.store,
        embedded_worker=args.embedded_worker,
    )
    app = create_app(ctx)
    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.log_level.lower())
    if ctx.closeable is not None:
        ctx.closeable.close()
    return 0


def cmd_worker(settings: Settings, args: argparse.Namespace) -> int:
    from notes_graphrag.api.dependencies import (
        build_gate,
        build_providers,
        build_storage,
        build_worker,
        check_schema,
        effective_settings,
    )
    from notes_graphrag.ingestion.worker import run_forever

    if args.store != "neo4j":
        print("worker requires the neo4j store: a separate process cannot share memory")
        return 2
    demo = _demo(settings, args.demo)
    s = effective_settings(settings, demo)
    store, _knowledge, closeable = build_storage(s, "neo4j")
    check_schema(store)
    providers = build_providers(s, demo=demo)
    gate = build_gate(s, store, providers)
    owner = f"worker:{s.workspace_id}:{threading.get_ident()}"
    worker = build_worker(s, store, providers, gate, owner)
    stop = threading.Event()
    try:
        run_forever(worker, store, stop, s.worker_poll_seconds)
    except KeyboardInterrupt:
        stop.set()
    finally:
        if closeable is not None:
            closeable.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="notes-graphrag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    mig = sub.add_parser("migrate", help="apply Neo4j schema migrations")
    mig.add_argument("--demo", action="store_true", help="use demo embedding dimensions")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--demo", action="store_true", help="fake providers, no LM Studio")
    serve.add_argument("--store", choices=["neo4j", "memory"], default=None)
    serve.add_argument("--embedded-worker", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    worker = sub.add_parser("worker", help="run an ingestion worker")
    worker.add_argument("--demo", action="store_true")

    args = parser.parse_args(argv)
    settings = Settings()
    _logging(settings)
    if args.cmd == "migrate":
        return cmd_migrate(settings, args.demo)
    if args.cmd == "serve":
        args.store = args.store or settings.store_backend
        return cmd_serve(settings, args)
    args.store = "neo4j"
    return cmd_worker(settings, args)


if __name__ == "__main__":
    sys.exit(main())
