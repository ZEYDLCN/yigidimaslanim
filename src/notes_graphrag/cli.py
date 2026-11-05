"""Command line entry points.

`serve` runs the API with an in-process worker over the in-memory store (development only;
data is lost on exit). The separate-worker + Neo4j topology is a later phase.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from notes_graphrag.api.app import create_app
from notes_graphrag.api.dependencies import build_context
from notes_graphrag.core.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="notes-graphrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run API + in-process worker (in-memory store)")
    serve.add_argument("--demo", action="store_true", help="use fake providers (no LM Studio)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app(build_context(settings, demo=args.demo))
    uvicorn.run(app, host=args.host, port=args.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
