from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok
from notes_graphrag.api.schemas import QuestionBody

router = APIRouter(prefix="/api/v1/questions")


@router.post("")
def ask(request: Request, body: QuestionBody) -> JSONResponse:
    r = ctx_of(request).answers.ask(body.question, include_graph=body.include_graph)
    return ok(
        request,
        {
            "status": r.status,
            "answer": r.answer,
            "citations": [asdict(c) for c in r.citations],
            "missing_information": r.missing_information,
            "graph": r.graph,
            "diagnostics": r.diagnostics,
        },
    )
