from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok
from notes_graphrag.api.views import job_view
from notes_graphrag.core.errors import NotFound

router = APIRouter(prefix="/api/v1/jobs")


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> JSONResponse:
    store = ctx_of(request).store
    job = store.get_job(job_id)
    if job is None:
        raise NotFound("job not found")
    note = store.get_note(job.note_id)
    if note is None or note.deleted_at is not None:
        raise NotFound("job not found")
    return ok(request, job_view(job))


@router.post("/{job_id}/retry")
def retry_job(request: Request, job_id: str) -> JSONResponse:
    ctx = ctx_of(request)
    ctx.notes.retry(job_id)
    job = ctx.store.get_job(job_id)
    assert job is not None
    return ok(request, job_view(job), 202)
