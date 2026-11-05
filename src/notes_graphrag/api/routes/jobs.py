from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, ok
from notes_graphrag.api.views import job_view
from notes_graphrag.core.errors import NotFound
from notes_graphrag.ingestion.service import utcnow

router = APIRouter(prefix="/api/v1/jobs")


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> JSONResponse:
    store = ctx_of(request).store
    job = store.jobs.get(job_id)
    if job is None or store.notes[job.note_id].deleted_at is not None:
        raise NotFound("job not found")
    return ok(request, job_view(job))


@router.post("/{job_id}/retry")
def retry_job(request: Request, job_id: str) -> JSONResponse:
    job = ctx_of(request).store.retry_job(job_id, utcnow())
    return ok(request, job_view(job), 202)
