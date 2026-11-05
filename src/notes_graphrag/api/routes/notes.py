from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from notes_graphrag.api.common import ctx_of, idempotency_key, ok
from notes_graphrag.api.schemas import NoteBody
from notes_graphrag.api.views import decode_cursor, encode_cursor, note_view
from notes_graphrag.core.errors import NotFound
from notes_graphrag.domain.state_machines import RevisionStatus
from notes_graphrag.ingestion.validation import decode_text, validate_filename
from notes_graphrag.persistence.models import Note, Submission

router = APIRouter(prefix="/api/v1/notes")
Key = Annotated[str, Depends(idempotency_key)]


def _accepted(request: Request, sub: Submission) -> JSONResponse:
    return ok(
        request,
        {
            "note_id": sub.note.note_id,
            "revision_id": sub.revision.revision_id,
            "job_id": sub.job.job_id,
            "status": sub.job.status.value.lower(),
            "replayed": sub.replayed,
        },
        202,
    )


def _live(request: Request, note_id: str) -> Note:
    note = ctx_of(request).store.get_note(note_id)
    if note is None or note.deleted_at is not None:
        raise NotFound("note not found")  # deleted notes look exactly like missing ones
    return note


@router.post("")
def create_note(request: Request, body: NoteBody, key: Key) -> JSONResponse:
    return _accepted(request, ctx_of(request).notes.create(body.title, body.text, key))


@router.post("/upload")
def upload_note(
    request: Request,
    key: Key,
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()],
) -> JSONResponse:
    ctx = ctx_of(request)
    validate_filename(file.filename or "")
    raw = file.file.read(ctx.settings.max_note_bytes + 1)  # bounded read
    text = decode_text(raw, ctx.settings)
    return _accepted(request, ctx.notes.create(title, text, key))


@router.get("")
def list_notes(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> JSONResponse:
    ctx = ctx_of(request)
    parts = decode_cursor(cursor, 2)
    after = (datetime.fromisoformat(parts[0]), parts[1]) if parts else None
    rows, more = ctx.store.list_notes(after, limit)
    nxt = encode_cursor(rows[-1].created_at.isoformat(), rows[-1].note_id) if more else None
    return ok(request, {"items": [note_view(ctx.store, n) for n in rows], "next_cursor": nxt})


@router.get("/{note_id}")
def get_note(request: Request, note_id: str) -> JSONResponse:
    return ok(request, note_view(ctx_of(request).store, _live(request, note_id)))


@router.get("/{note_id}/revisions/{revision_id}")
def get_revision(request: Request, note_id: str, revision_id: str) -> JSONResponse:
    _live(request, note_id)
    rev = ctx_of(request).store.get_revision(revision_id)
    if rev is None or rev.note_id != note_id or rev.status == RevisionStatus.CANCELLED:
        raise NotFound("revision not found")
    return ok(
        request,
        {
            "note_id": note_id,
            "revision_id": rev.revision_id,
            "title": rev.title,
            "status": rev.status.value,
            "text": rev.text,
            "offset_unit": "unicode_code_point",
        },
    )


@router.put("/{note_id}")
def replace_note(request: Request, note_id: str, body: NoteBody, key: Key) -> JSONResponse:
    sub = ctx_of(request).notes.replace(note_id, body.title, body.text, key)
    return _accepted(request, sub)


@router.delete("/{note_id}", status_code=204)
def delete_note(request: Request, note_id: str) -> Response:
    ctx_of(request).notes.delete(note_id)
    return Response(status_code=204)


@router.post("/{note_id}/reindex")
def reindex(request: Request, note_id: str, key: Key) -> JSONResponse:
    return _accepted(request, ctx_of(request).notes.reindex(note_id, key))
