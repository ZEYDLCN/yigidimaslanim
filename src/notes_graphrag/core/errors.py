"""Application errors carrying a stable machine-readable code."""

from __future__ import annotations


class AppError(Exception):
    code = "INTERNAL_ERROR"
    http_status = 500
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class ValidationFailed(AppError):
    code = "VALIDATION_FAILED"
    http_status = 422


class PayloadTooLarge(AppError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class Conflict(AppError):
    code = "CONFLICT"
    http_status = 409


class IdempotencyKeyReused(Conflict):
    code = "IDEMPOTENCY_KEY_REUSED"


class NoteDeleted(Conflict):
    code = "NOTE_DELETED"


class NotFound(AppError):
    code = "NOT_FOUND"
    http_status = 404


class ProviderUnavailable(AppError):
    code = "LM_STUDIO_UNAVAILABLE"
    http_status = 503
    retryable = True


class ProviderBadResponse(AppError):
    code = "PROVIDER_BAD_RESPONSE"
    http_status = 502


class StaleLease(AppError):
    code = "STALE_LEASE"
    http_status = 409


class KnowledgeBaseChanged(Conflict):
    code = "KNOWLEDGE_BASE_CHANGED"
    retryable = True


class QuestionTimeout(AppError):
    code = "QUESTION_TIMEOUT"
    http_status = 503
    retryable = True


class GenerationInvalid(AppError):
    code = "GENERATION_INVALID"
    http_status = 502
