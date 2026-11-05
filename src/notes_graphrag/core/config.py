"""Typed settings validated at startup. Secrets come from the environment only."""

from __future__ import annotations

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    workspace_id: str = "default"
    log_level: str = "INFO"

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_database: str = "neo4j"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")

    llm_provider: str = "lmstudio"
    lm_studio_instance_id: str = "local-lmstudio"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: SecretStr | None = None
    extraction_model: str = ""
    answer_model: str = ""
    embedding_provider: str = "lmstudio"
    embedding_base_url: str = "http://127.0.0.1:1234/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = ""
    embedding_dimensions: int = 1024
    embedding_profile: str = "qwen3-embedding-0.6b-local-v1"
    lm_studio_context_tokens: int = 8192
    extraction_max_output_tokens: int = 2048
    answer_max_output_tokens: int = 1024

    max_note_bytes: int = 1_048_576
    max_note_chars: int = 100_000
    max_title_chars: int = 200
    max_chunks_per_note: int = 128
    max_context_tokens: int = 4096
    chunk_target_tokens: int = 500
    chunk_overlap_tokens: int = 60
    max_provider_calls_per_job: int = 64
    job_max_attempts: int = 3
    job_lease_seconds: int = 180
    provider_timeout_seconds: float = 120
    question_timeout_seconds: float = 180
    inference_slot_wait_seconds: float = 30
    idempotency_retention_days: int = 30

    app_env_choices: tuple[str, ...] = Field(
        default=("development", "test", "production"), exclude=True
    )

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.app_env not in self.app_env_choices:
            raise ValueError("APP_ENV must be development, test or production")
        if self.llm_provider != "lmstudio" or self.embedding_provider != "lmstudio":
            raise ValueError("only the lmstudio provider is supported")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
        reserve = self.answer_max_output_tokens + 512
        if self.max_context_tokens + reserve > self.lm_studio_context_tokens:
            raise ValueError("MAX_CONTEXT_TOKENS plus output reserve exceeds loaded context")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk overlap must be smaller than chunk target")
        return self
