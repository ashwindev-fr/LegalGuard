"""Application configuration loaded from environment variables and .env file."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"


class Settings(BaseSettings):
    """Central configuration — loaded from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT

    # ── Neo4j ────────────────────────────────────────────────────────────
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "change_me"
    neo4j_database: str = "neo4j"

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM / SLM ───────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048

    # ── Directories ──────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"))
    model_dir: Path = Field(default=Path("./models"))
    cache_dir: Path = Field(default=Path("./cache"))

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k_vector: int = 20
    top_k_graph: int = 20
    top_k_lexical: int = 20
    top_k_final: int = 8

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"


# ── Singleton access ─────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings (created once)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_logging(settings: Settings | None = None) -> None:
    """Set up root logging based on application settings."""
    s = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, s.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
