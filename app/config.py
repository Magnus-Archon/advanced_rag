"""Central settings loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Gemini) ───────────────────────────────────
    gemini_api_key: str
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"  # 768-dim

    # ── Search (Tavily) ────────────────────────────────
    tavily_api_key: str
    tavily_search_depth: str = "basic"          # "basic" | "advanced"
    tavily_max_results: int = 10

    # ── Reranking ──────────────────────────────────────
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── DB ─────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://rag:ragpassword@localhost:5432/ragdb"
    database_sync_url: str = "postgresql://rag:ragpassword@localhost:5432/ragdb"

    # ── Redis ──────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── App ────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    max_context_chars: int = 20_000
    top_k_retrieval: int = 20
    top_k_rerank: int = 8
    reflection_enabled: bool = True
    multihop_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
