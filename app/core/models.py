"""Shared Pydantic models used across the system."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Search layer ──────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    trust_score: float = 0.65


# ── Document / chunk layer ────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    chunk_id: str
    url: str
    title: str
    text: str
    trust_score: float = 0.65
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(BaseModel):
    chunk: DocumentChunk
    score: float                # combined retrieval + rerank score
    relevance_score: float = 0.0


# ── API layer ─────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    multihop: bool = True
    reflect: bool = True
    debug: bool = False


class SourceCitation(BaseModel):
    title: str
    url: str
    trust_score: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    reflected: bool = False
    debug: Optional[dict[str, Any]] = None


# ── Internal pipeline ─────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    original_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    ranked_chunks: list[RankedChunk] = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    sources: list[SourceCitation] = Field(default_factory=list)
    reflection_note: str = ""
    hop: int = 1
