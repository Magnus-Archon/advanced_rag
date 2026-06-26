"""FastAPI route handlers."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.models import SearchRequest, SearchResponse
from app.core.pipeline import run_pipeline, _fetcher, _cache
from app.core.embeddings import embed_query
from app.generation.answerer import generate_answer_stream
from app.retrieval.context_builder import build_context
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.search.providers import SearchAggregator
from app.generation.query_expander import expand_query
from app.db.vector_store import upsert_chunks
from app.core.chunker import chunk_document
from app.core.embeddings import embed_texts
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()
_aggregator = SearchAggregator()


# ── POST /search ──────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, summary="RAG search")
async def search(request: SearchRequest) -> SearchResponse:
    """
    Full RAG pipeline:
    - Query expansion
    - Web search
    - Fetch + chunk + embed + store
    - Hybrid retrieve + rerank
    - LLM answer with citations
    - Optional reflection
    """
    try:
        return await run_pipeline(request)
    except Exception as exc:
        logger.error("pipeline_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


# ── POST /search/stream ───────────────────────────────────────────────────────

@router.post("/search/stream", summary="Streaming RAG search (SSE)")
async def search_stream(request: SearchRequest):
    """
    Streaming version: sends SSE events as the answer is generated.
    Events:
      - type: "status"  → pipeline stage updates
      - type: "token"   → answer token stream
      - type: "sources" → final source list
      - type: "done"    → completion
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            query = request.query.strip()

            yield {"event": "status", "data": json.dumps({"stage": "expanding_query"})}
            query_embedding = await embed_query(query)

            # Cache check
            cached = await _cache.get(query, query_embedding)
            if cached:
                yield {"event": "status", "data": json.dumps({"stage": "cache_hit"})}
                yield {"event": "token", "data": cached["answer"]}
                yield {"event": "sources", "data": json.dumps(cached["sources"])}
                yield {"event": "done", "data": "{}"}
                return

            expanded = await expand_query(query)
            yield {"event": "status", "data": json.dumps({"stage": "searching", "queries": expanded})}

            search_results = await _aggregator.search_many(expanded)
            yield {"event": "status", "data": json.dumps({"stage": "fetching_pages", "count": len(search_results)})}

            # Ingest
            urls = [r.url for r in search_results]
            url_to_title = {r.url: r.title for r in search_results}
            url_to_trust = {r.url: r.trust_score for r in search_results}
            page_texts = await _fetcher.fetch_many(urls)
            all_chunks = []
            for url, text in page_texts.items():
                chunks = chunk_document(text, url, url_to_title.get(url, ""), url_to_trust.get(url, 0.65))
                all_chunks.extend(chunks)

            if all_chunks:
                embeddings = await embed_texts([c.text for c in all_chunks])
                await upsert_chunks(all_chunks, embeddings)

            yield {"event": "status", "data": json.dumps({"stage": "retrieving"})}
            ranked = await hybrid_retrieve(query, query_embedding)
            ranked = await rerank(query, ranked, top_k=settings.top_k_rerank)
            context, sources = build_context(ranked)

            yield {"event": "status", "data": json.dumps({"stage": "generating_answer"})}

            # Stream tokens
            full_answer = ""
            async for token in generate_answer_stream(query, context):
                full_answer += token
                yield {"event": "token", "data": token}

            # Cache and emit sources
            source_dicts = [s.model_dump() for s in sources]
            await _cache.set(query, query_embedding, {
                "answer": full_answer,
                "sources": source_dicts,
                "reflected": False,
            })
            yield {"event": "sources", "data": json.dumps(source_dicts)}
            yield {"event": "done", "data": "{}"}

        except Exception as exc:
            logger.error("stream_error", error=str(exc), exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": settings.gemini_chat_model}
