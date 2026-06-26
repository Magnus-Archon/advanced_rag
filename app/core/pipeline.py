"""Main RAG pipeline orchestrator.

Ties together:
  query expansion → search → fetch → chunk → embed → upsert →
  hybrid retrieve → rerank → context build → generate → reflect
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.config import get_settings
from app.core.cache import SemanticCache
from app.core.chunker import chunk_document
from app.core.embeddings import embed_query, embed_texts
from app.core.models import (
    DocumentChunk,
    PipelineState,
    RankedChunk,
    SearchRequest,
    SearchResponse,
    SourceCitation,
)
from app.db.vector_store import upsert_chunks
from app.generation.answerer import generate_answer
from app.generation.followup import generate_followup_queries
from app.generation.query_expander import expand_query
from app.generation.reflector import reflect_and_verify
from app.retrieval.context_builder import build_context
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.search.fetcher import WebFetcher
from app.search.providers import SearchAggregator
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Module-level singletons (initialised once per process)
_aggregator = SearchAggregator()
_fetcher = WebFetcher(concurrency=8)
_cache = SemanticCache()


async def _ingest_pages(
    search_results,
) -> list[DocumentChunk]:
    """Fetch pages, extract text, chunk, embed, upsert. Returns all chunks."""
    urls = [r.url for r in search_results]
    url_to_title = {r.url: r.title for r in search_results}
    url_to_trust = {r.url: r.trust_score for r in search_results}

    page_texts = await _fetcher.fetch_many(urls)

    all_chunks: list[DocumentChunk] = []
    for url, text in page_texts.items():
        chunks = chunk_document(
            text=text,
            url=url,
            title=url_to_title.get(url, ""),
            trust_score=url_to_trust.get(url, 0.65),
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("no_chunks_ingested")
        return []

    # Embed all chunks (batched internally)
    texts = [c.text for c in all_chunks]
    embeddings = await embed_texts(texts)

    # Upsert to vector DB
    await upsert_chunks(all_chunks, embeddings)
    logger.info("ingestion_complete", chunks=len(all_chunks))
    return all_chunks


async def _retrieve_and_rerank(
    query: str,
    query_embedding: list[float],
) -> list[RankedChunk]:
    candidates = await hybrid_retrieve(query, query_embedding)
    if not candidates:
        return []
    reranked = await rerank(query, candidates, top_k=settings.top_k_rerank)
    return reranked


async def run_pipeline(request: SearchRequest) -> SearchResponse:
    """Execute the full RAG pipeline for a search request."""
    query = request.query.strip()
    state = PipelineState(original_query=query)

    # ── 0. Embed query ────────────────────────────────────────────────────────
    query_embedding = await embed_query(query)

    # ── 1. Semantic cache check ───────────────────────────────────────────────
    cached = await _cache.get(query, query_embedding)
    if cached:
        return SearchResponse(**cached)

    # ── 2. Query expansion ────────────────────────────────────────────────────
    expanded = await expand_query(query)
    state.expanded_queries = expanded
    logger.info("expanded_queries", queries=expanded)

    # ── 3. Web search (parallel across expanded queries) ──────────────────────
    search_results = await _aggregator.search_many(expanded)
    state.search_results = search_results

    # ── 4. Fetch + Chunk + Embed + Upsert ─────────────────────────────────────
    await _ingest_pages(search_results)

    # ── 5. Hybrid retrieve + rerank ───────────────────────────────────────────
    ranked = await _retrieve_and_rerank(query, query_embedding)
    state.ranked_chunks = ranked

    # ── 6. Multi-hop: if context is thin, do a second retrieval round ─────────
    if request.multihop and settings.multihop_enabled and len(ranked) < 3:
        logger.info("multihop_triggered", hop=2)
        context_preview, _ = build_context(ranked)
        followup_queries = await generate_followup_queries(query, context_preview)
        if followup_queries:
            extra_results = await _aggregator.search_many(followup_queries)
            await _ingest_pages(extra_results)
            extra_ranked = await _retrieve_and_rerank(query, query_embedding)
            # Merge and deduplicate by chunk_id
            seen = {rc.chunk.chunk_id for rc in ranked}
            for rc in extra_ranked:
                if rc.chunk.chunk_id not in seen:
                    ranked.append(rc)
                    seen.add(rc.chunk.chunk_id)
            ranked.sort(key=lambda x: x.score, reverse=True)
            ranked = ranked[: settings.top_k_rerank]
            state.ranked_chunks = ranked
            state.hop = 2

    # ── 7. Build context ──────────────────────────────────────────────────────
    context, sources = build_context(ranked)
    state.context = context
    state.sources = sources

    if not context.strip():
        return SearchResponse(
            answer="I was unable to retrieve sufficient information to answer your question. Please try rephrasing.",
            sources=[],
        )

    # ── 8. Generate answer ────────────────────────────────────────────────────
    answer = await generate_answer(query, context)
    state.answer = answer

    # ── 9. Reflection / verification ──────────────────────────────────────────
    reflected = False
    if request.reflect and settings.reflection_enabled:
        answer, reflection_note = await reflect_and_verify(query, answer, context)
        state.reflection_note = reflection_note
        reflected = True
        logger.info("reflection_complete", note=reflection_note[:80])

    # ── 10. Build response ────────────────────────────────────────────────────
    debug_data = None
    if request.debug:
        debug_data = {
            "expanded_queries": state.expanded_queries,
            "search_results_count": len(state.search_results),
            "chunks_retrieved": len(state.ranked_chunks),
            "hop": state.hop,
            "reflection_note": state.reflection_note,
            "top_chunks": [
                {
                    "url": rc.chunk.url,
                    "score": round(rc.score, 4),
                    "preview": rc.chunk.text[:120],
                }
                for rc in state.ranked_chunks[:5]
            ],
        }

    response = SearchResponse(
        answer=answer,
        sources=sources,
        reflected=reflected,
        debug=debug_data,
    )

    # ── 11. Cache result ──────────────────────────────────────────────────────
    await _cache.set(query, query_embedding, response.model_dump())

    return response
