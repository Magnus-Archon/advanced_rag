"""Main RAG pipeline orchestrator."""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.core.cache import SemanticCache
from app.core.chunker import chunk_document
from app.core.embeddings import embed_query, embed_texts
from app.core.models import (
    DocumentChunk, PipelineState, RankedChunk,
    SearchRequest, SearchResponse, SourceCitation,
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

_aggregator = SearchAggregator()
_fetcher    = WebFetcher(concurrency=8)
_cache      = SemanticCache()


async def _ingest_pages(search_results) -> list[DocumentChunk]:
    urls          = [r.url for r in search_results]
    url_to_title  = {r.url: r.title for r in search_results}
    url_to_trust  = {r.url: r.trust_score for r in search_results}

    page_texts = await _fetcher.fetch_many(urls)

    all_chunks: list[DocumentChunk] = []
    for url, text in page_texts.items():
        chunks = chunk_document(
            text=text, url=url,
            title=url_to_title.get(url, ""),
            trust_score=url_to_trust.get(url, 0.65),
        )
        for c in chunks:
            c.source_type = "web"
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("no_chunks_ingested")
        return []

    embeddings = await embed_texts([c.text for c in all_chunks])
    await upsert_chunks(all_chunks, embeddings)
    logger.info("ingestion_complete", chunks=len(all_chunks))
    return all_chunks


async def _retrieve_and_rerank(query, query_embedding, scope="all") -> list[RankedChunk]:
    candidates = await hybrid_retrieve(query, query_embedding, scope=scope)
    if not candidates:
        return []
    return await rerank(query, candidates, top_k=settings.top_k_rerank)


async def run_pipeline(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    scope = request.scope   # "web" | "files" | "all"
    state = PipelineState(original_query=query)

    # ── 0. Embed query ────────────────────────────────────────────────────────
    query_embedding = await embed_query(query)

    # ── 1. Cache check (skip for files-only so fresh docs always searched) ────
    if scope != "files":
        cached = await _cache.get(query, query_embedding)
        if cached:
            return SearchResponse(**cached)

    # ── 2. Query expansion + web search (skip when scope=files) ──────────────
    if scope in ("web", "all"):
        expanded = await expand_query(query)
        state.expanded_queries = expanded
        search_results = await _aggregator.search_many(expanded)
        state.search_results = search_results
        await _ingest_pages(search_results)

    # ── 3. Hybrid retrieve + rerank ───────────────────────────────────────────
    ranked = await _retrieve_and_rerank(query, query_embedding, scope=scope)
    state.ranked_chunks = ranked

    # ── 4. Multi-hop (web only; no point re-searching files) ─────────────────
    if scope in ("web", "all") and request.multihop and settings.multihop_enabled and len(ranked) < 3:
        logger.info("multihop_triggered", hop=2)
        context_preview, _ = build_context(ranked)
        followup_queries = await generate_followup_queries(query, context_preview)
        if followup_queries:
            extra_results = await _aggregator.search_many(followup_queries)
            await _ingest_pages(extra_results)
            extra_ranked = await _retrieve_and_rerank(query, query_embedding, scope=scope)
            seen = {rc.chunk.chunk_id for rc in ranked}
            for rc in extra_ranked:
                if rc.chunk.chunk_id not in seen:
                    ranked.append(rc)
                    seen.add(rc.chunk.chunk_id)
            ranked.sort(key=lambda x: x.score, reverse=True)
            ranked = ranked[:settings.top_k_rerank]
            state.ranked_chunks = ranked
            state.hop = 2

    # ── 5. Build context ──────────────────────────────────────────────────────
    context, sources = build_context(ranked)
    state.context = context
    state.sources = sources

    if not context.strip():
        return SearchResponse(
            answer="I was unable to retrieve sufficient information to answer your question. Please try rephrasing.",
            sources=[],
        )

    # ── 6. Generate answer ────────────────────────────────────────────────────
    answer = await generate_answer(query, context)
    state.answer = answer

    # ── 7. Reflection ─────────────────────────────────────────────────────────
    reflected = False
    if request.reflect and settings.reflection_enabled:
        answer, reflection_note = await reflect_and_verify(query, answer, context)
        state.reflection_note = reflection_note
        reflected = True

    # ── 8. Build response ─────────────────────────────────────────────────────
    debug_data = None
    if request.debug:
        debug_data = {
            "scope": scope,
            "expanded_queries": state.expanded_queries,
            "search_results_count": len(state.search_results),
            "chunks_retrieved": len(state.ranked_chunks),
            "hop": state.hop,
            "reflection_note": state.reflection_note,
            "top_chunks": [
                {"url": rc.chunk.url, "source_type": rc.chunk.source_type,
                 "score": round(rc.score, 4), "preview": rc.chunk.text[:120]}
                for rc in state.ranked_chunks[:5]
            ],
        }

    response = SearchResponse(answer=answer, sources=sources, reflected=reflected, debug=debug_data)

    if scope != "files":
        await _cache.set(query, query_embedding, response.model_dump())

    return response
