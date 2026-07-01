"""FastAPI route handlers.

Endpoints:
  POST /search                — full RAG pipeline
  POST /search/stream         — streaming SSE version
  POST /files/upload          — ingest one or more files
  GET  /files                 — list uploaded files
  DELETE /files               — delete a file's chunks
  GET  /health
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import sqlalchemy as sa
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.embeddings import embed_query, embed_texts
from app.core.models import SearchRequest, SearchResponse
from app.core.pipeline import _cache, _fetcher, run_pipeline
from app.core.chunker import chunk_document
from app.db.vector_store import delete_file_chunks, list_uploaded_files, upsert_chunks
from app.generation.answerer import generate_answer_stream
from app.generation.query_expander import expand_query
from app.retrieval.context_builder import build_context
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.reranker import rerank
from app.search.file_ingestor import SUPPORTED_EXTENSIONS, ingest_file
from app.search.providers import SearchAggregator
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()
_aggregator = SearchAggregator()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file


# ── POST /search ──────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        return await run_pipeline(request)
    except Exception as exc:
        logger.error("pipeline_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /search/stream ───────────────────────────────────────────────────────

@router.post("/search/stream")
async def search_stream(request: SearchRequest):
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            query = request.query.strip()
            scope = request.scope

            yield {"event": "status", "data": json.dumps({"stage": "embedding_query"})}
            query_embedding = await embed_query(query)

            if scope != "files":
                cached = await _cache.get(query, query_embedding)
                if cached:
                    yield {"event": "status", "data": json.dumps({"stage": "cache_hit"})}
                    yield {"event": "token",   "data": cached["answer"]}
                    yield {"event": "sources", "data": json.dumps(cached["sources"])}
                    yield {"event": "done",    "data": "{}"}
                    return

            if scope in ("web", "all"):
                yield {"event": "status", "data": json.dumps({"stage": "expanding_query"})}
                expanded = await expand_query(query)
                yield {"event": "status", "data": json.dumps({"stage": "searching", "queries": expanded})}
                search_results = await _aggregator.search_many(expanded)

                yield {"event": "status", "data": json.dumps({"stage": "fetching_pages", "count": len(search_results)})}
                urls         = [r.url for r in search_results]
                url_to_title = {r.url: r.title for r in search_results}
                url_to_trust = {r.url: r.trust_score for r in search_results}
                page_texts   = await _fetcher.fetch_many(urls)
                all_chunks   = []
                for url, text in page_texts.items():
                    chunks = chunk_document(text, url, url_to_title.get(url, ""), url_to_trust.get(url, 0.65))
                    for c in chunks:
                        c.source_type = "web"
                    all_chunks.extend(chunks)
                if all_chunks:
                    embeddings = await embed_texts([c.text for c in all_chunks])
                    await upsert_chunks(all_chunks, embeddings)

            yield {"event": "status", "data": json.dumps({"stage": "retrieving"})}
            ranked = await hybrid_retrieve(query, query_embedding, scope=scope)
            ranked = await rerank(query, ranked, top_k=settings.top_k_rerank)
            context, sources = build_context(ranked)

            yield {"event": "status", "data": json.dumps({"stage": "generating"})}
            full_answer = ""
            async for token in generate_answer_stream(query, context):
                full_answer += token
                yield {"event": "token", "data": token}

            source_dicts = [s.model_dump() for s in sources]
            if scope != "files":
                await _cache.set(query, query_embedding, {
                    "answer": full_answer, "sources": source_dicts, "reflected": False
                })
            yield {"event": "sources", "data": json.dumps(source_dicts)}
            yield {"event": "done",    "data": "{}"}

        except Exception as exc:
            logger.error("stream_error", error=str(exc), exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


# ── POST /files/upload ────────────────────────────────────────────────────────

@router.post("/files/upload", summary="Upload and ingest documents")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Upload one or more files (PDF, DOCX, TXT, MD).
    Each file is parsed, chunked, embedded and stored in the vector DB.
    After uploading, use scope='files' or scope='all' in /search to query them.
    """
    results = []
    errors  = []

    for upload in files:
        filename = upload.filename or "unknown"
        try:
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                errors.append({"filename": filename, "error": f"File exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit"})
                continue
            summary = await ingest_file(filename, data)
            results.append(summary)
        except ValueError as exc:
            errors.append({"filename": filename, "error": str(exc)})
        except Exception as exc:
            logger.error("upload_error", filename=filename, error=str(exc), exc_info=True)
            errors.append({"filename": filename, "error": "Internal error during ingestion"})

    return {
        "ingested": results,
        "errors":   errors,
        "total_ingested": len(results),
        "supported_formats": sorted(SUPPORTED_EXTENSIONS),
    }


# ── GET /files ────────────────────────────────────────────────────────────────

@router.get("/files", summary="List uploaded files")
async def list_files():
    """Return all files currently stored in the vector DB."""
    files = await list_uploaded_files()
    return {"files": files, "count": len(files)}


# ── DELETE /files ─────────────────────────────────────────────────────────────

@router.delete("/files", summary="Delete an uploaded file")
async def delete_file(file_url: str = Query(..., description="The file_url returned by /files/upload")):
    """Remove all chunks for a previously uploaded file."""
    count = await delete_file_chunks(file_url)
    if count == 0:
        raise HTTPException(status_code=404, detail="No chunks found for that file_url")
    return {"deleted_chunks": count, "file_url": file_url}


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "model": settings.gemini_chat_model}
