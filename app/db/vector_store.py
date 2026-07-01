"""Vector store: upsert chunks + cosine similarity search with scope filtering."""
from __future__ import annotations

from typing import Literal, Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import DocumentChunk, RankedChunk
from app.db.models import ChunkRecord, AsyncSessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)

Scope = Literal["web", "files", "all"]


async def upsert_chunks(
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[list[float]],
) -> None:
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for chunk, emb in zip(chunks, embeddings):
                existing = await session.get(ChunkRecord, chunk.chunk_id)
                if existing:
                    existing.content     = chunk.text
                    existing.embedding   = emb
                    existing.trust_score = chunk.trust_score
                    existing.source_type = chunk.source_type
                    existing.metadata_   = chunk.metadata
                else:
                    session.add(ChunkRecord(
                        id          = chunk.chunk_id,
                        url         = chunk.url,
                        title       = chunk.title,
                        content     = chunk.text,
                        trust_score = chunk.trust_score,
                        source_type = chunk.source_type,
                        embedding   = emb,
                        metadata_   = chunk.metadata,
                    ))
    logger.debug("upserted_chunks", n=len(chunks))


async def semantic_search(
    query_embedding: list[float],
    top_k: int = 20,
    scope: Scope = "all",
    session: AsyncSession | None = None,
) -> list[RankedChunk]:
    """Cosine-similarity search with optional source_type filter."""

    # Build optional WHERE clause
    if scope == "web":
        where = "WHERE source_type = 'web'"
    elif scope == "files":
        where = "WHERE source_type = 'upload'"
    else:
        where = ""

    sql = f"""
        SELECT id, url, title, content, trust_score, source_type, metadata,
               1 - (embedding <=> CAST(:emb AS vector)) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> CAST(:emb AS vector)
        LIMIT :k
    """

    async def _run(s: AsyncSession) -> list[RankedChunk]:
        rows = (await s.execute(
            sa.text(sql).bindparams(emb=str(query_embedding), k=top_k)
        )).fetchall()
        results: list[RankedChunk] = []
        for row in rows:
            chunk = DocumentChunk(
                chunk_id    = row.id,
                url         = row.url,
                title       = row.title or "",
                text        = row.content,
                trust_score = row.trust_score or 0.65,
                source_type = row.source_type or "web",
                metadata    = row.metadata or {},
            )
            results.append(RankedChunk(chunk=chunk, score=float(row.score)))
        return results

    if session:
        return await _run(session)
    async with AsyncSessionLocal() as s:
        return await _run(s)


async def list_uploaded_files() -> list[dict]:
    """Return distinct uploaded files (url, title) for the UI sidebar."""
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(sa.text(
            "SELECT DISTINCT url, title, metadata FROM chunks WHERE source_type='upload' ORDER BY url"
        ))).fetchall()
    return [{"url": r.url, "title": r.title, "metadata": r.metadata or {}} for r in rows]


async def delete_file_chunks(file_url: str) -> int:
    """Delete all chunks for an uploaded file. Returns count deleted."""
    async with AsyncSessionLocal() as s:
        async with s.begin():
            result = await s.execute(
                sa.text("DELETE FROM chunks WHERE source_type='upload' AND url=:url").bindparams(url=file_url)
            )
    deleted = result.rowcount
    logger.info("deleted_file_chunks", url=file_url, count=deleted)
    return deleted
