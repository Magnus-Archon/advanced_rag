"""Vector store operations: upsert chunks + cosine similarity search."""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import DocumentChunk, RankedChunk
from app.db.models import ChunkRecord, AsyncSessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def upsert_chunks(
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[list[float]],
) -> None:
    """Insert or update chunk records in the DB."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for chunk, emb in zip(chunks, embeddings):
                existing = await session.get(ChunkRecord, chunk.chunk_id)
                if existing:
                    existing.content = chunk.text
                    existing.embedding = emb
                    existing.trust_score = chunk.trust_score
                    existing.metadata_ = chunk.metadata
                else:
                    session.add(
                        ChunkRecord(
                            id=chunk.chunk_id,
                            url=chunk.url,
                            title=chunk.title,
                            content=chunk.text,
                            trust_score=chunk.trust_score,
                            embedding=emb,
                            metadata_=chunk.metadata,
                        )
                    )
    logger.debug("upserted_chunks", n=len(chunks))


async def semantic_search(
    query_embedding: list[float],
    top_k: int = 20,
    session: AsyncSession | None = None,
) -> list[RankedChunk]:
    """Cosine-similarity vector search. Returns top_k RankedChunks."""

    async def _run(s: AsyncSession) -> list[RankedChunk]:
        # pgvector cosine distance: <=> operator (lower = more similar)
        stmt = (
            sa.text(
                """
                SELECT id, url, title, content, trust_score, metadata,
                       1 - (embedding <=> CAST(:emb AS vector)) AS score
                FROM chunks
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :k
                """
            ).bindparams(
                emb=str(query_embedding),
                k=top_k,
            )
        )
        rows = (await s.execute(stmt)).fetchall()
        results: list[RankedChunk] = []
        for row in rows:
            chunk = DocumentChunk(
                chunk_id=row.id,
                url=row.url,
                title=row.title or "",
                text=row.content,
                trust_score=row.trust_score or 0.65,
                metadata=row.metadata or {},
            )
            results.append(RankedChunk(chunk=chunk, score=float(row.score)))
        return results

    if session:
        return await _run(session)

    async with AsyncSessionLocal() as s:
        return await _run(s)
