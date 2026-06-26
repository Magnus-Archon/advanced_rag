"""Cohere reranking layer.

Falls back to score-based ordering if Cohere is unavailable.
"""
from __future__ import annotations

import cohere
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.models import RankedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_cohere: cohere.AsyncClientV2 | None = None


def _get_cohere() -> cohere.AsyncClientV2 | None:
    global _cohere
    if not settings.cohere_api_key:
        return None
    if _cohere is None:
        _cohere = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    return _cohere


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def rerank(
    query: str,
    chunks: list[RankedChunk],
    top_k: int | None = None,
) -> list[RankedChunk]:
    """Rerank using Cohere or fall back to hybrid score ordering."""
    k = top_k or settings.top_k_rerank
    if not chunks:
        return []

    client = _get_cohere()
    if client is None:
        logger.warning("cohere_unavailable_using_fallback")
        return sorted(chunks, key=lambda x: x.score, reverse=True)[:k]

    try:
        documents = [rc.chunk.text[:2000] for rc in chunks]  # Cohere doc length limit
        response = await client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=k,
        )
        reranked: list[RankedChunk] = []
        for result in response.results:
            rc = chunks[result.index]
            reranked.append(
                RankedChunk(
                    chunk=rc.chunk,
                    score=result.relevance_score,
                    relevance_score=result.relevance_score,
                )
            )
        logger.info("rerank_done", returned=len(reranked))
        return reranked

    except Exception as exc:
        logger.warning("rerank_error_fallback", error=str(exc))
        return sorted(chunks, key=lambda x: x.score, reverse=True)[:k]
