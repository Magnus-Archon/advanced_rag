"""Hybrid retrieval: combines semantic vector search with BM25 keyword scoring.

Flow:
  1. semantic_search() → top-K chunks by cosine similarity
  2. BM25 scoring over the same candidate set
  3. Combined score = α * semantic + (1-α) * bm25_norm
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.core.models import DocumentChunk, RankedChunk
from app.db.vector_store import semantic_search
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

ALPHA = 0.70  # weight for semantic score vs BM25


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    raw = bm25.get_scores(_tokenize(query))
    max_score = max(raw) if raw.max() > 0 else 1.0
    return [float(s / max_score) for s in raw]


async def hybrid_retrieve(
    query: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[RankedChunk]:
    """Return top-K chunks ranked by hybrid score."""
    k = top_k or settings.top_k_retrieval

    # Step 1: semantic search (fetch 2× for candidate pool)
    candidates = await semantic_search(query_embedding, top_k=k * 2)
    if not candidates:
        return []

    texts = [rc.chunk.text for rc in candidates]

    # Step 2: BM25 scores
    bm25 = bm25_scores(query, texts)

    # Step 3: combine
    combined: list[RankedChunk] = []
    for rc, bm25_s in zip(candidates, bm25):
        hybrid = ALPHA * rc.score + (1 - ALPHA) * bm25_s
        # trust score acts as a multiplier (±10%)
        trust_boost = 0.9 + 0.2 * rc.chunk.trust_score   # [0.9, 1.1]
        final = hybrid * trust_boost
        combined.append(RankedChunk(chunk=rc.chunk, score=final, relevance_score=rc.score))

    combined.sort(key=lambda x: x.score, reverse=True)
    result = combined[:k]
    logger.info("hybrid_retrieve_done", returned=len(result))
    return result
