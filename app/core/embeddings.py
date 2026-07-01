"""Gemini embeddings with batching and retry.

Model: gemini-embedding-001, truncated to 768-dim via output_dimensionality.
(text-embedding-004 was deprecated by Google and now 404s on v1beta.)
Batch limit: 100 texts per call (API limit is 250; we stay conservative).

Note: gemini-embedding-001 only pre-normalizes the full 3072-dim output.
Since we request 768 dims, we L2-normalize client-side before storing /
querying so that pgvector's cosine distance operator behaves correctly.
"""
from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np
from google import genai
from google.genai import types as gentypes
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# google-genai uses a synchronous client; we run it in a thread pool
_client = genai.Client(api_key=settings.gemini_api_key)

_BATCH_SIZE = 100
EMBEDDING_DIM = settings.gemini_embedding_dim  # 768 by default


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize. Only the full 3072-dim gemini-embedding-001 output is
    pre-normalized by the API; truncated dims (e.g. 768) are not."""
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n == 0:
        return vec
    return (arr / n).tolist()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()

    def _sync():
        result = _client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=gentypes.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=settings.gemini_embedding_dim,
            ),
        )
        return [_normalize(e.values) for e in result.embeddings]

    return await loop.run_in_executor(None, _sync)


async def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts, batching as needed."""
    texts = list(texts)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        embeddings = await _embed_batch(batch)
        all_embeddings.extend(embeddings)
    logger.debug("embedded_texts", n=len(texts))
    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string (uses RETRIEVAL_QUERY task type)."""
    loop = asyncio.get_event_loop()

    def _sync():
        result = _client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=[query],
            config=gentypes.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=settings.gemini_embedding_dim,
            ),
        )
        return _normalize(result.embeddings[0].values)

    return await loop.run_in_executor(None, _sync)
