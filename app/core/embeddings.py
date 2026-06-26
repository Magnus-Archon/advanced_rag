"""Gemini embeddings with batching and retry.

Model: text-embedding-004  (768-dim)
Batch limit: 100 texts per call (API limit is 250; we stay conservative).
"""
from __future__ import annotations

import asyncio
from typing import Sequence

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
EMBEDDING_DIM = 768  # text-embedding-004


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()

    def _sync():
        result = _client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=gentypes.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]

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
            config=gentypes.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    return await loop.run_in_executor(None, _sync)
