"""Semantic cache: store and retrieve answers by query embedding similarity.

Flow:
  - On query: embed it, search Redis for a similar cached embedding
  - If cosine similarity > threshold → return cached answer
  - On answer generation: store embedding + answer in Redis
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_SIM_THRESHOLD = 0.92  # cosine similarity to consider a cache hit
_KEY_PREFIX = "rag:cache:"
_IDX_KEY = "rag:cache:index"  # list of all cache keys


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def get(
        self,
        query: str,
        query_embedding: list[float],
    ) -> Optional[dict]:
        """Return cached result if a similar query exists."""
        try:
            r = self._get_redis()
            keys = await r.lrange(_IDX_KEY, 0, -1)
            best_sim = 0.0
            best_data: Optional[str] = None

            for key in keys:
                raw = await r.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                sim = _cosine(query_embedding, entry["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_data = raw

            if best_sim >= _SIM_THRESHOLD and best_data:
                entry = json.loads(best_data)
                logger.info("cache_hit", similarity=round(best_sim, 3), query=query[:60])
                return entry["result"]

        except Exception as exc:
            logger.warning("cache_get_error", error=str(exc))
        return None

    async def set(
        self,
        query: str,
        query_embedding: list[float],
        result: dict,
    ) -> None:
        """Store a query + result in the cache."""
        try:
            r = self._get_redis()
            key = _KEY_PREFIX + hashlib.sha256(query.encode()).hexdigest()[:16]
            entry = {"embedding": query_embedding, "query": query, "result": result}
            await r.setex(key, settings.cache_ttl_seconds, json.dumps(entry))
            await r.lpush(_IDX_KEY, key)
            await r.expire(_IDX_KEY, settings.cache_ttl_seconds * 2)
            logger.debug("cache_set", query=query[:60])
        except Exception as exc:
            logger.warning("cache_set_error", error=str(exc))

    async def aclose(self) -> None:
        if self._redis:
            await self._redis.aclose()
