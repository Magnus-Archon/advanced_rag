"""Web search providers.

Primary: Brave Search API
Fallback: extensible via BaseSearchProvider.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.models import SearchResult
from app.utils.logger import get_logger
from app.utils.trust import score_domain_trust

logger = get_logger(__name__)
settings = get_settings()


class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, count: int = 10) -> list[SearchResult]:
        ...


class BraveSearchProvider(BaseSearchProvider):
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": settings.brave_api_key,
            },
            timeout=15.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def search(self, query: str, count: int = 10) -> list[SearchResult]:
        try:
            resp = await self._client.get(
                self.BASE_URL,
                params={"q": query, "count": count, "text_decorations": 0},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("brave_search_error", query=query, error=str(exc))
            return []

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            url = item.get("url", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("description", ""),
                    trust_score=score_domain_trust(url),
                )
            )
        logger.info("brave_search_ok", query=query[:60], n=len(results))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


# ── Aggregator ────────────────────────────────────────────────────────────────

class SearchAggregator:
    """Run multiple queries in parallel and deduplicate results."""

    def __init__(self, provider: Optional[BaseSearchProvider] = None) -> None:
        self._provider = provider or BraveSearchProvider()

    async def search_many(
        self,
        queries: list[str],
        count_per_query: int | None = None,
    ) -> list[SearchResult]:
        count = count_per_query or settings.brave_search_count
        tasks = [self._provider.search(q, count) for q in queries]
        batches = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        merged: list[SearchResult] = []
        for batch in batches:
            if isinstance(batch, Exception):
                logger.warning("search_batch_error", error=str(batch))
                continue
            for r in batch:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    merged.append(r)

        logger.info("search_aggregator_done", total_results=len(merged))
        return merged
