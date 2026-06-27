"""Web search providers.

Primary: Tavily Search API
Fallback: extensible via BaseSearchProvider.

Tavily advantages over Brave for RAG:
  - Returns clean pre-extracted content per result (no need to fetch every page)
  - Natively supports "advanced" depth for more thorough retrieval
  - Built-in answer synthesis (ignored here; we do our own RAG)
  - Results include `raw_content` when depth="advanced"
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from tavily import AsyncTavilyClient
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


class TavilySearchProvider(BaseSearchProvider):
    def __init__(self) -> None:
        self._client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def search(self, query: str, count: int = 10) -> list[SearchResult]:
        try:
            response = await self._client.search(
                query=query,
                search_depth=settings.tavily_search_depth,
                max_results=count,
                include_raw_content=False,   # we fetch pages ourselves via WebFetcher
                include_answer=False,        # we generate our own answer via LLM
            )
        except Exception as exc:
            logger.warning("tavily_search_error", query=query[:60], error=str(exc))
            return []

        results: list[SearchResult] = []
        for item in response.get("results", []):
            url = item.get("url", "")
            # Tavily provides a snippet in `content`; fall back to empty string
            snippet = item.get("content", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=snippet,
                    trust_score=score_domain_trust(url),
                )
            )

        logger.info("tavily_search_ok", query=query[:60], n=len(results))
        return results


# ── Aggregator ────────────────────────────────────────────────────────────────

class SearchAggregator:
    """Run multiple queries in parallel and deduplicate results."""

    def __init__(self, provider: Optional[BaseSearchProvider] = None) -> None:
        self._provider = provider or TavilySearchProvider()

    async def search_many(
        self,
        queries: list[str],
        count_per_query: int | None = None,
    ) -> list[SearchResult]:
        count = count_per_query or settings.tavily_max_results
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
