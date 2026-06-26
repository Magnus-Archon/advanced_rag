"""Async web page fetcher + trafilatura-based content extractor."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

import httpx
import trafilatura

from app.utils.logger import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/ragbot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)


async def fetch_page(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fetch raw HTML for a URL. Returns None on failure."""
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" not in ct and "text" not in ct:
            return None
        return resp.text
    except Exception as exc:
        logger.debug("fetch_failed", url=url[:80], error=str(exc))
        return None


def extract_text(html: str, url: str) -> Optional[str]:
    """Extract clean main-content text from HTML using trafilatura."""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        return text if text and len(text) > 100 else None
    except Exception as exc:
        logger.debug("extract_failed", url=url[:80], error=str(exc))
        return None


class WebFetcher:
    def __init__(self, concurrency: int = 8) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=_TIMEOUT,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def fetch_and_extract(self, url: str) -> Optional[str]:
        async with self._sem:
            html = await fetch_page(url, self._client)
            if not html:
                return None
            return extract_text(html, url)

    async def fetch_many(self, urls: list[str]) -> dict[str, str]:
        """Returns {url: cleaned_text} for successfully fetched pages."""
        tasks = {url: self.fetch_and_extract(url) for url in urls}
        results: dict[str, str] = {}
        coros = [(url, coro) for url, coro in tasks.items()]
        fetched = await asyncio.gather(*[c for _, c in coros], return_exceptions=True)
        for (url, _), text in zip(coros, fetched):
            if isinstance(text, str) and text:
                results[url] = text
        logger.info("fetch_many_done", requested=len(urls), successful=len(results))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()
