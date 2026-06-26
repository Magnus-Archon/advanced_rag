"""Query expansion: use Gemini to generate diverse sub-queries."""
from __future__ import annotations

import asyncio
import json
import re

from google import genai
from google.genai import types as gentypes
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_client = genai.Client(api_key=settings.gemini_api_key)

_SYSTEM = """You are a search query optimizer. Given a user question, generate 4 search queries:
1. The original query (slightly cleaned)
2. A broader / background context query
3. A specific / technical angle query
4. A "latest / recent news" variant (if applicable, otherwise a synonym/alternative phrasing)

Respond ONLY with a JSON array of 4 strings. No explanation, no markdown fences."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def expand_query(query: str) -> list[str]:
    """Return 4 search query variants for the given user query."""
    loop = asyncio.get_event_loop()

    def _sync() -> str:
        response = _client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=query,
            config=gentypes.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.3,
                max_output_tokens=300,
            ),
        )
        return response.text or "[]"

    try:
        raw = await loop.run_in_executor(None, _sync)
        raw = re.sub(r"```json|```", "", raw).strip()
        queries = json.loads(raw)
        if isinstance(queries, list) and queries:
            logger.info("query_expansion_done", n=len(queries), original=query[:60])
            return [str(q) for q in queries[:4]]
    except Exception as exc:
        logger.warning("query_expansion_error", error=str(exc))

    return [query]  # fallback
