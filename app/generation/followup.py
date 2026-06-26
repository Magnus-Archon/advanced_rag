"""Generate follow-up queries when initial context is weak (Gemini)."""
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

_SYSTEM = """You are a research planner. Given a question and a partial context,
identify what key information is missing and generate 2-3 targeted follow-up search queries.

Respond ONLY with a JSON array of query strings. No explanation, no markdown fences."""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
async def generate_followup_queries(query: str, partial_context: str) -> list[str]:
    """Generate follow-up queries for multi-hop retrieval."""
    loop = asyncio.get_event_loop()
    prompt = (
        f"ORIGINAL QUESTION: {query}\n\n"
        f"PARTIAL CONTEXT (may be incomplete):\n{partial_context[:3000]}\n\n"
        "What follow-up search queries would help fill the gaps?"
    )

    def _sync() -> str:
        response = _client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=prompt,
            config=gentypes.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.4,
                max_output_tokens=200,
            ),
        )
        return response.text or "[]"

    try:
        raw = await loop.run_in_executor(None, _sync)
        raw = re.sub(r"```json|```", "", raw).strip()
        queries = json.loads(raw)
        if isinstance(queries, list):
            logger.info("followup_queries_generated", n=len(queries))
            return [str(q) for q in queries[:3]]
    except Exception as exc:
        logger.warning("followup_query_error", error=str(exc))
    return []
