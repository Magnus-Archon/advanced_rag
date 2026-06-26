"""LLM answer generation via Gemini: citation-aware, grounded, hallucination-resistant."""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google import genai
from google.genai import types as gentypes
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_client = genai.Client(api_key=settings.gemini_api_key)

_SYSTEM = """You are a precise, citation-driven research assistant.

Rules (MUST follow):
1. Answer ONLY using information from the provided CONTEXT SOURCES.
2. Cite sources using [SOURCE N] notation inline. e.g., "According to [SOURCE 1], ..."
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the provided sources to answer this confidently."
4. Never fabricate facts, statistics, or claims not supported by the sources.
5. Be concise but thorough. Use bullet points or sections where appropriate.
6. At the end, list the sources you cited as: "**Sources cited:** [SOURCE 1], [SOURCE 3]..."

Context sources are labeled [SOURCE N] with their URL."""

_USER_TEMPLATE = """CONTEXT SOURCES:
{context}

---

USER QUESTION: {query}

Provide a comprehensive, well-cited answer based solely on the context above."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def generate_answer(query: str, context: str) -> str:
    """Generate a grounded, cited answer for the query."""
    prompt = _USER_TEMPLATE.format(context=context, query=query)
    loop = asyncio.get_event_loop()

    def _sync() -> str:
        response = _client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=prompt,
            config=gentypes.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.1,
                max_output_tokens=2000,
            ),
        )
        return response.text or ""

    answer = await loop.run_in_executor(None, _sync)
    logger.info("answer_generated", chars=len(answer))
    return answer


async def generate_answer_stream(query: str, context: str) -> AsyncGenerator[str, None]:
    """Streaming version — yields text deltas."""
    prompt = _USER_TEMPLATE.format(context=context, query=query)
    loop = asyncio.get_event_loop()

    # Gemini streaming is synchronous; we iterate in a thread and push to a queue
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _sync_stream():
        try:
            for chunk in _client.models.generate_content_stream(
                model=settings.gemini_chat_model,
                contents=prompt,
                config=gentypes.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=2000,
                ),
            ):
                if chunk.text:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _sync_stream)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token
