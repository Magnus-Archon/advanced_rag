"""Reflection layer: Gemini second pass to detect hallucinations and unsupported claims.

If issues are found, an improved answer is generated.
"""
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

_REFLECT_SYSTEM = """You are a rigorous fact-checker for AI-generated answers.

Given:
- CONTEXT: the source documents used to generate the answer
- ANSWER: the generated answer

Your job:
1. Identify any claims in the answer NOT supported by the context (potential hallucinations).
2. Identify any citations [SOURCE N] that are misused or inaccurate.
3. Rate answer quality: "good", "minor_issues", "major_issues"

Respond ONLY with JSON (no markdown fences):
{
  "quality": "good" | "minor_issues" | "major_issues",
  "unsupported_claims": ["..."],
  "citation_errors": ["..."],
  "should_regenerate": true | false
}"""

_REGEN_SYSTEM = """You are a precise, citation-driven research assistant.
The previous answer had issues. Generate a corrected answer using ONLY the context provided.
Be conservative — if unsure, say you don't have enough information rather than guessing."""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
async def reflect_and_verify(
    query: str,
    answer: str,
    context: str,
) -> tuple[str, str]:
    """
    Returns:
        final_answer: the (possibly improved) answer
        note: reflection summary
    """
    loop = asyncio.get_event_loop()

    # ── Step 1: reflection pass ───────────────────────────────────────────────
    def _reflect_sync() -> str:
        response = _client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=f"CONTEXT:\n{context[:8000]}\n\nANSWER:\n{answer}",
            config=gentypes.GenerateContentConfig(
                system_instruction=_REFLECT_SYSTEM,
                temperature=0.0,
                max_output_tokens=600,
            ),
        )
        return response.text or "{}"

    try:
        raw = await loop.run_in_executor(None, _reflect_sync)
        raw = re.sub(r"```json|```", "", raw).strip()
        reflection = json.loads(raw)
    except Exception as exc:
        logger.warning("reflection_parse_error", error=str(exc))
        return answer, "Reflection skipped due to parse error."

    quality = reflection.get("quality", "good")
    should_regen = reflection.get("should_regenerate", False)
    unsupported = reflection.get("unsupported_claims", [])
    note = f"Quality: {quality}. Unsupported: {unsupported}"

    logger.info("reflection_done", quality=quality, should_regen=should_regen)

    if not should_regen or quality == "good":
        return answer, note

    # ── Step 2: regeneration ──────────────────────────────────────────────────
    try:
        issues = "\n".join(f"- {c}" for c in unsupported)
        regen_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"ORIGINAL ANSWER (had issues):\n{answer}\n\n"
            f"IDENTIFIED PROBLEMS:\n{issues}\n\n"
            f"QUESTION: {query}\n\n"
            "Please provide a corrected, conservative answer citing only what the context supports."
        )

        def _regen_sync() -> str:
            response = _client.models.generate_content(
                model=settings.gemini_chat_model,
                contents=regen_prompt,
                config=gentypes.GenerateContentConfig(
                    system_instruction=_REGEN_SYSTEM,
                    temperature=0.05,
                    max_output_tokens=2000,
                ),
            )
            return response.text or answer

        improved = await loop.run_in_executor(None, _regen_sync)
        logger.info("answer_regenerated")
        return improved, note + " | Answer regenerated."
    except Exception as exc:
        logger.warning("regen_error", error=str(exc))
        return answer, note
