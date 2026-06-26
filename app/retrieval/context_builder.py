"""Build the final grounded context string for the LLM.

Rules:
  - Deduplicate by chunk_id
  - Group by source URL
  - Cap total at MAX_CONTEXT_CHARS
  - Each source block is annotated with [SOURCE N] for easy citation
"""
from __future__ import annotations

from app.config import get_settings
from app.core.models import RankedChunk, SourceCitation
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_context(
    ranked_chunks: list[RankedChunk],
) -> tuple[str, list[SourceCitation]]:
    """
    Returns:
        context_str: formatted context block
        sources: deduplicated ordered list of SourceCitation
    """
    # Deduplicate chunks
    seen_ids: set[str] = set()
    deduped: list[RankedChunk] = []
    for rc in ranked_chunks:
        if rc.chunk.chunk_id not in seen_ids:
            seen_ids.add(rc.chunk.chunk_id)
            deduped.append(rc)

    # Group by URL (preserve insertion order)
    url_to_chunks: dict[str, list[RankedChunk]] = {}
    for rc in deduped:
        url_to_chunks.setdefault(rc.chunk.url, []).append(rc)

    sources: list[SourceCitation] = []
    blocks: list[str] = []
    total_chars = 0
    source_index = 1

    for url, chunks in url_to_chunks.items():
        title = chunks[0].chunk.title or url
        trust = chunks[0].chunk.trust_score

        source_text = "\n\n".join(rc.chunk.text for rc in chunks)
        block = f"[SOURCE {source_index}] {title}\nURL: {url}\n\n{source_text}"

        if total_chars + len(block) > settings.max_context_chars:
            # Try to fit a trimmed version
            remaining = settings.max_context_chars - total_chars
            if remaining < 200:
                break
            block = block[:remaining]

        blocks.append(block)
        sources.append(SourceCitation(title=title, url=url, trust_score=trust))
        total_chars += len(block)
        source_index += 1

        if total_chars >= settings.max_context_chars:
            break

    context = "\n\n---\n\n".join(blocks)
    logger.info("context_built", sources=len(sources), chars=total_chars)
    return context, sources
