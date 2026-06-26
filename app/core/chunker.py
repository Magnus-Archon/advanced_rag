"""Semantic-aware document chunking.

Strategy:
  1. Split on Markdown / HTML headings to preserve section context.
  2. Further split oversized sections by token count with overlap.
  3. Each chunk carries its source URL and a stable chunk_id.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

from app.core.models import DocumentChunk
from app.utils.tokens import chunk_text, count_tokens
from app.utils.logger import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"(?m)^#{1,3} .+$")

# Token targets
CHUNK_SIZE = 1000
OVERLAP = 175


def _split_by_headings(text: str) -> list[str]:
    """Split text on markdown headings, keeping heading in each section."""
    parts: list[str] = []
    last = 0
    for m in _HEADING_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()].strip())
        last = m.start()
    parts.append(text[last:].strip())
    return [p for p in parts if p]


def _make_chunk_id(url: str, index: int) -> str:
    raw = f"{url}::chunk::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_document(
    text: str,
    url: str,
    title: str = "",
    trust_score: float = 0.65,
) -> list[DocumentChunk]:
    """Convert a document text into DocumentChunk objects."""
    sections = _split_by_headings(text)
    raw_chunks: list[str] = []

    for section in sections:
        if count_tokens(section) <= CHUNK_SIZE:
            raw_chunks.append(section)
        else:
            raw_chunks.extend(chunk_text(section, CHUNK_SIZE, OVERLAP))

    chunks: list[DocumentChunk] = []
    for i, raw in enumerate(raw_chunks):
        stripped = raw.strip()
        if not stripped:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(url, i),
                url=url,
                title=title,
                text=stripped,
                trust_score=trust_score,
                metadata={"section_index": i},
            )
        )

    logger.debug("chunked_document", url=url[:60], n_chunks=len(chunks))
    return chunks
