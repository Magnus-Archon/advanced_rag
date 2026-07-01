"""File ingestion: parse uploaded documents → chunks → embeddings → vector DB.

Supported formats:
  .pdf   — via PyMuPDF (fitz)
  .docx  — via python-docx
  .txt   — plain text
  .md    — markdown (same as txt; heading-aware chunker handles it)
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from app.core.chunker import chunk_document
from app.core.embeddings import embed_texts
from app.core.models import DocumentChunk
from app.db.vector_store import upsert_chunks
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_pdf(data: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def _parse_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _parse_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_PARSERS = {
    ".pdf":  _parse_pdf,
    ".docx": _parse_docx,
    ".txt":  _parse_text,
    ".md":   _parse_text,
}

SUPPORTED_EXTENSIONS = set(_PARSERS.keys())


def _file_url(filename: str, content_hash: str) -> str:
    """Stable synthetic URL that uniquely identifies an uploaded file."""
    return f"upload://{filename}#{content_hash[:8]}"


# ── Public API ────────────────────────────────────────────────────────────────

async def ingest_file(
    filename: str,
    data: bytes,
) -> dict:
    """
    Parse, chunk, embed and store a file. Returns summary dict.
    Raises ValueError for unsupported extensions.
    """
    ext = Path(filename).suffix.lower()
    if ext not in _PARSERS:
        raise ValueError(f"Unsupported file type: {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    # Extract text
    text = _PARSERS[ext](data)
    if not text or len(text.strip()) < 50:
        raise ValueError("File appears empty or unreadable.")

    content_hash = hashlib.sha256(data).hexdigest()
    file_url = _file_url(filename, content_hash)

    # Chunk
    chunks = chunk_document(
        text=text,
        url=file_url,
        title=filename,
        trust_score=1.0,        # user's own documents — full trust
    )
    # Tag all chunks as uploads
    for c in chunks:
        c.source_type = "upload"
        c.metadata["filename"] = filename
        c.metadata["content_hash"] = content_hash
        c.metadata["size_bytes"] = len(data)

    # Embed + upsert
    embeddings = await embed_texts([c.text for c in chunks])
    await upsert_chunks(chunks, embeddings)

    logger.info("file_ingested", filename=filename, chunks=len(chunks), url=file_url)
    return {
        "filename": filename,
        "file_url": file_url,
        "chunks": len(chunks),
        "characters": len(text),
        "content_hash": content_hash,
    }
