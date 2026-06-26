"""Token counting and text utilities.

Uses tiktoken when available; falls back to a character-based heuristic
(4 chars ≈ 1 token) so tests and offline environments still work.
"""
from __future__ import annotations

from typing import Optional

_enc = None


def _get_enc():
    global _enc
    if _enc is not None:
        return _enc
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _enc = None
    return _enc


# ── Public helpers ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    enc = _get_enc()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)  # 4 chars ≈ 1 token heuristic


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _get_enc()
    if enc is not None:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    # fallback: char-based
    char_limit = max_tokens * 4
    return text[:char_limit]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 175,
) -> list[str]:
    """Split text into overlapping token-based chunks."""
    enc = _get_enc()
    if enc is not None:
        tokens = enc.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += chunk_size - overlap
        return chunks

    # fallback: character-based splitting
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + char_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += char_size - char_overlap
    return chunks
