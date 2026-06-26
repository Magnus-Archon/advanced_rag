"""Unit tests — no external API calls needed."""
from __future__ import annotations

import pytest

from app.core.chunker import chunk_document, _split_by_headings
from app.utils.tokens import chunk_text, count_tokens, truncate_to_tokens
from app.utils.trust import score_domain_trust


# ── Token utilities ───────────────────────────────────────────────────────────

def test_count_tokens_basic():
    assert count_tokens("Hello world") > 0


def test_chunk_text_splits_correctly():
    long_text = "word " * 3000
    chunks = chunk_text(long_text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c) <= 520  # small buffer for edge cases


def test_truncate_to_tokens():
    text = "a " * 2000
    truncated = truncate_to_tokens(text, max_tokens=100)
    assert count_tokens(truncated) <= 100


# ── Chunker ───────────────────────────────────────────────────────────────────

def test_split_by_headings_finds_sections():
    text = "Intro text\n\n## Section A\nContent A\n\n## Section B\nContent B"
    parts = _split_by_headings(text)
    assert len(parts) >= 2


def test_chunk_document_basic():
    text = "This is a test document. " * 200
    chunks = chunk_document(text, url="https://example.com", title="Test")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.url == "https://example.com"
        assert c.chunk_id
        assert len(c.text) > 0


def test_chunk_document_preserves_headings():
    text = "## Introduction\nSome intro text.\n\n## Methods\nSome methods text. " * 20
    chunks = chunk_document(text, url="https://test.com", title="Paper")
    assert any("##" in c.text or "Introduction" in c.text for c in chunks)


# ── Trust scoring ─────────────────────────────────────────────────────────────

def test_trust_wikipedia():
    assert score_domain_trust("https://en.wikipedia.org/wiki/Python") == 1.0


def test_trust_gov():
    assert score_domain_trust("https://www.cdc.gov/flu") == 1.0


def test_trust_edu():
    assert score_domain_trust("https://mit.edu/research") == 1.0


def test_trust_reuters():
    assert score_domain_trust("https://www.reuters.com/article/abc") == 0.85


def test_trust_penalty():
    assert score_domain_trust("https://www.pinterest.com/pin/123") == 0.4


def test_trust_unknown():
    score = score_domain_trust("https://some-random-blog.io/post")
    assert 0.0 < score <= 0.75


# ── Context builder (no DB needed) ───────────────────────────────────────────

def test_context_builder_deduplication():
    from app.core.models import DocumentChunk, RankedChunk
    from app.retrieval.context_builder import build_context

    chunk = DocumentChunk(
        chunk_id="abc123",
        url="https://example.com",
        title="Example",
        text="Some content about Python.",
        trust_score=0.7,
    )
    # Duplicate chunks
    ranked = [
        RankedChunk(chunk=chunk, score=0.9),
        RankedChunk(chunk=chunk, score=0.85),  # same chunk_id → should be deduped
    ]
    context, sources = build_context(ranked)
    assert context.count("[SOURCE 1]") == 1
    assert len(sources) == 1


def test_context_builder_token_limit():
    from app.core.models import DocumentChunk, RankedChunk
    from app.retrieval.context_builder import build_context

    chunks = [
        RankedChunk(
            chunk=DocumentChunk(
                chunk_id=f"chunk{i}",
                url=f"https://example{i}.com",
                title=f"Source {i}",
                text="A " * 5000,  # Very long content
                trust_score=0.7,
            ),
            score=0.9 - i * 0.01,
        )
        for i in range(20)
    ]
    context, sources = build_context(chunks)
    assert len(context) <= 21000  # max_context_chars + small buffer
