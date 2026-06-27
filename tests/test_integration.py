"""Integration tests — require live API keys and running services.

Run with:
    pytest tests/test_integration.py -v -s

Set SKIP_INTEGRATION=1 to skip in CI.
"""
from __future__ import annotations

import os
import pytest

SKIP = os.getenv("SKIP_INTEGRATION", "1") == "1"
pytestmark = pytest.mark.skipif(SKIP, reason="Integration tests skipped (set SKIP_INTEGRATION=0 to run)")


@pytest.mark.asyncio
async def test_query_expansion():
    from app.generation.query_expander import expand_query
    queries = await expand_query("What is quantum computing?")
    assert isinstance(queries, list)
    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)


@pytest.mark.asyncio
async def test_embedding_shape():
    from app.core.embeddings import embed_query, EMBEDDING_DIM
    emb = await embed_query("test query")
    assert isinstance(emb, list)
    assert len(emb) == EMBEDDING_DIM  # 768 for text-embedding-004


@pytest.mark.asyncio
async def test_tavily_search():
    from app.search.providers import TavilySearchProvider
    provider = TavilySearchProvider()
    results = await provider.search("Python programming language", count=3)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_full_pipeline_smoke():
    from app.core.models import SearchRequest
    from app.core.pipeline import run_pipeline

    req = SearchRequest(query="What is the capital of France?", reflect=False, debug=True)
    response = await run_pipeline(req)
    assert response.answer
    assert isinstance(response.sources, list)
    assert "Paris" in response.answer or len(response.answer) > 10
