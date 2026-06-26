"""Quick CLI smoke-test — runs a full pipeline query and prints the result.

Usage:
    python scripts/smoke_test.py "What is retrieval-augmented generation?"
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import get_settings
from app.core.models import SearchRequest
from app.core.pipeline import run_pipeline
from app.utils.logger import configure_logging

configure_logging("INFO")


async def main(query: str):
    print(f"\n🔍 Query: {query}\n{'─' * 60}")
    req = SearchRequest(query=query, reflect=True, debug=True)
    resp = await run_pipeline(req)

    print(f"\n📝 Answer:\n{resp.answer}")
    print(f"\n📚 Sources ({len(resp.sources)}):")
    for i, s in enumerate(resp.sources, 1):
        print(f"  [{i}] {s.title[:60]} — {s.url[:80]} (trust={s.trust_score:.2f})")
    if resp.debug:
        print(f"\n🔧 Debug:")
        print(f"  Expanded queries: {resp.debug.get('expanded_queries', [])}")
        print(f"  Search results: {resp.debug.get('search_results_count', 0)}")
        print(f"  Chunks retrieved: {resp.debug.get('chunks_retrieved', 0)}")
        print(f"  Retrieval hop: {resp.debug.get('hop', 1)}")
        print(f"  Reflection: {resp.debug.get('reflection_note', 'N/A')}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    asyncio.run(main(q))
