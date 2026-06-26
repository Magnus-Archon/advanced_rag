# 🔍 RAG Search Engine

A production-grade Retrieval-Augmented Generation (RAG) + agentic web search system.  
Modular, lean, and fully async — no LangChain abstractions.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI  (/api/v1/search  |  /api/v1/search/stream)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │     Semantic Cache (Redis)       │ ◄── cache hit → return early
          └────────────────┬────────────────┘
                           │ miss
          ┌────────────────▼────────────────┐
          │     Query Expander (LLM)        │  → 4 query variants
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Search Aggregator (Brave API)  │  → parallel searches
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Web Fetcher + Trafilatura     │  → async fetch & extract
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Chunker (token-aware, 1000t)  │  → DocumentChunks
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Embeddings (text-embedding-004) │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Vector DB (PostgreSQL+pgvector)│  → upsert chunks
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Hybrid Retrieval               │
          │  (Semantic 70% + BM25 30%)      │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Multi-hop? (if context thin)   │ → follow-up queries + re-retrieve
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Cohere Reranker                │  → top-K most relevant
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Context Builder                │  → dedup, group, token-cap
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  LLM Answer Generation (GPT-4.1)│  → cited, grounded answer
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Reflection / Verification      │  → hallucination check + regen
          └────────────────┬────────────────┘
                           │
                    JSON Response
              { answer, sources, debug }
```

---

## Features

| Feature | Implementation |
|---|---|
| Query expansion | LLM generates 4 query variants |
| Web search | Brave Search API (parallel) |
| Page extraction | `httpx` + `trafilatura` |
| Chunking | Token-aware, heading-preserving, 1000t / 175t overlap |
| Embeddings | OpenAI `text-embedding-004` (768-dim) (768-dim) |
| Vector DB | PostgreSQL + `pgvector` (cosine IVFFlat index) |
| Hybrid retrieval | 70% semantic + 30% BM25 + trust boost |
| Reranking | Cohere `rerank-english-v3.0` (with fallback) |
| Context building | Dedup, group by source, 20k char cap |
| Answer generation | Gemini 2.0 Flash, citation-aware, hallucination-resistant |
| Reflection | Second LLM pass detects bad claims; regenerates if needed |
| Multi-hop retrieval | Follow-up queries if context is thin |
| Semantic cache | Redis embedding-similarity cache (cosine ≥ 0.92) |
| Source trust scoring | Heuristic domain-tier scoring |
| Streaming API | SSE endpoint with stage events + token stream |

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Postgres + Redis)
- API Keys:
  - [Google AI Studio](https://aistudio.google.com/app/apikey) — embeddings + LLM
  - [Brave Search](https://api.search.brave.com/app/subscriptions) — web search
  - [Cohere](https://dashboard.cohere.com/api-keys) — reranking (optional but recommended)

---

## Quickstart

### 1. Clone & configure

```bash
git clone <repo>
cd advanced_rag

cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start infrastructure (Docker)

```bash
docker-compose up -d db redis
# Wait for postgres and redis to be healthy:
docker-compose ps
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Initialise the database

```bash
python scripts/init_db.py
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Test it

```bash
# Smoke-test via CLI
python scripts/smoke_test.py "What is the James Webb Space Telescope?"

# Or via curl
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest advances in fusion energy", "debug": true}'
```

---

## Full Docker deployment (API + DB + Redis)

```bash
cp .env.example .env
# fill in keys

docker-compose up --build
```

The API is then available at `http://localhost:8000`.

---

## API Reference

### `POST /api/v1/search`

**Request:**
```json
{
  "query": "What is retrieval-augmented generation?",
  "multihop": true,
  "reflect": true,
  "debug": false
}
```

**Response:**
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is... [SOURCE 1] ... [SOURCE 2]",
  "sources": [
    {
      "title": "What is RAG? - AWS",
      "url": "https://aws.amazon.com/what-is/retrieval-augmented-generation/",
      "trust_score": 0.85
    }
  ],
  "reflected": true,
  "debug": {
    "expanded_queries": ["...", "...", "...", "..."],
    "search_results_count": 32,
    "chunks_retrieved": 8,
    "hop": 1,
    "reflection_note": "Quality: good. Unsupported: []",
    "top_chunks": [...]
  }
}
```

### `POST /api/v1/search/stream`

Same request body. Returns Server-Sent Events (SSE):

```
event: status
data: {"stage": "expanding_query"}

event: status
data: {"stage": "searching", "queries": [...]}

event: status
data: {"stage": "retrieving"}

event: token
data: Retrieval-Augmented Generation

event: token
data:  (RAG) combines...

event: sources
data: [{"title": "...", "url": "...", "trust_score": 0.85}]

event: done
data: {}
```

### `GET /api/v1/health`

```json
{"status": "ok", "model": "gemini-2.0-flash"}
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | required | Gemini API key |
| `GEMINI_CHAT_MODEL=gemini-2.0-flash
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-004` (768-dim) | Embedding model |
| `BRAVE_API_KEY` | required | Brave Search API key |
| `BRAVE_SEARCH_COUNT` | `10` | Results per query |
| `COHERE_API_KEY` | optional | Cohere rerank key |
| `DATABASE_URL` | see example | PostgreSQL async URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL |
| `CACHE_TTL_SECONDS` | `3600` | Semantic cache TTL |
| `MAX_CONTEXT_CHARS` | `20000` | Max LLM context size |
| `TOP_K_RETRIEVAL` | `20` | Candidates for retrieval |
| `TOP_K_RERANK` | `8` | Final chunks after rerank |
| `REFLECTION_ENABLED` | `true` | Enable reflection pass |
| `MULTIHOP_ENABLED` | `true` | Enable multi-hop retrieval |

---

## Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/test_unit.py -v

# Integration tests (requires live keys + running services)
SKIP_INTEGRATION=0 pytest tests/test_integration.py -v -s
```

---

## Project Structure

```
advanced_rag/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── config.py                # Pydantic settings
│   ├── api/
│   │   └── routes.py            # /search, /search/stream, /health
│   ├── core/
│   │   ├── models.py            # Shared Pydantic models
│   │   ├── pipeline.py          # Main orchestrator
│   │   ├── chunker.py           # Document chunking
│   │   ├── embeddings.py        # OpenAI embeddings
│   │   └── cache.py             # Semantic Redis cache
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM + pgvector
│   │   └── vector_store.py      # Upsert + cosine search
│   ├── search/
│   │   ├── providers.py         # Brave Search + aggregator
│   │   └── fetcher.py           # Async web fetcher + trafilatura
│   ├── retrieval/
│   │   ├── hybrid.py            # Semantic + BM25 hybrid
│   │   ├── reranker.py          # Cohere rerank
│   │   └── context_builder.py   # Context assembly
│   ├── generation/
│   │   ├── query_expander.py    # LLM query expansion
│   │   ├── answerer.py          # Citation-aware LLM answerer
│   │   ├── reflector.py         # Hallucination check + regen
│   │   └── followup.py          # Multi-hop follow-up queries
│   └── utils/
│       ├── logger.py            # Structured logging
│       ├── tokens.py            # tiktoken utilities
│       └── trust.py             # Domain trust scoring
├── scripts/
│   ├── init_db.py               # DB initialiser
│   └── smoke_test.py            # CLI end-to-end test
├── tests/
│   ├── test_unit.py             # Pure unit tests
│   └── test_integration.py      # Live integration tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Extending the System

### Add a new search provider

Implement `BaseSearchProvider` in `app/search/providers.py`:

```python
class MyProvider(BaseSearchProvider):
    async def search(self, query: str, count: int = 10) -> list[SearchResult]:
        ...
```

Pass it to `SearchAggregator(provider=MyProvider())`.

### Swap the vector DB

Replace `app/db/vector_store.py` with any store that implements `upsert_chunks` and `semantic_search`.

### Use a different reranker

Replace `app/retrieval/reranker.py`; the interface is just `rerank(query, chunks, top_k) → list[RankedChunk]`.

---

## Cost Estimates (per query, rough)

| Component | Cost |
|---|---|
| Query embedding | ~$0.00013 (768-dim) |
| 4× Brave searches | $0.005 (paid tier) |
| 100 chunk embeddings | ~$0.013 |
| Gemini answer (2k tokens) | ~$0.012 |
| Cohere rerank (8 docs) | ~$0.001 |
| Reflection pass | ~$0.008 |
| **Total (cold)** | **~$0.04 / query** |
| **Total (cache hit)** | **~$0.00013** |

Cache hit rate rises quickly for repeated/similar queries.
