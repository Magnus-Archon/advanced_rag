"""SQLAlchemy async models for PostgreSQL + pgvector."""
from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Text, Float, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id          = Column(String(16), primary_key=True)
    url         = Column(Text, nullable=False, index=True)
    title       = Column(Text, default="")
    content     = Column(Text, nullable=False)
    trust_score = Column(Float, default=0.65)
    source_type = Column(String(16), default="web", index=True)  # "web" | "upload"
    embedding   = Column(Vector(768))                             # text-embedding-004
    metadata_   = Column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index(
            "ix_chunks_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
