"""Run all pending DB migrations.

Safe to run multiple times (idempotent).

Usage:
    python scripts/migrate.py
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlalchemy as sa
from app.db.models import engine
from app.utils.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


MIGRATIONS = [
    # v1.1 — add source_type column
    (
        "add_source_type",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'source_type'
            ) THEN
                ALTER TABLE chunks
                    ADD COLUMN source_type VARCHAR(16) NOT NULL DEFAULT 'web';
                CREATE INDEX IF NOT EXISTS ix_chunks_source_type
                    ON chunks (source_type);
                RAISE NOTICE 'Migration add_source_type: applied';
            ELSE
                RAISE NOTICE 'Migration add_source_type: already applied, skipping';
            END IF;
        END$$;
        """
    ),
]


async def main() -> None:
    async with engine.begin() as conn:
        for name, sql in MIGRATIONS:
            logger.info("running_migration", name=name)
            await conn.execute(sa.text(sql))
            logger.info("migration_done", name=name)
    logger.info("all_migrations_complete")


if __name__ == "__main__":
    asyncio.run(main())
