"""One-shot DB initialiser — run before first startup if not using Docker.

Usage:
    python scripts/init_db.py
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.models import create_tables
from app.utils.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


async def main():
    logger.info("initialising_database")
    await create_tables()
    logger.info("database_ready")


if __name__ == "__main__":
    asyncio.run(main())
