"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db.models import create_tables
from app.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("startup", env=settings.app_env, model=settings.gemini_chat_model)
    await create_tables()
    logger.info("database_ready")
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Search Engine",
        description="Advanced RAG + agentic web search",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
