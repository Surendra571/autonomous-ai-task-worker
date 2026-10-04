"""Main FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.config.logging import get_logger, setup_logging
from app.db.session import init_db, get_session_factory
from app.api.routes import router as api_router
from app.finance import finance_router, seed_invoices

setup_logging()
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle."""
    logger.info("Starting Autonomous AI Task Worker", version=settings.APP_VERSION, env=settings.APP_ENV)

    # Initialize storage directories
    settings.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database tables and seed sample data
    try:
        await init_db()
        session_factory = get_session_factory()
        async with session_factory() as session:
            await seed_invoices(session)
    except Exception as e:
        logger.error("Failed to initialize database on startup", error=str(e))

    yield

    logger.info("Shutting down Autonomous AI Task Worker")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Autonomous AI Task Worker capable of executing multi-step business goals with observation, verification, and human-in-the-loop safety.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register main routes and simulated finance application
    application.include_router(api_router)
    application.include_router(finance_router)

    return application


app = create_app()
