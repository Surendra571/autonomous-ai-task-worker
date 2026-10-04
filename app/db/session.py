"""Database engine and session management with async SQLAlchemy."""

import os
from pathlib import Path
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from app.config.settings import settings
from app.config.logging import get_logger
from app.models.base import Base

logger = get_logger("db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Lazily initialize and return the async engine."""
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL

        # Ensure sqlite parent directory exists if using sqlite
        if "sqlite" in db_url:
            sqlite_path = db_url.split("///")[-1]
            if sqlite_path and not sqlite_path.startswith(":"):
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing database engine", url=db_url.split("@")[-1])
        _engine = create_async_engine(
            db_url,
            echo=settings.DEBUG,
            future=True,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database sessions."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connection() -> bool:
    """Verifies that the database is reachable."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database connectivity check failed", error=str(e))
        return False


async def init_db() -> None:
    """Initializes tables in the database."""
    global _engine, _session_factory
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        if settings.USE_SQLITE_FALLBACK and "postgresql" in settings.DATABASE_URL:
            logger.warning(
                "PostgreSQL connection failed during init, falling back to local SQLite",
                fallback=settings.FALLBACK_SQLITE_URL,
                error=str(e),
            )
            # Switch to sqlite fallback
            settings.DATABASE_URL = settings.FALLBACK_SQLITE_URL
            if _engine is not None:
                await _engine.dispose()
            _engine = None
            _session_factory = None

            # Re-init with sqlite
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Fallback SQLite database tables initialized successfully")
        else:
            raise e

