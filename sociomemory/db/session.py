"""
Database session management for SocioMemory

Supabase Connection Modes:
- Session Mode (port 5432): Limited connections (15-20 max), supports prepared statements
- Transaction Mode (port 6543): High concurrency (200+ clients), NO prepared statements

For production with high concurrency, use Transaction Mode (port 6543) in DATABASE_URL.
Example: postgresql+asyncpg://user:pass@db.xxx.supabase.co:6543/postgres

References:
- https://supabase.com/docs/guides/database/connection-management
- https://supabase.com/docs/guides/troubleshooting/supavisor-faq-YyP5tI
"""
import logging
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
from sqlalchemy.pool import NullPool

from sociomemory.config import get_settings

logger = logging.getLogger(__name__)

# Lazy-initialized engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def _is_transaction_mode(database_url: str) -> bool:
    """Check if database URL uses Supabase Transaction Mode (port 6543)."""
    parsed = urlparse(database_url.replace("+asyncpg", ""))
    return parsed.port == 6543


def get_engine() -> AsyncEngine:
    """Get or create the async engine.

    Connection pooling strategy:
    - Transaction Mode (port 6543): Use NullPool - let Supavisor handle pooling
      This avoids MaxClientsInSessionMode errors and scales to 200+ clients
    - Session Mode (port 5432): Use SQLAlchemy pooling with conservative limits

    For Transaction Mode, prepared statements MUST be disabled since Supavisor
    doesn't support them. This is handled via connect_args.

    Based on:
    - https://docs.sqlalchemy.org/en/20/core/pooling.html
    - https://supabase.com/docs/guides/database/connection-management
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url

        if _is_transaction_mode(db_url):
            # Transaction Mode: Let Supavisor handle all connection pooling
            # NullPool = no SQLAlchemy pooling, each request gets fresh connection
            # prepared_statement_cache_size=0 disables prepared statements (required for Supavisor)
            logger.info("Using Supabase Transaction Mode (port 6543) - Supavisor handles pooling")
            _engine = create_async_engine(
                db_url,
                echo=settings.api_debug,
                poolclass=NullPool,  # Let Supavisor pool, not SQLAlchemy
                connect_args={
                    "prepared_statement_cache_size": 0,  # Disable prepared statements
                    "statement_cache_size": 0,  # Also disable statement cache
                },
            )
        else:
            # Session Mode: Use SQLAlchemy pooling with conservative limits
            # Must stay within Supabase's pool_size limit (typically 15-20)
            logger.info("Using Supabase Session Mode (port 5432) - SQLAlchemy handles pooling")
            _engine = create_async_engine(
                db_url,
                echo=settings.api_debug,
                pool_pre_ping=True,
                pool_size=10,         # Conservative: stay under Supabase Session limit
                max_overflow=5,       # Allow small burst, total max: 15
                pool_timeout=60,      # Wait up to 60s for connection
                pool_recycle=1800,    # Recycle every 30 min
            )
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


# Backwards compatibility aliases
@property
def engine() -> AsyncEngine:
    """Engine property for backwards compatibility."""
    return get_engine()


@property
def AsyncSessionLocal() -> async_sessionmaker:
    """Session factory property for backwards compatibility."""
    return get_session_factory()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
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


async def init_db() -> None:
    """Initialize database - create tables."""
    from sociomemory.db.models import Base

    async with get_engine().begin() as conn:
        # Create pgvector extension if not exists
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


def reset_engine() -> None:
    """Reset engine and session factory (for testing)."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None


def set_engine(new_engine: AsyncEngine) -> None:
    """Set a custom engine (for testing)."""
    global _engine, _session_factory
    _engine = new_engine
    _session_factory = async_sessionmaker(
        new_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
