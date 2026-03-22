"""
API dependencies for SocioMemory
"""
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from sociomemory.config import get_settings
from sociomemory.db.session import get_db
from sociomemory.services.memory_engine import MemoryEngine

settings = get_settings()


async def verify_api_key(x_api_key: Annotated[Optional[str], Header()] = None) -> bool:
    """
    Verify API key if configured.

    If API_KEY is not set in config, all requests are allowed.
    """
    if not settings.api_key:
        return True

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True


async def get_memory_engine(
    db: Annotated[AsyncSession, Depends(get_db)]
) -> MemoryEngine:
    """Get memory engine instance with database session"""
    return MemoryEngine(db=db)


# Type aliases for dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db)]
AuthRequired = Annotated[bool, Depends(verify_api_key)]
Engine = Annotated[MemoryEngine, Depends(get_memory_engine)]
