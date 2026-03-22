"""
API v1 module
"""
from fastapi import APIRouter

from sociomemory.api.v1.memories import router as memories_router
from sociomemory.api.v1.stats import router as stats_router
from sociomemory.api.v1.answers import router as answers_router

router = APIRouter(prefix="/v1")
router.include_router(memories_router)
router.include_router(stats_router)
router.include_router(answers_router)

__all__ = ["router"]
