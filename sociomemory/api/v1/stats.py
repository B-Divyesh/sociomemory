"""
Memory Stats API endpoints
"""
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from sociomemory.api.deps import AuthRequired, Engine
from sociomemory.models.memory import MemoryStats

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/{user_id}", response_model=MemoryStats)
async def get_memory_stats(
    user_id: UUID,
    engine: Engine,
    _auth: AuthRequired
) -> MemoryStats:
    """
    Get memory statistics for a user.

    Returns:
        - total_memories: Total number of memories
        - by_type: Count by memory type (preference, behavior, fact, etc.)
        - by_platform: Count by source platform
        - avg_stability: Average FSRS stability score
        - avg_retrievability: Average FSRS retrievability score
        - total_accesses: Total memory access count
        - memories_accessed_today: Memories accessed in last 24h
        - oldest_memory: Timestamp of oldest memory
        - newest_memory: Timestamp of newest memory
    """
    try:
        return await engine.get_stats(user_id)
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{user_id}/health")
async def check_memory_health(
    user_id: UUID,
    engine: Engine,
    _auth: AuthRequired
) -> dict:
    """
    Check health status of user's memory system.

    Returns indicators for:
    - Memory count status (good/warning/empty)
    - Memory freshness (based on newest memory age)
    - Retrieval performance (based on avg retrievability)
    """
    try:
        stats = await engine.get_stats(user_id)

        # Determine health indicators
        memory_status = "empty"
        if stats.total_memories > 100:
            memory_status = "good"
        elif stats.total_memories > 0:
            memory_status = "growing"

        retrieval_status = "unknown"
        if stats.avg_retrievability >= 0.7:
            retrieval_status = "excellent"
        elif stats.avg_retrievability >= 0.5:
            retrieval_status = "good"
        elif stats.avg_retrievability >= 0.3:
            retrieval_status = "moderate"
        elif stats.avg_retrievability > 0:
            retrieval_status = "poor"

        freshness_status = "unknown"
        if stats.memories_accessed_today > 0:
            freshness_status = "active"
        elif stats.total_accesses > 0:
            freshness_status = "dormant"

        return {
            "user_id": str(user_id),
            "total_memories": stats.total_memories,
            "health": {
                "memory_status": memory_status,
                "retrieval_status": retrieval_status,
                "freshness_status": freshness_status,
            },
            "recommendations": _get_recommendations(
                memory_status, retrieval_status, freshness_status, stats
            )
        }
    except Exception as e:
        logger.error(f"Failed to check health for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _get_recommendations(
    memory_status: str,
    retrieval_status: str,
    freshness_status: str,
    stats: MemoryStats
) -> list[str]:
    """Generate recommendations based on health indicators."""
    recommendations = []

    if memory_status == "empty":
        recommendations.append("Start adding memories from your conversations")
    elif memory_status == "growing":
        recommendations.append("Memory system is building up - continue adding memories")

    if retrieval_status == "poor":
        recommendations.append("Consider reviewing and reinforcing important memories")
    elif retrieval_status == "moderate":
        recommendations.append("Access your memories more frequently to improve retrieval")

    if freshness_status == "dormant":
        recommendations.append("Your memories haven't been accessed recently - consider reviewing them")

    # Platform diversity
    if stats.by_platform and len(stats.by_platform) == 1:
        recommendations.append("Consider adding memories from other platforms for better context")

    return recommendations
