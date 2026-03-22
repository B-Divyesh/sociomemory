"""
Memory API endpoints
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from sociomemory.api.deps import AuthRequired, Engine
from sociomemory.models.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    AccessRecord,
    MemoryType,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    request: MemoryCreate,
    engine: Engine,
    _auth: AuthRequired
) -> MemoryResponse:
    """
    Create a new memory.

    - Generates embedding for the content
    - Initializes FSRS state for retrieval optimization
    - Optionally extracts entities and detects relations
    """
    try:
        return await engine.create_memory(request)
    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    user_id: UUID,
    query: str = Query(..., min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    threshold: float = Query(0.0, ge=0.0, le=1.0),  # Default 0.0 for hybrid search
    platforms: Optional[str] = Query(None, description="Comma-separated platform names"),
    memory_types: Optional[str] = Query(None, description="Comma-separated memory types"),
    source_id: Optional[str] = Query(None, description="Filter by source ID (e.g., chat_id for conversation-specific search)"),
    only_latest: bool = Query(True),
    only_valid: bool = Query(True),
    # Search strategy parameters
    mode: str = Query(
        "hyper",
        description="Search mode: 'hyper' (90%+ accuracy, query expansion + HopRAG + CoT reranking, ~$0.08/query), "
                    "'full' (80-85% accuracy, uses GPT-4o ~$0.03/query), "
                    "'hybrid' (BM25+vector, no LLM, ~60% accuracy, free), "
                    "'enhanced' (entity reranking, ~50% accuracy, free), "
                    "'basic' (vector only, ~30% accuracy, free). "
                    "For Q&A use cases, consider POST /api/v1/answers which adds Chain-of-Note reasoning + voting on top of search."
    ),
    similarity_weight: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity vs FSRS scores (0.0-1.0). "
                    "Default 0.5 balances similarity (50%) with FSRS (50%). "
                    "Use 1.0 for pure semantic ranking (best for benchmarks). "
                    "Use 0.5 for production with memory reinforcement."
    ),
    engine: Engine = None,
    _auth: AuthRequired = None
) -> MemorySearchResponse:
    """
    Search memories with configurable accuracy/cost tradeoff.

    **Search Modes:**

    | Mode | Accuracy | Cost | Latency | Description |
    |------|----------|------|---------|-------------|
    | **hyper** | 90%+ | ~$0.08/query | 3-6s | Query expansion + HopRAG + CoT reranking |
    | **full** | 80-85% | ~$0.03/query | 2-4s | Hybrid + GPT-4o reranking |
    | **hybrid** | ~60% | $0 | 200ms | BM25 + Vector + RRF fusion |
    | **enhanced** | ~50% | $0 | 150ms | Vector + entity/keyword bonuses |
    | **basic** | ~30% | $0 | 100ms | Pure vector similarity |

    **For 90%+ accuracy, use mode=hyper.**

    Hyper mode pipeline (state-of-the-art):
    1. Query expansion (generate 2-3 query variants)
    2. Multi-query hybrid search with deduplication
    3. HopRAG-style reasoning for multi-hop questions
    4. Enhanced temporal filtering
    5. Chain-of-Thought reranking
    """
    # Parse comma-separated filters
    platform_list = [p.strip() for p in platforms.split(",")] if platforms else None
    type_list = [MemoryType(t.strip()) for t in memory_types.split(",")] if memory_types else None

    request = MemorySearchRequest(
        user_id=user_id,
        query=query,
        limit=limit,
        threshold=threshold,
        platforms=platform_list,
        memory_types=type_list,
        source_id=source_id,  # Filter by source ID (e.g., chat_id)
        only_latest=only_latest,
        only_valid=only_valid
    )

    try:
        if mode == "hyper":
            # HYPER mode: Maximum accuracy with advanced RAG techniques
            # Expected: 90%+ accuracy
            # Uses: Query expansion + HopRAG + CoT reranking
            return await engine.search_memories_hyper(
                request,
                use_query_expansion=True,
                use_hoprag=True,
                use_temporal_filter=True,
            )
        elif mode == "full":
            # Full LOCAL parity: Hybrid + Temporal + GPT-4o reranking
            # Expected: 80-85% accuracy
            return await engine.search_memories_local_parity(
                request,
                use_hybrid=True,
                use_temporal_filter=True,
                use_llm_reranker=True,
                similarity_weight=similarity_weight,
            )
        elif mode == "hybrid":
            # Hybrid search only (no LLM reranking)
            # Expected: ~60% accuracy, free
            return await engine.search_memories_local_parity(
                request,
                use_hybrid=True,
                use_temporal_filter=True,
                use_llm_reranker=False,
                similarity_weight=similarity_weight,
            )
        elif mode == "enhanced":
            # Entity/keyword reranking (original enhanced)
            return await engine.search_memories_enhanced(request, similarity_weight=similarity_weight)
        else:
            # Basic vector search
            return await engine.search_memories(request, similarity_weight=similarity_weight)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: UUID,
    engine: Engine,
    _auth: AuthRequired
) -> MemoryResponse:
    """Get a single memory by ID"""
    memory = await engine.get_memory(memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    return memory


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    request: MemoryUpdate,
    engine: Engine,
    _auth: AuthRequired
) -> MemoryResponse:
    """
    Update a memory.

    Set invalidate=true to mark the memory as no longer valid (soft delete).
    """
    memory = await engine.update_memory(memory_id, request)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    return memory


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    engine: Engine,
    _auth: AuthRequired
) -> None:
    """
    Soft delete a memory (invalidate it).

    The memory is not actually deleted, but marked with valid_until timestamp
    and is_latest=false.
    """
    deleted = await engine.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )


@router.post("/{memory_id}/access", response_model=MemoryResponse)
async def record_memory_access(
    memory_id: UUID,
    record: AccessRecord,
    engine: Engine,
    _auth: AuthRequired
) -> MemoryResponse:
    """
    Record that a memory was accessed.

    This updates the FSRS state for the memory:
    - was_useful=true: Increases stability (memory is being reinforced)
    - was_useful=false: Decreases stability (memory wasn't helpful)

    Call this when a memory is retrieved and used in generating a response.
    """
    memory = await engine.record_access(
        memory_id=memory_id,
        was_useful=record.was_useful,
        context=record.context
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found"
        )
    return memory
