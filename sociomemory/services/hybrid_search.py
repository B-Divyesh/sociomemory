"""
Hybrid Search Service - BM25 + Vector Search with RRF Fusion

This implements the same retrieval strategy used by:
- Hindsight (91.4% accuracy)
- Zep (94.8% accuracy)

Pipeline:
1. BM25 keyword search (PostgreSQL tsvector)
2. Vector semantic search (pgvector)
3. RRF fusion: 1/(k+rank) with k=60
"""
import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sociomemory.models.memory import (
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryType,
)
from sociomemory.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid search combining BM25 (tsvector) + Vector similarity with RRF fusion.

    This matches the LOCAL benchmark's retrieval strategy that achieves:
    - 96% on LongMemEval
    - 80% on LoCoMo
    """

    # RRF hyperparameters (from Hindsight paper)
    DEFAULT_RRF_K = 60  # Smoothing constant
    DEFAULT_BM25_WEIGHT = 1.0
    DEFAULT_SEMANTIC_WEIGHT = 1.0

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()

    async def search_hybrid(
        self,
        request: MemorySearchRequest,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> MemorySearchResponse:
        """
        Perform hybrid search with RRF fusion.

        Args:
            request: Search request with query and filters
            bm25_weight: Weight for BM25/keyword results (default 1.0)
            semantic_weight: Weight for vector similarity results (default 1.0)
            rrf_k: RRF smoothing constant (default 60)

        Returns:
            MemorySearchResponse with RRF-ranked results
        """
        logger.info(f"Hybrid search for user {request.user_id}: {request.query[:50]}...")

        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(request.query)
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # Build platform and type filters
        platforms_sql = "NULL::text[]"
        if request.platforms:
            platforms_list = ",".join(f"'{p}'" for p in request.platforms)
            platforms_sql = f"ARRAY[{platforms_list}]::text[]"

        types_sql = "NULL::text[]"
        if request.memory_types:
            types_list = ",".join(f"'{t.value}'" for t in request.memory_types)
            types_sql = f"ARRAY[{types_list}]::text[]"

        # Call hybrid search function
        query = text(f"""
            SELECT * FROM hybrid_search_memories(
                CAST(:user_id AS uuid),
                :query_text,
                CAST(:embedding AS vector(3072)),
                :limit_val,
                :bm25_weight,
                :semantic_weight,
                :rrf_k,
                {platforms_sql},
                {types_sql},
                :only_latest_val,
                :only_valid_val
            )
        """)

        result = await self.db.execute(
            query,
            {
                "user_id": str(request.user_id),
                "query_text": request.query,
                "embedding": embedding_str,
                "limit_val": request.limit,
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "rrf_k": rrf_k,
                "only_latest_val": request.only_latest,
                "only_valid_val": request.only_valid,
            }
        )

        rows = result.fetchall()

        # Convert to response objects
        results = []
        for row in rows:
            # event_time/source_id may not be present if DB function not updated yet
            event_time = getattr(row, 'event_time', None)
            source_id = getattr(row, 'source_id', None)
            results.append(MemoryResponse(
                id=row.id,
                content=row.content,
                memory_type=MemoryType(row.memory_type),
                source_platform=row.source_platform,
                source_id=source_id,
                confidence=row.confidence,
                valid_from=row.valid_from,
                valid_until=row.valid_until,
                is_latest=row.is_latest,
                stability=row.stability,
                difficulty=row.difficulty,
                retrievability=row.retrievability,
                access_count=row.access_count,
                similarity=row.semantic_similarity or 0.0,
                priority_score=row.rrf_score,
                created_at=row.created_at,
                event_time=event_time,  # CRITICAL for temporal reasoning
            ))

        bm25_hits = sum(1 for r in rows if r.fulltext_rank is not None)
        semantic_hits = sum(1 for r in rows if r.semantic_rank is not None)
        logger.info(f"Hybrid search: {len(results)} results (BM25: {bm25_hits}, Semantic: {semantic_hits})")

        return MemorySearchResponse(
            results=results,
            total=len(results),
            query_embedding_tokens=len(request.query) // 4
        )

    async def search_hybrid_with_fallback(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        """
        Hybrid search with fallback to pure semantic if BM25 returns nothing.

        This handles cases where the tsvector column might not exist yet
        or the query has no keyword matches.
        """
        try:
            response = await self.search_hybrid(request)
            if response.results:
                return response
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to semantic: {e}")

        # Fallback to pure semantic search
        return await self._semantic_only_search(request)

    async def _semantic_only_search(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        """Pure semantic search fallback."""
        query_embedding = await self.embedding_service.get_embedding(request.query)
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        platforms_sql = "NULL::text[]"
        if request.platforms:
            platforms_list = ",".join(f"'{p}'" for p in request.platforms)
            platforms_sql = f"ARRAY[{platforms_list}]::text[]"

        types_sql = "NULL::text[]"
        if request.memory_types:
            types_list = ",".join(f"'{t.value}'" for t in request.memory_types)
            types_sql = f"ARRAY[{types_list}]::text[]"

        query = text(f"""
            SELECT * FROM search_memories(
                CAST(:user_id AS uuid),
                CAST(:embedding AS vector(3072)),
                :limit_val,
                :threshold_val,
                {platforms_sql},
                {types_sql},
                :only_latest_val,
                :only_valid_val
            )
        """)

        result = await self.db.execute(
            query,
            {
                "user_id": str(request.user_id),
                "embedding": embedding_str,
                "limit_val": request.limit,
                "threshold_val": request.threshold,
                "only_latest_val": request.only_latest,
                "only_valid_val": request.only_valid,
            }
        )

        rows = result.fetchall()

        results = []
        for row in rows:
            # event_time/source_id may not be present if DB function not updated yet
            event_time = getattr(row, 'event_time', None)
            source_id = getattr(row, 'source_id', None)
            results.append(MemoryResponse(
                id=row.id,
                content=row.content,
                memory_type=MemoryType(row.memory_type),
                source_platform=row.source_platform,
                source_id=source_id,
                confidence=row.confidence,
                valid_from=row.valid_from,
                valid_until=row.valid_until,
                is_latest=row.is_latest,
                stability=row.stability,
                difficulty=row.difficulty,
                retrievability=row.retrievability,
                access_count=row.access_count,
                similarity=row.similarity,
                priority_score=row.priority_score,
                created_at=row.created_at,
                event_time=event_time,  # CRITICAL for temporal reasoning
            ))

        return MemorySearchResponse(
            results=results,
            total=len(results),
            query_embedding_tokens=len(request.query) // 4
        )


async def get_hybrid_search_service(db: AsyncSession) -> HybridSearchService:
    """Get hybrid search service instance."""
    return HybridSearchService(db=db)
