"""
Answer API endpoint - generates answers from memories using Chain-of-Note reasoning.

v37: Added CRAG corrective re-search and aggregation limit boost.
"""
import logging

from fastapi import APIRouter, HTTPException, status

from sociomemory.api.deps import AuthRequired, Engine
from sociomemory.models.answer import AnswerRequest, AnswerResponse
from sociomemory.models.memory import MemorySearchRequest
from sociomemory.services.answer_service import AnswerService, detect_question_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/answers", tags=["answers"])

# Aggregation questions benefit from more results to avoid under-counting
AGGREGATION_SEARCH_LIMIT = 20


def _results_to_dicts(search_response) -> list[dict]:
    """Convert search response results to dicts for answer service."""
    return [
        {
            "id": str(r.id),
            "content": r.content,
            "memory_type": r.memory_type,
            "source_platform": r.source_platform,
            "source_id": r.source_id,
            "confidence": r.confidence,
            "similarity": r.similarity,
            "priority_score": r.priority_score,
            "valid_from": r.valid_from.isoformat() if r.valid_from else None,
            "valid_until": r.valid_until.isoformat() if r.valid_until else None,
            "is_latest": r.is_latest,
            "stability": r.stability,
            "difficulty": r.difficulty,
            "retrievability": r.retrievability,
            "access_count": r.access_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "event_time": r.event_time.isoformat() if r.event_time else None,
        }
        for r in search_response.results
    ]


def _merge_results(original: list[dict], new_results: list[dict]) -> list[dict]:
    """Merge search results, deduplicating by memory ID (keep highest similarity)."""
    seen: dict[str, dict] = {}
    for r in original:
        mid = r.get("id", "")
        if mid not in seen or (r.get("similarity", 0) or 0) > (seen[mid].get("similarity", 0) or 0):
            seen[mid] = r
    for r in new_results:
        mid = r.get("id", "")
        if mid not in seen or (r.get("similarity", 0) or 0) > (seen[mid].get("similarity", 0) or 0):
            seen[mid] = r
    # Return sorted by similarity descending
    merged = list(seen.values())
    merged.sort(key=lambda r: r.get("similarity", 0) or 0, reverse=True)
    return merged


async def _do_search(engine, user_id, query: str, limit: int, mode: str) -> list[dict]:
    """Execute a search and return results as dicts."""
    search_request = MemorySearchRequest(
        user_id=user_id,
        query=query,
        limit=limit,
        threshold=0.0,
        only_latest=True,
        only_valid=True,
    )
    if mode == "hyper":
        search_response = await engine.search_memories_hyper(search_request)
    else:
        search_response = await engine.search_memories(search_request)
    return _results_to_dicts(search_response)


@router.post("", response_model=AnswerResponse)
async def generate_answer(
    request: AnswerRequest,
    engine: Engine,
    _auth: AuthRequired,
) -> AnswerResponse:
    """
    Search memories and generate an answer with Chain-of-Note reasoning.

    This endpoint combines search + answer generation in a single call.
    It uses the same intelligence pipeline that achieved 85% on LongMemEval:
    - Question type detection (temporal, aggregation, preference, KU)
    - Chain-of-Note structured reasoning for complex queries
    - Simple prompt for straightforward factual questions
    - Self-consistency voting (3 votes + consensus) for complex questions
    - Entity verification to prevent hallucination
    - CRAG corrective re-search when answer is insufficient (v37)
    - Aggregation limit boost to 20 results (v37)

    If `search_results` is provided in the request, the search step is skipped
    and those results are used directly. This saves ~15s latency and ensures
    consistency when results were already retrieved.
    """
    try:
        # Detect question type early for aggregation boost
        q_type = detect_question_type(request.question)
        effective_limit = request.search_limit
        if q_type.is_aggregation:
            effective_limit = max(request.search_limit, AGGREGATION_SEARCH_LIMIT)

        if request.search_results is not None:
            results_dicts = request.search_results
            logger.info(
                f"Using {len(results_dicts)} pre-computed search results for answer generation"
            )

            # For aggregation with pre-computed results, do supplementary search
            # if we have fewer results than the aggregation limit
            if q_type.is_aggregation and len(results_dicts) < AGGREGATION_SEARCH_LIMIT:
                logger.info(
                    f"Aggregation detected with only {len(results_dicts)} pre-computed results, "
                    f"doing supplementary search for {AGGREGATION_SEARCH_LIMIT} results"
                )
                supplementary = await _do_search(
                    engine, request.user_id, request.question,
                    AGGREGATION_SEARCH_LIMIT, request.search_mode,
                )
                results_dicts = _merge_results(results_dicts, supplementary)
                logger.info(f"After supplementary search: {len(results_dicts)} total results")
        else:
            results_dicts = await _do_search(
                engine, request.user_id, request.question,
                effective_limit, request.search_mode,
            )

        # Step 2: Generate answer (singleton to reuse HTTP connections)
        answer_service = AnswerService.get_instance()
        result = await answer_service.generate_answer(
            question=request.question,
            search_results=results_dicts,
            question_date=request.question_date,
            enable_voting=request.enable_voting,
        )

        # Step 3: CRAG corrective re-search if answer service returned CRAG queries
        crag_queries = result.get("crag_queries")
        if crag_queries and len(crag_queries) > 0:
            logger.info(f"CRAG re-search triggered with {len(crag_queries)} queries")
            all_new_results: list[dict] = []
            for crag_query in crag_queries:
                new_results = await _do_search(
                    engine, request.user_id, crag_query,
                    5, request.search_mode,
                )
                all_new_results.extend(new_results)

            # Merge original + CRAG results
            merged = _merge_results(results_dicts, all_new_results)
            new_count = len(merged) - len(results_dicts)
            logger.info(
                f"CRAG added {new_count} new unique results "
                f"(total: {len(merged)}, was: {len(results_dicts)})"
            )

            # Re-generate answer with expanded context
            if new_count > 0:
                result = await answer_service.generate_answer(
                    question=request.question,
                    search_results=merged,
                    question_date=request.question_date,
                    enable_voting=request.enable_voting,
                    is_crag_retry=True,
                )
                result["crag_queries"] = crag_queries
                result["crag_iteration"] = 1
                result["search_results_count"] = len(merged)

        return AnswerResponse(**result)

    except Exception as e:
        logger.error(f"Failed to generate answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
