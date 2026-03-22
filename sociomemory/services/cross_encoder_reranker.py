"""
Cross-Encoder Reranking Service - Deterministic reranking via flashrank

Replaces LLM-based CoT reranking (which is non-deterministic even at temp=0)
with a lightweight ONNX cross-encoder model that produces identical results
for identical inputs.

Model: ms-marco-MiniLM-L-12-v2 (~34MB, ONNX, CPU-optimized)
Latency: ~30-50ms for 100 candidates on CPU
Deterministic: Same input always produces same output

v17: Core change for breaking through 80% accuracy barrier.
Research basis: Cross-encoder reranking is standard in SOTA RAG systems
(Emergence AI 86%, Hindsight 91.4% both use cross-encoder reranking).
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Singleton ranker instance (thread-safe, stateless inference)
_ranker = None


def get_ranker(
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str = "/opt/flashrank_cache",
    max_length: int = 512,
):
    """
    Get or create singleton Ranker instance.

    Uses ms-marco-MiniLM-L-12-v2 for best precision (~34MB).
    The model is loaded once and reused across all requests.

    Args:
        model_name: flashrank model name
        cache_dir: Directory where model is cached
        max_length: Maximum token length for passages (512 for cross-encoders)

    Returns:
        flashrank Ranker instance
    """
    global _ranker
    if _ranker is None:
        try:
            from flashrank import Ranker
            logger.info(f"Initializing flashrank ranker: {model_name}")
            _ranker = Ranker(
                model_name=model_name,
                cache_dir=cache_dir,
                max_length=max_length,
            )
            logger.info("flashrank ranker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize flashrank ranker: {e}")
            raise
    return _ranker


def cross_encoder_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    content_key: str = "content",
    return_all_scores: bool = False,
) -> List[Dict[str, Any]]:
    """
    Deterministic cross-encoder reranking using flashrank.

    Replaces the non-deterministic LLM CoT reranking with a fast,
    deterministic cross-encoder model. Same input always produces
    the same output, eliminating the 48.8% search variance.

    Args:
        query: The search query
        candidates: List of candidate dicts with content_key field
        top_k: Number of results to return (ignored if return_all_scores=True)
        content_key: Key in candidate dict containing text
        return_all_scores: If True, return ALL candidates sorted by score (for adaptive-k)

    Returns:
        Candidates reranked by cross-encoder score,
        with 'ce_score' field added to each result.
        If return_all_scores=True, returns all candidates sorted by score.
        Otherwise returns top_k candidates.
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        # Still score them for consistency
        try:
            ranker = get_ranker()
            from flashrank import RerankRequest

            passages = []
            for i, c in enumerate(candidates):
                text = c.get(content_key, "")
                passages.append({
                    "id": i,
                    "text": text[:2000],
                    "meta": {"original_index": i},
                })

            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            # Map scores back to candidates
            score_map = {}
            for result in results:
                idx = result["meta"]["original_index"]
                score_map[idx] = float(result["score"])  # Ensure native float

            scored = []
            for i, c in enumerate(candidates):
                c_copy = dict(c)
                c_copy["ce_score"] = score_map.get(i, 0.0)
                scored.append(c_copy)

            return scored

        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed for small set: {e}")
            return candidates

    try:
        ranker = get_ranker()
        from flashrank import RerankRequest

        # Build passages for flashrank
        # Each passage needs: id, text, meta (optional)
        passages = []
        for i, c in enumerate(candidates):
            text = c.get(content_key, "")
            # Truncate to ~2000 chars (model tokenizes to max_length=512 tokens internally)
            passages.append({
                "id": i,
                "text": text[:2000],
                "meta": {"original_index": i},
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        # Results come back sorted by score (highest first)
        # Map back to original candidates
        limit = len(results) if return_all_scores else top_k
        reranked = []
        for result in results[:limit]:
            idx = result["meta"]["original_index"]
            candidate = dict(candidates[idx])  # Copy to avoid mutation
            candidate["ce_score"] = float(result["score"])  # Ensure native float (not numpy)
            reranked.append(candidate)

        if reranked:
            logger.debug(
                f"Cross-encoder rerank: {len(candidates)} -> {len(reranked)} candidates, "
                f"top score: {reranked[0]['ce_score']:.4f}"
                f"{' (all scores returned)' if return_all_scores else ''}"
            )
        else:
            logger.warning("Cross-encoder returned no results, falling back to original order")
            return candidates[:top_k]

        return reranked

    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {e}", exc_info=True)
        # Fallback: return original order (preserves existing behavior)
        return candidates if return_all_scores else candidates[:top_k]
