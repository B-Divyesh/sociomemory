"""
LLM-based Reranker for SocioMemory

Uses Azure OpenAI to rerank retrieval candidates for improved accuracy.
Inspired by RankRAG and TEMPR approaches.

The key insight: LLMs can better understand query-document relevance
than simple vector similarity, especially for:
- Multi-hop queries (connecting multiple pieces of information)
- Temporal queries (understanding time context)
- Complex factual queries (entity relationships)
"""
import json
import logging
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from sociomemory.config import get_settings


logger = logging.getLogger(__name__)


# Reranking prompt designed for conversational memory retrieval
# Enhanced with query-type awareness and explicit synthesis instructions
RERANK_PROMPT = """You are a relevance scoring expert for conversational memory retrieval.

Given a question and a list of {num_passages} conversation memory passages, score EACH passage from 0.0 to 1.0 based on how likely it contains the EXACT information needed to answer the question.

**CRITICAL SCORING RULES:**

1. **EXACT MATCH = 1.0**: If a passage contains the SPECIFIC answer to the question (names, dates, facts, events), score it 1.0. DO NOT let recency or position influence this - an old passage with the exact answer beats a recent passage without it.

2. **For "When" questions**: Look for DATES, TIMES, or temporal references (yesterday, last week, May 2023). The passage with the actual date/time gets 1.0.

3. **For "What" questions about facts**: Look for the SPECIFIC fact being asked. Passages mentioning the exact topic/entity get higher scores.

4. **For "Who" questions**: Look for NAMES of people. The passage mentioning the specific person gets 1.0.

5. **For multi-hop questions**: Score higher passages that provide critical CONNECTING information, even if they don't contain the final answer directly.

**Scoring scale:**
- 1.0: Contains the EXACT answer or critical fact needed
- 0.8-0.9: Contains highly relevant information that directly helps answer
- 0.5-0.7: Contains related context but not the direct answer
- 0.2-0.4: Tangentially related topic
- 0.0-0.1: Irrelevant to the question

**WARNING**: DO NOT bias toward passages that merely mention similar topics. Focus on finding the passage with the ACTUAL ANSWER.

Question: {question}

Passages to score:
{passages}

IMPORTANT: Return EXACTLY {num_passages} scores as a JSON array, one for each passage in order. Focus on finding the passage with the EXACT answer.

JSON scores:"""


# Query-type specific reranking prompts for even better accuracy
RERANK_PROMPT_TEMPORAL = """You are a relevance scoring expert. The question asks about WHEN something happened.

Your task: Find the passage that contains the SPECIFIC DATE or TIME reference that answers this question.

**CRITICAL**: Look for:
- Explicit dates (May 7, 2023; last Tuesday; yesterday)
- Session dates in the passage headers
- Relative time references that can be calculated

Question: {question}

Passages to score:
{passages}

Score each passage 0.0-1.0. Give 1.0 ONLY to passages containing the actual date/time answer.
Return EXACTLY {num_passages} scores as a JSON array.

JSON scores:"""


RERANK_PROMPT_FACTUAL = """You are a relevance scoring expert. The question asks about a SPECIFIC FACT.

Your task: Find the passage that contains the EXACT factual information requested.

**CRITICAL**: Look for:
- Specific names, places, events mentioned
- Direct statements of fact matching the question
- Explicit answers, not just related topics

Question: {question}

Passages to score:
{passages}

Score each passage 0.0-1.0. Give 1.0 ONLY to passages containing the exact fact requested.
Return EXACTLY {num_passages} scores as a JSON array.

JSON scores:"""


RERANK_PROMPT_MULTIHOP = """You are a relevance scoring expert. This question requires connecting multiple pieces of information.

Your task: Score passages based on how they contribute to a CHAIN OF REASONING.

**CRITICAL**: Look for:
- Passages that provide key entities/facts mentioned in the question
- Passages that connect different pieces of information
- Evidence chains that lead to the answer

Question: {question}

Passages to score:
{passages}

Score each passage 0.0-1.0 based on its contribution to answering this multi-hop question.
Return EXACTLY {num_passages} scores as a JSON array.

JSON scores:"""


class LLMReranker:
    """Rerank retrieval candidates using LLM scoring."""

    def __init__(self, model: str = None):
        """Initialize the reranker with Azure OpenAI client.

        Args:
            model: Optional model override. Defaults to gpt-4o for best quality.
        """
        self.settings = get_settings()

        if self.settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_key,
                api_version="2024-05-01-preview",  # Updated API version for GPT-4o
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            # Use provided model or default to gpt-4o for best reranking quality
            self.model = model or "gpt-4o"
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = model or "gpt-4o"

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type to select the optimal reranking prompt.

        Returns: 'temporal', 'factual', 'multihop', or 'general'
        """
        query_lower = query.lower().strip()

        # Temporal queries - asking about WHEN
        temporal_starts = ['when ', 'what time ', 'how long ago ', 'at what time ']
        if any(query_lower.startswith(t) for t in temporal_starts):
            return 'temporal'

        # Multi-hop queries - requiring reasoning chains
        multihop_patterns = [
            'based on what', 'considering that', 'given that',
            'taking into account', 'if you combine', 'connecting',
            'what would', 'how would'
        ]
        if any(p in query_lower for p in multihop_patterns):
            return 'multihop'

        # Factual queries - asking about specific facts
        factual_starts = [
            'what is', 'what was', 'what are', 'what did', 'what does',
            'who is', 'who was', 'who are', 'who did',
            'where is', 'where was', 'where did',
            'which ', 'whose ', 'whom '
        ]
        if any(query_lower.startswith(f) for f in factual_starts):
            return 'factual'

        return 'general'

    def _get_prompt_for_query_type(self, query_type: str) -> str:
        """Get the appropriate reranking prompt based on query type."""
        if query_type == 'temporal':
            return RERANK_PROMPT_TEMPORAL
        elif query_type == 'factual':
            return RERANK_PROMPT_FACTUAL
        elif query_type == 'multihop':
            return RERANK_PROMPT_MULTIHOP
        else:
            return RERANK_PROMPT

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        content_key: str = "content",
        query_type: str = None,
    ) -> list[dict]:
        """
        Rerank candidates using LLM scoring with query-type awareness.

        Args:
            query: The search query
            candidates: List of candidate dicts (must have content_key field)
            top_k: Number of top results to return
            content_key: Key in candidate dict containing the text content
            query_type: Optional query type override ('temporal', 'factual', 'multihop', 'general')

        Returns:
            List of top_k candidates reranked by LLM relevance scores
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        num_candidates = len(candidates)

        # Detect query type if not provided
        detected_type = query_type or self._detect_query_type(query)
        logger.debug(f"Reranking with query type: {detected_type}")

        # Format passages for the prompt - include more context for better reranking
        passages_text = "\n".join([
            f"[{i+1}] {c.get(content_key, '')[:800]}"  # Increased from 500 to 800 chars
            for i, c in enumerate(candidates)
        ])

        # Select query-type specific prompt
        prompt_template = self._get_prompt_for_query_type(detected_type)
        prompt = prompt_template.format(
            question=query,
            passages=passages_text,
            num_passages=num_candidates,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise relevance scoring system. Return only valid JSON array of floats."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic scoring
                max_tokens=50 + num_candidates * 6,  # Scale with candidate count
            )

            scores_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if scores_text.startswith("```"):
                scores_text = scores_text.split("```")[1]
                if scores_text.startswith("json"):
                    scores_text = scores_text[4:]
                scores_text = scores_text.strip()

            # Parse JSON scores
            scores = json.loads(scores_text)

            if not isinstance(scores, list):
                logger.warning(f"LLM reranker returned non-list: {scores_text[:100]}")
                return candidates[:top_k]

            # Robust handling of score count mismatch
            if len(scores) != num_candidates:
                logger.debug(f"LLM reranker returned {len(scores)} scores for {num_candidates} candidates, adjusting")
                if len(scores) < num_candidates:
                    # Pad with 0.0 for missing scores
                    scores.extend([0.0] * (num_candidates - len(scores)))
                else:
                    # Truncate extra scores
                    scores = scores[:num_candidates]

            # Validate scores are numeric and in range
            validated_scores = []
            for i, score in enumerate(scores):
                try:
                    s = float(score)
                    validated_scores.append(max(0.0, min(1.0, s)))  # Clamp to [0, 1]
                except (TypeError, ValueError):
                    validated_scores.append(0.0)

            # Pair candidates with scores and sort
            scored = list(zip(candidates, validated_scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Add LLM score to candidates and return top_k
            results = []
            for candidate, score in scored[:top_k]:
                candidate["llm_score"] = score
                results.append(candidate)

            return results

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM reranker response: {e}")
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"LLM reranker error: {e}")
            return candidates[:top_k]

    async def rerank_with_reasoning(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        content_key: str = "content",
    ) -> tuple[list[dict], str]:
        """
        Rerank with explicit reasoning (useful for debugging/analysis).

        Returns:
            Tuple of (reranked candidates, reasoning text)
        """
        num_candidates = len(candidates)
        reasoning_prompt = f"""Analyze the relevance of these {num_candidates} passages to the question.
Think step by step about which passages contain the most relevant information.

Question: {query}

Passages:
{chr(10).join([f"[{i+1}] {c.get(content_key, '')[:300]}" for i, c in enumerate(candidates)])}

First, briefly explain which passages are most relevant and why.
Then provide your final ranking as a JSON array of EXACTLY {num_candidates} scores (0.0-1.0), one for each passage.

Analysis:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.0,
                max_tokens=500 + num_candidates * 6,
            )

            full_response = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[[\d.,\s]+\]', full_response)

            if json_match:
                scores = json.loads(json_match.group())

                # Handle score count mismatch
                if len(scores) < num_candidates:
                    scores.extend([0.0] * (num_candidates - len(scores)))
                elif len(scores) > num_candidates:
                    scores = scores[:num_candidates]

                # Validate and clamp scores
                validated_scores = []
                for score in scores:
                    try:
                        s = float(score)
                        validated_scores.append(max(0.0, min(1.0, s)))
                    except (TypeError, ValueError):
                        validated_scores.append(0.0)

                scored = list(zip(candidates, validated_scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                results = []
                for candidate, score in scored[:top_k]:
                    candidate["llm_score"] = score
                    results.append(candidate)
                return results, full_response

            return candidates[:top_k], full_response

        except Exception as e:
            logger.error(f"LLM reranker with reasoning error: {e}")
            return candidates[:top_k], f"Error: {e}"


# Singleton instance
_reranker: Optional[LLMReranker] = None


def get_reranker() -> LLMReranker:
    """Get or create singleton reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker()
    return _reranker
