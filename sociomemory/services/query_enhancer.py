"""
Query Enhancement Service for SocioMemory

Implements SOTA techniques from LongMemEval and Emergence.ai:
1. Time-aware query expansion (+6.8-11.3% for temporal queries)
2. Fact-augmented key extraction (+9.4% recall, +5.4% accuracy)
3. Chain-of-Note reading strategy (+10 points)

These techniques combined can boost accuracy from ~70% to 82%+.
"""
import json
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from openai import AsyncAzureOpenAI, AsyncOpenAI

from sociomemory.config import get_settings


# Time-aware query expansion prompt
TIME_EXPANSION_PROMPT = """Analyze this question and extract any temporal constraints.

Question: {question}

If the question contains temporal references (like "last week", "in March", "yesterday", "first time", "most recent"), extract:
1. The time range or temporal constraint
2. Keywords for temporal filtering

Return ONLY valid JSON:
{{"has_temporal": true/false, "temporal_type": "relative|absolute|ordinal|none", "time_hint": "recent|earliest|specific|none", "time_keywords": ["keyword1", "keyword2"]}}

Example outputs:
- "What did I eat last Friday?" -> {{"has_temporal": true, "temporal_type": "relative", "time_hint": "specific", "time_keywords": ["friday", "last"]}}
- "What's my favorite color?" -> {{"has_temporal": false, "temporal_type": "none", "time_hint": "none", "time_keywords": []}}
- "What was the first restaurant I mentioned?" -> {{"has_temporal": true, "temporal_type": "ordinal", "time_hint": "earliest", "time_keywords": ["first"]}}

JSON:"""


# Fact extraction prompt
FACT_EXTRACTION_PROMPT = """Extract key facts from this conversation memory that should be indexed for retrieval.

Memory: {content}

Extract as structured JSON with:
1. entities: People, places, organizations, products mentioned
2. facts: Key factual statements (what happened, preferences, decisions)
3. topics: Main topics discussed
4. keywords: Important searchable terms

Return ONLY valid JSON:
{{"entities": ["entity1", "entity2"], "facts": ["fact1", "fact2"], "topics": ["topic1"], "keywords": ["kw1", "kw2"]}}

JSON:"""


# Chain-of-Note reading prompt
CHAIN_OF_NOTE_PROMPT = """You are analyzing retrieved memories to answer a question.

Question: {question}

Retrieved Memories:
{memories}

First, extract relevant information from each memory as structured notes.
Then, synthesize an answer from these notes.

Return ONLY valid JSON:
{{
  "notes": [
    {{"memory_idx": 1, "relevant_info": "extracted info from memory 1", "relevance": "high|medium|low"}},
    ...
  ],
  "synthesis": "synthesized answer combining relevant notes",
  "confidence": 0.0-1.0,
  "answer_found": true/false
}}

JSON:"""


class QueryEnhancer:
    """Enhance queries and reading using GPT-4o."""

    def __init__(self, model: str = "gpt-4o"):
        self.settings = get_settings()
        self.model = model

        if self.settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_key,
                api_version="2024-05-01-preview",
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    async def expand_temporal_query(self, question: str) -> dict:
        """Extract temporal constraints from a question.

        Returns:
            Dict with temporal analysis:
            - has_temporal: bool
            - temporal_type: relative|absolute|ordinal|none
            - time_hint: recent|earliest|specific|none
            - time_keywords: list of temporal keywords
        """
        try:
            prompt = TIME_EXPANSION_PROMPT.format(question=question)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a temporal analysis assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            return result

        except Exception as e:
            return {
                "has_temporal": False,
                "temporal_type": "none",
                "time_hint": "none",
                "time_keywords": [],
                "error": str(e),
            }

    async def extract_facts(self, content: str) -> dict:
        """Extract indexable facts from memory content.

        Returns:
            Dict with:
            - entities: list of entities
            - facts: list of key facts
            - topics: list of topics
            - keywords: list of keywords
        """
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(content=content[:1500])

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            return result

        except Exception as e:
            return {
                "entities": [],
                "facts": [],
                "topics": [],
                "keywords": [],
                "error": str(e),
            }

    async def chain_of_note_read(
        self,
        question: str,
        memories: list[dict],
        content_key: str = "content",
    ) -> dict:
        """Use Chain-of-Note strategy to extract and synthesize answer.

        Args:
            question: The question to answer
            memories: List of retrieved memory dicts
            content_key: Key containing memory content

        Returns:
            Dict with:
            - notes: List of extracted notes per memory
            - synthesis: Synthesized answer
            - confidence: 0-1 confidence score
            - answer_found: Whether answer was found
        """
        if not memories:
            return {
                "notes": [],
                "synthesis": "No relevant memories found.",
                "confidence": 0.0,
                "answer_found": False,
            }

        try:
            # Format memories
            memories_text = "\n".join([
                f"[{i+1}] {mem.get(content_key, '')[:500]}"
                for i, mem in enumerate(memories[:10])  # Limit to 10
            ])

            prompt = CHAIN_OF_NOTE_PROMPT.format(
                question=question,
                memories=memories_text,
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a memory analysis assistant. Extract relevant information and synthesize answers. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            return result

        except Exception as e:
            return {
                "notes": [],
                "synthesis": f"Error: {e}",
                "confidence": 0.0,
                "answer_found": False,
                "error": str(e),
            }


# Lightweight temporal detection without LLM (for fast path)
def detect_temporal_fast(query: str) -> Tuple[bool, str]:
    """Fast regex-based temporal detection.

    Returns:
        Tuple of (has_temporal, time_hint)
    """
    query_lower = query.lower()

    # Ordinal patterns (first, earliest, last)
    if re.search(r'\b(first|earliest|initial|originally)\b', query_lower):
        return True, "earliest"

    if re.search(r'\b(last|latest|most recent|recently|newest)\b', query_lower):
        return True, "recent"

    # Relative time patterns
    if re.search(r'\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+|\d+\s+(days?|weeks?|months?|years?)\s+ago)\b', query_lower):
        return True, "specific"

    # Absolute time patterns
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', query_lower):
        return True, "specific"

    if re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', query_lower):
        return True, "specific"

    if re.search(r'\b\d{4}\b', query_lower):  # Year
        return True, "specific"

    return False, "none"


# Singleton instance
_enhancer: Optional[QueryEnhancer] = None


def get_query_enhancer(model: str = "gpt-4o") -> QueryEnhancer:
    """Get or create singleton query enhancer."""
    global _enhancer
    if _enhancer is None:
        _enhancer = QueryEnhancer(model=model)
    return _enhancer
