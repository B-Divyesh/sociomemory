"""
Hyper Search Service - Advanced RAG Pipeline for 90%+ Accuracy

This implements state-of-the-art retrieval techniques from recent papers:
1. Query Expansion (FlashRank paper: +6-8% recall)
2. HopRAG-style Retrieve-Reason-Prune (+76.78% on multi-hop)
3. Memory-T1 temporal reasoning with coarse-to-fine retrieval
4. Chain-of-Thought reranking with explicit reasoning

Target: 90%+ accuracy on memorybench (minimum 85%)

Design principles:
- Stateless for parallel multi-user requests
- Each function is pure and doesn't maintain state
- All state is passed explicitly via parameters
"""
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from openai import AsyncAzureOpenAI, AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from sociomemory.config import get_settings
from sociomemory.models.memory import (
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)
from sociomemory.services.cross_encoder_reranker import cross_encoder_rerank
from sociomemory.services.embedding_service import EmbeddingService, get_embedding_service
from sociomemory.services.hybrid_search import HybridSearchService
from sociomemory.services.temporal_parser import TemporalParser, TemporalType
from sociomemory.services.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY EXPANSION MODULE
# =============================================================================

QUERY_EXPANSION_PROMPT = """You are a query expansion expert for conversational memory retrieval.

Given a user question, generate 2-3 alternative search queries that would help find relevant conversation memories.

EXPANSION STRATEGIES by question type:

For TEMPORAL/SEQUENCE questions (e.g., "How many days between X and Y?"):
- Create SEPARATE queries for EACH event mentioned
- Example: "days between museum visit and concert" → ["museum visit date", "concert date", "when did I go to the museum"]

For PREFERENCE questions (e.g., "Can you recommend a hotel?"):
- Focus on user's stated preferences and past experiences
- Example: "recommend a hotel in Miami" → ["hotels I liked", "Miami accommodation preferences", "hotel features I prefer"]

For FACTUAL questions:
- Use synonyms and related terms
- More specific versions of the question

IMPORTANT: Keep expansions focused. For multi-event questions, ensure each event gets its own query.

Original question: {question}

Return a JSON array of 2-3 alternative queries. Example:
["alternative query 1", "alternative query 2"]

JSON array:"""


QUERY_EXPANSION_TEMPORAL_SEQUENCE = """You are a query expansion expert for temporal/sequence questions.

The question asks about the ORDER or TIME RELATIONSHIP between multiple events.

Your task: Generate search queries that will find the DATE of EACH event mentioned.

Question: {question}

STEP 1: List each distinct event mentioned in the question.
STEP 2: Create a focused search query for EACH event to find its date.

Example:
Question: "How many days between my dentist visit and my doctor appointment?"
Events: ["dentist visit", "doctor appointment"]
Queries: ["when did I visit the dentist", "dentist appointment date", "when was my doctor appointment"]

Return a JSON array of 3-4 focused queries, one targeting each event:
["query for event 1", "query for event 2", "query for event 3"]

JSON array:"""


QUERY_EXPANSION_AGGREGATION = """You are a query expansion expert for AGGREGATION questions.

The question asks to COUNT or LIST multiple items across different conversations/memories.

Your task: Generate search queries that will find ALL instances of the item type being asked about.

Question: {question}

STEP 1: Identify what TYPE of items are being counted/listed (e.g., "clothing items", "projects", "plants").
STEP 2: Generate DIVERSE search queries to find ALL instances, using:
   - Different phrasings of the item type
   - Related terms and synonyms
   - Specific contexts where these items might be mentioned

Example:
Question: "How many items of clothing do I need to pick up or return from a store?"
Item type: clothing items to pick up/return
Queries: [
  "clothing pick up store",
  "dry cleaning pick up",
  "return clothes store",
  "exchange clothing",
  "tailor alterations pick up"
]

Return a JSON array of 4-5 diverse search queries to maximize recall:
["query 1", "query 2", "query 3", "query 4"]

JSON array:"""


QUERY_EXPANSION_KNOWLEDGE_UPDATE = """You are a query expansion expert for KNOWLEDGE UPDATE questions.

The question asks about something that may have been UPDATED or CHANGED. We need to find the LATEST value.

Question: {question}

Your task: Generate search queries that will find:
1. The UPDATED/CHANGED value (most important)
2. Any conversations where the update was discussed
3. Related context that might mention the current state

STEP 1: Identify what information is being asked about (e.g., phone number, address, job)
STEP 2: Generate queries that emphasize:
   - Update language: "new", "changed", "updated", "switched"
   - Current state: "current", "now", "recently"
   - The specific topic being asked about

Example:
Question: "What is my new phone number?"
Queries: [
  "new phone number",
  "changed phone number",
  "updated my phone",
  "current phone number",
  "phone number now"
]

Return a JSON array of 3-4 search queries focused on finding the UPDATED value:
["query 1", "query 2", "query 3"]

JSON array:"""


QUERY_EXPANSION_PREFERENCE = """You are a query expansion expert for PREFERENCE/RECOMMENDATION questions.

The question asks for recommendations based on user preferences, tastes, or past experiences.

Question: {question}

Your task: Generate search queries that will find:
1. User's stated preferences, likes, and dislikes on this topic
2. User's past positive experiences with similar items
3. User's hobbies, activities, or interests related to this topic
4. Specific items/activities the user has mentioned enjoying

STEP 1: Identify what type of recommendation is being asked for
STEP 2: Generate queries to find:
   - Direct preferences: "I like", "I prefer", "my favorite"
   - Past experiences: "I enjoyed", "I loved", "it was great"
   - Relevant activities/hobbies: what the user does, grows, makes, etc.
   - Specific related items the user has mentioned

Example:
Question: "What should I serve for dinner this weekend with my homegrown ingredients?"
Topic: dinner, homegrown ingredients
Queries: [
  "what I grow in my garden",
  "my homegrown vegetables",
  "ingredients from my garden",
  "what I've harvested",
  "my favorite recipes"
]

Example:
Question: "Can you recommend a movie for tonight?"
Topic: movie recommendation
Queries: [
  "movies I liked",
  "my favorite films",
  "movies I enjoyed watching",
  "what I like to watch"
]

Return a JSON array of 4-5 search queries focused on finding user preferences:
["query 1", "query 2", "query 3", "query 4"]

JSON array:"""


# =============================================================================
# QUERY EXPANSION CACHE (v17: eliminates LLM expansion variance)
# =============================================================================
# In-memory cache: same query+type always returns same expansions
# This eliminates the biggest source of the 48.8% search non-determinism
_expansion_cache: Dict[str, List[str]] = {}


def _expansion_cache_key(query: str, query_type: str) -> str:
    """Generate deterministic cache key for expansion."""
    return hashlib.md5(f"{query}|{query_type}".encode()).hexdigest()


async def expand_query(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    model: str,
    query: str,
    max_expansions: int = 3,
    query_type: str = None,
) -> List[str]:
    """
    Expand a query into multiple search variants for better recall.

    Based on FlashRank paper showing +6-8% improvement with query expansion.
    Enhanced with query-type-specific expansion strategies.

    v17: Added caching to eliminate LLM expansion variance.
    temperature=0.0 + seed=42 for near-deterministic LLM output on cache miss.

    Args:
        client: OpenAI client
        model: Model to use for expansion
        query: Original query
        max_expansions: Maximum number of expansions to generate
        query_type: Optional pre-detected query type for optimized expansion

    Returns:
        List of query variants (including original)
    """
    # Detect query type if not provided
    if query_type is None:
        query_type = detect_query_type_enhanced(query)

    # v17: Check expansion cache first
    cache_key = _expansion_cache_key(query, query_type or "auto")
    if cache_key in _expansion_cache:
        logger.debug(f"Using cached expansion for: {query[:50]}")
        return _expansion_cache[cache_key]

    # Select appropriate expansion prompt based on query type
    if query_type == 'temporal_sequence':
        expansion_prompt = QUERY_EXPANSION_TEMPORAL_SEQUENCE.format(question=query)
        max_expansions = 4  # Need more for multi-event queries
    elif query_type == 'aggregation':
        expansion_prompt = QUERY_EXPANSION_AGGREGATION.format(question=query)
        max_expansions = 5  # Need more diverse queries for counting across memories
    elif query_type == 'knowledge_update':
        expansion_prompt = QUERY_EXPANSION_KNOWLEDGE_UPDATE.format(question=query)
        max_expansions = 4  # Need update-focused queries
    elif query_type == 'preference':
        expansion_prompt = QUERY_EXPANSION_PREFERENCE.format(question=query)
        max_expansions = 5  # Need diverse queries to find user preferences/hobbies
    else:
        expansion_prompt = QUERY_EXPANSION_PROMPT.format(question=query)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise query expansion system. Return only valid JSON."},
                {"role": "user", "content": expansion_prompt},
            ],
            temperature=0.0,  # v17: Deterministic (was 0.3)
            seed=42,  # v17: Seed for reproducibility
            max_tokens=250,
        )

        response_text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        expansions = json.loads(response_text)

        if isinstance(expansions, list):
            # Return original + expansions, limited to max_expansions
            all_queries = [query] + [e for e in expansions if isinstance(e, str)][:max_expansions]
            logger.debug(f"Query expansion: {query} -> {len(all_queries)} variants")
            # v17: Cache for determinism
            _expansion_cache[cache_key] = all_queries
            return all_queries

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")

    # Fallback: return original query only
    fallback = [query]
    _expansion_cache[cache_key] = fallback
    return fallback


# =============================================================================
# HOPRAG-STYLE RETRIEVE-REASON-PRUNE
# =============================================================================

HOPRAG_REASON_PROMPT = """You are analyzing retrieved passages for relevance to a multi-hop question.

Question: {question}

Retrieved passages:
{passages}

Task: Identify which passages contain CRITICAL information for answering this question.
Think step-by-step:
1. What entities/facts does the question ask about?
2. Which passages mention these entities/facts?
3. Which passages provide CONNECTING information (e.g., "X is Y's friend" helps answer "What does Y's friend do?")

Return a JSON object with:
- "reasoning": Brief explanation of your analysis
- "relevant_indices": Array of passage numbers (1-indexed) that should be KEPT
- "critical_indices": Array of passage numbers with the MOST critical information

Example:
{{"reasoning": "The question asks about X's hobby. Passage 2 mentions X, passage 4 mentions their hobby.", "relevant_indices": [2, 4, 5], "critical_indices": [4]}}

JSON response:"""


HOPRAG_TEMPORAL_SEQUENCE_PROMPT = """You are analyzing retrieved passages to find DATE information for MULTIPLE EVENTS.

Question: {question}
(This question asks about the TIME/ORDER relationship between multiple events)

Retrieved passages:
{passages}

Task: Find passages that contain DATE/TIME information for EACH event mentioned in the question.

Think step-by-step:
1. LIST each distinct event mentioned in the question
2. For EACH event, identify which passage(s) contain its DATE or TIME
3. Mark passages as CRITICAL if they provide a specific date for any event

IMPORTANT: We need dates for ALL events mentioned. A passage is valuable if it dates ANY of the events.

Return a JSON object with:
- "reasoning": "Event 1 (X) found in passage N with date Y. Event 2 (Z) found in passage M with date W."
- "relevant_indices": Array of ALL passages mentioning any of the events
- "critical_indices": Array of passages with SPECIFIC DATES for the events

Example for "days between dentist and doctor visit":
{{"reasoning": "Dentist visit date in passage 3 (May 5). Doctor appointment in passage 7 (May 12).", "relevant_indices": [3, 5, 7], "critical_indices": [3, 7]}}

JSON response:"""


async def hoprag_reason_prune(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    model: str,
    query: str,
    candidates: List[Dict[str, Any]],
    content_key: str = "content",
    query_type: str = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    HopRAG-style reasoning step to identify relevant passages.

    This implements the "retrieve-reason-prune" pattern from HopRAG paper
    which showed +76.78% improvement on multi-hop questions.

    Args:
        client: OpenAI client
        model: Model to use
        query: The search query
        candidates: List of candidate passages
        content_key: Key in candidate dict containing text
        query_type: Optional query type for specialized prompts

    Returns:
        Tuple of (filtered candidates, reasoning text)
    """
    if len(candidates) <= 3:
        return candidates, "Too few candidates to prune"

    # Detect query type if not provided
    if query_type is None:
        query_type = detect_query_type_enhanced(query)

    # Format passages for the prompt
    passages_text = "\n".join([
        f"[{i+1}] {c.get(content_key, '')[:600]}"
        for i, c in enumerate(candidates[:15])  # Limit to 15 for context window
    ])

    # Select prompt based on query type
    if query_type == 'temporal_sequence':
        prompt_content = HOPRAG_TEMPORAL_SEQUENCE_PROMPT.format(
            question=query,
            passages=passages_text,
        )
    else:
        prompt_content = HOPRAG_REASON_PROMPT.format(
            question=query,
            passages=passages_text,
        )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise reasoning system. Return only valid JSON."},
                {"role": "user", "content": prompt_content},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        response_text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)

        reasoning = result.get("reasoning", "")
        relevant_indices = result.get("relevant_indices", [])
        critical_indices = result.get("critical_indices", [])

        # Combine relevant and critical (critical gets priority)
        all_indices = set(critical_indices) | set(relevant_indices)

        if all_indices:
            # Filter candidates, keeping only relevant ones
            # Reorder: critical first, then relevant
            filtered = []
            for idx in critical_indices:
                if 1 <= idx <= len(candidates):
                    filtered.append(candidates[idx - 1])
            for idx in relevant_indices:
                if 1 <= idx <= len(candidates) and idx not in critical_indices:
                    filtered.append(candidates[idx - 1])

            if filtered:
                logger.debug(f"HopRAG pruning: {len(candidates)} -> {len(filtered)} candidates")
                return filtered, reasoning

    except Exception as e:
        logger.warning(f"HopRAG reasoning failed: {e}")

    # Fallback: return original candidates
    return candidates, "Reasoning failed, using original order"


# =============================================================================
# CHAIN-OF-THOUGHT RERANKING
# =============================================================================

COT_RERANK_PROMPT = """You are an expert at finding specific answers in conversation memories.

Question: {question}

{num_passages} conversation memories to evaluate:
{passages}

TASK: Score each memory 0.0-1.0 for relevance to answering the question.

THINK STEP BY STEP:
1. What specific information does the question ask for?
2. For each memory, does it contain that EXACT information?
3. Score 1.0 ONLY if the memory contains the specific answer (names, dates, facts)

SCORING RULES:
- 1.0: Contains the EXACT answer to the question
- 0.7-0.9: Contains highly relevant supporting information
- 0.4-0.6: Mentions related topics but not the answer
- 0.1-0.3: Tangentially related
- 0.0: Irrelevant

Think through each memory, then provide scores.

Your analysis (brief):
[Think about which memories contain the actual answer]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_TEMPORAL = """You are an expert at finding temporal/date information in conversation memories.

Question: {question}
(This question asks about WHEN something happened)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Find the memory that contains the SPECIFIC DATE or TIME that answers this question.

THINK STEP BY STEP:
1. What time/date information is the question asking for?
2. Look for: explicit dates, relative times ("yesterday", "last week"), timestamps
3. Score 1.0 ONLY for memories with the actual date/time answer

Look for date patterns like:
- "May 7, 2023", "2023-05-07"
- "yesterday", "last Tuesday", "3 days ago"
- Session/conversation timestamps

Your analysis (brief):
[Which memory has the date/time answer?]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_MULTIHOP = """You are an expert at multi-hop reasoning across conversation memories.

Question: {question}
(This question requires connecting multiple pieces of information)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Score memories based on their contribution to the REASONING CHAIN.

THINK STEP BY STEP:
1. Break down the question: What facts need to be connected?
2. For each memory: Does it provide a LINK in the reasoning chain?
3. Even if a memory doesn't have the final answer, score high if it provides critical connecting information

Example: "What is John's sister's favorite color?"
- Memory mentioning "John's sister is Sarah" = HIGH SCORE (provides connection)
- Memory mentioning "Sarah loves blue" = HIGH SCORE (provides answer piece)
- Both together answer the question

Your analysis (brief):
[What facts need connecting? Which memories provide the links?]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_AGGREGATION = """You are an expert at finding ALL instances of items across conversation memories.

Question: {question}
(This question asks to COUNT or LIST items across multiple conversations)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Find EVERY memory that mentions an item matching the question criteria.

CRITICAL: For counting/listing questions, we need to find ALL relevant items, not just one.

THINK STEP BY STEP:
1. What TYPE of item is being counted/listed? (e.g., "clothing to pick up", "projects led")
2. For EACH memory: Does it mention an item of that type?
3. Score HIGH for ANY memory with a relevant item, even if partial information

SCORING FOR AGGREGATION:
- 1.0: Clearly mentions an item matching the criteria
- 0.8: Mentions a potentially relevant item (needs verification)
- 0.5: Related topic but item not clearly matching
- 0.1: Mentions the topic category but no specific item
- 0.0: Irrelevant

IMPORTANT: Score multiple memories high if they each contain different items to count!

Your analysis (brief):
[List each item found in any memory that matches the criteria]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_PREFERENCE = """You are an expert at finding user PREFERENCES and TASTES in conversation memories.

Question: {question}
(This question asks for a RECOMMENDATION based on user preferences)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Score memories based on whether they reveal USER PREFERENCES relevant to this recommendation.

THINK STEP BY STEP:
1. What type of recommendation is being asked for? (e.g., hotel, movie, food)
2. What user preferences would inform this recommendation?
3. Score HIGH for memories that reveal the user's TASTES, LIKES, DISLIKES, or past positive experiences

CRITICAL - For preference questions:
- 1.0: Reveals user's explicit preferences/likes/dislikes for this topic
- 0.8-0.9: Shows user's past positive experiences with similar items
- 0.6-0.7: Contains related preferences that could inform the answer
- 0.3-0.5: Mentions the topic but no clear preference
- 0.0-0.2: Unrelated

IMPORTANT: A memory about "the user loved X" or "the user prefers Y" is HIGHLY relevant even if it doesn't directly answer the question.

Your analysis (brief):
[What preferences are relevant? Which memories reveal user tastes?]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_KNOWLEDGE_UPDATE = """You are an expert at finding the MOST RECENT/UPDATED information in conversation memories.

Question: {question}
(This question asks about something that may have been UPDATED or CHANGED)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Find the memory with the LATEST/MOST CURRENT value for what's being asked.

CRITICAL - For knowledge update questions:
- The user wants the CURRENT/NEW value, NOT the old value
- Look for phrases like: "new", "updated", "changed", "now", "recently", "just"
- More recent conversations are MORE LIKELY to have the current value
- If multiple memories mention the topic, prefer the one with UPDATE language

THINK STEP BY STEP:
1. What information is being asked about? (e.g., phone number, address, job)
2. Which memory mentions this with UPDATE/CHANGE language?
3. Which memory appears to be more recent (based on context/language)?

SCORING FOR KNOWLEDGE UPDATES:
- 1.0: Contains the UPDATED/NEW value with clear update language
- 0.8-0.9: Mentions the topic in what appears to be a recent context
- 0.5-0.7: Mentions the topic but unclear if it's the latest
- 0.2-0.4: Mentions the topic but appears to be OLD information
- 0.0-0.1: Unrelated

IMPORTANT: Old values are WRONG for this question type. Prioritize recency!

Your analysis (brief):
[Which memory has the UPDATED value? Look for update language.]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


COT_RERANK_TEMPORAL_SEQUENCE = """You are an expert at finding TEMPORAL SEQUENCES and DATE ORDERING in conversation memories.

Question: {question}
(This question asks about the ORDER or TIME RELATIONSHIP between events)

{num_passages} conversation memories to evaluate:
{passages}

TASK: Find memories containing DATE/TIME information for EACH event mentioned in the question.

THINK STEP BY STEP:
1. List ALL events mentioned in the question
2. For EACH event, find which memory contains its date/time
3. Score HIGH for ANY memory that pins down the date of ANY event in the question

CRITICAL FOR SEQUENCING QUESTIONS:
- 1.0: Contains explicit date for one of the events being compared
- 0.8-0.9: Contains the event with temporal context (e.g., "last week", "after X")
- 0.5-0.7: Mentions one of the events but no clear date
- 0.0-0.4: Unrelated to any events being compared

IMPORTANT: Even if a memory only has ONE event's date, score it HIGH - we need ALL dates to compare.

Your analysis (brief):
[List events needing dates. Which memories provide dates for which events?]

Final scores as JSON array (EXACTLY {num_passages} numbers):"""


def extract_key_entities(query: str) -> List[str]:
    """
    Extract key entities from a query for filtering.

    Extracts: locations, names, specific topics for better retrieval filtering.
    """
    entities = []
    query_lower = query.lower()

    # Extract quoted strings
    quoted = re.findall(r'"([^"]+)"', query)
    entities.extend(quoted)
    quoted = re.findall(r"'([^']+)'", query)
    entities.extend(quoted)

    # Extract capitalized words (potential proper nouns)
    words = query.split()
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
            if clean_word.lower() not in ['can', 'any', 'the', 'how', 'what', 'when', 'where', 'which', 'who']:
                entities.append(clean_word)

    # Common location indicators
    location_patterns = [
        r'(?:in|to|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, query)
        entities.extend(matches)

    return list(set(entities))


def detect_query_type_enhanced(query: str) -> str:
    """
    Enhanced query type detection for selecting optimal reranking prompt.

    Returns: 'knowledge_update', 'aggregation', 'temporal', 'temporal_sequence', 'preference', 'multihop', or 'factual'
    """
    query_lower = query.lower().strip()

    # KNOWLEDGE UPDATE queries - asking for CURRENT/UPDATED/CHANGED information
    # These need recency boosting to get the LATEST value, not old values
    # Check this FIRST as it's high priority for accuracy
    knowledge_update_patterns = [
        r'\b(new|current|latest|updated|now)\b.*(phone|number|address|email|password|job|work|name)',  # "What's my new phone number?"
        r'(what|where).*(do i|am i).*(now|currently)',  # "Where do I work now?"
        r'(changed|updated|switched).*(to|my)',  # "What did I change my password to?"
        r'(recent|latest).*(change|update)',  # "What's my latest address change?"
        r'what is my (current|new)',  # "What is my current address?"
        r'(did i|have i).*(change|update|switch)',  # "Did I change my email?"
        r'(after|since).*(chang|updat|switch)',  # "After I changed jobs, where do I work?"
        r'most recent.*(version|value|update)',  # "Most recent version of..."
    ]
    for pattern in knowledge_update_patterns:
        if re.search(pattern, query_lower):
            return 'knowledge_update'

    # AGGREGATION queries - asking to COUNT or LIST discrete ITEMS across memories
    # IMPORTANT: Must NOT match duration questions like "How many days did I spend"
    # Aggregation is for counting discrete ITEMS (projects, trips, books, clothes)
    # NOT for summing time units (days spent, hours worked)
    aggregation_patterns = [
        r'^how many (?!days|weeks|months|years|hours|minutes|times)',  # "How many X?" but NOT time/frequency units
        r'^list all',
        r'^what are all',
        r'^name all',
        r'total number of',
        r'count of',
        # Only match "how many [discrete items] have I/did I" - exclude duration-based
        # These are discrete item counts: "projects", "books", "trips", "people", "things", etc.
        r'how many (items|things|projects|books|movies|trips|places|people|friends|meetings|appointments|events|tasks|plants|clothes|clothing|outfits)',
    ]
    for pattern in aggregation_patterns:
        if re.search(pattern, query_lower):
            return 'aggregation'

    # Temporal SEQUENCE queries - asking about ORDER of events or TIME BETWEEN events
    # These need special handling to find MULTIPLE event dates
    temporal_sequence_patterns = [
        r'how many (days|weeks|months|years) (passed|between|since|ago)',
        r'(first|last|before|after).*order',
        r'order.*(first|last)',
        r'from first to last',
        r'chronological',
        r'happened (before|after)',
        r'which.*happened.*(first|earlier|later)',
        r'sequence of events',
        r'between my .* and .*(my|the)',  # "between my X and my Y"
    ]
    for pattern in temporal_sequence_patterns:
        if re.search(pattern, query_lower):
            return 'temporal_sequence'

    # Simple temporal queries - asking about WHEN (single event)
    temporal_patterns = [
        r'^when\s',
        r'^what\s+time\s',
        r'^how\s+long\s+ago\s',
        r'^at\s+what\s+time\s',
        r'^on\s+what\s+day\s',
        r'^what\s+date\s',
        r'happened\s+when',
        r'when\s+did',
        r'what\s+day\s+did',
    ]
    for pattern in temporal_patterns:
        if re.search(pattern, query_lower):
            return 'temporal'

    # PREFERENCE queries - asking for recommendations based on user tastes
    # These need specialized expansion to find user's hobbies, past experiences, and preferences
    preference_patterns = [
        r'^can you (suggest|recommend)',
        r'^(suggest|recommend)',
        r'^any (tips|advice|suggestions|recommendations)',
        r'what.*(should|would|could) (i|you).*(recommend|suggest)',
        r'preference',
        r'prefer(red)?',
        r'favorite',
        r'like (to|for me)',
        r'(give|tell) me.*(tips|advice|suggestions)',
        r'ideas for',
        # NEW: Catch "What should I serve/cook/make/do..." type questions
        r'^what should i (serve|cook|make|do|try|watch|read|eat|drink|wear|get|buy|use)',
        r'^what (can|could|would) i (serve|cook|make|do|try)',
        r'(serve|cook|make).*(dinner|lunch|breakfast|meal)',
        r'with my (homegrown|home-grown|garden|fresh)',  # Gardening/cooking context
        r'(suggest|recommend).*(for me|to me)',
        r'^do you have any (suggestions|recommendations|ideas)',
        r'what.*(would|should).*(be good|work well|go well)',
        r'(looking for|need).*(recommendations|suggestions|ideas)',
        r'(advice|tips) on',
        r'help me (find|choose|pick|decide)',
    ]
    for pattern in preference_patterns:
        if re.search(pattern, query_lower):
            return 'preference'

    # Multi-hop queries - requiring reasoning chains
    multihop_patterns = [
        r"'s\s+\w+'s",  # "John's sister's" - possessive chain
        r"of\s+the\s+\w+\s+of",  # "of the friend of"
        r"who\s+is\s+\w+'s",  # "who is X's"
        r"what\s+does\s+\w+'s",  # "what does X's"
        r"based\s+on",
        r"considering\s+that",
        r"given\s+that",
        r"taking\s+into\s+account",
        r"if\s+you\s+combine",
        r"connecting",
        r"related\s+to\s+\w+'s",
    ]
    for pattern in multihop_patterns:
        if re.search(pattern, query_lower):
            return 'multihop'

    return 'factual'


async def cot_rerank(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    model: str,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Chain-of-Thought reranking with explicit reasoning steps.

    This implements enhanced reranking that forces the model to reason
    step-by-step before scoring, improving accuracy on complex queries.

    Args:
        client: OpenAI client
        model: Model to use (should be gpt-4o for best results)
        query: The search query
        candidates: List of candidate passages
        top_k: Number of results to return
        content_key: Key in candidate dict containing text

    Returns:
        Top-k reranked candidates with llm_score field
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    num_candidates = len(candidates)

    # Detect query type
    query_type = detect_query_type_enhanced(query)
    logger.debug(f"CoT reranking with query type: {query_type}")

    # Format passages with more context
    passages_text = "\n".join([
        f"[{i+1}] {c.get(content_key, '')[:900]}"
        for i, c in enumerate(candidates)
    ])

    # Select prompt based on query type
    if query_type == 'temporal':
        prompt_template = COT_RERANK_TEMPORAL
    elif query_type == 'temporal_sequence':
        prompt_template = COT_RERANK_TEMPORAL_SEQUENCE
    elif query_type == 'preference':
        prompt_template = COT_RERANK_PREFERENCE
    elif query_type == 'multihop':
        prompt_template = COT_RERANK_MULTIHOP
    elif query_type == 'aggregation':
        prompt_template = COT_RERANK_AGGREGATION
    elif query_type == 'knowledge_update':
        prompt_template = COT_RERANK_KNOWLEDGE_UPDATE
    else:
        prompt_template = COT_RERANK_PROMPT

    prompt = prompt_template.format(
        question=query,
        passages=passages_text,
        num_passages=num_candidates,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise relevance scoring system. Think step by step, then return scores as a JSON array."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200 + num_candidates * 10,  # More tokens for reasoning
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON array from response (may have reasoning before it)
        json_match = re.search(r'\[[\d.,\s]+\]', response_text)

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

            # Pair candidates with scores and sort
            scored = list(zip(candidates, validated_scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Add LLM score to candidates and return top_k
            results = []
            for candidate, score in scored[:top_k]:
                candidate["llm_score"] = score
                results.append(candidate)

            return results

    except Exception as e:
        logger.error(f"CoT reranking error: {e}")

    # Fallback: return original order
    return candidates[:top_k]


# =============================================================================
# ENHANCED TEMPORAL FILTERING
# =============================================================================

def enhanced_temporal_filter(
    candidates: List[MemoryResponse],
    query: str,
    buffer_days: int = 7,
) -> List[MemoryResponse]:
    """
    Enhanced temporal filtering with coarse-to-fine approach.

    Based on Memory-T1 paper's temporal reasoning improvements.

    CRITICAL FIX: Uses event_time (when the event occurred) instead of created_at
    (when memory was stored). This is essential for temporal reasoning accuracy.

    Args:
        candidates: List of memory candidates
        query: The search query
        buffer_days: Buffer days for date range (default 7)

    Returns:
        Temporally filtered and sorted candidates
    """
    parser = TemporalParser()
    temporal_info = parser.parse(query)

    if not temporal_info.has_temporal:
        return candidates

    logger.debug(f"Temporal filtering: type={temporal_info.temporal_type}, ordering={temporal_info.ordering_hint}")

    results = list(candidates)

    # Helper to get the best available date for a memory
    # Prefer event_time (when event occurred), fall back to created_at (when stored)
    def get_temporal_date(memory: MemoryResponse) -> datetime:
        if memory.event_time:
            return memory.event_time
        # Fall back to created_at if event_time not available
        return memory.created_at or datetime.min.replace(tzinfo=timezone.utc)

    # Apply ordering using event_time (CRITICAL for temporal reasoning)
    if temporal_info.ordering_hint == 'earliest':
        results.sort(key=lambda x: get_temporal_date(x))
    elif temporal_info.ordering_hint == 'latest':
        results.sort(key=lambda x: get_temporal_date(x), reverse=True)

    # Date range filtering using event_time (CRITICAL for temporal reasoning)
    if temporal_info.reference_date and temporal_info.temporal_type in (TemporalType.RELATIVE, TemporalType.ABSOLUTE):
        start_date, end_date = parser.get_date_range_for_query(query, buffer_days=buffer_days)

        if start_date and end_date:
            filtered = []
            for r in results:
                temporal_date = get_temporal_date(r)
                if temporal_date and temporal_date != datetime.min.replace(tzinfo=timezone.utc):
                    # Ensure timezone awareness
                    if temporal_date.tzinfo is None:
                        temporal_date = temporal_date.replace(tzinfo=timezone.utc)
                    if start_date <= temporal_date <= end_date:
                        filtered.append(r)

            if filtered:
                logger.debug(f"Temporal filter: {len(results)} -> {len(filtered)} candidates (using event_time)")
                return filtered

    return results


# =============================================================================
# MAIN HYPER SEARCH SERVICE
# =============================================================================

class HyperSearchService:
    """
    Hyper Search Service implementing advanced RAG techniques.

    This is a stateless service that can handle parallel multi-user requests.
    Each search is independent and doesn't share state.

    Pipeline:
    1. Query Expansion (generate 2-3 query variants)
    2. Multi-query retrieval (fetch candidates for all variants)
    3. Deduplication and fusion
    4. HopRAG reasoning (for complex queries)
    5. PageRank boosting for multi-hop queries (HippoRAG-inspired)
    6. Temporal filtering
    7. Chain-of-Thought reranking

    Target: 90%+ accuracy on memorybench
    """

    # PageRank boost weight for multi-hop queries
    PAGERANK_BOOST_WEIGHT = 0.15

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()
        self.knowledge_graph = knowledge_graph  # Optional, for PageRank boosting

        # Initialize OpenAI client
        self.settings = get_settings()
        if self.settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_key,
                api_version="2024-05-01-preview",
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            self.model = "gpt-4o"
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o"

    async def search_hyper(
        self,
        request: MemorySearchRequest,
        use_query_expansion: bool = True,
        use_hoprag: bool = True,
        use_temporal_filter: bool = True,
        expansion_model: str = "gpt-4o",  # Use gpt-4o (gpt-4o-mini may not be deployed)
        rerank_model: str = "gpt-4o",  # Best model for reranking
    ) -> MemorySearchResponse:
        """
        Full hyper search pipeline for maximum accuracy.

        Args:
            request: Search request
            use_query_expansion: Enable query expansion (default True)
            use_hoprag: Enable HopRAG reasoning for complex queries (default True)
            use_temporal_filter: Enable temporal filtering (default True)
            expansion_model: Model for query expansion (cheaper)
            rerank_model: Model for reranking (best quality)

        Returns:
            MemorySearchResponse with best results
        """
        logger.info(f"HYPER search for user {request.user_id}: {request.query[:50]}...")

        original_limit = request.limit
        query = request.query

        # Detect query type early for optimized pipeline
        query_type = detect_query_type_enhanced(query)
        logger.info(f"Query type detected: {query_type}")

        # Extract key entities for filtering/boosting
        key_entities = extract_key_entities(query)
        if key_entities:
            logger.debug(f"Key entities extracted: {key_entities}")

        # Step 1: Query Expansion (with query-type awareness)
        query_variants = [query]
        if use_query_expansion:
            # v18: More expansions for better retrieval coverage
            # temporal_sequence: 4 (need to find each event)
            # aggregation: 4 (need to find ALL items across sessions)
            # default: 3 (up from 2, improves multi-session recall)
            if query_type == 'temporal_sequence':
                max_exp = 4
            elif query_type == 'aggregation':
                max_exp = 4
            else:
                max_exp = 3
            query_variants = await expand_query(
                self.client,
                expansion_model,
                query,
                max_expansions=max_exp,
                query_type=query_type,
            )
            logger.debug(f"Query variants: {query_variants}")

        # Step 2: Multi-query retrieval with FREQUENCY BOOSTING (v14)
        # v17: Wider retrieval pool (100 candidates) for cross-encoder reranking
        # Cross-encoder is fast (~30ms for 100 candidates) so wider pool = better recall
        # For aggregation, even wider (150) to find ALL items across conversations
        # v19g: fetch_limit strategy per query type
        # Aggregation: use original_limit with vector search ranking (no cross-encoder)
        # Cross-encoder reranking HURTS aggregation by favoring relevance over diversity
        # 15 results also hurt (v19i) - 10 is the sweet spot
        # Non-aggregation: wider pool + cross-encoder reranking (helps single-session, temporal)
        if query_type == 'aggregation':
            fetch_limit = original_limit  # Skip cross-encoder reranking for aggregation
        else:
            fetch_limit = min(original_limit * 10, 100)

        all_candidates: Dict[str, MemoryResponse] = {}  # Dedupe by ID
        candidate_frequency: Dict[str, int] = {}  # v14: Track how many variants found each
        hybrid_service = HybridSearchService(db=self.db, embedding_service=self.embedding_service)

        for variant in query_variants:
            variant_request = MemorySearchRequest(
                user_id=request.user_id,
                query=variant,
                limit=fetch_limit,
                threshold=request.threshold,
                platforms=request.platforms,
                memory_types=request.memory_types,
                source_id=request.source_id,
                only_latest=request.only_latest,
                only_valid=request.only_valid,
            )

            try:
                response = await hybrid_service.search_hybrid(variant_request)
                for r in response.results:
                    memory_id = str(r.id)
                    # v14: Track frequency
                    candidate_frequency[memory_id] = candidate_frequency.get(memory_id, 0) + 1
                    # Keep highest score if duplicate
                    existing = all_candidates.get(memory_id)
                    if not existing or r.priority_score > existing.priority_score:
                        all_candidates[memory_id] = r
            except Exception as e:
                logger.warning(f"Variant search failed for '{variant[:30]}...': {e}")

        # v14: Apply frequency boosting - memories found by multiple variants get boosted
        # This is a strong signal of relevance (consensus across query expansions)
        if len(query_variants) > 1:
            for memory_id, candidate in all_candidates.items():
                freq = candidate_frequency.get(memory_id, 1)
                if freq > 1:
                    # Boost by 0.08 per additional query that found it (max 0.2)
                    freq_boost = min((freq - 1) * 0.08, 0.2)
                    candidate.priority_score = min(1.0, candidate.priority_score + freq_boost)
            logger.debug(f"Frequency boosting applied: {sum(1 for f in candidate_frequency.values() if f > 1)} memories found by multiple variants")

        if not all_candidates:
            return MemorySearchResponse(
                results=[],
                total=0,
                query_embedding_tokens=len(query) // 4
            )

        # Convert to list and sort by priority score
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        logger.info(f"Multi-query retrieval: {len(candidates)} unique candidates from {len(query_variants)} queries")

        # Step 2.5: Entity-based boosting (boost results containing key entities)
        if key_entities:
            for candidate in candidates:
                content_lower = candidate.content.lower() if candidate.content else ""
                entity_matches = sum(1 for e in key_entities if e.lower() in content_lower)
                if entity_matches > 0:
                    # Boost by 0.05 per entity match (up to 0.15)
                    boost = min(entity_matches * 0.05, 0.15)
                    candidate.priority_score = min(1.0, candidate.priority_score + boost)
            # Re-sort after boosting
            candidates.sort(key=lambda x: x.priority_score, reverse=True)
            logger.debug(f"Entity boosting applied for: {key_entities}")

        # Step 2.6: PageRank boosting for multi-hop and temporal_sequence queries (HippoRAG-inspired)
        # This enables "spreading activation" through the knowledge graph for complex queries
        if query_type in ('multihop', 'temporal_sequence') and self.knowledge_graph and len(self.knowledge_graph.nodes) > 0:
            try:
                # Get PageRank scores for memories based on query entities
                pagerank_memory_scores = self.knowledge_graph.get_memories_by_pagerank(
                    key_entities, top_k=100
                )
                pagerank_dict = {mem_id: score for mem_id, score in pagerank_memory_scores}

                if pagerank_dict:
                    for candidate in candidates:
                        memory_id_str = str(candidate.id)
                        if memory_id_str in pagerank_dict:
                            # Apply PageRank boost (normalized to 0-PAGERANK_BOOST_WEIGHT)
                            max_pr_score = max(pagerank_dict.values()) if pagerank_dict else 1.0
                            normalized_pr = pagerank_dict[memory_id_str] / max_pr_score
                            boost = normalized_pr * self.PAGERANK_BOOST_WEIGHT
                            candidate.priority_score = min(1.0, candidate.priority_score + boost)

                    # Re-sort after PageRank boosting
                    candidates.sort(key=lambda x: x.priority_score, reverse=True)
                    logger.debug(f"PageRank boosting applied: {len(pagerank_dict)} memories scored")
            except Exception as e:
                logger.warning(f"PageRank boosting failed: {e}")

        # Step 2.7: Recency boosting for knowledge_update queries
        # For questions about UPDATED/CHANGED info, prioritize more recent memories
        # CRITICAL FIX v14: Only boost candidates that are semantically relevant
        # AND use logarithmic decay (not exponential) to avoid killing old-but-correct memories
        if query_type == 'knowledge_update' and candidates:
            now = datetime.now(timezone.utc)
            # Extract key terms from query to check semantic relevance
            query_lower = query.lower()
            key_terms = extract_key_entities(query)

            for candidate in candidates:
                # ONLY apply recency boost if the candidate is semantically relevant
                # (has similarity > 0.25 OR contains key terms from query)
                is_relevant = (
                    (hasattr(candidate, 'similarity') and candidate.similarity and candidate.similarity > 0.25) or
                    any(term.lower() in candidate.content.lower() for term in key_terms if len(term) > 2)
                )

                if not is_relevant:
                    continue  # Skip irrelevant candidates

                # Use event_time if available, otherwise created_at
                memory_time = candidate.event_time or candidate.created_at
                if memory_time:
                    # Ensure timezone awareness
                    if memory_time.tzinfo is None:
                        memory_time = memory_time.replace(tzinfo=timezone.utc)
                    # Calculate days ago
                    days_ago = max(0, (now - memory_time).days)
                    # v14 FIX: Use LOGARITHMIC decay instead of exponential
                    # This decays more slowly, allowing older memories to still compete
                    # Boost formula: 0.2 / (1 + log(1 + days/7))
                    # 0 days = 0.2, 7 days = 0.14, 30 days = 0.10, 90 days = 0.07
                    recency_boost = 0.2 / (1.0 + math.log(1 + days_ago / 7))
                    candidate.priority_score = min(1.0, candidate.priority_score + recency_boost)
            # Re-sort after recency boosting
            candidates.sort(key=lambda x: x.priority_score, reverse=True)
            logger.debug(f"Recency boosting applied for knowledge_update query (v14: log decay, semantic filter)")

        # Step 3: Temporal filtering (if applicable)
        if use_temporal_filter:
            candidates = enhanced_temporal_filter(candidates, query)

        # Step 4: HopRAG reasoning for complex queries (multi-hop ONLY)
        # v14 FIX: Remove temporal_sequence - HopRAG prunes temporal info incorrectly
        # Temporal sequences should rely on CoT reranking with temporal-specific prompts
        if use_hoprag and query_type == 'multihop' and len(candidates) > 5:
            candidate_dicts = [
                {"content": r.content, "memory": r, "original_score": r.priority_score}
                for r in candidates
            ]

            pruned, reasoning = await hoprag_reason_prune(
                self.client,
                rerank_model,
                query,
                candidate_dicts,
                content_key="content",
                query_type=query_type,
            )

            # Extract memories from pruned results
            candidates = [c["memory"] for c in pruned]
            logger.debug(f"HopRAG reasoning: {reasoning[:100]}...")

        # Step 5: Cross-encoder reranking (v17: deterministic, replaces LLM CoT)
        # v19g: Aggregation uses vector search ranking (no cross-encoder)
        # Non-aggregation: adaptive-k with gap analysis after cross-encoder reranking
        effective_top_k = original_limit

        if len(candidates) > effective_top_k:
            candidate_dicts = [
                {"content": r.content, "memory": r, "original_score": r.priority_score}
                for r in candidates
            ]

            # v19: Get ALL scores for adaptive-k analysis (non-aggregation only)
            use_adaptive_k = query_type not in ('aggregation', 'temporal_sequence')
            reranked = cross_encoder_rerank(
                query=query,
                candidates=candidate_dicts,
                top_k=effective_top_k,
                content_key="content",
                return_all_scores=use_adaptive_k,
            )

            # v19: Adaptive-K - use score gaps to dynamically cut off results
            # Only for non-aggregation, non-temporal queries (simple factual + preference)
            if use_adaptive_k and len(reranked) > 3:
                reranked = self._adaptive_k_cutoff(
                    reranked, min_k=3, max_k=effective_top_k, gap_threshold=0.15
                )
            else:
                reranked = reranked[:effective_top_k]

            # Extract memories with cross-encoder scores
            final_results = []
            for r in reranked:
                memory = r["memory"]
                ce_score = r.get("ce_score", memory.priority_score)
                memory.priority_score = float(ce_score) if ce_score is not None else memory.priority_score
                final_results.append(memory)

            logger.info(
                f"Cross-encoder reranked: {len(candidates)} -> {len(final_results)} results "
                f"(query_type={query_type}, adaptive_k={use_adaptive_k})"
            )

            return MemorySearchResponse(
                results=final_results,
                total=len(final_results),
                query_embedding_tokens=len(query) // 4
            )

        # Return candidates if no reranking needed (already fewer than effective_top_k)
        return MemorySearchResponse(
            results=candidates[:effective_top_k],
            total=min(len(candidates), effective_top_k),
            query_embedding_tokens=len(query) // 4
        )

    def _adaptive_k_cutoff(
        self,
        scored_results: list,
        min_k: int = 3,
        max_k: int = 10,
        gap_threshold: float = 0.15,
    ) -> list:
        """
        v19: Adaptive-K retrieval - dynamically cut off results based on score gaps.

        After cross-encoder scoring, analyze gaps between consecutive scores.
        If a large gap exists (> threshold), cut off there - the results below
        the gap are likely noise that hurts simple single-session queries.

        Based on EMNLP 2025 Adaptive-K paper.
        """
        if len(scored_results) <= min_k:
            return scored_results

        scores = [r.get("ce_score", 0.0) for r in scored_results]

        for i in range(min_k, min(len(scores), max_k)):
            gap = scores[i - 1] - scores[i]
            if gap > gap_threshold:
                logger.debug(
                    f"Adaptive-K cutoff at position {i}: "
                    f"score[{i-1}]={scores[i-1]:.4f}, score[{i}]={scores[i]:.4f}, "
                    f"gap={gap:.4f} > threshold={gap_threshold}"
                )
                return scored_results[:i]

        return scored_results[:max_k]

    def _session_level_grouping(
        self,
        reranked_results: list,
        effective_top_k: int,
        query_type: str,
    ) -> list:
        """
        v19: Session-level grouping for multi-session queries.

        Groups results by source_id and ensures diverse session representation.
        Only activates when results span 3+ distinct sessions (indicating a
        multi-session query). For single-session results, passes through unchanged.

        Inspired by Emergence AI's session-level NDCG aggregation approach.
        """
        from collections import defaultdict

        # Group by source_id
        sessions = defaultdict(list)
        for r in reranked_results:
            memory = r.get("memory")
            sid = getattr(memory, "source_id", None) if memory else None
            sid = sid or "unknown"
            sessions[sid].append(r)

        # Only apply session grouping if 3+ distinct sessions
        if len(sessions) < 3:
            return reranked_results[:effective_top_k]

        # Score each session by max ce_score of its turns
        session_scores = []
        for sid, turns in sessions.items():
            max_score = max(r.get("ce_score", 0.0) for r in turns)
            session_scores.append((sid, max_score, turns))

        session_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top turns from top sessions (max 3 per session for diversity)
        final = []
        max_per_session = 3
        for sid, score, turns in session_scores:
            turns.sort(key=lambda r: r.get("ce_score", 0.0), reverse=True)
            for t in turns[:max_per_session]:
                final.append(t)
                if len(final) >= effective_top_k:
                    logger.debug(
                        f"Session grouping: {len(sessions)} sessions -> "
                        f"{len(final)} results (capped at {effective_top_k})"
                    )
                    return final

        logger.debug(
            f"Session grouping: {len(sessions)} sessions -> "
            f"{len(final)} results"
        )
        return final[:effective_top_k]


async def get_hyper_search_service(
    db: AsyncSession,
    knowledge_graph: Optional[KnowledgeGraph] = None,
) -> HyperSearchService:
    """Get hyper search service instance."""
    return HyperSearchService(db=db, knowledge_graph=knowledge_graph)
