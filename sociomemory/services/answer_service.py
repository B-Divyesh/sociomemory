"""
Answer Service - Core answer generation with Chain-of-Note reasoning.

Ported from benchmarks/memorybench/src/providers/sociomemory/prompts.ts (v25).
This moves all prompt engineering intelligence into the core API so that
any caller gets the full Chain-of-Note + entity verification + self-consistency
voting pipeline, not just raw search results.

v37 improvements:
- CRAG (Corrective RAG): When answer is insufficient, generate targeted queries for re-search
- Decompose-and-Recount: Memory-by-memory extraction for aggregation questions
- Previous: v36 majority-vote consensus, insufficient-retry for both paths
"""
import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from sociomemory.config import get_settings
from sociomemory.models.answer import QuestionTypeInfo

logger = logging.getLogger(__name__)

# Newline constant for use in f-strings (Python 3.11 doesn't allow backslashes in f-string braces)
_NL = "\n"


def _date_line(question_date: Optional[str]) -> str:
    """Build the Today's Date line for prompts."""
    if question_date:
        return f"\nToday's Date: {question_date}"
    return ""


# SSA detection regex - matches questions about what the assistant said/did
SSA_PATTERN = re.compile(
    r"(?:you (?:recommend|suggest|creat|mention|told|gave|provid|describ))"
    r"|(?:(?:our|the) previous (?:conversation|chat|discussion|game|session))"
    r"|(?:remind me (?:of|about|what))"
    r"|(?:looking back at)",
    re.IGNORECASE,
)


def format_date(date_str: Optional[str]) -> Optional[str]:
    """Format a date string for display in context."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")  # e.g., "January 15, 2025"
    except (ValueError, TypeError):
        return None


def compress_conversation(content: str) -> str:
    """
    Strip assistant messages from conversation content, keeping only user messages.
    Assistant responses are typically 500-3000 chars of generic advice that add noise.
    User messages contain the actual personal information we need for answering.
    Keeps the session header and [user]: lines, compresses [assistant]: to a brief note.
    """
    lines = content.split("\n")
    compressed: list[str] = []
    in_assistant = False
    assistant_line_count = 0

    for line in lines:
        if line.startswith("[assistant]:") or line.startswith("[assistant]"):
            if not in_assistant:
                in_assistant = True
                assistant_line_count = 0
            assistant_line_count += 1
            # Keep first line of assistant response as context
            if assistant_line_count == 1:
                brief = line[:150]
                compressed.append(brief + ("..." if len(line) > 150 else ""))
            continue

        if line.startswith("[user]:") or line.startswith("[user]"):
            in_assistant = False
            assistant_line_count = 0

        if not in_assistant:
            compressed.append(line)

    return "\n".join(compressed)


def build_context(
    results: list[dict[str, Any]],
    sort_newest_first: bool = False,
    sort_chronological: bool = False,
    compress_assistant: bool = False,
) -> str:
    """Build context string from search results."""
    if not results:
        return "No relevant memories found."

    ordered = list(results)
    if sort_chronological:
        ordered.sort(
            key=lambda r: datetime.fromisoformat(
                (r.get("event_time") or r.get("created_at") or "1970-01-01T00:00:00").replace("Z", "+00:00")
            ).timestamp()
        )
    elif sort_newest_first:
        ordered.sort(
            key=lambda r: datetime.fromisoformat(
                (r.get("event_time") or r.get("created_at") or "1970-01-01T00:00:00").replace("Z", "+00:00")
            ).timestamp(),
            reverse=True,
        )

    memory_strings = []
    for i, result in enumerate(ordered):
        date_str = format_date(result.get("event_time")) or format_date(result.get("created_at"))
        content = result.get("content", "")
        if compress_assistant:
            content = compress_conversation(content)

        if date_str:
            memory_strings.append(f"[{i + 1}] ({date_str})\n{content}")
        else:
            memory_strings.append(f"[{i + 1}] {content}")

    return "\n\n".join(memory_strings)


def detect_question_type(question: str) -> QuestionTypeInfo:
    """Detect question type for routing to appropriate prompt."""
    q = question.lower()

    is_aggregation = (
        q.startswith("how many ")
        or "how much total" in q
        or "how much money" in q
        or "list all" in q
        or "what are all" in q
        or "name all" in q
        or "count" in q
        or "total number" in q
        or "total amount" in q
        or "in total" in q
        or "average" in q
    )

    is_temporal = (
        q.startswith("when ")
        or "how many days" in q
        or "how many weeks" in q
        or "how many months" in q
        or "how long" in q
        or "before" in q
        or "after" in q
        or "first" in q
        or "last" in q
        or "recent" in q
        or "order" in q
        or "ago" in q
    )

    is_knowledge_update = (
        "new " in q
        or "current " in q
        or "currently" in q
        or "now " in q
        or "updated " in q
        or "changed " in q
        or "switched " in q
        or "recently " in q
        or "most recent" in q
        or "increase" in q
        or "decrease" in q
        or ("what is my" in q and not is_aggregation)
        or ("where do i" in q and not is_aggregation)
        or ("where did" in q and "move" in q)
    )

    is_preference = (
        "recommend" in q
        or "suggest" in q
        or "tips" in q
        or "advice" in q
        or "preference" in q
        or "prefer" in q
        or "favorite" in q
        or "like for me" in q
        or "ideas for" in q
        or "what should i" in q
        or "can you suggest" in q
        or "any tips" in q
        or "helpful tips" in q
    )

    asks_old_value = (
        "initially" in q
        or "previously" in q
        or "used to" in q
        or "was my previous" in q
        or "before getting" in q
        or "before i " in q
        or ("old " in q and "how old" not in q)
    )

    is_complex = is_aggregation or is_temporal or is_preference

    return QuestionTypeInfo(
        is_aggregation=is_aggregation,
        is_temporal=is_temporal,
        is_knowledge_update=is_knowledge_update,
        is_preference=is_preference,
        is_complex=is_complex,
        asks_old_value=asks_old_value,
    )


def build_simple_prompt(
    question: str,
    results: list[dict[str, Any]],
    question_date: Optional[str],
    q_type: QuestionTypeInfo,
) -> str:
    """Simple prompt for straightforward single-session questions."""
    sort_newest = q_type.is_knowledge_update and not q_type.asks_old_value
    is_assistant_question = bool(SSA_PATTERN.search(question))
    do_compress = not is_assistant_question
    retrieved_context = build_context(results, sort_newest, False, do_compress)

    prompt = f"""Answer the question using ONLY the retrieved memories below.

Question: {question}{_date_line(question_date)}

Retrieved Memories:
{retrieved_context}

Instructions:
1. Read each memory carefully, noting any relevant dates and facts
2. Extract the specific answer from the relevant memory/memories"""

    if q_type.is_knowledge_update and q_type.asks_old_value:
        prompt += """
3. CRITICAL: The user is asking about a PREVIOUS/INITIAL/OLD value, NOT the current one
4. If the topic has been updated over time, use the EARLIEST/ORIGINAL value
5. Look for the FIRST mention of this topic, which represents the original state
6. Do NOT use the most recent/current value - the user specifically asks about what it WAS before"""
    elif q_type.is_knowledge_update:
        prompt += """
3. CRITICAL: The user is asking for UPDATED/CURRENT information
4. The memories are sorted NEWEST FIRST - Memory [1] has the LATEST date
5. If multiple memories mention this topic, ALWAYS use the value from the memory with the LATEST date
6. NEVER use an older value when a newer one exists
7. The answer MUST be the CURRENT value, NOT any old/previous value"""
    else:
        prompt += """
3. Read EVERY memory thoroughly from start to end - the answer is likely embedded within a conversation
4. Look for the user's own statements (lines starting with [user]:) - they often casually mention the answer
5. If not directly stated, infer from patterns, preferences, or context in the memories
6. You MUST provide an answer. Do NOT say "I don't know" or "the information is not enough" - the answer IS in the memories, find it"""

    ku_note = ", identifying the MOST RECENT value by date" if q_type.is_knowledge_update else ""
    ku_answer = " - provide the UPDATED/CURRENT value" if q_type.is_knowledge_update else ""
    prompt += f"""

Notes:
[Brief analysis of relevant memories{ku_note}]

Answer: [Your concise answer{ku_answer}]"""

    return prompt


def build_chain_of_note_prompt(
    question: str,
    results: list[dict[str, Any]],
    question_date: Optional[str],
    q_type: QuestionTypeInfo,
) -> str:
    """Chain-of-Note prompt for complex multi-session/temporal/preference questions."""
    sort_newest = q_type.is_knowledge_update and not q_type.is_temporal and not q_type.asks_old_value
    is_assistant_question = bool(SSA_PATTERN.search(question))
    do_compress = not is_assistant_question
    retrieved_context = build_context(results, sort_newest, q_type.is_temporal, do_compress)
    num_results = len(results)

    prompt = f"""You are a memory analysis assistant. Answer the question by carefully extracting and reasoning over facts from the retrieved conversation memories.

Question: {question}{_date_line(question_date)}

Retrieved Memories:
{retrieved_context}

=== STEP 1: FACT EXTRACTION ===
For each memory, extract ONLY the facts relevant to answering this question.
Format as bullet points:
- Memory [N] (date): [relevant fact 1]; [relevant fact 2]
- Skip memories with NO relevant information.
- IMPORTANT: Check ALL {num_results} memories, do not stop early.
"""

    # CRITICAL: Check isTemporal BEFORE isAggregation
    if q_type.is_temporal:
        qd = question_date or "[not provided]"
        prompt += f"""
=== STEP 2: TEMPORAL REASONING ===
From the extracted facts:
a) List ALL relevant events with their EXACT dates in YYYY-MM-DD format:
   - Event: [description], Date: YYYY-MM-DD (from Memory [N])
   - Include ALL events from ALL memories, even if they seem less relevant.
b) For time difference questions:
   - Start date: YYYY-MM-DD
   - End date: YYYY-MM-DD (use Today's Date: {qd} if asking "how long ago" or "since")
   - Calculation: [show day-by-day or month-by-month arithmetic]
   - Result: [X days/weeks/months]
c) For ordering questions:
   - Sort ALL events by date (earliest to latest)
   - List the COMPLETE ordered sequence
   - Answer based on position in chronological order
d) For "last Saturday/Monday" questions:
   - Today's Date: {qd}
   - Count backwards from today to find the most recent [day]
   - Most recent [day]: YYYY-MM-DD
   - Find which memory matches that date

Common errors to avoid:
- Using the wrong date (memory date vs question date vs event date)
- Off-by-one errors in date arithmetic
- Confusing "X ago" with "X from now"
- Not listing ALL events before ordering
- Forgetting to convert dates to YYYY-MM-DD before calculating
"""
    elif q_type.is_aggregation:
        ku_agg = ""
        if q_type.is_knowledge_update and not q_type.asks_old_value:
            ku_agg = "\nKNOWLEDGE UPDATE: If the question asks about CURRENT counts, use only the MOST RECENT information. Earlier memories may be outdated."
        elif q_type.is_knowledge_update and q_type.asks_old_value:
            ku_agg = "\nPREVIOUS VALUE: The user asks about a previous/initial state. Use the EARLIEST information, not the latest update."
        prompt += f"""
=== STEP 2: MEMORY-BY-MEMORY EXTRACTION ===
Go through EACH memory one at a time and extract matching items:

Memory [1]:
- Items found that match the question: [list each specific item, or "none"]
- Date context: [when was this mentioned]

Memory [2]:
- Items found that match the question: [list each specific item, or "none"]
- Date context: [when was this mentioned]

... continue for ALL {num_results} memories. Do NOT skip any.

=== STEP 2b: DEDUPLICATION ===
Now compile ALL items found across all memories into a single list:
- For each item, note which memory/memories mention it
- If the SAME item appears in multiple memories, count it ONLY ONCE
- If an item was REPLACED/UPDATED (e.g., old pet → new pet), decide if both should count based on the question
- IMPORTANT: Different items that sound similar are DIFFERENT (e.g., "hiking in Alps" and "hiking in Rockies" are 2 items)
{ku_agg}

Unique items after dedup (numbered):
1. [item] (Memory [N])
2. [item] (Memory [N])
...

=== STEP 2c: TEMPORAL FILTER ===
- Does the question have a time constraint? (e.g., "in the past month", "since January", "this year")
- If YES: filter the list above to only items within that time range
- If NO: keep all items
- Final filtered list: [numbered list]
- FINAL COUNT: [number]

Common errors to avoid:
- Counting the same item twice because it appears in multiple memories
- Missing items buried in conversation text - READ EACH MEMORY COMPLETELY
- Confusing related but different items (they ARE different, count separately)
- Ignoring time constraints in the question
"""
    elif q_type.is_preference:
        prompt += """
=== STEP 2: PREFERENCE ANALYSIS ===
From the extracted facts:
a) List the user's SPECIFIC stated preferences and experiences EXACTLY as mentioned in memories:
   - Quote directly: "I love/prefer/enjoy [X]" (from Memory [N])
   - Equipment/tools owned: exact names, brands, models
   - Past experiences with specific results (positive/negative)
b) Identify the MOST RELEVANT user context for this specific question:
   - What specific items/tools do they already own related to this question?
   - What specific experiences have they had related to this question?
   - What specific preferences have they expressed related to this question?
c) Generate ONLY 2-3 highly personalized recommendations:
   - EVERY recommendation must cite a specific detail from the user's memories
   - Do NOT include generic suggestions that could apply to anyone
   - Do NOT pad with extra recommendations beyond what's specifically supported by memories
   - If a recommendation doesn't reference a specific user detail, REMOVE IT

CRITICAL: The answer should demonstrate knowledge of THIS user's specific situation.
A good answer references specific brands, models, past experiences, or stated preferences.
A bad answer gives generic advice that any person asking the same question would receive.
"""

    prompt += """
=== STEP 3: ENTITY VERIFICATION ===
Before answering, verify key entities:
- List the KEY entities/items from the question (e.g., specific people, places, objects, events)
- For each, check: Does this EXACT entity appear in ANY memory?
- If the question mentions an entity (e.g., "iPad", "uncle", "table tennis", "Shinjuku") but memories only mention a DIFFERENT entity in the same category (e.g., "laptop", "friend", "tennis", "Harajuku") → this is a MISMATCH → say "the information is not enough"
- If the key entities DO appear in the memories → you MUST provide an answer, do NOT say "not enough"

=== STEP 4: ANSWER ===
Based on your analysis above, provide a concise, direct answer to the question."""

    if q_type.is_temporal:
        prompt += " Include the specific date(s) or time difference."
    elif q_type.is_aggregation:
        prompt += " State the number clearly. If you listed N items, the answer is N."
    elif q_type.is_preference:
        prompt += " Give 2-3 concise, highly personalized recommendations. Each MUST reference a specific user detail from the memories. Remove any generic advice."

    if q_type.is_knowledge_update and q_type.asks_old_value:
        prompt += " IMPORTANT: The user asks about a PREVIOUS/INITIAL/OLD value. Use the EARLIEST mention, NOT the most recent update."
    elif q_type.is_knowledge_update:
        prompt += " IMPORTANT: Use the MOST RECENT/CURRENT value from the memories. If a value was updated, use the latest version."

    prompt += """
IMPORTANT: If the key entities from the question ARE found in the memories, you MUST provide a specific answer. Only say "the information is not enough" if a critical entity is NOT mentioned in any memory.

Answer:"""

    return prompt


def build_consensus_prompt(
    question: str,
    results: list[dict[str, Any]],
    question_date: Optional[str],
    candidate_answers: list[str],
) -> str:
    """
    GSA-style consensus prompt: synthesize the best answer from multiple candidates.

    Research basis: "LLMs Can Generate a Better Answer by Aggregating Their Own Responses"
    (GSA, March 2025) - generative synthesis outperforms discriminative judging.
    """
    q_type = detect_question_type(question)
    is_assistant_question = bool(SSA_PATTERN.search(question))

    # Build compact memory summary for verification
    memory_lines = []
    for i, r in enumerate(results):
        date_str = format_date(r.get("event_time")) or format_date(r.get("created_at"))
        content = r.get("content", "")
        if is_assistant_question:
            content = content[:1500] + ("..." if len(content) > 1500 else "")
        else:
            content = compress_conversation(content)
            if len(content) > 1500:
                content = content[:1500] + "..."
        prefix = f"({date_str}) " if date_str else ""
        memory_lines.append(f"[{i + 1}] {prefix}{content}")
    memory_summary = "\n\n".join(memory_lines)

    # Extract answer portions from each candidate
    extracted_answers = []
    for i, a in enumerate(candidate_answers):
        match = re.search(r"Answer:\s*([\s\S]*?)$", a, re.IGNORECASE)
        extracted = match.group(1).strip() if match else a[-500:].strip()
        extracted_answers.append(
            f"--- Candidate {i + 1} ---\nFull reasoning:\n{a}\n\nExtracted answer: {extracted}"
        )

    # Type-specific verification instructions
    verify_instructions = ""
    if q_type.is_temporal:
        qd = question_date or "[not provided]"
        verify_instructions = f"""   - For dates/times: Redo the date arithmetic step by step using YYYY-MM-DD format
   - Check: Is the start date correct? Is the end date correct? Is the calculation right?
   - Today's date: {qd}
   - Watch for off-by-one errors and "ago" vs "from now" confusion"""
    elif q_type.is_aggregation:
        verify_instructions = """   - INDEPENDENT RECOUNT: Go through each source memory and list every matching item yourself:
     * Memory [1]: [items found]
     * Memory [2]: [items found]
     * ... (check ALL memories)
   - Compile YOUR count from the memories (don't just trust the candidates)
   - A single memory can mention MULTIPLE items - read each one completely
   - For money: list each amount and sum them yourself
   - Check time constraints (e.g., "past month", "last two months", "this year")
   - If YOUR count differs from the majority → YOUR count from source memories is AUTHORITATIVE
   - Trust YOUR direct count over the candidates' counts
   - IMPORTANT: If the question asks about something NOT mentioned in ANY memory, say "the information is not enough\""""
    elif q_type.is_preference:
        verify_instructions = """   - PERSONALIZATION CHECK: Count how many SPECIFIC user details the answer references:
     * Does it mention specific brands/models the user owns? (e.g., "your Sony A7R IV")
     * Does it reference specific past experiences? (e.g., "building on your lemon cake success")
     * Does it mention specific apps/tools they use? (e.g., "your Suica card")
   - If the answer has FEWER than 2 specific references to user context → IT IS TOO GENERIC
   - If the answer could apply to ANY person asking the same question → IT IS WRONG
   - Pick the candidate that references the MOST specific user details from the memories
   - If NO candidate is sufficiently personalized, rewrite using specific details from the memories
   - CONCISENESS: Keep final answer to 2-3 focused recommendations maximum
   - REMOVE any generic suggestions not grounded in specific user details from the memories
   - Every recommendation should be traceable to a specific memory"""

    if q_type.is_knowledge_update:
        verify_instructions += """
   - CRITICAL: The user asks for CURRENT/UPDATED information
   - Check: Which memory has the LATEST date? Does the answer use that value?
   - If an older value and newer value exist, the answer MUST use the NEWER one"""

    return f"""You are a consensus judge and fact-checker. Three separate reasoning attempts answered the same question. Your job is to find the correct answer.

Question: {question}{_date_line(question_date)}

=== THREE CANDIDATE ANSWERS ===
{(_NL + _NL).join(extracted_answers)}

=== SOURCE MEMORIES (for verification) ===
{memory_summary}

=== YOUR TASK ===
Step 1 - CONSENSUS: Compare the three extracted answers. Do 2 or more agree? What is the majority answer?

Step 2 - VERIFY: Check the majority answer against the source memories:
   - Is the answer directly supported by the memories?
{verify_instructions}

Step 3 - ENTITY VERIFICATION:
   - List the KEY ENTITIES from the question (specific people, places, objects, activities, companies)
   - For EACH key entity, search ALL source memories: Does this EXACT entity appear? (yes/no)
   - MISMATCH EXAMPLES that mean "not enough":
     * Question says "iPad" but memories only mention "laptop" → MISMATCH
     * Question says "uncle" but memories only mention "friend" or "aunt" → MISMATCH
     * Question says "table tennis" but memories only mention "tennis" → MISMATCH
     * Question says "Shinjuku" but memories only mention "Harajuku" → MISMATCH
     * Question says "Google" but memories never mention working at Google → MISMATCH
     * Question says "vintage films" but memories only mention "vintage cameras" → MISMATCH
   - If ANY key entity has a MISMATCH → say "the information is not enough"
   - If ALL key entities ARE found in the memories → you MUST provide a specific answer
   - Do NOT say "not enough" just because you're uncertain - if the entities match, ANSWER IT

Step 4 - DECIDE:
   - If the majority answer is verified correct → use it
   - If the majority answer has an error → check if a minority answer is correct
   - If all answers seem wrong → derive the correct answer from the memories yourself
   - Only say "information is not enough" if the specific entity/event asked about is NOT in any memory

Step 5 - OUTPUT: Write ONLY the final verified answer below. Be concise and direct.

Answer:"""


class AnswerService:
    """
    Generates answers from search results using Chain-of-Note reasoning
    and self-consistency voting with GSA consensus.

    This is the core intelligence that was previously only in the benchmark
    test harness (prompts.ts). Now it's available to any API caller.
    """

    _instance: "AnswerService | None" = None

    def __init__(self) -> None:
        settings = get_settings()
        if settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_key,
                api_version="2024-05-01-preview",
                azure_endpoint=settings.azure_openai_endpoint,
            )
        else:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.azure_openai_chat_deployment

    @classmethod
    def get_instance(cls) -> "AnswerService":
        """Get singleton instance to reuse HTTP client connections."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _llm_call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """Make a single LLM call."""
        params: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Only use seed for deterministic calls (temp=0). For voting calls (temp>0),
        # seed would make all votes identical, defeating self-consistency voting.
        if temperature == 0.0:
            params["seed"] = 42
        response = await self.client.chat.completions.create(**params)
        return (response.choices[0].message.content or "").strip()

    async def generate_crag_queries(
        self,
        question: str,
        search_results: list[dict[str, Any]],
    ) -> list[str]:
        """
        Generate CRAG (Corrective RAG) queries when the initial answer is insufficient.

        Analyzes what information is missing from current results and generates
        2-3 targeted queries to find the missing data.
        """
        # Build a brief summary of what we already have
        existing_topics = []
        for i, r in enumerate(search_results[:5]):
            content = r.get("content", "")[:200]
            existing_topics.append(f"Memory {i + 1}: {content}")
        existing_summary = _NL.join(existing_topics)

        prompt = f"""The user asked: "{question}"

We searched for relevant memories but could not find enough information to answer.

Existing search results (brief):
{existing_summary}

Generate exactly 3 alternative search queries that might find the MISSING information needed to answer the question.
Each query should target a DIFFERENT angle or rephrasing of what's needed.

Rules:
- Be specific and targeted, not generic
- Use different keywords/synonyms than the original question
- Focus on what's MISSING, not what we already found
- Each query should be a short phrase (3-8 words)

Output ONLY the 3 queries, one per line, no numbering or bullets:"""

        response = await self._llm_call(prompt, temperature=0.0, max_tokens=200)
        # Parse response into individual queries
        queries = [line.strip() for line in response.strip().split("\n") if line.strip()]
        # Take at most 3 queries
        return queries[:3]

    async def generate_answer(
        self,
        question: str,
        search_results: list[dict[str, Any]],
        question_date: Optional[str] = None,
        enable_voting: bool = True,
        is_crag_retry: bool = False,
    ) -> dict[str, Any]:
        """
        Generate an answer from search results using the full reasoning pipeline.

        1. Detect question type
        2. Build appropriate prompt (simple or Chain-of-Note)
        3. For complex questions with voting: 3 votes + consensus
        4. For simple questions: single answer at temp=0
        5. Insufficient-retry: if answer says "not enough" but entities likely match, retry
        6. CRAG: if still insufficient and not already a CRAG retry, generate targeted queries

        Returns dict with: answer, question_type, search_results_count, voting_used, duration_ms,
                          crag_queries (if insufficient), crag_iteration
        """
        start_time = time.time()

        q_type = detect_question_type(question)
        is_complex = q_type.is_complex or q_type.is_knowledge_update
        use_voting = enable_voting and is_complex

        if q_type.is_complex:
            prompt = build_chain_of_note_prompt(question, search_results, question_date, q_type)
        else:
            prompt = build_simple_prompt(question, search_results, question_date, q_type)

        if use_voting:
            # Self-consistency voting: 3 votes at temp=0.3, then consensus at temp=0
            votes = await asyncio.gather(
                self._llm_call(prompt, temperature=0.3, max_tokens=2000),
                self._llm_call(prompt, temperature=0.3, max_tokens=2000),
                self._llm_call(prompt, temperature=0.3, max_tokens=2000),
            )

            consensus_prompt = build_consensus_prompt(
                question, search_results, question_date, list(votes)
            )
            final_answer = await self._llm_call(consensus_prompt, temperature=0.0, max_tokens=2000)
        else:
            final_answer = await self._llm_call(prompt, temperature=0.0, max_tokens=2000)

        # Insufficient-retry: if the answer says "not enough", retry once with stronger
        # anti-insufficient instructions. This catches cases where the model is too conservative.
        # Applies to both simple AND voting paths (v36: 12/12 insufficients came from voting).
        crag_queries = None
        if _is_insufficient(final_answer) and len(search_results) > 0:
            retry_prompt = _build_retry_prompt(question, search_results, question_date, q_type)
            final_answer = await self._llm_call(retry_prompt, temperature=0.0, max_tokens=2000)
            logger.info(f"Insufficient-retry triggered for question: {question[:80]}")

            # If STILL insufficient after retry AND this is not already a CRAG re-search,
            # generate CRAG queries for the endpoint to re-search with
            if _is_insufficient(final_answer) and not is_crag_retry:
                crag_queries = await self.generate_crag_queries(question, search_results)
                logger.info(f"CRAG queries generated for question: {question[:80]} -> {crag_queries}")

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": final_answer,
            "question_type": q_type.model_dump(),
            "search_results_count": len(search_results),
            "voting_used": use_voting,
            "duration_ms": duration_ms,
            "crag_queries": crag_queries,
            "crag_iteration": 1 if is_crag_retry else 0,
        }


def _is_insufficient(answer: str) -> bool:
    """Check if the answer says insufficient information."""
    lower = answer.lower()
    return "not enough" in lower or "insufficient" in lower or "not mentioned" in lower


def _build_retry_prompt(
    question: str,
    results: list[dict[str, Any]],
    question_date: Optional[str],
    q_type: QuestionTypeInfo,
) -> str:
    """Build a retry prompt with stronger anti-insufficient instructions."""
    is_assistant_question = bool(SSA_PATTERN.search(question))
    do_compress = not is_assistant_question
    retrieved_context = build_context(results, False, False, do_compress)

    return f"""You previously said the information was not enough. Look again more carefully.

Question: {question}{_date_line(question_date)}

Retrieved Memories:
{retrieved_context}

IMPORTANT INSTRUCTIONS:
1. The answer IS likely in the memories above. Read EVERY word of EVERY memory.
2. Look for INDIRECT mentions - the user may have mentioned the answer casually in passing.
3. Look for [user]: lines - these contain the user's personal information.
4. If the topic is mentioned AT ALL in the memories, you MUST extract an answer from it.
5. Only say "the information is not enough" if the specific topic/entity is TRULY absent from ALL memories.
6. Even partial or approximate answers are better than saying "not enough".

Notes:
[Careful re-analysis of each memory for any relevant information]

Answer:"""
