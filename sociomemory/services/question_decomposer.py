"""
Question Decomposer Service for SocioMemory

Implements StepChain GraphRAG-inspired question decomposition for multi-hop reasoning:
1. Detect if a question requires multiple pieces of information
2. Decompose into sub-questions
3. Execute retrieval for each sub-question
4. Aggregate results for final answer

Key insight from research:
- Multi-session math questions ("How many X in total?") fail because
  they need to retrieve from multiple memories and aggregate
- Preference questions ("What should I...") need to understand user preferences
  from context before answering
- Complex temporal questions need step-by-step reasoning

StepChain approach: BFS traversal of sub-questions with evidence accumulation
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from openai import AsyncAzureOpenAI, AsyncOpenAI

from sociomemory.config import get_settings


logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Type of question determining decomposition strategy."""
    SIMPLE = "simple"              # Single fact retrieval
    AGGREGATION = "aggregation"    # Sum, count, total across memories
    COMPARISON = "comparison"      # Compare X with Y
    TEMPORAL_ORDER = "temporal_order"  # What happened first/last
    PREFERENCE = "preference"      # User preference inference
    MULTI_HOP = "multi_hop"        # Requires connecting multiple facts


@dataclass
class SubQuestion:
    """A sub-question derived from the original question."""
    question: str
    purpose: str  # What information this sub-question seeks
    depends_on: list[int] = field(default_factory=list)  # Indices of prior sub-questions


@dataclass
class DecompositionResult:
    """Result of question decomposition."""
    original_question: str
    question_type: QuestionType
    sub_questions: list[SubQuestion]
    aggregation_instruction: Optional[str] = None  # How to combine sub-answers
    needs_decomposition: bool = True


# Patterns for detecting question types
AGGREGATION_PATTERNS = [
    r'\b(total|sum|combined|altogether|in all|overall)\b',
    r'\bhow many\b.*\bin total\b',
    r'\bhow much\b.*\b(total|combined|altogether)\b',
    r'\bcount\b.*\b(all|both|every)\b',
    r'\b(add|adding)\b.*\btogether\b',
]

COMPARISON_PATTERNS = [
    r'\b(compare|comparing|versus|vs\.?|difference between)\b',
    r'\b(more|less|better|worse|higher|lower)\s+than\b',
    r'\b(which|what)\s+(is|was)\s+(more|less|better|worse)\b',
]

TEMPORAL_ORDER_PATTERNS = [
    r'\b(first|earliest|initial|originally)\b.*\b(then|after|before|later|next)\b',
    r'\border\b.*\b(from|of)\b',
    r'\bwhat happened\b.*\b(first|before|after)\b',
    r'\bsequence\b',
    r'\btimeline\b',
    r'\bchronological\b',
]

PREFERENCE_PATTERNS = [
    r'\bwhat should\b',
    r'\bcan you (suggest|recommend)\b',
    r'\bwhat would\s+(I|you)\s+(like|prefer|want)\b',
    r'\bany (suggestions|recommendations|tips)\b',
    r'\bshould I\b',
]

MULTI_HOP_PATTERNS = [
    r'\bbased on\b.*\bwhat\b',
    r'\bgiven that\b',
    r'\bconsidering\b.*\bwhat\b',
    r'\bif\b.*\bthen\b.*\bwhat\b',
    r'\b(who|what)\b.*\bthat\b.*\b(also|and)\b',
]


# Question decomposition prompt
DECOMPOSITION_PROMPT = """Analyze this question and determine if it requires multiple pieces of information to answer.

Question: {question}

If the question requires:
1. AGGREGATION (counting, summing across multiple items) - decompose into sub-questions for each item
2. COMPARISON (comparing two or more things) - decompose into sub-questions for each thing being compared
3. TEMPORAL_ORDER (sequence of events) - decompose into sub-questions for each event
4. MULTI_HOP (connecting facts) - decompose into sub-questions that build on each other
5. SIMPLE (single fact) - no decomposition needed

Return ONLY valid JSON:
{{
  "question_type": "simple|aggregation|comparison|temporal_order|preference|multi_hop",
  "needs_decomposition": true/false,
  "sub_questions": [
    {{"question": "sub-question 1", "purpose": "what info this seeks"}},
    {{"question": "sub-question 2", "purpose": "what info this seeks"}}
  ],
  "aggregation_instruction": "How to combine sub-answers (null if not applicable)"
}}

Examples:
- "How many fish are there in total in both aquariums?" ->
  {{"question_type": "aggregation", "needs_decomposition": true, "sub_questions": [{{"question": "How many fish are in the first aquarium?", "purpose": "count fish in aquarium 1"}}, {{"question": "How many fish are in the second aquarium?", "purpose": "count fish in aquarium 2"}}], "aggregation_instruction": "Add the fish counts together"}}

- "What is my favorite color?" ->
  {{"question_type": "simple", "needs_decomposition": false, "sub_questions": [], "aggregation_instruction": null}}

JSON:"""


# Preference reasoning prompt
PREFERENCE_PROMPT = """Based on the user's past conversations and preferences, determine what they would likely want.

Question: {question}

Retrieved memories about the user:
{memories}

Analyze the user's preferences, habits, and past choices to infer:
1. What specific preferences are relevant to this question?
2. What constraints or requirements do they typically have?
3. What would they likely prefer based on their history?

Return ONLY valid JSON:
{{
  "relevant_preferences": ["preference 1", "preference 2"],
  "constraints": ["constraint 1"],
  "inferred_preference": "Based on the memories, the user would likely prefer...",
  "confidence": 0.0-1.0
}}

JSON:"""


class QuestionDecomposer:
    """
    Decompose complex questions into sub-questions for better retrieval.

    Implements StepChain GraphRAG approach:
    1. Classify question type
    2. Decompose if needed
    3. Provide aggregation instructions
    """

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

    def detect_question_type_fast(self, question: str) -> QuestionType:
        """
        Fast regex-based question type detection (no LLM).

        Used for quick classification without API calls.
        """
        question_lower = question.lower()

        # Check patterns in order of specificity
        for pattern in AGGREGATION_PATTERNS:
            if re.search(pattern, question_lower):
                return QuestionType.AGGREGATION

        for pattern in TEMPORAL_ORDER_PATTERNS:
            if re.search(pattern, question_lower):
                return QuestionType.TEMPORAL_ORDER

        for pattern in COMPARISON_PATTERNS:
            if re.search(pattern, question_lower):
                return QuestionType.COMPARISON

        for pattern in PREFERENCE_PATTERNS:
            if re.search(pattern, question_lower):
                return QuestionType.PREFERENCE

        for pattern in MULTI_HOP_PATTERNS:
            if re.search(pattern, question_lower):
                return QuestionType.MULTI_HOP

        return QuestionType.SIMPLE

    def decompose_fast(self, question: str) -> DecompositionResult:
        """
        Fast rule-based decomposition (no LLM).

        Uses regex patterns to detect and decompose common question types.
        """
        question_type = self.detect_question_type_fast(question)

        if question_type == QuestionType.SIMPLE:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                sub_questions=[],
                needs_decomposition=False,
            )

        # For aggregation, try to extract items being counted
        if question_type == QuestionType.AGGREGATION:
            # Look for "both X and Y" or "in X and in Y" patterns
            both_match = re.search(r'both\s+(?:of\s+)?(?:my\s+)?(\w+)\s*(?:and|&)\s*(?:my\s+)?(\w+)', question.lower())
            if both_match:
                item1, item2 = both_match.groups()
                return DecompositionResult(
                    original_question=question,
                    question_type=question_type,
                    sub_questions=[
                        SubQuestion(
                            question=f"How many in my {item1}?",
                            purpose=f"Count items in {item1}",
                        ),
                        SubQuestion(
                            question=f"How many in my {item2}?",
                            purpose=f"Count items in {item2}",
                        ),
                    ],
                    aggregation_instruction="Add the two counts together",
                    needs_decomposition=True,
                )

        # For temporal order, extract events
        if question_type == QuestionType.TEMPORAL_ORDER:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                sub_questions=[
                    SubQuestion(
                        question="What events or activities happened?",
                        purpose="Identify events mentioned",
                    ),
                    SubQuestion(
                        question="When did each event happen?",
                        purpose="Get timestamps for ordering",
                    ),
                ],
                aggregation_instruction="Order events chronologically from earliest to latest",
                needs_decomposition=True,
            )

        # For comparison
        if question_type == QuestionType.COMPARISON:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                sub_questions=[
                    SubQuestion(
                        question=question.replace("compare", "describe").replace("vs", "and"),
                        purpose="Get details about items being compared",
                    ),
                ],
                aggregation_instruction="Compare the retrieved information",
                needs_decomposition=True,
            )

        # For preferences - need to understand user context
        if question_type == QuestionType.PREFERENCE:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                sub_questions=[
                    SubQuestion(
                        question="What are my relevant preferences and past choices?",
                        purpose="Understand user preferences",
                    ),
                    SubQuestion(
                        question=question,
                        purpose="Apply preferences to current question",
                    ),
                ],
                aggregation_instruction="Infer answer based on user preferences",
                needs_decomposition=True,
            )

        # Default for multi-hop
        return DecompositionResult(
            original_question=question,
            question_type=question_type,
            sub_questions=[
                SubQuestion(
                    question=question,
                    purpose="Primary retrieval",
                ),
            ],
            needs_decomposition=False,
        )

    async def decompose(self, question: str, use_llm: bool = True) -> DecompositionResult:
        """
        Decompose a question into sub-questions.

        Args:
            question: The question to decompose
            use_llm: Whether to use LLM for decomposition (slower but more accurate)

        Returns:
            DecompositionResult with sub-questions and aggregation instructions
        """
        if not use_llm:
            return self.decompose_fast(question)

        # Quick check: if simple question detected, skip LLM
        question_type = self.detect_question_type_fast(question)
        if question_type == QuestionType.SIMPLE:
            return DecompositionResult(
                original_question=question,
                question_type=question_type,
                sub_questions=[],
                needs_decomposition=False,
            )

        try:
            prompt = DECOMPOSITION_PROMPT.format(question=question)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a question analysis expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)

            # Parse question type
            q_type_str = result.get("question_type", "simple")
            try:
                q_type = QuestionType(q_type_str)
            except ValueError:
                q_type = QuestionType.SIMPLE

            # Parse sub-questions
            sub_qs = []
            for sq in result.get("sub_questions", []):
                sub_qs.append(SubQuestion(
                    question=sq.get("question", ""),
                    purpose=sq.get("purpose", ""),
                ))

            return DecompositionResult(
                original_question=question,
                question_type=q_type,
                sub_questions=sub_qs,
                aggregation_instruction=result.get("aggregation_instruction"),
                needs_decomposition=result.get("needs_decomposition", len(sub_qs) > 0),
            )

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}, falling back to fast method")
            return self.decompose_fast(question)

    async def reason_about_preferences(
        self,
        question: str,
        memories: list[dict],
        content_key: str = "content",
    ) -> dict:
        """
        Use LLM to reason about user preferences from retrieved memories.

        This helps answer preference-type questions like "What should I...?"

        Args:
            question: The preference question
            memories: Retrieved memories with user history
            content_key: Key containing memory content

        Returns:
            Dict with inferred preferences and reasoning
        """
        if not memories:
            return {
                "relevant_preferences": [],
                "constraints": [],
                "inferred_preference": "Unable to determine preferences from available memories.",
                "confidence": 0.0,
            }

        try:
            # Format memories
            memories_text = "\n".join([
                f"[{i+1}] {mem.get(content_key, '')[:400]}"
                for i, mem in enumerate(memories[:10])
            ])

            prompt = PREFERENCE_PROMPT.format(
                question=question,
                memories=memories_text,
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a preference analysis expert. Infer user preferences from their history. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)

        except Exception as e:
            logger.error(f"Preference reasoning failed: {e}")
            return {
                "relevant_preferences": [],
                "constraints": [],
                "inferred_preference": f"Error analyzing preferences: {e}",
                "confidence": 0.0,
            }

    async def aggregate_sub_answers(
        self,
        original_question: str,
        sub_answers: list[tuple[str, str]],  # List of (sub-question, answer) tuples
        aggregation_instruction: str,
    ) -> str:
        """
        Aggregate answers to sub-questions into a final answer.

        Args:
            original_question: The original question
            sub_answers: List of (sub-question, answer) tuples
            aggregation_instruction: How to combine the answers

        Returns:
            Aggregated final answer
        """
        if not sub_answers:
            return "Unable to find relevant information."

        # Format sub-answers
        sub_answers_text = "\n".join([
            f"Q: {sq}\nA: {ans}"
            for sq, ans in sub_answers
        ])

        prompt = f"""Original question: {original_question}

Sub-questions and answers:
{sub_answers_text}

Aggregation instruction: {aggregation_instruction}

Based on the sub-answers and aggregation instruction, provide the final answer to the original question.
Be concise and direct.

Final answer:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Answer aggregation failed: {e}")
            # Fallback: simple concatenation
            return f"Based on the available information: {'; '.join([ans for _, ans in sub_answers])}"


# Singleton instance
_decomposer: Optional[QuestionDecomposer] = None


def get_question_decomposer(model: str = "gpt-4o") -> QuestionDecomposer:
    """Get or create singleton question decomposer."""
    global _decomposer
    if _decomposer is None:
        _decomposer = QuestionDecomposer(model=model)
    return _decomposer
