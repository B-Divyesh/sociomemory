"""
Entity Extraction Service for SocioMemory

Extracts named entities from memory content using:
1. Regex patterns for common entities (dates, emails, names)
2. LLM-based extraction for complex entities
3. Entity embedding for semantic search

Entity Types:
- PERSON: People names
- ORG: Organizations, companies
- LOCATION: Places, addresses
- DATE: Dates, times, durations
- EVENT: Events, activities
- PREFERENCE: User preferences
- FACT: Factual statements
"""
import re
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from uuid import uuid4

from openai import AsyncAzureOpenAI, AsyncOpenAI

from sociomemory.config import get_settings


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    name: str
    entity_type: str
    confidence: float = 1.0
    context: str = ""
    start_pos: int = 0
    end_pos: int = 0
    extra_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "context": self.context,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "extra_data": self.extra_data,
        }


class EntityExtractor:
    """Extract named entities from text content."""

    # Regex patterns for common entities
    PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "URL": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
        "DATE_ISO": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        "DATE_MDY": r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b',
        "DATE_RELATIVE": r'\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b',
        "TIME": r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b',
        "MONEY": r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|EUR|GBP)',
        "PERCENTAGE": r'\b\d+(?:\.\d+)?%',
    }

    # Common name patterns (capitalized words that might be names)
    # Match both single names (Caroline, John) and full names (John Smith)
    NAME_PATTERN = r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b'

    # Common first names for better detection
    COMMON_NAMES = {
        'james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'joseph',
        'thomas', 'charles', 'christopher', 'daniel', 'matthew', 'anthony', 'mark',
        'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan',
        'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty', 'margaret', 'sandra',
        'alice', 'bob', 'charlie', 'dave', 'eve', 'frank', 'grace', 'helen', 'ivan',
        'julia', 'kate', 'laura', 'mike', 'nancy', 'oscar', 'peter', 'rachel', 'sam',
        'tom', 'uma', 'victor', 'wendy', 'xavier', 'yvonne', 'zach',
        'caroline', 'melanie', 'alex', 'emma', 'olivia', 'sophia', 'isabella',
    }

    # LLM extraction prompt
    EXTRACTION_PROMPT = """Extract all named entities from the following text. Return a JSON array of entities.

Entity types to extract:
- PERSON: Names of people
- ORG: Organizations, companies, teams
- LOCATION: Places, cities, countries, addresses
- DATE: Specific dates, times, durations
- EVENT: Events, meetings, activities
- PREFERENCE: User preferences or opinions
- FACT: Key factual statements about a person

For each entity, provide:
- name: The entity text
- type: One of the types above
- confidence: 0.0-1.0 confidence score

Text: {text}

Return ONLY valid JSON array, no explanation. Example:
[{{"name": "John Smith", "type": "PERSON", "confidence": 0.95}}]

JSON:"""

    def __init__(self, use_llm: bool = True):
        """Initialize entity extractor.

        Args:
            use_llm: Whether to use LLM for complex extraction (default True)
        """
        self.settings = get_settings()
        self.use_llm = use_llm
        self.client = None

        if use_llm:
            if self.settings.use_azure_openai:
                self.client = AsyncAzureOpenAI(
                    api_key=self.settings.azure_openai_key,
                    api_version="2024-02-01",
                    azure_endpoint=self.settings.azure_openai_endpoint,
                )
                self.model = self.settings.azure_openai_chat_deployment
            else:
                self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                self.model = "gpt-4o"  # Use gpt-4o (gpt-4o-mini may not be deployed)

    def extract_with_regex(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    name=match.group(),
                    entity_type=entity_type,
                    confidence=1.0,
                    context=text[max(0, match.start()-20):match.end()+20],
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        # Extract potential names (capitalized word sequences)
        false_positives = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'hey', 'hello', 'hi', 'good', 'great', 'nice', 'well', 'thanks',
        }

        for match in re.finditer(self.NAME_PATTERN, text):
            name = match.group()
            name_lower = name.lower()

            # Filter out common false positives
            if name_lower in false_positives:
                continue

            # Check if it's a known common name (higher confidence)
            first_word = name.split()[0].lower() if ' ' in name else name_lower
            is_common_name = first_word in self.COMMON_NAMES

            # Assign confidence based on whether it's a known name
            confidence = 0.9 if is_common_name else 0.6

            entities.append(ExtractedEntity(
                name=name,
                entity_type="PERSON",
                confidence=confidence,
                context=text[max(0, match.start()-20):match.end()+20],
                start_pos=match.start(),
                end_pos=match.end(),
            ))

        return entities

    async def extract_with_llm(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using LLM."""
        if not self.client:
            return []

        try:
            prompt = self.EXTRACTION_PROMPT.format(text=text[:2000])  # Limit text length

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            entities_data = json.loads(content)

            entities = []
            for ent in entities_data:
                if isinstance(ent, dict) and "name" in ent and "type" in ent:
                    entities.append(ExtractedEntity(
                        name=ent["name"],
                        entity_type=ent["type"],
                        confidence=ent.get("confidence", 0.9),
                        extra_data={"source": "llm"},
                    ))

            return entities

        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            return []
        except Exception as e:
            # Log error but don't fail
            print(f"LLM extraction error: {e}")
            return []

    async def extract_entities(
        self,
        text: str,
        use_regex: bool = True,
        use_llm: bool = None,
    ) -> list[ExtractedEntity]:
        """Extract all entities from text.

        Args:
            text: The text to extract entities from
            use_regex: Whether to use regex patterns (default True)
            use_llm: Whether to use LLM extraction (default: self.use_llm)

        Returns:
            List of extracted entities
        """
        entities = []

        if use_regex:
            regex_entities = self.extract_with_regex(text)
            entities.extend(regex_entities)

        if use_llm is None:
            use_llm = self.use_llm

        if use_llm and self.client:
            llm_entities = await self.extract_with_llm(text)
            entities.extend(llm_entities)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return entities

    def _deduplicate_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())

    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        """Extract important keywords from text for search enhancement.

        Simple TF-based keyword extraction without external dependencies.
        """
        # Tokenize and normalize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
            'same', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
            'there', 'then', 'once', 'not', 'its', 'their', 'your', 'his', 'over',
            'her', 'our', 'their', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
        }

        words = [w for w in words if w not in stop_words]

        # Count frequency
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency and return top-k
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_k]]


class RelationshipDetector:
    """Detect relationships between memories."""

    RELATION_TYPES = [
        "updates",      # New info updates old info
        "contradicts",  # New info contradicts old info
        "extends",      # New info adds to old info
        "relates_to",   # General semantic relationship
        "temporal",     # Temporal relationship (before/after)
    ]

    DETECTION_PROMPT = """Analyze the relationship between these two memory statements.

Memory 1 (older): {memory1}
Memory 2 (newer): {memory2}

Determine the relationship type:
- "updates": Memory 2 provides updated information about the same topic
- "contradicts": Memory 2 contradicts Memory 1
- "extends": Memory 2 adds new information to Memory 1
- "relates_to": Memories are about related topics
- "none": No meaningful relationship

Also provide a confidence score (0.0-1.0).

Return ONLY valid JSON:
{{"relation": "type", "confidence": 0.9, "explanation": "brief reason"}}

JSON:"""

    def __init__(self):
        self.settings = get_settings()
        if self.settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_key,
                api_version="2024-02-01",
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            self.model = self.settings.azure_openai_deployment
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o"  # Use gpt-4o (gpt-4o-mini may not be deployed)

    async def detect_relationship(
        self,
        memory1_content: str,
        memory2_content: str,
        memory1_timestamp: datetime = None,
        memory2_timestamp: datetime = None,
    ) -> dict:
        """Detect relationship between two memories.

        Args:
            memory1_content: Content of first (older) memory
            memory2_content: Content of second (newer) memory
            memory1_timestamp: Timestamp of first memory
            memory2_timestamp: Timestamp of second memory

        Returns:
            Dict with relation type, confidence, and explanation
        """
        # Determine which is older
        if memory1_timestamp and memory2_timestamp:
            if memory2_timestamp < memory1_timestamp:
                memory1_content, memory2_content = memory2_content, memory1_content

        try:
            prompt = self.DETECTION_PROMPT.format(
                memory1=memory1_content[:500],
                memory2=memory2_content[:500],
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a relationship analysis assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            return {
                "relation_type": result.get("relation", "none"),
                "confidence": result.get("confidence", 0.5),
                "explanation": result.get("explanation", ""),
            }

        except Exception as e:
            print(f"Relationship detection error: {e}")
            return {
                "relation_type": "none",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
            }

    def detect_temporal_relationship(
        self,
        memory1_timestamp: datetime,
        memory2_timestamp: datetime,
    ) -> dict:
        """Detect temporal relationship between memories.

        Returns:
            Dict with temporal relation (before, after, same_day, same_week, etc.)
        """
        if not memory1_timestamp or not memory2_timestamp:
            return {"temporal_relation": "unknown"}

        diff = memory2_timestamp - memory1_timestamp
        diff_days = abs(diff.days)

        if diff.total_seconds() == 0:
            relation = "same_time"
        elif diff_days == 0:
            relation = "same_day"
        elif diff_days <= 7:
            relation = "same_week"
        elif diff_days <= 30:
            relation = "same_month"
        elif diff_days <= 365:
            relation = "same_year"
        else:
            relation = "different_years"

        return {
            "temporal_relation": relation,
            "days_apart": diff_days,
            "is_before": diff.total_seconds() < 0,
        }
