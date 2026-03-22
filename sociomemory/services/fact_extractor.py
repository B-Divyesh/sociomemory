"""
Fact Extraction Service for SocioMemory

Extracts atomic facts from memories for improved retrieval accuracy.
Based on research showing that atomic fact storage improves retrieval by 15-20%.

Facts are single, verifiable statements like:
- "User visited MoMA on January 8, 2023"
- "User prefers ocean-view hotels"
- "John works at Google"

Each fact has:
- fact_text: The atomic statement
- fact_type: general, preference, event, relationship
- subject_entity: The subject (e.g., "User", "John")
- predicate: The action/relation (e.g., "visited", "prefers")
- object_entity: The object (e.g., "MoMA", "ocean-view hotels")
- event_time: When the fact's event occurred (if applicable)
"""
import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from openai import AsyncAzureOpenAI, AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sociomemory.config import get_settings
from sociomemory.db.models import FactORM, MemoryORM
from sociomemory.models.memory import FactCreate, FactResponse, FactType, FactSearchRequest, FactSearchResponse
from sociomemory.services.embedding_service import EmbeddingService, get_embedding_service
from sociomemory.services.temporal_parser import TemporalParser

logger = logging.getLogger(__name__)


FACT_EXTRACTION_PROMPT = """Extract atomic facts from the following text. Each fact should be a single, verifiable statement.

Text:
{text}

Extract facts in the following categories:
1. GENERAL - General factual statements
2. PREFERENCE - User preferences, likes, dislikes
3. EVENT - Time-bound occurrences with specific dates
4. RELATIONSHIP - Relationships between entities

For each fact, provide:
- fact_text: The atomic statement (one sentence)
- fact_type: One of general, preference, event, relationship
- subject: The subject entity (who/what the fact is about)
- predicate: The action or relation
- object: The object entity (if applicable)
- event_time: Date/time if this is a time-bound event (ISO format or null)

Return ONLY valid JSON array. Example:
[
  {{
    "fact_text": "User visited MoMA on January 8, 2023",
    "fact_type": "event",
    "subject": "User",
    "predicate": "visited",
    "object": "MoMA",
    "event_time": "2023-01-08"
  }},
  {{
    "fact_text": "User prefers ocean-view hotels",
    "fact_type": "preference",
    "subject": "User",
    "predicate": "prefers",
    "object": "ocean-view hotels",
    "event_time": null
  }}
]

JSON array:"""


class FactExtractor:
    """
    Extracts and manages atomic facts from memories.

    Facts enable more precise retrieval for:
    - Temporal queries (facts with event_time)
    - Preference queries (facts with fact_type=preference)
    - Entity-centric queries (facts with subject/object)
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or get_embedding_service()

        # Initialize OpenAI client
        self.settings = get_settings()
        if self.settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_key,
                api_version="2024-02-01",
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            self.model = self.settings.azure_openai_chat_deployment or "gpt-4o"  # Use gpt-4o (gpt-4o-mini may not be deployed)
        else:
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = "gpt-4o"  # Use gpt-4o (gpt-4o-mini may not be deployed)

        self.temporal_parser = TemporalParser()

    async def extract_facts_from_text(self, text: str) -> list[dict]:
        """
        Extract atomic facts from text using LLM.

        Args:
            text: The text to extract facts from

        Returns:
            List of fact dictionaries
        """
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(text=text[:3000])  # Limit text length

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact extraction assistant. Extract atomic, verifiable facts. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            facts = json.loads(content)

            if isinstance(facts, list):
                return facts
            return []

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse fact extraction response: {e}")
            return []
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    async def extract_and_store_facts(
        self,
        memory: MemoryORM,
    ) -> list[FactResponse]:
        """
        Extract facts from a memory and store them in the database.

        Args:
            memory: The memory ORM object to extract facts from

        Returns:
            List of created FactResponse objects
        """
        # Extract facts using LLM
        extracted_facts = await self.extract_facts_from_text(memory.content)

        if not extracted_facts:
            logger.debug(f"No facts extracted from memory {memory.id}")
            return []

        created_facts = []

        for fact_data in extracted_facts:
            try:
                # Parse event_time if present
                event_time = None
                if fact_data.get("event_time"):
                    try:
                        event_time = datetime.fromisoformat(fact_data["event_time"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        # Try temporal parser as fallback
                        temporal_info = self.temporal_parser.parse(fact_data["event_time"])
                        if temporal_info.reference_date:
                            event_time = temporal_info.reference_date

                # Map fact_type string to enum
                fact_type_str = fact_data.get("fact_type", "general").lower()
                try:
                    fact_type = FactType(fact_type_str)
                except ValueError:
                    fact_type = FactType.GENERAL

                # Generate embedding for the fact
                fact_text = fact_data.get("fact_text", "")
                if not fact_text:
                    continue

                embedding = await self.embedding_service.get_embedding(fact_text)

                # Create fact record
                fact = FactORM(
                    user_id=memory.user_id,
                    fact_text=fact_text,
                    fact_type=fact_type.value,
                    subject_entity=fact_data.get("subject"),
                    predicate=fact_data.get("predicate"),
                    object_entity=fact_data.get("object"),
                    event_time=event_time,
                    source_memory_id=memory.id,
                    embedding=embedding,
                    confidence=1.0,
                )

                self.db.add(fact)
                await self.db.flush()

                created_facts.append(self._to_response(fact))

            except Exception as e:
                logger.warning(f"Failed to create fact: {e}")
                continue

        if created_facts:
            await self.db.commit()
            logger.info(f"Extracted {len(created_facts)} facts from memory {memory.id}")

        return created_facts

    async def create_fact(self, request: FactCreate) -> FactResponse:
        """
        Create a fact directly (without extraction from memory).

        Args:
            request: FactCreate request

        Returns:
            Created FactResponse
        """
        # Generate embedding
        embedding = await self.embedding_service.get_embedding(request.fact_text)

        fact = FactORM(
            user_id=request.user_id,
            fact_text=request.fact_text,
            fact_type=request.fact_type.value,
            subject_entity=request.subject_entity,
            predicate=request.predicate,
            object_entity=request.object_entity,
            event_time=request.event_time,
            source_memory_id=request.source_memory_id,
            embedding=embedding,
            confidence=request.confidence,
        )

        self.db.add(fact)
        await self.db.commit()
        await self.db.refresh(fact)

        return self._to_response(fact)

    async def search_facts(
        self,
        request: FactSearchRequest,
    ) -> FactSearchResponse:
        """
        Search facts using vector similarity.

        Args:
            request: FactSearchRequest with query and filters

        Returns:
            FactSearchResponse with matching facts
        """
        from sqlalchemy import text

        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(request.query)
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # Build query with filters
        filters = ["user_id = CAST(:user_id AS uuid)", "embedding IS NOT NULL"]
        params = {"user_id": str(request.user_id), "embedding": embedding_str, "limit_val": request.limit}

        if request.fact_types:
            types_list = ",".join(f"'{t.value}'" for t in request.fact_types)
            filters.append(f"fact_type IN ({types_list})")

        if request.subject_filter:
            filters.append("LOWER(subject_entity) LIKE LOWER(:subject_filter)")
            params["subject_filter"] = f"%{request.subject_filter}%"

        if request.event_time_start:
            filters.append("event_time >= :event_time_start")
            params["event_time_start"] = request.event_time_start

        if request.event_time_end:
            filters.append("event_time <= :event_time_end")
            params["event_time_end"] = request.event_time_end

        where_clause = " AND ".join(filters)

        query = text(f"""
            SELECT
                id,
                user_id,
                fact_text,
                fact_type,
                subject_entity,
                predicate,
                object_entity,
                event_time,
                source_memory_id,
                confidence,
                created_at,
                (1 - (embedding <=> CAST(:embedding AS vector(3072)))) AS similarity
            FROM facts
            WHERE {where_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector(3072))
            LIMIT :limit_val
        """)

        result = await self.db.execute(query, params)
        rows = result.fetchall()

        results = []
        for row in rows:
            results.append(FactResponse(
                id=row.id,
                user_id=row.user_id,
                fact_text=row.fact_text,
                fact_type=FactType(row.fact_type),
                subject_entity=row.subject_entity,
                predicate=row.predicate,
                object_entity=row.object_entity,
                event_time=row.event_time,
                source_memory_id=row.source_memory_id,
                confidence=row.confidence,
                created_at=row.created_at,
            ))

        return FactSearchResponse(
            results=results,
            total=len(results),
        )

    async def get_facts_for_memory(self, memory_id: UUID) -> list[FactResponse]:
        """
        Get all facts extracted from a specific memory.

        Args:
            memory_id: The memory ID

        Returns:
            List of FactResponse objects
        """
        result = await self.db.execute(
            select(FactORM).where(FactORM.source_memory_id == memory_id)
        )
        facts = result.scalars().all()

        return [self._to_response(f) for f in facts]

    def _to_response(self, fact: FactORM) -> FactResponse:
        """Convert ORM model to response model."""
        return FactResponse(
            id=fact.id,
            user_id=fact.user_id,
            fact_text=fact.fact_text,
            fact_type=FactType(fact.fact_type),
            subject_entity=fact.subject_entity,
            predicate=fact.predicate,
            object_entity=fact.object_entity,
            event_time=fact.event_time,
            source_memory_id=fact.source_memory_id,
            confidence=fact.confidence,
            created_at=fact.created_at,
        )


async def get_fact_extractor(db: AsyncSession) -> FactExtractor:
    """Get fact extractor instance."""
    return FactExtractor(db=db)
