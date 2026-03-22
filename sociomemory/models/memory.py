"""
Pydantic models for Memory operations
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories"""
    EPISODE = "episode"      # Raw interaction data
    FACT = "fact"            # Extracted knowledge
    ENTITY = "entity"        # Named entity
    PREFERENCE = "preference"  # User preference
    BEHAVIOR = "behavior"    # User behavior pattern
    EVENT = "event"          # Time-bound occurrence
    CONTEXT = "context"      # Contextual information
    RELATIONSHIP = "relationship"  # Entity relationship


class RelationType(str, Enum):
    """Types of memory relationships"""
    UPDATES = "updates"      # New info replaces old
    EXTENDS = "extends"      # New info adds to old
    DERIVES = "derives"      # Inferred from patterns
    CONTRADICTS = "contradicts"  # Flagged for resolution


class MemoryBase(BaseModel):
    """Base memory fields"""
    # Support up to 500k chars - embedding service handles chunking with overlap
    content: str = Field(..., min_length=1, max_length=500000)
    memory_type: MemoryType = Field(default=MemoryType.FACT)
    source_platform: Optional[str] = Field(None, max_length=50)
    source_id: Optional[str] = Field(None, max_length=255)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[dict[str, Any]] = None


class MemoryCreate(MemoryBase):
    """Request model for creating a memory"""
    user_id: UUID
    extract_entities: bool = Field(default=False, description="Auto-extract entities from content")
    check_relations: bool = Field(default=False, description="Check for updates/contradictions")


class MemoryUpdate(BaseModel):
    """Request model for updating a memory"""
    content: Optional[str] = Field(None, min_length=1, max_length=500000)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[dict[str, Any]] = None
    invalidate: bool = Field(default=False, description="Mark memory as no longer valid")


class Memory(MemoryBase):
    """Full memory model with all fields"""
    id: UUID
    user_id: UUID
    embedding: Optional[list[float]] = None

    # Temporal fields (bi-temporal model)
    valid_from: datetime
    valid_until: Optional[datetime] = None
    is_latest: bool = True
    event_time: Optional[datetime] = None  # When the event occurred (extracted from content)

    # FSRS fields
    stability: float = 4.0  # 4.0 days initial stability (matches DB schema)
    difficulty: float = 0.3
    retrievability: float = 1.0
    last_accessed: datetime
    access_count: int = 0

    # Timestamps
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    id: UUID
    content: str
    memory_type: MemoryType
    source_platform: Optional[str] = None
    source_id: Optional[str] = None  # Session/source identifier for grouping
    confidence: float

    # Temporal (bi-temporal model)
    valid_from: datetime
    valid_until: Optional[datetime] = None
    is_latest: bool
    event_time: Optional[datetime] = None  # When the event occurred

    # FSRS state
    stability: float
    difficulty: float
    retrievability: float
    access_count: int

    # Search relevance (only in search results)
    similarity: Optional[float] = None
    priority_score: Optional[float] = None

    created_at: datetime

    model_config = {"from_attributes": True}


class MemorySearchRequest(BaseModel):
    """Request model for searching memories"""
    user_id: UUID
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)  # 0.0 to return all matches (typical similarity: 0.1-0.4)
    platforms: Optional[list[str]] = None
    memory_types: Optional[list[MemoryType]] = None
    source_id: Optional[str] = Field(None, description="Filter by source ID (e.g., chat_id for conversation-specific search)")
    only_latest: bool = Field(default=True, description="Only return latest versions")
    only_valid: bool = Field(default=True, description="Only return currently valid memories")


class MemorySearchResponse(BaseModel):
    """Response model for memory search"""
    results: list[MemoryResponse]
    total: int
    query_embedding_tokens: int = 0


class MemoryRelation(BaseModel):
    """Model for memory relationships"""
    id: UUID
    source_memory_id: UUID
    target_memory_id: UUID
    relation_type: RelationType
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}


class AccessRecord(BaseModel):
    """Request model for recording memory access"""
    was_useful: bool = Field(default=True, description="Whether the memory was useful")
    context: Optional[str] = Field(None, description="Context of access for logging")


class MemoryStats(BaseModel):
    """Memory statistics for a user"""
    user_id: UUID
    total_memories: int
    by_type: dict[str, int]
    by_platform: dict[str, int]
    avg_stability: float
    avg_retrievability: float
    total_accesses: int
    memories_accessed_today: int
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]


# =============================================================================
# FACT MODELS (for atomic fact extraction)
# =============================================================================

class FactType(str, Enum):
    """Types of extracted facts"""
    GENERAL = "general"          # General factual statement
    PREFERENCE = "preference"    # User preference
    EVENT = "event"              # Time-bound event
    RELATIONSHIP = "relationship"  # Entity relationship


class FactCreate(BaseModel):
    """Request model for creating a fact"""
    user_id: UUID
    fact_text: str = Field(..., min_length=1, max_length=2000)
    fact_type: FactType = Field(default=FactType.GENERAL)
    subject_entity: Optional[str] = Field(None, max_length=255)
    predicate: Optional[str] = Field(None, max_length=100)
    object_entity: Optional[str] = Field(None, max_length=255)
    event_time: Optional[datetime] = None
    source_memory_id: Optional[UUID] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FactResponse(BaseModel):
    """Response model for fact operations"""
    id: UUID
    user_id: UUID
    fact_text: str
    fact_type: FactType
    subject_entity: Optional[str] = None
    predicate: Optional[str] = None
    object_entity: Optional[str] = None
    event_time: Optional[datetime] = None
    source_memory_id: Optional[UUID] = None
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}


class FactSearchRequest(BaseModel):
    """Request model for searching facts"""
    user_id: UUID
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    fact_types: Optional[list[FactType]] = None
    subject_filter: Optional[str] = None
    event_time_start: Optional[datetime] = None
    event_time_end: Optional[datetime] = None


class FactSearchResponse(BaseModel):
    """Response model for fact search"""
    results: list[FactResponse]
    total: int
