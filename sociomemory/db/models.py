"""
SQLAlchemy ORM models for SocioMemory
"""
import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass


class MemoryORM(Base):
    """Memory table - stores all types of memories with FSRS fields"""
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Content
    memory_type = Column(String(50), nullable=False, default="fact")
    content = Column(Text, nullable=False)
    embedding = Column(Vector(3072))  # text-embedding-3-large dimensions

    # Temporal fields (Graphiti/Zep inspired bi-temporal model)
    valid_from = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    valid_until = Column(DateTime(timezone=True), nullable=True)  # NULL = still valid
    is_latest = Column(Boolean, default=True, nullable=False)
    # Bi-temporal: event_time = when the event occurred (extracted from content)
    # Different from created_at which is when the memory was stored
    event_time = Column(DateTime(timezone=True), nullable=True)

    # FSRS fields for retrieval optimization
    stability = Column(Float, default=4.0, nullable=False)  # 4.0 days initial stability
    difficulty = Column(Float, default=0.3, nullable=False)
    retrievability = Column(Float, default=1.0, nullable=False)
    last_accessed = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=0, nullable=False)

    # Source info
    source_platform = Column(String(50), nullable=True)  # telegram, twitter, reddit, etc.
    source_id = Column(String(255), nullable=True)  # Original message/post ID
    confidence = Column(Float, default=1.0, nullable=False)
    extra_data = Column(JSON, nullable=True)  # Additional metadata (renamed from 'metadata' - reserved in SQLAlchemy)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    source_relations = relationship(
        "MemoryRelationORM",
        foreign_keys="MemoryRelationORM.source_memory_id",
        back_populates="source_memory",
        cascade="all, delete-orphan"
    )
    target_relations = relationship(
        "MemoryRelationORM",
        foreign_keys="MemoryRelationORM.target_memory_id",
        back_populates="target_memory",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_memories_user_valid", "user_id", "is_latest", "valid_until"),
        Index("idx_memories_user_type", "user_id", "memory_type"),
        Index("idx_memories_user_platform", "user_id", "source_platform"),
    )


class MemoryRelationORM(Base):
    """Memory relationships - updates, extends, derives, contradicts"""
    __tablename__ = "memory_relations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    target_memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(50), nullable=False)  # updates, extends, derives, contradicts
    confidence = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    source_memory = relationship("MemoryORM", foreign_keys=[source_memory_id], back_populates="source_relations")
    target_memory = relationship("MemoryORM", foreign_keys=[target_memory_id], back_populates="target_relations")

    __table_args__ = (
        Index("idx_relations_source", "source_memory_id"),
        Index("idx_relations_target", "target_memory_id"),
    )


class EntityORM(Base):
    """Extracted entities from memories - serves as graph nodes"""
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False)  # person, organization, location, etc.
    embedding = Column(Vector(3072))
    extra_data = Column(JSON, nullable=True)  # Additional metadata (renamed from 'metadata' - reserved in SQLAlchemy)

    # Graph-specific fields for PageRank
    node_properties = Column(JSON, nullable=True, default={})
    mention_count = Column(Integer, default=0, nullable=False)
    last_mentioned_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "name", "entity_type", name="uq_entity_user_name_type"),
        Index("idx_entities_user_type", "user_id", "entity_type"),
        Index("idx_entities_mention_count", "user_id", "mention_count"),
    )


class EntityMentionORM(Base):
    """Links entities to memories where they are mentioned"""
    __tablename__ = "entity_mentions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    mention_context = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_mentions_entity", "entity_id"),
        Index("idx_mentions_memory", "memory_id"),
    )


class FactORM(Base):
    """Atomic facts extracted from memories for improved retrieval"""
    __tablename__ = "facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Fact content
    fact_text = Column(Text, nullable=False)
    fact_type = Column(String(50), default="general")  # general, preference, event, relationship

    # Structured extraction (optional, for advanced queries)
    subject_entity = Column(String(255), nullable=True)  # e.g., "User", "John"
    predicate = Column(String(100), nullable=True)       # e.g., "visited", "prefers"
    object_entity = Column(String(255), nullable=True)   # e.g., "MoMA", "ocean-view hotels"

    # Temporal information
    event_time = Column(DateTime(timezone=True), nullable=True)

    # Source tracking
    source_memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=True)

    # Embedding for semantic search
    embedding = Column(Vector(3072))

    # Confidence and metadata
    confidence = Column(Float, default=1.0, nullable=False)
    extra_data = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_facts_user", "user_id"),
        Index("idx_facts_user_type", "user_id", "fact_type"),
        Index("idx_facts_source_memory", "source_memory_id"),
        Index("idx_facts_event_time", "user_id", "event_time"),
        Index("idx_facts_subject", "user_id", "subject_entity"),
    )


class GraphEdgeORM(Base):
    """Knowledge graph edges representing relationships between entities"""
    __tablename__ = "graph_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Edge endpoints
    source_entity_name = Column(String(255), nullable=False)
    source_entity_type = Column(String(50), nullable=False)
    target_entity_name = Column(String(255), nullable=False)
    target_entity_type = Column(String(50), nullable=False)

    # Relationship info
    relation_type = Column(String(100), nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)

    # Source memory
    source_memory_id = Column(UUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True)

    # Edge properties
    edge_properties = Column(JSON, nullable=True, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "user_id", "source_entity_name", "source_entity_type",
            "target_entity_name", "target_entity_type", "relation_type",
            name="uq_graph_edge"
        ),
        Index("idx_graph_edges_user", "user_id"),
        Index("idx_graph_edges_source", "user_id", "source_entity_name", "source_entity_type"),
        Index("idx_graph_edges_target", "user_id", "target_entity_name", "target_entity_type"),
        Index("idx_graph_edges_relation", "user_id", "relation_type"),
    )
