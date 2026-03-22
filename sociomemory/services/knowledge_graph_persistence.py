"""
Knowledge Graph Persistence Service for SocioMemory

Handles loading and saving the knowledge graph to/from the database.
Enables persistent graph-based retrieval across sessions.

Tables used:
- entities: Stores graph nodes (with node_properties, mention_count)
- graph_edges: Stores graph edges with relationship info
- entity_mentions: Links entities to memories

Performance optimizations:
- Batch operations to reduce commits (see batch_add_entities, batch_add_relationships)
- Retry with exponential backoff for connection pool exhaustion
- Single commit at end of batch operations

Based on: https://docs.sqlalchemy.org/en/20/core/pooling.html
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

from sociomemory.db.models import EntityORM, EntityMentionORM, GraphEdgeORM
from sociomemory.services.knowledge_graph import KnowledgeGraph, GraphNode, GraphEdge
from sociomemory.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

# Retry configuration for connection pool exhaustion
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 10.0  # seconds


async def with_retry(coro_func, *args, **kwargs):
    """
    Execute an async function with exponential backoff retry.

    Handles SQLAlchemy connection pool exhaustion errors gracefully.
    Based on: https://sqlalche.me/e/20/3o7r
    """
    last_error = None
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            return await coro_func(*args, **kwargs)
        except (OperationalError, SATimeoutError) as e:
            error_msg = str(e)
            if "QueuePool limit" in error_msg or "connection timed out" in error_msg:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Connection pool exhausted (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                raise
        except Exception:
            raise

    logger.error(f"Failed after {MAX_RETRIES} retries: {last_error}")
    raise last_error


class PersistentKnowledgeGraph:
    """
    A knowledge graph with database persistence.

    This extends the in-memory KnowledgeGraph with:
    1. Loading graph from database on initialization
    2. Saving new entities and edges to database
    3. Syncing changes between memory and database
    """

    def __init__(
        self,
        db: AsyncSession,
        user_id: UUID,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db = db
        self.user_id = user_id
        self.embedding_service = embedding_service or get_embedding_service()
        self.graph = KnowledgeGraph()
        self._loaded = False

    async def load_from_db(self) -> None:
        """
        Load the knowledge graph from the database.

        Loads all entities and edges for the user.
        """
        if self._loaded:
            return

        try:
            # Load entities
            result = await self.db.execute(
                select(EntityORM).where(EntityORM.user_id == self.user_id)
            )
            entities = result.scalars().all()

            for entity in entities:
                node = GraphNode(
                    id=self.graph._make_entity_key(entity.name, entity.entity_type),
                    name=entity.name,
                    entity_type=entity.entity_type,
                    memory_ids=set(),  # Will be populated from entity_mentions
                    properties=entity.node_properties or {},
                    created_at=entity.created_at,
                    updated_at=entity.updated_at,
                )
                self.graph.nodes[node.id] = node
                self.graph.type_to_nodes[entity.entity_type].append(node.id)
                self.graph.name_to_node[entity.name.lower()] = node.id

            # Load entity mentions to populate memory_ids
            entity_ids = [e.id for e in entities]
            if entity_ids:
                mentions_result = await self.db.execute(
                    select(EntityMentionORM).where(EntityMentionORM.entity_id.in_(entity_ids))
                )
                mentions = mentions_result.scalars().all()

                # Build entity_id to node_id mapping
                entity_id_to_node = {}
                for entity in entities:
                    node_id = self.graph._make_entity_key(entity.name, entity.entity_type)
                    entity_id_to_node[entity.id] = node_id

                for mention in mentions:
                    node_id = entity_id_to_node.get(mention.entity_id)
                    if node_id and node_id in self.graph.nodes:
                        memory_id_str = str(mention.memory_id)
                        self.graph.nodes[node_id].memory_ids.add(memory_id_str)
                        self.graph.memory_to_nodes[memory_id_str].append(node_id)

            # Load edges
            edges_result = await self.db.execute(
                select(GraphEdgeORM).where(GraphEdgeORM.user_id == self.user_id)
            )
            edges = edges_result.scalars().all()

            for edge in edges:
                source_key = self.graph._make_entity_key(edge.source_entity_name, edge.source_entity_type)
                target_key = self.graph._make_entity_key(edge.target_entity_name, edge.target_entity_type)
                edge_id = f"{source_key}->{target_key}:{edge.relation_type}"

                graph_edge = GraphEdge(
                    id=edge_id,
                    source_id=source_key,
                    target_id=target_key,
                    relation_type=edge.relation_type,
                    memory_id=str(edge.source_memory_id) if edge.source_memory_id else None,
                    confidence=edge.confidence,
                    properties=edge.edge_properties or {},
                    created_at=edge.created_at,
                )

                self.graph.edges[edge_id] = graph_edge
                self.graph.outgoing_edges[source_key].append(edge_id)
                self.graph.incoming_edges[target_key].append(edge_id)

            self._loaded = True
            logger.info(f"Loaded knowledge graph for user {self.user_id}: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            raise

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        memory_id: Optional[str] = None,
        properties: dict = None,
    ) -> GraphNode:
        """
        Add an entity to both the in-memory graph and database.

        Args:
            name: Entity name
            entity_type: Type of entity (PERSON, LOCATION, etc.)
            memory_id: ID of the memory where this entity was found
            properties: Additional properties

        Returns:
            The GraphNode for this entity
        """
        # Add to in-memory graph
        node = self.graph.add_entity(name, entity_type, memory_id, properties)

        # Persist to database
        try:
            # Check if entity exists
            result = await self.db.execute(
                select(EntityORM).where(
                    EntityORM.user_id == self.user_id,
                    EntityORM.name == name,
                    EntityORM.entity_type == entity_type,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing entity
                existing.mention_count = (existing.mention_count or 0) + 1
                existing.last_mentioned_at = datetime.now(timezone.utc)
                if properties:
                    existing.node_properties = {**(existing.node_properties or {}), **properties}
                existing.updated_at = datetime.now(timezone.utc)
                entity_id = existing.id
            else:
                # Create new entity
                embedding = await self.embedding_service.get_embedding(name)
                entity = EntityORM(
                    user_id=self.user_id,
                    name=name,
                    entity_type=entity_type,
                    embedding=embedding,
                    node_properties=properties or {},
                    mention_count=1,
                    last_mentioned_at=datetime.now(timezone.utc),
                )
                self.db.add(entity)
                await self.db.flush()
                entity_id = entity.id

            # Add entity mention if memory_id provided
            if memory_id:
                mention = EntityMentionORM(
                    entity_id=entity_id,
                    memory_id=UUID(memory_id),
                    mention_context=properties.get("context", "") if properties else "",
                )
                self.db.add(mention)

            await self.db.commit()

        except Exception as e:
            logger.warning(f"Failed to persist entity {name}: {e}")
            await self.db.rollback()

        return node

    async def add_relationship(
        self,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relation_type: str,
        memory_id: Optional[str] = None,
        confidence: float = 1.0,
        properties: dict = None,
    ) -> GraphEdge:
        """
        Add a relationship to both the in-memory graph and database.

        Args:
            source_name: Name of source entity
            source_type: Type of source entity
            target_name: Name of target entity
            target_type: Type of target entity
            relation_type: Type of relationship
            memory_id: ID of the memory that established this relationship
            confidence: Confidence score (0-1)
            properties: Additional properties

        Returns:
            The GraphEdge for this relationship
        """
        # First ensure entities exist
        await self.add_entity(source_name, source_type, memory_id, properties)
        await self.add_entity(target_name, target_type, memory_id, properties)

        # Add to in-memory graph
        edge = self.graph.add_relationship(
            source_name, source_type,
            target_name, target_type,
            relation_type, memory_id,
            confidence, properties,
        )

        # Persist to database
        try:
            # Check if edge exists (unique constraint)
            result = await self.db.execute(
                select(GraphEdgeORM).where(
                    GraphEdgeORM.user_id == self.user_id,
                    GraphEdgeORM.source_entity_name == source_name,
                    GraphEdgeORM.source_entity_type == source_type,
                    GraphEdgeORM.target_entity_name == target_name,
                    GraphEdgeORM.target_entity_type == target_type,
                    GraphEdgeORM.relation_type == relation_type,
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update confidence if higher
                if confidence > existing.confidence:
                    existing.confidence = confidence
                if properties:
                    existing.edge_properties = {**(existing.edge_properties or {}), **properties}
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new edge
                db_edge = GraphEdgeORM(
                    user_id=self.user_id,
                    source_entity_name=source_name,
                    source_entity_type=source_type,
                    target_entity_name=target_name,
                    target_entity_type=target_type,
                    relation_type=relation_type,
                    confidence=confidence,
                    source_memory_id=UUID(memory_id) if memory_id else None,
                    edge_properties=properties or {},
                )
                self.db.add(db_edge)

            await self.db.commit()

        except Exception as e:
            logger.warning(f"Failed to persist edge {source_name}->{target_name}: {e}")
            await self.db.rollback()

        return edge

    async def batch_add_entities_and_relationships(
        self,
        entities: list[dict],
        relationships: list[dict],
        memory_id: Optional[str] = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Batch add entities and relationships with a SINGLE database commit.

        This is the preferred method for high-performance entity extraction.
        Reduces O(N²) commits to O(1) by batching all operations.

        Args:
            entities: List of dicts with keys: name, entity_type, properties
            relationships: List of dicts with keys: source_name, source_type,
                          target_name, target_type, relation_type, confidence
            memory_id: Memory ID for all entities/relationships

        Returns:
            Tuple of (added_nodes, added_edges)
        """
        added_nodes = []
        added_edges = []
        entity_db_ids = {}  # Map (name, type) -> entity_id for mentions

        try:
            # Phase 1: Add all entities to in-memory graph and collect DB operations
            for entity_data in entities:
                name = entity_data["name"]
                entity_type = entity_data["entity_type"]
                properties = entity_data.get("properties", {})

                # Add to in-memory graph
                node = self.graph.add_entity(name, entity_type, memory_id, properties)
                added_nodes.append(node)

                # Check if entity exists in DB
                result = await self.db.execute(
                    select(EntityORM).where(
                        EntityORM.user_id == self.user_id,
                        EntityORM.name == name,
                        EntityORM.entity_type == entity_type,
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.mention_count = (existing.mention_count or 0) + 1
                    existing.last_mentioned_at = datetime.now(timezone.utc)
                    if properties:
                        existing.node_properties = {**(existing.node_properties or {}), **properties}
                    existing.updated_at = datetime.now(timezone.utc)
                    entity_db_ids[(name, entity_type)] = existing.id
                else:
                    # Create new entity (get embedding)
                    embedding = await self.embedding_service.get_embedding(name)
                    entity = EntityORM(
                        user_id=self.user_id,
                        name=name,
                        entity_type=entity_type,
                        embedding=embedding,
                        node_properties=properties or {},
                        mention_count=1,
                        last_mentioned_at=datetime.now(timezone.utc),
                    )
                    self.db.add(entity)
                    await self.db.flush()  # Get ID without committing
                    entity_db_ids[(name, entity_type)] = entity.id

                # Add entity mention if memory_id provided
                if memory_id:
                    entity_id = entity_db_ids.get((name, entity_type))
                    if entity_id:
                        mention = EntityMentionORM(
                            entity_id=entity_id,
                            memory_id=UUID(memory_id),
                            mention_context=properties.get("context", "")[:500] if properties else "",
                        )
                        self.db.add(mention)

            # Phase 2: Add all relationships to in-memory graph and DB
            for rel_data in relationships:
                source_name = rel_data["source_name"]
                source_type = rel_data["source_type"]
                target_name = rel_data["target_name"]
                target_type = rel_data["target_type"]
                relation_type = rel_data["relation_type"]
                confidence = rel_data.get("confidence", 1.0)
                properties = rel_data.get("properties")

                # Add to in-memory graph
                edge = self.graph.add_relationship(
                    source_name, source_type,
                    target_name, target_type,
                    relation_type, memory_id,
                    confidence, properties,
                )
                added_edges.append(edge)

                # Check if edge exists in DB
                result = await self.db.execute(
                    select(GraphEdgeORM).where(
                        GraphEdgeORM.user_id == self.user_id,
                        GraphEdgeORM.source_entity_name == source_name,
                        GraphEdgeORM.source_entity_type == source_type,
                        GraphEdgeORM.target_entity_name == target_name,
                        GraphEdgeORM.target_entity_type == target_type,
                        GraphEdgeORM.relation_type == relation_type,
                    )
                )
                existing_edge = result.scalar_one_or_none()

                if existing_edge:
                    if confidence > existing_edge.confidence:
                        existing_edge.confidence = confidence
                    if properties:
                        existing_edge.edge_properties = {**(existing_edge.edge_properties or {}), **properties}
                    existing_edge.updated_at = datetime.now(timezone.utc)
                else:
                    db_edge = GraphEdgeORM(
                        user_id=self.user_id,
                        source_entity_name=source_name,
                        source_entity_type=source_type,
                        target_entity_name=target_name,
                        target_entity_type=target_type,
                        relation_type=relation_type,
                        confidence=confidence,
                        source_memory_id=UUID(memory_id) if memory_id else None,
                        edge_properties=properties or {},
                    )
                    self.db.add(db_edge)

            # Phase 3: Single commit for ALL operations
            await self.db.commit()
            logger.debug(
                f"Batch persisted {len(added_nodes)} entities and {len(added_edges)} relationships"
            )

        except Exception as e:
            logger.warning(f"Batch persist failed: {e}")
            await self.db.rollback()
            # Still return in-memory nodes/edges even if DB failed

        return added_nodes, added_edges

    def personalized_pagerank(
        self,
        query_entities: list[str],
        damping_factor: float = 0.85,
        iterations: int = 20,
    ) -> dict[str, float]:
        """
        Compute Personalized PageRank scores.

        Delegates to the in-memory graph.
        """
        return self.graph.personalized_pagerank(
            query_entities, damping_factor, iterations
        )

    def get_memories_by_pagerank(
        self,
        query_entities: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get memories ranked by PageRank relevance.

        Delegates to the in-memory graph.
        """
        return self.graph.get_memories_by_pagerank(query_entities, top_k)

    def get_related_entities(
        self,
        entity_name: str,
        entity_type: str = None,
        max_hops: int = 1,
    ) -> list[dict]:
        """
        Get related entities.

        Delegates to the in-memory graph.
        """
        return self.graph.get_related_entities(entity_name, entity_type, max_hops=max_hops)

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return self.graph.get_stats()

    @property
    def nodes(self) -> dict:
        """Access in-memory nodes."""
        return self.graph.nodes

    @property
    def edges(self) -> dict:
        """Access in-memory edges."""
        return self.graph.edges


async def get_persistent_knowledge_graph(
    db: AsyncSession,
    user_id: UUID,
    load: bool = True,
) -> PersistentKnowledgeGraph:
    """
    Get a persistent knowledge graph instance for a user.

    Args:
        db: Database session
        user_id: User ID
        load: Whether to load from database immediately (default True)

    Returns:
        PersistentKnowledgeGraph instance
    """
    graph = PersistentKnowledgeGraph(db, user_id)
    if load:
        await graph.load_from_db()
    return graph
