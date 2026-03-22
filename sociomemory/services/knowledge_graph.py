"""
Knowledge Graph Service for SocioMemory

Provides entity-centric knowledge graph for:
1. Multi-hop reasoning across memories
2. Entity relationship traversal
3. Time-aware graph queries
4. Improved retrieval through graph context

The graph structure:
- Nodes: Entities (people, places, events, etc.)
- Edges: Relationships between entities (from same memory, semantic link, etc.)
- Properties: Timestamps, confidence, memory references
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Set
from uuid import UUID, uuid4


@dataclass
class GraphNode:
    """A node in the knowledge graph representing an entity."""
    id: str
    name: str
    entity_type: str
    memory_ids: Set[str] = field(default_factory=set)
    properties: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_memory_reference(self, memory_id: str):
        """Add a reference to a memory that mentions this entity."""
        self.memory_ids.add(memory_id)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "memory_ids": list(self.memory_ids),
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class GraphEdge:
    """An edge in the knowledge graph representing a relationship."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    memory_id: Optional[str] = None
    confidence: float = 1.0
    properties: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "memory_id": self.memory_id,
            "confidence": self.confidence,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }


class KnowledgeGraph:
    """In-memory knowledge graph for entity relationships."""

    def __init__(self):
        # Node storage: entity_key -> GraphNode
        self.nodes: dict[str, GraphNode] = {}

        # Edge storage: edge_id -> GraphEdge
        self.edges: dict[str, GraphEdge] = {}

        # Index: source_id -> list of edge_ids
        self.outgoing_edges: dict[str, list[str]] = defaultdict(list)

        # Index: target_id -> list of edge_ids
        self.incoming_edges: dict[str, list[str]] = defaultdict(list)

        # Index: memory_id -> list of node_ids
        self.memory_to_nodes: dict[str, list[str]] = defaultdict(list)

        # Index: entity_type -> list of node_ids
        self.type_to_nodes: dict[str, list[str]] = defaultdict(list)

        # Index: name (lowercase) -> node_id for fast lookup
        self.name_to_node: dict[str, str] = {}

    def _make_entity_key(self, name: str, entity_type: str) -> str:
        """Create a unique key for an entity."""
        return f"{entity_type}:{name.lower()}"

    def add_entity(
        self,
        name: str,
        entity_type: str,
        memory_id: Optional[str] = None,
        properties: dict = None,
    ) -> GraphNode:
        """Add or update an entity in the graph.

        Args:
            name: Entity name
            entity_type: Type of entity (PERSON, LOCATION, etc.)
            memory_id: ID of the memory where this entity was found
            properties: Additional properties for the entity

        Returns:
            The GraphNode for this entity
        """
        key = self._make_entity_key(name, entity_type)

        if key in self.nodes:
            # Update existing node
            node = self.nodes[key]
            if memory_id:
                node.add_memory_reference(memory_id)
            if properties:
                node.properties.update(properties)
        else:
            # Create new node
            node = GraphNode(
                id=key,
                name=name,
                entity_type=entity_type,
                memory_ids={memory_id} if memory_id else set(),
                properties=properties or {},
            )
            self.nodes[key] = node
            self.type_to_nodes[entity_type].append(key)
            self.name_to_node[name.lower()] = key

        if memory_id:
            self.memory_to_nodes[memory_id].append(key)

        return node

    def add_relationship(
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
        """Add a relationship between two entities.

        Creates the entities if they don't exist.

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
        # Ensure entities exist
        source_node = self.add_entity(source_name, source_type, memory_id)
        target_node = self.add_entity(target_name, target_type, memory_id)

        # Create edge
        edge_id = f"{source_node.id}->{target_node.id}:{relation_type}"

        if edge_id in self.edges:
            # Update existing edge
            edge = self.edges[edge_id]
            edge.confidence = max(edge.confidence, confidence)
            if properties:
                edge.properties.update(properties)
        else:
            edge = GraphEdge(
                id=edge_id,
                source_id=source_node.id,
                target_id=target_node.id,
                relation_type=relation_type,
                memory_id=memory_id,
                confidence=confidence,
                properties=properties or {},
            )
            self.edges[edge_id] = edge
            self.outgoing_edges[source_node.id].append(edge_id)
            self.incoming_edges[target_node.id].append(edge_id)

        return edge

    def get_entity(self, name: str, entity_type: str = None) -> Optional[GraphNode]:
        """Get an entity by name and optionally type.

        Args:
            name: Entity name
            entity_type: Optional entity type filter

        Returns:
            GraphNode if found, None otherwise
        """
        if entity_type:
            key = self._make_entity_key(name, entity_type)
            return self.nodes.get(key)
        else:
            # Search by name only
            node_id = self.name_to_node.get(name.lower())
            if node_id:
                return self.nodes.get(node_id)
            return None

    def get_related_entities(
        self,
        entity_name: str,
        entity_type: str = None,
        relation_types: list[str] = None,
        max_hops: int = 1,
    ) -> list[dict]:
        """Get entities related to a given entity.

        Args:
            entity_name: Name of the source entity
            entity_type: Optional type of source entity
            relation_types: Optional filter for relationship types
            max_hops: Maximum number of hops to traverse (default 1)

        Returns:
            List of related entities with relationship info
        """
        start_node = self.get_entity(entity_name, entity_type)
        if not start_node:
            return []

        visited = {start_node.id}
        results = []
        current_level = [start_node.id]

        for hop in range(max_hops):
            next_level = []

            for node_id in current_level:
                # Get outgoing edges
                for edge_id in self.outgoing_edges.get(node_id, []):
                    edge = self.edges[edge_id]

                    if relation_types and edge.relation_type not in relation_types:
                        continue

                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        next_level.append(edge.target_id)
                        target_node = self.nodes[edge.target_id]
                        results.append({
                            "entity": target_node.to_dict(),
                            "relation": edge.to_dict(),
                            "hops": hop + 1,
                        })

                # Get incoming edges
                for edge_id in self.incoming_edges.get(node_id, []):
                    edge = self.edges[edge_id]

                    if relation_types and edge.relation_type not in relation_types:
                        continue

                    if edge.source_id not in visited:
                        visited.add(edge.source_id)
                        next_level.append(edge.source_id)
                        source_node = self.nodes[edge.source_id]
                        results.append({
                            "entity": source_node.to_dict(),
                            "relation": edge.to_dict(),
                            "hops": hop + 1,
                        })

            current_level = next_level
            if not current_level:
                break

        return results

    def get_memories_for_entity(self, entity_name: str, entity_type: str = None) -> list[str]:
        """Get all memory IDs that mention an entity.

        Args:
            entity_name: Name of the entity
            entity_type: Optional entity type

        Returns:
            List of memory IDs
        """
        node = self.get_entity(entity_name, entity_type)
        if node:
            return list(node.memory_ids)
        return []

    def get_entities_in_memory(self, memory_id: str) -> list[GraphNode]:
        """Get all entities mentioned in a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            List of GraphNodes
        """
        node_ids = self.memory_to_nodes.get(memory_id, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def find_path(
        self,
        source_name: str,
        target_name: str,
        source_type: str = None,
        target_type: str = None,
        max_hops: int = 3,
    ) -> Optional[list[dict]]:
        """Find a path between two entities in the graph.

        Uses BFS to find the shortest path.

        Args:
            source_name: Name of source entity
            target_name: Name of target entity
            source_type: Optional type of source entity
            target_type: Optional type of target entity
            max_hops: Maximum path length

        Returns:
            List of path steps (nodes and edges) or None if no path found
        """
        source = self.get_entity(source_name, source_type)
        target = self.get_entity(target_name, target_type)

        if not source or not target:
            return None

        if source.id == target.id:
            return [{"node": source.to_dict(), "edge": None}]

        # BFS
        queue = [(source.id, [{"node": source.to_dict(), "edge": None}])]
        visited = {source.id}

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Check outgoing edges
            for edge_id in self.outgoing_edges.get(current_id, []):
                edge = self.edges[edge_id]
                next_id = edge.target_id

                if next_id == target.id:
                    return path + [{"node": self.nodes[next_id].to_dict(), "edge": edge.to_dict()}]

                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [{"node": self.nodes[next_id].to_dict(), "edge": edge.to_dict()}]
                    queue.append((next_id, new_path))

            # Check incoming edges
            for edge_id in self.incoming_edges.get(current_id, []):
                edge = self.edges[edge_id]
                next_id = edge.source_id

                if next_id == target.id:
                    return path + [{"node": self.nodes[next_id].to_dict(), "edge": edge.to_dict()}]

                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [{"node": self.nodes[next_id].to_dict(), "edge": edge.to_dict()}]
                    queue.append((next_id, new_path))

        return None

    def get_subgraph_for_query(
        self,
        query_entities: list[str],
        max_hops: int = 2,
    ) -> dict:
        """Get a subgraph containing entities relevant to a query.

        Args:
            query_entities: List of entity names from the query
            max_hops: Maximum hops from query entities

        Returns:
            Dict with nodes, edges, and memory_ids
        """
        relevant_nodes = {}
        relevant_edges = {}
        memory_ids = set()

        for entity_name in query_entities:
            # Find the entity
            node = self.get_entity(entity_name)
            if not node:
                continue

            relevant_nodes[node.id] = node
            memory_ids.update(node.memory_ids)

            # Get related entities
            related = self.get_related_entities(entity_name, max_hops=max_hops)
            for rel in related:
                rel_node_data = rel["entity"]
                rel_node = self.nodes.get(rel_node_data["id"])
                if rel_node:
                    relevant_nodes[rel_node.id] = rel_node
                    memory_ids.update(rel_node.memory_ids)

                rel_edge_data = rel["relation"]
                edge = self.edges.get(rel_edge_data["id"])
                if edge:
                    relevant_edges[edge.id] = edge

        return {
            "nodes": [n.to_dict() for n in relevant_nodes.values()],
            "edges": [e.to_dict() for e in relevant_edges.values()],
            "memory_ids": list(memory_ids),
        }

    def get_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {t: len(nodes) for t, nodes in self.type_to_nodes.items()},
            "avg_edges_per_node": len(self.edges) / max(1, len(self.nodes)),
        }

    def clear(self):
        """Clear all data from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.outgoing_edges.clear()
        self.incoming_edges.clear()
        self.memory_to_nodes.clear()
        self.type_to_nodes.clear()
        self.name_to_node.clear()

    def personalized_pagerank(
        self,
        query_entities: list[str],
        damping_factor: float = 0.85,
        iterations: int = 20,
        convergence_threshold: float = 1e-6,
    ) -> dict[str, float]:
        """
        Compute Personalized PageRank scores for multi-hop reasoning.

        HippoRAG-inspired: Like human memory, activates related concepts
        through spreading activation from query entities.

        Args:
            query_entities: List of entity names from the query (seed nodes)
            damping_factor: Probability of following a link (vs random teleport)
            iterations: Maximum iterations
            convergence_threshold: Stop when scores change less than this

        Returns:
            Dict mapping node_id -> PageRank score
        """
        if not self.nodes:
            return {}

        # Step 1: Find seed nodes for personalization
        seed_nodes = set()
        for entity_name in query_entities:
            node = self.get_entity(entity_name)
            if node:
                seed_nodes.add(node.id)

        if not seed_nodes:
            # No query entities found, use uniform distribution
            return {node_id: 1.0 / len(self.nodes) for node_id in self.nodes}

        # Step 2: Initialize personalization vector (biased toward query entities)
        n = len(self.nodes)
        personalization = {}
        for node_id in self.nodes:
            if node_id in seed_nodes:
                personalization[node_id] = 1.0 / len(seed_nodes)
            else:
                personalization[node_id] = 0.0

        # Step 3: Initialize scores uniformly
        scores = {node_id: 1.0 / n for node_id in self.nodes}

        # Step 4: Power iteration
        for iteration in range(iterations):
            new_scores = {}
            max_diff = 0.0

            for node_id in self.nodes:
                # Teleport probability (biased toward seed nodes)
                teleport_prob = (1 - damping_factor) * personalization[node_id]

                # Contribution from incoming edges
                inlink_sum = 0.0

                # Get all neighbors (both incoming and outgoing for undirected traversal)
                neighbor_edge_ids = (
                    self.incoming_edges.get(node_id, []) +
                    self.outgoing_edges.get(node_id, [])
                )

                for edge_id in self.incoming_edges.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if edge:
                        source_id = edge.source_id
                        # Calculate out-degree of source
                        source_out_degree = len(self.outgoing_edges.get(source_id, []))
                        if source_out_degree > 0:
                            inlink_sum += scores[source_id] / source_out_degree

                # Also consider reverse edges (treat graph as undirected for memory retrieval)
                for edge_id in self.outgoing_edges.get(node_id, []):
                    edge = self.edges.get(edge_id)
                    if edge:
                        target_id = edge.target_id
                        # Calculate in-degree of target as "reverse out-degree"
                        target_in_degree = len(self.incoming_edges.get(target_id, []))
                        if target_in_degree > 0:
                            inlink_sum += scores[target_id] / target_in_degree

                new_scores[node_id] = teleport_prob + damping_factor * inlink_sum
                max_diff = max(max_diff, abs(new_scores[node_id] - scores[node_id]))

            scores = new_scores

            # Check convergence
            if max_diff < convergence_threshold:
                break

        return scores

    def get_memories_by_pagerank(
        self,
        query_entities: list[str],
        top_k: int = 10,
        min_score_threshold: float = 0.001,
    ) -> list[tuple[str, float]]:
        """
        Get memories ranked by Personalized PageRank relevance.

        This enables multi-hop reasoning by finding memories connected
        to query entities through the knowledge graph.

        Args:
            query_entities: List of entity names from the query
            top_k: Maximum number of memory IDs to return
            min_score_threshold: Minimum PageRank score to include

        Returns:
            List of (memory_id, score) tuples, sorted by score descending
        """
        # Compute PageRank scores for all entities
        pr_scores = self.personalized_pagerank(query_entities)

        if not pr_scores:
            return []

        # Aggregate scores by memory
        memory_scores: dict[str, float] = defaultdict(float)

        for node_id, score in pr_scores.items():
            if score < min_score_threshold:
                continue

            node = self.nodes.get(node_id)
            if node:
                for memory_id in node.memory_ids:
                    # Each memory gets the max PageRank score of its entities
                    memory_scores[memory_id] = max(memory_scores[memory_id], score)

        # Sort by score and return top K
        sorted_memories = sorted(
            memory_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_memories[:top_k]

    def get_entity_proximity_score(
        self,
        entity1_name: str,
        entity2_name: str,
        entity1_type: str = None,
        entity2_type: str = None,
    ) -> float:
        """
        Calculate proximity score between two entities based on graph distance.

        Used for reranking: memories containing entities close to query entities
        get boosted.

        Args:
            entity1_name: First entity name
            entity2_name: Second entity name
            entity1_type: Optional type for first entity
            entity2_type: Optional type for second entity

        Returns:
            Proximity score (0-1), where 1 = same entity, 0 = no path
        """
        if entity1_name.lower() == entity2_name.lower():
            return 1.0

        # Find path between entities
        path = self.find_path(
            entity1_name,
            entity2_name,
            entity1_type,
            entity2_type,
            max_hops=3
        )

        if not path:
            return 0.0

        # Score based on path length: 1-hop = 0.5, 2-hop = 0.33, 3-hop = 0.25
        return 1.0 / len(path)
