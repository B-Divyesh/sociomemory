-- Migration 003: Add knowledge graph persistence tables
-- Stores nodes (entities) and edges (relationships) for graph-based retrieval
-- Enables Personalized PageRank and multi-hop reasoning
--
-- Run this on Supabase SQL Editor
-- Date: 2026-01-25

-- =====================================================
-- Graph Nodes Table (extends existing entities table)
-- =====================================================
-- The existing 'entities' table serves as our node storage.
-- We add additional columns for graph-specific properties.

ALTER TABLE entities
ADD COLUMN IF NOT EXISTS node_properties JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS mention_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_mentioned_at TIMESTAMP WITH TIME ZONE;

-- Index for graph traversal
CREATE INDEX IF NOT EXISTS idx_entities_mention_count
ON entities(user_id, mention_count DESC);

-- =====================================================
-- Graph Edges Table
-- =====================================================
CREATE TABLE IF NOT EXISTS graph_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,

    -- Edge endpoints (reference entities by composite key)
    source_entity_name VARCHAR(255) NOT NULL,
    source_entity_type VARCHAR(50) NOT NULL,
    target_entity_name VARCHAR(255) NOT NULL,
    target_entity_type VARCHAR(50) NOT NULL,

    -- Relationship info
    relation_type VARCHAR(100) NOT NULL,  -- e.g., "friend_of", "works_at", "visited"
    confidence FLOAT DEFAULT 1.0,

    -- Source memory that established this relationship
    source_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,

    -- Edge properties
    edge_properties JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Unique constraint to prevent duplicate edges
    CONSTRAINT uq_graph_edge UNIQUE (
        user_id,
        source_entity_name,
        source_entity_type,
        target_entity_name,
        target_entity_type,
        relation_type
    )
);

-- Indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_graph_edges_user ON graph_edges(user_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(user_id, source_entity_name, source_entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(user_id, target_entity_name, target_entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relation ON graph_edges(user_id, relation_type);

-- Row Level Security
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own graph edges
-- Drop existing policy if any, then create
DROP POLICY IF EXISTS graph_edges_user_isolation ON graph_edges;
CREATE POLICY graph_edges_user_isolation ON graph_edges
    FOR ALL
    USING (user_id = auth.uid());

-- =====================================================
-- Helper function: Get entity connections for PageRank
-- =====================================================
CREATE OR REPLACE FUNCTION get_entity_connections(
    p_user_id UUID,
    p_entity_name VARCHAR(255),
    p_entity_type VARCHAR(50)
)
RETURNS TABLE (
    connected_entity_name VARCHAR(255),
    connected_entity_type VARCHAR(50),
    relation_type VARCHAR(100),
    direction VARCHAR(10)
) AS $$
BEGIN
    RETURN QUERY
    -- Outgoing edges
    SELECT
        ge.target_entity_name,
        ge.target_entity_type,
        ge.relation_type,
        'outgoing'::VARCHAR(10) as direction
    FROM graph_edges ge
    WHERE ge.user_id = p_user_id
      AND ge.source_entity_name = p_entity_name
      AND ge.source_entity_type = p_entity_type

    UNION ALL

    -- Incoming edges
    SELECT
        ge.source_entity_name,
        ge.source_entity_type,
        ge.relation_type,
        'incoming'::VARCHAR(10) as direction
    FROM graph_edges ge
    WHERE ge.user_id = p_user_id
      AND ge.target_entity_name = p_entity_name
      AND ge.target_entity_type = p_entity_type;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Helper function: Get memories for entity (via mentions)
-- =====================================================
CREATE OR REPLACE FUNCTION get_memories_for_entity(
    p_user_id UUID,
    p_entity_name VARCHAR(255),
    p_entity_type VARCHAR(50),
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT
        m.id as memory_id,
        m.content,
        m.created_at
    FROM memories m
    JOIN entity_mentions em ON em.memory_id = m.id
    JOIN entities e ON e.id = em.entity_id
    WHERE e.user_id = p_user_id
      AND e.name = p_entity_name
      AND e.entity_type = p_entity_type
      AND m.is_latest = true
      AND m.valid_until IS NULL
    ORDER BY m.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE graph_edges IS
'Knowledge graph edges representing relationships between entities.
Used for multi-hop reasoning and Personalized PageRank retrieval.';

COMMENT ON FUNCTION get_entity_connections IS
'Get all entities connected to a given entity (both directions).
Used for graph traversal in PageRank computation.';
