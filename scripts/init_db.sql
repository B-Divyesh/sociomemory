-- SocioMemory Database Initialization
-- Run this script to set up the database schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Memories table with FSRS fields
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,

    -- Content
    memory_type VARCHAR(50) NOT NULL DEFAULT 'fact',
    content TEXT NOT NULL,
    embedding VECTOR(3072),

    -- Temporal fields (Graphiti/Zep inspired)
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    valid_until TIMESTAMP WITH TIME ZONE,
    is_latest BOOLEAN DEFAULT TRUE NOT NULL,

    -- FSRS fields for retrieval optimization
    stability FLOAT DEFAULT 1.0 NOT NULL,
    difficulty FLOAT DEFAULT 0.3 NOT NULL,
    retrievability FLOAT DEFAULT 1.0 NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    access_count INTEGER DEFAULT 0 NOT NULL,

    -- Source info
    source_platform VARCHAR(50),
    source_id VARCHAR(255),
    confidence FLOAT DEFAULT 1.0 NOT NULL,
    extra_data JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Memory relationships (updates, extends, derives, contradicts)
CREATE TABLE IF NOT EXISTS memory_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,
    confidence FLOAT DEFAULT 1.0 NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Entities table
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    embedding VECTOR(3072),
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    UNIQUE(user_id, name, entity_type)
);

-- Entity mentions linking entities to memories
CREATE TABLE IF NOT EXISTS entity_mentions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    mention_context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_valid ON memories(user_id, is_latest, valid_until);
CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_user_platform ON memories(user_id, source_platform);

-- IMPORTANT: Vector index for 3072 dimensions (text-embedding-3-large)
-- pgvector has a 2000 dimension limit for BOTH ivfflat AND hnsw indexes on vector type
-- Solution: Use halfvec type (half-precision) which supports up to 4000 dimensions
-- Reference: https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes
-- Reference: https://github.com/pgvector/pgvector/issues/442
--
-- How it works:
-- - Data is stored as VECTOR(3072) at full precision
-- - Index is created on embedding::halfvec(3072) at half precision
-- - Queries must cast both sides to halfvec for index to be used
-- - Requires pgvector 0.7.0+ (Supabase supports this)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_memory_id);

CREATE INDEX IF NOT EXISTS idx_entities_user_id ON entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_user_type ON entities(user_id, entity_type);

-- HNSW index for entities embeddings (3072 dimensions) using halfvec
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
    USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_memory ON entity_mentions(memory_id);

-- Function: Search memories with FSRS-weighted priority scoring
-- IMPORTANT: Uses halfvec casting to leverage the HNSW index on 3072-dim vectors
-- Both sides of the <=> operator must be cast to halfvec for index usage
CREATE OR REPLACE FUNCTION search_memories(
    p_user_id UUID,
    p_query_embedding VECTOR(3072),
    p_limit INTEGER DEFAULT 10,
    p_threshold FLOAT DEFAULT 0.7,
    p_platforms TEXT[] DEFAULT NULL,
    p_memory_types TEXT[] DEFAULT NULL,
    p_only_latest BOOLEAN DEFAULT TRUE,
    p_only_valid BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    id UUID,
    memory_type VARCHAR,
    content TEXT,
    similarity FLOAT,
    retrievability FLOAT,
    priority_score FLOAT,
    source_platform VARCHAR,
    confidence FLOAT,
    valid_from TIMESTAMP WITH TIME ZONE,
    valid_until TIMESTAMP WITH TIME ZONE,
    is_latest BOOLEAN,
    stability FLOAT,
    difficulty FLOAT,
    access_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.memory_type,
        m.content,
        -- Cast both sides to halfvec for HNSW index usage
        (1 - (m.embedding::halfvec(3072) <=> p_query_embedding::halfvec(3072)))::FLOAT AS similarity,
        m.retrievability,
        -- Priority = similarity * 0.5 + FSRS_retrievability * 0.3 + recency_bonus * 0.2
        (
            (1 - (m.embedding::halfvec(3072) <=> p_query_embedding::halfvec(3072))) * 0.5 +
            m.retrievability * 0.3 +
            LEAST(1.0, 1.0 / (1 + EXTRACT(EPOCH FROM (NOW() - m.last_accessed)) / 86400)) * 0.2
        )::FLOAT AS priority_score,
        m.source_platform,
        m.confidence,
        m.valid_from,
        m.valid_until,
        m.is_latest,
        m.stability,
        m.difficulty,
        m.access_count,
        m.created_at
    FROM memories m
    WHERE m.user_id = p_user_id
      AND m.embedding IS NOT NULL
      AND (NOT p_only_latest OR m.is_latest = TRUE)
      AND (NOT p_only_valid OR m.valid_until IS NULL)
      AND (1 - (m.embedding::halfvec(3072) <=> p_query_embedding::halfvec(3072))) >= p_threshold
      AND (p_platforms IS NULL OR m.source_platform = ANY(p_platforms))
      AND (p_memory_types IS NULL OR m.memory_type = ANY(p_memory_types))
    ORDER BY priority_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Update memory access (for FSRS tracking)
CREATE OR REPLACE FUNCTION record_memory_access(
    p_memory_id UUID,
    p_was_useful BOOLEAN DEFAULT TRUE
)
RETURNS VOID AS $$
DECLARE
    v_current_stability FLOAT;
    v_current_difficulty FLOAT;
    v_days_since_access FLOAT;
    v_new_stability FLOAT;
    v_new_difficulty FLOAT;
    v_new_retrievability FLOAT;
BEGIN
    -- Get current FSRS state
    SELECT stability, difficulty,
           EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400
    INTO v_current_stability, v_current_difficulty, v_days_since_access
    FROM memories WHERE id = p_memory_id;

    -- Simplified FSRS update (full implementation in Python service)
    IF p_was_useful THEN
        -- Successful recall: increase stability
        v_new_stability := LEAST(365.0, v_current_stability * (1.0 + 0.1 * v_current_stability));
        v_new_difficulty := GREATEST(0.1, v_current_difficulty - 0.05);
    ELSE
        -- Failed recall: decrease stability
        v_new_stability := GREATEST(1.0, v_current_stability * 0.5);
        v_new_difficulty := LEAST(1.0, v_current_difficulty + 0.1);
    END IF;

    -- Calculate new retrievability (exponential decay)
    v_new_retrievability := POWER(0.9, v_days_since_access / GREATEST(v_new_stability, 0.1));

    -- Update memory
    UPDATE memories
    SET
        stability = v_new_stability,
        difficulty = v_new_difficulty,
        retrievability = v_new_retrievability,
        last_accessed = NOW(),
        access_count = access_count + 1,
        updated_at = NOW()
    WHERE id = p_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Invalidate memory (set valid_until)
CREATE OR REPLACE FUNCTION invalidate_memory(
    p_memory_id UUID
)
RETURNS VOID AS $$
BEGIN
    UPDATE memories
    SET
        valid_until = NOW(),
        is_latest = FALSE,
        updated_at = NOW()
    WHERE id = p_memory_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Get memory stats for user
CREATE OR REPLACE FUNCTION get_memory_stats(
    p_user_id UUID
)
RETURNS TABLE (
    total_memories BIGINT,
    by_type JSONB,
    by_platform JSONB,
    avg_stability FLOAT,
    avg_retrievability FLOAT,
    total_accesses BIGINT,
    memories_accessed_today BIGINT,
    oldest_memory TIMESTAMP WITH TIME ZONE,
    newest_memory TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT AS total_memories,
        (SELECT jsonb_object_agg(memory_type, cnt)
         FROM (SELECT memory_type, COUNT(*) as cnt FROM memories WHERE user_id = p_user_id GROUP BY memory_type) t
        ) AS by_type,
        (SELECT jsonb_object_agg(COALESCE(source_platform, 'unknown'), cnt)
         FROM (SELECT source_platform, COUNT(*) as cnt FROM memories WHERE user_id = p_user_id GROUP BY source_platform) t
        ) AS by_platform,
        AVG(m.stability)::FLOAT AS avg_stability,
        AVG(m.retrievability)::FLOAT AS avg_retrievability,
        SUM(m.access_count)::BIGINT AS total_accesses,
        COUNT(*) FILTER (WHERE m.last_accessed >= NOW() - INTERVAL '1 day')::BIGINT AS memories_accessed_today,
        MIN(m.created_at) AS oldest_memory,
        MAX(m.created_at) AS newest_memory
    FROM memories m
    WHERE m.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;
