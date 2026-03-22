-- Migration 002: Add facts table for atomic fact extraction
-- Facts are atomic statements extracted from memories for better retrieval
-- Example: "User visited MoMA on January 8, 2023"
--
-- Run this on Supabase SQL Editor
-- Date: 2026-01-25

-- Create facts table
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,

    -- Fact content
    fact_text TEXT NOT NULL,
    fact_type VARCHAR(50) DEFAULT 'general',  -- general, preference, event, relationship

    -- Structured extraction (optional, for advanced queries)
    subject_entity VARCHAR(255),  -- e.g., "User", "John"
    predicate VARCHAR(100),       -- e.g., "visited", "prefers", "lives in"
    object_entity VARCHAR(255),   -- e.g., "MoMA", "ocean-view hotels"

    -- Temporal information
    event_time TIMESTAMP WITH TIME ZONE,  -- When the fact's event occurred

    -- Source tracking
    source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,

    -- Embedding for semantic search (3072 dimensions for text-embedding-3-large)
    embedding VECTOR(3072),

    -- Confidence and metadata
    confidence FLOAT DEFAULT 1.0,
    extra_data JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Indexes for facts table
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_user_type ON facts(user_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_source_memory ON facts(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_facts_event_time ON facts(user_id, event_time) WHERE event_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(user_id, subject_entity) WHERE subject_entity IS NOT NULL;

-- Enable vector similarity search on facts
-- Use halfvec(3072) casting to bypass HNSW 2000 dimension limit
-- This matches the pattern used in memories and entities tables
CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts
USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Row Level Security for facts
ALTER TABLE facts ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own facts
-- Drop existing policy if any, then create
DROP POLICY IF EXISTS facts_user_isolation ON facts;
CREATE POLICY facts_user_isolation ON facts
    FOR ALL
    USING (user_id = auth.uid());

-- Comment
COMMENT ON TABLE facts IS
'Atomic facts extracted from memories for improved retrieval.
Each fact is a single, verifiable statement that can be independently searched.
Example: "User visited MoMA on January 8, 2023"';
