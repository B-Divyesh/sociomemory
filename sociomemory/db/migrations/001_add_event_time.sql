-- Migration 001: Add bi-temporal model support
-- This adds event_time column to track WHEN events in memories occurred
-- (separate from created_at which tracks WHEN the memory was stored)
--
-- Run this on Supabase SQL Editor
-- Date: 2026-01-25

-- Add event_time column to memories table
ALTER TABLE memories
ADD COLUMN IF NOT EXISTS event_time TIMESTAMP WITH TIME ZONE;

-- Add index for event_time queries (temporal reasoning)
CREATE INDEX IF NOT EXISTS idx_memories_user_event_time
ON memories(user_id, event_time)
WHERE event_time IS NOT NULL;

-- Add composite index for temporal range queries
CREATE INDEX IF NOT EXISTS idx_memories_temporal_range
ON memories(user_id, event_time, created_at)
WHERE event_time IS NOT NULL;

-- Comment explaining the bi-temporal model
COMMENT ON COLUMN memories.event_time IS
'When the event described in the memory occurred (extracted from content).
Different from created_at which is when the memory was stored.
NULL if no specific date could be extracted.';
