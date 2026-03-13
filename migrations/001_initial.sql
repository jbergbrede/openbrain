CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    summary TEXT,
    embedding vector,
    people TEXT[] DEFAULT '{}',
    topics TEXT[] DEFAULT '{}',
    action_items JSONB DEFAULT '[]',
    connections UUID[] DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'mcp',
    source_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memories_people ON memories USING GIN (people);
CREATE INDEX idx_memories_topics ON memories USING GIN (topics);
CREATE INDEX idx_memories_connections ON memories USING GIN (connections);
CREATE INDEX idx_memories_created_at ON memories (created_at DESC);
