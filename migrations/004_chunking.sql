-- Chunking on ingestion: add chunks table
-- NOTE: does NOT drop memories.embedding yet — run 005_drop_embedding.sql after backfill

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector,
    token_count INTEGER NOT NULL,
    UNIQUE (memory_id, chunk_index)
);

CREATE INDEX idx_chunks_memory_id ON chunks (memory_id);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
