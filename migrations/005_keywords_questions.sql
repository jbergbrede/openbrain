-- Add keywords and questions columns to memories
ALTER TABLE memories ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS questions TEXT[] DEFAULT '{}';

-- Add is_synthetic flag to chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

-- Drop the old unique constraint and replace with a partial index
-- (synthetic chunks share chunk_index=-1, so uniqueness only applies to real chunks)
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_memory_id_chunk_index_key;
CREATE UNIQUE INDEX IF NOT EXISTS chunks_memory_id_chunk_index_key
    ON chunks (memory_id, chunk_index) WHERE NOT is_synthetic;
