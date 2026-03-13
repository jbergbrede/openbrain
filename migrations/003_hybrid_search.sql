ALTER TABLE memories ADD COLUMN language VARCHAR(2) NOT NULL DEFAULT 'en';
ALTER TABLE memories ADD COLUMN content_english TEXT;
ALTER TABLE memories ADD COLUMN content_german TEXT;
ALTER TABLE memories ADD COLUMN search_vector TSVECTOR;

CREATE INDEX idx_memories_search_vector ON memories USING GIN (search_vector);

-- Backfill: treat existing rows as English
UPDATE memories
SET content_english = content,
    search_vector =
        setweight(to_tsvector('english', COALESCE(content, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(summary, '')), 'B');
