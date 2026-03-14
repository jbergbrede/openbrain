-- Drop embedding column from memories after backfill is complete
-- Run ONLY after scripts/backfill_chunks.py has been executed successfully

ALTER TABLE memories DROP COLUMN embedding;
