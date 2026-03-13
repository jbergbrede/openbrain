-- Fix double-encoded action_items: rows where action_items is a JSON string
-- (e.g. '"[]"') instead of a JSON array/object.
UPDATE memories
SET action_items = action_items::text::jsonb
WHERE jsonb_typeof(action_items) = 'string';
