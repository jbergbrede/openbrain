CREATE TABLE action_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    due_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_action_items_memory_id ON action_items (memory_id);
CREATE INDEX idx_action_items_status ON action_items (status);
CREATE INDEX idx_action_items_due_date ON action_items (due_date) WHERE due_date IS NOT NULL;
CREATE INDEX idx_action_items_status_due ON action_items (status, due_date);

-- Extract existing JSONB action items into new table
INSERT INTO action_items (memory_id, text, status, due_date, created_at)
SELECT
    m.id,
    ai->>'text',
    COALESCE(ai->>'status', 'open'),
    CASE WHEN ai->>'due_date' IS NOT NULL AND ai->>'due_date' != ''
         THEN (ai->>'due_date')::date ELSE NULL END,
    m.created_at
FROM memories m,
     jsonb_array_elements(m.action_items) AS ai
WHERE jsonb_array_length(m.action_items) > 0;
