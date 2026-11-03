CREATE TABLE IF NOT EXISTS domain_search_documents (
    document_key TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    source_type TEXT NOT NULL,
    record_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_domain_search_domain_source
    ON domain_search_documents (domain_id, source_id);
CREATE INDEX IF NOT EXISTS ix_domain_search_document
    ON domain_search_documents (domain_id, document_id, chunk_index);
CREATE INDEX IF NOT EXISTS ix_domain_search_record
    ON domain_search_documents (domain_id, record_id);

CREATE TABLE IF NOT EXISTS domain_search_outbox (
    id BIGSERIAL PRIMARY KEY,
    document_key TEXT NOT NULL REFERENCES domain_search_documents(document_key) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (event_type = 'index'),
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    last_error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_domain_search_outbox_pending
    ON domain_search_outbox (document_key) WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_domain_search_outbox_ready
    ON domain_search_outbox (available_at, id) WHERE processed_at IS NULL;
