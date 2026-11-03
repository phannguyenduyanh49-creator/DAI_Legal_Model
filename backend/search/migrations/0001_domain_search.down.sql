-- Rollback only the new Elasticsearch-backed document storage objects.
-- Existing Catalog tables and Qdrant collections are intentionally untouched.
DROP TABLE IF EXISTS domain_search_outbox;
DROP TABLE IF EXISTS domain_search_documents;
