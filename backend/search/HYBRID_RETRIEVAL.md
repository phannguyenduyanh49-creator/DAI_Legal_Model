# Hybrid retrieval

The stack launcher inherits `SEARCH_BACKEND` from the caller. With no explicit
setting, Core defaults to `qdrant`. Set `SEARCH_BACKEND=hybrid` to validate the
hybrid path; this does not change the default for other runs.

Request path: domain routing -> BGE-M3 query embedding -> concurrent Qdrant and
Elasticsearch BM25 searches -> reciprocal rank fusion (RRF, k=60) -> existing
relevance filtering / BGE reranker -> Evidence Builder -> existing prompt,
Qwen2.5-7B + LoRA V4, citation and verifier pipeline.

## Configuration

Use these settings in the terminal that will launch Core (not an already
running process). Supply the real database connection securely; no credentials
are stored in this document.

```powershell
Set-Location D:\Prj\DAI-Legal-Model
$env:SEARCH_BACKEND = 'hybrid'
$env:ELASTICSEARCH_URL = 'http://127.0.0.1:9200'
$env:ELASTICSEARCH_INDEX = 'dai_documents_v1'
# SEARCH_DATABASE_URL or CATALOG_DATABASE_URL must already contain a valid URL.
# If the previous value was the literal '<gia tri da duoc calibration>', remove
# that placeholder from this shell. No fabricated numeric threshold is needed:
Remove-Item Env:ELASTICSEARCH_MIN_SCORE -ErrorAction SilentlyContinue
```

Hybrid does not require a BM25 minimum: BM25 only supplies bounded candidates
to RRF, not answer acceptance. A supplied minimum must be finite/nonnegative.
The vector minimum, reranker threshold, evidence and verifier policies remain
unchanged. RRF is a rank score, never semantic confidence. Elasticsearch-only
mode still requires the explicit minimum under its existing contract.

Do not launch another stack while ports 8000/8001/8002 are occupied. Existing
services have not been restarted by this change. The current launcher shuts
down every child if one exits; restarting its Core child can also stop Qwen.
Plan a coordinated reload rather than killing a child independently.

## Integrity and operations

- ES results are domain-filtered and hydrated from PostgreSQL, with the domain
  checked again. Uploaded Qdrant records use a domain filter. Untagged legacy
  records rely on collections selected by the existing authoritative registry.
- Explicit foreign domain tags are rejected before fusion. No schema/data
  migrations or collection changes are performed.
- Fusion deduplication requires matching source, document, record, chunk and
  exact content. Inconsistent historical identities may remain separate; they
  are not guessed or repaired automatically.
- Branch raw scores, ranks, point IDs, fused scores and timings are logged with
  request IDs. Payload contents are not logged by the fusion implementation.
- Branch searches have a 30-second wait limit. Worker-thread cancellation does
  not interrupt network I/O; client socket timeouts remain important. Collection
  discovery and embedding precede fan-out and can still fail the request.
- A failed branch is reported as partial; both branches failing raises an error.
  No evidence or successful answer is fabricated on failure.
- Upload storage remains Qdrant in hybrid mode. Existing ES indexed records are
  read; this change does not dual-write or automatically backfill new uploads.
  Newly uploaded documents may therefore appear only in the vector branch.
- Rollback: set `SEARCH_BACKEND=qdrant` for the next controlled Core launch.
  No data rollback is required.

## Validation

Run `.\.venv\Scripts\python.exe -m pytest backend/tests tests -q -p no:cacheprovider`.
Tests cover fusion, original score/payload preservation, duplicate handling,
domain rejection, unchanged vector cutoff, invalid configuration, empty
results, partial failures and timeouts. They do not establish retrieval quality.
Final local result: **492 passed**, one existing Starlette/AnyIO deprecation
warning. Python syntax checks also passed. An initial timing-sensitive timeout
test failed on Windows worker startup; it was replaced with a deterministic
async scheduling test before rerunning the full suite successfully.

Read-only runtime checks on 2026-11-09 returned HTTP 200 for Core, Model,
Gateway, Qdrant and Elasticsearch. Core reported **qdrant**, readiness true;
Elasticsearch cluster health was yellow. These are checks of the old process,
not live validation of hybrid. No new model instance was loaded.

After controlled reload, verify `/api/v2/search/readiness` reports `hybrid` and
PostgreSQL/Qdrant/Elasticsearch available, then send fixed questions through
Gateway. Inspect both branch diagnostics, evidence, citations and outcome.
Latency and answer quality improvements remain unmeasured; use reviewed
relevance labels and identical baseline queries before making such claims.
