# Hybrid E2E preflight — 2026-11-09

This report records read-only checks of the **currently running Qdrant Core**
and preparation for a controlled hybrid reload. It is not a hybrid E2E result.

## Runtime and ownership

| Port | Service | Listener PID | Confirmed status |
| --- | --- | ---: | --- |
| 8000 | Core | 8956 | HTTP 200; `search_backend=qdrant`; readiness true |
| 8001 | Model API | 29104 | HTTP 200; Qwen2.5-7B-Instruct + LoRA V4 |
| 8002 | Gateway | 4604 | HTTP 200, healthy |
| 6333 | Qdrant | 7044 | HTTP 200 |
| 9200 | Elasticsearch | 7044 | HTTP 200; index `dai_documents_v1`, yellow |
| 5432 | PostgreSQL | 7044 | Listening; Core reports PostgreSQL available |

Read-only process ancestry: `run_gateway_stack.py` PID 20404 is a child of
venv wrapper PID 27308. It spawned venv child wrappers 23300 (Core), 28996
(Model), and 476 (Gateway), whose active Python processes are the listeners
above. The launcher terminates **all** children when any exits. Child-only
Core termination would therefore interrupt Qwen. PIDs must be rechecked
immediately before any reload.

`run_gateway_stack.py` now inherits `SEARCH_BACKEND`; without a setting, the
Core configuration defaults to `qdrant`. Hybrid requires the launch terminal
to set `SEARCH_BACKEND=hybrid` explicitly and provide `SEARCH_DATABASE_URL` or
`CATALOG_DATABASE_URL`. The current launcher process's environment is unknown;
the diagnostic shell used for this report had neither variable. No secrets
were read or printed.

## Read-only data inspection

| Store | Dataset | Count | Observation |
| --- | --- | ---: | --- |
| Qdrant | `legal_docs` | 3,927 | Legacy points; sample has no domain/source/document ID |
| Qdrant | `legal_docs_bge_m3` | 3,927 | Registry selects this collection for both legal domains; sampled points untagged |
| Qdrant | `domain_documents_tourism` | 700 | 700/700 tagged tourism |
| Qdrant | `domain_documents_healthcare` | 209 | 209/209 tagged healthcare, but 100 Education and 9 README rows are misassigned |
| Qdrant | `domain_documents_shopping` | 114 | 100 product + 14 instruction rows |
| Elasticsearch | `dai_documents_v1` | 8,778 | 3,927 legal, 3,927 administrative, 624 tourism, 100 healthcare, 100 education, 100 shopping |

Qdrant uploaded samples have domain, source, document, chunk and Evidence IDs.
Legacy legal vectors have no such tags; legal isolation relies on collection
selection, and the registry currently maps both legal domains to the same
vector collection. This remains a limitation for domain isolation claims.

A sampled Healthcare point had identical chunk text, source, document, chunk
and Evidence IDs in Qdrant and PostgreSQL/Elasticsearch. Qdrant stored its
record ID under `record_fields.record_id`; PostgreSQL exposed it as `record_id`.
The RRF identity key was corrected to read either location, with a regression
test. This does not modify stored records.

An Elasticsearch Education hit has `domain_id=education` but an Evidence ID
beginning `domain:healthcare:`. Its indexed document key also begins
`domain_documents_healthcare:`. This was verified through PostgreSQL hydration.
The Evidence Builder already rejects mismatched domain/source identity. Hybrid
fusion now excludes such uploaded hits earlier. It does not rewrite the ID.
Education retrieval may consequently have zero valid Evidence until the source
data receives a separately approved repair. Qdrant Healthcare still contains
misassigned Education/README rows and requires data repair; this task does not
mutate them.

The Elasticsearch BM25 search and PostgreSQL hydration were exercised directly
for legal, healthcare, education and tourism. Hits and metadata were returned.
This is **read-only component integration**, not Gateway/Core hybrid E2E.
BM25 scores are not calibrated probabilities or legal correctness judgments.

## Controlled reload proposal — approval required

The existing launcher cannot safely reload only Core. After explicit approval:

1. In the terminal that owns launcher PID 20404, stop the launcher with
   `Ctrl+C`. Wait until ports 8000/8001/8002 are free. Verify those PIDs still
   belong to this launcher before acting. This will unload and later reload
   Qwen once; PostgreSQL, Qdrant and Elasticsearch remain running.
2. In the launch terminal, set `SEARCH_BACKEND=hybrid`, ensure a valid
   `SEARCH_DATABASE_URL` or `CATALOG_DATABASE_URL` is set without printing it,
   and remove any uncalibrated `ELASTICSEARCH_MIN_SCORE` placeholder. Keep the
   same Model API configuration and LoRA V4.
3. Start `.\.venv\Scripts\python.exe run_gateway_stack.py` once. Confirm
   listeners, health and `/api/v2/search/readiness` reports all three stores
   available and `configured_backend=hybrid`.
4. Send bounded legal/tourism/shopping queries through Gateway. Capture trace
   IDs, branch scores/ranks, merged candidate IDs, reranker results, Evidence
   IDs, citations, verifier status and stage timings. Do not use unreviewed
   labels to claim answer correctness.
5. Run A01–A03 smoke with `--limit 3 --runs 1 --warmup 1 --timeout 120` in a
   new timestamped results directory. Stop before the full benchmark if a
   timeout or serious verification failure remains.
6. To compare Elasticsearch-only and Qdrant-only full runtime using identical
   code/queries requires separate controlled Core reloads. The current
   launcher would also restart Qwen each time. Request a separate decision
   after initial hybrid smoke instead of doing extra restarts implicitly.

Rollback: stop the same launcher terminal, remove `SEARCH_BACKEND` or set it
to `qdrant`, restart the launcher once, and verify Core readiness. No index or
collection changes are involved.

## Validation status

Hybrid Gateway E2E, three-mode full runtime comparison, reranker/Evidence
outcomes, citation pass rate, fallback under real backend outage, and benchmark
latencies: **not run pending reload approval**. The unit test for one failed
branch confirms explicit partial status, but is test-only.
