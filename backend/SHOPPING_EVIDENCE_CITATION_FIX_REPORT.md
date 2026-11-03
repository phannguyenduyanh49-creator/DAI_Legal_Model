# Shopping Evidence Identity and Citation Verification Report

## Scope and runtime status

This report records code-level fixes and automated tests for the user-provided Shopping trace `trace_legal_5c100ca42e9d42bc9abe861bdc232ffb`. That request and its three Qdrant point IDs were not replayed or independently inspected during this run.

Read-only listener checks found no listeners on ports 8000 (Core), 8001 (Model API), 8002 (Gateway), 6333 (Qdrant), or 5432 (PostgreSQL). No service was started, stopped, or restarted. No Qdrant/PostgreSQL data was read or changed. Consequently, the live payloads for points `2521e13a-357e-4368-bfb2-b8358d17e48d`, `6dffc11b-483f-4795-beaf-511f32516b0e`, and `11422170-a54d-4141-8821-566d654bd206` remain unverified.

## Findings

### Confirmed metadata loss in the code path

The Qdrant retrieval path in `backend/main.py` read `domain_id`, `source_id`, `evidence_id`, row metadata, and record fields into a retrieval candidate, but omitted the payload's `document_id` and `chunk_index`. `backend/rag/adapter.py` requires the document identity to be carried as Evidence metadata. The verifier requires uploaded Evidence to have a domain, `metadata.document_id`, and `source_id` matching the document identity, and requires the Evidence ID prefix to match that provenance. Thus a valid Qdrant payload with `source_id` present could become Evidence without `metadata.document_id` and fail closed as missing source identity. This matches the supplied `EVIDENCE_SOURCE_IDENTITY_MISSING` diagnostics.

The same omission existed in `backend/tourism_retrieval.py`: structured tourism selection reconstructed a reduced candidate and dropped the original document identity. This was confirmed by the first full-suite run; the existing tourism Evidence test failed under the new pre-generation identity check. The selector now retains document, record, point, and chunk identities.

The Reranker does not drop candidate metadata: `rerank_local_results` in `backend/main.py` shallow-copies the selected candidate and adds `rerank_score`. The `record_id=unknown` diagnostic did not prove an identity failure. Some uploaded workbooks have no source `record_id`; for those, the existing ingestion contract provides workbook `document_id`/`source_id`, per-chunk `evidence_id` and `chunk_index`, and, for structured sheets, `sheet_name`/`row_number`. No record ID is synthesized.

### Evidence identity policy

For uploaded documents, a usable source identity requires a nonempty `domain_id`, `document_id`, and independent `source_id`, with `source_id == document_id` under the current ingestion contract and an Evidence ID beginning `domain:{domain_id}:{document_id}:`. The Evidence ID/chunk index identifies a chunk within the document. `record_id` is preserved if the source provides it; it is optional when absent from the source, with sheet/row and chunk provenance retained where available. A shared workbook-level `source_id` across multiple rows/chunks is expected and is not used as a substitute for their individual Evidence IDs or row identity.

### Citation behavior

The supplied trace diagnostics report that many claims had no citation and a later answer had an `ORPHAN_CITATION`. Existing prompt instructions and parser already use the same canonical marker and fail closed for missing, malformed, unknown, or orphan citations. The parser was not weakened or changed. The shared generation/repair instruction now explicitly asks for a concise answer and requires the marker on the same line as the supported sentence, with no blank-line separation. Regression tests assert that correctly linked Vietnamese citations parse and orphan citations remain invalid.

The raw generated answers from the supplied trace were not available, so it is not possible to decide whether each citation failure was caused by answer formatting or by another output characteristic. Citation integrity remains separate from semantic grounding; no change marks an answer verified.

## Changes made

- `backend/main.py`: retain `document_id`, `chunk_index`, and source-provided `record_id` in Qdrant retrieval candidates. The existing candidate `point_id`, `source_id`, `evidence_id`, domain, sheet, and row values remain intact.
- `backend/tourism_retrieval.py`: retain document/record/point/chunk identity while converting structured tourism matches to candidates.
- `backend/rag/adapter.py`: propagate document, record, point, and chunk IDs to Evidence metadata. For `domain_upload`, no longer infer `source_id` from `document_id` or invent an Evidence ID from a filename/title when the payload is missing the corresponding identity.
- `backend/evidence/builder.py`: reject uploaded Evidence with missing or inconsistent source identity before prompt assembly; detect conflicts in record/chunk identity when duplicate Evidence IDs occur. It does not require a fabricated record ID.
- `backend/rag/reranker_diagnostics.py`: report an absent `record_id` as absent and show the available row/chunk identity instead of implying it is an unknown generated ID.
- `backend/answer/citation_contract.py`: make same-line, non-orphan citation placement and concise output explicit for initial generation and repair.
- Regression tests were added/updated in `backend/tests/test_domain_documents.py`, `test_mock_pipeline_flow.py`, `test_reranker_diagnostics.py`, `test_evidence_builder_integrity.py`, `test_tourism_retrieval.py`, and `test_verifier_structure.py`.

The verifier implementation and thresholds were not changed. Citation parsing remains strict. API schemas, Qwen/LoRA, BGE configuration, and stored data were not changed.

## Validation

Commands executed from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_domain_documents.py backend/tests/test_mock_pipeline_flow.py backend/tests/test_reranker_diagnostics.py backend/tests/test_evidence_builder_integrity.py backend/tests/test_tourism_retrieval.py backend/tests/test_verifier_structure.py backend/tests/test_verification_scorer_contract.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m py_compile backend/main.py backend/rag/adapter.py backend/rag/reranker_diagnostics.py backend/evidence/builder.py backend/tourism_retrieval.py backend/answer/citation_contract.py
```

Results:

- Focused metadata, Evidence, reranking, tourism, citation, and verifier tests: **145 passed**.
- Full backend suite after all changes: **459 passed**, with one Starlette/AnyIO deprecation warning.
- Python compile check: **passed**.
- An initial test invocation via the system `python` could not collect tests because that interpreter lacked project dependencies (`openpyxl`, `sqlalchemy`, and `httpx`). Re-running with the repository `.venv` succeeded. No dependencies were installed.
- One first full-suite run exposed the structured Tourism metadata loss described above; after preserving metadata and adding a regression assertion, the full suite passed.

## Remaining limitations and live retest

- No live Gateway/Core/Qdrant services were running, so the original Shopping request was **not rerun** and no live E2E success is claimed.
- The three supplied Qdrant points' actual payloads, including whether their original data has `record_id`, are not independently verified. The test suite covers both source-provided record IDs and valid source-level identity when the dataset has no record ID.
- The model's raw outputs for the two reported citation failures were not available. The stricter prompt reminder is covered by tests, but its effect on live Qwen output is unmeasured.
- `verified` still requires the configured verifier checks; a syntactically valid citation only establishes citation linkage, not semantic support or legal/product correctness.
- No production data changed. A Core API process reload is needed before the code changes affect a running Core process. Do not start another Model API/Qwen instance. After a controlled Core deployment, submit a new Gateway request for `iPhone 15 có giá bao nhiêu?` with `domain_hint: "shopping"`, inspect the same request's Core trace, and verify payload-to-Evidence IDs, claim links, verifier result, and Gateway response. Re-run only against the intended existing Shopping collection; do not re-index it.

## Manual restart/test instructions

No restart was performed. For a full local runtime test after explicit approval, use the repository's documented commands from the root, first confirming the three API ports are free and that Qdrant/PostgreSQL are in the expected state:

```powershell
docker compose -f backend/docker-compose.yml up -d postgres qdrant
.\.venv\Scripts\python.exe run_gateway_stack.py
```

The stack launcher starts Core, Model API, and Gateway together; it loads one Model API/Qwen instance. Do not launch it if any of those ports are already owned by another process. In another terminal, send the request through the Gateway at `POST http://127.0.0.1:8002/api/v2/legal-query` with JSON body `{"query":"iPhone 15 có giá bao nhiêu?","domain_hint":"shopping","allow_external":false}`. Check HTTP response and trace diagnostics; do not infer success from unit tests or the selected domain alone. If an existing deployment is already running, reload only its Core process using the operator's established process manager, then submit the same request through its existing Gateway; do not create a second Model API.
