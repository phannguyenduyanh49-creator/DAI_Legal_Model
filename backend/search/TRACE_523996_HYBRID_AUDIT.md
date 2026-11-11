# Audit trace `trace_legal_523996ccd9a24779bef74983a4836fd7`

## What was observed

- Read-only Core root and `/api/v2/search/readiness` returned HTTP 200 with
  `configured_backend=qdrant`, `ready=true`, Qdrant and PostgreSQL available.
  At inspection time, Core PID 28256 had started on 2026-11-10 at 10:26:25.
- Elasticsearch `dai_documents_v1` was available and contained 8,778 documents;
  a read-only `term` count for `domain_id=legal_ai` returned **3,927**.
- The exact trace ID was not found in saved stack logs, benchmark JSONL files
  or local chat history. The user subsequently supplied a diagnostic excerpt
  for this trace. Its original full response and process timestamp remain
  unavailable.
- The excerpt reports `Domain=legal_ai`, `STRICT`, Catalog `no_match`,
  BGE-M3/CUDA embedding (335.95 ms), and Qdrant `legal_docs_bge_m3` HTTP 200.
  Local exact retrieval found one accepted match (reported score 0.7188).
- It lists nine other Qdrant candidate IDs rejected as irrelevant. The
  retrieval stage reported two accepted contexts; the reranking stage selected
  one; Evidence Builder produced one Evidence. Sufficiency was `sufficient`.
- The model generated 241 tokens in 22,318.14 ms. Citation parsing found
  four claims, zero with Evidence, and one allowed Evidence; citation and
  verifier both failed. Final outcome: `answered_unverified`.

## Code path

`backend/core_gateway.py` forwards the request to Core. In
`backend/main.py::search_local_rag`, a domain whose retrieval strategy is
semantic/hybrid enters exactly one configured branch:

1. `SEARCH_BACKEND=hybrid`: encode with BGE-M3, query Qdrant and Elasticsearch
   in parallel through `retrieve_hybrid`, fuse with RRF, then process candidates.
2. `SEARCH_BACKEND=elasticsearch`: BM25 only.
3. Otherwise, with the currently observed `qdrant` config: query Qdrant only.

`find_exact_local_procedures` runs **after** the configured retrieval branch.
In Hybrid mode its separate exact lookup uses the Qdrant local index; it does
not prevent the two Hybrid branches from executing. If an exact match is
selected and `should_use_exact_match_shortcut` allows it, final context
selection skips the BGE reranker and may choose that Qdrant exact item instead
of an RRF candidate. This trace confirms an exact match was found, but
`selected_count=1` does not establish whether the shortcut or BGE reranker
selected it. The shortcut flag is unavailable in this excerpt.

The current Qdrant runtime does not call Elasticsearch or RRF. The supplied
excerpt shows Qdrant but no Elasticsearch or RRF event. This supports
Qdrant-only execution if the trace came from the currently running Core;
the trace timestamp/process identity is unavailable, so an earlier Hybrid
run cannot be excluded from the excerpt alone. Elasticsearch having legal
documents does not establish that this trace queried them.

## Per-trace measurements

| Measurement | Value for requested trace |
| --- | --- |
| Qdrant candidate count | Raw count not given; nine rejected IDs listed and HTTP 200 confirmed |
| Elasticsearch candidate count | Unknown; no ES call shown |
| RRF execution / fused count | Unknown; no RRF event shown |
| Accepted retrieval contexts | 2 |
| Selected after reranking stage | 1; shortcut versus BGE scoring not established |
| Final Evidence count | 1 |

Do not interpret “unknown” as zero. Earlier saved Hybrid traces belong to
**different request IDs** and cannot be used as evidence for this one.

## Telemetry improvement (working tree only)

Core now attaches `retrieval_diagnostics` to the existing pipeline details for
new Hybrid requests. It records each branch's status, error type, candidate
count and duration; RRF status, fused count and cross-store match count;
accepted count, exact lookup count, exact-match shortcut flag, reranked count
and built Evidence count. The Gateway forwards this additive field. When both
branches fail, their status remains available in the failure diagnostics.
No query text or document content is added to the diagnostics.

This code is **not loaded** in the currently running Core. No services were
restarted, thresholds altered, Evidence IDs changed or data modified. Focused
tests passed (27), and the full backend suite passed (496; one existing
Starlette/AnyIO deprecation warning). These tests do not validate this trace.

**Conclusion for this trace: HYBRID NOT VERIFIED.** The excerpt confirms a
Qdrant call, one Evidence and failed citation verification. It does not
confirm an Elasticsearch call or RRF. Core currently reports Qdrant; the
trace's historical process identity remains unknown.
