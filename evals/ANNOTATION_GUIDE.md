# Human retrieval annotation protocol

`golden_candidates.jsonl` is an immutable candidate queue, **not gold**. Its domain proposals and `auto_candidate_judgments` are discovery metadata. Do not give that file, preliminary retrieval scores, model outputs, or another annotator's decisions to annotators.

## Prepare a blinded, shared document pool

Before annotation, collect the union of:

1. The source-linked seed candidate for each query.
2. The top 10 unique documents from Qdrant.
3. The top 10 unique documents from Elasticsearch BM25.
4. The top 10 unique documents from Elasticsearch BM25 + BGE Reranker.

Use actual `document_key` values from the results/PG store; deduplicate the pool by `document_key`, preserving `record_id`, `document_id`, `source_id`, and source-content hash. Give both annotators the same pool, but hide retrieval scores, rank, system/config identity, proposed expected domain/type, and automatic relevance suggestions. Keep the pool-to-query manifest with the coordinator. The current runtime cannot yet produce the Qdrant and reranker pools, so annotation on the current queue would be incomplete for comparative evaluation.

Each reviewer gets a separate file and must not see the other review. Review query intent/domain first, then inspect each pooled source. To inspect a record directly from PostgreSQL, use:

```powershell
python -m evals.inspect_candidate --document-key <document-key>
```

Record stable annotator IDs, UTC ISO-8601 timestamps, exact source identity, and a short reason for each judgment. A label without an inspected source and provenance is invalid.

## Review sidecar format

Create two JSONL files, such as `reviews_a.jsonl` and `reviews_b.jsonl`. Each file has one row per query and does not include auto suggestions. Example shape (replace all placeholders with actual values):

```json
{
  "query_id": "Q-...",
  "annotator_id": "reviewer-identifier",
  "annotated_at": "2026-11-03T10:15:00Z",
  "review_status": "submitted",
  "query_status": "answerable",
  "expected_domain": "domain-id-decided-by-reviewer",
  "expected_record_type": "record-type-decided-by-reviewer",
  "document_judgments": [
    {
      "document_key": "actual-key-from-the-blinded-pool",
      "domain_id": "actual-domain-id-from-postgres",
      "document_id": "actual-document-id",
      "record_id": "actual-record-id-or-null",
      "source_id": "actual-source-id",
      "content_sha256": "sha256-of-current-postgres-content",
      "source_checked_at": "2026-11-03T10:10:00Z",
      "state": "relevant",
      "label": 2,
      "note": "Explain which inspected source content answers the query."
    }
  ],
  "notes": "Explain the domain/type decision and any ambiguity."
}
```

Valid states are `relevant` (label 2), `partially_relevant` (label 1), `irrelevant` (label 0), and `insufficient_information` (label `null`). Insufficient information is a distinct state; never encode it as irrelevant/0. When the query itself cannot be answered from the corpus, set `query_status` to `insufficient_information` and explain the missing source or field. `query_status` may also be `answerable` or `invalid_query`.

## Adjudication sidecar

After both independent reviews are locked, a third person writes `adjudications.jsonl`. The adjudicator must have a different ID, inspect disagreements against the original PostgreSQL records, and include the complete same document pool:

```json
{
  "query_id": "Q-...",
  "adjudicator_id": "adjudicator-identifier",
  "adjudicated_at": "2026-11-03T12:00:00Z",
  "adjudication_status": "approved",
  "query": "Final query text after review",
  "query_status": "answerable",
  "expected_domain": "domain-id-confirmed-by-adjudicator",
  "expected_record_type": "record-type-confirmed-by-adjudicator",
  "document_judgments": [],
  "expected_answer_points": [],
  "notes": "Explain disagreements, evidence reviewed, and final resolution."
}
```

Use `adjudication_status: "disputed"` when the evidence does not resolve a disagreement. Disputed or missing adjudications block the entire official export. An adjudicator must not be either reviewer.

## Source validation and gold export

Check the current queue's source references without annotations:

```powershell
python -m evals.validate_annotations --audit-only
```

After there are two reviews and one approved adjudication for every query candidate, validate all IDs and content hashes against PostgreSQL and create the official file:

```powershell
python -m evals.validate_annotations --reviews evals/annotations/reviews_a.jsonl --adjudications evals/annotations/adjudications.jsonl --output evals/gold_reviewed.jsonl
```

The exporter fails closed on absent/duplicate reviewers, incomplete or unequal document pools, invalid label/state pairs, missing notes/timestamps, disputed decisions, missing PostgreSQL records, ID/domain mismatches, or changed content hashes. It produces no output gold file unless the full eligible query set passes. Food's five blocked slots are excluded until actual Food source records are ingested and validated; they must not be fabricated.

The exporter derives `source_provenance.references` from the exact PostgreSQL rows judged, including actual IDs, title/URL where present, content hash, and source-check time. It exports only adjudicated human labels; `auto_candidate_judgments` is never read as relevance truth.

## Domain ambiguity and simulated datasets

Registry currently maps both `legal_ai` and `administrative_law` to the same 3,927-row Catalog. Reviewers must choose the expected domain from the query contract and source evidence, not copy the candidate's proposed domain. If the source set cannot distinguish the expected domain, mark the query invalid or insufficient for domain evaluation.

Healthcare HC-F records and the current Healthcare/Education/Shopping uploads include explicitly simulated values. Preserve those warnings during review; they may support retrieval tests of the stored records, but must not be described as real providers, courses, prices, or inventory.

## Benchmark and threshold policy

Use only `gold_reviewed.jsonl` for official retrieval metrics. Run the same approved sample IDs and pooled sources through Qdrant, Elasticsearch BM25, and Elasticsearch BM25 + BGE Reranker. Compare Qdrant vs BM25 and BM25 vs reranked outputs with the paired runner described in `QUALITY_BENCHMARK.md`. Do not calculate official metrics from the candidate queue or preliminary BM25 output.

Calibrate `ELASTICSEARCH_MIN_SCORE` only on a predeclared development split. Report precision, recall, empty-result rate, and false-positive rate for each threshold, freeze the selected value, then evaluate the held-out test split once. Do not choose a threshold just to reduce empty results.
