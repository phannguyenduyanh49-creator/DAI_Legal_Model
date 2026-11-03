# Retrieval and answer quality benchmark protocol

This protocol compares baseline and improved runs on exactly the same, independently reviewed sample set. The runner does not call models and never fills missing results from expected values.

## Gold JSONL

Each row has a unique `sample_id`, `query`, `expected_domain`, and `expected_behavior`; `relevant_evidence` maps immutable source/document IDs to graded relevance (`0` irrelevant, `1` relevant, `2` highly relevant). `expected_answer_points` contains stable point IDs and descriptions; `expected_citations` identifies source IDs supporting those points. Require `review_status: "expert_confirmed"`, at least two independent `reviewers`, an `adjudicator`, and `source_provenance.references` with the exact source title/version/URL or catalog identifier and access date. Synthetic fixtures are for tests only and must not be placed in the production gold set.

## Run result JSONL

Each baseline/improved row uses the same `sample_id` and includes `selected_domain`, ranked deduplicated `retrieved_evidence_ids`, `latency_ms` by stage (retrieval, reranking, generation, total), and `answer_review`. Answer review must be completed by reviewers against the raw answer and exact Evidence supplied to generation: `review_status: "expert_confirmed"`, integer `supported_claims`, `unsupported_claims`, `contradicted_claims`, `cited_claims`, `accurate_citations`, `expected_points_present`, and `expected_points_total`.

## Metrics and interpretation

The runner reports overall and per-domain Recall@K, Precision@K (hits divided by K, so missing slots count as non-relevant), MRR, graded NDCG@K, and domain accuracy. Answer metrics are unavailable unless every answer row has expert-confirmed review. Factual correctness/evidence coverage count supported material claims over all material claims; unsupported claim rate is unsupported claims over all material claims; citation accuracy is accurate cited claims over cited claims; completeness is expected answer points present over expected points total. These operational definitions should remain fixed for paired runs. Latencies report sample count, arithmetic mean, p50, and nearest-rank p95 per available stage; no absent stage is inferred.

## Command

```powershell
python -m evals.quality_benchmark --gold evals/gold_reviewed.jsonl --baseline runs/baseline.jsonl --improved runs/improved.jsonl --top-k 5 --output runs/paired-report.json
```

The command fails closed if gold provenance/review is missing, IDs are not paired exactly, or results are incomplete. Do not interpret a score on fewer than the predeclared review sample size as representative. Freeze dataset version, splits, runtime/config/model revisions, and hardware alongside every run. This repository currently has no gold file meeting the independent review contract; therefore it has no valid quality or baseline/improved score yet.
