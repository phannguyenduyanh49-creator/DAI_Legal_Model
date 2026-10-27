# Confidence and semantic grounding scorers

## Current implementation state

The verifier now has typed `ConfidenceScorer` and `GroundingScorer` interfaces.
No production implementation is configured. The default runtime therefore
reports confidence as `not_verified` when a policy threshold is required and
grounding as `not_evaluated`; the Gateway must continue withholding the answer.
The deterministic scorer objects in unit tests are contract test doubles, not
an evaluated legal model.

The repository contains general transformer/embedding dependencies, but no
configured NLI/entailment checkpoint, confidence model, labeled legal
evaluation set, or calibration artifact. Loading an arbitrary pretrained
classifier or reusing query/domain routing confidence would not be defensible.

## Experimental NLI candidate (not production-wired)

`TransformersNliGroundingScorer` is an opt-in adapter for
[`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).
The model card identifies MIT
licensing, sequence-classification labels `entailment`, `neutral`, and
`contradiction`, 512 token positions, and a 558 MB safetensors file. Its
Vietnamese XNLI accuracy is reported as 0.795. This indicates general
Vietnamese NLI capability, not legal grounding accuracy: its training is
MNLI/XNLI, not Vietnamese legal evidence. The XNLI number must not be reported
as legal verification quality.

The adapter loads only when explicitly constructed. `from_pretrained()` is
cache-only by default and defaults to CPU; it is not wired into the Core API,
does not download weights, and does not claim readiness unless a compatible
checkpoint is already available. It does not truncate a premise/hypothesis
pair beyond the model's position limit. A slow synchronous inference cannot be
interrupted safely mid-kernel; the adapter discards a late result and raises a
timeout, but that timeout does not guarantee immediate cancellation or bounded
latency. An isolated worker process would be needed for hard cancellation.

The adapter maps `entailment` to `supported`, `contradiction` to
`contradicted`, and `neutral` to `unsupported`. Neutral is not called partial
support. This three-way checkpoint cannot reliably distinguish a partially
supported composite claim from an unrelated/insufficient premise. Its softmax
is not a calibrated confidence score. Contradiction from any cited item wins;
otherwise a complete entailment from at least one cited item supports the
claim, and only-neutral evidence is unsupported. Each cited item is evaluated
as a separate premise; uncited Evidence is never inspected. Composite-claim
decomposition and conflicting-source adjudication remain limitations.
The three-way adapter also cannot distinguish a reviewer-labeled
`insufficient_evidence` case from an evaluated `unsupported` case. The
`GroundingAssessment` contract accepts `insufficient_evidence` and
`inconclusive` as distinct outcomes for scorers that can identify them; this
NLI adapter does not make that distinction and remains experimental.

Other candidates reviewed:

- [`ntphuc149/ViLegalBERT`](https://huggingface.co/ntphuc149/ViLegalBERT) is pretrained for Vietnamese legal text and reports
  legal benchmark results including VLSP-NLI, but its published checkpoint is
  a 135M masked-language-model encoder with a 256-token limit, not a ready NLI
  classification head. Its model card lists AGPL-3.0. It would need a trained
  legal NLI head and license review before integration.
- [ViLegalNLI](https://arxiv.org/abs/2605.00116) is a legal-domain dataset of 42,012 premise/hypothesis pairs from
  official statutory material. Its paper describes semi-automatic generation
  and binary entailment/non-entailment labels; that label set alone does not
  separate contradiction from neutral. A vetted checkpoint was not verified
  for this repository.
- The existing [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) scores query/passage relevance, not claim-to-Evidence
  entailment. Its score cannot substitute for grounding or calibrated
  confidence.

No weights were downloaded. On 2026-10-27 the existing local checkpoint was
loaded cache-only in a separate evaluation process; it remains unwired from
the production verifier. Adapter unit tests use an in-memory fake classifier;
they verify inputs, label mapping and fail-closed behavior, not model accuracy.

## Reproducible offline candidate benchmark

Run the NLI candidate against a frozen JSONL dataset. The command below uses
the already-cached checkpoint, selects CUDA explicitly, and writes outputs to
the operating-system temporary directory:

```powershell
.\.venv\Scripts\python.exe -m backend.verifier.evaluation.run_nli_benchmark `
  backend/verifier/evaluation/catalog_candidates.jsonl `
  --predictions (Join-Path $env:TEMP 'legal_ai_nli_candidate_predictions.jsonl') `
  --report (Join-Path $env:TEMP 'legal_ai_nli_candidate_report.json') `
  --device cuda
```

The runner uses `local_files_only=True`, refuses input truncation, records the
model revision, hardware/runtime, model-load time, evaluation-loop time,
per-pair inference latency, claim/Evidence pair counts, host RSS snapshots and
CUDA allocated/peak memory. Per-pair inference latency excludes tokenization;
evaluation-loop time includes tokenization and inference but excludes model
loading. Precision, recall, one-vs-rest false-positive rate, F1 and confusion
matrix are computed only from `expert_confirmed` labels. Without such labels,
the report explicitly returns `metrics unavailable` and a zero reviewed count.

### 2026-10-27 cache-only run

The available Catalog candidate file has 80 rows from 80 distinct source
records; every row is `gold_label=""` and `review_status="unreviewed"`. The
claim text is copied from a Catalog source field, so these candidates still
need expert review and do not yet represent independently authored legal
claims. No accuracy metric, precision, recall, false-positive rate, or
confusion matrix can be computed from them.

The cached checkpoint revision was
`8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`. On Python 3.12.10, PyTorch
2.11.0+cu128, 16 logical CPUs and an NVIDIA GeForce RTX 3060 (12 GiB), the
single run loaded the model in 2.43 s and processed 80 one-claim/one-Evidence
pairs in 2.08 s. It completed inference for 49 pairs and skipped 31 pairs
whose combined input exceeded the model limit rather than truncating Evidence.
For the 49 completed pairs, inference-only latency was 40.2 ms mean, 23.1 ms
median, and 141.5 ms p95. The run reported 539 MiB of model CUDA allocation
and 600 MiB peak CUDA allocation. Host process RSS was 707 MiB before load,
1,074 MiB after model load, and 1,814 MiB after evaluation; only 2.54 GiB of
host memory was available before loading. These are one-run resource
measurements, not a service-level latency guarantee. The CPU-only attempt was
stopped after approximately 86 seconds before it completed; its partial work
is not included in the reported benchmark.

The result is a runtime/input-coverage smoke test only. The reviewed sample
count is zero, so the candidate remains unevaluated for legal claim-grounding
quality. A proper comparison needs at least two qualified reviewers, an
adjudicated held-out set, and splits grouped by source record, legal
instrument/version and time. Run every candidate on the same immutable split;
do not tune thresholds on the final test split. Evaluator-model and
deterministic-rule baselines are not configured in this repository, so no
same-example accuracy comparison is currently available.

## Intended confidence meaning

A production confidence score must estimate the probability that the complete
answer satisfies a documented answer-level target: its substantive claims are
accurate for the applicable domain and supported by the exact evidence used.
It must be produced by a versioned scorer from answer/evidence signals, never
from a model's self-reported confidence. The score is a calibrated estimate,
not a proof of correctness. Citation-ID integrity and semantic entailment
remain separate gates.

The current domain thresholds are policy configuration only; this repository
does not demonstrate that they are calibrated:

| Domain | Configured threshold | Calibration status |
| --- | ---: | --- |
| `legal_ai` | 0.85 | Not evaluated |
| `general_assistant` | 0.50 | Not evaluated |
| `administrative_law` | 0.80 | Not evaluated |
| No fourth registered domain | — | — |

The `0.90` value in `domain_registry.py` is only used by a validation example
for an intentionally incomplete configuration; it is not an active domain.

Each domain needs representative, human-labeled answer/evidence examples. Split
by source/document and time to avoid leakage. Report sample size, Brier score,
reliability diagrams/expected calibration error, discrimination, and
risk-versus-coverage at the configured threshold. Review false-accept rates
separately for legal subdomains and high-impact claims. Threshold changes need
versioned evaluation results and policy approval; thresholds must not be
lowered just to increase acceptance.

## Intended grounding behavior

For every parsed substantive claim, the grounding adapter receives only the
non-empty Evidence objects referenced by that claim's valid citations. The
production contract distinguishes `supported`, `unsupported`, `contradicted`,
`partially_supported`, `insufficient_evidence`, `inconclusive`, and
`evaluator_unavailable`. Provider exceptions, timeouts, and malformed outputs
are represented through scorer status and do not become successful grounding.
The aggregate passes only when every claim is `supported`. Missing/invalid
citations prevent scorer invocation.

Any future entailment model must be tested on manually adjudicated legal
claim/evidence pairs including negation, exceptions, dates, jurisdiction,
partial support, and contradiction. A generic NLI model is not automatically
valid for Vietnamese legal text. Record per-label precision/recall/F1,
confusion matrix, calibration, subgroup results, model/checkpoint hash, and
known limitations. Human review remains necessary before production claims.

The evaluation record format and synthetic examples are in
`backend/verifier/evaluation/`. The fixture has six cases and is not a
manually labeled evaluation dataset; production precision, recall, F1, and
calibration metrics are unavailable.

## Timeout and failure contract

Adapters receive `timeout_seconds` and must enforce it for their provider or
inference call. The verifier catches surfaced `TimeoutError`, unavailable
provider errors, other exceptions, and malformed output. It emits stable
metadata without propagating provider internals or inventing a score. No
scorer failure can produce `passed`.
