# DAI Legal Model – Expanded Benchmark Dataset Report

**Report type:** Dataset and benchmark-harness readiness report  
**Status:** Dataset and test runner prepared; live API execution not performed.

## 1. Objective

Expand the DAI Legal Model benchmark coverage with a broader Vietnamese legal-question dataset and a repeatable API test harness. The dataset is intended to exercise common legal-information and retrieval scenarios across multiple domains.

## 2. Dataset inventory

- Total test questions: **100**
- Categories: **10**
- Planned repetitions: **1–3 per question**
- Maximum planned API requests at 3 repetitions: **300**

| Category | Questions |
|---|---:|
| Administrative procedures | 15 |
| Labor law | 14 |
| Civil law | 15 |
| Criminal law | 10 |
| Tax law | 10 |
| Land and housing | 10 |
| Intellectual property | 5 |
| Commercial law | 5 |
| Administrative law | 5 |
| Legal retrieval and RAG quality | 11 |
| **Total** | **100** |

## 3. Benchmark coverage

The question set covers:
- Legal procedure and document requirements.
- Employment contracts, termination, wages, and social insurance.
- Civil transactions, contracts, inheritance, and liability.
- Criminal-law concepts and sentencing principles.
- Tax declaration, invoices, tax payment, and refunds.
- Land-use rights, housing, and property transactions.
- Copyright, trademarks, and software licensing.
- Commercial contracts, arbitration, and force majeure.
- Administrative complaints and sanctions.
- Legal-source retrieval, effective-date checks, citations, ambiguous questions, and out-of-scope handling.

## 4. Metrics supported by the runner

The live API runner is designed to record:
- HTTP status and request success/failure.
- Per-request response latency.
- Mean latency, P50, and P95 over successful requests.
- Success rate and failed-request count.
- Mean latency grouped by question category.
- A limited response preview for troubleshooting.

## 5. Execution status

| Item | Status |
|---|---|
| 100-question dataset | Prepared |
| 10-category classification | Prepared |
| Python API benchmark runner | Prepared |
| JSON/CSV/Markdown output generation | Implemented |
| Live API requests | Not executed |
| Actual latency statistics | Not available |
| Legal-answer accuracy evaluation | Not performed |

**Important:** No API calls were run for this report. Therefore, this report does not claim any measured latency, throughput, success rate, or legal-answer accuracy. The counts above describe the prepared test dataset and harness only.

## 6. Planned execution

When the Core API environment is available:
1. Confirm the API endpoint and request schema.
2. Run one request per question (100 requests) as a smoke/performance pass.
3. Review failures and request payload compatibility.
4. If stable and authorized, repeat three times per question (up to 300 requests).
5. Review latency distribution and error categories.
6. Separately evaluate answer correctness and evidence grounding using a reviewed gold-standard set.

## 7. Reproducibility and output files

The accompanying benchmark package contains:
- `benchmark_questions.json`
- `benchmark_live.py`
- `README.md`

After a successful live run, the runner generates:
- `benchmark_live_results.json`
- `benchmark_live_results.csv`
- `BENCHMARK_LIVE_REPORT.md`
