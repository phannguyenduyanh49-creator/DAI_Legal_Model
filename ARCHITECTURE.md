# DAI-Legal-Model architecture (code inspection)

**Runtime status:** this describes code ownership and intended request flow. Services were not running during this audit, so arrows are not live integration claims.

## Request flow

```mermaid
flowchart TD
    U[User] --> FE[React/Vite FE]
    FE -->|POST query, domain_hint, permission| GW[Core Gateway :8002]
    GW --> QA[Query analysis]
    QA --> OR[Registry-driven Domain Router / Planner]
    OR --> POL[Domain Registry + MCP policy]
    POL -->|policy_domain, allowlist, request ID| CORE[Core API :8000]
    CORE --> CAT[Direct Catalog / PostgreSQL]
    CORE --> EMB[BGE-M3 query embedding]
    EMB --> QD[Qdrant collection(s) for domain]
    QD --> REL[Relevance filter + exact match + reranker]
    CAT --> EB[Evidence Builder]
    REL --> EB
    EB --> SUF[Data sufficiency / external-source policy]
    SUF -->|approved when needed| MCP[MCP stdio client/server / DVC]
    MCP --> EB
    EB --> PA[Prompt assembly + Evidence ID allowlist]
    PA -->|prompt| MODEL[Model API :8001 / Qwen + LoRA]
    MODEL --> CIT[Claim and citation parsing]
    CIT --> VER[Verifier: citation checks; scorer interfaces]
    VER --> OUT[Core policy outcome]
    OUT --> GW
    GW --> FE
```

## Module responsibilities

| Layer | Main files | Responsibility / boundary |
|---|---|---|
| Frontend | `FE/src/App.jsx`, `FE/src/api.js`, `FE/src/telemetry.js`, `FE/src/answerPresentation.js` | Chat state, domain/document controls, Gateway contract validation, presentation of source/outcome/diagnostics. |
| Gateway | `backend/core_gateway.py` | Request schema, analysis/router, registry policy enforcement, Core HTTP call, response validation/outcome/diagnostics. |
| Query/routing | `backend/query_analysis.py`, `backend/orchestrator.py`, `backend/domain_registry.py`, `backend/policy/middleware.py` | Query normalization/analysis, keyword routing and hint handling, domain config and MCP allowlist narrowing. |
| Core pipeline | `backend/main.py`, `backend/chat_session.py`, `backend/rag/pipeline_policy.py` | Session permission continuity, catalog/RAG/MCP sequence, sufficiency, prompt/model/verifier orchestration and SSE response. |
| Catalog | `backend/catalog/{database,models,schemas,repository,matcher,service}.py`, `backend/catalog/query_parser.py` | PostgreSQL schema and lookup, title/code parsing and matching. Live DB contents are not part of this code map. |
| Retrieval | `backend/main.py`, `backend/domain_documents.py`, `backend/rag/{adapter,relevance,diagnostics,data_sufficiency}.py` | Embedding, domain collection selection, Qdrant result filtering, source conversion, upload/chunking, retrieval and sufficiency diagnostics. |
| Evidence/prompt | `backend/evidence/{schemas,builder,normalizer}.py`, `backend/answer/{citation_contract,evidence_gate}.py`, `backend/prompt/assembler.py` | Normalize and deduplicate source evidence, preflight missing prompt Evidence, select Evidence under prompt budget, and format allowed citations. |
| Model | `backend/model_api.py`, `backend/answer/model_api_client.py` | Process-start Qwen/LoRA loading and generation API/client. Model quality/load state requires runtime proof. |
| Citation/verifier | `backend/answer/{citations,citation_generation,reference_policy}.py`, `backend/verifier/{verifier,scoring,transformers_nli}.py` | Parse claim-citation links, retry citation formatting, separate integrity/grounding/confidence states, and choose conservative response outcome. No production scorer is configured. |
| MCP | `backend/mcp/{client,contract,runtime,adapter,dvc_search}.py`, `mcp_server.py` | Per-call stdio lifecycle, caller/server allowlists, timeouts, DVC/local search and source conversion. External behavior was not live-tested. |
| Deployment/data | `run_gateway_stack.py`, `backend/docker-compose.yml`, `scripts/`, `data/`, `outputs/` | Process launcher, local PostgreSQL/Qdrant definitions, import/training artifacts. No migration or data import was run. |

## Trust boundaries and caveats

- Domain-specific Qdrant collections prevent ordinary cross-domain vector mixing in the collection-selection path. They do not provide per-user privacy: uploads are stored in shared domain collections and the current API has no authenticated tenant identity.
- MCP is an external retrieval/tool path, not a vector database. Core policy intersects requested tools with the registry allowlist; user permission is carried separately for external lookup.
- Prompt citation IDs are drawn from the prompt-selected Evidence set, and Core constructs a verifier bundle from those selected objects. This establishes a code-level data path, not semantic support.
- `Verifier()` has no production grounding or confidence scorer. `answered_unverified` can be an allowed product outcome; it is never evidence of `verified`.
- Gateway/Core/Model ports and database availability, actual collection payloads, source freshness, model readiness, user isolation, and E2E behavior remain unverified.

## Multi-domain implementation inventory

The table separates code configuration from real data availability and execution. `NOT_VERIFIED` means no live service/data check was possible in the current environment.

| Public domain ID | Registry and routing | Configured source / retrieval | Evidence, prompt and model path | Runtime acceptance |
|---|---|---|---|---|
| `legal_ai` | Registered. Legal route aliases such as `enterprise_law` map to this policy; rules now reside in Registry. | `catalog_enabled=true`; `legal_docs_bge_m3` and `domain_documents_legal_ai` are configured targets. Actual DB/collection contents are not verified. | Shared Core Evidence Builder, sufficiency gate, prompt assembly, citation checks and Model API. Production semantic grounding scorer is not configured. | Unit/contract coverage; live retrieval, model and E2E `NOT_VERIFIED`. |
| `administrative_law` | Registered. Administrative routing rule lives in Registry and maps to this policy. | `catalog_enabled=true`; uses configured `legal_docs_bge_m3` plus `domain_documents_administrative_law`. No live corpus/data was verified. | Same shared Core pipeline; policy allows procedure/DVC tools. | Unit/contract coverage; live retrieval and E2E `NOT_VERIFIED`. |
| `tourism` | Registered. Auto-routing now recognizes a bounded set of travel terms; explicit hint also routes here. | No base collection configured. Only `domain_documents_tourism` is eligible, populated through `.txt`/`.md` upload. No production tourism corpus was verified; the repository fixture is test-only. | Same Evidence/prompt/model code; MCP allowlist is empty. | Routing and mock upload/retrieval mapping tests pass; live source, generation and E2E `NOT_VERIFIED`. |
| `general_assistant` | Registered and selectable by explicit hint. Automatic no-keyword fallback now selects General; explicit legal keywords still route through the legal aliases. | No base collection; only its isolated upload collection could supply Evidence. No general corpus or independent source was verified. | Shared Evidence gate remains in force, so an empty source set blocks generation. Unimplemented weather/map tools were removed from its declared MCP allowlist. | Auto-routing and empty-collection contract tests pass; source coverage and E2E `NOT_VERIFIED`. |

`DomainConfig` now owns enabled state, explicit Catalog eligibility, a mandatory Evidence requirement, and route rules (route ID, category, strong/regular terms, exclusions, priority). `DomainRouterPlanner` scores those Registry rules and treats competing strong Legal/Tourism signals as ambiguous. A stale hint is overridden by clear content signals; unmatched queries use the configured General fallback. Scores are heuristic ranks, not calibrated probabilities. The legacy `/api/v2/domains/` response remains a list of enabled IDs; additive `/api/v2/domains/readiness` reports registration/configuration separately from unprobed data, retrieval, and model state, and never marks E2E verified from unit tests. The Frontend consumes that readiness response, labels unverified domains, and shows the canonical `domain_policy.domain_id` returned for each completed answer. Catalog eligibility no longer depends on collection-name equality. General and Tourism remain Evidence-gated; no source evidence means no model call. These changes establish code and contract behavior only: service health, actual data, and live E2E remain unverified.

## Phase 5 routing and readiness changes

- Routing rules are declared in `backend/domain_registry.py`; policy alias resolution uses the same Registry instead of a second hard-coded alias map.
- Legal rule coverage includes general Legal phrases and bounded English equivalents. Overlapping strong Legal and Tourism signals produce `needs_clarification` before Core/model invocation.
- `catalog_enabled` is an explicit per-domain setting used by Core. Evidence remains required for every domain. Domain disablement is reflected by routing, retrieval collection selection, upload checks, and readiness status.
- `/api/v2/domains/readiness` returns registered, enabled, configured, data, retrieval, model, E2E, and response-policy fields. Runtime fields stay unknown/not_verified until separately probed.
- FE shows readiness labels and reason text, preserves the existing enabled-ID API, and displays the canonical domain used in each response.
- General/Tourism missing-Evidence responses are domain-specific while retaining the same fail-closed preflight and no-model-call behavior.
