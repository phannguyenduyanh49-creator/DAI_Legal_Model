# Implementation report

**Date:** 2026-10-27  
**Scope:** Restore P0 code integrity from existing repository history, correct documentation and declare direct test/runtime dependencies. No database, model, Qdrant, or deployment changes.

## Work completed

### Restored missing modules without overwriting existing files

The prior audit found imports that could not resolve. They were not in `HEAD` or the current working tree, but the exact files existed in the untracked-files parent of the existing `backup-before-sync-main` stash. I verified that each destination was absent, extracted only those nine paths from `stash@{1}^3`, and verified each restored file hash against its source blob. The stash was not popped, dropped, or modified. An initial `git restore` attempt could not create `.git/index.lock` because `.git` is read-only; no checkout/restore operation changed the index.

Restored:

- `FE/src/answerPresentation.js`
- `backend/answer/citation_contract.py`
- `backend/answer/evidence_gate.py`
- `backend/catalog/query_parser.py`
- `backend/chat_session.py`
- `backend/domain_documents.py`
- `backend/mcp/client.py`
- `backend/rag/diagnostics.py`
- `backend/rag/relevance.py`

These are the previously authored implementations, including the text upload checks, per-domain Qdrant collections, MCP caller authorization, query parser, and user-facing status/source presentation. They were not recreated as placeholders.

### Documentation and dependency manifest

- `README.md`: resolved conflict markers by retaining the dataset/setup and API material coherently; removed an obsolete dataset command from the frontend start section; corrected verifier status claims; labeled internet deployment as documented but not verified; documented Gateway domain query; added PowerShell-compatible `npm.cmd` commands and the launcher readiness limitation.
- `requirements.txt`: declared installed direct runtime imports `FlagEmbedding==1.4.2`, `SQLAlchemy==2.1.4`, and `psycopg2-binary==2.9.13`.
- `requirements-dev.txt`: added the installed test runner `pytest==9.1.1` separately from runtime requirements.
- `AUDIT_REPORT.md`: retained the initial audit snapshot and appended the post-restoration results.
- Added `ARCHITECTURE.md`, `TEST_REPORT.md`, and `REMAINING_ISSUES.md` for the requested implementation/status handoff.

No new scorer, model, threshold, API request field, database schema, or retrieval policy was introduced. No packages were installed or downloaded.

### Local service bind defaults

- Core API, Model API, and Gateway now default to loopback when launched directly or with `run_gateway_stack.py`; explicit `*_HOST` overrides remain available.
- Core's model endpoint now defaults to the loopback Model API and follows `MODEL_API_PORT`, with `MODEL_API_URL` available for an explicit deployment URL.
- Added launcher regression tests for default binds and preserved overrides. This limits accidental network exposure for local runs; it does not add authentication or make an explicitly exposed deployment safe.

### Multi-domain contract fixes

- `DomainRegistryService.register_domain` now rejects duplicate IDs instead of silently replacing a configuration; `update_domain_policy` normalizes IDs consistently before writing them back.
- `DomainRouterPlanner` now recognizes a bounded set of tourism terms, allowing automatic Tourism selection when those terms occur. Retrieval still only targets the Tourism collection; no shared/legal fallback was added.
- Unmatched/no-keyword requests now select `general_assistant` instead of internal `general_legal` (which mapped them to Legal policy). The Evidence gate remains unchanged; General requests with no configured/uploaded Evidence will be withheld. Legal questions outside the keyword map may need a domain hint, which is a recall limitation requiring broader routing evaluation.
- Removed General Assistant MCP tools that do not exist in the runtime MCP contract. The policy boundary already filtered them, so this aligns declared configuration with effective behavior.
- Added five regression tests for duplicate IDs, normalized policy updates, unsupported tool declarations, auto Tourism routing to only its own eligible collection, and General fallback isolation. Updated a Gateway integration contract test for the expected General policy.

These fixes do not add domain datasets, domain-specific scorers, data readiness status, or live service behavior. The registry/keyword split remains a documented architecture gap.

## Verification

After restoring the modules, backend tests completed successfully: **279 passed, 1 existing Starlette deprecation warning**. After the bind configuration change, the full backend suite completed with **281 passed, 1 existing Starlette deprecation warning**. After the multi-domain fixes, focused tests completed **35 passed** and full suite **286 passed**, each with the same existing warning. Frontend tests: **18 passed**. Frontend production build: **passed** to a temporary directory outside the repository. `pip check`: no broken requirements.

See [TEST_REPORT.md](TEST_REPORT.md) for exact commands, initial failures, and tests not run.

## Remaining limits

No services were running, Docker control was denied, and no live HTTP request was made. Legal retrieval against current PostgreSQL/Qdrant data, Qwen model loading/inference, MCP external calls, authentication/security controls, semantic legal grounding quality, and production readiness remain unverified. See [REMAINING_ISSUES.md](REMAINING_ISSUES.md).

## Phase 5 — Domain routing, readiness and General Assistant

### Implemented

- Added Registry-owned `DomainRoutingRule` configuration with route IDs, legal/tourism category, regular and strong phrases, exclusions and priority. The Registry rejects duplicate route aliases and aliases that conflict with registered domain IDs. Disabled domains are excluded from the enabled list, route iteration, collection selection and document upload.
- Replaced the Orchestrator keyword map with scoring over Registry rules. Scores are heuristic ranks, not probabilities. Clear content can override a stale domain hint; competing strong Legal/Tourism evidence asks for clarification before Core/model invocation. Added general Legal phrases and bounded English examples to address tested legal wording outside the previous rule list.
- Added explicit `enabled`, `catalog_enabled`, and non-disableable `evidence_required` fields. Core Catalog selection now uses `catalog_enabled`, not collection-name equality.
- Added backward-compatible `/api/v2/domains/readiness`; existing `/api/v2/domains/` still returns enabled domain ID strings. The new contract distinguishes registry/configuration from data, retrieval, model and E2E state. Unprobed dependencies are `unknown`; E2E remains `not_verified`.
- Connected the Frontend to readiness state, labels domains as unverified when appropriate, shows readiness reason text, and displays the canonical selected `domain_policy.domain_id` for responses. Corrected the missing `ChevronUp` import on the existing collapsible control.
- General and Tourism no-Evidence messages are domain-specific. The common Evidence preflight remains fail-closed; it does not call the model without usable prompt Evidence.

### Regression coverage and results

- Added registry/router tests for legal paraphrase examples, normal business queries, stale Legal/Tourism hints, mixed strong-signal clarification, disabled domain filtering, explicit Legal retrieval targets, and readiness states.
- Added mocked Gateway test that verifies ambiguity returns `needs_clarification` without a Core request; added API test for the additive readiness endpoint and stable legacy list.
- Added fail-closed General/Tourism Evidence tests and Frontend readiness contract tests.
- Final focused backend run: **55 passed**; full backend suite: **302 passed**, each with 1 existing Starlette deprecation warning. Frontend: **20 passed**. Production build: **passed**, 1,865 modules transformed. See `TEST_REPORT.md` for commands.

### Limits

No runtime services were started, no live request ran, and no browser test ran. Readiness data/retrieval/model values are not actively probed; no domain is declared production-ready or E2E verified. Keyword routing still needs evaluation on a reviewed representative query set, and legal semantic grounding remains unconfigured in production. These changes do not alter verifier thresholds or remove the Evidence gate.
