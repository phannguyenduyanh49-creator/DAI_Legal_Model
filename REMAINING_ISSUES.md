# Remaining issues

Prioritized after restoring the missing source modules. No issue below is represented as fixed unless supported by a test or runtime observation.

## P0 — Resolve before claiming stack is runnable

### R-001: Core and dependent services still lack live startup proof

- **Evidence (rechecked 2026-10-27):** ports 5173, 8000–8002, 5432, and 6333 were closed. `docker ps` was denied access to the Docker config/named pipe. No service, health probe, or live request ran. Frontend tests/build, safe imports of the restored modules, `pip check`, and backend tests pass, but these do not establish process startup.
- **Impact:** database schemas/data, Qdrant collection compatibility, Qwen base checkpoint + LoRA pairing, and Gateway/Core connectivity are unverified.
- **Next:** run only in an approved environment with existing model process awareness; verify DB/Qdrant health and collections read-only; start/reuse the documented stack, confirm health, then run four Legal HTTP requests and retain redacted traces.
- **Acceptance:** actual service health and each request's domain, catalog/retrieval Evidence, prompt IDs, Qwen attempts, verifier outcome, trace ID and latency recorded.

## P1 — Address before any public demo

### R-002: APIs lack authentication/access control

- **Evidence:** Core, Model API, and Gateway now default to `127.0.0.1`, and the launcher preserves explicit host overrides. The Model API `/generate` and Gateway query endpoint still have no auth dependency visible in code. Core API CORS still uses wildcard origins with credentials. Session identity is a UUID cookie, not user authentication.
- **Impact:** loopback defaults reduce accidental network exposure for local launches, but do not protect explicitly exposed ports, tunnels, reverse proxies, or other callers on a shared host. CORS is not authorization.
- **Next:** decide the deployment boundary and identity provider/access proxy; keep Model API and databases private; add service-to-service authentication without breaking the public API contract; configure narrow CORS and secure cookie settings for HTTPS. Requires deployment review before public exposure.

### R-003: Static development database credentials

- **Evidence:** `backend/docker-compose.yml` and `backend/catalog/database.py` contain `legal_user` / `legal_password` defaults.
- **Impact:** unsafe if reused outside isolated local development.
- **Next:** use generated/local secrets or required environment variables in non-development deployments; rotate any credential that has been used on a reachable service.

### R-004: Domain uploads are shared per domain, not scoped per authenticated user

- **Evidence:** `backend/domain_documents.py` writes to `domain_documents_{domain_id}` and retrieval selects that domain collection; upload endpoints have no user/tenant authorization.
- **Impact:** user-uploaded content is visible to other requests in the same domain. This may be intended for a shared knowledge base but is unsafe to assume private.
- **Next:** product decision: shared domain corpus versus per-user/tenant documents. Add authorization and collection/payload isolation if uploads may contain private data; test cross-user and cross-domain retrieval.

### R-005: No readiness checks in stack launcher

- **Evidence:** `run_gateway_stack.py` delays one second between child launches and checks process exit codes, not HTTP health or model readiness.
- **Impact:** stack may appear started before Qwen or dependent services are ready.
- **Next:** add health readiness polling with a bounded timeout and actionable service-specific errors; test launcher logic with fake child processes and HTTP responders before live use.

## P2 — Required before quality claims

### R-006: No production semantic grounding or calibrated confidence scorer

- **Evidence:** `backend/main.py` constructs `Verifier()` with no scorers; the NLI adapter is experimental and legal benchmark labels are insufficient for deployment claims.
- **Impact:** citation integrity does not prove entailment, contradiction absence, current validity, or legal correctness. Eligible answers may be `answered_unverified` under existing policy.
- **Next:** evaluate a scorer on a reviewed, representative, held-out legal dataset; report confusion matrix, precision/recall/FPR by claim type, latency, and failure behavior; obtain policy review before wiring it into runtime.

### R-007: Legal retrieval and Catalog data quality are not measured

- **Evidence:** static matcher/retrieval code and passing tests exist, but no database/Qdrant data was read during this audit.
- **Impact:** collection existence, freshness, source provenance, retrieval recall, ambiguity and completeness are not established.
- **Next:** run read-only catalog/collection checks and a labeled retrieval evaluation after services are available. Do not import or recreate data without a reviewed idempotency/rollback plan.

### R-008: Clean-install portability is unverified

- **Evidence:** direct runtime dependencies are now listed, and local `pip check` passes; `requirements.txt` pins CUDA-specific PyTorch wheels and no clean installation was run.
- **Impact:** supported OS/GPU combinations and reproducible setup are unknown.
- **Next:** define supported platform/runtime matrix and test a clean disposable environment without replacing the current `.venv` or downloading heavyweight models unnecessarily.

### R-009: Current legal freshness/authority is not validated by the verifier

- **Evidence:** Evidence schema can preserve source metadata, but no live source data/version/validity check ran. Generic NLI is not a legal-currentness validator.
- **Impact:** a semantically supported claim may rely on superseded or wrong-scope text.
- **Next:** retain promulgation/effective dates and jurisdiction metadata from authoritative sources; require a separate legal-currentness policy/evaluation before `verified` semantics are claimed.

### R-010: Runtime readiness and routing recall remain unverified

- **Evidence (Phase 5 code/tests):** `DomainConfig` now owns route rules, enabled state, retrieval strategy, `catalog_enabled`, and the fail-closed Evidence requirement. `/api/v2/domains/` remains backward-compatible and returns enabled IDs; `/api/v2/domains/readiness` reports configured state separately while data/retrieval/model remain `unknown` and E2E remains `not_verified`. The FE labels these unknown states and shows the canonical selected policy domain. Registry routing covers tested legal variants, Tourism, stale hints, and mixed strong signals. Backend tests cover domain-specific empty-Evidence behavior without a model call.
- **Impact:** Keyword routing is still heuristic and only the tested phrases are covered; the score is not calibrated. General and Tourism have no verified base corpus, and `administrative_law` shares the configured Legal Qdrant collection. Readiness does not actively probe dependencies. Production data availability and answerability are unknown.
- **Next:** when services are available, add separately permissioned/read-only runtime probes and real trace checks for retrieval, Evidence, model and verifier stage continuity. Expand routing only against a reviewed set of representative queries. Do not mark any domain ready or E2E verified based on these tests.
