# Test report

**Date:** 2026-10-27. Test results below are from the current local checkout after restoring the nine previously authored missing modules.

## Results

| Area | Command | Result | Test type / notes |
|---|---|---|---|
| Backend full suite (initial after restore) | `\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider` | **279 passed**, 1 warning, 0 failed | Unit, contract, and mocked integration tests. Warning: Starlette `anyio.abc.BlockingPortal` deprecation. |
| Backend full suite (final) | `\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider` | **281 passed**, 1 warning, 0 failed | Includes the two new launcher bind regression tests. Same Starlette deprecation warning. |
| Multi-domain focused tests (final) | `\.venv\Scripts\python.exe -m pytest backend/tests/test_domain_registry_contract.py backend/tests/test_policy_mcp_integration.py backend/tests/test_domain_documents.py backend/tests/test_gateway_integration.py -q -p no:cacheprovider` | **35 passed**, 1 warning, 0 failed | Registry, routing, domain collection mapping, mocked Gateway/API and MCP policy contracts. |
| Backend full suite (multi-domain changes) | `\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider` | **286 passed**, 1 warning, 0 failed | Includes five added registry/routing regressions; one existing Gateway test now asserts General fallback. |
| Frontend | `npm.cmd test` from `FE/` | **18 passed**, 0 failed | Node tests for API transport and telemetry; no browser rendering. |
| Frontend production build | `npm.cmd run build -- --outDir C:\Users\Admin\AppData\Local\Temp\dai-legal-audit-build-restored-20261027` from `FE/` | **Passed**, 1,865 modules transformed | Vite build output was outside the repository; existing `FE/dist-old` was untouched. |
| Python dependency consistency | `\.venv\Scripts\python.exe -m pip check` | **No broken requirements found** | Checks installed environment, not a clean install. |
| Diff formatting | `git diff --check` | **Passed** | Git printed line-ending conversion warnings for preexisting CRLF/LF files. |
| Launcher bind configuration | `\.venv\Scripts\python.exe -m pytest backend/tests/test_citation_debug_configuration.py -q -p no:cacheprovider` | **4 passed** | Verifies loopback defaults and that explicit bind overrides are preserved; does not start services. |
| README conflict markers | `rg` marker scan | **None in source/docs outside historical `integration_staging` snapshot** | README Markdown fence count: 68, balanced. |

## Baseline failures and recovery evidence

Before restoring files, `python -m pytest backend/tests -q -p no:cacheprovider` stopped during collection with eight import errors; `import backend.main` failed first at missing `backend.catalog.query_parser`. The FE build failed to resolve `./answerPresentation`. A selected subset had three `test_dvc_search.py` failures because `backend.rag.relevance` was missing. These modules were found in the existing stash's untracked-files tree and restored only to previously absent destinations; each restored file's Git blob hash matched the stash source.

The initial PowerShell command `npm test` was blocked because execution policy rejects `npm.ps1`; `npm.cmd test` ran successfully.

The initial `import backend.main` probe failed before module restoration at `backend.catalog.query_parser`. A second direct import was not attempted: `backend.main` invokes `init_chat_database()` during import and constructs embedding/reranker models, which would write local chat data and may initialize heavyweight model resources. The restored modules are covered by the passing backend suite, but this does not substitute for a real service startup/readiness check.

## Not run

- Live Gateway → Core → Qdrant/PostgreSQL → Model API end-to-end requests, including the four Legal questions.
- Browser/UI rendering tests.
- Docker/container health tests: `docker ps` was denied access to the Docker named pipe/config.
- Qwen load/inference, GPU/VRAM checks, and model latency measurements. No model process was started.
- External MCP/DVC calls or permission-denial HTTP tests against a running stack.
- Clean environment dependency installation. No package installation/download was performed.
- Legal correctness, retrieval-quality, grounding, or confidence benchmark with expert-labeled examples.

Therefore these results establish code/test/build status only. They do not establish service readiness, live E2E success, or legal answer accuracy.

## Phase 5 domain routing/readiness verification — 2026-10-27

### Changes covered by tests

- Registry-owned route rules are exercised for enterprise formation/registration, general legal obligations, administrative procedures, an English legal question, ordinary business advice, Tourism, mismatched hints, and mixed strong Legal/Tourism signals.
- The mixed strong-signal case returns `needs_clarification` at Gateway and proves the Core HTTP client is not created for that request.
- The readiness contract is tested alongside the legacy domain list. Configured Legal reports `data_available`, `retrieval_ready`, and `model_ready` as `unknown`, and `e2e_verified` as `not_verified`. Disabled Tourism routing and retrieval are excluded and readiness reports retrieval unavailable.
- Core’s explicit `catalog_enabled` configuration and Legal collection targets are covered at contract level. Empty General/Tourism Evidence stays fail-closed, with `model_api=not_started` and verifier not started.
- Frontend API tests validate readiness shape and safe handling of unsupported policy values. The frontend shows a non-ready label/reason and the canonical response domain when data is returned by the API.

### Commands and results

| Area | Command | Result | Test type / notes |
|---|---|---|---|
| Focused domain/evidence/Gateway tests | `\.venv\Scripts\python.exe -m pytest backend/tests/test_domain_registry_contract.py backend/tests/test_policy_mcp_integration.py backend/tests/test_domain_documents.py backend/tests/test_evidence_gate.py backend/tests/test_gateway_integration.py -q -p no:cacheprovider` | **55 passed**, 1 Starlette deprecation warning | Final run after domain-disable and alias-collision checks; unit, API contract and mocked Gateway integration. |
| Backend full suite | `\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider` | **302 passed**, 1 Starlette deprecation warning | Final run; unit, contract and mocked integration; no live service calls. |
| Frontend tests | `npm.cmd test` from `FE/` | **20 passed**, 0 failed | API/telemetry Node tests; no browser rendering. |
| Frontend production build | `npm.cmd run build -- --outDir C:\Users\Admin\AppData\Local\Temp\dai-legal-phase5-build-20261027` from `FE/` | **Passed**, 1,865 modules transformed | Output outside repository. |
| Diff whitespace check | `git diff --check` | **Passed** | Existing line-ending conversion warnings only. |
| Service listener check | PowerShell `Get-NetTCPConnection` for 5173, 8000–8002, 5432, 6333 | **No matching listeners** | No service was started; this phase forbids starting dependencies. |

The initial new focused run exposed route coverage defects for general legal phrasing and mixed strong signals, plus an assertion that assumed a grounding field is populated before verifier execution. The Registry rules and ambiguity decision were corrected; the Evidence-gate contract was kept unchanged and the test now asserts `verification_status=not_verified`, `verifier_scope=not_evaluated`, and verifier not started. A prior UI branch referenced `ChevronUp` without importing it; the missing import was added while connecting the UI readiness state.

### Not run / not established

- No Gateway → Core → PostgreSQL/Qdrant → Model API live request or E2E trace ran. No service, Docker, database, Qdrant, or model process was started.
- No browser automation or actual rendered UI test ran. FE tests/build establish API contract and compile output, not runtime browser presentation.
- Registry keyword rules remain heuristic and are not an exhaustive legal classifier. Runtime data/retrieval/model readiness stays `unknown`; no domain is marked E2E verified.
- The API readiness endpoint reports configured runtime state but deliberately does not probe databases, Qdrant, or Model API.

## Phase 2 E2E readiness check — 2026-10-27

### Current environment facts

- Listener checks found ports **5173, 8000, 8001, 8002, 5432, 6333, 6334, 7860, and 11434 closed**. No Python/Node/PostgreSQL/Qdrant/Ollama process appeared; two `docker` processes were present, but no relevant service listener was open.
- `docker ps` could not connect: access denied to `C:\Users\Admin\.docker\config.json` and the Docker Engine named pipe. Container state could not be inspected; no container was started by this run.
- No repository `.env*` file was found. The checked environment-variable names for Catalog URL, Model API URL/port, Qdrant URL, MCP allowlist, and DVC token were absent; values were not printed. Code defaults point Catalog to local PostgreSQL, Qdrant to `localhost:6333`, and Model API to `127.0.0.1:8001`.
- The configured base model is `Qwen/Qwen2.5-7B-Instruct`; a Hugging Face cache snapshot directory exists. The configured LoRA V4 directory and `adapter_model.safetensors` exist. This does not prove the full base weights, adapter compatibility, available runtime memory, or successful inference.
- `nvidia-smi` reported an RTX 3060 with 12 GiB VRAM and about 1.1 GiB in use at the check. Qwen was not loaded. The Core module also loads BGE-M3 and a reranker onto CUDA during import.
- The Gateway exposes `GET /health`; Model API exposes `GET /` and `POST /generate`; Core exposes `GET /`. All were unavailable because no listener was running.

### Safe checks run in this phase

| Check | Result |
|---|---|
| `\.venv\Scripts\python.exe -m pip check` | No broken requirements found. |
| Imports of the eight restored leaf modules (`query_parser`, citation contract/gate, chat session, domain documents, MCP client, RAG diagnostics/relevance) | Passed. |
| Frontend tests (`npm.cmd test`, `FE/`) | 18 passed, 0 failed. |
| Frontend production build to `C:\Users\Admin\AppData\Local\Temp\dai-legal-phase2-build-20261027` | Passed, 1,865 modules transformed. |
| Backend full suite (`\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider`) | 281 passed, 1 existing Starlette deprecation warning. |
| README conflict marker scan | No matching conflict markers. |

`backend.main` was not imported directly: importing it creates/initializes `data/chat_history.db` and constructs BGE-M3/reranker models on CUDA. Avoiding those persistent/heavy side effects is intentional. The prior missing leaf-module import failure is resolved, and all restored leaf imports plus the backend suite pass; a real Core startup/import remains unverified.

### Live E2E status

**BLOCKED / NOT RUN.** PostgreSQL and Qdrant are unavailable and Docker access is denied. The services were not started, so none of the eight acceptance questions were submitted. There are no live HTTP statuses, trace IDs, Evidence counts, verifier results, or E2E latency measurements to report. No source-based legal answer assessment was possible. Mock/unit results above are not live integration results.

On the subsequent multi-domain check, the same relevant ports remained closed and Docker Engine access remained denied. Two `docker` client processes were visible; their container state could not be inspected, so they are not counted as running services.

## Multi-domain contract work — 2026-10-27

Confirmed and fixed: duplicate `DomainRegistryService.register_domain` calls silently replaced the previous config; policy updates using padded IDs could insert a second, unreachable registry key; Tourism had no automatic-routing keywords and was reachable only with a hint; General advertised MCP tool names that the runtime contract never registers; and generic no-keyword queries fell through to Legal policy. Fixes are covered by five new tests in `test_domain_registry_contract.py`, plus an updated Gateway policy contract assertion. The General MCP correction matches the prior effective runtime behavior, which already removed unknown tool names.

Focused tests: **35 passed**; full backend suite after changes: **286 passed, 1 existing Starlette deprecation warning**. Tourism test asserts router output and eligible retrieval collection name using a local fake collection set; it is a contract test, not actual Qdrant data. No production collection was created or queried. See `ARCHITECTURE.md` for domain-by-domain configured source and readiness distinctions.

The first focused run after changing the no-keyword fallback reported two failures: the proposed General test phrase contained `shuttle` and correctly matched the new Tourism terms, while a Gateway contract test still expected the old Legal fallback. The query was changed to a genuinely unmatched general-knowledge phrase, and the Gateway contract was updated to expect `general_assistant` with its existing empty MCP allowlist. The rerun and full suite then passed; no failure was suppressed.

Remaining architectural gaps: routing keywords are separate from the registry; legal questions with vocabulary outside the keyword map can now fall into General; the domain list API does not describe readiness; Catalog policy is coupled to the Legal collection name; no production semantic scorer is configured. General/tourism/admin source availability and every domain's E2E remain unverified.
