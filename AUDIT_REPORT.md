# Báo cáo audit DAI-Legal-Model

**Ngày audit:** 2026-10-27  
**Phạm vi:** kiểm tra mã nguồn, trạng thái Git, test an toàn và build frontend. Không sửa mã ứng dụng, cấu hình, dữ liệu hoặc migration. File này là báo cáo được yêu cầu.  
**Phân loại:** `PASS` chỉ cho phần có bằng chứng; `PARTIAL` đã có triển khai nhưng còn thiếu xác minh; `FAIL` có lỗi hiện hữu; `NOT TESTED` chưa có điều kiện kiểm tra; `NOT IMPLEMENTED` chỉ dùng khi đã xác minh không có triển khai.

## A. Tóm tắt điều hành

Repository hiện **không sẵn sàng chạy stack từ working tree này**:

1. `backend.main` dừng ngay khi import vì thiếu `backend.catalog.query_parser`; nhiều module khác được import hoặc được test tham chiếu cũng không có trong working tree.
2. Frontend test API/telemetry chạy qua, nhưng production build thất bại vì `FE/src/App.jsx` import `./answerPresentation` không tồn tại.
3. README có ba vùng conflict marker Git. Hướng dẫn và một số mô tả trạng thái verifier không khớp với mã hiện tại.
4. Không có service listener ở các port ứng dụng/cơ sở dữ liệu được kiểm tra. `docker ps` bị từ chối quyền truy cập, vì vậy PostgreSQL, Qdrant và container không được xác minh.
5. Verifier có kiểm tra citation và fail-closed khi scorer bắt buộc không có, nhưng `backend/main.py` khởi tạo `Verifier()` mà không inject confidence/grounding scorer. Do đó **không có bằng chứng hệ thống đánh giá semantic grounding hoặc confidence pháp lý trong runtime**. Chính sách hiện có thể trả `answered_unverified` khi citation, sufficiency và policy cho phép; trạng thái đó không phải `verified`.

**Đánh giá sẵn sàng:** chưa đủ điều kiện demo end-to-end từ working tree hiện tại. Một số unit/API transport tests pass; Core API import và frontend production build fail. Không có live Legal E2E. Không đủ bằng chứng cho đánh giá học thuật về legal correctness, càng chưa đủ cho triển khai công khai/production.

## B. Trạng thái theo module

| Module | Trạng thái | Bằng chứng | Vấn đề / ưu tiên |
|---|---|---|---|
| Frontend API và telemetry | `PASS` cho test được chạy | `npm.cmd test`: 18/18 pass; test kiểm tra request Gateway, outcome, telemetry lọc an toàn | Không chứng minh component chat render hay kết nối service thật |
| Frontend app/build | `FAIL` | Build báo không resolve `./answerPresentation` từ `FE/src/App.jsx` | P0: app bundle không build được từ working tree |
| Gateway | `PARTIAL` | Có request schema, domain routing, allowlist, timeout và diagnostic code; có mock/API tests trong repository | Không có listener/service runtime; chưa xác minh HTTP thực tế |
| Core API | `FAIL` | `python -c "import backend.main"` lỗi `ModuleNotFoundError: backend.catalog.query_parser` | P0: service không thể khởi tạo; các import thiếu khác chưa thể đi tới |
| Domain routing / policy | `PARTIAL` | Keyword router, domain hint, registry mapping alias pháp lý và policy MCP có trong code | Heuristic, không phải classifier ngữ nghĩa; E2E không chạy được |
| Catalog / PostgreSQL | `PARTIAL` | Matcher, repository, schema và optional lookup có trong code; test matcher pass | Không có PostgreSQL được xác minh; Catalog runtime/data match `NOT TESTED`; có credential mặc định trong source/compose |
| Qdrant / RAG / reranker | `PARTIAL` | Code có embedding, truy vấn collection, filter/relevance, rerank và stage diagnostics | Không có Qdrant listener/container được xác minh; module diagnostics/relevance/domain-document bị thiếu |
| Evidence Builder / prompt | `PARTIAL` | Builder, schema, prompt assembly và prompt Evidence selection có trong source | Import PromptAssembler phụ thuộc `backend.answer.citation_contract` đang thiếu; Evidence runtime `NOT TESTED` |
| Citation / verifier | `PARTIAL` | Claim citation parsing, ID linkage, domain metadata check và fail-closed coordinator có trong code | Không có grounding/confidence scorer production; không có legal evaluation đủ nhãn; runtime chưa khởi động |
| Model API / Qwen + LoRA | `NOT TESTED` | Cấu hình model `Qwen/Qwen2.5-7B-Instruct`, adapter V4 và 4-bit NF4 có trong code; thư mục adapter tồn tại | Không gọi model; không xác minh base checkpoint/VRAM/inference; API bind `0.0.0.0` và không thấy xác thực ở endpoint |
| MCP / DVC | `PARTIAL` | Contract, permission checks, timeout và server/tool code tồn tại | `backend.mcp.client` thiếu khiến Core import fail; external DVC không được gọi; runtime/permission live `NOT TESTED` |
| Upload tài liệu theo domain | `PARTIAL` | Gateway/Core có endpoint và giới hạn đọc tối đa 2 MiB + 1 byte | `backend.domain_documents` thiếu; không xác minh định dạng, indexing, kích thước cuối cùng hoặc domain isolation khi chạy |
| PostgreSQL, Qdrant, chat SQLite | `NOT TESTED` | Cấu hình compose và code lưu session tồn tại | Không có listener; Docker API không truy cập được; không truy vấn/ghi dữ liệu |
| Fine-tuning / datasets / evals | `PARTIAL` | Có train/evaluate scripts, local Arrow/XLSX/JSONL data và eval harness | Provenance/license/label quality và model accuracy chưa được xác minh trong audit này |
| Tài liệu dự án | `FAIL` | `README.md` có conflict markers ở dòng 3, 6, 8, 51, 386, 389, 470, 509, 535 | P1: tài liệu hợp nhất chưa hoàn tất; có tuyên bố deployment chưa được kiểm chứng và nội dung không nhất quán |

## C. Kiến trúc thực tế trong mã

```text
React/Vite (FE/src/App.jsx, FE/src/api.js)
       | POST /api/v2/legal-query; domain_hint; allow_external
       v
Core Gateway (backend/core_gateway.py :8002)
  Query schema -> QueryAnalysis -> DomainRouterPlanner
       -> PolicyEnforcementMiddleware / Domain Registry
       -> HTTP POST sang Core API :8000
       v
Core API (backend/main.py)
  session + policy -> Direct Catalog (nếu domain cho phép)
  -> BGE-M3 query embedding -> collection Qdrant theo domain
  -> relevance/exact match/rerank -> EvidenceBuilder
  -> sufficiency/policy -> PromptAssembler + prompt budget
  -> Model API :8001 (Qwen + LoRA)
  -> claim/citation parsing -> Verifier -> outcome policy -> Gateway response

Khi nguồn nội bộ không đủ và permission/policy cho phép:
Core API -> MCP client/server -> DVC search -> Evidence -> prompt/generation
```

Đây là luồng được suy ra từ code, **không phải trace runtime**. Core hiện không import được nên thứ tự thực thi toàn stack chưa được kiểm chứng trong phiên audit. Các module trên được mô tả trong README, nhưng README cũng có conflict marker và không thể dùng làm bằng chứng runtime.

### Luồng đã xác minh bằng code

- FE `askLegalQuery` gửi `query`, `allow_external`, và `domain_hint` nếu người dùng chọn; `FE/src/api.js` kiểm tra hình dạng response.
- Gateway xác thực domain hint qua policy registry, gọi query analyzer/router, áp policy MCP và chuyển `policy_domain` tới Core.
- Router dùng tập keyword; nếu hint non-legal mâu thuẫn với keyword pháp lý, code ưu tiên domain legal phù hợp. Alias như `enterprise_law` được policy middleware map về cấu hình `legal_ai`.
- Core có nhánh Direct Catalog cho cấu hình knowledge collection phù hợp, sau đó gọi local retrieval; code domain collection phụ thuộc `backend.domain_documents`, hiện thiếu.
- PromptAssembler giới hạn Evidence theo ký tự và tạo danh sách Evidence ID cho prompt. Core tạo verifier bundle từ `prompt_evidences`, nên verifier dự kiến nhận đúng Evidence được đưa vào prompt.
- `Verifier()` được tạo không có scorer. Citation integrity không tương đương entailment/grounding; `confidence_status` có thể `not_verified`, grounding `not_evaluated`.
- `reference_answer_failure_reason` có thể cho phép trả câu trả lời citation-linked dưới trạng thái `answered_unverified` nếu data sufficiency/policy cho phép. Nó không được xem là `answered`/`verified`.

## D. Kết quả kiểm tra

### Frontend tests

PowerShell `npm test` bị chặn bởi execution policy vì resolve `npm.ps1`. Dùng shim `npm.cmd` theo hướng dẫn Windows:

```powershell
npm.cmd test
```

**Kết quả:** 18 test, 18 pass, 0 fail, 0 skipped. Đây là Node unit tests cho `api.js` và `telemetry.js`, không phải browser/render/E2E.

### Frontend production build

```powershell
npm.cmd run build -- --outDir C:\Users\Admin\AppData\Local\Temp\dai-legal-audit-build-20261027
```

**Kết quả:** thất bại sau transform vì `FE/src/App.jsx` import `./answerPresentation` nhưng file không tồn tại. Dùng temp output để không ghi đè `FE/dist-old` hay thư mục build có sẵn.

### Backend full suite

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
```

**Kết quả:** pytest dừng ở collection với 8 lỗi import, không có kết quả tổng pass/fail cho toàn suite. Các module không tìm thấy gồm `backend.mcp.client`, `backend.catalog.query_parser`, `backend.chat_session`, `backend.domain_documents`, `backend.answer.evidence_gate`, `backend.rag.diagnostics`, và `backend.answer.citation_contract` (một số lỗi test cùng phụ thuộc một module).

### Backend import probe

```powershell
.\.venv\Scripts\python.exe -c "import backend.main"
```

**Kết quả:** thất bại ngay tại `backend/main.py:5` vì `backend.catalog.query_parser` không tồn tại. Đây là kiểm tra import, không phải khởi chạy service.

### Backend subset không phụ thuộc các module đang thiếu

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_verification_scorer_contract.py backend/tests/test_catalog_matcher.py backend/tests/test_catalog_optional_lookup.py backend/tests/test_data_sufficiency.py backend/tests/test_mcp_http_diagnostics.py backend/tests/test_nli_benchmark.py backend/tests/test_transformers_nli_adapter.py -q -p no:cacheprovider
```

**Kết quả:** 74 passed. Đây là unit/contract tests có test doubles; không chứng minh Qdrant, PostgreSQL, MCP, Qwen hoặc legal grounding accuracy.

Một nhóm rộng hơn có thêm `test_dvc_search.py` cho kết quả **63 passed, 3 failed**; cả ba lỗi gọi `backend.rag.relevance.procedure_relevance`, module hiện không tồn tại.

### Service availability và E2E

- Không tìm thấy listener ở các port kiểm tra: 5173–5175, 8000–8002, 5432, 6333, 7860, 8080, 11434.
- `docker ps --format ...` không chạy được vì quyền truy cập Docker named pipe/config bị từ chối. Tôi không đổi quyền và không khởi động container.
- Không khởi động API, model, Qdrant, PostgreSQL hoặc MCP. Không gửi bốn câu hỏi legal qua HTTP. Trace IDs, latency, Qwen attempts và kết quả retrieval thật: **NOT TESTED**.
- Không chạy browser test; không đánh dấu UI rendering là pass.

## E. Lỗi và rủi ro đã xác minh

### AUD-P0-01 — Working tree thiếu module cần cho startup và build

- **Mức:** P0
- **Bằng chứng:** `backend/main.py:5-45` import nhiều module không tồn tại; import probe dừng ở `backend.catalog.query_parser`. Pytest collection báo 8 import errors. `FE/src/App.jsx:19` import `./answerPresentation`, Vite build không resolve được.
- **Ảnh hưởng:** Core API không thể import; frontend không thể build. E2E hiện không khả thi.
- **Nguyên nhân:** module được import/test tham chiếu nhưng absent trong working tree; không xác định được từ code hiện tại liệu chúng bị xóa chủ đích hay chưa được đưa vào checkout.
- **Hướng xử lý đề xuất, chưa sửa:** khôi phục/đưa vào version control các module đúng theo thay đổi đã dự định; kiểm tra lại diff trước khi khôi phục để tránh ghi đè chỉnh sửa đang có; sau đó import probe, focused/full suite và frontend build.

### AUD-P1-01 — Tài liệu README còn merge conflict và mâu thuẫn

- **Mức:** P1
- **Bằng chứng:** `README.md` có conflict markers ở ba đoạn; nội dung cũng nêu deployment/public test như đã xác minh nhưng audit hiện không có bằng chứng runtime. Phần verifier nói “withheld” trong khi policy code có đường trả `answered_unverified`.
- **Ảnh hưởng:** người vận hành có thể làm theo hướng dẫn sai hoặc hiểu nhầm mức bảo đảm.
- **Hướng xử lý đề xuất, chưa sửa:** hợp nhất README, loại bỏ marker, cập nhật trạng thái verifier và phân biệt verified/unverified; chỉ ghi deployment verified khi có trace/test mới.

### AUD-P1-02 — Rủi ro bảo mật nếu expose stack ra mạng

- **Mức:** P1 trước khi public deployment; chưa xác minh khai thác hay trạng thái tunnel hiện tại.
- **Bằng chứng:** Gateway bind `0.0.0.0` theo mặc định (`backend/core_gateway.py`); Model API bind `0.0.0.0:8001` (`backend/model_api.py:34-35`). Không thấy auth middleware/API key trên Gateway `/api/v2/legal-query` hoặc Model API `/generate`. Core có CORS wildcard với credentials (`backend/main.py:291-292`). Session cookie đặt `SESSION_COOKIE_SECURE = False` (`backend/main.py:125-130`). PostgreSQL compose dùng user/password tĩnh `legal_user` / `legal_password` (`backend/docker-compose.yml:8-10`); database.py có cùng fallback (`backend/catalog/database.py:9-12`).
- **Ảnh hưởng:** nếu các port được expose ngoài máy hoặc qua tunnel như README mô tả, request generation, data access và session có thể không có ranh giới xác thực phù hợp. CORS không thay thế authorization.
- **Hướng xử lý đề xuất, chưa sửa:** xác minh topology deploy; đặt API sau access control/auth, chỉ expose Gateway cần thiết, không expose Model API trực tiếp; dùng secret ngoài source, CORS allowlist chính xác, bật Secure cookie khi HTTPS và đánh giá rate limit.

### AUD-P1-03 — Upload domain có thể là kho dùng chung; policy riêng tư chưa được xác minh

- **Mức:** P1 nếu cho phép người dùng tải tài liệu riêng tư
- **Bằng chứng:** Gateway/Core upload endpoint không thể xác minh end-to-end vì `backend.domain_documents` thiếu; route đọc tối đa 2 MiB + 1 byte rồi chuyển sang service. Tài liệu README mô tả collection theo domain, không theo user/tenant.
- **Ảnh hưởng:** không có bằng chứng tài liệu được cách ly theo người dùng; file có thể được dùng chung trong phạm vi domain. Kiểm tra type/content validation và giới hạn thực sự cũng bị chặn bởi module thiếu.
- **Hướng xử lý đề xuất, chưa sửa:** xác nhận đây là knowledge base chia sẻ có chủ đích hay cần tenant scope; audit service upload khi module được khôi phục; kiểm thử MIME/extension, byte limit, parse safety, authorization và cross-domain/cross-user retrieval.

### AUD-P1-04 — Runtime dependencies không đầy đủ trong manifest

- **Mức:** P1 cho reproducibility
- **Bằng chứng:** `requirements.txt` không khai báo một số dependency được import trực tiếp, gồm SQLAlchemy/driver PostgreSQL và FlagEmbedding; hiện môi trường `.venv` có thể chứa dependency ngoài manifest, nhưng chưa dựng environment sạch để xác minh.
- **Ảnh hưởng:** setup mới có thể lỗi ngay khi import/start. Không chạy cài đặt package trong audit.
- **Hướng xử lý đề xuất, chưa sửa:** đối chiếu import production với lock/requirements; pin đúng runtime dependencies và kiểm tra cài đặt trong environment sạch.

### AUD-P1-05 — Runtime launcher không chờ health readiness

- **Mức:** P1 reliability
- **Bằng chứng:** `run_gateway_stack.py` khởi chạy ba process cách nhau một giây, sau đó chỉ poll exit code; không gọi health endpoint để xác nhận API và Qwen ready.
- **Ảnh hưởng:** launcher có thể báo “started” khi model còn load hoặc service dependency chưa sẵn sàng; E2E có thể đua readiness.
- **Hướng xử lý đề xuất, chưa sửa:** thêm readiness probe có timeout và báo lỗi riêng cho service sau khi stack khôi phục; không tự khởi động model trong audit.

### AUD-P2-01 — Semantic grounding và confidence chưa được đo trong runtime

- **Mức:** P2 cho prototype, P0 trước khi tuyên bố legal `verified`
- **Bằng chứng:** `backend/main.py:109` khởi tạo `Verifier()` không scorer; `backend/verifier/scoring.py` xác nhận interface chứ không có production scorer. `SCORING.md` ghi NLI adapter là experimental. Bộ test contract dùng test doubles/synthetic cases.
- **Ảnh hưởng:** citation integrity chỉ kiểm tra claim-to-Evidence linkage; không chứng minh entailment hoặc đúng pháp luật. Không có legal precision/recall/FPR đủ nhãn để công bố.
- **Hướng xử lý đề xuất, chưa sửa:** xây tập claim–Evidence do reviewer pháp lý gán nhãn, đánh giá held-out theo domain/claim type, đánh giá latency/false-positive, rồi review scorer/policy trước production integration.

### AUD-P2-02 — Domain router dùng keyword và confidence heuristic

- **Mức:** P2
- **Bằng chứng:** `backend/orchestrator.py` cộng điểm keyword rồi chọn domain có số từ khóa cao nhất; hint hợp lệ được ưu tiên, trừ trường hợp legal keyword override hint non-legal.
- **Ảnh hưởng:** mơ hồ, từ đồng nghĩa, câu dài đa lĩnh vực và câu phủ định có thể phân loại sai; score không phải xác suất đã calibration.
- **Hướng xử lý đề xuất, chưa sửa:** đánh giá trên bộ routing có nhãn và confusion matrix; thêm clarification khi top domains mơ hồ; không gọi router score là confidence calibrated.

### AUD-P2-03 — Chất lượng retrieval/sufficiency chưa được chứng minh bằng dữ liệu đang chạy

- **Mức:** P2
- **Bằng chứng:** có bộ lọc score, relevance, reranker, Direct Catalog matcher và field-based sufficiency trong code. Không truy cập được Qdrant/PostgreSQL nên không biết collection, bản ghi, freshness, hit quality hoặc completeness hiện tại.
- **Ảnh hưởng:** code path có thể tồn tại nhưng không chứng minh đúng dữ liệu, đúng thủ tục hay Evidence đủ cho câu trả lời.
- **Hướng xử lý đề xuất, chưa sửa:** sau khi services và modules hoạt động, ghi trace cho bốn câu hỏi Legal; xác minh catalog hit, Qdrant collection/payload, Evidence thật, prompt IDs, sufficiency, model calls và policy outcome.

## F. Chất lượng AI — các chiều tách biệt

| Chiều | Trạng thái audit | Nhận xét |
|---|---|---|
| Retrieval relevance | `NOT TESTED` runtime; `PARTIAL` code/tests | Có matcher/filter/rerank code và unit tests; không có Qdrant/DB live để tính recall hoặc relevance |
| Evidence sufficiency | `PARTIAL` | Có field detector và required-field checks; heuristic phrase mapping, cần legal query evaluation và data thực |
| Citation integrity | `PARTIAL` | Có parser và ID checks; tests subset pass; Core hiện không import được |
| Semantic grounding | `NOT EVALUATED` runtime | Scorer interface/test doubles; production `Verifier()` không có grounding scorer |
| Confidence | `NOT EVALUATED` runtime | Không có calibrated confidence scorer; không dùng router/NLI softmax làm correctness probability |
| Legal correctness/currentness | `NOT TESTED` | Không có gold legal set/reviewer outcomes và không chạy request live; không được suy ra từ citation hoặc test |

Không tính điểm tổng hợp vì các chiều trên có ý nghĩa và bằng chứng khác nhau.

## G. Ma trận tiêu chí

| Tiêu chí | Trạng thái | Bằng chứng còn thiếu |
|---|---|---|
| FE request/response shape | `PASS` ở unit test | Browser render và Gateway live |
| FE production app | `FAIL` | Khôi phục module import bị thiếu, build lại |
| Core API startup | `FAIL` | Khôi phục module imports; import probe và health endpoint |
| Gateway/Core API contract | `PARTIAL` | HTTP integration với cả hai service đang chạy |
| Domain hint/routing | `PARTIAL` | Corpus có nhãn, confusion matrix, live requests |
| MCP permission/deny behavior | `PARTIAL` | Client module thiếu; test runtime/denial với server thật |
| Catalog lookup | `PARTIAL` | PostgreSQL, schema/data và trace query thực |
| Qdrant retrieval | `PARTIAL` | Collection tồn tại, dimension/payload counts, retrieved examples |
| Prompt Evidence matches verifier Evidence | `PASS` theo code path tĩnh | Runtime trace/model request chưa có |
| Citation checks | `PARTIAL` | Full test suite blocked; live response chưa có |
| Semantic support/confidence | `NOT IMPLEMENTED` production scorer | Labeled legal evaluation và reviewed scorer/policy |
| No-Evidence behavior | `PARTIAL` theo code/test fixtures | Live request qua Gateway chưa thực hiện |
| Bốn câu hỏi Legal E2E | `NOT TESTED` | Startup modules, services, data, model và HTTP route |
| Public deployment security | `NOT TESTED` topology; `FAIL` code-level controls hiện thấy | Auth/access proxy review, network exposure and cookie/CORS validation |

## H. Kế hoạch khắc phục đề xuất (chưa thực hiện)

### P0 — Khôi phục khả năng chạy và xác định source-of-truth

- **Mục tiêu:** đưa working tree về trạng thái import/build/test được, không ghi đè thay đổi hiện có.
- **Module:** `backend/catalog/query_parser.py`, `backend/domain_documents.py`, `backend/chat_session.py`, `backend/mcp/client.py`, `backend/rag/diagnostics.py`, `backend/rag/relevance.py`, `backend/answer/citation_contract.py`, `backend/answer/evidence_gate.py`, `FE/src/answerPresentation.js`; `README.md`.
- **Nghiệm thu:** `import backend.main` pass; FE production build pass; backend full suite collection pass; conflict marker scan rỗng; git diff xác nhận không mất thay đổi trước đó.

### P1 — Xác minh chức năng Legal trong môi trường thực

- **Mục tiêu:** test bốn truy vấn Legal, nguồn, Evidence, prompt, Qwen, citation/outcome và latency trên request mới.
- **Điều kiện:** service ports/readiness pass; PostgreSQL và Qdrant truy cập được; collection/schema/data kiểm chứng; model/adapter sẵn sàng; policy external giữ explicit permission.
- **Nghiệm thu:** ghi trace IDs và stage durations thật; mỗi câu hỏi có domain, catalog state, retrieval collection, Evidence IDs/sources, sufficiency, prompt Evidence count, model attempts, citation status và final outcome. Không coi `answered_unverified` là `verified`.

### P1 — Chặn public exposure trước khi deploy lại

- **Mục tiêu:** xác nhận Gateway là điểm vào duy nhất, bảo vệ model/database, kiểm soát origin/session và secrets.
- **Nghiệm thu:** threat model/access control được review; không có credential mặc định trong deploy; cookie Secure cho HTTPS; external access test âm/dương; logs không chứa prompt, token hoặc secret.

### P2 — Đo chất lượng retrieval và verifier

- **Mục tiêu:** tách retrieval recall, sufficiency, citation correctness, entailment, contradiction và confidence calibration.
- **Nghiệm thu:** bộ legal test có provenance/reviewer labels, split giữ nguyên cho candidate scorers, report confusion matrix/precision/recall/FPR và latency; scorer không vào production trước review.

### P3 — Hoàn thiện docs, UI và vận hành

- **Mục tiêu:** đồng bộ README/OpenAPI, accessibility/browser test, readiness health checks, build/test CI và monitoring.
- **Nghiệm thu:** tài liệu khớp code; health probes phản ánh model/database readiness; CI test/build xanh; UI tested trong browser; rollback deployment rõ ràng.

## I. Kết luận

1. **Đã có bằng chứng:** FE API/telemetry unit tests; một tập backend unit/contract tests (74 pass); code path cho domain routing/policy, Catalog matcher, RAG, Evidence, citation checks, MCP allowlist và Qwen API.
2. **Triển khai một phần:** Gateway/Core modules, retrieval, evidence, MCP, session handling, diagnostics và domain-specific document flow. Các phần lớn chưa được live test; một số required modules đang vắng mặt.
3. **Chưa xác minh:** Qdrant/PostgreSQL data và collection; model load/inference; actual external MCP; browser UI; legal answer quality; bốn truy vấn E2E; public deployment.
4. **Ba ưu tiên:** khôi phục module thiếu và build/import; khóa rủi ro public API/credentials/session trước khi mở mạng; chạy Legal E2E có trace trên dữ liệu và services đã xác minh.
5. **Mức sẵn sàng:** có thể trình bày một số unit tests và kiến trúc code sau khi giải thích rõ giới hạn; chưa đủ để demo end-to-end từ checkout này; chưa đủ cho đánh giá học thuật về legal correctness hoặc triển khai thực tế.

## Phụ lục — Trạng thái working tree

Tại đầu audit, working tree đã có nhiều file modified và untracked (bao gồm frontend, backend, README và artifacts evaluation). Các thay đổi hiện hữu được giữ lại; README được cập nhật có mục tiêu để xử lý conflict markers và thông tin sai lệch đã xác minh. Không stage, reset, checkout-over hoặc xóa file. Không có test/build artifact nào được tạo trong repository. `backend/main.zip` không bị đọc/ghi trong quá trình audit.

## Phụ lục cập nhật sau khôi phục P0 — 2026-10-27

Phần A–I phía trên ghi lại kết quả trước khi khôi phục. Khi tiếp tục theo yêu cầu mới, các module vắng mặt được tìm thấy trong parent untracked của stash `backup-before-sync-main` (`stash@{1}^3`). Tôi xác minh từng đường dẫn chưa tồn tại, trích xuất riêng chín blob vào working tree, rồi so sánh SHA của từng file với blob gốc trong stash. Không pop, drop, sửa stash hay ghi index. Các đường dẫn đã khôi phục:

- `FE/src/answerPresentation.js`
- `backend/answer/citation_contract.py`
- `backend/answer/evidence_gate.py`
- `backend/catalog/query_parser.py`
- `backend/chat_session.py`
- `backend/domain_documents.py`
- `backend/mcp/client.py`
- `backend/rag/diagnostics.py`
- `backend/rag/relevance.py`

Sau khôi phục: backend full suite **279 passed, 1 warning**; frontend tests **18 passed**; frontend production build **passed** vào thư mục tạm ngoài repository. `pip check` báo không có broken requirements. README conflict markers đã được loại bỏ, phần verifier/deployment được sửa thành mô tả có điều kiện. Các kết quả này thay thế trạng thái P0 tương ứng trong snapshot ban đầu; chúng không thay đổi kết luận rằng live E2E, database/Qdrant, Qwen runtime, public deployment security và semantic legal quality vẫn chưa được xác minh.

### Security follow-up — 2026-10-27

The Core API, Model API, and Gateway now bind to loopback by default, and the launcher preserves explicit host overrides. The Core-to-Model URL also defaults to loopback and follows `MODEL_API_PORT`. Regression tests cover launcher defaults and overrides; final backend suite: **281 passed, 1 existing warning**. This reduces accidental exposure for local runs but does not add authentication; the earlier public-deployment risk remains for tunnels, reverse proxies, and explicit non-loopback binds. The original audit snapshot above records the pre-change state.
