DAI-Legal-Model
AI Legal Assistant for Vietnamese administrative procedures

DAI-Legal-Model is an academic research prototype for answering Vietnamese legal and administrative-procedure questions. It combines a React/Vite frontend, FastAPI services, domain-scoped retrieval, a Qwen2.5-7B-Instruct model with a LoRA V4 adapter, and evidence/citation controls.
Disclaimer: This project is for academic and research use. It is not a substitute for advice from a qualified legal professional. Users should verify important procedures, fees, deadlines, and legal requirements against current official sources.

Contents
- Overview
- Key capabilities
- Architecture
- Request and retrieval flow
- Domain-scoped knowledge bases
- Retrieval infrastructure
- MCP and external retrieval
- Technology stack
- Project structure
- Requirements
- Setup
- Run the application
- API endpoints
- Tests and build
- Search-store migration (optional/staged)
- Datasets and model artifacts
- Limitations and safety notes
Overview
The application separates the user interface, request gateway, retrieval/orchestration pipeline, and model inference service.
Component	Responsibility
Frontend	Chat interface, domain selection, document upload, and answer/source presentation
Core Gateway	Frontend-facing API, domain routing, request handling, and response status
Core API	Query analysis, retrieval, evidence construction, prompt assembly, and orchestration
Retrieval layer	Domain-filtered catalog/vector/search retrieval and reranking where configured
Evidence and verifier	Evidence packaging, citation parsing, and citation/evidence checks
Model API	Local inference using Qwen2.5-7B-Instruct and the configured LoRA V4 adapter
MCP	Permission-controlled access to configured external administrative-procedure sources


The frontend submits legal queries to POST /api/v2/legal-query through the Gateway. The default local Gateway URL is http://127.0.0.1:8002.

A successful HTTP response does not necessarily mean the answer pipeline succeeded. Clients should inspect the response body's status and verification fields rather than relying only on the HTTP status code.
Key capabilities
- Multi-domain chat: registered domains can be selected from the frontend.
- Domain-scoped document upload: supported .txt, .md, and .xlsx files can be uploaded to a selected domain.
- Local-first retrieval: the system attempts configured local retrieval before considering external retrieval.
- Retrieval and reranking: the configured RAG path can use BGE-M3 embeddings, Qdrant, and a reranker.
- Evidence-based generation: retrieved material is assembled into evidence context for the model.
- Citation controls: citation formatting/parsing and verifier checks help detect unsupported or malformed citations.
- Permission-controlled MCP: external retrieval is gated by the application's permission policy.
- Staged search-store option: PostgreSQL and Elasticsearch components are available for a controlled migration path; Qdrant remains the default unless explicitly configured otherwise.
Architecture
Browser (React / Vite)
        |
        v
Core Gateway :8002
        |
        v
Core API :8000
  |       |        |
  |       |        +--> Domain / Catalog / RAG Retrieval
  |       |                       |
  |       |                       v
  |       |                Reranking (when configured)
  |       |                       |
  |       |                       v
  |       +----------------> Evidence Builder
  |                               |
  |                               v
  |                         Prompt Assembly
  |                               |
  v                               v
Diagnostics / Response <------ Model API :8001
                                  |
                                  v
                         Qwen2.5-7B-Instruct
                               + LoRA V4
                                  |
                                  v
                       Citation / Evidence Checks
Default local ports
Service	Port	Role
Frontend (Vite)	5173	Browser UI
Core API	8000	Retrieval and answer pipeline
Model API	8001	Local model inference
Core Gateway	8002	Frontend-facing API
Qdrant	6333	Local vector database
Elasticsearch	9200	Optional staged search backend


Request and retrieval flow
At a high level, a request follows this sequence:
1. The frontend sends the query and selected domain to the Gateway.
2. The Gateway forwards the request to the Core API.
3. Query analysis and domain policy determine the applicable retrieval path.
4. The retrieval layer searches the selected domain's configured knowledge source.
5. Retrieved candidates are ranked/filtered and assembled into an evidence bundle.
6. The prompt assembler passes the question and available evidence to the Model API.
7. The model generates a response.
8. Citation/evidence checks determine the response status returned to the frontend.
The exact retrieval components depend on the configured backend and domain. A citation-linked response should not automatically be interpreted as semantically verified or legally correct.
Domain-scoped knowledge bases
The Gateway exposes registered domains through:
GET /api/v2/domains/
The frontend allows a user to select a domain and upload supported files:
POST /api/v2/domains/{domain_id}/documents
Supported upload formats are UTF-8 .txt, .md, and .xlsx, with a maximum upload size of 2 MiB.

With the default Qdrant backend, uploaded chunks are stored in a collection named:
domain_documents_{domain_id}
With the explicitly configured Elasticsearch backend, PostgreSQL is the source of truth and Elasticsearch is used as the retrieval index. Retrieval remains domain-filtered.

Tourism is a sample non-legal domain for isolated development fixtures. It must not fall back to the legal catalog or legal vectors. Uploaded text does not receive a fabricated public URL; the UI identifies it by filename, document ID, and chunk location.
Retrieval infrastructure
Qdrant (default Local RAG backend)
Default address:
http://localhost:6333
The repository documents these legal collections and embedding configurations:
Pipeline	Collection	Embedding model	Vector size
Core API Local RAG	legal_docs_bge_m3	BAAI/bge-m3	1024
MCP search_legal_documents	legal_docs	bkai-foundation-models/vietnamese-bi-encoder	768


These collections use different embedding models and vector dimensions. Each query pipeline must use the matching collection and embedding model.
Important: backend/import_data.py recreates legal_docs. Do not run it against a populated database unless you intentionally want to replace that collection. Qdrant must be running before either retrieval path can be verified.

The documented DVC import command is:
python backend/import_dvc_to_qdrant.py
PostgreSQL and Elasticsearch (staged option)
The search-store implementation uses PostgreSQL as the source of truth and a transactional outbox to synchronize document chunks into Elasticsearch. Existing Qdrant collections are left untouched. The Core API continues to use Qdrant by default until the search migration is reviewed and the Elasticsearch minimum score is calibrated on labeled queries.

See Search-store migration (optional/staged) before enabling this path.
MCP and external retrieval
MCP is an external retrieval layer, not a replacement for the internal Local RAG database. The intended policy is local-first: use local evidence first, and request permission before accessing the configured external source when local evidence is insufficient.

The documented external source is the National Public Service Portal (Dịch vụ công Quốc gia — DVC). The MCP integration restricts the external host to:
dichvucong.gov.vn
The documented DVC endpoints are:
/api/v1/submitting/formality-top-search/list-by-citizen
/api/v1/configuring/formality/get-formality-by-citizen
Available MCP tools documented by the project include:
- search_legal_documents
- search_procedure
The Core API starts the MCP server as a subprocess when the applicable policy permits the requested tool. A separate MCP terminal is normally not required when running the full application.
Permission flow
User question
     |
     v
Local retrieval
     |
     v
Is local evidence sufficient?
     |                  |
    YES                 NO
     |                  |
     v                  v
Generate answer     Ask for permission
                        |
                 User allows external search?
                    |          |
                   YES         NO
                    |          |
                    v          v
                   MCP     Do not access
                    |      external source
                    v
                 DVC API
                    |
                    v
             External evidence
                    |
                    v
             Answer generation
The MCP integration described here is restricted to configured DVC endpoints; it is not unrestricted Google or general web search.

For a standalone MCP development check, the README's documented command is:
python mcp_server.py
Technology stack
Area	Technologies
Frontend	React 19, Vite, Tailwind CSS, Lucide React
Backend APIs	Python, FastAPI
Language model	Qwen2.5-7B-Instruct + LoRA V4
Embeddings	BAAI/bge-m3 (for the documented Core API legal collection)
Vector database	Qdrant
Reranking	Configured BGE reranker
External integration	MCP, National Public Service Portal (DVC)
Optional search store	PostgreSQL, Elasticsearch
Testing	Pytest, Node.js built-in test runner


Project structure
DAI-Legal-Model/
├── FE/                         # React/Vite frontend and Node tests
├── backend/
│   ├── answer/                 # Citation formatting, generation, diagnostics
│   ├── catalog/                # Catalog lookup and service logic
│   ├── evidence/               # Evidence construction and source summaries
│   ├── mcp/                    # MCP adapter and DVC integration
│   ├── rag/                    # Retrieval policy and query logic
│   ├── search/                 # PostgreSQL / Elasticsearch search store
│   ├── tests/                  # Backend tests
│   ├── verifier/               # Answer/evidence verification
│   ├── core_gateway.py         # Gateway API (port 8002)
│   ├── main.py                 # Core API (port 8000)
│   └── model_api.py            # Model inference API (port 8001)
├── configs/                    # Example local configuration
├── data/                       # Datasets and local data
├── evals/                      # Retrieval/answer evaluation utilities
├── gateway_openapi.json        # Gateway OpenAPI specification
├── run_gateway_stack.py        # Starts the three Python services
└── README.md
Requirements
- Windows PowerShell or another supported Python/Node environment.
- Python dependencies from requirements.txt.
- Pytest dependencies from requirements-dev.txt for backend tests.
- Node.js and npm for the frontend.
- Docker Desktop / Docker Engine when using the Compose-managed services.
- The required model weights, LoRA adapter, datasets, and retrieval services must be available and configured locally.
Do not commit API keys, passwords, personal data, local chat history, virtual environments, generated build output, or model checkpoints. Confirm that datasets may be redistributed before publishing them.
Setup
1. Create and activate a Python environment
Run from the repository root:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
If PowerShell blocks activation, use the environment's Python executable directly, for example:
.\.venv\Scripts\python.exe -m pytest
2. Install frontend dependencies
cd FE
npm.cmd install
3. Configure local paths and services
Before starting the backend, check the model-adapter path and project/data paths in the backend configuration. Some settings may use machine-specific absolute paths and need adjustment.

The frontend can use VITE_GATEWAY_API_URL in FE/.env:
VITE_GATEWAY_API_URL=http://127.0.0.1:8002
This is also the default Gateway URL for localhost. Keep secrets in environment variables or an ignored local .env file, not in tracked source files.
Run the application
1. Start PostgreSQL and Qdrant
From the repository root:
docker compose -f backend/docker-compose.yml up -d postgres qdrant
2. Start the Python API stack
.\.venv\Scripts\python.exe run_gateway_stack.py
The launcher starts:
- Core API on port 8000
- Model API on port 8001
- Core Gateway on port 8002
Keep this terminal open while testing. Press Ctrl+C to stop the services started by the script.

The launcher monitors child-process exit codes; it does not wait for health readiness.
3. Start the frontend
Open a second terminal:
cd FE
npm.cmd run dev
Open the local URL printed by Vite, typically:
http://localhost:5173
Runtime notes
- The Model API loads Qwen/Qwen2.5-7B-Instruct with 4-bit NF4/BF16 settings and the configured LoRA V4 adapter at startup.
- The Python APIs bind to 127.0.0.1 by default.
- CORE_API_HOST, MODEL_API_HOST, and CORE_GATEWAY_HOST can override service bind hosts.
- MODEL_API_URL can override the Core API's model-service URL.
- python model_api.py is a separate legacy entry point; it is not launched by run_gateway_stack.py.
- MCP is started by the Core API when the applicable permission and policy allow the requested tool.
- Loopback binding does not add user authentication. Do not expose the Gateway publicly without an appropriate access-control layer.
Generation limits
The documented environment settings include:
Variable	Default	Bounds / purpose
MODEL_MAX_NEW_TOKENS	256	Maximum response length; bounded to 1–512
PROMPT_MAX_EVIDENCE_CHARS	6000	Evidence prompt budget; bounded to 1,000–12,000


These settings can be overridden in the process environment.
API endpoints
Service	Method	Path	Purpose
Core Gateway (:8002)	GET	/health	Gateway health check
Core Gateway (:8002)	POST	/api/v2/legal-query	Submit a query
Core Gateway (:8002)	GET	/api/v2/domains/	List registered domains
Core Gateway (:8002)	POST	/api/v2/domains/{domain_id}/documents	Upload domain documents
Core API (:8000)	Various	See backend/main.py	Retrieval and answer pipeline
Model API (:8001)	POST	/generate	Generate a model response


Example Gateway request:
{
  "query": "What documents are required for enterprise registration?",
  "domain_hint": "enterprise_law",
  "allow_external": false
}
For exact request/response schemas, see gateway_openapi.json or the running service's OpenAPI documentation.
Tests and build
Backend tests
Run from the repository root:
python -m pytest backend/tests -q
Frontend tests and production build
Run from the FE directory:
npm.cmd test
npm.cmd run build
If Vite cannot clean an existing dist directory on Windows because of a file lock or permissions, stop the frontend dev server and try a separate output directory:
npm.cmd run build -- --outDir build-check
Generated directories such as FE/dist/ and FE/build-check/ should not be committed.
Search-store migration (optional/staged)
The PostgreSQL/Elasticsearch path is an optional staged migration. It does not replace Qdrant automatically, and API startup does not run migrations.

Start the development services:
docker compose -f backend/docker-compose.yml up -d postgres elasticsearch
Configure:
- SEARCH_DATABASE_URL (or an explicit CATALOG_DATABASE_URL) for PostgreSQL.
- ELASTICSEARCH_URL (default http://127.0.0.1:9200) for Elasticsearch.
- Optional ELASTICSEARCH_API_KEY or ELASTICSEARCH_USERNAME / ELASTICSEARCH_PASSWORD credentials through environment variables.
Search storage requires an explicitly configured PostgreSQL URL; it does not inherit the Catalog module's legacy default. Elasticsearch security is disabled only in the local Compose development setup and is bound to localhost. Use authenticated and secured settings outside local development.

Schema changes and indexing are explicit operations:
python -m backend.search.migrate                 # dry run
python -m backend.search.migrate --apply         # create new search tables
python -m backend.search.create_index            # create versioned mapping
python -m backend.search.reindex_qdrant          # read-only inventory/dry run
python -m backend.search.reindex_qdrant --apply  # copy validated payloads; does not alter Qdrant
python -m backend.search.sync_outbox             # retry pending index events once
Review dry-run domain counts and skipped/mismatched metadata before using --apply. The Elasticsearch path is selected only with SEARCH_BACKEND=elasticsearch and requires an explicitly calibrated, nonnegative ELASTICSEARCH_MIN_SCORE; do not choose an arbitrary value.

Readiness endpoint:
GET /api/v2/search/readiness
It probes the configured retrieval service and PostgreSQL. The migration has a matching down script, but rollback is destructive to the new search tables. Take an independent backup and obtain explicit operator approval before rollback.
Datasets and model artifacts
The repository documents these datasets:
Dataset	Purpose / use
duyet/vietnamese-legal-instruct	Legal-domain instruction fine-tuning
tmquan/dichvucong-gov-vn	Administrative-procedure knowledge source
data/legal_train_v3/	Prepared dataset documented for train_v3.py
data/dichvucong_procedures/	Local administrative-procedure data


The application requires the configured Qwen2.5-7B-Instruct model, LoRA V4 adapter, and relevant local datasets/services to be present before end-to-end use.

Datasets, vector indexes, model weights, and LoRA adapters may be large or subject to redistribution restrictions. Keep local-only artifacts out of Git unless the team has confirmed permission and intentionally selected a storage method such as Git LFS.
Limitations and safety notes
- Answer quality depends on retrieval coverage, evidence quality, and model behavior.
- Citation checks can detect some citation/evidence problems but do not guarantee semantic correctness or legal accuracy.
- A citation pass must not be interpreted as a semantic verification pass.
- Depending on policy and available checks, the application may return an answered_unverified response. Such a response must not be described as verified.
- External sources can change, become unavailable, or return incomplete information.
- Verify important procedures, fees, deadlines, and legal requirements against current official sources.
- The documented Cloudflare Tunnel deployment is optional; its current availability and access controls have not been verified. If used, the local PC must remain powered on and connected to the Internet.
Academic and research prototype — not professional legal advice.
