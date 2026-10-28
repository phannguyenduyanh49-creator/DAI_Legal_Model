# DAI-Legal-Model

> **AI Legal Assistant for Vietnamese administrative procedures**  
> Default stack: Qwen2.5-7B-Instruct + LoRA V4 + BGE-M3 RAG + Qdrant + Reranker + MCP/DVC + FastAPI + React/Vite
>
> Vietnamese legal-assistance research prototype combining a React/Vite frontend, FastAPI services, retrieval-augmented generation (RAG), a Qwen2.5-7B-Instruct model with a LoRA adapter, and evidence/citation verification.

> **Scope:** The project focuses on Vietnamese legal and administrative-procedure questions. It is an academic prototype, not a substitute for professional legal advice.

## Overview

- **Frontend:** React 19, Vite, Lucide React.
- **Core Gateway:** FastAPI service for legal-query requests, routing/planning, diagnostics, and response status.
- **Legacy/Core API:** FastAPI pipeline for catalog/RAG retrieval, evidence construction, prompt assembly, model calls, and verification.
- **Model API:** Qwen2.5-7B-Instruct with a locally available LoRA adapter.
- **Retrieval:** Catalog and RAG components, with Qdrant/embedding/reranking dependencies where configured.
- **Evidence controls:** Evidence bundles, explicit citations, and verifier checks are used to reduce unsupported answers.
- **External retrieval:** Requests to external sources must follow the application's permission policy.

## Architecture

```text
Browser (React/Vite)
        |
        v
Core Gateway :8002
        |
        v
Legacy/Core API :8000 ----> Catalog / RAG / Evidence Builder
        |                              |
        |                              v
        |                       Prompt Assembly
        |                              |
        v                              v
Diagnostics / Response <-------- Model API :8001
                                       |
                                       v
                              Qwen2.5-7B + LoRA
                                       |
                                       v
                             Answer / Citation Verifier
```

The frontend sends `POST /api/v2/legal-query` to the Gateway. On localhost, the default Gateway URL is `http://127.0.0.1:8002`. A successful HTTP response may still carry `status: "FAILED"`; clients should inspect the response body instead of relying only on the HTTP status.

## 📚 Datasets

### 1. Vietnamese Legal Instruction Dataset

```text
duyet/vietnamese-legal-instruct
```

Used primarily for legal-domain instruction fine-tuning.

### 2. National Public Service Portal Procedure Dataset

```text
tmquan/dichvucong-gov-vn
```

Used as the main RAG knowledge source for Vietnamese administrative procedures from the National Public Service Portal.

Local path:

```text
data/dichvucong_procedures/
```

### 3. Prepared V3 dataset

```text
data/legal_train_v3/
```

This is the processed dataset used by `train_v3.py`.

Large dataset files are stored in the repository using **Git LFS**.

---

## 🔎 RAG Pipeline

```text
User Question
     │
     ▼
Core API :8000
     │
     ▼
MCP search_legal_documents
     │
     ▼
Qdrant / legal_docs
     │
     ▼
Relevant legal procedure
     │
     ▼
Prompt + retrieved context
     │
     ▼
Model API :8001
     │
     ▼
Qwen2.5-7B-Instruct + LoRA V3
     │
     ▼
Generated Answer
```

The current Core API limits and reranks the retrieved context before sending it to the model to reduce irrelevant context, hallucination risk, inference latency, and request timeouts.

---

## 🔌 MCP & External Data Retrieval

MCP is used as the **external retrieval layer** when the Local RAG knowledge base does not contain enough reliable information to answer the user's question.

MCP server:

```text
mcp_server.py
```

### MCP tools

Available tools include:

- `search_legal_documents`
- `search_procedure`

The Core API starts the MCP server as a subprocess, so a separate MCP terminal is normally **not required** for the full application.

### 🌐 External Data Source — Dịch vụ công Quốc gia

The external retrieval layer connects to the **National Public Service Portal (Dịch vụ công Quốc gia — DVC)** for Vietnamese administrative-procedure information.

The DVC source is used to retrieve procedure information such as:

- Procedure name and identification
- Required documents
- Processing information
- Fees / charges when available
- Other available administrative-procedure details

The MCP server uses a whitelist for the external host:

```text
dichvucong.gov.vn
```

The main external DVC endpoints used by the project are:

```text
/api/v1/submitting/formality-top-search/list-by-citizen
/api/v1/configuring/formality/get-formality-by-citizen
```

### 🔄 External Retrieval Flow

External retrieval is **not the first retrieval path**. The system follows a Local-first strategy:

```text
User Question
      ↓
Synonym Mapping
      ↓
BGE-M3
      ↓
Qdrant
      ↓
Top-K
      ↓
BGE Reranker
      ↓
Enough Local Data?
   ├── YES
   │    ↓
   │  Local Legal Context
   │    ↓
   │  Qwen2.5-7B + LoRA V3
   │    ↓
   │  Answer
   │
   └── NO
        ↓
    Ask User Permission
        ↓
     User allows?
        ↓ YES
       MCP
        ↓
     DVC API
        ↓
   External Legal Context
        ↓
   Qwen2.5-7B + LoRA V3
        ↓
      Answer
```

### 🔐 Permission & Safety

When Local RAG is insufficient, the system asks the user for permission before accessing external data.

```text
Local RAG insufficient
        ↓
Ask: Allow external retrieval?
        ↓
   ┌────┴────┐
   │         │
  YES        NO
   │         │
   ↓         ↓
 MCP       Stop external
   │       retrieval
   ↓
 DVC API
```

The MCP server does not provide unrestricted web/Google search. External retrieval is restricted to the configured DVC source.

### MCP standalone test

```powershell
python mcp_server.py
```

---

## 🗄️ Qdrant

Qdrant is used as the local vector database for the Local RAG pipeline.

Default address:

```text
http://localhost:6333
```

Current collection:

```text
legal_docs_bge_m3
```

Embedding model:

```text
BAAI/bge-m3
```

Vector size:

```text
1024
```

Distance:

```text
COSINE
```

The runtime has a separate MCP local-search index:

| Pipeline | Collection | Embedding model | Configured vector size |
|---|---|---|---:|
| Core API Local RAG | `legal_docs_bge_m3` | `BAAI/bge-m3` | 1024 |
| MCP `search_legal_documents` | `legal_docs` | `bkai-foundation-models/vietnamese-bi-encoder` | 768 |

These collections use different embedding models and vector dimensions. Keep each
query pipeline paired with its matching collection. `backend/import_data.py`
recreates `legal_docs`; do not run it against a populated database unless you
intend to replace that collection. Qdrant must be running before either retrieval
pipeline can be verified.

The current DVC import workflow uses:

```powershell
python backend/import_dvc_to_qdrant.py
```

The system uses Qdrant as the primary local knowledge base before considering external MCP retrieval.

## 🌐 Web Frontend

Frontend technology:

- React
- Vite
- Tailwind CSS
- Lucide React

Local development server:

```text
http://localhost:5173
```

The browser tab is configured as:

```text
AI Pháp Lý
```

The frontend communicates with the Core API through the `/api/v1/chat/completions` endpoint using streaming responses.

---

## 🌍 Internet Deployment

This repository documents an optional **Cloudflare Tunnel** deployment while keeping the model on a local machine. The current deployment and network access controls were not verified in this audit.

Documented routing (current availability not verified):

```text
legal.donghai.uk
        │
        ▼
localhost:5173
Frontend
```

```text
api.donghai.uk
        │
        ▼
localhost:8000
Core API
```

The README has previously described the Cloudflare agent as a Windows Service; its current installation state was not checked.

Documented routing (not live-tested in this audit):

```text
Other Device
     │
     ▼
Internet
     │
     ▼
Cloudflare Tunnel
     │
     ├── Frontend :5173
     │
     └── Core API :8000
              │
              ▼
         MCP + Qdrant
              │
              ▼
       Model API :8001
              │
              ▼
     Qwen2.5-7B + LoRA V3
```

The repository documents this deployment path, but its current availability and access controls were not verified in this checkout.

> If this tunnel configuration is used, the local PC must remain powered on and connected to the Internet for the service to remain available.

---

## 📂 Project Structure

```text
.
├── FE/                         # React/Vite frontend and Node tests
├── backend/
│   ├── answer/                  # Citation formatting, model client, diagnostics
│   ├── catalog/                 # Catalog lookup and service logic
│   ├── rag/                     # Retrieval policy and data sufficiency
│   ├── verifier/                # Answer/evidence verification
│   ├── core_gateway.py          # Gateway API (port 8002)
│   ├── main.py                  # Legacy/Core API (port 8000)
│   └── model_api.py             # Model inference API (port 8001)
├── data/                        # Project datasets and local data
├── gateway_openapi.json         # Gateway OpenAPI specification
├── run_gateway_stack.py         # Starts the three Python services
├── model_api.py                 # Legacy model API entry point
└── README.md
```

## Requirements

- Windows PowerShell or another supported Python/Node environment.
- Python dependencies listed in `requirements.txt`.
- Pytest test-runner dependency listed in `requirements-dev.txt` for backend test runs.
- Node.js and npm for the frontend.
- Model weights/LoRA adapter and any required retrieval services/data must be available locally or configured for your environment.
- If using Qdrant, make sure its service and collection are configured before running queries.

Do not commit API keys, passwords, personal data, local chat history, virtual environments, generated build output, or model checkpoints. Confirm that datasets may be redistributed before publishing them.

## Setup (Windows / PowerShell)

### 1. Create and activate a Python environment

Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install this additional file when running the backend tests.
pip install -r requirements-dev.txt
```

If PowerShell blocks activation, you can call the environment's Python directly, for example `.\.venv\Scripts\python.exe -m pytest`.

### 2. Install frontend dependencies

```powershell
cd FE
npm.cmd install
```

### 3. Configure local paths and services

Before starting the backend, check the model-adapter path and project/data paths in the backend configuration. Some current settings use Windows absolute paths and need to be adjusted for your machine. Do not put secrets in tracked files; use local environment variables or an ignored `.env` file where supported.

The frontend can be configured with `VITE_GATEWAY_API_URL` in `FE/.env`:

```dotenv
VITE_GATEWAY_API_URL=http://127.0.0.1:8002
```

For localhost, the frontend defaults to this Gateway URL if the variable is not set.

## Run the system

From the repository root, with Python dependencies, model files, and retrieval services configured:

```powershell
python run_gateway_stack.py
```

This script starts:

| Service | Default port | Purpose |
|---|---:|---|
| Legacy/Core API | 8000 | Retrieval/evidence pipeline and model orchestration |
| Model API | 8001 | Local model inference |
| Core Gateway | 8002 | Frontend-facing legal-query API |

Keep this terminal open while testing. Press `Ctrl+C` to stop the services started by the script.

In a separate terminal, start the frontend:

```powershell
cd FE
npm.cmd run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`).

## Running the Full System

The default launcher starts the Legacy/Core API, Model API, and Core Gateway. The Model API loads `Qwen/Qwen2.5-7B-Instruct` with 4-bit NF4/BF16 settings and the configured LoRA V4 adapter at startup.

Start PostgreSQL and Qdrant, then start the API stack from the repository root:

```powershell
docker compose -f backend/docker-compose.yml up -d postgres qdrant
.\.venv\Scripts\python.exe run_gateway_stack.py
```

The APIs use ports 8000 (Legacy/Core), 8001 (Model API), and 8002 (Core Gateway). In another terminal, start the frontend with the command above. The launcher currently monitors child process exit codes; it does not wait for health readiness.

The Python APIs bind to `127.0.0.1` by default. The launcher passes `CORE_API_HOST`, `MODEL_API_HOST`, and `CORE_GATEWAY_HOST` through to the services; set an explicit host only when the deployment boundary and access controls are configured. `MODEL_API_URL` can override the Core API's model service URL. Loopback binding does not add user authentication, and the Gateway should not be exposed publicly without an access-control layer.

`python model_api.py` is a separate legacy entrypoint and is not launched by `run_gateway_stack.py`. MCP is started by the Core API when permission and policy allow the requested tool.

### Generation limits and verification status

`MODEL_MAX_NEW_TOKENS` controls the maximum response length (default `256`, bounded to `1..512`). `PROMPT_MAX_EVIDENCE_CHARS` controls the prompt evidence budget (default `6000`, bounded to `1000..12000`). These settings can be overridden in the process environment.

The verifier checks citation linkage but has no production semantic-grounding or calibrated confidence scorer configured. A citation pass is not a semantic verification pass. Existing policy may return a citation-linked `answered_unverified` answer when its conditions pass; that response must not be described as verified.

## API endpoints

| Service | Method | Path | Purpose |
|---|---|---|---|
| Core Gateway (:8002) | GET | `/health` | Gateway health check |
| Core Gateway (:8002) | POST | `/api/v2/legal-query` | Submit a legal query |
| Core Gateway (:8002) | GET | `/api/v2/domains/` | List registered domains |
| Legacy/Core API (:8000) | API routes | See `backend/main.py` | Retrieval and answer pipeline |
| Model API (:8001) | POST | `/generate` | Generate a model response |

The Gateway request body accepts `query`, optional `domain_hint`, and `allow_external`, for example:

```json
{
  "query": "What documents are required for enterprise registration?",
  "domain_hint": "enterprise_law",
  "allow_external": false
}
```

For exact schemas and response fields, use `gateway_openapi.json` or the running service's OpenAPI docs.
## Tests and production build

Run backend tests from the repository root:

```powershell
python -m pytest backend/tests -q
```

Run frontend tests and build:

```powershell
cd FE
npm.cmd test
npm.cmd run build
```

The frontend tests use Node's built-in test runner. If Vite cannot clean an existing `dist` directory on Windows due to a file lock or permissions, stop the frontend dev server and try a separate output directory:

```powershell
npm.cmd run build -- --outDir build-check
```

Generated directories such as `FE/dist/` and `FE/build-check/` should not be committed.

## Data and model artifacts

Datasets, vector indexes, model weights, and LoRA adapters may be large or subject to redistribution restrictions. Keep local-only artifacts out of Git unless the team has confirmed permission and intentionally selected a storage method such as Git LFS. The application will not run end-to-end until the required model files, data, and dependent services are configured.

## Limitations and disclaimer

- Answer quality depends on retrieval coverage, evidence quality, and model behavior.
- A verifier can catch some citation/evidence problems but cannot guarantee legal correctness.
- External sources may change, be unavailable, or return incomplete information.
- Users should verify important procedures, fees, deadlines, and legal requirements against current official sources.

This repository is intended for academic and research use only.

## Domain-scoped document knowledge bases

The Gateway lists registered domains at GET /api/v2/domains/. Select a domain in the chat controls, then upload UTF-8 .txt or .md files (up to 2 MiB) with POST /api/v2/domains/{domain_id}/documents before asking a question in that domain. Uploaded chunks are stored in `domain_documents_{domain_id}` and retrieval is restricted to that domain's configured collection. Tourism is a sample non-legal domain for isolated development fixtures; it does not fall back to legal catalog or legal vectors. Uploaded text has no fabricated public URL, so the UI identifies it by filename, document ID, and chunk location.
