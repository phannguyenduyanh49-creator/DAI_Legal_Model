# DAI-Legal-Model

<<<<<<< HEAD
> **AI Legal Assistant for Vietnamese administrative procedures**  
> Default stack: Qwen2.5-7B-Instruct + LoRA V4 + BGE-M3 RAG + Qdrant + Reranker + MCP/DVC + FastAPI + React/Vite
=======
Vietnamese legal-assistance research prototype combining a React/Vite frontend, FastAPI services, retrieval-augmented generation (RAG), a Qwen2.5-7B-Instruct model with a LoRA adapter, and evidence/citation verification.
>>>>>>> origin/main

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

## Repository layout

```text
<<<<<<< HEAD
outputs/qwen-legal-lora-v3/
```

Model checkpoints are intentionally excluded from Git because of their size.

---

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

The application has been exposed to the Internet using **Cloudflare Tunnel** while keeping the model running on the local RTX 4070 machine.

Current routing:

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

The Cloudflare agent is installed as a Windows Service.

Verified deployment flow:

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

The system has been tested from another device through the public domain.

> The local PC must remain powered on and connected to the Internet for the public demo to remain available.

---

## 📂 Project Structure

```text
DAI-Legal-Model/
│
=======
.
├── FE/                         # React/Vite frontend and Node tests
>>>>>>> origin/main
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
```

If PowerShell blocks activation, you can call the environment's Python directly, for example `.\.venv\Scripts\python.exe -m pytest`.

### 2. Install frontend dependencies

```powershell
cd FE
npm install
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
<<<<<<< HEAD
python prepare_legal_dataset.py
```

Inspect the prepared dataset:

```powershell
python check_dataset.py
```

---

## ▶️ Running the Full System

The default launcher starts the Legacy/Core API, the Model API, and the Core Gateway. It runs `backend.model_api`, which loads `Qwen/Qwen2.5-7B-Instruct` with 4-bit NF4/BF16 settings and the `outputs/qwen-legal-lora-v4` adapter.

Start PostgreSQL and Qdrant, then start the API stack from the repository root:

```powershell
docker compose -f backend/docker-compose.yml up -d postgres qdrant
.\.venv\Scripts\python.exe run_gateway_stack.py
```

The APIs listen on ports 8000 (Legacy/Core), 8001 (Model API), and 8002 (Core Gateway). The frontend uses the Gateway. In another terminal:

```powershell
cd FE
npm.cmd run dev
```

`python model_api.py` is a separate legacy entrypoint. It reads `outputs/active/registry.json` and falls back to the V3 adapter; it is not launched by `run_gateway_stack.py`. The two entrypoints use different adapter-selection behavior, so both are retained.

MCP is launched by the Legacy/Core API when permission and policy allow the requested tool.

### Generation limits and verification status

`MODEL_MAX_NEW_TOKENS` controls the maximum response length (default `256`, bounded to `1..512`). `PROMPT_MAX_EVIDENCE_CHARS` controls the prompt evidence budget (default `6000`, bounded to `1000..12000`). These settings can be overridden in the process environment and reset by removing the override.

The current verifier checks citation IDs and evidence references; it does not calculate semantic support or a confidence score. Domain policies define `verification_threshold`, so answers remain withheld with `confidence_status=not_verified` until a real confidence scorer is integrated. A passing citation check alone is not reported as passing the configured confidence threshold.
=======
cd FE
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`).

## API endpoints

| Service | Method | Path | Purpose |
|---|---|---|---|
| Core Gateway (:8002) | GET | `/health` | Gateway health check |
| Core Gateway (:8002) | POST | `/api/v2/legal-query` | Submit a legal query |
| Legacy/Core API (:8000) | API routes | See `backend/main.py` | Retrieval and answer pipeline |
| Model API (:8001) | POST | `/generate` | Generate a model response |

The Gateway request body currently accepts a query and an `allow_external` permission flag, for example:

```json
{
  "query": "Thủ tục đăng ký tạm trú cần giấy tờ gì?",
  "allow_external": false
}
```

For the exact schemas and response fields, use `gateway_openapi.json` or the running service's OpenAPI docs.
>>>>>>> origin/main

## Tests and production build

Run backend tests from the repository root:

```powershell
python -m pytest backend/tests -q
```

Run frontend tests and build:

```powershell
cd FE
npm test
npm run build
```

The frontend tests use Node's built-in test runner. If Vite cannot clean an existing `dist` directory on Windows due to a file lock or permissions, stop the frontend dev server and try a separate output directory:

```powershell
npm run build -- --outDir build-check
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
