# DAI-Legal-Model

Vietnamese legal-assistance research prototype combining a React/Vite frontend, FastAPI services, retrieval-augmented generation (RAG), a Qwen2.5-7B-Instruct model with a LoRA adapter, and evidence/citation verification.

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
