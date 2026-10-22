# DAI-Legal-Model

> **AI Legal Assistant for Vietnamese administrative procedures**  
> Default stack: Qwen2.5-7B-Instruct + LoRA V4 + BGE-M3 RAG + Qdrant + Reranker + MCP/DVC + FastAPI + React/Vite

## 📌 Overview

DAI-Legal-Model is an academic/research prototype for a Vietnamese legal assistant. The system combines a domain-adapted Large Language Model (LLM) with Retrieval-Augmented Generation (RAG), semantic reranking, and permission-controlled external retrieval so that answers can be grounded in Vietnamese administrative-procedure data.

The project currently supports:

- Vietnamese legal-domain question answering
- LoRA/QLoRA fine-tuning of Qwen2.5-7B-Instruct
- Semantic legal-document retrieval with Qdrant
- MCP tools for legal procedure search and DVC external retrieval
- Permission-gated fallback from Local RAG to external DVC retrieval
- Auto Fine-Tuning pipeline with validation, QLoRA training, evaluation, and adapter activation/rollback
- FastAPI Core API and Model API
- React/Vite web frontend
- Public Internet access through Cloudflare Tunnel
- Deployment on a local RTX 4070 12GB machine

---

## 🏗️ System Architecture

```text
USER
  ↓
FRONTEND (React / Vite)
  ↓
CORE API :8000
  ↓
ORCHESTRATOR
  ↓
SYNONYM MAPPING
  ↓
BGE-M3
  ↓
QUERY EMBEDDING
  ↓
QDRANT
  ↓
TOP-K
  ↓
BGE RERANKER
  ↓
ENOUGH DATA?
  ├── YES
  │    ↓
  │  LEGAL CONTEXT
  │    ↓
  │  QWEN2.5-7B + LORA V3
  │    ↓
  │  ANSWER
  │
  └── NO
       ↓
    ASK USER
       ↓
    USER ALLOW?
       ↓ YES
      MCP
       ↓
     DVC API
       ↓
   LEGAL CONTEXT
       ↓
  QWEN2.5-7B + LORA V3
       ↓
     ANSWER
```

### Core principle

**Local RAG is the primary retrieval path.** BGE-M3 converts the user query into an embedding, Qdrant retrieves candidate procedures, and BGE Reranker filters/reorders the candidates before the system decides whether the local evidence is sufficient.

**MCP is a fallback mechanism**, not the first retrieval path. When local evidence is insufficient, the system asks the user for permission before accessing the whitelisted DVC source.

**LoRA** adapts the model's legal-domain response behavior, while **RAG** supplies factual context for administrative-procedure questions.

## 🤖 Model

| Component | Configuration |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning | QLoRA / LoRA / PEFT |
| Adapter | `outputs/qwen-legal-lora-v3` |
| Quantization | 4-bit NF4 |
| Compute dtype | BF16 |
| GPU tested | NVIDIA RTX 4070 12GB |
| Training dataset | `duyet/vietnamese-legal-instruct` |

### V3 training

The V3 adapter was trained with:

- 4-bit NF4 quantization
- LoRA adapters
- BF16
- Gradient checkpointing
- Gradient accumulation
- Paged AdamW 8-bit optimizer
- Maximum sequence length: 2048
- Training data prepared under `data/legal_train_v3/`

Train with:

```powershell
python train_v3.py
```

The resulting adapter is saved to:

```text
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
├── backend/
│   ├── main.py
│   └── import_dvc_to_qdrant.py
│
├── FE/
│   └── src/
│       ├── App.jsx
│       └── api.js
│
├── auto_finetune/
│   ├── auto_activation.py
│   ├── auto_evaluator.py
│   ├── auto_trainer.py
│   ├── dataset_formatter.py
│   ├── dataset_manager.py
│   ├── dataset_validator.py
│   ├── run_auto.py
│   └── datasets/
│       └── current/
│
├── mcp_server.py
├── model_api.py
├── run_test_cases.py
├── test_reranker.py
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets, model checkpoints, local databases, and runtime-generated files should not be committed to the repository unless they are intentionally managed through Git LFS or another artifact-storage mechanism.

## 🔌 API Endpoints

| Service | Endpoint | Port | Purpose |
|---|---|---:|---|
| Core API | `/api/v1/chat/completions` | 8000 | Chat request and answer generation |
| Core API | `/api/v1/chat/history` | 8000 | Retrieve conversation/session history |
| Model API | `/generate` | 8001 | Qwen2.5-7B + LoRA inference |
| Qdrant | `/collections` | 6333 | Vector database status and collections |

The normal application flow is:

```text
Frontend → Core API → Local RAG
                     ↓
                MCP if allowed
                     ↓
                 Model API
```

## ⚙️ Installation

### 1. Clone repository

```powershell
git clone https://github.com/AnNguyen0806/DAI-Legal-Model.git
cd DAI-Legal-Model
```

### 2. Git LFS

Install Git LFS and initialize it:

```powershell
git lfs install
git lfs pull
```

The repository contains several GB of dataset files, so sufficient disk space and Git LFS quota are required.

### 3. Create Python environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scriptsctivate
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Dataset Preparation

Download the DVC procedure dataset:

```powershell
python download_dvc_dataset.py
```

Download the legal instruction dataset:

```powershell
python download_legal_dataset.py
```

Prepare the V3 training data:

```powershell
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

---

## 🎬 Demo Test Cases

Recommended scenarios for the academic demonstration:

1. **Local RAG** — `Đăng ký tạm trú cần những giấy tờ gì?`
2. **Synonym Mapping** — `ĐKKH cần những giấy tờ gì?`
3. **Insufficient Local Data** — ask a procedure that is not sufficiently covered by the local knowledge base.
4. **Permission + MCP** — when Local RAG is insufficient, demonstrate the user permission step before external retrieval.
5. **DVC procedure/fee query** — demonstrate retrieval from the DVC source after permission.
6. **Session History** — create a conversation, reload/request history, and continue the same session.
7. **Multi-session** — use different session IDs to demonstrate separated conversation contexts.
8. **Auto Fine-Tuning** — demonstrate the dataset → training → evaluation → activation/rollback pipeline.

## 🧪 Testing

### Test the trained model

```powershell
python test_trained_v3.py
```

### Test the MCP server

```powershell
python mcp_server.py
```

### Recommended RAG questions

Examples:

```text
Thủ tục đăng ký tạm trú cần những giấy tờ gì?

Đăng ký kết hôn mất bao lâu?

Chứng thực chữ ký mất bao lâu?

Lệ phí đăng ký tạm trú là bao nhiêu?

Đăng ký tạm trú thực hiện ở đâu?
```

When evaluating the system, pay attention to:

- Retrieval relevance
- Exact procedure matching
- Factual grounding
- Hallucination
- Answer completeness
- Response latency

---

## 👥 Multi-user & Session History

The application supports multiple independent chat sessions through `session_id`.

### Session flow

```text
User
  ↓
Frontend
  ↓
session_id
  ↓
Core API
  ↓
Session / Conversation History
  ↓
RAG + MCP + Model
  ↓
Answer
```

### Multi-user behavior

- Each conversation is associated with a `session_id`.
- Different sessions keep their conversation context separate.
- The frontend sends the session identifier with chat requests.
- The Core API uses the session identifier when handling conversation history.
- The architecture can serve multiple users/sessions without loading a separate Qwen model for every request.

### Conversation History

The frontend can request conversation history through:

```text
GET /api/v1/chat/history
```

Chat requests use:

```text
POST /api/v1/chat/completions
```

The session/history mechanism allows a user to continue a previous conversation while keeping different sessions logically separated.

## 🔄 Auto Fine-Tuning

The project includes an automated fine-tuning pipeline under `auto_finetune/`:

```text
Dataset
  ↓
Validation
  ↓
Formatting
  ↓
QLoRA Training
  ↓
Evaluation
  ↓
Activation / Rollback
```

A new dataset can be placed into the current dataset directory and processed by `run_auto.py`. The candidate adapter is activated only when evaluation passes; otherwise the previously active adapter is retained.

Run:

```powershell
python auto_finetune/run_auto.py
```

## 🔄 Current Development Status

### Completed

- [x] Qwen2.5-7B-Instruct inference
- [x] LoRA V3 fine-tuning
- [x] Auto Fine-Tuning pipeline
- [x] Vietnamese legal instruction dataset preparation
- [x] DVC procedure dataset integration
- [x] Qdrant vector database
- [x] BGE-M3 RAG retrieval pipeline
- [x] BGE Reranker
- [x] MCP legal search tools
- [x] Permission-controlled external retrieval
- [x] FastAPI Core API
- [x] FastAPI Model API
- [x] React/Vite frontend
- [x] Streaming AI responses
- [x] Frontend UI update
- [x] Session-based conversation history
- [x] Multi-session architecture
- [x] Git LFS dataset storage
- [x] Cloudflare Tunnel deployment
- [x] Public Internet testing from another device

### Next steps

- [ ] Add deterministic legal source references to generated answers
- [ ] Run a larger legal-question evaluation set
- [ ] Record retrieval and answer-quality results
- [ ] Create `start.bat` for one-click startup
- [ ] Revisit Docker packaging after native demo is stable
- [ ] Capture screenshots for the academic report
- [ ] Prepare presentation slides and demo script

---

## ⚠️ Limitations

- Retrieval quality depends on the quality and coverage of the local legal dataset.
- External DVC search may return noisy or partially mismatched procedures, so procedure matching remains an important limitation.
- The multi-stage pipeline can increase response latency compared with direct LLM inference.
- RAG reduces hallucination risk but cannot guarantee that generated answers are always correct.
- The current knowledge scope focuses strongly on Vietnamese administrative procedures and does not represent the complete body of Vietnamese law.
- External retrieval depends on the availability and response quality of the DVC source.

## ⚠️ Disclaimer

This project is an **academic/research prototype** and is not a substitute for professional legal advice.

Legal procedures, fees, required documents, processing times, and regulations may change. Users should verify important information against current official sources before taking legal or administrative action.

---

## 👨‍💻 Project

**DAI-Legal-Model**

Vietnamese Legal AI Assistant using:

```text
Qwen2.5-7B-Instruct
        +
LoRA / QLoRA
        +
RAG
        +
Qdrant
        +
MCP
        +
FastAPI
        +
React / Vite
        +
Cloudflare Tunnel
```
