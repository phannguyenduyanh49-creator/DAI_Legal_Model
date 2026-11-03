# GPU VRAM audit and controlled CPU/GPU experiment

Audit date: 2026-11-03. This report is based on source inspection and read-only
host diagnostics. No service was restarted, no model was loaded by this audit,
and no inference benchmark was run.

## Findings

### Model lifecycle and placement

- `backend.main` initializes the retrieval models at module import. In Qdrant
  mode, it loads `BAAI/bge-m3` once with `SentenceTransformer(...,
  device=EMBEDDING_DEVICE)`. `EMBEDDING_DEVICE` defaults to CUDA. In
  Elasticsearch mode, BGE-M3 initialization is skipped.
- `backend.main` loads `BAAI/bge-reranker-v2-m3` once at module import using
  `FlagReranker(..., devices=RERANKER_DEVICE)`. Its default is CUDA when CUDA
  is available. It is loaded in either search mode. Per-request retrieval does
  not reinitialize either model.
- `backend.model_api` loads `Qwen/Qwen2.5-7B-Instruct` and
  `outputs/qwen-legal-lora-v4` at module import. Source configuration remains
  4-bit NF4, BF16 compute, and `device_map="auto"`. It loads once per process.
- The prior implementation added an OS lock for the Model API, so another
  process using the same lock path fails before loading Qwen. The prior change
  also bounds generation concurrency and queue size. Those changes have not
  been activated in the currently running processes because they require a
  manual service start.
- The normal Core path calls `Verifier()` without confidence or grounding
  scorers. The experimental Transformers NLI adapter is not imported or loaded
  by normal inference; its weights load only through an explicit
  `from_pretrained` call in the experimental/evaluation path.
- There is no source evidence of per-request model loading or a second Qwen
  construction in the normal service path. Multiple Core worker processes
  would still each initialize their own BGE models; run one Core worker for
  this single-GPU setup.

### Root cause assessment

The source-supported cause of sustained GPU pressure is that Core can keep
BGE-M3 and the BGE reranker resident on CUDA while the separate Model API keeps
Qwen resident on CUDA. Each process also owns CUDA runtime/context allocations.
This explains why model memory may remain allocated while utilization is low.

This is a likely architectural explanation, not exact VRAM attribution. Under
Windows WDDM the available `nvidia-smi` output reports per-process memory as
`N/A`, so the measurements cannot say how much belongs to Qwen, BGE-M3, the
reranker, CUDA contexts, or desktop applications. The PIDs below are Python
processes; the current diagnostics did not independently establish their
command lines or service ownership.

## Read-only runtime measurements

At the latest snapshot, `nvidia-smi` reported:

- GPU: NVIDIA GeForce RTX 3060, 12,288 MiB total.
- Used: 11,621 MiB; available by subtraction: about 667 MiB.
- Utilization: 0%; temperature: 45 C; power: 49 W.
- Visible Python PIDs: 15284 and 20488; their individual GPU memory was `N/A`.
- `Get-Process` showed working sets of about 229 MiB and 26.5 MiB. Host RAM
  working set is not GPU memory attribution.
- CIM process inspection was denied. No listeners were detected at the
  configured ports 8000, 8001, 8002, 5173, 5432, 6333, or 9200 at audit time.

No live PyTorch allocator counters or model device maps were collected because
the API ports were not listening and importing the Model API would load Qwen.
Historical host snapshots of 11,799 MiB and 11,758 MiB are not comparable
before/after inference measurements. They do not demonstrate memory savings.

The last previously recorded answer request took about 53.5 seconds, with
about 47.9 seconds in two model attempts. This is a historical baseline, not a
measurement from this audit. Current source logs per-attempt duration, token
counts, generation duration, tokens/sec, and PyTorch allocated/reserved/peak
memory when the services are run. TTFT is not measured because Model API
generation is non-streaming.

## Existing controls and changes

The working tree already contains the following controls from prior work:

- `EMBEDDING_DEVICE=cpu|cuda` and `RERANKER_DEVICE=cpu|cuda` independently
  select Core model placement. Defaults preserve prior CUDA behavior, invalid
  values fail startup, and requesting unavailable CUDA does not silently fall
  back to CPU.
- `configs/cpu_ab_experiment.env.example` is an explicit, non-auto-loaded
  profile setting both BGE models to CPU. It does not change Qwen placement.
- Model API defaults to one active generation and a queue of two, with a
  bounded queue timeout and a retryable HTTP 503 when full.
- `MODEL_MAX_INPUT_TOKENS` defaults to 8192. Overlong formatted prompts are
  rejected with HTTP 413; relevant Evidence is not silently truncated.
- Startup/inference logs distinguish PyTorch process allocator counters from
  total GPU memory. Core logs selected devices, initialization duration,
  embedding dimension, and inference durations. Model API logs model/adapter,
  quantization, device-map layer counts, token counts, inference duration, and
  allocator counters.

No model identity, adapter, precision, quantization, retrieval threshold,
vector, API schema, or verifier policy was changed for this audit. No new
optimization was enabled because the CPU trade-off has not been measured on
the same queries and candidate set.

## Controlled A/B plan

Use the same fixed queries and indexed data in both runs. Record at least:

1. Model startup duration and selected/actual device for both BGE models.
2. CUDA allocator allocated/reserved counters in Core and Model API.
3. `nvidia-smi` total usage before and after startup and generation.
4. Embedding, retrieval, reranking, generation, and total latency.
5. Embedding dimensions and output checks; retrieval IDs/order; reranker
   scores/order; Evidence selection and citation/verifier outcomes.

Compare the normal CUDA profile to the supplied CPU-BGE profile. Keep Qwen
unchanged. Use a small, fixed dev query set first; do not claim retrieval
quality preservation until candidate IDs/ranks and scores have been compared.
The official relevance benchmark remains blocked until human labels are
approved.

## Manual restart commands

No services were restarted during the audit. When ready to run the experimental
Core placement, manually stop only the existing Core API from its owning
terminal with `Ctrl+C`. Leave the Model API and Gateway running. Then start a
new PowerShell terminal at the repository root and run:

```powershell
$env:EMBEDDING_DEVICE = "cpu"
$env:RERANKER_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m backend.main
```

This starts only Core API on its configured/default port 8000. Confirm its
startup log says `selected_device=cpu` / `actual_device=cpu` for both BGE
models before sending benchmark requests. Do not start `run_gateway_stack.py`
for this Core-only experiment because it also starts Model API and Gateway.

For rollback, stop that experimental Core process from its terminal using
`Ctrl+C`, then start Core without the experimental variables:

```powershell
Remove-Item Env:EMBEDDING_DEVICE, Env:RERANKER_DEVICE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m backend.main
```

This returns Core to the source defaults (CUDA when available). The Model API
does not need a restart for the BGE placement experiment. Do not run these
commands while another Core process owns port 8000.

The Model API changes from prior work (single-instance lock, bounded inference
queue, and input token guard) take effect only after that service is started
from the updated code. Only after manually confirming the old Model API process
has stopped and port 8001 is free, it can be started in its own terminal with:

```powershell
.\.venv\Scripts\python.exe -m backend.model_api
```

Do not start this while the old Model API process is still loading or serving.
The process lock is a second guard for updated instances, not a substitute for
confirming ownership of an already-running older process.

## Tests and remaining blockers

The existing test suites cover device configuration, no silent CUDA fallback,
the Model API admission/instance lock, retrieval and reranking behavior,
Evidence/citation/verifier policy, and API response contracts. Run before an
A/B rollout:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

No live E2E or CPU-vs-CUDA A/B result is claimed in this report. Remaining
blockers are per-process VRAM attribution under WDDM, absent listeners during
this audit, no PyTorch allocator snapshot from the running services, and no
approved labeled retrieval set. CPU BGE placement is an experiment, not yet a
production recommendation.
