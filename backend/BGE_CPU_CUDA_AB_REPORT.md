# BGE CPU vs CUDA A/B report

Status: **BLOCKED BEFORE EXPERIMENT**  
Audit date: 2026-11-03  
No API/model was started, no environment file was changed, and no Qdrant data
was read or modified during this attempt.

## Why the experiment did not run

Read-only listener checks found no service listening on:

| Service | Port | Observed |
| --- | ---: | --- |
| Core API | 8000 | No listener |
| Model API | 8001 | No listener |
| Gateway | 8002 | No listener |
| Qdrant | 6333 | No listener |
| PostgreSQL | 5432 | No listener |

Two Python processes, PIDs `15284` and `20488`, still appeared in the GPU
process table and were responsive to `Get-Process`. Their command lines and
service ownership could not be inspected in this environment. WDDM reported
per-process GPU memory as `N/A`. I therefore cannot rule out that one of these
processes owns an already loaded model. Starting Model API could load a second
Qwen instance, which is prohibited for this experiment.

At the snapshot (`nvidia-smi`, 2026-11-03 16:36:46), the RTX 3060 showed
11,641 / 12,288 MiB in use, 0% GPU utilization, 45 C, and 50 W. About 647 MiB
was free by subtraction. This is total WDDM GPU memory, not model-specific
memory. With the existing Python PIDs unidentified and little free VRAM, no
model startup or inference was attempted.

## Configuration inspected

`configs/cpu_ab_experiment.env.example` contains exactly:

```dotenv
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

It is explicitly marked as an experimental Core API profile and is not
automatically loaded. It does not change Qwen placement. No root or `backend`
`.env` file was found by the read-only file check. The running processes'
environment could not be read, so their effective settings remain unknown.

The CUDA baseline values are the existing source defaults when these variables
are unset and CUDA is available:

```text
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
```

The model IDs, embedding dimension/normalization, reranker score behavior,
retrieval thresholds, and Qwen configuration were not changed.

## Dataset and query set

No Qdrant collection or points were inspected because Qdrant was not listening.
No query set was run or frozen. Selecting 20 queries without access to current
indexed data would risk testing absent or fabricated records. The A/B remains
blocked until at least 20 representative queries can be selected from the
actual indexed corpus and frozen for both configurations.

## Results

| Measurement | CUDA baseline | CPU BGE | Result |
| --- | --- | --- | --- |
| BGE-M3 embedding latency | Not measured | Not measured | Blocked |
| Embedding dimension and output consistency | Not measured | Not measured | Blocked |
| Retrieval top-K IDs/scores | Not measured | Not measured | Blocked |
| Reranker latency/scores/order | Not measured | Not measured | Blocked |
| Core process RAM | Not measured | Not measured | Blocked |
| PyTorch allocated/reserved CUDA memory | Not measured | Not measured | No live Core allocator telemetry |
| Total GPU memory | 11,641 / 12,288 MiB host snapshot | Not measured | No before/after comparison |
| Qwen latency/device map | Not measured | Not measured | No request sent; no duplicate model loaded |

No retrieval consistency or performance improvement is claimed. The official
relevance set is also not yet human-approved, so this run could not certify
retrieval quality even if latency measurements were available.

## Manual startup requirements

Do not start the stack until the owner has identified PIDs `15284` and `20488`
and confirmed neither is an active Qwen/Model API process. Also confirm enough
VRAM is available for the existing CUDA baseline without terminating processes
or unloading models owned by someone else.

The documented dependencies and launcher are:

```powershell
docker compose -f backend/docker-compose.yml up -d postgres qdrant
```

After confirming no existing API/model process owns the configured ports or
GPU allocation, start the API stack from the repository root:

```powershell
Remove-Item Env:EMBEDDING_DEVICE, Env:RERANKER_DEVICE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe run_gateway_stack.py
```

This launcher starts Core API, Model API, and Gateway. For the CPU phase, keep
the same Model API and Gateway processes alive, stop only Core from its owning
terminal after collecting the CUDA run, and start Core in a new terminal with:

```powershell
$env:EMBEDDING_DEVICE = "cpu"
$env:RERANKER_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m backend.main
```

Do not launch a second full stack for the CPU phase; that would attempt to load
another Qwen instance. Stop the CPU Core manually and start it without those
variables to roll back. The Core startup logs must confirm both BGE models are
on CPU before running phase B.

## Remaining blockers

1. Identify the two existing Python GPU processes and establish that no Qwen
   instance would be duplicated.
2. Bring up PostgreSQL, Qdrant, Core, Model API, and Gateway manually after
   checking port/GPU ownership.
3. Confirm indexed domains and choose a fixed 20+ query set from real data.
4. Capture Core and Model API allocator telemetry plus host `nvidia-smi`
   measurements for each matched run.
5. Run both phases and compare retrieval IDs/scores, reranker ordering, latency,
   RAM, and verifier/citation outcomes.

No production defaults were changed. No service restart, model load, live
request, retrieval, or inference was performed in this attempt.
